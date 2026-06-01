"""Isolation Scorecard Generator.

Produces the comparative isolation analysis table showing which vector databases
fail which attack vectors — the key publishable finding.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from vectorbleed.experiments.base import ExperimentSummary

console = Console()


@dataclass
class IsolationScore:
    """Score for a single attack vector against a single database."""

    attack_vector: str
    database: str
    score: str  # "SECURE", "PARTIAL", "VULNERABLE", "NOT_TESTED"
    detection_rate: float
    details: str = ""


@dataclass
class IsolationScorecard:
    """Complete isolation scorecard across all databases and attack vectors."""

    databases: list[str]
    attack_vectors: list[str]
    scores: list[IsolationScore] = field(default_factory=list)
    overall_grades: dict = field(default_factory=dict)

    def add_score(self, score: IsolationScore) -> None:
        """Add a score to the scorecard."""
        self.scores.append(score)

    def compute_overall_grades(self) -> None:
        """Compute overall isolation grade per database."""
        for db in self.databases:
            db_scores = [s for s in self.scores if s.database == db]
            if not db_scores:
                self.overall_grades[db] = "NOT_TESTED"
                continue

            vulnerable_count = sum(1 for s in db_scores if s.score == "VULNERABLE")
            partial_count = sum(1 for s in db_scores if s.score == "PARTIAL")

            if vulnerable_count >= 3:
                self.overall_grades[db] = "F"
            elif vulnerable_count >= 2:
                self.overall_grades[db] = "D"
            elif vulnerable_count >= 1:
                self.overall_grades[db] = "C"
            elif partial_count >= 2:
                self.overall_grades[db] = "B"
            else:
                self.overall_grades[db] = "A"

    def display(self) -> None:
        """Display the scorecard as a rich table."""
        table = Table(title="VectorBleed Isolation Scorecard", show_lines=True)

        table.add_column("Attack Vector", style="bold")
        for db in self.databases:
            table.add_column(db, justify="center")

        for vector in self.attack_vectors:
            row = [vector]
            for db in self.databases:
                score = next(
                    (s for s in self.scores if s.attack_vector == vector and s.database == db),
                    None,
                )
                if score is None:
                    row.append("[dim]NOT TESTED[/dim]")
                elif score.score == "VULNERABLE":
                    row.append(f"[bold red]🔴 VULNERABLE[/bold red]\n({score.detection_rate:.0%})")
                elif score.score == "PARTIAL":
                    row.append(f"[yellow]🟡 PARTIAL[/yellow]\n({score.detection_rate:.0%})")
                elif score.score == "SECURE":
                    row.append(f"[green]🟢 SECURE[/green]\n({score.detection_rate:.0%})")
                else:
                    row.append("[dim]—[/dim]")
            table.add_row(*row)

        # Overall grade row
        grade_row = ["[bold]Overall Grade[/bold]"]
        for db in self.databases:
            grade = self.overall_grades.get(db, "?")
            color = {"A": "green", "B": "green", "C": "yellow", "D": "red", "F": "bold red"}.get(
                grade, "dim"
            )
            grade_row.append(f"[{color}]{grade}[/{color}]")
        table.add_row(*grade_row)

        console.print(table)

    def to_dict(self) -> dict:
        """Convert scorecard to dictionary."""
        return {
            "databases": self.databases,
            "attack_vectors": self.attack_vectors,
            "scores": [asdict(s) for s in self.scores],
            "overall_grades": self.overall_grades,
        }

    def save(self, output_path: Path) -> None:
        """Save scorecard to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_scorecard_from_results(
    experiment_results: dict[str, ExperimentSummary],
    database_name: str = "Pinecone Namespace",
) -> IsolationScorecard:
    """Build a scorecard from experiment results.

    Args:
        experiment_results: Dict mapping experiment_id to ExperimentSummary
        database_name: Name of the database being tested
    """
    scorecard = IsolationScorecard(
        databases=[database_name],
        attack_vectors=[
            "Proximity Probing",
            "Centroid Injection",
            "Score Side-Channel",
            "Framework Misconfiguration",
            "Embedding Inversion",
        ],
    )

    # Map experiment IDs to attack vector names
    exp_to_vector = {
        "exp1_proximity_probing": "Proximity Probing",
        "exp2_centroid_injection": "Centroid Injection",
        "exp3_score_sidechannel": "Score Side-Channel",
        "exp4_framework_misconfiguration": "Framework Misconfiguration",
        "exp5_embedding_inversion": "Embedding Inversion",
    }

    for exp_id, vector_name in exp_to_vector.items():
        summary = experiment_results.get(exp_id)
        if summary is None:
            scorecard.add_score(
                IsolationScore(
                    attack_vector=vector_name,
                    database=database_name,
                    score="NOT_TESTED",
                    detection_rate=0.0,
                )
            )
            continue

        # Determine score based on detection rate
        rate = summary.detection_rate
        if rate >= 0.5:
            score = "VULNERABLE"
        elif rate >= 0.1:
            score = "PARTIAL"
        else:
            score = "SECURE"

        scorecard.add_score(
            IsolationScore(
                attack_vector=vector_name,
                database=database_name,
                score=score,
                detection_rate=rate,
                details=f"{summary.cross_tenant_detections}/{summary.total_probes} probes detected cross-tenant data",
            )
        )

    scorecard.compute_overall_grades()
    return scorecard
