"""Report generation — JSON, CSV, and Markdown outputs."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console

from vectorbleed.analysis.scorecard import IsolationScorecard
from vectorbleed.defenses.mitigations import MitigationResult
from vectorbleed.experiments.base import ExperimentSummary

console = Console()


class ReportGenerator:
    """Generates comprehensive reports from experiment results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        experiment_results: dict[str, ExperimentSummary],
        scorecard: Optional[IsolationScorecard] = None,
        mitigation_results: Optional[list[MitigationResult]] = None,
    ) -> dict[str, Path]:
        """Generate all report formats."""
        paths = {}

        # JSON report
        paths["json"] = self._generate_json_report(experiment_results, scorecard, mitigation_results)

        # CSV summary
        paths["csv"] = self._generate_csv_summary(experiment_results)

        # Markdown report
        paths["markdown"] = self._generate_markdown_report(experiment_results, scorecard, mitigation_results)

        console.print(f"\n[bold green]Reports generated:[/bold green]")
        for fmt, path in paths.items():
            console.print(f"  {fmt}: {path}")

        return paths

    def _generate_json_report(
        self,
        experiment_results: dict[str, ExperimentSummary],
        scorecard: Optional[IsolationScorecard],
        mitigation_results: Optional[list[MitigationResult]],
    ) -> Path:
        """Generate comprehensive JSON report."""
        report = {
            "metadata": {
                "tool": "VectorBleed",
                "version": "0.1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "description": "Cross-Tenant Data Leakage Research Results",
            },
            "experiments": {},
            "scorecard": None,
            "mitigations": None,
        }

        for exp_id, summary in experiment_results.items():
            report["experiments"][exp_id] = summary.to_dict()

        if scorecard:
            report["scorecard"] = scorecard.to_dict()

        if mitigation_results:
            report["mitigations"] = [
                {
                    "name": m.mitigation_name,
                    "id": m.mitigation_id,
                    "effectiveness": m.effectiveness_score,
                    "overhead_ms": m.performance_overhead_ms,
                    "accuracy_impact": m.retrieval_accuracy_impact,
                    "cost_multiplier": m.cost_multiplier,
                    "blocks": m.blocks_experiments,
                    "tradeoffs": m.tradeoffs,
                }
                for m in mitigation_results
            ]

        output_path = self.output_dir / "vectorbleed_report.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def _generate_csv_summary(
        self,
        experiment_results: dict[str, ExperimentSummary],
    ) -> Path:
        """Generate CSV summary of all experiments."""
        output_path = self.output_dir / "vectorbleed_summary.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "experiment_id",
                "experiment_name",
                "total_probes",
                "cross_tenant_detections",
                "detection_rate",
                "avg_latency_ms",
                "max_similarity_score",
                "duration_seconds",
            ])

            for exp_id, summary in experiment_results.items():
                writer.writerow([
                    summary.experiment_id,
                    summary.experiment_name,
                    summary.total_probes,
                    summary.cross_tenant_detections,
                    f"{summary.detection_rate:.4f}",
                    f"{summary.avg_latency_ms:.2f}",
                    f"{summary.max_similarity_score:.4f}",
                    f"{summary.duration_seconds:.1f}",
                ])

        return output_path

    def _generate_markdown_report(
        self,
        experiment_results: dict[str, ExperimentSummary],
        scorecard: Optional[IsolationScorecard],
        mitigation_results: Optional[list[MitigationResult]],
    ) -> Path:
        """Generate Markdown report."""
        output_path = self.output_dir / "vectorbleed_report.md"

        lines = [
            "# VectorBleed Research Report",
            f"## Cross-Tenant Data Leakage in Multi-Tenant Vector Databases",
            f"",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Tool Version**: 0.1.0",
            f"",
            "---",
            "",
            "## Executive Summary",
            "",
        ]

        total_experiments = len(experiment_results)
        vulnerable_experiments = sum(
            1 for s in experiment_results.values() if s.detection_rate > 0.1
        )
        lines.append(f"- **Experiments conducted**: {total_experiments}")
        lines.append(f"- **Vulnerabilities detected**: {vulnerable_experiments}/{total_experiments}")
        lines.append("")

        # Experiment results
        lines.append("## Experiment Results")
        lines.append("")
        lines.append("| Experiment | Probes | Detections | Rate | Max Score |")
        lines.append("|---|---|---|---|---|")

        for summary in experiment_results.values():
            lines.append(
                f"| {summary.experiment_name} | {summary.total_probes} | "
                f"{summary.cross_tenant_detections} | {summary.detection_rate:.2%} | "
                f"{summary.max_similarity_score:.4f} |"
            )

        lines.append("")

        # Detailed findings per experiment
        for summary in experiment_results.values():
            lines.append(f"### {summary.experiment_name}")
            lines.append(f"")
            lines.append(f"**Description**: {summary.description}")
            lines.append(f"")
            lines.append(f"- Total probes: {summary.total_probes}")
            lines.append(f"- Cross-tenant detections: {summary.cross_tenant_detections}")
            lines.append(f"- Detection rate: {summary.detection_rate:.2%}")
            lines.append(f"- Average latency: {summary.avg_latency_ms:.1f}ms")
            lines.append(f"- Duration: {summary.duration_seconds:.1f}s")
            lines.append("")

        # Mitigation results
        if mitigation_results:
            lines.append("## Defense Mitigations")
            lines.append("")
            lines.append("| Mitigation | Effectiveness | Overhead | Cost | Blocks |")
            lines.append("|---|---|---|---|---|")

            for m in mitigation_results:
                blocks = ", ".join(m.blocks_experiments)
                lines.append(
                    f"| {m.mitigation_name} | {m.effectiveness_score:.0%} | "
                    f"{m.performance_overhead_ms:.2f}ms | {m.cost_multiplier:.1f}x | {blocks} |"
                )

            lines.append("")

        # Scorecard
        if scorecard:
            lines.append("## Isolation Scorecard")
            lines.append("")
            lines.append("| Attack Vector | " + " | ".join(scorecard.databases) + " |")
            lines.append("|---" + "|---" * len(scorecard.databases) + "|")

            for vector in scorecard.attack_vectors:
                row = [vector]
                for db in scorecard.databases:
                    score = next(
                        (s for s in scorecard.scores if s.attack_vector == vector and s.database == db),
                        None,
                    )
                    if score:
                        row.append(f"{score.score} ({score.detection_rate:.0%})")
                    else:
                        row.append("NOT TESTED")
                lines.append("| " + " | ".join(row) + " |")

            lines.append("")
            lines.append("**Overall Grades**: " + ", ".join(
                f"{db}: {grade}" for db, grade in scorecard.overall_grades.items()
            ))
            lines.append("")

        lines.append("---")
        lines.append("*Report generated by VectorBleed v0.1.0 — AetherGuard Research*")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path
