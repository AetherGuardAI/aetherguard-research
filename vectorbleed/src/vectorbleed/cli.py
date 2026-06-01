"""VectorBleed CLI — Cross-Tenant Vector Database Security Research Tool."""

import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel

from vectorbleed.config import get_settings

# Suppress noisy deprecation warnings from dependencies
warnings.filterwarnings("ignore", message="SelectableGroups dict interface is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="opentelemetry")

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """VectorBleed: Cross-Tenant Data Leakage Research Framework.

    Tests multi-tenant isolation in vector databases under adversarial conditions.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


@main.command()
@click.option("--output-dir", "-o", type=click.Path(), default="results/corpus", help="Output directory")
@click.pass_context
def generate_corpus(ctx: click.Context, output_dir: str) -> None:
    """Generate synthetic tenant document corpora using GPT-4."""
    from vectorbleed.corpus.generator import CorpusGenerator

    console.print(Panel("Generating Synthetic Tenant Corpora", style="bold blue"))

    generator = CorpusGenerator()
    result = generator.generate_corpus(output_dir=Path(output_dir))

    console.print(f"\n[bold green]Corpus generation complete![/bold green]")
    console.print(f"  Tenant A: {result['tenant_a']['count']} documents")
    console.print(f"  Tenant B: {result['tenant_b']['count']} documents")


@main.command()
@click.option("--corpus-dir", "-c", type=click.Path(exists=True), default="results/corpus", help="Corpus directory")
@click.pass_context
def setup_db(ctx: click.Context, corpus_dir: str) -> None:
    """Initialize Pinecone and upload tenant documents."""
    from vectorbleed.databases.base import TenantConfig
    from vectorbleed.databases.pinecone_db import PineconeAdapter
    from vectorbleed.embeddings import EmbeddingEngine

    settings = get_settings()
    console.print(Panel("Setting Up Multi-Tenant Vector Database", style="bold blue"))

    # Initialize components
    embedding_engine = EmbeddingEngine(settings)
    db = PineconeAdapter(settings)
    db.initialize()

    # Load corpora
    corpus_path = Path(corpus_dir)
    tenant_a_path = corpus_path / "tenant_a_financial.json"
    tenant_b_path = corpus_path / "tenant_b_healthcare.json"

    if not tenant_a_path.exists() or not tenant_b_path.exists():
        console.print("[red]Corpus files not found. Run 'vectorbleed generate-corpus' first.[/red]")
        sys.exit(1)

    with open(tenant_a_path) as f:
        tenant_a_docs = json.load(f)
    with open(tenant_b_path) as f:
        tenant_b_docs = json.load(f)

    # Configure tenants
    tenant_a = TenantConfig(tenant_id="tenant_financial", namespace=settings.tenant_a_namespace)
    tenant_b = TenantConfig(tenant_id="tenant_healthcare", namespace=settings.tenant_b_namespace)

    # Embed and upload Tenant A
    console.print(f"\n[bold]Embedding Tenant A ({len(tenant_a_docs)} docs)...[/bold]")
    texts_a = [doc["content"] for doc in tenant_a_docs]
    embeddings_a = embedding_engine.embed_batch(texts_a)
    count_a = db.upsert_documents(tenant_a, tenant_a_docs, embeddings_a)
    console.print(f"  [green]✓ Uploaded {count_a} vectors to namespace '{tenant_a.namespace}'[/green]")

    # Embed and upload Tenant B
    console.print(f"\n[bold]Embedding Tenant B ({len(tenant_b_docs)} docs)...[/bold]")
    texts_b = [doc["content"] for doc in tenant_b_docs]
    embeddings_b = embedding_engine.embed_batch(texts_b)
    count_b = db.upsert_documents(tenant_b, tenant_b_docs, embeddings_b)
    console.print(f"  [green]✓ Uploaded {count_b} vectors to namespace '{tenant_b.namespace}'[/green]")

    # Save embeddings for later use
    emb_dir = corpus_path / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    np.save(emb_dir / "tenant_a_embeddings.npy", embeddings_a)
    np.save(emb_dir / "tenant_b_embeddings.npy", embeddings_b)
    console.print(f"\n  [green]✓ Embeddings saved to {emb_dir}[/green]")

    # Print stats
    stats_a = db.get_namespace_stats(tenant_a)
    stats_b = db.get_namespace_stats(tenant_b)
    console.print(f"\n[bold]Index Stats:[/bold]")
    console.print(f"  Tenant A ({tenant_a.namespace}): {stats_a.get('vector_count', 0)} vectors")
    console.print(f"  Tenant B ({tenant_b.namespace}): {stats_b.get('vector_count', 0)} vectors")
    console.print(f"  Total: {stats_a.get('total_vector_count', 0)} vectors")

    console.print(f"\n[bold green]Database setup complete![/bold green]")


@main.command()
@click.option("--experiment", "-e", type=click.Choice(["1", "2", "3", "4", "5", "all"]), default="all")
@click.option("--corpus-dir", "-c", type=click.Path(exists=True), default="results/corpus")
@click.option("--output-dir", "-o", type=click.Path(), default="results/experiments")
@click.pass_context
def run_experiments(ctx: click.Context, experiment: str, corpus_dir: str, output_dir: str) -> None:
    """Run attack experiments against the multi-tenant setup."""
    from vectorbleed.databases.base import TenantConfig
    from vectorbleed.databases.pinecone_db import PineconeAdapter
    from vectorbleed.embeddings import EmbeddingEngine
    from vectorbleed.experiments import (
        CentroidInjectionExperiment,
        EmbeddingInversionExperiment,
        MisconfigurationExperiment,
        ProximityProbingExperiment,
        SideChannelExperiment,
    )

    settings = get_settings()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel("Running VectorBleed Attack Experiments", style="bold red"))

    # Initialize
    embedding_engine = EmbeddingEngine(settings)
    db = PineconeAdapter(settings)
    db.initialize()

    tenant_a = TenantConfig(tenant_id="tenant_financial", namespace=settings.tenant_a_namespace)
    tenant_b = TenantConfig(tenant_id="tenant_healthcare", namespace=settings.tenant_b_namespace)

    # Load embeddings if available
    emb_dir = Path(corpus_dir) / "embeddings"
    tenant_a_emb = None
    tenant_b_emb = None
    if (emb_dir / "tenant_a_embeddings.npy").exists():
        tenant_a_emb = np.load(emb_dir / "tenant_a_embeddings.npy")
        tenant_b_emb = np.load(emb_dir / "tenant_b_embeddings.npy")

    # Load victim documents for Exp 5
    victim_docs = None
    victim_path = Path(corpus_dir) / "tenant_b_healthcare.json"
    if victim_path.exists():
        with open(victim_path) as f:
            victim_docs = json.load(f)

    results = {}
    experiments_to_run = ["1", "2", "3", "4", "5"] if experiment == "all" else [experiment]

    for exp_num in experiments_to_run:
        if exp_num == "1":
            exp = ProximityProbingExperiment(db, embedding_engine, tenant_a, tenant_b, settings)
            summary = exp.run()
        elif exp_num == "2":
            exp = CentroidInjectionExperiment(db, embedding_engine, tenant_a, tenant_b, settings)
            summary = exp.run(tenant_a_embeddings=tenant_a_emb, tenant_b_embeddings=tenant_b_emb)
        elif exp_num == "3":
            exp = SideChannelExperiment(db, embedding_engine, tenant_a, tenant_b, settings)
            summary = exp.run()
        elif exp_num == "4":
            exp = MisconfigurationExperiment(db, embedding_engine, tenant_a, tenant_b, settings)
            summary = exp.run()
        elif exp_num == "5":
            exp = EmbeddingInversionExperiment(db, embedding_engine, tenant_a, tenant_b, settings)
            summary = exp.run(victim_documents=victim_docs)
        else:
            continue

        results[summary.experiment_id] = summary
        summary.save(output_path)

    console.print(f"\n[bold green]All experiments complete! Results saved to {output_path}[/bold green]")

    # Print summary table
    console.print(f"\n{'='*70}")
    console.print(f"{'Experiment':<35} {'Probes':<8} {'Detections':<12} {'Rate':<8}")
    console.print(f"{'-'*70}")
    for summary in results.values():
        console.print(
            f"{summary.experiment_name:<35} "
            f"{summary.total_probes:<8} "
            f"{summary.cross_tenant_detections:<12} "
            f"{summary.detection_rate:<8.2%}"
        )


@main.command()
@click.option("--corpus-dir", "-c", type=click.Path(exists=True), default="results/corpus")
@click.option("--output-dir", "-o", type=click.Path(), default="results/defenses")
@click.pass_context
def run_defenses(ctx: click.Context, corpus_dir: str, output_dir: str) -> None:
    """Test defense mitigations against the attack vectors."""
    from vectorbleed.databases.base import TenantConfig
    from vectorbleed.databases.pinecone_db import PineconeAdapter
    from vectorbleed.defenses.mitigations import DefenseTester
    from vectorbleed.embeddings import EmbeddingEngine

    settings = get_settings()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel("Testing Defense Mitigations", style="bold magenta"))

    embedding_engine = EmbeddingEngine(settings)
    db = PineconeAdapter(settings)
    db.initialize()

    tenant_a = TenantConfig(tenant_id="tenant_financial", namespace=settings.tenant_a_namespace)
    tenant_b = TenantConfig(tenant_id="tenant_healthcare", namespace=settings.tenant_b_namespace)

    tester = DefenseTester(db, embedding_engine, tenant_a, tenant_b)
    results = tester.run_all()

    # Save results
    report = [
        {
            "name": r.mitigation_name,
            "id": r.mitigation_id,
            "effectiveness": r.effectiveness_score,
            "overhead_ms": r.performance_overhead_ms,
            "accuracy_impact": r.retrieval_accuracy_impact,
            "cost_multiplier": r.cost_multiplier,
            "blocks": r.blocks_experiments,
            "tradeoffs": r.tradeoffs,
            "test_results": r.test_results,
        }
        for r in results
    ]

    with open(output_path / "defense_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(f"\n[bold green]Defense testing complete! Results saved to {output_path}[/bold green]")


@main.command()
@click.option("--db", "-d", type=click.Choice(["pinecone", "weaviate", "qdrant", "chroma", "all", "comparative"]),
              default="all", help="Which DB results to report on. 'comparative' combines all into one scorecard.")
@click.option("--output-dir", "-o", type=click.Path(), default="results/reports")
@click.pass_context
def generate_report(ctx: click.Context, db: str, output_dir: str) -> None:
    """Generate research reports from experiment results.

    Use --db to target a specific database's results, or 'comparative' to
    produce the combined isolation scorecard across all tested databases.
    """
    from vectorbleed.analysis.scorecard import IsolationScorecard, IsolationScore, build_scorecard_from_results
    from vectorbleed.experiments.base import ExperimentSummary
    from vectorbleed.reporting.report import ReportGenerator

    console.print(Panel("Generating Research Report", style="bold green"))

    base_output = Path(output_dir)

    # Map db names to their results directories
    db_dirs = {
        "pinecone": Path("results/experiments_pinecone"),
        "weaviate": Path("results/experiments_weaviate"),
        "qdrant": Path("results/experiments_qdrant"),
        "chroma": Path("results/experiments_chroma"),
    }

    # Also check the legacy path for pinecone
    if not db_dirs["pinecone"].exists() and Path("results/experiments").exists():
        db_dirs["pinecone"] = Path("results/experiments")

    db_display_names = {
        "pinecone": "Pinecone Namespace",
        "weaviate": "Weaviate Shard",
        "qdrant": "Qdrant Collection",
        "chroma": "ChromaDB Collection",
    }

    if db == "comparative":
        # Build combined scorecard across all databases
        _generate_comparative_report(db_dirs, db_display_names, base_output)
        return

    # Determine which DBs to report on
    if db == "all":
        dbs_to_report = [k for k, v in db_dirs.items() if v.exists()]
    else:
        dbs_to_report = [db]

    for db_name in dbs_to_report:
        exp_path = db_dirs[db_name]
        if not exp_path.exists():
            console.print(f"[yellow]No results found for {db_name} at {exp_path}. Skipping.[/yellow]")
            continue

        results = _load_experiment_results(exp_path)
        if not results:
            console.print(f"[yellow]No experiment JSON files in {exp_path}. Skipping.[/yellow]")
            continue

        display_name = db_display_names[db_name]
        scorecard = build_scorecard_from_results(results, display_name)

        console.print(f"\n[bold]{display_name} Scorecard:[/bold]")
        scorecard.display()

        # Generate per-DB reports
        db_report_dir = base_output / db_name
        reporter = ReportGenerator(db_report_dir)
        paths = reporter.generate_full_report(results, scorecard)
        console.print(f"  Reports saved to: {db_report_dir}/")

    console.print(f"\n[bold green]Report generation complete![/bold green]")


def _load_experiment_results(exp_path: Path) -> dict:
    """Load experiment results from a directory of JSON files."""
    from vectorbleed.experiments.base import ExperimentSummary

    results = {}
    for json_file in exp_path.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            summary = ExperimentSummary(
                experiment_name=data.get("experiment_name", ""),
                experiment_id=data.get("experiment_id", ""),
                description=data.get("description", ""),
                total_probes=data.get("total_probes", 0),
                cross_tenant_detections=data.get("cross_tenant_detections", 0),
                detection_rate=data.get("detection_rate", 0.0),
                avg_latency_ms=data.get("avg_latency_ms", 0.0),
                max_similarity_score=data.get("max_similarity_score", 0.0),
                results=[],
                start_time=data.get("start_time", ""),
                end_time=data.get("end_time", ""),
                duration_seconds=data.get("duration_seconds", 0.0),
                metadata=data.get("metadata", {}),
            )
            results[summary.experiment_id] = summary
    return results


def _generate_comparative_report(db_dirs: dict, db_display_names: dict, output_dir: Path) -> None:
    """Generate the comparative isolation scorecard across all databases."""
    from vectorbleed.analysis.scorecard import IsolationScorecard, IsolationScore

    console.print("[bold]Building Comparative Isolation Scorecard[/bold]\n")

    attack_vectors = [
        "Proximity Probing",
        "Centroid Injection",
        "Score Side-Channel",
        "Framework Misconfiguration",
        "Embedding Inversion",
    ]

    exp_to_vector = {
        "exp1_proximity_probing": "Proximity Probing",
        "exp2_centroid_injection": "Centroid Injection",
        "exp3_score_sidechannel": "Score Side-Channel",
        "exp4_framework_misconfiguration": "Framework Misconfiguration",
        "exp5_embedding_inversion": "Embedding Inversion",
    }

    databases_found = []
    scorecard = IsolationScorecard(databases=[], attack_vectors=attack_vectors)

    for db_name, exp_path in db_dirs.items():
        if not exp_path.exists():
            continue

        results = _load_experiment_results(exp_path)
        if not results:
            continue

        display_name = db_display_names[db_name]
        databases_found.append(display_name)
        scorecard.databases.append(display_name)

        for exp_id, vector_name in exp_to_vector.items():
            summary = results.get(exp_id)
            if summary is None:
                scorecard.add_score(IsolationScore(
                    attack_vector=vector_name,
                    database=display_name,
                    score="NOT_TESTED",
                    detection_rate=0.0,
                ))
                continue

            rate = summary.detection_rate
            if rate >= 0.5:
                score_label = "VULNERABLE"
            elif rate >= 0.1:
                score_label = "PARTIAL"
            else:
                score_label = "SECURE"

            scorecard.add_score(IsolationScore(
                attack_vector=vector_name,
                database=display_name,
                score=score_label,
                detection_rate=rate,
                details=f"{summary.cross_tenant_detections}/{summary.total_probes} detections",
            ))

    if not databases_found:
        console.print("[red]No experiment results found for any database.[/red]")
        return

    scorecard.compute_overall_grades()
    scorecard.display()

    # Save comparative scorecard
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard.save(output_dir / "comparative_scorecard.json")

    # Generate comparative markdown
    lines = [
        "# VectorBleed Comparative Isolation Report",
        "",
        "## Databases Tested",
        "",
    ]
    for db in databases_found:
        grade = scorecard.overall_grades.get(db, "?")
        lines.append(f"- **{db}**: Grade {grade}")
    lines.append("")
    lines.append("## Isolation Scorecard")
    lines.append("")
    lines.append("| Attack Vector | " + " | ".join(databases_found) + " |")
    lines.append("|---" + "|---" * len(databases_found) + "|")

    for vector in attack_vectors:
        row = [vector]
        for db in databases_found:
            s = next((x for x in scorecard.scores if x.attack_vector == vector and x.database == db), None)
            if s:
                row.append(f"{s.score} ({s.detection_rate:.0%})")
            else:
                row.append("NOT TESTED")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Overall Grades")
    lines.append("")
    for db, grade in scorecard.overall_grades.items():
        lines.append(f"- **{db}**: {grade}")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by VectorBleed v0.1.0 — AetherGuard Research*")

    with open(output_dir / "comparative_report.md", "w") as f:
        f.write("\n".join(lines))

    console.print(f"\n[bold green]Comparative report saved to {output_dir}/[/bold green]")
    console.print(f"  - comparative_scorecard.json")
    console.print(f"  - comparative_report.md")


@main.command()
@click.option("--corpus-dir", "-c", type=click.Path(exists=True), default="results/corpus")
@click.option("--output-dir", "-o", type=click.Path(), default="results/visualizations")
@click.pass_context
def visualize(ctx: click.Context, corpus_dir: str, output_dir: str) -> None:
    """Generate embedding space visualizations."""
    from vectorbleed.analysis.visualization import plot_embedding_space

    console.print(Panel("Generating Visualizations", style="bold cyan"))

    emb_dir = Path(corpus_dir) / "embeddings"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not (emb_dir / "tenant_a_embeddings.npy").exists():
        console.print("[red]Embeddings not found. Run 'vectorbleed setup-db' first.[/red]")
        sys.exit(1)

    tenant_a_emb = np.load(emb_dir / "tenant_a_embeddings.npy")
    tenant_b_emb = np.load(emb_dir / "tenant_b_embeddings.npy")

    # PCA visualization
    console.print("  Generating PCA visualization...")
    plot_embedding_space(
        tenant_a_emb, tenant_b_emb,
        output_path=output_path / "embedding_space_pca.png",
        method="pca",
        title="Multi-Tenant Embedding Space (PCA)",
    )

    # t-SNE visualization
    console.print("  Generating t-SNE visualization...")
    plot_embedding_space(
        tenant_a_emb, tenant_b_emb,
        output_path=output_path / "embedding_space_tsne.png",
        method="tsne",
        title="Multi-Tenant Embedding Space (t-SNE)",
    )

    console.print(f"\n[bold green]Visualizations saved to {output_path}[/bold green]")


@main.command()
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run the complete VectorBleed research pipeline.

    Executes: generate-corpus → setup-db → run-experiments → run-defenses → visualize → generate-report
    """
    console.print(Panel(
        "[bold]VectorBleed: Complete Research Pipeline[/bold]\n\n"
        "This will execute the full research workflow:\n"
        "1. Generate synthetic tenant corpora (GPT-4)\n"
        "2. Setup multi-tenant Pinecone database\n"
        "3. Run all 5 attack experiments\n"
        "4. Test defense mitigations\n"
        "5. Generate visualizations\n"
        "6. Produce research report",
        style="bold red",
    ))

    ctx.invoke(generate_corpus)
    ctx.invoke(setup_db)
    ctx.invoke(run_experiments, experiment="all")
    ctx.invoke(run_defenses)
    ctx.invoke(visualize)
    ctx.invoke(generate_report)

    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]VectorBleed research pipeline complete![/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print(f"\nResults: results/")
    console.print(f"Reports: results/reports/")
    console.print(f"Visualizations: results/visualizations/")


@main.command()
@click.option("--db", "-d", type=click.Choice(["pinecone", "weaviate", "qdrant", "chroma", "all"]), default="all",
              help="Which database(s) to clean up")
@click.pass_context
def cleanup(ctx: click.Context, db: str) -> None:
    """Delete all experiment data from vector databases."""
    from vectorbleed.databases.base import TenantConfig

    settings = get_settings()
    tenant_a = TenantConfig(tenant_id="tenant_financial", namespace=settings.tenant_a_namespace)
    tenant_b = TenantConfig(tenant_id="tenant_healthcare", namespace=settings.tenant_b_namespace)

    dbs_to_clean = ["pinecone", "weaviate", "qdrant", "chroma"] if db == "all" else [db]

    if not click.confirm(f"This will delete all VectorBleed data from: {', '.join(dbs_to_clean)}. Continue?"):
        return

    for db_name in dbs_to_clean:
        console.print(f"\n[bold]Cleaning up {db_name}...[/bold]")

        try:
            if db_name == "pinecone":
                from vectorbleed.databases.pinecone_db import PineconeAdapter
                adapter = PineconeAdapter(settings)
                adapter.initialize()
                adapter.delete_namespace(tenant_a)
                adapter.delete_namespace(tenant_b)
                console.print(f"  [green]✓ Pinecone: deleted namespaces {tenant_a.namespace}, {tenant_b.namespace}[/green]")
                adapter.teardown()

            elif db_name == "weaviate":
                from vectorbleed.databases.weaviate_db import WeaviateAdapter
                adapter = WeaviateAdapter(settings)
                adapter.initialize()
                # Delete the entire collection (cleaner than per-tenant)
                try:
                    adapter.client.collections.delete("VectorBleedDocs")
                    console.print(f"  [green]✓ Weaviate: deleted collection VectorBleedDocs[/green]")
                except Exception as e:
                    console.print(f"  [yellow]Weaviate: {e}[/yellow]")
                adapter.teardown()

            elif db_name == "qdrant":
                from vectorbleed.databases.qdrant_db import QdrantAdapter
                adapter = QdrantAdapter(settings)
                adapter.initialize()
                try:
                    adapter.client.delete_collection("vectorbleed_docs")
                    console.print(f"  [green]✓ Qdrant: deleted collection vectorbleed_docs[/green]")
                except Exception as e:
                    console.print(f"  [yellow]Qdrant: {e}[/yellow]")
                adapter.teardown()

            elif db_name == "chroma":
                from vectorbleed.databases.chroma_db import ChromaDBAdapter
                adapter = ChromaDBAdapter(settings)
                adapter.initialize()
                try:
                    adapter.client.delete_collection("vectorbleed_shared")
                    console.print(f"  [green]✓ ChromaDB: deleted collection vectorbleed_shared[/green]")
                except Exception:
                    pass
                try:
                    adapter.client.delete_collection(f"vectorbleed_{tenant_a.namespace}")
                    adapter.client.delete_collection(f"vectorbleed_{tenant_b.namespace}")
                    console.print(f"  [green]✓ ChromaDB: deleted tenant collections[/green]")
                except Exception:
                    pass
                adapter.teardown()

        except Exception as e:
            console.print(f"  [red]Failed to clean {db_name}: {e}[/red]")

    console.print(f"\n[bold green]Cleanup complete![/bold green]")


@main.command()
@click.option("--db", "-d", type=click.Choice(["pinecone", "weaviate", "qdrant", "chroma", "all"]), default="all",
              help="Which database(s) to test")
@click.option("--experiment", "-e", type=click.Choice(["1", "2", "3", "4", "5", "all"]), default="all")
@click.option("--corpus-dir", "-c", type=click.Path(exists=True), default="results/corpus")
@click.option("--output-dir", "-o", type=click.Path(), default="results")
@click.pass_context
def run_other_dbs(ctx: click.Context, db: str, experiment: str, corpus_dir: str, output_dir: str) -> None:
    """Run experiments against Pinecone, Weaviate, Qdrant, and/or ChromaDB.

    Uses existing corpus (run generate-corpus first).
    locally via Docker Compose.
    """
    from vectorbleed.databases.base import TenantConfig
    from vectorbleed.embeddings import EmbeddingEngine
    from vectorbleed.experiments import (
        CentroidInjectionExperiment,
        EmbeddingInversionExperiment,
        MisconfigurationExperiment,
        ProximityProbingExperiment,
        SideChannelExperiment,
    )

    settings = get_settings()
    corpus_path = Path(corpus_dir)
    base_output = Path(output_dir)

    console.print(Panel("Running VectorBleed on Additional Databases", style="bold red"))

    # Load corpus
    tenant_a_path = corpus_path / "tenant_a_financial.json"
    tenant_b_path = corpus_path / "tenant_b_healthcare.json"

    if not tenant_a_path.exists() or not tenant_b_path.exists():
        console.print("[red]Corpus not found. Run 'vectorbleed generate-corpus' first.[/red]")
        sys.exit(1)

    with open(tenant_a_path) as f:
        tenant_a_docs = json.load(f)
    with open(tenant_b_path) as f:
        tenant_b_docs = json.load(f)

    # Load pre-computed embeddings
    emb_dir = corpus_path / "embeddings"
    if not (emb_dir / "tenant_a_embeddings.npy").exists():
        console.print("[red]Embeddings not found. Run 'vectorbleed setup-db' first to generate embeddings.[/red]")
        sys.exit(1)

    tenant_a_emb = np.load(emb_dir / "tenant_a_embeddings.npy")
    tenant_b_emb = np.load(emb_dir / "tenant_b_embeddings.npy")

    embedding_engine = EmbeddingEngine(settings)

    tenant_a = TenantConfig(tenant_id="tenant_financial", namespace=settings.tenant_a_namespace)
    tenant_b = TenantConfig(tenant_id="tenant_healthcare", namespace=settings.tenant_b_namespace)

    dbs_to_test = ["pinecone", "weaviate", "qdrant", "chroma"] if db == "all" else [db]
    experiments_to_run = ["1", "2", "3", "4", "5"] if experiment == "all" else [experiment]

    for db_name in dbs_to_test:
        console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
        console.print(f"[bold cyan]  Testing: {db_name.upper()}[/bold cyan]")
        console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

        # Create adapter
        db_adapter = _create_adapter(db_name, settings)
        if db_adapter is None:
            console.print(f"[red]Failed to initialize {db_name}. Is it running? Skipping...[/red]")
            continue

        try:
            db_adapter.initialize()
        except Exception as e:
            console.print(f"[red]Failed to connect to {db_name}: {e}[/red]")
            console.print(f"[yellow]Make sure Docker containers are running. Skipping...[/yellow]")
            continue

        # Upload corpus (reusing pre-computed embeddings) — skip if already uploaded
        console.print(f"[bold]Checking {db_name} for existing data...[/bold]")
        try:
            stats = db_adapter.get_namespace_stats(tenant_a)
            if stats.get("vector_count", 0) >= len(tenant_a_docs):
                console.print(f"  [green]✓ Corpus already uploaded ({stats['vector_count']} vectors). Skipping upload.[/green]")
            else:
                console.print(f"  Uploading corpus to {db_name}...")
                count_a = db_adapter.upsert_documents(tenant_a, tenant_a_docs, tenant_a_emb)
                count_b = db_adapter.upsert_documents(tenant_b, tenant_b_docs, tenant_b_emb)
                console.print(f"  [green]✓ Tenant A: {count_a} vectors[/green]")
                console.print(f"  [green]✓ Tenant B: {count_b} vectors[/green]")
        except Exception as e:
            console.print(f"  [yellow]Could not check stats, uploading: {e}[/yellow]")
            try:
                count_a = db_adapter.upsert_documents(tenant_a, tenant_a_docs, tenant_a_emb)
                count_b = db_adapter.upsert_documents(tenant_b, tenant_b_docs, tenant_b_emb)
                console.print(f"  [green]✓ Tenant A: {count_a} vectors[/green]")
                console.print(f"  [green]✓ Tenant B: {count_b} vectors[/green]")
            except Exception as e2:
                console.print(f"[red]Failed to upload to {db_name}: {e2}[/red]")
                continue

        # Run experiments
        output_path = base_output / f"experiments_{db_name}"
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}
        for exp_num in experiments_to_run:
            try:
                if exp_num == "1":
                    exp = ProximityProbingExperiment(db_adapter, embedding_engine, tenant_a, tenant_b, settings)
                    summary = exp.run()
                elif exp_num == "2":
                    exp = CentroidInjectionExperiment(db_adapter, embedding_engine, tenant_a, tenant_b, settings)
                    summary = exp.run(tenant_a_embeddings=tenant_a_emb, tenant_b_embeddings=tenant_b_emb)
                elif exp_num == "3":
                    exp = SideChannelExperiment(db_adapter, embedding_engine, tenant_a, tenant_b, settings)
                    summary = exp.run()
                elif exp_num == "4":
                    exp = MisconfigurationExperiment(db_adapter, embedding_engine, tenant_a, tenant_b, settings)
                    summary = exp.run()
                elif exp_num == "5":
                    exp = EmbeddingInversionExperiment(db_adapter, embedding_engine, tenant_a, tenant_b, settings)
                    summary = exp.run(victim_documents=tenant_b_docs)
                else:
                    continue

                results[summary.experiment_id] = summary
                summary.save(output_path)
            except Exception as e:
                console.print(f"[red]Experiment {exp_num} failed on {db_name}: {e}[/red]")
                continue

        # Print summary for this DB
        if results:
            console.print(f"\n[bold green]{'='*70}[/bold green]")
            console.print(f"[bold green]  {db_name.upper()} Results Summary[/bold green]")
            console.print(f"[bold green]{'='*70}[/bold green]")
            console.print(f"{'Experiment':<35} {'Probes':<8} {'Detections':<12} {'Rate':<8}")
            console.print(f"{'-'*70}")
            for summary in results.values():
                console.print(
                    f"{summary.experiment_name:<35} "
                    f"{summary.total_probes:<8} "
                    f"{summary.cross_tenant_detections:<12} "
                    f"{summary.detection_rate:<8.2%}"
                )

        # Cleanup
        try:
            db_adapter.teardown()
        except Exception:
            pass

    console.print(f"\n[bold green]All database tests complete![/bold green]")
    console.print(f"Results saved to: {base_output}/experiments_<db_name>/")


def _create_adapter(db_name: str, settings):
    """Factory function to create database adapters."""
    if db_name == "pinecone":
        from vectorbleed.databases.pinecone_db import PineconeAdapter
        return PineconeAdapter(settings)
    elif db_name == "weaviate":
        from vectorbleed.databases.weaviate_db import WeaviateAdapter
        return WeaviateAdapter(settings)
    elif db_name == "qdrant":
        from vectorbleed.databases.qdrant_db import QdrantAdapter
        return QdrantAdapter(settings)
    elif db_name == "chroma":
        from vectorbleed.databases.chroma_db import ChromaDBAdapter
        return ChromaDBAdapter(settings)
    else:
        return None


if __name__ == "__main__":
    main()
