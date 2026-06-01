"""GPT-4 powered synthetic document corpus generator."""

import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from vectorbleed.config import Settings, get_settings
from vectorbleed.corpus.templates import (
    FINANCIAL_GENERATION_PROMPT,
    FINANCIAL_TOPICS,
    HEALTHCARE_GENERATION_PROMPT,
    HEALTHCARE_TOPICS,
)

console = Console()


class CorpusGenerator:
    """Generates synthetic multi-tenant document corpora using GPT-4."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_generation_model

    def generate_corpus(self, output_dir: Optional[Path] = None) -> dict:
        """Generate full corpus for both tenants. Returns paths to generated files."""
        output_dir = output_dir or self.settings.results_dir / "corpus"
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print("[bold blue]Generating synthetic tenant corpora...[/bold blue]")

        tenant_a_docs = self._generate_tenant_docs(
            topics=FINANCIAL_TOPICS[: self.settings.corpus_size_per_tenant],
            prompt_template=FINANCIAL_GENERATION_PROMPT,
            tenant_name="Tenant A (Financial Services)",
        )

        tenant_b_docs = self._generate_tenant_docs(
            topics=HEALTHCARE_TOPICS[: self.settings.corpus_size_per_tenant],
            prompt_template=HEALTHCARE_GENERATION_PROMPT,
            tenant_name="Tenant B (Healthcare)",
        )

        # Save corpora
        tenant_a_path = output_dir / "tenant_a_financial.json"
        tenant_b_path = output_dir / "tenant_b_healthcare.json"

        with open(tenant_a_path, "w") as f:
            json.dump(tenant_a_docs, f, indent=2)

        with open(tenant_b_path, "w") as f:
            json.dump(tenant_b_docs, f, indent=2)

        console.print(f"[green]✓ Tenant A corpus: {len(tenant_a_docs)} documents → {tenant_a_path}[/green]")
        console.print(f"[green]✓ Tenant B corpus: {len(tenant_b_docs)} documents → {tenant_b_path}[/green]")

        return {
            "tenant_a": {"path": str(tenant_a_path), "count": len(tenant_a_docs)},
            "tenant_b": {"path": str(tenant_b_path), "count": len(tenant_b_docs)},
        }

    def _generate_tenant_docs(
        self, topics: list[str], prompt_template: str, tenant_name: str
    ) -> list[dict]:
        """Generate documents for a single tenant."""
        documents = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Generating {tenant_name}...", total=len(topics))

            for i, topic in enumerate(topics):
                prompt = prompt_template.format(topic=topic)
                content = self._generate_document(prompt)

                documents.append({
                    "id": f"doc_{i:03d}",
                    "topic": topic,
                    "content": content,
                    "tenant": tenant_name,
                    "domain": "financial" if "Financial" in tenant_name else "healthcare",
                })

                progress.advance(task)

        return documents

    def _generate_document(self, prompt: str, max_retries: int = 3) -> str:
        """Generate a single document using GPT-4."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a document generator creating realistic synthetic enterprise documents for security research. Generate only the document content.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.8,
                    max_tokens=800,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                console.print(f"[yellow]Retry {attempt + 1}/{max_retries}: {e}[/yellow]")
                time.sleep(wait_time)

        return ""  # unreachable

    def load_corpus(self, corpus_path: Path) -> list[dict]:
        """Load a previously generated corpus from JSON."""
        with open(corpus_path) as f:
            return json.load(f)
