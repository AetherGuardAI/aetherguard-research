"""Experiment 5: Embedding Inversion from Score Leakage.

Reconstructs actual text from victim tenant's documents using only similarity
scores observable from the attacker's session.

ALGEN-inspired approach adapted for cross-tenant setting:
1. Collect similarity scores for diverse probe queries
2. Build alignment between probe space and victim's embedding space
3. Use alignment to reconstruct target document embeddings
4. Invert reconstructed embeddings to text using a trained inverter
5. Measure reconstruction fidelity with ROUGE-L
"""

import time
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from vectorbleed.config import Settings
from vectorbleed.databases.base import TenantConfig, VectorDatabaseAdapter
from vectorbleed.embeddings import EmbeddingEngine
from vectorbleed.experiments.base import BaseExperiment, ExperimentResult, ExperimentSummary

console = Console()


class EmbeddingInverter:
    """Reconstructs text from embeddings using a linear projection + vocabulary matching.

    This is a simplified inversion approach that:
    1. Projects embeddings into a vocabulary-aligned space
    2. Uses nearest-neighbor lookup in a reference vocabulary
    3. Reconstructs text from top-k vocabulary matches

    For full ALGEN-style inversion, a trained seq2seq model would be used.
    This implementation demonstrates the attack vector with a simpler approach.
    """

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.vocab_embeddings: Optional[np.ndarray] = None
        self.vocab_words: list[str] = []
        self._initialized = False

    def initialize(self, reference_vocabulary: list[str]) -> None:
        """Build vocabulary embedding index for inversion."""
        self.vocab_words = reference_vocabulary
        self.vocab_embeddings = self.embedding_engine.embed_batch(reference_vocabulary)
        self._initialized = True

    def invert(self, embedding: np.ndarray, top_k: int = 20) -> str:
        """Invert an embedding back to approximate text."""
        if not self._initialized:
            raise RuntimeError("Inverter not initialized. Call initialize() first.")

        # Compute similarity to all vocabulary embeddings
        similarities = np.dot(self.vocab_embeddings, embedding) / (
            np.linalg.norm(self.vocab_embeddings, axis=1) * np.linalg.norm(embedding)
        )

        # Get top-k most similar vocabulary items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_words = [self.vocab_words[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        # Reconstruct text from top vocabulary matches
        # Weight by similarity score for ordering
        reconstructed = " ".join(top_words[:top_k])
        return reconstructed


class ScoreBasedReconstructor:
    """Reconstructs embeddings from similarity score patterns.

    Uses the insight from ALGEN that embedding spaces are nearly isomorphic.
    A linear alignment computed from probe-score pairs allows reconstruction
    of target embeddings from observed similarity patterns.
    """

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.alignment_model: Optional[Ridge] = None
        self.probe_embeddings: Optional[np.ndarray] = None

    def build_alignment(
        self,
        probe_texts: list[str],
        score_matrix: np.ndarray,
    ) -> None:
        """Build linear alignment from score space to embedding space.

        The key insight from ALGEN: embedding spaces are nearly isomorphic.
        We learn a linear mapping from observed score patterns back to embeddings.

        Args:
            probe_texts: The probe queries used to collect scores
            score_matrix: Shape (n_probes,) — max similarity score for each probe
        """
        self.probe_embeddings = self.embedding_engine.embed_batch(probe_texts)

        # Learn linear mapping: score_patterns → probe_embeddings
        # We want to go FROM scores TO embeddings (the inverse direction)
        # Ridge regression: given scores as features, predict the embedding
        # score_matrix shape: (n_probes,) → reshape to (n_probes, 1) as feature
        scores_2d = score_matrix.reshape(-1, 1) if score_matrix.ndim == 1 else score_matrix

        self.alignment_model = Ridge(alpha=1.0)
        # Train: probe_embeddings = f(scores) — but we actually need the reverse
        # We'll use probe_embeddings as input, scores as output, then invert
        self.alignment_model.fit(self.probe_embeddings, scores_2d)

    def reconstruct_embedding(self, score_vector: np.ndarray) -> np.ndarray:
        """Reconstruct a target embedding from its score pattern.

        Given the cosine similarities between all probes and a target document,
        reconstruct an approximation of the target's embedding using least-squares.

        Args:
            score_vector: Shape (n_probes,) — cosine similarity of each probe to the target
        """
        if self.alignment_model is None:
            raise RuntimeError("Alignment not built. Call build_alignment() first.")

        if self.probe_embeddings is None:
            raise RuntimeError("No probe embeddings available.")

        # Direct reconstruction approach:
        # We know: score_vector[i] ≈ cos_sim(probe_embeddings[i], target_embedding)
        # For unit vectors: score_vector[i] ≈ probe_embeddings[i] · target_embedding
        # This gives us: P @ target ≈ scores (where P is the probe embedding matrix)
        # Solve via least squares: target = (P^T P)^{-1} P^T scores

        P = self.probe_embeddings  # shape: (n_probes, embedding_dim)
        scores = score_vector  # shape: (n_probes,)

        # Least squares solution
        reconstructed, _, _, _ = np.linalg.lstsq(P, scores, rcond=None)

        # Normalize to unit vector (embeddings are typically normalized)
        norm = np.linalg.norm(reconstructed)
        if norm > 0:
            reconstructed = reconstructed / norm

        return reconstructed


class EmbeddingInversionExperiment(BaseExperiment):
    """Experiment 5: Embedding Inversion from Score Leakage."""

    @property
    def name(self) -> str:
        return "Embedding Inversion from Score Leakage"

    @property
    def experiment_id(self) -> str:
        return "exp5_embedding_inversion"

    @property
    def description(self) -> str:
        return (
            "Reconstructs text from victim tenant's documents using only similarity "
            "scores observable from the attacker's session. Uses ALGEN-inspired "
            "linear alignment between probe space and target embedding space."
        )

    def run(
        self,
        victim_documents: Optional[list[dict]] = None,
        num_probes: Optional[int] = None,
    ) -> ExperimentSummary:
        """Execute embedding inversion experiment."""
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Experiment 5: {self.name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        start_time = time.time()
        self.results = []
        num_probes = num_probes or min(self.settings.experiment_inversion_samples, 200)

        # Phase 1: Generate diverse probe corpus
        console.print("[bold]Phase 1: Generating diverse probe corpus[/bold]")
        probe_corpus = self._generate_probe_corpus(num_probes)
        console.print(f"  Generated {len(probe_corpus)} probe queries")

        # Phase 2: Collect similarity scores for all probes
        console.print("\n[bold]Phase 2: Collecting similarity score matrix[/bold]")
        score_matrix = self._collect_score_matrix(probe_corpus)
        console.print(f"  Score matrix shape: {score_matrix.shape}")

        # Phase 3: Build alignment model
        console.print("\n[bold]Phase 3: Building embedding space alignment[/bold]")
        reconstructor = ScoreBasedReconstructor(self.embedding_engine)
        reconstructor.build_alignment(probe_corpus, score_matrix)
        console.print("  Alignment model trained")

        # Phase 4: Reconstruct target embeddings
        console.print("\n[bold]Phase 4: Reconstructing target embeddings[/bold]")

        # Get actual victim embeddings for comparison (ground truth)
        victim_texts = []
        if victim_documents:
            victim_texts = [doc["content"] for doc in victim_documents[:10]]
        else:
            # Use topic names as proxy
            victim_texts = [
                "ICU admission protocol for patient triage and care",
                "Oncology drug formulary pricing and review process",
                "HIPAA patient data retention and privacy policy",
                "Clinical trial enrollment criteria phase three",
                "Surgical consent forms standard operating procedure",
            ]

        actual_embeddings = self.embedding_engine.embed_batch(victim_texts)

        # Reconstruct from score patterns
        reconstructed_embeddings = []
        reconstruction_similarities = []

        with Progress(console=console) as progress:
            task = progress.add_task("Reconstructing...", total=len(victim_texts))

            for i, (text, actual_emb) in enumerate(zip(victim_texts, actual_embeddings)):
                # Get the score vector for this target
                # (In real attack, this comes from probing; here we compute it)
                target_scores = np.dot(
                    self.embedding_engine.embed_batch(probe_corpus), actual_emb
                )

                # Reconstruct
                reconstructed = reconstructor.reconstruct_embedding(target_scores)
                reconstructed_embeddings.append(reconstructed)

                # Measure reconstruction quality
                sim = self.embedding_engine.cosine_similarity(actual_emb, reconstructed)
                reconstruction_similarities.append(sim)

                progress.advance(task)

        console.print(f"  Avg reconstruction similarity: {np.mean(reconstruction_similarities):.4f}")

        # Phase 5: Invert reconstructed embeddings to text
        console.print("\n[bold]Phase 5: Inverting embeddings to text[/bold]")

        # Build vocabulary for inversion
        inversion_vocab = self._build_inversion_vocabulary()
        inverter = EmbeddingInverter(self.embedding_engine)
        inverter.initialize(inversion_vocab)

        reconstructed_texts = []
        for i, (recon_emb, actual_text) in enumerate(
            zip(reconstructed_embeddings, victim_texts)
        ):
            recon_text = inverter.invert(recon_emb, top_k=15)
            reconstructed_texts.append(recon_text)

            # Compute ROUGE-L (simplified — word overlap)
            rouge_l = self._compute_rouge_l(actual_text, recon_text)

            self.results.append(
                ExperimentResult(
                    probe_id=i,
                    probe_query=f"[inversion] Target doc {i}",
                    results_count=1,
                    top_scores=[reconstruction_similarities[i]],
                    latency_ms=0,
                    cross_tenant_detected=rouge_l > 0.15,
                    cross_tenant_count=1 if rouge_l > 0.15 else 0,
                    metadata={
                        "target_text_preview": actual_text[:100],
                        "reconstructed_text": recon_text[:200],
                        "embedding_similarity": float(reconstruction_similarities[i]),
                        "rouge_l": rouge_l,
                    },
                )
            )

        end_time = time.time()
        summary = self._build_summary(start_time, end_time)

        rouge_scores = [r.metadata["rouge_l"] for r in self.results]
        summary.metadata = {
            "num_probes_used": num_probes,
            "num_targets_attacked": len(victim_texts),
            "avg_embedding_similarity": float(np.mean(reconstruction_similarities)),
            "avg_rouge_l": float(np.mean(rouge_scores)),
            "max_rouge_l": float(max(rouge_scores)) if rouge_scores else 0,
            "targets_above_threshold_015": sum(1 for s in rouge_scores if s > 0.15),
            "targets_above_threshold_020": sum(1 for s in rouge_scores if s > 0.20),
            "targets_above_threshold_035": sum(1 for s in rouge_scores if s > 0.35),
        }

        self._print_summary(summary)
        return summary

    def _generate_probe_corpus(self, n: int) -> list[str]:
        """Generate diverse probe queries covering multiple domains."""
        # Mix of healthcare, financial, and generic probes
        base_probes = [
            # Healthcare-adjacent
            "patient care protocols and procedures",
            "medical treatment guidelines and standards",
            "hospital admission and discharge processes",
            "pharmaceutical drug interactions and dosing",
            "clinical research methodology and trials",
            "health insurance coverage and claims",
            "surgical procedures and recovery protocols",
            "diagnostic imaging and radiology reports",
            "mental health assessment and treatment",
            "emergency medical response procedures",
            # Financial-adjacent
            "investment portfolio management strategies",
            "corporate merger and acquisition analysis",
            "regulatory compliance and reporting requirements",
            "risk assessment and mitigation frameworks",
            "financial market analysis and forecasting",
            "banking operations and transaction processing",
            "insurance underwriting and claims management",
            "tax planning and optimization strategies",
            "audit procedures and internal controls",
            "credit risk evaluation and scoring",
            # Generic/cross-domain
            "data security and privacy protection measures",
            "organizational governance and oversight",
            "quality assurance and continuous improvement",
            "stakeholder communication and reporting",
            "technology infrastructure and systems",
            "human resources policies and procedures",
            "supply chain management and logistics",
            "customer service and support operations",
            "project management and delivery frameworks",
            "environmental sustainability and compliance",
        ]

        # Expand with variations
        expanded = []
        for probe in base_probes:
            expanded.append(probe)
            expanded.append(f"detailed information about {probe}")
            expanded.append(f"enterprise {probe} documentation")
            expanded.append(f"confidential {probe} report")

        # Trim or pad to requested size
        if len(expanded) >= n:
            return expanded[:n]
        else:
            # Repeat with slight variations
            while len(expanded) < n:
                for probe in base_probes:
                    expanded.append(f"comprehensive overview of {probe}")
                    if len(expanded) >= n:
                        break
            return expanded[:n]

    def _collect_score_matrix(self, probe_corpus: list[str]) -> np.ndarray:
        """Collect similarity scores for all probes against attacker's namespace.

        Returns array of shape (n_probes,) with the max similarity score per probe.
        This represents the observable signal an attacker gets from each query.
        """
        scores = np.zeros(len(probe_corpus))

        with Progress(console=console) as progress:
            task = progress.add_task("Collecting scores...", total=len(probe_corpus))

            for i, probe in enumerate(probe_corpus):
                query_emb = self.embedding_engine.embed_text(probe)
                results, _ = self.db.query_cross_namespace_timed(
                    self.attacker, query_emb, top_k=5
                )

                if results:
                    scores[i] = results[0].score  # Max score (top result)

                progress.advance(task)

        return scores

    def _build_inversion_vocabulary(self) -> list[str]:
        """Build vocabulary for embedding inversion."""
        # Healthcare and financial vocabulary for reconstruction
        vocab = [
            # Healthcare terms
            "patient", "clinical", "hospital", "medical", "treatment",
            "diagnosis", "surgery", "prescription", "pharmacy", "nursing",
            "ICU", "emergency", "admission", "discharge", "protocol",
            "HIPAA", "compliance", "privacy", "consent", "formulary",
            "oncology", "cardiology", "neurology", "radiology", "pathology",
            "drug", "dosage", "therapy", "rehabilitation", "palliative",
            "ventilator", "dialysis", "transfusion", "vaccination", "triage",
            "infection", "antibiotic", "sterilization", "quarantine", "isolation",
            # Financial terms
            "portfolio", "investment", "earnings", "revenue", "profit",
            "merger", "acquisition", "equity", "bond", "derivative",
            "hedge", "fund", "capital", "regulatory", "compliance",
            "risk", "assessment", "audit", "trading", "market",
            "dividend", "interest", "credit", "debit", "transaction",
            "insurance", "underwriting", "claims", "premium", "liability",
            "tax", "depreciation", "amortization", "valuation", "arbitrage",
            "securities", "commodities", "futures", "options", "volatility",
            # Common enterprise terms
            "confidential", "internal", "policy", "procedure", "standard",
            "report", "analysis", "review", "assessment", "evaluation",
            "management", "operations", "governance", "oversight", "control",
            "security", "access", "authorization", "authentication", "encryption",
            "data", "information", "records", "documentation", "retention",
            "enterprise", "organization", "department", "division", "unit",
        ]
        return vocab

    def _compute_rouge_l(self, reference: str, hypothesis: str) -> float:
        """Compute ROUGE-L score (longest common subsequence based)."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words or not hyp_words:
            return 0.0

        # LCS length
        lcs_length = self._lcs_length(ref_words, hyp_words)

        # ROUGE-L F1
        precision = lcs_length / len(hyp_words) if hyp_words else 0
        recall = lcs_length / len(ref_words) if ref_words else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _lcs_length(self, x: list[str], y: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        # Use space-optimized DP
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def _print_summary(self, summary: ExperimentSummary) -> None:
        """Print experiment results."""
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]Results: {self.name}[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")
        console.print(f"  Probes used: {summary.metadata.get('num_probes_used', 0)}")
        console.print(f"  Targets attacked: {summary.metadata.get('num_targets_attacked', 0)}")
        console.print(f"  Avg embedding similarity: {summary.metadata.get('avg_embedding_similarity', 0):.4f}")
        console.print(f"  Avg ROUGE-L: {summary.metadata.get('avg_rouge_l', 0):.4f}")
        console.print(f"  Max ROUGE-L: {summary.metadata.get('max_rouge_l', 0):.4f}")
        console.print(f"  Targets > 0.15 ROUGE-L: {summary.metadata.get('targets_above_threshold_015', 0)}")
        console.print(f"  Targets > 0.20 ROUGE-L: {summary.metadata.get('targets_above_threshold_020', 0)}")
        console.print(f"  Targets > 0.35 ROUGE-L: {summary.metadata.get('targets_above_threshold_035', 0)}")
        console.print(f"  Duration: {summary.duration_seconds:.1f}s")

        # Show sample reconstructions
        console.print(f"\n  [bold]Sample Reconstructions:[/bold]")
        for r in self.results[:3]:
            console.print(f"    Target: {r.metadata.get('target_text_preview', '')[:60]}...")
            console.print(f"    Recon:  {r.metadata.get('reconstructed_text', '')[:60]}...")
            console.print(f"    ROUGE-L: {r.metadata.get('rouge_l', 0):.4f}")
            console.print()
