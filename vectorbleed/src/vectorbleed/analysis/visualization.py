"""Embedding space visualization for VectorBleed experiments."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embedding_space(
    tenant_a_embeddings: np.ndarray,
    tenant_b_embeddings: np.ndarray,
    output_path: Optional[Path] = None,
    method: str = "pca",
    title: str = "Multi-Tenant Embedding Space",
) -> None:
    """Visualize the embedding space of two tenants in 2D.

    Args:
        tenant_a_embeddings: Embeddings for tenant A (financial)
        tenant_b_embeddings: Embeddings for tenant B (healthcare)
        output_path: Path to save the plot
        method: Dimensionality reduction method ("pca" or "tsne")
        title: Plot title
    """
    all_embeddings = np.vstack([tenant_a_embeddings, tenant_b_embeddings])
    n_a = len(tenant_a_embeddings)

    # Reduce to 2D
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
    else:
        reducer = PCA(n_components=2)

    coords_2d = reducer.fit_transform(all_embeddings)

    # Split back
    coords_a = coords_2d[:n_a]
    coords_b = coords_2d[n_a:]

    # Compute centroids
    centroid_a = np.mean(coords_a, axis=0)
    centroid_b = np.mean(coords_b, axis=0)
    global_centroid = np.mean(coords_2d, axis=0)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.scatter(coords_a[:, 0], coords_a[:, 1], c="blue", alpha=0.6, label="Tenant A (Financial)", s=50)
    ax.scatter(coords_b[:, 0], coords_b[:, 1], c="red", alpha=0.6, label="Tenant B (Healthcare)", s=50)

    # Centroids
    ax.scatter(*centroid_a, c="blue", marker="X", s=200, edgecolors="black", linewidths=2, label="Centroid A")
    ax.scatter(*centroid_b, c="red", marker="X", s=200, edgecolors="black", linewidths=2, label="Centroid B")
    ax.scatter(*global_centroid, c="purple", marker="*", s=300, edgecolors="black", linewidths=2, label="Global Centroid")

    # Draw boundary region
    midpoint = (centroid_a + centroid_b) / 2
    ax.axhline(y=midpoint[1], color="gray", linestyle="--", alpha=0.3)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_score_distribution(
    same_tenant_scores: list[float],
    cross_tenant_scores: list[float],
    output_path: Optional[Path] = None,
    title: str = "Similarity Score Distribution",
) -> None:
    """Plot distribution of similarity scores for same-tenant vs cross-tenant queries.

    Args:
        same_tenant_scores: Scores from queries matching own tenant
        cross_tenant_scores: Scores from queries targeting other tenant
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if same_tenant_scores:
        ax.hist(same_tenant_scores, bins=30, alpha=0.6, color="green", label="Same-Tenant Queries")
    if cross_tenant_scores:
        ax.hist(cross_tenant_scores, bins=30, alpha=0.6, color="red", label="Cross-Tenant Queries")

    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_latency_comparison(
    baseline_latencies: list[float],
    cross_tenant_latencies: list[float],
    output_path: Optional[Path] = None,
    title: str = "Query Latency: Baseline vs Cross-Tenant Probes",
) -> None:
    """Plot latency comparison between baseline and cross-tenant queries.

    Args:
        baseline_latencies: Latencies for same-tenant queries (ms)
        cross_tenant_latencies: Latencies for cross-tenant probes (ms)
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    axes[0].boxplot(
        [baseline_latencies, cross_tenant_latencies],
        labels=["Baseline", "Cross-Tenant"],
    )
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency Distribution")
    axes[0].grid(True, alpha=0.3)

    # Time series
    axes[1].plot(baseline_latencies, "g-", alpha=0.7, label="Baseline")
    axes[1].plot(cross_tenant_latencies, "r-", alpha=0.7, label="Cross-Tenant")
    axes[1].set_xlabel("Query Index")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
