# VectorBleed

**Cross-Tenant Data Leakage Through Embedding Space Proximity in Shared RAG Infrastructure**

The first empirical security audit of tenant isolation in production vector databases under adversarial conditions.

## Research Questions

1. Can a malicious tenant craft queries that exploit embedding space geometry to surface content from another tenant's namespace?
2. Can embedding inversion techniques reconstruct private documents from partially leaked vector representations?
3. Are there implementation-level misconfigurations in LangChain that bypass namespace isolation?
4. Does the attack surface differ across isolation models (Pinecone namespaces vs others)?

## Attack Vectors

| # | Attack | Description |
|---|--------|-------------|
| 1 | Proximity Probing | Craft queries targeting victim's content domain; observe score/latency anomalies |
| 2 | Centroid Injection | Inject documents at embedding space centroid to cross namespace boundaries |
| 3 | Score Side-Channel | Use similarity scores to reconstruct victim's topic clusters |
| 4 | Framework Misconfiguration | Audit LangChain default patterns that bypass DB-level isolation |
| 5 | Embedding Inversion | Reconstruct victim text from score patterns using linear alignment |

## Quick Start

```bash
cd vectorbleed
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your OpenAI + Pinecone API keys

vectorbleed run-all
```

See [SETUP_AND_RUN.md](SETUP_AND_RUN.md) for detailed instructions.

## CLI Commands

```
vectorbleed generate-corpus    # Generate synthetic tenant documents
vectorbleed setup-db           # Initialize Pinecone multi-tenant setup
vectorbleed run-experiments    # Run attack experiments (1-5 or all)
vectorbleed run-defenses       # Test defense mitigations
vectorbleed visualize          # Generate embedding space plots
vectorbleed generate-report    # Produce JSON/CSV/Markdown reports
vectorbleed run-all            # Complete pipeline
vectorbleed cleanup            # Delete vectors from Pinecone
```

## Architecture

```
vectorbleed/
├── src/vectorbleed/
│   ├── cli.py                  # CLI entry point
│   ├── config.py               # Settings (env vars)
│   ├── embeddings.py           # OpenAI embedding wrapper
│   ├── corpus/                 # GPT-4 document generation
│   ├── databases/              # Vector DB adapters (Pinecone)
│   ├── experiments/            # 5 attack experiments
│   ├── defenses/               # 5 mitigation tests
│   ├── analysis/               # Metrics, visualization, scorecard
│   └── reporting/              # JSON/CSV/Markdown reports
├── results/                    # Experiment outputs (gitignored)
├── notebooks/                  # Jupyter analysis
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

## Defense Mitigations Tested

| # | Mitigation | Blocks | Cost |
|---|-----------|--------|------|
| 1 | Physical Isolation (separate indexes) | Exp 1,2,3,5 | 10x |
| 2 | Differential Privacy (noise on embeddings) | Exp 1,2,3 | 1x |
| 3 | Score Suppression (no scores returned) | Exp 3,5 | 1x |
| 4 | Cryptographic Namespace (AetherGuard) | All 5 | 1.2x |
| 5 | Query Attestation (signed tokens) | Exp 1,2,3 | 1.1x |

## License

Research use only. See AetherGuard Research terms.
