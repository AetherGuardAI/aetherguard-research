# VectorBleed

**Cross-Tenant Data Leakage Through Embedding Space Proximity in Shared RAG Infrastructure**

The first empirical security audit of tenant isolation in production vector databases under adversarial conditions.

## Research Questions

1. Can a malicious tenant craft queries that exploit embedding space geometry to surface content from another tenant's namespace?
2. Can embedding inversion techniques reconstruct private documents from partially leaked vector representations?
3. Are there implementation-level misconfigurations in RAG frameworks (LangChain, LlamaIndex) that bypass namespace isolation?
4. Does the attack surface differ across isolation models (Pinecone namespaces, Weaviate multi-tenancy, Qdrant payload filters, ChromaDB collections)?

## Attack Vectors

| # | Attack | Description |
|---|--------|-------------|
| 1 | Proximity Probing | Craft queries targeting victim's content domain; observe score/latency anomalies |
| 2 | Centroid Injection | Inject documents at embedding space centroid to cross namespace boundaries |
| 3 | Score Side-Channel | Use similarity scores to reconstruct victim's topic clusters |
| 4 | Framework Misconfiguration | Audit LangChain, LlamaIndex, and native SDK retrieval patterns that bypass DB-level isolation |
| 5 | Embedding Inversion | Reconstruct victim text from score patterns using linear alignment |

## Supported Databases

| Database | Isolation Model | Adapter |
|----------|----------------|---------|
| Pinecone | Namespace-based | `pinecone_db.py` |
| Weaviate | Shard-per-tenant multi-tenancy | `weaviate_db.py` |
| Qdrant | Payload-filter isolation | `qdrant_db.py` |
| ChromaDB | Collection + metadata filter | `chroma_db.py` |

All five attack experiments run against each database, producing a comparative isolation scorecard.

## Quick Start

```bash
cd vectorbleed
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys (OpenAI + database credentials)

vectorbleed run-all
```

See [SETUP_AND_RUN.md](SETUP_AND_RUN.md) for detailed instructions.

## CLI Commands

```
vectorbleed generate-corpus              # Generate synthetic tenant documents via GPT-4
vectorbleed setup-db                     # Initialize Pinecone and upload tenant documents
vectorbleed run-experiments              # Run attack experiments (1-5 or all) against Pinecone
vectorbleed run-other-dbs --db all       # Run experiments against Weaviate, Qdrant, ChromaDB
vectorbleed run-defenses                 # Test defense mitigations
vectorbleed visualize                    # Generate embedding space plots (PCA + t-SNE)
vectorbleed generate-report --db comparative  # Produce comparative scorecard across all DBs
vectorbleed run-all                      # Complete pipeline (corpus → setup → experiments → defenses → viz → report)
vectorbleed cleanup --db all             # Delete experiment data from vector databases
```

### Key Options

- `run-experiments -e <1-5|all>` — Run specific or all experiments
- `run-other-dbs --db <pinecone|weaviate|qdrant|chroma|all>` — Target specific databases
- `generate-report --db <pinecone|weaviate|qdrant|chroma|all|comparative>` — Report scope
- `cleanup --db <pinecone|weaviate|qdrant|chroma|all>` — Selective cleanup

## Architecture

```
vectorbleed/
├── src/vectorbleed/
│   ├── cli.py                  # CLI entry point (Click)
│   ├── config.py               # Settings via pydantic-settings (VECTORBLEED_ env prefix)
│   ├── embeddings.py           # OpenAI embedding wrapper
│   ├── corpus/                 # GPT-4o-mini document generation
│   ├── databases/              # Vector DB adapters (Pinecone, Weaviate, Qdrant, ChromaDB)
│   ├── experiments/            # 5 attack experiments
│   ├── defenses/               # 5 mitigation tests
│   ├── analysis/               # Metrics, visualization, isolation scorecard
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

## Requirements

- Python 3.11–3.12
- OpenAI API key (embeddings + corpus generation)
- At least one vector database configured (Pinecone, Weaviate Cloud, Qdrant Cloud, or ChromaDB)

## License

Research use only. See AetherGuard Research terms.
