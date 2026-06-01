# VectorBleed — Setup and Run Guide

## Prerequisites

- Python 3.11 (LangChain requires <3.13)
- OpenAI API key (for embeddings + corpus generation)
- Pinecone API key (free tier works for research)
- ~$5-10 in OpenAI API credits (corpus generation + embeddings)

## Step 1: Create Virtual Environment (Python 3.11)

LangChain requires Python <3.13. Use the `py` launcher to target 3.11:

```bash
cd vectorbleed

# Create venv with Python 3.11
py -3.11 -m venv .venv

# Activate
.venv\Scripts\activate
```

Verify:
```bash
python --version
# Should show: Python 3.11.x
```

## Step 2: Install Dependencies

```bash
pip install -e .
```

This installs:
- **LangChain** + langchain-openai + langchain-pinecone (RAG framework)
- **Pinecone SDK** (vector database)
- **OpenAI SDK** (embeddings + corpus generation)
- **NumPy, SciPy, scikit-learn** (analysis)
- **Matplotlib, Seaborn, Pandas** (visualization + reporting)
- **Rich, Click** (CLI)

## Step 3: Configure Environment

```bash
# Copy the example env file
copy .env.example .env

# Edit .env with your API keys:
# - VECTORBLEED_OPENAI_API_KEY=sk-...
# - VECTORBLEED_PINECONE_API_KEY=pcsk_...
```

### Getting API Keys

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Set `VECTORBLEED_OPENAI_API_KEY` in .env

**Pinecone:**
1. Go to https://app.pinecone.io/
2. Sign up for free tier (sufficient for research)
3. Create an API key
4. Set `VECTORBLEED_PINECONE_API_KEY` in .env
5. Note: Free tier supports 1 index with up to 100K vectors — more than enough

## Step 4: Run the Complete Pipeline

### Option A: Run Everything at Once

```bash
python -m vectorbleed run-all
```

This executes the full pipeline:
1. Generates 100 synthetic documents (50 financial + 50 healthcare) via GPT-4
2. Creates Pinecone index and uploads documents to separate namespaces
3. Runs all 5 attack experiments using LangChain retrievers
4. Tests 5 defense mitigations
5. Generates embedding space visualizations
6. Produces research report (JSON + CSV + Markdown)

### Option B: Run Step by Step

```bash
# 1. Generate synthetic tenant documents
python -m vectorbleed generate-corpus

# 2. Setup Pinecone with multi-tenant namespaces
python -m vectorbleed setup-db

# 3. Run specific experiments
python -m vectorbleed run-experiments -e 1    # Proximity probing
python -m vectorbleed run-experiments -e 2    # Centroid injection
python -m vectorbleed run-experiments -e 3    # Side-channel
python -m vectorbleed run-experiments -e 4    # LangChain misconfiguration audit
python -m vectorbleed run-experiments -e 5    # Embedding inversion
python -m vectorbleed run-experiments -e all  # All experiments

# 4. Test defenses
python -m vectorbleed run-defenses

# 5. Generate visualizations
python -m vectorbleed visualize

# 6. Generate report
python -m vectorbleed generate-report
```

## Step 5: View Results

Results are saved to `results/`:

```
results/
├── corpus/
│   ├── tenant_a_financial.json     # Generated financial documents
│   ├── tenant_b_healthcare.json    # Generated healthcare documents
│   └── embeddings/
│       ├── tenant_a_embeddings.npy
│       └── tenant_b_embeddings.npy
├── experiments/
│   ├── exp1_proximity_probing.json
│   ├── exp2_centroid_injection.json
│   ├── exp3_score_sidechannel.json
│   ├── exp4_framework_misconfiguration.json
│   └── exp5_embedding_inversion.json
├── defenses/
│   └── defense_results.json
├── visualizations/
│   ├── embedding_space_pca.png
│   └── embedding_space_tsne.png
└── reports/
    ├── vectorbleed_report.json     # Full structured report
    ├── vectorbleed_summary.csv     # Summary table
    └── vectorbleed_report.md       # Markdown report
```

## Step 6: Jupyter Analysis (Optional)

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/
```

Or use Docker:
```bash
docker-compose up jupyter
# Access at http://localhost:8888 (token: vectorbleed)
```

## How LangChain Is Used

VectorBleed uses LangChain's `PineconeVectorStore` and retriever patterns — the same
abstractions enterprise RAG applications use. This makes findings directly applicable:

**Secure pattern (what apps SHOULD do):**
```python
from langchain_pinecone import PineconeVectorStore

vs = PineconeVectorStore(index=index, embedding=embeddings, namespace="tenant_a")
retriever = vs.as_retriever(search_kwargs={"k": 10, "filter": {"tenant_id": "tenant_a"}})
```

**Vulnerable pattern (what Experiment 4 tests):**
```python
vs = PineconeVectorStore(index=index, embedding=embeddings)  # NO namespace!
retriever = vs.as_retriever()  # Queries ALL tenants
```

## Cleanup

```bash
# Delete all vectors from Pinecone (preserves index)
python -m vectorbleed cleanup

# Delete the Pinecone index entirely (do this from Pinecone console)
```

## Estimated Costs

| Operation | Estimated Cost |
|-----------|---------------|
| Corpus generation (100 docs × GPT-4o-mini) | ~$0.50 |
| Embedding 100 documents | ~$0.01 |
| Embedding experiment probes (~500 queries) | ~$0.05 |
| Pinecone free tier | $0.00 |
| **Total** | **~$1-2** |

## Troubleshooting

**"Python version error" / LangChain won't install**
- LangChain requires Python <3.13. Use `py -3.11 -m venv .venv` to create the venv.

**"Pinecone index not ready"**
- Wait 30-60 seconds after creation. Serverless indexes take time to initialize.

**"OpenAI rate limit"**
- The tool has built-in retry with exponential backoff. If persistent, reduce `VECTORBLEED_CORPUS_SIZE_PER_TENANT`.

**"No results from queries"**
- Ensure `setup-db` completed successfully. Check Pinecone console for vector counts.

**"Import errors"**
- Make sure you're in the `.venv` (run `.venv\Scripts\activate`) and installed with `pip install -e .`

**"numpy/scipy binary mismatch"**
- If you see C-extension errors, run: `pip install --force-reinstall numpy scipy scikit-learn`

## Docker Alternative

```bash
# Build
docker build -t vectorbleed .

# Run full pipeline
docker run --env-file .env -v ${PWD}/results:/app/results vectorbleed run-all

# Run specific command
docker run --env-file .env -v ${PWD}/results:/app/results vectorbleed run-experiments -e 1
```


#Run all databases
python -m vectorbleed run-other-dbs --db all

#Generate report
python -m vectorbleed generate-report --db all

#Clean-up all databases
python -m vectorbleed cleanup --db all
