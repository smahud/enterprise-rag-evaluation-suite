# enterprise-rag-evaluation-suite

Benchmarking and synthetic evaluation harness for semantic search, reranking strategies, and context hallucination mitigation.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  Datasets   │────▶│  Retrieval   │────▶│  Reranking  │────▶│  Generation │
│  (Queries,  │     │  (Embedding, │     │  (Cross-    │     │  (LLM +     │
│   Corpus,   │     │   BM25,      │     │   Encoder,   │     │   Prompts)  │
│   Gold)     │     │   Hybrid)    │     │   LLM-based) │     │             │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                  ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
                  │  Metrics     │     │  Metrics    │     │  Metrics    │
                  │  (Recall@k,  │     │  (MRR,      │     │  (Faith-    │
                  │   nDCG,      │     │   nDCG@k)   │     │   fulness,  │
                  │   Latency)   │     │             │     │   Answer    │
                  └──────────────┘     └─────────────┘     │   Relevance,│
                                                         │   Halluc.   │
                                                         └─────────────┘
```

## Evaluation Dimensions

| Dimension | Metrics | Purpose |
|-----------|---------|---------|
| **Retrieval** | Recall@k, nDCG@k, MAP, Latency(p50/p99) | Find relevant docs |
| **Reranking** | MRR, nDCG@k, Precision@k | Improve ranking quality |
| **Generation** | Faithfulness, Answer Relevance, Hallucination Rate, Citation Accuracy | LLM output quality |
| **End-to-End** | RAGAS Score, Token Efficiency, Cost/Query | System-level health |

## Built-in Datasets

| Dataset | Domain | Size | Gold Standard |
|---------|--------|------|---------------|
| `msmarco` | Web search | 8.8M passages | Human judgments |
| `nq` | Open QA | 300k questions | Wikipedia pages |
| `hotpotqa` | Multi-hop | 113k questions | Supporting facts |
| `fever` | Fact verification | 185k claims | Evidence sentences |
| `custom` | Your data | Any | Your annotations |

## Reranking Strategies Tested

- **Cross-Encoder** (bge-reranker, ms-marco-MiniLM)
- **LLM-based** (GPT-4, Llama-3 as judge)
- **Hybrid** (BM25 + Dense + Reciprocal Rank Fusion)
- **Adaptive** (Query-type aware routing)

## Hallucination Detection

- **Self-consistency**: Multiple generations → consensus check
- **Citation verification**: Extract citations → verify against source
- **Entailment scoring**: NLI model (DeBERTa) for premise/hypothesis
- **Token probability**: Low-prob spans flagging

## Quick Start

```bash
# Install
pip install -e .

# Run full evaluation
rag-eval run \
  --dataset msmarco \
  --retriever bge-large-en-v1.5 \
  --reranker bge-reranker-large \
  --generator gpt-4o-mini \
  --output results/

# Generate report
rag-eval report results/ --format html
```

## Configuration

```yaml
retrieval:
  embedder: "BAAI/bge-large-en-v1.5"
  top_k: 50
  index_type: "hnsw"

reranking:
  model: "BAAI/bge-reranker-large"
  top_k: 10

generation:
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 512

evaluation:
  metrics: ["recall", "ndcg", "mrr", "faithfulness", "hallucination_rate"]
  sample_size: 1000
```

## Topics

`rag` `vector-search` `qdrant` `evals` `python` `llm-evaluation` `retrieval` `reranking` `hallucination-detection`