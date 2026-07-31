# Semantic Art Search Engine — Evaluation Pipeline

Evaluation script that benchmarks and compares two ChromaDB embedding collections using two complementary query sources: fixed Ground Truth queries and VLM-generated queries. Results are exported to three CSV files for downstream analysis.

> Built for the **Modelling Week MS 2026** project (Management Solutions).

---

## Overview

```
directed_queries.json  ──► GT queries (fixed, reproducible)
full_result.json       ──► VLM queries (LLaVA × 4 prompt strategies)
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ChromaDB A (mpnet)      ChromaDB B (CLIP)
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    Precision@K · Recall@K · MRR
                    nDCG@K · Hit Rate@K · MAP
                    exact_rank · exact_hit_at_k
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        gt_results.csv   gt_summary.csv   gt_compare.csv
```

| Output file | Content |
|-------------|---------|
| `gt_results.csv` | One row per query × embedding with all metric values |
| `gt_summary.csv` | Aggregated metrics: global, by source, by query type, by category |
| `gt_compare.csv` | Direct A vs B comparison with per-metric delta (A − B) |

---

## Query Sources

### Ground Truth (GT)
Fixed queries loaded from `directed_queries.json`. Each entry maps an artwork ID to a hand-crafted search query and a query type label. GT queries are fully reproducible — no model inference is involved. The exact query image is always counted as relevant; if `full_result.json` is provided, same-artist and same-tag artworks are added as soft positives.

### VLM Queries
A random sample of artworks is passed to LLaVA via Ollama, which generates one query per image per prompt strategy. Four strategies are used:

| Strategy | Description |
|----------|-------------|
| `short` | 1–2 word query focused on the most distinctive visual element |
| `long_specific` | 2–3 sentence detailed description of subject, composition, and mood |
| `style` | One sentence focused exclusively on artistic style and technique |
| `era` | One sentence focused exclusively on historical period |

For VLM queries, only same-artist and same-tag artworks are counted as relevant — the query image itself is excluded to avoid trivial exact-match retrieval.

---

## Dynamic K (Kneedle Algorithm)

Rather than using a fixed evaluation cutoff, the script detects the elbow of the cosine distance curve for each query result set and uses that position as K. This adapts the evaluation to the actual score distribution of each query, avoiding over-counting near-random results in long tails. K is subject to a configurable minimum (`k_min`).

---

## Metrics

All metrics are computed at the dynamic K determined per query:

| Metric | Description |
|--------|-------------|
| `Precision@K` | Fraction of top-K results that are relevant |
| `Recall@K` | Fraction of all relevant artworks retrieved in top K |
| `MRR` | Reciprocal rank of the first relevant result |
| `nDCG@K` | Normalised Discounted Cumulative Gain (binary relevance) |
| `Hit Rate@K` | 1 if at least one relevant result appears in top K |
| `MAP` | Mean Average Precision |
| `exact_rank` | 1-indexed position of the query image in the result list |
| `exact_hit_at_k` | 1 if the query image appears within the top K |

---

## Requirements

```bash
pip install chromadb sentence-transformers tqdm numpy ollama Pillow
```

Ollama must be running with LLaVA available (required only if `vlm_sample > 0`):

```bash
ollama serve          # keep running in a separate terminal
ollama pull llava
```

---

## Input Files

### `directed_queries.json`
A JSON array where each entry must contain at minimum:

```json
[
  {
    "image_id":   "SK-A-1234",
    "query":      "A gloomy room lit by a candle",
    "query_type": "scene"
  }
]
```

Optional fields: `category`, `notes`.

### `full_result.json`
The merged VLM output JSON produced after combining all machines' `description.json` files. Used for two purposes: VLM query generation and enriching GT relevance sets with same-artist / same-tag artworks. Set `--full-json none` to disable and fall back to exact-hit-only relevance.

---

## Configuration

All parameters can be set via command-line arguments or by editing `DEFAULT_CONFIG` at the top of the script.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gt-json` | `directed_queries.json` | Ground truth query file |
| `--full-json` | `full_result.json` | Full metadata JSON (`none` to disable) |
| `--images-dir` | `./rijksmuseum` | Root image folder (`images_dir/category/filename`) |
| `--vlm-sample` | `20` | Images per VLM strategy (`0` to disable VLM) |
| `--vlm-model` | `llava` | Ollama model tag |
| `--seed` | `42` | Random seed for reproducible VLM sampling |
| `--category` | `None` | Category filter for VLM sampling (e.g. `painting`) |
| `--min-relevant` | `2` | Minimum relevant artworks required for VLM eligibility |
| `--chroma-path-a` | `./vector_database_1_full` | ChromaDB path for embedding A |
| `--collection-a` | `rijksmuseum_llava` | Collection name for embedding A |
| `--embed-model-a` | `sentence-transformers/all-mpnet-base-v2` | Encoder for embedding A |
| `--label-a` | `A` | Label used in output columns for embedding A |
| `--chroma-path-b` | `./vector_database_2_full` | ChromaDB path for embedding B |
| `--collection-b` | `rijksmuseum_artworks` | Collection name for embedding B |
| `--embed-model-b` | `openai/clip-vit-base-patch32` | Encoder for embedding B |
| `--label-b` | `B` | Label used in output columns for embedding B |
| `--pool-size` | `300` | Candidate pool size per query |
| `--k-min` | `3` | Minimum dynamic K value |
| `--output-csv` | `./gt_results.csv` | Detailed results output path |
| `--summary-csv` | `./gt_summary.csv` | Summary output path |
| `--compare-csv` | `./gt_compare.csv` | Comparison table output path |

---

## Usage

### Default configuration

```bash
python evaluation.py
```

### Custom parameters

```bash
python evaluation.py \
    --gt-json       directed_queries.json \
    --full-json     full_result.json \
    --vlm-sample    20 \
    --label-a       mpnet \
    --label-b       clip \
    --pool-size     300 \
    --k-min         3 \
    --output-csv    ./gt_results.csv \
    --summary-csv   ./gt_summary.csv \
    --compare-csv   ./gt_compare.csv
```

### GT only (no VLM)

```bash
python evaluation.py --vlm-sample 0
```

### Disable full metadata (exact hit relevance only)

```bash
python evaluation.py --full-json none
```

---

## Output Structure

```
.
├── evaluation.py
├── directed_queries.json     ← GT query file (input)
├── full_result.json          ← merged VLM metadata (input)
├── gt_results.csv            ← one row per query × embedding
├── gt_summary.csv            ← aggregated metrics per embedding
└── gt_compare.csv            ← A vs B delta comparison
```

### `gt_results.csv` columns

`embedding`, `source`, `image_id`, `file`, `category`, `query_type`, `query`, `n_relevant`, `k_final`, `k_kneedle`, `dist_at_k`, `exact_rank`, `exact_hit_at_k`, `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`, `hit_rate_at_k`, `ap_at_k`, `latency_vlm_s`, `latency_chroma_s`

### `gt_summary.csv` scope values

| Scope | Grouping |
|-------|---------|
| `GLOBAL` | All queries combined |
| `SOURCE` | Split by `gt` / `vlm` |
| `SOURCE_QTYPE` | Split by source × query type |
| `CATEGORY` | Split by artwork category |

### `gt_compare.csv`
One row per (scope, category) combination with metric values for A and B side by side, plus `delta_<metric> = A − B`. A positive delta means embedding A outperforms B on that metric.


