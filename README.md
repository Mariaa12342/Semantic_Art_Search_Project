# Semantic Art Search Engine — VLM Pipeline

A four-phase Retrieval-Augmented Generation (RAG) pipeline for semantic artwork retrieval over the [Rijksmuseum dataset](https://www.kaggle.com/datasets/richliao/imageNet). Instead of filtering by author or year, you can query artworks by concept, mood, scene, or visual content — entirely in natural language.

> Built for the **Modelling Week MS 2026** project (Management Solutions).

---

## Pipeline Overview

```
Images
  └─► Phase 1 ── LLaVA (VLM via Ollama) ──► description.json
        └─► Phase 2 ── Cross with CSV metadata ──► result.json
              └─► Phase 3 ── Sentence Transformer ──► ChromaDB
                    └─► Phase 4 ── Cosine similarity search
```

| Phase | Input | Output | Key component |
|-------|-------|--------|---------------|
| 1 – VLM Extraction | Images (JPG/PNG/…) | `description.json` | LLaVA via Ollama |
| 2 – Metadata Cross | `description.json` + CSV | `result.json` | pandas |
| 3 – Embedding Ingest | `result.json` | ChromaDB collection | `all-mpnet-base-v2` |
| 4 – Semantic Search | Free-text query | Ranked artwork list | ChromaDB + cosine similarity |

---

## How It Works

Each image is sent to **LLaVA**, a local vision-language model running via Ollama, which produces a structured JSON description covering general summary, dominant colours, mood, scene context, and semantic tags. That text is then enriched with the artwork's title and artist name from the CSV metadata, and encoded into a 768-dimensional vector using `all-mpnet-base-v2`. At search time, the free-text query is encoded with the same model and the most similar artworks are retrieved from ChromaDB via cosine similarity.

Because both the indexed documents and the queries are text, they live in the same embedding space — no cross-modal alignment is needed. The intermediate JSON files also make it straightforward to inspect what the model understood about each image and to diagnose retrieval failures.

---

## Requirements

### Python dependencies

```bash
pip install ollama chromadb sentence-transformers pandas
```

### Ollama + LLaVA

Install [Ollama](https://ollama.com/) and pull the model:

```bash
ollama pull llava
ollama serve          # keep this running in a separate terminal
```

> Developed and tested with an **NVIDIA RTX 2060**. LLaVA benefits significantly from GPU acceleration; CPU-only execution is possible but considerably slower.

---

## Dataset Layout

```
sample/
├── drawings/
│   ├── SK-A-1234.jpg
│   └── ...
├── paintings/
│   └── ...
├── photographs/
│   └── ...
├── prints/
│   └── ...
└── metadata_sample.csv      ← must contain at least: id, title, artist, date
```

The image filename stem (e.g. `SK-A-1234`) is used as the join key between images and the CSV metadata.

---

## Configuration

All tuneable parameters are grouped at the top of `semantic_art_search.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_PATH` | `"sample"` | Root folder containing category sub-folders |
| `OUTPUT_JSON` | `"description.json"` | Phase 1 output |
| `MODEL` | `"llava"` | Ollama model tag |
| `DELAY_SECONDS` | `1` | Pause between Ollama calls (reduces thermal throttling) |
| `OLLAMA_HOST` | `"http://localhost:11434"` | Ollama API endpoint |
| `MAX_RETRIES` | `3` | JSON parse retry attempts per image |
| `MAX_IMAGES_PER_CATEGORY` | `None` | Set an integer for quick test runs |
| `METADATA_CSV` | `"sample/metadata_sample.csv"` | Path to the metadata CSV |
| `CROSSED_JSON` | `"result.json"` | Phase 2 output |
| `CHROMA_PATH` | `"chroma_db"` | ChromaDB persistence directory |
| `MODEL_EMBED` | `"all-mpnet-base-v2"` | Sentence Transformer model (768 dims) |
| `COLLECTION_NAME` | `"rijksmuseum_llava"` | ChromaDB collection name |

---

## Running the Pipeline

### Full run (all four phases)

```bash
python semantic_art_search.py
```

### Resume an interrupted Phase 1

Phase 1 writes `description.json` incrementally after every image. If the run is interrupted, simply re-run the script — already-processed images are skipped automatically.

### Skip Phase 1 (use an existing `description.json`)

Comment out the `run_vlm_extraction()` call near the bottom of the file:

```python
# run_vlm_extraction()   # ← commented out
run_cross_data()
run_chroma_ingestion()
```

### Run a custom search query

Call `search_art` with any free-text query at the end of the script, or import it:

```python
from semantic_art_search import search_art
search_art("A stormy sea with sailing ships", n_results=5)
```

---

## Generated Artefacts

| File / Folder | Description |
|---------------|-------------|
| `description.json` | LLaVA analysis for every image, organised by category |
| `result.json` | Cross-referenced data: VLM descriptions + CSV metadata |
| `chroma_db/` | Persistent ChromaDB vector store (768-dim cosine index) |

---

## Text Representation

The string embedded for each artwork is assembled in `build_text_from_analysis()` in the following order:

```
{title}. by {artist}. {general_description}. {mood}. {scene_context}. {tags}
```

Title and artist are placed first to give them the highest semantic weight in the embedding, improving retrieval precision for queries that mention specific works or artists.

---

## Evaluation

The technical document accompanying this project defines the following evaluation framework:

- **Precision@K** — fraction of the top-K results that are genuinely relevant to the query.
- **Recall@K** — fraction of all relevant artworks in the corpus retrieved within the top K.
- **Relevance@K** — mean binary relevance score over the top-K list.
- **Qualitative inspection** — side-by-side comparison of returned images vs. query intent.

Human relevance judgements were used to build the ground-truth labels for the four reference queries included in the script.

---

## Project Structure

```
.
├── semantic_art_search.py    # This pipeline
├── sample/
│   ├── drawings/
│   ├── paintings/
│   ├── photographs/
│   ├── prints/
│   └── metadata_sample.csv
├── description.json          # Generated by Phase 1
├── result.json               # Generated by Phase 2
└── chroma_db/                # Generated by Phase 3
```

---

