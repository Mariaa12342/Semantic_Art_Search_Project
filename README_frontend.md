# Semantic Art Search Engine — Frontend & API

A browser-based interface and REST API that expose both ingestion pipelines as a single interactive search tool over the Rijksmuseum dataset.

> Built for the **Modelling Week MS 2026** project (Management Solutions).

---

## Overview

```
Browser (index.html)
  └─► POST /api/search  ──► app.py (FastAPI)
                                ├─► pipeline="image" → CLIP text encoder → ChromaDB (image vectors)
                                └─► pipeline="text"  → all-mpnet-base-v2  → ChromaDB (VLM text vectors)
```

The user types a natural-language query, selects a pipeline, and the backend returns the top-K most similar artworks ranked by cosine similarity. Results are displayed as an image grid with title, artist, date, and similarity score. Clicking any card opens a full-screen lightbox with the artwork's VLM description (when available).

---

## Files

| File | Description |
|------|-------------|
| `app.py` | FastAPI backend — query embedding, ChromaDB retrieval, image serving |
| `static/index.html` | Single-page frontend — search UI, results grid, lightbox |

---

## Requirements

```bash
pip install fastapi uvicorn chromadb sentence-transformers transformers torch pandas
```

---

## Project Layout

```
.
├── app.py
├── static/
│   └── index.html
├── sample/                        # Artwork images (category sub-folders)
│   ├── drawings/
│   ├── paintings/
│   ├── photographs/
│   ├── prints/
│   └── metadata.csv               # CLIP pipeline metadata (id, title, artist, date)
├── vector_database_2_full/        # ChromaDB — CLIP image embeddings
└── chroma_db/                     # ChromaDB — VLM text embeddings
```

---

## Configuration

All paths and model identifiers are set at the top of `app.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGES_FOLDER` | `./sample` | Root folder containing artwork images |
| `IMAGE_DB_PATH` | `./vector_database_2_full` | ChromaDB for the CLIP pipeline |
| `IMAGE_COLLECTION` | `"rijksmuseum_artworks"` | Collection name (CLIP) |
| `CLIP_MODEL_ID` | `"openai/clip-vit-base-patch32"` | CLIP model for query embedding |
| `CLIP_CSV_PATH` | `./sample/metadata.csv` | Artwork metadata for the CLIP pipeline |
| `TEXT_DB_PATH` | `./chroma_db` | ChromaDB for the VLM pipeline |
| `TEXT_COLLECTION` | `"rijksmuseum_llava"` | Collection name (VLM) |
| `TEXT_MODEL_ID` | `"all-mpnet-base-v2"` | Sentence Transformer for query embedding |

---

## Running

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in a browser.

Both ChromaDB collections must exist before starting the server. Run the ingestion scripts for each pipeline first if they have not been built yet.

---

## API Reference

### `GET /api/health`

Returns the availability and document count of each pipeline's ChromaDB collection.

```json
{
  "image": {"available": true, "count": 25000},
  "text":  {"available": true, "count": 1000}
}
```

### `POST /api/search`

**Request body:**

```json
{
  "query":     "A gloomy room lit by a candle",
  "pipeline":  "image",
  "n_results": 6
}
```

`pipeline` must be `"image"` (CLIP) or `"text"` (VLM + mpnet).  
`n_results` is clamped to [1, 50].

**Response:**

```json
{
  "query":    "A gloomy room lit by a candle",
  "pipeline": "image",
  "results": [
    {
      "id":          "SK-A-1234",
      "similarity":  87.3,
      "title":       "Interior with candlelight",
      "artist":      "Rembrandt van Rijn",
      "date":        "1640",
      "category":    "paintings",
      "description": "A dimly lit interior...",
      "filepath":    "..."
    }
  ]
}
```

Only results whose image file exists on disk are included. The text pipeline additionally filters out the `"photomechanical print"` category.

### `GET /api/image/{pipeline}/{item_id}`

Serves the artwork image file from disk. `item_id` is the filename stem (e.g. `SK-A-1234`). The `.jpg` extension is appended automatically if absent.

---

## Frontend Features

- **Pipeline toggle** — switch between Image (CLIP) and Text (VLM + mpnet); re-runs the last query automatically on switch.
- **N results** — configurable result count, clamped to 50.
- **Min. similarity threshold** — client-side filter; cards below the threshold are hidden after retrieval.
- **Skeleton loading** — placeholder cards shown while the API request is in flight.
- **Lightbox** — click any card to open a full-screen view with the artwork image, metadata, VLM description, and the active pipeline label.
- **Lazy image loading** — card images use `loading="lazy"` with an inline error fallback.
- **Health status bar** — shows live document counts for both databases on page load.

---

## Authors

Laura, Daniel, Javier, María, Andrés, Juan & Gabriel — Máster en Ingeniería Matemática, Universidad Complutense de Madrid  
Modelling Week MS 2026 · Management Solutions
