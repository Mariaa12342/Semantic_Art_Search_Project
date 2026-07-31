#!/usr/bin/env python3
"""
Semantic Art Search Engine
==========================
A four-phase Retrieval-Augmented Generation (RAG) pipeline for semantic
artwork retrieval using the Rijksmuseum dataset.

Pipeline:
    Image → LLaVA (VLM) → Text Description → Embeddings → ChromaDB → Search

Phases:
    1. VLM Extraction  – Send each image to LLaVA via Ollama; store structured
                         JSON descriptions per artwork.
    2. Metadata Cross  – Join the VLM output JSON with the CSV metadata file
                         (title, artist, date) on artwork ID.
    3. Embedding Ingest– Build rich text strings from each artwork's description
                         and metadata, encode them with a Sentence Transformer,
                         and persist the vectors in ChromaDB.
    4. Semantic Search – Encode a free-text query with the same embedding model
                         and retrieve the top-K most similar artworks via
                         cosine similarity.
"""

import base64
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import ollama
    import chromadb
    import pandas as pd
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Install libraries: pip install ollama chromadb sentence-transformers pandas")
    sys.exit(1)


# ════════════════════════════════════════════════════════════
#   CONFIGURATION  ← edit these variables
# ════════════════════════════════════════════════════════════

# --- PHASE 1: VLM ---
IMAGE_PATH              = "sample"          # Root folder with category sub-folders
OUTPUT_JSON             = "description.json"  # VLM output (Phase 1 → Phase 2)
MODEL                   = "llava"           # Ollama model tag
DELAY_SECONDS           = 1                 # Pause between consecutive Ollama calls
OLLAMA_HOST             = "http://localhost:11434"
MAX_RETRIES             = 3                 # Re-attempts on malformed JSON responses
MAX_IMAGES_PER_CATEGORY = None              # None = all; set an int for test mode

# --- PHASE 2: CSV CROSS ---
METADATA_CSV = "sample/metadata_sample.csv"  # Must contain an "id" column
CROSSED_JSON = "result.json"                 # Phase 2 → Phase 3

# --- PHASES 3 & 4: EMBEDDINGS AND SEARCH ---
CHROMA_PATH     = "chroma_db"           # Persistent ChromaDB storage directory
MODEL_EMBED     = "all-mpnet-base-v2"   # Sentence Transformer model (768-dim)
COLLECTION_NAME = "rijksmuseum_llava"   # ChromaDB collection name

# ════════════════════════════════════════════════════════════


# Maps file extensions to their MIME type for base64 encoding
MIME_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
}

# Primary prompt: instructs the VLM to return a strict JSON object.
# Low temperature (0.1) reduces hallucinations and keeps output deterministic.
PROMPT = """Look at this image and output ONLY a JSON object. No explanations, no markdown, no code blocks.
Start your response with { and end with }

Use this exact structure (replace values with real observations):
{"descripcion_general": "1-2 sentence summary", "elementos_principales": ["object1", "object2"], "colores_predominantes": ["color1", "color2", "color3"], "escena_o_contexto": "indoor/outdoor setting", "estado_de_animo": "mood or atmosphere", "texto_visible": null, "calidad_imagen": "buena", "etiquetas": ["tag1", "tag2", "tag3", "tag4", "tag5"]}

Remember: ONLY the JSON object, nothing else. Start with {"""

# Retry prompt: more explicit schema for cases where the first attempt fails
PROMPT_RETRY = """Output ONLY valid JSON for this image. Start with { end with }. No other text:
{"descripcion_general":"...","elementos_principales":["...","..."],"colores_predominantes":["...","...","..."],"escena_o_contexto":"...","estado_de_animo":"...","texto_visible":null,"calidad_imagen":"buena","etiquetas":["...","...","...","...","..."]}"""


# ══════════════════════════════════════════
# PHASE 1: INGESTION WITH LLaVA
# ══════════════════════════════════════════

def find_images(folder: Path) -> list[Path]:
    """
    Collect all image files in a directory (case-insensitive extension match).

    Args:
        folder: Directory to scan.

    Returns:
        Sorted, deduplicated list of image paths.
    """
    images = []
    for ext in MIME_TYPES:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def find_categories(root: Path) -> list[Path]:
    """
    Return all immediate sub-directories of *root*, sorted alphabetically.
    Each sub-directory is treated as an artwork category (e.g. drawings,
    paintings, photographs, prints).

    Args:
        root: Root image folder.

    Returns:
        List of category directory paths.
    """
    return sorted([p for p in root.iterdir() if p.is_dir()])


def check_ollama_and_model() -> ollama.Client:
    """
    Verify that Ollama is reachable and that the configured model is available.

    Returns:
        An authenticated Ollama client instance.

    Raises:
        SystemExit: If the Ollama server is unreachable or the model is missing.
    """
    try:
        client    = ollama.Client(host=OLLAMA_HOST)
        available = [m.model for m in client.list().models]
        matches   = [m for m in available if m.startswith(MODEL.split(":")[0])]
        if not matches:
            print(f"❌ Model '{MODEL}' not found in Ollama.")
            print(f"   Available models: {available or '(none)'}")
            print(f"   Download it with: ollama pull {MODEL}")
            sys.exit(1)
        return client
    except Exception:
        print(f"❌ Could not connect to Ollama at {OLLAMA_HOST}")
        print(f"   Is Ollama running? Start it with: ollama serve")
        sys.exit(1)


def clean_and_parse_json(raw: str) -> dict:
    """
    Parse a JSON string that may be wrapped in Markdown code fences or contain
    Python-style literals (None, True, False). Attempts multiple recovery
    strategies before raising.

    Strategy order:
        1. Strip code fences and parse directly.
        2. Extract the first {...} block with a regex.
        3. Normalize Python literals and retry the regex.

    Args:
        raw: Raw string returned by the VLM.

    Returns:
        Parsed Python dict.

    Raises:
        json.JSONDecodeError: If all recovery strategies fail.
    """
    # Strip optional ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw).strip()

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract first JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Attempt 3: convert Python literals to JSON-valid equivalents
    fixed = re.sub(r"(?<![\\])'", '"', raw)
    fixed = re.sub(r":\s*None\b",  ": null",  fixed)
    fixed = re.sub(r":\s*True\b",  ": true",  fixed)
    fixed = re.sub(r":\s*False\b", ": false", fixed)
    match2 = re.search(r"\{.*\}", fixed, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(0))
        except json.JSONDecodeError:
            pass

    return json.loads(fixed)  # Final attempt; raises on failure


def call_model(client: ollama.Client, image_data: str, prompt: str) -> str:
    """
    Send a single image to LLaVA via Ollama and return the raw text response.

    Args:
        client:     Ollama client instance.
        image_data: Base64-encoded image bytes.
        prompt:     Instruction prompt for the VLM.

    Returns:
        Raw model response string.
    """
    response = client.generate(
        model=MODEL,
        prompt=prompt,
        images=[image_data],
        options={"temperature": 0.1, "num_predict": 512},
    )
    return response.response.strip()


def analyze_image(client: ollama.Client, image_path: Path) -> dict:
    """
    Analyze a single image with LLaVA, retrying with a stricter prompt on
    JSON parse failures.

    Args:
        client:     Ollama client instance.
        image_path: Path to the image file.

    Returns:
        Parsed analysis dict matching the PROMPT schema.

    Raises:
        json.JSONDecodeError: If the model fails to produce valid JSON after
                              MAX_RETRIES attempts.
    """
    image_data = base64.b64encode(image_path.read_bytes()).decode()
    last_raw   = ""
    for attempt in range(1, MAX_RETRIES + 1):
        # Use the stricter retry prompt on second and subsequent attempts
        prompt = PROMPT if attempt == 1 else PROMPT_RETRY
        try:
            raw      = call_model(client, image_data, prompt)
            last_raw = raw
            return clean_and_parse_json(raw)
        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                print(f"↩️  attempt {attempt} failed, retrying...", end=" ", flush=True)
    raise json.JSONDecodeError(
        f"Invalid JSON after {MAX_RETRIES} attempts. Last response: {last_raw!r}",
        last_raw, 0
    )


def build_done_set(results: dict) -> set[str]:
    """
    Build a set of already-processed image keys from a partial results dict.
    Used when resuming an interrupted Phase 1 run. Only successfully processed
    images (no "error" key) are included, so failed ones will be retried.

    Args:
        results: Partially-filled Phase 1 output dict.

    Returns:
        Set of "category/filename" strings for completed images.
    """
    done = set()
    for cat_name, cat_data in results.get("categories", {}).items():
        for entry in cat_data.get("images", []):
            if "error" not in entry:
                done.add(f"{cat_name}/{entry['file']}")
    return done


def run_vlm_extraction():
    """
    Phase 1 – VLM Extraction.

    Iterates over all category sub-folders under IMAGE_PATH, sends each image
    to LLaVA, and writes the structured descriptions to OUTPUT_JSON.

    Features:
    - Resume support: skips images already present in an existing OUTPUT_JSON.
    - Retry logic: re-attempts images that previously produced errors.
    - Test mode: set MAX_IMAGES_PER_CATEGORY to limit images per category.
    - Incremental writes: OUTPUT_JSON is updated after every image so that a
      crash does not lose progress.
    """
    print("\n" + "=" * 50)
    print("🎨 PHASE 1: VLM Extraction LLaVA -> JSON")
    if MAX_IMAGES_PER_CATEGORY:
        print(f"   ⚠️  Test mode: max {MAX_IMAGES_PER_CATEGORY} images per category")
    print("=" * 50)

    root = Path(IMAGE_PATH)
    if not root.exists() or not root.is_dir():
        print(f"❌ Folder not found: {root}"); sys.exit(1)

    categories   = find_categories(root)
    all_images   = {cat: find_images(cat) for cat in categories}
    total_images = sum(len(v) for v in all_images.values())

    print(f"📂 Root folder  : {root.name}")
    print(f"📁 Categories   : {len(categories)} ({', '.join(c.name for c in categories)})")
    print(f"🖼️  Images       : {total_images} total")
    print(f"🤖 Model        : {MODEL}")
    print(f"💾 Output       : {OUTPUT_JSON}")
    print(f"\n🔌 Connecting to Ollama at {OLLAMA_HOST} ...")
    client = check_ollama_and_model()

    # Load existing results if resuming, otherwise start fresh
    output_path = Path(OUTPUT_JSON)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        already_done = build_done_set(results)
        # Remove previously failed entries so they will be retried
        for cat_name in results.get("categories", {}):
            results["categories"][cat_name]["images"] = [
                r for r in results["categories"][cat_name]["images"] if "error" not in r
            ]
        print(f"↩️  Resuming — {len(already_done)} OK, retrying failed ones\n")
    else:
        results      = {
            "metadata": {
                "root_folder": str(root.resolve()),
                "model":       MODEL,
                "date":        datetime.now().isoformat(),
            },
            "categories": {},
        }
        already_done = set()

    # Ensure every category exists in the results structure
    for cat in categories:
        if cat.name not in results["categories"]:
            results["categories"][cat.name] = {"total_images": len(all_images[cat]), "images": []}

    processed = errors = 0
    for cat in categories:
        images  = all_images[cat]
        if MAX_IMAGES_PER_CATEGORY:
            images = images[:MAX_IMAGES_PER_CATEGORY]
        pending = [img for img in images if f"{cat.name}/{img.name}" not in already_done]
        if not pending:
            continue
        print(f"📁 [{cat.name}] — {len(pending)} images pending out of {len(images)}")
        for i, img_path in enumerate(pending, 1):
            print(f"  [{i}/{len(pending)}] {img_path.name} ...", end=" ", flush=True)
            try:
                analysis = analyze_image(client, img_path)
                results["categories"][cat.name]["images"].append({
                    "file":         img_path.name,
                    "path":         str(img_path.resolve()),
                    "category":     cat.name,
                    "processed_at": datetime.now().isoformat(),
                    "analysis":     analysis,
                })
                print("✅"); processed += 1
            except json.JSONDecodeError:
                print(f"⚠️  Invalid JSON after {MAX_RETRIES} attempts")
                results["categories"][cat.name]["images"].append({
                    "file":     img_path.name,
                    "path":     str(img_path.resolve()),
                    "category": cat.name,
                    "error":    "Invalid JSON",
                })
                errors += 1
            except Exception as e:
                print(f"❌ Error: {e}")
                results["categories"][cat.name]["images"].append({
                    "file":     img_path.name,
                    "path":     str(img_path.resolve()),
                    "category": cat.name,
                    "error":    str(e),
                })
                errors += 1

            # Persist progress after every image
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            if i < len(pending):
                time.sleep(DELAY_SECONDS)
        print()

    print(f"✅ Phase 1 completed — {processed} OK, {errors} errors")
    print(f"💾 JSON saved at {output_path.resolve()}")


# ══════════════════════════════════════════
# PHASE 2: JSON + CSV CROSS
# ══════════════════════════════════════════

def load_csv(path: str) -> pd.DataFrame:
    """
    Load the metadata CSV with automatic encoding and delimiter detection.

    Tries common encodings (utf-8-sig, utf-8, cp1252, latin1) and common
    delimiters (auto, semicolon, comma, tab) until a valid parse that includes
    the required "id" column is found.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with the "id" column cast to stripped strings.

    Raises:
        The last caught exception if no combination succeeds.
    """
    encodings  = ("utf-8-sig", "utf-8", "cp1252", "latin1")
    separators = (None, ";", ",", "\t")
    last_error = None
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, encoding=enc, engine="python",
                                 on_bad_lines="skip", sep=sep)
                if "id" not in df.columns:
                    raise ValueError(f"Column 'id' not found. Columns: {list(df.columns)}")
                df["id"] = df["id"].astype(str).str.strip()
                print(f"   CSV loaded (encoding={enc}, sep={repr(sep)})")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as e:
                last_error = e
    raise last_error


def run_cross_data():
    """
    Phase 2 – Metadata Cross-reference.

    Merges the VLM descriptions from OUTPUT_JSON with structured metadata
    (title, artist, date) from METADATA_CSV, joining on artwork ID (filename
    stem). The result is written to CROSSED_JSON for ingestion in Phase 3.

    Images that appear only in the JSON (no matching CSV row) are silently
    discarded; a count is reported at the end.
    """
    print("\n" + "=" * 50)
    print("🔗 PHASE 2: JSON + CSV Metadata Cross")
    print("=" * 50)

    json_path = Path(OUTPUT_JSON)
    csv_path  = Path(METADATA_CSV)
    if not json_path.exists():
        print(f"❌ {OUTPUT_JSON} not found. Run Phase 1 first."); return
    if not csv_path.exists():
        print(f"❌ {METADATA_CSV} not found."); return

    with open(json_path, "r", encoding="utf-8-sig") as f:
        data_json = json.load(f)

    print(f"   Loading CSV: {METADATA_CSV}")
    df_csv    = load_csv(str(csv_path))
    # Build a fast lookup dict: artwork_id → {title, artist, date, ...}
    csv_index = df_csv.set_index("id").to_dict(orient="index")
    csv_ids   = set(csv_index.keys())

    result = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model":        MODEL,
            "csv_source":   str(csv_path.resolve()),
        },
        "categories": {},
    }
    total_common = total_json_only = 0
    json_ids_all = set()

    for cat_name, cat_data in data_json.get("categories", {}).items():
        crossed_images = []
        for img in cat_data.get("images", []):
            if "error" in img:
                continue
            img_id = os.path.splitext(img["file"])[0]  # Strip extension to get ID
            json_ids_all.add(img_id)
            if img_id in csv_ids:
                csv_row = csv_index[img_id]
                crossed_images.append({
                    "id":       img_id,
                    "file":     img["file"],
                    "path":     str(Path(IMAGE_PATH) / cat_name / img["file"]),
                    "title":    csv_row.get("title"),
                    "artist":   csv_row.get("artist"),
                    "date":     csv_row.get("date"),
                    "analysis": img.get("analysis", {}),
                })
                total_common += 1
            else:
                total_json_only += 1

        if crossed_images:
            result["categories"][cat_name] = {
                "total":  len(crossed_images),
                "images": sorted(crossed_images, key=lambda x: x["id"]),
            }

    result["metadata"].update({
        "total_common":    total_common,
        "total_json_only": total_json_only,
        "total_csv_only":  len(csv_ids - json_ids_all),
    })

    with open(Path(CROSSED_JSON), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"   Images in JSON:          {len(json_ids_all)}")
    print(f"   Rows in CSV:             {len(csv_ids)}")
    print(f"   Common IDs (cross):      {total_common}")
    print(f"   JSON only (no CSV):      {total_json_only}")
    print(f"   CSV only (no image):     {len(csv_ids - json_ids_all)}")
    print(f"✅ Phase 2 completed — result saved at {Path(CROSSED_JSON).resolve()}")


# ══════════════════════════════════════════
# PHASE 3: TEXT -> EMBEDDINGS -> CHROMADB
# ══════════════════════════════════════════

def to_str(val) -> str:
    """
    Safely convert a value to a plain string.
    Lists are joined with commas; None becomes an empty string.

    Args:
        val: Any value (str, list, None, …).

    Returns:
        String representation.
    """
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val) if val else ""


def build_text_from_analysis(analysis: dict, title: str = "", artist: str = "") -> str:
    """
    Assemble a single rich-text string that will be embedded for an artwork.

    The text prioritises title and artist name at the front (high semantic
    weight), followed by the general description, mood, scene context, and
    tags. This ordering gives the embedding model the most discriminative
    features first, improving retrieval quality.

    Fields included (in order):
        1. title            – e.g. "The Night Watch"
        2. artist           – e.g. "by Rembrandt van Rijn"
        3. descripcion_general – 1-2 sentence VLM summary
        4. estado_de_animo  – mood / atmosphere
        5. escena_o_contexto – scene context (indoor/outdoor …)
        6. etiquetas        – comma-separated semantic tags

    Args:
        analysis: The analysis dict from the VLM (Phase 1).
        title:    Artwork title from CSV metadata.
        artist:   Artist name from CSV metadata.

    Returns:
        Period-separated text string ready for embedding.
    """
    parts = []
    if title:
        parts.append(to_str(title))
    if artist and to_str(artist).lower() != "anonymous":
        parts.append(f"by {to_str(artist)}")
    description = to_str(analysis.get("descripcion_general", ""))
    if description:
        parts.append(description)
    mood = to_str(analysis.get("estado_de_animo", ""))
    if mood:
        parts.append(mood)
    context = to_str(analysis.get("escena_o_contexto", ""))
    if context:
        parts.append(context)
    tags = ", ".join(to_str(e) for e in analysis.get("etiquetas", []))
    if tags:
        parts.append(tags)
    return ". ".join(parts)


def run_chroma_ingestion():
    """
    Phase 3 – Embedding Generation and ChromaDB Ingestion.

    For each artwork in CROSSED_JSON:
        1. Build a rich text string via build_text_from_analysis().
        2. Encode it with MODEL_EMBED (all-mpnet-base-v2, 768 dimensions).
        3. Store the (id, embedding, metadata, document) tuple in ChromaDB.

    The collection uses cosine similarity ("hnsw:space": "cosine").
    Incremental ingestion is supported: artworks already in the collection are
    skipped, so the function is idempotent.
    """
    print("\n" + "=" * 50)
    print("🧠 PHASE 3: Embedding Generation -> ChromaDB")
    print("=" * 50)

    crossed_path = Path(CROSSED_JSON)
    if not crossed_path.exists():
        print(f"❌ {CROSSED_JSON} not found. Run Phase 2 first."); return

    with open(crossed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build (id, text, metadata) tuples for every valid artwork
    valid_works = []
    for cat_name, cat_data in data.get("categories", {}).items():
        for img in cat_data.get("images", []):
            analysis = img.get("analysis", {})
            text     = build_text_from_analysis(
                analysis,
                img.get("title", ""),
                img.get("artist", ""),
            )
            valid_works.append({
                "id":   img["id"],
                "text": text,
                "meta": {
                    "file":        str(img.get("file", "")),
                    "category":    cat_name,
                    "image_path":  str(img.get("path", "")),
                    "title":       str(img.get("title") or ""),
                    "artist":      str(img.get("artist") or ""),
                    "date":        str(img.get("date") or ""),
                    "description": str(analysis.get("descripcion_general", "")),
                },
            })

    if not valid_works:
        print("⚠️  No valid images in crossed JSON to vectorize."); return

    print(f"📖 {len(valid_works)} valid works found")
    print(f"🔤 Loading embedding model '{MODEL_EMBED}'...")
    embedder = SentenceTransformer(MODEL_EMBED)

    test_dim = len(embedder.encode(["test"])[0])
    print(f"📐 Vector dimension: {test_dim}")

    print(f"🗄️  Initializing ChromaDB at '{CHROMA_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection    = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity for retrieval
    )

    # Skip artworks that are already indexed
    existing_ids = set(collection.get()["ids"])
    new_works    = [o for o in valid_works if o["id"] not in existing_ids]

    if not new_works:
        print(f"✅ ChromaDB already has {collection.count()} documents. Nothing to add.")
        return

    print(f"⚙️  Vectorizing {len(new_works)} new works...")
    batch_size = 100
    for i in range(0, len(new_works), batch_size):
        batch      = new_works[i: i + batch_size]
        embeddings = embedder.encode(
            [o["text"] for o in batch],
            normalize_embeddings=True,  # L2 normalization; cosine sim = dot product
        ).tolist()
        collection.add(
            ids        = [o["id"]   for o in batch],
            embeddings = embeddings,
            metadatas  = [o["meta"] for o in batch],
            documents  = [o["text"] for o in batch],
        )
        print(f"   Inserted: {min(i + batch_size, len(new_works))}/{len(new_works)}")

    print(f"✅ ChromaDB ready — {collection.count()} documents total")


# ══════════════════════════════════════════
# PHASE 4: SEARCH (RETRIEVAL)
# ══════════════════════════════════════════

def search_art(query: str, n_results: int = 5):
    """
    Phase 4 – Semantic Search.

    Encodes *query* with the same embedding model used during ingestion and
    retrieves the top-K most similar artworks from ChromaDB using cosine
    similarity. Results are printed in ranked order with similarity score,
    metadata, and the VLM description summary.

    Args:
        query:     Free-text search query (e.g. "A gloomy room lit by a candle").
        n_results: Number of top results to return.
    """
    print("\n" + "=" * 50)
    print(f"🔍 SEARCHING: '{query}'")
    print("=" * 50)

    embedder      = SentenceTransformer(MODEL_EMBED)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"❌ Collection '{COLLECTION_NAME}' not found. Run Phase 3 first.")
        return

    # Encode query with the same model and normalization as during ingestion
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
    results      = collection.query(
        query_embeddings=[query_vector],
        n_results=min(n_results, collection.count()),
    )

    if not results["ids"][0]:
        print("No results found.")
        return

    for i, (distance, metadata) in enumerate(
        zip(results["distances"][0], results["metadatas"][0]), 1
    ):
        # ChromaDB returns cosine distance; convert to similarity percentage
        similarity = (1 - distance) * 100
        print(f"🏅 Top {i} | Similarity: {similarity:.1f}%")
        print(f"   File     : {metadata['file']} | Category: {metadata['category']}")
        print(f"   Title    : {metadata.get('title', 'N/A')} | "
              f"Artist: {metadata.get('artist', 'N/A')} | "
              f"Date: {metadata.get('date', 'N/A')}")
        print(f"   Path     : {metadata['image_path']}")
        print(f"   Summary  : {metadata['description']}\n")


# ══════════════════════════════════════════
# FULL PIPELINE ENTRY POINT
# ══════════════════════════════════════════

if __name__ == "__main__":

    # ── Phase 1: LLaVA → description.json ────────────────────────────────────
    # Comment out this line if description.json already exists and is complete.
    run_vlm_extraction()

    # ── Clean previous Phase 2/3 artifacts ───────────────────────────────────
    # Ensures a fresh cross and re-ingestion after a new VLM run.
    for path in [Path(CROSSED_JSON), Path(CHROMA_PATH)]:
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            print(f"🗑️  Deleted: {path}")

    # ── Phase 2: description.json + CSV → result.json ────────────────────────
    run_cross_data()

    # ── Phase 3: result.json → Embeddings → ChromaDB ─────────────────────────
    run_chroma_ingestion()

    # ── Phase 4: Sample semantic queries ─────────────────────────────────────
    print("\n📋 Running test searches:")
    search_art("A gloomy room lit by a candle",        n_results=3)
    search_art("Melancholic scene related to religion", n_results=3)
    search_art("Joy",                                   n_results=3)
    search_art("A building of marble and limestone",    n_results=3)
