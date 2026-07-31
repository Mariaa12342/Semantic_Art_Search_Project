#!/usr/bin/env python3
"""
Semantic Art Search Engine
==========================
Evaluation Script — Dual Embedding Comparison with Ground Truth and VLM Queries

This script evaluates and compares two ChromaDB embedding collections using
two complementary query sources:

    1. Ground Truth (GT) — fixed queries loaded from a JSON file.
       Fully reproducible; no VLM involvement.

    2. VLM Queries — a random sample of artworks from full_result.json,
       where the VLM (LLaVA via Ollama) generates one query per image per
       prompt strategy. Four strategies are used:
           - short:         1-2 word query focusing on the most distinctive element.
           - long_specific: 2-3 sentence detailed description.
           - style:         one sentence focused on artistic style and technique.
           - era:           one sentence focused on historical period.

Both query sources are evaluated against both embedding collections (A and B)
using the same retrieval metrics, enabling a direct side-by-side comparison.
Results are tagged with a `source` column ("gt" or "vlm") for downstream
filtering.

Relevance definition:
    - GT queries:  the exact image_id is always relevant, plus artworks by the
                   same artist or sharing tags (if full_result.json is provided).
    - VLM queries: artworks by the same artist or sharing tags (excluding the
                   query image itself).

Dynamic K (Kneedle algorithm):
    Rather than using a fixed K, the script detects the elbow of the distance
    curve for each query result set and uses that as the evaluation cutoff,
    subject to a configurable minimum (k_min).

Metrics computed per query:
    Precision@K, Recall@K, MRR, nDCG@K, Hit Rate@K, AP@K,
    exact_rank, exact_hit_at_k.

Output files:
    gt_results.csv   — one row per (query × embedding).
    gt_summary.csv   — aggregated metrics: global, by source, by query type,
                       by category, for each embedding.
    gt_compare.csv   — direct A vs B comparison with delta per metric.

Dependencies:
    pip install chromadb sentence-transformers tqdm numpy ollama Pillow

Usage (default configuration):
    python evaluation.py

Usage (custom parameters):
    python evaluation.py \\
        --gt-json         directed_queries.json \\
        --full-json       full_result.json \\
        --images-dir      ./rijksmuseum \\
        --vlm-sample      20 \\
        --vlm-model       llava \\
        --seed            42 \\
        --chroma-path-a   ./vector_database_1_full \\
        --collection-a    rijksmuseum_llava \\
        --embed-model-a   sentence-transformers/all-mpnet-base-v2 \\
        --label-a         mpnet \\
        --chroma-path-b   ./vector_database_2_full \\
        --collection-b    rijksmuseum_artworks \\
        --embed-model-b   openai/clip-vit-base-patch32 \\
        --label-b         clip \\
        --pool-size       300 \\
        --k-min           3 \\
        --output-csv      ./gt_results.csv \\
        --summary-csv     ./gt_summary.csv \\
        --compare-csv     ./gt_compare.csv
"""

import argparse
import base64
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import chromadb
except ImportError:
    raise ImportError("pip install chromadb")

try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")


# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Path to the ground truth JSON file (list of {image_id, query, query_type}).
    "gt_json":          "directed_queries.json",

    # Path to the full VLM output JSON (full_result.json).
    # Required for VLM query generation and for enriching GT relevance sets
    # with same-artist / same-tag artworks. Set to None to use exact hit only.
    "full_json":        "full_result.json",

    # Root folder containing artwork images, organised as:
    #   images_dir / <category> / <filename>
    "images_dir":       "./rijksmuseum",

    # Number of images sampled per VLM prompt strategy.
    # Total VLM evaluations = vlm_sample × 4 strategies × 2 embeddings.
    # Set to 0 to disable VLM query generation entirely.
    "vlm_sample":       20,
    "vlm_model":        "llava",
    "random_seed":      42,

    # Optional category filter for VLM sampling (None = all categories).
    "category_filter":  None,

    # Minimum number of relevant artworks required for an image to be included
    # in the VLM sample (avoids images with no meaningful ground truth).
    "min_relevant":     2,

    # Embedding A — LLaVA text descriptions + all-mpnet-base-v2.
    "chroma_path_a":    "./vector_database_1_full",
    "collection_a":     "rijksmuseum_llava",
    "embed_model_a":    "sentence-transformers/all-mpnet-base-v2",
    "label_a":          "A",

    # Embedding B — CLIP image embeddings.
    "chroma_path_b":    "./vector_database_2_full",
    "collection_b":     "rijksmuseum_artworks",
    "embed_model_b":    "openai/clip-vit-base-patch32",
    "label_b":          "B",

    # Candidate pool size passed to ChromaDB per query.
    # Dynamic K is chosen from within this pool via the Kneedle algorithm.
    "pool_size":        300,
    "k_min":            3,

    # Output file paths.
    "output_csv":       "./gt_results.csv",
    "summary_csv":      "./gt_summary.csv",
    "compare_csv":      "./gt_compare.csv",
}


# ══════════════════════════════════════════════════════════════════════════════
#   VLM PROMPT STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_STRATEGIES = [
    {
        "name":        "short",
        "description": "Very short query (1-2 words)",
        "prompt": (
            "Look at this image. "
            "Write an EXTREMELY short search query of ONLY 1 or 2 words that someone "
            "would type to find this artwork in a museum image database. "
            "Pick the single most distinctive visual element "
            "(e.g. 'windmill', 'woman portrait', 'naval battle'). "
            "Return ONLY the 1-2 word query. No explanation, no quotes, no punctuation."
        ),
    },
    {
        "name":        "long_specific",
        "description": "Long, detailed and specific query",
        "prompt": (
            "Look at this image. "
            "Write a long, detailed and specific search query (2-3 sentences) that someone "
            "would type to find this EXACT artwork in a museum image database. "
            "Describe with precision: the main subject, secondary elements, composition, "
            "colors, lighting, clothing, objects, setting and mood. "
            "Return ONLY the search query, no explanation, no quotes."
        ),
    },
    {
        "name":        "style",
        "description": "Query focused on artistic style",
        "prompt": (
            "Look at this image. "
            "Write a search query (1 sentence) focused EXCLUSIVELY on the artistic STYLE "
            "of this artwork: art movement (e.g. baroque, impressionism, romanticism), "
            "medium and technique (oil painting, etching, watercolor), brushwork, "
            "color palette and overall aesthetic. "
            "Do NOT describe the subject or content of the image. "
            "Return ONLY the search query, no explanation, no quotes."
        ),
    },
    {
        "name":        "era",
        "description": "Query focused on historical period",
        "prompt": (
            "Look at this image. "
            "Write a search query (1 sentence) focused EXCLUSIVELY on the historical "
            "PERIOD of this artwork: the estimated century or date range when it was "
            "created (e.g. '17th century', 'Dutch Golden Age 1600s', 'late medieval'), "
            "supported by era-specific clues visible in the image such as clothing, "
            "architecture or technology. "
            "Do NOT describe the general subject beyond what dates the image. "
            "Return ONLY the search query, no explanation, no quotes."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#   GROUND TRUTH LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_gt(gt_json_path: str) -> list[dict]:
    """
    Load the ground truth JSON file and validate required fields.

    Each entry must contain at minimum: image_id, query, query_type.
    The function raises a descriptive error if any entry is malformed so that
    problems in the GT file are caught before the (potentially long) evaluation
    run begins.

    Args:
        gt_json_path: Path to the ground truth JSON file.

    Returns:
        List of GT entry dicts.

    Raises:
        ValueError: If the JSON is not a list or any entry is missing required fields.
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"GT JSON must be a list. Got: {type(data)}")

    required = {"image_id", "query", "query_type"}
    for i, entry in enumerate(data):
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"GT entry {i} is missing fields: {missing}")

    print(f"  {len(data)} ground truth queries loaded.")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#   METADATA LOADING AND RELEVANCE INDEXING
# ══════════════════════════════════════════════════════════════════════════════

def load_full_images(json_path: str) -> list[dict]:
    """
    Load all artwork records from full_result.json by flattening the nested
    category structure into a flat list.

    A '_category' key is injected into each record so that the category is
    accessible without traversing the outer dict again.

    Args:
        json_path: Path to the merged VLM output JSON.

    Returns:
        Flat list of artwork dicts, each enriched with '_category'.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = []
    for cat, cat_data in data.get("categorias", {}).items():
        for img in cat_data.get("images", []):
            img["_category"] = cat
            images.append(img)
    return images


def build_relevance_index(all_images: list[dict]) -> dict[str, dict]:
    """
    Build a relevance index that maps each artwork ID to the set of other
    artworks that are considered relevant to it.

    Relevance is defined as sharing the same non-anonymous artist OR sharing
    at least one semantic tag. The index stores three sets per artwork:
        - relevant:  union of by_artist and by_tags (excluding the artwork itself).
        - by_artist: other artworks by the same artist.
        - by_tags:   other artworks sharing at least one tag.

    This index is used at evaluation time to determine which retrieved results
    count as true positives for a given query.

    Args:
        all_images: Flat list of artwork dicts from load_full_images().

    Returns:
        Dict mapping artwork ID (filename stem) to relevance sets.
    """
    artist_idx: dict[str, list[str]] = defaultdict(list)
    tag_idx:    dict[str, list[str]] = defaultdict(list)

    for img in all_images:
        iid    = Path(img.get("file", "")).stem
        artist = (img.get("artist") or "").strip().lower()
        if artist and artist != "anonymous":
            artist_idx[artist].append(iid)
        for tag in (img.get("analysis", {}).get("etiquetas", []) or []):
            if isinstance(tag, str):
                tag_idx[tag.strip().lower()].append(iid)

    index: dict[str, dict] = {}
    for img in all_images:
        iid    = Path(img.get("file", "")).stem
        artist = (img.get("artist") or "").strip().lower()

        by_artist: set[str] = set()
        if artist and artist != "anonymous":
            by_artist = set(artist_idx[artist]) - {iid}

        by_tags: set[str] = set()
        for tag in (img.get("analysis", {}).get("etiquetas", []) or []):
            if isinstance(tag, str):
                by_tags.update(tag_idx[tag.strip().lower()])
        by_tags.discard(iid)

        index[iid] = {
            "relevant":  by_artist | by_tags,
            "by_artist": by_artist,
            "by_tags":   by_tags,
        }
    return index


def get_relevant_gt(image_id: str, rel_index: "dict | None") -> set:
    """
    Return the relevance set for a GT query.

    The exact image_id is always included as relevant (exact match).
    If a relevance index is available, same-artist and same-tag artworks
    are added as additional soft positives.

    Args:
        image_id:  Artwork ID that the GT query was derived from.
        rel_index: Relevance index from build_relevance_index(), or None.

    Returns:
        Set of relevant artwork IDs.
    """
    relevant = {image_id}
    if rel_index and image_id in rel_index:
        relevant |= rel_index[image_id]["relevant"]
    return relevant


def get_relevant_vlm(image_id: str, rel_index: dict) -> set:
    """
    Return the relevance set for a VLM query.

    Unlike GT queries, the query image itself is NOT included as a relevant
    result — the query was generated FROM that image, so retrieving it back
    would be a trivial exact match rather than a meaningful semantic hit.
    Only same-artist and same-tag artworks are considered relevant.

    Args:
        image_id:  Artwork ID that the VLM query was generated from.
        rel_index: Relevance index from build_relevance_index().

    Returns:
        Set of relevant artwork IDs (excluding image_id itself).
    """
    return rel_index.get(image_id, {}).get("relevant", set())


# ══════════════════════════════════════════════════════════════════════════════
#   VLM QUERY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def image_to_base64(image_path: Path) -> str:
    """
    Read an image file and return its contents as a base64-encoded string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64-encoded image bytes as a UTF-8 string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_query_from_image(
    image_path: Path,
    vlm_model:  str,
    prompt:     str,
) -> tuple[str, float]:
    """
    Send an image to the VLM via Ollama and return the generated query text
    together with the measured inference latency.

    Args:
        image_path: Path to the artwork image file.
        vlm_model:  Ollama model tag (e.g. 'llava').
        prompt:     Instruction prompt for the VLM.

    Returns:
        Tuple of (query_text, latency_seconds).
    """
    img_b64 = image_to_base64(image_path)
    t0      = time.perf_counter()
    response = ollama.chat(
        model=vlm_model,
        messages=[{
            "role":    "user",
            "content": prompt,
            "images":  [img_b64],
        }],
    )
    latency = time.perf_counter() - t0
    return response["message"]["content"].strip(), latency


def build_vlm_entries(
    all_images: list[dict],
    rel_index:  dict,
    config:     dict,
) -> list[dict]:
    """
    Sample a random subset of artworks and generate VLM queries for each
    prompt strategy.

    For each strategy a separate random sample is drawn (with a strategy-
    specific seed offset to ensure independent samples across strategies).
    Only artworks that have at least `min_relevant` relevant artworks in the
    index are eligible for sampling, to avoid queries with no meaningful
    ground truth.

    Args:
        all_images: Flat list of artwork dicts from load_full_images().
        rel_index:  Relevance index from build_relevance_index().
        config:     Configuration dict (uses vlm_sample, vlm_model,
                    images_dir, random_seed, category_filter, min_relevant).

    Returns:
        List of VLM entry dicts in the same format as GT entries, with
        additional fields: source='vlm', vlm_latency_s.
    """
    n = config["vlm_sample"]
    if n <= 0:
        return []

    cat      = config.get("category_filter")
    pool     = [i for i in all_images if i.get("_category") == cat] if cat else all_images
    min_rel  = config["min_relevant"]
    eligible = [
        i for i in pool
        if len(rel_index.get(
            Path(i.get("file", "")).stem, {}
        ).get("relevant", set())) >= min_rel
    ]

    if not eligible:
        print(f"  No images meet min_relevant={min_rel}. VLM query generation skipped.")
        return []

    images_dir = Path(config["images_dir"])
    entries    = []

    for s_idx, strat in enumerate(PROMPT_STRATEGIES):
        rng    = random.Random(config["random_seed"] + s_idx)
        sample = rng.sample(eligible, min(n, len(eligible)))

        print(f"\n  VLM strategy '{strat['name']}' — {len(sample)} images")

        for img in tqdm(sample, desc=f"[VLM-{strat['name']}]", unit="img"):
            img_id   = Path(img.get("file", "unknown")).stem
            category = img.get("_category", "")
            filename = Path(img.get("file", "")).name
            img_path = images_dir / category / filename

            if not img_path.exists():
                tqdm.write(f"  Image not found: {img_path}. Skipping.")
                continue

            try:
                query_text, vlm_latency = generate_query_from_image(
                    img_path, config["vlm_model"], strat["prompt"]
                )
            except Exception as e:
                tqdm.write(f"  VLM error ({img_id}): {e}. Skipping.")
                continue

            entries.append({
                "image_id":      img_id,
                "file":          f"{category}/{filename}",
                "query":         query_text,
                "query_type":    strat["name"],
                "category":      category,
                "notes":         "",
                "source":        "vlm",
                "vlm_latency_s": round(vlm_latency, 3),
            })

    print(f"\n  {len(entries)} VLM queries generated in total.")
    return entries


# ══════════════════════════════════════════════════════════════════════════════
#   CHROMADB RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def query_chroma(
    collection,
    embed_model: SentenceTransformer,
    query_text:  str,
    pool_size:   int,
) -> tuple[list[str], list[float], float]:
    """
    Encode a query string and retrieve the top pool_size results from ChromaDB.

    Args:
        collection:  ChromaDB Collection object.
        embed_model: SentenceTransformer used to encode the query.
        query_text:  Free-text search query.
        pool_size:   Number of candidates to retrieve.

    Returns:
        Tuple of (retrieved_ids, distances, latency_seconds).
        Distances are cosine distances in [0, 2]; lower means more similar.
    """
    embedding = embed_model.encode(query_text).tolist()
    t0        = time.perf_counter()
    result    = collection.query(
        query_embeddings=[embedding],
        n_results=pool_size,
        include=["distances"],
    )
    latency = time.perf_counter() - t0
    return result["ids"][0], result["distances"][0], latency


# ══════════════════════════════════════════════════════════════════════════════
#   DYNAMIC K (KNEEDLE ALGORITHM)
# ══════════════════════════════════════════════════════════════════════════════

def _kneedle(distances: list) -> int:
    """
    Detect the elbow point of a distance curve using the Kneedle algorithm.

    The algorithm normalises the distance curve to [0,1] × [0,1] and finds
    the point of maximum perpendicular distance from the line connecting the
    first and last points. This corresponds to the 'knee' where the marginal
    gain from including the next result drops off sharply.

    Args:
        distances: List of cosine distances in ascending order.

    Returns:
        1-indexed elbow position.
    """
    n = len(distances)
    if n < 3:
        return n

    x  = np.arange(n, dtype=float)
    y  = np.array(distances, dtype=float)
    xr = x[-1] - x[0]
    yr = y[-1] - y[0]
    xn = (x - x[0]) / xr if xr else x
    yn = (y - y[0]) / yr if yr else y

    dx, dy = xn[-1] - xn[0], yn[-1] - yn[0]
    ll     = math.hypot(dx, dy)
    if ll == 0:
        return n // 2

    perp = np.abs(
        dy * xn - dx * yn + (xn[-1] * yn[0] - xn[0] * yn[-1])
    ) / ll
    return max(1, int(np.argmax(perp)) + 1)


def dynamic_k(distances: list, k_min: int = 3) -> dict:
    """
    Compute the dynamic evaluation cutoff K for a single query result set.

    K is set to the Kneedle elbow position, clamped to be at least k_min
    and at most len(distances).

    Args:
        distances: List of cosine distances from ChromaDB.
        k_min:     Minimum allowed K value.

    Returns:
        Dict with keys: k_kneedle (raw elbow), k_final (clamped),
        dist_at_k (distance at the chosen cutoff).
    """
    k1      = _kneedle(distances)
    k_final = max(k_min, min(k1, len(distances)))
    return {
        "k_kneedle": k1,
        "k_final":   k_final,
        "dist_at_k": round(distances[k_final - 1], 6) if distances else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#   RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _precision(relevant: set, retrieved: list, k: int) -> float:
    """Fraction of the top-K retrieved results that are relevant."""
    return sum(1 for r in retrieved[:k] if r in relevant) / k if k else 0.0


def _recall(relevant: set, retrieved: list, k: int) -> float:
    """Fraction of all relevant artworks that appear in the top-K results."""
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant)


def _rr(relevant: set, retrieved: list) -> float:
    """Reciprocal rank of the first relevant result (MRR contribution)."""
    return next(
        (1.0 / (i + 1) for i, r in enumerate(retrieved) if r in relevant), 0.0
    )


def _ndcg(relevant: set, retrieved: list, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.

    Binary relevance is assumed (relevant = 1, not relevant = 0).
    IDCG is computed assuming the maximum possible number of relevant
    results are ranked at the top.
    """
    dcg  = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(retrieved[:k]) if r in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def _hit(relevant: set, retrieved: list, k: int) -> float:
    """1.0 if at least one relevant result appears in the top K, else 0.0."""
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def _ap(relevant: set, retrieved: list, k: int) -> float:
    """
    Average Precision at K.

    Computes the mean of precision values at each position where a relevant
    result is found, up to rank K.
    """
    hits, total = 0, 0.0
    for i, r in enumerate(retrieved[:k], 1):
        if r in relevant:
            hits  += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def compute_metrics(relevant: set, retrieved: list, k: int) -> dict:
    """
    Compute all retrieval metrics for a single query.

    Args:
        relevant:  Set of relevant artwork IDs for this query.
        retrieved: Ordered list of retrieved artwork IDs from ChromaDB.
        k:         Evaluation cutoff (dynamic K).

    Returns:
        Dict with keys: precision_at_k, recall_at_k, mrr, ndcg_at_k,
        hit_rate_at_k, ap_at_k.
    """
    return {
        "precision_at_k": round(_precision(relevant, retrieved, k), 4),
        "recall_at_k":    round(_recall(relevant,   retrieved, k), 4),
        "mrr":            round(_rr(relevant,        retrieved),    4),
        "ndcg_at_k":      round(_ndcg(relevant,      retrieved, k), 4),
        "hit_rate_at_k":  round(_hit(relevant,        retrieved, k), 4),
        "ap_at_k":        round(_ap(relevant,         retrieved, k), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#   EMBEDDING EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_embedding(
    label:       str,
    collection,
    embed_model: SentenceTransformer,
    entries:     list,
    rel_index,
    pool_size:   int,
    k_min:       int,
) -> list:
    """
    Evaluate one embedding collection over all query entries (GT and VLM).

    For each entry the function:
        1. Determines the relevant set based on the entry's source field.
        2. Retrieves pool_size candidates from ChromaDB.
        3. Computes dynamic K via the Kneedle algorithm.
        4. Computes all retrieval metrics at that K.
        5. Records the exact rank of the query image in the result list.

    Args:
        label:       Human-readable label for this embedding (e.g. 'A', 'mpnet').
        collection:  ChromaDB Collection to query.
        embed_model: SentenceTransformer used to encode queries.
        entries:     Combined list of GT and VLM entry dicts.
        rel_index:   Relevance index from build_relevance_index(), or None.
        pool_size:   Number of candidates to retrieve per query.
        k_min:       Minimum dynamic K value.

    Returns:
        List of result row dicts, one per evaluated query.
    """
    rows = []

    for entry in tqdm(entries, desc=f"[{label}]", unit="query"):
        image_id   = entry["image_id"]
        query_text = entry["query"]
        query_type = entry.get("query_type", "")
        category   = entry.get("category", "")
        source     = entry.get("source", "gt")

        # Build the relevance set according to the query source.
        if source == "gt":
            relevant = get_relevant_gt(image_id, rel_index)
        else:
            relevant = get_relevant_vlm(image_id, rel_index) if rel_index else set()

        try:
            retrieved_ids, distances, chroma_latency = query_chroma(
                collection, embed_model, query_text, pool_size
            )
        except Exception as e:
            tqdm.write(f"  ChromaDB error ({image_id}): {e}. Skipping.")
            continue

        k_info = dynamic_k(distances, k_min=k_min)
        k      = k_info["k_final"]

        # Exact rank: 1-indexed position of the query image in results,
        # or None if it does not appear within the pool.
        try:
            exact_rank = retrieved_ids.index(image_id) + 1
        except ValueError:
            exact_rank = None

        metrics = compute_metrics(relevant, retrieved_ids, k)

        rows.append({
            "embedding":        label,
            "source":           source,
            "image_id":         image_id,
            "file":             entry.get("file", ""),
            "category":         category,
            "query_type":       query_type,
            "query":            query_text,
            "n_relevant":       len(relevant),
            "k_final":          k,
            "k_kneedle":        k_info["k_kneedle"],
            "dist_at_k":        k_info["dist_at_k"],
            "exact_rank":       exact_rank if exact_rank is not None else "",
            "exact_hit_at_k":   1 if (exact_rank is not None and exact_rank <= k) else 0,
            **metrics,
            "latency_vlm_s":    round(entry.get("vlm_latency_s", 0.0), 3),
            "latency_chroma_s": round(chroma_latency, 4),
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#   SUMMARY AND COMPARISON TABLES
# ══════════════════════════════════════════════════════════════════════════════

METRIC_KEYS = [
    "precision_at_k", "recall_at_k", "mrr",
    "ndcg_at_k", "hit_rate_at_k", "ap_at_k",
    "exact_hit_at_k", "latency_chroma_s", "latency_vlm_s",
]


def summary_row(scope: str, label: str, subset: list) -> dict:
    """
    Aggregate a subset of result rows into a single summary row.

    Computes mean and standard deviation for every metric key, plus
    auxiliary statistics (mean K, mean number of relevants, exact rank
    statistics, percentage of query images found within the pool).

    Args:
        scope:  Grouping level label (e.g. 'GLOBAL', 'SOURCE', 'CATEGORY').
        label:  Value of the grouping key (e.g. 'gt', 'painting').
        subset: List of result row dicts belonging to this group.

    Returns:
        Single summary dict ready to be written to gt_summary.csv.
    """
    row = {
        "scope":     scope,
        "embedding": subset[0]["embedding"] if subset else "",
        "category":  label,
        "n":         len(subset),
    }
    row["mean_k"]     = round(float(np.mean([r["k_final"]    for r in subset])), 1)
    row["std_k"]      = round(float(np.std( [r["k_final"]    for r in subset])), 1)
    row["mean_n_rel"] = round(float(np.mean([r["n_relevant"] for r in subset])), 1)

    exact_ranks = [r["exact_rank"] for r in subset if isinstance(r["exact_rank"], int)]
    row["mean_exact_rank"]   = round(float(np.mean(exact_ranks)), 2) if exact_ranks else ""
    row["pct_found_in_pool"] = round(len(exact_ranks) / len(subset), 4) if subset else 0.0

    for mk in METRIC_KEYS:
        vals = [r[mk] for r in subset if isinstance(r[mk], (int, float))]
        row[f"mean_{mk}"] = round(float(np.mean(vals)), 4) if vals else 0.0
        row[f"std_{mk}"]  = round(float(np.std( vals)), 4) if vals else 0.0

    row["MAP"] = row["mean_ap_at_k"]
    return row


def build_summary(rows: list) -> list:
    """
    Build a multi-level summary table from a flat list of result rows.

    Aggregation levels produced:
        GLOBAL          — all rows combined.
        SOURCE          — split by source (gt / vlm).
        SOURCE_QTYPE    — split by source × query_type.
        CATEGORY        — split by artwork category.

    Args:
        rows: List of result row dicts from evaluate_embedding().

    Returns:
        List of summary row dicts for one embedding.
    """
    srows = [summary_row("GLOBAL", "all", rows)]

    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        if sub:
            srows.append(summary_row("SOURCE", src, sub))

    for src in sorted({r["source"] for r in rows}):
        for qt in sorted({r["query_type"] for r in rows if r["source"] == src}):
            sub = [r for r in rows if r["source"] == src and r["query_type"] == qt]
            if sub:
                srows.append(summary_row("SOURCE_QTYPE", f"{src}-{qt}", sub))

    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        if sub:
            srows.append(summary_row("CATEGORY", cat, sub))

    return srows


def build_compare_table(
    sum_rows_a: list,
    sum_rows_b: list,
    label_a:    str,
    label_b:    str,
) -> list:
    """
    Build a direct A vs B comparison table with per-metric deltas.

    For every (scope, category) combination present in either summary,
    the table contains one row with the metric values for A and B side by
    side, plus delta = A - B for each metric. A positive delta means A
    outperforms B on that metric.

    Args:
        sum_rows_a: Summary rows for embedding A.
        sum_rows_b: Summary rows for embedding B.
        label_a:    Label for embedding A (used as column suffix).
        label_b:    Label for embedding B (used as column suffix).

    Returns:
        List of comparison row dicts ready to be written to gt_compare.csv.
    """
    key_metrics = [
        "mean_precision_at_k", "mean_recall_at_k", "mean_mrr",
        "mean_ndcg_at_k", "mean_hit_rate_at_k", "MAP",
        "mean_exact_hit_at_k", "pct_found_in_pool",
        "mean_latency_chroma_s",
    ]

    index_a  = {(r["scope"], r["category"]): r for r in sum_rows_a}
    index_b  = {(r["scope"], r["category"]): r for r in sum_rows_b}
    all_keys = sorted(set(index_a) | set(index_b), key=lambda x: (x[0], x[1]))

    rows = []
    for (scope, cat) in all_keys:
        ra  = index_a.get((scope, cat), {})
        rb  = index_b.get((scope, cat), {})
        row = {
            "scope":        scope,
            "category":     cat,
            f"n_{label_a}": ra.get("n", ""),
            f"n_{label_b}": rb.get("n", ""),
        }
        for mk in key_metrics:
            va = ra.get(mk, float("nan"))
            vb = rb.get(mk, float("nan"))
            row[f"{mk}_{label_a}"] = va
            row[f"{mk}_{label_b}"] = vb
            try:
                row[f"delta_{mk}"] = round(float(va) - float(vb), 4)
            except (TypeError, ValueError):
                row[f"delta_{mk}"] = ""
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(config: dict):
    """
    Orchestrate the full evaluation pipeline.

    Steps:
        1.  Load GT entries from directed_queries.json.
        2.  Load full artwork metadata and build relevance index.
        3.  Generate VLM queries (if vlm_sample > 0).
        4.  Combine GT and VLM entries.
        5.  Connect to ChromaDB A and B.
        6.  Load embedding models A and B.
        7.  Evaluate embedding A over all entries.
        8.  Evaluate embedding B over all entries.
        9.  Write detailed results to gt_results.csv.
        10. Compute and write summary tables to gt_summary.csv.
        11. Compute and write comparison table to gt_compare.csv.
        12. Print a formatted console summary.

    Args:
        config: Configuration dict (see DEFAULT_CONFIG for all keys).
    """
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    print("\n" + "=" * 68)
    print("  DUAL EMBEDDING EVALUATION — Ground Truth + VLM + Two Embeddings")
    print("=" * 68)

    # 1. Ground truth
    print("\n> Loading ground truth...")
    gt_entries = load_gt(config["gt_json"])
    for e in gt_entries:
        e.setdefault("source", "gt")
        e.setdefault("vlm_latency_s", 0.0)

    # 2. Full metadata and relevance index
    rel_index  = None
    all_images = []
    if config.get("full_json"):
        print("\n> Loading full artwork metadata...")
        try:
            all_images = load_full_images(config["full_json"])
            rel_index  = build_relevance_index(all_images)
            print(f"  {len(all_images)} artworks indexed.")
        except FileNotFoundError:
            print(
                f"  {config['full_json']} not found. "
                "Relevance reduced to exact hit only; VLM disabled."
            )
    else:
        print("\n> full_json not configured. Exact hit only for GT; VLM disabled.")

    # 3. VLM query generation
    vlm_entries = []
    if config["vlm_sample"] > 0 and all_images:
        print(f"\n> Checking VLM availability: {config['vlm_model']} ...")
        try:
            ollama.show(config["vlm_model"])
            print("  Available.")
            print(
                f"\n> Generating VLM queries "
                f"({config['vlm_sample']} images × {len(PROMPT_STRATEGIES)} strategies)..."
            )
            vlm_entries = build_vlm_entries(all_images, rel_index or {}, config)
        except Exception as e:
            print(f"  VLM not available: {e}")
            print("  Make sure 'ollama serve' is running.")
    elif config["vlm_sample"] > 0 and not all_images:
        print("\n> VLM skipped (no full_json available).")
    else:
        print("\n> VLM disabled (vlm_sample=0).")

    # 4. Combine entries
    all_entries = gt_entries + vlm_entries
    print(
        f"\n  Total queries to evaluate: {len(all_entries)} "
        f"(GT: {len(gt_entries)}, VLM: {len(vlm_entries)})"
    )

    if not all_entries:
        print("No queries to evaluate. Check the configuration.")
        return

    # 5. ChromaDB A
    print(f"\n> Connecting to ChromaDB A ({config['label_a']})...")
    client_a     = chromadb.PersistentClient(path=config["chroma_path_a"])
    collection_a = client_a.get_collection(name=config["collection_a"])
    n_docs_a     = collection_a.count()
    print(f"  '{config['collection_a']}': {n_docs_a} documents")

    # 6. ChromaDB B
    print(f"\n> Connecting to ChromaDB B ({config['label_b']})...")
    client_b     = chromadb.PersistentClient(path=config["chroma_path_b"])
    collection_b = client_b.get_collection(name=config["collection_b"])
    n_docs_b     = collection_b.count()
    print(f"  '{config['collection_b']}': {n_docs_b} documents")

    # 7. Embedding models
    print(f"\n> Loading embedding model A: {config['embed_model_a']} ...")
    embed_a = SentenceTransformer(config["embed_model_a"])

    if config["embed_model_a"] == config["embed_model_b"]:
        print("\n> Embedding model B is identical to A — reusing.")
        embed_b = embed_a
    else:
        print(f"\n> Loading embedding model B: {config['embed_model_b']} ...")
        embed_b = SentenceTransformer(config["embed_model_b"])

    pool_size_a = min(config["pool_size"], n_docs_a)
    pool_size_b = min(config["pool_size"], n_docs_b)

    # 8. Evaluate A
    print(f"\n> Evaluating embedding A ({config['label_a']}) — {len(all_entries)} queries...\n")
    rows_a = evaluate_embedding(
        label       = config["label_a"],
        collection  = collection_a,
        embed_model = embed_a,
        entries     = all_entries,
        rel_index   = rel_index,
        pool_size   = pool_size_a,
        k_min       = config["k_min"],
    )

    # 9. Evaluate B
    print(f"\n> Evaluating embedding B ({config['label_b']}) — {len(all_entries)} queries...\n")
    rows_b = evaluate_embedding(
        label       = config["label_b"],
        collection  = collection_b,
        embed_model = embed_b,
        entries     = all_entries,
        rel_index   = rel_index,
        pool_size   = pool_size_b,
        k_min       = config["k_min"],
    )

    all_rows = rows_a + rows_b
    if not all_rows:
        print("No results produced. Check the configuration.")
        return

    # 10. Detailed CSV
    out = Path(config["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n> Detailed results  → {out}")

    # 11. Summary CSV
    sum_rows_a = build_summary(rows_a)
    sum_rows_b = build_summary(rows_b)
    all_sum    = sum_rows_a + sum_rows_b

    sum_path = Path(config["summary_csv"])
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_sum[0].keys()))
        w.writeheader()
        w.writerows(all_sum)
    print(f"> Summary           → {sum_path}")

    # 12. Comparison CSV
    compare_rows = build_compare_table(
        sum_rows_a, sum_rows_b, config["label_a"], config["label_b"]
    )
    cmp_path = Path(config["compare_csv"])
    with open(cmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        w.writeheader()
        w.writerows(compare_rows)
    print(f"> A vs B comparison → {cmp_path}\n")

    # 13. Console: global summary
    def _print_global(label: str, g: dict):
        print(f"  {'-'*28} {label} {'-'*28}")
        print(f"  Queries evaluated   : {g['n']}")
        print(f"  K (mean ± std)      : {g['mean_k']} ± {g['std_k']}")
        print(f"  Relevants (mean)    : {g['mean_n_rel']}")
        print(f"  Target in pool      : {g['pct_found_in_pool']:.1%}")
        print(f"  Exact rank (mean)   : {g.get('mean_exact_rank', 'N/A')}")
        print(f"  Exact Hit@K         : {g['mean_exact_hit_at_k']}")
        print(f"  Precision@K         : {g['mean_precision_at_k']} ± {g['std_precision_at_k']}")
        print(f"  Recall@K            : {g['mean_recall_at_k']} ± {g['std_recall_at_k']}")
        print(f"  MRR                 : {g['mean_mrr']} ± {g['std_mrr']}")
        print(f"  nDCG@K              : {g['mean_ndcg_at_k']} ± {g['std_ndcg_at_k']}")
        print(f"  Hit Rate@K          : {g['mean_hit_rate_at_k']} ± {g['std_hit_rate_at_k']}")
        print(f"  MAP                 : {g['MAP']}")
        print(f"  VLM latency         : {g['mean_latency_vlm_s']}s ± {g['std_latency_vlm_s']}s")
        print(f"  ChromaDB latency    : {g['mean_latency_chroma_s']}s ± {g['std_latency_chroma_s']}s")

    print("=" * 68)
    print("  GLOBAL SUMMARY")
    print("=" * 68)
    _print_global(config["label_a"], sum_rows_a[0])
    print()
    _print_global(config["label_b"], sum_rows_b[0])

    # 14. Console: source × query_type breakdown
    label_a, label_b = config["label_a"], config["label_b"]
    sqt_a   = {r["category"]: r for r in sum_rows_a if r["scope"] == "SOURCE_QTYPE"}
    sqt_b   = {r["category"]: r for r in sum_rows_b if r["scope"] == "SOURCE_QTYPE"}
    all_sqt = sorted(sqt_a.keys() | sqt_b.keys())

    if all_sqt:
        print("\n" + "=" * 96)
        print("  BREAKDOWN BY SOURCE × QUERY TYPE")
        print("=" * 96)
        hdr = (
            f"  {'source-query_type':<22}{'emb':<8}{'n':>4}{'K':>6}"
            f"{'P@K':>7}{'R@K':>7}{'MRR':>7}{'nDCG':>7}"
            f"{'Hit@K':>7}{'ExHit':>7}{'MAP':>7}"
        )
        print(hdr)
        print("  " + "-" * 94)
        for sqt in all_sqt:
            for lbl, rows_dict in [(label_a, sqt_a), (label_b, sqt_b)]:
                r = rows_dict.get(sqt)
                if r:
                    print(
                        f"  {sqt:<22}{lbl:<8}{r['n']:>4}{r['mean_k']:>6.1f}"
                        f"{r['mean_precision_at_k']:>7}{r['mean_recall_at_k']:>7}"
                        f"{r['mean_mrr']:>7}{r['mean_ndcg_at_k']:>7}"
                        f"{r['mean_hit_rate_at_k']:>7}{r['mean_exact_hit_at_k']:>7}"
                        f"{r['MAP']:>7}"
                    )
            print("  " + "." * 94)
        print("=" * 96)

    print("\nDual evaluation completed.")


# ══════════════════════════════════════════════════════════════════════════════
#   ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments, falling back to DEFAULT_CONFIG for any
    argument not explicitly provided.

    Returns:
        Parsed argparse.Namespace object.
    """
    p = argparse.ArgumentParser(
        description=(
            "Evaluate and compare two embedding collections using fixed GT queries "
            "and VLM-generated queries."
        )
    )
    p.add_argument("--gt-json",       default=DEFAULT_CONFIG["gt_json"])
    p.add_argument("--full-json",     default=DEFAULT_CONFIG["full_json"],
                   help="Full metadata JSON. Use 'none' to disable.")
    p.add_argument("--images-dir",    default=DEFAULT_CONFIG["images_dir"],
                   help="Root image folder (combined with img['file']).")
    p.add_argument("--vlm-sample",    type=int, default=DEFAULT_CONFIG["vlm_sample"],
                   help="Images per VLM strategy (0 = disable VLM).")
    p.add_argument("--vlm-model",     default=DEFAULT_CONFIG["vlm_model"])
    p.add_argument("--seed",          type=int, default=DEFAULT_CONFIG["random_seed"])
    p.add_argument("--category",      default=DEFAULT_CONFIG["category_filter"],
                   help="Category filter for VLM sampling (e.g. 'painting').")
    p.add_argument("--min-relevant",  type=int, default=DEFAULT_CONFIG["min_relevant"])
    # Embedding A
    p.add_argument("--chroma-path-a", default=DEFAULT_CONFIG["chroma_path_a"])
    p.add_argument("--collection-a",  default=DEFAULT_CONFIG["collection_a"])
    p.add_argument("--embed-model-a", default=DEFAULT_CONFIG["embed_model_a"])
    p.add_argument("--label-a",       default=DEFAULT_CONFIG["label_a"])
    # Embedding B
    p.add_argument("--chroma-path-b", default=DEFAULT_CONFIG["chroma_path_b"])
    p.add_argument("--collection-b",  default=DEFAULT_CONFIG["collection_b"])
    p.add_argument("--embed-model-b", default=DEFAULT_CONFIG["embed_model_b"])
    p.add_argument("--label-b",       default=DEFAULT_CONFIG["label_b"])
    # Dynamic K
    p.add_argument("--pool-size",     type=int, default=DEFAULT_CONFIG["pool_size"])
    p.add_argument("--k-min",         type=int, default=DEFAULT_CONFIG["k_min"])
    # Output
    p.add_argument("--output-csv",    default=DEFAULT_CONFIG["output_csv"])
    p.add_argument("--summary-csv",   default=DEFAULT_CONFIG["summary_csv"])
    p.add_argument("--compare-csv",   default=DEFAULT_CONFIG["compare_csv"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fj   = args.full_json
    if fj and fj.lower() in ("none", ""):
        fj = None

    config = {
        "gt_json":         args.gt_json,
        "full_json":       fj,
        "images_dir":      args.images_dir,
        "vlm_sample":      args.vlm_sample,
        "vlm_model":       args.vlm_model,
        "random_seed":     args.seed,
        "category_filter": (
            None if (args.category or "") in ("all", "", "none") else args.category
        ),
        "min_relevant":    args.min_relevant,
        "chroma_path_a":   args.chroma_path_a,
        "collection_a":    args.collection_a,
        "embed_model_a":   args.embed_model_a,
        "label_a":         args.label_a,
        "chroma_path_b":   args.chroma_path_b,
        "collection_b":    args.collection_b,
        "embed_model_b":   args.embed_model_b,
        "label_b":         args.label_b,
        "pool_size":       args.pool_size,
        "k_min":           args.k_min,
        "output_csv":      args.output_csv,
        "summary_csv":     args.summary_csv,
        "compare_csv":     args.compare_csv,
    }
    run(config)