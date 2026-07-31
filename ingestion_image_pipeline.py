#!/usr/bin/env python3
"""
Semantic Art Search Engine
==========================
Script 05 – Image Embedding Pipeline (CLIP) — Ingestion and Retrieval

This script implements the image-based search pipeline using OpenAI's CLIP
(Contrastive Language-Image Pretraining) model. Unlike the text pipeline
(Script 04 / LLaVA), this approach encodes images directly into a shared
embedding space where both images and free-text queries can be compared via
cosine similarity — no VLM description step is required.

Pipeline:
    Images ──► CLIP image encoder ──► normalised 512-dim vectors ──► ChromaDB
    Query  ──► CLIP text  encoder ──► normalised 512-dim vector  ──► cosine search

Phases:
    Ingestion  – Recursively scans the sample folder, encodes every image with
                 CLIP, and persists the normalised embeddings in ChromaDB.
    Retrieval  – Encodes a free-text prompt with the same CLIP text encoder and
                 retrieves the most similar images via cosine similarity.

Because CLIP was trained with contrastive objectives on (image, caption) pairs,
its image and text encoders share the same latent space. This means a textual
query such as "a gloomy room lit by a candle" can be directly compared against
image embeddings without any intermediate description generation.

Input:
    sample/   – Directory with per-type image sub-folders (Script 03 output).

Output:
    vector_database_2_full/  – Persistent ChromaDB collection of image embeddings.

Dependencies:
    pip install torch transformers chromadb Pillow tqdm

Usage:
    python script_05_clip_pipeline.py
"""

import os
from pathlib import Path

import chromadb
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_model(model_id: str = "openai/clip-vit-base-patch32"):
    """
    Download (on first run) or load the CLIP model and its processor from
    Hugging Face Hub.

    The base patch-32 variant produces 512-dimensional embeddings and offers
    a good balance between retrieval quality and inference speed on consumer
    GPUs.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        Tuple of (CLIPModel, CLIPProcessor).
    """
    print(f"Loading CLIP model: {model_id} ...")
    model     = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
#  INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_database(db_path: str, collection_name: str):
    """
    Open (or create) a persistent ChromaDB collection configured for cosine
    similarity search.

    ChromaDB's HNSW index with cosine space allows sub-linear nearest-neighbour
    lookups, which scales well as the number of indexed artworks grows.

    Args:
        db_path:         Local directory where ChromaDB persists its data.
        collection_name: Name of the collection to get or create.

    Returns:
        ChromaDB Collection object.
    """
    print(f"Initialising ChromaDB at: {db_path}")
    client     = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def get_image_paths(main_folder: Path, limit: int = 1_000_000) -> list[Path]:
    """
    Recursively collect all image files under *main_folder*.

    Only files with extensions .png, .jpg, or .jpeg are included.
    The result is capped at *limit* paths to support test runs on a subset
    of the dataset.

    Args:
        main_folder: Root directory to scan (e.g. the sample/ folder).
        limit:       Maximum number of paths to return.

    Returns:
        List of image Path objects, up to *limit* entries.

    Raises:
        FileNotFoundError: If no images are found under *main_folder*.
    """
    valid_extensions = {".png", ".jpg", ".jpeg"}
    image_paths = [
        p for p in main_folder.rglob("*")
        if p.suffix.lower() in valid_extensions
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in '{main_folder}' or its sub-directories."
        )

    return image_paths[:limit]


def process_and_ingest_images(
    paths: list,
    collection,
    model: CLIPModel,
    processor: CLIPProcessor,
) -> None:
    """
    Encode each image with the CLIP vision encoder and store the resulting
    normalised embedding in ChromaDB.

    Embeddings are L2-normalised so that cosine similarity reduces to a dot
    product, which ChromaDB computes efficiently via its HNSW index.

    Metadata stored alongside each embedding:
        - filename  – bare file name (used as the ChromaDB document ID).
        - filepath  – absolute path for downstream image display.
        - category  – parent directory name (artwork type).

    Images that cannot be opened or encoded are skipped with a warning rather
    than aborting the entire ingestion run.

    Args:
        paths:      List of image Path objects to process.
        collection: ChromaDB collection to insert into.
        model:      Loaded CLIPModel instance.
        processor:  Loaded CLIPProcessor instance.
    """
    ids         = []
    embeddings  = []
    metadatas   = []

    for img_path in tqdm(paths, desc="Vectorising images"):
        filename = img_path.name
        try:
            image  = Image.open(img_path).convert("RGB")
            inputs = processor(
                text=[""], images=image, return_tensors="pt", padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)

            # L2-normalise so cosine similarity equals dot product at query time.
            features            = outputs.image_embeds
            normalised_features = features / features.norm(p=2, dim=-1, keepdim=True)

            ids.append(filename)
            embeddings.append(normalised_features.squeeze().tolist())
            metadatas.append({
                "filename": filename,
                "filepath": str(img_path),
                "category": img_path.parent.name,
            })

        except Exception as e:
            print(f"⚠️  Skipping {filename}: {e}")

    if ids:
        print("\nSaving embeddings to ChromaDB...")
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"✅ Ingestion complete — {collection.count()} items in collection.")


def run_ingestion(
    db_path: str,
    collection_name: str,
    images_folder: Path,
    limit: int = 1_000_000,
) -> None:
    """
    Orchestrate the full ingestion phase: initialise the database, load CLIP,
    discover images, and encode + store them.

    Args:
        db_path:         ChromaDB persistence directory.
        collection_name: Target collection name.
        images_folder:   Root folder containing per-type image sub-directories.
        limit:           Maximum number of images to ingest.
    """
    collection        = initialize_database(db_path, collection_name)
    model, processor  = load_clip_model()

    print(f"\nSearching for images in: {images_folder}")
    paths = get_image_paths(images_folder, limit=limit)
    print(f"{len(paths)} images will be processed.")

    process_and_ingest_images(paths, collection, model, processor)


# ─────────────────────────────────────────────────────────────────────────────
#  RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def connect_to_database(db_path: str, collection_name: str):
    """
    Connect to an existing ChromaDB collection.

    Unlike initialize_database, this function uses get_collection (not
    get_or_create_collection) and will raise an exception if the collection
    does not exist, making it explicit that ingestion must be run first.

    Args:
        db_path:         ChromaDB persistence directory.
        collection_name: Name of the collection to retrieve.

    Returns:
        ChromaDB Collection object.
    """
    print(f"Connecting to ChromaDB at: {db_path}")
    client     = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    return collection


def get_text_embedding(prompt: str, model: CLIPModel, processor: CLIPProcessor) -> list:
    """
    Encode a free-text query into a normalised CLIP text embedding.

    CLIP's text and image encoders share the same latent space, so the
    resulting vector can be directly compared against image embeddings stored
    in ChromaDB using cosine similarity.

    A dummy black image is passed alongside the text because CLIPProcessor
    requires both modalities; the image input does not affect the text
    embedding computation.

    Args:
        prompt:    Free-text search query.
        model:     Loaded CLIPModel instance.
        processor: Loaded CLIPProcessor instance.

    Returns:
        L2-normalised text embedding as a plain Python list.
    """
    dummy_image = Image.new("RGB", (224, 224), (0, 0, 0))
    inputs      = processor(
        text=[prompt], images=dummy_image, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    text_features      = outputs.text_embeds
    normalised_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return normalised_features.squeeze().tolist()


def search_artworks(
    prompt: str,
    collection,
    model: CLIPModel,
    processor: CLIPProcessor,
    n_results: int = 5,
) -> None:
    """
    Run a semantic search against the ChromaDB image collection and print the
    top-K results.

    ChromaDB returns cosine distances (0 = identical, 2 = opposite). Results
    are displayed with the raw distance value alongside the file path so that
    retrieval quality can be assessed visually.

    Args:
        prompt:     Free-text search query.
        collection: ChromaDB collection to query.
        model:      Loaded CLIPModel instance.
        processor:  Loaded CLIPProcessor instance.
        n_results:  Number of top results to retrieve and display.
    """
    print(f"\n🔍 Searching: '{prompt}' ...")

    query_embedding = get_text_embedding(prompt, model, processor)
    results         = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    print(f"--- Top {n_results} Results ---")
    if not results["ids"][0]:
        print("No results found.")
        return

    for i, (artwork_id, distance, metadata) in enumerate(
        zip(results["ids"][0], results["distances"][0], results["metadatas"][0]), 1
    ):
        filepath = metadata.get("filepath", "Unknown path")
        print(f"{i}. File    : {artwork_id}")
        print(f"   Distance: {distance:.4f}  (cosine; lower = more similar)")
        print(f"   Path    : {filepath}")
        print("-" * 30)


def run_retrieval(
    db_path: str,
    collection_name: str,
    prompts: list[str],
    n_results: int = 3,
) -> None:
    """
    Orchestrate the retrieval phase: connect to the database, load CLIP, and
    run a semantic search for each prompt in *prompts*.

    Args:
        db_path:         ChromaDB persistence directory.
        collection_name: Collection to query.
        prompts:         List of free-text search queries.
        n_results:       Number of top results to display per query.
    """
    collection       = connect_to_database(db_path, collection_name)
    model, processor = load_clip_model()

    for prompt in prompts:
        search_artworks(prompt, collection, model, processor, n_results=n_results)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CURRENT_DIR     = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    DB_PATH         = str(CURRENT_DIR / "vector_database_2_full")
    COLLECTION_NAME = "rijksmuseum_artworks"

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # Skip ingestion if the database already exists to avoid re-processing all
    # images on subsequent runs. Delete the directory manually to force a full
    # re-ingestion.
    if Path(DB_PATH).exists():
        print(f"Database already found at '{DB_PATH}' — skipping ingestion.")
    else:
        SAMPLE_FOLDER = CURRENT_DIR / "sample"
        try:
            run_ingestion(DB_PATH, COLLECTION_NAME, SAMPLE_FOLDER, limit=100_000_000)
        except Exception as e:
            print(f"\n❌ Critical error during ingestion: {e}")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TEST_PROMPTS = [
        "A gloomy room lit by a candle",
        "Melancholic scene related to religion",
        "Joy",
        "Feet",
        "Why?",
        "Fruit in a vase",
        "Football",
    ]
    try:
        run_retrieval(DB_PATH, COLLECTION_NAME, TEST_PROMPTS, n_results=3)
    except Exception as e:
        print(f"\n❌ Critical error during retrieval: {e}")
