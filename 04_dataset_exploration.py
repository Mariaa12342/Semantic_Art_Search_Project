#!/usr/bin/env python3
"""
Semantic Art Search Engine
==========================
Script 04 – Exploratory Data Analysis and Vector Clustering

This script performs a deep dive into the Rijksmuseum dataset through:
    1. Metadata Analysis: Temporal and categorical distributions.
    2. NLP & Networks: Tag co-occurrence networks with Louvain community 
       detection and lemmatized word clouds.
    3. Computer Vision Analysis: Visual clustering of embeddings (CLIP/LLaVA) 
       using KMeans and PCA, including cluster-specific image sampling.

Key Features:
    - Deterministic: Fixed random seeds ensure reproducible results.
    - Robust: Handles both string and dictionary-based tags from various VLMs.
    - Compatible: Supports English and Spanish JSON key conventions.

Usage:
    python 04_dataset_exploration.py
"""

import json
import os
import re
import sys
from collections import Counter
from itertools import combinations

import chromadb
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import STOPWORDS, WordCloud

# ─────────────────────────────────────────────────────────────────────────────
#  1. GLOBAL CONFIGURATION & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

# Fix random seed for reproducibility across all libraries
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

JSON_INPUT = "full_results.json"
IMAGE_ROOT = "./rijksmuseum"
NETWORK_STOPWORDS = {"art", "artwork", "photography", "image", "vintage", "illustration"}

def extract_tag_string(tag) -> str:
    """Safely extracts a string from a tag entry (handles both str and dict)."""
    if isinstance(tag, str):
        return tag.strip().lower()
    elif isinstance(tag, dict):
        for key in ("tag", "name", "label", "value"):
            if key in tag:
                return str(tag[key]).strip().lower()
        return " ".join(str(v) for v in tag.values()).strip().lower()
    return ""

def process_date(date_str) -> int:
    """Extracts a representative year from a raw date string via regex."""
    if not date_str or pd.isna(date_str):
        return None
    years = re.findall(r"\b\d{4}\b", str(date_str))
    if len(years) == 1:
        return int(years[0])
    elif len(years) > 1:
        return int(sum(map(int, years)) / len(years))
    return None

def resolve_image_path(filepath, image_id, root_dir=IMAGE_ROOT):
    """Fallback strategy to locate image files if metadata paths are missing."""
    if filepath and os.path.exists(filepath):
        return filepath
    if filepath:
        filename = os.path.basename(filepath)
        for root, _, files in os.walk(root_dir):
            if filename in files:
                return os.path.join(root, filename)
    if image_id:
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"):
            candidate = f"{image_id}{ext}"
            for root, _, files in os.walk(root_dir):
                if candidate in files:
                    return os.path.join(root, candidate)
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  2. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

try:
    with open(JSON_INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {JSON_INPUT}")
except FileNotFoundError:
    print(f"❌ ERROR: {JSON_INPUT} not found.")
    sys.exit()

records = []
categories_data = data.get("categorias", data.get("categories", {}))

for cat_name, content in categories_data.items():
    images = content.get("images", content.get("imagenes", []))
    for img in images:
        analysis = img.get("analysis", img.get("analisis", {}))
        records.append({
            "category": cat_name,
            "id": img.get("id"),
            "title": img.get("title", img.get("titulo")),
            "date": img.get("date", img.get("fecha")),
            "tags": analysis.get("etiquetas", []),
            "filepath": img.get("filepath", "")
        })

df = pd.DataFrame(records)
df["estimated_year"] = df["date"].apply(process_date)
df["century"] = df["estimated_year"].apply(
    lambda x: f"{int(x // 100 + 1)}th C." if pd.notna(x) else "Unknown"
)

# ─────────────────────────────────────────────────────────────────────────────
#  3. VISUALIZATION: FREQUENCY & TEMPORAL EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

# --- Category Distribution ---
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df, y="category", hue="category",
    order=df["category"].value_counts().index,
    palette="Set2", legend=False
)
plt.title("Artwork Frequency by Category")
plt.tight_layout()
plt.show()

# --- Temporal Heatmap ---
if df["estimated_year"].notna().any():
    temporal = df[df["century"] != "Unknown"].groupby(["century", "category"]).size().unstack(fill_value=0)
    if not temporal.empty:
        plt.figure(figsize=(12, 6))
        sns.heatmap(temporal, annot=True, fmt="d", cmap="YlGnBu")
        plt.title("Temporal Evolution: Categories across Centuries")
        plt.tight_layout()
        plt.show()

# ─────────────────────────────────────────────────────────────────────────────
#  4. NLP ANALYSIS: NETWORK & WORD CLOUD
# ─────────────────────────────────────────────────────────────────────────────

all_tags_list = df["tags"].dropna().tolist()

# --- Thematic Clustering (Network) ---
tag_pairs = []
for tags in all_tags_list:
    clean = [extract_tag_string(t) for t in tags]
    clean = [t for t in clean if t and t not in NETWORK_STOPWORDS]
    if len(clean) > 1:
        tag_pairs.extend(list(combinations(sorted(set(clean)), 2)))

filtered_pairs = {k: v for k, v in Counter(tag_pairs).items() if v >= 20}

if filtered_pairs:
    G = nx.Graph()
    for (n1, n2), w in filtered_pairs.items():
        G.add_edge(n1, n2, weight=w)

    # DETERMINISTIC Louvain Partition
    partition = community_louvain.best_partition(G, random_state=RANDOM_SEED)
    
    plt.figure(figsize=(15, 11))
    pos = nx.spring_layout(G, k=1.0, iterations=100, seed=RANDOM_SEED)
    
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=list(partition.values()), cmap=plt.cm.tab10, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=[G[u][v]["weight"] * 0.2 for u, v in G.edges()], edge_color="gainsboro", alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    
    plt.title("Thematic Tag Clustering (Louvain Community Detection)")
    plt.axis("off")
    plt.show()

# --- Word Cloud ---
try:
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    processed_words = []
    for tags in all_tags_list:
        for t in tags:
            w = extract_tag_string(t)
            if w and w not in NETWORK_STOPWORDS:
                processed_words.append(lemmatizer.lemmatize(w))
    
    if processed_words:
        # DETERMINISTIC WordCloud
        wc = WordCloud(width=1000, height=500, background_color="white", 
                       random_state=RANDOM_SEED, collocations=False).generate(" ".join(processed_words))
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()
except Exception as e:
    print(f"⚠️ WordCloud skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  5. VECTOR ANALYSIS: CHROMADB CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def analyze_chroma_clusters(db_path, collection_name, n_clusters=5):
    """Performs deterministic clustering and visual sampling from ChromaDB."""
    if not os.path.exists(db_path):
        print(f"⚠️ DB path {db_path} not found. Skipping.")
        return

    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(name=collection_name)
    except:
        return

    data = collection.get(include=["embeddings", "metadatas"])
    X = np.array(data["embeddings"])
    metas = data["metadatas"]
    ids = data["ids"]

    # 1. Deterministic KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
    cluster_labels = kmeans.fit_predict(X)

    # 2. PCA Visualization
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=cluster_labels, palette="tab10", s=60, alpha=0.7)
    plt.title(f"Visual Cluster PCA: {collection_name}")
    plt.show()

    # 3. Deterministic Visual Sampling
    for c_id in range(n_clusters):
        cluster_indices = [i for i, l in enumerate(cluster_labels) if l == c_id]
        if not cluster_indices: continue
        
        # Consistent sampling
        sample_size = min(5, len(cluster_indices))
        selected = np.random.RandomState(RANDOM_SEED).choice(cluster_indices, sample_size, replace=False)
        
        fig, axes = plt.subplots(1, sample_size, figsize=(15, 3))
        fig.suptitle(f"Cluster #{c_id} Representative Samples")
        if sample_size == 1: axes = [axes]
        
        for ax, idx in zip(axes, selected):
            img_path = resolve_image_path(metas[idx].get("filepath"), ids[idx])
            try:
                if img_path:
                    ax.imshow(Image.open(img_path))
                else:
                    ax.text(0.5, 0.5, "Image Not Found", ha='center')
            except:
                ax.text(0.5, 0.5, "Load Error", ha='center')
            ax.axis("off")
        plt.show()

# Run Analysis for both pipelines
analyze_chroma_clusters("./vector_database_2_full", "rijksmuseum_artworks") # CLIP
analyze_embeddings_llava = analyze_chroma_clusters("./vector_database_1_full", "rijksmuseum_llava")

print("✨ Exploration and Analysis Complete.")