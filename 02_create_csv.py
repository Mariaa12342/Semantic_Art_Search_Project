#!/usr/bin/env python3
"""
Semantic Art Search Engine
==========================
Script 02 – Reproducible Stratified Sampling

This script creates a balanced subset of the cleaned metadata CSV by drawing
a fixed number of samples per artwork type. The subset is used as input for
Script 03 (image copying) and subsequently for VLM description generation.

Sampling is stratified by artwork type so that every category (drawing,
painting, photograph, photomechanical print) is equally represented,
regardless of its original frequency in the dataset.

A fixed random seed guarantees that the same subset is produced on every run,
making experiments fully reproducible across machines and team members.

Input:
    metadata_clean.csv          – Cleaned CSV produced by Script 01.

Output:
    sample/metadata_sample.csv  – Stratified sample ready for Script 03.

Usage:
    python script_02_sample.py
"""

import pandas as pd


# ════════════════════════════════════════════════════════════
#   CONFIGURATION  ← edit these variables
# ════════════════════════════════════════════════════════════

# Random seed — keeps the sample identical across runs and machines.
SEED = 42

# Number of artworks to draw per artwork type.
# If a type has fewer rows than N_PER_TYPE, all its rows are included.
N_PER_TYPE = 250

# Path to the cleaned metadata CSV (output of Script 01).
CSV_PATH = "metadata_clean.csv"

# Destination path for the sampled CSV.
# The parent directory (sample/) must exist before running this script.
OUTPUT_PATH = "sample/metadata_sample.csv"

# ════════════════════════════════════════════════════════════


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print(f"SUBSET — {N_PER_TYPE} SAMPLES PER TYPE")
print("=" * 60)
print(f"\nOriginal dataset: {len(df)} rows")
print(f"\nDistribution by type:")
print(df["type"].value_counts().to_string())


# ── 2. STRATIFIED SAMPLING ───────────────────────────────────────────────────

# Draw N_PER_TYPE rows from each artwork type independently.
# Using min(N_PER_TYPE, len(subset)) prevents errors when a type has fewer
# available rows than the requested sample size.
parts = []
for artwork_type in df["type"].unique():
    subset = df[df["type"] == artwork_type]
    n      = min(N_PER_TYPE, len(subset))
    parts.append(subset.sample(n=n, random_state=SEED))

sample = pd.concat(parts).reset_index(drop=True)

print(f"\nSubset distribution (seed={SEED}, n={N_PER_TYPE} per type):")
print(sample["type"].value_counts().to_string())
print(f"\nTotal rows in subset: {len(sample)}")


# ── 3. SAVE ───────────────────────────────────────────────────────────────────

sample.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Subset saved at: {OUTPUT_PATH}")