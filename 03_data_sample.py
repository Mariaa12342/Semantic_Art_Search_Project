#!/usr/bin/env python3
"""
Semantic Art Search Engine
==========================
Script 03 – Sample Folder Assembly

This script builds the self-contained sample directory that every machine
will use as input for VLM description generation (Script 04 / LLaVA pipeline).

It reads the sampled metadata CSV produced by Script 02, creates one
sub-directory per artwork type inside the output folder, and copies each
artwork's image file from the full dataset into the corresponding sub-directory.
The metadata CSV is also copied into the output folder so that the sample
directory is fully self-contained and can be transferred to other machines.

Output folder structure:
    sample/
    ├── metadata_sample.csv
    ├── drawing/
    ├── painting/
    ├── photograph/
    └── photomechanical print/

Input:
    sample/metadata_sample.csv  – Sampled CSV produced by Script 02.
    ./<type>/<id>.jpg            – Full-resolution images from the dataset.

Output:
    sample/                      – Self-contained sample directory.

Usage:
    python script_03_build_sample.py
"""

import os
import shutil

import pandas as pd


# ════════════════════════════════════════════════════════════
#   CONFIGURATION  ← edit these variables
# ════════════════════════════════════════════════════════════

# Root folder containing the original per-type image sub-directories.
# Use "." if this script is run from the same directory as drawing/, painting/, etc.
BASE_DIR = "."

# Path to the sampled metadata CSV (output of Script 02).
SAMPLE_CSV = "sample/metadata_sample.csv"

# Destination directory where the sample folder will be assembled.
OUTPUT_DIR = "sample"

# ════════════════════════════════════════════════════════════


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

df = pd.read_csv(SAMPLE_CSV)

print("=" * 60)
print("SAMPLE FOLDER ASSEMBLY")
print("=" * 60)
print(f"\nTotal artworks to copy: {len(df)}")
print(f"Types: {df['type'].unique().tolist()}")


# ── 2. CREATE FOLDER STRUCTURE ───────────────────────────────────────────────

# Create the output root and one sub-directory per artwork type.
# exist_ok=True makes the script safe to re-run without raising errors
# if the directories already exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)
for artwork_type in df["type"].unique():
    os.makedirs(os.path.join(OUTPUT_DIR, artwork_type), exist_ok=True)

print(f"\n✓ Sub-directories created in: {OUTPUT_DIR}")


# ── 3. COPY IMAGES ───────────────────────────────────────────────────────────

# Probe extensions in order of likelihood; stop at the first match.
# shutil.copy2 preserves the original file metadata (timestamps, permissions).
EXTENSIONS = [".jpg", ".jpeg", ".png"]

copied      = 0
not_found   = []

for _, row in df.iterrows():
    artwork_type = row["type"]
    art_id       = row["id"]

    # Locate the source file by probing each supported extension.
    source = None
    for ext in EXTENSIONS:
        candidate = os.path.join(BASE_DIR, artwork_type, art_id + ext)
        if os.path.exists(candidate):
            source = candidate
            break

    if source is None:
        not_found.append(art_id)
        continue

    destination = os.path.join(OUTPUT_DIR, artwork_type, os.path.basename(source))
    shutil.copy2(source, destination)
    copied += 1

print(f"\n✓ Images copied: {copied} / {len(df)}")
if not_found:
    print(f"⚠  Not found ({len(not_found)}): {not_found}")


# ── 4. COPY CSV ───────────────────────────────────────────────────────────────

# Include the metadata CSV in the output folder so the sample directory
# is fully self-contained and can be transferred to other machines as-is.
shutil.copy2(SAMPLE_CSV, os.path.join(OUTPUT_DIR, "metadata_sample.csv"))
print(f"✓ Metadata CSV copied to: {OUTPUT_DIR}")


# ── 5. FINAL STRUCTURE REPORT ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL STRUCTURE")
print("=" * 60)
for artwork_type in df["type"].unique():
    folder = os.path.join(OUTPUT_DIR, artwork_type)
    count  = len(os.listdir(folder))
    print(f"  {artwork_type}/  →  {count} images")
print(f"  metadata_sample.csv")