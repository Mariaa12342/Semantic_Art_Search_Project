#!/usr/bin/env python3
"""
Semantic Art Search Engine
==========================
Script 01 – Dataset Cleaning and Image Verification

This script performs the first stage of the data preparation pipeline:

    1. Load the raw metadata CSV from the dataset folder.
    2. Analyse and report missing values before any transformation.
    3. Clean the dataset by dropping rows with critical missing fields,
       filling non-critical nulls, and removing residual index columns.
    4. Verify that each artwork's image file is present on disk.
    5. Drop rows whose image cannot be found.
    6. Save the cleaned, image-verified DataFrame to a new CSV file.

Input:
    full_dataset/metadata_sample.csv  – Raw Rijksmuseum metadata CSV.

Output:
    metadata_clean.csv  – Cleaned CSV ready for sampling (Script 02).

Usage:
    python script_01_clean_data.py
"""

import os

import pandas as pd
from tqdm import tqdm


# ════════════════════════════════════════════════════════════
#   CONFIGURATION  ← edit these variables
# ════════════════════════════════════════════════════════════

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

# Root folder that contains the per-type image sub-directories
# (drawing/, painting/, photograph/, photomechanical print/)
BASE_DIR    = os.path.join(SCRIPT_DIR, "full_dataset")

# Path to the raw metadata CSV
CSV_PATH    = os.path.join(BASE_DIR, "metadata_sample.csv")

# Destination path for the cleaned CSV
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "metadata_clean.csv")

# ════════════════════════════════════════════════════════════


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("MISSING VALUE REPORT — BEFORE CLEANING")
print("=" * 60)
print(f"\nOriginal dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}\n")


# ── 2. MISSING VALUE ANALYSIS ────────────────────────────────────────────────

missing_counts = df.isnull().sum()
missing_pct    = (missing_counts / len(df) * 100).round(2)
print(pd.DataFrame({"Missings": missing_counts, "% of total": missing_pct}).to_string())
print(f"\nTotal cells with missing value: {missing_counts.sum()}")


# ── 3. CLEANING ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CLEANING")
print("=" * 60)

actions = []

# Drop rows where any of the three identifying fields are null.
# These records cannot be reliably linked to an image or queried.
before = len(df)
df = df.dropna(subset=["id", "title", "artist"])
dropped = before - len(df)
if dropped > 0:
    actions.append(f"→ Dropped {dropped} rows with missing 'id', 'title', or 'artist'.")

# Fill missing creation dates with a placeholder so no row is excluded
# solely because the date is unknown.
date_missing = df["date"].isnull().sum()
if date_missing > 0:
    df["date"] = df["date"].fillna("unknown")
    actions.append(f"→ {date_missing} null values in 'date' filled with 'unknown'.")

# Same strategy for artwork type.
type_missing = df["type"].isnull().sum()
if type_missing > 0:
    df["type"] = df["type"].fillna("unknown")
    actions.append(f"→ {type_missing} null values in 'type' filled with 'unknown'.")

# Remove the residual integer index column that pandas adds when a CSV is
# saved without index=False.
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    actions.append("→ Column 'Unnamed: 0' removed (residual index).")

if actions:
    for action in actions:
        print(action)
else:
    print("→ No cleaning actions required.")


# ── 4. IMAGE VERIFICATION ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("IMAGE VERIFICATION ON DISK")
print("=" * 60)


def check_image(row: pd.Series) -> str:
    """
    Check whether the image file for a given artwork exists on disk.

    The function probes three common extensions (.jpg, .jpeg, .png) under
    the sub-directory that matches the artwork's type. No pixel-level
    validation is performed; only file existence is checked to keep the
    verification fast across ~25 000 records.

    Args:
        row: A DataFrame row that must contain 'type' and 'id' fields.

    Returns:
        'ok'      – at least one matching image file was found.
        'missing' – no image file was found for any probed extension.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(BASE_DIR, str(row["type"]), str(row["id"]) + ext)
        if os.path.exists(path):
            return "ok"
    return "missing"


tqdm.pandas(desc="Verifying images")
df["image_status"] = df.progress_apply(check_image, axis=1)

status_counts = df["image_status"].value_counts()
print(f"\n{status_counts.to_string()}")

df_missing = df[df["image_status"] == "missing"]

if not df_missing.empty:
    print("\n=== MISSING (in CSV but no image on disk) ===")
    print(df_missing[["id", "type", "title"]].to_string())

    # Secondary check: re-probe with the same logic to rule out false positives
    # (e.g. unexpected casing or whitespace in the file name).
    print("\n=== FALSE MISSING CHECK ===")
    for _, row in df_missing.iterrows():
        found = any(
            os.path.exists(os.path.join(BASE_DIR, str(row["type"]), str(row["id"]) + ext))
            for ext in [".jpg", ".jpeg", ".png"]
        )
        status = "FOUND (false missing?)" if found else "NOT found on disk"
        print(f"  {row['id']} ({row['type']}) → {status}")


# ── 5. DROP ROWS WITHOUT IMAGE ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DROPPING ROWS WITHOUT IMAGE ON DISK")
print("=" * 60)

before_img   = len(df)
df           = df[df["image_status"] == "ok"].copy()
df           = df.drop(columns=["image_status"])
dropped_imgs = before_img - len(df)

print(f"→ {dropped_imgs} rows dropped (no image found on disk).")
print(f"→ Remaining rows with valid image: {len(df)}")


# ── 6. FINAL REPORT ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MISSING VALUE REPORT — AFTER CLEANING")
print("=" * 60)
print(f"\nFinal dimensions: {df.shape[0]} rows × {df.shape[1]} columns")

missing_after     = df.isnull().sum()
missing_pct_after = (missing_after / len(df) * 100).round(2)
print(pd.DataFrame({"Missings": missing_after, "% of total": missing_pct_after}).to_string())


# ── 7. SAVE ───────────────────────────────────────────────────────────────────

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Clean dataset saved at: {OUTPUT_PATH}")
print(f"  Final row count: {len(df)}")