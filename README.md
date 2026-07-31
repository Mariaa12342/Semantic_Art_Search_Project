# Semantic Art Search Engine — Data Preprocessing Pipeline

Scripts for cleaning, sampling, and assembling the Rijksmuseum dataset before VLM description generation. This preprocessing pipeline must be executed in order before running the main RAG pipeline.

> Built for the **Modelling Week MS 2026** project (Management Solutions).

---

## Pipeline Overview

```
metadata.csv  +  full image dataset
  └─► Script 01 ── Clean & verify ──► metadata_clean.csv
        └─► Script 02 ── Stratified sample ──► sample/metadata_sample.csv
              └─► Script 03 ── Copy images ──► sample/
                    └─► Script 04 ── EDA & visualisations
```

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| 01 – Clean | `full_dataset/metadata_sample.csv` | `metadata_clean.csv` | Remove nulls, verify images on disk |
| 02 – Sample | `metadata_clean.csv` | `sample/metadata_sample.csv` | Stratified sample by artwork type |
| 03 – Build sample | `sample/metadata_sample.csv` + images | `sample/` | Assemble self-contained image folder |
| 04 – EDA | `full_results.json` | Visualisations | Exploratory analysis of VLM output |

---

## How It Works

**Script 01** loads the raw metadata CSV, reports missing values, drops rows with critical nulls (`id`, `title`, `artist`), fills non-critical nulls (`date`, `type`) with `"unknown"`, removes residual index columns, and verifies that each artwork's image file is present on disk. Rows without an image are dropped. The result is a clean, image-verified CSV.

**Script 02** reads the cleaned CSV and draws a fixed number of artworks per type using stratified random sampling with a fixed seed (`SEED = 42`). This guarantees that every artwork category is equally represented and that the same subset is reproduced on every run across all machines.

**Script 03** builds the self-contained `sample/` directory by creating one sub-folder per artwork type and copying each sampled image into it. The metadata CSV is also copied so the folder can be transferred to other machines as-is.

**Script 04** performs exploratory data analysis on the merged VLM output JSON once all machines have finished processing, producing four visualisations: artwork frequency by category, temporal evolution by century, thematic clustering via Louvain community detection, and a semantic word cloud.

---

## Requirements

```bash
pip install pandas tqdm matplotlib seaborn networkx python-louvain
pip install wordcloud nltk    # required only for Script 04 word cloud
```

---

## Dataset Layout

The full dataset must be organised as follows before running Script 01:

```
full_dataset/
├── drawing/
│   ├── RP-T-1234.jpg
│   └── ...
├── painting/
│   └── ...
├── photograph/
│   └── ...
├── photomechanical print/
│   └── ...
└── metadata_sample.csv
```

The image filename stem (e.g. `RP-T-1234`) must match the `id` column in the metadata CSV.

---

## Configuration

Each script has a `CONFIGURATION` block at the top. The key variables are:

**Script 01**

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | `<script_dir>/full_dataset` | Root folder with per-type image sub-directories |
| `CSV_PATH` | `BASE_DIR/metadata_sample.csv` | Raw metadata CSV |
| `OUTPUT_PATH` | `<script_dir>/metadata_clean.csv` | Cleaned CSV output path |

**Script 02**

| Variable | Default | Description |
|----------|---------|-------------|
| `SEED` | `42` | Random seed for reproducibility |
| `N_PER_TYPE` | `250` | Number of artworks to sample per type |
| `CSV_PATH` | `metadata_clean.csv` | Cleaned CSV from Script 01 |
| `OUTPUT_PATH` | `sample/metadata_sample.csv` | Sampled CSV output path |

**Script 03**

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | `"."` | Root folder containing the original per-type image sub-directories |
| `SAMPLE_CSV` | `sample/metadata_sample.csv` | Sampled CSV from Script 02 |
| `OUTPUT_DIR` | `sample` | Destination directory for the assembled sample |

**Script 04**

| Variable | Default | Description |
|----------|---------|-------------|
| `JSON_PATH` | `full_results.json` | Merged VLM output JSON |
| `GRAPH_THRESHOLD` | `20` | Minimum co-occurrence count for tag pairs in the network graph |
| `NETWORK_STOPWORDS` | `{art, artwork, …}` | Tags excluded from the co-occurrence network |

---

## Running the Pipeline

### Step 1 — Clean the metadata

```bash
python script_01_clean_data.py
```

Produces `metadata_clean.csv` in the same directory as the script.

### Step 2 — Create a stratified sample

```bash
python script_02_sample.py
```

Produces `sample/metadata_sample.csv`. The `sample/` directory must exist beforehand:

```bash
mkdir sample
```

### Step 3 — Assemble the sample folder

```bash
python script_03_build_sample.py
```

Copies the sampled images into `sample/<type>/` and places the metadata CSV inside `sample/`. The folder is now self-contained and ready to be transferred to other machines.

### Step 4 — Exploratory analysis (after VLM processing)

Run this script once all machines have finished generating descriptions and the JSON outputs have been merged into `full_results.json`:

```bash
python script_04_eda.py
```

---

## Output Structure

After running Scripts 01–03 the working directory will contain:

```
.
├── script_01_clean_data.py
├── script_02_sample.py
├── script_03_build_sample.py
├── script_04_eda.py
├── metadata_clean.csv          ← Script 01 output
└── sample/
    ├── metadata_sample.csv     ← Script 02 & 03 output
    ├── drawing/
    ├── painting/
    ├── photograph/
    └── photomechanical print/
```

---

## Distributed Processing Note

The sample folder produced by Script 03 is designed to be split across multiple machines for parallel VLM description generation. Each machine receives a subset of the images and runs the LLaVA pipeline independently. The resulting JSON files are merged into `full_results.json` before running Script 04.

---

## Authors

Juan, María, Laura, Andrés, Daniel, Gabriel & Javier — Máster en Ingeniería Matemática, Universidad Complutense de Madrid  
Modelling Week MS 2026 · Management Solutions
