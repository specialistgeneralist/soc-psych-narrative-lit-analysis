# 'Narrative' Definition Analysis Pipeline

This repository implements a pipeline for analyzing a large number of definitions of 'narrative' in high dimensional embedding space, leveraging encoder transformer models. The goal is to identify clusters of similar definitions and synthesize (with generative transformer models)representative summaries for each cluster, as well as meta-summaries across clusters.

The pipeline has two basic paths:

 1) Path 1: Semantic Embedding → TriMAP → Visualisation (with cluster colouring)

 2) Path 2: Semantic Embedding → HDBSCAN → Cluster Synthesis → Meta-Synthesis

This repository contains a modular Python pipeline that:

1) Reads a tab-separated input file (TSV) containing short texts (by default in the `text_normalised` column).

2) Computes OpenAI embedding vectors (configurable “small” and “large” models).

3) Caches embeddings locally as Parquet for reproducibility.

4) Runs TriMAP to create a 2D projection for visualization.

5) Runs HDBSCAN clustering in embedding space and overlays labels on the 2D plot.

6) Selects representative items per cluster (nearest to centroid) and generates per-cluster summaries plus an optional meta-synthesis using Anthropic models.

## Repository layout

The main entry point is `src/main.py`.

Experiment settings live in YAML files. This repo includes two config locations:

- `configs/*.yaml` (original configs)

Prompts used for summarization live in `prompts/`.

## Install

Create and activate a virtual environment (example with Python 3.11), then install dependencies:

`python3.11 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements-lock.txt`

(note, this will install specific versions of dependencies as of the last lock; to update, run `pip freeze > requirements-lock.txt` after installing new packages or updating versions)

## Configuration and secrets

1) Create a `.env` file from the example:

`cp .env.example .env`

2) Set at least:

- `OPENAI_API_KEY`

- `ANTHROPIC_API_KEY`

The code loads environment variables via `python-dotenv` (see `src/pipeline/config.py`). Do not commit `.env`.

## Input data

The CLI reads the input with `pandas.read_csv(..., sep="\t")`, so your input should be tab-separated.

You need at least a text column, and optionally an id column:

- Text column: `text_normalised` by default (configurable)

- ID column: `uid` by default (optional; if missing, row indices are used)

By default, `src/configs/baseline.yaml` points at `../../data/{...}.tsv`. Confirm the path and column names in the config before running. 

## Quickstart

After creating `.env`, run an end-to-end pipeline:

`python -m src.main --config configs/test.yaml --do all`

Notes:

- If you prefer to use the configs in `configs/`, pass them explicitly, e.g. `--config configs/baseline.yaml`.

- Outputs go to `outputs/` by default (see YAML `paths:`).

- You can optionally pass a `--tag` argument to add a custom tag to output filenames (e.g. `--tag test_run`), which can help keep outputs organized across runs. By default, if no tag is provided, then the stem of the config file is used as the tag (e.g. `baseline` for `configs/baseline.yaml`).

## Running individual stages

Compute embeddings only (large model):

`python -m src.main --config src/configs/baseline.yaml --do embeddings --models large --tag large_only`

Run TriMAP + HDBSCAN given existing embeddings:

`python -m src.main --config src/configs/baseline.yaml --do dimred,cluster vis_only`

Plot from existing coords/labels without recomputing:

`python -m src.main --config src/configs/baseline.yaml --do visualize --tag demo`

Summarize clusters (requires labels from clustering):

`python -m src.main --config src/configs/baseline.yaml --do summarize --tag demo`

## Tuning

We assume in the following embeddings already exist.

Grid-search TriMAP parameters:

`python -m src.main --config src/configs/baseline.yaml --tune-trimap --tag tune_tri`

Grid-search HDBSCAN parameters:

`python -m src.main --config src/configs/baseline.yaml --tune-hdbscan --tag tune_hdb`

Results are saved under `outputs/results/*tuning*.json`.

### Ranking TriMAP, HDBSCANtuning outcomes

After a tuning run, you can rank the outcomes in a JSON results file with:

`python src/tools/rank_trimap_results.py outputs/results/<tag>_tuning_trimap.json --top 20`

`python src/tools/rank_hdbscan_results.py outputs/results/<tag>_tuning_hdbscan.json --top 20`

Ranking for HDBSCAN by default is based on Silhouette scores only. You can optionally use a combined metric between Silhouette scores and fraction of points labelled as noise (e.g., `silhouette - noise_fraction`), which may be preferable in some cases to avoid solutions that label most points as noise. By default, the weighting is 0.5 for the noise-fraction, but you can adjust this with the `--noise-weight` argument (e.g., `--w-noise 0.7` to weight noise fraction more heavily). e.g.

`python src/tools/rank_hdbscan_results.py outputs/results/<tag>_tuning_hdbscan.json --top 20 --composite --w-noise 0.7`

## Outputs

By default the pipeline writes:

- Embeddings: `outputs/embeddings/embeddings_small.parquet`, `outputs/embeddings/embeddings_large.parquet`

- TriMAP metrics: `outputs/results/<tag>_trimap.json`

- HDBSCAN metrics: `outputs/results/<tag>_hdbscan.json`

- 2D coordinates: `outputs/results/<tag>_coords_<model>.npy` plus a CSV next to the plot

- Plot: `outputs/plots/<tag>_trimap_hdbscan.png`

- Labels: `outputs/results/<tag>_labels_<model>.npy`

- Summaries: `outputs/summaries/<tag>_cluster_XX_summary.md`, `outputs/summaries/<tag>_meta_summary.md`, plus an index JSON in `outputs/results/`

## Reproducibility

The pipeline fixes random seeds via `seed` in the YAML config.

Additionally, it writes a run metadata JSON including effective config and package versions to `outputs/results/<tag>_meta.json`.

## Notes

- Distance metric defaults to `euclidean` for TriMAP which is equivalent to cosine on L2-normalised embeddings which we receive from OpenAI end-point.

- HDBSCAN labels `-1` for noise points, which are colored grey in the 2D plot.

- Representative texts per cluster are the top-k closest to the centroid (cosine similarity) in the original embedding space.

- Trustworthiness is used as a goodness-of-fit proxy for TriMAP; silhouette on clustered points for HDBSCAN.

## Security / cost notes

This pipeline calls paid APIs (OpenAI embeddings and Anthropic text generation). Be mindful of dataset size and model choice.

Keep keys in `.env` (never commit them), and consider adding organization/project-level restrictions on your API keys.

