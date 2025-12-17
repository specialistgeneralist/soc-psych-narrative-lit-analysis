# Semantic Embedding, TriMAP + HDBSCAN, and Cluster Synthesis Pipeline

This repository provides a modular, reproducible Python pipeline to:

- Ingest a TSV file where the text to embed is in column `text_normalised` (configurable).
- Compute OpenAI embedding vectors using both small and large models (configurable).
- Store embeddings locally (Parquet format) for reproducibility.
- Perform topology-preserving dimensionality reduction to 2D using TriMAP.
- Cluster high-dimensional embeddings via HDBSCAN, then project cluster labels to 2D.
- Select representative texts per cluster (closest to centroid) and summarize with Anthropic Claude.
- Save figures, metrics, and metadata for use in papers.

## Install

1) Create and activate a virtual environment, then install deps:

- Python 3.10+

`pip install -r requirements.txt`

2) Copy `.env.example` to `.env` and fill out API keys and optional overrides:

`cp .env.example .env`

Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

Review other params (model names, default data location etc.).

## Input data

Provide a TSV with at least a `text_normalised` column.

By default, the pipeline expects `data/input_example.csv`. Override via CLI `--input` or in `.env`.

## Running

Basic end-to-end run (embeddings -> TriMAP -> HDBSCAN -> visualization -> summaries):

`python -m src.main --do all --models small,large --tag demo`

- The pipeline will prefer the large embeddings for downstream steps if both are computed.
- Outputs go to `outputs/` by default (configurable in `.env`).

Run individual stages (e.g., only embeddings using the small model):

`python -m src.main --do embeddings --models small --tag small_only`

Run TriMAP and HDBSCAN using existing embeddings (computed previously):

`python -m src.main --do dimred,cluster --models large --tag vis_only`

Generate cluster summaries (requires labels from clustering):

`python -m src.main --do summarize --models large --tag demo`

To run visualisation on its own, after all the prior steps:

`python -m src.main --do visualize --models large --tag tuned_version`

## Tuning

Grid search TriMAP parameters (trustworthiness metric reported):

`python -m src.main --tune-trimap --do embeddings --models large --tag tune_tri`

Grid search HDBSCAN parameters (silhouette on core points, number of clusters, noise fraction):

`python -m src.main --tune-hdbscan --do embeddings --models large --tag tune_hdb`

Results saved under `outputs/results/*tuning*.json`.

### Ranking HDBSCAN tuning outcomes

After a tuning run, you can rank the outcomes in the .json results file with the provided script `src.tools.rank_hdbscan_results`.

What it does:

 - Reads either {"records": [...]} or a plain list of records from your JSON.
 - By default sorts lexicographically by:
    - silhouette (descending), then
    - frac_noise (ascending), then
    - n_clusters (descending).
 - Optionally uses a composite score: score = silhouette − wnoise*fracnoise + wclusters*normalized(nclusters).
 - Prints a tab-separated table and can also save the top-N to JSON.

Usage examples:

 - Rank by defaults and print top 20:

`python -m src.tools.rank_hdbscan_results tune_hdb_tuning_hdbscan.json --top 20`

 - Save top 50 to a file:

`python -m src.tools.rank_hdbscan_results tune_hdb_tuning_hdbscan.json --top 50 --out top50.json`
 - Use a composite score emphasizing low noise:

`python -m src.tools.rank_hdbscan_results tune_hdb_tuning_hdbscan.json --top 20 --composite --w-noise 0.7 --w-clusters 0.1`

Notes:

 - The script normalizes sentinel values: pcadim=-1 and minsamples=-1 are shown as None.
 - If silhouette is NaN, those rows are pushed to the bottom.

## Outputs

- Embeddings: `outputs/embeddings/embeddings_small.parquet`, `embeddings_large.parquet`
- TriMAP metrics: `outputs/results/<tag>_trimap.json`
- HDBSCAN metrics: `outputs/results/<tag>_hdbscan.json`
- 2D coordinates: `outputs/results/<tag>_coords_<model>.npy` and CSV next to the plot
- Plot: `outputs/plots/<tag>_trimap_hdbscan.png`
- Labels: `outputs/results/<tag>_labels_<model>.npy`
- Summaries: `outputs/summaries/cluster_XX_summary.md` and an index JSON

## Running specific experiments

A specific set of params for HDBSCAN/trimap etc. can be run to override defaults (in `config.py`). To do this, pass an experiment file via `--experiment`:

`python -m src.main --do all --models large --experiment experiments/my_experiment.json`

The tag will in the first instance be taken from the experiment file, but can be overridden via `--tag`.

An example experiment file is provided in `experiments/demo.json`.

## Reproducibility

- Random seeds are fixed via `SEED` in `.env` (TriMAP and general RNGs).
- All configuration, versions, and key parameters are serialized to JSON in `outputs/results`.
- Embeddings are stored verbatim (and L2-normalized copies) to ensure deterministic downstream analyses.

## Notes

- Distance metric defaults to `euclidean` for TriMAP which is equivalent to cosine on L2-normalised embeddings which we receive from OpenAI end-point. For HDBSCAN, we use `manhattan` as per optimisation experiments.
- HDBSCAN labels `-1` for noise points, which are colored grey in the 2D plot.
- Representative texts per cluster are the top-k closest to the centroid (cosine similarity) in the original embedding space.
- Trustworthiness is used as a goodness-of-fit proxy for TriMAP; silhouette on clustered points for HDBSCAN.

## Citation

If you use this pipeline, please cite the TriMAP and HDBSCAN papers and the embedding/LLM providers as appropriate.
