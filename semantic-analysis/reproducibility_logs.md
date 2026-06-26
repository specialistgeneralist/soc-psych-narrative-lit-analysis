# Reproducibility: Logs

NOTES:
 - Below we provide logs of actual runs of the system to produce the results and figures in the main paper.
 - This log file should be read in conjunction with the main paper, the supplementary information, and the README.md file.

## Stage 0: Obtain embeddings

AIM: Obtain .parquet compressed embedding vectors under OpenAI embedding end-point (and Google Gemini for robustness experiments below).

python -m src.main --config configs/baseline_openai.yaml --do embeddings --models large 

python -m src.main --config configs/baseline_gemini.yaml --do embeddings 


## STAGE 1: TUNE

AIM: Obtain optimal hyper-parameter configurations for HDBSCAN and TriMap.

### Baseline_gemini tune trimap / hbdscan

GEMINI / TriMAP - Run param search:

python -m src.main --tune-trimap --config configs/baseline_gemini.yaml --tag baseline_gemini_trimap_tune

.. run ranking:

python src/tools/rank_trimap_results.py outputs/results/baseline_gemini_trimap_tune_tuning_trimap.json --out outputs/results/baseline_gemini_trimap_tune_tuning_trimap_ranked.json
    rank    n_inliers       n_outliers      n_random        trustworthiness@10
    1       20      5       3       0.878940
    2       20      15      3       0.871468
    3       20      10      2       0.871065

GEMINI / HDBSCAN - Run Param search:

python -m src.main --tune-hdbscan --config configs/baseline_gemini.yaml --tag baseline_gemini_hdbscan_tune

.. run ranking:

python src/tools/rank_hdbscan_results.py outputs/results/baseline_gemini_hdbscan_tune_tuning_hdbscan.json --out outputs/results/baseline_gemini_hdbscan_tune_tuning_hdbscan_ranked.json --composite --w-noise 0.5

score   pca_dim metric  min_cluster_size        min_samples     cluster_selection_method        cluster_selection_epsilon       n_clusters      frac_noise      silhouette      silhouette_eval silhouette_eval_pca_dim
 0.58212        2       manhattan       5       8       leaf    0.0     5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       5       8       leaf    0.01    5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       5       8       leaf    0.1     5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       5       8       eom     0.0     5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       5       8       eom     0.01    5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       5       8       eom     0.1     5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       10      8       leaf    0.0     5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       10      8       leaf    0.01    5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       10      8       leaf    0.1     5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       10      8       eom     0.0     5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       10      8       eom     0.01    5.0     0.178295        0.671270        0.087989        20
 0.58212        2       manhattan       10      8       eom     0.1     5.0     0.178295        0.671270        0.087989        20
 0.56116        2       manhattan       5       12      leaf    0.0     5.0     0.255814        0.689065        0.096825        20
 0.56116        2       manhattan       5       12      leaf    0.01    5.0     0.255814        0.689065        0.096825        20
 0.56116        2       manhattan       5       12      leaf    0.1     5.0     0.255814        0.689065        0.096825        20
 0.56116        2       manhattan       5       12      eom     0.0     5.0     0.255814        0.689065        0.096825        20
 0.56116        2       manhattan       5       12      eom     0.01    5.0     0.255814        0.689065        0.096825        20
 0.56116        2       manhattan       5       12      eom     0.1     5.0     0.255814        0.689065        0.096825        20
 0.55564        2       manhattan       10      12      leaf    0.0     4.0     0.201550        0.656416        0.096321        20
 0.55564        2       manhattan       10      12      leaf    0.01    4.0     0.201550        0.656416        0.096321        20

Note: these are stable to composite weighting 0.3 or 0.7

### Baseline_openai tune trimap / hbdscan

OPENAI / TriMAP - Run param search:

python -m src.main --tune-trimap --config configs/baseline_openai.yaml --tag baseline_openai_trimap_tune

.. run ranking:

python src/tools/rank_trimap_results.py outputs/results/baseline_openai_trimap_tune_tuning_trimap.json --out outputs/results/baseline_openai_trimap_tune_tuning_trimap_ranked.json
    rank    n_inliers       n_outliers      n_random        trustworthiness@10
    1       10      10      2       0.876235
    2       10      15      5       0.875061
    3       15      10      10      0.874316
    4       10      3       2       0.873647

OPENAI / HDBSCAN - Run param search:

python -m src.main --tune-hdbscan --config configs/baseline_openai.yaml --tag baseline_openai_hdbscan_tune

.. run ranking:

python src/tools/rank_hdbscan_results.py outputs/results/baseline_openai_hdbscan_tune_tuning_hdbscan.json --out outputs/results/baseline_openai_hdbscan_tune_tuning_hdbscan_ranked.json --composite --w-noise 0.5

score   pca_dim metric  min_cluster_size        min_samples     cluster_selection_method        cluster_selection_epsilon       n_clusters      frac_noise      silhouette      silhouette_eval silhouette_eval_pca_dim
  0.62017        2       manhattan       10      4       leaf    0.0     5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       10      4       leaf    0.01    5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       10      4       leaf    0.1     5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       10      4       eom     0.0     5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       10      4       eom     0.01    5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       10      4       eom     0.1     5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       15      4       leaf    0.0     5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       15      4       leaf    0.01    5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       15      4       leaf    0.1     5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       15      4       eom     0.0     5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       15      4       eom     0.01    5.0     0.093023        0.666680        0.064115        20
  0.62017        2       manhattan       15      4       eom     0.1     5.0     0.093023        0.666680        0.064115        20
  0.60243        2       euclidean       15      4       leaf    0.0     5.0     0.100775        0.652819        0.066482        20   <--- variant 1
  0.60243        2       euclidean       15      4       leaf    0.01    5.0     0.100775        0.652819        0.066482        20
  0.60243        2       euclidean       15      4       leaf    0.1     5.0     0.100775        0.652819        0.066482        20
  0.60243        2       euclidean       15      4       eom     0.0     5.0     0.100775        0.652819        0.066482        20
  0.60243        2       euclidean       15      4       eom     0.01    5.0     0.100775        0.652819        0.066482        20
  0.60243        2       euclidean       15      4       eom     0.1     5.0     0.100775        0.652819        0.066482        20
  0.58939        2       euclidean       5       8       leaf    0.0     4.0     0.124031        0.651404        0.099747        20   <--- variant 2
  0.58939        2       euclidean       5       8       leaf    0.01    4.0     0.124031        0.651404        0.099747        20


## STAGE 2a: Main Runs :: with tuned params

AIM: Obtain main results under OpenAI and Gemini embeddings using optimal configuration from tuning.

### openai
Note:
 - we use the --save-dimred-map version so that we can save the exact 2D locations of the final coordinates
 - we can re-use these in subsequent projections.
 
python -m src.main --do dimred,cluster,summarize,visualise, --config configs/baseline_openai.yaml --trimap-n-iters 1000 --save-dimred-map
    --> Loaded config from configs/baseline_openai.yaml
    --> Loading input data from data/04112025_conceptualised_articles_103_cleaned_normalised_nocounts.csv with text column 'text_normalised' and id column 'uid'
    TRIMAP(n_inliers=10, n_outliers=10, n_random=2, distance=pre-computed, weight_temp=0.5, lr=0.1, n_iters=1000, apply_pca=True, opt_method=dbd, verbose=True, return_seq=False)
    running TriMap on 129 points with dimension 129
    using distance matrix
    running TriMap with dbd
    Iteration:  100 / 1000, Loss: 0.091, Violated triplets: 11.4683
    Iteration:  200 / 1000, Loss: 0.089, Violated triplets: 11.5443
    Iteration:  300 / 1000, Loss: 0.089, Violated triplets: 11.6659
    Iteration:  400 / 1000, Loss: 0.088, Violated triplets: 11.8103
    Iteration:  500 / 1000, Loss: 0.088, Violated triplets: 11.8103
    Iteration:  600 / 1000, Loss: 0.088, Violated triplets: 11.8559
    Iteration:  700 / 1000, Loss: 0.088, Violated triplets: 11.8787
    Iteration:  800 / 1000, Loss: 0.088, Violated triplets: 11.9319
    Iteration:  900 / 1000, Loss: 0.088, Violated triplets: 11.9547
    Iteration: 1000 / 1000, Loss: 0.088, Violated triplets: 11.9547
    Elapsed time: 0:00:00.650216
     --> Saved fixed dimensionality-reduction coordinate map to outputs/results/baseline_openai_dimred_map_large.json

### gemini

python -m src.main --do dimred,cluster,summarize,visualise --config configs/baseline_gemini.yaml --trimap-n-iters 1000
    --> Loaded config from configs/baseline_gemini.yaml
    --> Loading input data from data/04112025_conceptualised_articles_103_cleaned_normalised_nocounts.csv with text column 'text_normalised' and id column 'uid'
    TRIMAP(n_inliers=20, n_outliers=5, n_random=3, distance=pre-computed, weight_temp=0.5, lr=0.1, n_iters=1000, apply_pca=True, opt_method=dbd, verbose=True, return_seq=False)
    running TriMap on 129 points with dimension 129
    using distance matrix
    running TriMap with dbd
    Iteration:  100 / 1000, Loss: 0.125, Violated triplets: 16.2490
    Iteration:  200 / 1000, Loss: 0.124, Violated triplets: 16.5049
    Iteration:  300 / 1000, Loss: 0.123, Violated triplets: 16.7683
    Iteration:  400 / 1000, Loss: 0.123, Violated triplets: 16.9564
    Iteration:  500 / 1000, Loss: 0.123, Violated triplets: 17.0242
    Iteration:  600 / 1000, Loss: 0.123, Violated triplets: 17.0919
    Iteration:  700 / 1000, Loss: 0.123, Violated triplets: 17.1069
    Iteration:  800 / 1000, Loss: 0.123, Violated triplets: 17.1521
    Iteration:  900 / 1000, Loss: 0.123, Violated triplets: 17.1747
    Iteration: 1000 / 1000, Loss: 0.123, Violated triplets: 17.2274
    Elapsed time: 0:00:00.676705


## STAGE 2b: Variant OpenAI runs

AIM: Run the same, but with variants 1 and 2 of the top 20 configurations.

Variants 1 and 2 from the table above.

### variant 1

% python -m src.main --do dimred,cluster,summarize,visualise, --config configs/openai_variant_1.yaml --trimap-n-iters 1000
 --> Loaded config from configs/openai_variant_1.yaml
 --> Loading input data from data/04112025_conceptualised_articles_103_cleaned_normalised_nocounts.csv with text column 'text_normalised' and id column 'uid'
TRIMAP(n_inliers=10, n_outliers=3, n_random=3, distance=pre-computed, weight_temp=0.5, lr=0.1, n_iters=1000, apply_pca=True, opt_method=dbd, verbose=True, return_seq=False)
running TriMap on 129 points with dimension 129
using distance matrix
running TriMap with dbd
Iteration:  100 / 1000, Loss: 0.098, Violated triplets: 12.4501
Iteration:  200 / 1000, Loss: 0.095, Violated triplets: 12.5206
Iteration:  300 / 1000, Loss: 0.094, Violated triplets: 12.6850
Iteration:  400 / 1000, Loss: 0.093, Violated triplets: 12.5206
Iteration:  500 / 1000, Loss: 0.093, Violated triplets: 12.5675
Iteration:  600 / 1000, Loss: 0.093, Violated triplets: 12.6145
Iteration:  700 / 1000, Loss: 0.093, Violated triplets: 12.6615
Iteration:  800 / 1000, Loss: 0.093, Violated triplets: 12.6615
Iteration:  900 / 1000, Loss: 0.093, Violated triplets: 12.6615
Iteration: 1000 / 1000, Loss: 0.092, Violated triplets: 12.7320
Elapsed time: 0:00:00.612359

### variant 2

% python -m src.main --do dimred,cluster,summarize,visualise, --config configs/openai_variant_2.yaml --trimap-n-iters 1000
 --> Loaded config from configs/openai_variant_2.yaml
 --> Loading input data from data/04112025_conceptualised_articles_103_cleaned_normalised_nocounts.csv with text column 'text_normalised' and id column 'uid'
TRIMAP(n_inliers=10, n_outliers=3, n_random=3, distance=pre-computed, weight_temp=0.5, lr=0.1, n_iters=1000, apply_pca=True, opt_method=dbd, verbose=True, return_seq=False)
running TriMap on 129 points with dimension 129
using distance matrix
running TriMap with dbd
Iteration:  100 / 1000, Loss: 0.063, Violated triplets: 12.0272
Iteration:  200 / 1000, Loss: 0.060, Violated triplets: 11.6749
Iteration:  300 / 1000, Loss: 0.059, Violated triplets: 11.7689
Iteration:  400 / 1000, Loss: 0.059, Violated triplets: 12.0742
Iteration:  500 / 1000, Loss: 0.059, Violated triplets: 12.0272
Iteration:  600 / 1000, Loss: 0.058, Violated triplets: 12.0742
Iteration:  700 / 1000, Loss: 0.058, Violated triplets: 12.1682
Iteration:  800 / 1000, Loss: 0.058, Violated triplets: 12.1447
Iteration:  900 / 1000, Loss: 0.058, Violated triplets: 12.2622
Iteration: 1000 / 1000, Loss: 0.058, Violated triplets: 12.3561
Elapsed time: 0:00:00.626262

## STAG 3: Visualisations --> using fixed 2D locations for visual comparison

AIM: produce base figure  visualisations for OpenAI (baseline, variants 1 and 2), Gemini.

We saved the 2D projection from the baseline_openai condition above. We can re-use these exact positions to explore visually the impact of different clustering params on the cluster contours:

### Within OpenAI

python -m src.main --do visualise --config configs/openai_variant_1.yaml --use-dimred-map outputs/results/baseline_openai_dimred_map_large.json

python -m src.main --do visualise --config configs/openai_variant_2.yaml --use-dimred-map outputs/results/baseline_openai_dimred_map_large.json

### Beyond to Gemini embeddings:

Note, here, we should be cautious about interpretation -- the gemini topological space is not equivalent to that of the OpenAI space.  Hence, fixing 2D down-scaled projections (from OpenAI space) and over-laying the clusters (derived from Gemini space) should not be over-interpreted.

python -m src.main --do visualise --config configs/baseline_gemini.yaml --use-dimred-map outputs/results/baseline_openai_dimred_map_large.json
 --> Loaded config from configs/baseline_gemini.yaml
 --> Loading input data from data/04112025_conceptualised_articles_103_cleaned_normalised_nocounts.csv with text column 'text_normalised' and id column 'uid'
WARNING: using fixed dimensionality-reduction coordinates saved for model key 'large' with current model key 'google_gemini_embedding_001'. This is useful for visual alignment, but it is not a projection of the current embedding topology.


## STAGE 4a: Robustness Experiments (all points)
Note: doesn't include embedding comparison, this is now separated out

% python -m src.robustness.run_robustness --spec configs/robustness.yaml
 --> Loaded config from configs/baseline_openai.yaml
 --> Loaded config from configs/openai_variant_1.yaml
 --> Loaded config from configs/openai_variant_2.yaml
Robustness outputs written to outputs/robustness

## STAGE 4b: Robustness - embedddings comparisons

## Embedding based on all, centre 10, centre 5 points

python -m src.robustness.run_embedding_comparison --spec configs/embedding_comparison.yaml --representative-top-k all

python -m src.robustness.run_embedding_comparison --spec configs/embedding_comparison.yaml --representative-top-k 10

python -m src.robustness.run_embedding_comparison --spec configs/embedding_comparison.yaml --representative-top-k 5
