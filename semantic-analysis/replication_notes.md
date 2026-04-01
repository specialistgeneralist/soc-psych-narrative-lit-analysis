# Replication

## Full analysis run:

 - Run TriMAP tuning full grid search:

python -m src.main --tune-trimap --config configs/baseline.yaml --tag tune_trimap_tune2

 - Run HDBSCAN tuning doing full grid search and rank outcomes:

python -m src.main --tune-hdbscan --config configs/baseline.yaml --tag tune_hdbscan_tune2 
python src/tools/rank_hdbscan_results.py outputs/results/tune_hdbscan_tune2_tuning_hdbscan.json --out outputs/results/tune_hdbscan_tune2_tuning_hdbscan_ranked.json --composite

 - Run comparison weighting schemes on the results to check stability of the top-ranked result:

python src/tools/rank_hdbscan_results.py outputs/results/tune_hdbscan_tune2_tuning_hdbscan.json --out outputs/results/tune_hdbscan_tune2_tuning_hdbscan_ranked_w0.json --composite --w-noise 0.00
(.. and so on)

 - Run full clustering/summarising/vis on baseline and variants after optimisation/tuning, as final run (trimap n_iters set to 1000 for stability):

python -m src.main --do dimred,cluster,summarize,visualise --config configs/baseline.yaml --trimap-n-iters 1000
python -m src.main --do dimred,cluster,summarize,visualise --config configs/variant_1.yaml --trimap-n-iters 1000
python -m src.main --do dimred,cluster,summarize,visualise --config configs/variant_2.yaml --trimap-n-iters 1000

Details follow --

## Tuning TRIMAP

We use baseline yaml settings to run the tuning over:

```
trimap_grid:
  n_inliers: [5, 10, 15]
  n_outliers: [3, 5, 10]
  n_random: [3, 5, 10]
```

- Run TriMAP tuning doing full grid search and rank outcomes:

python -m src.main --tune-trimap --config configs/baseline.yaml --tag tune_trimap_tune2
python src/tools/rank_trimap_results.py outputs/results/tune_trimap_tune2_tuning_trimap.json --out outputs/results/tune_trimap_tune2_tuning_trimap_ranked.json

rank    n_inliers       n_outliers      n_random        trustworthiness@10
1       10      3       3       0.878189   ** baseline settings
2       15      10      10      0.876474
3       15      3       10      0.874439
4       15      5       10      0.873845
5       15      3       5       0.873613
6       15      10      3       0.872096
7       10      10      3       0.869242
8       10      3       10      0.868579
9       15      10      5       0.867958
10      10      5       3       0.867534
11      10      10      10      0.865376
12      15      5       3       0.865212
13      15      3       3       0.863750
14      10      10      5       0.861401
15      10      5       10      0.861012
16      10      5       5       0.860615
17      15      5       5       0.856340
18      5       10      3       0.855370
19      5       3       5       0.852495
20      10      3       5       0.852160


## Tuning HDBSCAN

We use the baseline.yaml settings to run the tuning over:

```
hdbscan_grid:
  min_cluster_size: [5, 10, 15, 20, 25]
  min_samples: [1, 2, 4, 8]
  selection_method: [leaf, eom]
  cluster_selection_epsilon: [0.0, 0.01, 0.1]
  allow_single_cluster: [false]
  metric: [euclidean, manhattan]
  pca_dim: [null, 2, 3, 4, 5]
```

Notes:

 - pca_dim null means no PCA, i.e. clustering in the original (high-D) embedding space.


Run tuning:

% python -m src.main --tune-hdbscan --config configs/baseline.yaml --tag tune_hdbscan_tune2 

Rank the results (top 20):

 - note: we use `--composite` which, by default, subtracts a some constant times the noise fraction to build a `score`. By default, this noise weighting is 0.5, effectively meaning that the score is `silhouette - 0.5 * frac_noise`. In other words, 2/3 Silhouette (higher better), and 1/3 fraction of points allocated to no cluster (noise).

% python src/tools/rank_hdbscan_results.py outputs/results/tune_hdbscan_tune2_tuning_hdbscan.json --out outputs/results/tune_hdbscan_tune2_tuning_hdbscan_ranked.json --composite

score   pca_dim metric  min_cluster_size        min_samples     cluster_selection_method        cluster_selection_epsilon       n_clusters      frac_noise      silhouette      silhouette_eval silhouette_eval_pca_dim
 0.61884        2       euclidean       25      1       leaf    0.0     3.0     0.069767        0.653727        0.124494        20 ** baseline
 0.61884        2       euclidean       25      1       leaf    0.01    3.0     0.069767        0.653727        0.124494        20
 0.61884        2       euclidean       25      1       leaf    0.1     3.0     0.069767        0.653727        0.124494        20
 0.61884        2       euclidean       25      2       leaf    0.0     3.0     0.069767        0.653727        0.124494        20
 0.61884        2       euclidean       25      2       leaf    0.01    3.0     0.069767        0.653727        0.124494        20
 0.61884        2       euclidean       25      2       leaf    0.1     3.0     0.069767        0.653727        0.124494        20
 0.61535        2       manhattan       10      4       leaf    0.0     5.0     0.085271        0.657988        0.065506        20 ** variant_1
 0.61535        2       manhattan       10      4       leaf    0.01    5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       10      4       leaf    0.1     5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       10      4       eom     0.0     5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       10      4       eom     0.01    5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       10      4       eom     0.1     5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       15      4       leaf    0.0     5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       15      4       leaf    0.01    5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       15      4       leaf    0.1     5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       15      4       eom     0.0     5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       15      4       eom     0.01    5.0     0.085271        0.657988        0.065506        20
 0.61535        2       manhattan       15      4       eom     0.1     5.0     0.085271        0.657988        0.065506        20
 0.60953        2       manhattan       10      1       leaf    0.0     5.0     0.069767        0.644418        0.062186        20 ** variant_2
 0.60953        2       manhattan       10      1       leaf    0.01    5.0     0.069767        0.644418        0.062186        20

### Influence of noise weighting

In the following we show the top ranked (by `score`) results for different noise weightings (0.0, 0.25, 0.5, 0.75, 1.0, 1.25) in the composite score. We can see that, for weighting greater than 0.25, the top-ranked result settles to the 0.5 weighting set. Below this level, we have 21% of records in the noise (unclustered set) - or around a fifth of the surveys. For a small drop in Silhouette score, we can accomplish a dramatically improved coverage with the baseline settings which produces 3 clusters, and is stable for a large range of noise weightings.

0.00: 0.69605        2       manhattan       5       8       leaf    0.0     6.0     0.209302        0.696045        0.035924        20
0.25: 0.64372        2       manhattan       5       8       leaf    0.0     6.0     0.209302        0.696045        0.035924        20
0.50: 0.61884        2       euclidean       25      1       leaf    0.0     3.0     0.069767        0.653727        0.124494        20
0.75: 0.60140        2       euclidean       25      1       leaf    0.0     3.0     0.069767        0.653727        0.124494        20
1.00: 0.58396        2       euclidean       25      1       leaf    0.0     3.0     0.069767        0.653727        0.124494        20
1.25: 0.56652        2       euclidean       25      1       leaf    0.0     3.0     0.069767        0.653727        0.124494        20

Run each with e.g.:

% python src/tools/rank_hdbscan_results.py outputs/results/tune_hdbscan_tune2_tuning_hdbscan.json --out outputs/results/tune_hdbscan_tune2_tuning_hdbscan_ranked.json --composite --w-noise 0.25

### Discussion

Turning back to the main results table above, we can see that there are three distinct blocks in the top-20 (by score) results (focussing on the frac_noise and silhouette score cols).

All top 20 results have pca_dim=2, indicating that a 2D rendering of the space facilitates better clustering than the original high-D space which, at over 3000 dimensions, and only 130 approx. records, may be too sparse for effective clustering. For other parameters, the top 20 comprises both euclidean or manhattan, min_cluster_size ranging from 10 to 25, min_samples ranging from 1 to 4, leaf and eom, and cluster_selection_epsilon ranging over 3 to 5. In other words, despite very little change in Silhouette score, there is large variation possible in the input parameters, suggesting that the clustering identified is robust to specific choices of paramaters.

For completeness, we study a full clustering and labelling run from each of the three regimes above. We sample the top ranked item from each of the three regimes noting that the specific choice of item within regime is not important since for a given set of parameters there is a convergent outcome.


## Run final baseline pipeline:

Here, for the final visualisation, we take the TriMAP iteration count out to 1000 (rather than default 400) to ensure convergence and stability of the visualisation.

Note: here we provide no --tag since we want to strictle respect the config file base-name as the output handle (see code for details)

python -m src.main --do dimred,cluster,summarize,visualise --config configs/baseline.yaml  --trimap-n-iters 1000
python -m src.main --do dimres,cluster,summarize,visualise --config configs/variant_1.yaml --trimap-n-iters 1000
python -m src.main --do dimres,cluster,summarize,visualise --config configs/variant_2.yaml --trimap-n-iters 1000

Final baseline logs:

```
% python -m src.main --do dimred,cluster,summarize,visualise --config configs/baseline.yaml --trimap-n-iters 1000
TRIMAP(n_inliers=10, n_outliers=3, n_random=3, distance=euclidean, weight_temp=0.5, lr=0.1, n_iters=1000, apply_pca=True, opt_method=dbd, verbose=True, return_seq=False)
running TriMap on 129 points with dimension 3072
pre-processing
applied PCA
found nearest neighbors
sampled triplets
running TriMap with dbd
Iteration:  100 / 1000, Loss: 0.092, Violated triplets: 12.2387
Iteration:  200 / 1000, Loss: 0.089, Violated triplets: 12.4971
Iteration:  300 / 1000, Loss: 0.088, Violated triplets: 12.4031
Iteration:  400 / 1000, Loss: 0.087, Violated triplets: 12.4031
Iteration:  500 / 1000, Loss: 0.086, Violated triplets: 12.5440
Iteration:  600 / 1000, Loss: 0.086, Violated triplets: 12.5206
Iteration:  700 / 1000, Loss: 0.086, Violated triplets: 12.5910
Iteration:  800 / 1000, Loss: 0.086, Violated triplets: 12.6850
Iteration:  900 / 1000, Loss: 0.086, Violated triplets: 12.7085
Iteration: 1000 / 1000, Loss: 0.086, Violated triplets: 12.7320
Elapsed time: 0:00:01.514706
```

### Discussion

One would now compare the plots across baseline vs. variants, and the cluster summaries. Ideally we can map cluster summaries from the n=5 cases back to the n=3.