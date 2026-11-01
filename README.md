# FedAvg under statistical heterogeneity on PlantVillage: code, logs and predictions

This repository accompanies the paper

> A. Samdouni, K. Taji, K. Hicham, F. Ghanimi. **Statistical Heterogeneity, Resource and Energy Cost in Federated Learning for Plant Disease Diagnosis: A Multi-Seed Empirical Characterization of Federated Averaging** (submitted).
>
> Contact: Ayoub Samdouni, ayoub.samdouni@uit.ac.ma

It contains the training scripts used for the experiments, the per-round logs of all 13 runs, the test-set predictions of the 13 final models, the outputs of the post-hoc diagnostics, and the scripts that compute every number, table and figure of the results section from these files. The analysis runs on a laptop in a few seconds; no GPU and no dataset download are needed for it.

## Setup in brief

- Dataset: PlantVillage, 38 classes, 54,305 colour leaf images. Stratified split with seed 42: 46,159 training images, 8,146 test images. The test set is the same in every run.
- Model: lightweight CNN with inverted-residual blocks and CBAM-lite attention, 1,084,472 parameters, 224 × 224 input.
- Federated learning: FedAvg, 5 clients, 30 rounds. Each client trains on a fixed subset of at most 3,000 of its images, with batch size 32 and Adam (learning rate 1e-4).
- Non-IID clients: Dirichlet label partition with α = 0.5 or α = 0.1 (Hsu et al., 2019).
- Process isolation: every client update and every evaluation runs in its own short-lived process (`multiprocessing`, spawn). This keeps host memory flat across rounds.

## The 13 runs

| Scenario | Partition | Local epochs E | Seed | Log file (`results/logs/`) | Accuracy at round 30 (%) |
|---|---|---|---|---|---|
| S1 | IID | 3 | 42 | `fl_results.csv` | 97.04 |
| S2 | α = 0.5 | 3 | 42 | `fl_results_dirichlet_a05.csv` | 94.97 |
| S2 | α = 0.5 | 3 | 43 | `fl_results_dirichlet_a05_seed43.csv` | 93.43 |
| S2 | α = 0.5 | 3 | 44 | `fl_results_dirichlet_a05_seed44.csv` | 94.37 |
| S3 | α = 0.1 | 3 | 42 | `fl_results_dirichlet_a01.csv` | 84.34 |
| S3 | α = 0.1 | 3 | 43 | `fl_results_dirichlet_a01_seed43.csv` | 82.68 |
| S3 | α = 0.1 | 3 | 44 | `fl_results_dirichlet_a01_seed44.csv` | 78.17 |
| S4 | α = 0.1 | 1 | 42 | `fl_results_dirichlet_a01_epoch1.csv` | 76.39 |
| S4 | α = 0.1 | 1 | 43 | `fl_results_dirichlet_a01_epoch1_seed43.csv` | 69.92 |
| S4 | α = 0.1 | 1 | 44 | `fl_results_dirichlet_a01_epoch1_seed44.csv` | 66.28 |
| S5 | α = 0.5 | 1 | 42 | `fl_results_dirichlet_a05_e1.csv` | 80.49 |
| S5 | α = 0.5 | 1 | 43 | `fl_results_dirichlet_a05_e1_seed43.csv` | 81.23 |
| S5 | α = 0.5 | 1 | 44 | `fl_results_dirichlet_a05_e1_seed44.csv` | 81.01 |

Each log has one row per round, with these columns:
- `accuracy` and `loss` on the test set;
- `eval_time_sec` and `infer_time_per_image_ms`;
- `total_train_time_sec`, the sum over the five clients;
- `total_comm_bytes`, the bytes uploaded in the round;
- `parent_ram_gb`, the memory of the coordinating process.

## Repository layout

```
training/        scripts that produced seven of the 13 runs (named after their log files;
                 the IID run is federated_plantvillage.py)
diagnostics/
  legacy_tools.py   post-hoc checks: parameter count, frozen image order, re-evaluation of
                    saved models, BatchNorm re-estimation, per-epoch timing
  edge_profile.py   FLOPs, model size, CPU latency and TFLite (float32 / int8) accuracy
analysis/
  runs.py            list of the 13 runs and file locations
  run_statistics.py  round-level results, Welch / paired t-tests, two-way ANOVA
  per_class.py       accuracy, macro-F1, bootstrap CIs, per-class F1, collapsed classes
  mcnemar.py         exact McNemar tests with Holm correction
  energy.py          estimated client-training energy (upper bound from time x 400 W)
  make_figures.py    Figures 2-4
results/
  logs/           per-round logs of the 13 runs
  predictions/    predicted class of each of the 8,146 test images for the 13 final models,
                  plus test_labels.npz (true labels and class names, in the same order)
  diagnostics/    outputs of legacy_tools.py (re-evaluation, BatchNorm checks at rounds 5
                  and 30, epoch timing) and their logs
  summary/        tables written by the analysis scripts
figures/          figures written by make_figures.py
```

## Reproducing the results (no GPU needed)

```bash
pip install -r requirements.txt
cd analysis
python run_statistics.py   # Sections 5.1-5.3, Table 2, Table S3 -> results/summary/final_runs.csv, penalties.json
python per_class.py        # Section 5.5, Table 2, Table S4    -> results/summary/perclass.json
python mcnemar.py          # McNemar tests, Holm correction    -> results/summary/mcnemar.csv
python energy.py           # Section 5.4, Table 3 (energy)     -> results/summary/energy.csv
python make_figures.py     # Figures 2-4                       -> figures/
```

Among other things, `run_statistics.py` prints the following:
- mean accuracy at round 30: 94.25 ± 0.77% for S2 and 81.73 ± 3.19% for S3;
- Welch test between S2 and S3: p = 0.017;
- cost of E = 1 against E = 3 at equal communication: 10.9 pp (p = 0.018) at α = 0.1 and 13.3 pp (p = 0.002) at α = 0.5;
- effect at equal local computation: +3.9 pp in favour of E = 1 at α = 0.1 (p = 0.08);
- interaction in the two-way ANOVA: p = 0.50.

The analysis was tested with Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn 1.9 and Matplotlib 3.11. With these versions the outputs match the files in `results/summary/` and `figures/` exactly.

## Re-running the experiments (GPU)

The training scripts are the code that produced the runs; scripts are included for seven of the 13 runs (S1; S2 with seeds 42, 43 and 44; S4 with seeds 43 and 44; S5 with seed 44). They were run on Google Colab, mostly on an A100 GPU; some replicate runs used an L4. Working comments were removed and the console messages translated to English; the program logic is unchanged. The paths at the top of each script point to the authors' Google Drive and were set separately for each run.

1. **Get the data.** Download the colour images of PlantVillage, for example the `raw/color` folder of <https://github.com/spMohanty/PlantVillage-Dataset>. Put the `color` folder, with its 38 class sub-folders, at the path set in `DATA_PATH`.
2. **Fix the image order.** The train/test split depends on the order of the image list. The original runs listed the folder directly. Before the replicate runs, that order was frozen with:
   ```bash
   python diagnostics/legacy_tools.py freeze --data <path>/color --out <path>/legacy
   ```
   The replicate scripts (seed 43/44 of S2, seed 44 of S4 and S5) read this `listing_order.npz`. Re-evaluating the saved final models on the split rebuilt from this file reproduced every logged accuracy to within two test images:
   - for the first nine runs, see `results/diagnostics/verify_results.csv`. Its `match` column uses a one-image tolerance, and S5 with seed 42 differs by two images;
   - for the four replicate runs, compare `results/summary/perclass.json` with the logs.
3. **Run a script**, for example:
   ```bash
   python training/federated_dirichlet_a05_seed43.py
   ```
   The replicate scripts resume from the last completed round if the session is interrupted.

**About the seeds.** `SEED = 42` always controls the train/test split.
- In the seed-42 runs, the same seed also controlled the Dirichlet partition and the client subsets. Weight initialization was not seeded.
- In the seed-43/44 runs, `DIRICHLET_SEED` controls the partition, the client subsets and the weight initialization.

GPU training is not bit-for-bit deterministic, so re-runs will give close, not identical, curves.

## Diagnostics

`diagnostics/legacy_tools.py` reuses the original data pipeline and model. Here is what each subcommand was used for in the paper:

| Command | Purpose | Output in this repository |
|---|---|---|
| `params` | parameter count, upload size per client | 1,084,472 parameters, 4,337,888 bytes |
| `freeze` | save the image listing order and a hash of the test set | `results/diagnostics/listing_info.json` |
| `verify` | re-evaluate saved final models and compare with the logged accuracy; save predictions | `results/diagnostics/verify_results.csv`, `results/predictions/` |
| `bncheck` | test accuracy before and after re-estimating BatchNorm statistics | `results/diagnostics/bn_round5/`, `bn_round30/` |
| `timing` | per-epoch timing of one client update (E = 3) | `results/diagnostics/timing.json` |

The predictions of the four replicate runs added last were produced with the same functions (`build_model`, `load_into`, `eval_model`).

`diagnostics/edge_profile.py profile` measures FLOPs, file sizes and CPU latency. `edge_profile.py tflite_acc` measures the accuracy of the float32 and int8 TFLite models on the full test set. Both were run on a CPU-only Colab runtime.

## Notes

- Per-class results and McNemar tests use the predictions in `results/predictions/`, which come from re-evaluating the saved final models. The round-30 accuracies in the paper are the logged values. The two differ by at most two test images.
- The timing figures in the paper are medians over rounds 2-30 (`tt_client_med` in `results/summary/final_runs.csv`) of runs that ran on the same A100 GPU: the seed-42 runs, and the seed-43 runs at α = 0.1.
- Energy was not metered. `energy.py` multiplies the logged client training time by the maximum board power of the A100 GPU (400 W, NVIDIA data sheet), which gives an upper bound for the GPU board only, and uses the nine runs executed on the A100.
- The dataset is not redistributed here. See its source for the terms of use.

## Citation

If you use this code or these results, please cite the paper (reference to be added after publication).

## License

The code is released under the MIT License (see `LICENSE`).
