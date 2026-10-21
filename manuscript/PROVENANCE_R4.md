# Revision 4 — where every number comes from

All results in the Revision-4 manuscript were produced in **one** computational
environment (recorded in `results/r4/environment.json`; full package list in
`environment/requirements-lock.txt`). `reproduce_all.sh` regenerates everything;
console logs of the run used for the manuscript are in `results/r4/logs/`.

`manuscript/build_manuscript_r4.py` and `manuscript/build_response_letter_r4.py`
read every number from the files below and assert that the results still show the
pattern each sentence describes; if a result changes so that a sentence would be
false, the build stops.

## New analyses (reviewer comments, round 4)

| Comment | Manuscript item | Command | Result file |
|---|---|---|---|
| 1 | Section 4.6.1, Table 8d: direct transfer synthetic → MSU/ORNL without retraining | `python -m scripts.r4_cross_domain_transfer` | `results/r4/cross_domain_transfer.json` |
| 2 | Section 4.5.1, Table 7b, Fig. 14b: KS/PSI/confidence drift detection, re-centring, shadow retraining | `python -m scripts.r4_drift_detection` | `results/r4/drift_detection.json` |
| 2 | Section 4.5.1: single-sensor drift sensitivity limit | `python -m scripts.r4_drift_single_sensor` | `results/r4/drift_single_sensor.json` |
| 4 | Section 4.9.1, Table 10b, Fig. 16: τ sweep, calibration (ECE, Brier), nested τ selection | `python -m scripts.r4_threshold_calibration` | `results/r4/threshold_calibration.json` |
| 5 | Section 4.6.2, Table 8e: event-disjoint (recording-grouped) and time-disjoint validation; synthetic row-order check | `python -m scripts.r4_disjoint_validation` | `results/r4/disjoint_validation.json` |
| 7 | Section 4.3.1, Table 5b, Fig. 12b: SHAP deletion/insertion and remove-and-retrain faithfulness | `python -m scripts.r4_shap_faithfulness` | `results/r4/shap_faithfulness.json` |
| 8 | Section 4.6.3, Table 8f: data-complexity measures, capacity curve, controlled comparisons, generator clipping artefacts | `python -m scripts.r4_dataset_difficulty` | `results/r4/dataset_difficulty.json` |
| 9 | Section 4.1, Tables 3b and 3d: fully specified CNN-LSTM / Transformer baselines | `python -m scripts.dl_baselines` | `results/r4/dl_baselines.json` |
| 7, 8 | Section 4.6.2: nearest-neighbour / recorded-event check | `python -m scripts.r4_dataset_difficulty` | `results/r4/dataset_difficulty.json` (`msu_nn_events_*`) |
| 10 | Section 3.9: environment | `python scripts/r4_environment_report.py` | `results/r4/environment.json`, `environment/requirements-lock.txt` |

## Re-generated earlier results (same scripts as before, final environment)

| Manuscript item | Command | Output |
|---|---|---|
| Deployed model, Fig. 8, single-split accuracy | `python -m src.train` | `artifacts/metrics.json` |
| Tables 2 and 3, Fig. 9 | `python -m scripts.cv_evaluation` | `artifacts/cv_results.json`, `artifacts/statistical_significance.json` |
| Table 3c, Section 4.9, Table 10 | `python run_repeated_cv_and_policy_eval.py`, `python scripts/corrected_resampled_ttest.py` | `results/repeated_cv_and_policy_eval.json`, `results/corrected_ttest.json` |
| Fig. 9c, Section 4.3 stability | `python -m scripts.shap_stability` | `results/r4/shap_stability.json` |
| Table 4, Fig. 10 | `python -m scripts.roc_auc_analysis` | console log |
| Table 6, Fig. 13 | `python -m scripts.ablation_study` | `artifacts/ablation_study_results.json` |
| Table 7, Fig. 14 | `python -m scripts.noise_robustness_test --model lightgbm` | `artifacts/robustness_results.json` |
| Figs. 4–7 | `python -m scripts.dos_attack`, `simulate_fdi`, `simulate_overload`, `simulate_tamper` | console logs |
| Table 8, Fig. 15 | `python -m scripts.msu_ics_dataset_eval --data_dir data/external/msu_ics_power/{multiclass,triple,binary}` | `artifacts/msu_external_validation_results_*.json` |
| Tables 8b, 8c | `python -m scripts.msu_shap_analysis`, `python -m scripts.msu_ablation_study` | `results/msu_shap_multiclass.json`, `results/msu_ablation_multiclass.json` |
| Table 9, Section 4.8 | `python -m scripts.latency_benchmark --model lightgbm --n_runs 200` (machine otherwise idle) | `artifacts/latency_benchmark_results.json` |
| Section 4.9 policy latency | `python -m scripts.r4_policy_latency` (machine otherwise idle) | `results/r4/policy_latency.json` |
| Fig. 9b | `python -m scripts.r4_fig_dl_baselines` | `results/r4/figures/dl_baselines.png` |

## Data

- Synthetic dataset: `data/external/enhanced_realistic_smartgrid.csv` (gitignored), recovered from
  `predictions.csv` as described in `manuscript/PROVENANCE.md`.
- MSU/ORNL: the official host (uah.edu) is not reachable from the analysis machine. The three original
  archives (`multiclass.7z`, `triple.7z`, `binaryAllNaturalPlusNormalVsAttacks.7z`) were taken from the public
  mirror https://github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse (folder `Class/`) and
  extracted to `data/external/msu_ics_power/{multiclass,triple,binary}/`. MD5:
  multiclass.7z `671c073ed89aafb117ded08a99ae7f8d`, triple.7z `2b4ae3dc094bb472f6b9f312c4afc3f0`,
  binary `9aa81e6efb865f3f0961ad504be69b17`. That these are the files behind Table 8 is checked by
  re-running Table 8 (see `results/r4/logs/msu_eval_*.txt`).
- The triple and binary releases contain the same 78,377 rows, in the same order, as the multiclass release
  (checked in `scripts/r4_disjoint_validation.py`); in the multiclass release each scenario is one contiguous,
  time-ordered run of rows per recording file, which defines the events used for event-disjoint validation.
