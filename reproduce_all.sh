#!/usr/bin/env sh
# Regenerates every result reported in the manuscript (Revision 4) in ONE
# computational environment (see environment/requirements-lock.txt).
# Run from the project root with the pinned environment active.
# Logs go to results/r4/logs/.
set -e
PY=${PY:-python}
L=results/r4/logs
mkdir -p $L
step() { name=$1; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > $L/$name.txt 2>&1; }

# --- synthetic dataset: existing analyses -----------------------------------
step train_deployed        $PY -m src.train
step cv_evaluation         $PY -m scripts.cv_evaluation
step repeated_cv_policy    $PY run_repeated_cv_and_policy_eval.py
step corrected_ttest       $PY scripts/corrected_resampled_ttest.py
step ablation              $PY -m scripts.ablation_study
step robustness            $PY -m scripts.noise_robustness_test --model lightgbm
step roc_auc               $PY -m scripts.roc_auc_analysis
step sim_dos               $PY -m scripts.dos_attack
step sim_fdi               $PY -m scripts.simulate_fdi
step sim_overload          $PY -m scripts.simulate_overload
step sim_tamper            $PY -m scripts.simulate_tamper
step shap_stability        $PY -m scripts.shap_stability
step dl_baselines          $PY -m scripts.dl_baselines
step fig_dl_baselines      $PY -m scripts.r4_fig_dl_baselines

# --- MSU/ORNL external dataset ----------------------------------------------
for v in multiclass triple binary; do
  step msu_eval_$v         $PY -m scripts.msu_ics_dataset_eval --data_dir data/external/msu_ics_power/$v
done
step msu_shap              $PY -m scripts.msu_shap_analysis --data_dir data/external/msu_ics_power/multiclass
step msu_ablation          $PY -m scripts.msu_ablation_study --data_dir data/external/msu_ics_power/multiclass

# --- Revision-4 analyses (reviewer comments 1-8) -----------------------------
step r4_transfer           $PY -m scripts.r4_cross_domain_transfer
step r4_drift              $PY -m scripts.r4_drift_detection
step r4_drift_single       $PY -m scripts.r4_drift_single_sensor
step r4_threshold          $PY -m scripts.r4_threshold_calibration
step r4_disjoint           $PY -m scripts.r4_disjoint_validation
step r4_faithfulness       $PY -m scripts.r4_shap_faithfulness
step r4_gap                $PY -m scripts.r4_dataset_difficulty

# --- latency last, on an otherwise idle machine -------------------------------
step latency               $PY -m scripts.latency_benchmark --model lightgbm --n_runs 200
step policy_latency        $PY -m scripts.r4_policy_latency
step environment           $PY scripts/r4_environment_report.py
echo done
