# Revision 3 — where every new number comes from

Every number added in this revision is read by `manuscript/build_manuscript.py` and
`manuscript/build_response_letter.py` from the result files below, never typed by hand.
The build also checks that the results still show the pattern the text describes. If a
result file changes so that a sentence would no longer be true, the build stops with an
error instead of producing a wrong document.

| Manuscript item | Command (run from the project root) | Result file / console log |
|---|---|---|
| Table 3c, Section 3.6.1: repeated CV (10 seeds, n = 50) and Wilcoxon tests | `python run_reviewer_experiments.py` | `results/reviewer_experiments.json`, `results/reviewer_experiments_output.txt` |
| Table 3c: corrected resampled t-test (Nadeau–Bengio) | `python scripts/corrected_resampled_ttest.py` | `results/corrected_ttest.json` |
| Section 4.9, Table 10: response accuracy, false-block rate, missed-response rate, policy latency | `python run_reviewer_experiments.py` (seed-42 out-of-fold predictions passed through `src/api.py::compute_preventive_action`) | `results/reviewer_experiments.json` → `prevention_benchmark` |
| Section 4.7, Table 8b: MSU/ORNL SHAP ranking and group shares | `python -m scripts.msu_shap_analysis --data_dir data/external/msu_ics_power/multiclass` | `results/msu_shap_multiclass.json`, `results/msu_shap_multiclass_output.txt` |
| Section 4.7, Table 8c: MSU/ORNL feature-group ablation | `python -m scripts.msu_ablation_study --data_dir data/external/msu_ics_power/multiclass` | `results/msu_ablation_multiclass.json`, `results/msu_ablation_multiclass_output.txt` |

## Reproduction checks (built into the scripts)

- Seed 42 of the repeated CV runs the same protocol as `scripts/cv_evaluation.py`. It reproduces
  every value in Table 2 and every Wilcoxon result in Table 3 exactly (see the reproduction check
  in `results/reviewer_experiments_output.txt`).
- Both MSU/ORNL scripts first re-run the Table 8 LightGBM 37-class pipeline, which reproduces
  0.8307 ± 0.0022 exactly, before computing SHAP values or ablations.

## Data

- `data/external/enhanced_realistic_smartgrid.csv` is gitignored. It was recovered from
  `predictions.csv` by dropping the `prediction` column. The `overlo` label spelling in that
  file is normalized to `overload`, the spelling used by the deployed label encoder and
  `src/api.py`. The exact reproduction of Tables 2 and 3 confirms that this is the original
  dataset.
- The MSU/ORNL multiclass data (gitignored) is the public `multiclass.7z` from
  http://www.ece.uah.edu/~thm0009/icsdatasets/, extracted to
  `data/external/msu_ics_power/multiclass/`.

## Rebuilding the documents

```bash
python manuscript/build_manuscript.py        # Manuscript_R3_clean.docx, Manuscript_R3_highlighted.docx
python manuscript/build_response_letter.py   # Response_to_Reviewers_R3.docx
```

The source is the Revision-2 Word file in `manuscript/source/`. `manuscript_revision3.md` and
`response_to_reviewers.md` at the project root are text-only pandoc exports of the built Word
files.
