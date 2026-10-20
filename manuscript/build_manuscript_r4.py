"""
Build the Revision-4 manuscript (.docx) from the Revision-3 clean Word file.

Every number introduced or updated in this revision is read from the result
files written by the analysis scripts (see manuscript/PROVENANCE_R4.md); none is
typed in by hand. Each block also asserts that the results still show the
pattern its sentence describes, so a changed result stops the build instead of
producing a wrong sentence.

Input : manuscript/source/Manuscript_R3_clean.docx   (place it there yourself)
Output: manuscript/Manuscript_R4_clean.docx         (no highlighting)
        manuscript/Manuscript_R4_highlighted.docx   (this revision's changes in yellow)

Usage (from the project root):  python manuscript/build_manuscript_r4.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from docx_edit import clone_table, insert_after, norm, remove_all_highlights, set_text  # noqa: E402

SOURCE = HERE / "source" / "Manuscript_R3_clean.docx"
R = ROOT / "results"
R4 = R / "r4"
FIG = R4 / "figures"
MODEL = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}


def J(path):
    path = Path(path)
    if not path.exists() and PARTIAL:
        return None
    return json.loads(path.read_text())


PARTIAL = False  # set True only to test-build while results are still being produced


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------
# Document helpers (same approach as the Revision-3 build)
# --------------------------------------------------------------------------
def runs_to_md(p):
    runs = [r for r in p.runs if r.text]
    if not runs or runs[-1].bold:
        return p.text
    out, i = [], 0
    while i < len(runs):
        if runs[i].bold:
            j = i
            while j < len(runs) and runs[j].bold:
                j += 1
            chunk = "".join(r.text for r in runs[i:j])
            core = chunk.rstrip()
            out.append(f"**{core}**{chunk[len(core):]}")
            i = j
        else:
            out.append(runs[i].text)
            i += 1
    return "".join(out)


class Doc:
    def __init__(self, path, mark):
        self.d = Document(str(path))
        self.mark = mark
        remove_all_highlights(self.d)

    def para(self, prefix):
        hits = [p for p in self.d.paragraphs if norm(p.text).startswith(norm(prefix))]
        assert len(hits) == 1, f"{len(hits)} paragraphs start with {prefix!r}"
        return hits[0]

    def table_after_caption(self, caption_prefix):
        """The table immediately preceding a caption paragraph."""
        p = self.para(caption_prefix)._p
        prev = p.getprevious()
        while prev is not None and prev.tag != qn("w:tbl"):
            prev = prev.getprevious()
        from docx.table import Table
        return Table(prev, self.d._body)

    def replace(self, prefix, md):
        set_text(self.para(prefix), md, self.mark)

    def edit(self, prefix, *pairs):
        p = self.para(prefix)
        text = runs_to_md(p)
        for old, new in pairs:
            assert text.count(old) == 1, f"{old!r} occurs {text.count(old)}x in {prefix!r}"
            text = text.replace(old, new)
        set_text(p, text, self.mark)

    def after(self, anchor, template, md):
        if isinstance(anchor, str):
            anchor = self.para(anchor)
        if isinstance(template, str):
            template = self.para(template)
        return insert_after(anchor, template, md, self.mark)

    def figure(self, anchor, image, width_in, caption_md, caption_template="Fig. 14. Macro-F1"):
        """Insert an image paragraph + caption after `anchor`; returns the caption paragraph."""
        if isinstance(anchor, str):
            anchor = self.para(anchor)
        blank = self.para(caption_template)._p.getprevious()  # the image paragraph of an existing figure
        el = copy.deepcopy(blank)
        for child in list(el):
            if child.tag != qn("w:pPr"):
                el.remove(child)
        anchor_el = anchor._p if isinstance(anchor, Paragraph) else anchor._tbl
        anchor_el.addnext(el)
        p = Paragraph(el, anchor._parent)
        run = p.add_run()
        run.add_picture(str(image), width=Inches(width_in))
        if self.mark:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return insert_after(p, self.para(caption_template), caption_md, self.mark)

    def table(self, anchor, template_caption, rows, caption_md):
        """Clone the table above `template_caption` with new rows after `anchor`, plus a caption."""
        tpl = self.table_after_caption(template_caption)
        if isinstance(anchor, str):
            anchor = self.para(anchor)
        t = clone_table(tpl, anchor, rows, self.mark)
        return insert_after(t, self.para(template_caption), caption_md, self.mark)

    def replace_image(self, caption_prefix, image):
        p = self.para(caption_prefix)._p.getprevious()
        blip = next(p.iter(qn("a:blip")))
        part = self.d.part.related_parts[blip.get(qn("r:embed"))]
        part._blob = Path(image).read_bytes()


# --------------------------------------------------------------------------
# Result files
# --------------------------------------------------------------------------
def load():
    n = {
        "transfer": J(R4 / "cross_domain_transfer.json"),
        "tau": J(R4 / "threshold_calibration.json"),
        "drift": J(R4 / "drift_detection.json"),
        "single": J(R4 / "drift_single_sensor.json"),
        "disjoint": J(R4 / "disjoint_validation.json"),
        "gap": J(R4 / "dataset_difficulty.json"),
        "faith": J(R4 / "shap_faithfulness.json"),
        "stab": J(R4 / "shap_stability.json"),
        "dl": J(R4 / "dl_baselines.json"),
        "env": J(R4 / "environment.json"),
        "rep": J(R / "repeated_cv_and_policy_eval.json"),
        "msu": {v: J(ROOT / "artifacts" / f"msu_external_validation_results_{v}.json")
                for v in ("multiclass", "triple", "binary")},
        "cv": J(ROOT / "artifacts" / "model_cv_summary.json"),
        "cv_folds": J(ROOT / "artifacts" / "cv_results.json"),
        "wilcoxon": J(ROOT / "artifacts" / "statistical_significance.json"),
        "ttest": J(R / "corrected_ttest.json"),
        "abl": J(ROOT / "artifacts" / "ablation_study_results.json"),
        "rob": J(ROOT / "artifacts" / "robustness_results.json"),
        "lat": J(ROOT / "artifacts" / "latency_benchmark_results.json"),
        "msu_shap": J(R / "msu_shap_multiclass.json"),
        "msu_abl": J(R / "msu_ablation_multiclass.json"),
        "auc": J(ROOT / "artifacts" / "roc_auc_scores.json"),
        "pol": J(R4 / "policy_latency.json"),
    }
    return n


def tau_row(n, t):
    return next(s for s in n["tau"]["sweep"] if abs(s["tau"] - t) < 1e-9)


def drift_row(n, s):
    return next(r for r in n["drift"]["detection"] if abs(r["shift"] - s) < 1e-9)


def mit_row(n, s):
    return next(r for r in n["drift"]["mitigation"] if abs(r["shift"] - s) < 1e-9)


# --------------------------------------------------------------------------
# Comment 1 -- direct cross-domain transfer (Section 3.2.2, new Section 4.6.1)
# --------------------------------------------------------------------------
def c1_numbers(n):
    t = n["transfer"]
    t1p, t1u = t["T1_deployed_18feat_mapping_P"], t["T1_deployed_18feat_mapping_PU"]
    t2p, t2u = t["T2_synthetic_10feat_binary_mapping_P"], t["T2_synthetic_10feat_binary_mapping_PU"]
    r = t["R_in_domain_10feat_retrained_cv"]
    rev = t["R_in_domain_10feat_event_disjoint"]
    assert rev["roc_auc_mean"] > 0.65
    aucs = [x["roc_auc"] for x in (t1p, t1u, t2p, t2u)]
    bas = [x["balanced_accuracy"] for x in (t1p, t1u, t2p, t2u)]
    # the text says: every no-retraining variant is at chance level, the retrained reference is not
    assert all(0.45 <= a <= 0.55 for a in aucs) and all(0.45 <= b <= 0.55 for b in bas), (aucs, bas)
    assert r["roc_auc_mean"] > 0.7
    assert 0.4 <= t1u["fraction_predicted_attack"] <= 0.6
    assert abs(t1u["attack_recall"] - t1u["false_alarm_rate_non_attack"]) < 0.06  # independent of the true class
    best128 = max(n["msu"]["binary"]["summary"].items(), key=lambda kv: kv[1]["macro_f1_mean"])
    dj = n["disjoint"]["binary"]["B_event_disjoint_groupkfold5"]
    best128ev = max(dj.items(), key=lambda kv: kv[1]["macro_f1_mean"])
    return {"t1p": t1p, "t1u": t1u, "t2p": t2p, "t2u": t2u, "r": r, "rev": rev, "best128ev": best128ev, "chance": t["chance"]["all_attack"],
            "auc_lo": min(aucs), "auc_hi": max(aucs), "ba_lo": min(bas), "ba_hi": max(bas),
            "best128": best128, "prev": t["chance"]["attack_prevalence"]}


def c1(D: Doc, n):
    c = c1_numbers(n)
    D.edit("To assess generalization beyond the controlled synthetic environment, the trained pipeline",
           ("To assess generalization beyond the controlled synthetic environment, the trained pipeline is "
            "additionally, and independently, validated on",
            "To assess whether the proposed methodology carries over to independently collected data, the same "
            "modeling pipeline is additionally retrained and evaluated on"),
           ("a substantially stronger and different claim that is not made in this study.",
            "a substantially stronger and different claim. We nevertheless tested this stronger claim directly "
            "(Section 4.6.1): applied to MSU/ORNL without retraining, the "
            "synthetic-trained model performed at chance level. The study therefore makes no claim of model "
            "transferability; all MSU/ORNL results other than those of Section 4.6.1 come from models retrained on "
            "MSU/ORNL data."))

    h = D.after("Second, model ranking is not consistent across datasets", "3.6.1 STATISTICAL POWER",
                "4.6.1 DIRECT CROSS-DOMAIN TRANSFER WITHOUT RETRAINING")
    p = D.after(h, "Two findings stand out.",
                "The results above were obtained by retraining the pipeline on MSU/ORNL. To test whether a model "
                "trained on the synthetic dataset can be used on MSU/ORNL data without retraining, a common "
                "feature space and a common label space are needed, because the two datasets share neither. The "
                "only label space they share is binary (attack vs. non-attack, the MSU/ORNL binary release, "
                f"{pct(c['prev'])} attack rows). For the features, ten of the 18 synthetic features have a "
                "physical counterpart in the PMU channels of relay R1 and the neighbouring relay R2: voltage, "
                "current, frequency, power factor (cosine of the phase-A voltage-current angle), active and "
                "reactive power and demand (computed from them), three-phase voltage imbalance, and feeder "
                "voltage and current. Magnitudes were converted to the synthetic operating scale per unit, "
                "dividing by their label-free MSU/ORNL median and multiplying by the synthetic normal-class "
                "median; dimensionless quantities were either kept as they are (mapping P) or rescaled in the same "
                "way (mapping PU). The eight synthetic features with no PMU counterpart (THD, breaker state, packet "
                "rate, packet error rate, the two temperatures, line load, and tap position) were set to their "
                "synthetic normal-class median, i.e. treated as missing. Two models were applied without any "
                "retraining: the deployed five-class LightGBM, with any non-normal prediction counted as an attack "
                "(T1), and a binary LightGBM trained on the synthetic data restricted to the ten mapped features "
                "(T2). As a reference, the same ten mapped features were used to retrain and test LightGBM on "
                "MSU/ORNL itself (R), both with the stratified random folds of Table 8 and with whole recordings held out "
                "(Section 4.6.2). Table 8d reports the results.")
    rows = [["Model (training data)", "Mapping", "Macro-F1", "Balanced accuracy", "ROC-AUC"],
            ["Majority class ('attack' for every row)", "--", f3(c["chance"]["macro_f1"]),
             f3(c["chance"]["balanced_accuracy"]), "--"],
            ["T1: deployed 18-feature model (synthetic, no retraining)", "P", f3(c["t1p"]["macro_f1"]),
             f3(c["t1p"]["balanced_accuracy"]), f3(c["t1p"]["roc_auc"])],
            ["T1: deployed 18-feature model (synthetic, no retraining)", "PU", f3(c["t1u"]["macro_f1"]),
             f3(c["t1u"]["balanced_accuracy"]), f3(c["t1u"]["roc_auc"])],
            ["T2: 10-feature binary model (synthetic, no retraining)", "P", f3(c["t2p"]["macro_f1"]),
             f3(c["t2p"]["balanced_accuracy"]), f3(c["t2p"]["roc_auc"])],
            ["T2: 10-feature binary model (synthetic, no retraining)", "PU", f3(c["t2u"]["macro_f1"]),
             f3(c["t2u"]["balanced_accuracy"]), f3(c["t2u"]["roc_auc"])],
            ["R: same 10 features, retrained on MSU/ORNL (5-fold CV)", "--", f3(c["r"]["macro_f1_mean"]),
             f3(c["r"]["balanced_accuracy_mean"]), f3(c["r"]["roc_auc_mean"])],
            ["R: same 10 features, retrained, whole recordings held out (event-disjoint 5-fold)", "--",
             f3(c["rev"]["macro_f1_mean"]), f3(c["rev"]["balanced_accuracy_mean"]), f3(c["rev"]["roc_auc_mean"])],
            [f"Table 8: all 128 features, retrained on MSU/ORNL ({MODEL[c['best128'][0]]}, 5-fold CV)", "--",
             f3(c["best128"][1]["macro_f1_mean"]), "--", "--"],
            [f"Table 8e: all 128 features, retrained, event-disjoint ({MODEL[c['best128ev'][0]]})", "--",
             f3(c["best128ev"][1]["macro_f1_mean"]), "--", "--"]]
    cap = D.table(p, "Table 2. Model Comparison", rows,
                  "Table 8d. Direct Cross-Domain Transfer to the MSU/ORNL Binary Task Without Retraining "
                  f"(n = {n['transfer']['n_msu']:,} MSU/ORNL rows; no MSU/ORNL row is used to train T1 or T2)")
    p2 = D.after(cap, "Two findings stand out.",
                 "Without retraining, both synthetic-trained models are at chance level under both mappings: "
                 f"balanced accuracy {f3(c['ba_lo'])}-{f3(c['ba_hi'])} and ROC-AUC {f3(c['auc_lo'])}-"
                 f"{f3(c['auc_hi'])}, i.e. their attack scores carry no information about the MSU/ORNL labels. "
                 f"Under mapping P the deployed model labels {pct(1 - c['t1p']['fraction_predicted_attack'])} of all "
                 "MSU/ORNL rows as normal, and its Macro-F1 "
                 f"({f3(c['t1p']['macro_f1'])}) is below that of the trivial all-attack predictor "
                 f"({f3(c['chance']['macro_f1'])}); under mapping PU it labels about half of the rows as attacks, "
                 "independently of their true class. The failure is not caused by the mapped features being "
                 "uninformative: retrained on MSU/ORNL, the same ten features reach a ROC-AUC of "
                 f"{f3(c['r']['roc_auc_mean'])} (row R), and still {f3(c['rev']['roc_auc_mean'])} when whole recordings "
                 "are held out. The decision boundaries learned on the synthetic "
                 "data therefore do not transfer to the MSU/ORNL testbed, whose attacks are not expressed as the "
                 "synthetic attack signatures (packet-rate floods, saturated voltages, or thermal ramps; "
                 "Section 4.6.3). We conclude that the evidence supports methodology-level external validity "
                 "(the same pipeline, retrained, works on real PMU data) but not model transferability, and we have "
                 "revised the Abstract, the contribution list, and the Conclusion accordingly.")
    return p2


# --------------------------------------------------------------------------
# Comment 3 -- what the 98.1% means (terminology, scope) ; Comment 4 -- tau
# --------------------------------------------------------------------------
def c3_numbers(n):
    P = n["rep"]["prevention_benchmark"]
    t07 = tau_row(n, 0.7)
    # the seed-42 out-of-fold predictions of both scripts must agree
    assert abs(P["response_accuracy"] - t07["response_accuracy"]) < 1e-12, (P["response_accuracy"], t07)
    assert P["false_blocks"] == t07["false_blocks"] and P["missed_responses"] == t07["missed_responses"]
    return P


def c3(D: Doc, n):
    P = c3_numbers(n)
    D.edit("Scope of experimental evaluation.",
           ("Section 4.9 evaluates its response accuracy,", "Section 4.9 evaluates its action-selection accuracy,"),
           ("This scoping is stated here to avoid any implication that the reported metrics constitute evidence of "
            "field-deployed prevention efficacy.",
            "This scoping is stated here to avoid any implication that the reported metrics constitute evidence of "
            "field-deployed prevention efficacy. No closed-loop validation was performed: it would require a "
            "power-system simulator or hardware-in-the-loop testbed in which the selected actions change the "
            "subsequent grid state and telemetry, and neither was available for this study. Accordingly, the "
            f"{pct(P['response_accuracy'])} reported in Section 4.9 is an action-selection accuracy -- the fraction of "
            "samples for which the policy engine selects the action that the ground-truth class calls for -- and not "
            "a measure of attacks actually prevented."))
    D.replace("4.9 PREVENTION/POLICY-ENGINE PERFORMANCE ANALYSIS",
              "4.9 OFFLINE (OPEN-LOOP) EVALUATION OF THE POLICY-RESPONSE LAYER")
    D.edit("Evaluation protocol and metrics.",
           ("Response accuracy is the fraction of samples whose selected action exactly matches the ground-truth "
            "action",
            "Action-selection accuracy is the fraction of samples "
            "whose selected action exactly matches the ground-truth action"))
    D.edit("Response accuracy and false-block rate.",
           ("**Response accuracy and false-block rate.**", "**Action-selection accuracy and false-block rate.**"),
           ("(response accuracy 98.1%)", f"(action-selection accuracy {pct(P['response_accuracy'])})"))
    D.edit("These results establish that the policy engine",
           ("These results establish that the policy engine selects the intended action given the classifier's "
            "actual, imperfect predictions.",
            "These results establish that the policy engine selects the intended action given the classifier's "
            "actual, imperfect predictions; they are an open-loop, offline measurement of action selection and do "
            "not show that any attack was prevented."),
           ("response accuracy would be correspondingly lower.",
            "action-selection accuracy would be correspondingly lower."))
    D.edit("Table 10. Ground-Truth Class-to-Action Mapping",
           ("Used to Evaluate the Prevention Layer", "Used to Evaluate the Policy-Response Layer (Offline)"))
    t9 = D.table_after_caption("Table 9. Deployment Characteristics")
    for row in t9.rows:
        if norm(row.cells[0].text).startswith("Operational validation"):
            from docx_edit import set_cell
            set_cell(row.cells[1], "Controlled synthetic + external real-world (MSU/ORNL, retrained); policy layer "
                                   "evaluated offline for action selection (Section 4.9); no closed-loop validation",
                     D.mark)


def c4(D: Doc, n, anchor):
    T = n["tau"]
    s07, s05, s09, s095 = (tau_row(n, t) for t in (0.7, 0.5, 0.9, 0.95))
    sw = T["sweep"]
    lo = [s for s in sw if 0.5 - 1e-9 <= s["tau"] <= 0.9 + 1e-9]
    ra_min, ra_max = min(s["response_accuracy"] for s in lo), max(s["response_accuracy"] for s in lo)
    cal = T["calibration"]
    sel = [x["selected_tau"] for x in T["nested_selection"]]
    out_sel = [x["outer_at_selected"]["response_accuracy"] for x in T["nested_selection"]]
    out_07 = [x["outer_at_0.7"]["response_accuracy"] for x in T["nested_selection"]]
    gap = max(abs(a - b) for a, b in zip(out_sel, out_07))
    # patterns the text relies on
    assert ra_max - ra_min < 0.002
    assert s05["false_block_rate"] > s07["false_block_rate"] > s09["false_block_rate"]
    assert s05["missed_response_rate"] < s07["missed_response_rate"] < s09["missed_response_rate"]
    assert min(sel) <= 0.7 <= max(sel) and gap < 0.001
    assert cal["ece"] < 0.05

    h = D.after(anchor, "3.6.1 STATISTICAL POWER", "4.9.1 CONFIDENCE-THRESHOLD SENSITIVITY AND CALIBRATION")
    p = D.after(h, "Two findings stand out.",
                "The threshold τ = 0.7 was set as a design value in config.yaml; it was not tuned on the test data. "
                "To show its effect, the same seed-42 out-of-fold predictions were passed through the deployed policy "
                "function for every τ from 0.20 to 0.95 in steps of 0.05, and we measured action-selection accuracy, "
                "false-block rate, missed-response rate, the fraction of attack samples given a wrong automated "
                "action, and the detection performance of the gated detector (a prediction below τ raises no alert "
                "and is counted as normal). Table 10b and Fig. 16 summarise the sweep.")
    rows = [["τ", "Action-selection accuracy", "False-block rate", "Missed-response rate", "Gated Macro-F1"]]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        s = tau_row(n, t)
        rows.append([f"{t:.2f}" + (" (deployed)" if abs(t - 0.7) < 1e-9 else ""), pct(s["response_accuracy"], 2),
                     pct(s["false_block_rate"], 2), pct(s["missed_response_rate"], 1), f4(s["gated_macro_f1"])])
    cap = D.table(p, "Table 2. Model Comparison", rows,
                  "Table 10b. Policy-Layer Behaviour as a Function of the Confidence Threshold τ (Seed-42 Out-of-Fold "
                  "LightGBM Predictions, n = 30,000; the full sweep from τ = 0.20 is shown in Fig. 16)")
    p2 = D.after(cap, "Two findings stand out.",
                 "Three observations follow. First, τ governs a trade-off between false blocks and missed responses "
                 f"rather than overall correctness: raising τ from 0.50 to 0.90 lowers the false-block rate from "
                 f"{pct(s05['false_block_rate'], 2)} ({s05['false_blocks']} normal samples) to "
                 f"{pct(s09['false_block_rate'], 2)} ({s09['false_blocks']}) but raises the missed-response rate from "
                 f"{pct(s05['missed_response_rate'])} to {pct(s09['missed_response_rate'])}, while action-selection "
                 f"accuracy stays between {pct(ra_min, 2)} and {pct(ra_max, 2)} over the whole range. Detection "
                 f"performance of the gated detector is likewise flat (Macro-F1 {f4(s05['gated_macro_f1'])} at τ = 0.50, "
                 f"{f4(s07['gated_macro_f1'])} at 0.70, {f4(s09['gated_macro_f1'])} at 0.90) and falls only at τ = 0.95 "
                 f"({f4(s095['gated_macro_f1'])}). The reason is the second observation: the classifier's probabilities "
                 f"are well calibrated (expected calibration error {f3(cal['ece'])}, multiclass Brier score "
                 f"{f3(cal['brier'])}; Fig. 16, right) and concentrated near 1, so only {pct(s07['fraction_below_tau'])} "
                 "of all predictions fall below 0.7, and these are exactly the unreliable ones: predictions with a top-"
                 f"class probability of at least 0.7 are correct in {pct(cal['accuracy_conf_ge_0.7'])} of cases, those "
                 f"below 0.7 in only {pct(cal['accuracy_conf_lt_0.7'])}. Third, a data-driven choice does not improve on "
                 "the design value. In a nested procedure, τ was selected inside each outer training fold by an inner "
                 "five-fold cross-validation (criterion: maximum action-selection accuracy) and then applied to the "
                 f"untouched outer test fold; the selected values ranged from {min(sel):.2f} to {max(sel):.2f}, and on "
                 "every outer fold the action-selection accuracy obtained with the selected τ differed from that "
                 f"obtained with τ = 0.7 by at most {100 * gap:.2f} percentage points. τ = 0.7 lies inside the range a "
                 "data-driven procedure selects and was retained; operators who weigh a missed response more heavily "
                 "than a false block can lower it to about 0.5 at the cost shown in Table 10b. These conclusions hold "
                 "for the well-calibrated synthetic-data model; for a model deployed on real telemetry, calibration "
                 "and τ would have to be re-assessed on site data.")
    return D.figure(p2, FIG / "threshold_sensitivity_calibration.png", 6.0,
                    "**Fig. 16. Confidence-Threshold Sensitivity (left) and Reliability Diagram (right).** Left: "
                    "action-selection accuracy, gated detection Macro-F1, 1 − missed-response rate, and false-block "
                    "rate (right axis) against τ; the dotted line marks the deployed τ = 0.7. Right: observed accuracy "
                    "against mean predicted top-class probability in 15 bins; points close to the diagonal indicate "
                    "well-calibrated probabilities.")


# --------------------------------------------------------------------------
# Comment 2 -- drift detection and mitigation experiment (new Section 4.5.1)
# --------------------------------------------------------------------------
def c2(D: Doc, n):
    d, sg = n["drift"], n["single"]
    st = d["settings"]
    z = drift_row(n, 0.0)
    shifts = [r for r in d["detection"] if r["shift"] > 0]
    assert all(r["ks_detection_rate"] == 1.0 and r["psi_detection_rate"] == 1.0 for r in shifts)
    c05, c02 = drift_row(n, 0.05), drift_row(n, 0.02)
    assert c02["conf_detection_rate"] < 0.2 and c05["conf_detection_rate"] > 0.8
    m = d["mitigation"]
    g = d["no_drift_guard"]
    assert g["candidate_val_f1"] <= g["deployed_val_f1"]  # healthy model is not replaced
    m01, m20 = mit_row(n, 0.01), mit_row(n, 0.20)
    assert all(all(x["promoted"] for x in r["M2_shadow"]) for r in m)
    n2000 = [next(x for x in r["M2_shadow"] if x["n_labelled"] == 2000)["active_model_eval_f1"] for r in m]
    n250 = [next(x for x in r["M2_shadow"] if x["n_labelled"] == 250)["active_model_eval_f1"] for r in m]
    m1 = [r["M1_recentering"] for r in m]
    m0 = [r["M0_no_action"] for r in m]
    S = sg["summary"]
    rows_ss = {(r["feature"], r["k_sigma"]): r for r in sg["rows"]}
    ll05 = rows_ss[("line_load_percent", 0.05)]
    assert ll05["ks_detection_rate"] < 0.1 and ll05["macro_f1_drop"] > 0.1

    D.edit("The model degrades gracefully under Gaussian sensor noise",
           ("Section 4.10 outlines a drift-detection and mitigation strategy for this failure mode.",
            "Section 4.5.1 therefore tests, rather than only proposes, a drift-detection and mitigation strategy "
            "for this failure mode."))
    h = D.after("The model degrades gracefully under Gaussian sensor noise", "3.6.1 STATISTICAL POWER",
                "4.5.1 DRIFT DETECTION AND MITIGATION EXPERIMENT")
    p = D.after(h, "Two findings stand out.",
                "The drift-monitoring and shadow-retraining strategy was implemented and evaluated with the deployed model, test split, and shift operator of Table 7 (every feature x is "
                "replaced by x + s·mean(x)), adding the smaller shifts s = 1% and 2%. The 6,000-row test split was "
                "divided into a 'recent' slice (2,000 rows), a validation slice (1,000 rows), and an evaluation slice "
                "(3,000 rows). Detection is label-free: a reference window of "
                f"{st['n_ref']:,} rows drawn from the training split is compared with monitoring windows of "
                f"{st['W']} rows drawn from the recent and evaluation slices, and a window raises an alarm if (KS) any "
                f"of the 18 two-sample Kolmogorov-Smirnov tests has p < {st['alpha']}/18 (Bonferroni), (PSI) any "
                f"feature's Population Stability Index exceeds {st['psi_limit']}, or (CONF) the window's mean top-class "
                "probability falls more than three standard errors below its value on the unshifted validation slice. "
                f"Each setting was evaluated on {st['n_windows']} random windows. Mitigation was evaluated on the "
                "evaluation slice: (M0) no action; (M1) label-free re-centring, which subtracts the per-feature mean "
                "difference between the recent window and the reference window before applying the unchanged model; "
                "and (M2) shadow retraining, in which a candidate model with the deployed configuration is trained on "
                "the original training split plus n labelled rows of the recent shifted slice (n = 250, 500, 1,000, "
                "2,000), and is promoted only if it outperforms the deployed model on the validation slice. Table 7b "
                "and Fig. 14b report the results.")
    rows = [["Shift s", "Alarm rate KS / PSI / CONF", "Macro-F1, no action (M0)", "Re-centring (M1)",
             "Shadow retraining, n = 250 / 2,000 (M2)"]]
    for r in d["detection"]:
        s = r["shift"]
        if s == 0:
            rows.append(["0 (no drift)", f"{pct(r['ks_detection_rate'])} / {pct(r['psi_detection_rate'])} / "
                         f"{pct(r['conf_detection_rate'])}", f3(g["deployed_eval_f1"]), "--",
                         f"not promoted ({f3(g['candidate_val_f1'])} vs. {f3(g['deployed_val_f1'])} on validation)"])
            continue
        mm = mit_row(n, s)
        a = next(x for x in mm["M2_shadow"] if x["n_labelled"] == 250)["active_model_eval_f1"]
        b = next(x for x in mm["M2_shadow"] if x["n_labelled"] == 2000)["active_model_eval_f1"]
        rows.append([f"{100 * s:.0f}%", f"{pct(r['ks_detection_rate'], 0)} / {pct(r['psi_detection_rate'], 0)} / "
                     f"{pct(r['conf_detection_rate'], 0)}", f3(mm["M0_no_action"]), f3(mm["M1_recentering"]),
                     f"{f3(a)} / {f3(b)}"])
    cap = D.table(p, "Table 2. Model Comparison", rows,
                  "Table 7b. Drift Detection (Alarm Rate per 250-Row Window; at s = 0 the False-Alarm Rate) and "
                  "Mitigation (Macro-F1 on the 3,000-Row Evaluation Slice) Under the Table 7 Shift Operator")
    p2 = D.after(cap, "Two findings stand out.",
                 "Both distribution tests detected every tested shift, including s = 1%, in every window "
                 f"(first-window detection), with per-window false-alarm rates of {pct(z['ks_detection_rate'])} (KS) "
                 f"and {pct(z['psi_detection_rate'])} (PSI) without drift, i.e. on average one false alarm every "
                 f"{z['ks_mean_delay_windows']:.0f} and {z['psi_mean_delay_windows']:.0f} windows. The confidence proxy "
                 f"was much less sensitive: it raised alarms in {pct(c02['conf_detection_rate'])} of windows at s = 2% "
                 f"and in {pct(c05['conf_detection_rate'])} at s = 5%, although Macro-F1 had already fallen to "
                 f"{f3(c02['macro_f1_deployed'])} at s = 2%, because the model stays confident under shift (mean "
                 f"top-class probability {f3(c02['mean_top_class_probability'])} at s = 2%); it is therefore unsuitable "
                 "as the primary drift signal. Even a 1% shift is harmful (Macro-F1 "
                 f"{f3(m01['M0_no_action'])}) because the operator shifts every feature by a fraction of its mean, which "
                 "for features with a small coefficient of variation is a large shift in standard-deviation units "
                 "(for frequency, 1% of 50 Hz is about ten standard deviations). Both mitigations recovered most of the "
                 f"loss. Label-free re-centring restored Macro-F1 to {f3(min(m1))}-{f3(max(m1))} across all shifts, "
                 "which is expected because the simulated drift is a pure offset; it would not correct drifts that change "
                 "the shape of a distribution. Shadow retraining restored Macro-F1 to "
                 f"{f3(min(n2000))}-{f3(max(n2000))} with 2,000 labelled recent rows and to {f3(min(n250))}-"
                 f"{f3(max(n250))} with 250, was promoted in every drifted setting, and was correctly not promoted "
                 f"without drift (validation Macro-F1 {f3(g['candidate_val_f1'])} vs. {f3(g['deployed_val_f1'])} for "
                 "the deployed model).")
    p3 = D.after(p2, "Two findings stand out.",
                 "The shift operator of Table 7 moves all 18 features at once and is easy to detect. To find the "
                 "sensitivity limit, we also offset one feature at a time by k standard deviations (k = 0.05, 0.1, "
                 "0.25, 0.5, 1; 300 windows per setting). The KS monitor detected single-feature offsets of 0.5 "
                 f"standard deviations in {pct(S['0.5']['ks_detection_rate_mean'])} of windows on average (minimum "
                 f"{pct(S['0.5']['ks_detection_rate_min'])} over the 18 features), 0.25 standard deviations in "
                 f"{pct(S['0.25']['ks_detection_rate_mean'])}, and 0.1 standard deviations in only "
                 f"{pct(S['0.1']['ks_detection_rate_mean'])}. Offsets this small were mostly harmless (mean Macro-F1 "
                 f"loss over the 18 features {f3(S['0.1']['macro_f1_drop_mean'])} at 0.1 standard deviations), but not "
                 "always: an offset of "
                 "0.05 standard deviations on line_load_percent (0.84 percentage points of line load) was detected in "
                 f"only {pct(ll05['ks_detection_rate'])} of windows yet reduced Macro-F1 by "
                 f"{f3(ll05['macro_f1_drop'])}. This sensitivity to a tiny offset reflects the generator artefact "
                 "analysed in Section 4.6.3 -- many overload rows sit exactly on the clipping bound of line load -- and "
                 "it identifies a case that distribution monitoring alone cannot catch. These experiments therefore "
                 "show that the proposed monitoring detects the drifts of Table 7 promptly and that both mitigations "
                 "recover most of the loss, but they were carried out on simulated drift in the synthetic data; "
                 "detection and recovery under real sensor ageing or recalibration, and the labelling effort that "
                 "shadow retraining needs in the field, remain to be established (Section 4.10).")
    return D.figure(p3, FIG / "drift_detection_mitigation.png", 6.0,
                    "**Fig. 14b. Drift Detection and Mitigation.** Left: alarm rate per 250-row window of the KS, PSI "
                    "and confidence monitors against the shift magnitude (at 0%, the false-alarm rate). Right: Macro-F1 "
                    "on the evaluation slice without action, after label-free re-centring, and after shadow retraining "
                    "with 250-2,000 labelled recent rows.")


# --------------------------------------------------------------------------
# Comment 10 -- one computational environment (Section 3.9)
# Comment 9  -- complete deep-learning baseline specification (Sections 3.9, 4.1)
# --------------------------------------------------------------------------
def env_sentence(n):
    e = n["env"]
    pk = e["packages"]
    return (f"Python {e['python']}, NumPy {pk['numpy']}, pandas {pk['pandas']}, SciPy {pk['scipy']}, scikit-learn "
            f"{pk['sklearn']}, XGBoost {pk['xgboost']}, LightGBM {pk['lightgbm']}, CatBoost {pk['catboost']}, SHAP "
            f"{pk['shap']}, and TensorFlow {pk['tensorflow']} (CPU)")


def c10(D: Doc, n, repro):
    e = n["env"]
    D.edit("The code used to generate all results reported in this study",
           ("and prevention-layer analyses, is available at",
            "and prevention-layer analyses, as well as the drift, transfer, event-disjoint, faithfulness, "
            "data-complexity, and threshold analyses, is available at"),
           ("CYBER-PHYSICAL-SYSTEMS.",
            "CYBER-PHYSICAL-SYSTEMS. The repository provides the package lock file of the computational environment "
            "(environment/requirements-lock.txt), a single script that regenerates every table and figure "
            "(reproduce_all.sh), and a file that maps each reported result to the command and result file that "
            "produced it (manuscript/PROVENANCE_R4.md)."))
    D.replace("Software environment:",
              "Software environment: every result reported in this study was generated in a single computational "
              "environment: "
              f"{env_sentence(n)}, on Linux ({e['platform']}) with a {e['physical_cpus']}-core "
              f"{e['cpu_model']} CPU and {e['ram_gb']:.0f} GB RAM. The complete package list is provided as a lock "
              "file (environment/requirements-lock.txt) in the code repository, and a single script "
              "(reproduce_all.sh) regenerates every table and figure. The latency benchmarks of Sections 4.8 and 4.9 "
              "depend on hardware and were measured on the machine above.")


def c9(D: Doc, n):
    dl = n["dl"]
    sel = dl["selection"]
    sm = dl["summary"]
    cl, tr = sel["cnn_lstm"]["selected"], sel["transformer"]["selected"]
    D.edit("Hyperparameters (fixed across all folds",
           ("CNN-LSTM and Transformer (Section 4, deep-learning baseline comparison): trained for up to 15 epochs with "
            "early stopping (patience = 3 on training loss), batch size 256, Adam optimizer (learning rate 1e-3); "
            "architectural details are given alongside their results.",
            "CNN-LSTM and Transformer (Section 4.1): the complete specification -- layers, parameter counts, "
            "optimizer, learning rate, batch size, epochs, regularisation, early stopping, and the hyperparameter-"
            "selection procedure -- is given in Table 3d and implemented in scripts/dl_baselines.py."))
    D.edit("To address whether tree-ensemble models remain competitive",
           ("(same folds, same fold-local oversampling, same scaler-fit-on-train-only discipline).",
            "(same folds, same fold-local oversampling, same scaler-fit-on-train-only discipline). Inside every "
            "outer training fold, 10% of the rows (stratified) were set aside before oversampling as a validation "
            "set for early stopping, so the outer test fold was never used for training, early stopping, or model "
            "selection. The complete configuration of both networks is given in Table 3d, and Table 3b reports the "
            "results."))
    rows = [["Item", "CNN-LSTM", "Transformer encoder"],
            ["Input", "18 standardized features as a length-18 sequence, 1 channel",
             "18 standardized features as a length-18 sequence, 1 channel"],
            ["Layers", f"Conv1D({cl['width']}, k=3, ReLU) → Conv1D({2 * cl['width']}, k=3, ReLU) → MaxPool1D(2) → "
                       f"LSTM({2 * cl['width']}) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(5, softmax)",
             f"Dense({tr['width']}) per-feature embedding + learned feature-position embedding → 2 × [multi-head "
             f"self-attention (4 heads, key dim {tr['width'] // 4}, dropout 0.1) + residual + LayerNorm → FFN("
             f"{2 * tr['width']}, ReLU) → Dense({tr['width']}) + residual + LayerNorm] → global average pooling → "
             "Dropout(0.3) → Dense(64, ReLU) → Dense(5, softmax)"],
            ["Trainable parameters", f"{sm['cnn_lstm']['n_params']:,}", f"{sm['transformer']['n_params']:,}"],
            ["Optimizer, loss", "Adam; sparse categorical cross-entropy", "Adam; sparse categorical cross-entropy"],
            ["Learning rate (selected)", f"{cl['lr']:g}", f"{tr['lr']:g}"],
            ["Batch size; maximum epochs", "256; 50", "256; 50"],
            ["Regularisation", "L2 1e-4 on Conv/LSTM/Dense kernels; dropout 0.3",
             "L2 1e-4 on Dense kernels; dropout 0.1 (attention) and 0.3"],
            ["Early stopping", "val. loss, patience 5, best weights restored", "val. loss, patience 5, best weights restored"],
            ["Epochs run per fold (best epoch)",
             ", ".join(f"{a} ({b})" for a, b in zip(sm["cnn_lstm"]["epochs_run"], sm["cnn_lstm"]["best_epoch"])),
             ", ".join(f"{a} ({b})" for a, b in zip(sm["transformer"]["epochs_run"], sm["transformer"]["best_epoch"]))],
            ["Hyperparameter selection", "grid lr ∈ {1e-3, 3e-4} × width F ∈ {16, 32}, scored by validation Macro-F1 on "
                                         "outer fold 1's training part only, then fixed for all folds",
             "grid lr ∈ {1e-3, 3e-4} × width d ∈ {32, 64}, same procedure"],
            ["Seeds / determinism", "tf.keras.utils.set_random_seed(42); op determinism enabled",
             "tf.keras.utils.set_random_seed(42); op determinism enabled"]]
    cap_b = "Table 3b. Tree-Ensemble vs. Deep-Learning Baselines"
    tb = D.table_after_caption(cap_b)
    return rows, tb


# --------------------------------------------------------------------------
# Comment 5 -- event- and time-disjoint validation (new Section 4.6.2; Section 3.2.1)
# --------------------------------------------------------------------------
def c5_numbers(n):
    dj = n["disjoint"]
    out = {"order": dj["synthetic_order_check"], "struct": dj["multiclass"]["structure"], "v": {}}
    for v in ("multiclass", "triple", "binary"):
        r = dj[v]
        out["v"][v] = {m: {"A": r["A_stratified_random_5fold"][m]["macro_f1_mean"],
                           "A_sd": r["A_stratified_random_5fold"][m]["macro_f1_std"],
                           "B": r["B_event_disjoint_groupkfold5"][m]["macro_f1_mean"],
                           "B_sd": r["B_event_disjoint_groupkfold5"][m]["macro_f1_std"],
                           "C": r["C_time_disjoint_within_event"][m]["macro_f1_mean"]}
                       for m in ("lightgbm", "random_forest")}
        # stratified random 5-fold here must equal Table 8 (same folds, same models)
        for m in ("lightgbm", "random_forest"):
            t8 = n["msu"][v]["summary"][m]["macro_f1_mean"]
            assert abs(out["v"][v][m]["A"] - t8) < 1e-9, (v, m, out["v"][v][m]["A"], t8)
    return out


def c5(D: Doc, n, anchor):
    c = c5_numbers(n)
    o = c["order"]
    assert o["identifier_columns"] == [] and abs(o["z"]) < 2
    D.edit("Obtaining large-scale, real-world, labeled smart grid",
           ("representative of adversarial activity.",
            "representative of adversarial activity. Each row is an independently generated operating point: the "
            "dataset contains no scenario, event, or time identifier, and its rows are not ordered in time (in file "
            f"order the class label changes {o['label_runs_in_file_order']:,} times, against "
            f"{o['expected_runs_if_independent']:,.0f} expected for independently ordered rows). Scenario- or "
            "time-disjoint splitting is therefore not defined for this dataset; it is applied to the MSU/ORNL data, "
            "whose rows do come from recorded events (Section 4.6.2)."))
    h = D.after(anchor, "3.6.1 STATISTICAL POWER", "4.6.2 EVENT- AND TIME-DISJOINT VALIDATION ON MSU/ORNL")
    s = c["struct"]
    p = D.after(h, "Two findings stand out.",
                "The stratified random five-fold protocol of Table 8 assigns individual rows to folds. In the MSU/ORNL "
                f"release, however, the {s['n']:,} rows come from {s['n_files']} recording files, and inside each file "
                "every scenario is one contiguous, time-ordered run of rows; we treat each such run as one executed "
                f"event ({s['n_events']} events, median {s['rows_per_event_median']:.0f} rows per event). Under "
                "random splitting, neighbouring and therefore highly similar rows of the same event fall into both "
                "the training and the test fold. To remove this, we repeated the evaluation with (i) event-disjoint "
                "five-fold cross-validation, grouping folds by recording file (GroupKFold, three complete recordings "
                "per test fold), so that no part of any test event is seen in training and every scenario is tested "
                "on executions the model has never seen; and (ii) a time-disjoint split, in which the first 80% of "
                "every event's rows (in recorded order) are used for training, the next 5% are discarded as a gap, and "
                "the last 15% are used for testing. The coarser 3-class and binary releases contain "
                "the same rows in the same order, so the same events and groups apply. Random Forest and LightGBM, "
                "the best models in Table 8, were evaluated with the Table 8 hyperparameters; Table 8e reports the "
                "results.")
    rows = [["Label granularity", "Protocol", "LightGBM Macro-F1", "Random Forest Macro-F1"]]
    names = {"multiclass": "37-class", "triple": "3-class", "binary": "Binary"}
    for v in ("multiclass", "triple", "binary"):
        x = c["v"][v]
        rows.append([names[v], "Stratified random 5-fold (Table 8)", f"{f4(x['lightgbm']['A'])} ± {f4(x['lightgbm']['A_sd'])}",
                     f"{f4(x['random_forest']['A'])} ± {f4(x['random_forest']['A_sd'])}"])
        rows.append([names[v], "Event-disjoint 5-fold (by recording)", f"{f4(x['lightgbm']['B'])} ± {f4(x['lightgbm']['B_sd'])}",
                     f"{f4(x['random_forest']['B'])} ± {f4(x['random_forest']['B_sd'])}"])
        rows.append([names[v], "Time-disjoint within event (80/5/15)", f4(x["lightgbm"]["C"]), f4(x["random_forest"]["C"])])
    cap = D.table(p, "Table 6. Feature-Group Ablation Results", rows,
                  "Table 8e. MSU/ORNL Results Under Stratified Random, Event-Disjoint, and Time-Disjoint Validation")
    return cap, c


# --------------------------------------------------------------------------
# Comment 6 -- specific novelty (Section 1.1) and the contribution list
# --------------------------------------------------------------------------
def c6(D: Doc, n):
    D.replace("Individually, several of the framework's components have precedent",
              "The building blocks of the framework are not new: ensemble tree-based classifiers for smart-grid "
              "intrusion detection [2,3,10], SHAP-based explanations of security models [8,31], and REST-based "
              "deployment of machine-learning services are each documented in the literature reviewed in Section 2, "
              "and we do not propose a new learning algorithm. The specific novelty of this work lies in three "
              "results that, to the best of our knowledge, no study in Table 1 reports for a smart-grid IDPS. "
              "(1) A tested, mechanistic account of the synthetic-to-real performance gap. Most smart-grid IDS "
              "studies are evaluated only on synthetic or single-testbed data. We evaluate the same pipeline on a "
              "synthetic and a real PMU dataset, show directly that the synthetic-trained model does not transfer "
              "without retraining (Section 4.6.1), validate the real-data result on held-out recorded events "
              "(Section 4.6.2), and explain the gap with the same instruments on both datasets -- SHAP attribution, "
              "feature-group ablation, and data-complexity measures -- which identify the generator artefacts and "
              "the communication-layer evidence that make the synthetic task easier (Sections 4.4, 4.6.3, 4.7). "
              "(2) Explanations that are tested, not only displayed. The SHAP rankings that the paper uses to "
              "interpret attacks are checked for stability across folds and inputs (Section 4.3) and for "
              "faithfulness to the model by deletion/insertion and remove-and-retrain tests on both datasets "
              "(Section 4.3.1). (3) An operational layer that is measured, with its limits stated. The drift "
              "monitors and shadow retraining that keep the detector valid are evaluated experimentally, including "
              "the drift sizes they cannot detect (Section 4.5.1), and the policy-response layer is evaluated "
              "offline for action selection, with a sensitivity and calibration analysis of its confidence "
              "threshold (Sections 4.9, 4.9.1). All three rest on a leakage-safe protocol with corrected "
              "statistical tests (Sections 3.4, 3.6, 3.6.1), and all results are produced by the same trained "
              "models in a single, documented computational environment. Equally important to the positioning is "
              "what the paper does not claim: it does not claim transferability of a trained model across grids, "
              "field-validated or closed-loop attack prevention, or superiority of one tree ensemble over the "
              "others.")
    D.edit("Development and experimental evaluation of an AI-based intrusion detection engine",
           ("evaluated post hoc for response accuracy, false-block rate, and execution latency (Section 4.9).",
            "evaluated offline (open loop) for action-selection accuracy, false-block rate, execution latency, and "
            "sensitivity to its confidence threshold (Sections 4.9, 4.9.1); closed-loop prevention is not evaluated."))
    D.edit("Independent external validation on the public Mississippi State University",
           ("Independent external validation on", "External validation of the methodology (retrained) on"),
           ("with SHAP-based explainability extended to this external dataset (Section 4.7).",
            "with event- and time-disjoint validation (Section 4.6.2), SHAP-based explainability extended to this "
            "dataset (Section 4.7), a direct test of model transfer without retraining (Section 4.6.1), and a "
            "data-complexity analysis of the synthetic-to-real performance gap (Section 4.6.3)."))
    D.edit("Systematic robustness evaluation under Gaussian sensor noise",
           ("together with a proposed drift-detection and mitigation strategy for operational deployment "
            "(Section 4.10).",
            "together with an experimental evaluation of drift detection (Kolmogorov-Smirnov, Population Stability "
            "Index, and model-confidence monitors) and mitigation (label-free re-centring and shadow retraining) "
            "(Section 4.5.1)."))
    D.edit("Incorporation of SHAP-based explainable AI mechanisms",
           ("to support operational decision-making.",
            "to support operational decision-making, with the stability and the faithfulness of the explanations "
            "tested by cross-fold rank correlation, deletion/insertion curves, and remove-and-retrain (Sections 4.3, "
            "4.3.1)."))


def declarations(D: Doc):
    """Scientific Reports uses a 'Competing interests' statement."""
    D.replace("Conflicts of Interest", "Competing interests")
    D.replace("The authors declare that they have no conflicts of interest",
              "The authors declare no competing interests.")


def c5_text(D: Doc, n, cap, c):
    v = c["v"]
    nn = n["gap"]["msu_nn_events_37class"]
    mc, tr, bi = v["multiclass"], v["triple"], v["binary"]
    for x in (mc, tr):
        for m in ("lightgbm", "random_forest"):
            assert x[m]["B"] < x[m]["A"] - 0.15 and x[m]["C"] < x[m]["A"] - 0.15, x
    assert nn["nn_from_same_event_fraction"] > 0.5
    assert nn["knn1_accuracy_neighbour_from_other_event"] < nn["knn1_accuracy_any_neighbour"] - 0.3
    best = lambda x, k: max(x["lightgbm"][k], x["random_forest"][k])
    chance37 = 1 / 37
    p = D.after(cap, "Two findings stand out.",
                "Holding out whole events changes the MSU/ORNL results substantially. On the 37-class task, Macro-F1 "
                f"falls from {f3(mc['lightgbm']['A'])} (LightGBM) and {f3(mc['random_forest']['A'])} (Random Forest) "
                f"under random splitting to {f3(mc['lightgbm']['B'])} and {f3(mc['random_forest']['B'])} when complete "
                f"recordings are held out, and to {f3(mc['lightgbm']['C'])} and {f3(mc['random_forest']['C'])} when "
                "the last part of every event is held out; this is above the chance level of a 37-class task "
                f"(about {f3(chance37)}) but far below Table 8. The 3-class task falls from "
                f"{f3(best(tr, 'A'))} to {f3(best(tr, 'B'))} (best model, event-disjoint), and the binary task from "
                f"{f3(best(bi, 'A'))} to {f3(best(bi, 'B'))}. The time-disjoint split, in which the model has seen "
                "the beginning of the very same event, gives almost the same values as the event-disjoint split, so "
                "the drop is not caused by differences between recordings. It is caused by the strong similarity of "
                "neighbouring rows: for "
                f"{pct(nn['nn_from_same_event_fraction'])} of {nn['n_query']:,} randomly chosen rows, the nearest "
                "neighbour among all other rows (standardized features) belongs to the same recorded event, and a "
                "1-nearest-neighbour rule labels "
                f"{pct(nn['knn1_accuracy_any_neighbour'])} of rows correctly when that neighbour may come from the same "
                f"event but only {pct(nn['knn1_accuracy_neighbour_from_other_event'])} when it must come from another "
                "event. Under random row-level splitting, therefore, most test rows have a near-copy from the same "
                "event in the training fold, and Table 8 measures largely the recognition of a known event rather than "
                "the detection of a new one. The event-disjoint values of Table 8e are the appropriate estimate of "
                "performance on unseen events, and they, not the Table 8 values, are used in this paper as the estimate "
                "of real-world performance. They also show that, in this dataset, a single "
                "row of PMU and log features carries little information about which of the 37 scenarios is "
                "executing; event-level features computed over windows of consecutive rows, which this study does "
                "not use, are the obvious next step. The synthetic dataset is not affected by this issue, because its "
                "rows are generated independently (Section 3.2.1).")
    return p


def section46_edits(D: Doc, n, c5):
    v = c5["v"]["multiclass"]
    ev = max(v["lightgbm"]["B"], v["random_forest"]["B"])
    D.edit("Two findings stand out. First, performance on the real MSU/ORNL dataset",
           ("and should be read as an honest bound on real-world generalization rather than a failure of the approach.",
            "and should be read as an upper bound on real-world performance rather than a failure of the approach. "
            "Section 4.6.2 shows that even these values are optimistic: random row-level splitting places rows of the "
            "same recorded event in both training and test folds, and with whole events held out the best 37-class "
            f"Macro-F1 is {f3(ev)}."))
    D.edit("Several converging factors plausibly explain this gap.",
           ("Several converging factors plausibly explain this gap.",
            "Several converging factors plausibly explain this gap; Section 4.6.3 tests them experimentally."),
           ("we regard the 83.1% MSU/ORNL Macro-F1, rather than the 92.8% synthetic figure, as the more representative "
            "estimate of performance on real grid telemetry, and we make no claim that a model trained on the synthetic "
            "dataset would transfer directly to a real substation without retraining on local data (Section 4.10).",
            "we regard the event-disjoint MSU/ORNL results of Section 4.6.2, rather than the 92.8% synthetic figure or "
            "the random-split 83.1%, as the more representative estimate of performance on real grid telemetry, and we "
            "make no claim that a model trained on the synthetic dataset transfers to a real substation without "
            "retraining on local data; Section 4.6.1 tests this directly and finds that it does not."))


def section46_captions(D: Doc):
    D.edit("Table 8. External Validation Results on the MSU/ORNL",
           ("(n = 78,377 real samples)",
            "(n = 78,377 real samples; stratified random row-level 5-fold cross-validation, retrained on MSU/ORNL; "
            "see Table 8e for event-disjoint results)"))
    D.edit("To assess generalization beyond the controlled synthetic dataset, the identical modeling pipeline",
           ("Table 8 reports the results.",
            "Table 8 reports the results of this stratified random row-level protocol; Section 4.6.2 repeats the "
            "evaluation with whole recorded events held out."))
    t1 = D.table_after_caption("Table 1. Comparative Analysis")
    from docx_edit import set_cell
    for row in t1.rows:
        if norm(row.cells[0].text).startswith("Proposed Work"):
            set_cell(row.cells[3], "Yes (MSU/ORNL, retrained; event-disjoint; direct transfer tested)", D.mark)


# --------------------------------------------------------------------------
# Comment 7 -- SHAP faithfulness (new Section 4.3.1, Table 5b, Fig. 12b); stability values re-run
# --------------------------------------------------------------------------
def stability_numbers(n):
    s = n["stab"]
    tops = [tuple(t) for t in s["top5_per_fold"]]
    same_set = all(set(t) == set(tops[0]) for t in tops)
    modal = max(set(tops), key=tops.count)
    return {"mean": s["mean_pairwise_spearman"], "min": s["min_pairwise_spearman"], "same_set": same_set,
            "same_order": tops.count(modal), "top": list(modal), "p01": s["perturbation"]["0.01"],
            "p05": s["perturbation"]["0.05"]}


def c7(D: Doc, n):
    S = stability_numbers(n)
    assert S["same_set"]
    F = n["faith"]
    sy, ms = F["synthetic"], F["msu_37class"]
    for r, b in ((sy, "mean"), (ms, "mean"), (sy, "marginal"), (ms, "marginal")):
        a = r["auc"][b]
        assert min(a[o]["deletion_auc_mean"] for o in ("shap_global", "gain_global")) > a["shap_instance"]["deletion_auc_mean"]
        assert max(a[o]["insertion_auc_mean"] for o in ("shap_global", "gain_global", "random")) < a["shap_instance"]["insertion_auc_mean"]
        assert a["shap_instance"]["deletion_auc_mean"] < a["random"]["deletion_auc_mean"] < a["shap_reverse"]["deletion_auc_mean"]
        assert a["shap_instance"]["insertion_auc_mean"] > a["random"]["insertion_auc_mean"]
    D.edit("A natural question for any post-hoc explanation method",
           ("the top-5 ranked features were identical (packet_error_rate, power_factor, phase_imbalance_percent, "
            "breaker_closed, substation_temp_c, in near-identical order) in 4 of 5 folds.",
            f"the top-5 ranked features were the same five features in all five folds ({', '.join(S['top'])}), in "
            f"identical order in {S['same_order']} of 5 folds."),
           ("averaged 0.947 (min 0.901 across all pairs)",
            f"averaged {f3(S['mean'])} (min {f3(S['min'])} across all pairs)"),
           ("correlated with the clean-input ranking at 0.988 ± 0.001 (1% noise) and 0.971 ± 0.001 (5% noise)",
            f"correlated with the clean-input ranking at {f3(S['p01']['mean'])} ± {f3(S['p01']['std'])} (1% noise) and "
            f"{f3(S['p05']['mean'])} ± {f3(S['p05']['std'])} (5% noise)"),
           ("Together, these results indicate the explanations in this section characterize a stable property of the "
            "model rather than a single-split artifact.",
            "Together, these results indicate the explanations in this section characterize a stable property of the "
            "model rather than a single-split artifact. Stability does not, however, show that an explanation is "
            "faithful, i.e. that the features it ranks highest are the ones the model's decision actually depends on; "
            "Section 4.3.1 tests this directly."))
    anchor = D.para("Table 5. SHAP-Based Attack Interpretation Summary")
    h = D.after(anchor, "3.6.1 STATISTICAL POWER", "4.3.1 FAITHFULNESS OF THE SHAP EXPLANATIONS")
    p = D.after(h, "Two findings stand out.",
                "Faithfulness was tested with deletion and insertion curves and with remove-and-retrain (ROAR), on "
                "both datasets. For each explained test sample and its predicted class c, the features were replaced "
                "by a baseline value one at a time in a given order (deletion), or restored one at a time starting "
                "from the all-baseline input (insertion), and the model's predicted probability of c was recorded "
                "after every step. If an ordering ranks the features the decision depends on first, deletion lowers "
                "the probability quickly (low area under the deletion curve) and insertion raises it quickly (high "
                "area under the insertion curve). Five orderings were compared: the sample's own |SHAP| values for "
                "class c (instance-level SHAP), the same ranking reversed, the global mean-|SHAP| ranking, LightGBM's "
                "gain importance, and random orders. Two baselines were used: the training-fold mean, and values "
                "drawn from random training rows, which keeps every replaced feature within its observed range. "
                f"The synthetic analysis used the five seed-42 folds ({sy['auc']['mean']['random']['n_samples']:,} "
                "explained test samples); the MSU/ORNL analysis used the 37-class LightGBM models of Table 8 "
                f"({ms['auc']['mean']['random']['n_samples']:,} samples, features removed in steps of four). For ROAR, "
                "the globally highest-ranked SHAP features were removed and the model was retrained and re-evaluated "
                "on the same folds, compared with removing the same number of randomly chosen features.")
    names = {"shap_instance": "Instance-level SHAP", "shap_global": "Global mean |SHAP|", "gain_global": "LightGBM gain",
             "random": "Random order", "shap_reverse": "Instance-level SHAP, reversed"}
    rows = [["Dataset", "Feature order", "Deletion AUC, mean baseline (↓)", "Insertion AUC, mean baseline (↑)",
             "Deletion AUC, marginal baseline (↓)"]]
    for dn, r in (("Synthetic", sy), ("MSU/ORNL 37-class", ms)):
        for o in ("shap_instance", "shap_global", "gain_global", "random", "shap_reverse"):
            rows.append([dn, names[o], f3(r["auc"]["mean"][o]["deletion_auc_mean"]),
                         f3(r["auc"]["mean"][o]["insertion_auc_mean"]), f3(r["auc"]["marginal"][o]["deletion_auc_mean"])])
    cap = D.table(p, "Table 2. Model Comparison", rows,
                  "Table 5b. SHAP Faithfulness: Area Under the Deletion and Insertion Curves of the Predicted-Class "
                  "Probability (LightGBM, Held-Out Samples)")
    vs, vm = sy["auc"]["mean"]["shap_vs_random"], ms["auc"]["mean"]["shap_vs_random"]
    fs, fm = sy["prediction_flip_rate"], ms["prediction_flip_rate"]
    ro, rm = sy["roar"], ms["roar"]
    base_s = ro["all_features"]
    ks = [k for k in (1, 2, 3, 5) if f"k{k}" in ro]
    assert all(ro[f"k{k}"]["remove_top_shap"] < ro[f"k{k}"]["remove_random_mean"] for k in ks if k != 2)
    assert abs(ro["k2"]["remove_top_shap"] - ro["k2"]["remove_random_mean"]) < 0.005
    roar_txt = ", ".join(f"{f3(ro[f'k{k}']['remove_top_shap'])} (k = {k})" for k in ks) + ", against " + \
        ", ".join(f3(ro[f"k{k}"]["remove_random_mean"]) for k in ks)
    m8 = rm["k8"]
    p2 = D.after(cap, "Two findings stand out.",
                 "On both datasets the instance-level SHAP ordering is the most faithful of the five (Table 5b, Fig. "
                 "12b): it gives the lowest deletion and the highest insertion area, the reversed ordering gives the "
                 "highest deletion area, and random orders lie in between. The ordering of the methods is the same with "
                 "the marginal baseline. Per sample, SHAP deletion was faster than random deletion for "
                 f"{pct(vs['deletion_shap_lower_fraction'])} of synthetic and {pct(vm['deletion_shap_lower_fraction'])} "
                 f"of MSU/ORNL samples (Wilcoxon signed-rank p = {vs['deletion_wilcoxon_p']:.1e} and "
                 f"{vm['deletion_wilcoxon_p']:.1e}). Replacing only the single top-SHAP feature by its mean changed the "
                 f"predicted class of {pct(fs['shap_top1'])} of synthetic samples, against {pct(fs['random_top1'])} for "
                 f"a random feature; for the top three features the rates were {pct(fs['shap_top3'])} and "
                 f"{pct(fs['random_top3'])} (MSU/ORNL: {pct(fm['shap_top3'])} vs. {pct(fm['random_top3'])}). "
                 "Remove-and-retrain, which tests whether the retrained model can do without the features, agrees "
                 f"on the synthetic data except at k = 2: starting from a Macro-F1 of {f3(base_s)} with all features, "
                 f"removing the k highest-ranked SHAP features and retraining gives {roar_txt} for k random features. "
                 f"On MSU/ORNL, removing the eight highest-ranked features costs little ({f3(m8['remove_top_shap'])}, "
                 f"against {f3(m8['remove_random_mean'])} for eight random features and {f3(rm['all_features'])} with "
                 "all features), because the 128 PMU channels are highly redundant and a retrained model recovers the "
                 "same information from correlated channels, as already seen for apparent impedance in Section 4.7. "
                 "We therefore regard the SHAP rankings used in this paper as faithful to the trained models, with "
                 "the caveat that on redundant real-world features a high SHAP rank shows what a model uses, not what "
                 "it cannot do without. Two further qualifications apply. Faithfulness tests measure agreement between an explanation and the model, not "
                 "between the explanation and the physical cause of an attack: a feature that the model uses because "
                 "of a generator artefact (Section 4.6.3) will be ranked as faithfully as a physically meaningful one. "
                 "And replacing features by a baseline creates inputs that may lie off the data distribution, which "
                 "is why the marginal baseline and remove-and-retrain, which avoids such inputs, are reported as well.")
    return D.figure(p2, FIG / "shap_faithfulness_curves.png", 6.0,
                    "**Fig. 12b. SHAP Faithfulness: Deletion (left) and Insertion (right) Curves.** Mean predicted "
                    "probability of the originally predicted class as features are replaced by the training mean "
                    "(deletion) or restored from it (insertion) in the order given by each method; top: synthetic "
                    "dataset, bottom: MSU/ORNL 37-class. A faster fall under deletion and a faster rise under insertion "
                    "indicate a more faithful ranking.")


# --------------------------------------------------------------------------
# Comment 8 -- is the synthetic dataset simply easier? (new Section 4.6.3, Table 8f)
# --------------------------------------------------------------------------
def c8(D: Doc, n, anchor, c5):
    G = n["gap"]
    s5, sb, m37, mb = G["synthetic_5class"], G["synthetic_binary"], G["msu_37class"], G["msu_binary"]
    sat = G["synthetic_saturation"]
    dj = n["disjoint"]
    ev37 = max(dj["multiclass"]["B_event_disjoint_groupkfold5"][m]["macro_f1_mean"] for m in ("lightgbm", "random_forest"))
    evb = max(dj["binary"]["B_event_disjoint_groupkfold5"][m]["macro_f1_mean"] for m in ("lightgbm", "random_forest"))
    cs, cm = s5["complexity"], m37["complexity"]
    # the text relies on: every complexity measure says the synthetic task is easier, by a wide margin
    assert cs["F1_max_fisher_ratio"] > 10 * cm["F1_max_fisher_ratio"]
    assert cs["N1_boundary_fraction"] < cm["N1_boundary_fraction"] and cs["knn1_loo_error"] < cm["knn1_loo_error"]
    assert sb["complexity"]["N1_boundary_fraction"] < mb["complexity"]["N1_boundary_fraction"]
    cap = lambda r, k: r["capacity"][k]["macro_f1_mean"]
    assert cap(s5, "logistic_regression") > 0.75 and cap(m37, "logistic_regression") < 0.3
    assert cap(sb, "lightgbm") > cap(mb, "lightgbm") > evb
    lr_share_s = (cap(s5, "logistic_regression") - cap(s5, "majority_class")) / (cap(s5, "lightgbm") - cap(s5, "majority_class"))
    lr_share_m = (cap(m37, "logistic_regression") - cap(m37, "majority_class")) / (cap(m37, "lightgbm") - cap(m37, "majority_class"))
    n30, t18 = G["msu_37class_n30000"]["lightgbm"]["macro_f1_mean"], G["msu_37class_top18"]["lightgbm"]["macro_f1_mean"]
    n30b, t18b = G["msu_binary_n30000"]["lightgbm"]["macro_f1_mean"], G["msu_binary_top18"]["lightgbm"]["macro_f1_mean"]
    wo = sat["without_rows_on_bounds"]
    ind, maj = sat["bound_indicators_only_lightgbm"]["macro_f1_mean"], sat["bound_indicators_only_majority"]["macro_f1_mean"]
    assert ind > maj + 0.2
    assert max(n30, t18) < cap(s5, "lightgbm") - 0.05
    per = sat["per_feature_class_fraction_on_bound"]

    h = D.after(anchor, "3.6.1 STATISTICAL POWER", "4.6.3 IS THE SYNTHETIC DATASET EASIER? A DATA-COMPLEXITY ANALYSIS")
    p = D.after(h, "Two findings stand out.",
                "The factors offered in Section 4.6 to explain the performance gap were tested with identical code on "
                "both datasets (standardized features). (i) Three data-complexity measures from the literature on "
                "classification difficulty: the maximum Fisher discriminant ratio of a single feature (F1, averaged "
                "over one-vs-rest class problems; higher = easier), the fraction of points on the class boundary of a "
                "minimum spanning tree (N1; lower = easier), and the leave-one-out 1-nearest-neighbour error (lower = "
                "easier), the last two on a stratified subsample of 4,000 rows. (ii) A model-capacity curve: Macro-F1 of "
                "models of increasing capacity, from a majority-class predictor and logistic regression over decision "
                "trees of depth 1, 2, 4 and 8 to the LightGBM of Tables 2 and 8, under the same five-fold protocol. On an "
                "easy distribution, low-capacity models already come close to the ceiling. (iii) Controlled comparisons "
                "that remove known confounders: the same binary attack-versus-non-attack task on both datasets, the "
                "MSU/ORNL data subsampled to the 30,000 rows of the synthetic dataset, and restricted to its 18 "
                "highest-ranked SHAP features (Table 8b) to match the synthetic feature count. (iv) Artefacts of the "
                "synthetic generator, which clips every feature to a fixed range, so that some rows lie exactly on a "
                "range bound. Table 8f summarises (i)-(iii).")
    rows = [["Measure", "Synthetic, 5 classes", "Synthetic, binary", "MSU/ORNL, 37 classes", "MSU/ORNL, binary"],
            ["Fisher ratio F1 (↑ easier)", f3(cs["F1_max_fisher_ratio"]), f3(sb["complexity"]["F1_max_fisher_ratio"]),
             f3(cm["F1_max_fisher_ratio"]), f3(mb["complexity"]["F1_max_fisher_ratio"])],
            ["Boundary fraction N1 (↓ easier)", f3(cs["N1_boundary_fraction"]), f3(sb["complexity"]["N1_boundary_fraction"]),
             f3(cm["N1_boundary_fraction"]), f3(mb["complexity"]["N1_boundary_fraction"])],
            ["1-NN error (↓ easier)", f3(cs["knn1_loo_error"]), f3(sb["complexity"]["knn1_loo_error"]),
             f3(cm["knn1_loo_error"]), f3(mb["complexity"]["knn1_loo_error"])]]
    labels = [("majority_class", "Macro-F1: majority class"), ("logistic_regression", "Macro-F1: logistic regression"),
              ("tree_depth1", "Macro-F1: tree, depth 1"), ("tree_depth2", "Macro-F1: tree, depth 2"),
              ("tree_depth4", "Macro-F1: tree, depth 4"), ("tree_depth8", "Macro-F1: tree, depth 8"),
              ("lightgbm", "Macro-F1: LightGBM (random 5-fold)")]
    for k, lab in labels:
        rows.append([lab] + [f3(cap(r, k)) for r in (s5, sb, m37, mb)])
    rows.append(["Macro-F1: best model, event-disjoint (Table 8e)", "n/a (independent rows)", "n/a (independent rows)",
                 f3(ev37), f3(evb)])
    rows.append(["Macro-F1: LightGBM, 30,000 rows", "as above (30,000 rows)", "as above (30,000 rows)", f3(n30), f3(n30b)])
    rows.append(["Macro-F1: LightGBM, top-18 features", "as above (18 features)", "as above (18 features)", f3(t18), f3(t18b)])
    cap_t = D.table(p, "Table 2. Model Comparison", rows,
                    "Table 8f. Data-Complexity Measures, Model-Capacity Curve, and Controlled Comparisons for the "
                    "Synthetic and MSU/ORNL Datasets (Macro-F1, Five-Fold Cross-Validation)")
    top_bounds = sorted(((f, c, v) for f, d in per.items() for c, v in d.items() if c != "normal"),
                        key=lambda x: -x[2])[:3]
    bt = "; ".join(f"{pct(v)} of {c} rows on a bound of {f} (normal rows: {pct(per[f]['normal'])})" for f, c, v in top_bounds)
    p2 = D.after(cap_t, "Two findings stand out.",
                 "Every measure says that the synthetic task is much easier. The best single feature separates the "
                 f"synthetic classes with a Fisher ratio of {f3(cs['F1_max_fisher_ratio'])}, against "
                 f"{f3(cm['F1_max_fisher_ratio'])} for MSU/ORNL; {pct(cs['N1_boundary_fraction'])} of synthetic points "
                 f"lie on a class boundary, against {pct(cm['N1_boundary_fraction'])}. A logistic regression already "
                 f"reaches a Macro-F1 of {f3(cap(s5, 'logistic_regression'))} on the synthetic data, "
                 f"{pct(lr_share_s, 0)} of the way from the majority-class predictor to LightGBM, whereas on MSU/ORNL it "
                 f"reaches {f3(cap(m37, 'logistic_regression'))} ({pct(lr_share_m, 0)} of the way). The gap persists "
                 "when the confounders named in Section 4.6 are removed. With the same binary label space, LightGBM "
                 f"reaches {f3(cap(sb, 'lightgbm'))} on synthetic data against {f3(cap(mb, 'lightgbm'))} on MSU/ORNL "
                 f"under random splitting (Random Forest, the best MSU/ORNL binary model in Table 8: "
                 f"{f3(n['msu']['binary']['summary']['random_forest']['macro_f1_mean'])}) and {f3(evb)} under "
                 "event-disjoint validation. Matching the synthetic dataset's "
                 f"size or feature count does not close the gap either: with 30,000 rows the MSU/ORNL 37-class Macro-F1 "
                 f"falls to {f3(n30)}, and with its 18 highest-ranked features it is {f3(t18)}, so neither the number "
                 "of rows nor the number of features explains the gap. Label granularity "
                 "therefore accounts for only part of the difference; most of it is in the data.")
    p3 = D.after(p2, "Two findings stand out.",
                 "Part of this ease is an artefact of how the synthetic data were generated. Because the generator clips "
                 "every feature to a fixed range, many rows sit exactly on a range bound, and how often they do depends "
                 f"strongly on the class: {bt}. A LightGBM that is given only the 36 binary indicators 'on the lower "
                 "bound' and 'on the upper bound' of each continuous feature, and none of the measured values, reaches a "
                 f"Macro-F1 of {f3(ind)} (majority class: {f3(maj)}). These indicators have no physical meaning; a "
                 "classifier can exploit them only on generated data. When every row that lies on any bound is removed, "
                 f"{wo['n_rows']:,} rows remain, most of them normal, and LightGBM reaches {f3(wo['lightgbm']['macro_f1_mean'])} "
                 "on them, so the classes remain separable without the artefact, but less so. In summary, the synthetic "
                 "dataset is easier than the real one, for two documented reasons: its attack classes are generated as "
                 "distinct, well-separated signatures (including features such as packet rate that MSU/ORNL does not "
                 "contain, Section 4.7), and its generator leaves class-dependent clipping artefacts. The synthetic "
                 f"Macro-F1 of {f3(cap(s5, 'lightgbm'))} should therefore be read as performance on a controlled, "
                 "easier benchmark and not as an estimate of performance on real telemetry, for which the event-disjoint "
                 "MSU/ORNL values of Table 8e are the better guide.")
    return p3


# --------------------------------------------------------------------------
# Section 4.10 -- limitations brought in line with the new results
# --------------------------------------------------------------------------
def limitations(D: Doc, n, c5):
    S = n["single"]["summary"]
    ev37 = max(c5["v"]["multiclass"][m]["B"] for m in ("lightgbm", "random_forest"))
    D.edit("Two further implications follow.",
           ("per-class explanations for the 37 scenarios, a stability check analogous to Section 4.3 were not performed "
            "on MSU/ORNL, and are noted",
            "per-class explanations for the 37 scenarios and a stability check analogous to Section 4.3 were not "
            "performed on MSU/ORNL (its faithfulness was tested, Section 4.3.1) and are noted"))
    D.edit("Several limitations should be considered when interpreting these results.",
           ("The substantially lower performance on the real MSU/ORNL dataset (Sections 4.6, 4.7) should be treated as "
            "the more representative estimate of real-world performance.",
            "The data-complexity analysis of Section 4.6.3 shows that the synthetic task is substantially easier than "
            "the real one, partly because of class-dependent clipping artefacts of the generator. The substantially "
            "lower performance on the real MSU/ORNL dataset, in particular under event-disjoint validation (Section "
            f"4.6.2; best 37-class Macro-F1 {f3(ev37)}), should be treated as the more representative estimate of "
            "real-world performance."),
           ("should not be assumed to transfer to another without retraining and re-validation on site-specific data.",
            "should not be assumed to transfer to another without retraining and re-validation on site-specific data; "
            "Section 4.6.1 shows directly that the synthetic-trained model does not transfer to MSU/ORNL. All "
            "MSU/ORNL models classify single rows; features computed over windows of consecutive measurements, which "
            "may recover performance on unseen events, were not evaluated."))
    D.replace("Second, the distribution-shift results (Section 4.5) show",
              "Second, the drift experiments of Section 4.5.1 show that Kolmogorov-Smirnov and PSI monitoring detects "
              "the distribution shifts of Table 7 and that label-free re-centring and shadow retraining recover most of "
              "the lost performance, but they were carried out with simulated offsets on synthetic data. Drifts that "
              "change the shape of a distribution, gradual sensor ageing, and the labelling effort that shadow "
              "retraining needs in the field were not tested, and single-feature offsets of 0.1 standard deviations "
              f"were detected in only {pct(S['0.1']['ks_detection_rate_mean'])} of windows on average; most such offsets "
              "were harmless, but one (line load) was not. Re-centring corrects pure offsets only.")
    D.edit("Fourth, the SHAP analysis of the MSU/ORNL dataset (Section 4.7)",
           ("is based on a random sample of held-out rows, and was not accompanied by a stability check on that dataset.",
            "is based on a random sample of held-out rows, and was not accompanied by a stability check on that dataset "
            "(its faithfulness was tested, Section 4.3.1). It explains models evaluated under random row-level "
            "splitting, whose performance Section 4.6.2 shows to be optimistic."))
    D.edit("Sixth, as scoped in Section 3.1, the prevention/policy layer is evaluated post hoc",
           ("the prevention/policy layer is evaluated post hoc (Section 4.9) by passing cross-validated detection "
            "predictions through the deployed policy function;",
            "the prevention/policy layer is evaluated offline and open-loop (Section 4.9) by passing cross-validated "
            "detection predictions through the deployed policy function, so the reported action-selection accuracy is "
            "not a rate of prevented attacks, and its confidence threshold was analysed only for the well-calibrated "
            "synthetic-data model (Section 4.9.1);"))
    D.replace("Finally, the SHAP explanations of Section 4.3 were validated for stability",
              "Finally, the SHAP explanations were tested for stability (Section 4.3) and for faithfulness to the "
              "model on both datasets (Section 4.3.1). Faithfulness to the model is not evidence that a highly ranked "
              "feature is the physical cause of an attack, and robustness of the explanations to adversarially "
              "crafted inputs was not tested. Also not addressed is a systematic comparison against deep-learning "
              "architectures beyond the CNN-LSTM and Transformer baselines of Section 4.1 -- specifically GNNs and "
              "TabNet, for the reasons stated there.")


def c9_finish(D: Doc, n, spec_rows, tb):
    dl = n["dl"]["summary"]
    lg, cl, tr = dl["lightgbm_same_90pct_split"], dl["cnn_lstm"], dl["transformer"]
    assert lg["macro_f1_mean"] > max(cl["macro_f1_mean"], tr["macro_f1_mean"])
    t2 = n["cv"]
    rows = [["Model", "Macro-F1 (mean ± std)", "Accuracy (mean ± std)", "Mean train time / fold"]]
    rows.append(["LightGBM (Table 2; full training fold)", f"{f3(t2['lightgbm']['f1'][0])} ± "
                 f"{f3(t2['lightgbm']['f1'][1])}", f"{pct(t2['lightgbm']['acc'][0])} ± {pct(t2['lightgbm']['acc'][1])}",
                 "--"])
    for name, r in (("LightGBM (same 90% training part as the networks)", lg), ("CNN-LSTM", cl),
                    ("Transformer encoder", tr)):
        rows.append([name, f"{f3(r['macro_f1_mean'])} ± {f3(r['macro_f1_std'])}",
                     f"{pct(r['accuracy_mean'])} ± {pct(r['accuracy_std'])}", f"{r['train_seconds_mean']:.0f} s"])
    new = clone_table(tb, tb, rows, D.mark)
    tb._tbl.getparent().remove(tb._tbl)
    D.edit("Table 3b. Tree-Ensemble vs. Deep-Learning Baselines",
           ("(5-fold CV, identical protocol)",
            "(5-fold CV, identical protocol; networks specified in Table 3d; training times on the machine of Section "
            "3.9 are indicative only)"))
    cap = D.para("Table 3b. Tree-Ensemble vs. Deep-Learning Baselines")
    cap3d = D.table(cap, "Table 10. Ground-Truth Class-to-Action Mapping", spec_rows,
                    "Table 3d. Complete Specification of the Deep-Learning Baselines (scripts/dl_baselines.py)")
    wc, wt = cl["wilcoxon_vs_lightgbm"], tr["wilcoxon_vs_lightgbm"]
    D.replace("LightGBM outperformed both deep-learning baselines on every one of the five folds",
              f"Trained on the same 90% of each training fold, LightGBM reached a Macro-F1 of {f3(lg['macro_f1_mean'])}, "
              f"against {f3(cl['macro_f1_mean'])} for the CNN-LSTM and {f3(tr['macro_f1_mean'])} for the Transformer, "
              f"and it was better on {wc['lightgbm_wins']} and {wt['lightgbm_wins']} of the five folds, respectively "
              f"(paired Wilcoxon signed-rank test, p = {wc['p']:.4f} and p = {wt['p']:.4f}). With n = 5 paired folds, "
              "p = 0.0625 is the smallest attainable p-value, so a consistent directional advantage cannot be confirmed "
              "at α = 0.05 -- the same statistical-power caveat that applies to Table 3; because of the training cost of "
              "the networks, this comparison was not extended to repeated cross-validation. On 18 tabular "
              "features and about 30,000 rows, gradient-boosted trees remain at least as accurate as these deep "
              "architectures, consistent with the broader empirical finding for small-to-medium tabular data.")
    D.replace_image("Fig. 9b. Macro-F1 of tree-ensemble vs. deep-learning models", FIG / "dl_baselines.png")
    D.replace("Fig. 9b. Macro-F1 of tree-ensemble vs. deep-learning models",
              "**Fig. 9b. Macro-F1 of LightGBM vs. the Deep-Learning Baselines Under Identical 5-Fold CV (error bars = ± 1 "
              "std).** All three models are trained on the same 90% of each training fold (the remaining 10% is the "
              "networks' early-stopping set) and tested on the same outer fold.")
    return cap3d


# --------------------------------------------------------------------------
# Comment 10 -- earlier tables re-generated in the final environment: every
# cell is rewritten from the new result files, and only cells whose value
# changed are touched (and therefore highlighted)
# --------------------------------------------------------------------------
def pm4(m, s):
    return f"{m:.4f} ± {s:.4f}"


def pmpct(m, s):
    return f"{100 * m:.2f}% ± {100 * s:.2f}%"


def sync_cells(D: Doc, caption, cells, changes):
    from docx_edit import set_cell
    t = D.table_after_caption(caption)
    for (label, col), text in cells.items():
        labels = label if isinstance(label, tuple) else (label,)
        row = [r for r in t.rows
               if all(norm(r.cells[i].text) == norm(lb) or (len(lb) > 3 and norm(r.cells[i].text).startswith(norm(lb)))
                      for i, lb in enumerate(labels))]
        assert len(row) == 1, (caption, label)
        cell = row[0].cells[col]
        if norm(cell.text) != norm(text):
            changes.append((caption.split(". ")[0], " / ".join(labels), col, cell.text.strip(), text))
            set_cell(cell, text, D.mark)


def sync_table2_3(D: Doc, n, changes):
    cv = n["cv"]
    cells = {}
    for k, lab in MODEL.items():
        r = cv[k]
        cells[(lab, 1)] = pm4(*r["f1"])
        cells[(lab, 2)] = pmpct(*r["acc"])
        cells[(lab, 3)] = pm4(*r["precision"])
        cells[(lab, 4)] = pm4(*r["recall"])
    sync_cells(D, "Table 2. Model Comparison", cells, changes)
    w = n["wilcoxon"]["comparisons"]
    cells = {}
    for k in ("random_forest", "xgboost", "catboost"):
        c = w[f"lightgbm_vs_{k}"]
        lab = f"LightGBM vs. {MODEL[k]}"
        cells[(lab, 1)] = f"{c['statistic']:.1f}"
        cells[(lab, 2)] = f"{c['p_value']:.4g}"
        cells[(lab, 3)] = "Yes" if c["p_value"] < 0.05 else "No"
    sync_cells(D, "Table 3. Paired Wilcoxon", cells, changes)
    if any(ch[0].startswith("Table 2") for ch in changes):
        D.replace_image("Fig. 9. Model Comparison Across", ROOT / "artifacts" / "visualizations" / "cv_model_comparison.png")


def swap(D: Doc, prefix, pairs):
    """D.edit, but only for pairs whose value actually changed (unchanged values stay unhighlighted)."""
    pairs = [(o, nw) for o, nw in pairs if o != nw]
    if pairs:
        D.edit(prefix, *pairs)
    return pairs


def p3(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def sync_section361(D: Doc, n, changes):
    rep, tt = n["rep"], n["ttest"]["comparisons"]
    S, W = rep["summary"], rep["wilcoxon_vs_best"]
    assert rep["best_model"] == "lightgbm"
    assert tt["random_forest"]["p"] >= 0.05 and tt["xgboost"]["p"] >= 0.05 and tt["catboost"]["p"] < 0.001
    assert all(W[k]["p"] < 0.001 for k in W)
    sw = swap(D, "The n = 5 paired observations underlying Table 3",
              [("0.9300 ± 0.0045 for LightGBM", f"{pm4(S['lightgbm']['mean'], S['lightgbm']['std'])} for LightGBM"),
               ("0.9275 ± 0.0042 for Random Forest", f"{pm4(S['random_forest']['mean'], S['random_forest']['std'])} for Random Forest"),
               ("0.9275 ± 0.0046 for XGBoost", f"{pm4(S['xgboost']['mean'], S['xgboost']['std'])} for XGBoost"),
               ("0.8947 ± 0.0047 for CatBoost", f"{pm4(S['catboost']['mean'], S['catboost']['std'])} for CatBoost")])
    changes += [("Section 3.6.1 text", o, nw) for o, nw in sw]
    d_rf, d_xg = W["random_forest"]["mean_diff"], W["xgboost"]["mean_diff"]
    if f"{d_rf:.4f}" == f"{d_xg:.4f}":
        small = f"{d_rf:.4f} Macro-F1, or {d_rf * 100:.2f} percentage points)"
    else:
        small = (f"{d_rf:.4f} and {d_xg:.4f} Macro-F1, or {d_rf * 100:.2f} and {d_xg * 100:.2f} "
                 f"percentage points)")
    sw = swap(D, "With n = 50, the Wilcoxon test is significant",
              [("(mean difference 0.035, p < 0.001)", f"(mean difference {W['catboost']['mean_diff']:.3f}, {p3(tt['catboost']['p'])})"),
               ("Random Forest (p = 0.105)", f"Random Forest ({p3(tt['random_forest']['p'])})"),
               ("XGBoost (p = 0.093)", f"XGBoost ({p3(tt['xgboost']['p'])})"),
               ("in 40 and 41 of the 50 folds", f"in {W['random_forest']['best_wins']} and {W['xgboost']['best_wins']} of the 50 folds"),
               ("small (0.0024 Macro-F1, about a quarter of a percentage point)", f"small ({small}")])
    changes += [("Section 3.6.1 text", o, nw) for o, nw in sw]
    cells = {}
    for k in ("random_forest", "xgboost", "catboost"):
        lab = f"LightGBM vs. {MODEL[k]}"
        cells[(lab, 1)] = f"+{W[k]['mean_diff']:.4f}"
        cells[(lab, 2)] = f"{W[k]['best_wins']} / 50"
        cells[(lab, 3)] = f"{W[k]['W']:.1f} ({p3(W[k]['p'])})"
        cells[(lab, 4)] = f"{tt[k]['t']:.2f} ({p3(tt[k]['p'])})"
    sync_cells(D, "Table 3c. Repeated Stratified", cells, changes)
    dl = n["dl"]["summary"]
    D.edit("The repeated analysis covers the tree-ensemble comparison",
           ("(approximately 109 s and 154 s per fold for the CNN-LSTM and Transformer, respectively, versus about 2 s for "
            "LightGBM)",
            f"(about {dl['cnn_lstm']['train_seconds_mean']:.0f} s and {dl['transformer']['train_seconds_mean']:.0f} s per "
            f"fold for the CNN-LSTM and Transformer, respectively, versus about "
            f"{dl['lightgbm_same_90pct_split']['train_seconds_mean']:.0f} s for LightGBM, Table 3b)"))


# --------------------------------------------------------------------------
# Response letter (manuscript/build_response_letter_r4.py calls this)
# --------------------------------------------------------------------------
COMMENTS = (HERE / "reviewer_comments_r4.txt").read_text(encoding="utf-8").strip().splitlines()


def response_sections(n):
    c1 = c1_numbers(n)
    P = c3_numbers(n)
    T = n["tau"]
    s05, s07, s09 = (tau_row(n, t) for t in (0.5, 0.7, 0.9))
    sel = [x["selected_tau"] for x in T["nested_selection"]]
    d = n["drift"]
    z, c02, c05 = drift_row(n, 0.0), drift_row(n, 0.02), drift_row(n, 0.05)
    m = d["mitigation"]
    m0 = [r["M0_no_action"] for r in m]
    m1 = [r["M1_recentering"] for r in m]
    n2000 = [next(x for x in r["M2_shadow"] if x["n_labelled"] == 2000)["active_model_eval_f1"] for r in m]
    n250 = [next(x for x in r["M2_shadow"] if x["n_labelled"] == 250)["active_model_eval_f1"] for r in m]
    g = d["no_drift_guard"]
    S1 = n["single"]["summary"]
    ll05 = next(r for r in n["single"]["rows"] if r["feature"] == "line_load_percent" and r["k_sigma"] == 0.05)
    c5 = c5_numbers(n)
    v = c5["v"]
    nn = n["gap"]["msu_nn_events_37class"]
    G = n["gap"]
    sat = G["synthetic_saturation"]
    cap = lambda r, k: G[r]["capacity"][k]["macro_f1_mean"]
    F = n["faith"]
    fs, fm = F["synthetic"]["auc"]["mean"], F["msu_37class"]["auc"]["mean"]
    stab = stability_numbers(n)
    dl = n["dl"]["summary"]
    sel_dl = n["dl"]["selection"]
    e = n["env"]
    best = lambda x, k: max(x["lightgbm"][k], x["random_forest"][k])
    out = []
    P_ = lambda md: out.append(("p", md))
    B_ = lambda md: out.append(("b", md))
    blank = lambda: out.append(("blank", ""))

    P_("**Response to Reviewers**")
    P_("*Manuscript: Explainable AI-Based Intrusion Detection for Smart Grid Cyber-Physical Systems*")
    P_("Dear Editor,")
    P_("We thank the editor and the reviewer for this further round of comments. The reviewer asked that the "
       "remaining issues be resolved scientifically rather than by discussion or by acknowledging limitations. We have "
       "therefore run a new experiment for every comment that can be answered experimentally (comments 1, 2, 4, 5, 7, "
       "8 and 9), re-generated every earlier result in a single computational environment (comment 10), and, where an "
       "experiment was not possible (closed-loop validation, comment 3), narrowed the claims and said so explicitly. "
       "Several of the new results are not in our favour, and we report them as found: the synthetic-trained model "
       "does not transfer to the MSU/ORNL data without retraining (comment 1), the MSU/ORNL results of the previous "
       "version are optimistic because rows of the same recorded event fell into training and test folds (comment 5), "
       "and the synthetic dataset is substantially easier than real telemetry, partly because of generator artefacts "
       "(comment 8). The Abstract, contributions, discussion, limitations and Conclusion have been revised to match "
       "these results.")
    P_("Every number in the revised manuscript and in this letter was produced by the scripts in the project's code "
       "repository and is read from their result files by the script that builds the manuscript; the file "
       "manuscript/PROVENANCE_R4.md lists, for every new table and figure, the command and the result file. Reviewer "
       "comments are reproduced in italics, each followed by our response and the location of the changes. In the "
       "version with changes highlighted, all text added or changed in this revision is marked in yellow.")
    P_("**Summary of the main changes**")
    for b in [
        f"New Section 4.6.1 and Table 8d: direct test of the synthetic-trained model on MSU/ORNL without retraining "
        f"(chance level, ROC-AUC {f3(c1['auc_lo'])}-{f3(c1['auc_hi'])}); transferability claims removed.",
        "New Section 4.5.1, Table 7b and Fig. 14b: experimental evaluation of KS/PSI/confidence drift detection, "
        "label-free re-centring and shadow retraining, including its sensitivity limit.",
        "Sections 3.1 and 4.9: no closed-loop validation was possible; the 98.1% is renamed action-selection accuracy "
        "and all prevention claims are limited to offline action selection.",
        "New Section 4.9.1, Table 10b and Fig. 16: τ sweep, calibration (ECE, Brier score) and nested data-driven "
        "selection of τ.",
        f"New Section 4.6.2 and Table 8e: event-disjoint and time-disjoint validation on MSU/ORNL (best 37-class Macro-F1 "
        f"{f3(best(v['multiclass'], 'B'))} instead of {f3(best(v['multiclass'], 'A'))}); the event-disjoint values are "
        "now the reported estimate for unseen events.",
        "Rewritten Section 1.1: the specific novelty, and what the paper does not claim.",
        "New Section 4.3.1, Table 5b and Fig. 12b: deletion/insertion and remove-and-retrain faithfulness tests of the "
        "SHAP explanations on both datasets.",
        "New Section 4.6.3 and Table 8f: data-complexity measures, model-capacity curve, controlled comparisons and "
        "generator clipping artefacts.",
        "New Table 3d: complete specification of the deep-learning baselines, which were re-implemented and re-run "
        "(Table 3b and Fig. 9b updated).",
        "Section 3.9: one final computational environment for every reported result, with a lock file and a single "
        "reproduction script; values that changed on re-running are listed under comment 10.",
        "Abstract shortened to the 200-word limit of Scientific Reports and rewritten; 'Conflicts of Interest' renamed "
        "'Competing interests'."]:
        B_(b)
    P_("**Reviewer 4**")
    P_("*\"The authors have responded to all reviewer comments, but the following issues should be addressed further "
       "to achieve actual scientific resolution, rather than only adding discussion or acknowledging limitations.\"*")
    P_("**Response:** We thank the reviewer for pressing these points. Each is answered below with a new experiment, "
       "or, where that was impossible, with an explicit narrowing of the claim.")

    def comment(i, resp, changes, bullets=()):
        P_(f"**Comment {i}**")
        P_(f"*\"{COMMENTS[i - 1]}\"*")
        for j, r in enumerate(resp):
            P_(("**Response:** " if j == 0 else "") + r)
        for b in bullets:
            B_(b)
        P_("**Changes in the manuscript:** " + changes)
        blank()

    comment(1, [
        "No, not in the previous version: every MSU/ORNL result there came from models retrained on MSU/ORNL. We have "
        "now carried out the direct test. Because the datasets share neither features nor labels, we mapped ten "
        "synthetic features onto their physical PMU counterparts (voltage, current, frequency, power factor, active and "
        "reactive power, demand, voltage imbalance, feeder voltage and current), converted magnitudes to the synthetic "
        "operating scale using label-free medians, set the eight features without a counterpart to their synthetic "
        "normal-class median, and used the only shared label space, attack versus non-attack. Two models were applied "
        "with no retraining: the deployed five-class LightGBM and a binary LightGBM trained on the synthetic data "
        "restricted to the ten mapped features, each under two scaling variants.",
        f"All four combinations are at chance level: balanced accuracy {f3(c1['ba_lo'])}-{f3(c1['ba_hi'])}, ROC-AUC "
        f"{f3(c1['auc_lo'])}-{f3(c1['auc_hi'])}, and Macro-F1 at or below that of the trivial all-attack predictor "
        f"({f3(c1['chance']['macro_f1'])}). The failure is not due to the mapped features being uninformative: retrained "
        f"on MSU/ORNL, the same ten features reach a ROC-AUC of {f3(c1['r']['roc_auc_mean'])}, and "
        f"{f3(c1['rev']['roc_auc_mean'])} with whole recordings held out. We have therefore removed every claim of "
        "cross-domain generalisation or model transferability. The manuscript now states that the evidence supports "
        "methodology-level external validity (the same pipeline, retrained, works on real PMU data) but not "
        "transferability of a trained model."],
        "Section 3.2.2; new Section 4.6.1 and Table 8d; Section 4.6 (discussion of the performance gap); Section 1.1; contributions "
        "list; Abstract; Section 4.10 (first limitation); Conclusion.")

    comment(2, [
        "Yes. The design is now implemented and tested with the deployed model, its test split and the shift operator of "
        "Table 7, adding 1% and 2% shifts. Detection is label-free: windows of 250 incoming rows are compared with a "
        "2,000-row training reference by (i) 18 two-sample Kolmogorov-Smirnov tests with Bonferroni correction, (ii) "
        "the Population Stability Index (> 0.2), and (iii) the model's mean confidence. Mitigation compares no action, "
        "label-free re-centring, and shadow retraining with 250-2,000 labelled recent rows, promoted only if it beats "
        "the deployed model on a held-back validation slice.",
        f"KS and PSI detected every tested shift, including 1%, in every window, with false-alarm rates of "
        f"{pct(z['ks_detection_rate'])} and {pct(z['psi_detection_rate'])} per window without drift. The confidence "
        f"proxy proposed in the previous version is much weaker ({pct(c02['conf_detection_rate'])} of windows at 2% and "
        f"{pct(c05['conf_detection_rate'])} at 5%), because the model stays confident under shift, so we no longer "
        f"recommend it as a primary signal. Without action, Macro-F1 fell to {f3(min(m0))}-{f3(max(m0))}; re-centring "
        f"restored {f3(min(m1))}-{f3(max(m1))}, and shadow retraining {f3(min(n2000))}-{f3(max(n2000))} with 2,000 "
        f"labelled rows ({f3(min(n250))}-{f3(max(n250))} with 250). It was promoted in every drifted setting and "
        f"correctly not promoted without drift ({f3(g['candidate_val_f1'])} vs. {f3(g['deployed_val_f1'])}).",
        "We also report the limit of the method rather than only its successes. When a single sensor drifts by k standard "
        f"deviations, KS detection falls to {pct(S1['0.25']['ks_detection_rate_mean'])} of windows at k = 0.25 and "
        f"{pct(S1['0.1']['ks_detection_rate_mean'])} at k = 0.1. Most such small drifts are harmless, but a 0.05 "
        f"standard-deviation offset of line load was detected in only {pct(ll05['ks_detection_rate'])} of windows and "
        f"still reduced Macro-F1 by {f3(ll05['macro_f1_drop'])}. These "
        "experiments use simulated drift on synthetic data; drift in field data remains to be tested, as stated in "
        "Section 4.10."],
        "New Section 4.5.1, Table 7b and Fig. 14b; Section 4.5 (last paragraph); Section 4.10 (second limitation, "
        "rewritten); contributions list; Abstract; Conclusion.")

    comment(3, [
        f"It represents only the correctness of the policy/action mapping. The {pct(P['response_accuracy'])} is the "
        "fraction of the 30,000 cross-validated predictions for which the deployed policy function selects the action "
        "that the sample's true class calls for. It is not a rate of prevented attacks. Closed-loop validation would "
        "require a power-system simulator or hardware-in-the-loop testbed in which the selected actions change the "
        "subsequent grid state and telemetry. Neither was available to us, and we did not want to present an "
        "open-loop replay as closed-loop evidence.",
        "We have therefore limited the claims. The metric is renamed 'action-selection accuracy' throughout. Section "
        "4.9 is retitled 'Offline (open-loop) evaluation of the policy-response layer'. Section 3.1 states that no "
        "closed-loop validation was performed. Table 9, the Abstract, the contribution list, Section 1.1 and the "
        "Conclusion now describe the layer as evaluated offline for action selection, and the paper explicitly does "
        "not claim field-validated or closed-loop prevention."],
        "Section 3.1 (scope statement); Section 4.9 (title, protocol, results paragraphs); Table 9 (operational "
        "validation row); Table 10 caption; Section 1.1; contributions list; Section 4.10 (sixth limitation); Abstract; "
        "Conclusion.")

    comment(4, [
        "τ = 0.7 was a design value; it was not tuned on the test data. We now report its effect. The same out-of-fold "
        "predictions were passed through the deployed policy function for every τ from 0.20 to 0.95 (Table 10b, Fig. "
        f"16). Raising τ from 0.5 to 0.9 lowers the false-block rate from {pct(s05['false_block_rate'], 2)} to "
        f"{pct(s09['false_block_rate'], 2)}, but raises the missed-response rate from "
        f"{pct(s05['missed_response_rate'])} to {pct(s09['missed_response_rate'])}. Action-selection accuracy and "
        f"gated detection Macro-F1 remain almost constant (Macro-F1 {f4(s05['gated_macro_f1'])} at 0.5, "
        f"{f4(s07['gated_macro_f1'])} at 0.7, {f4(s09['gated_macro_f1'])} at 0.9).",
        f"The calibration analysis explains why: the probabilities are well calibrated (ECE "
        f"{f3(T['calibration']['ece'])}, Brier score {f3(T['calibration']['brier'])}), and only "
        f"{pct(s07['fraction_below_tau'])} of predictions fall below 0.7. Predictions above 0.7 are correct "
        f"{pct(T['calibration']['accuracy_conf_ge_0.7'])} of the time, those below only "
        f"{pct(T['calibration']['accuracy_conf_lt_0.7'])}. Finally, τ was selected in a nested cross-validation inside "
        f"each training fold. The selected values ranged from {min(sel):.2f} to {max(sel):.2f}, and on every outer fold "
        "they gave essentially the same result as τ = 0.7. This is the experimental justification for keeping 0.7, and "
        "it shows operators what lowering τ costs."],
        "New Section 4.9.1, Table 10b and Fig. 16; Section 4.10 (sixth limitation).")

    comment(5, [
        "In the previous version we did not, and the reviewer's concern is justified. The MSU/ORNL release consists of "
        f"{c5['struct']['n_files']} recording files, and in each file every scenario is one contiguous, time-ordered run "
        f"of rows ({c5['struct']['n_events']} events in total). We therefore repeated the evaluation with two new "
        "protocols. The first is event-disjoint five-fold cross-validation, which holds out complete recordings, so "
        "every test event is unseen. The second is time-disjoint: within every event, the model trains on the first "
        "80% of rows and is tested on the last 15%, after a 5% gap.",
        f"The results change substantially (Table 8e). Taking the better of LightGBM and Random Forest in each case, the 37-class Macro-F1 "
        f"falls from {f3(best(v['multiclass'], 'A'))} "
        f"to {f3(best(v['multiclass'], 'B'))} (event-disjoint) and {f3(best(v['multiclass'], 'C'))} (time-disjoint). "
        f"The 3-class task falls from {f3(best(v['triple'], 'A'))} to {f3(best(v['triple'], 'B'))}, and the binary task "
        f"from {f3(best(v['binary'], 'A'))} to {f3(best(v['binary'], 'B'))}. The cause is near-duplicate neighbouring "
        f"rows: for {pct(nn['nn_from_same_event_fraction'])} of rows, the nearest neighbour comes from the same event, "
        f"and 1-NN accuracy drops from {pct(nn['knn1_accuracy_any_neighbour'])} to "
        f"{pct(nn['knn1_accuracy_neighbour_from_other_event'])} once same-event neighbours are excluded. We now report "
        "the event-disjoint values as the estimate for unseen events, and the Abstract, Section 4.6, the limitations and "
        "the Conclusion use them. Table 8 is retained and labelled as the random row-level result. The synthetic "
        "dataset has no scenario, event or time structure: its rows are generated independently, which we verified "
        f"({o_runs(c5)}), so the question applies to MSU/ORNL only."],
        "Section 3.2.1 (structure of the synthetic data); new Section 4.6.2 and Table 8e; Table 8 caption and Section 4.6 "
        "text; Table 1; Section 4.10 (first limitation); Abstract; Conclusion.")

    comment(6, [
        "We agree that the building blocks are known, and Section 1.1 has been rewritten to say exactly what is new. "
        "In short, there are three specific results that, to our knowledge, no study in Table 1 reports for a "
        "smart-grid IDPS. (1) A tested account of the synthetic-to-real gap: direct transfer is tested and fails; the "
        "real-data result is validated on held-out events; and the gap is explained with the same instruments on both "
        "datasets (SHAP, ablation, data-complexity measures), which identify generator artefacts and missing "
        "communication-layer evidence. (2) Explanations that are tested for faithfulness, not only shown, on both "
        "datasets. (3) An operational layer that is measured, including its limits: drift monitoring and shadow "
        "retraining are tested experimentally, and the policy layer is evaluated offline together with a "
        "threshold-calibration analysis.",
        "Section 1.1 also states what the paper does not claim: transferability of a trained model, closed-loop "
        "prevention, or superiority of one tree ensemble. We believe this clearer positioning, backed by the new "
        "experiments, addresses the reviewer's question more directly than the previous statement that the novelty "
        "lay in the combination."],
        "Section 1.1 (rewritten); contributions list; Table 1 (proposed-work row).")

    ro = F["synthetic"]["roar"]
    comment(7, [
        "We have added the requested quantitative analysis (Section 4.3.1, Table 5b, Fig. 12b) on both datasets: "
        "deletion and insertion curves with two baselines (training mean and marginal resampling), compared across five "
        "feature orderings (instance-level SHAP, its reverse, global SHAP, LightGBM gain, random), and a remove-and-"
        f"retrain test. On the synthetic data, the deletion area is {f3(fs['shap_instance']['deletion_auc_mean'])} for "
        f"the SHAP order, {f3(fs['random']['deletion_auc_mean'])} for random orders and "
        f"{f3(fs['shap_reverse']['deletion_auc_mean'])} for the reversed SHAP order. On MSU/ORNL the values are "
        f"{f3(fm['shap_instance']['deletion_auc_mean'])}, {f3(fm['random']['deletion_auc_mean'])} and "
        f"{f3(fm['shap_reverse']['deletion_auc_mean'])}, and insertion shows the same ordering. Remove-and-retrain gives "
        "the following results for the top-k SHAP features against k random features: "
        + "; ".join(f"k = {k}: {f3(ro[f'k{k}']['remove_top_shap'])} vs. {f3(ro[f'k{k}']['remove_random_mean'])}"
                    for k in (1, 2, 3, 5)) + ".",
        "We have also qualified the claims. The tests show that the explanations are faithful to the model, not that "
        "the top-ranked features are the physical cause of an attack, and a feature used because of a generator "
        "artefact is explained just as faithfully. The stability check of Section 4.3 was re-run in the final "
        f"environment. Cross-fold stability is unchanged (mean Spearman {f3(stab['mean'])}). The perturbation check, "
        f"re-implemented, gives {f3(stab['p01']['mean'])} (1% noise) and {f3(stab['p05']['mean'])} (5% noise); the "
        "previous version reported 0.988 and 0.971."],
        "Section 4.3 (stability paragraph); new Section 4.3.1, Table 5b and Fig. 12b; contributions list; Section 4.10 "
        "(last limitation); Abstract; Conclusion.")

    comment(8, [
        "We tested this directly (Section 4.6.3, Table 8f), and the answer is that the synthetic data are indeed easier. "
        "We say so plainly in the revision rather than argue otherwise. With identical code on both datasets, every "
        f"data-complexity measure favours the synthetic data. The Fisher ratio is "
        f"{f3(G['synthetic_5class']['complexity']['F1_max_fisher_ratio'])} vs. "
        f"{f3(G['msu_37class']['complexity']['F1_max_fisher_ratio'])}, and the boundary fraction N1 is "
        f"{f3(G['synthetic_5class']['complexity']['N1_boundary_fraction'])} vs. "
        f"{f3(G['msu_37class']['complexity']['N1_boundary_fraction'])}. A logistic regression already reaches "
        f"{f3(cap('synthetic_5class', 'logistic_regression'))} on the synthetic data but "
        f"{f3(cap('msu_37class', 'logistic_regression'))} on MSU/ORNL. The gap remains when label granularity, sample "
        f"size and feature count are matched: on the binary task LightGBM scores {f3(cap('synthetic_binary', 'lightgbm'))} "
        f"vs. {f3(cap('msu_binary', 'lightgbm'))} (Random Forest, the best MSU/ORNL binary model: "
        f"{f3(n['msu']['binary']['summary']['random_forest']['macro_f1_mean'])}), and the MSU/ORNL 37-class score falls to "
        f"{f3(G['msu_37class_n30000']['lightgbm']['macro_f1_mean'])} with 30,000 rows and is "
        f"{f3(G['msu_37class_top18']['lightgbm']['macro_f1_mean'])} with 18 features.",
        "We also identified a concrete artefact of the synthetic generator: because it clips features to fixed ranges, "
        "many rows sit exactly on a bound, at class-dependent rates. A model given only these 'on-bound' indicators, and "
        f"no measured values, reaches a Macro-F1 of {f3(sat['bound_indicators_only_lightgbm']['macro_f1_mean'])}. The "
        "synthetic results are therefore now presented as performance on a controlled, easier benchmark, and the "
        "event-disjoint MSU/ORNL results as the guide to real-world performance."],
        "New Section 4.6.3 and Table 8f; Section 4.6 (second and third paragraphs); Section 4.10 (first limitation); "
        "Abstract; Conclusion.")

    cl, tr = sel_dl["cnn_lstm"]["selected"], sel_dl["transformer"]["selected"]
    comment(9, [
        "Yes. New Table 3d gives the complete specification, and scripts/dl_baselines.py implements it. It covers the "
        "layers of both networks, trainable parameters "
        f"({dl['cnn_lstm']['n_params']:,} and {dl['transformer']['n_params']:,}), the Adam optimizer and loss, the "
        f"selected learning rates ({cl['lr']:g}, {tr['lr']:g}) and widths, batch size 256, at most 50 epochs, L2 "
        "regularisation and dropout, and early stopping on a validation loss with patience 5. It also states the "
        "epochs actually run in each fold, the hyperparameter grid with its selection procedure (selection on the "
        "training part of fold 1 only), and the seeds. The validation set for early stopping is carved out of each "
        "outer training fold before oversampling, so the test fold is never used.",
        "In preparing this table we found that the previous baselines were stopped on the training loss after at most 15 "
        "epochs. Rather than document that configuration, we re-implemented and re-ran both baselines with this "
        "specification. The new results replace the old ones in Table 3b and Fig. 9b: CNN-LSTM "
        f"{f3(dl['cnn_lstm']['macro_f1_mean'])} ± {f3(dl['cnn_lstm']['macro_f1_std'])}, Transformer "
        f"{f3(dl['transformer']['macro_f1_mean'])} ± {f3(dl['transformer']['macro_f1_std'])}, and LightGBM on the same "
        f"training data {f3(dl['lightgbm_same_90pct_split']['macro_f1_mean'])}. The conclusion of Section 4.1 is "
        "unchanged."],
        "Section 3.9 (hyperparameters); Section 4.1 (first and second paragraphs); Table 3b (updated); new Table 3d; "
        "Fig. 9b (updated); Section 3.6.1 (training times).")

    pk = e["packages"]
    lat = latency_numbers(n)
    comment(10, [
        "They were not fully consistent before, and we have fixed this. The previous version combined results from a "
        "Windows environment (Python 3.12, scikit-learn 1.8, XGBoost 3.4, SHAP 0.52) and a Linux environment (Python "
        "3.11, scikit-learn 1.9, XGBoost 3.2, SHAP 0.51, SciPy 1.17). For this revision, every reported result, old and "
        f"new, was regenerated in one environment: {env_sentence(n)}, on Linux. The full package list is provided as "
        "environment/requirements-lock.txt, and reproduce_all.sh regenerates every table and figure.",
        "Re-running everything reproduced every other reported value exactly. The values that changed, all by small "
        "amounts and mostly for XGBoost (whose version changed from 3.2 to 3.4.1), are updated and highlighted in the "
        "manuscript, and none changes a conclusion:"],
        "Section 3.9 (rewritten); Code Availability; Section 4.8 and Table 9; Section 4.9 (policy latency); the updated "
        "values listed above.", bullets=changes_bullets(n) + [
        f"Section 4.8 and Table 9 (latency, re-measured on the final machine): mean {lat['mean_ms']:.2f} ms (previously "
        f"1.70 ms on a Windows workstation), p99 {lat['p99_ms']:.2f} ms (2.70 ms), throughput {lat['tp1']:,.0f} samples/s "
        f"at batch size 1 (610) and {lat['tp512']:,.0f} samples/s at batch size 512 (46,336); policy latency "
        f"{n['pol']['latency_us']['mean']:.2f} μs per decision (1.21 μs).",
        f"Section 4.3 (SHAP perturbation stability, re-implemented): {f3(stab['p01']['mean'])} and "
        f"{f3(stab['p05']['mean'])} at 1% and 5% noise (previously 0.988 and 0.971).",
        "Section 4.1, Table 3b and Fig. 9b: the re-implemented deep-learning baselines (comment 9)."])

    P_("We hope that the revised manuscript now resolves the reviewer's comments. We thank the reviewer again for "
       "comments that led to substantive corrections of the paper.")
    P_("Yours sincerely,")
    P_("The authors")
    return out


def o_runs(c5):
    o = c5["order"]
    return (f"in file order the class label changes {o['label_runs_in_file_order']:,} times, against "
            f"{o['expected_runs_if_independent']:,.0f} expected for independent rows")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
CHANGES_FILE = R4 / "manuscript_value_changes.json"


def build(n, mark, out, steps=None):
    D = Doc(SOURCE, mark)
    changes = []
    run = lambda name: steps is None or name in steps
    if run("c6"):
        c6(D, n)
        declarations(D)
    if run("c1"):
        a = c1(D, n)
        if run("c5"):
            cap, c5n = c5(D, n, a)
            a = c5_text(D, n, cap, c5n)
            if run("c8"):
                c8(D, n, a, c5n)
            section46_edits(D, n, c5n)
            section46_captions(D)
            if run("limits"):
                limitations(D, n, c5n)
    if run("c3"):
        c3(D, n)
        c4(D, n, D.para("These results establish that the policy engine"))
    if run("c2"):
        c2(D, n)
    if run("c7"):
        c7(D, n)
    if run("c9"):
        spec, tb = c9(D, n)
        c9_finish(D, n, spec, tb)
    if run("sync"):
        sync_table2_3(D, n, changes)
        sync_section361(D, n, changes)
        sync_rest(D, n, changes)
    if run("front"):
        abstract(D, n)
        conclusion(D, n)
    if run("c10"):
        c10(D, n, repro_sentence(changes))
        CHANGES_FILE.write_text(json.dumps(changes, indent=1, ensure_ascii=False))
    D.d.save(str(out))
    return changes


def main():
    n = load()
    for mark, name in ((True, "Manuscript_R4_highlighted.docx"), (False, "Manuscript_R4_clean.docx")):
        ch = build(n, mark, HERE / name)
        print(f"Saved manuscript/{name} ({len(ch)} re-run values changed)")



# --------------------------------------------------------------------------
# Abstract (Scientific Reports: at most 200 words, no references) and Conclusion
# --------------------------------------------------------------------------
def headline(n):
    c1 = c1_numbers(n)
    c5 = c5_numbers(n)["v"]
    m = n["drift"]["mitigation"]
    best = lambda x, k: max(x["lightgbm"][k], x["random_forest"][k])
    return {"lgb": n["cv"]["lightgbm"]["f1"], "acc": n["cv"]["lightgbm"]["acc"],
            "msuA": c5["multiclass"]["lightgbm"]["A"], "msuB": best(c5["multiclass"], "B"),
            "auc_lo": c1["auc_lo"], "auc_hi": c1["auc_hi"],
            "m0": min(r["M0_no_action"] for r in m),
            "m2": min(next(x for x in r["M2_shadow"] if x["n_labelled"] == 2000)["active_model_eval_f1"] for r in m),
            "asa": c3_numbers(n)["response_accuracy"], "c1": c1, "c5": c5}


def abstract(D: Doc, n):
    h = headline(n)
    txt = ("Smart grids couple communication networks with physical power systems, exposing them to cyber-physical "
           "attacks such as denial of service, false data injection and tampering. We present an explainable intrusion detection and prevention framework that combines multidomain "
           "features, tree-ensemble classifiers, SHAP explanations, a rule-based response layer and a real-time service, "
           "and test each of its claims explicitly. On a 30,000-sample synthetic dataset, LightGBM reached a Macro-F1 "
           f"of {f3(h['lgb'][0])} under leakage-safe cross-validation, statistically equivalent to Random Forest and "
           "XGBoost; a data-complexity analysis shows that this dataset is much easier than real telemetry, partly "
           "because of generator artefacts. Retrained on the real MSU/ORNL power-system dataset, the pipeline reached "
           f"{f3(h['msuA'])} for 37 classes with random splits but at most {f3(h['msuB'])} on held-out recorded events, and "
           "the synthetic-trained model transferred to it at chance level (ROC-AUC "
           f"{f3(h['auc_lo'])}-{f3(h['auc_hi'])}). Deletion and insertion tests showed that the SHAP rankings are faithful "
           "to the models. Kolmogorov-Smirnov and population-stability monitors detected simulated sensor drift, and "
           f"shadow retraining restored Macro-F1 from as low as {f3(h['m0'])} to at least {f3(h['m2'])}. Offline, the "
           f"response layer selected the intended action for {pct(h['asa'])} of samples; closed-loop prevention was not "
           "evaluated.")
    words = len(txt.split())
    assert words <= 200, words
    D.replace("Smart grids integrate digital communication infrastructure", txt)
    return words


def conclusion(D: Doc, n):
    h = headline(n)
    W, tt = n["rep"]["wilcoxon_vs_best"], n["ttest"]["comparisons"]
    stab = stability_numbers(n)
    abl = n["abl"]
    d = n["drift"]
    z = drift_row(n, 0.0)
    S1 = n["single"]["summary"]
    lat = latency_numbers(n)
    P = c3_numbers(n)
    T = n["tau"]
    gs = n["msu_shap"]["group_share"]
    cyber = gs["PMU relay status flags"] + gs["Control-panel / relay log"] + gs["Snort IDS log"]
    comm_drop = ablation_drop(n, "communication")
    D.replace("This paper presented an explainable AI-based intrusion detection engine",
              "This paper presented an explainable AI-based intrusion detection engine for smart grid cyber-physical "
              "systems, the core of an IDPS whose rule-based policy-response layer is described in Section 3.1, and "
              "tested explicitly what such a system can and cannot be shown to do. Under leakage-safe stratified "
              "five-fold cross-validation on the synthetic dataset, LightGBM achieved the highest mean performance "
              f"(Macro-F1 {f3(h['lgb'][0])} ± {f3(h['lgb'][1])}, accuracy {pct(h['acc'][0])} ± {pct(h['acc'][1])}), but "
              "repeated cross-validation with 50 paired folds showed its advantage over Random Forest and XGBoost to be "
              f"small ({W['random_forest']['mean_diff']:.4f} and {W['xgboost']['mean_diff']:.4f} Macro-F1) and not "
              "significant once fold dependence is accounted for, whereas CatBoost was significantly worse. Fully "
              "specified CNN-LSTM and Transformer baselines did not exceed LightGBM. Feature-group ablation identified "
              f"communication-domain features as the most influential group ({f3(comm_drop)} Macro-F1 drop), and the "
              f"SHAP explanations were stable across folds (mean Spearman ρ = {f3(stab['mean'])}) and faithful to the "
              "model under deletion and insertion tests on both datasets. A data-complexity analysis showed, however, "
              "that the synthetic task is much easier than real telemetry, partly because of class-dependent clipping "
              "artefacts of its generator, so the synthetic results describe a controlled benchmark rather than "
              "expected field performance.")
    D.replace("Critically, independent external validation on the public MSU/ORNL",
              "On the real MSU/ORNL Power System Attack Dataset, the retrained pipeline reached a 37-class Macro-F1 of "
              f"{f3(h['msuA'])} under random row-level splitting (LightGBM), but at most {f3(h['msuB'])} for any model when whole recorded events were "
              "held out, because neighbouring rows of the same event are near-duplicates; the event-disjoint values are "
              "the appropriate estimate for unseen events. Applied without retraining, the synthetic-trained model "
              f"performed at chance level (ROC-AUC {f3(h['auc_lo'])}-{f3(h['auc_hi'])}), so the study supports "
              "methodology-level external validity but not model transferability. SHAP attribution and ablation on "
              "MSU/ORNL showed that its models rely almost entirely on PMU voltage and current measurements, with "
              f"cyber-side log signals receiving {pct(cyber)} of the attribution. Kolmogorov-Smirnov and PSI drift "
              "monitors detected all simulated shifts of Table 7, with per-window false-alarm rates of "
              f"{pct(z['ks_detection_rate'])} and {pct(z['psi_detection_rate'])}, and label-free re-centring and shadow "
              "retraining recovered most of the loss, although single-sensor offsets of 0.1 standard deviations were "
              f"detected in only {pct(S1['0.1']['ks_detection_rate_mean'])} of windows. Single-sample inference took "
              f"{lat['mean_ms']:.2f} ms on average ({lat['p99_ms']:.2f} ms p99). Offline, the policy layer selected the "
              f"intended action for {pct(P['response_accuracy'])} of samples with a false-block rate of "
              f"{pct(P['false_block_rate'], 2)}, insensitive to τ between 0.5 and 0.9 because the model is well "
              f"calibrated (ECE {f3(T['calibration']['ece'])}); it was not validated in closed loop. Event-level features "
              "for real telemetry, closed-loop testing on a hardware-in-the-loop testbed, and validation on operational "
              "utility data are the next steps.")


def ablation_drop(n, group):
    return n["abl"][f"without_{group}"]["macro_f1_drop_vs_baseline"]


def latency_numbers(n):
    L = n["lat"]
    s = L["single_sample_latency_ms"]
    bt = {int(k): v for k, v in L["batch_throughput"].items()}
    return {"mean_ms": s["mean"], "median_ms": s["median"], "p95_ms": s["p95"], "p99_ms": s["p99"],
            "tp1": bt[1]["throughput_samples_per_sec"], "tp512": bt[512]["throughput_samples_per_sec"],
            "size_mb": L["model_artifact_kb"] / 1024, "hw": L["hardware"]}


def sync_rest(D: Doc, n, changes):
    from docx_edit import set_cell
    # Table 4 / Fig. 10 -- ROC AUC
    auc = n["auc"]
    names = {"dos": "DoS", "tamper": "Tamper", "overload": "Overload", "normal": "Normal", "fdi": "FDI"}
    before = len(changes)
    sync_cells(D, "Table 4. AUC Performance", {(names[k], 1): f"{v:.3f}" for k, v in auc.items()}, changes)
    lo, hi = min(auc.values()), max(auc.values())
    rng = f"{lo:.3f}-{hi:.3f}"
    swap(D, "All five classes achieve AUC above 0.98", [("(0.984-0.997)", f"({rng})")])
    swap(D, "Fig. 10. ROC Curves", [("0.984-0.997", rng)])
    if len(changes) > before:
        D.replace_image("Fig. 10. ROC Curves", ROOT / "artifacts" / "visualizations" / "roc_auc_curve.png")
    # Table 6 / Fig. 13 -- feature-group ablation
    A = n["abl"]
    before = len(changes)
    cells = {("All features (baseline)", 2): pm4(A["all_features"]["macro_f1_mean"], A["all_features"]["macro_f1_std"])}
    for g, lab in (("communication", "Without communication group"), ("electrical", "Without electrical group"),
                   ("equipment_health", "Without equipment-health"), ("power_quality", "Without power-quality group")):
        r = A[f"without_{g}"]
        cells[(lab, 2)] = pm4(r["macro_f1_mean"], r["macro_f1_std"])
        cells[(lab, 3)] = f"-{r['macro_f1_drop_vs_baseline']:.4f}"
    sync_cells(D, "Table 6. Feature-Group Ablation Results", cells, changes)
    old = {"communication": "0.192", "electrical": "0.135", "equipment_health": "0.082", "power_quality": "0.049"}
    new = {g: f"{ablation_drop(n, g):.3f}" for g in old}
    swap(D, "Communication-domain features (only 2 of 18 total features)",
         [("drop (0.192)", f"drop ({new['communication']})"), ("second (0.135 drop)", f"second ({new['electrical']} drop)"),
          ("equipment-health (0.082)", f"equipment-health ({new['equipment_health']})"),
          ("power-quality indicators (0.049)", f"power-quality indicators ({new['power_quality']})")])
    swap(D, "These results bear directly on the performance gap of Section 4.6.",
         [("compared with 0.192 for", f"compared with {new['communication']} for")])
    if len(changes) > before:
        D.replace_image("Fig. 13. Macro-F1 Drop", ROOT / "artifacts" / "visualizations" / "ablation_study.png")
    # Table 7 / Fig. 14 -- robustness
    rob = n["rob"]
    lv = lambda kind, l: next(x for x in rob[kind] if abs(x["level"] - l) < 1e-9)
    before = len(changes)
    cells = {}
    for kind, lab, lvls in (("noise", "Gaussian noise", (0.0, 0.2)), ("missing", "Missing data", (0.0, 0.3)),
                            ("shift", "Distribution shift", (0.0, 0.05, 0.2))):
        for l in lvls:
            key = "0% (baseline)" if l == 0 else f"{100 * l:.0f}%"
            x = lv(kind, l)
            cells[((lab, key), 2)] = f"{100 * x['accuracy']:.1f}%"
            cells[((lab, key), 3)] = f"{x['macro_f1']:.3f}"
    sync_cells(D, "Table 7. Robustness Under Telemetry Corruption", cells, changes)
    if len(changes) > before:
        D.replace_image("Fig. 14. Macro-F1 Degradation", ROOT / "artifacts" / "visualizations" / "robustness_degradation.png")
    # Table 8 / Fig. 15 -- MSU/ORNL random-split results
    before = len(changes)
    cells = {}
    for v, lab in (("multiclass", "37-class"), ("triple", "3-class"), ("binary", "Binary")):
        S = n["msu"][v]["summary"]
        best = max(S, key=lambda m: S[m]["macro_f1_mean"])
        for m in MODEL:
            cells[((lab, MODEL[m]), 2)] = pm4(S[m]["macro_f1_mean"], S[m]["macro_f1_std"])
            if v == "multiclass":
                cells[((lab, MODEL[m]), 3)] = f"{100 * S[m]['accuracy_mean']:.1f}% ± {100 * S[m]['accuracy_std']:.1f}%"
        t = D.table_after_caption("Table 8. External Validation Results")
        tagged = [r.cells[1].text for r in t.rows if norm(r.cells[0].text).startswith(lab) and "(best)" in r.cells[1].text]
        assert tagged and tagged[0].startswith(MODEL[best]), (v, best, tagged)
    sync_cells(D, "Table 8. External Validation Results", cells, changes)
    if len(changes) > before:
        D.replace_image("Fig. 15. External Validation", ROOT / "artifacts" / "visualizations" / "msu_external_validation_multiclass.png")
    # Table 8b -- MSU/ORNL SHAP ranking
    rk = n["msu_shap"]["ranking"][:10]
    tot = sum(r["mean_abs_shap"] for r in n["msu_shap"]["ranking"])
    cells = {}
    for i, r in enumerate(rk, 1):
        cells[(str(i), 1)] = r["feature"]
        cells[(str(i), 3)] = f"{r['mean_abs_shap']:.3f}"
        cells[(str(i), 4)] = f"{100 * r['share']:.1f}%"
    sync_cells(D, "Table 8b. Top-10 Features", cells, changes)
    # Table 8c -- MSU/ORNL ablation
    MA = n["msu_abl"]
    cells = {("All 128 features", 2): pm4(MA["baseline"]["mean"], MA["baseline"]["std"])}
    for g, lab in (("PMU current phasors", "Without current phasors"), ("PMU voltage phasors", "Without voltage phasors"),
                   ("Cyber-side (logs + relay status)", "Without cyber-side columns"),
                   ("PMU frequency / frequency delta", "Without frequency"),
                   ("PMU apparent impedance", "Without apparent impedance")):
        r = MA["ablations"][g]
        cells[(lab, 2)] = pm4(r["mean"], r["std"])
        cells[(lab, 3)] = f"{r['drop']:+.4f}"
    sync_cells(D, "Table 8c. Feature-Group Ablation on the MSU/ORNL", cells, changes)
    # Section 4.8 / Table 9 -- latency, re-measured on the final machine
    lat = latency_numbers(n)
    e = n["env"]
    hw = f"{e['cpu_model']}, {e['physical_cpus']} physical / {e['logical_cpus']} logical cores, {e['ram_gb']:.1f} GB RAM, Linux"
    t9 = D.table_after_caption("Table 9. Deployment Characteristics")
    vals = {"Model artifact size": f"{lat['size_mb']:.2f} MB", "Measuring hardware": hw,
            "Single-sample latency (mean)": f"{lat['mean_ms']:.2f} ms", "Single-sample latency (p99)": f"{lat['p99_ms']:.2f} ms",
            "Throughput @ batch = 1": f"{lat['tp1']:,.0f} samples/sec", "Throughput @ batch = 512": f"{lat['tp512']:,.0f} samples/sec"}
    for row in t9.rows:
        for k, v in vals.items():
            if norm(row.cells[0].text).startswith(k) and norm(row.cells[1].text) != norm(v):
                changes.append(("Table 9", k, 1, row.cells[1].text.strip(), v))
                set_cell(row.cells[1], v, D.mark)
    D.replace("The proposed framework is implemented as a FastAPI-based REST service.",
              "The proposed framework is implemented as a FastAPI-based REST service. Real inference latency was "
              "benchmarked end-to-end (StandardScaler transform + LightGBM predict) on real held-out test rows, on the "
              f"machine of Section 3.9 ({hw}; Table 9). Single-sample latency averaged {lat['mean_ms']:.2f} ms (median "
              f"{lat['median_ms']:.2f} ms, p95 {lat['p95_ms']:.2f} ms, p99 {lat['p99_ms']:.2f} ms), and batch throughput "
              f"scaled from {lat['tp1']:,.0f} samples/second at batch size 1 to {lat['tp512']:,.0f} samples/second at "
              "batch size 512, indicating that the pipeline is well within real-time requirements for smart-grid "
              "telemetry rates and can additionally support high-throughput batch scenarios such as periodic bulk "
              "re-scoring. The additional latency contributed by the policy layer is measured separately in "
              "Section 4.9.")
    pl = n["pol"]["latency_us"]
    D.edit("Response latency. Because the policy lookup",
           ("The mean per-call latency was 1.21 μs (99th percentile 2.13 μs) on the machine described in Section 3.9. "
            "Although this machine differs from the one used for Table 9, the policy stage is roughly three orders of "
            "magnitude faster than the 1.70 ms mean ML-inference latency",
            f"The mean per-call latency was {pl['mean']:.2f} μs (99th percentile {pl['p99']:.2f} μs) on the machine "
            "described in Section 3.9, the same machine as Table 9; the policy stage is therefore roughly three orders "
            f"of magnitude faster than the {lat['mean_ms']:.2f} ms mean ML-inference latency"))


def repro_sentence(changes):
    if not changes:
        return "reproduced every previously reported value."
    tabs = sorted({c[0] for c in changes})
    return ("reproduced every previously reported value except the following, which are updated and highlighted: "
            + ", ".join(tabs) + " (details in the response letter).")


COLS = {"Table 2": {1: "Macro-F1", 2: "accuracy", 3: "precision", 4: "recall"},
        "Table 3": {1: "W", 2: "p", 3: "significant"},
        "Table 3c": {1: "mean ΔMacro-F1", 2: "folds won", 3: "Wilcoxon W (p)", 4: "corrected t (p)"},
        "Table 8": {2: "Macro-F1", 3: "accuracy"}}


def changes_bullets(n):
    ch = json.loads(CHANGES_FILE.read_text()) if CHANGES_FILE.exists() else []
    by = {}
    for c in ch:
        if c[0] == "Table 9":
            continue
        if len(c) == 5:
            tab, label, col, old, new = c
            by.setdefault(tab, []).append(f"{label}, {COLS.get(tab, {}).get(col, 'column ' + str(col))}: {old} → {new}")
        else:
            by.setdefault(c[0], []).append(f"'{c[1]}' → '{c[2]}'")
    names = {"Table 2": "Table 2 (and Fig. 9)", "Table 8": "Table 8 (and Fig. 15)"}
    return [f"{names.get(t, t)}: " + "; ".join(v) + "." for t, v in by.items()]


def changes_sentence(n):
    ch = json.loads(CHANGES_FILE.read_text()) if CHANGES_FILE.exists() else []
    if not ch:
        return "none."
    by = {}
    for c in ch:
        tab, old, new = c[0], c[-2], c[-1]
        label = c[1] if len(c) == 5 else "text"
        if tab == "Table 9":  # latency: re-measured on new hardware, described separately
            continue
        by.setdefault(tab, []).append(f"{label}: {old} → {new}")
    return " ".join(f"{t}: " + "; ".join(v) + "." for t, v in by.items())


if __name__ == "__main__":
    main()
