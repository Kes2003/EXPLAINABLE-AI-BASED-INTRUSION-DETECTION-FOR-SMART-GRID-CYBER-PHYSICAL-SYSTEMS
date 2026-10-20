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
    return json.loads(Path(path).read_text())


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
        "cv": J(ROOT / "artifacts" / "cv_results.json"),
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
            "a substantially stronger and different claim. Because the reviewers asked whether such transfer holds, "
            "we also tested it directly (Section 4.6.1): applied to MSU/ORNL without retraining, the "
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
            "Action-selection accuracy (termed response accuracy in the previous version) is the fraction of samples "
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
                "The monitoring and shadow-retraining design proposed in the previous version was implemented and "
                "evaluated with the deployed model, test split, and shift operator of Table 7 (every feature x is "
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
    D.replace("Software environment:",
              "Software environment: every result reported in this version -- including all values carried over from "
              "the previous versions -- was regenerated in a single computational environment: "
              f"{env_sentence(n)}, on Linux ({e['platform']}) with a {e['physical_cpus']}-core "
              f"{e['cpu_model']} CPU and {e['ram_gb']:.0f} GB RAM. The complete package list is provided as a lock "
              "file (environment/requirements-lock.txt) in the code repository, and a single script "
              "(reproduce_all.sh) regenerates every table and figure. Earlier versions of this manuscript reported "
              "results produced in two different environments (Python 3.12 with scikit-learn 1.8, XGBoost 3.4, SHAP "
              "0.52 on a Windows machine, and Python 3.11 with scikit-learn 1.9, XGBoost 3.2, SHAP 0.51 and SciPy 1.17 "
              "on a Linux machine). Re-running the whole pipeline in the final environment " + repro + " The latency "
              "benchmark of Section 4.8 depends on hardware and was therefore re-measured on the machine above; the "
              "previous measurement on a Windows workstation is no longer reported.")


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
            "selection. The baselines were re-implemented for this version with a fully documented configuration "
            "(Table 3d) and re-run; Table 3b reports the new results, which replace those of the previous version."))
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
                "performance on unseen events, and they replace the Table 8 values wherever the previous version used "
                "the latter as an estimate of real-world performance. They also show that, in this dataset, a single "
                "row of PMU and log features carries little information about which of the 37 scenarios is "
                "executing; event-level features computed over windows of consecutive rows, which this study does "
                "not use, are the obvious next step. The synthetic dataset is not affected by this issue, because its "
                "rows are generated independently (Section 3.2.1).")
    return p
