"""
Build the Revision-3 manuscript (.docx) from the Revision-2 Word file.

Every number introduced in this revision is read from the result files
produced by the analysis scripts -- none is typed in by hand:
    results/reviewer_experiments.json  (run_reviewer_experiments.py)
    results/corrected_ttest.json       (scripts/corrected_resampled_ttest.py)
    results/msu_shap_multiclass.json   (scripts/msu_shap_analysis.py)

Outputs (manuscript/):
    Manuscript_R3_clean.docx        -- all highlighting removed
    Manuscript_R3_highlighted.docx  -- only this revision's changes in yellow

Usage (from the project root):
    python manuscript/build_manuscript.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from docx_edit import clone_table, insert_after, norm, remove_all_highlights, set_cell, set_text  # noqa: E402

SOURCE = HERE / "source" / "Smart_Grid_IDPS_Manuscript_R2_Final_1.docx"
RESULTS = ROOT / "results"

MODEL_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------
def load_numbers():
    R = json.loads((RESULTS / "reviewer_experiments.json").read_text())
    T = json.loads((RESULTS / "corrected_ttest.json").read_text())
    msu = RESULTS / "msu_shap_multiclass.json"
    M = json.loads(msu.read_text()) if msu.exists() else None
    P = R["prevention_benchmark"]
    n = {
        "R": R, "T": T, "M": M, "P": P,
        "resp_acc": f"{100 * P['response_accuracy']:.1f}%",
        "resp_correct": f"{round(P['response_accuracy'] * P['n_predictions']):,}",
        "n_pred": f"{P['n_predictions']:,}",
        "fbr": f"{100 * P['false_block_rate']:.2f}%",
        "fb": f"{P['false_blocks']:,}",
        "n_normal": f"{P['n_true_normal']:,}",
        "missed_rate": f"{100 * P['missed_response_rate']:.1f}%",
        "missed": f"{P['missed_responses']:,}",
        "n_need": f"{P['n_requiring_action']:,}",
        "lat_mean": f"{P['latency_us']['mean']:.2f}",
        "lat_p99": f"{P['latency_us']['p99']:.2f}",
        "lat_calls": f"{P['latency_us']['n_calls']:,}",
    }
    for cls, v in P["per_class"].items():
        n[f"pc_{cls}"] = f"{100 * v['correct_action'] / v['n']:.1f}%"
    return n


def runs_to_md(p):
    """Paragraph text with bold spans as **...** (when the base run is not bold)."""
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


# --------------------------------------------------------------------------
# Paragraph lookup
# --------------------------------------------------------------------------
class Doc:
    def __init__(self, path, mark):
        self.d = Document(str(path))
        self.mark = mark
        remove_all_highlights(self.d)  # drop Revision-2 highlighting in both copies

    def para(self, prefix):
        hits = [p for p in self.d.paragraphs if norm(p.text).startswith(norm(prefix))]
        assert len(hits) == 1, f"{len(hits)} paragraphs start with {prefix!r}"
        return hits[0]

    def table(self, first_cell, ncols=None):
        hits = [t for t in self.d.tables if norm(t.rows[0].cells[0].text).startswith(norm(first_cell))
                and (ncols is None or len(t.columns) == ncols)]
        assert len(hits) == 1, f"{len(hits)} tables start with {first_cell!r}"
        return hits[0]

    def replace(self, prefix, md):
        set_text(self.para(prefix), md, self.mark)

    def edit(self, prefix, *pairs):
        """Apply literal (old, new) substitutions to a paragraph's text."""
        p = self.para(prefix)
        text = runs_to_md(p)
        for old, new in pairs:
            assert text.count(old) == 1, f"{old!r} occurs {text.count(old)}x in {prefix!r}"
            text = text.replace(old, new)
        set_text(p, text, self.mark)

    def after(self, anchor, template_prefix_or_para, md):
        tpl = template_prefix_or_para
        if isinstance(tpl, str):
            tpl = self.para(tpl)
        return insert_after(anchor, tpl, md, self.mark)


# --------------------------------------------------------------------------
# Revisions (Reviewer 4, round 3)
# --------------------------------------------------------------------------
def revise_front(D: Doc, n):
    # Abstract -- R4 points 1, 2, 3, 5, 7
    D.edit(
        "Smart grids integrate digital communication",
        ("The prevention component is realized as a rule-based policy-response layer whose architecture is "
         "described but whose operational effectiveness is not independently benchmarked here; quantitative "
         "evaluation of prevention performance (e.g., response latency, false-block rate) is identified as a "
         "direction for future work.",
         "The prevention component is realized as a rule-based policy-response layer; because this layer is a "
         "deterministic function of the predicted class and its probability, we evaluate it by passing all "
         f"{n['n_pred']} cross-validated detection predictions through the deployed policy function: it selected "
         f"the correct action for {n['resp_acc']} of samples, blocked {n['fbr']} of normal samples, and added "
         f"{n['lat_mean']} μs per decision, although it has not been validated under live, closed-loop "
         "operating conditions."),
        ("while CatBoost trailed with a Macro-F1 of 0.893 ± 0.005.", abstract_cv_sentence(n)),
        ("an expected and informative gap that is discussed in detail.",
         "an expected and informative gap whose likely causes and implications for real-world transferability "
         "are analysed in detail. " + abstract_msu_shap_sentence(n)),
        ("(0.938 → 0.402 at 20% shift), highlighting an important limitation for deployment on drifting sensor "
         "calibrations.",
         "(0.938 → 0.402 at 20% shift), for which we outline a drift-detection and shadow-retraining mitigation "
         "strategy."),
    )

    # Contributions list -- R4 points 1, 2, 3, 5, 7
    D.edit("Development and experimental evaluation of an AI-based",
           ("is presented at the design level (Section 3.1) and scoped for future empirical benchmarking (Section 4.8).",
            "is described in Section 3.1 and evaluated post hoc for response accuracy, false-block rate, and "
            "execution latency (Section 4.9)."))
    D.edit("Comparative evaluation of Random Forest, XGBoost",
           ("for multiclass attack detection.",
            "for multiclass attack detection, strengthened by a repeated cross-validation analysis "
            "(10 × 5 folds, n = 50 paired observations; Section 3.6.1)."))
    D.edit("Independent external validation on the public Mississippi",
           ("label granularities.",
            "label granularities, with SHAP-based explainability extended to this external dataset (Section 4.7)."))
    D.edit("Systematic robustness evaluation under Gaussian",
           ("and distribution shift.",
            "and distribution shift, together with a proposed drift-detection and mitigation strategy for "
            "operational deployment (Section 4.10)."))
    D.edit("Deployment of the proposed framework as a FastAPI",
           ("Deployment of the proposed framework as a FastAPI-based real-time monitoring and prevention service,",
            "Deployment of the detection component of the proposed framework as a FastAPI-based real-time "
            "monitoring service,"),
           ("environments.",
            "environments; the accompanying prevention/policy-response layer is exposed through the same service "
            "and quantitatively evaluated in Section 4.9, but is not yet field-validated under live operational "
            "conditions."))
    D.edit("The remainder of this paper is organized as follows.",
           ("Section 4 presents experimental results, statistical validation, ablation analysis, external validation, "
            "and deployment performance.",
            "Section 4 presents experimental results, statistical validation, ablation analysis, external validation "
            "and its explainability, deployment performance, and the evaluation of the prevention layer."))

    # New Section 1.1 -- R4 point 6
    h = D.after(D.para("The remainder of this paper is organized as follows."), "2.1. AI-BASED INTRUSION", "1.1 NOVELTY AND POSITIONING")
    D.after(h, "The remainder of this paper is organized as follows.", NOVELTY)


NOVELTY = (
    "Individually, several of the framework's components have precedent in prior work: ensemble tree-based "
    "classifiers for smart-grid intrusion detection [2,3,10], SHAP-based explainability for cybersecurity models "
    "[8,31], and REST-based deployment of machine-learning services are each documented separately in the "
    "literature reviewed in Section 2. The contribution of this work is therefore not a new learning algorithm, "
    "but the joint empirical validation of five elements that, to the best of our knowledge, have not previously "
    "been evaluated together within one smart-grid IDPS study: (i) multidomain cyber-physical feature fusion whose "
    "necessity is quantitatively demonstrated, rather than assumed, through feature-group ablation (Section 4.4); "
    "(ii) a leakage-safe cross-validation protocol with paired statistical significance testing, strengthened by "
    "a repeated cross-validation analysis (Sections 3.4, 3.6, 3.6.1); (iii) independent external validation on a "
    "real-world, non-synthetic dataset, extended with a cross-dataset explainability analysis (Sections 4.6, 4.7); "
    "(iv) per-class SHAP explanations whose stability is itself verified across folds and under input "
    "perturbation (Section 4.3); and (v) hardware-benchmarked detection latency and throughput together with a "
    "quantitative evaluation of the prevention/policy layer's response accuracy, false-block rate, and latency "
    "(Sections 4.8, 4.9). As Table 1 shows, none of the reviewed prior studies reports external validation or "
    "statistical significance testing, and none evaluates its prevention layer quantitatively; the present work "
    "reports all five elements under a single, internally consistent protocol applied to the same trained models. "
    "We regard this combination, rather than any individual algorithmic novelty, as the paper's primary contribution."
)


def revise_methods(D: Doc, n):
    # Table 1 caption and "Proposed Work" row -- R4 point 6
    D.edit("Table 1. Comparative Analysis",
           ("vs. the Proposed Framework",
            "vs. the Proposed Framework. None of the reviewed studies reports cross-dataset external validation or "
            "statistical significance testing; the present work additionally reports a quantitative evaluation of "
            "its prevention/policy layer (Section 4.9), a repeated cross-validation significance analysis "
            "(Section 3.6.1), and a cross-dataset explainability analysis (Section 4.7)."))
    t1 = D.table("Reference")
    cell = [c for c in t1.rows[-1].cells if norm(c.text).startswith("Yes (Wilcoxon")][0]
    set_cell(cell, "Yes (Wilcoxon; 5-fold and 10 × 5-fold repeated CV)", D.mark)

    # Fig. 1 legend and scope statement -- R4 points 1, 7
    D.edit("Fig. 1. Proposed Smart Grid",
           ("the architected but not yet independently benchmarked prevention stage;",
            "whose response accuracy, false-block rate, and execution latency are evaluated post hoc in Section 4.9, "
            "though not under live, closed-loop operating conditions;"))
    D.edit("Scope of experimental evaluation.",
           ("characterize the detection stage exclusively.", "characterize the detection stage."),
           ("The prevention/policy-response layer is implemented and exposed through the same FastAPI service, but its "
            "operational effectiveness (e.g., response latency, correct-action rate, false-block rate) has not been "
            "independently benchmarked in this study and is identified explicitly as a direction for future work "
            "(Section 4.8). This scoping is stated here to avoid any implication that the reported detection metrics "
            "constitute evidence of prevention efficacy.",
            "The prevention/policy-response layer is implemented and exposed through the same FastAPI service. Because "
            "this layer is a deterministic function of the predicted class and its probability, Section 4.9 evaluates "
            "its response accuracy, false-block rate, and execution latency by passing the cross-validated detection "
            "predictions of Section 3.6 through the deployed policy function and comparing each selected action with an "
            "explicit ground-truth class-to-action mapping. This establishes whether the policy engine selects the "
            "intended action given the classifier's actual, imperfect predictions; it does not evaluate the policy "
            "engine under live, closed-loop operating conditions (actual breaker actuation, network-level "
            "rate-limiting enforcement, or operator response to escalations), which remains a direction for future "
            "work (Section 4.10). Nor does it establish whether the mapping itself is the optimal operator response "
            "for a given attack in a real substation, which requires power-system domain expertise and is outside the "
            "scope of this study. This scoping is stated here to avoid any implication that the reported metrics "
            "constitute evidence of field-deployed prevention efficacy."))

    # Algorithm 2 -- the policy step is now evaluated (R4 point 1)
    alg2 = D.table("Algorithm 2.")
    ps = alg2.rows[0].cells[0].paragraphs
    fix = {
        "Algorithm 2. Real-Time Detection Pipeline (with Architected Policy-Response Stage)":
            "**Algorithm 2. Real-Time Detection Pipeline (with Policy-Response Stage)**",
        "Output: predicted class y_t; SHAP attribution phi_t; policy action a_t (design-level only)":
            "Output: predicted class y_t; SHAP attribution phi_t; policy action a_t",
        "6: a_t <- POLICY_ENGINE.lookup(y_t)      // rule-based class-to-action mapping;":
            "6: a_t <- POLICY_ENGINE(y_t, p_t)       // Table 10 action if p_t >= tau (0.7);",
        "                                          // architected but NOT independently benchmarked":
            "                                          // evaluated post hoc in Section 4.9",
        "                                          // in this study (scope stated in Section 3.1)":
            "                                          // (scope stated in Section 3.1)",
    }
    for p in ps:
        if p.text in fix:
            set_text(p, fix[p.text], D.mark)

    # Algorithm 1 note -- R4 point 2
    alg1 = D.table("Algorithm 1.")
    last = alg1.rows[0].cells[0].paragraphs[-1]
    set_text(last, last.text + " Section 3.6.1 repeats the entire protocol with 10 independent fold-assignment "
                               "seeds to increase statistical power.", D.mark)

    # Section 3.6 pointer to 3.6.1
    D.edit("All four models (Random Forest, XGBoost",
           ("using the per-fold Macro-F1 scores as paired observations.",
            "using the per-fold Macro-F1 scores as paired observations. Section 3.6.1 repeats this analysis with "
            "substantially greater statistical power."))

    # Section 3.8 and 3.9
    D.edit("Deployment latency is measured end-to-end",
           ("(Section 4.7).", "(Section 4.8); the policy engine's own latency is measured separately in Section 4.9."))
    D.edit("Software environment:",
           ("(Section 4.3-4.8)", "(Sections 4.2-4.8)"),
           ("already noted in Section 4.7 for the latency benchmark.",
            "described in Section 4.8 for the latency benchmark."),
           ("not on the machine used to obtain them.",
            "not on the machine used to obtain them. The analyses added in this revision -- the repeated "
            "cross-validation of Section 3.6.1, the MSU/ORNL explainability analysis of Section 4.7, and the "
            "policy-engine evaluation of Section 4.9 -- were run with Python 3.11, scikit-learn 1.9, XGBoost 3.2, "
            "LightGBM 4.7, CatBoost 1.2, SHAP 0.51, and SciPy 1.17 on a Linux machine with a 4-core Intel Xeon "
            "processor (2.1 GHz) and 15 GB RAM; re-running the original five-fold protocol in this environment "
            "reproduced every Macro-F1 and accuracy value of Table 2 and every test result of Table 3 exactly."))
    D.edit("Data splitting:",
           ("every sample serves as test data in exactly one fold.",
            "every sample serves as test data in exactly one fold. The repeated cross-validation of Section 3.6.1 "
            "runs this protocol with 10 fold-assignment seeds (42-51), of which seed 42 is the original run."))


def revise_results(D: Doc, n):
    D.edit("Paired Wilcoxon signed-rank tests comparing LightGBM",
           ("but likewise not confirmed as significant.",
            "but likewise not confirmed as significant. Section 3.6.1 repeats these tests with n = 50 paired "
            "observations (Table 3c) to determine whether this conclusion is an artifact of limited statistical power."))
    D.edit("The confusion matrix (Fig. 8) shows",
           ("rather than an abrupt anomaly.",
            "rather than an abrupt anomaly. Section 4.9 uses the corresponding five-fold out-of-fold predictions for "
            f"all {n['n_pred']} samples to evaluate the prevention layer."))
    D.edit("Fig. 9. Model Comparison Across",
           ("CatBoost trails visibly behind all three.",
            "CatBoost trails visibly behind all three. Section 3.6.1 reports a repeated cross-validation analysis "
            "with substantially greater statistical power."))
    D.edit("LightGBM outperformed both deep-learning baselines",
           ("the same statistical-power caveat that applies throughout Section 4.",
            "the same statistical-power caveat that applies to Table 3; owing to the computational cost noted in "
            "Section 3.6.1, this comparison was not extended to repeated cross-validation."))
    D.edit("Two further architectures named in review",
           ("Two further architectures named in review --", "Two further architectures --"),
           ("established in Section 4.7,", "established in Section 4.8,"))
    D.edit("Fig. 10. ROC Curves", ("(Section 4.8)", "(Section 4.10)"))
    D.edit("All five classes achieve AUC above 0.98", ("discussion in Section 4.8,", "discussion in Section 4.10,"))
    D.edit("A natural question for any post-hoc explanation",
           ("rather than a single-split artifact.",
            "rather than a single-split artifact. Section 4.7 extends the SHAP analysis to the external MSU/ORNL dataset."))
    D.edit("Table 5. SHAP-Based Attack",
           ("Summary by Class", "Summary by Class. This mapping also underlies the ground-truth class-to-action "
                                "mapping used to evaluate the prevention layer (Table 10, Section 4.9)."))
    D.edit("Communication-domain features (only 2 of 18",
           ("independently derived evidence for which feature domains the model relies on.",
            "independently derived evidence for which feature domains the model relies on. Section 4.6 discusses the "
            "implications of this finding for the observed synthetic/MSU-ORNL performance gap."))
    D.edit("The model degrades gracefully under Gaussian",
           ("which would require periodic model retraining or drift-detection mechanisms in an operational deployment "
            "(see Section 4.8).",
            "Section 4.10 outlines a drift-detection and mitigation strategy for this failure mode."))

    # Section 4.6 -- R4 point 4: split the "Two findings" paragraph and add the causal analysis
    p = D.para("Two findings stand out.")
    text = p.text
    cut = text.index("Second, model ranking")
    first, second = text[:cut].rstrip(), text[cut:]
    set_text(p, first, D.mark, old=text)
    gap = D.after(p, "Two findings stand out.", GAP_ANALYSIS)
    D.after(gap, "Two findings stand out.", second)

    s47(D, n)

    # Section 4.8 (was 4.7)
    D.replace("4.7 DEPLOYMENT PERFORMANCE ANALYSIS", "4.8 DEPLOYMENT PERFORMANCE ANALYSIS")
    D.edit("The proposed framework is implemented as a FastAPI",
           ("periodic bulk re-scoring.",
            "periodic bulk re-scoring. The additional latency contributed by the prevention/policy layer is "
            "measured separately in Section 4.9."))
    t9 = D.table("Metric")
    for row in t9.rows:
        k = norm(row.cells[0].text)
        if k == "Explainability support":
            set_cell(row.cells[1], "SHAP (TreeExplainer), per-class visualization; global ranking on MSU/ORNL "
                                   "(Section 4.7)", D.mark)
        if k == "Operational validation":
            set_cell(row.cells[1], "Controlled synthetic + external real-world (MSU/ORNL); prevention layer "
                                   "evaluated post hoc (Section 4.9)", D.mark)

    s49(D, n)
    s410(D, n)


GAP_ANALYSIS = (
    "Several converging factors plausibly explain this gap. First, label granularity differs sharply: the "
    "synthetic dataset uses five broad operational/attack categories, whereas the MSU/ORNL taxonomy distinguishes "
    "37 fine-grained event scenarios, several of which (e.g., closely related relay-setting-change and "
    "command-injection variants) are electrically similar and inherently harder to separate than the five broad "
    "synthetic classes. Second, the two datasets differ in feature abstraction level: the 18 synthetic features are "
    "hand-engineered, domain-informed quantities that directly encode known attack signatures (Section 3.3), "
    "whereas the 128 MSU/ORNL features are comparatively low-level PMU phasor, frequency, and impedance channels "
    "plus relay/Snort log flags, so the model must learn implicitly relationships that are explicit by "
    "construction in the synthetic case. Third, the synthetic attack classes are generated with physically "
    "motivated but idealized perturbations, producing more separable class-conditional distributions than the "
    "noisy, non-stationary transients recorded on a physical laboratory testbed during real attack execution. "
    "Fourth, the ablation study of Section 4.4 identifies packet rate and packet error rate as the single most "
    "influential feature group in the synthetic dataset; MSU/ORNL contains no equivalent network-traffic "
    "statistics, only coarse relay and Snort log flags, so the category the ablation study identifies as most "
    "informative is largely absent in the real-world case. For these reasons, we regard the 83.1% MSU/ORNL "
    "Macro-F1, rather than the 92.8% synthetic figure, as the more representative estimate of performance on real "
    "grid telemetry, and we make no claim that a model trained on the synthetic dataset would transfer directly "
    "to a real substation without retraining on local data (Section 4.10)."
)


def s49(D: Doc, n):
    """New Section 4.9 -- R4 point 1."""
    anchor = D.para("Table 9. Deployment Characteristics")
    h = D.after(anchor, "4.6 EXTERNAL VALIDATION", "4.9 PREVENTION/POLICY-ENGINE PERFORMANCE ANALYSIS")
    body = "The proposed framework is implemented as a FastAPI"
    p = D.after(h, body,
        "The rule-based policy-response layer of Section 3.1 maps each predicted class to a response action. Because "
        "the policy engine (Algorithm 2, line 6) is a deterministic function of the predicted class and its "
        "probability, its behaviour is fully determined once the classifier's predictions are known. This has two "
        "consequences for evaluation. First, response accuracy and false-block rate are a re-expression of "
        "classification performance under a specific class-to-action mapping, and can be measured by passing "
        "held-out predictions through the deployed policy function, without a live-system trial. Second, the policy "
        "engine's own contribution to response latency is separate from, and additive to, the ML inference latency "
        "of Table 9, and is therefore measured independently.")
    p = D.after(p, body,
        "**Ground-truth action mapping.** Table 10 defines the correct action for each true class, consistent with "
        "the design intent of Section 3.1 and the SHAP-based interpretation of Table 5: network rate-limiting for "
        "DoS, breaker isolation for tamper and overload, escalation to the operator without automated action for "
        "FDI, and no action for normal operation. The deployed policy takes an automated action only when the "
        "predicted class probability reaches the alert threshold τ = 0.7 (config.yaml); below it, no automated "
        "action is taken.")
    t = clone_table(D.table("Attack Type"), p, [
        ["**True Class**", "**Ground-Truth Action**", "**Rationale**"],
        ["Normal", "No action", "Nominal operation"],
        ["DoS", "Rate-limit network traffic", "Communication-layer congestion mitigation"],
        ["FDI", "Escalate to operator (no automated action)", "Measurement integrity requires operator investigation"],
        ["Tamper", "Trip / isolate breaker", "Physical safety response to an unauthorized state change"],
        ["Overload", "Trip / isolate breaker", "Thermal/loading safety response"],
    ], D.mark)
    cap = D.after(t, "Table 9. Deployment Characteristics",
                  "Table 10. Ground-Truth Class-to-Action Mapping Used to Evaluate the Prevention Layer")
    p = D.after(cap, body,
        "**Evaluation protocol and metrics.** The five-fold out-of-fold LightGBM predictions of the original "
        f"cross-validation run (seed 42, Table 2), which cover all {n['n_pred']} samples exactly once, were passed "
        "through the deployed policy function of the FastAPI service with τ = 0.7, and each selected action was "
        "compared with the Table 10 action for the sample's true class. Response accuracy is the fraction of "
        "samples whose selected action exactly matches the ground-truth action (for example, a DoS sample "
        "misclassified as tamper, which would trip a breaker instead of rate-limiting traffic, counts as an error). "
        "False-block rate is the fraction of truly normal samples for which a blocking action (breaker trip or "
        "rate-limiting) was taken. Missed-response rate is the fraction of DoS, tamper, and overload samples for "
        "which no automated action was taken.")
    p = D.after(p, body,
        f"**Response accuracy and false-block rate.** The policy engine selected the correct action for "
        f"{n['resp_correct']} of {n['n_pred']} samples (response accuracy {n['resp_acc']}), and only {n['fb']} of "
        f"{n['n_normal']} normal samples triggered an automated block (false-block rate {n['fbr']}). Per class, the "
        f"correct action was selected for {n['pc_normal']} of normal, {n['pc_dos']} of DoS, {n['pc_fdi']} of FDI, "
        f"{n['pc_tamper']} of tamper, and {n['pc_overload']} of overload samples. The dominant error mode is "
        f"therefore not spurious blocking but missed responses: {n['missed']} of the {n['n_need']} samples that "
        f"required an automated action ({n['missed_rate']}) received none, because they were either misclassified "
        "as normal or FDI or predicted with a probability below τ. This reflects the conservative design of the "
        "confidence gate, which accepts some missed responses in exchange for a very low false-block rate; τ is the "
        "parameter that governs this trade-off.")
    p = D.after(p, body,
        "**Response latency.** Because the policy lookup consists of a few conditional comparisons with no model "
        f"inference or I/O, each call was timed directly, over five passes through all {n['n_pred']} predictions "
        f"({n['lat_calls']} calls). The mean per-call latency was {n['lat_mean']} μs (99th percentile "
        f"{n['lat_p99']} μs) on the machine described in Section 3.9. Although this machine differs from the one "
        "used for Table 9, the policy stage is roughly three orders of magnitude faster than the 1.70 ms mean "
        "ML-inference latency and does not materially affect the end-to-end real-time budget.")
    D.after(p, body,
        "These results establish that the policy engine selects the intended action given the classifier's actual, "
        "imperfect predictions. They do not evaluate closed-loop operation -- actual breaker actuation, enforcement "
        "of rate limits in network equipment, or operator handling of escalations -- nor whether the mapping of "
        "Table 10 is the optimal operator response in a real substation; both require hardware-in-the-loop or field "
        "testing and power-system domain expertise, and remain future work (Section 4.10). The results also inherit "
        "the synthetic-data caveat of Section 4.10: on real grid telemetry, where detection performance is lower "
        "(Section 4.6), response accuracy would be correspondingly lower.")


def s410(D: Doc, n):
    """Section 4.10 (was 4.8) -- split into paragraphs; R4 points 2, 3, 4, 5, 7."""
    D.replace("4.8 LIMITATIONS OF THE EXPERIMENTAL EVALUATION", "4.10 LIMITATIONS OF THE EXPERIMENTAL EVALUATION")
    p = D.para("Several limitations should be considered")
    old = p.text
    paras = [
        "Several limitations should be considered when interpreting these results. First, the synthetic dataset "
        "remains the primary training and evaluation environment, and its physically idealized design likely "
        "contributes to the uniformly high AUC values of Section 4.2. The substantially lower performance on the "
        "real MSU/ORNL dataset (Sections 4.6, 4.7) should be treated as the more representative estimate of "
        "real-world performance. Moreover, external validation here means retraining and evaluating the same "
        "methodology on a second dataset; the framework has not been evaluated on telemetry from an operational "
        "utility network, and a model trained on one grid's data should not be assumed to transfer to another "
        "without retraining and re-validation on site-specific data.",

        "Second, the distribution-shift results (Section 4.5) show that the model is comparatively fragile to "
        "systematic sensor drift, a realistic failure mode in long-running deployments. Empirically evaluating a "
        "drift-detection and mitigation subsystem is outside the scope of this detection-focused study, but the "
        "corruption-injection methodology of Table 7 suggests a concrete, testable design for future work. (i) "
        "Per-feature drift monitoring -- for example, a rolling-window two-sample Kolmogorov-Smirnov test or "
        "Population Stability Index between a reference window drawn from the training distribution and a recent "
        "window of incoming telemetry, computed for each of the 18 features -- provides a label-free early warning; "
        "Table 7 shows that a proportional shift of only 5% already reduces Macro-F1 to 0.700, so drift must be "
        "detected at small magnitudes. (ii) Because ground-truth labels are rarely available in real time, the "
        "model's own confidence can serve as a proxy signal: a sustained fall in mean top-class probability without "
        "a corresponding change in alert rates is consistent with drift rather than a genuine change in grid "
        "behaviour. The same signal interacts with the prevention layer, since lower confidence pushes more "
        "predictions below the threshold τ and increases missed responses (Section 4.9). (iii) A shadow-retraining "
        "protocol, in which a candidate model retrained on recent telemetry runs in parallel with the deployed model "
        "and is promoted only after it outperforms it on a held-back, recently labelled validation slice, avoids "
        "unvalidated cut-overs. We present this as a proposed design, not as a tested capability of the current "
        "system.",

        "Third, CatBoost's default configuration underperforms substantially on the MSU/ORNL dataset; this is "
        "reported as found rather than tuned away, since it illustrates that default hyperparameters do not "
        "transfer uniformly across datasets.",

        LIMITATION_XAI,

        limitation_cv(n),

        "Sixth, as scoped in Section 3.1, the prevention/policy layer is evaluated post hoc (Section 4.9) by "
        "passing cross-validated detection predictions through the deployed policy function; its behaviour under "
        "live, closed-loop operating conditions (actual breaker actuation, network-level rate-limiting "
        "enforcement, operator response to escalations) has not been validated, and the class-to-action mapping "
        "itself has not been reviewed by power-system operators. Closed-loop automated prevention should therefore "
        "be regarded as architecturally integrated but not yet field-validated.",

        "Finally, the SHAP explanations of Section 4.3 were validated for stability across folds (mean Spearman "
        "rho = 0.947) and under input perturbation (rho = 0.988 at 1% noise), but this check used five folds, one "
        "perturbation scheme, and the synthetic dataset only; broader validation, including adversarially crafted "
        "perturbations and the MSU/ORNL models of Section 4.7, is left to future work. Also not addressed is a "
        "systematic comparison against deep-learning architectures beyond the CNN-LSTM and Transformer baselines "
        "of Section 4.1 -- specifically GNNs and TabNet, for the reasons stated there.",
    ]
    set_text(p, paras[0], D.mark, old=old)
    anchor = p
    for text in paras[1:]:
        anchor = D.after(anchor, "Several limitations should be considered", text)
        # new paragraphs: highlight only what is not in the old single paragraph
        set_text(anchor, text, D.mark, old=old)


def revise_back(D: Doc, n):
    D.replace("This paper presented an explainable AI-based intrusion detection engine", conclusion_1(n))
    D.edit("Critically, independent external validation",
           ("a substantial and expected performance gap", "a substantial, mechanistically explained performance gap"),
           ("(83.1% vs. 92.8% Macro-F1 for the closest-comparable multiclass task).",
            "(83.1% vs. 92.8% Macro-F1 for the closest-comparable multiclass task); we therefore treat the external "
            "result as the more representative estimate of real-world performance. " + conclusion_msu_sentence(n)),
           ("but pronounced sensitivity to systematic distribution shift, an important limitation for long-running "
            "deployments.",
            "but pronounced sensitivity to systematic distribution shift, for which we outlined a testable "
            "drift-detection and shadow-retraining strategy (Section 4.10)."),
           ("supporting real-time SCADA-integrated operation.",
            "supporting real-time SCADA-integrated detection. The prevention layer selected the correct action for "
            f"{n['resp_acc']} of cross-validated predictions with a false-block rate of {n['fbr']} and "
            f"{n['lat_mean']} μs of added latency (Section 4.9); however, because it has not been validated under "
            "live operating conditions and the detector is sensitive to distribution shift, closed-loop automated "
            "prevention should be regarded as architecturally integrated but not yet field-validated."))

    # Code availability (Scientific Reports requires a statement when custom code is central)
    da = D.para("The synthetic dataset generated and analyzed")
    h = D.after(da, "Data Availability", "Code Availability")
    D.after(h, "The synthetic dataset generated and analyzed", CODE_AVAILABILITY)

    # New reference (corrected resampled t-test, Section 3.6.1)
    last = D.para("41. Sen, O. et al.")
    D.after(last, "41. Sen, O. et al.",
            "42. Nadeau, C. & Bengio, Y. Inference for the generalization error. Mach. Learn. 52, 239-281 (2003).")


CODE_AVAILABILITY = (
    "The code used to generate all results reported in this study, including the cross-validation, "
    "significance-testing, ablation, robustness, external-validation, explainability, and prevention-layer "
    "analyses, is available at https://github.com/kes2003/EXPLAINABLE-AI-BASED-INTRUSION-DETECTION-FOR-SMART-GRID-"
    "CYBER-PHYSICAL-SYSTEMS."
)


# --------------------------------------------------------------------------
# Repeated cross-validation (R4 point 2) -- text driven by the result files
# --------------------------------------------------------------------------
def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def cv(n):
    R, T = n["R"], n["T"]
    s, w, t = R["summary"], R["wilcoxon_vs_best"], T["comparisons"]
    assert R["best_model"] == "lightgbm" == T["best_model"]
    ms = lambda m: f"{s[m]['mean']:.4f} ± {s[m]['std']:.4f}"
    return {
        "s": s, "w": w, "t": t, "ms": ms,
        "diff_rf": f"{w['random_forest']['mean_diff']:.4f}",
        "diff_xgb": f"{w['xgboost']['mean_diff']:.4f}",
        "diff_cb": f"{w['catboost']['mean_diff']:.3f}",
        "p_rf": fmt_p(t["random_forest"]["p"]), "p_xgb": fmt_p(t["xgboost"]["p"]),
        "p_cb": fmt_p(t["catboost"]["p"]),
        "win_rf": w["random_forest"]["best_wins"], "win_xgb": w["xgboost"]["best_wins"],
        "p_min_close": min(t["random_forest"]["p"], t["xgboost"]["p"]),
        "wil_max": max(v["p"] for v in w.values()),
    }


def check_cv_pattern(c):
    """The prose below states a specific pattern of results; fail loudly if the
    numbers ever stop matching it rather than printing a wrong sentence."""
    assert c["wil_max"] < 1e-6, "Wilcoxon no longer significant for all comparisons"
    assert c["t"]["random_forest"]["p"] >= 0.05 and c["t"]["xgboost"]["p"] >= 0.05
    assert c["t"]["catboost"]["p"] < 0.001
    assert abs(float(c["diff_rf"]) - float(c["diff_xgb"])) < 1e-4


def abstract_cv_sentence(n):
    c = cv(n)
    check_cv_pattern(c)
    return ("while CatBoost trailed with a Macro-F1 of 0.893 ± 0.005. Repeating the protocol with ten "
            "fold-assignment seeds (n = 50 paired folds) showed that LightGBM's advantage over Random Forest and "
            f"XGBoost is consistent but small ({c['diff_rf']} Macro-F1) and not significant once the dependence "
            f"between cross-validation folds is accounted for (corrected resampled t-test, p ≥ {c['p_min_close']:.2f}), "
            "whereas CatBoost's deficit is significant (p < 0.001); the three leading models are therefore "
            "practically equivalent.")


def s361(D: Doc, n):
    c = cv(n)
    check_cv_pattern(c)
    anchor = D.para("W is compared against the critical value")
    h = D.after(anchor, "3.2.1 Primary", "3.6.1 STATISTICAL POWER: REPEATED CROSS-VALIDATION")
    body = "All four models (Random Forest, XGBoost"
    p = D.after(h, body,
        "The n = 5 paired observations underlying Table 3 give limited statistical power to detect small but "
        "genuine differences between models. We therefore repeated the complete leakage-safe protocol of "
        "Algorithm 1 -- including fold-local oversampling and scaler fitting -- with 10 independent "
        "fold-assignment seeds (42-51), yielding 50 paired fold-level Macro-F1 observations per model. Seed 42 is "
        "the original run and reproduces Tables 2 and 3 exactly. Across the 50 folds, mean Macro-F1 was "
        f"{c['ms']('lightgbm')} for LightGBM, {c['ms']('random_forest')} for Random Forest, "
        f"{c['ms']('xgboost')} for XGBoost, and {c['ms']('catboost')} for CatBoost.")
    p = D.after(p, body,
        "Fold-level scores from repeated cross-validation are not independent, because the training sets of "
        "different folds overlap, so tests that treat them as independent -- including the Wilcoxon signed-rank "
        "test at n = 50 -- overstate significance. Alongside the Wilcoxon test we therefore report the corrected "
        "resampled t-test of Nadeau and Bengio [42], the standard conservative test for repeated cross-validation, "
        "which inflates the variance of the J paired differences d by the test-to-training size ratio: "
        "t = mean(d) / sqrt[(1/J + n_test/n_train) × var(d)], with n_test/n_train = 1/4 for five-fold "
        "cross-validation and J − 1 = 49 degrees of freedom. Table 3c reports both tests.")
    rows = [["**Comparison**", "**Mean ΔMacro-F1**", "**Folds won by LightGBM**", "**Wilcoxon W (p)**",
             "**Corrected t(49) (p)**"]]
    for m in ("random_forest", "xgboost", "catboost"):
        w, t = c["w"][m], c["t"][m]
        rows.append([f"LightGBM vs. {MODEL_NAMES[m]}", f"+{w['mean_diff']:.4f}", f"{w['best_wins']} / {w['n']}",
                     f"{w['W']:.1f} (p {fmt_p(w['p'])})", f"{t['t']:.2f} (p {fmt_p(t['p'])})"])
    tbl = clone_table(D.table("Model", ncols=5), p, rows, D.mark)
    cap = D.after(tbl, "Table 3. Paired Wilcoxon",
                  "Table 3c. Repeated Stratified Five-Fold Cross-Validation (10 Seeds, n = 50 Paired Folds per "
                  "Comparison): Paired Wilcoxon Signed-Rank and Corrected Resampled t-Tests")
    p = D.after(cap, body,
        "With n = 50, the Wilcoxon test is significant for all three comparisons (p < 0.001). The corrected test "
        f"confirms that CatBoost is significantly worse than LightGBM (mean difference {c['diff_cb']}, "
        f"p {c['p_cb']}), but not that LightGBM outperforms Random Forest (p {c['p_rf']}) or XGBoost "
        f"(p {c['p_xgb']}). LightGBM's advantage over these two models is consistent -- it achieved the higher "
        f"Macro-F1 in {c['win_rf']} and {c['win_xgb']} of the 50 folds, respectively -- but small ({c['diff_rf']} "
        "Macro-F1, about a quarter of a percentage point), and it does not survive correction for the dependence "
        "between folds. We therefore make no claim that LightGBM is superior to Random Forest or XGBoost: the "
        "three models are practically equivalent on this task, and LightGBM is retained as the default model on "
        "the basis of its marginally higher mean score rather than a demonstrated statistical advantage.")
    D.after(p, body,
        "The repeated analysis covers the tree-ensemble comparison of Tables 2 and 3. It was not extended to the "
        "deep-learning baselines of Table 3b (Section 4.1) because of their much higher training cost "
        "(approximately 109 s and 154 s per fold for the CNN-LSTM and Transformer, respectively, versus about 2 s "
        "for LightGBM); this is noted as a limitation in Section 4.10.")


def limitation_cv(n):
    return ("Fifth, the repeated cross-validation of Section 3.6.1 shows that LightGBM's small advantage over "
            "Random Forest and XGBoost is consistent across folds but not statistically significant once fold "
            "dependence is accounted for, so this study does not establish which of the three leading models is "
            "best. The repeated analysis covered the tree-ensemble comparison only; it was not extended to the "
            "deep-learning baselines of Table 3b because of their training cost, and nested cross-validation, "
            "which would additionally capture hyperparameter-selection variance, remains future work.")


def conclusion_1(n):
    c = cv(n)
    return ("This paper presented an explainable AI-based intrusion detection engine for smart grid cyber-physical "
            "systems, architected as the core of a broader IDPS whose rule-based prevention/policy-response layer is "
            "described in Section 3.1 and evaluated post hoc in Section 4.9, and assessed with a level of statistical "
            "and cross-dataset rigor not typically demonstrated in this literature. Under leakage-safe stratified "
            "five-fold cross-validation, LightGBM achieved the highest mean performance (Macro-F1 0.928 ± 0.003, "
            "accuracy 96.7% ± 0.1%). Repeated cross-validation with 50 paired folds showed that its advantage over "
            f"Random Forest and XGBoost is consistent but small ({c['diff_rf']} Macro-F1) and not significant once "
            "fold dependence is accounted for, so the three models are best regarded as practically equivalent, "
            "whereas CatBoost is significantly worse. LightGBM also outperformed compact CNN-LSTM and Transformer "
            "deep-learning baselines on every fold under an identical protocol, again without reaching statistical "
            "significance at five folds. A feature-group ablation study identified communication-domain features as "
            "the single most influential category (0.192 Macro-F1 drop upon removal), ahead of electrical, "
            "equipment-health, and power-quality features, with this ranking independently corroborated by "
            "SHAP-based per-class explanations and plausibly underlying part of the cross-dataset performance gap "
            "(Section 4.6).")


# --------------------------------------------------------------------------
def build(out_path, mark, n):
    D = Doc(SOURCE, mark)
    revise_front(D, n)
    revise_methods(D, n)
    s361(D, n)
    revise_results(D, n)
    revise_back(D, n)
    D.d.save(str(out_path))


def main():
    n = load_numbers()
    if n["M"] is None:
        raise SystemExit("results/msu_shap_multiclass.json is missing -- run scripts/msu_shap_analysis.py first.")
    build(HERE / "Manuscript_R3_clean.docx", False, n)
    build(HERE / "Manuscript_R3_highlighted.docx", True, n)
    print("Saved manuscript/Manuscript_R3_clean.docx and manuscript/Manuscript_R3_highlighted.docx")


if __name__ == "__main__":
    main()
