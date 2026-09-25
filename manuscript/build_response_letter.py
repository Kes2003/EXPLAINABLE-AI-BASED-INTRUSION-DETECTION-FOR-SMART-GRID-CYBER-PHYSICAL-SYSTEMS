"""
Build the point-by-point response to reviewers (.docx) for Revision 3.

All numbers come from build_manuscript.load_numbers(), i.e. from the same
result files as the manuscript, so the letter and the manuscript cannot
disagree.

Usage (from the project root):
    python manuscript/build_response_letter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_manuscript as bm  # noqa: E402

TITLE = "Explainable AI-Based Intrusion Detection for Smart Grid Cyber-Physical Systems"

REVIEWER_4 = [
    "The prevention/policy-response component of the proposed IDPS is described in the manuscript, but its actual "
    "operational performance has not been experimentally evaluated. The authors are requested to provide suitable "
    "experimental results for response accuracy, response latency, and false-block rate, or clearly limit the "
    "claims regarding the prevention capability of the proposed framework.",
    "The statistical significance analysis is performed using only five cross-validation folds. This provides "
    "limited statistical power for drawing strong conclusions regarding the superiority of one model over another. "
    "The authors are requested to either strengthen the statistical evaluation using repeated or nested "
    "cross-validation or moderate the claims regarding model superiority.",
    "The SHAP-based explainability and feature-ablation analyses are mainly evaluated using the synthetic dataset. "
    "Since the MSU/ORNL dataset is used for external validation, the authors are requested to clarify why similar "
    "explainability analysis has not been performed on the external real-world dataset and discuss the "
    "implications of this limitation.",
    "A considerable performance gap is observed between the synthetic dataset and the MSU/ORNL real-world dataset. "
    "The authors should provide a more detailed discussion explaining the reasons for this performance degradation "
    "and clearly state the limitations regarding the transferability and generalization of the proposed framework "
    "to real smart-grid environments.",
    "The robustness analysis indicates a substantial degradation in performance under systematic distribution "
    "shift, with Macro-F1 decreasing from 0.938 to 0.402 at 20% distribution shift. This is an important concern "
    "for practical deployment. The authors are requested to provide a more detailed discussion of how sensor drift "
    "and distribution changes can be detected and mitigated in a real-time deployment scenario.",
    "The novelty of the manuscript has been better explained in the revised version; however, the main "
    "contribution still appears to be the integration of several existing techniques. The authors are requested "
    "to clearly distinguish the proposed framework from existing smart-grid IDPS approaches and explicitly state "
    "the specific technical novelty and original contribution of the work.",
    "The manuscript presents the framework as a deployment-oriented IDPS, while the prevention functionality has "
    "not yet been independently benchmarked and the system shows sensitivity to distribution shift. The authors "
    "should therefore ensure that the statements regarding real-world readiness, deployment capability, and "
    "practical effectiveness are appropriately supported by the experimental evidence.",
]


def responses(n):
    return [
        # 1 -- prevention layer
        (f"We thank the reviewer for this important point and have added a quantitative evaluation of the "
         f"prevention layer in the new Section 4.9. Because the policy engine is a deterministic function of the "
         f"predicted class and its probability (Algorithm 2, line 6), we passed all {n['n_pred']} five-fold "
         f"out-of-fold predictions of the original cross-validation run through the deployed policy function "
         f"(confidence threshold τ = 0.7) and compared every selected action with an explicit ground-truth "
         f"class-to-action mapping (new Table 10). Response accuracy, defined strictly as an exact match with the "
         f"ground-truth action, is {n['resp_acc']}; the false-block rate on normal traffic is {n['fbr']} "
         f"({n['fb']} of {n['n_normal']} normal samples). We also report the missed-response rate, "
         f"{n['missed_rate']} ({n['missed']} of {n['n_need']} DoS, tamper, and overload samples received no "
         f"automated action), which shows that the confidence gate trades some missed responses for a very low "
         f"false-block rate. The policy engine's own latency, timed over {n['lat_calls']} calls, is "
         f"{n['lat_mean']} μs per decision on average (99th percentile {n['lat_p99']} μs), roughly three orders of "
         f"magnitude below the 1.70 ms ML-inference latency. We have also limited the claims accordingly: "
         f"Sections 3.1 and 4.9 state explicitly that this evaluation does not cover closed-loop operation (actual "
         f"breaker actuation, rate-limit enforcement, operator handling of escalations) or the optimality of the "
         f"mapping itself, and the Abstract, contributions list, and Conclusion describe closed-loop prevention as "
         f"architecturally integrated but not yet field-validated.",
         ["Section 3.1 (scope statement, Fig. 1 legend)", "Algorithm 2 (line 6)", "Section 4.9 and Table 10 (new)",
          "Section 4.10 (sixth limitation)", "Abstract, contributions list, Conclusion"]),
        # 2 -- statistical power
        (response_cv(n),
         ["Section 3.6.1 and Table 3c (new)", "Algorithm 1 note, Sections 3.6 and 3.9",
          "Section 4.10 (fifth limitation)", "Abstract, contributions list, Conclusion", "Reference 42 (new)"]),
        # 3 -- explainability on MSU/ORNL
        (response_xai(n),
         ["Section 4.7 and Tables 8b and 8c (new)", "Section 3.7", "Section 4.10 (fourth limitation)",
          "Abstract, Conclusion"]),
        # 4 -- performance gap
        ("We have added a detailed analysis of the performance gap to Section 4.6. It identifies four converging "
         "causes: (i) label granularity (5 broad synthetic classes vs. 37 fine-grained MSU/ORNL scenarios, several "
         "of them electrically similar); (ii) feature abstraction (18 hand-engineered, domain-informed features "
         "vs. 128 low-level PMU and log channels); (iii) the more separable, idealized class-conditional "
         "distributions of synthetic data compared with noisy, non-stationary testbed transients; and (iv) the "
         "absence in MSU/ORNL of network-traffic statistics equivalent to packet rate and packet error rate, the "
         "feature group our ablation study (Section 4.4) found most informative. "
         + gap_xai_link(n) +
         "We now state explicitly that the 83.1% MSU/ORNL Macro-F1, rather than the 92.8% synthetic figure, is "
         "the more representative estimate of real-world performance, and Section 4.10 now states the "
         "transferability limits directly: external validation in this study means retraining the same "
         "methodology on a second dataset, the framework has not been evaluated on telemetry from an operational "
         "utility network, and a model trained on one grid's data should not be assumed to transfer to another "
         "without site-specific retraining and re-validation.",
         ["Section 4.6 (new paragraph)", "Section 4.10 (first limitation)", "Abstract, Conclusion"]),
        # 5 -- distribution shift
        ("We have expanded the second limitation in Section 4.10 into a concrete, testable detection-and-mitigation "
         "design: (i) label-free, per-feature drift monitoring (rolling-window two-sample Kolmogorov-Smirnov tests "
         "or Population Stability Index against a reference window from the training distribution), sized to "
         "detect small shifts because Table 7 shows that a 5% shift already reduces Macro-F1 to 0.700; (ii) the "
         "model's own confidence as a proxy signal, since a sustained fall in mean top-class probability without "
         "a change in alert rates indicates drift rather than changed grid behaviour; this section also notes "
         "that drift-induced confidence loss would increase missed responses in the prevention layer "
         "(Section 4.9); and (iii) shadow retraining, in which a candidate model retrained on recent telemetry "
         "runs in parallel with the deployed model and is promoted only after outperforming it on recently "
         "labelled data. We present this explicitly as a proposed design for future empirical validation, not "
         "as a tested capability.",
         ["Section 4.10 (second limitation)", "Section 4.5 (last paragraph)", "Abstract, Conclusion"]),
        # 6 -- novelty
        ("We have added Section 1.1, \"Novelty and Positioning\". It states plainly that the contribution is not "
         "a new learning algorithm, and identifies it as the joint empirical validation of five elements under "
         "one consistent protocol: (i) multidomain feature fusion whose necessity is demonstrated by ablation; "
         "(ii) leakage-safe cross-validation with paired significance testing, now strengthened by repeated "
         "cross-validation; (iii) independent external validation on real-world data, extended with "
         "cross-dataset explainability; (iv) SHAP explanations whose stability is verified across folds and "
         "under perturbation; and (v) hardware-benchmarked detection latency together with a quantitative "
         "evaluation of the prevention layer. The Table 1 legend now makes the comparison explicit: none of the "
         "reviewed smart-grid IDPS studies reports external validation or significance testing, and none "
         "evaluates its prevention layer quantitatively.",
         ["Section 1.1 (new)", "Table 1 legend and 'Proposed Work' row", "Contributions list"]),
        # 7 -- deployment-readiness claims
        ("We have revised every statement about deployment and prevention so that it matches the evidence. The "
         "Abstract and contributions list now describe the detection component as deployed and "
         "hardware-benchmarked, and the prevention layer as quantitatively evaluated post hoc (Section 4.9) but "
         "not field-validated. The Conclusion states that, because the prevention layer has not been validated "
         "under live operating conditions and the detector is sensitive to distribution shift, closed-loop "
         "automated prevention should be regarded as architecturally integrated but not yet field-validated. "
         "Section 4.10 adds explicit limits on transferability to operational networks, and Section 4.9 notes "
         "that prevention accuracy on real telemetry would be lower than on the synthetic data.",
         ["Abstract", "Contributions list", "Sections 3.1, 4.9, 4.10", "Conclusion"]),
    ]


def build(out_path):
    n = bm.load_numbers()
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)

    def para(text="", bold=False, italic=False, size=None, align=None, color=None, space_after=6):
        p = doc.add_paragraph()
        for seg, b, i in bm_parse(text):
            r = p.add_run(seg)
            r.bold = bold or b
            r.italic = italic or i
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = color
        if align is not None:
            p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        return p

    para("Response to Reviewers", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(f"Manuscript: {TITLE}", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
    para("Dear Editor,")
    para("We thank the editor and the reviewers for their careful evaluation of our manuscript. We have addressed "
         "every remaining comment. For Reviewer 4, we carried out new analyses where they were requested "
         "(points 1-3) and expanded the discussion where that was requested (points 4-7). Every new result was "
         "produced by the scripts in the project's code repository (see the new Code Availability statement). "
         "As a check, re-running the original cross-validation in the new analysis environment reproduced every "
         "value in Tables 2 and 3 exactly. Reviewer comments are reproduced below in italics, each followed by "
         "our response and the location of the changes in the revised manuscript. In the version with changes "
         "highlighted, all text added or changed in this revision is marked in yellow.")
    para("Summary of the main changes", bold=True, space_after=4)
    for item in summary_items(n):
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)

    para("")
    para("Reviewer 1", bold=True, size=12)
    para("\"ok now\"", italic=True)
    para("**Response:** We thank Reviewer 1 for confirming that the previous revisions address their concerns. "
         "No further changes were requested.")
    para("Reviewer 2", bold=True, size=12)
    para("\"All corrections are incorporated\"", italic=True)
    para("**Response:** We thank Reviewer 2 for confirming that all previously requested corrections have been "
         "incorporated. No further changes were requested.")

    para("Reviewer 4", bold=True, size=12)
    para("Recommendation: Minor Revision. We thank Reviewer 4 for these constructive comments, which have "
         "substantially strengthened the evaluation of the prevention layer, the statistical analysis, and the "
         "external-validation discussion.", italic=True)
    for i, (comment, (resp, where)) in enumerate(zip(REVIEWER_4, responses(n)), 1):
        para(f"Comment {i}", bold=True, space_after=2)
        para(f"\"{comment}\"", italic=True, color=RGBColor(0x40, 0x40, 0x40))
        para("**Response:** " + resp)
        para("**Changes in the manuscript:** " + "; ".join(where) + ".", space_after=12)

    para("We believe these revisions fully address the reviewers' comments, and we thank them again for their "
         "guidance.")
    para("Sincerely,", space_after=2)
    para("Aravind Pitchai (corresponding author), on behalf of all authors")
    doc.save(out_path)


def bm_parse(text):
    from docx_edit import parse_md
    return parse_md(text)


def response_cv(n):
    c = bm.cv(n)
    bm.check_cv_pattern(c)
    return ("We agree and have strengthened the analysis with repeated cross-validation (new Section 3.6.1 and "
            "Table 3c). The complete leakage-safe protocol of Algorithm 1 was repeated with 10 independent "
            "fold-assignment seeds, giving n = 50 paired fold-level observations per comparison instead of 5; the "
            "first seed is the original run and reproduces Tables 2 and 3 exactly. Because fold scores from "
            "repeated cross-validation are not independent (training sets overlap), a Wilcoxon test at n = 50 "
            "overstates significance, so we additionally report the corrected resampled t-test of Nadeau and "
            "Bengio (new reference 42). The Wilcoxon test is significant for all comparisons (p < 0.001). The "
            f"corrected test confirms that CatBoost is significantly worse than LightGBM (p {c['p_cb']}), but not "
            f"that LightGBM outperforms Random Forest (p {c['p_rf']}) or XGBoost (p {c['p_xgb']}): LightGBM wins "
            f"{c['win_rf']} and {c['win_xgb']} of the 50 folds, but its mean advantage is only {c['diff_rf']} "
            "Macro-F1. We have therefore moderated the claims throughout: the manuscript now states that LightGBM, "
            "Random Forest, and XGBoost are practically equivalent on this task, that LightGBM is retained as the "
            "default model only because of its marginally higher mean score, and that the study does not establish "
            "which of the three is best (Abstract, Section 3.6.1, Section 4.10, Conclusion).")


def summary_items(n):
    c = bm.cv(n)
    return [
        f"New Section 4.9 and Table 10: quantitative evaluation of the prevention/policy layer (response accuracy "
        f"{n['resp_acc']}, false-block rate {n['fbr']}, missed-response rate {n['missed_rate']}, policy latency "
        f"{n['lat_mean']} μs per decision).",
        "New Section 3.6.1 and Table 3c: repeated cross-validation (10 seeds, n = 50 paired folds) with Wilcoxon "
        "and corrected resampled t-tests; the claims of model superiority have been moderated accordingly.",
        "New Section 4.7 and Tables 8b and 8c: SHAP explainability and feature-group ablation on the external "
        "MSU/ORNL dataset, with a discussion of their implications and limits.",
        "Section 4.6: new analysis of the causes of the synthetic/MSU-ORNL performance gap; Section 4.10: explicit "
        "limits on transferability to operational networks.",
        "Section 4.10: a concrete, testable design for detecting and mitigating sensor drift and distribution shift.",
        "New Section 1.1 and revised Table 1: explicit statement of the specific contribution and how it differs "
        "from prior smart-grid IDPS work.",
        "Abstract, contributions list, and Conclusion: deployment and prevention claims aligned with the evidence.",
        "Also added: the environment used for the new analyses (Section 3.9), a Code Availability statement, and "
        "one new reference [42].",
    ]


def gap_xai_link(n):
    m = bm.msu(n)
    return ("The new SHAP analysis of Section 4.7 corroborates the fourth cause directly: on MSU/ORNL the "
            f"relay, control-panel, and Snort log signals together receive only {m['cyber']} of the attribution, "
            f"and removing all cyber-side columns reduces Macro-F1 by only {m['d_cyber']}, whereas communication "
            "features are the most influential group on the synthetic data (0.192 drop when removed). ")


def response_xai(n):
    m = bm.msu(n)
    return ("We agree that neither analysis should be limited to the synthetic dataset, and we have performed both "
            "on MSU/ORNL (new Section 4.7, Tables 8b and 8c). The external-validation pipeline (Section 4.6) does not "
            "store its fitted models, but it is deterministic; re-running its LightGBM configuration on the same five "
            f"folds reproduced the Table 8 result exactly (Macro-F1 {m['f1']}). (1) SHAP: each fold's model was "
            "explained with TreeExplainer on its own held-out test fold "
            f"({m['rows']} explained samples in total). Because the dataset's 37 classes make a per-class figure "
            "impractical and its 128 features are channel identifiers rather than named domain quantities, we report "
            "a global ranking aggregated into seven schema groups. PMU voltage and current phasors account for "
            f"{m['phasors']} of the attribution, apparent impedance for {m['imp']}, and frequency for {m['freq']}, "
            f"while the relay, control-panel, and Snort log signals together account for only {m['cyber']} (the "
            "Snort indicators receive none at all). (2) Feature-group ablation, with the same folds: removing the "
            f"current or voltage phasors reduces Macro-F1 by {m['d_curr']} and {m['d_volt']}, whereas removing all "
            f"16 cyber-side columns reduces it by only {m['d_cyber']} -- compared with 0.192 for the "
            "communication-domain group on the synthetic data. Removing apparent impedance does not reduce "
            "performance at all despite its SHAP share, because it is derived from the voltage and current phasors; "
            "we discuss this as an illustration of why attribution and ablation are complementary. Together these "
            "results corroborate one of the causes of the synthetic/real performance gap -- the communication-domain "
            "evidence that is most informative on the synthetic data is effectively absent from MSU/ORNL -- and show "
            "that the multidomain premise of our approach could not be exercised on that dataset. On the implications "
            "(Section 4.7): the explanations identify which measurement channels drive a decision, but turning them "
            "into operator-facing statements requires the dataset documentation and domain expertise, so "
            "explainability is only as actionable as the underlying feature schema. Section 4.10 lists the "
            "remaining limitations (global rather than per-class explanations, a sampled explanation set, and no "
            "explanation-stability check on MSU/ORNL).")


if __name__ == "__main__":
    out = HERE / "Response_to_Reviewers_R3.docx"
    build(out)
    print(f"Saved {out}")
