# src/explain.py
import joblib, json, os, numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser

MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"  # optional
OUT_DIR = "artifacts/explanations"

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    le = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None
    return model, scaler, le

def explain_rows(model, scaler, X_df, feature_names, top_k=3, class_names=None, plot_dir=None):
    explainer = shap.TreeExplainer(model)  # works for RF/XGB/LGB
    X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
    if scaler is not None:
        # NOTE: scaler expects same columns order as training features.
        X = scaler.transform(X)
    shap_vals = explainer.shap_values(X)  # for multiclass: old shap = list of (n,f) arrays; new shap = single (n,f,classes) array

    # Normalize to a (n_classes, n_samples, n_features) array so per-class plots work
    # regardless of which SHAP version/API produced the output.
    if isinstance(shap_vals, list):
        shap_per_class = np.array(shap_vals)  # (n_classes, n, f)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_per_class = np.transpose(shap_vals, (2, 0, 1))  # (n,f,classes) -> (classes, n, f)
    else:
        shap_per_class = np.abs(shap_vals)[np.newaxis, ...]  # (1, n, f) for binary/regression

    # Aggregate across classes (for the top-feature text explanation, unchanged behaviour)
    shap_arr = np.sum(np.abs(shap_per_class), axis=0)

    results = []
    for i in range(X.shape[0]):
        row_vals = shap_arr[i]
        top_idx = np.argsort(-row_vals)[:top_k]
        top_feats = [(feature_names[j], float(row_vals[j])) for j in top_idx]
        human = "; ".join([f"{name} (impact={val:.3f})" for name, val in top_feats])
        results.append({"index": i, "top_features": top_feats, "explanation_text": human})

    # NEW: per-class mean(|SHAP|) bar chart -- addresses Reviewer 4 pt.12 / Reviewer 5's
    # ask for more than a single global feature-importance bar chart.
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        n_classes = shap_per_class.shape[0]
        for c in range(n_classes):
            mean_abs = np.abs(shap_per_class[c]).mean(axis=0)  # (n_features,)
            order = np.argsort(-mean_abs)[:10]
            label = class_names[c] if class_names is not None and c < len(class_names) else f"class_{c}"
            plt.figure(figsize=(7, 4.5))
            plt.barh([feature_names[j] for j in order][::-1], mean_abs[order][::-1], color="#4C72B0")
            plt.xlabel("mean(|SHAP value|)")
            plt.title(f"Top SHAP features -- '{label}' class")
            plt.tight_layout()
            out_path = os.path.join(plot_dir, f"shap_summary_{label}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {out_path}")

    return results

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="artifacts/demo_explain_input.csv",
                         help="CSV with full feature columns (see scripts to regenerate: "
                              "sample rows from data/external/enhanced_realistic_smartgrid.csv)")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, scaler, le = load_model()

    df = pd.read_csv(args.input)
    # If demo_stream format: we only have some columns; ensure we match model training features.
    # TODO: replace below with your actual feature list order or load models/features.json
    feature_list = []
    if os.path.exists("models/features.json"):
        feature_list = json.load(open("models/features.json"))["features"]
    else:
        # Fallback: pick numeric columns except 't','scenario','pred_label','prob'
        feature_list = [c for c in df.columns if c not in ("t","scenario","pred_label","prob") and df[c].dtype in [int,float]]
    X = df[feature_list]
    class_names = list(le.classes_) if le is not None else None
    explanations = explain_rows(model, scaler, X, feature_list, top_k=3,
                                 class_names=class_names, plot_dir=os.path.join(args.out, "plots"))
    out_path = os.path.join(args.out, "demo_explanations.json")
    with open(out_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print("Saved explanations to", out_path)

if __name__ == "__main__":
    main()
