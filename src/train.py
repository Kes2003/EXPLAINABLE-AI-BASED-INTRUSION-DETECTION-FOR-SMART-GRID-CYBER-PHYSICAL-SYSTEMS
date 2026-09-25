from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

# Fix matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

try:
    from .preprocess import Preprocessor
    from .model import train_eval, train_eval_presplit, save_model
    from .utils import ensure_dir, save_json, set_global_seed
except ImportError:
    from preprocess import Preprocessor
    from model import train_eval, train_eval_presplit, save_model
    from utils import ensure_dir, save_json, set_global_seed


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def oversample_minority_classes(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    if label_col not in df.columns:
        return df

    counts = df[label_col].value_counts()
    if counts.empty:
        return df

    max_count = counts.max()
    classes = counts.index.tolist()

    balanced_frames = []
    rng = np.random.default_rng(random_state)

    for cls in classes:
        cls_df = df[df[label_col] == cls]
        n = len(cls_df)
        if n == 0:
            continue
        if n < max_count:
            n_extra = max_count - n
            idx = rng.integers(0, n, size=n_extra)
            extra = cls_df.iloc[idx]
            balanced_cls_df = pd.concat([cls_df, extra], ignore_index=True)
        else:
            balanced_cls_df = cls_df
        balanced_frames.append(balanced_cls_df)

    balanced_df = pd.concat(balanced_frames, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced_df


def generate_visualizations(
    df: pd.DataFrame,
    features: list[str],
    preproc: Preprocessor,
    model,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, Any],
    artifacts_dir: Path,
) -> None:
    """Generate all visualization plots"""
    
    vis_dir = artifacts_dir / "visualizations"
    ensure_dir(vis_dir)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("  Generating attack distribution chart...")
    # 1. Attack Distribution
    if "label" in df.columns:
        plt.figure(figsize=(10, 6))
        counts = df["label"].value_counts()
        ax = counts.plot(kind="bar", color='steelblue', edgecolor='black')
        plt.title("Attack Type Distribution (after oversampling)", fontsize=14, fontweight='bold')
        plt.xlabel("Attack Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for i, v in enumerate(counts):
            ax.text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(vis_dir / "01_attack_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: 01_attack_distribution.png")

    print("  Generating feature correlation heatmap...")
    # 2. Feature Correlations
    plt.figure(figsize=(12, 10))
    corr = df[features].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, 
                cbar_kws={'label': 'Correlation'}, linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(vis_dir / "02_feature_correlations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 02_feature_correlations.png")

    print("  Generating confusion matrix...")
    # 3. Confusion Matrix
    if preproc.label_encoder is not None:
        class_names = preproc.label_encoder.classes_
    else:
        class_names = [str(c) for c in sorted(np.unique(y_test))]

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(vis_dir / "03_model_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 03_model_performance.png")

    print("  Generating feature importance chart...")
    # 4. Feature Importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx_sorted = np.argsort(importances)[::-1]
        top_n = min(15, len(features))
        top_idx = idx_sorted[:top_n]
        top_features = [features[i] for i in top_idx]
        top_values = importances[top_idx]

        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_values, color=colors, edgecolor='black')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title("Top Feature Importances", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_values)):
            plt.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(vis_dir / "04_feature_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: 04_feature_analysis.png")
    else:
        print("    ⚠ Feature importances not available for this model type")

    print("  Generating system architecture diagram...")
    # 5. Smart Grid Architecture
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    
    text_content = """
    Smart Grid IDS Architecture
    
    ════════════════════════════════════════════════════════════
    
    DATA SOURCES:
      • Electrical Sensors (voltage, current, frequency, power)
      • Power Quality Monitors (THD, phase imbalance)
      • Network Telemetry (packet rates, error rates)
      • Asset Health Sensors (temperatures, load, tap position)
    
    ════════════════════════════════════════════════════════════
    
    PROCESSING PIPELINE:
      1. Data Collection → Real-time telemetry from smart grid
      2. Preprocessing → Feature scaling and normalization
      3. ML Detection → Random Forest / XGBoost / LightGBM
      4. Classification → Normal, DoS, FDI, Tamper, Overload
      5. Prevention → Automated response recommendations
    
    ════════════════════════════════════════════════════════════
    
    PREVENTION ACTIONS:
      → DoS Attack: Network rate limiting
      → Tampering: Breaker trip + operator alert
      → Overload: Load shedding + relay trip
      → FDI: Operator investigation required
    
    ════════════════════════════════════════════════════════════
    """
    
    ax.text(0.05, 0.95, text_content, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(vis_dir / "05_smart_grid_architecture.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 05_smart_grid_architecture.png")

    print("  Generating performance summary...")
    # 6. Performance Summary
    report = metrics.get("classification_report", {})
    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})
    macro_f1 = macro_avg.get("f1-score", None)
    weighted_f1 = weighted_avg.get("f1-score", None)
    accuracy = report.get("accuracy", None)

    labels = []
    values = []

    if accuracy is not None:
        labels.append("Accuracy")
        values.append(accuracy)
    if macro_f1 is not None:
        labels.append("Macro F1")
        values.append(macro_f1)
    if weighted_f1 is not None:
        labels.append("Weighted F1")
        values.append(weighted_f1)

    if labels:
        plt.figure(figsize=(10, 6))
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=2)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Score", fontsize=12)
        plt.title("Model Performance Summary", fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(vis_dir / "06_project_summary_fixed.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: 06_project_summary_fixed.png")
    else:
        print("    ⚠ No summary metrics available")
    
    print(f"\n✓ All visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train intrusion detection model")
    parser.add_argument("--data", type=str, required=False, help="Path to CSV dataset")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow training on synthetic or non-real datasets")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    features = cfg["features"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    default_model_type = str(cfg.get("default_model", model_cfg.get("type", "random_forest")))

    artifacts_dir = Path(args.artifacts_dir)
    models_dir = Path(args.models_dir)
    ensure_dir(artifacts_dir)
    ensure_dir(models_dir)

    # Default dataset: if no --data provided, use the real dataset in data/external
    if not args.data:
        suggested = Path("data") / "external" / "enhanced_realistic_smartgrid.csv"
        if suggested.exists():
            data_path = suggested
        else:
            raise SystemExit("No --data provided and no default real dataset found at data/external/enhanced_realistic_smartgrid.csv")
    else:
        data_path = Path(args.data)

    # Enforce use of 'real' dataset by default
    if not args.allow_synthetic:
        # allow if file path contains 'external' or 'real' or 'realistic'
        path_str = str(data_path).lower()
        if "external" not in path_str and "real" not in path_str and "realistic" not in path_str:
            raise SystemExit(
                "By default training only on real/external datasets is allowed.\n"
                "If you really want to train on synthetic data, pass --allow-synthetic flag."
            )

    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

    # IMPORTANT (data-leakage fix): split into train/test FIRST, then oversample
    # ONLY the training portion. Oversampling before the split (the previous
    # behaviour) can place duplicated copies of the same minority-class row in
    # both the train and test sets, which inflates test-set metrics. Splitting
    # first guarantees the test set is untouched, non-duplicated real data.
    from sklearn.model_selection import train_test_split as _tts
    print("Splitting into train/test BEFORE oversampling (avoids train/test leakage)...")
    train_df, test_df = _tts(
        df,
        test_size=train_cfg.get("test_size", 0.2),
        stratify=df["label"] if train_cfg.get("stratify", True) else None,
        shuffle=train_cfg.get("shuffle", True),
        random_state=seed,
    )

    print("Oversampling minority classes (training split only)...")
    train_df_balanced = oversample_minority_classes(train_df, label_col="label", random_state=seed)
    print(f"Training rows after oversampling: {len(train_df_balanced):,}  |  Test rows (untouched): {len(test_df):,}")

    preproc = Preprocessor(
        features=features,
        scaler_path=models_dir / f"scaler_{default_model_type}.joblib",
        label_encoder_path=models_dir / f"label_encoder_{default_model_type}.joblib",
    )

    print("Preprocessing data...")
    X_train, y_train = preproc.fit_transform(train_df_balanced)
    X_test = preproc.transform(test_df)
    y_test = preproc.label_encoder.transform(test_df["label"].astype(str).values)
    print(f"Preprocessed: X_train = {X_train.shape}, X_test = {X_test.shape}")

    print(f"Training {default_model_type} model...")
    model, metrics, y_test, y_pred, y_proba = train_eval_presplit(
        X_train, y_train, X_test, y_test,
        model_cfg=model_cfg,
        artifacts_dir=artifacts_dir,
    )
    # df_balanced kept only for the visualization step below (class-distribution chart)
    df_balanced = train_df_balanced

    # Save under model.joblib and model_{type}.joblib for selection
    model_path = models_dir / "model.joblib"
    model_typed_path = models_dir / f"model_{default_model_type}.joblib"
    save_model(model, model_path)
    save_model(model, model_typed_path)
    print(f"Model saved to: {model_path} and {model_typed_path}")

    # Also save scaler & label_encoder under typed names
    # Preprocessor already saved them to the configured paths in fit_transform
    # For a consistent fallback, we also ensure untyped copies exist
    try:
        # copy typed to untyped if present
        typed_scaler = models_dir / f"scaler_{default_model_type}.joblib"
        untyped_scaler = models_dir / "scaler.joblib"
        if typed_scaler.exists():
            import shutil
            shutil.copy(typed_scaler, untyped_scaler)
    except Exception:
        pass

    try:
        typed_le = models_dir / f"label_encoder_{default_model_type}.joblib"
        untyped_le = models_dir / "label_encoder.joblib"
        if typed_le.exists():
            import shutil
            shutil.copy(typed_le, untyped_le)
    except Exception:
        pass

    # Save metrics and feature names
    save_json(metrics, artifacts_dir / "metrics.json")
    save_json({"features": features}, models_dir / "features.json")
    print(f"Saved metrics to {artifacts_dir / 'metrics.json'}")

    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(
        df=df_balanced,
        features=features,
        preproc=preproc,
        model=model,
        y_test=y_test,
        y_pred=y_pred,
        metrics=metrics,
        artifacts_dir=artifacts_dir,
    )

    print("\nTraining complete!")
    print(f"Model accuracy: {metrics.get('classification_report', {}).get('accuracy', 'N/A'):.4f}")


if __name__ == "__main__":
    main()