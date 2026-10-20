# scripts/dl_baselines.py
"""
Deep-learning baselines (manuscript Section 4.1, Table 3b), fully specified.

Protocol: the five seed-42 folds of Algorithm 1. Inside every outer training
fold, 10% of the rows (stratified, seed 42) are held out as a validation set
BEFORE oversampling; the remaining 90% are oversampled (fold-local) and used
to fit the StandardScaler and the network. The validation set drives early
stopping; the outer test fold is never used for training, early stopping or
model selection.

Architectures (input: the 18 standardized features, treated as a length-18
sequence with one channel):
  CNN-LSTM    Conv1D(F, 3, ReLU, same) -> Conv1D(2F, 3, ReLU, same) ->
              MaxPool1D(2) -> LSTM(2F) -> Dropout(0.3) -> Dense(64, ReLU) ->
              Dropout(0.3) -> Dense(5, softmax)
  Transformer per-feature linear embedding to d (Dense(d) on each scalar)
              + learned feature-position embedding -> 2 x [MultiHeadAttention
              (4 heads, key_dim d/4, dropout 0.1) + residual + LayerNorm ->
              FFN(2d, ReLU) -> Dense(d) + residual + LayerNorm] ->
              GlobalAveragePooling1D -> Dropout(0.3) -> Dense(64, ReLU) ->
              Dense(5, softmax)
Training: Adam, categorical cross-entropy (sparse labels), batch size 256,
at most 50 epochs, EarlyStopping(monitor=val_loss, patience=5,
restore_best_weights=True), L2 weight decay 1e-4 on Dense/Conv/LSTM kernels.
Hyperparameter selection (grid, done ONCE on the training part of outer fold
1, scored by validation Macro-F1): learning rate in {1e-3, 3e-4} x width
F in {16, 32} (CNN-LSTM) / d in {32, 64} (Transformer). The selected setting
is then fixed for all five folds.

Seeds: tf.keras.utils.set_random_seed(42) before every model and
tf.config.experimental.enable_op_determinism().

Usage:  python -m scripts.dl_baselines
Output: results/r4/dl_baselines.json
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.r4_common import LIGHTGBM_CV, build_model, fresh_label_encoder, load_synthetic, save, synthetic_folds
from src.train import oversample_minority_classes
from src.utils import set_global_seed

L2 = tf.keras.regularizers.l2(1e-4)
GRID = {"cnn_lstm": [{"lr": lr, "width": w} for lr in (1e-3, 3e-4) for w in (16, 32)],
        "transformer": [{"lr": lr, "width": w} for lr in (1e-3, 3e-4) for w in (32, 64)]}
MAX_EPOCHS, BATCH, PATIENCE = 50, 256, 5


def cnn_lstm(n_feat, n_cls, width):
    inp = tf.keras.Input((n_feat, 1))
    x = tf.keras.layers.Conv1D(width, 3, padding="same", activation="relu", kernel_regularizer=L2)(inp)
    x = tf.keras.layers.Conv1D(2 * width, 3, padding="same", activation="relu", kernel_regularizer=L2)(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.LSTM(2 * width, kernel_regularizer=L2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=L2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(n_cls, activation="softmax")(x))


class FeaturePosition(tf.keras.layers.Layer):
    def __init__(self, n_feat, d, **kw):
        super().__init__(**kw)
        self.emb = tf.keras.layers.Embedding(n_feat, d)
        self.n_feat = n_feat

    def call(self, x):
        return x + self.emb(tf.range(self.n_feat))


def transformer(n_feat, n_cls, d):
    inp = tf.keras.Input((n_feat, 1))
    x = tf.keras.layers.Dense(d, kernel_regularizer=L2)(inp)
    x = FeaturePosition(n_feat, d)(x)
    for _ in range(2):
        a = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d // 4, dropout=0.1)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + a)
        f = tf.keras.layers.Dense(2 * d, activation="relu", kernel_regularizer=L2)(x)
        f = tf.keras.layers.Dense(d, kernel_regularizer=L2)(f)
        x = tf.keras.layers.LayerNormalization()(x + f)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=L2)(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(n_cls, activation="softmax")(x))


BUILD = {"cnn_lstm": cnn_lstm, "transformer": transformer}


def split_fold(df, features, le, tr, seed=42):
    """Outer training fold -> (train 90% oversampled, validation 10%), scaler fit on the oversampled part."""
    tr_df = df.iloc[tr].reset_index(drop=True)
    fit_df, val_df = train_test_split(tr_df, test_size=0.1, stratify=tr_df["label"], random_state=seed)
    bal = oversample_minority_classes(fit_df.reset_index(drop=True), label_col="label", random_state=seed)
    sc = StandardScaler().fit(bal[features].values)
    return sc, (sc.transform(bal[features].values), le.transform(bal["label"].values)), \
        (sc.transform(val_df[features].values), le.transform(val_df["label"].values))


def fit(arch, hp, train, val, n_cls):
    tf.keras.utils.set_random_seed(42)
    model = BUILD[arch](train[0].shape[1], n_cls, hp["width"])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp["lr"]), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    t0 = time.time()
    h = model.fit(train[0][..., None], train[1], validation_data=(val[0][..., None], val[1]), epochs=MAX_EPOCHS,
                  batch_size=BATCH, callbacks=[es], verbose=0, shuffle=True)
    return model, time.time() - t0, len(h.history["loss"]), int(np.argmin(h.history["val_loss"])) + 1


def main():
    tf.config.experimental.enable_op_determinism()
    set_global_seed(42)
    _, df, features = load_synthetic()
    le = fresh_label_encoder(df["label"].values)
    n_cls = len(le.classes_)
    folds = synthetic_folds(df, 42)

    # hyperparameter selection on outer fold 1's training part only
    _, tr1, va1 = split_fold(df, features, le, folds[0][0])
    selection = {}
    for arch, grid in GRID.items():
        scores = []
        for hp in grid:
            m, secs, ep, best_ep = fit(arch, hp, tr1, va1, n_cls)
            f1 = f1_score(va1[1], m.predict(va1[0][..., None], verbose=0).argmax(1), average="macro")
            scores.append({**hp, "val_macro_f1": float(f1), "epochs": ep, "best_epoch": best_ep, "seconds": secs})
            print(f"[select] {arch} {hp}: val Macro-F1 {f1:.4f} ({ep} epochs, {secs:.0f}s)", flush=True)
        best = max(scores, key=lambda s: s["val_macro_f1"])
        selection[arch] = {"grid": scores, "selected": {"lr": best["lr"], "width": best["width"]}}

    res = {"selection": selection, "folds": {}}
    lgb_f1 = []
    for arch in GRID:
        res["folds"][arch] = []
    for k, (tr, te) in enumerate(folds, 1):
        sc, train, val = split_fold(df, features, le, tr)
        Xte = sc.transform(df.iloc[te][features].values)
        yte = le.transform(df.iloc[te]["label"].values)
        for arch in GRID:
            hp = selection[arch]["selected"]
            m, secs, ep, best_ep = fit(arch, hp, train, val, n_cls)
            p = m.predict(Xte[..., None], verbose=0).argmax(1)
            res["folds"][arch].append({"fold": k, "macro_f1": float(f1_score(yte, p, average="macro")),
                                       "accuracy": float(accuracy_score(yte, p)), "epochs_run": ep,
                                       "best_epoch": best_ep, "train_seconds": secs,
                                       "n_params": int(m.count_params())})
            print(f"fold {k} {arch}: Macro-F1 {res['folds'][arch][-1]['macro_f1']:.4f} ({ep} epochs, {secs:.0f}s)",
                  flush=True)
        # LightGBM on exactly the same 90% training part (for a like-for-like reference)
        lg = build_model(LIGHTGBM_CV)
        t0 = time.time(); lg.fit(train[0], train[1]); lsec = time.time() - t0
        lgb_f1.append({"macro_f1": float(f1_score(yte, lg.predict(Xte), average="macro")),
                       "accuracy": float(accuracy_score(yte, lg.predict(Xte))), "train_seconds": lsec})

    summary = {}
    for arch, rows in list(res["folds"].items()) + [("lightgbm_same_90pct_split", lgb_f1)]:
        f1s = [r["macro_f1"] for r in rows]
        summary[arch] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                         "accuracy_mean": float(np.mean([r["accuracy"] for r in rows])),
                         "accuracy_std": float(np.std([r["accuracy"] for r in rows])),
                         "train_seconds_mean": float(np.mean([r["train_seconds"] for r in rows]))}
        if "n_params" in rows[0]:
            summary[arch]["n_params"] = rows[0]["n_params"]
            summary[arch]["epochs_run"] = [r["epochs_run"] for r in rows]
            summary[arch]["best_epoch"] = [r["best_epoch"] for r in rows]
    for arch in GRID:
        a = [r["macro_f1"] for r in lgb_f1]
        b = [r["macro_f1"] for r in res["folds"][arch]]
        w = wilcoxon(a, b)
        summary[arch]["wilcoxon_vs_lightgbm"] = {"W": float(w.statistic), "p": float(w.pvalue),
                                                 "lightgbm_wins": int(sum(x > y for x, y in zip(a, b)))}
    res["summary"] = summary
    res["lightgbm_same_split_folds"] = lgb_f1
    res["tensorflow"] = tf.__version__
    for k, v in summary.items():
        print(k, v)
    save("dl_baselines", res)


if __name__ == "__main__":
    main()
