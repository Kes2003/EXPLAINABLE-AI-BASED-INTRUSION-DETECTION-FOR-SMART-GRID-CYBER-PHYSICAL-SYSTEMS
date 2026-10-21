# scripts/msu_sequence_models.py
"""
Deep sequence models for MSU/ORNL, evaluated event-disjoint.

Outer loop: GroupKFold(5) by recording file, exactly as msu_improved_eval.py.
Each row is classified from a causal window: the row itself and up to W-1
rows recorded before it IN THE SAME EVENT (shorter histories are left-padded
by repeating the event's first row, plus a 0/1 "padded" channel). Inputs are
the 128 raw columns and the physics features, standardised with a scaler fit
on the outer training rows only.

Inside each outer training fold, 2 of the 12 training recordings are held out
for early stopping (patience 4 on validation Macro-F1, at most MAX_EPOCHS);
the outer test recordings are used only for the final score.

Architectures:
  cnn  causal Conv1D(d,k=3) -> ReLU -> Conv1D(d,k=3,dilation 2) -> ReLU ->
       Conv1D(d,k=3,dilation 4) -> ReLU -> last time step -> Dropout(0.3) -> Linear
  gru  Linear(d) -> 2-layer GRU(d, dropout 0.2) -> last time step ->
       Dropout(0.3) -> Linear
Training: AdamW (lr 1e-3, weight decay 1e-4), batch 256, class-balanced
cross-entropy (inverse class frequency in the training rows), seed 42.

Usage:  python -m scripts.msu_sequence_models MODEL [variant ...]  (MODEL: cnn | gru)
Output: results/msu_improved/msu_sequence_models.json (resumable)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from scripts.msu_improve_features import phys_features
from scripts.msu_improved_eval import VARIANTS, data, summarise
from scripts.r4_common import ROOT

OUT = ROOT / "results" / "msu_improved" / "msu_sequence_models.json"
W = int(os.environ.get("SEQ_W", 32))
D = 64
MAX_EPOCHS, PATIENCE, BATCH = 30, 4, 256


def window_index(ev):
    """idx[t, j] = row used at window position j for row t (j = W-1 is t itself)."""
    n = len(ev)
    start = np.zeros(n, dtype=np.int64)
    new = np.r_[True, ev[1:] != ev[:-1]]
    start[new] = np.where(new)[0]
    start = np.maximum.accumulate(start)
    offs = np.arange(-(W - 1), 1)
    idx = np.arange(n)[:, None] + offs[None, :]
    pad = idx < start[:, None]
    return np.where(pad, start[:, None], idx), pad


class CNN(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(f if i == 0 else D, D, 3, dilation=d) for i, d in enumerate((1, 2, 4))])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(D, c))

    def forward(self, x):  # x: (B, W, F)
        h = x.transpose(1, 2)
        for conv in self.convs:
            h = torch.relu(conv(nn.functional.pad(h, ((conv.kernel_size[0] - 1) * conv.dilation[0], 0))))
        return self.head(h[:, :, -1])


class GRU(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.inp = nn.Linear(f, D)
        self.gru = nn.GRU(D, D, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(D, c))

    def forward(self, x):
        out, _ = self.gru(torch.relu(self.inp(x)))
        return self.head(out[:, -1])


def batches(X, idx, pad, rows, shuffle, rng):
    order = rng.permutation(rows) if shuffle else rows
    for i in range(0, len(order), BATCH):
        r = order[i:i + BATCH]
        xb = np.concatenate([X[idx[r]], pad[r][:, :, None].astype(np.float32)], axis=2)
        yield r, torch.from_numpy(xb)


def predict(model, X, idx, pad, rows):
    model.eval()
    out = []
    with torch.no_grad():
        for _, xb in batches(X, idx, pad, rows, False, None):
            out.append(model(xb).argmax(1).numpy())
    return np.concatenate(out)


def fit_fold(kind, Xraw, y, idx, pad, tr, te, groups, n_classes, log):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    fit_rows, val_rows = next(GroupShuffleSplit(1, test_size=2, random_state=42).split(tr, y[tr], groups[tr]))
    fit_rows, val_rows = tr[fit_rows], tr[val_rows]
    X = StandardScaler().fit(Xraw[tr]).transform(Xraw).astype(np.float32)
    np.clip(X, -20, 20, out=X)
    model = (CNN if kind == "cnn" else GRU)(X.shape[1] + 1, n_classes)
    cnt = np.bincount(y[fit_rows], minlength=n_classes).astype(np.float32)
    w = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / n_classes)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    yt = torch.from_numpy(y)
    best, best_state, wait = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for r, xb in batches(X, idx, pad, fit_rows, True, rng):
            opt.zero_grad()
            lossf(model(xb), yt[r]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vf = f1_score(y[val_rows], predict(model, X, idx, pad, val_rows), average="macro")
        if vf > best:
            best, wait = vf, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        log(f"      epoch {epoch + 1}: val F1 {vf:.4f}")
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    # refit on all training recordings is not done: the early-stopped model is scored as is
    p = predict(model, X, idx, pad, te)
    return f1_score(y[te], p, average="macro"), accuracy_score(y[te], p), best, epoch + 1


def main():
    kind = sys.argv[1]
    variants = sys.argv[2:] or VARIANTS
    torch.set_num_threads(4)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    ref, feats, ev, labels = data()
    Xraw = pd.concat([ref[feats], phys_features(ref)], axis=1).values.astype(np.float64)
    idx, pad = window_index(ev)
    groups = ref["_file"].values
    for v in variants:
        key = f"rows/{v}/{kind}_w{W}"
        if key in res:
            continue
        t = time.time()
        y = LabelEncoder().fit_transform(labels[v])
        c = int(y.max() + 1)
        f1s, accs, info = [], [], []
        for k, (tr, te) in enumerate(GroupKFold(5).split(Xraw, y, groups)):
            f, a, vbest, ep = fit_fold(kind, Xraw, y, idx, pad, tr, te, groups, c, lambda m: print(m, flush=True))
            f1s.append(f); accs.append(a); info.append({"val_macro_f1": float(vbest), "epochs": ep})
            print(f"   {key} fold {k + 1}: test F1 {f:.4f} (val {vbest:.4f}, {ep} epochs)", flush=True)
        res[key] = summarise(f1s, accs, {"window": W, "folds": info, "seconds": round(time.time() - t)})
        print(key, round(res[key]["macro_f1_mean"], 4), flush=True)
        OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
