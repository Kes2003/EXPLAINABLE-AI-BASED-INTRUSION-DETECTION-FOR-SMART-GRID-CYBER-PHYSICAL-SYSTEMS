# scripts/msu_improve_features.py
"""
Feature engineering for MSU/ORNL under event-disjoint evaluation.

Every feature here is computed from a row and the rows recorded BEFORE it in
the same recording file (causal), so a deployed detector could compute it on
a live stream. Nothing uses labels, row position, event boundaries or any
statistic fitted on test recordings.

Feature groups:
  raw       the 128 original columns
  phys      inter-relay differences: wrapped phase-angle differences between
            relays for the same phasor, magnitude ratios between relays,
            per-relay V/I ratio, and zero/negative-sequence-like imbalance
            of the three phase magnitudes
  temporal  causal lag-1 / lag-k differences and trailing rolling mean, std,
            min, max over the previous w rows of the same file
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RELAYS = ["R1", "R2", "R3", "R4"]


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def phys_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for k in range(1, 13):
        kind = "V" if k in (1, 2, 3, 7, 8, 9) else "I"
        pa = [f"{r}-PA{k}:{kind}H" for r in RELAYS]
        pm = [f"{r}-PM{k}:{kind}" for r in RELAYS]
        for i in range(4):
            for j in range(i + 1, 4):
                out[f"dPA{k}_{RELAYS[i]}{RELAYS[j]}"] = wrap_deg(df[pa[i]].values - df[pa[j]].values)
                out[f"rPM{k}_{RELAYS[i]}{RELAYS[j]}"] = df[pm[i]].values / (np.abs(df[pm[j]].values) + 1e-6)
    for r in RELAYS:
        # within-relay angle differences between phases (A-B, B-C) for V and I
        for a, b in [(1, 2), (2, 3), (4, 5), (5, 6)]:
            ka = "V" if a <= 3 else "I"
            out[f"{r}_dPA{a}{b}"] = wrap_deg(df[f"{r}-PA{a}:{ka}H"].values - df[f"{r}-PA{b}:{ka}H"].values)
        # voltage-current angle (power-factor angle) per phase
        for v, i in [(1, 4), (2, 5), (3, 6)]:
            out[f"{r}_pf{v}"] = wrap_deg(df[f"{r}-PA{v}:VH"].values - df[f"{r}-PA{i}:IH"].values)
            out[f"{r}_Z{v}"] = df[f"{r}-PM{v}:V"].values / (np.abs(df[f"{r}-PM{i}:I"].values) + 1e-6)
        # three-phase magnitude imbalance (std / mean) for V and I
        V = np.stack([df[f"{r}-PM{k}:V"].values for k in (1, 2, 3)], 1)
        I = np.stack([df[f"{r}-PM{k}:I"].values for k in (4, 5, 6)], 1)
        out[f"{r}_Vimb"] = V.std(1) / (np.abs(V.mean(1)) + 1e-6)
        out[f"{r}_Iimb"] = I.std(1) / (np.abs(I.mean(1)) + 1e-6)
        out[f"{r}_Vmin_rel"] = V.min(1) / (np.abs(V.mean(1)) + 1e-6)
        out[f"{r}_Imax_rel"] = I.max(1) / (np.abs(I.mean(1)) + 1e-6)
    return pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def temporal_features(df: pd.DataFrame, cols, key="_event", windows=(5, 20), lags=(1, 5)) -> pd.DataFrame:
    """Causal features within each stream given by `key`; the first rows of a
    stream use whatever history exists (lag differences are 0 there)."""
    g = df.groupby(key, sort=False)[cols]
    parts = []
    for L in lags:
        parts.append((df[cols] - g.shift(L)).add_suffix(f"_d{L}"))
    for w in windows:
        r = g.rolling(w, min_periods=1)
        mean = r.mean().reset_index(level=0, drop=True).sort_index()
        std = r.std().reset_index(level=0, drop=True).sort_index()
        mn = r.min().reset_index(level=0, drop=True).sort_index()
        mx = r.max().reset_index(level=0, drop=True).sort_index()
        parts += [(df[cols] - mean).add_suffix(f"_dev{w}"), std.add_suffix(f"_std{w}"),
                  (mx - mn).add_suffix(f"_rng{w}")]
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def expanding_features(df: pd.DataFrame, cols, key="_event") -> pd.DataFrame:
    """Causal statistics over everything recorded so far in the stream given by
    `key` (from its first row up to and including the current row)."""
    g = df.groupby(key, sort=False)[cols]
    first = g.transform("first")
    cmax, cmin = g.cummax(), g.cummin()
    n = df.groupby(key, sort=False).cumcount() + 1
    csum = g.cumsum()
    csq = (df[cols] ** 2).groupby(df[key], sort=False).cumsum()
    mean = csum.div(n, axis=0)
    std = np.sqrt(np.maximum(csq.div(n, axis=0) - mean ** 2, 0.0))
    parts = [(df[cols] - first).add_suffix("_dfirst"), (cmax - first).add_suffix("_cmaxd"),
             (first - cmin).add_suffix("_cmind"), (df[cols] - mean).add_suffix("_dcmean"),
             std.add_suffix("_cstd")]
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
