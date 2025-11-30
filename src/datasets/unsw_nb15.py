from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

NUMERIC_ONLY_COLUMNS = None  # will be inferred


def load_unsw_nb15(csv_paths: List[str | Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df


def prepare_unsw_for_classification(df: pd.DataFrame, label_column: str = "label") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # If the dataset uses 'attack_cat' and 'label' (0/1), prefer attack_cat when available
    y = None
    if "attack_cat" in df.columns:
        y = df["attack_cat"].astype(str)
    elif label_column in df.columns:
        y = df[label_column].astype(str)
    else:
        # Create a binary label if none provided based on 'label' presence
        raise ValueError("UNSW-NB15 CSV must include 'attack_cat' or 'label' column")

    # Keep numeric features only; drop obvious identifiers if present
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["id", "No.", "label"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
    X = df[numeric_cols].copy()

    # Replace inf and impute
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y, numeric_cols




