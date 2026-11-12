"""Unica fonte di verita' per split temporali e fold interni."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import DEV_END, VALIDATION_END, TEST_END, H1_2026_END


def temporal_splits(P: pd.DataFrame) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(P["Data_Presentazione"], errors="raise")
    masks = {
        "development": (dates < DEV_END).to_numpy(),
        "validation": ((dates >= DEV_END) & (dates < VALIDATION_END)).to_numpy(),
        "final_train": (dates < VALIDATION_END).to_numpy(),
        "test": ((dates >= VALIDATION_END) & (dates < TEST_END)).to_numpy(),
        "refit_2026": (dates < TEST_END).to_numpy(),
        "h1_2026": ((dates >= TEST_END) & (dates < H1_2026_END)).to_numpy(),
        "excluded_future": (dates >= H1_2026_END).to_numpy(),
    }
    if np.any(masks["development"] & masks["validation"]):
        raise AssertionError("Gli split temporali si sovrappongono")
    if np.any(masks["final_train"] & masks["test"]):
        raise AssertionError("Train finale e test si sovrappongono")
    if np.any(masks["refit_2026"] & masks["h1_2026"]):
        raise AssertionError("Rifit pre-2026 e valutazione H1 2026 si sovrappongono")
    return masks


def inner_fold_ids(
    train_mask: np.ndarray,
    incomplete: np.ndarray,
    practice_ids: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Assegna ogni pratica di train a un fold, stratificando per incompletezza.

    Code_Practice e' la chiave: nessun raggruppamento per particella viene applicato.
    Lo stesso vettore viene passato a tutte le feature supervisionate, TF-IDF incluso.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    incomplete = np.asarray(incomplete, dtype=int)
    practice_ids = np.asarray(practice_ids)
    if not (len(train_mask) == len(incomplete) == len(practice_ids)):
        raise ValueError("Lunghezze non coerenti nella costruzione dei fold")
    if len(np.unique(practice_ids)) != len(practice_ids):
        raise ValueError("Code_Practice deve identificare univocamente ogni riga")
    idx = np.flatnonzero(train_mask)
    if len(idx) < n_splits:
        raise ValueError("Numero di pratiche insufficiente per i fold richiesti")
    counts = np.bincount(incomplete[idx], minlength=2)
    if counts.min() < n_splits:
        raise ValueError("Ogni classe deve avere almeno n_splits pratiche")
    fold_ids = np.full(len(train_mask), -1, dtype=np.int16)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, held) in enumerate(cv.split(idx, incomplete[idx])):
        fold_ids[idx[held]] = fold
    if np.any(fold_ids[idx] < 0):
        raise AssertionError("Assegnazione incompleta dei fold interni")
    return fold_ids
