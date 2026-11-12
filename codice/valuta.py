"""Metriche di ranking e bootstrap a livello di pratica."""
from __future__ import annotations

import numpy as np


def aggiungi_identita(colonne: dict[str, np.ndarray], npr: int, nd: int):
    """Aggiunge esplicitamente l'identita' nominale del documento candidato."""
    out = dict(colonne)
    out["doc_identita"] = np.tile(np.arange(nd, dtype=float), (npr, 1))
    return out


def _ordine(scores, depositati, seed):
    scores = np.asarray(scores, dtype=float)
    depositati = np.asarray(depositati, dtype=bool)
    if scores.shape != depositati.shape:
        raise ValueError("scores e depositati devono avere la stessa forma")
    rng = np.random.default_rng(seed)
    # Serve solo a rompere i pareggi esatti in modo riproducibile.
    ranked = scores + rng.random(scores.shape) * 1e-12
    ranked[depositati] = -np.inf
    return np.argsort(-ranked, axis=1, kind="stable")


def metriche(scores, target, depositati, seed=0, kk=(3, 5, 10)):
    target = np.asarray(target, dtype=bool)
    ordine = _ordine(scores, depositati, seed)
    ordinati = np.take_along_axis(target, ordine, axis=1)
    incomplete = target.sum(1) > 0
    if not incomplete.any():
        raise ValueError("Nessuna pratica incompleta nel campione di valutazione")
    relevant = target[incomplete].sum(1)
    out = {"n_pratiche": int(len(target)), "n_incomplete": int(incomplete.sum())}
    for k in kk:
        hits = ordinati[:, :k].sum(1)
        out[f"R@{k}"] = float((hits[incomplete] / relevant).mean() * 100)
        out[f"P@{k}"] = float((hits[incomplete] / k).mean() * 100)
        out[f"Pnc@{k}"] = float((hits / k).mean() * 100)
        out[f"H@{k}"] = float((hits[incomplete] > 0).mean() * 100)
    cumulative = np.cumsum(ordinati[incomplete], axis=1)
    out["Rprec"] = float(
        (cumulative[np.arange(len(relevant)), relevant - 1] / relevant).mean() * 100
    )
    out["MRR"] = float(
        (1.0 / (np.argmax(ordinati[incomplete] > 0, axis=1) + 1)).mean() * 100
    )
    return out


def top_k(scores, depositati, k=10, seed=0):
    order = _ordine(scores, depositati, seed)[:, :k]
    if np.take_along_axis(np.asarray(depositati, bool), order, axis=1).any():
        raise AssertionError("Un documento gia' depositato e' entrato nella graduatoria")
    return order


def bootstrap_metriche(scores, target, depositati, n_boot=2000, seed=0):
    """IC percentile 95%, ricampionando intere pratiche e non singole coppie."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    estimates = {}
    n = len(target)
    completed = 0
    while completed < n_boot:
        idx = rng.integers(0, n, n)
        if not np.asarray(target)[idx].any():
            continue
        values = metriche(scores[idx], target[idx], depositati[idx], seed=seed)
        for name, value in values.items():
            if name.startswith("n_"):
                continue
            estimates.setdefault(name, []).append(value)
        completed += 1
    return {
        name: {
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
        for name, values in estimates.items()
    }
