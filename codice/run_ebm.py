"""Entrypoint esplicito della pipeline EBM v2.

Nessun addestramento parte senza il comando ``evaluate`` e un identificativo di run.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

import feature2
import testo
import universo
import valuta
from config import (
    CATEGORICAL, DATA_DIR, EBM_PARAMS, FEATURES, IMPACT_M_GENERAL,
    IMPACT_M_OPERATION, INNER_FOLDS, OUT_DIR, ROOT, SEED,
)
from provenienza import hashes, package_versions
from split import inner_fold_ids, temporal_splits


def prepara_dati():
    snapshot = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    current = hashes(DATA_DIR.glob("*.parquet"))
    if current != snapshot["sha256"]:
        raise AssertionError("Lo snapshot dati non coincide con dati/manifest.json")
    D = universo.definitivo(feature2.carica(DATA_DIR))
    if D["nd"] != 149:
        raise AssertionError(f"Universo inatteso: {D['nd']} documenti invece di 149")
    return D


def costruisci_matrice(D, train_mask):
    incomplete = D["TAR"].any(1)
    fold_ids = inner_fold_ids(
        train_mask,
        incomplete,
        D["P"].Codice_Pratica.to_numpy(),
        INNER_FOLDS,
        SEED,
    )
    columns = feature2.costruisci_out_of_fold(D, train_mask, fold_ids)
    normalized = np.array([testo.norma_pulita(x) for x in D["P"].Oggetto], dtype=object)
    columns["int_knn_semantico_car"] = testo.colonna_out_of_fold(
        D, train_mask, normalized, fold_ids
    )
    columns = valuta.aggiungi_identita(columns, D["npr"], D["nd"])
    missing = set(FEATURES) - set(columns)
    extra_target = set(name for name in columns if name == "int_prior_istruttore")
    if missing or extra_target:
        raise AssertionError(f"Schema feature non valido; mancanti={missing}, vietate={extra_target}")
    matrix = np.stack([columns[name] for name in FEATURES], axis=2)
    if not np.isfinite(matrix).all():
        raise ValueError("La matrice contiene NaN o infiniti")
    return matrix, fold_ids


def addestra_e_prevedi(D, matrix, train_mask, eval_mask):
    n_features = len(FEATURES)
    X_train = matrix[train_mask].reshape(-1, n_features)
    y_train = D["TAR"][train_mask].ravel().astype(int)
    deposited_train = D["DEP"][train_mask].ravel()
    keep = ~deposited_train
    feature_types = ["nominal" if name in CATEGORICAL else "continuous" for name in FEATURES]
    model = ExplainableBoostingClassifier(
        feature_names=list(FEATURES), feature_types=feature_types, **EBM_PARAMS
    )
    model.fit(X_train[keep], y_train[keep])
    scores = model.predict_proba(
        matrix[eval_mask].reshape(-1, n_features)
    )[:, 1].reshape(eval_mask.sum(), D["nd"])
    return model, scores, feature_types


def _split_for_stage(masks, stage):
    if stage == "validation":
        return masks["development"], masks["validation"]
    if stage == "test":
        return masks["final_train"], masks["test"]
    if stage == "h1_2026":
        return masks["refit_2026"], masks["h1_2026"]
    raise ValueError(stage)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate(stage, run_id, n_boot, acknowledge_test=False):
    if stage in ("test", "h1_2026") and not acknowledge_test:
        raise SystemExit(
            f"La coorte '{stage}' e' gia' stata consultata e non e' blind. "
            "Ripetere con --acknowledge-retrospective-test."
        )
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise SystemExit("run-id puo' contenere solo lettere, numeri, '-' e '_'")
    final_dir = OUT_DIR / run_id
    if final_dir.exists():
        raise SystemExit(f"Il run esiste gia' e non verra' sovrascritto: {final_dir}")
    temp_dir = OUT_DIR / f".tmp-{run_id}-{os.getpid()}"
    if temp_dir.exists():
        raise SystemExit(f"Directory temporanea gia' presente: {temp_dir}")
    temp_dir.mkdir(parents=True)
    try:
        D = prepara_dati()
        masks = temporal_splits(D["P"])
        train_mask, eval_mask = _split_for_stage(masks, stage)
        matrix, fold_ids = costruisci_matrice(D, train_mask)
        model, scores, feature_types = addestra_e_prevedi(D, matrix, train_mask, eval_mask)
        y_eval, dep_eval = D["TAR"][eval_mask], D["DEP"][eval_mask]
        point = valuta.metriche(scores, y_eval, dep_eval, seed=SEED)
        intervals = valuta.bootstrap_metriche(
            scores, y_eval, dep_eval, n_boot=n_boot, seed=SEED
        )
        metrics = {"point": point, "bootstrap_95": intervals}
        np.save(temp_dir / "scores.npy", scores)
        np.save(temp_dir / "eval_mask.npy", eval_mask)
        np.save(temp_dir / "inner_fold_ids.npy", fold_ids)
        _write_json(temp_dir / "metrics.json", metrics)
        with (temp_dir / "model.pkl").open("wb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame({
            "term": model.term_names_,
            "importance": model.term_importances(),
        }).sort_values("importance", ascending=False).to_csv(
            temp_dir / "term_importance.csv", index=False
        )

        order = valuta.top_k(scores, dep_eval, k=10, seed=SEED)
        eval_rows = np.flatnonzero(eval_mask)
        rows = []
        for local_i, practice_i in enumerate(eval_rows):
            for rank, doc_i in enumerate(order[local_i], start=1):
                rows.append({
                    "Codice_Pratica": D["P"].iloc[practice_i].Codice_Pratica,
                    "rank": rank,
                    "documento": D["DOC"][doc_i],
                    "score": float(scores[local_i, doc_i]),
                    "target": bool(y_eval[local_i, doc_i]),
                })
        pd.DataFrame(rows).to_csv(temp_dir / "top10.csv", index=False)

        source_paths = [p for p in ROOT.glob("*.py") if p.name != "__pycache__"]
        artifact_paths = [
            temp_dir / "scores.npy", temp_dir / "eval_mask.npy",
            temp_dir / "inner_fold_ids.npy", temp_dir / "metrics.json", temp_dir / "top10.csv",
            temp_dir / "model.pkl", temp_dir / "term_importance.csv",
        ]
        manifest = {
            "schema_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "retrospective_test": stage != "validation",
            "run_id": run_id,
            "seed": SEED,
            "inner_folds": INNER_FOLDS,
            "features": list(FEATURES),
            "feature_types": feature_types,
            "ebm_params": EBM_PARAMS,
            "impact_encoding": {
                "definition": (
                    "m-estimate(P(target | documento assente, contesto)) "
                    "- P(target | documento assente)"
                ),
                "m_general": IMPACT_M_GENERAL,
                "m_operation": IMPACT_M_OPERATION,
                "training_values_out_of_fold": True,
                "operation_aggregation": "maximum across individual operations",
                "change_use_non_dichiarato_value": 0.0,
            },
            "counts": {
                "train_practices": int(train_mask.sum()),
                "eval_practices": int(eval_mask.sum()),
                "eval_incomplete": int(y_eval.any(1).sum()),
                "documents": D["nd"],
            },
            "packages": package_versions(),
            "data_hashes": hashes(DATA_DIR.glob("*.parquet")),
            "source_hashes": hashes(source_paths),
            "artifact_hashes": hashes(artifact_paths),
        }
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(final_dir)
    except Exception:
        # Non si pubblica mai un run parziale. La directory temporanea resta per diagnosi.
        raise
    print(json.dumps({"run_dir": str(final_dir), "metrics": metrics}, indent=2))


def check():
    D = prepara_dati()
    masks = temporal_splits(D["P"])
    counts = {
        name: {"practices": int(mask.sum()), "incomplete": int(D["TAR"][mask].any(1).sum())}
        for name, mask in masks.items()
    }
    folds = inner_fold_ids(
        masks["development"], D["TAR"].any(1), D["P"].Codice_Pratica.to_numpy(),
        INNER_FOLDS, SEED,
    )
    print(json.dumps({
        "practices": D["npr"], "documents": D["nd"], "splits": counts,
        "development_inner_fold_sizes": [int((folds == i).sum()) for i in range(INNER_FOLDS)],
        "features": list(FEATURES),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Controlla dati e split senza addestrare il modello")
    ev = sub.add_parser("evaluate", help="Esegue un run completo e non sovrascrivibile")
    ev.add_argument("--stage", choices=("validation", "test", "h1_2026"), required=True)
    ev.add_argument("--run-id", required=True)
    ev.add_argument("--bootstrap", type=int, default=2000)
    ev.add_argument("--acknowledge-retrospective-test", action="store_true")
    args = parser.parse_args()
    if args.command == "check":
        check()
    else:
        evaluate(args.stage, args.run_id, args.bootstrap, args.acknowledge_retrospective_test)


if __name__ == "__main__":
    main()
