"""Verifica read-only di hash, schema, artefatti e metriche di un run pubblicato."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import feature2
import universo
import valuta
from config import DATA_DIR, FEATURES, ROOT, SEED
from provenienza import hashes


def verify(run_dir: Path):
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = (
        "scores.npy", "eval_mask.npy", "inner_fold_ids.npy", "metrics.json", "top10.csv",
        "model.pkl", "term_importance.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise AssertionError(f"Artefatti mancanti: {missing}")
    if manifest["features"] != list(FEATURES):
        raise AssertionError("Lo schema feature del run non coincide con il codice corrente")
    if hashes(DATA_DIR.glob("*.parquet")) != manifest["data_hashes"]:
        raise AssertionError("Gli hash dei dati non coincidono")
    current_sources = hashes(p for p in ROOT.glob("*.py") if p.name != "__pycache__")
    if current_sources != manifest["source_hashes"]:
        raise AssertionError("Gli hash del codice non coincidono")
    artifact_paths = [run_dir / name for name in required]
    if hashes(artifact_paths) != manifest["artifact_hashes"]:
        raise AssertionError("Uno o piu' artefatti sono stati modificati")

    D = universo.definitivo(feature2.carica(DATA_DIR))
    scores = np.load(run_dir / "scores.npy", allow_pickle=False)
    eval_mask = np.load(run_dir / "eval_mask.npy", allow_pickle=False)
    if scores.shape != (int(eval_mask.sum()), D["nd"]):
        raise AssertionError("Forma dei punteggi non coerente")
    recomputed = valuta.metriche(
        scores, D["TAR"][eval_mask], D["DEP"][eval_mask], seed=SEED
    )
    recorded = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["point"]
    for key in recomputed:
        if not np.isclose(recomputed[key], recorded[key], rtol=0, atol=1e-12):
            raise AssertionError(f"Metrica non riproducibile: {key}")
    print(json.dumps({"status": "ok", "run_dir": str(run_dir)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    verify(args.run_dir.resolve())


if __name__ == "__main__":
    main()
