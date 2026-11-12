"""Hash e metadati riproducibili dei run."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "interpret")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashes(paths):
    return {path.name: sha256(path) for path in sorted(paths)}


def package_versions():
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
