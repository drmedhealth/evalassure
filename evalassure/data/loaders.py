"""Dataset loading (synthetic generators, CSV, JSONL, YAML) and stratified splitting."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.context import Split
from . import synthetic


def _to_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _encode_like(rows: list[dict[str, str]], schema: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """Encode rows with an existing feature schema (so production data lines up with training)."""
    feats = []
    header = set(rows[0])
    for name in schema:
        if name in header:
            feats.append(np.array([_to_float(r[name]) for r in rows]))
        elif "=" in name and name.split("=", 1)[0] in header:
            col, cat = name.split("=", 1)
            feats.append(np.array([1.0 if r[col] == cat else 0.0 for r in rows]))
        else:  # column missing in this file -> all missing
            feats.append(np.full(len(rows), np.nan))
    return feats, list(schema)


def load_csv(path: str | Path, target: str, sensitive: list[str] | None = None,
             drop: list[str] | None = None, schema: list[str] | None = None) -> dict[str, Any]:
    """Load a CSV. Numeric columns are used as-is; text columns are one-hot encoded.
    Sensitive columns are kept *out* of the feature matrix but used for fairness checks.
    `schema`: feature names from another file (e.g. training) to encode this file the same way."""
    sensitive, drop = sensitive or [], drop or []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    if target not in rows[0]:
        raise KeyError(f"target column '{target}' not found in {path}. Columns: {list(rows[0])}")
    missing_sens = [s for s in sensitive if s not in rows[0]]
    if missing_sens:
        raise KeyError(f"sensitive column(s) {missing_sens} not found in {path}. Columns: {list(rows[0])}")
    cols = [c for c in rows[0] if c not in {target, *sensitive, *drop}]
    feats: list[np.ndarray] = []
    names: list[str] = []
    if schema is not None:
        feats, names = _encode_like(rows, schema)
        cols = []
    for c in cols:
        raw = [r[c] for r in rows]
        num = np.array([_to_float(v) for v in raw])
        non_empty = [v for v in raw if v not in ("", None)]
        if non_empty and np.isnan(num[[i for i, v in enumerate(raw) if v not in ("", None)]]).mean() > 0.5:
            for cat in sorted(set(non_empty))[:50]:  # one-hot (capped)
                feats.append(np.array([1.0 if v == cat else 0.0 for v in raw]))
                names.append(f"{c}={cat}")
        else:
            feats.append(num)
            names.append(c)
    yraw = [r[target] for r in rows]
    yn = np.array([_to_float(v) for v in yraw])
    y = yn if not np.isnan(yn).any() else np.array(yraw)
    if y.dtype.kind == "f" and np.all(np.mod(y, 1) == 0) and len(np.unique(y)) <= 20:
        y = y.astype(int)
    return {"X": np.column_stack(feats), "y": y, "feature_names": names,
            "sensitive": {s: np.array([r[s] for r in rows]) for s in sensitive},
            "description": f"CSV dataset {Path(path).name}"}


def load_dataset(spec: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if "generator" in spec:
        gen = synthetic.GENERATORS.get(spec["generator"])
        if gen is None:
            raise KeyError(f"Unknown generator '{spec['generator']}'. Available: {sorted(synthetic.GENERATORS)}")
        data = gen(**spec.get("params", {}))
    elif "path" in spec:
        data = load_csv(base_dir / spec["path"] if not Path(spec["path"]).is_absolute() else spec["path"],
                        target=spec["target"], sensitive=spec.get("sensitive"), drop=spec.get("drop"))
        if "production_path" in spec:
            pp = Path(spec["production_path"])
            p = load_csv(pp if pp.is_absolute() else base_dir / pp, target=spec["target"],
                         sensitive=spec.get("sensitive"), drop=spec.get("drop"), schema=data["feature_names"])
            data["production"] = {"X": p["X"], "y": p["y"], "sensitive": p["sensitive"]}
    else:
        raise ValueError("data spec needs `generator:` or `path:`")
    if spec.get("inject"):
        data = synthetic.inject_issues(data, **spec["inject"])
    return data


def stratified_split(data: dict[str, Any], test_size: float = 0.2, val_size: float = 0.2,
                     seed: int = 42, stratify: bool = True) -> dict[str, Split]:
    X, y = data["X"], data["y"]
    sens = data.get("sensitive", {})
    rng = np.random.default_rng(seed)
    n = len(X)
    idx_train, idx_val, idx_test = [], [], []
    groups = [np.where(y == c)[0] for c in np.unique(y)] if stratify and len(np.unique(y)) <= 20 else [np.arange(n)]
    for g in groups:
        g = rng.permutation(g)
        nt, nv = int(round(len(g) * test_size)), int(round(len(g) * val_size))
        idx_test.extend(g[:nt]); idx_val.extend(g[nt:nt + nv]); idx_train.extend(g[nt + nv:])
    full = Split(X=X, y=y, sensitive=sens)
    out = {name: full.subset(np.array(sorted(ix), dtype=int))
           for name, ix in (("train", idx_train), ("val", idx_val), ("test", idx_test))}
    if "production" in data:
        p = data["production"]
        out["production"] = Split(X=p["X"], y=p.get("y"), sensitive=p.get("sensitive", {}))
    return out


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load test cases from .jsonl, .json or .yaml."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("//")]
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if isinstance(data, dict):
        data = data.get("cases", [])
    return list(data)
