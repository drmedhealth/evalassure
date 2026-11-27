"""Build models from scenario specs. Supports built-ins, your own Python code, pickles and LLM providers."""
from __future__ import annotations

import importlib
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any

from .automation import BUILTIN_WORKFLOWS
from .guardrails import GUARDS
from .llm import LLM_PROVIDERS
from .ml import BUILTIN_MODELS, make_trainer


def import_object(ref: str, base_dir: Path | None = None) -> Any:
    """'package.module:attr' or 'path/to/file.py:attr'."""
    if ":" not in ref:
        raise ValueError(f"Expected 'module:attribute', got '{ref}'")
    mod_name, attr = ref.rsplit(":", 1)
    if mod_name.endswith(".py"):
        path = Path(mod_name)
        if not path.is_absolute() and base_dir:
            path = base_dir / path
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    else:
        if base_dir and str(base_dir) not in sys.path:
            sys.path.insert(0, str(base_dir))
        module = importlib.import_module(mod_name)
    return getattr(module, attr)


def build_ml_model(spec: dict[str, Any], X_train, y_train, base_dir: Path,
                   feature_names: list[str] | None = None, task: str = "classification") -> tuple[Any, Any, dict[str, Any]]:
    """Returns (fitted_model, trainer_or_None, trainer_params)."""
    params = dict(spec.get("params", {}))
    if "endpoint" in spec:  # already-deployed model behind HTTP (see models/remote.py)
        from .remote import build_remote_model

        names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        return build_remote_model(spec, X_train, y_train, names, task), None, {}
    if "builtin" in spec:
        kind = spec["builtin"]
        if kind not in BUILTIN_MODELS:
            raise KeyError(f"Unknown builtin model '{kind}'. Available: {sorted(BUILTIN_MODELS)}")
        trainer = make_trainer(kind)
        return trainer(X_train, y_train, **params), trainer, params
    if "python" in spec:
        obj = import_object(spec["python"], base_dir)
        if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "predict")):
            model = obj(**params)  # class or factory function
        else:
            model = obj  # already-built model instance
        trainer = None
        if spec.get("fit", True) and hasattr(model, "fit"):
            model.fit(X_train, y_train)
            cls = type(model)

            def trainer(X, y, sample_weight=None, **p):  # retrainable bring-your-own model
                m = cls(**{**params, **p})
                try:
                    return m.fit(X, y, sample_weight=sample_weight) if sample_weight is not None else m.fit(X, y)
                except TypeError:
                    return m.fit(X, y)
        return model, trainer, params
    if "pickle" in spec:
        path = Path(spec["pickle"])
        path = path if path.is_absolute() else base_dir / path
        # Only load pickles you trust: unpickling can execute arbitrary code.
        with open(path, "rb") as fh:
            return pickle.load(fh), None, {}
    raise ValueError("model spec needs one of: builtin | python | pickle | endpoint")


def build_llm(spec: dict[str, Any], base_dir: Path) -> Any:
    if "python" in spec:
        from .llm import CallableLLM

        obj = import_object(spec["python"], base_dir)
        model = obj if hasattr(obj, "generate") else CallableLLM(obj, name=spec.get("name", spec["python"]))
    else:
        provider = spec.get("provider", "mock")
        if provider not in LLM_PROVIDERS:
            raise KeyError(f"Unknown LLM provider '{provider}'. Available: {sorted(LLM_PROVIDERS)}")
        model = LLM_PROVIDERS[provider](**spec.get("params", {}))
    for g in spec.get("guardrails", []):
        name, cfg = (g, {}) if isinstance(g, str) else (g["name"], g.get("params", {}))
        model = GUARDS[name](model, **cfg)
    return model


def build_workflow(spec: dict[str, Any], base_dir: Path) -> Any:
    params = spec.get("params", {})
    if "python" in spec:
        obj = import_object(spec["python"], base_dir)
        return obj(**params) if isinstance(obj, type) else obj
    kind = spec.get("workflow", "invoice")
    if kind not in BUILTIN_WORKFLOWS:
        raise KeyError(f"Unknown workflow '{kind}'. Available: {sorted(BUILTIN_WORKFLOWS)}")
    return BUILTIN_WORKFLOWS[kind](**params)
