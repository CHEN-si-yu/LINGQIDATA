from __future__ import annotations

import importlib
import pkgutil

_LOADED = False


def ensure_builtin_factors_loaded() -> None:
    global _LOADED

    if _LOADED:
        return

    from . import factors as factors_package

    for module_info in pkgutil.walk_packages(
        factors_package.__path__,
        prefix=f"{factors_package.__name__}.",
    ):
        if module_info.name.startswith("_"):
            continue
        # Skip package-level __init__ files
        if module_info.ispkg:
            continue
        importlib.import_module(module_info.name)

    _LOADED = True
