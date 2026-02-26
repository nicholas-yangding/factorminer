"""Central operator registry mapping names to implementations and specs.

Combines the ``OperatorSpec`` definitions from ``core.types`` with the concrete
NumPy / PyTorch function implementations from each category module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from factorminer.core.types import OPERATOR_REGISTRY as SPEC_REGISTRY
from factorminer.core.types import OperatorSpec, OperatorType

from factorminer.operators.arithmetic import ARITHMETIC_OPS
from factorminer.operators.statistical import STATISTICAL_OPS
from factorminer.operators.timeseries import TIMESERIES_OPS
from factorminer.operators.crosssectional import CROSSSECTIONAL_OPS
from factorminer.operators.smoothing import SMOOTHING_OPS
from factorminer.operators.regression import REGRESSION_OPS
from factorminer.operators.logical import LOGICAL_OPS

try:
    import torch

    _TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH = False

# ---------------------------------------------------------------------------
# Build unified registry: name -> (OperatorSpec, np_fn, torch_fn)
# ---------------------------------------------------------------------------

_ALL_IMPL_TABLES: List[Dict[str, Tuple[Callable, Callable]]] = [
    ARITHMETIC_OPS,
    STATISTICAL_OPS,
    TIMESERIES_OPS,
    CROSSSECTIONAL_OPS,
    SMOOTHING_OPS,
    REGRESSION_OPS,
    LOGICAL_OPS,
]

# Merge implementation tables
_IMPL: Dict[str, Tuple[Callable, Callable]] = {}
for table in _ALL_IMPL_TABLES:
    _IMPL.update(table)

# The full registry: name -> (spec, numpy_fn, torch_fn)
OPERATOR_REGISTRY: Dict[str, Tuple[OperatorSpec, Callable, Optional[Callable]]] = {}

for name, spec in SPEC_REGISTRY.items():
    if name in _IMPL:
        np_fn, torch_fn = _IMPL[name]
        OPERATOR_REGISTRY[name] = (spec, np_fn, torch_fn)
    else:
        # Spec exists but no implementation yet -- register with None fns
        OPERATOR_REGISTRY[name] = (spec, None, None)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_operator(name: str) -> OperatorSpec:
    """Look up an operator spec by name."""
    if name not in OPERATOR_REGISTRY:
        raise KeyError(
            f"Unknown operator '{name}'. "
            f"Available: {sorted(OPERATOR_REGISTRY.keys())}"
        )
    return OPERATOR_REGISTRY[name][0]


def get_impl(name: str, backend: str = "numpy") -> Callable:
    """Return the implementation function for a given operator and backend."""
    if name not in OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator '{name}'")
    spec, np_fn, torch_fn = OPERATOR_REGISTRY[name]
    if backend == "torch" or backend == "gpu":
        if torch_fn is None:
            raise NotImplementedError(f"No PyTorch implementation for '{name}'")
        return torch_fn
    if np_fn is None:
        raise NotImplementedError(f"No NumPy implementation for '{name}'")
    return np_fn


def execute_operator(
    name: str,
    *inputs: Any,
    params: Optional[Dict[str, Any]] = None,
    backend: str = "numpy",
) -> Union[np.ndarray, "torch.Tensor"]:
    """Execute an operator by name.

    Parameters
    ----------
    name : str
        Operator name (e.g. ``"Add"``, ``"Mean"``).
    *inputs : array-like
        Positional data inputs (1, 2, or 3 depending on arity).
    params : dict, optional
        Extra keyword parameters (e.g. ``{"window": 20}``).
    backend : str
        ``"numpy"`` or ``"torch"`` / ``"gpu"``.

    Returns
    -------
    np.ndarray or torch.Tensor
    """
    fn = get_impl(name, backend)
    kw = params or {}
    return fn(*inputs, **kw)


def list_operators(grouped: bool = True) -> Union[List[str], Dict[str, List[str]]]:
    """List all registered operator names.

    Parameters
    ----------
    grouped : bool
        If True, return a dict mapping category name -> list of op names.
        If False, return a flat sorted list.
    """
    if not grouped:
        return sorted(OPERATOR_REGISTRY.keys())

    groups: Dict[str, List[str]] = {}
    for name, (spec, _, _) in OPERATOR_REGISTRY.items():
        cat = spec.category.name
        groups.setdefault(cat, []).append(name)
    for cat in groups:
        groups[cat].sort()
    return groups


def implemented_operators() -> List[str]:
    """Return names of operators that have at least a NumPy implementation."""
    return sorted(name for name, (_, np_fn, _) in OPERATOR_REGISTRY.items() if np_fn is not None)
