"""Evaluation metrics and baselines for the competitive learning network."""

import numpy as np

RIM_MAX_FT = 8
MID_MAX_FT = 20
KINDS = ["cartesian", "polar"]


def _check_kind(kind: str) -> None:
    """Raise ValueError unless kind is "cartesian" or "polar"."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def prototype_distances(weights: np.ndarray, kind: str) -> np.ndarray:
    """Return each prototype's distance from the basket in feet."""
    _check_kind(kind)
    if kind == "polar":
        return weights[:, 0].copy()
    return np.hypot(weights[:, 0], weights[:, 1])
