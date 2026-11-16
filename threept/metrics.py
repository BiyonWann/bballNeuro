"""Evaluation metrics and baselines for the competitive learning network."""

import numpy as np

RIM_MAX_FT = 8
MID_MAX_FT = 20
KINDS = ["cartesian", "polar"]
ARC_RADIUS_FT = 23.75


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


def prototype_angles(weights: np.ndarray, kind: str) -> np.ndarray:
    """Return each prototype's angle in radians, 0 toward the top of the key."""
    _check_kind(kind)
    if kind == "polar":
        return weights[:, 1] / ARC_RADIUS_FT
    return np.arctan2(weights[:, 0], weights[:, 1])


def cell_allocation(distances: np.ndarray) -> dict:
    """Return how many cells sit at the rim, in the mid-range and beyond 20 ft."""
    is_rim = distances < RIM_MAX_FT
    is_beyond = distances > MID_MAX_FT
    is_mid = ~is_rim & ~is_beyond
    return {
        "rim": int(is_rim.sum()),
        "mid_range": int(is_mid.sum()),
        "beyond_20": int(is_beyond.sum()),
    }
