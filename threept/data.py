"""Load, clean and reshape NBA shot data into model inputs."""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = {"2000-01": 2000, "2008-09": 2008, "2015-16": 2015, "2024-25": 2024}
DATA_URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/shotdetail_{year}.tar.xz"
ZONES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
]
ARC_RADIUS_FT = 23.75
SHOT_COLUMNS = ["GAME_ID", "LOC_X", "LOC_Y", "SHOT_TYPE", "SHOT_ZONE_BASIC"]
# A little past the baseline, so shots right behind the rim do not wrap to +-pi.
MAX_ANGLE = np.pi / 2 + 0.25


def download_season(start_year: int, data_dir: str | Path = "data") -> Path:
    """Return the path to a season's shot CSV, downloading it only if it is not cached."""
    data_dir = Path(data_dir)
    csv_name = f"shotdetail_{start_year}.csv"
    csv_path = data_dir / csv_name
    if csv_path.exists():
        return csv_path
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / f"shotdetail_{start_year}.tar.xz"
    urllib.request.urlretrieve(DATA_URL.format(year=start_year), archive_path)
    with tarfile.open(archive_path, "r:xz") as archive:
        archive.extract(csv_name, path=data_dir, filter="data")
    archive_path.unlink()
    return csv_path


def load_season(start_year: int, data_dir: str | Path = "data") -> pd.DataFrame:
    """Return a season's shots with only the columns the project uses."""
    csv_path = download_season(start_year, data_dir)
    shots = pd.read_csv(csv_path, usecols=SHOT_COLUMNS)
    # read_csv keeps file order, so reorder to the documented column order.
    return shots[SHOT_COLUMNS]


def clean_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Return only shots in the six official zones that have both coordinates."""
    in_zone = shots["SHOT_ZONE_BASIC"].isin(ZONES)
    has_location = shots["LOC_X"].notna() & shots["LOC_Y"].notna()
    cleaned = shots[in_zone & has_location]
    return cleaned.reset_index(drop=True)


def add_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of shots with court feet and polar features added."""
    featured = shots.copy()
    featured["x_ft"] = featured["LOC_X"] / 10
    featured["y_ft"] = featured["LOC_Y"] / 10
    points_ft = featured[["x_ft", "y_ft"]].to_numpy()
    polar = court_to_inputs(points_ft, "polar")
    featured["distance_ft"] = polar[:, 0]
    featured["arc_ft"] = polar[:, 1]
    # court_to_inputs already applied the angle clip, so recover the angle from arc_ft.
    featured["angle"] = featured["arc_ft"] / ARC_RADIUS_FT
    featured["is_three"] = featured["SHOT_TYPE"].str.startswith("3PT")
    return featured


def _check_kind(kind: str) -> None:
    """Raise ValueError unless kind is "cartesian" or "polar"."""
    if kind not in ["cartesian", "polar"]:
        raise ValueError(f"kind must be 'cartesian' or 'polar', not {kind!r}")


def court_to_inputs(points_ft: np.ndarray, kind: str) -> np.ndarray:
    """Return model inputs for (n, 2) court points: [x_ft, y_ft] or [distance_ft, arc_ft]."""
    _check_kind(kind)
    points_ft = np.asarray(points_ft, dtype=float)
    if kind == "cartesian":
        return points_ft.copy()
    x_ft = points_ft[:, 0]
    y_ft = points_ft[:, 1]
    distance_ft = np.hypot(x_ft, y_ft)
    angle = np.clip(np.arctan2(x_ft, y_ft), -MAX_ANGLE, MAX_ANGLE)
    return np.column_stack([distance_ft, ARC_RADIUS_FT * angle])


def inputs_to_court(points: np.ndarray, kind: str) -> np.ndarray:
    """Return (n, 2) court points [x_ft, y_ft] for model inputs, the inverse of court_to_inputs."""
    _check_kind(kind)
    points = np.asarray(points, dtype=float)
    if kind == "cartesian":
        return points.copy()
    distance_ft = points[:, 0]
    angle = points[:, 1] / ARC_RADIUS_FT
    # The angle is measured from the +y axis, so x uses sin and y uses cos.
    return np.column_stack([distance_ft * np.sin(angle), distance_ft * np.cos(angle)])
