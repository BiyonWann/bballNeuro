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
# a little past the baseline so shots behind the rim do not wrap
MAX_ANGLE = np.pi / 2 + 0.25


def download_season(start_year: int, data_dir: str | Path = "data") -> Path:
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
    csv_path = download_season(start_year, data_dir)
    shots = pd.read_csv(csv_path, usecols=SHOT_COLUMNS)
    # read_csv keeps file order so put the columns in our order
    return shots[SHOT_COLUMNS]


def clean_shots(shots: pd.DataFrame) -> pd.DataFrame:
    in_zone = shots["SHOT_ZONE_BASIC"].isin(ZONES)
    has_location = shots["LOC_X"].notna() & shots["LOC_Y"].notna()
    cleaned = shots[in_zone & has_location]
    return cleaned.reset_index(drop=True)


def add_features(shots: pd.DataFrame) -> pd.DataFrame:
    featured = shots.copy()
    featured["x_ft"] = featured["LOC_X"] / 10
    featured["y_ft"] = featured["LOC_Y"] / 10
    points_ft = featured[["x_ft", "y_ft"]].to_numpy()
    polar = court_to_inputs(points_ft, "polar")
    featured["distance_ft"] = polar[:, 0]
    featured["arc_ft"] = polar[:, 1]
    # arc_ft already has the clip so the angle comes from it
    featured["angle"] = featured["arc_ft"] / ARC_RADIUS_FT
    featured["is_three"] = featured["SHOT_TYPE"].str.startswith("3PT")
    return featured


def _check_kind(kind: str) -> None:
    if kind not in ["cartesian", "polar"]:
        raise ValueError(f"kind must be 'cartesian' or 'polar', not '{kind}'")


def court_to_inputs(points_ft: np.ndarray, kind: str) -> np.ndarray:
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
    _check_kind(kind)
    points = np.asarray(points, dtype=float)
    if kind == "cartesian":
        return points.copy()
    distance_ft = points[:, 0]
    angle = points[:, 1] / ARC_RADIUS_FT
    # angle starts at the top of the key so x uses sin
    return np.column_stack([distance_ft * np.sin(angle), distance_ft * np.cos(angle)])


def split_by_game(
    shots: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    game_ids = shots["GAME_ID"].unique()
    n_test_games = round(test_fraction * len(game_ids))
    test_games = rng.choice(game_ids, size=n_test_games, replace=False)
    is_test = shots["GAME_ID"].isin(test_games)
    train = shots[~is_test].reset_index(drop=True)
    test = shots[is_test].reset_index(drop=True)
    return train, test


def sample_shots(shots: pd.DataFrame, n: int = 20000, seed: int = 0) -> pd.DataFrame:
    n_sampled = min(n, len(shots))
    rng = np.random.default_rng(seed)
    sample = shots.sample(n=n_sampled, random_state=rng)
    return sample.reset_index(drop=True)


def get_inputs(shots: pd.DataFrame, kind: str = "cartesian") -> np.ndarray:
    points_ft = shots[["x_ft", "y_ft"]].to_numpy(dtype=float)
    return court_to_inputs(points_ft, kind)


def prepare_season(
    start_year: int, seed: int = 0, n_train: int = 20000, data_dir: str | Path = "data"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = load_season(start_year, data_dir)
    shots = clean_shots(shots)
    shots = add_features(shots)
    train, test = split_by_game(shots, seed=seed)
    train = sample_shots(train, n=n_train, seed=seed)
    return train, test
