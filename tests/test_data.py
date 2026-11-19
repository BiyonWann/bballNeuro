import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from threept import data

FIXTURE = Path(__file__).parent / "fixtures" / "shots_sample.csv"


def fail_download(url, filename=None):
    raise AssertionError(f"tests must not download: {url}")


@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlretrieve", fail_download)
    monkeypatch.setattr(urllib.request, "urlopen", fail_download)


@pytest.fixture
def cached_dir(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "shotdetail_2000.csv")
    return tmp_path


def test_constants():
    assert data.SEASONS == {"2000-01": 2000, "2008-09": 2008, "2015-16": 2015, "2024-25": 2024}
    assert data.DATA_URL == (
        "https://github.com/shufinskiy/nba_data/raw/main/datasets/shotdetail_{year}.tar.xz"
    )
    assert data.ZONES == [
        "Restricted Area",
        "In The Paint (Non-RA)",
        "Mid-Range",
        "Left Corner 3",
        "Right Corner 3",
        "Above the Break 3",
    ]
    assert data.ARC_RADIUS_FT == 23.75
    assert data.MAX_ANGLE == np.pi / 2 + 0.25


def test_download_season_returns_cached_file(cached_dir, no_network):
    path = data.download_season(2000, data_dir=cached_dir)

    assert path == cached_dir / "shotdetail_2000.csv"


def test_download_season_accepts_str_dir(cached_dir, no_network):
    path = data.download_season(2000, data_dir=str(cached_dir))

    assert path == cached_dir / "shotdetail_2000.csv"


@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    shutil.copy(FIXTURE, source_dir / "shotdetail_2000.csv")
    archive = source_dir / "shotdetail_2000.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(source_dir / "shotdetail_2000.csv", arcname="shotdetail_2000.csv")
    requested_urls = []

    def copy_archive(url, filename=None):
        requested_urls.append(url)
        shutil.copy(archive, filename)

    monkeypatch.setattr(urllib.request, "urlretrieve", copy_archive)
    return requested_urls


def test_download_season_downloads_and_extracts(tmp_path, fake_download):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = data.download_season(2000, data_dir=data_dir)

    assert fake_download == [data.DATA_URL.format(year=2000)]
    assert path == data_dir / "shotdetail_2000.csv"
    assert path.read_text() == FIXTURE.read_text()


def test_download_season_creates_data_dir(tmp_path, fake_download):
    data_dir = tmp_path / "new"

    path = data.download_season(2000, data_dir=data_dir)

    assert data_dir.is_dir()
    assert path.read_text() == FIXTURE.read_text()


def test_load_season_reads_five_columns(cached_dir, no_network):
    shots = data.load_season(2000, data_dir=cached_dir)

    assert list(shots.columns) == ["GAME_ID", "LOC_X", "LOC_Y", "SHOT_TYPE", "SHOT_ZONE_BASIC"]


def test_load_season_row_count(cached_dir, no_network):
    shots = data.load_season(2000, data_dir=cached_dir)

    assert len(shots) == 40


@pytest.fixture
def loaded_fixture(cached_dir, no_network):
    return data.load_season(2000, data_dir=cached_dir)


def test_clean_shots_drops_backcourt(loaded_fixture):
    shots = data.clean_shots(loaded_fixture)

    assert "Backcourt" not in set(shots["SHOT_ZONE_BASIC"])


def test_clean_shots_drops_missing_coordinates():
    shots = pd.DataFrame(
        {
            "GAME_ID": [1, 1, 1],
            "LOC_X": [10.0, np.nan, 20.0],
            "LOC_Y": [50.0, 60.0, np.nan],
            "SHOT_TYPE": ["2PT Field Goal", "2PT Field Goal", "2PT Field Goal"],
            "SHOT_ZONE_BASIC": ["Mid-Range", "Mid-Range", "Mid-Range"],
        }
    )

    cleaned = data.clean_shots(shots)

    assert list(cleaned["LOC_X"]) == [10.0]
    assert list(cleaned["LOC_Y"]) == [50.0]


def test_clean_shots_keeps_six_zones(loaded_fixture):
    shots = data.clean_shots(loaded_fixture)

    assert set(shots["SHOT_ZONE_BASIC"]) == set(data.ZONES)
    assert len(shots) == 38


def test_clean_shots_drops_unknown_zone():
    shots = pd.DataFrame(
        {
            "GAME_ID": [1, 1],
            "LOC_X": [10.0, 20.0],
            "LOC_Y": [50.0, 60.0],
            "SHOT_TYPE": ["2PT Field Goal", "2PT Field Goal"],
            "SHOT_ZONE_BASIC": ["Mid-Range", "Unknown"],
        }
    )

    cleaned = data.clean_shots(shots)

    assert list(cleaned["SHOT_ZONE_BASIC"]) == ["Mid-Range"]


def test_clean_shots_resets_index(loaded_fixture):
    shots = data.clean_shots(loaded_fixture)

    assert list(shots.index) == list(range(38))


def test_court_to_inputs_cartesian():
    points = np.array([[3, 4], [-22, 0]])

    inputs = data.court_to_inputs(points, "cartesian")

    assert inputs.dtype == float
    assert inputs.tolist() == [[3.0, 4.0], [-22.0, 0.0]]


def test_court_to_inputs_polar():
    points = np.array([[0.0, 23.75], [22.0, 0.0], [3.0, 4.0]])

    inputs = data.court_to_inputs(points, "polar")

    expected = [[23.75, 0.0], [22.0, 23.75 * np.pi / 2], [5.0, 23.75 * np.arctan2(3, 4)]]
    assert inputs == pytest.approx(np.array(expected))


@pytest.mark.parametrize("x_ft, sign", [(0.5, 1), (-0.5, -1)])
def test_court_to_inputs_polar_clips(x_ft, sign):
    points = np.array([[x_ft, -3.0]])

    inputs = data.court_to_inputs(points, "polar")

    assert inputs[0, 0] == pytest.approx(np.hypot(0.5, 3.0))
    assert inputs[0, 1] == pytest.approx(sign * 23.75 * data.MAX_ANGLE)


def test_inputs_to_court_polar():
    inputs = np.array([[23.75, 0.0], [22.0, 23.75 * np.pi / 2]])

    points = data.inputs_to_court(inputs, "polar")

    assert points == pytest.approx(np.array([[0.0, 23.75], [22.0, 0.0]]))


@pytest.mark.parametrize("kind", ["cartesian", "polar"])
def test_round_trip(kind):
    points = np.array([[0.0, 23.75], [-22.0, 0.0], [22.0, 0.0], [3.0, 4.0], [-10.0, 15.0]])

    round_trip = data.inputs_to_court(data.court_to_inputs(points, kind), kind)

    assert round_trip == pytest.approx(points)


def shots_at(locations):
    n_shots = len(locations)
    return pd.DataFrame(
        {
            "GAME_ID": [1] * n_shots,
            "LOC_X": [location[0] for location in locations],
            "LOC_Y": [location[1] for location in locations],
            "SHOT_TYPE": ["2PT Field Goal"] * n_shots,
            "SHOT_ZONE_BASIC": ["Mid-Range"] * n_shots,
        }
    )


def test_add_features_feet():
    shots = data.add_features(shots_at([(30, 40)]))

    assert shots.loc[0, "x_ft"] == pytest.approx(3.0)
    assert shots.loc[0, "y_ft"] == pytest.approx(4.0)
    assert shots.loc[0, "distance_ft"] == pytest.approx(5.0)


def test_add_features_at_basket():
    shots = data.add_features(shots_at([(0, 0)]))

    assert shots.loc[0, "distance_ft"] == 0.0


@pytest.mark.parametrize(
    "location, distance_ft, angle",
    [((0, 237.5), 23.75, 0.0), ((220, 0), 22.0, np.pi / 2), ((-220, 0), 22.0, -np.pi / 2)],
)
def test_add_features_angle(location, distance_ft, angle):
    shots = data.add_features(shots_at([location]))

    assert shots.loc[0, "distance_ft"] == pytest.approx(distance_ft)
    assert shots.loc[0, "angle"] == pytest.approx(angle)
    assert shots.loc[0, "arc_ft"] == pytest.approx(23.75 * angle)


@pytest.mark.parametrize("location, sign", [((5, -30), 1), ((-5, -30), -1)])
def test_add_features_clips_angle_behind_rim(location, sign):
    shots = data.add_features(shots_at([location]))

    assert shots.loc[0, "angle"] == pytest.approx(sign * data.MAX_ANGLE)
    assert shots.loc[0, "arc_ft"] == pytest.approx(sign * 23.75 * data.MAX_ANGLE)


def test_add_features_keeps_angle_inside_clip():
    shots = data.add_features(shots_at([(220, -220 * np.tan(0.1))]))

    assert shots.loc[0, "angle"] == pytest.approx(np.pi / 2 + 0.1)


def test_add_features_clip_keeps_distance():
    shots = data.add_features(shots_at([(0, -30)]))

    assert shots.loc[0, "distance_ft"] == pytest.approx(3.0)
    assert shots.loc[0, "angle"] == pytest.approx(data.MAX_ANGLE)


def test_add_features_matches_court_to_inputs():
    shots = data.add_features(shots_at([(0, 237.5), (-230, -26), (26, -10), (157, 203), (5, 80)]))

    points_ft = shots[["x_ft", "y_ft"]].to_numpy()
    expected = data.court_to_inputs(points_ft, "polar")
    assert shots[["distance_ft", "arc_ft"]].to_numpy() == pytest.approx(expected)


def test_add_features_is_three():
    shots = shots_at([(0, 250), (0, 100)])
    shots["SHOT_TYPE"] = ["3PT Field Goal", "2PT Field Goal"]

    featured = data.add_features(shots)

    assert list(featured["is_three"]) == [True, False]


def test_add_features_does_not_mutate():
    shots = shots_at([(30, 40)])
    original = shots.copy()

    featured = data.add_features(shots)

    pd.testing.assert_frame_equal(shots, original)
    new_columns = ["x_ft", "y_ft", "distance_ft", "angle", "arc_ft", "is_three"]
    assert set(featured.columns) == set(original.columns) | set(new_columns)


@pytest.fixture
def xy_only():
    return pd.DataFrame({"x_ft": [3.0, -22.0, 0.0], "y_ft": [4.0, 0.0, 23.75]})


def test_get_inputs_cartesian(xy_only):
    inputs = data.get_inputs(xy_only, "cartesian")

    assert inputs.shape == (3, 2)
    assert inputs.dtype == float
    assert inputs.tolist() == [[3.0, 4.0], [-22.0, 0.0], [0.0, 23.75]]


def test_get_inputs_polar(xy_only):
    inputs = data.get_inputs(xy_only, "polar")

    expected = data.court_to_inputs(xy_only[["x_ft", "y_ft"]].to_numpy(), "polar")
    assert inputs == pytest.approx(expected)


def test_get_inputs_default_kind_is_cartesian(xy_only):
    inputs = data.get_inputs(xy_only)

    assert inputs.tolist() == [[3.0, 4.0], [-22.0, 0.0], [0.0, 23.75]]


@pytest.mark.parametrize("function", [data.court_to_inputs, data.inputs_to_court])
def test_bad_kind_raises(function):
    with pytest.raises(ValueError):
        function(np.array([[1.0, 2.0]]), "spherical")


def test_get_inputs_bad_kind_raises(xy_only):
    with pytest.raises(ValueError):
        data.get_inputs(xy_only, "spherical")


def game_shots(n_games):
    game_ids = []
    for game_id in range(1, n_games + 1):
        game_ids = game_ids + [game_id, game_id, game_id]
    return pd.DataFrame({"GAME_ID": game_ids, "shot_id": list(range(len(game_ids)))})


def test_split_no_shared_games():
    train, test = data.split_by_game(game_shots(20), test_fraction=0.2, seed=0)

    assert set(train["GAME_ID"]) & set(test["GAME_ID"]) == set()
    assert list(train.index) == list(range(len(train)))
    assert list(test.index) == list(range(len(test)))


def test_split_keeps_every_row():
    train, test = data.split_by_game(game_shots(20), test_fraction=0.2, seed=0)

    shot_ids = list(train["shot_id"]) + list(test["shot_id"])
    assert sorted(shot_ids) == list(range(60))


@pytest.mark.parametrize(
    "n_games, test_fraction, n_test_games", [(20, 0.2, 4), (20, 0.5, 10), (7, 0.2, 1)]
)
def test_split_test_game_count(n_games, test_fraction, n_test_games):
    train, test = data.split_by_game(game_shots(n_games), test_fraction=test_fraction, seed=0)

    assert test["GAME_ID"].nunique() == n_test_games
    assert len(test) == 3 * n_test_games
    assert list(test.index) == list(range(3 * n_test_games))


def test_split_zero_fraction():
    train, test = data.split_by_game(game_shots(20), test_fraction=0.0, seed=0)

    assert len(test) == 0
    assert len(train) == 60


def test_split_same_seed():
    _, first = data.split_by_game(game_shots(20), seed=3)
    _, second = data.split_by_game(game_shots(20), seed=3)

    assert set(first["GAME_ID"]) == set(second["GAME_ID"])


def test_split_different_seed():
    _, first = data.split_by_game(game_shots(20), seed=0)
    _, second = data.split_by_game(game_shots(20), seed=1)

    assert set(first["GAME_ID"]) != set(second["GAME_ID"])


def test_sample_shots_size():
    sample = data.sample_shots(game_shots(20), n=10, seed=0)

    assert len(sample) == 10
    assert list(sample.index) == list(range(10))
    assert sample["shot_id"].nunique() == 10


def test_sample_shots_n_too_large():
    sample = data.sample_shots(game_shots(20), n=100, seed=0)

    assert list(sample.index) == list(range(60))
    assert sorted(sample["shot_id"]) == list(range(60))


def test_sample_shots_n_equal():
    sample = data.sample_shots(game_shots(20), n=60, seed=0)

    assert sorted(sample["shot_id"]) == list(range(60))


def test_sample_shots_seed():
    first = data.sample_shots(game_shots(20), n=10, seed=0)
    second = data.sample_shots(game_shots(20), n=10, seed=0)
    other = data.sample_shots(game_shots(20), n=10, seed=1)

    assert list(first["shot_id"]) == list(second["shot_id"])
    assert list(first["shot_id"]) != list(other["shot_id"])


def test_prepare_season_runs_offline(cached_dir, no_network):
    train, test = data.prepare_season(2000, seed=0, n_train=10, data_dir=cached_dir)

    assert len(train) == 10
    assert "Backcourt" not in set(train["SHOT_ZONE_BASIC"]) | set(test["SHOT_ZONE_BASIC"])
    assert train["LOC_X"].notna().all()
    for column in ["x_ft", "y_ft", "distance_ft", "angle", "arc_ft", "is_three"]:
        assert column in train.columns
        assert column in test.columns


def test_prepare_season_test_is_whole_games(cached_dir, loaded_fixture):
    clean = data.clean_shots(loaded_fixture)

    train, test = data.prepare_season(2000, seed=0, n_train=10, data_dir=cached_dir)

    assert set(train["GAME_ID"]) & set(test["GAME_ID"]) == set()
    assert test["GAME_ID"].nunique() == 1
    assert len(test) == clean["GAME_ID"].isin(set(test["GAME_ID"])).sum()


def test_prepare_season_small_train(cached_dir, no_network):
    train, test = data.prepare_season(2000, seed=0, n_train=20000, data_dir=cached_dir)

    assert len(train) + len(test) == 38


def test_prepare_season_same_seed(cached_dir, no_network):
    first_train, first_test = data.prepare_season(2000, seed=0, n_train=10, data_dir=cached_dir)
    second_train, second_test = data.prepare_season(2000, seed=0, n_train=10, data_dir=cached_dir)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_prepare_season_different_seed(cached_dir, no_network):
    _, first_test = data.prepare_season(2000, seed=0, n_train=20000, data_dir=cached_dir)
    _, second_test = data.prepare_season(2000, seed=1, n_train=20000, data_dir=cached_dir)

    assert set(first_test["GAME_ID"]) == {20000001}
    assert set(second_test["GAME_ID"]) == {22400001}


@pytest.mark.parametrize(
    "seed, loc_x, loc_y",
    [
        (0, [236, 10, 26, 223, 45], [72, 13, -10, 11, -1]),
        (1, [37, -7, -58, 0, 157], [12, 19, 28, 0, 212]),
    ],
)
def test_prepare_season_samples_with_seed(cached_dir, no_network, seed, loc_x, loc_y):
    train, _ = data.prepare_season(2000, seed=seed, n_train=5, data_dir=cached_dir)

    assert list(train["LOC_X"]) == loc_x
    assert list(train["LOC_Y"]) == loc_y


def test_prepare_season_test_fraction(tmp_path, no_network):
    shots = game_shots(20)
    n_shots = len(shots)
    shots["LOC_X"] = [10] * n_shots
    shots["LOC_Y"] = [150] * n_shots
    shots["SHOT_TYPE"] = ["2PT Field Goal"] * n_shots
    shots["SHOT_ZONE_BASIC"] = ["Mid-Range"] * n_shots
    shots.to_csv(tmp_path / "shotdetail_2000.csv", index=False)

    _, test = data.prepare_season(2000, seed=0, data_dir=tmp_path)

    assert test["GAME_ID"].nunique() == 4
