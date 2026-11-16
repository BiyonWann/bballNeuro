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


@pytest.mark.parametrize("function_name", ["court_to_inputs", "inputs_to_court"])
def test_bad_kind_raises(function_name):
    function = getattr(data, function_name)

    with pytest.raises(ValueError):
        function(np.array([[1.0, 2.0]]), "spherical")
