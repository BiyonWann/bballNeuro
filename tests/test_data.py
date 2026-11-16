import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
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
