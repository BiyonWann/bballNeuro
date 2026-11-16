import shutil
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
