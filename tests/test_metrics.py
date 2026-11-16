from threept import metrics


def test_group_thresholds():
    assert metrics.RIM_MAX_FT == 8
    assert metrics.MID_MAX_FT == 20
