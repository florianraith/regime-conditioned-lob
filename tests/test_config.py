import pytest

from lobgen.common import config_sample6m as cfg

days = cfg.available_days()
pytestmark = pytest.mark.skipif(not days, reason="raw data not found")


def test_usable_days():
    assert len(days) == 121
    assert days == cfg.available_days("vstoxx")


def test_split_a_counts():
    s = cfg.split_a_days(days)
    counts = {k: len(v) for k, v in s.items()}
    assert counts == {"train_a": 60, "stress": 14, "embargo": 4, "train_b": 19, "val": 10, "test": 14, "train": 79}


@pytest.mark.parametrize("fold,train,test", [(0, 37, 18), (1, 56, 19), (2, 76, 20), (3, 97, 18)])
def test_split_b_counts(fold, train, test):
    f = cfg.SPLIT_B[fold]
    assert len(f["train"].days(days)) == train
    assert len(f["val"].days(days)) == 5
    assert len(f["test"].days(days)) == test
    assert f["train"].end < f["val"].start <= f["val"].end < f["test"].start


def test_split_c_counts():
    assert [len(s.days(days)) for s in cfg.SPLIT_C.values()] == [16, 1, 2, 1]
