from pytest import mark, raises
import torch
from torchmdnet.utils import make_splits


def sum_lengths(*args):
    return sum(map(len, args))


def test_make_splits_outputs():
    result = make_splits(100, 0.7, 0.2, 0.1, 1234)
    assert len(result) == 3
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert isinstance(result[2], torch.Tensor)
    assert len(result)[0] == 70
    assert len(result)[1] == 20
    assert len(result)[2] == 10
    assert sum_lengths(*result) == len(torch.unique(torch.cat(result)))
    assert max(map(max, result)) == 99
    assert min(map(min, result)) == 0


@mark.parametrize("dset_len", [5, 1000])
@mark.parametrize("ratio1", [0.0, 0.3])
@mark.parametrize("ratio2", [0.0, 0.3])
@mark.parametrize("ratio3", [0.0, 0.3])
def test_make_splits_ratios(dset_len, ratio1, ratio2, ratio3):
    train, val, test = make_splits(dset_len, ratio1, ratio2, ratio3, 1234)
    assert sum_lengths(train, val, test) <= dset_len
    assert len(train) == round(ratio1 * dset_len)
    assert len(val) == round(ratio2 * dset_len)
    # simply multiplying and rounding ratios can lead to values larger than dset_len,
    # which make_splits should account for by removing one sample from the test set
    if (
        round(ratio1 * dset_len) + round(ratio2 * dset_len) + round(ratio3 * dset_len)
        > dset_len
    ):
        assert len(test) == round(ratio3 * dset_len) - 1
    else:
        assert len(test) == round(ratio3 * dset_len)


def test_make_splits_sizes():
    assert sum_lengths(*make_splits(100, 70, 20, 10, 1234)) == 100
    assert sum_lengths(*make_splits(100, 70, 20, None, 1234)) == 100
    assert sum_lengths(*make_splits(100, 70, None, 10, 1234)) == 100
    assert sum_lengths(*make_splits(100, None, 20, 10, 1234)) == 100
    assert sum_lengths(*make_splits(100, 70, 20, 0.1, 1234)) == 100
    assert sum_lengths(*make_splits(100, 70, 20, 0.05, 1234)) == 95


def test_make_splits_errors():
    with raises(AssertionError):
        make_splits(100, 0.5, 0.5, 0.5, 1234)
    with raises(AssertionError):
        make_splits(100, 50, 50, 50, 1234)
    with raises(AssertionError):
        make_splits(100, None, None, 5, 1234)
    with raises(AssertionError):
        make_splits(100, 60, 60, None, 1234)
