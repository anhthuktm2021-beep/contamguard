import pytest

from contamguard.scoring.basic import mean_logprob, score


def test_mean_logprob_basic(): # thử với n = 3
    assert mean_logprob([-1.0, -2.0, -3.0]) == -2.0


def test_score_single_token(): # thử với n = 1
    assert score([-0.5]) == -0.5


def test_empty_list_raises(): # thử với mẫu rỗng (n = 0)
    with pytest.raises(ValueError):
        score([])