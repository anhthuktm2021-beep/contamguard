import pytest

from contamguard.scoring.basic import mean_logprob, min_k_prob, score


# ------------------------ test cho mean_logprob ---------------------------
def test_mean_logprob_basic():  # thử với n = 3
    assert mean_logprob([-1.0, -2.0, -3.0]) == -2.0


def test_score_single_token():  # thử với n = 1
    assert score([-0.5]) == -0.5


def test_empty_list_raises():  # thử với mẫu rỗng (n = 0)
    with pytest.raises(ValueError):
        score([])


# ------------------------ test cho min_k_prob ---------------------------
def test_min_k_basic():
    # k=0.5 -> lấy 2 token thấp nhất: -5.0 và -4.0, trung bình -4.5
    assert min_k_prob([-0.1, -0.2, -5.0, -4.0], k=0.5) == -4.5


def test_min_k_default_k():
    # 10 token, k=0.2 -> lấy 2 token thấp nhất
    logprobs = [-1.0] * 8 + [-6.0, -4.0]
    assert min_k_prob(logprobs) == -5.0


def test_min_k_single_token():
    assert min_k_prob([-2.5]) == -2.5


def test_min_k_small_k_takes_at_least_one():
    # 3 token, k=0.01 -> 0.03 token, ép lên 1: lấy -9.0
    assert min_k_prob([-1.0, -9.0, -2.0], k=0.01) == -9.0


def test_min_k_k_equals_one_is_mean():
    data = [-1.0, -2.0, -3.0]
    assert min_k_prob(data, k=1.0) == mean_logprob(data)


def test_min_k_empty_raises():
    with pytest.raises(ValueError):
        min_k_prob([])


def test_min_k_invalid_k_raises():
    with pytest.raises(ValueError):
        min_k_prob([-1.0], k=0)
    with pytest.raises(ValueError):
        min_k_prob([-1.0], k=1.5)
