import math


def mean_logprob(token_logprobs: list[float]) -> float:
    """Trung bình log-prob của các token.

    Gần 0 nghĩa là mô hình đoán trúng nhiều (đáng ngờ đã thấy văn bản).
    Càng âm nghĩa là mô hình càng "ngạc nhiên".
    """
    if len(token_logprobs) == 0:
        raise ValueError("token_logprobs không được rỗng")
    return sum(token_logprobs) / len(token_logprobs)


def min_k_prob(token_logprobs: list[float], k: float = 0.2) -> float:
    """Trung bình log-prob của k% token có log-prob thấp nhất (Min-K% Prob).

    k nằm trong (0, 1]. Luôn lấy tối thiểu 1 token.
    """
    if len(token_logprobs) == 0:
        raise ValueError("token_logprobs không được rỗng")
    if not 0 < k <= 1:
        raise ValueError("k phải nằm trong khoảng (0, 1]")

    n_select = max(1, math.ceil(len(token_logprobs) * k))
    lowest = sorted(token_logprobs)[:n_select]
    return sum(lowest) / len(lowest)


def score(token_logprobs: list[float]) -> float:
    # Hàm chấm điểm chuẩn của dự án. Hiện dùng mean log-prob.
    return mean_logprob(token_logprobs)
