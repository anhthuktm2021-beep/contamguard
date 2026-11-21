def mean_logprob(token_logprobs: list[float]) -> float:
    """Trung bình log-prob của các token.

    Gần 0 nghĩa là mô hình đoán trúng nhiều (đáng ngờ đã thấy văn bản).
    Càng âm nghĩa là mô hình càng "ngạc nhiên".
    """
    if len(token_logprobs) == 0:
        raise ValueError("token_logprobs không được rỗng")
    return sum(token_logprobs) / len(token_logprobs)


def score(token_logprobs: list[float]) -> float:
    # Hàm chấm điểm chuẩn của dự án. Hiện dùng mean log-prob.
    return mean_logprob(token_logprobs)