from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"


@dataclass
class TokenLogprobs:
    tokens: list[str]
    logprobs: list[float]
    mu: list[float] | None = None  # chừa chỗ cho Min-K%++
    sigma: list[float] | None = None


class TransformersBackend:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()  # chế độ suy luận

    @torch.no_grad()  # tắt gradient cho nhẹ và nhanh
    def get_logprobs(self, text: str, add_bos: bool = True) -> TokenLogprobs:
        ids = self.tokenizer(text, return_tensors="pt").input_ids  # [1, n]

        if add_bos:
            # Qwen không tự thêm BOS. Ta chèn <|endoftext|> vào đầu
            # để token thật đầu tiên cũng có log-prob.
            bos = torch.tensor([[self.tokenizer.eos_token_id]])
            ids = torch.cat([bos, ids], dim=1)

        ids = ids.to(self.device)

        logits = self.model(ids).logits  # [1, n, V]

        # Logits tại vị trí i dự đoán token i+1.
        # Nên bỏ vị trí cuối (không có token đứng sau) và
        # đối chiếu với input_ids bỏ token đầu.
        logits = logits[:, :-1].float()  # float32 cho log_softmax chính xác
        targets = ids[:, 1:]  # [1, n-1]

        log_probs = torch.log_softmax(logits, dim=-1)  # [1, n-1, V]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, n-1]

        tokens = self.tokenizer.convert_ids_to_tokens(targets[0])
        return TokenLogprobs(tokens=tokens, logprobs=token_lp[0].tolist())
