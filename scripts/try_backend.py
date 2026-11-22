from contamguard.backends.transformers_backend import TransformersBackend
from contamguard.scoring.basic import mean_logprob, min_k_prob

backend = TransformersBackend()
result = backend.get_logprobs("The capital of France is Paris.")

for tok, lp in zip(result.tokens, result.logprobs):
    print(f"{tok!r:>15}  {lp:8.3f}")

print("mean:", mean_logprob(result.logprobs))
print("min-k:", min_k_prob(result.logprobs, k=0.2))
