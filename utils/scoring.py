from __future__ import annotations

import math
import re

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def true_prob_from_logprobs(output) -> float:
    """Sum exp(logprob) over every top-k token whose decoded text is 'true'."""
    step0 = output.outputs[0].logprobs[0]
    p_true = 0.0
    for token_id, lp in step0.items():
        tok = lp.decoded_token.strip().lower()
        if tok == "true":
            p_true += math.exp(lp.logprob)
    return p_true


def ab_probs_from_logprobs(output) -> tuple[float, float]:
    """Returns (p_A, p_B), raw probability mass on tokens 'A' and 'B'."""
    step0 = output.outputs[0].logprobs[0]
    p_a, p_b = 0.0, 0.0
    for token_id, lp in step0.items():
        tok = lp.decoded_token.strip()
        if tok == "A":
            p_a += math.exp(lp.logprob)
        elif tok == "B":
            p_b += math.exp(lp.logprob)
    return p_a, p_b


def parse_verbalized_confidence(raw_text: str) -> float:
    match = _NUMBER_RE.search(raw_text)
    if not match:
        return 0.0
    return max(0.0, min(10.0, float(match.group(1))))


def extract_atomics(options: dict) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for opt in options.values():
        for field in ("a1", "a2"):
            val = opt[field]
            if val not in seen:
                seen.add(val)
                result.append(val)
    return result
