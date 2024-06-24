from dataclasses import dataclass
from typing import List


def is_seq2seq_model(model_name: str) -> bool:
    return any([(t in model_name) for t in ["t5", "T0"]])


def is_decoder_only_model(model_name: str) -> bool:
    return any([(t in model_name) for t in ["gpt", "opt", "bloom", "llama", "mistral", "gemma"]])


def is_chat_model(model_name: str) -> bool:
    return "Instruct" in model_name or "chat" in model_name or "it" in model_name


@dataclass
class QueryCost:
    toks_prompt: int
    toks_sampled: int


@dataclass
class TokenCandidate:
    tok: str
    tok_id: int
    logit: float
    prob: float


@dataclass
class LogitInfo:
    prev_tok: str
    curr_tok: str
    curr_tok_id: int
    candidates: List[TokenCandidate]  # sorted by logit!
    ce: float


# For UI: when user inputs just comma (842) or period (1200) for purposes of
# watermark context we actually want the token "comma in text" (28723) and
# "period in text" (28725)
def fix_isolated_punctuation(name: str, tok: int) -> int:
    if name == "mistralai/Mistral-7B-Instruct-v0.1":
        if tok == 842:
            return 28723
        elif tok == 1200:
            return 28725
        else:
            return tok
    else:
        raise NotImplementedError(
            f"Fixing isolated punctuation not implemented for this model: {name}"
        )
