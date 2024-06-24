import functools
import os
from typing import Any

from src.models.psp.embed_sentences import embed_all, similarity
from src.models.psp.models import load_model


class PspModel:

    def __init__(self) -> None:
        # Load
        # https://github.com/martiansideofthemoon/ai-detection-paraphrases/blob/main/dipper_paraphrases/utils.py
        path = "src/models/psp/model.para.lc.100.pt"
        sp_path = "src/models/psp/paranmt.model"
        if os.path.exists(path):
            self.model = load_model(path, sp_model=sp_path)
            self.model.eval()
            embedder = functools.partial(embed_all, model=self.model, disable=True)
            self.sim_fn = functools.partial(self._get_sim_vectors, embedder)
        else:
            raise ValueError(f"No model at {path}")

    def _get_sim_vectors(self, embedder: Any, sequence: Any) -> Any:
        gen_vec = embedder(sentences=[sequence])[0]
        return gen_vec

    def get_psp(self, original: str, paraphrase: str) -> float:
        original_emb = self.sim_fn(original)
        paraphrase_emb = self.sim_fn(paraphrase)
        psp = similarity(original_emb, paraphrase_emb)
        return psp
