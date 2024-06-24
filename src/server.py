from typing import Any, List

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.config import MetaConfig, ServerConfig
from src.models import HfModel
from src.watermarks import BaseWatermark, get_watermark


class ForceLongProcessor(LogitsProcessor):
    def __init__(
        self,
        *args: Any,
        eos_token: int,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.eos_token = eos_token

    def __call__(self, input_ids: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        for b in range(input_ids.shape[0]):
            if input_ids.shape[1] < 600:
                logits[b][self.eos_token] = -1e7
        return logits


class Server:
    def __init__(self, meta_cfg: MetaConfig, server_cfg: ServerConfig) -> None:
        self.seed, self.device, self.out_root_dir = (
            meta_cfg.seed,
            meta_cfg.device,
            meta_cfg.out_root_dir,
        )
        self.model = HfModel(meta_cfg, server_cfg.model)
        self.cfg = server_cfg

        # You can have one or more watermarks
        self.watermarks = []
        for seeding_scheme in self.cfg.watermark.generation.seeding_scheme.strip().split(";"):
            cfg_watermark = self.cfg.watermark.model_copy(deep=True)
            cfg_watermark.generation.seeding_scheme = seeding_scheme
            watermark: BaseWatermark = get_watermark(
                meta_cfg, cfg_watermark, self.model.tokenizer, self.cfg.model.name
            )
            self.watermarks.append(watermark)

    def generate(
        self, prompts: List[str], disable_watermark: bool = False, return_model_inputs: bool = False
    ) -> Any:
        if disable_watermark:
            return self.model.generate(
                prompts, LogitsProcessorList([]), return_model_inputs=return_model_inputs
            )
        else:
            # Pick a random watermark
            watermark = self.watermarks[np.random.randint(len(self.watermarks))]
            return self.model.generate(
                prompts,
                LogitsProcessorList([watermark.spawn_logits_processor()]),
                return_model_inputs=return_model_inputs,
            )

    def detect(self, completions: List[str]) -> List[dict]:
        output_dicts = []
        for completion in completions:
            # z_scores = []  # For debug
            output_dict = None
            for watermark in self.watermarks:
                curr = watermark.detect([completion])[0]
                # z_scores.append(curr["z_score"])  # For debug
                if output_dict is None or curr["z_score"] > output_dict["z_score"]:
                    output_dict = curr  # max z score
            # print(f"Z Scores Per Key: {z_scores}") # For debug
            output_dicts.append(output_dict)
        assert output_dicts is not None
        return output_dicts
