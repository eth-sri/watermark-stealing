from typing import List

from transformers import LogitsProcessor

from src.config import MetaConfig, WatermarkConfig


class BaseWatermark:
    def __init__(self, meta_cfg: MetaConfig, watermark_cfg: WatermarkConfig) -> None:
        self.device = meta_cfg.device
        self.cfg = watermark_cfg

    def spawn_logits_processor(self) -> LogitsProcessor:
        # We assume something like this will always exist
        raise NotImplementedError("BaseWatermark.spawn_logits_processor not implemented")

    def detect(self, completions: List[str]) -> List[dict]:
        raise NotImplementedError("BaseWatermark.detect not implemented")
