from typing import Optional

from transformers import PreTrainedTokenizer

from src.config import MetaConfig, WatermarkConfig, WatermarkScheme
from src.watermarks.base_watermark import BaseWatermark
from src.watermarks.kgw_watermark import KgwWatermark


def get_watermark(
    meta_cfg: MetaConfig,
    watermark_cfg: WatermarkConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model_name: Optional[str] = None,
) -> BaseWatermark:
    if watermark_cfg.scheme == WatermarkScheme.KGW:
        if tokenizer is None or model_name is None:
            raise ValueError("KGW watermark requires tokenizer and model_name to be passed")
        return KgwWatermark(meta_cfg, watermark_cfg, tokenizer, model_name)
    else:
        raise ValueError(f"Unknown watermark scheme: {watermark_cfg.scheme}")
