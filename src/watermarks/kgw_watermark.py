from typing import Any, List, Optional

import torch
from transformers import PreTrainedTokenizer

from src.config import MetaConfig, WatermarkConfig
from src.models import fix_isolated_punctuation
from src.watermarks.base_watermark import BaseWatermark
from src.watermarks.kgw import (  # type: ignore
    HardWatermarkLogitsProcessor,
    WatermarkDetector,
    WatermarkLogitsProcessor,
)

# NOTE ^ type ignored as external wm submodules don't typecheck


class KgwWatermark(BaseWatermark):
    def __init__(
        self,
        meta_cfg: MetaConfig,
        watermark_cfg: WatermarkConfig,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
    ) -> None:
        super().__init__(meta_cfg, watermark_cfg)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.vocab = list(self.tokenizer.get_vocab().values())

    @staticmethod
    def get_prevctx_width(seeding_scheme: str) -> int:
        if seeding_scheme == "selfhash":
            return 3
        elif seeding_scheme == "lefthash":
            return 1
        elif seeding_scheme == "gptwm":
            return 0
        elif seeding_scheme.startswith("ff") or seeding_scheme.startswith("hard-"):
            uses_self_salt = int(seeding_scheme.split("-")[3] == "True")
            return int(seeding_scheme.split("-")[2]) - uses_self_salt
        else:
            raise ValueError(f"Unknown KGW seeding scheme: {seeding_scheme}")

    def spawn_logits_processor(self) -> WatermarkLogitsProcessor:
        if self.cfg.generation.seeding_scheme.startswith("hard-"):
            return HardWatermarkLogitsProcessor(
                vocab=self.vocab,
                gamma=self.cfg.generation.gamma,
                delta=self.cfg.generation.delta,
                seeding_scheme=self.cfg.generation.seeding_scheme,
                device=self.device,
                tokenizer=self.tokenizer,  # needed just for debug
            )
        else:
            return WatermarkLogitsProcessor(
                vocab=self.vocab,
                gamma=self.cfg.generation.gamma,
                delta=self.cfg.generation.delta,
                seeding_scheme=self.cfg.generation.seeding_scheme,
                device=self.device,
                tokenizer=self.tokenizer,  # needed just for debug
            )

    def detect(self, completions: List[str]) -> List[dict]:
        detector = WatermarkDetector(
            vocab=self.vocab,
            seeding_scheme=self.cfg.generation.seeding_scheme,
            gamma=self.cfg.generation.gamma,
            device=self.device,
            tokenizer=self.tokenizer,
            normalizers=self.cfg.detection.normalizers,
            z_threshold=self.cfg.detection.z_threshold,
            ignore_repeated_ngrams=self.cfg.detection.ignore_repeated_ngrams,
        )
        detector_results: List[Any] = []
        for completion in completions:
            # TODO KGW internal returns a nasty dict, repack into a class
            if len(completion) <= KgwWatermark.get_prevctx_width(
                self.cfg.generation.seeding_scheme
            ):
                detector_results.append(None)
                # not enough length for /any/ signal
            else:
                detector_results.append(detector.detect(completion))
        return detector_results

    # Used as an oracle to check accuracy, which of toks_text toks is green
    # under the context of ctx_text?
    def get_greenness_dict(self, ctx_text: str, toks: List[int]) -> Optional[dict[int, bool]]:
        ctx = self.tokenizer([ctx_text], add_special_tokens=False)["input_ids"][0]
        ctx = [fix_isolated_punctuation(self.model_name, c) for c in ctx]
        return self.get_greenness_dict_ints(ctx, toks)

    def get_greenness_dict_ints(self, ctx: List[int], toks: List[int]) -> Optional[dict[int, bool]]:
        ctx_tensor = torch.tensor(ctx, device=self.device, dtype=torch.long)

        # Only possible if there is enough context
        if len(ctx_tensor) < KgwWatermark.get_prevctx_width(self.cfg.generation.seeding_scheme):
            return None

        # Two cases, in the "self salt" case we need to do rejection sampling
        processor = self.spawn_logits_processor()
        if not processor.self_salt:
            greenlist = processor._get_greenlist_ids(ctx_tensor)
        else:
            # We put fake logit 1 for toks of interest and 0 for rest
            # Rejection sampling (instructed to process exactly this many toks)
            # will prioritize these and return those that are green
            logits = torch.zeros((len(self.vocab),), dtype=torch.float, device=self.device)
            toks_tensor = torch.tensor(toks, device=self.device)
            logits.scatter_(
                0, toks_tensor, torch.ones(len(toks), dtype=torch.float, device=self.device)
            )
            greenlist = processor._score_rejection_sampling(
                ctx_tensor, logits, tail_rule=f"fixed_compute_{len(toks)}"
            )
        # Build the final green list: initialize to false and add those present
        return dict([(tok, tok in greenlist) for tok in toks])
