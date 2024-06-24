from typing import Any, List, Optional

from src.config import AttackerConfig, AttackerGenerationConfig, MetaConfig
from src.server import Server


class BaseAttacker:
    def __init__(self, meta_cfg: MetaConfig, attacker_cfg: AttackerConfig) -> None:
        self.seed, self.device, self.out_root_dir = (
            meta_cfg.seed,
            meta_cfg.device,
            meta_cfg.out_root_dir,
        )
        self.cfg = attacker_cfg
        self.model: Any = None
        pass

    def query_server_and_save(self, server: Server) -> None:
        raise NotImplementedError("BaseAttacker.query not implemented")

    def load_queries_and_learn(self, base: bool) -> None:
        raise NotImplementedError("BaseAttacker.learn not implemented")

    def generate(
        self, prompts: List[str], cfg_gen: Optional[AttackerGenerationConfig], reseed: bool
    ) -> List[str]:
        raise NotImplementedError("BaseAttacker.generate not implemented")
