from src.attackers.base_attacker import BaseAttacker
from src.attackers.our_attacker import OurAttacker
from src.attackers.sadasivan_attacker import SadasivanAttacker
from src.config import AttackerAlgo, WsConfig


def get_attacker(cfg: WsConfig) -> BaseAttacker:
    if cfg.attacker.algo == AttackerAlgo.OUR:
        return OurAttacker(
            cfg.meta,
            cfg.attacker,
            cfg.server.watermark.scheme,
            cfg.server.watermark.generation.seeding_scheme,
        )
    elif cfg.attacker.algo == AttackerAlgo.SADASIVAN:
        return SadasivanAttacker(cfg.meta, cfg.attacker)
    else:
        raise ValueError(f"Unknown attacker algo: {cfg.attacker.algo}")
