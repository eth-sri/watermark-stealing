import datetime
import gc
import json
import os
import sys

import neptune
import torch
from neptune.utils import stringify_unsupported

from src.attackers import get_attacker
from src.config.meta_config import get_pydantic_models_from_path
from src.evaluator import Evaluator
from src.gradio import run_gradio
from src.server import Server
from src.utils import create_open


def main(cfg_path: str) -> None:
    cfgs = get_pydantic_models_from_path(cfg_path)
    print(f"Number of configs: {len(cfgs)}")
    for cfg in cfgs:
        out_dir = cfg.get_result_path()
        with create_open(f"{out_dir}/config.txt", "w") as f:
            json.dump(cfg.model_dump(mode="json"), indent=4, fp=f)

        if cfg.meta.use_neptune:
            mode = "scrub" if "dipper" in cfg.attacker.model.name else "spoof"
            model = "llama" if "llama" in cfg.meta.out_root_dir else "mistral"
            run = neptune.init_run(
                project=cfg.meta.neptune_project,
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                name=f"{mode}-{model}",
                monitoring_namespace="monitoring-namespace",
            )  # your credentials
            # Set up logging
            run["cfg_path"] = cfg_path
            run["config"] = stringify_unsupported(cfg.model_dump(mode="json"))
        else:
            run = None

        server = Server(cfg.meta, cfg.server)
        attacker = get_attacker(cfg)

        if not attacker.cfg.querying.skip:
            attacker.query_server_and_save(server)

        if not attacker.cfg.learning.skip:
            attacker.load_queries_and_learn(base=False)
            attacker.load_queries_and_learn(base=True)

        evaluator = Evaluator(
            cfg.meta.seed,
            cfg.evaluator,
            server,
            verbose=True,
            neptune_project=cfg.meta.neptune_project,
            run=run,
        )
        if not cfg.evaluator.skip:
            # Server needed only for scrubbing (to generate original watermarked completions)
            evaluator.run_eval(server, attacker, out_dir=out_dir)

        if not cfg.gradio.skip:
            run_gradio(cfg, server, attacker, evaluator)

        if cfg.meta.use_neptune:
            assert run is not None
            run.stop()

        # Clean up
        server = None  # type: ignore
        attacker = None  # type: ignore
        evaluator = None  # type: ignore
        run = None
        gc.collect()
        torch.cuda.empty_cache()

    print("Done")


if __name__ == "__main__":
    print(f"{datetime.datetime.now()}")
    if len(sys.argv) != 2:
        raise ValueError(
            f"Exactly one argument expected (the path to the config file), got {len(sys.argv)}."
        )
    main(sys.argv[1])
