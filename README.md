# Watermark Stealing  😈💧 <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

This repository contains the code accompanying our ICML 2024 paper: 

> Nikola Jovanović, Robin Staab, and Martin Vechev. 2024. _Watermark Stealing in Large Language Models._ In Proceedings of ICML ’24.

For an overview of the work, check out our project website: **[watermark-stealing.org](https://watermark-stealing.org)**.

## Basic Setup

To set up the project, clone the repository and execute the steps from `setup.sh`. Running all steps will install Conda, create the `ws` environment, install the dependencies listed in `env.yaml`, install Flash attention, and download the PSP model needed for scrubbing evaluation. On top of this, make sure to set the `OAI_API_KEY` environment variable to your OpenAI API key (to use GPT-as-a-judge evaluation).

## Repository Structure 

The project structure is as follows.
- `main.py` is the main entry point for the code.
- `src/` contains the rest of the code, namely:
    - `src/attackers` contains all our attacker code for all 3 steps of the watermark stealing attack (see below under "Running the Code").
    - `src/config` contains definitions of our Pydantic configuration files. Refer to `ws_config.py` for detailed explanations of each field.
    - `src/models` contains model classes for our server, attacker, judge, and PSP models. 
    - `src/utils` contains utility functions for file handling, logging, and the use of GPT as a judge.
    - `src/watermarks` contains watermark implementations to be used on the server. 
    - `evaluator.py` implements all evaluation code for the attacker; we are primarily interested in the `targeted` evaluation mode. 
    - `gradio.py` contains the (experimental) Gradio interface used for debugging; this is not used in our experiments.
    - `server.py` contains the code for the server, i.e., the watermarked model.
- `configs/` contains YAML configuration files (corresponding to `src/config/ws_config.py`) for our main experiments reported in the paper. 
- `data/` holds static data files for some datasets used in the experiments.

## Running the Code

Our code can be run by providing a path to a YAML configuration file. For example:

```
python3 main.py configs/spoofing/llama7b/mistral_selfhash.yaml
```

This example will run watermarking stealing with `Llama-7B` as the watermarked server model using the `KGW2-SelfHash` scheme, and `Mistral-7B` as the attacker model, evaluated on a _spoofing_ attack. If `use_neptune` is set to true the experiment will be logged in neptune.ai; to enable this, set the `NEPTUNE_API_TOKEN` environment variable and replace `ORG/PROJ` in `src/config/ws_config.py` with your project ID to set it as default, or add it to the config file for each of your runs.

This executes the following three key steps, also visible in each config file:

1) `querying`: The attacker queries the watermarked server with a set of prompts and saves the resulting responses as `json` files. This step can be skipped by downloading all watermarked server outputs used in our experimental evaluation from [this link](https://drive.google.com/file/d/1UrPUAJ-ZyHiMdL3uL9WUG0h8e2hPQN8v/view?usp=sharing), and setting `skip: true` in the relevant section of the config file (done by default). Extract the archive such that `out_mistral`, `out_llama` and `out_llama13b` are in the root of the project.
2) `learning`: The attacker loads the responses and uses our algorithm to learn an internal model of the watermarking rules.
3) `generation`: The attacker mounts a _scrubbing_ or a _spoofing_ attack using the logit processors defined in `src/attackers/processors.py`. The `evaluator` section of the config file defines the relevant evaluation setting. To evaluate a scrubbing attack, first execute a _server run_ (see `server_*.yaml` files) to produce watermarked responses and log them as a neptune experiment, whose ID should be set in the `get_server_prompts_from` field of the config file of the main run. The code can be easily extended to use local storage if neptune is not available.

To obtain the results reported in the paper, we have postprocessed the results of the runs such as above to compute the FPR/FNR metrics under a specific FPR setting (as detailed in the paper). We have also recomputed the PPL of all texts using `Llama-13B` for consistency across experiments.

## Contact

Nikola Jovanović, nikola.jovanovic@inf.ethz.ch<br>
Robin Staab, robin.staab@inf.ethz.ch<br>
Martin Vechev

## Citation

If you use our code please cite the following.

```
@inproceedings{jovanovic2024watermarkstealing,
    author = {Jovanović, Nikola and Staab, Robin and Vechev, Martin},
    title = {Watermark Stealing in Large Language Models},
    journal = {{ICML}},
    year = {2024}
}
```