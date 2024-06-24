import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Iterator, List, Tuple

import openai
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LogitsProcessorList, PreTrainedTokenizer

from src.config import MetaConfig, ModelConfig
from src.models.utils import LogitInfo
from src.utils import print


class OpenAIModel:
    def __init__(self, meta_cfg: MetaConfig, model_cfg: ModelConfig) -> None:
        self.cfg = model_cfg
        self.client = openai.OpenAI(api_key=os.environ["OAI_API_KEY"])
        self.system_prompt = textwrap.dedent(
            """
        You are a helpful assistant.
        """
        )
        self.model_name = model_cfg.name[7:]
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1", padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.period_token_id = tokenizer("Game over.", add_special_tokens=False).input_ids[-1]
        return tokenizer

    def generate(
        self,
        prompts: List[str],
        logit_processors: LogitsProcessorList,
        reseed: bool = True,  # irrelevant
        return_model_inputs: bool = False,
    ) -> Any:
        if len(logit_processors) > 0:
            raise RuntimeError("Cant do logit processors on an openAI model")
        if return_model_inputs:
            raise RuntimeError("Cant return model inputs on an openAI model")
        completions_str = list(self._query_api(prompts))
        print(f"Model Output: {completions_str}", color="purple")

        return completions_str, None

    def get_ppls_and_logitinfo(
        self, prompts: List[str], completions: List[str], logit_processors: LogitsProcessorList
    ) -> Tuple[torch.Tensor, List[List[LogitInfo]]]:
        raise NotImplementedError("Cant get PPL from OpenAI")

    # TODO: unify with gpt_judge.py
    def _query_api(self, inputs: List[str], **kwargs: Any) -> Iterator[str]:
        print(f"Querying api with {inputs}")
        max_workers = 8
        base_timeout = 240

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(inputs)))
            retry_ctr = 0
            timeout = base_timeout

            while len(ids_to_do) > 0 and retry_ctr <= len(inputs):
                # executor.map will apply the function to every item in the iterable (prompts), returning a generator that yields the results
                results = executor.map(
                    lambda id: (
                        id,
                        inputs[id],
                        self.client.chat.completions.create(  # type: ignore
                            model=self.model_name, messages=inputs[id], temperature=1
                        )
                        .choices[0]
                        .message.content,
                    ),
                    ids_to_do,
                    timeout=timeout,
                )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Queries",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield answer
                        # answered_prompts.append()
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except openai.RateLimitError as r:
                    print(f"Rate Limit: {r}")
                    time.sleep(10)
                    continue
                except Exception as e:
                    print(f"Exception: {e}")
                    if getattr(e, "type", "default") == "invalid_prompt" and len(ids_to_do) == 1:
                        # We skip invalid prompts for GPT -> This only works with bs=1
                        ids_to_do = []
                        yield "invalid_prompt_error"
                    else:
                        time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(base_timeout, timeout)
                retry_ctr += 1
