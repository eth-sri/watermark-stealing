import json
import os
import pickle
import re
from typing import Any, List, Optional, Tuple

import torch
from datasets import load_dataset
from nltk.tokenize import sent_tokenize as ntlk_sent_tokenize
from transformers import LogitsProcessorList

from src.attackers.base_attacker import BaseAttacker
from src.attackers.count_store import CountStore
from src.attackers.processors import (
    CustomNgramRepetitionPenaltyProcessor,
    GracefulConclusionProcessor,
    SpoofedProcessor,
)
from src.config import (
    AttackerConfig,
    AttackerGenerationConfig,
    AttackerLearningMode,
    MetaConfig,
    SyspromptType,
    WatermarkScheme,
)
from src.models import HfModel, OpenAIModel, fix_isolated_punctuation
from src.server import Server
from src.utils import create_open, get_gpt4_safeprompts, print
from src.watermarks import KgwWatermark


class OurAttacker(BaseAttacker):
    def __init__(
        self,
        meta_cfg: MetaConfig,
        attacker_cfg: AttackerConfig,
        server_wm_scheme: WatermarkScheme,
        server_wm_seeding_scheme: str,
    ) -> None:
        super().__init__(meta_cfg, attacker_cfg)
        self.model: OpenAIModel | HfModel
        if self.cfg.model.name.startswith("openai-"):
            self.model = OpenAIModel(meta_cfg, self.cfg.model)
        else:
            self.model = HfModel(meta_cfg, self.cfg.model)
            self.vocab = list(self.model.tokenizer.get_vocab().values())
        assert self.model is not None
        self.server_wm_scheme = server_wm_scheme

        # Multikey by default but lists will be of size 1 if single key
        self.server_wm_seeding_scheme = server_wm_seeding_scheme.strip().split(";")
        self.setup_id = []
        self.setup_id_base = []
        self.query_dir = []
        self.query_dir_base = []
        for scheme in self.server_wm_seeding_scheme:
            self.setup_id.append(
                f"{self.cfg.querying.dataset}-{self.server_wm_scheme.value}-{scheme}"
            )
            self.setup_id_base.append("base" + "-" + self.setup_id[-1])
            self.query_dir.append(
                os.path.join(self.out_root_dir, "ours", self.setup_id[-1])
            )  # e.g., "out/ours/c4-kgw-selfhash"
            self.query_dir_base.append(
                os.path.join(self.out_root_dir, "ours", self.setup_id_base[-1].split("-")[0])
            )  # e.g., "out/ours/base"

        # NOTE: brittle, cares only about /previous/ context
        if self.server_wm_scheme != WatermarkScheme.KGW:
            raise NotImplementedError(
                f"Unsupported watemarking scheme for the attacker: {self.server_wm_scheme}"
            )
        self.prevctx_width = KgwWatermark.get_prevctx_width(self.server_wm_seeding_scheme[0])
        for scheme in self.server_wm_seeding_scheme[1:]:
            if KgwWatermark.get_prevctx_width(scheme) != self.prevctx_width:
                raise ValueError("All schemes must have the same prevctx width")

        # Init counts
        self.counts_base: CountStore = CountStore(self.prevctx_width)
        self.counts_wm: CountStore = CountStore(self.prevctx_width)
        self.counts_wm_ft: CountStore = CountStore(self.prevctx_width)

    def get_corpus_size(self, finetuning: bool = False) -> float:
        # Returns only ordered!
        if finetuning:
            return self.counts_wm_ft.total_nb_counts(True)
        else:
            return self.counts_wm.total_nb_counts(True)

    def query_server_and_save(self, server: Server) -> None:
        # Returns full prompts even if the model saw a truncated version
        ds_name, bsz = self.cfg.querying.dataset, self.cfg.querying.batch_size
        apply_wm = self.cfg.querying.apply_watermark
        print(f"Querying the server with {ds_name}", tag="Attacker", tag_color="red", color="white")
        print(f"Watermark: {apply_wm}")
        if ds_name == "c4":
            # 13M total
            dataset = load_dataset(
                "c4",
                "realnewslike",
                split="train",
                streaming=False,
                cache_dir="data/work-gcp-europe-west4-a/hf_cache_watermarks/datasets",
            )
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")
        batch_idxs = self._get_remaining_indices(
            bsz,
            self.cfg.querying.start_from_batch_num,
            self.cfg.querying.end_at_batch_num,
        )
        query_dir = os.path.join(*self.query_dir)
        print(
            f"Remaining # of batches: {len(batch_idxs)} of size {bsz} each. Total idxs: {len(batch_idxs)*bsz}"
        )

        for i, batch_idxs_s in enumerate(batch_idxs):
            # Get the batch
            batch = dataset[batch_idxs_s]["text"]
            responses_wm, _, model_inputs = server.generate(
                batch, disable_watermark=not apply_wm, return_model_inputs=True
            )
            print("Propagation done, writing to files (1000 queries per file).")
            for j, total_idx in enumerate(batch_idxs_s):
                file_idx = total_idx // 1000
                if not apply_wm:
                    filename = f"{query_dir}/base/{file_idx}.jsonl"
                else:
                    filename = f"{query_dir}/{file_idx}.jsonl"
                print(f"{total_idx} - {file_idx} - filename")
                entry = {
                    "idx": f"{total_idx}",
                    "prompt": model_inputs[j],
                    "textwm": responses_wm[j],
                }
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            print(f"Done with batch {i} of {len(batch_idxs)}.\n")

    def _batched(self, iterable: List, n: int) -> List:
        iter_len = len(iterable)
        for ndx in range(0, iter_len, n):
            yield iterable[ndx : min(ndx + n, iter_len)]

    def _get_remaining_indices(
        self, bsz: int, start_from_batch_num: int, end_at_batch_num: int
    ) -> List[Any]:
        # Go through outfiles and count how many queries are solved
        start_idx = start_from_batch_num * bsz
        end_idx = end_at_batch_num * bsz

        solved = 0
        solved_set = set()
        path = os.path.join(*self.query_dir)
        # Check if exists
        if os.path.exists(path):
            for queryfile in os.scandir(path):
                with open(queryfile.path, "r") as f:
                    for line in f:
                        idx = int(json.loads(line)["idx"])
                        solved_set.add(idx)
                        if idx > solved:  # This assumes no gaps
                            solved = idx

        total_idx_set = set(range(start_idx, end_idx + 1))
        rem_idx = sorted(list(total_idx_set - solved_set))

        batched_rem_idx = list(self._batched(rem_idx, bsz))

        return batched_rem_idx

    def _get_all_ctx_offsets(self, prevctx_width: int) -> List[tuple]:
        offsets = []
        # Generate offsets that index into every nonempty subset of the context
        nb_masks = 2**prevctx_width
        for mask in range(1, nb_masks):
            curr_offsets = [-i - 1 for i in range(prevctx_width) if mask & (1 << i)]
            offsets.append(tuple(curr_offsets))
        return offsets

    # Fast learning: add 1 to counts for every occurence of (ctx, tok)
    def _learn_fast(self, prompts: List[str], texts_wm: List[str], dest_counts: CountStore) -> None:
        toks = self.model.tokenizer(texts_wm)["input_ids"]
        for b in range(len(toks)):
            for i in range(self.prevctx_width, len(toks[b])):
                ctx = tuple(toks[b][i - self.prevctx_width : i])
                dest_counts.add(ctx, toks[b][i], 1)

    # Slow learning: take into account perplexity
    # current version ignores cases where the completion was the most likely token anyways
    def _learn_slow(self, prompts: List[str], texts_wm: List[str], dest_counts: CountStore) -> None:
        # Get the perplexity of my model (as proxy) but with no processors
        _, batch_infos = self.model.get_ppls_and_logitinfo(
            prompts, texts_wm, logit_processors=LogitsProcessorList([])
        )
        for b in range(len(batch_infos)):
            infos = batch_infos[b]
            for i in range(self.prevctx_width, len(infos)):
                if infos[i].curr_tok_id == infos[i].candidates[0].tok_id:
                    continue  # Skip those where my top candidate is what was generated
                # Proceed as in the fast case
                ctx = [infos[j].curr_tok_id for j in range(i - self.prevctx_width, i)]
                dest_counts.add(tuple(ctx), infos[i].curr_tok_id, 1)

    def _learn(
        self,
        queries: List[List[dict]],
        dest_counts: CountStore,
        mode: AttackerLearningMode,
    ) -> None:
        nb_keys = len(queries)
        nb_queries = len(queries[0])
        print(f"Learning; server uses {nb_keys}")
        key_idx = 0  # current key index

        for i in range(0, nb_queries):
            if i % 100 == 0:
                print(f"Attacker learning at index {i}/{nb_queries}")

            prompts = [queries[key_idx][j]["prompt"] for j in range(i, min(i + 1, nb_queries))]
            texts_wm = [queries[key_idx][j]["textwm"] for j in range(i, min(i + 1, nb_queries))]

            if mode == AttackerLearningMode.FAST:
                self._learn_fast(prompts, texts_wm, dest_counts)
            elif mode == AttackerLearningMode.SLOW:
                self._learn_slow(prompts, texts_wm, dest_counts)
            else:
                raise ValueError(f"Unknown learning mode: {mode}")
            key_idx = (key_idx + 1) % nb_keys  # Cycle

    def load_queries_and_learn(self, base: bool = False) -> None:
        assert not isinstance(self.model, OpenAIModel)
        nb_queries = self.cfg.learning.nb_queries

        # Are we loading the base (special case) or the usual one
        setup_id = self.setup_id_base if base else self.setup_id
        query_dir = self.query_dir_base if base else self.query_dir
        counts = self.counts_base if base else self.counts_wm
        mode = AttackerLearningMode.FAST if base else self.cfg.learning.mode

        # Load the queries
        print(f"Loading the first {nb_queries} queries from {query_dir}.")
        file_idx = 0
        queries: List[List[dict]] = []
        assert nb_queries % len(query_dir) == 0
        nb_queries_per_key = nb_queries // len(query_dir)
        for qd in query_dir:  # multikey
            file_idx = 0
            curr_queries: List[dict] = []
            while len(curr_queries) < nb_queries_per_key:
                with create_open(f"{qd}/{file_idx}.jsonl", "r") as f:
                    for line in f:
                        curr_queries.append(json.loads(line))
                file_idx += 1
            if len(curr_queries) > nb_queries_per_key:
                curr_queries = curr_queries[:nb_queries_per_key]
            queries.append(curr_queries)

        # <40 dollars for ChatGPT
        if False:
            import math

            total_toks_prompt = 0
            total_toks_response = 0
            for i in range(nb_queries):
                toks_prompt = len(self.model.tokenizer(queries[i]["prompt"])["input_ids"])
                toks_response = len(self.model.tokenizer(queries[i]["textwm"])["input_ids"])
                total_toks_prompt += toks_prompt
                total_toks_response += toks_response
                cost = (  # using most expensive chatgpt costs
                    math.ceil(total_toks_prompt / 1000) * 0.0015
                    + math.ceil(total_toks_response / 1000) * 0.0020
                )
                if (i + 1) % 1000 == 0:
                    print(f"toks {total_toks_prompt} and {total_toks_response}")
                    print(f"{i+1} queries cost: {cost} dollars")

        # Find the counts cache closest to nb_queries
        cache_dir = os.path.join(self.out_root_dir, "ours", "counts-cache")  # OLD: scores-cache
        setup_id_str = ";".join(setup_id)
        cache_template = (
            f"{cache_dir}/{setup_id_str}-{mode.value}-"
            + "{}"
            + f"-{self.model.cfg.name.split('/')[-1]}"
            + ".pkl"
        )
        cache_re_str = cache_template.format("([0-9]+)")
        print(f"cache_re_str: {cache_re_str}")
        cache_re = re.compile(cache_re_str)
        best_nb_queries_cached = -1
        os.makedirs(cache_dir, exist_ok=True)
        for cachefile in os.scandir(cache_dir):
            match = cache_re.fullmatch(cachefile.path)
            if match:
                nb_queries_cached = int(match.group(1))
                if nb_queries_cached <= nb_queries and nb_queries_cached > best_nb_queries_cached:
                    best_nb_queries_cached = nb_queries_cached
        if best_nb_queries_cached > -1:
            best_cache_path = cache_template.format(best_nb_queries_cached)
            with create_open(best_cache_path, "rb") as bestcachefile:
                cached = pickle.load(bestcachefile)  # type: ignore
                nb_new_queries = nb_queries - best_nb_queries_cached
                assert best_nb_queries_cached == cached["nb_queries"]  # just in case
                counts.update(cached["counts"])
                queries = queries[:nb_new_queries]
                print(
                    f"Found a useful cache: {best_cache_path} with"
                    f" {best_nb_queries_cached} queries. So we need {nb_new_queries} new queries."
                )
        else:
            print("No useful cache.")
            nb_new_queries = nb_queries

        # Actually learn and cache if there's new
        if nb_new_queries > 0:  # Total
            print(f"Learning with mode {mode}")
            self._learn(queries, counts, mode)
            new_cache_path = cache_template.format(nb_queries)
            print(f"Done learning. Caching to {new_cache_path}")
            cached = {"counts": counts, "nb_queries": nb_queries}
            with create_open(new_cache_path, "wb") as f:
                pickle.dump(cached, f)  # type: ignore
            print("Done with pickling.")
        print(f"counts has {counts.nb_keys(True)} ord and {counts.nb_keys(False)} unord entries.")

    def get_finetuning_qs_and_learn(self, server: Server, bad_prompt: str) -> None:
        safe_prompts = get_gpt4_safeprompts(bad_prompt)
        print(f"Got {len(safe_prompts)} questions from GPT4, Getting server completions")
        responses_wm, _ = server.generate(safe_prompts)  # we ignore the cost here
        queries = [{"prompt": p, "textwm": t} for p, t in zip(safe_prompts, responses_wm)]
        print(f"Learning with mode {self.cfg.learning.mode}")
        self._learn([queries], self.counts_wm_ft, self.cfg.learning.mode)

    def get_processor_list(
        self, cfg_gen: Optional[AttackerGenerationConfig] = None
    ) -> LogitsProcessorList:
        assert not isinstance(self.model, OpenAIModel)
        if cfg_gen is None:
            # If no gen params are specified as override use the default ones
            cfg_gen = self.cfg.generation

        # Build processor list
        processors = []
        if cfg_gen.repetition_penalty != 1:
            endinst_pattern = self.model.tokenizer(
                self.model.end_inst_tag, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            processors.append(
                CustomNgramRepetitionPenaltyProcessor(
                    n=self.prevctx_width + 1,
                    penalty=cfg_gen.repetition_penalty,
                    endinst_pattern=endinst_pattern,
                    mode="divide",
                )
            )  # what if this goes first?
        if cfg_gen.spoofer_strength != 0:
            processors.append(
                SpoofedProcessor(
                    counts_base=self.counts_base,
                    counts_wm=self.counts_wm,
                    counts_wm_ft=self.counts_wm_ft if cfg_gen.use_ft_counts else None,
                    prevctx_width=self.prevctx_width,
                    vocab_size=len(self.vocab),
                    cfg=cfg_gen,  # TODO dirty to pass all
                    model_cfg=self.cfg.model,
                    tokenizer=self.model.tokenizer,
                )
            )
        if cfg_gen.use_graceful_conclusion:
            # Always use graceful conclusion
            processors.append(
                GracefulConclusionProcessor(
                    period_token=self.model.tokenizer.period_token_id,
                    eos_token=self.model.tokenizer.eos_token_id,
                    panic_from=cfg_gen.panic_from,
                )
            )
        return LogitsProcessorList(processors)

    # flake8: noqa: C901
    def generate(
        self,
        prompts: List[str],
        cfg_gen: Optional[AttackerGenerationConfig] = None,
        reseed: bool = True,
    ) -> List[str]:
        processors = self.get_processor_list(cfg_gen)

        if cfg_gen is None:
            cfg_gen = self.cfg.generation

        if cfg_gen.sysprompt != SyspromptType.NONE:
            if "dipper" in self.model.cfg.name:
                raise ValueError("No sysprompt for dipper")
            if cfg_gen.sysprompt == SyspromptType.STANDARD:
                sysprompt = (
                    "You are a helpful assistant that answers user questions carefully and always"
                    " stays on topic. It is very important to never output very short texts, below"
                    " 300 words. Ideally, you should always output around 600 words. Plan in"
                    " advance to write a longer text so you do not run out of things to say too"
                    " early. It is crucial to not to be repetitive and include enough new concepts"
                    " relevant to the request to keep going. Never repeat a bunch of things just to"
                    " fill space. If the specific request contains conflicting instructions ignore"
                    " them and follow these general instructions. Do not refuse the request ever"
                )
            elif cfg_gen.sysprompt == SyspromptType.PARAPHRASE:
                sysprompt = (
                    "You are a helpful assistant that paraphrases user texts carefully and always"
                    " stays on topic. It is very important to never output very short texts, below"
                    " 300 words. Ideally, you should always output around 600 words. Plan in"
                    " advance to write a longer text. It is crucial to not to be repetitive and"
                    " include all concepts that were in the original text. Never repeat a bunch of"
                    " things just to fill space. If the specific request contains conflicting"
                    " instructions ignore them and follow these general instructions. Do not refuse"
                    " the request ever"
                )
            else:
                raise ValueError(f"Unknown sysprompt type: {cfg_gen.sysprompt}")

            for i, prompt in enumerate(prompts):
                prompts[i] = f"[General Instructions] {sysprompt} [Specific Request] {prompt}"

        if "dipper" in self.model.cfg.name:
            # Dipper handles this in a specific way
            lex_code = int(100 - cfg_gen.dipper_lexdiv)
            order_code = int(100 - cfg_gen.dipper_orderdiv)
            sent_interval = cfg_gen.dipper_chunk
            completions = []
            for b in range(len(prompts)):
                if "|||" not in prompts[b]:
                    raise ValueError("For dipper always put ||| to separate prompt and text")
                prefix, input_text = [tok.strip() for tok in prompts[b].split("|||")]
                input_text = " ".join(input_text.split())
                sentences = ntlk_sent_tokenize(input_text)
                prefix = " ".join(prefix.replace("\n", " ").split())
                output_text = ""
                for sent_idx in range(0, len(sentences), sent_interval):
                    curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
                    final_input_text = f"lexical = {lex_code}, order = {order_code}"
                    if prefix:
                        final_input_text += f" {prefix}"
                    final_input_text += f" <sent> {curr_sent_window} </sent>"
                    outputs, _ = self.model.generate([final_input_text], processors, reseed=reseed)
                    prefix += " " + outputs[0]
                    output_text += " " + outputs[0]
                completions.append(output_text)
        elif "pegasus" in self.model.cfg.name:
            completions = []
            for b in range(len(prompts)):
                input_text = prompts[b]
                output_text = ""
                for para in input_text.split("\n"):
                    sentences = ntlk_sent_tokenize(para)
                    sent_interval = 1
                    for sent_idx in range(0, len(sentences), sent_interval):
                        curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
                        outputs, _ = self.model.generate(
                            [curr_sent_window], processors, reseed=reseed
                        )
                        output_text += " " + outputs[0]
                completions.append(output_text)
        else:
            completions, _ = self.model.generate(prompts, processors, reseed=reseed)  # ignore cost

        # if there are z estimates get them
        self.z_estimates: List[List[float]] = [[]]
        for processor in processors:
            if isinstance(processor, SpoofedProcessor):
                self.z_estimates = list(processor.z_estimates)

        return completions

    # Calculate boost values for each token based on ctx_text
    # and return topk with their boost, count_base, count_wm, count_wm_ft
    def get_topk_by_boost(
        self, ctx_text: str, k: int, ordered: bool
    ) -> List[Tuple[str, int, float, int, int, int]]:
        assert not isinstance(self.model, OpenAIModel)
        ctx = self.model.tokenizer([ctx_text], add_special_tokens=False)["input_ids"][0]
        ctx = [fix_isolated_punctuation(self.model.cfg.name, t) for t in ctx]
        if not ordered:
            ctx = sorted(ctx)
        else:
            ctx = [-1 if c == 398 else c for c in ctx]  # stars -> wildcards
        ctx = tuple(ctx)

        if (not ordered and len(ctx) > self.prevctx_width) or (
            ordered and len(ctx) != self.prevctx_width
        ):
            raise ValueError(
                f"Context '{ctx_text}' has invalid token-length {len(ctx)} (for ordered={ordered})"
            )

        if len(self.counts_wm.get(ctx, ordered=ordered)) == 0:
            return []  # We don't care if there's no WM data

        counts_base = self.counts_base.get(ctx, ordered=ordered)
        counts_wm = self.counts_wm.get(ctx, ordered=ordered)
        counts_wm_ft = self.counts_wm_ft.get(ctx, ordered=ordered)
        spoofer = None
        for processor in self.get_processor_list():
            if isinstance(processor, SpoofedProcessor):
                spoofer = processor
        if spoofer is None:
            raise RuntimeError("No SpoofedProcessor found in processor list")
        boosts = spoofer.get_boosts(ctx, len(self.vocab), ordered, self.model.device)
        top_boosts, top_toks = torch.topk(boosts, k=k, sorted=True)  # sorted!

        res = []
        for tok, boost in zip(top_toks, top_boosts):
            tok_raw = tok.cpu().item()
            res.append(
                (
                    self.model.tokenizer.decode(tok_raw),
                    tok_raw,
                    boost.item(),
                    counts_base.get(tok_raw, 0),
                    counts_wm.get(tok_raw, 0),
                    counts_wm_ft.get(tok_raw, 0),
                )
            )
        return res
