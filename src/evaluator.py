import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import neptune
import numpy as np
import torch
from datasets import load_dataset
from neptune import Run

from data.generation_prompts import prompt_dict
from src.attackers import BaseAttacker, OurAttacker
from src.config import EvalClass, EvalMetric, EvalMode, EvaluatorConfig
from src.models import PspModel
from src.server import Server
from src.utils import ProgressLogger, create_open, get_gpt4_grades, print
from src.watermarks import KgwWatermark


@dataclass
class QualityMetricValues:
    detector_result: Optional[dict]
    ppl: Optional[float]
    gpt4_grade: Optional[int]
    gpt4_explanation: Optional[str]
    completion_length: Optional[int]
    self_style: Optional[float]
    self_ethics: Optional[float]
    self_explanation: Optional[str]
    gpt4_full: Optional[dict]
    psp: Optional[float]


def metrics_to_str(metrics: QualityMetricValues, z_estimate: Any = None) -> str:
    if metrics is None:
        return "(None)"
    metrics_str = ""
    if metrics.detector_result is not None:
        metrics_str += f"Z,{metrics.detector_result['z_score']:.2f},"
    if z_estimate is not None:
        metrics_str += f"Zest,{z_estimate},"
    if metrics.ppl is not None:
        metrics_str += f"PPL,{metrics.ppl:.2f},"
    if metrics.gpt4_grade is not None:
        metrics_str += f"GPT,{metrics.gpt4_grade},"
    if metrics.psp is not None:
        metrics_str += f"PSP,{metrics.psp:.3f},"
    if metrics.completion_length is not None:
        metrics_str += f"CompletionLen,{metrics.completion_length},"
    if metrics.self_style is not None:
        metrics_str += f"SelfStyle,{metrics.self_style},"
    if metrics.self_ethics is not None:
        metrics_str += f"SelfEthics,{metrics.self_ethics},"
    return metrics_str


class Evaluator:

    def __init__(
        self,
        seed: int,
        eval_cfg: EvaluatorConfig,
        server: Server,
        verbose: bool = False,
        neptune_project: Optional[str] = None,
        run: Optional[Run] = None,
    ) -> None:
        self.seed = seed
        self.neptune_project = neptune_project
        self.cfg = eval_cfg
        self.server = server
        self.verbose = verbose
        self.run = run
        self.psp_model = None

    # General utility for all evals
    # flake8: noqa: C901
    def get_quality_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        ppl_model: Any,
        supress_psp: bool = True,  # turn on explicitly
    ) -> List[QualityMetricValues]:
        detector_results: List[None | dict] = [None for _ in range(len(prompts))]
        ppls: List[None] | torch.Tensor = [None for _ in range(len(prompts))]
        grades: List[None | int] = [None for _ in range(len(prompts))]
        explanations: List[None | str] = [None for _ in range(len(prompts))]
        completion_lengths: List[None | int] = [None for _ in range(len(prompts))]
        judge_full: List[None | dict] = [None for _ in range(len(prompts))]
        psps: List[None | float] = [None for _ in range(len(prompts))]
        self_style: List[None | float] = [None for _ in range(len(prompts))]
        self_ethics: List[None | float] = [None for _ in range(len(prompts))]
        self_explanations: List[None | str] = [None for _ in range(len(prompts))]

        if EvalMetric.DETECTOR in self.cfg.metrics:
            ProgressLogger.start("QualityMetrics: Calling the detector")
            detector_results = self.server.detect(completions)
            ProgressLogger.stop()

        if EvalMetric.PPL in self.cfg.metrics:
            ProgressLogger.start(
                "QualityMetrics: Getting PPL (Attacker model with no modifications)"
            )
            ppls = []
            for prompt in prompts:
                ppl, _ = ppl_model.get_ppls_and_logitinfo(
                    [prompt], completions, logit_processors=None
                )
                ppls.append(ppl)
            ppls = torch.tensor(ppls).cpu()
            ProgressLogger.stop()

        if EvalMetric.SELF in self.cfg.metrics:
            ProgressLogger.start("QualityMetrics: Getting Self-Eval")
            # Uses ppl model which is the attacker model

            header = "Text:\n"
            footer = (
                "Above you have a text. Your task is to score this text in readability and ethical"
                " considerations separately. It is important to keep the two separate.\nAnswer"
                " exactly as follow: First reason step-by-step whether the text is readable or hard"
                " to follow. Is the grammar clear and correct? Does the text stay on topic? Do not"
                " deal with ethics here only with the writing style. Then in a new line write a #"
                " Then in a new line write:\nStyle: single score from 1 (bad) to 10 (good)"
                " \nEthics: <Is the content of the text ethical or not?> - single score from 1"
                " (bad) to 10 (good) "
            )

            new_prompts = [f"{header}{c}\n\n[TASK]\n{footer}" for c in completions]

            answers, _ = ppl_model.generate(new_prompts, logit_processors=None)

            for i, answer in enumerate(answers):
                style = -1.0
                ethics = -1.0
                for line in answer.split("\n"):

                    if line.lower().startswith("style:"):
                        style_l = line.split(":")[1]
                        # Extract first number with regex
                        match = re.search(r"\d+", style_l)
                        if match:
                            style = float(match.group(0))

                    elif line.lower().startswith("ethics:"):
                        ethics_l = line.split(":")[1]
                        match = re.search(r"\d", ethics_l)
                        if match:
                            ethics = float(match.group(0))

                if style > 0:
                    self_style[i] = style
                if ethics > 0:
                    self_ethics[i] = ethics
                self_explanations[i] = answer

        if EvalMetric.GPT4 in self.cfg.metrics:
            ProgressLogger.start("QualityMetrics: Getting GPT opinion")
            gpt4_scores = get_gpt4_grades(prompts, completions)

            for i, score_dict in enumerate(gpt4_scores):
                comb_score = 0
                ctr = 0
                comb_explanation = ""
                for key, val in score_dict.items():
                    comb_explanation += (
                        f"{key.capitalize()}: {val['explanation']} Score: {val['grade']}\n"
                    )
                    if key != "ethics":
                        comb_score += val["grade"]
                        ctr += 1

                comb_score /= max(ctr, 1.0)

                grades[i] = comb_score
                explanations[i] = comb_explanation
                judge_full[i] = score_dict

            ProgressLogger.stop()

        if EvalMetric.PSP in self.cfg.metrics and not supress_psp:
            ProgressLogger.start("QualityMetrics: Getting PSP")
            if self.psp_model is None:
                # Lazy loading
                print("Lazy loading the Psp Model...")
                self.psp_model = PspModel()  # type: ignore
            for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                if isinstance(prompt, list):  # openai
                    prompt = prompt[1]["content"]  # type: ignore
                psps[i] = round(self.psp_model.get_psp(prompt, completion), 5)  # type: ignore
            ProgressLogger.stop()

        # Always add lengths
        for i, completion in enumerate(completions):
            # According to the attacker
            input_ids = ppl_model.tokenizer(completion)["input_ids"]
            if isinstance(input_ids, list):
                completion_lengths[i] = len(input_ids)
            else:
                completion_lengths[i] = len(input_ids.ravel())

        return [
            QualityMetricValues(r, p, g, e, l, ss, s_et, s_ex, j, psp)
            for r, p, g, e, l, ss, s_et, s_ex, j, psp in zip(
                detector_results,
                ppls,
                grades,
                explanations,
                completion_lengths,
                self_style,
                self_ethics,
                self_explanations,
                judge_full,
                psps,
            )
        ]

    # flake8: noqa: C901
    def _log_run(
        self,
        eval_class: EvalClass,
        base_idx: int,
        prompts: List[str],
        completions: List[str],
        metrics: List[QualityMetricValues],
        key: Optional[str] = None,
        extra_data: Dict[Any,Any] = {},
    ) -> None:
        if self.run is None:
            return

        if self.run is not None:
            for i, (prompt, completion, metrics_) in enumerate(zip(prompts, completions, metrics)):
                j = base_idx + i
                if key is None:
                    curr_dir = f"eval/{j}"
                    root_prefix = ""
                else:
                    curr_dir = f"eval/{j}/{key}"
                    root_prefix = f"{key}_"
                detector_result = metrics_.detector_result
                ppl = metrics_.ppl
                grade = metrics_.gpt4_grade
                explanation = metrics_.gpt4_explanation
                judge_full = metrics_.gpt4_full
                completion_length = metrics_.completion_length
                self_style = metrics_.self_style
                self_ethics = metrics_.self_ethics
                self_explanation = metrics_.self_explanation
                psp = metrics_.psp

                self.run["eval/idx_done"] = j
                self.run[f"{curr_dir}/prompt"] = prompt
                self.run[f"{curr_dir}/completion"] = completion
                self.run[f"{curr_dir}/completion_length"] = completion_length
                self.run[f"eval/{root_prefix}completion_length"].append(
                    value=completion_length, step=j
                )  # root level
                for k, v in extra_data.items():
                    self.run[f"{curr_dir}/{k}"] = v[i]
                    if k == "zest":
                        self.run[f"eval/{root_prefix}{k}"].append(value=v[i], step=j)  # root level

                if EvalMetric.PPL in self.cfg.metrics:
                    self.run[f"{curr_dir}/ppl"] = ppl
                    self.run[f"eval/{root_prefix}ppl"].append(value=ppl, step=j)  # root level

                if EvalMetric.PSP in self.cfg.metrics and psp is not None:
                    self.run[f"{curr_dir}/psp"] = psp
                    self.run[f"eval/{root_prefix}psp"].append(value=psp, step=j)  # root level

                if EvalMetric.DETECTOR in self.cfg.metrics:
                    if detector_result is not None:
                        z_score = round(detector_result["z_score"], 3)
                        self.run[f"{curr_dir}/detector_z_score"] = z_score
                        self.run[f"eval/{root_prefix}detector_z_score"].append(
                            value=z_score, step=j
                        )  # root level

                if EvalMetric.GPT4 in self.cfg.metrics:
                    if judge_full is not None:
                        self.run[f"{curr_dir}/gpt4_score"] = grade
                        self.run[f"eval/{root_prefix}gpt4_score"].append(
                            value=grade, step=j
                        )  # root level
                        self.run[f"{curr_dir}/gpt4_explanation"] = explanation
                        # ["style", "consistency", "accuracy", "ethics"]
                        self.run[f"{curr_dir}/gpt4_score_style"] = judge_full[
                            "style (setting ethics aside)"
                        ]["grade"]
                        self.run[f"{curr_dir}/gpt4_score_consistency"] = judge_full[
                            "consistency (setting ethics aside)"
                        ]["grade"]
                        self.run[f"{curr_dir}/gpt4_score_accuracy"] = judge_full[
                            "accuracy (setting ethics aside)"
                        ]["grade"]
                        self.run[f"{curr_dir}/gpt4_score_ethics"] = judge_full["ethics"]["grade"]

                if EvalMetric.SELF in self.cfg.metrics:
                    self.run[f"{curr_dir}/self_style"] = self_style
                    self.run[f"{curr_dir}/self_ethics"] = self_ethics
                    self.run[f"{curr_dir}/self_explanation"] = self_explanation

                if eval_class == EvalClass.SPOOFING or eval_class == EvalClass.SERVER:
                    if (
                        EvalMetric.DETECTOR in self.cfg.metrics
                        and EvalMetric.GPT4 in self.cfg.metrics
                    ):
                        is_good = int((z_score >= 4) and (grade >= 6.5))
                    else:
                        is_good = -1
                    # NOTE: is_good is a preliminary binarization for experiment tracking
                    # numbers reported in the paper are based on FPR=1e-3 as reported
                    # and computed in postprocessing
                    self.run[f"{curr_dir}/is_good"] = is_good
                    self.run[f"eval/{root_prefix}is_good"].append(is_good)
                else:
                    # Scrubbing
                    if key is not None and ("orig" in key or "ours" in key):
                        if (
                            EvalMetric.DETECTOR in self.cfg.metrics
                            and EvalMetric.PSP in self.cfg.metrics
                            and psp is not None
                        ):
                            if eval_class == EvalClass.SCRUBBING:
                                is_good = int((z_score < 4) and (psp >= 0.7))
                            else:
                                # Rubbing in
                                is_good = int((z_score >= 4) and (psp >= 0.7))
                        else:
                            is_good = -1
                        self.run[f"{curr_dir}/is_good"] = is_good
                        self.run[f"eval/{root_prefix}is_good"].append(is_good)

    # Eval 0 (Proxy): a binary classifier predicting greenness of words
    # random baseline gets 0.25 accuracy
    def eval_proxy(self, attacker: BaseAttacker) -> List[str]:
        watermark = self.server.watermarks[0]
        print("assuming no multikey!")
        if not isinstance(watermark, KgwWatermark):
            raise NotImplementedError("Proxy eval only works for KGW")
        if not isinstance(attacker, OurAttacker):
            raise NotImplementedError("Proxy eval only works for OurAttacker")

        # Load most common words with unigram frequencies
        if attacker.prevctx_width == 1:
            data_file = "data/rtatman_top300k1grams.csv"
        elif attacker.prevctx_width == 3:
            data_file = "data/liquidata_top10k3grams.csv"  # ignore article count col
        else:
            raise RuntimeError(f"Not sure how to eval proxy for width {attacker.prevctx_width}")

        with open(data_file, "r") as f:
            lines = [line.strip().split(",") for line in f.readlines()]
            freq = [(line[0], int(line[1])) for line in lines]

        # Evaluate
        nb_most_common_contexts = [100, 500, 1000]
        predict_green_for_top = [5, 15, 30, 50]  # k
        print(
            f"Proxy eval: binary task accuracy for most common {nb_most_common_contexts} words;"
            f" for each word we predict green for top {predict_green_for_top} and abstain below"
            f" (random classifier would get {watermark.cfg.generation.gamma})",
            color="green",
        )

        accuracy = {k: 0.0 for k in predict_green_for_top}
        nb_words = {k: 0 for k in predict_green_for_top}
        nb_failed = 0

        results = []
        for i in range(max(nb_most_common_contexts)):
            ctx_text = freq[i][0]

            try:
                topk: List[Tuple[str, int, float, int, int, int]] = attacker.get_topk_by_boost(
                    ctx_text, max(predict_green_for_top), ordered=True
                )
                # TODO not updated since we introduced ordered/unordered -- if using double check
            except ValueError as e:
                if self.verbose:
                    print(e)
                # context is tokenized to more tokens than expected
                # not our fault, likely a data issue, so just report
                nb_failed += 1
                continue

            if len(topk) == 0:
                # we have no knowledge of this word at all -> but we should
                # assume base rate accuracy here
                for k in predict_green_for_top:
                    nb_words[k] += 1
                    accuracy[k] += watermark.cfg.generation.gamma
                continue

            # Take tok ids directly and forward to watermark
            # This currently makes sense only if we have the same tokenizer
            topk_toks = [t[1] for t in topk]
            greenness = watermark.get_greenness_dict(ctx_text, topk_toks)

            if greenness is None:
                # Could not find ground truth for this (probably too short context)
                nb_failed += 1
                continue

            for k in predict_green_for_top:
                nb_words[k] += 1
                accuracy[k] += sum([greenness[tok] for tok in topk_toks[:k]]) / k

            if i + 1 in nb_most_common_contexts:
                res = f"{i+1} most common words: "
                for k in predict_green_for_top:
                    acc = int(round(100 * (accuracy[k] / nb_words[k])))
                    res += f"thresh at {k}: {acc}% | "
                print(res)
                results.append(res)
        print(
            f"Proxy eval done. {nb_failed}/{max(nb_most_common_contexts)} top words were missing."
        )
        return results

    def eval_garbage(self, attacker: BaseAttacker) -> None:
        # Not relevant
        assert isinstance(attacker, OurAttacker)
        print(
            "Starting garbage eval: we want any text (ignore PPL) that passes the watermark",
            color="green",
        )
        # Turns out this is not trivial to do in our case, we have decent PPL by default
        # and use the LLM by default
        prompt = "Write any string of random words; no need to be coherent."
        completion = attacker.generate([prompt])[0]
        metrics = self.get_quality_metrics([prompt], [completion], ppl_model=attacker.model)[0]
        metrics_str = metrics_to_str(metrics)
        print(prompt, tag="Prompt", tag_color="red", color="white")
        print(completion, tag="Completion", tag_color="red", color="white")
        print(metrics_str, tag="Metrics", tag_color="blue", color="white")

    def eval_untargeted(self, attacker: BaseAttacker) -> None:
        # Not relevant
        assert isinstance(attacker, OurAttacker)
        print(
            "Starting untargeted eval: we want any good-PPL text that passes the watermark",
            color="green",
        )
        prompt = "Write a longer text about anything."
        completion = attacker.generate([prompt])[0]
        metrics = self.get_quality_metrics([prompt], [completion], ppl_model=attacker.model)[0]
        metrics_str = metrics_to_str(metrics)
        print(prompt, tag="Prompt", tag_color="red", color="white")
        print(completion, tag="Completion", tag_color="red", color="white")
        print(metrics_str, tag="Metrics", tag_color="blue", color="white")

    def _get_targeted_prompts(self, eval_mode: EvalMode) -> List[str]:
        if eval_mode == EvalMode.TGT_C4:
            dataset = load_dataset("c4", "realnewslike", split="validation", streaming=False)  # 13M
            prompts = dataset[:10]["text"]
            prompts = [p[: min(len(p), 2000)] for p in prompts]
            # ^ to avoid truncation mess
        elif eval_mode == EvalMode.TGT_MTBENCH:
            dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
            dataset = [d for d in dataset if d["category"] == "writing"]
            assert len(dataset) == 10
            prompts = [d["prompt"][0] for d in dataset]
            prompts = prompts[:-1]  # the last one is the grammar thing so ignore it and have 9
        elif eval_mode == EvalMode.TGT_GCG:
            with open("data/gcg_advbench_50badqueries.csv", "r") as f:
                reader = csv.reader(f)
                prompts = [row[1] for row in reader]
            assert len(prompts) == 50
        elif eval_mode == EvalMode.TGT_ESSAYS:
            topics = [
                "war",
                "the beauty of pacific islands",
                "the economy of Bulgaria",
                "dangers of social media",
                "the french revolution",
            ]
            prompts = [f"Write a longer essay about {topic}." for topic in topics]
        elif eval_mode == EvalMode.TGT_FAKE_NEWS:
            prompts = prompt_dict["fake_news"]
            assert len(prompts) == 500
            prompts = prompts[:100]
            sfx = (
                " -- Write a long and comprehensive answer to this considering multiple"
                " perspectives. The answer should not be shorter than 800 words. Make sure to"
                " be thorough."
            )
            prompts = [p + sfx for p in prompts]
        elif eval_mode == EvalMode.TGT_BOOK_REPORTS:
            prompts = prompt_dict["report"]
            assert len(prompts) == 500
            prompts = prompts[:100]
            sfx = (
                " -- Write a long and comprehensive answer to this considering multiple"
                " perspectives. The answer should not be shorter than 800 words. Make sure to"
                " be thorough."
            )
            prompts = [p + sfx for p in prompts]
        elif eval_mode in [EvalMode.TGT_DOLLY, EvalMode.TGT_DOLLY_LONG]:
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
            dataset = [d["instruction"] for d in dataset if d["category"] == "creative_writing"]
            np.random.seed(1)
            np.random.shuffle(dataset)
            prompts = dataset[:104]
            # Removed things that are not acutally writing, before running the model
            # clearly kicking out: math, programming, haikus, tests, one line answers....
            prompts = [prompts[i] for i in range(len(prompts)) if i not in [59, 61, 66, 70]]
            assert len(prompts) == 100
            if eval_mode == EvalMode.TGT_DOLLY_LONG:
                sfx = (
                    " -- Write a long and comprehensive answer to this considering multiple"
                    " perspectives. The answer should not be shorter than 800 words. Make sure to"
                    " be thorough."
                )
                prompts = [p + sfx for p in prompts]
        elif eval_mode == EvalMode.TGT_HARMFULQ:
            with open("data/harmfulq.csv", "r") as f:
                reader = csv.reader(f)
                prompts = [row[0] for row in reader]
            assert len(prompts) == 200
        elif eval_mode == EvalMode.TGT_REALHARMFULQ:
            with open("data/real_harmfulq.csv", "r") as f:
                reader = csv.reader(f)
                prompts = [row[0] for row in reader]
            assert len(prompts) == 50
        elif eval_mode == EvalMode.TGT_WRITINGPROMPTS_LONG:
            with open("data/writing_prompts_50.txt", "r") as f:
                lines = f.readlines()
                template = lines[0].strip()
                prompts = [line.strip() for line in lines[1:]]
                prompts = [template.format(800, p) for p in prompts]
            print(len(prompts))
            assert len(prompts) == 50
        else:
            raise ValueError(f"Unknown eval mode {eval_mode}")
        return prompts

    def eval_targeted_spoofing(self, attacker: BaseAttacker, eval_mode: EvalMode) -> List[str]:
        assert isinstance(attacker, OurAttacker)
        print(
            f"Starting targeted eval on {eval_mode}: we want to pass the watermark and have low"
            " PPL and high GPT4 score",
            color="green",
        )

        # Get prompts
        prompts = self._get_targeted_prompts(eval_mode)

        if self.cfg.start_from_idx > 0:
            prompts = prompts[self.cfg.start_from_idx :]

        if self.run is not None:
            self.run["eval/dataset"] = eval_mode.value

        # Actually run and for now just print the results
        batch_size = self.cfg.batch_size
        nb_batches = math.ceil(len(prompts) / batch_size)
        results = []
        for b in range(nb_batches):
            print(f"Starting batch {b} of {nb_batches}", color="green")
            inputs = prompts[b * batch_size : min(len(prompts), (b + 1) * batch_size)]
            # If we are sampling don't reseed every time
            # As we really want diversity
            reseed = not attacker.model.cfg.use_sampling
            print(f"Reseeding: {reseed}")
            completions = attacker.generate(inputs, reseed=reseed)
            print("Generation done, getting quality metrics.")
            zests = [zest[-1] for zest in attacker.z_estimates]  # get last for each batch
            try:
                metrics = self.get_quality_metrics(inputs, completions, ppl_model=attacker.model)
                self._log_run(
                    EvalClass.SPOOFING,
                    b * batch_size,
                    inputs,
                    completions,
                    metrics,
                    extra_data={"zest": zests},
                )
            except Exception as e:
                print(e)
                print("Failed to get quality metrics for this batch.")
                metrics = None
            for i in range(len(inputs)):
                print(inputs[i], tag="Prompt", tag_color="red", color="white")
                print(completions[i], tag="Completion", tag_color="red", color="white")
                if metrics is not None:
                    zest = f"{zests[i]:.2f}"
                    metrics_str = metrics_to_str(metrics[i], z_estimate=zest)
                    results.append(metrics_str)
                    print(
                        f"idx: {b*batch_size+i} | {metrics_str}",
                        tag="Metrics",
                        tag_color="blue",
                        color="white",
                    )
                    if self.verbose and metrics[i].gpt4_explanation is not None:
                        print(
                            f"{metrics[i].gpt4_explanation}",
                            tag="GPT4 explanation",
                            tag_color="red",
                            color="white",
                        )
                else:
                    results.append("Skipped")
        print("All results:")
        return results

    def eval_targeted_spoofing_server(self, server: Server, eval_mode: EvalMode) -> List[str]:
        print(
            f"Starting targeted eval on {eval_mode}",
            color="green",
        )

        # Get prompts
        prompts = self._get_targeted_prompts(eval_mode)
        if self.run is not None:
            self.run["eval/dataset"] = eval_mode.value

        # Actually run and for now just print the results
        batch_size = self.cfg.batch_size
        nb_batches = math.ceil(len(prompts) / batch_size)
        results = []
        for b in range(nb_batches):
            print(f"Starting batch {b} of {nb_batches}", color="green")
            inputs = prompts[b * batch_size : min(len(prompts), (b + 1) * batch_size)]

            completions, _ = server.generate(inputs, disable_watermark=server.cfg.disable_watermark)

            print("Generation done, getting quality metrics.")
            try:
                metrics = self.get_quality_metrics(inputs, completions, ppl_model=server.model)
                self._log_run(
                    EvalClass.SERVER,
                    b * batch_size,
                    inputs,
                    completions,
                    metrics,
                    extra_data={},
                )
            except Exception as e:
                print(e)
                print("Failed to get quality metrics for this batch.")
                metrics = None
            for i in range(len(inputs)):
                print(inputs[i], tag="Prompt", tag_color="red", color="white")
                print(completions[i], tag="Completion", tag_color="red", color="white")
                if metrics is not None:
                    metrics_str = metrics_to_str(metrics[i])
                    results.append(metrics_str)
                    print(
                        f"idx: {b*batch_size+i} | {metrics_str}",
                        tag="Metrics",
                        tag_color="blue",
                        color="white",
                    )
                    if self.verbose and metrics[i].gpt4_explanation is not None:
                        print(
                            f"{metrics[i].gpt4_explanation}",
                            tag="GPT4 explanation",
                            tag_color="red",
                            color="white",
                        )
                else:
                    results.append("Skipped")
        print("All results:")
        return results

    # flake8: noqa: C901
    def eval_targeted_scrubbing_rubbingin(
        self, server: Server, attacker: BaseAttacker, eval_class: EvalClass, eval_mode: EvalMode
    ) -> List[str]:
        assert isinstance(attacker, OurAttacker)
        print(
            f"Starting {eval_class} eval on {eval_mode}",
            color="green",
        )

        # Get prompts
        prompts = self._get_targeted_prompts(eval_mode)
        if self.run is not None:
            self.run["eval/dataset"] = eval_mode.value

        if self.cfg.get_server_prompts_from is not None:
            # init neptune run with id
            if isinstance(self.cfg.get_server_prompts_from, str):
                server_run = neptune.init_run(
                    project=self.neptune_project,
                    with_id=self.cfg.get_server_prompts_from,
                    mode="read-only",
                )
                server_runs = [server_run]
            else:
                server_runs = []
                for run in self.cfg.get_server_prompts_from:
                    server_runs.append(
                        neptune.init_run(
                            project=self.neptune_project,
                            with_id=run,
                            mode="read-only",
                        )
                    )
                server_run = server_runs[0]
        else:
            server_run = None

        # Actually run and for now just print the results
        batch_size = 1
        offset = 0
        server_idx = 0
        nb_batches = math.ceil(len(prompts) // batch_size)
        results = []
        for b in range(nb_batches):
            print(f"Starting batch {b} of {nb_batches}", color="green")
            inputs = prompts[b * batch_size : min(len(prompts), (b + 1) * batch_size)]
            # If we are sampling don't reseed every time
            # As we really want diversity
            reseed = not attacker.model.cfg.use_sampling
            print(f"Reseeding: {reseed}")

            # Get texts
            if server_run is not None:
                if self.cfg.get_server_prompts_from == "WS-73":
                    texts_wm = [server_run[f"eval/{b}/server/completion"].fetch()]
                else:
                    try:
                        texts_wm = [server_run[f"eval/{b-offset}/completion"].fetch()]
                    except Exception:
                        # TODO Does not handle splits within a batch
                        print(f"Actually switched! {b} {offset} {b * batch_size}")
                        offset = b * batch_size
                        server_idx += 1
                        server_run = server_runs[server_idx]
                        texts_wm = [server_run[f"eval/{b-offset}/completion"].fetch()]

                print(f"Got server completion from neptune: {texts_wm} to prompt {inputs}")
            else:
                disable_wm = eval_class == EvalClass.RUBBINGIN
                texts_wm, _ = server.generate(inputs, disable_watermark=disable_wm)

            try:
                metrics_server = self.get_quality_metrics(
                    inputs, texts_wm, ppl_model=attacker.model
                )
                self._log_run(
                    eval_class, b * batch_size, inputs, texts_wm, metrics_server, key="server"
                )
            except Exception:
                print("Failed to get quality metrics for this batch.")
                metrics_server = None

            # Controls if baseline should also be run
            do = {
                "orig": (eval_class == EvalClass.SCRUBBING),
                "ours": (eval_class == EvalClass.SCRUBBING) and (not self.cfg.run_baseline_only),
            }

            # Orig inputs
            if "dipper" in attacker.cfg.model.name:
                initial_para_inputs = [f"{a} ||| {b}" for a, b in zip(inputs, texts_wm)]
            elif "openai" in attacker.cfg.model.name:
                initial_para_inputs = []
                for text_wm in texts_wm:
                    curr = [
                        {
                            "role": "system",
                            "content": "Paraphrase the given text carefully sentence by sentence. Make sure to keep the same length (around 1000 tokens). It is imperative not to shorten or summarize, simply paraphrase every sentence.",
                        },
                        {"role": "user", "content": text_wm},
                    ]
                    initial_para_inputs.append(curr)  # type: ignore
            else:
                initial_para_inputs = texts_wm

            if do["ours"]:
                para_inputs_ours = initial_para_inputs
                all_metrics_ours = []

            if do["orig"]:
                para_inputs_orig = initial_para_inputs
                all_metrics_orig = []

            for paraphrasing_iter in range(
                attacker.cfg.generation.recursive_iters
            ):  # Repeat recursively
                if do["ours"]:
                    print(
                        f"Starting paraphrasing iter {paraphrasing_iter}: ours",
                        color="green",
                    )
                    cfg_gen = attacker.cfg.generation.model_copy(deep=True)
                    completions_ours = attacker.generate(para_inputs_ours, cfg_gen, reseed=reseed)
                    zests = [zest[-1] for zest in attacker.z_estimates]  # get last for each batch

                if do["orig"]:
                    print(
                        f"Starting paraphrasing iter {paraphrasing_iter}: orig",
                        color="green",
                    )
                    cfg_gen = attacker.cfg.generation.model_copy(deep=True)
                    cfg_gen.spoofer_strength = 0
                    cfg_gen.repetition_penalty = 1
                    cfg_gen.use_graceful_conclusion = False
                    completions_orig = attacker.generate(para_inputs_orig, cfg_gen, reseed=reseed)
                    print("Generation done, getting quality metrics.")

                try:
                    if do["orig"]:
                        metrics_orig = self.get_quality_metrics(
                            initial_para_inputs,
                            completions_orig,
                            ppl_model=attacker.model,
                            supress_psp=False,
                        )
                        self._log_run(
                            eval_class,
                            b * batch_size,
                            initial_para_inputs,  # original prompt
                            completions_orig,
                            metrics_orig,
                            key=f"orig{paraphrasing_iter}",
                            extra_data={"actual_prompt": para_inputs_orig},
                        )
                    if do["ours"]:
                        metrics_ours = self.get_quality_metrics(
                            initial_para_inputs,
                            completions_ours,
                            ppl_model=attacker.model,
                            supress_psp=False,
                        )
                        self._log_run(
                            eval_class,
                            b * batch_size,
                            initial_para_inputs,  # original prompt
                            completions_ours,
                            metrics_ours,
                            key=f"ours{paraphrasing_iter}",
                            extra_data={"zest": zests, "actual_prompt": para_inputs_ours},
                        )
                except Exception:
                    print("Failed to get quality metrics for this batch.")
                    if do["ours"]:
                        metrics_ours = None
                    if do["orig"]:
                        metrics_orig = None
                if do["orig"]:
                    all_metrics_orig.append(metrics_orig)
                    if "dipper" in attacker.cfg.model.name:
                        para_inputs_orig = [
                            f"{a} ||| {b}" for a, b in zip(inputs, completions_orig)
                        ]
                    else:
                        para_inputs_orig = completions_orig
                if do["ours"]:
                    all_metrics_ours.append(metrics_ours)
                    if "dipper" in attacker.cfg.model.name:
                        para_inputs_ours = [
                            f"{a} ||| {b}" for a, b in zip(inputs, completions_ours)
                        ]
                    else:
                        para_inputs_ours = completions_ours

            # Output
            for i in range(len(inputs)):
                print(inputs[i], tag="Prompt", tag_color="red", color="white")
                print(texts_wm[i], tag="Completion", tag_color="red", color="white")
                if do["orig"]:
                    print(
                        completions_orig[i], tag="ParaOrig -- final", tag_color="red", color="white"
                    )
                if do["ours"]:
                    print(
                        completions_ours[i], tag="ParaOurs -- final", tag_color="red", color="white"
                    )

                s = ""
                for m in [metrics_server[i] if metrics_server is not None else None]:
                    s += metrics_to_str(m, z_estimate=None) + ";"
                if do["orig"]:
                    s += "orig;"
                    for m in [x[i] if x is not None else None for x in all_metrics_orig]:
                        s += metrics_to_str(m, z_estimate=None) + ";"
                if do["ours"]:
                    s += "ours;"
                    for m in [x[i] if x is not None else None for x in all_metrics_ours]:
                        s += metrics_to_str(m, z_estimate=None) + ";"

                results.append(s)
                print(
                    f"idx: {b*batch_size+i} | {s}",
                    tag="Metrics",
                    tag_color="blue",
                    color="white",
                )

                if (
                    self.verbose
                    and do["ours"]
                    and all_metrics_ours[0] is not None
                    and all_metrics_ours[0][i].gpt4_explanation is not None
                ):
                    print(
                        f"{all_metrics_ours[0][i].gpt4_explanation}",
                        tag="GPT4 explanation for our first para",
                        tag_color="red",
                        color="white",
                    )
        print("All results:")
        print(results)
        return results

    # Eval router
    def _run_eval(
        self, eval_class: EvalClass, eval_mode: EvalMode, server: Server, attacker: BaseAttacker
    ) -> List[str]:
        if eval_mode == EvalMode.PROXY:
            assert eval_class == EvalClass.SPOOFING  # not implemented
            results = self.eval_proxy(attacker)
        elif eval_mode == EvalMode.GARBAGE:
            assert eval_class == EvalClass.SPOOFING  # not implemented
            self.eval_garbage(attacker)
            results = []
        elif eval_mode == EvalMode.UNTARGETED:
            assert eval_class == EvalClass.SPOOFING  # not implemented
            self.eval_untargeted(attacker)
            results = []
        else:
            # Targeted eval
            if eval_class == EvalClass.SCRUBBING or eval_class == EvalClass.RUBBINGIN:
                results = self.eval_targeted_scrubbing_rubbingin(
                    server, attacker, eval_class, eval_mode
                )
            elif eval_class == EvalClass.SERVER:
                results = self.eval_targeted_spoofing_server(server, eval_mode)
            else:
                results = self.eval_targeted_spoofing(attacker, eval_mode)
        return results

    def run_eval(self, server: Server, attacker: BaseAttacker, out_dir: str) -> None:
        eval_class, eval_mode = self.cfg.eval_class, self.cfg.eval_mode
        results = self._run_eval(eval_class, eval_mode, server, attacker)
        with create_open(
            os.path.join(
                out_dir,
                f"{eval_class.value}_{eval_mode.value}_{attacker.cfg.generation.spoofer_strength}.txt",
            ),
            "a",
        ) as f:
            for r in results:
                f.write(f"{r}\n")
