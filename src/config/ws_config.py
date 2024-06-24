from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import Field, validator

from src.config.utils import PydanticBaseModelWithOptionalDefaultsPath as PBMwODP

"""
    Pre-detection normalizers from watermarks/kgw/normalizers.py
"""


class WatermarkDetectionNormalizer(Enum):
    UNICODE = "unicode"
    HOMOGLYPHS = "homoglyphs"
    TRUECASE = "truecase"


"""
    "our" = watermark stealing
        (use this)
    "sadasivan" = Can AI-Gen Text Be Reliably Detected?
        (evaluated before configs so does not integrate well with rest of the codebase)
"""


class AttackerAlgo(Enum):
    OUR = "our"
    SADASIVAN = "sadasivan"


"""
    Slow learning takes into account perplexity, fast does not
"""


class AttackerLearningMode(Enum):
    FAST = "fast"
    SLOW = "slow"


"""
    Type of attack we are evaluating (or a server run)
"""


class EvalClass(Enum):
    SPOOFING = "spoofing"
    SERVER = "server"
    SCRUBBING = "scrubbing"
    RUBBINGIN = "rubbingin"


"""
    Evaluation protocol, TGT = Targeted (only relevant)
"""


class EvalMode(Enum):
    PROXY = "proxy-greenlist-prediction"
    GARBAGE = "garbage"
    UNTARGETED = "untargeted"
    TGT_C4 = "c4realnews-val-10"
    TGT_MTBENCH = "mtbench-writing-10"
    TGT_GCG = "gcg-advbench-50"
    TGT_ESSAYS = "essays"
    TGT_DOLLY = "dolly-writing-100"
    TGT_HARMFULQ = "harmfulq-200"
    TGT_REALHARMFULQ = "realharmfulq-50"
    TGT_DOLLY_LONG = "dolly-writing-100-long"
    TGT_WRITINGPROMPTS_LONG = "writing-prompts-long"
    TGT_FAKE_NEWS = "fake-news"
    TGT_BOOK_REPORTS = "book-reports"


class EvalMetric(Enum):
    DETECTOR = "detector"
    PPL = "ppl"
    GPT4 = "gpt4"
    PSP = "psp"
    SELF = "self"


class WatermarkScheme(Enum):
    KGW = "kgw"


class SyspromptType(Enum):
    NONE = None
    STANDARD = "standard"
    PARAPHRASE = "paraphrase"


class MetaConfig(PBMwODP, extra="forbid"):  # type: ignore
    device: str = Field(..., description="Device to run on (cuda/cpu)")
    use_neptune: bool = Field(False, description="If neptune should be used")
    neptune_project: str = Field(
        "ORG/PROJ", description="Neptune project name; replace with your project if using neptune"
    )
    seed: int = Field(
        ..., description="Seed, applied before every unwatermarked/watermark generate call"
    )
    out_root_dir: str = Field(..., description="Directory to save outputs to")
    result_dir: str = Field(
        "results/default", description="Directory to save evaluation results to"
    )

    def short_str(self) -> str:
        return f"seed={self.seed}"


class ModelConfig(PBMwODP, arbitrary_types_allowed=True):  # type: ignore
    skip: bool = Field(..., description="If this model should be loaded or skipped for speed")
    name: str = Field(description="Name of the model (from hf or openai-)")
    use_fp16: bool = Field(False, description="Load and use in FP16 precision")
    use_flashattn2: bool = Field(False, description="Use flash attention")
    prompt_max_len: int = Field(None, description="Max length of the prompt")
    response_max_len: int = Field(None, description="Max length of the response")
    n_beams: int = Field(1, description="Number of beams for beam search")
    use_sampling: bool = Field(False, description="Use multinomial sampling instead of greedy")
    sampling_temp: float = Field(0.7, description="Temperature for multinomial sampling")

    def short_str(self) -> str:
        return f"name={self.name},n_beams={self.n_beams},sample={self.use_sampling},temp={self.sampling_temp}"


class WatermarkGenerationConfig(PBMwODP, extra="forbid"):  # type: ignore
    seeding_scheme: str = Field(..., description="Seeding scheme, see prf_schemes.py")
    gamma: float = Field(..., description="Fraction of green tokens")
    delta: float = Field(..., description="Logit boost for green tokens")


class WatermarkDetectionConfig(PBMwODP, extra="forbid"):  # type: ignore
    normalizers: List[WatermarkDetectionNormalizer] = Field(
        ..., description="Preprocessors/normalizers to apply"
    )
    ignore_repeated_ngrams: bool = Field(
        ..., description="If repetitions should be ignored when counting hits"
    )
    z_threshold: float = Field(..., description="Min z-score to consider a text watermarked")


class WatermarkConfig(PBMwODP, extra="forbid"):  # type: ignore
    scheme: WatermarkScheme = Field(..., description="Watermark scheme to use")
    generation: WatermarkGenerationConfig
    detection: WatermarkDetectionConfig


class ServerConfig(PBMwODP, extra="forbid"):  # type: ignore
    model: ModelConfig = Field(..., description="Model to use for the server")
    watermark: WatermarkConfig = Field(..., description="Watermark to use for the server")
    disable_watermark: bool = Field(False, description="Whether to disable the watermark")

    def short_str(self) -> str:
        return f"model=[{self.model.short_str()}],wm=[{self.watermark.scheme.value}]"


class AttackerQueryingConfig(PBMwODP, extra="forbid"):  # type: ignore
    skip: bool = Field(..., description="If the attacker should query the server")
    dataset: str = Field(..., description="Dataset to use for querying")
    batch_size: int = Field(..., description="Batch size to use for querying")
    start_from_batch_num: int = Field(..., description="Batch number to start querying from")
    end_at_batch_num: int = Field(..., description="Batch number to end querying at")
    apply_watermark: bool = Field(..., description="If the server should apply the watermark")


class AttackerLearningConfig(PBMwODP, extra="forbid"):  # type: ignore
    skip: bool = Field(
        ..., description="If the attacker should learn from loaded server completions"
    )
    mode: AttackerLearningMode = Field(..., description="Mode to use for learning")
    nb_queries: int = Field(..., description="How many queries to load from server completions")

    def short_str(self) -> str:
        return f"mode={self.mode.value},nb_queries={self.nb_queries}"


class AttackerGenerationConfig(PBMwODP, extra="forbid"):  # type: ignore
    spoofer_strength: float = Field(..., description="Strength of the spoofer (0 = off)")
    w_abcd: float = Field(..., description="Weight of the abcd spoofer")
    w_partials: float = Field(..., description="Weight of the partials spoofer")
    w_empty: float = Field(..., description="Weight of the empty spoofer")
    w_future: float = Field(..., description="Weight of the future ppl spoofer")
    min_wm_count_nonempty: int = Field(
        ...,
        description="Min data (wm counts) threshold for empty ctx",
    )
    min_wm_mass_empty: float = Field(
        ...,
        description="Min data (wm mass) threshold for non-empty ctx",
    )
    # Future params
    future_num_cands: int = Field(
        ..., description="Number of wm-nice candidates to propose to roberta per mask width"
    )
    future_num_options_per_fillin: int = Field(
        ..., description="Number of options to get from roberta per mask width and per position"
    )
    future_prefix_size: int = Field(..., description="Prefix size to give to roberta")
    future_local_w_decay: float = Field(..., description="A way to value bigger mask width less")

    panic_from: int = Field(..., description="Start panicking from this many tokens")
    repetition_penalty: float = Field(
        ..., description="Repetition penalty of the spoofer (1 = off)"
    )
    use_ft_counts: bool = Field(..., description="If the spoofer should use finetuning counts")
    use_graceful_conclusion: bool = Field(..., description="If we should use the GC processor")
    sysprompt: SyspromptType = Field(
        SyspromptType.NONE, description="If we append the standard spoofing sysprompt"
    )
    dipper_chunk: int = Field(3, description="Chunk size for dipper")
    dipper_lexdiv: int = Field(60, description="Lexdiv to use for dipper")
    dipper_orderdiv: int = Field(20, description="Orderdiv to use for dipper")
    recursive_iters: int = Field(1, description="Recursive paraphrasing sets >1")
    prevent_eos_if_zest_bad: bool = Field(..., description="If we should prevent EOS if zest bad")
    clip_at: float = Field(2.0, description="Clip ratios at this value")

    @validator("dipper_lexdiv")
    def valid_lexdiv(cls: Any, v: int) -> int:
        if v not in [0, 20, 40, 60, 80, 100]:
            raise ValueError(f"Lexdiv {v} not in [0, 20, 40, 60, 80, 100]")
        return v

    @validator("dipper_orderdiv")
    def valid_orderdiv(cls: Any, v: int) -> int:
        if v not in [0, 20, 40, 60, 80, 100]:
            raise ValueError(f"Orderdiv {v} not in [0, 20, 40, 60, 80, 100]")
        return v

    def short_str(self) -> str:
        return f"sp_str={self.spoofer_strength},w_abcd={self.w_abcd},w_p={self.w_partials},w_e={self.w_empty},w_f={self.w_future},rep_p={self.repetition_penalty},use_ftc={self.use_ft_counts}"


class AttackerConfig(PBMwODP, extra="forbid"):  # type: ignore
    algo: AttackerAlgo = Field(..., description="Algorithm to use for the attacker")
    model: ModelConfig = Field(..., description="Model to use for the attacker")
    querying: AttackerQueryingConfig = Field(..., description="Querying config for the attacker")
    learning: AttackerLearningConfig = Field(..., description="Learning config for the attacker")
    generation: AttackerGenerationConfig = Field(
        ..., description="Generation config for the attacker"
    )

    def short_str(self) -> str:
        # USe sampling
        return f"algo={self.algo.value},model=[{self.model.short_str()}],learn=[{self.learning.short_str()}],gen=[{self.generation.short_str()}]"


class GradioConfig(PBMwODP, extra="forbid"):  # type: ignore
    skip: bool = Field(..., description="If the app should be run (after all work is done)")
    make_public: bool = Field(..., description="If the app should be public")
    port: int = Field(..., description="Port to run the app on")
    default_prompt: str = Field(..., description="Default prompt to use for the gradio app")


class EvaluatorConfig(PBMwODP, extra="forbid"):  # type: ignore
    skip: bool = Field(False, description="If the evaluator should be run")
    get_server_prompts_from: Optional[Union[str, List[str]]] = Field(
        None, description="Neptune run id"
    )
    run_baseline_only: bool = Field(
        False, description="If scrubbing should run base+ours or just base"
    )
    batch_size: int = Field(1, description="Batch size to use for evaluation")
    metrics: List[EvalMetric] = Field(..., description="Metrics to use (also in gradio)")
    eval_class: EvalClass = Field(..., description="Eval class to run -- spoofing or scrubbing")
    eval_mode: EvalMode = Field(..., description="Eval to run -- mostly targeted dataset")
    start_from_idx: int = Field(-1, description="Idx to start from")

    def short_str(self) -> str:
        return "met=" + "-".join([x.value for x in self.metrics])


class WsConfig(PBMwODP):
    meta: MetaConfig
    gradio: GradioConfig = Field(..., description="Gradio config")
    server: ServerConfig
    attacker: AttackerConfig
    evaluator: EvaluatorConfig

    def get_result_path(self) -> str:
        return f"{self.meta.result_dir}/att=[{self.attacker.short_str()}],serv=[{self.server.short_str()}],meta=[{self.meta.short_str()}],eval=[{self.evaluator.short_str()}]"
