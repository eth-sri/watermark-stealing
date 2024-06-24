from typing import Any, List, Optional, Tuple

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    LlamaTokenizerFast,
    LogitsProcessorList,
    PegasusForConditionalGeneration,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    TemperatureLogitsWarper,
)

from src.config import MetaConfig, ModelConfig
from src.models.utils import (
    LogitInfo,
    QueryCost,
    TokenCandidate,
    is_chat_model,
    is_decoder_only_model,
    is_seq2seq_model,
)
from src.utils import ProgressLogger, print


class HfModel:
    def __init__(self, meta_cfg: MetaConfig, model_cfg: ModelConfig) -> None:
        self.device, self.seed = meta_cfg.device, meta_cfg.seed
        self.cfg = model_cfg
        self.model: Optional[AutoModel] = None
        self.end_inst_tag = "[/INST]"
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer()
        if not self.cfg.skip:
            self._load_model()

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        # Load tokenizer
        if "llama" in self.cfg.name:
            tokenizer = LlamaTokenizerFast.from_pretrained(
                self.cfg.name, torch_dtype=torch.float16, padding_side="left"
            )
            tokenizer.pad_token = tokenizer.eos_token
        elif "dipper" in self.cfg.name:
            tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
            # Used to be T5Tokenizer but we need this for offset_mapping
        elif "pegasus" in self.cfg.name:
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.name, padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.period_token_id = tokenizer("Game over.", add_special_tokens=False).input_ids[-1]

        return tokenizer

    def _load_model(self) -> None:
        # Load LM
        if "pegasus" in self.cfg.name:
            self.model = PegasusForConditionalGeneration.from_pretrained(self.cfg.name)
        elif is_seq2seq_model(self.cfg.name):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.name)
        elif is_decoder_only_model(self.cfg.name):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.name,
                torch_dtype=torch.float16 if self.cfg.use_fp16 else torch.float32,
                use_flash_attention_2=self.cfg.use_flashattn2,
                device_map="auto",
            )
        elif "dipper" in self.cfg.name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.cfg.name)
        else:
            raise ValueError(f"Unknown model name: {self.cfg.name}")

        if self.device == "cuda":
            self.model.to("cuda")
        self.model.eval()

    def _chatify(self, prompt: str, completion: Optional[str] = None) -> str:
        if completion is not None:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )

    def _encode_batch(
        self,
        prompts: List[str],
        completions: Optional[List[str]] = None,
        prevent_truncation: bool = False,
        return_offsets: bool = False,
        return_inputs: bool = False,
    ) -> BatchEncoding:
        # Tokenize
        if is_chat_model(self.cfg.name):
            if completions is not None:
                inputs = [self._chatify(p, c) for p, c in zip(prompts, completions)]
            else:
                inputs = [self._chatify(p) for p in prompts]

            # Some templates are wrong (e.g., mistral) and add a trailing space
            inputs = [i.strip() for i in inputs]

            add_special_tokens = False  # already done by chatify
        else:
            inputs = prompts
            add_special_tokens = True

        batchenc: BatchEncoding = self.tokenizer(
            inputs,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            truncation=not prevent_truncation,
            padding=len(inputs) > 1,
            max_length=self.cfg.prompt_max_len,
            return_offsets_mapping=return_offsets,
        ).to(self.device)

        if batchenc["input_ids"].shape[-1] == self.cfg.prompt_max_len:
            print(
                f"At least one prompt was truncated to {self.cfg.prompt_max_len}",
                tag="!!!",
                color="red",
                tag_color="red",
            )
        if return_inputs:
            # Needed for PPL computation
            return inputs, batchenc
        else:
            return batchenc

    def generate(
        self,
        prompts: List[str],
        logit_processors: LogitsProcessorList,
        reseed: bool = True,
        return_model_inputs: bool = False,
    ) -> Any:
        if self.model is None:
            raise RuntimeError("Call _load_model before using the model")
        inputs, batchenc = self._encode_batch(prompts, return_offsets=True, return_inputs=True)
        print(f"Model Input: {prompts} with length {batchenc['input_ids'].shape[1]}", color="blue")

        actual_inputs = self._get_actual_input_str(inputs, batchenc)
        if hasattr(batchenc, "offset_mapping"):
            del batchenc["offset_mapping"]
        if reseed:
            torch.manual_seed(self.seed)

        ProgressLogger.start("Calling model.generate")
        print(batchenc["input_ids"].shape)
        completions = self.model.generate(
            **batchenc,
            max_new_tokens=self.cfg.response_max_len,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=self.cfg.n_beams,
            do_sample=self.cfg.use_sampling,
            temperature=self.cfg.sampling_temp,
            logits_processor=logit_processors,
        )
        ProgressLogger.stop()

        if is_decoder_only_model(self.cfg.name):
            completions = completions[:, batchenc["input_ids"].shape[-1] :]

        if completions.shape[-1] == self.cfg.response_max_len:
            print(
                f"At least one response seems to be truncated to {self.cfg.response_max_len}",
                tag="!!!",
                color="red",
                tag_color="red",
            )

        completions_str: List[str] = self.tokenizer.batch_decode(
            completions, skip_special_tokens=True
        )
        print(f"Model Output: {completions_str} with shape {completions.shape}", color="purple")

        # Compute cost and return
        cost = QueryCost(batchenc["input_ids"].numel(), completions.numel())
        if return_model_inputs:
            return completions_str, cost, actual_inputs
        return completions_str, cost

    def _get_actual_input_str(self, inputs: List[str], batchenc: BatchEncoding) -> List[str]:
        last_idx = -1 if "dipper" not in self.cfg.name else -2
        idxs = batchenc["offset_mapping"][
            :, last_idx, 1
        ]  # cutoff was here - Location of last token in the input_text
        actual_inputs = []
        for inp, idx in zip(inputs, idxs):
            if idx < len(inp):
                print(
                    f"Cutting is indeed done: (in text character lengths) idx={idx} and"
                    f" len(inp)={len(inp)}"
                )
            inp = inp[: min(len(inp) - 8, idx)]
            inp = inp[10:]

            actual_inputs.append(inp)

        return actual_inputs

    def get_ppls_and_logitinfo(
        self, prompts: List[str], completions: List[str], logit_processors: LogitsProcessorList
    ) -> Tuple[torch.Tensor, List[List[LogitInfo]]]:
        if self.model is None:
            raise RuntimeError("Call _load_model before using the model")
        if "dipper" in self.cfg.name:
            raise ValueError("We dont do ppl with DIPPER")
        batch_size = len(prompts)
        if batch_size > 1:
            raise NotImplementedError("Batched PPL seems to leak memory, do only B=1 for now")

        if not is_chat_model(self.cfg.name):
            raise NotImplementedError("PPL computation only implemented for chat models")

        with torch.no_grad():
            bsz = len(prompts)
            convos, batchenc = self._encode_batch(
                prompts,
                completions,
                prevent_truncation=True,
                return_offsets=True,
                return_inputs=True,
            )
            input_ids = batchenc["input_ids"]  # (B, maxlen)
            offset_mapping = batchenc["offset_mapping"]  # (B, maxlen, 2)

            # Find start indices of completion parts (in token indexing)
            labels = input_ids.clone().detach()
            completion_mask = torch.ones_like(labels).bool()

            len_assistant_tag = len(self.end_inst_tag)
            for b in range(batch_size):
                text_idx_completion_start = convos[b].find(self.end_inst_tag) + len_assistant_tag
                idx_completion_start = (
                    (offset_mapping[b][:, 0] >= text_idx_completion_start).nonzero()[0, 0].item()
                )
                completion_mask[b, :idx_completion_start] = 0
            labels[~completion_mask] = -100

            # Get loss and logits, processors need to be applied manually!
            # (B, maxlen, vocab_size)
            raw_logits: torch.Tensor = self.model(input_ids=input_ids, labels=labels).logits
            logit_list = []
            for i in range(raw_logits.shape[1]):
                if i < idx_completion_start - 1 or logit_processors is None:
                    # Not our response so doesn't even matter, will be dropped anyways
                    # Or no logit processors
                    logit_list.append(raw_logits[:, i, :])
                else:
                    logit_list.append(logit_processors(input_ids[:, : i + 1], raw_logits[:, i, :]))
            logits = torch.stack(logit_list, dim=1)  # again (B, maxlen, vocab_size)
            assert raw_logits.shape == logits.shape

            # Manually compute the PPL for each batch element, as there's no way to get this from HF API
            # NOTE: last logit is for unseen next token and first label is for first token so drop them and align
            shift_logits = (
                logits[..., :-1, :].contiguous().permute((0, 2, 1))
            )  # (B, vocab_size, maxlen-1)
            shift_labels = labels[..., 1:].contiguous()  # (B, maxlen-1)
            shift_completion_mask = completion_mask[..., 1:].contiguous()  # (B, maxlen-1)

            ce = torch.nn.CrossEntropyLoss(reduction="none")
            ces = ce(shift_logits, shift_labels)  # (B, maxlen-1)
            ces[~shift_completion_mask] = 0  # should already be true
            avg_ce_losses = ces.sum(dim=1) / shift_completion_mask.sum(dim=1)  # (B)
            ppls = torch.exp(avg_ce_losses)  # (B)
            # loss = (ces.sum() / shift_completion_mask.sum()).item()

            # Get logit debugging info
            # For each token: get top10 sorted decoded logits
            top_logits, top_tok_ids = torch.topk(logits, 10, dim=2, sorted=True)  # (B, maxlen, 10)
            batch_logitinfos = []
            for b in range(bsz):
                logitinfos = []
                # Go through all tokens
                for i in range(input_ids.shape[1]):
                    if completion_mask[b, i] == 0:
                        continue  # Skip the instruction part

                    # Get probs
                    # NOTE: assumes the PPL model uses just temperature (no topk/topp/beamsearch)
                    warper = TemperatureLogitsWarper(self.cfg.sampling_temp)
                    probs_full = warper(input_ids[b, :i], logits[b, i - 1, :]).softmax(0)

                    prev_tok = self.tokenizer.decode(input_ids[b, i - 1])
                    curr_tok = self.tokenizer.decode(input_ids[b, i])
                    curr_tok_id = input_ids[b, i].cpu().item()
                    candidates = []
                    for k in range(10):
                        cand = self.tokenizer.decode(top_tok_ids[b, i - 1, k]).replace(
                            "\\", r"\\\\"
                        )
                        cand_id = top_tok_ids[b, i - 1, k].cpu().item()
                        # TODO: mypy doesn't know that torch.topk(.)[1] is LongTensor and even if
                        # we manually assert that t is LongTensor, t[0] again becomes Tensor
                        cand_id = int(cand_id)  # was int anyways, just for mypy

                        logit = top_logits[b, i - 1, k].cpu().item()
                        prob = probs_full[cand_id].cpu().item()
                        # NOTE: cand is in because warper is monotone
                        candidates.append(TokenCandidate(cand, cand_id, logit, prob))
                    logitinfos.append(
                        LogitInfo(
                            prev_tok, curr_tok, curr_tok_id, candidates, ces[b, i - 1].cpu().item()
                        )
                    )
                batch_logitinfos.append(logitinfos)
            return ppls.cpu(), batch_logitinfos
