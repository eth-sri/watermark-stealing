import re
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

import gradio as gr
from src.attackers import BaseAttacker, OurAttacker
from src.config import SyspromptType, WsConfig
from src.evaluator import Evaluator, QualityMetricValues
from src.models import LogitInfo, fix_isolated_punctuation
from src.server import Server
from src.utils import ProgressLogger
from src.watermarks import KgwWatermark
from src.watermarks.kgw import hashint  # type: ignore

"""
    Experimental debug UI, no guarantees
"""


def run_gradio(cfg: WsConfig, server: Server, attacker: BaseAttacker, evaluator: Evaluator) -> None:
    demo = _build_ui(cfg, server, attacker, evaluator)
    demo.queue()
    demo.launch(share=cfg.gradio.make_public, max_threads=5, server_port=cfg.gradio.port)


# UI + Hooks (Callbacks are below)
def _build_ui(  # noqa: C901
    cfg: WsConfig, server: Server, attacker: BaseAttacker, evaluator: Evaluator
) -> gr.Blocks:
    if not isinstance(attacker, OurAttacker):
        raise RuntimeError(f"Gradio only works for OurAttacker not {type(attacker)}")
    if not isinstance(server.watermarks[0], KgwWatermark):
        raise RuntimeError(f"Gradio only works for KGW not {type(server.watermarks[0])}")

    nb_metric_rows = 7  # NOTE this probably does not matter?

    css = "#title {text-align: center; margin: auto; width: 50%}"
    with gr.Blocks(css=css, theme=gr.themes.Glass(), title="💧🤏😈") as demo:
        with gr.Row():
            title = """## 💧🤏😈 <span style="color:#c20eb7"> Stealing LLM Watermarks </span>"""
            gr.Markdown(title, visible=True, elem_id="title")

        # NOTE: these should be State but State does not have change handlers so it's invisible md
        srvr_nowm_output_state = gr.Markdown(value="", visible=False)
        srvr_wm_output_state = gr.Markdown(value="", visible=False)
        att_output_state = gr.Markdown(value="", visible=False)

        with gr.Row():
            with gr.Column(scale=1):
                # Logit debugger + exploring counts
                with gr.Tab("LogitDebugger"):
                    gr.Markdown("Visualize logit info for the current attacker generation.")
                    dbg_info = gr.State(value={})
                    dbg_slider = gr.Slider(label="Token", minimum=0, maximum=1000, step=1, value=0)
                    with gr.Row():
                        dbg_prevtok = gr.Textbox(
                            label="Prev. Token", interactive=False, lines=1, max_lines=1
                        )
                        dbg_currtok = gr.Textbox(
                            label="Sampled Token", interactive=False, lines=1, max_lines=1
                        )
                        dbg_CE = gr.Textbox(
                            label="CE(Sampled Token)", interactive=False, lines=1, max_lines=1
                        )
                    dbg_barplots = [gr.Plot() for _ in range(4)]
                    dbg_infocontainers = dbg_barplots + [dbg_prevtok, dbg_currtok, dbg_CE]  # !

                # Server generations
                with gr.Tab("Server"):
                    cfg_wmgen = cfg.server.watermark.generation
                    gr.Markdown(
                        f"LM: {cfg.server.model.name} {'(float16 mode)' if cfg.server.model.use_fp16 else ''} |"
                        f" SCHEME: {cfg.server.watermark.scheme} with"
                        f" {cfg_wmgen.seeding_scheme} and gamma={cfg_wmgen.gamma} and"
                        f" delta={cfg_wmgen.delta}"
                    )
                    with gr.Row():
                        srvr_prompt = gr.Textbox(
                            label="Prompt (may get truncated -- see terminal)",
                            interactive=True,
                            lines=5,
                            max_lines=5,
                            value=cfg.gradio.default_prompt,
                        )
                    with gr.Row():
                        srvr_btn = gr.Button("Generate")
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Tab("Watermarked Model"):
                                srvr_wm_output = gr.Markdown(
                                    label="Output With Watermark (Colored)"
                                )
                        with gr.Column(scale=1):
                            srvr_wm_metrics = gr.Dataframe(
                                headers=["Metric", "Value"],
                                interactive=False,
                                row_count=nb_metric_rows,
                                col_count=2,
                            )
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Tab("(Base Model - No Watermark)"):
                                srvr_nowm_output = gr.Markdown(
                                    label="Output Without Watermark (Colored)"
                                )
                        with gr.Column(scale=1):
                            srvr_nowm_metrics = gr.Dataframe(
                                headers=["Metric", "Value"],
                                interactive=False,
                                row_count=nb_metric_rows,
                                col_count=2,
                            )
                    srvr_gencontainers = [
                        srvr_nowm_output_state,
                        srvr_wm_output_state,
                        srvr_nowm_output,
                        srvr_wm_output,
                    ]  # !
                with gr.Tab("DetectorOnly"):
                    with gr.Row():
                        detectoronly_prompt = gr.Textbox(
                            label="Prompt",
                            interactive=True,
                            lines=5,
                            value="Some text",
                        )
                    with gr.Row():
                        detectoronly_text = gr.Textbox(
                            label="Text",
                            interactive=True,
                            lines=5,
                            value="Some text",
                        )
                    with gr.Row():
                        detectoronly_detect_btn = gr.Button("Detect")
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Tab("Highlighted Text"):
                                detectoronly_text_colored = gr.Markdown(label="Text (Colored)")
                        with gr.Column(scale=1):
                            detectoronly_metrics = gr.Dataframe(
                                headers=["Metric", "Value"],
                                interactive=False,
                                row_count=nb_metric_rows,
                                col_count=2,
                            )

            # Attacker side
            with gr.Column(scale=1):
                with gr.Tab("Attacker"):
                    gr.Markdown(
                        "Attacker LM:"
                        f" {cfg.attacker.model.name} {'(float16 mode)' if cfg.attacker.model.use_fp16 else ''}"
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            att_ft_get = gr.Button("Get FT counts", min_width=100, visible=False)
                            att_ft_clear = gr.Button(
                                "Clear FT counts", min_width=100, visible=False
                            )
                        with gr.Column(scale=4):
                            with gr.Row():
                                # This generally does not change after init
                                _ = gr.Number(
                                    label="Corpus Sz (sum counts)",
                                    value=attacker.get_corpus_size(False),
                                    interactive=False,
                                    min_width=100,
                                    visible=False,
                                )
                                # This can change from 0
                                att_ftcorpus_sz = gr.Number(
                                    label="FT Corpus Sz (sum counts)",
                                    value=attacker.get_corpus_size(True),
                                    interactive=False,
                                    min_width=100,
                                    visible=False,
                                )

                    # Attacker model
                    with gr.Row():
                        att_prompt = gr.Textbox(
                            label="Prompt",
                            interactive=True,
                            lines=5,
                            max_lines=100,
                            value=cfg.gradio.default_prompt,
                        )
                    with gr.Row():
                        att_strength = gr.Number(label="Strength", value=0.0, interactive=True)
                        att_reppenalty = gr.Number(
                            label="Rep. Penalty", value=1.0, interactive=True
                        )
                        att_use_ft_counts = gr.Checkbox(
                            label="Use FT counts", value=True, interactive=True, visible=False
                        )
                        with gr.Column():
                            att_sysprompt = gr.Radio(
                                [None, "standard", "paraphrase"], label="Sysprompt"
                            )
                            att_use_graceful_conclusion = gr.Checkbox(
                                label="Use Graceful Conclusion", value=False, interactive=True
                            )
                    with gr.Row():
                        att_w_future = gr.Number(
                            label="w_FUTURE", value=0.0, interactive=True, visible=False
                        )
                    with gr.Row():
                        att_future_num_cands = gr.Number(
                            label="F-num_cands", value=5, interactive=True, visible=False
                        )
                        att_future_num_options = gr.Number(
                            label="F-num_options", value=10, interactive=True, visible=False
                        )
                        att_future_prefix_size = gr.Number(
                            label="F-prefix_size", value=10, interactive=True, visible=False
                        )
                        att_future_local_w_decay = gr.Number(
                            label="F-w_decay", value=0.9, interactive=True, visible=False
                        )

                    att_params = [
                        att_prompt,
                        att_strength,
                        att_reppenalty,
                        att_use_ft_counts,
                        att_use_graceful_conclusion,
                        att_sysprompt,
                        att_w_future,
                        att_future_num_cands,
                        att_future_num_options,
                        att_future_prefix_size,
                        att_future_local_w_decay,
                    ]  # !
                    with gr.Row():
                        att_btn = gr.Button("Generate")
                        att_populate_logitdebugger = gr.Checkbox(
                            label="Logitdebugger?", value=False, interactive=True
                        )
                        att_running_Z = gr.Checkbox(
                            label="Running Z?", value=False, interactive=True
                        )

                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Tab("Attacker Model"):
                                att_output = gr.Markdown(label="Attacker's Output (Colored)")
                        with gr.Column(scale=1):
                            att_metrics = gr.Dataframe(
                                headers=["Metric", "Value"],
                                interactive=False,
                                row_count=nb_metric_rows,
                                col_count=2,
                            )
                            att_gpt4_explanation = gr.Markdown(label="GPT4 judgement")

        with gr.Row():
            gr.Markdown(
                "Explore attacker's knowledge. (use * for wildcards in Ordered mode)", visible=True
            )
        with gr.Row():
            inspect_token = gr.Textbox(
                label="Token(s) to inspect",
                interactive=True,
                lines=1,
                max_lines=1,
                value="The",
            )
            inspect_k = gr.Number(label="K", value=10, interactive=True)
            inspect_col_sort = gr.Number(label="Col# to Sort", value=2, interactive=True)
            inspect_btn = gr.Button("Inspect")
            inspect_ordered = gr.Checkbox(label="Ordered?", value=True, interactive=True)

        with gr.Row():
            headers: List[Any] = []
            inspect_result = gr.Dataframe(
                headers=headers, interactive=False, row_count=10, col_count=len(headers)
            )
        with gr.Row():
            inspect_token2 = gr.Textbox(
                label="Token(s) to inspect",
                interactive=True,
                lines=1,
                max_lines=1,
                value="The",
            )
            inspect_k2 = gr.Number(label="K", value=10, interactive=True)
            inspect_col_sort2 = gr.Number(label="Col# to Sort", value=8, interactive=True)
            inspect_btn2 = gr.Button("Inspect")
            inspect_ordered2 = gr.Checkbox(label="Ordered?", value=True, interactive=True)
        with gr.Row():
            headers2: List[Any] = []
            inspect_result2 = gr.Dataframe(
                headers=headers2, interactive=False, row_count=10, col_count=len(headers2)
            )
        #####
        #####
        #####
        # Helpers for callbacks
        #####
        #####
        #####

        def _color_text(
            text: str, token_mask: List[int], offset_mapping: List[Tuple[int, int]], z_at_T: Any
        ) -> str:
            colors = {
                "-1": ["#000000", "#777777"],  # black
                "0": ["#FF0000", "#880000"],  # red
                "1": ["#008000", "#03ad55"],  # green
            }
            colored = ""
            for idx in range(len(token_mask)):
                span, mask = offset_mapping[idx], token_mask[idx]
                piece = text[span[0] : span[1]].replace("\n", "\\n<br>")
                color = colors[str(mask)][idx % 2]
                colored += f'<span style="color:{color}">{piece}</span>'
                if z_at_T is not None and idx >= 3:
                    zest = attacker.z_estimates[0]
                    colored += f"[{z_at_T[idx-3]:.1f};{zest[idx]:.1f}]"

            # Also temporarily plot this so we can see
            if z_at_T is not None:
                plt.clf()
                plt.plot(np.arange(1, 1 + len(zest[3:-1])), zest[3:-1], label="estimate")
                plt.plot(np.arange(1, 1 + z_at_T.shape[0]), z_at_T, label="real")
                plt.legend()
                plt.savefig(f"zest_{z_at_T[-1]:.2f}.png")

            return colored

        def _get_formatted_metric_dataframe(metrics: QualityMetricValues) -> List[Any]:
            dr = metrics.detector_result
            assert dr is not None
            formatted = [
                ["Toks Counted (T)", int(dr["num_tokens_scored"])],
                ["#Toks in Greenlist", int(dr["num_green_tokens"])],
                ["% in Greenlist", f"{dr['green_fraction']:.1%}"],
                ["z-score", f"{dr['z_score']:.2f}"],
                ["p-value", f"{dr['p_value']:0.3e}"],
                ["Prediction", "Watermarked" if dr["prediction"] else "Human"],
                ["Confidence", f"{dr['confidence']*100:.2f}%" if dr["prediction"] else "/"],
                ["PPL", f"{metrics.ppl:.2f}" if metrics.ppl is not None else "[skipped]"],
                [
                    "GPT-4 Grade",
                    int(metrics.gpt4_grade) if metrics.gpt4_grade is not None else "[skipped]",
                ],
                ["PSP", f"{metrics.psp:.4f}" if metrics.psp is not None else "[skipped]"],
            ]
            return formatted  # 10 rows

        def _get_colored_text_and_metric_df(
            prompt: str, raw_output: str, get_gpt4_exp: bool = False, running_z: bool = False
        ) -> Any:
            metrics = evaluator.get_quality_metrics(
                [prompt], [raw_output], ppl_model=attacker.model, supress_psp=False
            )[0]
            if metrics.detector_result is not None:
                colored_text = _color_text(
                    raw_output,
                    metrics.detector_result["token_mask"],
                    metrics.detector_result["offset_mapping"],
                    metrics.detector_result["z_score_at_T"] if running_z else None,
                )
                metric_df = _get_formatted_metric_dataframe(metrics)
                gpt4_exp = metrics.gpt4_explanation
            else:
                colored_text = raw_output
                metric_df = [[]]
                gpt4_exp = "/"
            if get_gpt4_exp:
                return colored_text, metric_df, gpt4_exp
            else:
                return colored_text, metric_df

        #####
        # Callbacks
        #####

        # When the debugger slider changes underline the correct token
        # Returns: raw_output again
        def on_dbg_slider_change(tok_idx: int, raw_output: str) -> Any:
            marker = "text-decoration:solid underline black 4px;"
            raw_output = raw_output.replace(marker, "")  # Remove old
            span = list(re.finditer('<span style="', raw_output))[tok_idx].span()
            return raw_output[: span[1]] + marker + raw_output[span[1] :]

        # (Hack) Whenever the att. output changes [update the barplots]
        # Returns: all plots, prevtok, currtok, CE
        def on_att_output_change(dbg_info: dict[str, List[LogitInfo]], tok_idx: int) -> Any:
            # Prepare an array of colors, 1 for each logit
            # Also hash tokens to indices to consistenly assign colors
            all_colors = np.concatenate([mpl.cm.tab20(np.arange(20)), mpl.cm.tab20b(np.arange(20))])  # type: ignore
            np.random.RandomState(0).shuffle(all_colors)
            MAX_TOK_HASH = -1
            tok_hash = {}

            if len(dbg_info) == 0:
                return gr.Plot(), gr.Plot(), gr.Plot(), gr.Plot(), "/", "/", "/"

            # Build the barplots, 1 info at a time
            barplots = []
            for k in [
                "repetitionPenalty+spoofer+future (FULL)",
                "repetitionPenalty+spoofer",
                "spoofer",
                "base",
            ]:
                info = dbg_info[k][tok_idx]  # Focus only on tok_idx
                fig, ax = plt.subplots(figsize=(9, 3))

                # Get a color for each candidate
                candidates_raw = [cand.tok for cand in info.candidates]
                colors = []
                for tok in candidates_raw:
                    if tok not in tok_hash:
                        MAX_TOK_HASH += 1
                        tok_hash[tok] = MAX_TOK_HASH
                    colors.append(all_colors[tok_hash[tok]])

                # Put hatch only on the sampled one
                hatch = ["*" if tok == info.curr_tok else None for tok in candidates_raw]

                # Get height (based on logit)
                logits = [cand.logit for cand in info.candidates]
                if min(logits) > 0:
                    bottom = 0.0
                    heights = logits
                else:
                    bottom = min(logits) - 5
                    heights = [s - min(logits) + 5 for s in logits]

                # Actually plot
                ax.bar(
                    x=candidates_raw,
                    height=heights,
                    bottom=bottom,
                    align="center",
                    color=colors,
                    hatch=hatch,
                )
                ax.set_ylabel("logit")
                ax.tick_params(axis="x", labelrotation=45)
                ax.set_title(f"{k}")
                fig.tight_layout()
                barplots.append(fig)

                # If this is the full run, also look at probabilities
                """
                if k == "full":
                    fig, ax = plt.subplots(figsize=(9, 3))
                    probs = [cand.prob for cand in info.candidates]
                    ax.bar(
                        x=candidates_raw,
                        height=probs,
                        bottom=0,
                        align="center",
                        color=colors,
                        hatch=hatch,
                    )
                    ax.set_ylabel("probability")
                    ax.tick_params(axis="x", labelrotation=45)
                    ax.set_title(f"{k} -- probs")
                    fig.tight_layout()
                    barplots.append(fig)
                """
                plt.close()

            # Return all (info is the last one but doesn't matter as all same)
            return barplots + [info.prev_tok, info.curr_tok, f"{info.ce:.2f}"]

        # When the inspect button is clicked, update the inspect result
        # Returns: inspect dataframe (list of lists)
        def on_inspect_btn_click(ctx_text: str, k: float, col_sort: int, ordered: bool) -> Any:
            k_int = int(k)  # for some reason gradio sends a float here
            try:
                topk = attacker.get_topk_by_boost(ctx_text, k_int, ordered)
            except ValueError as e:
                return [["Tokenization Error", f"{e}"]]

            if len(topk) == 0:
                return [["No Results", ""]]
            assert isinstance(server.watermarks[0], KgwWatermark)  # ...
            greenness = server.watermarks[0].get_greenness_dict(ctx_text, [r[1] for r in topk])

            # Needed for later
            ctx = attacker.model.tokenizer([ctx_text], add_special_tokens=False)["input_ids"][0]
            ctx = [fix_isolated_punctuation(attacker.model.cfg.name, t) for t in ctx]
            if not ordered:
                ctx = sorted(ctx)
            else:
                ctx = [-1 if c == 398 else c for c in ctx]  # stars -> wildcards
            ctx = tuple(ctx)

            # Build rows
            rows: List[List[Any]] = []
            nb_green = 0
            nb_red = 0
            for idx, topk_entry in enumerate(topk):
                tok_text, tok, boost, count_base, count_wm, count_wm_ft = topk_entry
                row = [tok, tok_text, boost, count_base, count_wm, count_wm_ft]
                if greenness is None:
                    row.append("❓")
                    row.append("/")
                else:
                    if greenness[tok]:
                        row.append("✅")
                        nb_green += 1
                    else:
                        row.append("❌")
                        nb_red += 1
                    row.append(nb_green / (nb_green + nb_red))

                total_base = sum(list(attacker.counts_base.get(ctx, ordered=ordered).values()))
                total_wm = sum(list(attacker.counts_wm.get(ctx, ordered=ordered).values()))
                mass_base = count_base / total_base
                mass_wm = count_wm / total_wm

                # Compute ratio
                if (ctx_text == "" and count_wm < 100) or (ctx_text != "" and count_wm < 2):
                    ratio = 0.0  # not enough data to be sure (or even division by zero)
                elif mass_base == 0:
                    ratio = 1e9  # tmp, will be set to max
                else:
                    ratio = mass_wm / mass_base
                diff = mass_wm - mass_base
                row.extend([mass_base, mass_wm, ratio, diff])
                rows.append(row)

            # fix maxratio to be equal to max if 1e9
            ratios = [float(r[10]) for r in rows if float(r[10]) < 1e9]
            maxratio = max(ratios)
            for i in range(len(rows)):
                if rows[i][10] == 1e9:
                    rows[i][10] = maxratio + 1e-6  # fix

            rows = sorted(rows, key=lambda v: v[int(col_sort) - 1], reverse=True)

            if ctx_text == "":
                # Compute avg below
                avg = 0
                FINAL = 0
                avgprob = 0
                isg_avg = 0
                for i in range(len(rows)):
                    hes = hashint(torch.tensor(rows[i][0])).item()
                    rows[i].append(f"{hes:_}")

                    avg += hes
                    avgprob += (1 - hes / 1000000) ** 3  # p^ 3
                    rows[i].append(f"{avg // (i+1):_}")
                    rows[i].append(f"{avgprob / (i+1):.2f}")

                    processor = server.watermarks[0].spawn_logits_processor()
                    tok = rows[i][0]
                    isg = tok in processor._get_greenlist_ids(
                        torch.tensor([270, 270, 270, tok], device="cuda")  # 999887, "at"
                    )
                    isg_avg += isg
                    rows[i].append(f"{isg} (avg={isg_avg/(i+1):.2f})")

                    p = 1 - hes / 1000000
                    final = p**3 * int(isg) + (1 - p**3) * 0.25
                    rows[i].append(f"{final:.2f}")
                    FINAL += final
                    rows[i].append(f"{FINAL / (i+1):.2f}")

            # Add idx from bot
            for i in range(len(rows)):
                nw = [f"#{i + 1}"]
                nw.extend(rows[i])
                rows[i] = nw

            # Build and return
            headers = [
                "Idx",
                "TokId",
                "Token",
                "Boost (str=1)",
                "CountsBase",
                "CountsWm",
                "CountsWmFt",
                "IsGreen?",
                "CumAcc",
                "mass_base",
                "mass_wm",
                "mass_ratio",
                "mass_diff",
            ]
            if ctx_text == "":
                headers.extend(
                    [
                        "hashint",
                        "Avg hashint ABOVE",
                        "P(win) ABOVE",
                        "SELFGREEN W/ BIG (avg ABOVE)",
                        "Final % (base is 25%)",
                        "Final % ABOVE (base is 25%)",
                    ]
                )

            return gr.Dataframe(
                headers=headers, interactive=False, col_count=len(headers), value=rows
            )

        def on_inspect_btn2_click(ctx_text: str, k: float, col_sort: int, ordered: bool) -> Any:
            return on_inspect_btn_click(ctx_text, k, col_sort, ordered)

        # Generate on the server -> update nonwm/wm state but also the textboxes
        # Returns: nowm output, wm output, nowm output, wm output
        def on_srvr_btn_click(prompt: str) -> Any:
            output_wm = server.generate([prompt])[0][0]
            output_nowm = server.generate([prompt], disable_watermark=True)[0][0]
            return output_nowm, output_wm, output_nowm, output_wm

        # When the corresponding state changes update the metrics
        # Returns: colored output, metric dataframe (list of lists)
        def on_srvr_nowm_output_state_change(prompt: str, raw_output: str) -> Any:
            return _get_colored_text_and_metric_df(prompt, raw_output)

        # When the corresponding state changes update the metrics
        # Returns: colored output, metric dataframe (list of lists)
        def on_srvr_wm_output_state_change(prompt: str, raw_output: str) -> Any:
            return _get_colored_text_and_metric_df(prompt, raw_output)

        # Do the same trick for the attacker but with more params
        # Returns: raw output x2
        def on_att_btn_click(
            prompt: str,
            strength: float,
            reppenalty: float,
            use_ft_counts: bool,
            use_graceful_conclusion: bool,
            att_sysprompt: str,
            att_w_future: float,
            att_future_num_cands: float,
            att_future_num_options: float,
            att_future_prefix_size: float,
            att_future_local_w_decay: float,
        ) -> Any:
            # Load and override
            cfg_gen = cfg.attacker.generation.model_copy(deep=True)
            cfg_gen.spoofer_strength = strength
            cfg_gen.repetition_penalty = reppenalty
            cfg_gen.use_ft_counts = use_ft_counts
            cfg_gen.use_graceful_conclusion = use_graceful_conclusion
            cfg_gen.sysprompt = SyspromptType(att_sysprompt)
            # cfg_gen.w_future = att_w_future
            # cfg_gen.future_num_cands = int(att_future_num_cands)
            # cfg_gen.future_num_options_per_fillin = int(att_future_num_options)
            # cfg_gen.future_prefix_size = int(att_future_prefix_size)
            # cfg_gen.future_local_w_decay = att_future_local_w_decay

            output = attacker.generate([prompt], cfg_gen)[0]
            return output, output

        # Returns: colored output, metric dataframe (list of lists), dbg_info, current token for dbg idx
        # dbg_info is dict[str, LogitInfo]
        def on_att_output_state_change(
            prompt: str,
            strength: float,
            reppenalty: float,
            use_ft_counts: bool,
            use_graceful_conclusion: bool,
            att_sysprompt: str,
            att_w_future: float,
            att_future_num_cands: float,
            att_future_num_options: float,
            att_future_prefix_size: float,
            att_future_local_w_decay: float,
            raw_output: str,
            populate_logitdebugger: bool,
            running_z: bool,
        ) -> Any:
            if populate_logitdebugger:
                dbg_info: Dict[str, List[LogitInfo]] = {}
                ProgressLogger.start("Getting dbg info...")

                # Full with all features
                cfg_gen = cfg.attacker.generation.model_copy(deep=True)
                cfg_gen.spoofer_strength = strength
                cfg_gen.repetition_penalty = reppenalty
                cfg_gen.use_ft_counts = use_ft_counts
                cfg_gen.use_graceful_conclusion = use_graceful_conclusion
                cfg_gen.sysprompt = SyspromptType(att_sysprompt)
                # cfg_gen.w_future = att_w_future
                # cfg_gen.future_num_cands = int(att_future_num_cands)
                # cfg_gen.future_num_options_per_fillin = int(att_future_num_options)
                # cfg_gen.future_prefix_size = int(att_future_prefix_size)
                # cfg_gen.future_local_w_decay = att_future_local_w_decay

                _, batch_infos = attacker.model.get_ppls_and_logitinfo(
                    [prompt], [raw_output], logit_processors=attacker.get_processor_list(cfg_gen)
                )
                dbg_info["repetitionPenalty+spoofer+future (FULL)"] = batch_infos[0]

                # Spoofer only
                cfg_gen.w_future = 0
                _, batch_infos = attacker.model.get_ppls_and_logitinfo(
                    [prompt], [raw_output], logit_processors=attacker.get_processor_list(cfg_gen)
                )
                dbg_info["repetitionPenalty+spoofer"] = batch_infos[0]

                # Base
                cfg_gen.repetition_penalty = 0
                _, batch_infos = attacker.model.get_ppls_and_logitinfo(
                    [prompt], [raw_output], logit_processors=attacker.get_processor_list(cfg_gen)
                )
                dbg_info["spoofer"] = batch_infos[0]

                # Base
                cfg_gen.spoofer_strength = 0
                _, batch_infos = attacker.model.get_ppls_and_logitinfo(
                    [prompt], [raw_output], logit_processors=attacker.get_processor_list(cfg_gen)
                )
                dbg_info["base"] = batch_infos[0]

                ProgressLogger.stop()
            else:
                dbg_info = {}  # ignore

            # Slider
            slider = gr.Slider(
                label="Token",
                minimum=0,
                maximum=(len(dbg_info["base"]) - 1 if len(dbg_info) > 0 else 0),  # max token
                step=1,
                value=0,
            )
            # Get the rest and return
            colored_output, metric_df, gpt4_explanations = _get_colored_text_and_metric_df(
                prompt, raw_output, get_gpt4_exp=True, running_z=running_z
            )
            return colored_output, metric_df, dbg_info, slider, gpt4_explanations

        # Actually run FT queries
        # Returns: size of FT corpus
        def on_att_ft_get_click(prompt: str) -> Any:
            attacker.get_finetuning_qs_and_learn(server, prompt)
            return attacker.get_corpus_size(finetuning=True)

        # Clear the FT corpus
        # Returns: size of FT corpus
        def on_att_ft_clear_click() -> Any:
            attacker.counts_wm_ft.clear()
            return 0

        # Detector Only
        def on_detectoronly_detect_btn_click(prompt: str, text: str) -> Any:
            colored_text, metric_df = _get_colored_text_and_metric_df(prompt, text)
            return colored_text, metric_df

        #####
        #####
        #####
        # Now hook up callbacks
        #####
        #####
        #####

        # When the debugger slider changes underline the correct token
        dbg_slider.change(
            fn=on_dbg_slider_change, inputs=[dbg_slider, att_output], outputs=[att_output]
        )

        # (Hack) Whenever the att. output changes update the barplots
        att_output.change(
            fn=on_att_output_change, inputs=[dbg_info, dbg_slider], outputs=dbg_infocontainers
        )

        # When the inspect button is clicked, update the inspect result
        inspect_btn.click(
            fn=on_inspect_btn_click,
            inputs=[inspect_token, inspect_k, inspect_col_sort, inspect_ordered],
            outputs=[inspect_result],
        )
        inspect_btn2.click(
            fn=on_inspect_btn2_click,
            inputs=[inspect_token2, inspect_k2, inspect_col_sort2, inspect_ordered2],
            outputs=[inspect_result2],
        )

        # Generate on the server -> update nonwm/wm state but also the textboxes
        srvr_btn.click(fn=on_srvr_btn_click, inputs=[srvr_prompt], outputs=srvr_gencontainers)

        # When the corresponding state changes update the metrics
        srvr_nowm_output_state.change(
            fn=on_srvr_nowm_output_state_change,
            inputs=[srvr_prompt, srvr_nowm_output_state],
            outputs=[srvr_nowm_output, srvr_nowm_metrics],
        )

        srvr_wm_output_state.change(
            fn=on_srvr_wm_output_state_change,
            inputs=[srvr_prompt, srvr_wm_output_state],
            outputs=[srvr_wm_output, srvr_wm_metrics],
        )

        # Do the same trick for the attacker but with more params
        att_btn.click(
            fn=on_att_btn_click, inputs=att_params, outputs=[att_output_state, att_output]
        )

        att_output_state.change(
            fn=on_att_output_state_change,
            inputs=att_params + [att_output_state, att_populate_logitdebugger, att_running_Z],
            outputs=[att_output, att_metrics, dbg_info, dbg_slider, att_gpt4_explanation],
        )

        # Get attacker FT info
        att_ft_get.click(fn=on_att_ft_get_click, inputs=[att_prompt], outputs=[att_ftcorpus_sz])
        att_ft_clear.click(fn=on_att_ft_clear_click, inputs=[], outputs=[att_ftcorpus_sz])

        # Detectoronly
        detectoronly_detect_btn.click(
            fn=on_detectoronly_detect_btn_click,
            inputs=[detectoronly_prompt, detectoronly_text],
            outputs=[detectoronly_text_colored, detectoronly_metrics],
        )

    return demo
