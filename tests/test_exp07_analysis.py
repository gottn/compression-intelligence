"""exp07 analysis on a synthetic fixture: item outcomes are built first and the scores derived from them, so the
unpaired values, paired CIs and verdicts must agree with numbers fixed by construction."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from exp07_arc_format.analyze_format import (
    EXP04_SCORES,
    diff_pts,
    drift_difference,
    expectations,
    figure,
    format_table,
    paired_ci,
    prov_check,
    reproduction_row,
    rule1_verdict,
    rule2_verdict,
    sanity_table,
    score_index,
    scores_long,
    summary_markdown,
    verdict_table,
)
from exp07_arc_format.run_battery import CHAT_GROUP, chat_folder

N = 500  # docs per synthetic task
OLMO, SMOL = "allenai/OLMo-2-0425-1B", "HuggingFaceTB/SmolLM2-1.7B"
SMOL_I = f"{SMOL}-Instruct"
OLMO_LAB, SMOL_LAB, SMOLI_LAB = "allenai__OLMo-2-0425-1B", "HuggingFaceTB__SmolLM2-1.7B", "HuggingFaceTB__SmolLM2-1.7B-Instruct"


def ones(k: int) -> str:
    return "1" * k + "0" * (N - k)


def group(tasks: dict[str, dict[str, int]], leaves: dict[str, list[str]] | None = None,
          md5: str | None = None) -> tuple[dict[str, object], dict[str, object]]:
    """(done.json, items.json) of one group from correct-answer counts per task and metric; a task listed in
    `leaves` is a group whose items are split evenly over those leaf names."""
    items, metrics = {}, {}
    for task, counts in tasks.items():
        names = (leaves or {}).get(task, [task])
        per = N // len(names)
        for j, leaf in enumerate(names):
            items[leaf] = {"doc_id": list(range(per))}
            for m, k in counts.items():
                kk = k // len(names) + (1 if j < k % len(names) else 0)
                items[leaf][m] = "1" * kk + "0" * (per - kk)
        m_out = {}
        for m, k in counts.items():
            p = k / N
            m_out[f"{m},none"], m_out[f"{m}_stderr,none"] = p, math.sqrt(p * (1 - p) / (N - 1))
        metrics[task] = {"metrics": m_out, "n_effective": N}
    return {"status": "ok", "tasks": metrics, "template_md5": md5,
            "truncation": {t: {"n_over": 0} for t in tasks}}, items


def arc(e_norm: int, e_acc: int, c_norm: int = 150, c_acc: int = 140) -> dict[str, dict[str, int]]:
    return {"arc_easy": {"acc_norm": e_norm, "acc": e_acc}, "arc_challenge": {"acc_norm": c_norm, "acc": c_acc}}


MMLU_LEAVES = {"mmlu": ["mmlu_a", "mmlu_b"], "mmlu_continuation": ["mmlu_a_continuation", "mmlu_b_continuation"]}


def checkpoint(label: str, b1: int, b2: int, b3: int, letter: int, cont: int, hs: int = 300,
               b3_folders: tuple[str, ...] = (CHAT_GROUP,), md5: str = "tpl", piqa_wino: bool = False):
    out = {}
    out["b1_arc_0shot"] = group(arc(b1, b1 - 10))
    out["b2_arc_25shot"] = group(arc(b2, b2 - 5))
    for f in b3_folders:
        out[f] = group(arc(b3, b3 - 5), md5=md5)
    out["b4_mmlu_letter"] = group({"mmlu": {"acc": letter}}, MMLU_LEAVES)
    out["b4_mmlu_cont"] = group({"mmlu_continuation": {"acc": cont}}, MMLU_LEAVES)
    out["b5_hellaswag"] = group({"hellaswag": {"acc_norm": hs, "acc": hs - 40}})
    if piqa_wino:
        out["b0_piqa_wino"] = group({"piqa": {"acc_norm": 380}, "winogrande": {"acc": 320}})
    return {(label, f): v for f, v in out.items()}


def entry(label: str, model: str, base: str, recipe: str = "public", step: int = 0) -> dict[str, object]:
    return {"label": label, "model": model, "recipe": recipe, "step": step, "marks": "-", "base": base,
            "is_base": model == base}


@pytest.fixture(scope="module")
def fixture() -> dict[str, object]:
    runs = {}
    olmo_b3 = tuple(chat_folder(m) for m in ("allenai/OLMo-2-0425-1B-SFT", "allenai/OLMo-2-0425-1B-DPO",
                                             "allenai/OLMo-2-0425-1B-Instruct"))
    runs |= checkpoint(OLMO_LAB, 365, 375, 360, 190, 170, b3_folders=olmo_b3, md5="olmo")
    # mathcot: ARC-E loss 9 / 6 / 3 points under B1 / B2 / B3 (mixed); tulu: letter +7, continuation +0.4 (format;
    # the unpaired MMLU floor at n = 500 is about 6 points)
    runs |= checkpoint("exp07__mathcot__step01500", 320, 345, 345, 185, 168, md5="olmo", piqa_wino=True)
    runs |= checkpoint("exp07__tulu__step01500", 364, 374, 361, 225, 172, md5="olmo", piqa_wino=True)
    # SmolLM2-Instruct: 10 / 6 / 7 points (K); MMLU letter +9, continuation +7 (survives)
    runs |= checkpoint(SMOL_LAB, 360, 370, 350, 180, 160, b3_folders=(chat_folder(SMOL_I),), md5="smol")
    runs |= checkpoint(SMOLI_LAB, 310, 340, 315, 225, 195, md5="smol")
    done = {k: v[0] for k, v in runs.items()}
    items = {k: v[1] for k, v in runs.items()}
    entries = [entry(OLMO_LAB, OLMO, OLMO), entry("exp07__mathcot__step01500", "/r/mathcot/ckpt/step01500", OLMO,
                                                   "mathcot", 1500),
               entry("exp07__tulu__step01500", "/r/tulu/ckpt/step01500", OLMO, "tulu", 1500),
               entry(SMOL_LAB, SMOL, SMOL), entry(SMOLI_LAB, SMOL_I, SMOL)]
    sl = scores_long(done)
    idx = score_index(sl)
    templates = {k: v.get("template_md5") for k, v in done.items()}
    trunc = {k: 0 for k in done}
    exp04 = pd.read_csv(EXP04_SCORES)
    table = format_table(entries, idx, items, templates, trunc, exp04)
    return {"done": done, "items": items, "sl": sl, "idx": idx, "table": table, "exp04": exp04}


def test_scores_long_reads_values_and_standard_errors(fixture: dict[str, object]) -> None:
    sl = fixture["sl"]
    r = sl[(sl.label == OLMO_LAB) & (sl.folder == "b1_arc_0shot") & (sl.task == "arc_easy") & (sl.metric == "acc_norm")]
    assert len(r) == 1 and r.value.iloc[0] == pytest.approx(365 / N) and r.se.iloc[0] > 0
    assert set(sl.metric) == {"acc", "acc_norm"}


def test_losses_and_paired_cis_match_the_construction(fixture: dict[str, object]) -> None:
    t = fixture["table"].set_index("label")
    mc, tu, sm = t.loc["exp07__mathcot__step01500"], t.loc["exp07__tulu__step01500"], t.loc[SMOLI_LAB]
    assert (mc.arce_loss_b1_norm, mc.arce_loss_b2_norm, mc.arce_loss_b3_norm) == pytest.approx((9.0, 6.0, 3.0))
    assert mc.arce_loss_b1_acc == pytest.approx(9.0) and mc.arce_amp_b1 == pytest.approx(0.0)
    assert mc.arce_loss_b2_norm_pmean == pytest.approx(6.0) and mc.arce_loss_b2_norm_plo < 6 < mc.arce_loss_b2_norm_phi
    assert mc.arce_loss_b1_norm_pn == N and mc.arce_normgap_b1 == pytest.approx(2.0)
    assert (sm.arce_loss_b1_norm, sm.arce_loss_b2_norm, sm.arce_loss_b3_norm) == pytest.approx((10.0, 6.0, 7.0))
    assert (tu.mmlu_gain_letter, tu.mmlu_gain_cont) == pytest.approx((7.0, 0.4))
    assert tu.mmlu_cont_over_letter == pytest.approx(0.4 / 7) and tu.mmlu_gain_letter_plo < 7 < tu.mmlu_gain_letter_phi
    assert sm.template_matched and mc.template_matched and mc.b2_truncated_docs == 0
    assert list(t.index) == [SMOLI_LAB, "exp07__mathcot__step01500", "exp07__tulu__step01500"]


def test_verdicts(fixture: dict[str, object]) -> None:
    t = fixture["table"].set_index("label")
    assert t.loc["exp07__mathcot__step01500"].rule1 == "mixed" and t.loc[SMOLI_LAB].rule1 == "K"
    assert t.loc["exp07__tulu__step01500"].rule1 == "" and t.loc["exp07__mathcot__step01500"].rule2 == ""
    assert t.loc["exp07__tulu__step01500"].rule2 == "format" and t.loc[SMOLI_LAB].rule2 == "survives"
    assert rule1_verdict(6.0, 2.2) == "F" and rule1_verdict(1.0, 9.0) == "F" and rule1_verdict(5.0, 5.0) == "K"
    assert rule1_verdict(4.9, 9.0) == "mixed" and rule1_verdict(math.nan, 9.0) == "missing"
    assert rule2_verdict(1.0, 2.0, 0.0, 2.0) == "no gain" and rule2_verdict(math.nan, 1, 1, 1) == "missing"
    assert len(verdict_table(fixture["table"])) == 3


def test_forgetting_without_arc_uses_exp04_for_public_and_the_battery_for_retrained(fixture: dict[str, object]) -> None:
    t, exp04 = fixture["table"].set_index("label"), fixture["exp04"]
    sm = t.loc[SMOLI_LAB]
    # scores.csv keeps 5 decimals per task, so the recomputed forgetting matches exp04 to ~1e-3 %
    assert sm.forget6 == pytest.approx(float(exp04[exp04.model == SMOL_I].forgetting.iloc[0]), abs=2e-3)
    mc = t.loc["exp07__mathcot__step01500"]
    base = exp04[exp04.model == OLMO].iloc[0]
    post5 = np.mean([150 / N, 300 / N, 380 / N, 320 / N, 185 / N])
    base5 = np.mean([base.arc_challenge, base.hellaswag, base.piqa, base.winogrande, base.mmlu])
    assert mc.forget5_no_arce == pytest.approx(100 * (base5 - post5) / base5)
    assert mc.forget4_no_arc_floor > 0 and mc.general == pytest.approx(np.mean([320, 150, 300, 380, 320, 185]) / N)


def test_paired_ci_edge_cases_and_signs() -> None:
    b = {"arc_easy": {"doc_id": [0, 1, 2, 3], "acc": "1100"}}
    c = {"arc_easy": {"doc_id": [1, 2, 3, 4], "acc": "0011"}}
    m, lo, hi, n = paired_ci(b, c, "arc_easy", "acc", +1)
    assert n == 3 and m == pytest.approx(100 * (1 - 1) / 3) and lo < m < hi  # docs 1, 2, 3: (1-0) + (0-1) + (0-1)
    assert paired_ci(b, c, "arc_easy", "acc", -1)[0] == pytest.approx(-m)
    assert math.isnan(paired_ci(None, c, "arc_easy", "acc", 1)[0]) and paired_ci(b, c, "mmlu", "acc", 1)[3] == 3
    assert diff_pts((0.7, 0.01), (0.6, 0.01), +1) == pytest.approx((10.0, 100 * 1.96 * math.sqrt(2) * 0.01))


def test_reproduction_rule() -> None:
    steps = np.arange(0, 1520, 20)
    d06 = pd.DataFrame({"step": steps, "dbpb_fresh": steps / 600})
    noise = np.where(steps >= 760, 0.05 * np.cos(steps), 0.0)
    d07 = pd.DataFrame({"step": steps, "dbpb_fresh": steps / 600 + noise})
    diff = drift_difference(d07, d06)
    r = reproduction_row("tulu", diff, (0.597, 0.0047), (0.5974, 0.0047), 0.5889)
    assert r["n_sigma"] == 37 and r["sigma_run"] == pytest.approx(float(np.sqrt(np.mean(noise[38:75] ** 2))))
    assert r["d_1500"] == pytest.approx(0.05 * math.cos(1500)) and r["ok_general"] and r["reproduced"] == r["ok_drift"]
    far = reproduction_row("tulu", diff.assign(d=np.where(diff.step == 1500, 1.0, diff.d)), (0.597, 0.0047),
                           (0.5974, 0.0047), 0.5889)
    assert not far["ok_drift"] and not far["reproduced"]
    off = reproduction_row("mathcot", diff, (0.50, 0.0047), (0.5639, 0.0047), 0.5889)
    assert not off["ok_general"]


def test_prov_check_and_sanity(fixture: dict[str, object]) -> None:
    old = {"recipe": "tulu", "dataset": "d", "config": None, "revision": "r", "kept": 3, "candidates_scanned": 4,
           "drops": {"too_long": 1}, "tokens": {"sum": 10}, "loss_tokens": {"sum": 7}}
    assert prov_check(dict(old), old)["identical"]
    assert not prov_check({**old, "tokens": {"sum": 11}}, old)["identical"]
    s = sanity_table(fixture["idx"], fixture["exp04"], [OLMO_LAB])
    assert len(s) == 4 and s.diff_pts.notna().all()


def test_expectations_summary_and_figure(fixture: dict[str, object]) -> None:
    t = fixture["table"]
    repro = pd.DataFrame([{"recipe": r, "reproduced": True, "dbpb_exp06": 2.0, "dbpb_exp07": 2.01, "d_1500": 0.01,
                           "sigma_run": 0.02, "n_sigma": 37, "max_abs_d": 0.03, "r_trajectory": 0.999,
                           "general_exp06": 0.59, "general_exp07": 0.59, "general_diff_pct": 0.0,
                           "general_floor_pct": 2.2} for r in ("tulu", "mathcot")])
    e = expectations(t, repro).set_index("id")
    assert e.loc["E1"].met and e.loc["E2"].met and e.loc["E3"].met
    md = summary_markdown(t, repro, pd.DataFrame(), pd.DataFrame(), expectations(t, repro))
    assert "| mathcot @1500 (exp07) |" in md and "mixed" in md and "format" in md
    fig = figure(t)
    assert isinstance(fig, Figure) and len(fig.axes) == 2
