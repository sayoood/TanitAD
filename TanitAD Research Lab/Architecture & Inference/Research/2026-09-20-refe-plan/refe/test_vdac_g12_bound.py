#!/usr/bin/env python3
"""HC G12 memory bound (MM 2026-11-12): head_retained_gb's sanity bound scales with the declared cell, bound = 1.0 x (2.0 / cell)^2 GB (2 m -> 1.0, 1 m -> 4.0).  CPU only.
Run:  CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python refe/test_vdac_g12_bound.py   ->   ZZTEST_PASS <n>
MEASURED on the pod 2026-11-12: HC-s0 preflight head_retained_gb 1.231 (cell 1.0, 5,120 queries); HA-s0 / HA-SHUF 0.463 (cell 2.0, 1,280 queries).
  * gate side:  gate_checks.judge_g12 on the parsed namespace of vdac_argv("HC-s0") (cell 1.0) / vdac_argv("HA-s0") (cell 2.0): 1.231 @ 1.0 PASS, 4.5 @ 1.0 RED, 1.5 @ 2.0 RED (test_launch_gate.py:1602's
    case, unchanged), 0.463 @ 2.0 PASS; a trainer bound that disagrees with the declared cell is RED.
  * MUTATION:   gate_checks.py with the bound hard-coded back to 1.0 must make (1.231, cell 1.0) RED; the trainer's source with `<= 1.0` back must be caught by the text pin.
  * trainer side: the G12 line reads the BUILT cell (aux.cfg.cell), not argv; the bound key is written next to head_retained_gb.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_checks as C  # noqa: E402
import launch_gate as LG  # noqa: E402

N = [0]
TABLE: list = []
GROUPS = ("q", "pos_mlp", "blocks.0", "blocks.1", "norm", "out")


def ok(cond, msg):
    N[0] += 1
    assert cond, msg


def red(caught, what, how):
    N[0] += 1
    assert caught, f"MUTATION NOT CAUGHT: {what}"
    TABLE.append((what, "RED", how))


def ns_of(arm):
    return C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), LG.vdac_argv(arm))[0]


def g12_of(ns, **mut):
    g = {"ok": True, "w_aux": ns.w_aux_drv, "feed": ns.aux_drv_feed, "labelled_frames_in_batch": 8, "batch_frames": 8, "mutate": [], "scene_proj_calls": 2, "tap_is_visual_ctx": True,
         "tap_is_scene_ctx": False, "score_to_head_grads_all_none": True, "bce_declared": 0.70, "huber_declared": 0.4, "bce_w0": 0.70, "huber_w0": 0.4,
         "per_group_declared": {k: 1.5 for k in GROUPS}, "per_group_w0": {k: 0.0 for k in GROUPS}, "bce_near_ln2": True, "adam_head_params": 43, "adam_head_params_with_state": 43,
         "head_retained_gb": None, "reference_replay_identical": True, "scorer_params_bit_identical_2steps": True, "identity_probe_reads_one_ulp": True}
    g.update(mut)
    return g


def main() -> int:
    ns_hc, ns_ha = ns_of("HC-s0"), ns_of("HA-s0")
    ok(ns_hc.aux_drv_cell == 1.0 and ns_ha.aux_drv_cell == 2.0, "HC-s0 declares cell 1.0, HA-s0 cell 2.0")
    ok(C.judge_g12(g12_of(ns_hc), ns_hc) == [] and C.judge_g12(g12_of(ns_ha), ns_ha) == [], "baseline G12 (memory not measured) is accepted for both")
    # gate side
    ok(C.judge_g12(g12_of(ns_hc, head_retained_gb=1.231, head_retained_bound_gb=4.0), ns_hc) == [], "MEASURED HC value 1.231 @ cell 1.0 (trainer bound 4.0) PASSES")
    ok(C.judge_g12(g12_of(ns_hc, head_retained_gb=1.231), ns_hc) == [], "1.231 @ cell 1.0 passes with no trainer bound recorded (older dicts)")
    r = C.judge_g12(g12_of(ns_hc, head_retained_gb=4.5, head_retained_bound_gb=4.0), ns_hc)
    red(any("4.5" in x and "> 4 GB" in x for x in r), "cell 1.0, 4.5 GB", f"names value and bound: {r}")
    r = C.judge_g12(g12_of(ns_ha, head_retained_gb=1.5, head_retained_bound_gb=1.0), ns_ha)
    red(any("> 1 GB" in x for x in r), "cell 2.0, 1.5 GB (test_launch_gate.py:1602, fragment '> 1 GB')", f"{r}")
    ok(C.judge_g12(g12_of(ns_ha, head_retained_gb=0.463, head_retained_bound_gb=1.0), ns_ha) == [], "MEASURED HA value 0.463 @ cell 2.0 PASSES")
    r = C.judge_g12(g12_of(ns_hc, head_retained_gb=1.231, head_retained_bound_gb=1.0), ns_hc)
    red(any("head_retained_bound_gb" in x for x in r), "trainer bound 1.0 at declared cell 1.0 (hard-coded in the trainer)", f"{r}")
    r = C.judge_g12(g12_of(ns_ha, head_retained_gb=2.0, head_retained_bound_gb=4.0), ns_ha)
    red(any("head_retained_bound_gb" in x for x in r), "trainer bound 4.0 at declared cell 2.0 (a loosened bound cannot hide)", f"{r}")
    # mutation: gate_checks with the bound hard-coded back to 1.0
    src = (HERE / "gate_checks.py").read_text(encoding="utf-8")
    line = "    bound = 1.0 * (2.0 / cell_g) ** 2"
    ok(src.count(line) == 1, "the gate's bound line exists once")
    mod = types.ModuleType("gate_checks_mut")
    mod.__file__ = str(HERE / "gate_checks.py")
    exec(compile(src.replace(line, "    bound = 1.0"), "gate_checks_mut", "exec"), mod.__dict__)
    r = mod.judge_g12(g12_of(ns_hc, head_retained_gb=1.231), ns_hc)
    red(any("1.231" in x for x in r), "gate_checks bound hard-coded back to 1.0: (1.231, cell 1.0) goes RED", f"{r}")
    ok(C.judge_g12(g12_of(ns_hc, head_retained_gb=1.231), ns_hc) == [], "control: the unmutated gate accepts the same dict")
    # trainer side
    tsrc = (HERE / "scorer_finetune.py").read_text(encoding="utf-8")
    t_set = 'g["head_retained_gb"], g["head_retained_bound_gb"] = None, 1.0 * (2.0 / float(aux.cfg.cell)) ** 2'
    t_cmp = 'g["head_retained_gb"] <= g["head_retained_bound_gb"])'

    def judge_t(s):
        return s.count(t_set) == 1 and s.count(t_cmp) == 1 and "head_retained_gb\"] <= 1.0" not in s
    ok(judge_t(tsrc), "the trainer sets the bound from the BUILT cell (aux.cfg.cell) and compares against it")
    red(not judge_t(tsrc.replace(t_cmp, 'g["head_retained_gb"] <= 1.0)')), "trainer: `<= 1.0` hard-coded again", "text check")
    red(not judge_t(tsrc.replace("float(aux.cfg.cell)", "float(a.aux_drv_cell)")), "trainer: bound read from argv instead of the built cell", "text check")
    print(f"{'mutation / check':80s} result  how")
    for what, res, how in TABLE:
        print(f"{what[:80]:80s} {res:7s} {how[:110]}")
    print(f"ZZTEST_PASS {N[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
