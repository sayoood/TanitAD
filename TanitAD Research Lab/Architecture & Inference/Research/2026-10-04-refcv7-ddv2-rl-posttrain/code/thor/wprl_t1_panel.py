#!/usr/bin/env python
"""WP-RL stage 3: the T1 PRIMARY panel (SPEC_RL sec. 6.1) from battery roll dumps, through the
battery's OWN instrument (`r7_panel.build_panel_dump` / `analyze` / `cross_paired`).

    python wprl_t1_panel.py --base <dump_base_sS> --arm rl=<dump> --arm rloff=<dump> ...
                            --seed S --out <dir>

For ONE inference seed: the BASE (cold start) dump is the panel's `os`; each arm's `os` is added as
an arm `b_<name>` on the SAME windows (pairing VERIFIED per window by the battery: `g` and the
model-free arms bit-identical, else VOID). Then the four-family analysis and the paired cells:
RL - RLOFF (the RL effect), RL - BASE (the total effect), RLOFF - BASE, RL-SHUF - RLOFF (validity),
RL-s1 - RLOFF, RL-s1 - RL (training floor). Interval: paired episode-cluster bootstrap (another
draw of EPISODES only). Runs where the battery runs (the dev box: its lead block path is D:).
"""
import argparse
import json
import os
import sys
from pathlib import Path

BATTERY = Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code")

PAIRS = [("b_rloff", "b_rl", "rl_minus_rloff"), ("os", "b_rl", "rl_minus_base"),
         ("os", "b_rloff", "rloff_minus_base"), ("b_rloff", "b_rlshuf", "rlshuf_minus_rloff"),
         ("b_rloff", "b_rl_s1", "rls1_minus_rloff"), ("b_rl", "b_rl_s1", "rls1_minus_rl"),
         ("os", "b_rl_s1", "rls1_minus_base"), ("ha0_ext", "b_rl", "rl_minus_ha0ext"),
         ("ha0_ext", "os", "base_minus_ha0ext")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", default=[], help="name=dump_dir (rl, rloff, rlshuf, rl_s1)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--labels", default="D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(BATTERY))
    import r7_panel as RP
    arms = {f"b_{k}": v for k, v in (x.split("=", 1) for x in a.arm)}
    for lab in arms:
        RP.TIERS.setdefault(lab, "T1")
    out = Path(a.out)
    panel = out / f"panel_s{a.seed}"
    rec = {"seed": a.seed, "base": a.base, "arms": arms,
           "build": RP.build_panel_dump(a.base, str(panel), None, baselines=arms)}
    if not rec["build"]["void_gates_3_4"]["pass"]:
        rec["VOID"] = rec["build"]["void_gates_3_4"]["failures"]
        (out / f"t1_panel_s{a.seed}.json").write_text(json.dumps(rec, indent=1, default=str))
        raise SystemExit(f"[t1] VOID: {rec['VOID'][:3]}")
    rec["analysis"] = RP.analyze(str(panel), str(out / f"analysis_s{a.seed}.json"), a.labels,
                                 n_boot=a.n_boot, seed=0)
    pairs = [p for p in PAIRS if p[0] in (["os", "ha0_ext"] + list(arms)) and p[1] in (["os"] + list(arms))]
    rec["paired"] = RP.cross_paired(str(panel), pairs, n_boot=a.n_boot, seed=0)
    rec["_tier"] = "T1 (self-action OPEN loop; S2 0.5-2 s)"
    rec["_estimator"] = "paired episode-cluster bootstrap (another draw of EPISODES only)"
    (out / f"t1_panel_s{a.seed}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("analysis",)}, indent=1, default=str)[:8000])


if __name__ == "__main__":
    main()
