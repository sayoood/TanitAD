"""CPU SMOKE, part 2: the inference-seed replicate and the bar evaluator on the smoke's 3 windows.
Code-path validation only (fp32 CPU trunk, 3 windows) -- no number here is a result."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import run_battery as RB  # noqa: E402
import refcv6_roll as RR  # noqa: E402
import refcv6_panel as P  # noqa: E402


def main():
    out = HERE.parent / "raw" / "smoke_cpu"
    kit_ids = RB._clip_ids_by_index(RB.KIT_EVAL)
    win = RR.s2_window_list(P.BASELINES["b_refcv4b"], set(kit_ids))
    windows = {c: win[c][:1] for c in sorted(win)[:3]}
    _orig = L.build_model

    def _fp32(*a, **k):
        m, c, ar, r = _orig(*a, **k)
        for mod in m.modules():
            lv = getattr(mod, "memory_levers", None)
            if isinstance(lv, dict) and "bf16" in lv:
                lv["bf16"] = False
                lv["channels_last"] = False
        return m, c, ar, r
    L.build_model = _fp32
    RB.roll_one(str(L.KIT / "ckpt/ckpt_step1000.pt"), str(L.KIT / "ckpt/config.json"), 1,
                str(out / "dump_s1"), windows, None, device="cpu")
    d1 = str(out / "dump_s1")
    P.build_panel_dump(d1, str(out / "panel_s1"), os.path.join(d1, "refcv6_extras.npz"))
    cells = {0: json.load(open(out / "cross_paired_s0.json")),
             1: P.cross_paired(str(out / "panel_s1"), RB.PAIRS, n_boot=200)}
    rep = P.seed_replicate(str(out / "panel_s0"), str(out / "panel_s1"), n_boot=200)
    s6 = {0: P.cross_paired(str(out / "s6_s0"), RB.PAIRS, n_boot=200)}
    P.build_s6_dump(d1, str(out / "s6_s1"))
    s6[1] = P.cross_paired(str(out / "s6_s1"), RB.PAIRS, n_boot=200)
    bars = RB.bar_verdicts(cells, s6, is_milestone=True)
    print("[smoke2] replicate max |path diff| m:", rep["max_abs_path_diff_m"],
          "ADE", rep["families"]["ADE"]["ade_m"])
    for b in bars:
        print("[smoke2]", b["id"], b["verdict"], json.dumps(b["per_inference_seed"]))
    print("[smoke2] DONE")


if __name__ == "__main__":
    main()
