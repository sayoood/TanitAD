"""CPU SMOKE of the whole battery pipeline on 1 clip x 3 windows (code-path validation only).

No number printed here is a result: it runs the kit checkpoint on CPU on 3 windows to prove that the
roll (refcv3_arm.run_dump fed like the trainer), the panel dump (pairing gates), the analysis, the
cross-model paired families, the refcv6 tactical metrics, the acceptance instruments and the 6 s
read all EXECUTE and emit their artifacts, before any GPU time is spent.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import torch  # noqa: E402
import run_battery as RB  # noqa: E402
import refcv6_roll as RR  # noqa: E402
import refcv6_panel as P  # noqa: E402


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "raw" / "smoke_cpu")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    kit_ids = RB._clip_ids_by_index(RB.KIT_EVAL)
    win = RR.s2_window_list(P.BASELINES["b_refcv4b"], set(kit_ids))
    cs = sorted(win)[:3]
    windows = {c: win[c][:1] for c in cs}            # 3 windows from 3 clips (>= 2 episodes)
    # ⚠️ SMOKE ONLY: the trunk runs fp32 on CPU (this CPU has no native bf16; autocast-bf16 on it
    # took 540 s for one 3-row forward). Numerics are irrelevant here -- only code paths are tested.
    _orig_build = L.build_model

    def _build_fp32(*a, **k):
        m, c, ar, r = _orig_build(*a, **k)
        for mod in m.modules():
            lv = getattr(mod, "memory_levers", None)
            if isinstance(lv, dict) and "bf16" in lv:
                lv["bf16"] = False
                lv["channels_last"] = False
        r["smoke_note"] = "trunk bf16/NHWC switched OFF for the CPU smoke"
        return m, c, ar, r
    L.build_model = _build_fp32
    t0 = time.time()
    rec = RB.roll_one(str(L.KIT / "ckpt/ckpt_step1000.pt"), str(L.KIT / "ckpt/config.json"), 0,
                      str(out / "dump_s0"), windows, None, device="cpu")
    print("[smoke] roll", json.dumps({k: rec[k] for k in ("n_windows", "n_episodes", "skipped",
                                                         "wall_s")}), flush=True)
    d = str(out / "dump_s0")
    pd = str(out / "panel_s0")
    prec = P.build_panel_dump(d, pd, os.path.join(d, "refcv6_extras.npz"))
    print("[smoke] panel", json.dumps({"void": prec["void_gates_3_4"], "arms": prec["arms"],
                                       "n": prec["n_windows"]}), flush=True)
    an = P.analyze(pd, str(out / "analysis_s0.json"), RB.KIT_LABELS, n_boot=200)
    cp = P.cross_paired(pd, RB.PAIRS, n_boot=200)
    json.dump(cp, open(out / "cross_paired_s0.json", "w"), indent=1)
    man = json.load(open(os.path.join(pd, "manifest.json"), encoding="utf-8"))
    clip_eid = {int(e["episode_index"]): int(e["file_index"]) for e in man["episodes"]}
    tac = P.tactical_v6(os.path.join(d, "refcv6_extras.npz"), clip_eid, n_boot=200)
    json.dump(tac, open(out / "tactical_v6_s0.json", "w"), indent=1)
    acc = P.acceptance(an, os.path.join(d, "refcv6_extras.npz"), clip_eid, pd, n_boot=200)
    json.dump(acc, open(out / "acceptance_s0.json", "w"), indent=1, default=str)
    s6 = P.build_s6_dump(d, str(out / "s6_s0"))
    print("[smoke] s6", json.dumps(s6, default=str), flush=True)
    if s6["n_windows"]:
        P.analyze(str(out / "s6_s0"), str(out / "analysis_s6_s0.json"), RB.KIT_LABELS, n_boot=200)
    print("[smoke] cross-paired ADE os-ha0_ext:",
          cp["pairs"]["os_minus_ha0ext"].get("families", {}).get("ADE", {}).get("ade_m"))
    print("[smoke] tflip", acc.get("tflip", {}).get("verdict"), "obedience",
          acc.get("obedience", {}).get("verdict"))
    print(f"[smoke] DONE in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
