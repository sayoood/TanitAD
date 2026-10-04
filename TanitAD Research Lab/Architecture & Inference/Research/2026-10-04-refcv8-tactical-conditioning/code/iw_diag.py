"""I-W DIAGNOSTIC (not a registered phase; measures WHY SPEC_WPB's I-W failed on Thor, 2026-10-04 17:22 UTC).

MEASURED by the registered I-W (arms/I_W.json, 218 grid windows, GPU):
- seams with no extension: bit-identical (PASS);
- allocation 32 + prior-free group, emission off: sel_idx identical on 218 / 218 windows, BUT traj differs on 205 and
  the base fan on 218, and the base scores by up to 2.1e-5 > 1e-5 (FAIL).

Hypothesis (to measure, not assume): extending the candidate axis 117 -> 266 changes the decoder's GEMM shapes, so
cuBLAS picks other kernels (TF32 tiles) and the BASE candidates' floats move in the low bits. It is a numerical
difference, not a logic one.

This script reads, per window and per variant, the MAGNITUDE (metres / score units), not only "differs":
* control C0: T0 against T0, a second forward of the same model. This is the GPU determinism floor and must read
  exactly 0.
* variants: allocation 32 only, prior-free only, and both. Each is run under the run's default TF32 setting and with
  TF32 OFF (matmul and cuDNN, float32 precision 'highest').
* outputs per variant: max |delta traj| (m), max |delta base fan| (m), max |delta base score|, and the count of
  windows where sel_idx differs.

One Thor GPU job under the lock (<= 40 min): --max-windows windows (default 48), 4 model builds, 8 passes.
-> <out>/I_W_diag.json. Nothing is trained; no tree file is edited.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wpb_arms as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1-code", default=W.R1_CODE_DEFAULT)
    ap.add_argument("--cache", default="/home/nvidia/refcv8_r1/cache")
    ap.add_argument("--out", default="/home/nvidia/refcv8_wpb/arms")
    ap.add_argument("--ckpt", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt")
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-windows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--timing-steps", type=int, default=0)
    ap.add_argument("--eval-seeds", default="0")
    ap.add_argument("--ctrl", action="store_true")
    ap.add_argument("--budget-s", type=float, default=2400.0)
    ap.add_argument("--max-eval-windows", type=int, default=0)
    ap.add_argument("--arm", default="T0")
    a = ap.parse_args()
    import torch
    t0 = time.time()
    A, H, L1 = W._r1(a)
    D, _sel = A.load_split(a.cache, "eval")
    grid = [j for j, f in enumerate(D["_flags"]) if "grid" in f][: a.max_windows]
    table = json.loads((Path(a.r1_code).parent / "raw" / "r1_nav_ann_table.json").read_text())
    nav, _ = A.nav_ann(D, table)
    default_tf32 = {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                    "cudnn": bool(torch.backends.cudnn.allow_tf32),
                    "fp32_precision": torch.get_float32_matmul_precision()}

    def set_tf32(on: bool):
        if on:
            torch.backends.cuda.matmul.allow_tf32 = default_tf32["matmul"]
            torch.backends.cudnn.allow_tf32 = default_tf32["cudnn"]
            torch.set_float32_matmul_precision(default_tf32["fp32_precision"])
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")

    def run(model, OUT, feed, tr, args):
        outs = []
        for j in grid:
            ix = np.array([j])
            feed.cur = A.make_feed(D, ix, a.device)
            b = A.make_batch(D, ix, None, None, nav)
            OUT.clear()
            torch.manual_seed(0)
            with torch.no_grad():
                H.forward_like_trainer(model, b, a.device, args.mode, tr)
            o = OUT["out"]
            r = {k: o[k].detach().double().cpu() for k in ("traj", "sel_idx", "anchor_traj", "sel_score_v3")}
            r["nb"] = int(o["r8_n_base"]) if "r8_n_base" in o else r["anchor_traj"].shape[1]
            outs.append(r)
        return outs

    def compare(x_list, y_list):
        dt, df, ds, nsel = [], [], [], 0
        for x, y in zip(x_list, y_list):
            nb = y["nb"]
            dt.append(float((x["traj"] - y["traj"]).abs().max()))
            df.append(float((x["anchor_traj"] - y["anchor_traj"][:, :nb]).abs().max()))
            ds.append(float((x["sel_score_v3"] - y["sel_score_v3"][:, :nb]).abs().max()))
            nsel += int(not np.array_equal(x["sel_idx"].numpy(), y["sel_idx"].numpy()))
        return {"max_abs_dtraj_m": max(dt), "median_abs_dtraj_m": float(np.median(dt)),
                "max_abs_dfan_base_m": max(df), "max_abs_dscore_base": max(ds), "n_sel_idx_differ": nsel,
                "n_windows_traj_bitdiff": int(sum(1 for v in dt if v > 0.0))}

    rep = {"what": "I-W diagnostic: magnitudes of the extended-fan difference (MEASURED)", "n_windows": len(grid),
           "default_tf32": default_tf32, "rows": {}}
    base = {}
    m0, _c, args, tr, _H, feed, _T, OUT = W.build_arm(a, A, "T0")
    m0.eval()
    for tf in (True, False):
        set_tf32(tf)
        base[tf] = run(m0, OUT, feed, tr, args)
        rep["rows"][f"C0_T0_vs_T0_tf32_{'default' if tf else 'off'}"] = compare(base[tf], run(m0, OUT, feed, tr, args))
    rep["rows"]["T0_tf32_default_vs_off"] = compare(base[True], base[False])
    del m0
    from tanitad.refs import refcv8_conditioning as C
    for name, extra in (("alloc32_only", dict(n_alloc=32)), ("prior_free_only", dict(prior_free_group=True)),
                        ("alloc32_prior_free", dict(n_alloc=32, prior_free_group=True))):
        m, _c, args, tr, _H, feed, _T, OUT = A.build(a)
        m.enable_refcv8(C.R8Config(**dict(W._R8_COMMON, **extra)))
        m.to(a.device).eval()
        for tf in (True, False):
            set_tf32(tf)
            rep["rows"][f"{name}_tf32_{'default' if tf else 'off'}"] = compare(base[tf], run(m, OUT, feed, tr, args))
        del m
        print(f"[iw-diag] {name} done {time.time() - t0:.0f}s", flush=True)
    set_tf32(True)
    rep["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out) / "I_W_diag.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
