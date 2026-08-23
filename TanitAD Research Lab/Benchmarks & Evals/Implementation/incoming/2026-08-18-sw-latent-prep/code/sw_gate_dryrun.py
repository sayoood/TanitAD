"""Gate-day REHEARSAL runner — loads a v6 S-W snapshot through the REAL dumper
load path and smoke-tests the model pass on the local GPU. NO gate quantity is
produced here: sigma comes only from the real chain (v6_dump_sw_latents.py ->
e_wc2_sigma_star.py -> v6_chain.py admission) on the real val40 substrate.

WHY THIS EXISTS (C111 class): an analysis-time failure after the compute is
paid for destroys the run's output. This runner pays the 2-second failures NOW
(imports, strict load, param count, geometry, forward pass, the dumper's own
vision-only control) so nothing is discovered at gate time.

It deliberately calls ONLY functions imported from the production scripts —
`v6_dump_sw_latents.preflight/load_v6_stack/_forward_latents/
vision_only_control` — so what is rehearsed is what will run.

USAGE (dev box):
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe sw_gate_dryrun.py \
        --stack-root C:/Users/Admin/wt-tanitad-local/stack \
        --ckpt C:/Users/Admin/tanitad-caches/v6F-snapshots/v6F_sw_step010000.fp16.pt \
        --args-from C:/Users/Admin/tanitad-caches/v6F-snapshots/sw_config.json \
        --expect-params 336542025 --out dryrun_10k.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("sw_gate_dryrun")
    ap.add_argument("--stack-root", required=True,
                    help="<clone>/stack — taniteval is resolved as its sibling")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--args-from", required=True,
                    help="the RUN'S OWN config.json (banked copy) — never a "
                         "place to type a geometry")
    ap.add_argument("--expect-params", type=int, default=336_542_025)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-batches", default="4,8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    stack = Path(a.stack_root).resolve()
    for p in (stack, stack / "scripts", stack.parent / "taniteval"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    rec: dict = {"_what": "S-W gate rehearsal: load-proof + forward smoke. "
                          "NOT a gate artifact; no sigma in here by design.",
                 "ckpt": a.ckpt, "args_from": a.args_from}

    # 1 — the dumper's own preflight (imports for steps 1 AND 2 of the recipe)
    import v6_dump_sw_latents as SW
    fails = SW.preflight()
    rec["preflight_failures"] = fails
    if fails:
        _emit(rec, a.out)
        print("[dryrun] PREFLIGHT FAILED — fix imports before anything else.")
        return 3

    # 2 — strict load through the REAL path (normalise snapshot -> args-from ->
    #     build_stack_from_args -> load_state_dict(strict=True))
    import torch
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) \
        else "cpu"
    t0 = time.time()
    stack_m, run_args, step, prov = SW.load_v6_stack(
        a.ckpt, device=device, args_from=a.args_from)
    rec["load"] = {"seconds": round(time.time() - t0, 1), "device": device,
                   "provenance": prov}

    # 3 — the param count is a GATE here: a mismatch is a finding, never
    #     worked around.
    n = prov["n_params"]
    rec["param_check"] = {"expected": a.expect_params, "got": n,
                          "match": n == a.expect_params}
    if n != a.expect_params:
        _emit(rec, a.out)
        print(f"[dryrun] PARAM MISMATCH: built {n:,} vs expected "
              f"{a.expect_params:,} — STOP, this is a finding.")
        return 4

    # 4 — geometry facts read from the RUN args (never re-typed)
    geo = {k: run_args.get(k) for k in
           ("stage", "in_channels", "frame_h", "frame_w", "window",
            "projection", "frame_hfov", "v2_val_cache")}
    rec["run_args_geometry"] = geo

    # 5 — forward smoke at the model's true geometry, random frames.
    #     `vision_only_control` is the dumper's OWN control and is valid on any
    #     frames: it asserts an architectural property (v0/actions cannot reach
    #     the latent blocks), not a data property.
    C = int(stack_m.cfg.encoder.in_channels)
    H, W = int(run_args["frame_h"]), int(run_args["frame_w"])
    Wm = int(stack_m.cfg.predictor.window)
    smokes = []
    for b in [int(x) for x in a.smoke_batches.split(",") if x.strip()]:
        g = torch.Generator().manual_seed(0)
        frames = torch.rand(b, Wm, C, H, W, generator=g).to(device)
        acts2 = torch.randn(b, Wm, 2, generator=g).to(device) * 0.1
        v0 = (torch.rand(b, generator=g) * 15).to(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        got = SW._forward_latents(stack_m, frames, acts2, v0)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        ctl = SW.vision_only_control(stack_m, frames, acts2, v0, got)
        smokes.append({
            "batch": b, "forward_s": round(dt, 2),
            "windows_per_s": round(b / dt, 2),
            "cuda_max_mem_gb": (round(torch.cuda.max_memory_allocated() / 2**30,
                                      2) if device == "cuda" else None),
            "blocks": {k: list(got[k].shape) for k in
                       ("pooled", "pooled_seq", "ctx", "z_tac", "waypoints")},
            "sel_emitted": "sel" in got,
            "vision_only_control": {k: ctl[k] for k in
                                    ("ok", "vacuous", "blocks")
                                    if k in ctl},
        })
    rec["forward_smoke"] = smokes
    ok_ctl = all(s["vision_only_control"].get("ok") for s in smokes)
    rec["verdict"] = {
        "strict_load": True, "params_match": True,
        "vision_only_control_all_ok": ok_ctl,
        "sel_absent_as_expected": all(not s["sel_emitted"] for s in smokes)
        if prov.get("scorer_class") is None else None,
    }
    _emit(rec, a.out)
    print(f"[dryrun] OK — {n:,} params strict-loaded (step {step}), "
          f"forward smoke {[s['batch'] for s in smokes]} passed, "
          f"vision-only control ok={ok_ctl}")
    return 0


def _emit(rec, out):
    js = json.dumps(rec, indent=1, default=str)
    if out:
        Path(out).write_text(js, encoding="utf-8")
    print(js)


if __name__ == "__main__":
    raise SystemExit(main())
