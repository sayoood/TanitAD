"""Bank REF-C's OWN LATENTS on the canonical 881 val40 windows — one inference pass.

Pre-registration: `…/Implementation/incoming/2026-08-04-lambda-findability/PREREG_E_EXP2.md`
(blob pinned in that package's `raw/prereg_pin.json` BEFORE this ran).

WHY THIS EXISTS. E-EXP-1 measured a CEILING: applying HAD's published per-window radial
scale λ to the already-selected trajectory recovers +0.1590 m (base) / +0.1651 m (XL) of
the 0.4728 / 0.4714 selection ADE — 56.5 % / 53.7 % of each arm's own oracle gap. That is
an ORACLE λ. The open question is FINDABILITY, and the two cheapest feature sets have
already failed (a single global λ recovers nothing; a `v0`-decile lookup is separably
WORSE than doing nothing). This dumps the third and most expensive feature set — the
model's own latents — so the question can be answered with 0 further GPU.

⛔ THE GATE THAT MAKES THE DUMP MEANINGFUL. The latent rows are only usable if they
correspond, row for row, to the banked fan the λ* target is computed from. So this asserts
`fan` / `gt` / `sel` / `eid` / `v0` **bit-identical** against the bank and REFUSES to write
a usable dump otherwise (it still writes, with `instrument_fail` populated, so the failure
is inspectable rather than invisible). That assert is also what makes a drifted pod
checkout safe here: a different model cannot reproduce the fan bit-for-bit.

⛔ RASTER GATE (R-2026-08-02-a): base ACCEPTS a wrong token count and returns a plausible
wrong number SILENTLY. Asserted before a single window is scored.

WHAT IS DUMPED — the list is CLOSED by the pre-registration:
    pooled        [n, F]      encoder pooled feature, LAST frame        VISION ONLY
    pooled_seq    [n, W, F]   the same for all W frames of the window   VISION ONLY
    ctx           [n, d_ctx]  StrategicCtx GRU summary                  VISION ONLY
    measurement   [n, d_m]    the ego+nav embedding — ⚠️ THE ECHO PATH, labelled as such

Run (Thor, tanitad-edge — the INFERENCE venv):
    PYTHONPATH=~/TanitAD/stack:~/TanitAD/taniteval OMP_NUM_THREADS=6 \
    ~/venvs/tanitad-edge/bin/python refc_dump_latents.py \
        --ckpt ~/models/refc-base/ckpt.pt --preset base \
        --val ~/valdata/physicalai-val-0c5f7dac3b11 \
        --bank ~/lambda_findability/fan_emitted_refc-base-30k.pt \
        --out ~/lambda_findability/latents_refc-base-30k.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")

WINDOW, STRIDE = 8, 8
NAV_MODE = "follow_constant"


def build_model(ckpt: str, preset: str):
    """Identical build to `refc_s1_dump_emitted.build_model` — same flag, so the emitted
    fan is the SAME fan. `sel_score_emitted` adds ZERO parameters (asserted by the
    missing/unexpected check in `main`)."""
    from taniteval.loaders import _apply_overrides
    from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                   refc_smoke_config, refc_xl_config)
    presets = {"small": refc_small_config, "base": refc_config,
               "xl": refc_xl_config, "smoke": refc_smoke_config}
    cfg = presets[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
    cfg.sel_score_emitted = True
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
    return model, cfg, ck.get("step"), missing, unexpected


def assert_raster(cfg, feats):
    gh, gw = cfg.encoder.grid_shape
    exp, got = (gh * 32, gw * 32), tuple(feats.shape[-2:])
    if got != exp:
        raise RuntimeError(f"C-raster REFUSES: val raster {got} but the arm declares "
                           f"grid_shape {(gh, gw)} => trained at {exp} "
                           f"(R-2026-08-02-a: base returns a wrong number SILENTLY)")
    return got, (gh, gw), gh * gw


class _PooledTap:
    """Captures the encoder's pooled output for every frame of every window.

    ⛔ Works for BOTH `RefCModel.forward` branches without assuming which one ran:
    with `hierarchy` the encoder is called once on `b*W` frames, without it once on `b`.
    The batch dimension is divided by the chunk's own `b` to recover W, so a config change
    cannot silently mis-shape the dump.
    """

    def __init__(self):
        self.buf = None

    def __call__(self, _mod, _inp, out):
        self.buf = out[1].detach()          # (fmap, pooled) -> pooled [B*, F]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val", required=True)
    ap.add_argument("--bank", required=True, help="the banked fan these latents must match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import driving_diagnostic as dd
    from taniteval import data
    from taniteval.refc_eval import resolve_nav

    t0 = time.time()
    model, cfg, step, missing, unexpected = build_model(a.ckpt, a.preset)
    fails = []
    if missing or unexpected:
        fails.append(f"state_dict missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    model = model.to(a.device).eval()
    steps = cfg.decoder.diffusion_steps
    K_MAX = max(dd.WP_STEPS)

    tap = _PooledTap()
    model.encoder.register_forward_hook(tap)

    files = data.list_val_episodes(a.val, a.episodes)
    dropped, eps = [], []
    for i, f in enumerate(files):
        try:
            eps.append(data.RawEp(data.load_episode(str(f), mmap=True), i))
        except Exception as exc:              # episode id keeps its ORIGINAL index
            dropped.append({"file": Path(f).name, "episode_id": i,
                            "error": type(exc).__name__})
    raster, grid, n_tok = assert_raster(cfg, eps[0].feats)
    print(f"[lat] raster {raster} grid {grid} = {n_tok} tokens :: C-raster PASS", flush=True)

    FAN, SEL, GT, EID, V0 = [], [], [], [], []
    POOL, PSEQ, CTX, MEAS = [], [], [], []
    for ep in eps:
        fr, poses = ep.feats, ep.poses.float()
        starts = list(range(0, fr.shape[0] - WINDOW - K_MAX, STRIDE))
        for i in range(0, len(starts), a.batch):
            ch = starts[i:i + a.batch]
            b = len(ch)
            last = torch.tensor([t + WINDOW - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WINDOW]) for t in ch]) \
                .to(a.device).float().div_(255.0)
            v0 = poses[last, 3].to(a.device)
            nav_cmd, _ = resolve_nav(model, fw, v0, steps, NAV_MODE,
                                     poses=poses, last=last)
            with torch.no_grad():
                o = model(fw, nav_cmd=nav_cmd, v0=v0, steps=steps)
            # --- the latents (list CLOSED by the pre-registration) -------------
            POOL.append(o["pooled"].float().cpu())
            MEAS.append(o["measurement"].float().cpu())
            CTX.append(o["ctx"].float().cpu() if "ctx" in o else torch.zeros(b, 0))
            pt = tap.buf
            if pt is None:
                raise RuntimeError("the encoder tap never fired — the forward did not "
                                   "call `model.encoder`, so `pooled_seq` would be a "
                                   "silent zero block")
            if pt.shape[0] == b * WINDOW:            # hierarchy: all W frames
                PSEQ.append(pt.reshape(b, WINDOW, -1).float().cpu())
            elif pt.shape[0] == b:                   # non-hierarchy: last frame only
                PSEQ.append(pt.reshape(b, 1, -1).float().cpu())
            else:
                raise RuntimeError(f"encoder tap returned batch {pt.shape[0]} for a "
                                   f"chunk of {b} windows x {WINDOW} frames — the "
                                   f"pooled sequence cannot be reshaped safely")
            tap.buf = None
            # --- the identity keys, for the bit-identity gate -------------------
            FAN.append(o["anchor_traj"].float().cpu())
            SEL.append(o["sel_idx"].cpu())
            GT.append(dd.gt_ego_waypoints(ep.poses, last))
            EID.extend([ep.episode_id] * b)
            V0.append(poses[last, 3])
        print(f"[lat] ep{ep.episode_id} {len(starts)} windows ({time.time() - t0:.0f}s)",
              flush=True)

    d = dict(pooled=torch.cat(POOL), pooled_seq=torch.cat(PSEQ),
             ctx=torch.cat(CTX), measurement=torch.cat(MEAS),
             fan=torch.cat(FAN), sel=torch.cat(SEL), gt=torch.cat(GT).float(),
             eid=EID, v0=torch.cat(V0).float(),
             wp_steps=list(dd.WP_STEPS), ckpt=a.ckpt, ckpt_step=step, steps=steps,
             nav_mode=NAV_MODE, raster=raster, grid_shape=grid, n_tokens=n_tok,
             host=os.uname().nodename, episodes_scored=len(eps),
             episodes_dropped=dropped, torch_version=torch.__version__,
             sd_missing=len(missing), sd_unexpected=len(unexpected),
             hierarchy=bool(getattr(cfg, "hierarchy", False)),
             provenance=(
                 "REF-C latents on the canonical 881 val40 windows, one inference pass, "
                 "no training. `pooled`/`pooled_seq`/`ctx` are VISION ONLY; `measurement` "
                 "is the EGO+NAV embedding and is the v0-echo path by construction. Built "
                 "with the same `sel_score_emitted=True` config as the banked fan emitter, "
                 "so the emitted fan is the same fan — asserted bit-for-bit below."))

    # ---- THE GATE: these latents must belong to the banked fan ---------------
    bk = torch.load(a.bank, map_location="cpu", weights_only=False)
    ctl = {
        "n_windows_match": int(bk["fan"].shape[0]) == int(d["fan"].shape[0]),
        "n_windows": int(d["fan"].shape[0]),
        "n_episodes": len(set(d["eid"])),
        "eid_match": list(bk["eid"]) == list(d["eid"]),
        "fan_bit_identical": bool(torch.equal(bk["fan"].float(), d["fan"])),
        "fan_max_abs_diff": float((bk["fan"].float() - d["fan"]).abs().max()),
        "gt_bit_identical": bool(torch.equal(bk["gt"].float(), d["gt"])),
        "sel_bit_identical": bool(torch.equal(bk["sel"], d["sel"])),
        "v0_bit_identical": bool(torch.equal(bk["v0"].float(), d["v0"])),
    }
    if not ctl["n_windows_match"] or not ctl["eid_match"]:
        fails.append("window grid does not match the bank")
    for k in ("fan_bit_identical", "gt_bit_identical", "sel_bit_identical",
              "v0_bit_identical"):
        if not ctl[k]:
            fails.append(k + " FAILED")
    if ctl["n_windows"] != 881 or ctl["n_episodes"] != 40:
        fails.append(f"counts {ctl['n_windows']}/{ctl['n_episodes']} != 881/40")
    d["controls_vs_bank"] = ctl
    d["instrument_fail"] = fails
    torch.save(d, a.out)
    print(json.dumps(ctl, indent=2), flush=True)
    print(f"[lat] pooled {tuple(d['pooled'].shape)} pooled_seq "
          f"{tuple(d['pooled_seq'].shape)} ctx {tuple(d['ctx'].shape)} meas "
          f"{tuple(d['measurement'].shape)} -> {a.out} ({time.time() - t0:.0f}s)",
          flush=True)
    if fails:
        print("[lat] ⛔ INSTRUMENT-FAIL: " + "; ".join(fails), flush=True)
        return 2
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
