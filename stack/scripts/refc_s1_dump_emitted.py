"""S1b's measurement — score the EMITTED fan, and its control, in ONE forward.

Pre-registration: `PREREG_S1_CLIMBOUT.md` §4.3 (blob pinned in raw/prereg_pin.json
BEFORE this ran).

WHAT IT MEASURES. `AnchoredDiffusionDecoder._decode(kv, cond, x_in, t)` returns the
confidence OF `x_in` and the offset that improves it; the loop emits `x = x_in +
off`. So the readout E-SEL banked as `refined_logits` is the confidence of the
estimate the LAST pass CONSUMED, and the trajectories that actually leave the
decoder are scored by NO head. With `sel_score_emitted = True` the decoder runs one
extra conf-only pass on the emitted fan and returns BOTH:

    refined_logits   = conf(X_k)   the EMITTED fan          <- S1b's treatment
    prefinal_logits  = conf(X_k-1) what S1 ranks today      <- its own control

⭐ BOTH COME FROM THE SAME FORWARD, so the paired contrast is exact and cannot be
confounded by float non-determinism between two runs on two hosts.

⛔ THE CONTROL THAT MUST HOLD. S1b keeps the extra pass's CONFIDENCE and DISCARDS
its OFFSET, so `anchor_traj` is bit-unchanged. The published oracle-in-fan (0.1914
base / 0.1640 XL) is defined on that fan and every D-SEL contrast is paired against
it, so a changed fan would silently re-baseline the whole comparison. Asserted here
against E-SEL's banked fan, bit-for-bit.

⛔ RASTER GATE (R-2026-08-02-a): base ACCEPTS a wrong token count and returns a
plausible wrong number SILENTLY. Asserted before a single window is scored.

Run (Thor, tanitad-edge - the INFERENCE venv):
    PYTHONPATH=~/TanitAD/stack:~/TanitAD/taniteval OMP_NUM_THREADS=6 \
    ~/venvs/tanitad-edge/bin/python refc_s1_dump_emitted.py \
        --ckpt ~/models/refc-base/ckpt.pt --preset base \
        --val ~/valdata/physicalai-val-0c5f7dac3b11 \
        --bank ~/fan_refined_refc-base-30k.pt \
        --out ~/s1_climbout/raw/fan_emitted_refc-base-30k.pt
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


def build_model(ckpt: str, preset: str, t_emit: int = -1):
    from taniteval.loaders import _apply_overrides
    from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                   refc_smoke_config, refc_xl_config)
    presets = {"small": refc_small_config, "base": refc_config,
               "xl": refc_xl_config, "smoke": refc_smoke_config}
    cfg = presets[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
    # S1b. ZERO parameters, so the checkpoint loads with no missing/unexpected
    # keys — asserted below, because a silent `strict=False` drop is exactly how
    # a "0-parameter" claim stops being true.
    cfg.sel_score_emitted = True
    cfg.sel_score_emitted_t = t_emit
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
    return model, cfg, ck.get("step"), missing, unexpected


def assert_raster(cfg, feats):
    gh, gw = cfg.encoder.grid_shape
    exp, got = (gh * 32, gw * 32), tuple(feats.shape[-2:])
    if got != exp:
        raise RuntimeError(f"C-raster REFUSES: val raster {got} but the arm "
                           f"declares grid_shape {(gh, gw)} => trained at {exp} "
                           f"(R-2026-08-02-a: base returns a wrong number SILENTLY)")
    return got, (gh, gw), gh * gw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val", required=True)
    ap.add_argument("--bank", required=True, help="E-SEL's bank, for the controls")
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
    # ⭐ POST-HOC (added after E-S1-0's first pass, and labelled as such): the
    # conf head is supervised by `loss_cls` at t=0 ONLY, so WHICH timestep token
    # the emitted readout is evaluated under is itself an axis — and it is free.
    # A second model, same weights, differing ONLY in that token.
    m_t0, _, _, _, _ = build_model(a.ckpt, a.preset, t_emit=0)
    m_t0 = m_t0.to(a.device).eval()
    if missing or unexpected:
        raise RuntimeError(f"S1b claims 0 parameters but the checkpoint load "
                           f"reports missing={list(missing)[:5]} "
                           f"unexpected={list(unexpected)[:5]}")
    model = model.to(a.device).eval()
    steps = cfg.decoder.diffusion_steps
    K_MAX = max(dd.WP_STEPS)

    files = data.list_val_episodes(a.val, a.episodes)
    dropped, eps = [], []
    for i, f in enumerate(files):
        try:
            eps.append(data.RawEp(data.load_episode(str(f), mmap=True), i))
        except Exception as exc:                 # episode id keeps its ORIGINAL index
            dropped.append({"file": Path(f).name, "episode_id": i,
                            "error": type(exc).__name__})
    raster, grid, n_tok = assert_raster(cfg, eps[0].feats)
    print(f"[s1b] raster {raster} grid {grid} = {n_tok} tokens :: C-raster PASS",
          flush=True)

    FAN, LOG, EMIT, EMIT0, PRE, SEL, GT, CV, EID, V0, SPD, HDG = (
        [], [], [], [], [], [], [], [], [], [], [], [])
    for ep in eps:
        fr, poses = ep.feats, ep.poses.float()
        starts = list(range(0, fr.shape[0] - WINDOW - K_MAX, STRIDE))
        for i in range(0, len(starts), a.batch):
            ch = starts[i:i + a.batch]
            last = torch.tensor([t + WINDOW - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WINDOW]) for t in ch]) \
                .to(a.device).float().div_(255.0)
            v0 = poses[last, 3].to(a.device)
            nav_cmd, _ = resolve_nav(model, fw, v0, steps, NAV_MODE,
                                     poses=poses, last=last)
            with torch.no_grad():
                o = model(fw, nav_cmd=nav_cmd, v0=v0, steps=steps)
                o0 = m_t0(fw, nav_cmd=nav_cmd, v0=v0, steps=steps)
            # the two forwards must differ ONLY in the readout's time token
            assert torch.equal(o["anchor_traj"], o0["anchor_traj"])
            assert torch.equal(o["prefinal_logits"], o0["prefinal_logits"])
            EMIT0.append(o0["refined_logits"].float().cpu())
            FAN.append(o["anchor_traj"].float().cpu())
            LOG.append(o["anchor_logits"].float().cpu())
            EMIT.append(o["refined_logits"].float().cpu())   # conf(X_k)  TREATMENT
            PRE.append(o["prefinal_logits"].float().cpu())   # conf(X_k-1) CONTROL
            SEL.append(o["sel_idx"].cpu())
            GT.append(dd.gt_ego_waypoints(ep.poses, last))
            CV.append(dd.baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            V0.append(poses[last, 3])
            SPD.append(ep.poses[last, 3])
            HDG.append(dd.net_heading_change_deg(ep.poses, last))
        print(f"[s1b] ep{ep.episode_id} {len(starts)} windows "
              f"({time.time() - t0:.0f}s)", flush=True)

    d = dict(fan=torch.cat(FAN), logits=torch.cat(LOG),
             emitted_logits=torch.cat(EMIT), emitted_t0_logits=torch.cat(EMIT0),
             prefinal_logits=torch.cat(PRE),
             refined_logits=torch.cat(EMIT),     # schema-compatible alias
             sel=torch.cat(SEL), gt=torch.cat(GT).float(),
             cv=torch.cat(CV).float(), eid=EID, v0=torch.cat(V0).float(),
             speed=torch.cat(SPD).float(), head_deg=torch.cat(HDG).float(),
             wp_steps=list(dd.WP_STEPS), ckpt=a.ckpt, ckpt_step=step,
             steps=steps, nav_mode=NAV_MODE,
             n_anchors=int(torch.cat(LOG).shape[1]),
             raster=raster, grid_shape=grid, n_tokens=n_tok,
             host=os.uname().nodename, episodes_scored=len(eps),
             episodes_dropped=dropped, torch_version=torch.__version__,
             sd_missing=len(missing), sd_unexpected=len(unexpected),
             provenance=(
                 "sel_score_emitted=True. `emitted_logits` = conf(X_k), the "
                 "confidence of the fan that is EMITTED; `prefinal_logits` = "
                 "conf(X_k-1), the readout S1 ranks on today and the one E-SEL "
                 "banked as `refined_logits`. Both from the SAME forward. "
                 "UNSUPERVISED either way: these weights never trained a refined "
                 "readout as a ranker, so this bounds S1b, not S1."))

    # ---- controls against E-SEL's bank ------------------------------------
    b = torch.load(a.bank, map_location="cpu", weights_only=False)
    ctl = {
        "n_windows_match": int(b["fan"].shape[0]) == int(d["fan"].shape[0]),
        "eid_match": list(b["eid"]) == list(d["eid"]),
        "gt_bit_identical": bool(torch.equal(b["gt"].float(), d["gt"])),
        "fan_bit_identical": bool(torch.equal(b["fan"].float(), d["fan"])),
        "fan_max_abs_diff": float((b["fan"].float() - d["fan"]).abs().max()),
        "logits_bit_identical": bool(torch.equal(b["logits"].float(), d["logits"])),
        "prefinal_reproduces_esel_refined": bool(
            torch.equal(b["refined_logits"].float(), d["prefinal_logits"])),
        "prefinal_max_abs_diff": float(
            (b["refined_logits"].float() - d["prefinal_logits"]).abs().max()),
        "argmax_logits_equals_sel": float(
            (d["logits"].argmax(1) == d["sel"]).float().mean()),
        "emitted_differs_from_prefinal": not bool(
            torch.equal(d["emitted_logits"], d["prefinal_logits"])),
        "emitted_t0_differs_from_emitted": not bool(
            torch.equal(d["emitted_t0_logits"], d["emitted_logits"])),
    }
    d["controls_vs_esel_bank"] = ctl
    torch.save(d, a.out)
    print(json.dumps(ctl, indent=2), flush=True)
    print(f"[s1b] {d['fan'].shape[0]} windows x {d['n_anchors']} anchors -> "
          f"{a.out} ({time.time() - t0:.0f}s)", flush=True)
    if not (ctl["fan_bit_identical"] and ctl["prefinal_reproduces_esel_refined"]):
        print("[s1b] ⛔ A CONTROL FAILED — the treatment is not isolated; do not "
              "quote a delta from this bank until it is explained", flush=True)
        return 2
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
