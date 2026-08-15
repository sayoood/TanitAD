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

WHAT IS DUMPED — the LATENT list is CLOSED by the pre-registration:
    pooled        [n, F]      encoder pooled feature, LAST frame        VISION ONLY
    pooled_seq    [n, W, F]   the same for all W frames of the window   VISION ONLY
    ctx           [n, d_ctx]  StrategicCtx GRU summary                  VISION ONLY
    measurement   [n, d_m]    the ego+nav embedding — ⚠️ THE ECHO PATH, labelled as such

⭐ ALSO DUMPED, and NOT a latent: the E-WC2 ENDPOINT TARGETS (`--endpoint-steps`).
`gt_endpoint` / `endpoint_steps` / `endpoint_valid` are GROUND TRUTH derived from
poses — they add nothing to the model-side surface the pre-registration closed, and
they are what makes `scripts/e_wc2_sigma_star.py` (V6F_PLANNER_DESIGN.md §5.2) a
0-GPU analysis instead of a second inference pass. Default `20,60` = 2.0 s and 6.0 s,
the two horizons §5.2 requires.

⛔ THE K_MAX CONFLICT, AND WHY THE 6 s ENDPOINT IS MASKED RATHER THAN WIDENING THE GRID.
The window grid is `range(0, T - WINDOW - K_MAX, STRIDE)` with `K_MAX = max(WP_STEPS)
= 20`. Raising K_MAX to 60 so every window has a 6 s future would RE-SELECT WINDOWS —
the 881-window grid would shrink, the bit-identity gate against the banked fan would
fail, and cross-arm comparability (parity is sacred) would be gone. So the grid is
LEFT ALONE and the endpoints that run past the end of an episode are marked
`endpoint_valid=False` and written as NaN. E-WC2 EXCLUDES them per horizon with n
reported; ⛔ they are never imputed (the P1/P2 per-target-exclusion rule).

Run (Thor, tanitad-edge — the INFERENCE venv):
    PYTHONPATH=~/TanitAD/stack:~/TanitAD/taniteval OMP_NUM_THREADS=6 \
    ~/venvs/tanitad-edge/bin/python refc_dump_latents.py \
        --ckpt ~/models/refc-base/ckpt.pt --preset base \
        --val ~/valdata/physicalai-val-0c5f7dac3b11 \
        --bank ~/lambda_findability/fan_emitted_refc-base-30k.pt \
        --endpoint-steps 20,60 \
        --out ~/lambda_findability/latents_refc-base-30k.pt

⭐ THE DUMPS FROM 2026-08-04 ARE ALREADY IN THE REPO AND ARE MISSING ONLY THE ENDPOINTS.
`…/incoming/2026-08-04-lambda-findability/raw/latents_refc-{base,xl}-30k.pt` carry the
881×40 surface with `instrument_fail: []`. Because the endpoints are GROUND TRUTH, they
can be added with **NO model and NO GPU** — only the 40 val episodes' pose arrays:

    python refc_dump_latents.py --backfill-endpoints \
        --dump-in .../raw/latents_refc-xl-30k.pt \
        --val <the val40 corpus> --endpoint-steps 20,60 \
        --out .../raw/latents_refc-xl-30k-ep.pt

⛔ The backfill REFUSES unless the rebuilt per-window `eid` matches the banked `eid`
element-for-element AND the recomputed 2 s endpoint is bit-identical to the banked `gt`
column — otherwise every latent would be regressed onto a NEIGHBOUR's endpoint and σ
would come back inflated, i.e. a wrong answer that looks like a measurement.

Then, 0 GPU, anywhere:
    python scripts/e_wc2_sigma_star.py --dump latents_refc-base-30k.pt \
        --features pooled,ctx --out ewc2_sigma_star_refc-base.json
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
#: The grid's future budget. ⛔ PARITY: this is `max(driving_diagnostic.WP_STEPS)`
#: and must NOT be raised to reach a 6 s endpoint — that would re-select windows
#: and break both the 881-window grid and the fan bit-identity gate. Asserted
#: against `dd.WP_STEPS` in `main` (module-level so the endpoint backfill, which
#: never imports the model stack, uses the identical grid).
K_MAX_GRID = 20


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


def gt_endpoints_masked(poses, last, steps):
    """GT ego-frame endpoint at each horizon + a per-window validity mask.

    ``(endpoint [b, He, 2], valid [b, He])``. The poses are PADDED with the final
    pose repeated so ``driving_diagnostic.gt_ego_waypoints`` — the ONE
    implementation of the ego-frame transform, reused rather than re-derived — can
    be called unchanged for horizons that run past the end of the episode. Those
    entries are then masked ``False`` and OVERWRITTEN WITH NaN, so a consumer that
    ignores the mask fails loudly instead of silently reading "the ego stopped at
    the last recorded pose" as a 6 s goal.
    """
    import driving_diagnostic as dd
    T = int(poses.shape[0])
    pad = max(int(k) for k in steps)
    poses_pad = torch.cat([poses, poses[-1:].expand(pad, poses.shape[1])], dim=0)
    ep = dd.gt_ego_waypoints(poses_pad, last, wp_steps=[int(k) for k in steps])
    valid = torch.stack([(last + int(k)) < T for k in steps], dim=1)
    ep = ep.clone()
    ep[~valid] = float("nan")
    return ep, valid


def window_starts(n_frames: int) -> list[int]:
    """The canonical window grid. ONE definition, used by the dump pass and by
    the endpoint backfill, so the two cannot drift apart."""
    return list(range(0, int(n_frames) - WINDOW - K_MAX_GRID, STRIDE))


def backfill_endpoints(banked: dict, eps, steps, *, strict: bool = True) -> dict:
    """⭐ Add E-WC2's endpoint targets to an ALREADY-BANKED dump — **0 GPU, no model**.

    The endpoints are GROUND TRUTH derived from poses, so a dump that is missing
    them does NOT need a re-inference pass: it needs the 40 val episodes' pose
    arrays and nothing else. ``eps`` is any sequence of objects exposing
    ``.poses`` [T, 4] and ``.episode_id`` (``taniteval.data.RawEp`` satisfies it);
    ``.n_frames`` is used for the grid when present, else ``poses.shape[0]``.

    ⛔ THE ALIGNMENT GATE, and why it is not optional. E-WC2 fits banked LATENT
    rows against these endpoint rows. If the rebuilt grid is off by one window,
    every latent is regressed onto a NEIGHBOUR's endpoint and σ comes back
    inflated — a wrong answer that looks like a measurement. So two things are
    asserted before anything is written:
      1. the rebuilt per-window ``eid`` equals the banked ``eid``, element for
         element;
      2. wherever an endpoint horizon COINCIDES with one of the banked fan's
         waypoints, the recomputed endpoint is **bit-identical** to the banked
         ``gt`` column. That is a per-row fingerprint of `last`, the ego-frame
         transform and the pose source all at once.
    """
    import driving_diagnostic as dd
    eid_bank = [int(x) for x in banked["eid"]]
    GTE, EVAL, EIDS = [], [], []
    for ep in eps:
        poses = ep.poses if torch.is_tensor(ep.poses) else torch.as_tensor(ep.poses)
        poses = poses.float()
        n_frames = int(getattr(ep, "n_frames", poses.shape[0]))
        starts = window_starts(n_frames)
        if not starts:
            continue
        last = torch.tensor([t + WINDOW - 1 for t in starts])
        gte, gval = gt_endpoints_masked(poses, last, steps)
        GTE.append(gte.float())
        EVAL.append(gval)
        EIDS.extend([int(ep.episode_id)] * len(starts))
    if not GTE:
        raise RuntimeError("no windows rebuilt — the episodes are too short or "
                           "the wrong corpus was passed")
    out = dict(banked)
    out["gt_endpoint"] = torch.cat(GTE)
    out["endpoint_valid"] = torch.cat(EVAL)
    out["endpoint_steps"] = [int(k) for k in steps]

    ctl = {"rebuilt_windows": len(EIDS), "banked_windows": len(eid_bank),
           "eid_match": EIDS == eid_bank}
    fails = []
    if not ctl["eid_match"]:
        fails.append(f"rebuilt eid does not match the banked eid "
                     f"({len(EIDS)} vs {len(eid_bank)} windows) — the endpoint "
                     f"rows would not correspond to the latent rows")
    wps = [int(x) for x in banked.get("wp_steps", [])]
    for i, k in enumerate(out["endpoint_steps"]):
        if k in wps and ctl["eid_match"]:
            same = bool(torch.equal(out["gt_endpoint"][:, i],
                                    banked["gt"].float()[:, wps.index(k)]))
            ctl[f"endpoint_{k}_matches_gt"] = same
            if not same:
                fails.append(f"recomputed endpoint at step {k} is not "
                             f"bit-identical to the banked gt column — the pose "
                             f"source, `last`, or the ego frame differs")
    ctl["valid_frac"] = {str(k): round(float(out["endpoint_valid"][:, i]
                                             .float().mean()), 4)
                         for i, k in enumerate(out["endpoint_steps"])}
    ctl["fails"] = fails
    out["endpoint_backfill_controls"] = ctl
    out["endpoint_provenance"] = (
        "endpoints BACKFILLED from poses only — no model, no GPU, no "
        "re-inference. The latent rows are the original banked ones; the "
        "alignment gate (eid element-for-element + bit-identical endpoint at a "
        "coinciding fan waypoint) is in `endpoint_backfill_controls`.")
    if strict and fails:
        raise AssertionError("endpoint backfill REFUSES: " + "; ".join(fails))
    return out


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


def main_backfill(a) -> int:
    """⭐ `--backfill-endpoints`: add E-WC2's endpoints to a banked dump, 0 GPU.

    Reads ONLY the val episodes' pose arrays. No checkpoint, no CUDA, no model
    import — so it can run on any CPU box and does not need a training pause.
    """
    from taniteval import data

    t0 = time.time()
    banked = torch.load(a.dump_in, map_location="cpu", weights_only=False)
    ep_steps = [int(x) for x in str(a.endpoint_steps).split(",") if x.strip()]
    files = data.list_val_episodes(a.val, a.episodes)
    eps = []
    for i, f in enumerate(files):
        try:
            eps.append(data.RawEp(data.load_episode(str(f), mmap=True), i))
        except Exception as exc:
            print(f"[bf] ⛔ episode {i} ({Path(f).name}) failed to load: "
                  f"{type(exc).__name__} — the grid will not match and the "
                  f"backfill will refuse", flush=True)
    out = backfill_endpoints(banked, eps, ep_steps, strict=not a.no_strict)
    torch.save(out, a.out)
    print(json.dumps(out["endpoint_backfill_controls"], indent=2), flush=True)
    print(f"[bf] gt_endpoint {tuple(out['gt_endpoint'].shape)} steps {ep_steps} "
          f"-> {a.out} ({time.time() - t0:.0f}s, 0 GPU)", flush=True)
    return 2 if out["endpoint_backfill_controls"]["fails"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val", required=True)
    ap.add_argument("--bank", help="the banked fan these latents must match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backfill-endpoints", action="store_true",
                    help="0-GPU mode: add gt_endpoint/endpoint_valid to an "
                         "ALREADY-BANKED dump (--dump-in) from poses only. No "
                         "checkpoint, no CUDA, no training pause needed.")
    ap.add_argument("--dump-in", help="the banked dump to backfill")
    ap.add_argument("--no-strict", action="store_true",
                    help="write even when the alignment gate fails (the "
                         "controls are still recorded) — for inspection only")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--endpoint-steps", default="20,60",
                    help="E-WC2 endpoint horizons in 10 Hz steps (default "
                         "20,60 = 2.0 s and 6.0 s, the two §5.2 requires). "
                         "Does NOT change the window grid — see the K_MAX note.")
    a = ap.parse_args(argv)
    if a.backfill_endpoints:
        if not a.dump_in:
            ap.error("--backfill-endpoints requires --dump-in")
        return main_backfill(a)
    for req in ("ckpt", "bank"):
        if not getattr(a, req):
            ap.error(f"--{req} is required for the inference pass")
    ep_steps = [int(x) for x in str(a.endpoint_steps).split(",") if x.strip()]

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
    assert K_MAX == K_MAX_GRID, (
        f"K_MAX_GRID {K_MAX_GRID} != max(dd.WP_STEPS) {K_MAX} — the dump pass "
        f"and the endpoint backfill would build DIFFERENT window grids")

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
    GTE, EVAL = [], []                      # E-WC2 endpoint targets + validity
    for ep in eps:
        fr, poses = ep.feats, ep.poses.float()
        starts = window_starts(fr.shape[0])       # ONE grid definition
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
            # --- E-WC2 endpoint targets (GT, not a latent; masked, not imputed) -
            gte, gval = gt_endpoints_masked(ep.poses, last, ep_steps)
            GTE.append(gte.float())
            EVAL.append(gval)
        print(f"[lat] ep{ep.episode_id} {len(starts)} windows ({time.time() - t0:.0f}s)",
              flush=True)

    d = dict(pooled=torch.cat(POOL), pooled_seq=torch.cat(PSEQ),
             ctx=torch.cat(CTX), measurement=torch.cat(MEAS),
             fan=torch.cat(FAN), sel=torch.cat(SEL), gt=torch.cat(GT).float(),
             eid=EID, v0=torch.cat(V0).float(),
             gt_endpoint=torch.cat(GTE), endpoint_valid=torch.cat(EVAL),
             endpoint_steps=ep_steps,
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
                 "so the emitted fan is the same fan — asserted bit-for-bit below. "
                 "`gt_endpoint`/`endpoint_valid`/`endpoint_steps` are GROUND TRUTH "
                 "endpoint targets for E-WC2 (V6F_PLANNER_DESIGN.md §5.2), not "
                 "latents: ego-frame displacement at each horizon, NaN + valid=False "
                 "where the horizon runs past the end of the episode. The window grid "
                 "is UNCHANGED (K_MAX = max(WP_STEPS) = 20) so the 881-window parity "
                 "and the fan bit-identity gate hold."))

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
    # ---- E-WC2 endpoint self-control -----------------------------------------
    # Where an endpoint horizon COINCIDES with one of the fan's waypoints, the two
    # must be BIT-IDENTICAL: same window grid, same `last`, same ego-frame
    # convention. This is the cheap proof that the endpoint block belongs to the
    # same rows as the fan — without it a frame-convention or off-by-one error in
    # the 6 s target would be invisible (the 6 s horizon has nothing to check
    # against on its own).
    wps = list(dd.WP_STEPS)
    ctl["endpoint_steps"] = list(ep_steps)
    ctl["endpoint_valid_frac"] = {
        str(k): round(float(d["endpoint_valid"][:, i].float().mean()), 4)
        for i, k in enumerate(ep_steps)}
    for i, k in enumerate(ep_steps):
        if k in wps:
            same = bool(torch.equal(d["gt_endpoint"][:, i],
                                    d["gt"][:, wps.index(k)]))
            ctl[f"endpoint_{k}_matches_gt"] = same
            if not same:
                fails.append(f"endpoint step {k} != gt[:, {wps.index(k)}] — the "
                             f"endpoint block is not on the fan's rows/frame")
    if 60 not in ep_steps:
        fails.append("no 6.0 s (step 60) endpoint — V6F_PLANNER_DESIGN.md §5.2 "
                     "requires sigma at 2 s AND 6 s; E-WC2 will refuse a verdict")

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
          f"{tuple(d['measurement'].shape)} gt_endpoint "
          f"{tuple(d['gt_endpoint'].shape)} steps {ep_steps} -> {a.out} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if fails:
        print("[lat] ⛔ INSTRUMENT-FAIL: " + "; ".join(fails), flush=True)
        return 2
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
