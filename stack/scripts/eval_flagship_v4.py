"""eval_flagship_v4.py -- the v4-aware held-out eval harness.

WHY THIS EXISTS (2026-07-23): no v4-aware held-out eval driver existed before
this file. ``eval_flagship_v16.py`` STRICT-loads the v1.5/v1.6 head, which is
architecturally incompatible with v4's ``FlagshipV4Head`` (dense 20-step
horizons, factorised LAT x LON x DIST selection, the lambda_plan seam) -- and
nothing emitted a ``windows_<key>.pt`` for a v4 checkpoint, so
``taniteval.driving.from_windows()``'s episode-cluster-bootstrap primary
(ade@2s + miss@2m) was unreachable and CONTINUE/RESTART decisions on the whole
flagship line were BLOCKED (GATE_PROTOCOL.md).

TWO MODES. Per GATE_PROTOCOL.md O-03, MODE A must be run and must PASS before
MODE B's output is ever trusted to judge a checkpoint.

MODE A -- ``--canary-only`` (auto-selected when the checkpoint has no 'head'
    key, i.e. a plain flagship WorldModel like flagship4b-speedjerk-30k).
    Runs ONLY the WM-integrity canary: the deterministic operative-predictor
    rollout under TRUE actions -> grounding -> SE(2) -> ADE@2s. This is the
    SAME quantity ``train_flagship_v4.canary_rollout`` computes, and it is the
    SAME quantity behind flagship v1's registry headline (0.4522 heldout /
    0.4271 full-set) -- v1 has no separate "planner", so its canonical
    TanitEval number already IS this rollout (MODEL_REGISTRY.md 1.2: "the
    intent-free operative path that produces the trajectory ADE@2s scores").
    Use this against flagship4b-speedjerk-30k FIRST to prove the harness's
    encode / rollout / grounding / SE(2) plumbing is correct before it ever
    touches a v4 checkpoint.

MODE B -- a real v4/v4.1 checkpoint (keys: model, grounding, head[, goal_head]).
    Runs the PLANNER PATH (FlagshipV4Head-selected trajectory, lambda_plan=1,
    NOT fed true future actions) over the val cache, at BOTH:
      (i)  the head's own DENSE horizons (1..20 steps, train-loop-comparable
           -- this is what the trainer's in-loop ``evaluate_planner`` reports
           every ``--eval-every``, and it is NOT the same statistic as the
           historical "ade_0_2s": a mean over 20 dense steps 0.1-2.0s is
           diluted by the small early-horizon errors and reads LOWER than a
           mean over just the 4 endpoint waypoints).
      (ii) the historical 4-WAYPOINT convention (steps 5/10/15/20 = 0.5-2s,
           the ONLY convention any other arm in MODEL_REGISTRY.md is quoted
           in) -- persisted to windows_<key>.pt for
           ``taniteval.driving.from_windows()``'s episode-cluster-bootstrap
           ``ade_0_2s`` / ``miss_2m`` (the gate's actual primary metric).
    ALSO runs the WM canary on the (now jointly fine-tuned) trunk
    (-> wm_canary_ade_2s secondary) and reads seam_norm_ratio_max off the
    head's own forward telemetry (-> seam_norm_ratio_max secondary).

Usage (eval pod; PYTHONPATH must include this dir's parent AND this dir):

  # MODE A -- validate the harness against the KNOWN v1 number
  python3 eval_flagship_v4.py \\
      --ckpt /root/models/flagship-30k/ckpt.pt --canary-only \\
      --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \\
      --key v1-validation --out /root/taniteval/results/v1-validation.json

  # MODE B -- the real v4.1 gate eval (only AFTER mode A passes)
  python3 eval_flagship_v4.py \\
      --ckpt /root/models/flagship-v4.1-10k/ckpt_step10000.pt \\
      --anchors-dense /root/models/flagship-v4.1-10k/flagship_v4_anchors_dense.pt \\
      --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \\
      --key flagship-v4.1-10k --out /root/taniteval/results/flagship-v4.1-10k.json

  # ... and the same checkpoint WITHOUT the goal oracle (the deployable path)
  #     add:  --goal-mode produced

  # MODE B on the V2 COMPRESSED corpus -- THE v5 GATE PATH (2026-07-28)
  python3 eval_flagship_v4.py \\
      --ckpt /workspace/experiments/flagship-v5-.../ckpt_best.pt \\
      --anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt \\
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \\
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \\
      --v2-subframe 176x624 --require-parity \\
      --key flagship-v5-... --out .../flagship-v5-....json

⭐ TWO CORPUS FORMATS (2026-07-28). ``--val-cache`` is the RAW EPCACHE
(``ep_*.pt``) every historical v4 number was produced on. ``--v2-val-cache`` is
the V2 COMPRESSED corpus (``<clip_id>.v2ep.pt``) — **the only format v5 has**,
because at 120 deg / 256x640 the raw epcache is ~697 GB for one split and fits
on no host in the fleet. Before this existed, ``build_v2_providers`` was called
from exactly TWO files in the repo, both trainers: a v5 checkpoint was
**trainable but not evaluable on its own corpus**, so no gate could be run on
it. Exactly one of the two flags, never both.

⛔ ``--v2-subframe`` IS NOT OPTIONAL PAPERWORK. The v5 corpus is built at
256x640 and the model is trained on a CENTRED SLICE of it (the rig-clean fix).
Scoring the checkpoint on the un-sliced parent is the ``ego=`` failure in
geometry — trained with a capability, scored without it — and it does not
crash: same corpus, same clips, different pixels, plausible ADE. The frame this
harness resolves is therefore cross-checked against the ``geometry`` block in
the checkpoint's OWN ``config.json`` (``parity.assert_eval_frame_matches_run``)
and a mismatch is a REFUSAL that names the flag value which reproduces the run.

⚠️ GOAL PROVENANCE (``--goal-mode``, 2026-07-26 -- see ``goal_modes.py``).
MODE B's goal channels ``route`` / ``route_graded`` / ``vt_band`` are minted per
window from the ego's own FUTURE poses, so every v4 MODE-B number published
before this flag existed is a **goal-oracle number**: an upper bound, not a
deployed capability (``GATE_PROTOCOL.md`` 0.8, ``V4_FLAGSHIP_DESIGN.md``
558-560). ``--goal-mode`` makes the source explicit and stamps it into the
output JSON in every mode:

  ``oracle``   the DEFAULT and unchanged, bit-identical to the historical path
  ``produced`` two-pass: the model's own ``goal_head`` reads the encoded
               observation window and its output is fed back -- no future
  ``neutral``  the head's learned no-goal-given rows (the control)

The default is deliberately NOT changed: the historical record must stay
reproducible while it is being corrected. (``vt_speed`` is NOT among the oracle
fields despite what 0.8 says -- ``_goal_inputs`` overwrites it with the last
OBSERVED speed ``v0``; see ``goal_modes.py``.)
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# VAL-PARITY GUARD (2026-07-25). `--val-cache` used to be globbed with no
# integrity check at all, so a truncated or substituted val cache produced a
# plausible-looking WRONG ADE that nothing downstream could detect. One shared
# guard (`tanitad.data.parity`, the same module the trainers use) asserts the
# leaky-split refusal + the registered episode-count deployment before a single
# episode is read. `train_flagship_v4` already guards its own --val-cache; this
# is its EVAL-side mirror.
import goal_modes  # noqa: E402  (--goal-mode: oracle | produced | neutral)
import goal_provenance  # noqa: E402  (goal ORACLE disclosure)
from tanitad.data import parity  # noqa: E402

WP_STEPS = (5, 10, 15, 20)           # 0.5/1/1.5/2 s @10 Hz -- the ONLY convention
                                      # any other MODEL_REGISTRY.md row is quoted in
K_MAX = max(WP_STEPS)
REGISTRY_V1_HELDOUT = 0.4522          # MODEL_REGISTRY.md 1.2, 8-split episode
                                      # jackknife heldout mean (flagship-30k)
REGISTRY_V1_FULLSET = 0.4271          # same row, plain corpus-wide mean -- the
                                      # methodology-matched target for a plain-
                                      # mean canary rollout (no split/bootstrap)
VALIDATION_TOL = 0.05                 # metres -- "small tolerance" per the brief


# ============================================================================
# shared setup -- the v1 trunk architecture EVERY flagship arm shares
# ============================================================================
def _eval_cfg(frame=None):
    """CLAUDE.md source of truth: speed_input, action_dim=3, grad-ckpt OFF.

    ⭐ ``frame`` (2026-07-28) sizes the ENCODER for the frame this eval reads.
    It must be threaded into every config the harness builds, because
    ``load_v1_from_ck`` / ``load_v4_from_ck`` STRICT-load: an encoder built at
    256x256 against a checkpoint trained at 176x624 fails on
    ``encoder.pos_embed`` with a shape error whose cause is three files away.
    ``None`` (default) is byte-identical to the pre-2026-07-28 behaviour."""
    from tanitad.config import flagship4b_config
    cfg = flagship4b_config()
    cfg.speed_input = True
    cfg.predictor = dc.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dc.replace(cfg.tactical_pred, action_dim=3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    if frame is not None:
        from tanitad.geometry import apply_frame
        apply_frame(cfg, frame)
    return cfg


def _plan(cfg):
    """Byte-identical to train_flagship_v4.train()'s call, so the val cache
    windows exactly the way the real run's own in-loop eval windowed it."""
    from tanitad.train.flagship_losses import horizon_plan
    return horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)


def resolve_eval_frames(a, cfg, *, label: str = "eval_flagship_v4"):
    """⭐ THE EVAL-SIDE RIG-CLEAN SEAM. Returns ``(cache_frame, model_frame)``.

    ⛔ NOTHING IS REIMPLEMENTED. This delegates to
    ``train_flagship_v4.resolve_v2_frames``, the SAME function the trainer uses,
    so the evaluator and the trainer cannot resolve ``--v2-subframe`` two
    different ways. If they could, the rig-clean fix would be exactly as easy to
    lose on the eval side as it was on the train side.

    With no geometry flags at all this returns ``(CANONICAL_256, CANONICAL_256)``
    and leaves ``cfg`` byte-identical (``flagship4b_config()`` IS the deployed
    frame), so every existing raw-epcache eval is unchanged."""
    from train_flagship_v4 import resolve_v2_frames
    return resolve_v2_frames(a, cfg, label=label)


def build_v2_val_episodes(a, *, cache_frame, train_frame, verbose: bool = True):
    """⭐ WHERE A V2 EVAL READS FRAMES — the eval-side twin of
    ``train_flagship_v4.build_v2_data``.

    WHY THIS EXISTS (2026-07-28). v5 trains at 120° / 256x640, where the RAW
    epcache is ~697 GB for the val split alone and fits on no host in the fleet,
    so v5's corpus can only be a v2 compressed cache. Until this function
    existed, ``build_v2_providers`` was called from EXACTLY TWO files in the
    repo, both trainers — no evaluator, nothing in ``taniteval/``. A v5
    checkpoint was therefore **trainable but not evaluable on its own corpus**,
    and no gate could be run on it.

    ⛔ THIS IS NOT A THIRD DECODE PATH. The repo already carries three
    implementations of the v2 payload decode — ``scripts/v2_compressed.py``
    (the builder + its round-trip validator), ``scripts/slice_v2_cache.py``
    (the re-emitter) and ``tanitad/data/v2_dataset.py`` (the loader the trainers
    read through). This function calls the THIRD one, the same object the
    trainer's seam calls, and adds no decode of its own. Adding a fourth is how
    the rig-clean fix came to be applied to a function nobody's training path
    reached.

    The two arguments that matter are the same two the trainer's seam turns on:
    ``frame=`` (the providers deliver the SLICED raster) and ``parent=`` (the
    binding compares the shape they hand back against the frame the eval
    declares) — so a sub-frame that is configured but never applied is a REFUSAL
    before the first window, not a silently wrong published ADE.
    """
    from tanitad.data.v2_dataset import build_v2_providers
    dirs = a.v2_val_cache if isinstance(a.v2_val_cache, (list, tuple)) \
        else [a.v2_val_cache]
    rec = parity.assert_v2_parity_cache(
        dirs, label="--v2-val-cache", require=bool(getattr(a, "require_parity",
                                                           False)))
    slice_frame = None if train_frame == cache_frame else train_frame
    eps = build_v2_providers(dirs, lru_size=int(getattr(a, "v2_lru", 64)),
                             frame=slice_frame, verbose=verbose)
    if not eps:
        raise SystemExit(
            f"[v4-eval] no *.v2ep.pt under {dirs} — does --v2-val-cache point "
            f"at the split dir?")
    binding = parity.assert_v2_geometry_matches(
        rec, train_frame, label="--v2-val-cache", providers=eps,
        parent=cache_frame)
    return eps, {"val_parity": rec, "geometry_binding": binding}


def load_val_episodes(a, *, cache_frame, train_frame, verbose: bool = True):
    """THE ONE PLACE the evaluator resolves its val episodes, either format.

    Returns ``(episodes, provenance)``. ``--val-cache`` is the raw epcache
    (every historical v4 number); ``--v2-val-cache`` is the v2 compressed
    corpus (v5). Exactly one, never both — episode identity is a POSITION in
    one and a CLIP ID in the other."""
    if getattr(a, "v2_val_cache", None):
        return build_v2_val_episodes(a, cache_frame=cache_frame,
                                     train_frame=train_frame, verbose=verbose)
    from tanitad.data.mixing import load_episode
    rec = parity.assert_val_cache(a.val_cache, label='--val-cache')
    files = sorted(Path(a.val_cache).glob("ep_*.pt"))
    if not files:
        raise SystemExit(f"[v4-eval] no ep_*.pt under {a.val_cache}")
    eps = [load_episode(str(p), mmap=True) for p in files]
    return eps, {"val_parity": rec, "geometry_binding": None}


def assert_val_corpus_args(a) -> bool:
    """Resolve WHICH corpus format this eval reads. Returns ``True`` for v2.

    Mirrors ``train_flagship_v4.assert_corpus_args`` decision-for-decision so
    the trainer and the evaluator cannot disagree about what a corpus is."""
    raw = bool(getattr(a, "val_cache", None))
    v2 = bool(getattr(a, "v2_val_cache", None))
    if raw and v2:
        raise SystemExit(
            "[v4-eval] --val-cache (raw epcache) and --v2-val-cache (v2 "
            "compressed) are two CORPUS FORMATS, not two sources to mix. Pass "
            "exactly one.")
    if not raw and not v2:
        raise SystemExit(
            "[v4-eval] one of --val-cache (raw epcache) or --v2-val-cache "
            "(v2 compressed, the v5 corpus) is required.")
    return v2


def build_val_dataset_base(eps, cfg, plan):
    """Plain FlagshipWindowDataset (v1/v2.1 keys only) -- used for MODE A so the
    validation exercises the MINIMUM moving parts (no v4 label minting).

    Takes the ALREADY-RESOLVED episode list (``load_val_episodes``) so the two
    corpus formats meet at one seam instead of two globs."""
    from train_flagship4b import FlagshipWindowDataset
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    print(f"[v4-eval] MODE A val dataset (base): {len(eps)} episodes, "
          f"{len(ds)} windows (window={cfg.predictor.window} "
          f"max_horizon={plan.max_horizon})", flush=True)
    return ds


def build_val_dataset_v4(eps, cfg, plan):
    """FlagshipV4Dataset (mints v3 factorised + strategic labels on the fly) --
    needed for MODE B because the head's _goal_inputs reads vt_band/route/
    route_graded off the batch."""
    from flagship_v4_data import FlagshipV4Dataset
    ds = FlagshipV4Dataset(eps, window=cfg.predictor.window,
                           max_horizon=plan.max_horizon, maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels)
    print(f"[v4-eval] MODE B val dataset (v4): {len(eps)} episodes, "
          f"{len(ds)} windows (window={cfg.predictor.window} "
          f"max_horizon={plan.max_horizon})", flush=True)
    return ds


def run_canary(world, grounding, ds_val, device, episodes, stride, batch):
    """Thin wrapper around train_flagship_v4.canary_rollout -- reused, not
    reimplemented, so this harness inherits the SAME rollout/grounding/SE(2)
    mechanics the design already anchors flagship v1's 0.452 against."""
    from train_flagship_v4 import canary_rollout
    t0 = time.time()
    out = canary_rollout(world, grounding, ds_val, device, horizons=WP_STEPS,
                         k_max=K_MAX, episodes=episodes, stride=stride,
                         batch=batch, amp=(str(device) == "cuda"))
    out["wallclock_s"] = round(time.time() - t0, 1)
    return out


# ============================================================================
# MODE A -- load a plain v1-shaped checkpoint (model + grounding, no head)
# ============================================================================
def load_v1_from_ck(ck: dict, device, frame=None):
    """Inline equivalent of v15_prep.load_frozen_v1, taking an ALREADY-loaded
    ckpt dict (avoids reading a 3+ GB file twice). Refuses a non-speed trunk
    the same way (near-identical-name inversion risk, CLAUDE.md source of
    truth)."""
    from tanitad.models.fourbrain import WorldModel
    from tanitad.train.flagship_losses import build_grounding

    sd = ck["model"]
    a_dim = sd["predictor.act_emb.0.weight"].shape[1]
    if a_dim != 3:
        raise SystemExit(
            f"REFUSING: predictor action_dim={a_dim}, not 3. This must be the "
            "speed arm (flagship4b-speedjerk-30k), NOT the no-speed ablation "
            "control flagship4b-phase0-30k (CLAUDE.md source of truth).")
    cfg = _eval_cfg(frame)
    world = WorldModel(cfg)
    world.load_state_dict(sd)                             # STRICT
    world = world.to(device).eval()
    for p in world.parameters():
        p.requires_grad_(False)
    grounding = build_grounding(world.state_dim, device=device)
    grounding.load_state_dict(ck["grounding"])            # STRICT
    grounding.eval()
    for p in grounding.parameters():
        p.requires_grad_(False)
    step = int(ck.get("step", -1))
    print(f"[v4-eval] MODE A: loaded v1-shaped ckpt, step={step}, "
          f"state_dim={world.state_dim} (FROZEN)", flush=True)
    return world, grounding, step


# ============================================================================
# MODE B -- load a v4 checkpoint (model + grounding + head [+ goal_head])
# ============================================================================
def load_v4_from_ck(ck: dict, device, head_config_path=None,
                    anchors_dense_path=None, cond_imagination_override=None,
                    frame=None):
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.flagship_v4 import FlagshipV4Head, V4Config, v4_config
    from tanitad.refs.refc import DecoderConfig
    from tanitad.train.flagship_losses import build_grounding

    a_dim = ck["model"]["predictor.act_emb.0.weight"].shape[1]
    if a_dim != 3:
        raise SystemExit(f"REFUSING: predictor action_dim={a_dim}, not 3 -- "
                         "not a speed-input v4 trunk.")

    cfg = _eval_cfg(frame)
    world = WorldModel(cfg)
    world.load_state_dict(ck["model"])                    # STRICT
    world = world.to(device).eval()
    for p in world.parameters():
        p.requires_grad_(False)

    grounding = build_grounding(world.state_dim, device=device)
    grounding.load_state_dict(ck["grounding"])            # STRICT
    grounding.eval()
    for p in grounding.parameters():
        p.requires_grad_(False)

    hcfg = v4_config()
    src = "v4_config() defaults (NO sibling config.json found -- risk of "\
          "architecture mismatch if the real run overrode any field)"
    if head_config_path and Path(head_config_path).exists():
        hj = json.loads(Path(head_config_path).read_text())
        hc = dict(hj.get("head_cfg", hj))
        dec = hc.get("decoder")
        if isinstance(dec, dict):
            hc["decoder"] = DecoderConfig(**dec)
        for tk in ("horizons", "imag_read"):
            if tk in hc and isinstance(hc[tk], list):
                hc[tk] = tuple(hc[tk])
        # ⭐ vision-rank compat. A config.json written BEFORE the rank-16 lever
        # existed has no `vision_rank` key, and its checkpoint's factorised heads
        # are 2048-wide with no projection weights. Reconstructing it at the new
        # default would build a rank-16 head and fail the STRICT load. Absence of
        # the key is therefore read as "legacy raw arm" and routed through the
        # explicit override — a REPRODUCTION of an old arm, not a new design
        # choice, and it says so in the reason string.
        if "vision_rank" not in hc:
            from tanitad.models.vision_rank import (LEGACY_RAW_REASON,
                                                    RAW_STATE_DIM)
            hc["vision_rank"] = RAW_STATE_DIM
            hc["allow_raw_vision"] = True
            hc["vision_rank_reason"] = LEGACY_RAW_REASON
        hcfg = V4Config(**hc)
        src = f"sibling config.json ({head_config_path})"
    if cond_imagination_override is not None:
        hcfg.cond_imagination = cond_imagination_override
        src += f" [cond_imagination OVERRIDDEN to {cond_imagination_override}]"
    hcfg.state_dim = world.state_dim
    hcfg.window = cfg.predictor.window

    # ⭐ E-F FIX — INFER THE VISION RANK FROM THE CHECKPOINT, NOT FROM A CONFIG KEY.
    #
    # The compat branch above only fires when a sibling `config.json` EXISTS and
    # lacks a `vision_rank` key. With no config.json at all, `v4_config()` supplies
    # the NEW default (rank 16) and builds a projection the old checkpoint has no
    # weights for — 2 missing keys and 3 shape mismatches against a STRICT load,
    # which is every committed v4 number made unreproducible.
    #
    # The checkpoint itself answers this unambiguously and works in BOTH branches:
    # if it carries no `vision_rank_proj.*` tensor it is a pre-lever arm and its
    # factorised heads are raw-`state_dim` wide. Reading it from the weights also
    # cannot drift out of sync with a config file the way a key can.
    _hsd = ck.get("head") or ck.get("head_state") or {}
    if isinstance(_hsd, dict) and _hsd:
        _proj_w = next((v for k, v in _hsd.items()
                        if k.endswith("vision_rank_proj.proj.weight")), None)
        if _proj_w is None:
            from tanitad.models.vision_rank import (LEGACY_RAW_REASON,
                                                    RAW_STATE_DIM)
            if int(getattr(hcfg, "vision_rank", 0) or 0) < RAW_STATE_DIM:
                hcfg.vision_rank = RAW_STATE_DIM
                hcfg.allow_raw_vision = True
                hcfg.vision_rank_reason = LEGACY_RAW_REASON
                src += " [vision_rank INFERRED FROM CHECKPOINT: legacy raw arm]"
        else:
            # rank-carrying checkpoint: trust the weights over any config value
            hcfg.vision_rank = int(_proj_w.shape[0])
            src += f" [vision_rank INFERRED FROM CHECKPOINT: {hcfg.vision_rank}]"

    head = FlagshipV4Head(hcfg).to(device)
    if anchors_dense_path and Path(anchors_dense_path).exists():
        anc = torch.load(anchors_dense_path, map_location=device,
                         weights_only=False)
        head.load_anchors(
            (anc["anchors"] if isinstance(anc, dict) else anc).to(device))
        print(f"[v4-eval] loaded TRAINED dense anchors from {anchors_dense_path}",
              flush=True)
    else:
        print("[v4-eval] WARNING: no --anchors-dense found -- scoring against "
              "the head's DEFAULT (seed-0 FPS) anchor buffer. If the real run "
              "loaded a trained anchors file (check config.json "
              "args.anchors_dense) this will NOT reproduce its numbers.",
              flush=True)
    # STRICT — with ONE narrowly-scoped exemption, and it is not a loosening.
    #
    # `vision_rank_proj.basis_loaded` (models/vision_rank.py:155) is a scalar bool
    # BUFFER registered after the v4 from-scratch arm was trained. It records whether
    # a PCA basis was seeded; `VisionRankProj.forward` reads ONLY `is_raw`, `mu` and
    # `proj`, so this flag CANNOT move a computed value. A 30k checkpoint that predates
    # the buffer is therefore load-compatible in every way that affects a number, and
    # refusing it would strand a completed arm on a provenance flag.
    #
    # ⛔ This must never become a general `strict=False`. That would silently accept a
    # genuine architecture mismatch — exactly the failure this gate exists to catch, and
    # exactly what happened upstream of here: with no `--head-config` the loader fell
    # back to CURRENT defaults and the head arrived with five imagination tensors the
    # checkpoint never had. Anything missing or unexpected outside the inert set below
    # still raises.
    _INERT_BUFFERS = {"vision_rank_proj.basis_loaded"}
    _missing, _unexpected = head.load_state_dict(ck["head"], strict=False)
    _hard_missing = [k for k in _missing if k not in _INERT_BUFFERS]
    if _hard_missing or _unexpected:
        raise RuntimeError(
            "FlagshipV4Head state_dict does not match this checkpoint. "
            f"missing={_hard_missing} unexpected={list(_unexpected)}. "
            "If the missing keys are imagination/vision-rank tensors, the head was "
            "built from CURRENT defaults because no sibling config.json was found — "
            "pass --head-config <the run's config.json>.")
    for _k in _missing:
        print(f"[v4-eval] inert buffer absent from checkpoint (pre-dates it, reads "
              f"nothing in forward): {_k}", flush=True)
    head = head.to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)

    # GOAL PRODUCER (2026-07-26). The ONLY model-side producer of a goal on a v4
    # checkpoint is `GoalScalarHead` -> (ttm, curv_3s, curv_5s, tspeed_5s). It is
    # loaded here so `--goal-mode produced` has something to read; it is NOT used
    # by `--goal-mode oracle`, which is why loading it cannot move the oracle
    # numbers. `in_dim` is taken from the CHECKPOINT's own first-layer shape
    # rather than assumed, because the design lets it read either z_strat (128)
    # or the operative readout state (state_dim).
    goal_head = None
    if isinstance(ck.get("goal_head"), dict) and ck["goal_head"]:
        from tanitad.models.strategic_goal import (GoalScalarConfig,
                                                   GoalScalarHead)
        gsd = ck["goal_head"]
        w0 = gsd.get("net.0.weight")
        gcfg = GoalScalarConfig(in_dim=int(w0.shape[1]), hidden=int(w0.shape[0]),
                                n_out=int(gsd["net.2.weight"].shape[0]))
        goal_head = GoalScalarHead(gcfg)
        goal_head.load_state_dict(gsd)                    # STRICT
        goal_head = goal_head.to(device).eval()
        for p in goal_head.parameters():
            p.requires_grad_(False)
        print(f"[v4-eval] loaded goal_head (GoalScalarHead in_dim="
              f"{gcfg.in_dim} hidden={gcfg.hidden} n_out={gcfg.n_out}) -- "
              f"--goal-mode produced is AVAILABLE on this checkpoint",
              flush=True)
    else:
        print("[v4-eval] NO goal_head in this checkpoint -- --goal-mode "
              "produced is NOT available (no model-side goal producer exists); "
              "oracle/neutral only unless --goal-fallback is passed",
              flush=True)

    step = int(ck.get("step", -1))
    print(f"[v4-eval] MODE B: loaded v4 head cfg from {src}\n"
          f"  n_anchors={hcfg.n_anchors} horizons={hcfg.horizons[0]}.."
          f"{hcfg.horizons[-1]} (n={len(hcfg.horizons)}) "
          f"cond(states/imag/vtarget/route)="
          f"{hcfg.cond_states}/{hcfg.cond_imagination}/{hcfg.cond_vtarget}/"
          f"{hcfg.cond_route} factorised={hcfg.factorised} step={step}",
          flush=True)
    return world, grounding, head, step, hcfg, goal_head


@torch.no_grad()
def build_c2_scorer(scorer: str | None, world, grounding, device):
    """Resolve ``--select-rule c2-wm-ref``'s scoring world model. -> dict | None.

    ``scorer`` is a path to a v1-shaped checkpoint, or the literal ``"self"``.
    ⛔ ``None`` raises: this rule's SIGN depends on the scorer (MEASURED
    -0.2918 m under v1's world model, **+0.2090 m WORSE** self-scoring), so it
    may never be chosen by omission — see
    :mod:`tanitad.models.wm_reference_select`.
    """
    from tanitad.models.wm_reference_select import SELF_SCORING, resolve_scorer_tag
    tag = resolve_scorer_tag(scorer)
    if tag == SELF_SCORING:
        print("[v4-eval] ⛔ --c2-scorer self: the arm scores its OWN fan. This "
              "configuration was MEASURED separated-WORSE (+0.2090 "
              "[+0.0550, +0.3642], 881 windows / 40 clusters). Diagnostic only.",
              flush=True)
        return {"world": world, "step_readout": grounding.step["op"], "tag": tag,
                "self_scoring": True}
    t0 = time.time()
    ck = torch.load(tag, map_location="cpu", weights_only=False)
    w_s, g_s, s_step = load_v1_from_ck(ck, device)
    del ck
    print(f"[v4-eval] C2 scorer loaded from {tag} (step={s_step}) in "
          f"{time.time() - t0:.1f}s", flush=True)
    return {"world": w_s, "step_readout": g_s.step["op"], "tag": tag,
            "self_scoring": False, "scorer_step": s_step}


def apply_c2_selection(out: dict, horizons, ref, tag: str) -> dict:
    """Swap the head's argmax pick for C2's, keeping every derived key in step.

    Mutates ``out`` and returns the selection telemetry. ``out["anchor_traj"]``
    — the fan itself — is NOT touched, so ``oracle_ade`` and every coverage
    diagnostic are invariant by construction: this changes WHICH candidate is
    deployed and nothing else. Every key the head derives from the pick
    (``traj``, ``wp_seq``, ``waypoints``) is recomputed here, because leaving one
    of them describing the OLD pick is exactly how a re-selection silently
    half-lands.
    """
    import torch as _t
    from tanitad.models.wm_reference_select import select_by_wm_reference
    idx, _cost, tele = select_by_wm_reference(
        out["anchor_traj"], ref, horizons=tuple(horizons),
        baseline_idx=out["sel_idx"], scorer=tag)
    ar = _t.arange(out["anchor_traj"].shape[0], device=idx.device)
    out["sel_idx"] = idx
    out["traj"] = out["anchor_traj"][ar, idx]
    out["wp_seq"] = out["traj"]
    out["waypoints"] = {k: out["traj"][:, i] for i, k in enumerate(horizons)}
    return tele


def collect_planner(world, grounding, head, ds_val, device, dd, episodes,
                    stride, batch, wp_steps=WP_STEPS, goal_mode="oracle",
                    goal_head=None, goal_fallback=False,
                    select_rule="as-trained", c2=None, agree_dump=None,
                    oracle_channels=()):
    """v4 PLANNER PATH: head-selected trajectory (lambda_plan=1, NOT fed true
    future actions), re-encoding the CURRENT (jointly fine-tuned) trunk.

    ``goal_mode`` (2026-07-26) selects WHERE THE GOAL COMES FROM — see
    ``goal_modes.py``. ``"oracle"`` (the default, unchanged) delegates to
    ``train_flagship_v4._goal_inputs`` verbatim and is bit-identical to every
    v4 MODE-B number published before this flag existed. ``"produced"`` is the
    two-pass deployable path: the model's own ``goal_head`` reads the encoded
    observation window, and its output — not the ego's future — is fed back in.

    Returns ``(data, diag)``. ``data`` is windows_<key>.pt-ready
    (pred/gt/cv/eid/speed/head_deg/wp_steps/method) at the historical
    4-waypoint resolution, via driving_diagnostic's exact GT/CV/head_deg
    convention -- the SAME convention every other MODEL_REGISTRY.md row uses,
    so this arm is directly comparable. ``diag`` carries both the head's own
    DENSE-horizon quantities (train-loop-comparable) and a self-computed
    4-waypoint oracle/ADE (a cross-check against taniteval.driving's number
    computed from the SAME persisted windows via a completely different code
    path).

    ``select_rule`` (2026-07-28, DEFAULT ``"as-trained"`` — unchanged behaviour):
    ``"c2-wm-ref"`` replaces the head's argmax pick with C2, i.e. the fan
    candidate closest to ONE world-model reference roll-out
    (:mod:`tanitad.models.wm_reference_select`). It re-selects inside the SAME
    frozen fan, so ``oracle_ade`` is untouched by construction and only the pick
    moves. ``c2`` is :func:`build_c2_scorer`'s dict. The rule is OFF by default
    because it is MEASURED separated-WORSE on one of the two arms it has been
    measured on."""
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device
    from tanitad.models.flagship_v15 import SPEED_SCALE, v15_losses
    from tanitad.models.wm_reference_select import wm_reference_rollout
    import refb_labels

    assert goal_mode in goal_modes.GOAL_MODES, goal_mode
    if select_rule not in ("as-trained", "c2-wm-ref"):
        raise SystemExit(f"[v4-eval] unknown --select-rule {select_rule!r}")
    if select_rule == "c2-wm-ref" and c2 is None:
        raise SystemExit("[v4-eval] --select-rule c2-wm-ref needs --c2-scorer")
    c2_tele: dict = {}
    agree = goal_modes.GoalAgreement() if goal_mode == "produced" else None
    goal_rec: dict = {}
    head.eval()
    horizons = head.cfg.horizons
    if not set(wp_steps) <= set(horizons):
        raise SystemExit(f"[v4-eval] wp_steps {wp_steps} not a subset of the "
                         f"head's own horizons {horizons}")
    wp_pos = [horizons.index(k) for k in wp_steps]

    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    if not sel:
        raise SystemExit("[v4-eval] no windows selected -- check "
                         "--episodes/--stride against the val cache size")

    # DENSE PATH (2026-07-26). `taniteval.lateral.block` — the 10 Hz lateral /
    # longitudinal decomposition GATE_PROTOCOL.md 0 makes the horizon-honest
    # co-primary read against — REQUIRES `pred_dense`/`gt_dense` and SKIPS
    # without them, and `taniteval.rollout.collect` gained them on 2026-07-25
    # while this planner path did not. The head already emits the full dense
    # plan (`out["traj"]` at its own 1..20 horizons) and the dense target is
    # already built (`traj_tgt`): keeping them is persistence, not compute.
    # ADDITIVE ONLY — `pred`/`gt` keep their exact 4-waypoint meaning, so every
    # existing consumer and every historical comparison is untouched.
    dense_ok = tuple(horizons) == tuple(range(1, len(horizons) + 1))
    if not dense_ok:
        print(f"[v4-eval] head horizons {horizons} are not a contiguous 1..K at "
              f"10 Hz -> NOT emitting pred_dense/gt_dense (a non-10 Hz surface "
              f"labelled as one is exactly the mismatch this guard exists to "
              f"prevent).", flush=True)
    PD, GD = [], []
    P, G, C, EID, SPD, HDG = [], [], [], [], [], []
    dense_ade_sum = dense_oracle_sum = dense_selgap_sum = dense_missfde_sum = 0.0
    wp_oracle_sum = wp_ade_sum = 0.0
    seam_ratios: list[float] = []
    pose_cache: dict[int, torch.Tensor] = {}
    n = 0
    t0 = time.time()

    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        items = [ds_val[i] for i in idx]
        b = _to_device(default_collate(items), device)
        v0 = b["pose_last"][:, 3].float()
        traj_tgt = refb_labels.waypoint_targets(
            b["pose_last"].float(), b["future_poses"][:, :max(horizons)].float(),
            horizons)
        st = world.encode_window(b["frames"])
        # --- WHERE THE GOAL COMES FROM (pass 1 when goal_mode == "produced") --
        goal_kw, rec = goal_modes.resolve_goal(
            goal_mode, head=head, batch=b, v0=v0, states=st,
            goal_head=goal_head, allow_fallback=goal_fallback,
            oracle_channels=oracle_channels)
        if agree is not None:
            agree.update(goal_kw, b, rec.pop("scalars", None))
        goal_rec = {k: v for k, v in rec.items() if k != "scalars"}
        # --- pass 2: the planner head, on whatever goal that resolved to ------
        out = head(st, v0, lambda_plan=1.0, **goal_kw)

        # --- OPTIONAL C2 re-selection over the SAME frozen fan ---------------
        # One extra roll-out per WINDOW (not per candidate): the world model is
        # rolled once under the observed action, zero-order-held, and the fan
        # candidate closest to that reference is taken. Nothing about the fan,
        # the goal or the decoder changes; only `sel_idx` and the trajectory
        # read off it. Placed BEFORE v15_losses so every diagnostic below
        # (ade / sel_gap / rank_acc / dense / 4wp) describes the DEPLOYED pick.
        if select_rule == "c2-wm-ref":
            with torch.no_grad():
                st_s = (st if c2["self_scoring"]
                        else c2["world"].encode_window(b["frames"]))
                aw = b["actions"].float()
                if world.predictor.cfg.action_dim == 3:
                    vch = (v0 / SPEED_SCALE)[:, None, None]
                    aw = torch.cat([aw, vch.expand(-1, aw.shape[1], -1)], dim=-1)
                ref = wm_reference_rollout(c2["world"].predictor, st_s, aw,
                                           c2["step_readout"], max(horizons))
                tele = apply_c2_selection(out, horizons, ref, c2["tag"])
            for k_t, v_t in tele.items():
                if isinstance(v_t, (int, float)) and not isinstance(v_t, bool):
                    c2_tele[k_t] = c2_tele.get(k_t, 0.0) + float(v_t) * (
                        1.0 if k_t.startswith("n_") else len(idx))
                else:
                    c2_tele[k_t] = v_t

        lg = v15_losses(out, head.decoder.anchors, traj_tgt)

        bs = len(idx)
        dense_ade_sum += float(lg["ade"]) * bs
        dense_oracle_sum += float(lg["oracle_ade"]) * bs
        dense_selgap_sum += float(lg["sel_gap"]) * bs
        fde_dense = (out["traj"][:, -1] - traj_tgt[:, -1]).norm(dim=-1)
        dense_missfde_sum += float((fde_dense > 2.0).float().sum())

        seam = out.get("telemetry", {}).get("seam_norm_ratio_max")
        if seam is not None:
            seam_ratios.append(float(seam))

        # ---- 4-waypoint sub-selection: the historical convention -----------
        pred4 = out["traj"][:, wp_pos]                             # [b,4,2]
        tgt4 = traj_tgt[:, wp_pos]                                  # [b,4,2]
        fan4 = out["anchor_traj"][:, :, wp_pos, :]                  # [b,N,4,2]
        fan_err4 = (fan4 - tgt4[:, None]).norm(dim=-1).mean(dim=-1)  # [b,N]
        wp_oracle_sum += float(fan_err4.min(dim=1).values.sum())
        wp_ade_sum += float((pred4 - tgt4).norm(dim=-1).mean(dim=-1).sum())
        P.append(pred4.float().cpu())
        if dense_ok:
            PD.append(out["traj"].float().cpu())
            GD.append(traj_tgt.float().cpu())
        n += bs

        for i in idx:
            e_i, t = ds_val.index[i]
            po = pose_cache.get(e_i)
            if po is None:
                po = torch.as_tensor(ds_val.episodes[e_i].poses,
                                     dtype=torch.float32)
                pose_cache[e_i] = po
            last = torch.tensor([t + ds_val.window - 1])
            G.append(dd.gt_ego_waypoints(po, last, wp_steps=wp_steps))
            C.append(dd.baseline_waypoints(po, last,
                                           wp_steps=wp_steps)["constant_velocity"])
            HDG.append(dd.net_heading_change_deg(po, last))
            EID.append(int(ds_val.episodes[e_i].episode_id))
        SPD.append(v0.float().cpu())
        if b0 % (batch * 10) == 0:
            print(f"  [v4-eval] planner-path {n}/{len(sel)} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # NOTE (2026-07-26): the two suffixes below are built OUTSIDE the f-string.
    # A multi-line expression inside a replacement field is PEP 701, i.e.
    # Python >= 3.12 ONLY, and the designated n>=200 eval host (pod2) runs
    # 3.11.10 — where the previous form raised `SyntaxError: unterminated
    # string literal` at import time and made EVERY v4 eval path (including
    # the registered closed-loop co-primary, which imports load_v4_from_ck
    # from this module) un-runnable on that host. Same string, no PEP 701.
    _oracle_tag = " [GOAL ORACLE]" if goal_mode == "oracle" else ""
    _fb = goal_rec.get("fallback")
    _fb_tag = (" [FALLBACK=" + str(_fb) + "]") if _fb else ""
    _sel_tag = ("" if select_rule == "as-trained" else
                f", SELECT-RULE=C2 wm-reference (scorer={c2['tag']})")
    if c2_tele:
        for k_t in list(c2_tele):
            if not k_t.startswith("n_") and isinstance(c2_tele[k_t], float):
                c2_tele[k_t] = round(c2_tele[k_t] / max(n, 1), 6)
        c2_tele["_read"] = ("selected_frac 1.000 = the rule is UNCONDITIONAL. A "
                            "gated variant must report its firing rate here or "
                            "its whole-set value is unknown (C19).")
    data = {
        "pred": torch.cat(P), "gt": torch.cat(G).float(),
        "cv": torch.cat(C).float(), "eid": EID,
        "speed": torch.cat(SPD).float(), "head_deg": torch.cat(HDG).float(),
        "wp_steps": list(wp_steps),
        "method": (f"flagship-v4: joint-trained trunk (re-encoded via "
                  f"world.encode_window) + FlagshipV4Head dense-"
                  f"{len(horizons)}-step anchored planner (argmax-conf, "
                  f"lambda_plan=1.0, {head.decoder.anchors.shape[0]} anchors), "
                  f"4wp sub-selected at steps {wp_steps}, "
                  f"goal_mode={goal_mode}"
                  f"{_oracle_tag}{_fb_tag}{_sel_tag}"),
    }
    if dense_ok:
        # `gt_dense` comes from `refb_labels.waypoint_targets` (the head's own
        # target), NOT from `dd.gt_ego_waypoints` (which produced the sparse
        # `gt`). The two paths agree — that agreement is exactly what the
        # existing `cross_check_ade_0_2s_selfcomputed_vs_driving_py` block
        # measures — but they are different code, so it is stamped, not assumed.
        data.update(pred_dense=torch.cat(PD), gt_dense=torch.cat(GD),
                    dense_steps=list(horizons), dt_s=0.1,
                    dense_provenance=(
                        "pred_dense = head out['traj'] at its own dense "
                        "horizons; gt_dense = refb_labels.waypoint_targets "
                        "(NOT driving_diagnostic.gt_ego_waypoints, which built "
                        "the sparse `gt`). pred[:, j] == "
                        "pred_dense[:, wp_steps[j]-1] by construction."))
    # PER-WINDOW goal dump (produced mode only). `report()` summarises these and
    # drops them, which is why the route threshold could not be swept without a
    # re-run. Costs one torch.save of tensors already in memory.
    if agree is not None and agree_dump is not None:
        agree.dump(agree_dump)

    diag = {
        "n_windows": n,
        "select_rule": select_rule,
        "c2_selection": c2_tele or None,
        "goal_mode": goal_mode,
        "goal_mode_record": goal_rec,
        "goal_agreement_vs_oracle": agree.report() if agree is not None else None,
        "wallclock_s": round(time.time() - t0, 1),
        "dense_headhorizons_ade_2s": dense_ade_sum / n,
        "dense_headhorizons_oracle_ade": dense_oracle_sum / n,
        "dense_headhorizons_sel_gap": dense_selgap_sum / n,
        "dense_headhorizons_miss_at_2m": dense_missfde_sum / n,
        "wp4_oracle_ade_0_2s": wp_oracle_sum / n,
        "wp4_ade_0_2s_selfcomputed": wp_ade_sum / n,
        "seam_norm_ratio_max": max(seam_ratios) if seam_ratios else None,
        "n_seam_samples": len(seam_ratios),
        "horizons_dense": list(horizons),
        "wp_steps": list(wp_steps),
    }
    return data, diag


def _dig_metric(ev: dict, metric: str):
    """Best-effort read of a metric's point estimate from a merged results
    JSON, trying the same node paths run_gate.py._cluster_node/_read_eval_metric
    tries (cluster_bootstrap.model / driving.cluster_bootstrap.model /
    headline / driving.headline / full_set.model / driving.full_set.model)."""
    for path in (("cluster_bootstrap", "model"),
                ("driving", "cluster_bootstrap", "model"),
                ("headline",), ("driving", "headline"),
                ("full_set", "model"), ("driving", "full_set", "model")):
        node = ev
        ok = True
        for k in path:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and isinstance(node, dict) and metric in node:
            v = node[metric]
            return v.get("mean") if isinstance(v, dict) else v
    return None


# ============================================================================
# main
# ============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        "eval_flagship_v4", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-cache", default=None,
                    help="RAW EPCACHE val split (ep_*.pt) -- every historical "
                         "v4 number. Mutually exclusive with --v2-val-cache.")
    ap.add_argument("--v2-val-cache", default=None, nargs="+",
                    help="⭐ V2 COMPRESSED val split (<clip_id>.v2ep.pt) -- THE "
                         "v5 CORPUS. v5 trains at 120deg / 256x640, where the "
                         "raw epcache is ~697 GB and fits on no host in the "
                         "fleet, so a v5 checkpoint can only be scored through "
                         "this path. Mutually exclusive with --val-cache.")
    ap.add_argument("--v2-lru", type=int, default=64,
                    help="decoded-payload LRU per v2 cache dir")
    # w120-class checkpoints: declare the CACHE geometry (same flags and the same
    # resolution path as train_flagship_v4 — frame_from_args getattr-defaults keep
    # every existing invocation byte-identical when these are absent).
    ap.add_argument("--frame-h", type=int, default=None)
    ap.add_argument("--frame-w", type=int, default=None)
    ap.add_argument("--frame-hfov", type=float, default=None)
    ap.add_argument("--f-ref", type=float, default=None)
    ap.add_argument("--projection", default=None)
    ap.add_argument("--v2-subframe", default=None,
                    help="⭐ HxW (e.g. 176x624) -- the CENTRED SUB-FRAME the "
                         "MODEL reads out of the cache, or 'none' to read the "
                         "cache exactly as built. THIS MUST MATCH THE RUN: a "
                         "checkpoint trained on a slice and scored on the "
                         "parent is the `ego=` failure in geometry. The frame "
                         "is cross-checked against the checkpoint's own "
                         "config.json, so omitting it is a REFUSAL, not a "
                         "wrong number.")
    ap.add_argument("--require-parity", action="store_true",
                    help="refuse an UNREGISTERED v2 val cache instead of "
                         "warning (the raw path's val guard already refuses an "
                         "unregistered episode count)")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-config", default=None,
                    help="sibling config.json (default: auto-detect "
                         "<ckpt-dir>/config.json)")
    ap.add_argument("--anchors-dense", default=None,
                    help="trained dense-anchor buffer. STRONGLY recommended to "
                         "pass explicitly -- the local path will differ from "
                         "whatever config.json's args.anchors_dense recorded "
                         "(that path lives on the TRAINING pod, not this one)")
    ap.add_argument("--cond-imagination", choices=("auto", "true", "false"),
                    default="auto")
    ap.add_argument("--goal-mode", choices=goal_modes.GOAL_MODES,
                    default="oracle",
                    help="WHERE THE GOAL COMES FROM (MODE B only). 'oracle' "
                         "(DEFAULT, unchanged): route/route_graded/vt_band "
                         "minted from the ego's own FUTURE poses -- the "
                         "historical path, bit-identical to every published v4 "
                         "MODE-B number, an UPPER BOUND not a deployable "
                         "result. 'produced': two-pass -- the model's own "
                         "goal_head reads the encoded observation window and "
                         "its output is fed back (no future, no label) -- THE "
                         "DEPLOYABLE PATH. 'neutral': the head's learned "
                         "no-goal-given rows, the control that makes the "
                         "oracle-vs-produced gap readable. Always recorded in "
                         "the output JSON as goal_provenance.goal_mode.")
    ap.add_argument("--goal-fallback", action="store_true",
                    help="allow --goal-mode produced to fall back to the "
                         "NEUTRAL rows on a checkpoint with no goal_head. The "
                         "result is stamped fallback='neutral' and is NOT a "
                         "produced-goal number. Off by default: silently "
                         "substituting a different goal source is the failure "
                         "this flag exists to prevent.")
    ap.add_argument("--select-rule", choices=("as-trained", "c2-wm-ref"),
                    default="as-trained",
                    help="WHICH candidate of the emitted fan is deployed. "
                         "'as-trained' (DEFAULT, unchanged) = the head's own "
                         "argmax. 'c2-wm-ref' = the candidate closest to ONE "
                         "world-model reference roll-out (C2). MEASURED on 881 "
                         "windows / 40 clusters: -0.2918 [-0.4233, -0.1598] "
                         "with v1's world model as the scorer, but +0.2090 "
                         "[+0.0550, +0.3642] WORSE when an arm scores its own "
                         "fan -- hence OFF by default and hence --c2-scorer.")
    ap.add_argument("--c2-scorer", default=None,
                    help="path to the v1-shaped checkpoint that SCORES the fan "
                         "under --select-rule c2-wm-ref, or the literal 'self' "
                         "to ask for the (measured-worse) self-scoring "
                         "diagnostic on purpose. There is no default: the sign "
                         "of the rule depends on this argument.")
    ap.add_argument("--canary-only", action="store_true",
                    help="force MODE A even if the ckpt has a 'head' key")
    ap.add_argument("--oracle-channels", default=None,
                    help="DIAGNOSTIC ONLY. Comma-separated goal channels to take from the "
                         "ORACLE while the rest stay produced, e.g. 'vt_band,vt_speed'. "
                         "Exists to ATTRIBUTE the oracle-vs-produced ADE gap per channel "
                         "(route is already measured at <=2.6%% of it). "
                         "⛔ An arm using this is fed a future-derived quantity and is NOT "
                         "a deployable number.")
    ap.add_argument("--route-thr", type=float, default=None,
                    help="OVERRIDE the produced-goal route decision threshold on "
                         "|tanh(curv_5s / CURV_TURN_PER_M)|. Default (unset) = the "
                         "historical tanh(1.0)=0.7616, so no published number moves "
                         "unless you opt in. MEASURED 2026-07-29: 0.35 lifts balanced "
                         "route accuracy 0.4242 -> 0.5493 with no training, but that "
                         "value was CHOSEN ON THE EVAL SPLIT and is an upper bound "
                         "until confirmed elsewhere.")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-bench", action="store_true",
                    help="skip the legacy taniteval.bench.run call")
    ap.add_argument("--skip-driving", action="store_true",
                    help="skip the taniteval.driving tier-0 panel")
    ap.add_argument("--results-dir", default=None,
                    help="dir for windows_<key>.pt / driving_<key>.json "
                         "(default: dirname(--out))")
    a = ap.parse_args(argv)

    # Fail on the ARGUMENTS, not 3 GB of checkpoint later: --select-rule
    # c2-wm-ref has no default scorer, because the rule's sign depends on it.
    if a.select_rule == "c2-wm-ref" and a.c2_scorer is None:
        from tanitad.models.wm_reference_select import resolve_scorer_tag
        try:
            resolve_scorer_tag(None)
        except ValueError as ex:
            raise SystemExit(f"[v4-eval] --select-rule c2-wm-ref: {ex}")
    if a.select_rule == "as-trained" and a.c2_scorer is not None:
        raise SystemExit("[v4-eval] --c2-scorer given without --select-rule "
                         "c2-wm-ref: it would be silently ignored.")
    use_v2 = assert_val_corpus_args(a)
    if a.v2_subframe and not use_v2:
        raise SystemExit(
            "[v4-eval] --v2-subframe applies to --v2-val-cache only. The raw "
            "epcache holds whatever frame it was built at and there is no "
            "loader slice on that path.")
    if a.select_rule == "c2-wm-ref" and (a.v2_subframe or a.frame_h or a.frame_w):
        # The C2 scorer is a SEPARATE v1-shaped checkpoint trained at the
        # deployed 256x256 frame; it encodes the same batch as the arm. On a
        # non-deployed frame its encoder cannot take those tensors at all, and
        # a rule whose SIGN depends on its scorer must not be reached broken.
        raise SystemExit(
            "[v4-eval] --select-rule c2-wm-ref is not available on a "
            "non-deployed frame: the C2 scorer is a separate 256x256 v1 trunk "
            "and it encodes THE SAME batch as the arm. Score the fan "
            "'as-trained', or train a scorer at this frame first.")

    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[v4-eval] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path(a.results_dir) if a.results_dir else out_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # ⭐ GEOMETRY FIRST, BEFORE THE CHECKPOINT. `cfg` sizes the encoder's
    # positional embedding, the loaders below are STRICT, and the frame is read
    # from flags that a gate command can silently omit — so it is resolved,
    # printed and cross-checked against the run's own config.json before 3 GB of
    # weights are touched.
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg)
    plan = _plan(cfg)

    # The run's own config.json — the ONLY artifact that records the frame the
    # checkpoint was TRAINED on. Read early: it is a few KB and it decides
    # whether this eval may proceed at all.
    ckpt_dir = Path(a.ckpt).parent
    head_cfg_path = a.head_config or (ckpt_dir / "config.json")
    run_cfg = None
    if Path(head_cfg_path).exists():
        try:
            run_cfg = json.loads(Path(head_cfg_path).read_text())
        except Exception as ex:                              # pragma: no cover
            print(f"[v4-eval] WARNING: could not parse {head_cfg_path}: {ex}",
                  flush=True)
    frame_check = parity.assert_eval_frame_matches_run(
        run_cfg, model_frame, label="--ckpt vs eval frame",
        cache_frame=(cache_frame if use_v2 else None))
    if not frame_check["checked"]:
        print(f"[v4-eval] ⚠ FRAME UNVERIFIED: {frame_check['note']}", flush=True)

    print(f"[v4-eval] loading checkpoint {a.ckpt} ...", flush=True)
    t_load0 = time.time()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    print(f"[v4-eval] ckpt loaded in {time.time() - t_load0:.1f}s, "
          f"keys={sorted(ck.keys()) if isinstance(ck, dict) else type(ck)}",
          flush=True)
    is_v4 = isinstance(ck, dict) and ("head" in ck) and not a.canary_only

    val_eps, val_prov = load_val_episodes(a, cache_frame=cache_frame,
                                          train_frame=model_frame)
    corpus_report = {
        "corpus_format": ("v2 compressed (<clip_id>.v2ep.pt)" if use_v2
                          else "raw epcache (ep_%05d.pt)"),
        "val_cache": (list(a.v2_val_cache) if use_v2 else a.val_cache),
        "require_parity": bool(a.require_parity),
        "eval_frame": model_frame.report(),
        "cache_frame": (cache_frame.report() if use_v2 else None),
        "v2_subframe_arg": a.v2_subframe,
        "frame_matches_checkpoint": frame_check,
        "val_parity": val_prov["val_parity"],
        "geometry_binding": val_prov["geometry_binding"],
    }

    if not is_v4:
        # ------------------------------- MODE A -------------------------------
        ds_val = build_val_dataset_base(val_eps, cfg, plan)
        world, grounding, step = load_v1_from_ck(ck, device, frame=model_frame)
        can = run_canary(world, grounding, ds_val, device, a.episodes,
                         a.stride, a.batch)
        delta_heldout = can["canary_ade@2s"] - REGISTRY_V1_HELDOUT
        delta_fullset = can["canary_ade@2s"] - REGISTRY_V1_FULLSET
        reproduces = abs(delta_fullset) <= VALIDATION_TOL
        result = {
            "mode": "MODE_A_canary_only_validation",
            "evidence_class": "MEASURED (ours; artifact = this JSON)",
            "ckpt": a.ckpt, "ckpt_step": step, "key": a.key,
            "val_cache": a.val_cache, "episodes": a.episodes,
            "corpus": corpus_report,
            "stride": a.stride, "batch": a.batch,
            "n_windows": can["n"], "wallclock_s": can["wallclock_s"],
            "canary_ade_2s_MEASURED": can["canary_ade@2s"],
            "registry_reference": {
                "ade_0_2s_heldout_8split_jackknife": REGISTRY_V1_HELDOUT,
                "ade_0_2s_full_set_plain_mean": REGISTRY_V1_FULLSET,
                "source": "Project Steering/MODEL_REGISTRY.md section 1.2 "
                          "(flagship4b-speedjerk-30k, TanitEval key "
                          "flagship-30k, step 29999)",
                "note": ("canary_rollout computes a PLAIN corpus-wide mean "
                        "over all selected windows -- the methodology-"
                        "matched comparison is the FULL-SET figure (0.4271), "
                        "not the 8-split episode-jackknife heldout mean "
                        "(0.4522, a different statistical construction).")},
            "delta_vs_full_set": round(delta_fullset, 4),
            "delta_vs_heldout": round(delta_heldout, 4),
            "tolerance_m": VALIDATION_TOL,
            "HARNESS_VALIDATED": bool(reproduces),
            "verdict": (
                "HARNESS VALIDATED -- reproduces the registry v1 number "
                "within tolerance; safe to proceed to a v4 checkpoint "
                "(GATE_PROTOCOL O-03 satisfied)."
                if reproduces else
                "HARNESS NOT VALIDATED -- does NOT reproduce v1's known "
                "number within tolerance. DO NOT proceed to score any v4 "
                "checkpoint with this harness until the discrepancy is "
                "found and fixed."),
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"\n[v4-eval] -> {out_path}", flush=True)
        return 0 if reproduces else 1

    # ---------------------------------- MODE B --------------------------------
    ds_val = build_val_dataset_v4(val_eps, cfg, plan)
    anchors_path = a.anchors_dense
    if anchors_path is None and isinstance(run_cfg, dict):
        cand = (run_cfg.get("args") or {}).get("anchors_dense")
        if cand and Path(cand).exists():
            anchors_path = cand
    cond_imag_override = {"auto": None, "true": True,
                          "false": False}[a.cond_imagination]

    world, grounding, head, step, hcfg, goal_head = load_v4_from_ck(
        ck, device, head_config_path=head_cfg_path,
        anchors_dense_path=anchors_path, frame=model_frame,
        cond_imagination_override=cond_imag_override)
    del ck  # free the raw state-dict copy before the eval loop

    if a.goal_mode == "oracle":
        print("[v4-eval] --goal-mode oracle (DEFAULT): route/route_graded/"
              "vt_band come from the ego's own FUTURE poses. This reproduces "
              "the historical numbers and is an UPPER BOUND, not a deployable "
              "result (GATE_PROTOCOL.md 0.8).", flush=True)
    else:
        print(f"[v4-eval] --goal-mode {a.goal_mode}: NO future-derived quantity "
              f"enters the evaluated forward pass.", flush=True)

    import driving_diagnostic as dd

    print("[v4-eval] running WM canary (wm_canary_ade_2s secondary)...",
          flush=True)
    can = run_canary(world, grounding, ds_val, device, a.episodes, a.stride,
                     a.batch)
    print(f"[v4-eval] canary_ade@2s={can['canary_ade@2s']:.4f} n={can['n']} "
          f"({can['wallclock_s']:.0f}s)", flush=True)

    c2 = (build_c2_scorer(a.c2_scorer, world, grounding, device)
          if a.select_rule == "c2-wm-ref" else None)

    print("[v4-eval] running the planner path (the gate primary)...",
          flush=True)
    if a.route_thr is not None:
        goal_modes.set_route_thr(a.route_thr)
        print(f"[v4-eval] ⚠ ROUTE THRESHOLD OVERRIDDEN to {a.route_thr} "
              f"(default is tanh(1.0)=0.7616). This number is NOT comparable to "
              f"published v4 produced-goal results measured at the default.",
              flush=True)

    data, diag = collect_planner(world, grounding, head, ds_val, device, dd,
                                 a.episodes, a.stride, a.batch,
                                 goal_mode=a.goal_mode, goal_head=goal_head,
                                 goal_fallback=a.goal_fallback,
                                 select_rule=a.select_rule, c2=c2,
                                 agree_dump=(results_dir
                                             / f"goalagree_{a.key}.pt"),
                                 oracle_channels=tuple(
                                     c.strip() for c in a.oracle_channels.split(",")
                                     if c.strip()) if a.oracle_channels else ())

    wp = results_dir / f"windows_{a.key}.pt"
    _keys = ["pred", "gt", "cv", "eid", "speed", "head_deg", "wp_steps"]
    # additive dense keys (see collect_planner) — present only when the head's
    # horizons really are a contiguous 10 Hz 1..K surface.
    _keys += [k for k in ("pred_dense", "gt_dense", "dense_steps", "dt_s",
                          "dense_provenance") if k in data]
    torch.save({k: data[k] for k in _keys}, wp)
    print(f"[v4-eval] windows -> {wp} (enables the episode-cluster-bootstrap "
          f"primary via taniteval.driving.from_windows())", flush=True)

    res_json_path = results_dir / f"{a.key}.json"
    res: dict = {}
    if not a.skip_bench:
        try:
            from taniteval import bench
            res = bench.run(data)
        except Exception as ex:
            print(f"[v4-eval] bench.run FAILED (non-fatal -- driving.py still "
                 f"runs below): {type(ex).__name__}: {str(ex)[:300]}",
                 flush=True)
    res.setdefault("key", a.key)
    res["method"] = data["method"]
    res["ckpt"] = a.ckpt
    res["ckpt_step"] = step
    res["v4_diagnostics"] = diag
    # CORPUS + FRAME PROVENANCE. Stamped so no v5 artifact can be read without
    # knowing WHICH corpus format and WHICH frame produced it — the two facts a
    # v2-capable evaluator newly makes variable.
    res["corpus"] = corpus_report
    # GOAL PROVENANCE (2026-07-26, PC1 item #5). Stamped for EVERY mode, so no
    # artifact can be read without knowing where its goal came from. Under the
    # default `--goal-mode oracle` this is the same ORACLE disclosure as before
    # (route / route_graded / vt_band minted per window from the episode's full
    # future poses); under `produced` it records the deployable path.
    # (MODE A returns above, so this line is only ever reached in MODE B.)
    res["goal_provenance"] = goal_modes.provenance(
        a.goal_mode, cfg=head.cfg, fallback=diag["goal_mode_record"].get("fallback"),
        extra={"goal_mode_record": diag["goal_mode_record"],
               "goal_agreement_vs_oracle": diag["goal_agreement_vs_oracle"]})
    res["wm_canary_ade_2s"] = can["canary_ade@2s"]
    res["wm_canary_n"] = can["n"]
    # SELECTION PROVENANCE — stamped for every mode so no artifact can be read
    # without knowing WHICH candidate of the fan it deployed.
    res["select_rule"] = a.select_rule
    res["c2_scorer"] = (None if c2 is None else
                        {"tag": c2["tag"], "self_scoring": c2["self_scoring"],
                         "step": c2.get("scorer_step")})
    res_json_path.write_text(json.dumps(res, indent=2, default=str),
                             encoding="utf-8")
    print(f"[v4-eval] -> {res_json_path}", flush=True)

    if not a.skip_driving:
        try:
            from taniteval import driving as tdriving
            tdriving.run_and_save(a.key, res_dir=results_dir)
        except Exception as ex:
            print(f"[v4-eval] taniteval.driving FAILED: {type(ex).__name__}: "
                 f"{str(ex)[:300]}", flush=True)

    merged = json.loads(res_json_path.read_text())
    ade_02s = _dig_metric(merged, "ade_0_2s")
    miss_2m = _dig_metric(merged, "miss_2m")
    if miss_2m is None:
        miss_2m = _dig_metric(merged, "miss_rate@2m")

    def _sec(value, thr, direction):
        if value is None:
            return {"value": None, "threshold": thr, "pass": None,
                   "note": "NOT COMPUTED"}
        ok = (value <= thr) if direction == "<=" else (value >= thr)
        return {"value": value, "threshold": thr, "pass": bool(ok)}

    summary = {
        "key": a.key, "ckpt": a.ckpt, "ckpt_step": step,
        "evidence_class": "MEASURED (ours; artifacts = "
                          f"{res_json_path.name}, {wp.name})",
        # GATE_PROTOCOL.md 0.8: a privileged-input primary may be reported but
        # may NOT be quoted as the model's performance. The verdict therefore
        # carries its own goal provenance, next to the number it qualifies.
        "goal_mode": a.goal_mode,
        "goal_provenance_short": (
            "ORACLE (route/route_graded/vt_band from the ego's own FUTURE "
            "poses) -- report as 'MODE B, goal-oracle inputs, ADE@2s = X', "
            "NEVER as 'the flagship achieves X'"
            if a.goal_mode == "oracle" else
            f"{a.goal_mode.upper()} -- no future-derived quantity entered the "
            f"evaluated forward pass"),
        "primary_is_deployable_capability": bool(a.goal_mode != "oracle"),
        "gate_primary_ade_0_2s": _sec(ade_02s, 0.60, "<="),
        "kill_secondaries": {
            "wm_canary_ade_2s": _sec(can["canary_ade@2s"], 0.55, "<="),
            "oracle_in_fan": {
                **_sec(diag["wp4_oracle_ade_0_2s"], 0.30, "<="),
                "note": "4-waypoint resolution (steps 5/10/15/20), comparable "
                        "to v1.5-ab's 0.3073 -- NOT the dense-20 "
                        "'oracle_ade@2s' the in-loop train log prints"},
            "miss_at_2m": _sec(miss_2m, 0.10, "<="),
            "seam_norm_ratio_max": _sec(diag["seam_norm_ratio_max"], 1.0, "<="),
            "encoder_touching_levers": {
                "value": 2, "threshold": 2, "pass": True,
                "evidence_class": "PUBLISHED (V4_FLAGSHIP_DESIGN.md / "
                                  "--print-launch design audit)",
                "note": "static design fact (lambda_plan + strategic = 2 of "
                        "2 encoder-touching levers, door CLOSED per "
                        "MODEL_REGISTRY.md retraction 07-21); not a GPU "
                        "measurement, not re-derived here"},
            "speed_benefit_recovered_frac": {
                "value": None, "pass": None,
                "note": "NOT BUILT this session -- new metric (P8), needs "
                        "its own definition off the two in-repo train logs "
                        "per V4_FLAGSHIP_DESIGN.md 17.3; no emitter exists "
                        "yet anywhere in the codebase"},
            "deploy_tick_p99_ms": {
                "value": None, "pass": None,
                "note": "NOT MEASURED this session -- needs the "
                        "efficiency.py latency-panel harness (CUDA-graph "
                        "capture, batch-1 profiling under gpu_lock.sh "
                        "exclusivity); out of scope for a correctness-first "
                        "harness build, flagged as the first thing to cut "
                        "per V4_FLAGSHIP_DESIGN.md 8 if it misses"},
            "nonav_route_beats_majority": {
                "value": None, "pass": None,
                "note": "NOT REACHABLE on this checkpoint -- v4.1's "
                        "goal_head (GoalScalarHead) only regresses "
                        "CONTINUOUS scalars (ttm/curv_3s/curv_5s/tspeed_5s); "
                        "no ROUTE classifier exists yet (P6 strategic "
                        "planner not landed). taniteval.hierarchy.py's "
                        "vision_route_beats_majority needs a nav-"
                        "conditioned route head this checkpoint does not "
                        "have -- this is the produced-goal fallback "
                        "(V4_FLAGSHIP_DESIGN.md 2.6) territory"},
        },
        "diagnostics_dense_headhorizons_train_loop_comparable": {
            k: diag[k] for k in (
                "dense_headhorizons_ade_2s", "dense_headhorizons_oracle_ade",
                "dense_headhorizons_sel_gap", "dense_headhorizons_miss_at_2m")},
        "goal_agreement_vs_oracle": diag["goal_agreement_vs_oracle"],
        "cross_check_ade_0_2s_selfcomputed_vs_driving_py": {
            "selfcomputed_from_forward_pass": diag["wp4_ade_0_2s_selfcomputed"],
            "driving_py_from_persisted_windows": ade_02s,
            "agree_within_1pct": bool(
                ade_02s is not None and
                abs(diag["wp4_ade_0_2s_selfcomputed"] - ade_02s)
                < 0.01 * max(ade_02s, 1e-6)),
            "note": "two independent code paths over the SAME forward-pass "
                    "output (direct tensor math here vs. persisted-tensor "
                    "reload + taniteval.driving.tier0 there); disagreement "
                    "would indicate an ego-frame convention mismatch"},
    }
    diag_out = results_dir / f"{a.key}_v4_diagnostics.json"
    diag_out.write_text(json.dumps(summary, indent=2, default=str),
                        encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"\n[v4-eval] -> {diag_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
