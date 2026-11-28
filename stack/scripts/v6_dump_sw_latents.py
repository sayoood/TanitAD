"""⭐ STEP 1 of the selector-admission recipe — dump **v6 S-W latents** in E-WC2's
dump contract. This is the one script the PI's *"eventually we need a tactical
selector"* was blocked on, and it was a SCRIPT, not a decision.

WHY THIS EXISTS. `E4_SELECTOR_RESOLUTION.md` (2026-08-17) repaired the reader half
of SEL-1's pre-registered reopening path and MEASURED, at three probes, that the
producer half did not exist: nothing in `stack/scripts/` dumps v6 S-W latents in
`e_wc2_sigma_star.DUMP_CONTRACT`. `refc_dump_latents.build_model` builds a
`RefCModel` and cannot load a v6 checkpoint; `probe_latent_state.py` reads v6
checkpoints but emits P1/P2 retention, not an E-WC2 dump. `v6_chain.py admission`
therefore emitted step 1 as `⛔ NOT BUILT`. This file is that step.

THE FOUR-STEP RECIPE (`v6_chain.sw_admission_recipe`), for orientation:
    1. THIS SCRIPT           — GPU, ~10-25 min at the S-W → S-T boundary
    2. e_wc2_sigma_star.py   — CPU, no GPU, no model
    3. v6_chain.py admission — adjudicate vs the PRE-REGISTERED thresholds
    4. v6_chain.py commands --step S-T:goal — only reachable if 3 says FUNDED


⛔ THE CONTRACT IS DERIVED FROM THE CONSUMER'S SOURCE, NOT FROM PROSE
=====================================================================
The defect this programme just repaired (C94) was **a fixture that modelled the
CONSUMER'S EXPECTATION instead of the PRODUCER'S OUTPUT** — `test_v6_chain.py`
hand-wrote `{"sigma_2s_m": …}`, the shape the reader wanted, so the join was
never exercised and a green suite certified a connection that did not exist. So
every key below is justified against the line of the consumer that reads it, and
`tests/test_v6_dump_sw_latents.py` runs **this producer → the real estimator →
the real chain reader** end-to-end on a planted σ rather than asserting shapes.

⭐ **AND ONE COUPLING THAT IS EASY TO MISS, MEASURED HERE.** The chain's admission
gate resolves `references_and_ratios.sigma_perax_2s_m`
(`v6_chain.SW_SIGMA_LOCATIONS`). `e_wc2_sigma_star.run` writes that key **only
inside** `if refs.get("available") and vstep in sig` (e_wc2_sigma_star.py:799-810),
and `fan_references` returns `available: False` when the dump carries no
`fan`/`gt` (:443-449). ⇒ **a latent-only dump produces an admission artifact with
NO σ in it at all, and the gate stays dead** — the exact failure E4 repaired, one
door along. The fan is therefore NOT optional here, and `max(wp_steps)` must be
20 or `run` emits `MISMATCH` and no σ (:800-806).


WHAT IS DUMPED, AND WHY EACH BLOCK IS ADMISSIBLE
================================================
    pooled      [n, d_op]   `z_op`      — operative latent at the window's LAST
                                          frame. VISION ONLY.
    pooled_seq  [n, W, d_op] `z_op_win` — the same for all W model frames.
                                          VISION ONLY.
    ctx         [n, d_str]  `z_str`     — the STRATEGIC summary; the v6 analogue
                                          of REF-C's `ctx` (its StrategicCtx GRU
                                          summary). VISION ONLY.
    z_tac       [n, d_tac]  `z_tac`     — the TACTICAL summary. VISION ONLY, but
                                          it has no built-in class in E-WC2, so
                                          using it needs `--declare
                                          z_tac=VISION_ONLY`.
    v0          [n]                     — ego speed at t0. MEASURED_PRESENT
                                          (admissible, PI 2026-08-16) and it
                                          carries the ANTI-ECHO OBLIGATION.

⛔ **VISION-ONLY IS MEASURED HERE, NOT ASSERTED FROM THE DIAGRAM.** `z_op` comes
from `encode_window(frames)` and `z_tac`/`z_str` from adapters over it
(v6.py:4045-4066) — `actions` and `v0` reach only the predictors and the
emission. That is an argument; the control is the measurement. `--vision-only-
control` (default ON) re-runs the first batch with `v0` **and** `actions`
PERMUTED ACROSS THE BATCH and requires `pooled`/`pooled_seq`/`ctx`/`z_tac` to be
**bit-identical**. Same shape as the v0-shuffle in `probe_latent_state.py
--speed-echo-control`: a claim about what an input cannot reach, tested by
changing that input. A failure is an `instrument_fail`, not a warning.

⛔ `measurement` (REF-C's ego+nav ECHO block) HAS NO v6 ANALOGUE AND IS NOT
EMITTED. v6 has no nav input at all. A block named `measurement` would be read by
E-WC2 as the labelled-inadmissible control and there is nothing here to put in it.


THE GRID — CANONICAL, AND WHY THE MODEL WINDOW DOES NOT MOVE IT
===============================================================
`WINDOW=8`, `STRIDE=8`, `K_MAX_GRID=20` → the 881-window val40 grid. The grid
constants and `window_starts` are IMPORTED from `refc_dump_latents`, so the two
producers cannot drift; and `EpisodeWindowDataset`'s own index rule
(`t_max = frames - window - max_horizon`, `range(t_max)`, data/_contract.py:118-121)
is that same grid when the dataset is built at `window=8, max_horizon=20`, which
is what :func:`build_val_grid` does.

⚠️ **v6's predictor window is 6, not 8** (`train_v6_staged.py:3513`), and that is
NOT allowed to re-select windows. Parity is sacred: a different grid would break
`eid` correspondence with the banked REF-C dumps, with §3.1's surface, and with
`refc_dump_latents --backfill-endpoints`. ⇒ **the grid stays at WINDOW=8 and the
model is fed the LAST `cfg.predictor.window` frames of each window**, which ends
on the same frame (`last = t + 8 - 1`), so `pose_last`, the ego frame, `gt` and
`gt_endpoint` are all on identical rows. A model window > 8 is REFUSED (frames
cannot be fabricated), never silently truncated.

⛔ THE K_MAX CONFLICT, unchanged from `refc_dump_latents`: the 6 s endpoint does
not exist for the last ~5 windows of an episode. Those rows are `endpoint_valid=
False` and NaN — never imputed, and the grid is never widened to reach them.


⛔ `sel` IS EMITTED ONLY IF THE ARM REALLY HAS A SCORER
======================================================
`sel` is *"the incumbent selector's chosen index"*, and `sel_ade` — the σ/ADE
denominator — is computed from it. On an S-W arm built with `--selector none`
there is no `cand_score`, so `emit` produces no `sel_*` key at all (MEASURED on
the production stack, `E4_SELECTOR_RESOLUTION.md` §2). **Fabricating one would be
this session's own root-cause class in a new costume**: candidate 0 is arbitrary
and argmin-over-candidates is the ORACLE, so either would manufacture a
denominator and with it a §5.2 verdict. So `sel` is ABSENT with
`sel_absent_reason` recorded, and the consequences are stated rather than hidden:

  * `e_wc2_sigma_star`'s OWN §5.2 verdict becomes `NO_VERDICT` (a `validate_dump`
    contract problem becomes a guard) — correct, because σ/ADE is not computable;
  * ⭐ the **S-W admission gate is unaffected**: `SW_LATENT_ADMISSION` is defined
    on ABSOLUTE metres (σ ≤ 0.80 m FUNDED · > 1.41 m REFUSED), and
    `sigma_perax_2s_m` is written before the guards run. Both halves are pinned
    by executed tests, not argued here.

When the arm DOES carry a scorer, `sel = plan["sel_score"].argmax(-1)` — the
incumbent rule, verbatim from `GoalDistanceScorer.forward`'s docstring
(v6.py:1581-1582) and from `train_v6_staged.py:1567-1572`.


PROVENANCE, AND THE ONE THING THAT IS NEVER TYPED
=================================================
The architecture AND the frame geometry are rebuilt from the CHECKPOINT'S OWN
run args (`tanitad.eval.v6_probe_trunk._run_args` → `build_stack_from_args`,
`strict=True`), so `--v2-subframe` / `--frame-h` / `--frame-w` are read from the
run rather than re-typed. `--args-from` is the ONE escape hatch, for a
weights-only snapshot that travelled without its `config.json`; it is a place to
point at a run's record, never a place to type a geometry (the E1 rule).

USAGE
    python stack/scripts/v6_dump_sw_latents.py --preflight-only
    PYTHONPATH=<stack>:<taniteval> OMP_NUM_THREADS=6 python \\
        stack/scripts/v6_dump_sw_latents.py \\
            --ckpt /workspace/experiments/v6F-SW-30k/ckpt.pt \\
            --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \\
            --out /workspace/experiments/v6F-SW-30k/ewc2_sw_dump.pt
then, 0 GPU:
    python stack/scripts/e_wc2_sigma_star.py --dump <that> --features pooled,ctx \\
        --out /workspace/experiments/v6F-SW-30k/ewc2_sw_latents.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:                       # scripts/ importable
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:                # the stack root
    sys.path.insert(0, str(_HERE.parent))

# ⛔ THE GRID IS IMPORTED, NEVER RE-DERIVED. One definition shared with the
# REF-C producer and with `--backfill-endpoints`, so the two dumps land on the
# same rows and `eid` corresponds element-for-element.
from refc_dump_latents import (K_MAX_GRID, STRIDE,  # noqa: E402
                               WINDOW, gt_endpoints_masked, window_starts)

#: §5.2 requires σ at 2 s AND 6 s. 20 also COINCIDES with `max(WP_STEPS)`, which
#: is what makes the endpoint-vs-`gt` bit-identity self-control possible.
ENDPOINT_STEPS_DEFAULT: tuple[int, ...] = (20, 60)

#: The pre-registered surface (`e_wc2_sigma_star.PREREG`), restated as the
#: producer's own target so a short dump is flagged HERE and not three steps later.
N_EPISODES, N_WINDOWS = 40, 881


# --------------------------------------------------------------------------- #
# 0. PREFLIGHT — fail in 2 seconds, not 11 minutes in                          #
# --------------------------------------------------------------------------- #
#: ⛔ EVERY import this script or its DOWNSTREAM STEP needs, probed at startup.
#: MEASURED 2026-08-11: `t1_eval.py` rolled both arms over all 40 episodes
#: (~11 min/arm) and then died in `analyze()` on `from taniteval import selgap`,
#: reporting `T1_EXIT=NO_ARMS_PRODUCED` for a 100 %-complete run. An
#: analysis-time import that fails AFTER the rollout destroys the output while
#: the compute is already paid for. `e_wc2_sigma_star` and `taniteval.ci` are in
#: this list even though THIS script never calls them: they are step 2, and a
#: dump that cannot be analysed on the box that produced it is a wasted pause.
PREFLIGHT_MODULES: tuple[tuple[str, str], ...] = (
    ("torch", "the model"),
    ("numpy", "the estimator's arrays"),
    ("driving_diagnostic", "WP_STEPS + gt_ego_waypoints (the ego frame)"),
    ("refc_dump_latents", "the canonical window grid + the endpoint mask"),
    ("tanitad.eval.v6_probe_trunk", "the v6 checkpoint construction path"),
    ("train_v6_staged", "build_stack_from_args — the trainer's own wiring"),
    ("eval_flagship_v4", "resolve_eval_frames / load_val_episodes"),
    ("train_flagship4b", "FlagshipWindowDataset — the windowing contract"),
    ("e_wc2_sigma_star", "STEP 2 — the estimator this dump feeds"),
)


def preflight(*, verbose: bool = True) -> list[str]:
    """Import every module steps 1 AND 2 need. Returns the failures, empty = OK.

    Also probes `taniteval/taniteval/ci.py` through
    :func:`e_wc2_sigma_star._load_ci` — the episode-cluster bootstrap lives in a
    SIBLING package of `stack/`, so `PYTHONPATH=<stack>` alone resolves it to a
    `ModuleNotFoundError` at the end of the analysis rather than at its start.
    """
    import importlib

    fails: list[str] = []
    for mod, why in PREFLIGHT_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:                      # noqa: BLE001
            fails.append(f"{mod} ({why}): {type(exc).__name__}: {exc}")
    try:
        import e_wc2_sigma_star as _E
        _E._load_ci()
    except Exception as exc:                          # noqa: BLE001
        fails.append(f"taniteval.ci (STEP 2's interval estimator): "
                     f"{type(exc).__name__}: {exc} — PYTHONPATH needs BOTH the "
                     f"stack root and its sibling `taniteval` root")
    if verbose:
        print(f"[swdump] preflight: {len(PREFLIGHT_MODULES) + 1} probes, "
              f"{len(fails)} failed", flush=True)
        for f in fails:
            print(f"[swdump]   ⛔ {f}", flush=True)
    return fails


# --------------------------------------------------------------------------- #
# 1. the checkpoint -> a frozen V6Stack, rebuilt from ITS OWN run args          #
# --------------------------------------------------------------------------- #
def load_v6_stack(ckpt: str, *, device: str = "cpu",
                  args_from: str | None = None) -> tuple:
    """``(stack, run_args, step, provenance)`` — the trained stack, frozen.

    Goes through :func:`tanitad.eval.v6_probe_trunk.load_v6_from_ck`, i.e. the
    run's own `config["args"]` → `build_stack_from_args` → `load_state_dict(
    strict=True)`. ⛔ Never relax the strict load: a non-strict load leaves
    probe tensors at random init and emits numbers that LOOK like results.

    ``args_from`` (a `config.json`, or a directory holding one) is the ONE
    escape hatch, for a weights-only snapshot that travelled without its config
    — the `~/ckpt_snaps/v6F_sw_step*.fp16.pt` case. It supplies the RUN's
    record; it is not a place to type a geometry.
    """
    from tanitad.eval.v6_probe_trunk import is_v6_checkpoint, load_v6_from_ck

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    prov: dict = {"ckpt": str(ckpt), "args_from": args_from,
                  "was_v6_layout": bool(is_v6_checkpoint(ck))}
    if not isinstance(ck, dict):
        raise SystemExit(f"[swdump] ⛔ {ckpt} is not a dict checkpoint")
    if "stack" not in ck:
        # a weights-only snapshot: either {"model": sd} or a bare state_dict.
        sd = ck.get("model") if isinstance(ck.get("model"), dict) else ck
        if not all(torch.is_tensor(v) for v in sd.values()):
            raise SystemExit(
                f"[swdump] ⛔ {ckpt} carries neither a 'stack' state-dict nor a "
                f"bare tensor state-dict — top-level keys {sorted(ck)[:12]}")
        # ⚠️ `step` defaults to -1, not None: `load_v6_from_ck` does
        # `int(ck.get("step", -1))`, and a present-but-None key makes that a
        # TypeError deep inside the loader rather than a missing-provenance note.
        step_in = ck.get("step") if isinstance(ck.get("step"), int) else -1
        ck = {"stack": sd, "step": step_in}
        prov["normalised_to_stack_layout"] = True
        prov["step_from_snapshot"] = step_in
    if args_from:
        p = Path(args_from)
        p = p / "config.json" if p.is_dir() else p
        if not p.is_file():
            raise SystemExit(f"[swdump] ⛔ --args-from {args_from} does not "
                             f"resolve to a config.json")
        cfg = json.loads(p.read_text(encoding="utf-8"))
        if "args" not in cfg:
            raise SystemExit(f"[swdump] ⛔ {p} carries no 'args' block — that is "
                             f"the run record `build_stack_from_args` needs")
        ck["config"] = cfg
        prov["args_from_resolved"] = str(p)
    trunk, step = load_v6_from_ck(ck, device, ckpt_path=ckpt)
    stack = trunk.stack
    from tanitad.eval.v6_probe_trunk import _run_args
    run_args = _run_args(ck, ckpt)
    prov |= {"ckpt_step": int(step), "device": str(device),
             "n_params": int(sum(p.numel() for p in stack.parameters())),
             "n_state_keys": int(len(stack.state_dict())),
             "model_window": int(stack.cfg.predictor.window),
             "plan_steps": int(stack.cfg.plan_steps),
             "n_candidates": int(stack.cfg.n_candidates),
             "d_op": int(stack.cfg.d_op), "d_tac": int(stack.cfg.d_tac),
             "d_str": int(stack.cfg.d_str),
             "has_scorer": stack.cand_score is not None,
             "scorer_class": (None if stack.cand_score is None
                              else type(stack.cand_score).__name__)}
    return stack, run_args, int(step), prov


# --------------------------------------------------------------------------- #
# 2. the canonical val40 grid                                                  #
# --------------------------------------------------------------------------- #
def build_val_grid(run_args: dict, *, in_channels: int, episodes: int = N_EPISODES
                   ) -> tuple:
    """``(ds, grid, provenance)`` — the canonical 881-window val40 surface.

    ⛔ NOTHING IS REIMPLEMENTED. The frames come through
    ``eval_flagship_v4.resolve_eval_frames`` + ``load_val_episodes``, the SAME
    seam ``train_v6_staged.train`` calls, so the frame a dump assumes and the
    frame the trainer applied cannot be spelled two different ways.

    ⛔ The dataset is built at ``window=WINDOW(8)`` and ``max_horizon=
    K_MAX_GRID(20)`` — the CANONICAL grid — not at the model's own window. See
    the module docstring: the model window changes what the encoder SEES, never
    which windows EXIST.
    """
    from eval_flagship_v4 import _eval_cfg, _plan, load_val_episodes
    from train_flagship4b import FlagshipWindowDataset

    a = Namespace(**run_args)
    cfg_eval = _eval_cfg()
    cache_frame, model_frame = resolve_frames(a, cfg_eval)
    plan = _plan(cfg_eval)
    if plan.maneuver_h > K_MAX_GRID:
        raise SystemExit(
            f"[swdump] ⛔ the eval plan's maneuver_h {plan.maneuver_h} exceeds "
            f"the canonical grid's K_MAX {K_MAX_GRID}; FlagshipWindowDataset "
            f"asserts maneuver_h <= max_horizon, and raising max_horizon would "
            f"RE-SELECT WINDOWS and break parity.")
    val_eps, val_prov = load_val_episodes(a, cache_frame=cache_frame,
                                          train_frame=model_frame)
    ds = FlagshipWindowDataset(val_eps, window=WINDOW, max_horizon=K_MAX_GRID,
                               maneuver_h=plan.maneuver_h,
                               channels=in_channels)
    grid = select_grid(ds, episodes=episodes)
    prov = {
        "n_episodes_loaded": len(val_eps),
        "episodes_requested": int(episodes),
        "grid": {"window": WINDOW, "stride": STRIDE, "k_max": K_MAX_GRID,
                 "definition": "refc_dump_latents.window_starts / "
                               "EpisodeWindowDataset.index (they AGREE at "
                               "window=8, max_horizon=20)",
                 "n_windows": len(grid)},
        "model_frame": {"h": int(model_frame.height), "w": int(model_frame.width)},
        "cache_frame": {"h": int(cache_frame.height), "w": int(cache_frame.width)},
        "val_provenance": _jsonable(val_prov),
        "corpus": ("--v2-val-cache " + str(run_args.get("v2_val_cache"))
                   if run_args.get("v2_val_cache")
                   else "--val-cache " + str(run_args.get("val_cache"))),
        "⛔ parity": "episode selection is load_val_episodes' — this script "
                     "NEVER re-selects episodes. The train corpus "
                     "(physicalai-train-e438721ae894, skip-hash f09e44db) is "
                     "not read at all.",
    }
    return ds, grid, prov


def resolve_frames(a: Namespace, cfg_eval):
    """`resolve_eval_frames`, imported — never a second `--v2-subframe` reading."""
    from eval_flagship_v4 import resolve_eval_frames
    return resolve_eval_frames(a, cfg_eval, label="v6_dump_sw_latents")


def select_grid(ds, *, episodes: int = N_EPISODES) -> list[int]:
    """Dataset indices on the canonical grid: episode < ``episodes``, t % STRIDE == 0.

    Identical rule to ``probe_latent_state.collect_grid`` (:554-556). Asserted
    against ``window_starts`` per episode so the two producers' grids cannot
    silently diverge — a grid that drifted by one window would regress every
    latent onto a NEIGHBOUR's endpoint and inflate σ, i.e. a wrong answer that
    looks like a measurement.
    """
    grid = [i for i, (e, t) in enumerate(ds.index)
            if e < episodes and t % STRIDE == 0]
    if not grid:
        raise SystemExit(f"[swdump] ⛔ the grid selected 0 windows over "
                         f"{len(ds.index)} dataset rows — wrong corpus?")
    for e in sorted({ds.index[i][0] for i in grid}):
        want = window_starts(int(ds.episodes[e].frames.shape[0]))
        got = [ds.index[i][1] for i in grid if ds.index[i][0] == e]
        if got != want:
            raise AssertionError(
                f"[swdump] ⛔ episode {e}: the dataset grid {got[:4]}…({len(got)}) "
                f"!= refc_dump_latents.window_starts {want[:4]}…({len(want)}) — "
                f"the two producers would land on different rows")
    return grid


# --------------------------------------------------------------------------- #
# 3. pose-side targets — GROUND TRUTH, no model, no GPU                        #
# --------------------------------------------------------------------------- #
def pose_targets(ds, grid: list[int], endpoint_steps=ENDPOINT_STEPS_DEFAULT
                 ) -> dict:
    """``gt`` / ``gt_endpoint`` / ``endpoint_valid`` / ``cv`` / ``v0`` / ``eid``.

    Everything here derives from the episodes' POSE ARRAYS — no frames, no
    model, no GPU — which is why the 6 s endpoint costs nothing beyond reading
    the poses. ``gt_endpoints_masked`` is imported from ``refc_dump_latents``:
    ONE implementation of the pad-then-mask rule, so a horizon that runs past
    the end of an episode is NaN + ``valid=False`` in both producers and is
    never imputed in either.

    ⚠️ ``future_poses`` from the dataset covers only ``max_horizon = 20`` steps,
    so the 6 s endpoint CANNOT come from the batch — it is read from the
    episode's own pose array at ``last + 60``.
    """
    import driving_diagnostic as dd

    wps = list(dd.WP_STEPS)
    steps = [int(k) for k in endpoint_steps]
    n = len(grid)
    eids = [int(ds.index[i][0]) for i in grid]
    ts = [int(ds.index[i][1]) for i in grid]
    gt = torch.zeros(n, len(wps), 2)
    cv = torch.zeros(n, len(wps), 2)
    gte = torch.zeros(n, len(steps), 2)
    gval = torch.zeros(n, len(steps), dtype=torch.bool)
    v0 = torch.zeros(n)
    ep_raw: dict[int, int] = {}
    for e in sorted(set(eids)):
        rows = [r for r in range(n) if eids[r] == e]
        poses = torch.as_tensor(ds.episodes[e].poses).float()
        last = torch.tensor([ts[r] + WINDOW - 1 for r in rows])
        gt[rows] = dd.gt_ego_waypoints(poses, last, wps)
        cv[rows] = dd.baseline_waypoints(poses, last, wps)["constant_velocity"]
        e_ep, e_val = gt_endpoints_masked(poses, last, steps)
        gte[rows] = e_ep.float()
        gval[rows] = e_val
        v0[rows] = poses[last, 3]
        ep_raw[e] = int(getattr(ds.episodes[e], "episode_id", e))
    return {"eid": eids, "t_start": ts, "gt": gt, "cv": cv,
            "gt_endpoint": gte, "endpoint_valid": gval, "v0_from_poses": v0,
            "wp_steps": wps, "endpoint_steps": steps,
            "episode_id_raw": ep_raw}


# --------------------------------------------------------------------------- #
# 4. THE MODEL PASS — the only part that needs the GPU                         #
# --------------------------------------------------------------------------- #
def _lift3(a2: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """`train_v6_staged._lift3`, imported — the SPEED_SCALE contract, one copy."""
    from train_v6_staged import _lift3 as lift
    return lift(a2, v0)


def _forward_latents(stack, frames: torch.Tensor, acts2: torch.Tensor,
                     v0: torch.Tensor) -> dict:
    """One real `V6Stack.forward`; the blocks E-WC2 consumes, on the CPU."""
    with torch.no_grad():
        out = stack(frames=frames, actions=_lift3(acts2, v0), v0=v0)
    plan = out["plan"]
    got = {"pooled": out["z_op"].float().cpu(),
           "pooled_seq": out["z_op_win"].float().cpu(),
           "ctx": out["z_str"].float().cpu(),
           "z_tac": out["z_tac"].float().cpu(),
           "waypoints": plan["waypoints"].float().cpu()}
    if "sel_score" in plan:
        got["sel"] = plan["sel_score"].float().argmax(-1).cpu()
    return got


#: the blocks the vision-only control must find bit-identical under a permuted
#: `v0`/`actions`. `waypoints` is deliberately NOT here — the emission READS v0
#: by design (it is the unicycle rollout's integration constant), so requiring
#: it to be invariant would be a wrong test, not a stricter one.
VISION_ONLY_BLOCKS: tuple[str, ...] = ("pooled", "pooled_seq", "ctx", "z_tac")


def vision_only_control(stack, frames, acts2, v0, base: dict, *,
                        seed: int = 0) -> dict:
    """⭐ MEASURE the vision-only claim; do not assert it from the diagram.

    Re-runs the SAME frames with `v0` and `actions` PERMUTED ACROSS THE BATCH.
    Every block in :data:`VISION_ONLY_BLOCKS` must come back BIT-IDENTICAL. Same
    construction as `probe_latent_state.py --speed-echo-control`: a claim about
    what an input cannot reach is tested by changing that input.

    A degenerate batch (all rows equal, or b < 2) makes the control vacuous —
    every permutation is the identity — so that is reported as `vacuous: True`
    and does NOT count as a pass.
    """
    b = int(v0.shape[0])
    rec: dict = {"batch": b, "seed": int(seed)}
    if b < 2:
        return rec | {"vacuous": True, "ok": False,
                      "reason": "batch < 2 — every permutation is the identity"}
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(b, generator=g).to(v0.device)
    moved = bool(not torch.equal(v0, v0[perm])
                 or not torch.equal(acts2, acts2[perm]))
    if not moved:
        return rec | {"vacuous": True, "ok": False,
                      "reason": "the permutation changed neither v0 nor the "
                                "actions — this batch carries no evidence"}
    alt = _forward_latents(stack, frames, acts2[perm], v0[perm])
    per = {k: bool(torch.equal(base[k], alt[k])) for k in VISION_ONLY_BLOCKS}
    rec |= {"vacuous": False, "blocks": per, "ok": all(per.values()),
            "permutation_changed_inputs": moved,
            "_what": "v0 and actions permuted across the batch; the VISION-ONLY "
                     "blocks must be bit-identical (they are functions of "
                     "`frames` alone: encode_window -> adapters)"}
    if not rec["ok"]:
        rec["max_abs_diff"] = {k: float((base[k] - alt[k]).abs().max())
                               for k, ok in per.items() if not ok}
    return rec


def collect_latents(stack, ds, grid: list[int], *, device: str = "cpu",
                    batch: int = 4, endpoint_steps=ENDPOINT_STEPS_DEFAULT,
                    run_control: bool = True, log=print) -> dict:
    """⭐ THE PRODUCER. A frozen v6 stack + the canonical grid -> the E-WC2 dump.

    This is the function `tests/test_v6_dump_sw_latents.py` drives with a REAL
    `V6Stack` and hands to the REAL estimator — the whole point being that the
    producer's actual output is what gets validated, never a fixture's idea of it.
    """
    from torch.utils.data import default_collate

    cfg = stack.cfg
    if not cfg.shared_encoder:
        # ⛔ Refuse UP FRONT, not in the first batch. The E-ENC arm (b) needs
        # `own_frames_tac`/`own_frames_str` and `V6Stack.forward` raises without
        # them — after the corpus has mounted. This dumper has no defensible
        # choice of per-layer frame to make on its own, so it says so instead.
        raise SystemExit(
            "[swdump] ⛔ this checkpoint is the E-ENC arm (b) "
            "(shared_encoder=False): `forward` needs own_frames_tac/"
            "own_frames_str and there is no single right frame for this "
            "producer to invent. Extend the dumper deliberately rather than "
            "letting it pick one.")
    w_model, w_grid = int(cfg.predictor.window), int(ds.window)
    if w_model > w_grid:
        raise SystemExit(
            f"[swdump] ⛔ the model's window {w_model} exceeds the canonical "
            f"grid window {w_grid} — frames cannot be fabricated, and widening "
            f"the grid would RE-SELECT WINDOWS (parity is sacred). Refusing.")
    tgt = pose_targets(ds, grid, endpoint_steps)
    wps, steps = tgt["wp_steps"], tgt["endpoint_steps"]
    #: `plan_target = gt_ego_waypoints(..., range(1, plan_steps+1))`
    #: (train_v6_staged.py:2938) is what the fan is scored against, so fan index
    #: k-1 IS step k. Subsampling to WP_STEPS is what makes `max(wp_steps)*DT`
    #: come out at the 2.0 s verdict horizon `e_wc2_sigma_star.run` requires.
    wp_idx = [k - 1 for k in wps]
    if max(wp_idx) >= int(cfg.plan_steps):
        raise SystemExit(f"[swdump] ⛔ the fan is {cfg.plan_steps} steps long "
                         f"and WP_STEPS needs index {max(wp_idx)}")

    n = len(grid)
    acc: dict[str, list] = {k: [] for k in
                            ("pooled", "pooled_seq", "ctx", "z_tac", "fan",
                             "sel", "v0")}
    control: dict | None = None
    t0 = time.time()
    stack.eval()
    for b0 in range(0, n, batch):
        idx = grid[b0:b0 + batch]
        b = default_collate([ds[i] for i in idx])
        frames = b["frames"].to(device).float()[:, -w_model:]
        acts2 = b["actions"][..., :2].to(device).float()[:, -w_model:]
        v0 = b["pose_last"][:, 3].to(device).float()
        got = _forward_latents(stack, frames, acts2, v0)
        if control is None and run_control:
            control = vision_only_control(stack, frames, acts2, v0, got)
        for k in ("pooled", "pooled_seq", "ctx", "z_tac"):
            acc[k].append(got[k])
        acc["fan"].append(got["waypoints"][:, :, wp_idx])
        acc["v0"].append(v0.detach().cpu())
        if "sel" in got:
            acc["sel"].append(got["sel"])
        if (b0 // max(batch, 1)) % 20 == 0:
            log(f"[swdump] {min(b0 + batch, n)}/{n} windows "
                f"({time.time() - t0:.0f}s)")

    d: dict = {
        "eid": tgt["eid"],
        "pooled": torch.cat(acc["pooled"]),
        "pooled_seq": torch.cat(acc["pooled_seq"]),
        "ctx": torch.cat(acc["ctx"]),
        "z_tac": torch.cat(acc["z_tac"]),
        "v0": torch.cat(acc["v0"]),
        "gt_endpoint": tgt["gt_endpoint"], "endpoint_valid": tgt["endpoint_valid"],
        "endpoint_steps": steps,
        "fan": torch.cat(acc["fan"]), "gt": tgt["gt"], "cv": tgt["cv"],
        "wp_steps": wps,
        "t_start": tgt["t_start"],
        "episode_id_raw": tgt["episode_id_raw"],
        "model_window": w_model, "grid_window": w_grid,
        # v6 has NO nav input at all; E-WC2 stamps this into `surface`, so say
        # what it is rather than leaving a REF-C-shaped blank.
        "nav_mode": "n/a — v6 takes no nav input (REF-C's `follow_constant` has "
                    "no analogue here)",
    }
    if acc["sel"]:
        d["sel"] = torch.cat(acc["sel"]).long()
    else:
        d["sel_absent_reason"] = (
            "this arm has NO cand_score, so `emit` produces no sel_* key and "
            "there is no incumbent selection to record (E4_SELECTOR_"
            "RESOLUTION.md §2, MEASURED on the production stack). ⛔ NOT "
            "fabricated: candidate 0 is arbitrary and argmin-over-candidates is "
            "the ORACLE — either would manufacture the σ/ADE denominator and "
            "with it a §5.2 verdict. e_wc2_sigma_star will therefore emit "
            "NO_VERDICT (σ/ADE is not computable) while STILL writing "
            "references_and_ratios.sigma_perax_2s_m, which is the absolute-"
            "metres quantity v6_chain.SW_LATENT_ADMISSION adjudicates.")

    d["controls"] = producer_controls(d, tgt, stack=stack, control=control,
                                      wall_s=round(time.time() - t0, 1))
    # ⭐ E-WC2 carries `controls_vs_bank` through to `surface` verbatim, so the
    # producer's own controls are auditable from the analysis artifact. There is
    # no v6 "bank" to compare against — the controls are SELF-controls, and the
    # key is reused so the audit path is the same one.
    d["controls_vs_bank"] = d["controls"]
    d["instrument_fail"] = d["controls"]["fails"]
    return d


def producer_controls(d: dict, tgt: dict, *, stack=None, control=None,
                      wall_s: float | None = None) -> dict:
    """The self-controls, and the `instrument_fail` list E-WC2 refuses on.

    Four things are checked, each because its failure is SILENT otherwise:

    1. ⛔ **row alignment** — `v0` read off the dataset's `pose_last` must be
       bit-identical to `v0` read off the episode pose array at `last`. They
       come from two independent index paths (the batch loop and
       :func:`pose_targets`); if they ever disagree, every latent sits on a
       different window's targets and σ is a wrong number that looks right.
    2. ⛔ **the ego frame / the fan's rows** — where an endpoint horizon
       COINCIDES with a fan waypoint (step 20), `gt_endpoint[:, i]` must equal
       `gt[:, wp_steps.index(20)]` BIT-IDENTICALLY. The 6 s horizon has nothing
       to check against on its own, so this is what pins its frame too.
    3. ⛔ **vision-only** — see :func:`vision_only_control`.
    4. ⛔ **the pre-registered surface** — 881 windows / 40 episodes, and a 6 s
       endpoint present (§5.2 requires 2 s AND 6 s).
    """
    import numpy as np

    fails: list[str] = []
    eid = [int(x) for x in d["eid"]]
    n, n_ep = len(eid), len(set(eid))
    steps, wps = list(d["endpoint_steps"]), list(d["wp_steps"])
    ctl: dict = {
        "n_windows": n, "n_episodes": n_ep,
        "grid": {"window": int(d["grid_window"]), "stride": STRIDE,
                 "k_max": K_MAX_GRID, "model_window": int(d["model_window"]),
                 "_note": "the grid is CANONICAL (WINDOW=8); the model is fed "
                          "the LAST model_window frames, which end on the same "
                          "frame, so `last`, the ego frame and every target row "
                          "are unchanged"},
        "endpoint_steps": steps,
        "endpoint_valid_frac": {
            str(k): round(float(d["endpoint_valid"][:, i].float().mean()), 4)
            for i, k in enumerate(steps)},
        "wp_steps": wps,
        "fan_wp_index_map": {str(k): k - 1 for k in wps},
        "reference_horizon_s": round(max(wps) * 0.1, 4),
        "has_scorer": bool(stack is not None and stack.cand_score is not None),
        "wall_s": wall_s,
    }
    # 1 — row alignment
    same_v0 = bool(torch.equal(d["v0"].float(), tgt["v0_from_poses"].float()))
    ctl["v0_batch_matches_poses"] = same_v0
    if not same_v0:
        diff = float((d["v0"].float() - tgt["v0_from_poses"].float()).abs().max())
        ctl["v0_max_abs_diff"] = diff
        fails.append(f"v0 from the dataset's pose_last and v0 from the episode "
                     f"pose array at `last` disagree (max {diff:.6g}) — the "
                     f"latent rows and the target rows are NOT the same windows")
    # 2 — the ego frame, pinned where the horizons coincide
    for i, k in enumerate(steps):
        if k in wps:
            same = bool(torch.equal(d["gt_endpoint"][:, i].float(),
                                    d["gt"][:, wps.index(k)].float()))
            ctl[f"endpoint_{k}_matches_gt"] = same
            if not same:
                fails.append(f"endpoint step {k} != gt[:, {wps.index(k)}] — the "
                             f"endpoint block is not on the fan's rows/frame")
    # 3 — vision-only, MEASURED
    if control is not None:
        ctl["vision_only_invariance"] = control
        if not control.get("ok"):
            fails.append(
                "the VISION-ONLY control did not pass: " +
                json.dumps({k: v for k, v in control.items()
                            if k in ("blocks", "vacuous", "reason")}) +
                " — a latent block that moves when v0/actions move is not "
                "vision-only, and its σ would be a leak magnitude")
    # 4 — the pre-registered surface
    if 60 not in steps:
        fails.append("no 6.0 s (step 60) endpoint — V6F_PLANNER_DESIGN.md §5.2 "
                     "requires sigma at 2 s AND 6 s; E-WC2 will refuse a verdict")
    if abs(ctl["reference_horizon_s"] - 2.0) > 1e-9:
        fails.append(
            f"the fan's last waypoint is {ctl['reference_horizon_s']} s, not the "
            f"2.0 s verdict horizon — e_wc2_sigma_star emits `MISMATCH` and NO "
            f"`sigma_perax_2s_m`, so the chain's admission gate would read "
            f"nothing (v6_chain.SW_SIGMA_LOCATIONS)")
    if n != N_WINDOWS or n_ep != N_EPISODES:
        fails.append(f"counts {n}/{n_ep} != {N_WINDOWS}/{N_EPISODES} — the "
                     f"pre-registered surface is not met and E-WC2 will refuse "
                     f"a verdict (the σ is still written and inspectable)")
    # a NaN in a feature block would propagate into the ridge as a silent NaN σ
    for blk in VISION_ONLY_BLOCKS:
        if blk in d and not bool(torch.isfinite(d[blk]).all()):
            bad = int((~torch.isfinite(d[blk])).sum())
            fails.append(f"feature block {blk} carries {bad} non-finite values")
    ctl["nan_free_feature_blocks"] = [b for b in VISION_ONLY_BLOCKS if b in d]
    ctl["fails"] = fails
    return ctl


# --------------------------------------------------------------------------- #
# 5. provenance                                                                #
# --------------------------------------------------------------------------- #
PROVENANCE = (
    "v6 S-W latents on the canonical 881-window val40 grid, ONE inference pass, "
    "no training. `pooled` (z_op), `pooled_seq` (z_op_win), `ctx` (z_str) and "
    "`z_tac` are VISION ONLY — functions of `frames` alone via encode_window -> "
    "adapters — and that is MEASURED by the v0/action permutation control, not "
    "asserted. `v0` is MEASURED_PRESENT (admissible, PI 2026-08-16) and carries "
    "the anti-echo obligation. `gt_endpoint`/`endpoint_valid`/`endpoint_steps` "
    "are GROUND TRUTH endpoint targets for E-WC2 (V6F_PLANNER_DESIGN.md §5.2): "
    "ego-frame displacement at each horizon, NaN + valid=False where the horizon "
    "runs past the end of the episode. The window grid is the REF-C producer's, "
    "imported (K_MAX = max(WP_STEPS) = 20), so the 881-window parity holds and "
    "`eid` corresponds element-for-element. The model is fed the LAST "
    "`predictor.window` frames of each canonical 8-frame window: same last "
    "frame, same `last`, same ego frame, same rows.")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


# --------------------------------------------------------------------------- #
# 6. CLI                                                                       #
# --------------------------------------------------------------------------- #
def build_args(argv=None) -> Namespace:
    ap = argparse.ArgumentParser(
        "v6_dump_sw_latents",
        description="STEP 1 of v6_chain.sw_admission_recipe — dump v6 S-W "
                    "latents in e_wc2_sigma_star's DUMP_CONTRACT.")
    ap.add_argument("--ckpt", help="the v6 S-W checkpoint")
    ap.add_argument("--args-from", help="a config.json (or its directory) for a "
                                        "weights-only snapshot that travelled "
                                        "without its run record")
    ap.add_argument("--v2-val-cache", nargs="+",
                    help="override the run's val corpus (default: the run's own)")
    ap.add_argument("--val-cache", help="raw epcache form of the same override")
    ap.add_argument("--v2-lru", type=int, default=0,
                    help="0 = keep the run's own --v2-lru")
    ap.add_argument("--out", help="write the dump here")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--endpoint-steps", default="20,60",
                    help="E-WC2 endpoint horizons in 10 Hz steps. Does NOT "
                         "change the window grid — see the K_MAX note.")
    ap.add_argument("--no-vision-only-control", action="store_true",
                    help="skip the v0/action permutation control (it is one "
                         "extra forward on ONE batch; there is no good reason)")
    ap.add_argument("--no-strict", action="store_true",
                    help="write even when a control fails (the controls are "
                         "still recorded) — for inspection only")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--print-contract", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    if a.print_contract:
        import e_wc2_sigma_star as E
        print(json.dumps(E.DUMP_CONTRACT, indent=2, ensure_ascii=False))
        return 0

    # ⛔ PREFLIGHT FIRST — before the checkpoint, before the corpus, before CUDA.
    fails = preflight()
    if fails:
        print("[swdump] ⛔ PREFLIGHT FAILED — refusing to start a GPU pass whose "
              "analysis step cannot run.", flush=True)
        return 3
    if a.preflight_only:
        print("[swdump] preflight OK", flush=True)
        return 0
    for req in ("ckpt", "out"):
        if not getattr(a, req):
            raise SystemExit(f"[swdump] --{req} is required")

    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[swdump] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    ep_steps = [int(x) for x in str(a.endpoint_steps).split(",") if x.strip()]

    t0 = time.time()
    stack, run_args, step, prov = load_v6_stack(a.ckpt, device=device,
                                                args_from=a.args_from)
    print(f"[swdump] stack: {prov['n_params']:,} params / "
          f"{prov['n_state_keys']} keys · step {step} · window "
          f"{prov['model_window']} · scorer {prov['scorer_class']}", flush=True)

    if a.v2_val_cache:
        run_args["v2_val_cache"] = list(a.v2_val_cache)
        run_args["val_cache"] = None
    if a.val_cache:
        run_args["val_cache"] = a.val_cache
        run_args["v2_val_cache"] = None
    if a.v2_lru:
        run_args["v2_lru"] = int(a.v2_lru)

    ds, grid, grid_prov = build_val_grid(
        run_args, in_channels=int(stack.cfg.encoder.in_channels),
        episodes=a.episodes)
    print(f"[swdump] grid: {len(grid)} windows / "
          f"{len({ds.index[i][0] for i in grid})} episodes "
          f"(dataset holds {len(ds)} rows)", flush=True)

    d = collect_latents(stack, ds, grid, device=device, batch=a.batch,
                        endpoint_steps=ep_steps,
                        run_control=not a.no_vision_only_control)
    d |= {"ckpt": a.ckpt, "ckpt_step": step, "stage": run_args.get("stage"),
          "build": prov, "grid_provenance": grid_prov,
          "torch_version": torch.__version__,
          "host": os.uname().nodename if hasattr(os, "uname") else "windows",
          "provenance": PROVENANCE}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, a.out)
    print(json.dumps(_jsonable(d["controls"]), indent=2, ensure_ascii=False),
          flush=True)
    print(f"[swdump] pooled {tuple(d['pooled'].shape)} pooled_seq "
          f"{tuple(d['pooled_seq'].shape)} ctx {tuple(d['ctx'].shape)} z_tac "
          f"{tuple(d['z_tac'].shape)} fan {tuple(d['fan'].shape)} gt_endpoint "
          f"{tuple(d['gt_endpoint'].shape)} -> {a.out} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if "sel" not in d:
        print("[swdump] ⚠️  no `sel` (this arm has no scorer) — e_wc2_sigma_star "
              "will emit NO_VERDICT for its OWN §5.2 ratio test and STILL write "
              "references_and_ratios.sigma_perax_2s_m, which is what "
              "`v6_chain.py admission` adjudicates.", flush=True)
    if d["instrument_fail"]:
        print("[swdump] ⛔ INSTRUMENT-FAIL: " + "; ".join(d["instrument_fail"]),
              flush=True)
        return 0 if a.no_strict else 2
    print(f"[swdump] ✅ next: python scripts/e_wc2_sigma_star.py --dump {a.out} "
          f"--features pooled,ctx --out <sw_dir>/ewc2_sw_latents.json",
          flush=True)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
