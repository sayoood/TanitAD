#!/usr/bin/env python3
"""Pseudo-simulation on ONE small-validation arm -> a per-window ``.npz``.

⭐ WHY THIS EXISTS AND WHY IT IS NOT ``panel_run.py``.
``…/2026-07-27-pseudosim-arm-panel/scripts/panel_run.py`` is the shipped panel
driver and would have been reused verbatim, but it cannot evaluate the arms this
validation is about. Three gaps, each verified in its source:

* it loads episodes with ``load_episode`` over ``ep_*.pt`` only — there is **no
  v2-compressed path**, and the wide corpus is a v2 cache of ``<clip>.v2ep.pt``;
* it never passes ``frame=`` to :func:`pseudosim.pseudo_evaluate`, so every warp
  would be the DEPLOYED 256x256 pinhole. ⛔ On v5's cylindrical frame that
  misplaces source pixels by a mean of 46.3 px against a true shift of 42.7 px —
  the arm would be scored on an observation its camera could never produce;
* its ``--kind v4`` arm uses the **ORACLE** goal, while this validation probes
  the **deployed no-goal surface** (``--heldout-goal dropped``), which is what
  the v5 run's own mid-run gate stops on.

⛔ NOTHING NEW IS IMPLEMENTED HERE. The planner is the trainer's own
:class:`tanitad.train.heldout_gate.DeployableSurfacePlanner`; the goal kwargs come
from :func:`tanitad.train.heldout_goal.make_goal_kwargs_fn`; the frame seam is
:func:`train_flagship_v4.resolve_v2_frames`; the scoring is
:mod:`taniteval.pseudosim`. This file is wiring, and the ``.npz`` it writes has
the **same schema** ``panel_run._save_pw`` writes, so
``…/2026-07-27-pseudosim-arm-panel/scripts/panel_combine.py`` consumes it
unmodified and the numbers stay directly comparable to the published 20-arm panel.

⚠️ THE GRID IS THE PANEL'S FULL 21-POINT GRID (``pseudosim.default_grid()``), not
``heldout_gate.probe_grid()``'s 3-point mid-run subset. The mid-run grid is
7x cheaper and is the right choice for a probe that runs every 1 000 steps; it is
the WRONG choice here, because every MDE anchor this validation pre-registered was
measured on the 21-point grid and a different substrate is a different metric.

usage (pod2, after the training chain is done -- never on a training pod):
  PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \\
  python3 smallval_pseudosim.py --arm A_old --ckpt <ckpt.pt> \\
      --anchors-dense <anchors.pt> --corpus raw \\
      --val-dir /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \\
      --episodes 120 --out-dir /workspace/smallval/ps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


#: The tree this driver MUST resolve every TanitAD module from.
STACK = os.environ.get("SMALLVAL_STACK", "/workspace/TanitAD/stack")


def pin_stack(stack: str = STACK) -> dict:
    """⛔ THE SHADOWING GUARD — hard-fail, because the failure is SILENT otherwise.

    MEASURED on pod2 2026-07-27: **~15 `taniteval` submodules hardcode**
    ``sys.path.insert(0, "/root/TanitAD/stack")`` (``bench``, ``closedloop``,
    ``corridor``, ``data``, ``driving``, ``blind_baseline``, …). On pod2 that
    path is a **12 MB stale tree with no `.git`, no
    `tanitad/train/heldout_gate.py` and no `resolve_v2_frames`** — so merely
    importing ``taniteval.pseudosim`` re-points ``tanitad`` at pre-v5 code, and
    the first symptom is ``ModuleNotFoundError: tanitad.train.heldout_gate``
    *after* the checkpoint has loaded. A run that imported a module which HAPPENS
    to exist in both trees would not have errored at all — it would have
    evaluated on the wrong code and published a number.

    ``taniteval/__init__.py`` already ships the cure: ``TANITEVAL_STACK_OVERRIDE``
    is inserted at ``sys.path[0]`` and ``tanitad`` is imported **there, first**, so
    the ``sys.modules`` cache wins over every later insert. This function sets it
    (before ``taniteval`` is imported anywhere) and then **verifies the outcome**
    rather than trusting it: an env var that is set but ineffective is exactly the
    'staging that reports success' shape this program keeps getting burned by.
    """
    os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", stack)
    for p in (f"{stack}/scripts", stack, str(Path(stack).parent / "taniteval")):
        if os.path.isdir(p):
            sys.path.insert(0, p)
    import tanitad                                            # noqa: PLC0415
    import taniteval                                          # noqa: PLC0415,F401
    bad = []
    if not str(Path(tanitad.__file__).resolve()).startswith(str(Path(stack).resolve())):
        bad.append(f"tanitad -> {tanitad.__file__}")
    import importlib
    for mod in ("tanitad.train.heldout_gate", "tanitad.train.heldout_goal",
                "tanitad.data.v2_dataset", "tanitad.data.parity",
                "train_flagship_v4", "eval_flagship_v4"):
        try:
            m = importlib.import_module(mod)
        except Exception as ex:                               # noqa: BLE001
            bad.append(f"{mod} -> IMPORT FAILED {ex!r}")
            continue
        f = str(Path(getattr(m, "__file__", "")).resolve())
        if not f.startswith(str(Path(stack).resolve())):
            bad.append(f"{mod} -> {f}")
    if bad:
        raise SystemExit(
            "[ps] ⛔ STACK SHADOWING — refusing to evaluate on code that is not "
            f"{stack}. Offenders:\n  " + "\n  ".join(bad) +
            "\n⇒ export TANITEVAL_STACK_OVERRIDE=" + stack + " BEFORE python3 "
            "starts, and make sure no stale tree (e.g. /root/TanitAD/stack) "
            "precedes it. See pin_stack.__doc__.")
    return {"stack": stack, "tanitad": tanitad.__file__,
            "TANITEVAL_STACK_OVERRIDE": os.environ["TANITEVAL_STACK_OVERRIDE"],
            "verified_modules": 6}


def _md5(p) -> str | None:
    if not p or not Path(p).is_file():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class BlindWrapper:
    """⭐ THE ANTI-C13 CONTROL, verbatim from ``panel_run.BlindWrapper``.

    The same planner on a DESTROYED observation. ``sighted - blind`` is the
    dynamic-range demonstration this validation registered BEFORE any arm
    finished: a composite that cannot separate an arm from itself-with-no-image
    cannot possibly resolve a change in what the image SHOWS, and every null on
    such an instrument is ``INSTRUMENT-BLIND``, not ``NO DIFFERENCE``."""

    def __init__(self, inner):
        self.inner = inner
        self.provenance = dict(getattr(inner, "provenance", {}) or {})
        self.provenance["blind"] = "observation zeroed (torch.zeros_like)"

    def traj(self, fw, v0, goal=None):
        return self.inner.traj(torch.zeros_like(fw), v0, goal)


def _save_pw(pw, path):
    """The no-GPU recompute path. SCHEMA-IDENTICAL to ``panel_run._save_pw``."""
    np.savez_compressed(
        path,
        traj=pw["traj"].numpy().astype(np.float32),
        ref_path=pw["ref_path"].numpy().astype(np.float32),
        ref_yaw=pw["ref_yaw"].numpy().astype(np.float32),
        v0=pw["v0"].numpy().astype(np.float32),
        pt_dlat=pw["pt_dlat"].numpy().astype(np.float32),
        pt_dyaw=pw["pt_dyaw"].numpy().astype(np.float32),
        pt_dlon=pw["pt_dlon"].numpy().astype(np.float32),
        anchor=pw["anchor"].numpy().astype(np.int64),
        ep_i=pw["ep_i"].numpy().astype(np.int64),
        eid=np.asarray(pw["eid"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchors-dense", required=True)
    ap.add_argument("--head-config", default=None)
    ap.add_argument("--corpus", choices=("raw", "v2"), required=True)
    ap.add_argument("--val-dir", default=None, help="raw epcache split dir")
    ap.add_argument("--v2-val-cache", nargs="+", default=None)
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--frame-h", type=int, default=None)
    ap.add_argument("--frame-w", type=int, default=None)
    ap.add_argument("--frame-hfov", type=float, default=None)
    ap.add_argument("--projection", default=None)
    ap.add_argument("--v2-subframe", default=None)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--goal-option", default="dropped")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    pin = pin_stack()
    print(f"[ps:{a.arm}] stack pinned: {pin}", flush=True)
    from taniteval import pseudosim as PS

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- the grid + its envelope proof, BEFORE any checkpoint is loaded ------ #
    grid = PS.default_grid()
    proof = PS.assert_grid_in_envelope(grid)
    print(f"[ps:{a.arm}] grid={grid.describe()} envelope "
          f"verdict={proof['EXTRAPOLATION_VERDICT']!r}", flush=True)

    # --- episodes + the FRAME the pixels were built at ----------------------- #
    from tanitad.config import flagship4b_config
    cfg = flagship4b_config()
    frame = None
    if a.corpus == "raw":
        from tanitad.data.mixing import load_episode
        files = sorted(Path(a.val_dir).glob("ep_*.pt"))[:a.episodes]
        if not files:
            raise SystemExit(f"[ps] no ep_*.pt under {a.val_dir}")
        episodes = [load_episode(str(p), mmap=True) for p in files]
        corpus_id = str(a.val_dir)
    else:
        # ⭐ the frame seam, from the TRAINER's own resolver so eval and train
        # cannot disagree about what "176x624" is.
        from train_flagship_v4 import resolve_v2_frames
        from tanitad.data.v2_dataset import build_v2_providers
        cache_frame, train_frame = resolve_v2_frames(a, cfg, label="smallval_pseudosim")
        slice_frame = None if train_frame == cache_frame else train_frame
        eps = build_v2_providers(a.v2_val_cache, lru_size=a.v2_lru,
                                 frame=slice_frame, verbose=True)
        if not eps:
            raise SystemExit(f"[ps] no *.v2ep.pt under {a.v2_val_cache}")
        episodes = eps[:a.episodes]
        frame = train_frame                       # ⛔ never None on a v2 arm
        corpus_id = " ".join(a.v2_val_cache)
    Ts = [int(e.poses.shape[0]) for e in episodes]
    print(f"[ps:{a.arm}] {len(episodes)} episodes, T in [{min(Ts)},{max(Ts)}], "
          f"frame={frame}, dev={device}", flush=True)

    # --- the checkpoint + the DEPLOYABLE surface ----------------------------- #
    from eval_flagship_v4 import load_v4_from_ck
    from tanitad.train.heldout_gate import DeployableSurfacePlanner
    from tanitad.train.heldout_goal import make_goal_kwargs_fn

    t0 = time.time()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(a.head_config or Path(a.ckpt).parent / "config.json"),
        anchors_dense_path=a.anchors_dense)
    del ck
    gfn = make_goal_kwargs_fn(a.goal_option, head.cfg, goal_head=goal_head)
    planner = DeployableSurfacePlanner(world, head, device=device,
                                       goal_kwargs_fn=gfn,
                                       goal_option=a.goal_option)
    if a.blind:
        planner = BlindWrapper(planner)
    print(f"[ps:{a.arm}] planner ready in {time.time() - t0:.0f}s "
          f"(ckpt step {step})", flush=True)

    # --- run ------------------------------------------------------------------ #
    t1 = time.time()
    pw = PS.pseudo_evaluate(planner, episodes, grid, device=device,
                            stride=a.stride, horizon=a.horizon,
                            batch=a.batch, verbose=True, frame=frame)
    elapsed = time.time() - t1
    if pw.get("_empty"):
        raise SystemExit(f"[ps:{a.arm}] pseudo_evaluate produced NO windows")

    _save_pw(pw, out / f"pw_{a.arm}.npz")
    meta = {
        "arm": a.arm, "blind": bool(a.blind), "goal_option": a.goal_option,
        "corpus": a.corpus, "corpus_id": corpus_id,
        "ckpt": a.ckpt, "ckpt_md5": _md5(a.ckpt), "ckpt_step": int(step),
        "anchors_dense": a.anchors_dense, "anchors_md5": _md5(a.anchors_dense),
        "n_episodes": len(episodes), "T_min": min(Ts), "T_max": max(Ts),
        "stride": a.stride, "horizon": a.horizon,
        "grid": grid.describe(), "envelope_proof": proof,
        "frame": (frame.to_dict() if hasattr(frame, "to_dict") else frame),
        "n_rows": int(pw["traj"].shape[0]),
        "planner_calls": pw.get("planner_calls"),
        "rollout_steps_executed": pw.get("rollout_steps_executed", 0),
        "warp": pw.get("warp"), "traffic_mode": PS.TRAFFIC_MODE_LOG_REPLAY,
        "protocol": PS.PROTOCOL,
        "surface": getattr(planner, "provenance", None),
        "elapsed_s": round(elapsed, 1),
        "torch": torch.__version__, "python": sys.version.split()[0],
    }
    (out / f"meta_{a.arm}.json").write_text(json.dumps(meta, indent=2,
                                                      default=str))
    print(f"[ps:{a.arm}] DONE rows={meta['n_rows']} eps={len(episodes)} "
          f"in {elapsed:.0f}s -> {out}/pw_{a.arm}.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
