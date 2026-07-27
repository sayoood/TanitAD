#!/usr/bin/env python3
"""Block A — ablate a TRAINED speed input inside ONE checkpoint.

REF-C is the only planner family in the panel that ALREADY consumes the ego
speed: ``RefCModel.forward(frames, nav_cmd, v0, …)`` scales it ``v0 / 10.0``
(``stack/tanitad/refs/refc.py:760,786-787``) — the same ``SPEED_SCALE = 10.0``
contract as the flagship's operative action channel. And REF-C is separated
BELOW ``cv_holdv0`` at all three scales. So the question "does a trained speed
input reach a planner's output at all?" can be answered CAUSALLY, within one
checkpoint, with no training: run the same weights with the real ``v0`` and with
``v0 = 0``.

⭐ ``v0 = 0`` IS IN-DISTRIBUTION, NOT OOD. REF-C trains with
``ego_dropout = 0.5`` (``refc.py:287``) — ``v0`` is Bernoulli-zeroed on HALF of
all training steps as a shortcut guard. ⚠️ The same fact bounds the conclusion:
REF-C was trained NOT to lean on ``v0``, so a small effect here is partly by
design and must not be read as "speed inputs do nothing in general".

⚠️ NOTHING IS REIMPLEMENTED. This imports ``panel_run`` (the published panel
driver) for the loader, the adapters, the envelope proof and the G4 falsifier,
and ``taniteval.pseudosim`` for the grid, the evaluation and the dump format.
The ONLY new code is a three-line subclass that zeroes ``v0``.

Host: ``tanitad-eval`` (idle A40). ⛔ pod1 is TRAINING and pod2 is building —
neither is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get("EGOIN_LIB", "/workspace/_egoin/lib"))
sys.path.insert(0, os.path.join(
    os.environ.get("EGOIN_LIB", "/workspace/_egoin/lib"), "stackscripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))

# ⚠️ IMPORT ORDER IS LOAD-BEARING. ``panel_run`` inserts ``/root/taniteval`` and
# ``/root/TanitAD/stack`` at sys.path[0] at module level. On the eval pod those
# are STALE (``/root/taniteval/taniteval`` has no ``clhorizon``), so importing
# panel_run first silently shadows the shipped packages and the run dies with
# ``ImportError: cannot import name 'clhorizon'``. MEASURED 2026-07-28: all four
# Block-A arms failed in 1 s each this way. Binding the packages FIRST fixes it —
# ``sys.modules['taniteval'].__path__`` then wins for every later submodule import.
import tanitad                              # noqa: E402,F401  bind FIRST
import taniteval                            # noqa: E402,F401  bind FIRST
from taniteval import pseudosim as PS       # noqa: E402
from taniteval.ood import ENV_YAW_MAX       # noqa: E402
from taniteval import clhorizon as _CH      # noqa: E402,F401  panel_run needs it
from taniteval import closedloop as _CL     # noqa: E402,F401
from taniteval import loaders as _LD        # noqa: E402,F401

import panel_run as PR                      # noqa: E402  the published driver

_want = os.environ.get("EGOIN_LIB", "/workspace/_egoin/lib")
assert taniteval.__file__.startswith(_want), (
    f"taniteval resolved to {taniteval.__file__}, not the shipped {_want} — "
    f"a stale pod package would silently change the instrument")
assert tanitad.__file__.startswith(_want), (
    f"tanitad resolved to {tanitad.__file__}, not the shipped {_want}")


class V0BlindRefC(PR.RefCPlanner):
    """THE ABLATION, and it is three lines. Identical weights, identical rows,
    identical grid — the planner is simply handed ``v0 = 0``.

    ``v0`` is zeroed BEFORE ``resolve_nav`` as well as before the model call, so
    the arm is blind to the ego speed on EVERY path the planner consumes it,
    not just the decoder FiLM. The harness's own ``v0`` bookkeeping (which the
    metric's diagnostics read) is untouched — ``pseudo_evaluate`` records the
    real ``v0`` regardless of what the planner was shown."""
    kind = "refc_v0blind"

    def traj(self, fw, v0, goal):
        return super().traj(fw, torch.zeros_like(v0), goal)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--refc-nav", default="produced")
    ap.add_argument("--v0", default="real", choices=["real", "zero"])
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = PS.default_grid()

    # ---- G3: the envelope assertion, BEFORE any checkpoint is loaded ------- #
    proof = PS.assert_grid_in_envelope(grid)
    print(f"[egoin:{a.arm}] envelope frac_windows="
          f"{proof['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']} "
          f"verdict={proof['EXTRAPOLATION_VERDICT']!r}", flush=True)

    # ---- G4: the DELIBERATELY FAILING input, exercised on THIS host -------- #
    g4 = {}
    try:
        PS.assert_grid_in_envelope(PS.GridSpec(dyaw_deg=(ENV_YAW_MAX,)))
        g4["edge_value_accepted"] = True
    except PS.EnvelopeViolation:
        g4["edge_value_accepted"] = False
    try:
        PS.assert_grid_in_envelope(PS.GridSpec(dyaw_deg=(ENV_YAW_MAX * 1.0001,)))
        g4["just_outside_raises"] = False
    except PS.EnvelopeViolation:
        g4["just_outside_raises"] = True
    try:
        PS.GridSpec(dlat_m=(1.0,))
        g4["lateral_refused"] = False
    except PS.LateralAxisRefused:
        g4["lateral_refused"] = True
    g4["G4_PASS"] = bool(g4["edge_value_accepted"] and g4["just_outside_raises"]
                         and g4["lateral_refused"])
    print(f"[egoin:{a.arm}] G4 falsifier: {g4}", flush=True)
    assert g4["G4_PASS"], "G4 falsifier did not fire on this host"

    from tanitad.data.mixing import load_episode
    eps_files = sorted(Path(a.val_dir).glob("ep_*.pt"))[:a.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps_files]
    print(f"[egoin:{a.arm}] {len(episodes)} val episodes, dev {device}", flush=True)

    from taniteval.loaders import load as load_arm
    t0 = time.time()
    L = load_arm({"arch": "refc", "ckpt": a.ckpt, "config_preset": a.refc_preset},
                 device)
    cls = PR.RefCPlanner if a.v0 == "real" else V0BlindRefC
    planner = cls(L["model"], a.refc_nav, a.horizon)
    print(f"[egoin:{a.arm}] planner {cls.__name__} ready in "
          f"{time.time() - t0:.0f}s", flush=True)

    meta = {
        "arm": a.arm, "kind": "refc", "v0_mode": a.v0,
        "V0_ABLATION": ("the planner is handed v0 = 0 on EVERY path it consumes "
                        "it (resolve_nav AND the decoder). IN-DISTRIBUTION: "
                        "REF-C trains with ego_dropout = 0.5, so v0 = 0 is what "
                        "half of its training steps saw."
                        if a.v0 == "zero" else
                        "the real ego speed, i.e. the SHIPPED arm -- run again "
                        "here as the fidelity control"),
        "ckpt": a.ckpt, "ckpt_md5": PR._md5(a.ckpt), "ckpt_step": L.get("step"),
        "refc_preset": a.refc_preset, "refc_nav_mode": a.refc_nav,
        "refc_denoise_steps": planner.steps,
        "refc_n_anchors": int(L["model"].cfg.anchors.n_anchors),
        "refc_ego_dropout_at_train": float(L["model"].cfg.ego_dropout),
        "envelope_proof": proof, "G4_falsifier_exercised": g4,
        "grid": grid.describe(), "stride": a.stride, "horizon": a.horizon,
        "n_episodes": len(episodes), "val_dir": a.val_dir,
        "traffic_mode": PS.TRAFFIC_MODE_LOG_REPLAY,
        "traffic_mode_note": PS.TRAFFIC_MODE_NOTE, "protocol": PS.PROTOCOL,
        "host": "tanitad-eval", "python": sys.version.split()[0],
        "torch": torch.__version__,
        "goal_provenance": "none (REF-C produces its own route from the image)",
    }

    # the densify identity, asserted numerically like the published driver
    from taniteval.closedloop import densify_plan
    probe = {5: torch.tensor([[1.0, 0.1]]), 10: torch.tensor([[2.0, 0.3]]),
             15: torch.tensor([[3.0, 0.6]]), 20: torch.tensor([[4.0, 1.0]])}
    err = float((densify_plan(probe, a.horizon)[:, -1] - probe[20]).abs().max())
    meta["densify_endpoint_max_err"] = err
    assert err == 0.0, f"densify moved the 2 s endpoint by {err}"

    t1 = time.time()
    pw = PS.pseudo_evaluate(planner, episodes, grid, device=device,
                            stride=a.stride, horizon=a.horizon, batch=a.batch,
                            verbose=True)
    meta.update(planner_calls=int(pw["planner_calls"]),
                rollout_steps_executed=int(pw["rollout_steps_executed"]),
                wallclock_s=round(time.time() - t1, 1),
                n_rows=int(pw["traj"].shape[0]),
                refc_fed_command_hist={n: int(c) for n, c in zip(
                    ("follow", "left", "right", "straight"),
                    planner.nav_hist.tolist()) if c},
                refc_nav_note=planner.nav_note)

    PR._save_pw(pw, str(out_dir / f"pw_{a.arm}.npz"))
    node = PS.emit(pw, arm=a.arm, n_boot=2000)
    node.pop("_per_window", None)
    node.pop("_per_window_composite", None)
    node["_meta"] = meta
    (out_dir / f"arm_{a.arm}.json").write_text(
        json.dumps(node, indent=2, default=str), encoding="utf-8")
    ci = (node.get("composite", {}) or {}).get("ci") or {}
    print(f"[egoin:{a.arm}] PSS={ci.get('mean')} [{ci.get('lo')},{ci.get('hi')}] "
          f"n_win={ci.get('n_windows')} n_ep={ci.get('n_episodes')} "
          f"{meta['wallclock_s']}s", flush=True)
    print("EGOIN_ARM_DONE " + a.arm, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
