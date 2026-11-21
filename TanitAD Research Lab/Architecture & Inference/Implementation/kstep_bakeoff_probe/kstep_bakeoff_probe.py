"""K-step rollout bake-off — matched-compute directional probe (Arch backlog P0 #2).

WHAT
----
The ``kstep_rollout`` bake-off lever (``train.rollout_k``) flipped from *planned*
to *runnable* on 2026-07-09 (``train_worldmodel._rollout_loss`` + ``future_actions``
in the window contract). This script executes the FIRST measured arm of that lever:
two arms trained at **matched compute** on REAL comma2k19 camera data, identical in
every config field except ``train.rollout_k`` (verified by the harness'
``lever_diff``), each scored through the integrated D1-D3 gate runner.

    arm A: rollout_k = 1   (single-step baseline)
    arm B: rollout_k = 2   (recursive 2-step rollout loss, backlog falsifier target)

HONEST SCOPE (P8, matches STATE.md doctrine)
--------------------------------------------
This is a **reduced-scale directional probe**, NOT the decision-grade sweep. The
probe config shrinks the operative stack (d256 / enc-depth-6 / pred-depth-4 /
128 px / no tactical / no H15) so two arms fit the Wednesday agent's wall-clock on
the local RTX 4060. It answers two questions with measured numbers:
  1. does the K-step mechanism train end-to-end on real data without instability?
  2. first read: does recursive rollout move D2 direction-acc / D3 ratio at matched
     steps? Falsifier (backlog): D2 >= +0.02 or the item is dropped.
Decision-grade needs matched-compute arms at the *operative* scale from the pod2
step-8k checkpoint (pod2 Phase C) — the D-gates on a small under-trained model may
be BLOCKED, in which case the doctrine (D-004) yields NO architecture claim and the
comparison is read only on the diagnostic metrics.

USAGE
-----
  python kstep_bakeoff_probe.py --data-root C:/Users/Admin/tanitad-data/comma2k19/extracted \
      --episodes 28 --steps 1500 --out <out_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# --- make the stack + its scripts importable ------------------------------- #
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]                       # .../TanitAD
_STACK = _REPO / "stack"
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import (EncoderConfig, PredictorConfig, ReadoutConfig,   # noqa: E402
                            base250cam_config)
from tanitad.eval.bakeoff import default_levers, lever_diff                  # noqa: E402
from tanitad.models.fourbrain import WorldModel                              # noqa: E402
from tanitad.train.train_worldmodel import _build_datasets, train            # noqa: E402
from evaluate_checkpoint import evaluate                                     # noqa: E402


def probe_config(rollout_k: int, steps: int, seed: int, out_dir: str):
    """The shared reduced-but-REAL config; the ONLY field that differs across
    arms is train.rollout_k (asserted against the bake-off lever below)."""
    cfg = base250cam_config()                  # real 9-ch camera contract
    cfg.encoder = EncoderConfig(in_channels=9, image_size=128, patch_size=16,
                                d_model=256, depth=6, n_heads=4)
    cfg.predictor = PredictorConfig(d_model=256, depth=4, n_heads=4, window=8,
                                    horizons=(1, 2, 4), action_dim=2)
    cfg.tactical_pred = None                   # operative-only probe (lever is operative)
    cfg.readout = ReadoutConfig(grid=4, d_readout=64)
    cfg.h15.enabled = False                    # drop imagination-field cost for the probe
    cfg.loss.sigreg.n_slices = 128
    cfg.train.lr = 3e-4
    cfg.train.batch_size = 32                  # *window 8 = 256 SigReg rows (F-2 floor)
    cfg.train.steps = steps
    cfg.train.warmup_steps = max(50, steps // 20)
    cfg.train.seed = seed
    cfg.train.save_every = steps + 1           # no mid checkpoints (probe)
    cfg.train.rollout_k = rollout_k
    cfg.train.out_dir = out_dir
    return cfg


def _assert_one_factor(k_base: int, k_variant: int):
    """The probe's two arms must differ ONLY in train.rollout_k — reuse the
    bake-off harness' own diff so a hidden confound cannot slip in."""
    a = probe_config(k_base, 10, 0, "x")
    b = probe_config(k_variant, 10, 0, "x")
    changed = lever_diff(a, b)
    assert changed == ["train.rollout_k"], f"not one-factor: {changed}"
    # also confirm the shipped lever really is train.rollout_k -> K=4
    lev = {l.name: l for l in default_levers()}["kstep_rollout"]
    assert lev.fields == ("train.rollout_k",)


def run_arm(name: str, rollout_k: int, args, git_hash: str) -> dict:
    from tanitad.data.comma2k19 import CORPUS_META
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = probe_config(rollout_k, args.steps, args.seed, str(out_dir))

    t0 = time.time()
    metrics = train(cfg, n_episodes=args.episodes, data="comma2k19",
                    data_root=args.data_root, amp=True)
    train_s = time.time() - t0

    # rebuild the SAME route-split val episodes (cached -> cheap) for the gates
    _tr, val_ds = _build_datasets(cfg, args.episodes, "comma2k19", args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = WorldModel(cfg)
    world.load_state_dict(torch.load(out_dir / "model.pt", map_location="cpu"))
    gates = evaluate(world, val_ds.episodes, device,
                     exp_id=f"p0-kstep-{name}", git_hash=git_hash,
                     corpus_meta=CORPUS_META)

    return {
        "arm": name, "rollout_k": rollout_k, "seed": args.seed,
        "train_wallclock_s": round(train_s, 1),
        "n_params": metrics["n_params"],
        "final_train_log": metrics["final"],
        "instruments_train": metrics["instruments"],
        "gate_summary": gates["summary"],
        "gates": {g["gate"]: {
            "status": g["gate"] in gates["summary"] and gates["summary"][g["gate"]],
            "admissible": g["admissible"], "passed": g["passed"],
            "metrics": g["metrics"], "verdict": g["verdict"],
        } for g in gates["gates"]},
        "spectral": gates.get("spectral", {}),
        "d3_horizon_s": gates.get("d3_horizon_s"),
        "n_eval_windows": gates.get("n_eval_windows"),
    }


def _cmp(res_a: dict, res_b: dict) -> dict:
    """Directional read on the K-step lever (measured, honest about BLOCKED).

    IMPORTANT (found in the smoke run): the D2 gate can PASS via the P4
    forward-dynamics probe ([prev_state (v,yaw) + action] -> displacement), which
    does NOT touch the operative predictor's imagined latents. The K-step rollout
    lever only reshapes the IMAGINATION path, so it must be read on the
    imagination-path signals, not on D2 gate status:
      - P1 direction_acc  (imagination probe, A3) + its fit R^2 (is it fittable?)
      - D3 imagined/oracle ratio (even when BLOCKED it is the diagnostic)
      - imag_rel_diagnostic (||z_hat-z|| / ||z-z_prev||; <1 beats persistence)
    """
    def g(res, gate, key):
        return res["gates"].get(gate, {}).get("metrics", {}).get(key)

    def block(gate, key, lower_better=None):
        va, vb = g(res_a, gate, key), g(res_b, gate, key)
        d = (None if va is None or vb is None else round(vb - va, 4))
        row = {"k1": va, "k2": vb, "delta_k2_minus_k1": d}
        return row

    out = {
        "P1_imag_direction_acc": block("D2", "direction_acc"),
        "P1_imag_probe_fit_r2": block("D2", "p1_fit_r2"),
        "P4_forward_dyn_dir_acc(lever-independent)": block(
            "D2", "p4_forward_dynamics_dir_acc"),
        "D3_imagined_oracle_ratio": block("D3", "ratio"),
        "imag_rel_diagnostic": block("D2", "imag_rel_diagnostic"),
        "D2_status": {"k1": res_a["gate_summary"].get("D2"),
                      "k2": res_b["gate_summary"].get("D2")},
        "D3_status": {"k1": res_a["gate_summary"].get("D3"),
                      "k2": res_b["gate_summary"].get("D3")},
    }

    # falsifier = imagination-path direction accuracy, but only trustworthy when
    # the P1 probe actually fits (>=0.9) on BOTH arms — else the lever read is on
    # an unfit probe and yields no claim (D-004 doctrine).
    d = out["P1_imag_direction_acc"]["delta_k2_minus_k1"]
    f1 = out["P1_imag_probe_fit_r2"]["k1"]
    f2 = out["P1_imag_probe_fit_r2"]["k2"]
    p1_fits = (f1 is not None and f2 is not None and f1 >= 0.9 and f2 >= 0.9)
    if d is None:
        out["falsifier_verdict"] = "INDETERMINATE (a metric missing)"
    elif not p1_fits:
        out["falsifier_verdict"] = (
            f"NO CLAIM on P1 dir-acc — imagination probe under-fits at this scale "
            f"(fit_r2 k1={f1}, k2={f2} < 0.9); K-step read defers to decision-grade "
            f"trained arms (pod2 Phase C). Diagnostic dir-acc delta={d:+.4f}.")
    elif d >= 0.02:
        out["falsifier_verdict"] = f"K=2 PASSES falsifier (P1 dir-acc {d:+.4f} >= +0.02)"
    else:
        out["falsifier_verdict"] = (
            f"K=2 FAILS falsifier (P1 dir-acc {d:+.4f} < +0.02) at this scale — "
            f"record negative; retest only at operative scale before dropping")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--episodes", type=int, default=28)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-base", type=int, default=1)
    ap.add_argument("--k-variant", type=int, default=2)
    ap.add_argument("--git-hash", default="unknown")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _assert_one_factor(args.k_base, args.k_variant)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    print(f"[kstep] arm A (K={args.k_base}) ...", flush=True)
    res_a = run_arm(f"k{args.k_base}", args.k_base, args, args.git_hash)
    print(f"[kstep] arm B (K={args.k_variant}) ...", flush=True)
    res_b = run_arm(f"k{args.k_variant}", args.k_variant, args, args.git_hash)

    comparison = _cmp(res_a, res_b)
    report = {
        "experiment": "p0-kstep-bakeoff-probe",
        "hardware": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else "cpu"),
        "cost_usd": 0.0,
        "config_note": ("reduced-but-real probe: d256/enc6/pred4/128px/9ch, "
                        "no tactical, no H15; ONLY train.rollout_k differs"),
        "steps_per_arm": args.steps, "episodes": args.episodes,
        "arms": [res_a, res_b],
        "comparison": comparison,
    }
    out = Path(args.out) / "kstep_bakeoff_result.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n=== K-step bake-off probe — comparison ===")
    print(json.dumps(comparison, indent=2, default=str))
    print(f"\n[kstep] full report -> {out}")


if __name__ == "__main__":
    main()
