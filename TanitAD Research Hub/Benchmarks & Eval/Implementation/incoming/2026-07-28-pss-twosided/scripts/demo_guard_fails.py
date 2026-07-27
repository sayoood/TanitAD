#!/usr/bin/env python3
"""⛔ DEMONSTRATE THE E1 GUARD FAILING — and that the bug it catches HAS TEETH.

A guard that cannot fail is worse than none (class C13). This script produces the
failing value as an ARTIFACT, on a real ``TacticalPolicy``, in three parts:

1. **The bug has teeth.** A 4-brain built with ``v2_ego_to_planners = true`` and
   NON-ZERO ``ego_emb`` weights (i.e. a *trained* one) emits a MEASURABLY
   DIFFERENT plan when the ego port is fed vs dropped. If this number were 0 the
   whole escalation would be cosmetic — it is not, and the number is published.
2. **The guard REFUSES that state** (``EgoInputDropped``), naming the call site.
3. **Warn mode stamps the defect into the node** rather than only logging it.

Plus the two controls that stop the demo being vacuous:
4. a policy WITHOUT the lever is untouched (the guard is a provable no-op for
   every arm in the published panel), and
5. an EXPLICIT zero ego is accepted, because an ego-ablated arm is a real
   experiment and must be sayable in code.

CPU only, no checkpoint, no corpus, seconds to run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[6]
for p in (str(_REPO / "stack"), str(_REPO / "stack" / "scripts"),
          str(_REPO / "taniteval")):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

from tanitad.config import flagship4b_config          # noqa: E402
from tanitad.ego_plan import EgoInputDropped          # noqa: E402
from tanitad.models.fourbrain import WorldModel       # noqa: E402
from taniteval import ego_guard as EG                 # noqa: E402


def build(ego_lever: bool, seed: int = 0):
    torch.manual_seed(seed)
    c = flagship4b_config()
    c.encoder.d_model = 64
    c.encoder.depth = 1
    c.encoder.n_heads = 4
    c.encoder.patch = 32
    c.encoder.img_size = 64
    c.readout.d_readout = 64
    c.predictor.d_model = 64
    c.predictor.depth = 1
    c.predictor.n_heads = 4
    for pol in (c.tactical_policy, c.strategic_policy):
        pol.d_model = 64
        pol.depth = 1
        pol.n_heads = 4
    c.v2_ego_to_planners = bool(ego_lever)
    return WorldModel(c).eval()


def main():
    out = {
        "_what": ("the E1 guard's FAILING VALUE, demonstrated on a real "
                  "TacticalPolicy, plus the magnitude of the bug it catches"),
        "_evidence_class": "MEASURED (ours; this JSON is the artifact)",
        "_context": ("flagship-v2corpus-30k is training on pod1 with "
                     "v2_ego_to_planners = true; before 2026-07-28 every eval "
                     "path in the repo called the policy with two positional "
                     "arguments and dropped ego= silently."),
    }
    m = build(True)
    b, w = 4, m.tactical_policy.window
    torch.manual_seed(1)
    states = torch.randn(b, w, m.state_dim)
    nav = torch.zeros(b, dtype=torch.long)

    # --- a TRAINED ego_emb ------------------------------------------------- #
    with torch.no_grad():
        for pol in (m.tactical_policy, m.strategic_policy):
            pol.ego_emb.weight.normal_(0.0, 0.5)
            pol.ego_emb.bias.normal_(0.0, 0.1)

    poses = torch.zeros(64, 4)
    poses[:, 2] = torch.linspace(0.0, 0.6, 64)          # yaw
    poses[:, 3] = torch.linspace(2.0, 18.0, 64)         # speed m/s
    ego = EG.ego_from_poses(poses, torch.tensor([10, 20, 30, 40]),
                            pose_scale=10.0)

    def _shift():
        with torch.no_grad():
            ctx_f = m.strategic_policy(states, nav, ego=ego)["ctx"]
            ctx_n = m.strategic_policy(states, nav, ego=None)["ctx"]
            wp_f = m.tactical_policy(states, ctx_f, ego=ego)["waypoints"]
            wp_n = m.tactical_policy(states, ctx_n, ego=None)["waypoints"]
        hs = sorted(wp_f)
        dxy = torch.stack([(wp_f[h] - wp_n[h]).norm(dim=-1) for h in hs], 1)
        return {
            "waypoint_horizons": [int(h) for h in hs],
            "mean_waypoint_shift_m": round(float(dxy.mean()), 6),
            "max_waypoint_shift_m": round(float(dxy.max()), 6),
            "terminal_waypoint_shift_m": round(float(dxy[:, -1].mean()), 6),
            "ctx_l2_shift": round(float((ctx_f - ctx_n).norm(dim=-1).mean()), 6),
        }

    # --- (1) DOES DROPPING THE EGO CHANGE THE PLAN? ------------------------- #
    # (1a) on a FRESHLY BUILT brain, with a trained ego_emb but the shipped
    # ZERO-INIT FiLM. ⚠️ This is 0.0 by construction and it is NOT the answer:
    # `FiLM.to_scale_shift` is zero-initialised (`predictor.py:25-26`), so the
    # WHOLE cond path — ctx and any ego graft — is numerically dead at init.
    # Measuring the bug on a fresh build would have "proved" it harmless.
    at_init = _shift()
    # (1b) on a TRAINED-LIKE brain: the FiLM is non-zero after any training, so
    # the seam is live. This is the state `flagship-v2corpus-30k` is in.
    with torch.no_grad():
        for mod in m.modules():
            if type(mod).__name__ == "FiLM":
                mod.to_scale_shift.weight.normal_(0.0, 0.05)
                mod.to_scale_shift.bias.normal_(0.0, 0.05)
    trained = _shift()
    out["1_the_bug_has_teeth"] = {
        "ego_vector_contract": "[v0 / pose_scale, yr0]  (flagship_losses.py:202-210)",
        "ego_fed_example": [round(float(x), 4) for x in ego[0].tolist()],
        "a_fresh_build_FiLM_zero_init": dict(
            at_init,
            _reading=("0.0 BY CONSTRUCTION — FiLM.to_scale_shift is zero-init "
                      "(predictor.py:25-26), so a fresh brain's whole cond path "
                      "is dead. ⚠️ A demo run only on a fresh build would have "
                      "'proved' this bug harmless. Found while writing the "
                      "demo; it is escalation E6 of the source report, hit in "
                      "practice.")),
        "b_trained_like_FiLM_nonzero": dict(
            trained,
            _reading=("NON-ZERO => scoring an ego-trained checkpoint with "
                      "ego=None measures a DIFFERENT MODEL, silently. This is "
                      "the state pod1's arm will be in.")),
    }

    # --- (2) THE GUARD REFUSES ---------------------------------------------- #
    try:
        EG.assert_planner_ego(m, None, where="demo.eval_path")
        out["2_guard_refuses"] = {"RAISED": False,
                                  "VERDICT": "⛔ THE GUARD DID NOT FIRE"}
    except EgoInputDropped as exc:
        out["2_guard_refuses"] = {
            "RAISED": True, "exception": "tanitad.ego_plan.EgoInputDropped",
            "message": str(exc),
            "_reading": "this is the state every eval path was in until today"}

    # --- (3) WARN MODE STAMPS THE DEFECT ------------------------------------ #
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prov = EG.assert_planner_ego(m, None, where="demo.eval_path",
                                     mode=EG.MODE_WARN)
    out["3_warn_mode_records_rather_than_hides"] = {
        "warnings_emitted": [str(x.message)[:160] for x in caught],
        "node_stamp_ego_input_DROPPED": prov["ego_input_DROPPED"],
        "_reading": ("a number produced in warn mode is identifiable as "
                     "ego-blind FROM THE JSON ALONE — a stderr line in a "
                     "multi-hour eval log is not a record"),
    }

    # --- (4) CONTROL: a policy without the lever is untouched --------------- #
    plain = build(False)
    out["4_control_no_lever_is_a_provable_no_op"] = {
        "capability": EG.planner_ego_capability(plain)["ego_input_on_planners"],
        "guard_result": EG.assert_planner_ego(
            plain, None, where="demo.published_arm")["ego_input_DROPPED"],
        "_reading": ("every arm in the 2026-07-27 panel and every checkpoint in "
                     "MODEL_REGISTRY has ego_emb is None, so adding the call "
                     "changes NO published number"),
    }

    # --- (5) CONTROL: the explicit ablation is accepted --------------------- #
    out["5_control_explicit_zero_ego_is_a_real_ablation"] = {
        "guard_result": EG.assert_planner_ego(
            m, torch.zeros(b, 2), where="demo.ablation",
            ego_source="deliberate v0=0 ablation")["ego_input_DROPPED"],
        "_reading": ("ego=0 is IN-distribution when the run used "
                     "v2_ego_dropout and is a DIFFERENT object from ego=None "
                     "(which skips the ego_emb bias too)"),
    }

    dest = Path(__file__).resolve().parents[1] / "artifacts" / "guard_demo.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:2600])
    print(f"\n[demo] wrote {dest}")
    print("GUARD_DEMO_DONE", flush=True)


if __name__ == "__main__":
    main()
