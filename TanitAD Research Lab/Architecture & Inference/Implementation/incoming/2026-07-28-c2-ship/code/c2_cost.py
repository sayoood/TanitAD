"""What does C2 COST per window? Measured, on the real architecture.

No checkpoint is needed: wall-clock depends on shapes and ops, not on weight
values. The predictor/readout are built from the SAME config the eval path uses
(`eval_flagship_v4._eval_cfg` -> flagship4b_config with action_dim=3), so the
shapes are the deployed ones (state_dim 2048, window 8, d_model 768, depth 10).

Reported against the only honest reference: the per-CANDIDATE roll (rule A1),
which is the alternative imagination-scoring design.
"""
import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

from tanitad.models.wm_reference_select import wm_reference_rollout  # noqa: E402
from tanitad.models.flagship_v15 import imagine_candidates           # noqa: E402

K, WIN, N_CAND = 20, 8, 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build():
    import dataclasses as dc
    from tanitad.config import flagship4b_config
    from tanitad.models.fourbrain import WorldModel
    from tanitad.train.flagship_losses import build_grounding
    cfg = flagship4b_config()
    cfg.speed_input = True
    cfg.predictor = dc.replace(cfg.predictor, action_dim=3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    w = WorldModel(cfg).to(DEV).eval()
    for p in w.parameters():
        p.requires_grad_(False)
    g = build_grounding(w.state_dim, device=DEV).eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return w, g, cfg


def timeit(fn, warmup=2, iters=10):
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main(out_path):
    world, grounding, cfg = build()
    sr = grounding.step["op"]
    S = world.state_dim
    R = {
        "_experiment": "C2 cost per window, MEASURED on the real architecture",
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
        "_host": platform.node(), "_device": DEV,
        "_gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None,
        "_torch": torch.__version__,
        "_shapes": {"state_dim": S, "window": WIN, "k": K,
                    "d_model": cfg.predictor.d_model,
                    "depth": cfg.predictor.depth,
                    "action_dim": cfg.predictor.action_dim,
                    "n_candidates": N_CAND},
        "_note": "random weights: wall-clock depends on shapes/ops, not values",
    }

    # ---- analytic: how many predictor steps does each rule need? -------------
    R["predictor_steps_per_window"] = {
        "C2_wm_reference (this rule)": K,
        "A1_per_candidate_imagination": N_CAND * K,
        "ratio_A1_over_C2": N_CAND,
        "_read": "C2 is ONE roll-out per window; A1 is one per candidate.",
    }

    # ---- wall-clock: the reference roll, at eval batch sizes -----------------
    rows = {}
    for B in (1, 4, 8, 16, 32):
        st = torch.randn(B, WIN, S, device=DEV)
        aw = torch.randn(B, WIN, 3, device=DEV)
        with torch.no_grad():
            s = timeit(lambda: wm_reference_rollout(world.predictor, st, aw, sr, K))
        rows[f"batch_{B}"] = {"wallclock_s": round(s, 5),
                              "ms_per_window": round(1000 * s / B, 4)}
    R["C2_reference_rollout"] = rows

    # ---- the honest comparison: the per-candidate roll, same k --------------
    B = 4
    st = torch.randn(B, WIN, S, device=DEV)
    aw = torch.randn(B, WIN, 3, device=DEV)
    ca = torch.randn(B, N_CAND, K, 2, device=DEV)
    v0n = torch.rand(B, device=DEV)
    with torch.no_grad():
        a1 = timeit(lambda: imagine_candidates(world.predictor, st, aw, ca,
                                               tuple(range(1, K + 1)), v0n),
                    warmup=1, iters=3)
    c2_b4 = R["C2_reference_rollout"]["batch_4"]["wallclock_s"]
    R["A1_per_candidate_rollout_batch4"] = {
        "wallclock_s": round(a1, 5), "ms_per_window": round(1000 * a1 / B, 4),
        "C2_is_cheaper_by": round(a1 / c2_b4, 1),
    }

    # ---- the foreign-scorer surcharge: a SECOND encode of the frame window --
    # C2 under a FOREIGN world model (the configuration that wins) must encode
    # the same frames a second time. That is the real deployment cost, and it is
    # bigger than the roll.
    ch = cfg.encoder.in_channels
    res = getattr(cfg.encoder, "image_size", None) or 224
    try:
        enc_rows = {}
        for B in (1, 4, 16):
            fr = torch.randn(B, WIN, ch, res, res, device=DEV)
            with torch.no_grad():
                e = timeit(lambda: world.encode_window(fr), warmup=1, iters=3)
            enc_rows[f"batch_{B}"] = {"wallclock_s": round(e, 5),
                                      "ms_per_window": round(1000 * e / B, 3)}
            del fr
        R["foreign_scorer_second_encode"] = {
            "frame_shape": [WIN, ch, res, res], "by_batch": enc_rows,
            "_read": "paid ONLY when the scorer is a different world model than "
                     "the one that produced the fan — which is the measured-"
                     "better configuration. Self-scoring reuses `states` and "
                     "pays 0 here, but is separated-WORSE (+0.2090).",
        }
    except Exception as ex:
        R["foreign_scorer_second_encode"] = {"ERROR": f"{type(ex).__name__}: {ex}"}

    # ---- INHERITED cross-check from the producing run's own artifact --------
    R["_cross_check_from_the_producing_run"] = {
        "source": "…/2026-07-26-v5-imagination-selection/raw/v5_v1.json",
        "evidence_class": "INHERITED (not re-measured here)",
        "gpu": "NVIDIA A40",
        "imagination_wallclock_s": 440.4,
        "n_rollouts": 225536,
        "k": 20,
        "ms_per_rollout_amortised": round(1000 * 440.4 / 225536, 4),
        "implied_C2_only_881_windows_s": round(440.4 * 881 / 225536, 2),
        "_read": "the producing run rolled 881x256 candidates AND the 881 C2 "
                 "reference rolls inside the same loop. C2 alone is 1/256 of "
                 "that work.",
    }
    Path(out_path).write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c2_cost.json")
