"""EXPLORATORY context for the A4 L2 lever (NOT pre-registered; reported as context, never as a result).

usage: python lever_context.py <tag_dir>

Question: is the L2 gain specific to refcv6, or does ANY learned plan blended with the causal hold
beat the echo? The SAME cross-fit (`lever_panel.cross_fit_blend`, 2-fold episode parity, 0.05 grid)
is applied to the banked baselines' `os` (refcv4b, refcv5-v2 seed 0) on the SAME panel windows.
Writes `<tag_dir>/levers/context_baseline_blends.json`.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_panel as P  # noqa: E402
import lever_panel as LP  # noqa: E402


def main():
    root = Path(sys.argv[1])
    man, A, eid = P.load_panel(str(root / "panel_s0"))
    dt = float(man["grid"]["dt_s"])
    G = A["g"]
    a_echo, a_ha = LP.ade(A["ha0_ext"], G, dt), LP.ade(A["ha"], G, dt)
    out = {"status": "EXPLORATORY (not pre-registered; context for SPEC A4 L2)", "n_windows": int(len(eid)),
           "n_episodes": int(len(set(eid.tolist()))), "arms": {}}
    for arm in ("os", "b_refcv4b", "b_refcv5v2_s0"):
        bl, w = LP.cross_fit_blend(A[arm], A["ha"], G, eid)
        a_bl, a_arm = LP.ade(bl, G, dt), LP.ade(A[arm], G, dt)
        out["arms"][arm] = {"weights": w, "arm_minus_echo": LP.paired(a_arm, a_echo, eid),
                            "blend_minus_echo": LP.paired(a_bl, a_echo, eid),
                            "blend_minus_ha": LP.paired(a_bl, a_ha, eid)}
    (root / "levers").mkdir(exist_ok=True)
    json.dump(out, open(root / "levers" / "context_baseline_blends.json", "w", encoding="utf-8"), indent=1)
    for arm, v in out["arms"].items():
        print(f"{arm:14s} arm−echo {LP._c(v['arm_minus_echo'])} | blend(arm,ha)−echo {LP._c(v['blend_minus_echo'])} "
              f"| w {v['weights']['w_fit_on_fold0_scores_fold1']} / {v['weights']['w_fit_on_fold1_scores_fold0']}")


if __name__ == "__main__":
    main()
