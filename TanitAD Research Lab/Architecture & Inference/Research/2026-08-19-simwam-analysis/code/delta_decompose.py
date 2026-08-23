"""Is the predicted delta MIS-SCALED, or is it NOISE? Decompose direction vs magnitude.

C137 corrected the diagnosis: every arm IS action-conditioned, but no arm predicts
the right MAGNITUDE of latent change -- delta/true_move spans 0.020x (`lewm`) to
25.8x (`fixed`), with champ30k at 0.264x one tick out.

"Too small" and "noise" look identical in an MSE, so decompose:

    cos      = <delta, dz_true> / (||delta|| ||dz_true||)     -- DIRECTION
    alpha*   = <delta, dz_true> / <delta, delta>              -- the best rescale
    EM(1)    = 1 - ||delta - dz||^2 / ||dz||^2                -- as shipped
    EM(a*)   = 1 - ||a* delta - dz||^2 / ||dz||^2  ==  cos^2  -- the ceiling a pure
                                                                rescale can reach

Readings:
  cos ~ 0, alpha* ~ 0   -> the delta carries NO directional signal. A rescale
                           cannot help. (This is what predictor.py records for
                           v6F: "the error fell monotonically to alpha=0".)
  cos high, alpha* >> 1 -> direction RIGHT, magnitude too SMALL: a scalar fixes
                           most of it, and the head init/scale is the lever.
  cos high, alpha* ~ 1  -> already calibrated; the loss is elsewhere.

⭐ CONTROLS, because a correlation-like statistic invites exactly the failures of
2026-08-22: a SHUFFLED-TARGET arm (pair each delta with another window's true
movement) must read cos ~ 0, and a HOLD control (delta := 0) must read EM = 0
EXACTLY. If the shuffle does not read ~0 the probe is measuring structure that
survives permutation, i.e. leakage, and the panel is void.

T0-DIAGNOSTIC. Held-out val clips.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")

VAL = SP / "sp2/cache/physicalai-val-w120-256x640cyl"
OUT = SP / "delta_decompose.json"
ARMS = ("champ30k", "lewm", "lewm_o1", "lewm_o1_detach", "fixed")


def main() -> int:
    import v7tiny_g2 as G
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    clips = sorted(VAL.glob("*.v2ep.pt"))[:6]
    assert clips, f"no val clips under {VAL}"
    rng = np.random.default_rng(0)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "arms": {}}

    print("\n  DELTA DECOMPOSITION — is the prediction mis-scaled, or noise?\n")
    print(f"  {'arm':<16}{'h':>3}{'cos':>9}{'alpha*':>9}{'EM(1)':>9}{'EM(a*)':>9}"
          f"{'cos SHUF':>10}{'n':>6}   reading")
    print("  " + "-" * 96)

    for arm in ARMS:
        try:
            world, step = G.load_arm(arm, dev)
        except Exception as e:
            print(f"  {arm:<16} NOT LOADABLE ({type(e).__name__})")
            rep["arms"][arm] = {"error": type(e).__name__}
            continue
        W = int(world.window)
        hs = sorted(int(h) for h in world.stack.cfg.predictor.horizons)
        D = {h: [] for h in hs}
        T = {h: [] for h in hs}
        with torch.no_grad():
            for c in clips:
                z, act, spd = G.encode_clip(world, c, dev, 100)
                zt = z.float()
                for i in range(0, max(1, len(zt) - W - max(hs) - 1), 5):
                    zs = zt[i:i + W][None].to(dev)
                    aa = act[i:i + W][None].to(dev)
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    outs = world.predictor(zs, torch.cat([aa, vv], -1))
                    z_last = zt[i + W - 1].to(dev)
                    for h in hs:
                        j = i + W - 1 + h
                        if h not in outs or j >= len(zt):
                            continue
                        D[h].append((outs[h].reshape(-1)[:z_last.numel()] - z_last).cpu().numpy())
                        T[h].append((zt[j].to(dev) - z_last).cpu().numpy())
        rows = {}
        for h in hs:
            if len(D[h]) < 10:
                continue
            d = np.stack(D[h]).astype(np.float64)
            t = np.stack(T[h]).astype(np.float64)
            # ⛔ MEAN-CENTRE BOTH before any correlation-like statistic.
            # The shuffled control fired at 0.0664 on champ30k h=1 WITHOUT this:
            # every delta points broadly one way and every true movement points
            # broadly one way (the ego drives forward), so ANY pairing scores
            # positive. That common component is the "everything reproduces the
            # MEAN" failure of 2026-08-22 in a cosine costume. Centring removes
            # it, and the shuffled control is what verifies the removal worked.
            d = d - d.mean(0, keepdims=True)
            t = t - t.mean(0, keepdims=True)
            num = float((d * t).sum())
            cos = num / max(float(np.linalg.norm(d) * np.linalg.norm(t)), 1e-30)
            alpha = num / max(float((d * d).sum()), 1e-30)
            em1 = 1.0 - float(((d - t) ** 2).sum()) / max(float((t ** 2).sum()), 1e-30)
            ema = 1.0 - float(((alpha * d - t) ** 2).sum()) / max(float((t ** 2).sum()), 1e-30)
            # CONTROL: pair each delta with a DIFFERENT window's true movement
            perm = rng.permutation(len(t))
            ts = t[perm]
            cos_s = float((d * ts).sum()) / max(
                float(np.linalg.norm(d) * np.linalg.norm(ts)), 1e-30)
            reading = ("NOISE — no directional signal" if abs(cos) < 0.05 else
                       "direction RIGHT, too SMALL" if cos >= 0.05 and alpha > 1.5 else
                       "direction RIGHT, too LARGE" if cos >= 0.05 and alpha < 0.67 else
                       "direction right, calibrated")
            rows[str(h)] = {"cos": round(cos, 4), "alpha_star": round(alpha, 4),
                            "em_alpha1": round(em1, 4), "em_alpha_star": round(ema, 4),
                            "cos_shuffled_control": round(cos_s, 4),
                            "n": len(d), "reading": reading}
            print(f"  {arm:<16}{h:>3}{cos:>9.4f}{alpha:>9.4f}{em1:>9.4f}{ema:>9.4f}"
                  f"{cos_s:>10.4f}{len(d):>6}   {reading}")
        rep["arms"][arm] = {"step": int(step), "by_horizon": rows}
        del world
        torch.cuda.empty_cache()

    # ---- control audit: any shuffled cos that is not ~0 voids that row --------
    bad = [(a, h) for a, v in rep["arms"].items() if "by_horizon" in v
           for h, r in v["by_horizon"].items() if abs(r["cos_shuffled_control"]) > 0.05]
    rep["shuffled_control_violations"] = bad
    ch = rep["arms"].get("champ30k", {}).get("by_horizon", {}).get("1")
    if bad:
        rep["verdict"] = (f"⛔ the SHUFFLED control is non-zero on {bad} — those rows measure "
                          f"structure that survives permutation and are VOID.")
    elif ch:
        rep["verdict"] = (
            f"champ30k h=1: cos {ch['cos']:.4f}, alpha* {ch['alpha_star']:.2f}. "
            + ("The delta carries NO directional signal, so no rescale can help — the head is "
               "emitting noise." if abs(ch["cos"]) < 0.05 else
               f"Direction is real; a single scalar rescale would lift explained movement "
               f"{ch['em_alpha1']:+.4f} -> {ch['em_alpha_star']:+.4f}. The magnitude, not the "
               f"direction, is the defect."))
    else:
        rep["verdict"] = "champ30k h=1 not measured"
    print(f"\n  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
