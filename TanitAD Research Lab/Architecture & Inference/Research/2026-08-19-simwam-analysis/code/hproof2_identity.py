"""H-PROOF-2 — is champ30k's predictor literally an IDENTITY MAP beyond one tick?

E-PROOF-1 raw gives explained-movement **+0.1303** [+0.1079, +0.1512] at h=1 but
**exactly -0.0 with a ZERO-WIDTH CI** at h=2 ([-0.0, +0.0]) and h=4 ([-0.0, -0.0]).

    EM = 1 - ||z_hat - z_plus||^2 / ||z - z_plus||^2

is identically 0 iff `z_hat == z`, i.e. the predictor returns its INPUT.  That is
a reading, not a measurement, and a zero-width CI at exactly 0.0000 is the same
shape as the lambda-selected-on-test failure that manufactured four false probe
results on 2026-08-22.  So it gets tested, not reported.

THE DIRECT TEST, on held-out val clips:

    identity_ratio(h) = mean ||z_hat(h) - z_last||  /  mean ||z_plus(h) - z_last||

  ~0  => the prediction IS the last input: an identity map (H-PROOF-2 SUPPORTED)
  ~1  => the prediction moves as far as the truth does (REFUTED; the zero has
         another cause -- a clamp, a horizon-indexing bug in the harness, or a
         genuine tie)

⭐ THE CONTROL THAT MAKES IT READABLE: the same quantity for arm `fixed`, which is
known action-SENSITIVE (divergence 474.9x).  If `fixed` also reads ~0 the probe is
measuring something about the harness rather than about champ30k, and the result
must be discarded.  A number with no control is what this file exists to avoid.

T0-DIAGNOSTIC.  Held-out val clips only.
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
OUT = SP / "h_proof2_identity.json"


def main() -> int:
    import v7tiny_g2 as G
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    clips = sorted(VAL.glob("*.v2ep.pt"))[:8]
    assert clips, f"no val clips under {VAL}"
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "hypothesis": "H-PROOF-2",
           "n_clips": len(clips), "arms": {}}

    print(f"\n  H-PROOF-2 - identity test on {len(clips)} held-out val clips")
    print(f"  ratio = mean||z_hat - z_last|| / mean||z_true - z_last||   "
          f"(~0 = identity, ~1 = moves like truth)\n")
    print(f"  {'arm':<10}{'h':>4}{'||z_hat-z_last||':>19}{'||z_true-z_last||':>20}"
          f"{'ratio':>9}   reading")
    print("  " + "-" * 84)

    for arm in ("champ30k", "fixed"):
        try:
            world, step = G.load_arm(arm, dev)
        except Exception as e:
            print(f"  {arm}: NOT LOADABLE ({type(e).__name__}) - reported, not skipped")
            rep["arms"][arm] = {"error": f"{type(e).__name__}"}
            continue
        W = int(world.window)
        hs = sorted(int(h) for h in world.stack.cfg.predictor.horizons)
        acc = {h: {"pred": [], "true": []} for h in hs}
        with torch.no_grad():
            for c in clips:
                z, act, spd = G.encode_clip(world, c, dev, 120)
                zt = z.float()
                for i in range(0, max(1, len(z) - W - max(hs) - 1), 5):
                    zs = zt[i:i + W][None].to(dev)
                    aa = act[i:i + W][None].to(dev)
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    outs = world.predictor(zs, torch.cat([aa, vv], -1))
                    z_last = zt[i + W - 1].to(dev)
                    # ⛔ `predictor` returns a DICT KEYED BY HORIZON, not a list.
                    # An earlier version indexed it positionally as outs[k+1];
                    # that was right for h=1 and h=2 only because the keys happen
                    # to be 1 and 2, and it SILENTLY DROPPED h=4 via a len() guard.
                    # Key explicitly, and fail loudly if a horizon is absent.
                    for h in hs:
                        if h not in outs:
                            raise KeyError(f"horizon {h} absent from predictor "
                                           f"output keys {sorted(outs)}")
                        zh = outs[h].reshape(-1)[: z_last.numel()]
                        j = i + W - 1 + h
                        if j >= len(zt):
                            continue
                        acc[h]["pred"].append(float(torch.linalg.vector_norm(zh - z_last)))
                        acc[h]["true"].append(float(torch.linalg.vector_norm(
                            zt[j].to(dev) - z_last)))
        rows = {}
        for h in hs:
            if not acc[h]["pred"]:
                continue
            mp = float(np.mean(acc[h]["pred"]))
            mt = float(np.mean(acc[h]["true"]))
            r = mp / max(mt, 1e-12)
            reading = ("IDENTITY (returns its input)" if r < 0.05
                       else "near-identity" if r < 0.25
                       else "moves like truth" if r > 0.6 else "partial")
            rows[str(h)] = {"pred_norm": round(mp, 5), "true_norm": round(mt, 5),
                            "identity_ratio": round(r, 5), "reading": reading,
                            "n": len(acc[h]["pred"])}
            print(f"  {arm:<10}{h:>4}{mp:>19.5f}{mt:>20.5f}{r:>9.4f}   {reading}")
        rep["arms"][arm] = {"step": int(step), "by_horizon": rows}
        del world
        torch.cuda.empty_cache()

    ch = rep["arms"].get("champ30k", {}).get("by_horizon", {})
    fx = rep["arms"].get("fixed", {}).get("by_horizon", {})
    long_h = [h for h in ch if h != "1"]
    ch_flat = all(ch[h]["identity_ratio"] < 0.05 for h in long_h) if long_h else False
    fx_flat = all(fx[h]["identity_ratio"] < 0.05 for h in fx if h != "1") if fx else False
    if ch_flat and fx_flat:
        rep["verdict"] = ("⛔ DISCARD — the action-SENSITIVE control `fixed` also reads ~0, so this "
                          "probe is measuring the harness (a horizon-indexing or readout-shape "
                          "mismatch), NOT champ30k. H-PROOF-2 stays OPEN and the instrument is "
                          "the thing to fix.")
    elif ch_flat and not fx_flat:
        rep["verdict"] = ("H-PROOF-2 SUPPORTED — champ30k's predictor returns its input beyond "
                          "h=1 while the control moves. The two-term recipe learned the TRIVIAL "
                          "solution; 'beats HOLD at h=1' is its only non-trivial behaviour.")
    elif not ch_flat:
        rep["verdict"] = ("H-PROOF-2 REFUTED — champ30k's prediction does move away from its "
                          "input, so the exactly-zero EM at h>=2 has another cause. Investigate "
                          "the EM computation itself before quoting those zeros again.")
    else:
        rep["verdict"] = "INDETERMINATE — control missing or horizons did not align"
    print(f"\n  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
