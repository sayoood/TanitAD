"""E-DEC-1c: LEAVE-ONE-EPISODE-OUT paired test — encoder tokens vs z_op.

The episode bootstrap at 12 clusters was degenerate (CI +-1.6 against an effect
of ~0.2), because every resample REFITS lambda and the PCA basis on ~12 noisy
clusters. LOEO is the stable design: fit on 11 episodes, score the held-out one,
and pair the two feature columns ON THAT SAME EPISODE. The episode-level
variance -- the dominant term -- cancels inside each pair.

Reported per pair: mean paired delta, SE over episodes, a t-like ratio, and the
SIGN COUNT (how many of 12 episodes favour the first column), which is
distribution-free and does not care about R2's heavy tail.

CONTROL: enc-vs-enc must be exactly 0 on every episode, or the pairing is broken.
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
OUT = SP / "e_dec1c_loeo.json"


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P

    dev = torch.device("cuda")
    arm = sys.argv[1] if len(sys.argv) > 1 else "champ30k"
    clips = sorted((SP / "sp2/cache/physicalai-val-w120-256x640cyl").glob("*.v2ep.pt"))
    F = 120
    world, st = G.load_arm(arm, dev)
    ZOP, ENC, PIX, PO = [], [], [], []
    for c in clips:
        z, _, _ = G.encode_clip(world, c, dev, F)
        ZOP.append(z.numpy().astype(np.float64))
        ENC.append(np.asarray(P.encode_tokens_meanpooled(world, c, dev, F), dtype=np.float64))
        d = torch.load(c, map_location="cpu", weights_only=False)
        PO.append(d["poses"].numpy().astype(np.float64)[:F])
        PIX.append(P.pooled_frames(c, F))
    DN = P.dinov3_encode(clips, F, dev)
    del world
    torch.cuda.empty_cache()

    T = {"speed": [p[:, 3:4] for p in PO],
         "d_ego": [np.concatenate([np.diff(p[:, :2], axis=0), np.zeros((1, 2))]) for p in PO]}
    COL = {"enc": ENC, "z_op": ZOP, "dino": DN, "pix": PIX}
    n = len(clips)

    def loeo(X, Y):
        out = []
        for e in range(n):
            idx = [i for i in range(n) if i != e]
            Xf = [np.asarray(X[i])[:len(Y[i])] for i in idx]
            Yf = [Y[i][:len(np.asarray(X[i]))] for i in idx]
            Xs = [np.asarray(X[e])[:len(Y[e])]]
            Ys = [Y[e][:len(np.asarray(X[e]))]]
            fn = getattr(P, "probe_fit_score", None)
            out.append(fn(Xf, Yf, Xs, Ys, 128) if fn else P.probe(Xf + Xs, Yf + Ys, 128)[0])
        return np.array(out, dtype=np.float64)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "arm": arm, "step": int(st), "n_episodes": n,
           "estimator": "leave-one-episode-out PAIRED differences",
           "used_probe_fit_score": bool(getattr(P, "probe_fit_score", None)),
           "targets": {}}
    print(f"\n  E-DEC-1c LOEO PAIRED · {arm}@{st} · {n} episodes"
          f"  (fit/score API: {'probe_fit_score' if rep['used_probe_fit_score'] else 'probe fallback'})\n",
          flush=True)
    print(f"  {'target':<9}{'pair':<14}{'mean d':>9}{'SE':>8}{'t':>7}{'favours 1st':>13}  reading")
    print("  " + "-" * 78)
    for tn, Y in T.items():
        R = {k: loeo(COL[k], Y) for k in COL}
        rep["targets"][tn] = {}
        for a, b in (("enc", "z_op"), ("enc", "pix"), ("z_op", "pix"),
                     ("enc", "dino"), ("enc", "enc")):
            d = R[a] - R[b]
            m = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            t = m / max(se, 1e-12)
            k = int((d > 0).sum())
            rd = ("CONTROL — must be 0" if a == b else
                  f"{a} BEATS {b}" if t > 2.2 else f"{a} LOSES to {b}" if t < -2.2
                  else "not separable")
            rep["targets"][tn][f"{a}-{b}"] = {"mean_delta": round(m, 4), "se": round(se, 4),
                                              "t": round(t, 2), "n_favouring_first": k,
                                              "n": len(d)}
            print(f"  {tn:<9}{a + ' - ' + b:<14}{m:>+9.4f}{se:>8.4f}{t:>7.2f}"
                  f"{k:>9}/{len(d)}  {rd}", flush=True)
        print()

    bad = [tn for tn in T if abs(rep["targets"][tn]["enc-enc"]["mean_delta"]) > 1e-12]
    rep["pairing_control_violations"] = bad
    sp, de = rep["targets"]["speed"], rep["targets"]["d_ego"]
    rd = [t for t, r in (("speed", sp), ("d_ego", de)) if r["enc-z_op"]["t"] > 2.2]
    lp = [t for t, r in (("speed", sp), ("d_ego", de)) if r["enc-pix"]["t"] > 2.2]
    if bad:
        rep["verdict"] = f"⛔ VOID — self-pairing control non-zero on {bad}"
    else:
        rep["verdict"] = (
            f"readout destroys signal on {rd or 'NO target'}; encoder beats raw pixels on "
            f"{lp or 'NO target'}. "
            + ("⭐ FIX THE READOUT — the trunk HAS content the 4x4 pooling throws away."
               if rd else
               ("The encoder beats pixels, but the readout is not the binding loss."
                if lp else
                "The encoder does NOT beat raw pixels ⇒ the OBJECTIVE is the defect.")))
    print(f"  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
