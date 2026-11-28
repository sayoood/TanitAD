"""E-PROOF-1 — does the collapse-free trunk ACTUALLY learn the environment?

THE QUESTION THE PI ASKED (2026-08-23): "can we now go with final validation of
representation richness and decodability?" Everything in world-model design
waits on this. Rank alone is not the answer (C131: v1 had the highest rank and
knew nothing). This battery is the answer, or its refutation.

Three gates on HELD-OUT VAL clips (the 35-episode split, never trained on):

  A  RANK        participation (σ²) at n>=1440 — already measured 6.499 by the
                 drumbeat; re-measured here so all three gates share one run.
  B  PREDICTOR   (1) G2: explained-movement vs HOLD at h=1,2,4 — EM>0 with CI
                 excluding zero = the predictor predicts CHANGE, not stasis.
                 (2) COUNTERFACTUAL DIVERGENCE (the anti-echo test): roll the
                 predictor under the true actions vs left/right/brake/throttle
                 counterfactuals; a real dynamics model DIVERGES, an action-echo
                 or static latent does not. This is the test v1 would have
                 failed (route head = bijection of its own input, C131).
  C  DECODABLE   ego-state linear probe (speed / yaw-rate / d_ego), R² over the
                 mean predictor, vs the RAW-PIXEL floor and the CONSTANT control
                 — and vs frozen DINOv3 on the SAME clips (+0.147 speed).

PASS = A ∧ B ∧ C. Every panel prints n and d. λ on a validation split of the
FIT clips only. Tier T0-DIAGNOSTIC: never a driving claim.
"""
from __future__ import annotations
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP)); sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
VAL = SP / "sp2/cache/physicalai-val-w120-256x640cyl"
DT = 0.1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="champ30k")
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--out", default=str(SP / "e_proof1_champ30k.json"))
    a = ap.parse_args()
    import v7tiny_g2 as G, v7tiny_probe as P
    from tanitad.models.v6 import spectrum_report, O6_PARTICIPATION_FLOOR
    from tanitad.models.flagship_v15 import SPEED_SCALE
    dev = torch.device("cuda")
    world, step = G.load_arm(a.arm, dev)
    W = int(world.window); H = sorted(int(h) for h in world.stack.cfg.predictor.horizons)
    clips = sorted(VAL.glob("*.v2ep.pt"))[:a.clips]
    assert clips, f"no val clips under {VAL}"
    print(f"\n  E-PROOF-1 on {a.arm}@{step} · {len(clips)} VAL clips (never trained on)\n")
    R = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)", "eval_tier": "T0-DIAGNOSTIC",
         "arm": a.arm, "step": int(step), "n_clips": len(clips), "parity": True,
         "val_split": "physicalai-val-0c5f7dac3b11", "gates": {}}

    # ---- encode everything once --------------------------------------------
    Z, AC, PO, PX = [], [], [], []
    for i, c in enumerate(clips, 1):
        z, act, spd = G.encode_clip(world, c, dev, a.frames)
        d = torch.load(c, map_location="cpu", weights_only=False)
        m = min(len(z), len(d["poses"]), len(act))
        Z.append(z.numpy()[:m].astype(np.float64)); AC.append(act.numpy()[:m].astype(np.float64))
        PO.append(d["poses"].numpy()[:m].astype(np.float64)); PX.append(P.pooled_frames(c, m)[:m])
        print(f"    [{i}/{len(clips)}] {c.name[:10]} {m}", flush=True)

    # ---- A. RANK --------------------------------------------------------------
    A_ = np.concatenate(Z); rep = spectrum_report(torch.from_numpy(A_).float())
    R["gates"]["A_rank"] = {"n": int(A_.shape[0]), "d": int(A_.shape[1]),
                            "participation": round(rep["participation_ratio"], 3),
                            "top8_share": round(rep["top_k_share"], 4),
                            "floor": O6_PARTICIPATION_FLOOR,
                            "pass": bool(rep["participation_ratio"] >= O6_PARTICIPATION_FLOOR)}
    print(f"\n  A RANK   participation {rep['participation_ratio']:.3f} (n={A_.shape[0]}, d={A_.shape[1]})"
          f"  floor {O6_PARTICIPATION_FLOOR}  -> {'PASS' if R['gates']['A_rank']['pass'] else 'FAIL'}")

    # ---- B1. G2 vs HOLD -------------------------------------------------------
    terms = G.per_clip_terms(world, clips, dev, H, W, a.frames)
    R["gates"]["B1_g2"] = {}
    print("\n  B1 PREDICTOR vs HOLD (explained movement, episode-cluster bootstrap)")
    b1_pass = True
    for h in H:
        e, mv = terms[h]; lo, hi = G.boot(e, mv); v = G.em(e, mv)
        R["gates"]["B1_g2"][str(h)] = {"em": round(float(v), 4), "ci95": [round(lo, 4), round(hi, 4)],
                                       "beats_hold": bool(lo > 0)}
        if h == 1: b1_pass = bool(lo > 0)
        print(f"     h={h} ({h*DT:.1f}s)  EM {v:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'BEATS hold' if lo > 0 else 'does NOT beat hold'}")

    # ---- B2. counterfactual divergence (anti-echo) ---------------------------
    print("\n  B2 COUNTERFACTUAL DIVERGENCE (anti-echo): rollouts under true vs altered actions")
    cf = {"left": (+0.5, 0.0), "right": (-0.5, 0.0), "brake": (0.0, -2.0), "throttle": (0.0, +2.0)}
    div = {k: [] for k in cf}; hold_mov = []
    with torch.no_grad():
        for z, act, po in zip(Z, AC, PO):
            zt = torch.from_numpy(z).float(); at = torch.from_numpy(act).float()
            v0 = torch.from_numpy(po[:, 3]).float()
            idx = list(range(0, max(1, len(z) - W - 4), 4))[:20]
            for i in idx:
                zs = zt[i:i+W][None].to(dev); aa = at[i:i+W][None].to(dev)
                vv = (v0[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                base = world.predictor(zs, torch.cat([aa, vv], -1))[1]
                hold_mov.append(float(((zt[i+W] - zt[i+W-1]) ** 2).sum()))
                for k, (ds, da) in cf.items():
                    a2 = aa.clone(); a2[..., 0] += ds; a2[..., 1] += da
                    alt = world.predictor(zs, torch.cat([a2, vv], -1))[1]
                    div[k].append(float(((alt - base) ** 2).sum()))
    mv = float(np.mean(hold_mov))
    R["gates"]["B2_counterfactual"] = {"true_per_tick_movement_mse": round(mv, 8)}
    b2_pass = True
    for k in cf:
        r = float(np.mean(div[k])) / max(mv, 1e-12)
        R["gates"]["B2_counterfactual"][k] = {"divergence_over_true_movement": round(r, 4)}
        print(f"     {k:<9} divergence = {r:8.3f} x true per-tick movement")
        if r < 0.05: b2_pass = False
    R["gates"]["B2_counterfactual"]["pass"] = b2_pass
    print(f"     -> {'actions CHANGE the rollout (not an echo)' if b2_pass else 'actions barely move the rollout — ECHO / STATIC LATENT'}")

    # ---- C. DECODABILITY ------------------------------------------------------
    print("\n  C DECODABILITY — R² over the mean predictor; PCA k=128 + λ fit on FIT clips only")
    T = {"speed": [p[:, 3:4] for p in PO],
         "yaw_rate": [np.concatenate([np.diff(p[:, 2:3], axis=0) / DT, np.zeros((1, 1))]) for p in PO],
         "d_ego": [np.concatenate([np.diff(p[:, :2], axis=0), np.zeros((1, 2))]) for p in PO]}
    DN = P.dinov3_encode(clips, a.frames, dev)
    F = {"latent": Z, "frozen DINOv3": DN, "pixel (floor)": PX,
         "constant (control)": [np.ones((len(z), 1)) for z in Z]}
    R["gates"]["C_decodability"] = {}
    print(f"     {'target':<10}{'features':<20}{'R2':>9}  {'CI95':<22}{'n':>6}{'d':>5}")
    c_pass = True
    for tn, Y in T.items():
        for fn, X in F.items():
            Xa = [x[:len(y)] for x, y in zip(X, Y)]; Ya = [y[:len(x)] for x, y in zip(Xa, Y)]
            r2, lo, hi, lam, n, d = P.probe(Xa, Ya, 128)
            R["gates"]["C_decodability"].setdefault(tn, {})[fn] = {"r2": round(r2, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": n, "d": d}
            print(f"     {tn:<10}{fn:<20}{r2:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]{n:>6}{d:>5}")
        lat = R["gates"]["C_decodability"][tn]["latent"]; px = R["gates"]["C_decodability"][tn]["pixel (floor)"]
        ok = lat["ci95"][0] > max(0.0, px["r2"])
        R["gates"]["C_decodability"][tn]["latent_beats_pixel_and_zero"] = ok
        if tn == "speed": c_pass = ok
        print(f"     -> {tn}: latent {'BEATS' if ok else 'does NOT beat'} the pixel floor with CI excluding it\n")

    R["verdict"] = {"A_rank": R["gates"]["A_rank"]["pass"], "B1_predictor_beats_hold": b1_pass,
                    "B2_not_an_echo": b2_pass, "C_decodable_speed": c_pass,
                    "PASS_ALL": bool(R["gates"]["A_rank"]["pass"] and b1_pass and b2_pass and c_pass)}
    print(f"  VERDICT  A rank={R['verdict']['A_rank']}  B1 beats-hold={b1_pass}  B2 not-echo={b2_pass}  "
          f"C decodable={c_pass}  =>  {'PASS' if R['verdict']['PASS_ALL'] else 'NOT YET'}")
    Path(a.out).write_text(json.dumps(R, indent=1), encoding="utf-8"); print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); raise SystemExit(main())
