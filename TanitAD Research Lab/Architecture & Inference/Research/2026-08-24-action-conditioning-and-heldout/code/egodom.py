"""E-DEC-41 — IS THE LATENT DOMINATED BY EGO/SCENE TRANSLATION, AND WHAT SURVIVES
DRIFT REMOVAL?

⛔ THE TWO QUESTIONS THAT PICK BETWEEN THE TWO FIXES. E-DEC-40 measured that Δz is
64 % predictable from `z_t` alone — the latent drifting along its own manifold —
with the action contributing ~4 % of that, confined to PCA directions 8-16. Two
mechanisms explain it and they imply DIFFERENT fixes:

  (a) THE LATENT'S VARIANCE IS DOMINATED BY EGO / SCENE TRANSLATION. If z mostly
      encodes "where am I along this road", Δz is "how far did I move" — which is
      predictable from position + speed, and speed is already in z_t. Other agents
      would then be a LOW-VARIANCE RESIDUAL. Consistent with n_agents R² ~0.22,
      absent lead range, and the action living in low-variance directions.
      ⇒ FIX: factor ego out, or train on an ego-compensated latent.

  (b) O5 IS SATISFIABLE BY DRIFT. Predicting z_{t+k} rewards learning the smooth
      trajectory and nothing forces the residual.
      ⇒ FIX: give O5 a DRIFT FLOOR, exactly as C149 gave the METRIC a
      constant-predictor floor. The objective becomes unsatisfiable by drift.

PART 1 — HOW MUCH OF z_t IS EGO? Predict `z_t`'s top PCA directions from:
    ego [v0, steer, accel]      the ego state alone
    scene [n_agents, occ x3, n_free]   the world
    both
    constant                    reads the no-information value EXACTLY

PART 2 — ⭐⭐ WHAT SURVIVES DRIFT REMOVAL? Fit drift ĝ(z_t) ≈ Δz on the FIT clips
only, then form the RESIDUAL r = Δz − ĝ(z_t) on the held-out clip and ask what
predicts IT:
    action        does the ego's command explain the residual?
    scene delta   does the world's change explain it?
    constant      the no-information value

⭐⭐⭐ **THIS IS THE NUMBER THAT DECIDES THE NEXT GPU-DAY.** If the residual carries
action or scene signal, a DRIFT-FLOOR objective has something real to concentrate
capacity on, and fix (b) is worth training. **If the residual is NOISE, then no
objective can help — the LATENT does not encode the world's dynamics at all, and
the next step is the ENCODER, not the loss.** O11 already showed what happens when
an objective demands information that is not present: it manufactures it.

CONTROLS: constant reading exactly 0.0000; a TIME-SHUFFLED control on every cell;
`n` and the function class stated; NONLINEAR RFF+ridge with clip-disjoint λ, since
a linear null here would say nothing (the programme has made that error before).
⚠️ The drift model ĝ is fit on the FIT CLIPS ONLY and applied to the held-out
clip — fitting it on all clips would remove the residual we are trying to measure.

T0-DIAGNOSTIC. Held-out, lead-matched. MEASURED (ours; dev-box RTX 4060).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
LEAD = Path(os.environ.get("SPD_CORPUS",
                           str(SP / "sp2/cache/physicalai-val130-heldout")))
LABELS = Path(os.environ.get("SPD_LABELS", str(SP / "sp2/val130_agents.jsonl")))
OUT = Path(os.environ.get("SPD_OUT", str(SP / "egodom.json")))
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k").split(",")
MIN_LEAD, N_CLIPS, F, K, N_DIR = 20, 20, 100, 4, 8


def main() -> int:
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r

    dev = torch.device("cuda")
    LAB = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])

    def n_lead(cid):
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0
                          for x in LAB.get(cid, {}).get(i, [])))

    def scene_feats(cid, m):
        from tanitad.data.psg_targets import PSG_N_COLS, azimuth_column
        out = np.zeros((m, 5))
        for i in range(m):
            a = LAB.get(cid, {}).get(i, [])
            cols = np.zeros(PSG_N_COLS)
            for q in a:
                c = azimuth_column(float(q["cx"]), float(q["cy"]))
                if c is not None:
                    cols[c] += 1
            out[i] = [len(a), np.log1p(cols[:3].sum()), np.log1p(cols[3:5].sum()),
                      np.log1p(cols[5:].sum()), (cols == 0).sum()]
        return out

    clips = [c for c in sorted(LEAD.glob("*.v2ep.pt"))
             if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]
    clips = [c for c in clips
             if n_lead(torch.load(c, map_location="cpu",
                                  weights_only=False)["clip_id"]) >= MIN_LEAD][:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-41 · IS THE LATENT EGO-DOMINATED, AND WHAT SURVIVES DRIFT"
          f" REMOVAL?\n  arms {present} · {len(clips)} lead-matched held-out clips"
          f" · k={K}\n", flush=True)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT, LEAD-MATCHED", "k": K,
           "function_class": "RFF+ridge (NONLINEAR, convex), clip-disjoint lambda",
           "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        Z, DZ, EGO, SC, DSC, PIX = [], [], [], [], [], []
        for c in clips:
            d, _r, _o, n_all, _ = G.frames_of(c)
            z, act, spd = G.encode_clip(w, c, dev, F)
            zt = z.float().numpy().astype(np.float64)
            a = act.float().numpy().astype(np.float64)
            v = spd.float().numpy().astype(np.float64).reshape(-1, 1)
            m = min(len(zt) - K, len(a) - K, len(v) - K)
            sc = scene_feats(d["clip_id"], min(n_all, F))
            if m < 25 or len(sc) < m + K:
                continue
            i0 = np.arange(m)
            # ⭐⭐ RAW PIXELS, added after E-DEC-41's first pass. PART 1 found that
            # NEITHER the 3 ego channels NOR the 5 scene scalars explain z_t's
            # top-8 directions (+0.0221 t 1.49 / -0.0050 t -0.35). ⚠️ BUT THAT IS
            # PARTLY A STATEMENT ABOUT MY PROBE'S INPUTS: 3 + 5 low-dimensional
            # labels cannot explain 8 latent directions even if the content IS
            # there. Raw pixels are HIGH-dimensional and label-free, so they
            # separate the two readings: if PIXELS explain the top directions, the
            # latent is an APPEARANCE code and our labels were simply too thin to
            # see it. If pixels ALSO fail, the top variance is SELF-GENERATED
            # structure that corresponds to nothing in the input — which is the
            # E-DEC-7 degeneracy shape (a non-collapsed optimum with no external
            # content), now with the ego part absent too.
            import io as _io
            from PIL import Image as _Im
            _im = [torch.from_numpy(np.asarray(
                _Im.open(_io.BytesIO(_r[_o[q]:_o[q+1]])).convert("RGB")).copy())
                .permute(2,0,1).float()/255.0 for q in range(min(n_all, F))]
            _px = torch.nn.functional.adaptive_avg_pool2d(
                torch.stack(_im)[:, -3:], (8, 8)).reshape(len(_im), -1).numpy()
            PIX.append(_px[i0].astype(np.float64))
            Z.append(zt[i0]); DZ.append(zt[i0 + K] - zt[i0])
            EGO.append(np.concatenate([v[i0], a[i0]], 1))
            SC.append(sc[i0]); DSC.append(sc[i0 + K] - sc[i0])
        del w
        torch.cuda.empty_cache()

        def panel(TARGETS, cols, title, key):
            print(f"  --- {title} ---")
            print(f"  {'column':<22}{'mean r':>10}{'shuffled':>11}"
                  f"{'true-shuf':>11}{'t':>7}")
            rng = np.random.default_rng(0)
            rep["arms"][arm][key] = {}
            for cname, X in cols.items():
                tr, sh = [], []
                for j in range(len(TARGETS)):
                    Y = TARGETS[j]
                    Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
                    for i in range(len(X)):
                        Xtr = [X[q] for q in range(len(X)) if q != i]
                        for Yv, sink in ((Y, tr), (Ysh, sh)):
                            ytr = [Yv[q] for q in range(len(Yv)) if q != i]
                            pred, _ = rff_fold(Xtr, ytr, X[i])
                            sink.append(within_clip_r(pred, Yv[i].ravel()))
                tr, sh = np.array(tr), np.array(sh)
                dd = tr - sh
                t = float(dd.mean()) / max(float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
                rep["arms"][arm][key][cname] = {
                    "mean_r": round(float(tr.mean()), 4),
                    "shuffled": round(float(sh.mean()), 4),
                    "true_minus_shuffled": round(float(dd.mean()), 4),
                    "t": round(t, 2), "n_scores": len(tr)}
                print(f"  {cname:<22}{tr.mean():>+10.4f}{sh.mean():>+11.4f}"
                      f"{dd.mean():>+11.4f}{t:>7.2f}", flush=True)
            print()

        rep["arms"][arm] = {"step": int(st), "n_rows": int(sum(len(x) for x in Z))}

        # ---- PART 1: how much of z_t is EGO? -----------------------------
        Za = np.concatenate(Z); mu = Za.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(Za - mu, full_matrices=False)
        ev = sv ** 2 / (sv ** 2).sum()
        print(f"  === {arm} (step {st}) · {len(Za)} rows ===")
        print(f"  z_t spectrum: top-1 {ev[0]:.1%} · top-8 {ev[:8].sum():.1%}\n")
        T1 = [[(zz - mu) @ Vt[j][:, None] for zz in Z] for j in range(N_DIR)]
        panel(T1, {"ego [v0,steer,accel]": EGO,
                   "scene [n_ag,occ,free]": SC,
                   "RAW PIXELS 8x8": PIX,
                   "both ego+scene": [np.concatenate([e, s], 1) for e, s in zip(EGO, SC)],
                   "constant (control)": [np.ones((len(x), 1)) for x in Z]},
              "PART 1 · what explains z_t's top-8 directions?", "z_t")

        # ---- PART 2: what survives DRIFT REMOVAL? ------------------------
        # ⛔ the drift model is fit on the FIT CLIPS ONLY, per held-out fold.
        DZa = np.concatenate(DZ); dmu = DZa.mean(0, keepdims=True)
        _, _, DVt = np.linalg.svd(DZa - dmu, full_matrices=False)
        RES = []
        for j in range(N_DIR):
            Yj = [(dz - dmu) @ DVt[j][:, None] for dz in DZ]
            rj = []
            for i in range(len(Z)):
                Xtr = [Z[q] for q in range(len(Z)) if q != i]
                ytr = [Yj[q] for q in range(len(Yj)) if q != i]
                pred, _ = rff_fold(Xtr, ytr, Z[i])
                rj.append((Yj[i].ravel() - pred)[:, None])
            RES.append(rj)
        panel(RES, {"action [steer,accel,v0]": EGO, "scene delta": DSC,
                    "constant (control)": [np.ones((len(x), 1)) for x in Z]},
              "PART 2 · what predicts the DRIFT-REMOVED RESIDUAL?  ⭐ THE DECIDER",
              "residual")

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
