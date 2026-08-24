"""E-DEC-40 — WHAT IS ACTUALLY IN THE LATENT TRANSITION Δz?

⛔ THE QUESTION EVERY NEGATIVE TONIGHT POINTS AT. Four independent results say the
same thing from different angles:
  * the predictor is action-independent (E-DEC-30/33, 32 of 32 arms);
  * the action is NOT recoverable from Δz (E-DEC-39, and flat at k=2 and k=4);
  * forcing action-dependence yields only the DEGENERATE ẑ = f(z) + λa (O11);
  * raw 8×8 pixels match or beat the latent on every physics-relevant target.
**If Δz carried ego motion, the action would be recoverable from it. It is not.
So what IS Δz?** This decomposes it instead of inferring it.

THE DECOMPOSITION. Predict Δz = z_{t+k} − z_t from each of:

    z_t                  the PRESENT latent — how much of the "change" is just a
                         deterministic function of where we already are (i.e. the
                         latent drifting along its own manifold, carrying no new
                         information at all)
    action [steer,accel,v0]   the ego's own command
    scene delta          Δn_agents, Δocc_left/center/right, Δn_free_cols — the
                         world actually changing
    all of the above     the union
    constant             reads the no-information value EXACTLY

⭐⭐ THE HEADLINE IS THE RESIDUAL. If Δz is largely unexplained by ANY of these,
it is noise — and that single fact would explain every result above at once: the
predictor cannot predict noise, the action cannot be recovered from noise, and O5
is minimised by predicting the mean, which is exactly what 32 of 32 arms do.

⚠️⚠️ THE TRAP THIS FILE MUST NOT FALL INTO, AND THE PROGRAMME HAS FALLEN INTO IT
BEFORE. From `CLAUDE.md`: *"I wrote '~98 % unpredictable ⇒ unreachable by ANY
predictor' from a ridge fit — in 2048 dims with nonlinear scene motion that is a
WEAK lower bound and the claim was unsupported."* ⇒ This uses the **NONLINEAR**
RFF+ridge probe, states the function class in the verdict, and carries a
**TIME-SHUFFLED** control (structure surviving a shuffle is leakage, not
dynamics). A residual measured by a linear map would say nothing.

⚠️ Δz is 2048-dimensional. Scoring "fraction of Δz explained" per-dimension and
averaging is the wrong statistic (it weights 2048 mostly-noise directions
equally). This scores the TOP PCA DIRECTIONS of Δz — the ones that carry its
variance — and reports how many directions cover 90 % of it, so the reader can
see whether Δz is low-dimensional structure or high-dimensional noise.

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
OUT = Path(os.environ.get("SPD_OUT", str(SP / "deltaz.json")))
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k,splitp30k").split(",")
MIN_LEAD, N_CLIPS, F = 20, 20, 100
K = int(os.environ.get("SPD_K", "4"))
N_DIR = 8            # top PCA directions of Δz to score


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
             if torch.load(c, map_location="cpu",
                           weights_only=False)["clip_id"] in LAB]
    clips = [c for c in clips
             if n_lead(torch.load(c, map_location="cpu",
                                  weights_only=False)["clip_id"]) >= MIN_LEAD][:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-40 · WHAT IS ACTUALLY IN THE LATENT TRANSITION Δz?"
          f"\n  arms {present} · {len(clips)} lead-matched held-out clips · k={K}"
          f"\n  ⭐ THE HEADLINE IS THE RESIDUAL — what NOTHING explains\n", flush=True)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT, LEAD-MATCHED", "k": K,
           "function_class": "RFF+ridge (NONLINEAR, convex), clip-disjoint lambda. "
                             "A residual measured by a LINEAR map would say nothing "
                             "— the programme has made that error before.",
           "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        DZ, COL = [], {"z_t (drift)": [], "action": [], "scene delta": [],
                       "all three": [], "constant (control)": []}
        for c in clips:
            d, _raw, _off, n_all, _ = G.frames_of(c)
            z, act, spd = G.encode_clip(w, c, dev, F)
            zt = z.float().numpy().astype(np.float64)
            a = act.float().numpy().astype(np.float64)
            v = spd.float().numpy().astype(np.float64).reshape(-1, 1)
            m = min(len(zt) - K, len(a) - K, len(v) - K)
            if m < 25:
                continue
            i0 = np.arange(m)
            sc = scene_feats(d["clip_id"], min(n_all, F))
            if len(sc) < m + K:
                continue
            dsc = sc[i0 + K] - sc[i0]
            act3 = np.concatenate([a[i0], v[i0]], 1)
            DZ.append(zt[i0 + K] - zt[i0])
            COL["z_t (drift)"].append(zt[i0])
            COL["action"].append(act3)
            COL["scene delta"].append(dsc)
            COL["all three"].append(np.concatenate([zt[i0], act3, dsc], 1))
            COL["constant (control)"].append(np.ones((m, 1)))
        del w
        torch.cuda.empty_cache()

        # ⭐ score the TOP PCA DIRECTIONS of Δz, not all 2048 dims equally
        DZa = np.concatenate(DZ)
        mu = DZa.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(DZa - mu, full_matrices=False)
        ev = sv ** 2 / (sv ** 2).sum()
        n90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
        print(f"  === {arm} (step {st}) · {len(DZ)} clips · {len(DZa)} rows ===")
        print(f"  Δz spectrum: top-1 {ev[0]:.1%} · top-8 {ev[:8].sum():.1%} · "
              f"{n90} directions cover 90 %  (of {DZa.shape[1]} dims)")
        dirs = Vt[:N_DIR]
        Y = {j: [ (dz - mu) @ dirs[j][:, None] for dz in DZ ] for j in range(N_DIR)}
        rng = np.random.default_rng(0)
        rep["arms"][arm] = {"step": int(st), "n_rows": int(len(DZa)),
                            "dz_top1_var": round(float(ev[0]), 4),
                            "dz_top8_var": round(float(ev[:8].sum()), 4),
                            "dz_dirs_for_90pct": n90, "columns": {}}
        print(f"\n  {'column':<22}{'mean r (top-8 dirs)':>21}{'shuffled':>11}"
              f"{'true-shuf':>11}{'t':>7}")
        for cname, X in COL.items():
            tr_all, sh_all = [], []
            for j in range(N_DIR):
                Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y[j]]
                for i in range(len(X)):
                    Xtr = [X[q] for q in range(len(X)) if q != i]
                    for Yv, sink in ((Y[j], tr_all), (Ysh, sh_all)):
                        ytr = [Yv[q] for q in range(len(Yv)) if q != i]
                        pred, _ = rff_fold(Xtr, ytr, X[i])
                        sink.append(within_clip_r(pred, Yv[i].ravel()))
            tr, sh = np.array(tr_all), np.array(sh_all)
            dd = tr - sh
            t = float(dd.mean()) / max(float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            rep["arms"][arm]["columns"][cname] = {
                "mean_r": round(float(tr.mean()), 4),
                "shuffled_r": round(float(sh.mean()), 4),
                "true_minus_shuffled": round(float(dd.mean()), 4),
                "t": round(t, 2), "n_scores": len(tr)}
            print(f"  {cname:<22}{tr.mean():>+21.4f}{sh.mean():>+11.4f}"
                  f"{dd.mean():>+11.4f}{t:>7.2f}", flush=True)
        print()
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
