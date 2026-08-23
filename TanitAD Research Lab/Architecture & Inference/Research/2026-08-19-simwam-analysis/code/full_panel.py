"""E-DEC-9 / E-DEC-10 FULL PANEL — the read-out with the kill-gate built in.

Arms (matched: same geometry, steps, O5+O6; O1 OFF; differing ONLY in the
external target, or its absence):
    rdw8     no external target      -- the incumbent objective
    o7w1p0   frozen DINOv3 cells     -- teacher (E-DEC-9)
    o8w1p0   raw-pixel cell patches  -- teacher-free (E-DEC-10)

⛔ MEASURING DECODABILITY ALONE WOULD REPRODUCE THE O1 MISTAKE. O1 bought the
thing being measured (action response) while destroying the thing that worked
(directional correctness), and O3 bought nothing while destroying everything.
So the panel reports, TOGETHER:

  A  RANK          participation on BOTH held-out sets (val, held24)
  B  PREDICTOR     mean-centred cos vs a 200-draw PERMUTATION NULL at h=1,2,4
                   -- the ADMISSIBLE statistic (C137 retired divergence)
  C  EGO           speed / d_ego, held-out val
  D  ENVIRONMENT   lead_gap_m / n_agents (in-sample; contrast is the readable part)

⭐ THE DECISIVE ROW IS B AT h>=2. Every arm to date is an IDENTITY MAP there
(H-PROOF-2, ratio 0.0002). E-DEC-7 predicts that is because the latent held
nothing worth predicting. If the latent now carries the scene and the predictor
STILL cannot move at h>=2, the predictor is a SEPARATE defect, not downstream of
the representation -- and that is the single most useful thing this run can say.

KILL-GATE (committed in advance): an arm is REJECTED if participation falls
below the rdw8 baseline on BOTH held-out sets, or cos-z drops below 2 at h=1,
however good its decodability looks.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
VAL = SP / "sp2/cache/physicalai-val-w120-256x640cyl"
HELD = SP / "sp2/cache/v7tiny-heldout24-w120-256x640cyl"
LEAD = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
LABELS = SP / "sp2/lead130_agents.jsonl"
OUT = SP / "e_dec9_10_full_panel.json"
ARMS = ["rdw8", "o7w1p0", "o8w1p0"]
F = 100


def env_t(ag, m):
    lead, n = [], []
    for i in range(m):
        a = ag.get(i, [])
        inl = [x["cx"] for x in a if abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0]
        lead.append(min(inl) if inl else 80.0)
        n.append(float(len(a)))
    return {"lead_gap_m": np.array(lead)[:, None], "n_agents": np.array(n)[:, None]}


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P
    from tanitad.models.v6 import spectrum_report
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    missing = [a for a in ARMS if a not in present]
    print(f"\n  FULL PANEL · arms {present}" + (f"  MISSING {missing}" if missing else "") + "\n",
          flush=True)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)", "eval_tier": "T0-DIAGNOSTIC",
           "arms_missing": missing, "arms": {}}

    LAB = {}
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])

    val = sorted(VAL.glob("*.v2ep.pt"))[:12]
    held = sorted(HELD.glob("*.v2ep.pt"))[:12]
    lead = [c for c in sorted(LEAD.glob("*.v2ep.pt"))[:24]
            if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]

    # ---- A rank + B predictor, per arm --------------------------------------
    print(f"  {'arm':<10}{'partic val':>11}{'held24':>9}"
          f"{'cos h1':>9}{'z h1':>7}{'cos h2':>9}{'z h2':>7}{'cos h4':>9}{'z h4':>7}")
    print("  " + "-" * 80, flush=True)
    for arm in present:
        w, st = G.load_arm(arm, dev)
        W = int(w.window)
        hs = sorted(int(h) for h in w.stack.cfg.predictor.horizons)

        def partic(cl):
            A = np.concatenate([G.encode_clip(w, c, dev, 120)[0].numpy() for c in cl])
            return float(spectrum_report(torch.from_numpy(A).float())["participation_ratio"])
        pv, ph = partic(val), partic(held)

        D = {h: [] for h in hs}
        T = {h: [] for h in hs}
        with torch.no_grad():
            for c in val[:6]:
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                for i in range(0, max(1, len(zt) - W - max(hs) - 1), 5):
                    zs = zt[i:i + W][None].to(dev)
                    aa = act[i:i + W][None].to(dev)
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    o = w.predictor(zs, torch.cat([aa, vv], -1))
                    zl = zt[i + W - 1].to(dev)
                    for h in hs:
                        j = i + W - 1 + h
                        if h in o and j < len(zt):
                            D[h].append((o[h].reshape(-1)[:zl.numel()] - zl).cpu().numpy())
                            T[h].append((zt[j].to(dev) - zl).cpu().numpy())
        row = {"participation_val": round(pv, 3), "participation_heldout24": round(ph, 3),
               "step": int(st), "predictor": {}}
        cells = []
        for h in hs:
            d = np.stack(D[h]).astype(np.float64)
            t = np.stack(T[h]).astype(np.float64)
            d -= d.mean(0, keepdims=True)
            t -= t.mean(0, keepdims=True)
            cos = float((d * t).sum()) / max(float(np.linalg.norm(d) * np.linalg.norm(t)), 1e-30)
            null = [float((d * t[rng.permutation(len(t))]).sum())
                    / max(float(np.linalg.norm(d) * np.linalg.norm(t)), 1e-30) for _ in range(200)]
            z_ = (cos - float(np.mean(null))) / max(float(np.std(null)), 1e-12)
            row["predictor"][str(h)] = {"cos": round(cos, 4), "z": round(z_, 2), "n": len(d)}
            cells += [f"{cos:>9.4f}", f"{z_:>7.2f}"]
        rep["arms"][arm] = row
        print(f"  {arm:<10}{pv:>11.2f}{ph:>9.2f}" + "".join(cells), flush=True)
        del w
        torch.cuda.empty_cache()

    # ---- C ego + D environment ----------------------------------------------
    for block, clips, held_out in (("EGO (held-out val)", val, True),
                                   ("ENV (in-sample)", lead, False)):
        COLS, PO, TG = {}, [], {}
        for arm in present:
            w, _ = G.load_arm(arm, dev)
            col = []
            for c in clips:
                z, _, _ = G.encode_clip(w, c, dev, F)
                col.append(z.numpy().astype(np.float64))
                if arm == present[0]:
                    d = torch.load(c, map_location="cpu", weights_only=False)
                    m = len(col[-1])
                    PO.append(d["poses"].numpy().astype(np.float64)[:m])
                    if not held_out:
                        for k, v in env_t(LAB[d["clip_id"]], m).items():
                            TG.setdefault(k, []).append(v)
            COLS[f"z_op {arm}"] = col
            del w
            torch.cuda.empty_cache()
        if held_out:
            TG = {"speed": [p[:, 3:4] for p in PO],
                  "d_ego": [np.concatenate([np.diff(p[:, :2], axis=0), np.zeros((1, 2))])
                            for p in PO]}
        COLS["pixel (floor)"] = [P.pooled_frames(c, len(p)) for c, p in zip(clips, PO)]
        COLS["frozen DINOv3"] = P.dinov3_encode(clips, F, dev)
        COLS["constant (control)"] = [np.ones((len(p), 1)) for p in PO]
        n = len(PO)

        def loeo(X, Y):
            o = []
            for e in range(n):
                idx = [i for i in range(n) if i != e]
                Xf = [np.asarray(X[i])[:len(Y[i])] for i in idx]
                Yf = [Y[i][:len(np.asarray(X[i]))] for i in idx]
                Xs = [np.asarray(X[e])[:len(Y[e])]]
                Ys = [Y[e][:len(np.asarray(X[e]))]]
                fn = getattr(P, "probe_fit_score", None)
                o.append(fn(Xf, Yf, Xs, Ys, 128) if fn else P.probe(Xf + Xs, Yf + Ys, 128)[0])
            return np.array(o, dtype=np.float64)

        print(f"\n  === {block} · {n} clips ===")
        print(f"  {'target':<12}{'column':<22}{'R2':>9}{'vs rdw8':>10}{'t':>7}{'favours':>9}",
              flush=True)
        rep.setdefault("blocks", {})[block] = {}
        for tn, Y in TG.items():
            R = {k: loeo(v, Y) for k, v in COLS.items()}
            base = R[f"z_op {present[0]}"]
            rep["blocks"][block][tn] = {}
            for k in COLS:
                dd = R[k] - base
                mm = float(dd.mean())
                se = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
                t = mm / max(se, 1e-12)
                rep["blocks"][block][tn][k] = {"r2": round(float(R[k].mean()), 4),
                                               "delta_vs_base": round(mm, 4), "t": round(t, 2),
                                               "n_favouring": int((dd > 0).sum()), "n": len(dd)}
                print(f"  {tn:<12}{k:<22}{float(R[k].mean()):>+9.4f}{mm:>+10.4f}{t:>7.2f}"
                      f"{int((dd > 0).sum()):>6}/{len(dd)}")
            print()

    # ---- the kill-gate, applied ---------------------------------------------
    base = rep["arms"].get(present[0], {})
    verdicts = {}
    for arm in present[1:]:
        r = rep["arms"][arm]
        rank_dead = (r["participation_val"] < base["participation_val"]
                     and r["participation_heldout24"] < base["participation_heldout24"])
        pred_dead = r["predictor"]["1"]["z"] < 2.0
        h2 = r["predictor"].get("2", {})
        verdicts[arm] = {
            "REJECTED_by_kill_gate": bool(rank_dead or pred_dead),
            "reason": ("rank fell on BOTH held-out sets" if rank_dead else
                       "predictor cos z<2 at h=1" if pred_dead else "passes the gate"),
            "predicts_beyond_h1": bool(h2.get("z", 0) > 2.0),
        }
    rep["kill_gate"] = verdicts
    rep["verdict"] = "; ".join(
        f"{a}: {'REJECTED — ' + v['reason'] if v['REJECTED_by_kill_gate'] else 'passes gate'}"
        f"{' · PREDICTS BEYOND h=1' if v['predicts_beyond_h1'] else ' · still identity at h>=2'}"
        for a, v in verdicts.items())
    print(f"  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
