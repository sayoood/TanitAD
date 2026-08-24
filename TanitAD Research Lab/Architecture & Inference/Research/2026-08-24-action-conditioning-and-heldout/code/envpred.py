"""E-DEC-28 — DOES THE **PREDICTED** LATENT CARRY THE ENVIRONMENT? (PI mandate)

⛔ THE GAP THIS CLOSES. Every environment number in this campaign — `n_agents`,
`lead_range_m`, the whole C150/E-DEC-24/25 line — was probed from the **ENCODED**
latent z_t. The predictor was only ever scored on *latent-space accuracy* (cos,
nrmse): does ẑ land near z. **Nobody has ever asked whether ẑ CARRIES THE SCENE.**
So "environment learning in the predictor" is, to date, unmeasured.

THE PI'S FRAMING (2026-08-24): not driving capability — that is the trajectory
work — but whether the REPRESENTATION encodes the causal structure of driving:
"if the vehicle in front decelerates, the ego must react on it". That is a T0
question about what the latent and its prediction contain.

WHAT IS PROBED. Roll the h=1 head k times from the observed window to get
ẑ_{t+k}, then probe it against the environment AT TIME t+k:

    n_agents(t+k)          how many agents are in the scene then
    lead_range_m(t+k)      how far the lead is then      (lead-present only, C150)
    lead_closing(t+k)      d(range)/dt — THE DECELERATION SIGNAL the PI named;
                           positive = the gap is opening, negative = closing

⭐⭐ THE CONTROL THAT DECIDES IT, AND WITHOUT WHICH THE RESULT IS MEANINGLESS.
The environment at t+k is heavily autocorrelated with t: a probe on the ENCODED
z_t already predicts much of it *without any prediction at all*. So the panel
carries, on the identical rows:

    z_t          (ENCODED, present)   — the "no prediction" baseline. **ẑ_{t+k}
                                        must BEAT this or the predictor added
                                        nothing.** This is the whole test.
    zhat_{t+k}   (PREDICTED)          — the quantity of interest
    z_{t+k}      (ENCODED, future)    — the CEILING: what the encoder carries at
                                        that time when it is allowed to look
    constant                          — reads exactly 0.0000 by construction
    pixels_t     (raw input at t)     — the raw-input floor (standing rule)

A `zhat` that beats `constant` proves nothing; a `zhat` that beats `z_t` proves
the predictor propagates scene structure forward.

⚠️ SCOPE. T0-DIAGNOSTIC. This says what the representation CONTAINS, never that a
planner would use it. And it is IN-SAMPLE until the held-out corpus lands.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
# ⭐ E-DEC-31: corpus/labels/out are now ENV-OVERRIDABLE so the HELD-OUT read
# is a RE-POINT of the identical instrument, not a forked copy that could
# drift from it. Defaults reproduce the in-sample run byte-for-byte.
LEAD = pathlib.Path(os.environ.get("SPD_CORPUS",
    str(SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl")))
LABELS = pathlib.Path(os.environ.get("SPD_LABELS",
    str(SP / "sp2/lead130_agents.jsonl")))
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "envpred.json")))
SPLIT = os.environ.get("SPD_SPLIT", "IN-SAMPLE")
ARMS = ["splitp30k", "rdw8p30k"]
N_CLIPS, F = 24, 100
KS = (1, 3, 6)


def scene(ag, m):
    """per-frame n_agents, lead range (nan when no lead), and closing rate.

    ⛔ `n_agents` is now NaN on an UNLABELLED frame, not 0. The lead targets were
    already nan-masked; `n_agents` was not, so an unlabelled frame entered the
    panel as a confident "zero agents in view". Harmless at 0.00 % coverage loss
    (in-sample) and decisive at 4.90 % (held-out). Same defect as C150.
    """
    n, rng_ = [], []
    for i in range(m):
        if i not in ag:
            n.append(np.nan); rng_.append(np.nan); continue
        a = ag.get(i, [])
        inl = [x["cx"] for x in a if abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0]
        n.append(float(len(a)))
        rng_.append(min(inl) if inl else np.nan)
    r = np.array(rng_)
    clo = np.full(m, np.nan)
    clo[1:] = r[1:] - r[:-1]          # + opening, - closing (the PI's decel signal)
    return np.array(n), r, clo


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    LAB = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])
    # ⛔⛔ THE SAME SELECTION MATCHING AS spatialenv.py, AND HERE IT FIXES TWO
    # THINGS AT ONCE. (1) C153: the in-sample corpus is `slotprobe-LEAD130`, chosen
    # for lead presence (130/130 clips vs 70/122 held-out), and comparing absolute
    # R2 across that gap reversed a verdict — splitp30k n_agents read -0.3515
    # unmatched and +0.1220 MATCHED, i.e. ABOVE frozen DINOv3, exactly as
    # in-sample. (2) POWER: unmatched, only 6 of 23 held-out clips carried >=20
    # lead-present rows, so `lead_range_m` and `lead_closing` — THE targets for
    # the PI's "the vehicle in front decelerates" question — were UNMEASURABLE and
    # were reported as absent rather than negative. Matching makes them powered.
    # Default 0 reproduces the unmatched run exactly.
    MIN_LEAD = int(os.environ.get("SPD_MIN_LEAD", "0"))

    def _lead_frames(cid):
        ag = LAB.get(cid, {})
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0
                          for x in ag.get(i, [])))

    _all = [c for c in sorted(LEAD.glob("*.v2ep.pt"))
            if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]
    if MIN_LEAD > 0:
        _kept = [c for c in _all
                 if _lead_frames(torch.load(c, map_location="cpu",
                                            weights_only=False)["clip_id"]) >= MIN_LEAD]
        print(f"  LEAD-MATCHED SELECTION: {len(_kept)}/{len(_all)} clips carry "
              f">= {MIN_LEAD} lead-present frames", flush=True)
        _all = _kept
    clips = _all[:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-28 · does the PREDICTED latent carry the environment?"
          f"\n  arms {present} · {len(clips)} clips · rolled k = {KS}\n", flush=True)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "min_lead_frames": int(os.environ.get("SPD_MIN_LEAD", "0")), "split": SPLIT,
           "corpus": str(LEAD), "labels": str(LABELS),
           "method": "h=1 head rolled k times -> zhat_{t+k}; probed against the "
                     "environment AT t+k. The decisive control is z_t (ENCODED, "
                     "present): zhat must BEAT it or the predictor added nothing.",
           "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        W = int(w.window)
        rep["arms"][arm] = {"step": int(st), "k": {}}
        for K in KS:
            COL = {"z_t (ENCODED present)": [], "zhat_t+k (PREDICTED)": [],
                   "z_t+k (ENCODED future, CEILING)": [], "pixels_t (floor)": []}
            TG = {"n_agents": [], "lead_range_m": [], "lead_closing": []}
            MK = {"n_agents": [], "lead_range_m": [], "lead_closing": []}
            for c in clips:
                d, raw, off, n_all, _ = G.frames_of(c)
                n = min(n_all, F)
                imgs = [torch.from_numpy(np.asarray(
                    Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")).copy())
                    .permute(2, 0, 1).float() / 255.0 for i in range(n)]
                if float(imgs[0].abs().mean()) == 0.0:
                    raise SystemExit(f"[FATAL] {c.name} all-zero frames")
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                nag, rr, clo = scene(LAB[d["clip_id"]], n)
                zc, zh, zf, px = [], [], [], []
                ya, yr, yc, ma, mr, mc = [], [], [], [], [], []
                with torch.no_grad():
                    # stride 1, not 2: the smoke at 10 clips gave ~225 fit rows
                    # against d_eff 128 and the CEILING column read BELOW the
                    # baseline, which is a small-n artefact, not a result. Stride 1
                    # over 24 clips gives ~2,100 rows (~17:1) and the ceiling can
                    # then be read as a ceiling.
                    for i in range(0, len(zt) - W - K, 1):
                        j = i + W - 1                 # last OBSERVED frame index
                        tgt = j + K                   # the predicted time
                        if tgt >= n:
                            break
                        win = zt[i:i + W][None].to(dev).clone()
                        vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                        for s in range(K):
                            aa = act[i + s:i + s + W][None].to(dev)
                            if aa.shape[1] != W:
                                break
                            o = w.predictor(win, torch.cat([aa, vv], -1))[1]
                            o = o.reshape(1, -1)[:, :zt.shape[1]]
                            win = torch.cat([win[:, 1:], o[None]], dim=1)
                        zc.append(zt[j].numpy()); zh.append(o.reshape(-1).cpu().numpy())
                        zf.append(zt[tgt].numpy())
                        px.append(torch.nn.functional.adaptive_avg_pool2d(
                            imgs[j][None, -3:], (8, 8)).reshape(-1).numpy())
                        ya.append(nag[tgt]); yr.append(rr[tgt]); yc.append(clo[tgt])
                        ma.append(not np.isnan(nag[tgt]))
                        mr.append(not np.isnan(rr[tgt])); mc.append(not np.isnan(clo[tgt]))
                if len(ya) < 20:
                    continue
                COL["z_t (ENCODED present)"].append(np.stack(zc).astype(np.float64))
                COL["zhat_t+k (PREDICTED)"].append(np.stack(zh).astype(np.float64))
                COL["z_t+k (ENCODED future, CEILING)"].append(np.stack(zf).astype(np.float64))
                COL["pixels_t (floor)"].append(np.stack(px).astype(np.float64))
                TG["n_agents"].append(np.nan_to_num(np.array(ya))[:, None])
                MK["n_agents"].append(np.array(ma))
                TG["lead_range_m"].append(np.nan_to_num(np.array(yr))[:, None])
                TG["lead_closing"].append(np.nan_to_num(np.array(yc))[:, None])
                MK["lead_range_m"].append(np.array(mr)); MK["lead_closing"].append(np.array(mc))
            COL["constant (control)"] = [np.ones((len(y), 1)) for y in TG["n_agents"]]

            def loeo(X, Y):
                o = []
                for i in range(len(X)):
                    Xf = [X[j] for j in range(len(X)) if j != i]
                    Yf = [Y[j] for j in range(len(Y)) if j != i]
                    o.append(P.probe(Xf + [X[i]], Yf + [Y[i]], 128)[0])
                return np.array(o, dtype=np.float64)

            print(f"  === {arm} · rolled k={K} · {len(TG['n_agents'])} clips ===")
            print(f"  {'target':<16}{'column':<34}{'R2':>9}{'vs z_t':>10}{'t':>8}{'favours':>9}")
            rep["arms"][arm]["k"][str(K)] = {}
            for tn, Y in TG.items():
                mk = MK.get(tn)
                if mk is not None:
                    keep = [i for i in range(len(Y)) if int(mk[i].sum()) >= 20]
                    if len(keep) < 6:
                        print(f"  {tn:<16}SKIPPED — {len(keep)} clips with >=20 rows"); continue
                    Yl = [Y[i][mk[i]] for i in keep]
                    Cl = {k: [v[i][mk[i]] for i in keep] for k, v in COL.items()}
                else:
                    keep, Yl, Cl = list(range(len(Y))), Y, COL
                nrow = sum(len(y) for y in Yl)
                R = {k: loeo(v, Yl) for k, v in Cl.items()}
                base = R["z_t (ENCODED present)"]
                rep["arms"][arm]["k"][str(K)][tn] = {"n_clips": len(keep), "n_rows": nrow,
                                                     "columns": {}}
                for k in Cl:
                    dd = R[k] - base
                    mm = float(dd.mean())
                    se = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
                    t = mm / max(se, 1e-12)
                    rep["arms"][arm]["k"][str(K)][tn]["columns"][k] = {
                        "r2": round(float(R[k].mean()), 4), "delta_vs_z_t": round(mm, 4),
                        "t": round(t, 2), "n_favouring": int((dd > 0).sum()), "n": len(dd),
                        "beats_no_prediction": bool(t > 2.0)}
                    print(f"  {tn:<16}{k:<34}{float(R[k].mean()):>+9.4f}{mm:>+10.4f}"
                          f"{t:>8.2f}{int((dd > 0).sum()):>6}/{len(dd)}")
                print()
        del w
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
