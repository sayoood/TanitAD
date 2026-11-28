"""E-DEC-49 (H3) — DOES THE ACTION PREDICT THE **EGO'S OWN** FUTURE, AND DOES OUR
PREDICTION CAPTURE IT?

⭐ THE QUESTION E-DEC-48b LEFT OPEN, AND IT DECIDES WHETHER MANDATE 3 IS
REFRAMEABLE OR DEAD. E-DEC-48b measured that the action adds NOTHING to predicting
the FUTURE SCENE (marginal −0.1678, t −3.50, against a positive control at
t 12.58) — other traffic evolves independently of what we do. But it never tested
the one thing the action certainly DOES determine: **the ego's own future state.**

Two possible worlds, and they demand opposite responses:

  A. the action predicts Δspeed strongly, but our ẑ does NOT
     -> the world model is failing at THE ONE THING the action determines.
        FIXABLE — condition and score on the EGO subspace, not the whole latent.
  B. the action predicts Δspeed AND ẑ captures it
     -> action-conditioning already works where it CAN work, and the physics
        mandate was mis-specified rather than unmet.

⚠️ If the action does NOT predict Δspeed either, the corpus's action channel is
suspect and every action-conditioning claim in this campaign rests on a signal we
have never verified is real. **That is the check this panel makes possible and no
earlier panel did.**

TARGET: Δspeed(t → t+k) — the quantity the accel command most directly causes.

COLUMNS
    spd_t                  autocorrelation baseline — also the POSITIVE CONTROL
    action_t               [steer, accel] alone
    spd_t + action_t       ⭐ the marginal of the action GIVEN present speed
    z_t (ENCODED)          does the latent carry the ego's future at all?
    zhat_{t+k} (PREDICTED) ⭐⭐ does OUR WORLD MODEL capture it?
    constant               reads EXACTLY 0.0000

⭐ THE TWO READABLE QUANTITIES:
    (spd+action) − spd     the action's MARGINAL contribution to the EGO's future
    zhat − z_t             what the PREDICTION adds over the present latent

CONTROLS: constant at exactly 0.0000; TIME-SHUFFLED on every cell; NONLINEAR
RFF+ridge with clip-disjoint λ; n printed; verdict gated on the positive control,
per C159. T0-DIAGNOSTIC, held-out, lead-matched.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
LEAD = pathlib.Path(os.environ.get(
    "SPD_CORPUS", str(SP / "sp2/cache/physicalai-val130-heldout")))
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k,postrain10k,splitp30k").split(",")
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "egofuture.json")))
N_CLIPS, F, K = 20, 100, 4


def main() -> int:
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print("\n  E-DEC-49 (H3) - DOES THE ACTION PREDICT THE EGO'S OWN FUTURE?")
    print("  target = delta-speed(t -> t+k), the quantity the accel command causes\n",
          flush=True)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT", "k": K,
           "target": "delta speed(t -> t+k)", "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        W = int(w.window)
        SPD, ACT, ZT, ZH, Y = [], [], [], [], []
        with torch.no_grad():
            for c in clips:
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                a = act.float().numpy().astype(np.float64)
                v = spd.float().numpy().astype(np.float64).reshape(-1, 1)
                s_, a_, zt_, zh_, y_ = [], [], [], [], []
                for i in range(0, len(zt) - W - K, 1):
                    j = i + W - 1
                    tgt = j + K
                    if tgt >= len(v):
                        break
                    win = zt[i:i + W][None].to(dev).clone()
                    aa = act[i:i + W][None].to(dev).float()
                    if aa.shape[1] != W:
                        break
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    o = w.predictor(win, torch.cat([aa, vv], -1))[1]
                    zh_.append(o.reshape(-1)[:zt.shape[1]].cpu().numpy())
                    zt_.append(zt[j].numpy())
                    s_.append(v[j]); a_.append(a[j])
                    y_.append(v[tgt] - v[j])          # the EGO's own change
                if len(y_) < 25:
                    continue
                SPD.append(np.stack(s_).astype(np.float64))
                ACT.append(np.stack(a_).astype(np.float64))
                ZT.append(np.stack(zt_).astype(np.float64))
                ZH.append(np.stack(zh_).astype(np.float64))
                Y.append(np.stack(y_).astype(np.float64))
        del w
        torch.cuda.empty_cache()
        if len(Y) < 8:
            continue

        COL = {"spd_t (POSITIVE CONTROL)": SPD, "action_t": ACT,
               "spd_t + action_t": [np.concatenate([s, a], 1) for s, a in zip(SPD, ACT)],
               "z_t (ENCODED)": ZT, "zhat_t+k (PREDICTED)": ZH,
               "constant (control)": [np.ones((len(y), 1)) for y in Y]}
        rng = np.random.default_rng(0)
        Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
        res = {}
        for cn, X in COL.items():
            tr, sh = [], []
            for i in range(len(X)):
                Xtr = [X[q] for q in range(len(X)) if q != i]
                for Yv, sink in ((Y, tr), (Ysh, sh)):
                    ytr = [Yv[q] for q in range(len(Yv)) if q != i]
                    pred, _ = rff_fold(Xtr, ytr, X[i])
                    sink.append(within_clip_r(pred, Yv[i].ravel()))
            res[cn] = (np.array(tr), np.array(sh))

        nrow = sum(len(y) for y in Y)
        print(f"  === {arm} (step {st}) - {len(Y)} clips, {nrow} rows ===")
        print(f"  {'column':<26}{'r':>9}{'shuf':>9}{'t-shuf':>9}{'t':>7}")
        rep["arms"][arm] = {"step": int(st), "n_clips": len(Y), "n_rows": nrow,
                            "columns": {}}
        for cn, (tr, sh) in res.items():
            dd = tr - sh
            t = float(dd.mean()) / max(float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            rep["arms"][arm]["columns"][cn] = {
                "r": round(float(tr.mean()), 4), "shuffled": round(float(sh.mean()), 4),
                "true_minus_shuffled": round(float(dd.mean()), 4), "t": round(t, 2)}
            print(f"  {cn:<26}{tr.mean():>+9.4f}{sh.mean():>+9.4f}"
                  f"{dd.mean():>+9.4f}{t:>7.2f}", flush=True)

        sp_ = res["spd_t (POSITIVE CONTROL)"][0]
        sa_ = res["spd_t + action_t"][0]
        z_ = res["z_t (ENCODED)"][0]
        zh = res["zhat_t+k (PREDICTED)"][0]
        c_ = res["constant (control)"][0]
        def tt(d):
            return float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)
        t_ctrl = tt(sp_ - c_)
        t_marg = tt(sa_ - sp_)
        t_pred = tt(zh - z_)
        rep["arms"][arm]["positive_control_t"] = round(t_ctrl, 2)
        rep["arms"][arm]["action_marginal_given_speed"] = {
            "delta": round(float((sa_ - sp_).mean()), 4), "t": round(t_marg, 2)}
        rep["arms"][arm]["prediction_adds_over_z_t"] = {
            "delta": round(float((zh - z_).mean()), 4), "t": round(t_pred, 2)}
        print()
        if t_ctrl < 2.0:
            v = (f"NO VERDICT - the positive control (spd_t) reads t {t_ctrl:+.2f}; "
                 f"this panel cannot answer")
        elif t_marg > 2.0 and t_pred < 2.0:
            v = ("WORLD A - the ACTION predicts the ego's future but OUR PREDICTION "
                 "DOES NOT. Failing at the one thing the action determines. FIXABLE: "
                 "condition and score on the EGO subspace.")
        elif t_marg > 2.0:
            v = ("WORLD B - the action predicts the ego's future AND the prediction "
                 "captures it. Action-conditioning works where it CAN work.")
        else:
            v = ("⚠️ THE ACTION DOES NOT PREDICT THE EGO'S OWN FUTURE EITHER - the "
                 "corpus action channel is suspect and every action-conditioning "
                 "claim in this campaign rests on an unverified signal.")
        rep["arms"][arm]["verdict"] = v
        print(f"  ctrl t {t_ctrl:+.2f} | action marginal t {t_marg:+.2f} | "
              f"prediction adds t {t_pred:+.2f}")
        print(f"  => {v}\n", flush=True)

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
