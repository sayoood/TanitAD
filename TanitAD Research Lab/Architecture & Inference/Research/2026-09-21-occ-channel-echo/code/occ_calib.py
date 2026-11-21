"""Does the occ channel carry INFORMATION, or only CALIBRATION, over the head's own box?

`occ_echo.py` established two things and left one confound standing.

ESTABLISHED. The target is an EXACT geometric identity: `occ == |atan2(cy, cx)| > 60.0 deg` on the
GT box centre, agreement **1.0000 on the fit half AND 1.0000 on the scored half**, half-angle swept
20-90 deg and chosen fit-side only. That is the docstring's claim (`agent_slots.py:82-87`)
reproduced independently from the data.

THE CONFOUND. ARM H applied that predicate to the head's OWN predicted centre as a HARD 0/1 at a
clip of 0.001, and lost to the head's `occ_logit` (0.449 vs 0.288, CI [0.017, 0.362]). But a hard
predicate is punished at `-log(0.001) = 6.9` for every error it makes, while a soft logit HEDGES
near the boundary. ⇒ **O beating H is exactly what you would see if the occ channel were a
perfectly CALIBRATED re-read of the head's own box and carried no extra information at all.** A
calibration win reported as an information win is the nav-echo failure with a softmax on top.

⭐ THE DISCRIMINATOR: give the box arm the same right to hedge, then ask again.

  H1  hard predicate on the head's own azimuth          (as before — the confounded arm)
  H2  LOGISTIC on the head's own azimuth                 1 feature: |atan2(cy_pred, cx_pred)|
  H3  LOGISTIC on the head's own azimuth + range + cx,cy so the box arm gets the WHOLE box
  O   the head's `occ_logit`

  O beats H3  -> the channel carries information NO soft read of the head's own box supplies. REAL.
  O ~= H2/H3  -> the channel is a CALIBRATED ECHO of the box. Redundant; say so.

⛔ Every logistic is fit on the FIT episodes only and scored on the disjoint half, and the
BASE-RATE constant remains the no-information control. n, d and the episode counts are printed.
⚠️ The function class is stated: H2/H3 are LOGISTIC. Beating them is evidence against a LINEAR soft
read of the box, not against every possible one.
⛔ CPU only. Rows are cached to `occ_rows.npz` so further arms cost no forward pass.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import random
import statistics as st
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))

CACHE = pathlib.Path("occ_rows.npz")
A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 60
B = 4000
SEED = 20260921
EPS = 1e-6
TH = math.radians(60.0)


def ll(p, y):
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def build():
    import torch
    import s1_pass as SP                                  # noqa: E402
    from tanitad.models.agent_slots import match_slots    # noqa: E402
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1500)
           if corp.eligibility(w) is None][:N_WIN]
    R = []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        sl = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
              if torch.is_tensor(v) and v.dim() >= 1}
        m = match_slots({k: v[None] for k, v in sl.items()},
                        {"box": item["agent_box"][None].float(),
                         "valid": item["agent_valid"][None], "cls": item["agent_cls"][None]})
        r, c = m["rows"][0].tolist(), m["cols"][0].tolist()
        occ = item["agent_occ"].float().numpy()
        tb = item["agent_box"].float().numpy()
        pb = sl["box"].numpy()
        ph = torch.sigmoid(sl["occ_logit"]).numpy()
        for i, j in zip(r, c):
            if occ[j] < 0.0:
                continue
            R.append([float(e_i), float(occ[j] > 0.5), float(ph[i]),
                      abs(math.atan2(float(tb[j, 1]), float(tb[j, 0]))),
                      abs(math.atan2(float(pb[i, 1]), float(pb[i, 0]))),
                      float(np.hypot(pb[i, 0], pb[i, 1])),
                      float(pb[i, 0]), float(pb[i, 1])])
    A = np.asarray(R, dtype=np.float64)
    np.savez_compressed(CACHE, rows=A)
    return A


def logistic(Xf, yf, Xs, iters=400, lam=1e-3):
    """Plain IRLS-free gradient fit — small d, so a fixed-step Newton is fine and deterministic."""
    d = Xf.shape[1]
    b = np.zeros(d)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xf @ b, -30, 30)))
        W = np.clip(p * (1.0 - p), 1e-6, None)
        H = Xf.T @ (Xf * W[:, None]) + lam * np.eye(d)
        g = Xf.T @ (yf - p) - lam * b
        step = np.linalg.solve(H, g)
        b = b + step
        if float(np.abs(step).max()) < 1e-9:
            break
    return 1.0 / (1.0 + np.exp(-np.clip(Xs @ b, -30, 30))), b


def main() -> int:
    A = np.load(CACHE)["rows"] if CACHE.exists() else build()
    epi, y, p_head, az_gt, az_pr, rng_pr, cx_pr, cy_pr = (A[:, k] for k in range(8))
    eps_ = sorted(set(epi.tolist()))
    half = set(eps_[: len(eps_) // 2])
    fit = np.array([e in half for e in epi])
    sc = ~fit
    assert fit.sum() > 20 and sc.sum() > 20, "ZZABORT a split side is too small"

    base = float(y[fit].mean())
    clip = 0.001
    hard = np.where(az_pr > TH, 1.0 - clip, clip)

    X2f = np.stack([az_pr[fit], np.ones(int(fit.sum()))], 1)
    X2s = np.stack([az_pr[sc], np.ones(int(sc.sum()))], 1)
    p_h2, b2 = logistic(X2f, y[fit], X2s)
    X3f = np.stack([az_pr[fit], rng_pr[fit], cx_pr[fit], cy_pr[fit],
                    np.ones(int(fit.sum()))], 1)
    X3s = np.stack([az_pr[sc], rng_pr[sc], cx_pr[sc], cy_pr[sc],
                    np.ones(int(sc.sum()))], 1)
    p_h3, b3 = logistic(X3f, y[fit], X3s)

    ys = y[sc]
    arms = {"BASE_RATE_control": ll(np.full(ys.shape, base), ys),
            "H1_hard_predicate_own_box": ll(hard[sc], ys),
            "H2_logistic_own_azimuth": ll(p_h2, ys),
            "H3_logistic_whole_own_box": ll(p_h3, ys),
            "O_head_occ_logit": ll(p_head[sc], ys)}
    es = epi[sc]
    rgen = random.Random(SEED)
    keys = sorted(set(es.tolist()))
    idx = {k: np.where(es == k)[0] for k in keys}

    def paired(a, b_):
        d = arms[a] - arms[b_]
        ms = sorted(float(np.concatenate([d[idx[k]] for k in rgen.choices(keys, k=len(keys))]).mean())
                    for _ in range(B))
        return [round(ms[int(0.025 * B)], 5), round(ms[int(0.975 * B)], 5)]

    res = {"_what": "does the occ channel carry INFORMATION, or only CALIBRATION, over its own box?",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000",
           "_function_class": "H2/H3 are LOGISTIC; beating them is evidence against a LINEAR soft "
                              "read of the head's own box, not against every possible read",
           "n_supervised_pairs": int(A.shape[0]), "n_episodes": len(eps_),
           "n_scored_pairs": int(sc.sum()), "n_scored_episodes": len(keys),
           "base_rate_FIT": round(base, 5),
           "logloss": {k: round(float(v.mean()), 5) for k, v in arms.items()},
           "H2_coefficients": {"azimuth": round(float(b2[0]), 4),
                               "bias": round(float(b2[1]), 4)},
           "d_H3": int(X3f.shape[1])}
    for a in ("H1_hard_predicate_own_box", "H2_logistic_own_azimuth",
              "H3_logistic_whole_own_box"):
        ci = paired(a, "O_head_occ_logit")
        res.setdefault("O_vs", {})[a] = {
            "gain_logloss_arm_minus_O": round(float((arms[a] - arms["O_head_occ_logit"]).mean()), 5),
            "CI95": ci, "separated": bool(ci[0] > 0 or ci[1] < 0)}

    h3 = res["O_vs"]["H3_logistic_whole_own_box"]
    res["_VERDICT"] = (
        "⭐ INFORMATION — the occ channel beats the best LINEAR SOFT read of the head's own whole "
        "box, so the win is not calibration alone" if h3["separated"] and h3["gain_logloss_arm_minus_O"] > 0
        else "⛔ the occ channel LOSES to a soft read of the head's own box"
        if h3["separated"] else
        "⚠️ CALIBRATION, NOT INFORMATION — once the box arm is allowed to hedge, the occ channel is "
        "statistically indistinguishable from it ⇒ a CALIBRATED ECHO, and the earlier win over the "
        "HARD predicate was about the 0/1 form, not about what the head knows")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("occ_calib.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
