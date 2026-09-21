"""Is the head's `occluded` skill REAL, or is it an ECHO of the box it already emits?

`occ_supervision.py` measured the head beating its base rate on `occluded` by a wide margin
(logloss 0.288 vs 0.675, CI [0.352, 0.431], 848 pairs / 18 episodes, coverage 1482/1482). Taken
alone that reads as the head's strongest demonstrated capability — at the field with the SECOND
LOWEST alignment.

⛔ BUT THE DOCSTRING SAYS THE TARGET IS A GEOMETRIC FUNCTION. `agent_slots.py:82-87`: `occ` IS
`bev_raster.fov_mask`'s predicate, and that predicate is (`bev_raster.py:355`)
`|cell_azimuth| <= half_angle`. On a box centre that is `|atan2(cy, cx)| <= th`. ⇒ **a head that
predicts (cx, cy) well gets `occluded` FOR FREE**, and scoring the channel on its own would be the
flagship-v1 route head all over again: an exact bijection of an input it already has, scored 1.0000
and read as skill (CLAUDE.md, the nav-echo defect).

⭐ THE DISCRIMINATING CONTROL, and it is the whole point of this file:

  ARM G  — the predicate applied to the **GT** box centre.  Establishes the identity: if this
           reproduces `occ`, the target IS geometry and nothing else.
  ARM H  — the predicate applied to the **HEAD'S OWN PREDICTED** box centre. This is what the head
           could compute from what it already emits, with NO occ channel at all.
  ARM O  — the head's actual `occ_logit`.

  O beats H  -> the occ channel carries information the head's own box does NOT express. REAL.
  O ~= H     -> the channel is REDUNDANT — an echo of the box, and the head-change SPEC should say
                so rather than bank it as a capability.
  O worse    -> worse than a free geometric read of its own output: a defect, not a skill.

⚠️ The half-angle is MEASURED, not assumed: swept over a grid and reported at its best agreement
with the GT flag, so ARM G is the strongest form of the identity rather than a guess at 60 deg.
⚠️ A hard 0/1 predicate cannot be scored by log-loss without clipping, so both predicate arms are
scored at a shared clip and the clip is printed — a hyper-parameter, so it is chosen on the FIT
split only, exactly like lambda elsewhere in this directory.
⛔ CPU only, read-only, no GPU.
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
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402
from tanitad.models.agent_slots import match_slots        # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 60
B = 4000
SEED = 20260921
EPS = 1e-6


def ll(p: float, y: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1500)
           if corp.eligibility(w) is None][:N_WIN]

    rows = []           # (episode, y, p_head, az_gt, az_pred)
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
        if not r:
            continue
        occ = item["agent_occ"].float().numpy()
        tb = item["agent_box"].float().numpy()
        pb = sl["box"].numpy()
        ph = torch.sigmoid(sl["occ_logit"]).numpy()
        for i, j in zip(r, c):
            if occ[j] < 0.0:
                continue
            rows.append((int(e_i), float(occ[j] > 0.5), float(ph[i]),
                         abs(math.atan2(float(tb[j, 1]), float(tb[j, 0]))),
                         abs(math.atan2(float(pb[i, 1]), float(pb[i, 0])))))
    assert len(rows) > 100, f"ZZABORT too few supervised pairs: {len(rows)}"

    eps_ = sorted({r[0] for r in rows})
    half = set(eps_[: len(eps_) // 2])
    fit = [r for r in rows if r[0] in half]
    sco = [r for r in rows if r[0] not in half]
    assert fit and sco, "ZZABORT a split side is empty"

    # ---- ARM G: the identity. Half-angle swept, chosen on the FIT split only.
    best_th, best_agree = None, -1.0
    for deg in np.arange(20.0, 90.01, 0.25):
        th = math.radians(float(deg))
        a = st.mean(float((r[3] > th) == (r[1] > 0.5)) for r in fit)
        if a > best_agree:
            best_th, best_agree = float(deg), a
    th_r = math.radians(best_th)
    agree_sc = st.mean(float((r[3] > th_r) == (r[1] > 0.5)) for r in sco)

    # ---- the clip for the hard predicates, also FIT-side only
    best_clip, best_m = None, float("inf")
    for cl in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        v = st.mean(ll(1.0 - cl if r[3] > th_r else cl, r[1]) for r in fit)
        if v < best_m:
            best_clip, best_m = cl, v

    def pack(key):
        d = collections.defaultdict(list)
        for r in sco:
            d[r[0]].append(key(r))
        return d

    base = sum(r[1] for r in fit) / len(fit)
    arms = {
        "BASE_RATE_control": pack(lambda r: ll(base, r[1])),
        "G_predicate_on_GT_box": pack(lambda r: ll(1.0 - best_clip if r[3] > th_r
                                                   else best_clip, r[1])),
        "H_predicate_on_HEAD_box": pack(lambda r: ll(1.0 - best_clip if r[4] > th_r
                                                     else best_clip, r[1])),
        "O_head_occ_logit": pack(lambda r: ll(r[2], r[1])),
    }
    res = {"_what": "is the head's `occluded` skill real, or an echo of the box it already emits?",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000",
           "_predicate": "occ == |atan2(cy, cx)| > half_angle   (bev_raster.py:355 on a box centre)",
           "n_supervised_pairs": len(rows), "n_episodes": len(eps_),
           "n_scored_pairs": len(sco), "n_scored_episodes": len({r[0] for r in sco}),
           "half_angle_deg_chosen_on_FIT": round(best_th, 3),
           "identity_agreement_FIT": round(best_agree, 5),
           "identity_agreement_SCORED": round(agree_sc, 5),
           "clip_chosen_on_FIT": best_clip,
           "logloss": {k: round(st.mean(x for v in d.values() for x in v), 5)
                       for k, d in arms.items()}}

    # ---- O vs H, paired by episode: the question the file exists to answer
    rng = random.Random(SEED)
    hd, od = arms["H_predicate_on_HEAD_box"], arms["O_head_occ_logit"]
    keys = sorted(set(hd) & set(od))
    ms = sorted(
        st.mean(h - o for k in rng.choices(keys, k=len(keys))
                for h, o in zip(hd[k], od[k]))
        for _ in range(B)) if len(keys) > 1 else None
    ci = [round(ms[int(0.025 * B)], 5), round(ms[int(0.975 * B)], 5)] if ms else None
    gain = res["logloss"]["H_predicate_on_HEAD_box"] - res["logloss"]["O_head_occ_logit"]
    res["O_minus_H"] = {"gain_logloss_H_minus_O": round(gain, 5), "CI95": ci,
                        "separated": bool(ci and (ci[0] > 0 or ci[1] < 0))}
    sep = res["O_minus_H"]["separated"]
    res["_VERDICT"] = (
        "⭐ REAL — the occ channel beats a free geometric read of the head's OWN box, so it carries "
        "information the box does not express" if sep and gain > 0 else
        "⛔ WORSE THAN FREE — the head's occ channel loses to the predicate applied to its own "
        "predicted box, i.e. the channel degrades information the head already had"
        if sep and gain < 0 else
        "⚠️ REDUNDANT — the occ channel is statistically indistinguishable from the predicate on "
        "the head's own box ⇒ an ECHO, not an added capability; report it as such")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("occ_echo.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
