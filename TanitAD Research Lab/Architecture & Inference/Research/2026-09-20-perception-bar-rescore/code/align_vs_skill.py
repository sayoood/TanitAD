"""Does ALIGNMENT predict ACCURACY? A test of my own statistic, not of the head.

`148ceb9` measured an alignment spectrum spanning 0.025 → 56.7 across one layer's fields and
flagged a tension: `v_rel_y` sits at **0.088** (apparently near-dead) while `box_quality` reports
`vel_gain_mps` **2.8738** beating a zero-velocity floor (`cb274b0`). If a near-zero-alignment field
still predicts well, then ALIGNMENT IS A WEAK DIAGNOSTIC and the spectrum means less than it looks.

⭐ VELOCITY IS THE CLEAN PLACE TO TEST IT, for a reason that matters: the Hungarian cost uses
centre, cls and presence (`agent_slots.py:475-477`) — it does NOT use velocity. So scoring the two
velocity components on matcher-assigned pairs is NOT circular, whereas scoring CENTRE that way
partly is. `v_rel_x` (alignment 7.258) and `v_rel_y` (0.088) are 82× apart and scored identically.

⛔ THE FLOOR IS THE CONTROL AND IT IS PER COMPONENT: a ZERO predictor for that component. A head
that cannot beat "predict no motion" has added nothing — the rule that caught four estimator bugs
on 2026-08-22.
⚠️ Centre is reported too, with its circularity flagged, because the prereg deliberately scores
through the TRAINING matcher (a 2 m greedy matcher "would pair only lucky hits and flatter the
head") — but the VERDICT rests on velocity.
⛔ CPU only.
"""
from __future__ import annotations

import collections
import json
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
N_WIN = 30
B = 4000
SEED = 20260921
ALIGN = {"v_rel_x": 7.258, "v_rel_y": 0.08807, "cx": 24.77866, "cy": 56.7131}


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:N_WIN]

    acc = {k: collections.defaultdict(list) for k in ("v_rel_x", "v_rel_y", "cx", "cy")}
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        sl = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
              if torch.is_tensor(v) and v.dim() >= 2}
        m = match_slots({k: v[None] for k, v in sl.items()},
                        {"box": item["agent_box"][None].float(),
                         "valid": item["agent_valid"][None], "cls": item["agent_cls"][None]})
        r, c = m["rows"][0].tolist(), m["cols"][0].tolist()
        if not r:
            continue
        tb = item["agent_box"].float()
        rates = item["agent_rates"].float()
        rmask = item["agent_rates_mask"].numpy().astype(bool)
        pr = sl.get("rates")
        for i, j in zip(r, c):
            acc["cx"][e_i].append((abs(float(sl["box"][i, 0] - tb[j, 0])),
                                   abs(float(tb[j, 0]))))
            acc["cy"][e_i].append((abs(float(sl["box"][i, 1] - tb[j, 1])),
                                   abs(float(tb[j, 1]))))
            ok = bool(rmask[j][0]) if rmask.ndim > 1 else bool(rmask[j])
            if pr is not None and ok:
                acc["v_rel_x"][e_i].append((abs(float(pr[i, 0] - rates[j, 0])),
                                            abs(float(rates[j, 0]))))
                acc["v_rel_y"][e_i].append((abs(float(pr[i, 1] - rates[j, 1])),
                                            abs(float(rates[j, 1]))))

    rng = random.Random(SEED)
    res = {"_what": "does alignment predict accuracy? skill vs a ZERO floor, per component",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000",
           "_non_circular": "velocity is NOT in the Hungarian cost; centre partly is — flagged",
           "fields": {}}
    for f, by in acc.items():
        by = {k: v for k, v in by.items() if v}
        if not by:
            continue
        head = st.mean(a for v in by.values() for a, _ in v)
        floor = st.mean(b for v in by.values() for _, b in v)
        skill = 1.0 - head / floor if floor else float("nan")
        keys = list(by)
        ms = sorted(
            (lambda d: 1.0 - st.mean(a for a, _ in d) / st.mean(b for _, b in d)
             if st.mean(b for _, b in d) else 0.0)(
                [x for k in rng.choices(keys, k=len(keys)) for x in by[k]])
            for _ in range(B)) if len(keys) > 1 else None
        res["fields"][f] = {
            "alignment": ALIGN[f], "n_pairs": sum(len(v) for v in by.values()),
            "n_episodes": len(keys),
            "MAE_head": round(head, 4), "MAE_zero_floor": round(floor, 4),
            "skill_vs_floor": round(skill, 4),
            "CI95": [round(ms[int(0.025 * B)], 4), round(ms[int(0.975 * B)], 4)] if ms else None,
            "beats_floor": bool(ms and ms[int(0.025 * B)] > 0)}

    vx, vy = res["fields"].get("v_rel_x"), res["fields"].get("v_rel_y")
    if vx and vy:
        res["_VERDICT"] = (
            "⛔ ALIGNMENT IS A WEAK DIAGNOSTIC — the 0.088 field beats its floor too, so low "
            "alignment does NOT imply a dead field" if vy["beats_floor"] else
            "⭐ alignment TRACKS skill — the 82x-lower field fails its floor while its sibling "
            "passes, so the spectrum is diagnostic")
    print(json.dumps(res, indent=1))
    print("\n" + res.get("_VERDICT", "(velocity unavailable)"))
    pathlib.Path("align_vs_skill.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
