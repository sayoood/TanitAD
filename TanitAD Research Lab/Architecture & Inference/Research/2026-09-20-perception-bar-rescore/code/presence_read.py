"""The DETECTION half of `D-S1-DEP-BOX` — does the box head know HOW MANY agents are there?

`S1-GATE-PRED` is defined on predicted occupancy from the BEV map AND the 3-D box heads. The map
half was discharged; of the box half, only the LOCALISATION read exists (near-forward centre
5.9539 m, `cb274b0`). This adds the detection read.

⛔ WHY NOT AP, AND WHY NOT THE MATCHER. Conventional AP is not defined here: `match_slots` pairs
min(n_target, n_query), and with <= 32 padded targets against 100 queries EVERY valid target is
matched by construction (`n_matched == n_target` on 100 % of 498 rows). Worse, scoring presence at
predicting "was matched" would be CIRCULAR — `agent_slots.py:475-477` puts
`w["presence"] * -sigmoid(presence_logit)` INTO the Hungarian cost, so which queries get matched
already depends on presence. Same trap as enriching a DAC pack with P2's own criterion.

⭐ SO THE READ IS MATCHER-FREE: per window compare the head's COUNT of confident slots against the
true target count. No assignment, no Hungarian, no presence in the ground truth.

⛔ AND IT CARRIES THE CONTROL THE PROGRAMME REQUIRES: a CONSTANT predictor at the mean true count
reads the no-information value. A head that cannot beat that has added nothing — the rule that
caught four estimator bugs on 2026-08-22.
⚠️ `presence` carries a prior bias of logit(0.05) (`agent_slots.py:273`), so an UNTRAINED head
predicts ~5 active slots of 100 by construction; that is reported so a reader can see whether the
count is learned or inherited from the bias.
⛔ CPU only, no GPU.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import statistics as st
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 40
B = 4000
SEED = 20260920


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:N_WIN]

    rows = []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        sl = out["perception"]["box_slots"]
        if "presence_logit" not in sl:
            print("⛔ no presence_logit in box_slots:", sorted(sl))
            return 3
        p = torch.sigmoid(sl["presence_logit"][0].float().cpu())
        n_true = int(item["agent_valid"].numpy().astype(bool).sum())
        rows.append({"ep": int(e_i), "n_true": n_true,
                     "n_pred": int((p > 0.5).sum()), "p_mean": float(p.mean()),
                     "p_max": float(p.max()), "n_q": int(p.numel())})
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} windows", flush=True)

    assert rows, "control: zero windows scored"
    n_q = rows[0]["n_q"]
    true = [r["n_true"] for r in rows]
    pred = [r["n_pred"] for r in rows]
    const = st.mean(true)                      # the no-information predictor

    mae_head = st.mean(abs(a - b) for a, b in zip(pred, true))
    mae_const = st.mean(abs(const - b) for b in true)

    # episode-clustered bootstrap on the DIFFERENCE (const - head): > 0 means the head helps
    by = collections.defaultdict(list)
    for r in rows:
        by[r["ep"]].append(abs(const - r["n_true"]) - abs(r["n_pred"] - r["n_true"]))
    keys = list(by)
    rng = random.Random(SEED)
    ms = sorted(st.mean(x for k in rng.choices(keys, k=len(keys)) for x in by[k])
                for _ in range(B)) if len(keys) > 1 else None
    ci = [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B)], 3)] if ms else None

    res = {"_what": "detection half of D-S1-DEP-BOX: does the box head know HOW MANY agents?",
           "_evidence_class": "MEASURED (ours), CPU, no GPU, A8 ckpt_5000",
           "_matcher_free": "counts only -- no Hungarian, so presence cannot score its own cost",
           "n_windows": len(rows), "n_episodes": len(keys), "n_queries": n_q,
           "mean_true_targets": round(st.mean(true), 2),
           "mean_pred_confident_slots": round(st.mean(pred), 2),
           "mean_presence_prob": round(st.mean(r["p_mean"] for r in rows), 4),
           "max_presence_prob": round(max(r["p_max"] for r in rows), 4),
           "MAE_head": round(mae_head, 3),
           "MAE_constant_control": round(mae_const, 3),
           "improvement_over_constant": round(mae_const - mae_head, 3),
           "CI95_improvement_episode_clustered": ci,
           "beats_constant": bool(ci and ci[0] > 0),
           "_prior_bias_note": "presence carries logit(0.05); an untrained head would sit near "
                               f"{0.05*n_q:.0f} of {n_q} confident slots"}
    res["_VERDICT"] = ("the head's COUNT beats the constant control" if res["beats_constant"]
                       else "⛔ the head's count does NOT beat a constant predictor at this n")
    print(json.dumps(res, indent=1))
    pathlib.Path("presence_read.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
