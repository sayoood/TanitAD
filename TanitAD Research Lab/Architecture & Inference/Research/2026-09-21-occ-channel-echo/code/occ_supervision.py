"""Is `occluded` SUPERVISED on our corpus at all — and if so, does the head beat its base rate?

`c920f15` closed `l`/`w` as genuine defects and left `occluded` (alignment 0.284) as the ONE
unexamined field, with the note "no target channel identified". This answers that directly, and the
answer is a precondition for every other reading of the field.

⭐ WHY THIS IS NOT THE SAME PROBE AS `quiet_fields.py`. That probe asks "is the target predictable
from target-side geometry". For `occluded` the prior question dominates, because the loss MASKS the
field: `agent_slots.py:602` computes `om = ot >= 0.0` and only supervises where the flag is present,
and `targets_from_join` initialises `occ` to **-1.0** (`:730`) and overwrites it only from column 5
of the join (`:739`). The trainer's own default is the same sentinel
(`refc_v3_train.py:3922`, `:4345`: `torch.full_like(..., -1.0)`). ⇒ a corpus whose join carries no
`occ` column trains this field on ZERO examples while the head keeps emitting it, and every
downstream reading of its output would be a reading of an UNTRAINED channel.

⛔ THE THREE OUTCOMES ARE COMMITTED IN ADVANCE:
  coverage == 0            -> the field is EMITTED AND NEVER SUPERVISED here. Not a "quiet" field
                              at all; the head-change SPEC must either drop it or supply the label.
  coverage > 0, head loses -> supervised and failing against its own base rate: a GENUINE DEFECT in
                              the same class as `l`/`w`.
  coverage > 0, head wins  -> the field works; alignment 0.284 does not imply a dead field (which is
                              the same lesson `3e3dac9` taught for `v_rel_y`).

⚠️ AND THE SEMANTICS TRAVEL WITH IT. `agent_slots.py:82-87`: `occ` IS `bev_raster.fov_mask`'s
predicate (0/7,680 cells disagree), so this field means OUT OF THE FRONT CAMERA'S FIELD while the
track continues — never object-object occlusion. A win here is evidence about P4 ("does the latent
carry agents the camera cannot see"), and must not be reported as occlusion reasoning.

⛔ CONTROLS, per the 2026-08-22 discipline that caught four estimator bugs:
  * the BASE-RATE constant is the control and it must read the no-information value — for a binary
    target that is the entropy of the base rate, and the constant is fit on the FIT half only;
  * fit/score split is EPISODE-DISJOINT;
  * n, the number of episodes, and the base rate are printed;
  * the head's own probability SPREAD is reported, because a near-constant emission is the
    presence-degeneracy pattern (`286e0d3`) and would otherwise hide inside an accuracy number.
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


def logloss(p: float, y: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    wis = [w for w in SP.trainer_windows(corp.ds, 1500)
           if corp.eligibility(w) is None][:N_WIN]

    # ---- PASS 1: coverage, with NO model pass. This is the question that gates the rest.
    n_valid = n_flag = n_pos = 0
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        occ = item["agent_occ"].float().numpy()
        val = item["agent_valid"].numpy().astype(bool)
        n_valid += int(val.sum())
        m = val & (occ >= 0.0)
        n_flag += int(m.sum())
        n_pos += int((occ[m] > 0.5).sum())
    cov = n_flag / n_valid if n_valid else 0.0
    base = n_pos / n_flag if n_flag else float("nan")
    print(f"coverage: {n_flag}/{n_valid} valid targets carry an occ flag ({cov:.4%}); "
          f"positives {n_pos} -> base rate {base if n_flag else float('nan')}")

    # ---- PASS 2: what does the head EMIT, and does it beat the base rate where supervised?
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    probs_all = []                                   # every matched slot's emitted p
    by = collections.defaultdict(list)               # episode -> (p, y) on SUPERVISED pairs
    used = 0
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        sl = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
              if torch.is_tensor(v) and v.dim() >= 1}
        if "occ_logit" not in sl:
            print("ZZABORT the head emits no occ_logit:", sorted(sl))
            return 3
        m = match_slots({k: v[None] for k, v in sl.items()},
                        {"box": item["agent_box"][None].float(),
                         "valid": item["agent_valid"][None], "cls": item["agent_cls"][None]})
        r, c = m["rows"][0].tolist(), m["cols"][0].tolist()
        if not r:
            continue
        used += 1
        occ = item["agent_occ"].float().numpy()
        p_all = torch.sigmoid(sl["occ_logit"]).numpy()
        for i, j in zip(r, c):
            probs_all.append(float(p_all[i]))
            if occ[j] >= 0.0:
                by[int(e_i)].append((float(p_all[i]), float(occ[j] > 0.5)))

    res = {"_what": "is `occluded` supervised on this corpus, and does the head beat its base rate?",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000",
           "_semantics": ("occ IS bev_raster.fov_mask's predicate (agent_slots.py:82-87) -> OUT OF "
                          "THE FRONT CAMERA'S FIELD, never object-object occlusion"),
           "_mask_site": "agent_slots.py:602 `om = ot >= 0.0`; sentinel -1.0 set at :730",
           "windows_used": used,
           "coverage": {"valid_targets": n_valid, "with_occ_flag": n_flag,
                        "fraction": round(cov, 6),
                        "positives": n_pos,
                        "base_rate": round(base, 6) if n_flag else None},
           "head_emission": {"n_matched_slots": len(probs_all),
                             "p_mean": round(st.mean(probs_all), 5) if probs_all else None,
                             "p_sd": round(st.pstdev(probs_all), 5) if len(probs_all) > 1 else None,
                             "p_min": round(min(probs_all), 5) if probs_all else None,
                             "p_max": round(max(probs_all), 5) if probs_all else None}}

    if n_flag == 0:
        res["_VERDICT"] = (
            "⛔ `occluded` IS EMITTED AND NEVER SUPERVISED ON THIS CORPUS — the join carries no occ "
            "column, every target hits the -1 sentinel, and the BCE term sees ZERO examples. Its "
            "output is an UNTRAINED channel and no reading of it (alignment included) is a reading "
            "about perception. The head change must DROP it or SUPPLY the label.")
    else:
        pairs = {k: v for k, v in by.items() if v}
        eps_ = sorted(pairs)
        half = set(eps_[: len(eps_) // 2])
        fit = [x for k in eps_ if k in half for x in pairs[k]]
        sc = {k: v for k, v in pairs.items() if k not in half}
        if not fit or not sc:
            res["_VERDICT"] = "INCONCLUSIVE — the episode-disjoint split leaves one side empty"
        else:
            const = sum(y for _, y in fit) / len(fit)      # base rate, FIT SIDE ONLY
            head = st.mean(logloss(p, y) for v in sc.values() for p, y in v)
            ctrl = st.mean(logloss(const, y) for v in sc.values() for p, y in v)
            rng = random.Random(SEED)
            keys = list(sc)
            ms = sorted(
                (lambda d: st.mean(logloss(const, y) for _, y in d)
                 - st.mean(logloss(p, y) for p, y in d))(
                    [x for k in rng.choices(keys, k=len(keys)) for x in sc[k]])
                for _ in range(B)) if len(keys) > 1 else None
            ci = [round(ms[int(0.025 * B)], 5), round(ms[int(0.975 * B)], 5)] if ms else None
            beats = bool(ci and ci[0] > 0)
            res["skill"] = {"fit_side_base_rate_control": round(const, 5),
                            "n_scored_pairs": sum(len(v) for v in sc.values()),
                            "n_scored_episodes": len(sc),
                            "logloss_head": round(head, 5),
                            "logloss_base_rate_control": round(ctrl, 5),
                            "gain": round(ctrl - head, 5), "CI95": ci,
                            "head_beats_base_rate": beats}
            res["_VERDICT"] = (
                "⭐ the head BEATS its base rate on `occluded` ⇒ alignment 0.284 does NOT mark a "
                "dead field, and this is evidence on P4 (agents outside the camera's field)"
                if beats else
                "⛔ the head DOES NOT beat the base rate on a field it IS supervised on ⇒ a GENUINE "
                "DEFECT in the same class as l/w")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("occ_supervision.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
