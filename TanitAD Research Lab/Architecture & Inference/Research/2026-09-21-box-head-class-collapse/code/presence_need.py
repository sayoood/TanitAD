"""Is the tactical head's presence DEGENERATE, or does it have NOTHING TO DISCRIMINATE?

`731ecd7` measured `sigmoid(presence_logit)` on `core.agent_head` at mean **0.76671**, sd 0.05494,
range [0.59251, 0.86197], 320/320 slots above the 0.5 gate — and concluded that presence "supplies
the tactical decoder with no per-slot weighting whatever". ⛔ THAT CONCLUSION HAS AN UNTESTED
PREMISE, and it is the same premise `3e3dac9` caught for `v_rel_y`: **a near-constant field is only
a defect if there was something to vary ABOUT.**

⭐ THE PREMISE IS CHECKABLE AND CHEAP. This head has **16 queries**. `match_slots` keeps the
`n_queries` NEAREST targets (`refc_agents.py:296-300` documents exactly that), so if a typical
window carries **more than 16** visible agents, then EVERY slot corresponds to a real agent, there
are no empty slots, and a uniformly high presence is **CORRECT** — not degenerate. The DETR ∅-logit
only has work to do when slots outnumber objects.

  n_visible >= 16 in ~every window  -> presence has nothing to discriminate ⇒ my claim is WRONG and
                                        must be corrected before it propagates.
  windows with n_visible < 16 exist -> presence SHOULD separate the real slots from the empty ones
                                        there, and whether it does is measurable.

⛔ AND THE MATCHER CANNOT BE USED AS-IS, BECAUSE IT WOULD MAKE THE ANSWER CIRCULAR. `match_slots`'s
Hungarian cost includes `w["presence"] * -sigmoid(presence_logit)` (`agent_slots.py:475-477`), so it
PREFERS high-presence slots — "matched slots have higher presence" would then be a property of the
matcher, not of the head. ⇒ this file matches on **CENTRE ONLY**, by its own Hungarian on centre
distance, with presence never entering the assignment. Same discipline that made velocity the clean
place to test alignment in `align_vs_skill.py`.

⚠️ VISIBILITY IS THE RIGHT DENOMINATOR, NOT THE RAW JOIN. `refc_agents.filter_targets_to_visible`
is documented MANDATORY before `match_slots` for a camera-only head, because
`targets_from_join` marks EVERY agent in the record valid with no azimuth or range filter. Counting
raw valid targets would overstate what this head could possibly be asked to represent.
⛔ CPU only, forward hook, read-only, no GPU.
"""
from __future__ import annotations

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
from tanitad.models.agent_slots import SLOT_SLICES        # noqa: E402
from tanitad.refs.refc_agents import filter_targets_to_visible   # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 60
B = 4000
SEED = 20260921
NEAR_M = 4.0        # a slot "corresponds to" an agent only if its centre lands within this
NEAR_SWEEP = (2.0, 4.0, 8.0, 16.0)   # the threshold is arbitrary, so it is SWEPT, not assumed


def centre_only_match(pred_xy, tgt_xy):
    """Hungarian on CENTRE DISTANCE ALONE — presence never enters the assignment."""
    if len(tgt_xy) == 0 or len(pred_xy) == 0:
        return {}
    from scipy.optimize import linear_sum_assignment
    d = np.linalg.norm(pred_xy[:, None, :] - tgt_xy[None, :, :], axis=-1)
    r, c = linear_sum_assignment(d)
    return {int(i): float(d[i, j]) for i, j in zip(r, c)}


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    root = getattr(mp, "model", None) or getattr(mp, "net", None)
    if root is None:
        for a in dir(mp):
            o = getattr(mp, a, None)
            if isinstance(o, torch.nn.Module):
                root = o
                break
    named = dict(root.named_modules())
    hname, hmod = [(n, m) for n, m in named.items()
                   if isinstance(m, torch.nn.Linear) and n.endswith("agent_head.head")][0]
    holder = named[hname.rsplit(".", 1)[0]]
    n_q = int(getattr(holder, "n_queries", 0) or 0)
    dec = holder
    print(f"hooking {hname}; declared queries = {n_q}")

    outs = []
    h = hmod.register_forward_hook(lambda m, i, o: outs.append(o.detach()))
    wis = [w for w in SP.trainer_windows(corp.ds, 1500)
           if corp.eligibility(w) is None][:N_WIN]

    n_vis, n_raw_valid = [], []
    rows = []          # (episode, presence, corresponds_to_a_real_agent)
    slack_windows = 0
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        outs.clear()
        mp.forward(item, str(corp.clip_ids[e_i]))
        if not outs:
            continue
        raw = outs[-1].float().reshape(-1, outs[-1].shape[-1])
        assert raw.shape[0] == n_q, f"ZZABORT hooked {raw.shape[0]} vs declared {n_q}"
        pres = torch.sigmoid(raw[..., SLOT_SLICES["presence"]].squeeze(-1)).numpy()

        tgt = {"box": item["agent_box"][None].float(),
               "valid": item["agent_valid"][None],
               "cls": item["agent_cls"][None]}
        n_raw_valid.append(int(tgt["valid"][0].sum()))
        vis = filter_targets_to_visible({k: v.clone() for k, v in tgt.items()})
        v = vis["valid"][0].numpy().astype(bool)
        nv = int(v.sum())
        n_vis.append(nv)
        if nv >= n_q:
            continue                     # no empty slot exists here: nothing to discriminate
        slack_windows += 1
        dd = dec.decode(raw[None])       # the head's own decode, for centre in metres
        pxy = dd["box"][0][:, :2].detach().numpy()
        txy = vis["box"][0].numpy()[v][:, :2]
        m = centre_only_match(pxy, txy)
        for i in range(n_q):
            rows.append((int(e_i), float(pres[i]),
                         float(m[i]) if i in m else float("inf")))
    h.remove()

    res = {"_what": "is the tactical head's presence degenerate, or is there nothing to discriminate?",
           "_evidence_class": "MEASURED (ours), CPU, forward hook, A8 ckpt_5000",
           "_non_circular": ("matched on CENTRE DISTANCE ONLY; match_slots' own cost includes "
                             "-sigmoid(presence_logit) (agent_slots.py:475-477) and would make "
                             "this circular"),
           "_visibility": "targets filtered by refc_agents.filter_targets_to_visible first",
           "declared_queries": n_q,
           "visible_agents_per_window": {
               "n_windows": len(n_vis),
               "mean": round(st.mean(n_vis), 3) if n_vis else None,
               "median": int(st.median(n_vis)) if n_vis else None,
               "min": min(n_vis) if n_vis else None, "max": max(n_vis) if n_vis else None,
               "windows_with_FEWER_than_n_queries": slack_windows,
               "share_with_slack": round(slack_windows / len(n_vis), 4) if n_vis else None},
           "raw_valid_per_window_before_visibility_filter": {
               "mean": round(st.mean(n_raw_valid), 3) if n_raw_valid else None,
               "max": max(n_raw_valid) if n_raw_valid else None}}

    def analyse(thr):
        """⛔ THE POINT ESTIMATE AND THE INTERVAL MUST COVER THE SAME SET.

        MEASURED on the first run of this file: a point estimate over ALL rows read 0.04514 while a
        bootstrap resampling only the episodes that carry BOTH groups read [0.01135, 0.04469] -- the
        point sat ABOVE its own upper bound, which is impossible for one quantity and is the tell
        that two different populations were being described. Both are now restricted to `keys`.
        """
        by = {}
        for e, pr, d in rows:
            by.setdefault(e, []).append((pr, d <= thr))
        keys = [k for k, v in by.items() if any(x[1] for x in v) and any(not x[1] for x in v)]
        if len(keys) < 2:
            return {"threshold_m": thr, "note": "fewer than 2 usable episodes", "n_episodes": len(keys)}
        sub = [x for k in keys for x in by[k]]
        on = [pr for pr, k in sub if k]
        off = [pr for pr, k in sub if not k]
        rng = random.Random(SEED)
        ms = sorted(
            (lambda d: st.mean([pr for pr, k in d if k]) - st.mean([pr for pr, k in d if not k]))(
                [x for k in rng.choices(keys, k=len(keys)) for x in by[k]])
            for _ in range(B))
        ci = [round(ms[int(0.025 * B)], 5), round(ms[int(0.975 * B)], 5)]
        return {"threshold_m": thr, "n_episodes": len(keys),
                "n_at_agent": len(on), "n_empty": len(off),
                "presence_at_agent": round(st.mean(on), 5),
                "presence_empty": round(st.mean(off), 5),
                "difference": round(st.mean(on) - st.mean(off), 5),
                "CI95": ci, "separated": bool(ci[0] > 0),
                "_same_set": "point estimate and CI both restricted to the bootstrap's episodes"}

    if not rows:
        res["_VERDICT"] = (
            "⭐⭐ CORRECTION TO `731ecd7` — THERE IS NOTHING FOR PRESENCE TO DISCRIMINATE. Every "
            f"sampled window carries at least {n_q} visible agents, so every one of this head's "
            "slots corresponds to a real agent and a uniformly high presence is CORRECT, not "
            "degenerate. The DETR ∅-logit only has work when slots outnumber objects.")
    else:
        sweep = [analyse(t) for t in NEAR_SWEEP]
        res["threshold_sweep"] = sweep
        main_row = next((r for r in sweep if r.get("threshold_m") == NEAR_M), sweep[0])
        res["slack_windows_only"] = main_row
        sep = bool(main_row.get("separated"))
        n_sep = sum(1 for r in sweep if r.get("separated"))
        res["_VERDICT"] = (
            "⭐ PRESENCE DOES DISCRIMINATE where it has the chance — slots sitting on a real "
            f"agent carry higher presence than empty ones, separated at {n_sep} of "
            f"{len(sweep)} thresholds. ⇒ `731ecd7`'s 'no per-slot weighting' reading is TOO "
            "STRONG and is CORRECTED: the field is weak, not inert."
            if sep else
            "⛔ PRESENCE DOES NOT DISCRIMINATE EVEN WHERE IT COULD — in windows that DO have "
            "empty slots, presence at a real agent is not separated from presence with no agent "
            "near. ⇒ `731ecd7` stands, now on a non-circular test.")
        _unused = None
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["_VERDICT"])
    pathlib.Path("presence_need.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
