"""⛔ CORRECTION: the landed alignment spectrum was measured on THE WRONG HEAD.

`148ceb9` (`presence_alignment.py`) selected the layer to hook as *"the Linear whose `out_features`
== the total slot width"*, breaking at the FIRST match. `SLOT_SLICES` sums to **21**, and the first
`Linear(*, 21)` in this model is **`core.agent_head.head`** — the 2-D :class:`AgentSlotDecoder`,
which runs with **16 queries**.

⛔ BUT EVERY SKILL, FLOOR AND DEFECT NUMBER IN THIS INVESTIGATION IS SCORED ON A DIFFERENT HEAD.
`s1_pass.py:307` reads `out["perception"]["box_slots"]`, produced at
`refcv6_perception_branch.py:426` by **`box_dec`**, a :class:`Box3DSlotDecoder`
(`box3d_head.py:227`). That class subclasses `AgentSlotDecoder` and **REPLACES `self.head`** with
`nn.Linear(d_model, SLOT3D_WIDTH)` (`:247`) — a different width — and in this build it emits
**100 slots**, not 16. MEASURED: matched row indices from `match_slots` reach **99**, while the
hooked tensor was `[1, 16, 256]`.

⇒ the spectrum 0.025 → 56.7 describes a head nobody scores, and joining it to skill numbers from
`box_dec` compared two different modules. That is the `df` / Thor `free` / cgroup /
cylindrical-FOV family exactly: **a true measurement quoted outside its scope**, and it is worse
than a wrong number because both halves were individually correct.

⭐ THE FIX IS TO SELECT THE MODULE BY THE OUTPUT BEING SCORED, NOT BY A WIDTH THAT TWO MODULES
SHARE A PREFIX OF. This file hooks `box_dec.head` BY NAME and, as a guard that cannot pass by
accident, ASSERTS that the captured slot count equals the slot count of the `box_slots` the same
forward emitted. A width match is what failed; an identity between the hooked tensor and the scored
tensor cannot.

The statistic is unchanged from `148ceb9` so the two are comparable:

    alignment(w) = var_slots(w.f) / ( |w|^2 * mean_dim_var(f) )

⛔ CPU only, forward hook, read-only.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402
from tanitad.models.agent_slots import SLOT_SLICES        # noqa: E402
from tanitad.models.box3d_head import SLOT3D_SLICES, SLOT3D_WIDTH   # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 12

LANDED = {"cy": 56.7131, "cx": 24.77866, "v_rel_x": 7.258, "l": 1.38458,
          "w": 0.2462, "occluded": 0.284, "v_rel_y": 0.08807,
          "yaw_rate_rel": 0.02459}


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
    assert root is not None, "ZZABORT no model on ModelPass"

    named = dict(root.named_modules())
    wrong = [(n, m) for n, m in named.items()
             if isinstance(m, torch.nn.Linear)
             and m.out_features == sum(int(SLOT_SLICES[k].stop - SLOT_SLICES[k].start)
                                       for k in SLOT_SLICES)]
    right = [(n, m) for n, m in named.items()
             if isinstance(m, torch.nn.Linear) and "box_dec" in n and n.endswith("head")]
    assert right, f"ZZABORT no box_dec head found; candidates: {[n for n in named if 'box_dec' in n]}"
    rname, rmod = right[0]
    print(f"WRONG (what 148ceb9 hooked): {[n for n, _ in wrong]}")
    print(f"RIGHT (what is scored)     : {rname}  in={rmod.in_features} out={rmod.out_features}")
    assert rmod.out_features == SLOT3D_WIDTH, \
        f"ZZABORT box_dec head is {rmod.out_features} wide, SLOT3D_WIDTH is {SLOT3D_WIDTH}"

    feats = []
    h = rmod.register_forward_hook(lambda m, i, o: feats.append(i[0].detach()))
    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:N_WIN]
    per = {k: [] for k in SLOT3D_SLICES}
    used, n_slots_seen = 0, set()
    W = rmod.weight.detach().float()
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        feats.clear()
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        if not feats:
            continue
        f = feats[-1].float().reshape(-1, feats[-1].shape[-1])
        # ⭐ THE GUARD A WIDTH MATCH COULD NOT GIVE: the hooked tensor must have the
        # SAME number of slots as the box_slots this very forward emitted.
        n_scored = int(out["perception"]["box_slots"]["box"].shape[-2])
        assert f.shape[0] == n_scored, \
            f"ZZABORT hooked {f.shape[0]} slots but box_slots carries {n_scored}"
        n_slots_seen.add(n_scored)
        used += 1
        mdv = float(f.var(dim=0, unbiased=False).mean())
        for k, sl in SLOT3D_SLICES.items():
            w = W[sl]
            v = float((f @ w.T).var(dim=0, unbiased=False).mean())
            wn = float((w * w).sum(dim=1).mean())
            per[k].append(v / (wn * mdv) if wn * mdv else 0.0)
    h.remove()
    assert used, "ZZABORT no windows produced features"

    al = {k: round(st.mean(v), 5) for k, v in per.items() if v}
    ordered = dict(sorted(al.items(), key=lambda kv: -kv[1]))
    comp = {k: {"CORRECTED_box_dec": al[k], "LANDED_wrong_head": LANDED[k],
                "ratio": round(al[k] / LANDED[k], 3) if LANDED[k] else None}
            for k in LANDED if k in al}
    res = {"_what": "the alignment spectrum, measured on the head that is ACTUALLY SCORED",
           "_evidence_class": "MEASURED (ours), CPU, forward hook, A8 ckpt_5000",
           "_correction_of": "148ceb9 / presence_alignment.py",
           "_wrong_module": [n for n, _ in wrong],
           "_right_module": rname,
           "_guard": "the hooked slot count is asserted EQUAL to the emitted box_slots count",
           "slots_per_window": sorted(n_slots_seen), "windows_used": used,
           "alignment_by_field_CORRECTED": ordered,
           "vs_landed": comp}
    pres = al.get("presence")
    others = {k: v for k, v in al.items() if k != "presence"}
    med = sorted(others.values())[len(others) // 2] if others else None
    res["presence_alignment"] = pres
    res["median_other_alignment"] = round(med, 5) if med else None
    res["ratio_presence_to_median"] = round(pres / med, 4) if (pres and med) else None
    print(json.dumps(res, indent=1, ensure_ascii=False))
    pathlib.Path("align_fix.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
