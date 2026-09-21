"""Does the presence weight vector SEE the variation the features carry?

The chain so far: presence is near-constant across slots (`286e0d3`); the decoder emits 100
distinct boxes (`a15de2d`); the projection weights are healthy and heavier than their siblings
(`3a8064d`).

⭐ HALF THE REMAINING QUESTION FALLS OUT BY LOGIC, NOT COMPUTE. Box and presence are slices of ONE
linear layer's output on ONE per-slot feature vector (`agent_slots.py:368`). Boxes vary across
slots ⇒ the FEATURES vary across slots. So "the features are near-constant" is ELIMINATED, and the
surviving reading is geometric: `w_presence` is nearly ORTHOGONAL to the directions the features
actually vary in.

⇒ THIS MEASURES THAT, with the other fields of the same layer as the control. For each field the
statistic is dimensionless, so a bigger weight vector cannot flatter itself:

    alignment(w) = var_slots(w·f) / ( |w|^2 * mean_dim_var(f) )

  ~1  -> the field picks up its fair share of the available variation
  ~0  -> the field's direction is orthogonal to where the features move

⛔ The normalisation is the point: `3a8064d` measured presence's row norm at 1.20x the sibling
median, so if presence STILL shows less output variation, alignment — not magnitude — is the cause.
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

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 12


def main() -> int:
    total = sum(int(SLOT_SLICES[k].stop - SLOT_SLICES[k].start) for k in SLOT_SLICES)
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")

    # find the slot head by OUT features, then hook its INPUT
    target, root = None, getattr(mp, "model", None) or getattr(mp, "net", None)
    if root is None:
        for a in dir(mp):
            o = getattr(mp, a, None)
            if isinstance(o, torch.nn.Module):
                root = o
                break
    assert root is not None, "ZZABORT could not find the model on ModelPass"
    for name, m in root.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == total:
            target = (name, m)
            break
    assert target is not None, f"ZZABORT no Linear with out_features == {total}"
    print(f"hooking {target[0]}  in={target[1].in_features} out={target[1].out_features}")

    feats = []
    h = target[1].register_forward_hook(lambda mod, inp, out: feats.append(inp[0].detach()))

    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:N_WIN]
    used = 0
    per_field = {k: [] for k in SLOT_SLICES}
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        feats.clear()
        mp.forward(item, str(corp.clip_ids[e_i]))
        if not feats:
            continue
        f = feats[-1].float()
        f = f.reshape(-1, f.shape[-1])          # [slots, D]
        if f.shape[0] < 2:
            continue
        used += 1
        mean_dim_var = float(f.var(dim=0, unbiased=False).mean())
        W = target[1].weight.detach().float()
        for k, sl in SLOT_SLICES.items():
            w = W[sl]                            # [rows, D]
            proj = f @ w.T                       # [slots, rows]
            v = float(proj.var(dim=0, unbiased=False).mean())
            wn = float((w * w).sum(dim=1).mean())
            per_field[k].append(v / (wn * mean_dim_var) if wn * mean_dim_var else 0.0)
    h.remove()
    assert used, "ZZABORT no windows produced features"

    al = {k: round(st.mean(v), 5) for k, v in per_field.items() if v}
    pres = al.get("presence")
    others = {k: v for k, v in al.items() if k != "presence"}
    med = sorted(others.values())[len(others) // 2]
    res = {"_what": "does w_presence see the variation the features carry?",
           "_evidence_class": "MEASURED (ours), CPU, forward hook, A8 ckpt_5000",
           "_statistic": "alignment(w) = var_slots(w.f) / (|w|^2 * mean_dim_var(f))",
           "_control": "the other fields of the SAME layer, same normalisation",
           "windows_used": used, "slots": None,
           "alignment_by_field": dict(sorted(al.items(), key=lambda kv: -kv[1])),
           "presence_alignment": pres,
           "median_other_alignment": round(med, 5),
           "ratio_presence_to_median": round(pres / med, 4) if med else None}
    res["_VERDICT"] = (
        "⭐ PRESENCE IS MISALIGNED — its direction picks up far less of the available feature "
        "variation than its siblings, despite a LARGER norm ⇒ the degeneracy is GEOMETRIC"
        if res["ratio_presence_to_median"] is not None and res["ratio_presence_to_median"] < 0.3
        else "presence picks up a comparable share of the variation ⇒ misalignment is NOT the "
             "explanation and the cause is elsewhere")
    print(json.dumps(res, indent=1))
    print("\n" + res["_VERDICT"])
    pathlib.Path("presence_alignment.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
