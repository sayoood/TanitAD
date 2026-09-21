"""The box head's CLASS output has collapsed to ONE class — and that is why SIZE is a constant.

`size_floor.py` measured the head's `l`/`w` as statistically indistinguishable from a single GLOBAL
MEDIAN, while the SAME median table keyed on the GROUND-TRUTH class beats the head 2.6x on `l` and
3.2x on `w`. And the table keyed on the head's OWN predicted class read **identical to the global
one, to five decimals** — which can only happen if the head's class prediction carries no
discrimination at all.

⭐ THIS FILE ESTABLISHES THAT DIRECTLY rather than inferring it from the coincidence, and it widens
the read from MATCHED pairs to EVERY SLOT, because "collapsed on the targets it was scored on" and
"collapsed everywhere" are different claims and only the second one indicts the head.

⛔ THE MECHANISM IS NAMED BY THE CODE'S OWN COMMENT. `agent_slots.py:230-232` introduces
`NO_OBJECT_W = 0.1` with the reason: *"unmatched slots vastly outnumber matched ones, and an
unweighted BCE simply learns 'always empty'"*. The `cls` term two hundred lines later
(`agent_slots.py:591`) is a **plain `cross_entropy` with no `weight=`**, applied to a target
distribution this file measures — and the same argument that earned presence its down-weight was
never applied to class. ⇒ the prediction is that `cls` learns "always <majority>", which is exactly
what `size_floor.py` ran into from the other side.

⚠️ THE CAVEAT THAT TRAVELS WITH IT, AND IT IS NOT SMALL: this is `ckpt_5000` — 5,000 steps. A
collapse at 5,000 steps may be a training-DURATION artifact rather than a loss-DESIGN defect, and
this file cannot separate them. What it CAN do is state the imbalance, confirm the loss is
unweighted, and show the collapse is total rather than partial. The separating experiment is a
longer run or a class-weighted arm, and that needs GPU.
⛔ CPU only, read-only.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES, SLOT_LOSS_W   # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 20


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:N_WIN]

    all_slots = collections.Counter()
    margins, used = [], 0
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        lg = out["perception"]["box_slots"]["cls_logits"][0].float().cpu()
        used += 1
        all_slots.update(lg.argmax(-1).tolist())
        s = lg.softmax(-1)
        top2 = s.topk(2, dim=-1).values
        margins.extend((top2[:, 0] - top2[:, 1]).tolist())

    # the GT distribution over MATCHED pairs, from the cached size rows
    A = np.load("size_rows.npz")["rows"]
    gt = collections.Counter(A[:, 5].astype(int).tolist())
    pr = collections.Counter(A[:, 6].astype(int).tolist())
    n = int(A.shape[0])
    maj_k, maj_n = gt.most_common(1)[0]
    acc = float((A[:, 5] == A[:, 6]).mean())

    res = {"_what": "has the box head's class output collapsed, and is the cls loss unweighted?",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000",
           "_loss_design": {
               "cls_weight_in_SLOT_LOSS_W": SLOT_LOSS_W["cls"],
               "cls_criterion": "nn.functional.cross_entropy(..., reduction='sum') at "
                                "agent_slots.py:591 — NO `weight=` argument, i.e. UNWEIGHTED",
               "presence_has_a_down_weight": "NO_OBJECT_W = 0.1 (agent_slots.py:232), introduced "
                                             "because 'an unweighted BCE simply learns always "
                                             "empty' (:230-231)"},
           "matched_pairs": {
               "n": n, "n_classes_in_vocab": len(AGENT_CLASSES),
               "GT_distribution": {AGENT_CLASSES[k]: v for k, v in sorted(gt.items())},
               "HEAD_distribution": {AGENT_CLASSES[k]: v for k, v in sorted(pr.items())},
               "distinct_classes_predicted": len(pr),
               "top1_accuracy": round(acc, 5),
               "majority_class_baseline": round(maj_n / n, 5),
               "majority_class": AGENT_CLASSES[maj_k],
               "accuracy_equals_majority_baseline": abs(acc - maj_n / n) < 1e-9,
               "imbalance_ratio_majority_to_rarest": round(
                   maj_n / min(gt.values()), 1)},
           "all_slots_not_only_matched": {
               "windows": used, "n_slot_predictions": sum(all_slots.values()),
               "distinct_classes_predicted": len(all_slots),
               "distribution": {AGENT_CLASSES[k]: v for k, v in sorted(all_slots.items())},
               "mean_top1_minus_top2_softmax_margin": round(float(np.mean(margins)), 5)}}

    # the class the safety case cares about
    ped = AGENT_CLASSES.index("person")
    res["vulnerable_road_users"] = {
        "GT_person_targets": int(gt.get(ped, 0)),
        "predicted_person": int(pr.get(ped, 0)),
        "note": "every GT `person` is emitted as the majority class when the head is collapsed"}

    total_collapse = (res["matched_pairs"]["distinct_classes_predicted"] == 1
                      and res["all_slots_not_only_matched"]["distinct_classes_predicted"] == 1)
    res["_VERDICT"] = (
        "⛔⛔ TOTAL CLASS COLLAPSE — the head emits ONE class of "
        f"{len(AGENT_CLASSES)} on every matched pair AND on every slot, its top-1 accuracy IS the "
        "majority-class baseline, and the cls loss is UNWEIGHTED on a target this imbalanced. The "
        "size defect is DOWNSTREAM of this: a median lookup on the TRUE class beats the head's own "
        "size regression 2.6x (l) and 3.2x (w)."
        if total_collapse else
        "the class head is not totally collapsed — read the distributions before concluding")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("cls_collapse.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
