"""The class census on the join refcv6 ACTUALLY trains against — the B1 corpus, not parity.

⛔⛔ WHY THIS EXISTS: I BUILT `H-BOXCLS-1`'s WEIGHT VECTOR FROM THE WRONG CORPUS.
`agent_cls_weights_train2400.json` was censused over `train2400_agents.jsonl.xz`, the join for the
**parity** line (`physicalai-train-e438721ae894`). MEASURED 2026-09-22: that join covers **193 of
4,719** clips of refcv6's corpus — **4.09 %** — while `b1_train_plus_eval_agents.jsonl.xz` covers
**4,566 / 4,719 = 96.76 %** with **0** clips outside it.

⚠️ AND THE ARGUMENT AGAINST THE VECTOR IS MY OWN. `e172c65` refused a held-out EVAL split as a
proxy for TRAIN frequencies because class frequencies are **sampling-sensitive** (2.26x across two
disjoint 62-clip halves; up to 2.167x against the measured train vector). A vector censused on a
4 %-overlapping corpus is that same refusal, with a larger sample and a better disguise.

⭐ THE CONTROLS, both of which must hold or this census is not admissible:
  1. every clip read must be INSIDE the v7 corpus (the join claims 0 outside -- re-assert it here
     rather than inheriting it);
  2. the counted frames/boxes must reproduce the independent tally taken while measuring coverage
     (875,657 frames / 28,958,699 boxes). A partial read would otherwise yield a plausible vector
     from a partial corpus -- exactly the failure the parity census's control was built to catch.

⛔ CPU only, read-only. No raw clip id is printed or stored -- digests only.
"""
from __future__ import annotations

import collections
import json
import lzma
import math
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/Projects/TanitAD/stack")

from tanitad.data import parity                                  # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES             # noqa: E402

JOIN = pathlib.Path("C:/Users/Admin/a40-rescue/b1_train_plus_eval_agents.jsonl.xz")
V7 = pathlib.Path("D:/Projects/TanitAD-artifacts/_s2build-copy-20260919/v7_full/s2_labels_v7.jsonl")
PARITY_ART = pathlib.Path(
    "D:/Projects/TanitAD/stack/tanitad/data/agent_cls_weights_train2400.json")

# ⛔ LITERALS from the independent coverage pass. Not recomputed here on purpose.
EXPECT_FRAMES = 875_657
EXPECT_BOXES = 28_958_699
EXPECT_CLIPS = 4_566


def inv_freq(counts: dict) -> dict:
    pres = {c: n for c, n in counts.items() if n > 0}
    raw = {c: 1.0 / n for c, n in pres.items()}
    m = sum(raw.values()) / len(raw)
    return {c: v / m for c, v in raw.items()}


def main() -> int:
    v7 = set()
    for line in V7.open(encoding="utf-8"):
        if line.strip():
            v7.add(parity.clip_digest(json.loads(line)["clip_id"]))

    cls = collections.Counter()
    clips, frames, boxes, no_cls, outside = set(), 0, 0, 0, 0
    with lzma.open(JOIN, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            frames += 1
            dg = parity.clip_digest(str(d.get("clip_id", "")))
            clips.add(dg)
            if dg not in v7:
                outside += 1
            for a in d.get("agents") or []:
                boxes += 1
                c = a.get("cls")
                if c is None:
                    no_cls += 1
                else:
                    cls[str(c)] += 1

    ctrl = {"frames": (frames, EXPECT_FRAMES, frames == EXPECT_FRAMES),
            "boxes": (boxes, EXPECT_BOXES, boxes == EXPECT_BOXES),
            "clips": (len(clips), EXPECT_CLIPS, len(clips) == EXPECT_CLIPS),
            "frames_outside_v7": (outside, 0, outside == 0)}
    for k, (got, want, ok) in ctrl.items():
        print(f"  CONTROL {k:18s} got {got:>12,}  expect {want:>12,}  {'OK' if ok else 'FAIL'}")
    if not all(v[2] for v in ctrl.values()):
        print("ZZABORT a control failed; a partial read would give a plausible but wrong vector")
        return 3

    counts = {c: int(cls.get(c, 0)) for c in AGENT_CLASSES}
    unknown = {k: v for k, v in cls.items() if k not in AGENT_CLASSES}
    w_b1 = inv_freq(counts)

    par = json.loads(PARITY_ART.read_text(encoding="utf-8"))
    w_par = {c: float(v) for c, v in par["weights_inv_freq_mean1"].items()}
    c_par = {c: int(v) for c, v in par["counts"].items()}
    tot_b1, tot_par = sum(counts.values()), sum(c_par.values())

    rows = []
    for c in sorted(AGENT_CLASSES, key=lambda k: -counts[k]):
        sh_b1 = counts[c] / tot_b1
        sh_par = c_par[c] / tot_par
        rows.append({
            "class": c, "b1_count": counts[c], "parity_count": c_par[c],
            "b1_share": round(sh_b1, 6), "parity_share": round(sh_par, 6),
            "share_ratio_b1_over_parity": round(sh_b1 / sh_par, 3) if sh_par else None,
            "b1_weight": round(w_b1[c], 6), "parity_weight": round(w_par[c], 6),
            "weight_ratio_parity_over_b1": round(w_par[c] / w_b1[c], 3) if w_b1[c] else None})

    maj = max(counts, key=counts.get)
    rare = min((c for c in counts if counts[c]), key=lambda c: counts[c])
    worst = max(AGENT_CLASSES, key=lambda c: abs(math.log(w_par[c] / w_b1[c])))
    max_ratio = max(max(w_par[c] / w_b1[c], w_b1[c] / w_par[c]) for c in AGENT_CLASSES)

    res = {
        "_what": ("the class census on the join refcv6 ACTUALLY trains against (B1), and how far "
                  "the BANKED parity vector is from it"),
        "_evidence_class": "MEASURED (ours), CPU, read-only",
        "_source": str(JOIN), "_corpus": "B1 train+eval, the v7/SAM3 line refcv6 trains on",
        "_controls": {k: {"got": v[0], "expect": v[1], "ok": v[2]} for k, v in ctrl.items()},
        "n_clips": len(clips), "n_frames": frames, "n_boxes": tot_b1,
        "boxes_without_cls": no_cls, "unknown_classes": unknown,
        "counts": counts,
        "share": {c: round(counts[c] / tot_b1, 6) for c in AGENT_CLASSES},
        "imbalance_majority_to_rarest": round(counts[maj] / counts[rare], 1),
        "majority": maj, "rarest": rare,
        "WEIGHT_VECTOR_inv_freq_mean1": {c: round(w_b1[c], 6) for c in sorted(w_b1)},
        "classes_absent": [c for c in AGENT_CLASSES if counts[c] == 0],
        "vs_banked_parity_vector": {
            "_question": ("is the banked train2400 vector usable for refcv6, or is it the "
                          "sampling-sensitivity refusal applied to my own artifact?"),
            "per_class": rows, "worst_class": worst,
            "max_weight_ratio": round(max_ratio, 3),
            "parity_imbalance": par["imbalance_majority_to_rarest"],
            "b1_imbalance": round(counts[maj] / counts[rare], 1)},
    }
    res["_VERDICT"] = (
        f"B1 census MEASURED on {tot_b1:,} boxes over {len(clips):,} clips "
        f"({frames:,} frames). Imbalance {res['imbalance_majority_to_rarest']}:1 "
        f"({maj} : {rare}) vs parity's {par['imbalance_majority_to_rarest']}:1. The banked "
        f"parity vector differs from the corpus refcv6 trains on by up to "
        f"{max_ratio:.3f}x ({worst}).")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print()
    print(res["_VERDICT"])
    pathlib.Path("C:/Users/Admin/qland/work/pbox/b1_cls_census.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
