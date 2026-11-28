"""The TRAIN class census `H-BOXCLS-1` needs — from the join that already existed.

⛔ THIS EXISTS BECAUSE I REPORTED A BLOCKER THAT WAS NOT THERE. `e172c65` states *"no TRAIN agent
join exists on the dev box or on Thor (two probes); building one is a pod-side step over the
2,376-episode parity corpus"*. Both halves are FALSE:
`C:/Users/Admin/tanitad-data/joins/joins/train2400_agents.jsonl.xz` (136.7 MB, 2026-09-05) is the
canonical train join, and CLAUDE.md already CITES its summary — *"the 2,308-episode train join
(433,040 frames / 12,122,129 boxes)"* — as the basis for `--agent-queries 100`.
⚠️ My two probes were `D:/Projects/TanitAD-artifacts` and Thor. Neither is where joins live. *Absence
at one location is not absence*, and two probes in the wrong places are still one wrong place.

⭐ THE CONTROL THAT MAKES THIS READABLE: the counts below must reproduce the meta's own summary
(2,308 / 433,040 / 12,122,129) EXACTLY. A census that silently reads half a file would otherwise
produce a plausible weight vector from a partial corpus — and the whole reason a proxy was refused
is that class frequencies are sampling-sensitive.

⭐ AND IT SETTLES THE PROXY QUESTION WITH DATA. `cls_freq_split.py` measured eval's two 62-clip
halves disagreeing by up to 2.26x and concluded a held-out split could not proxy for TRAIN. That
was the right call on the evidence, but it was never a measurement OF train. Now both exist, so the
eval-vs-train gap is reported rather than assumed.
⛔ CPU only, read-only.
"""
from __future__ import annotations

import collections
import json
import lzma
import math
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
from tanitad.models.agent_slots import AGENT_CLASSES      # noqa: E402

JOIN = pathlib.Path("C:/Users/Admin/tanitad-data/joins/joins/train2400_agents.jsonl.xz")
META = pathlib.Path(str(JOIN) + ".meta.json")
EVAL_A = {"automobile": 283417, "heavy_truck": 10056, "bus": 3170, "other_vehicle": 1502,
          "trailer": 3276, "person": 59575, "rider": 8840, "stroller": 230, "animal": 171,
          "protruding_object": 1577}
EVAL_B = {"automobile": 296717, "heavy_truck": 7000, "bus": 4789, "other_vehicle": 1431,
          "trailer": 3969, "person": 98562, "rider": 12449, "stroller": 475, "animal": 273,
          "protruding_object": 1450}


def inv_freq(counts):
    """Inverse frequency over the classes PRESENT, normalised to mean 1 (scale is absorbed by the
    loss's own denominator, so an unnormalised comparison would report a shared constant)."""
    pres = {c: n for c, n in counts.items() if n > 0}
    raw = {c: 1.0 / n for c, n in pres.items()}
    m = sum(raw.values()) / len(raw)
    return {c: v / m for c, v in raw.items()}


def main() -> int:
    meta = json.loads(META.read_text(encoding="utf-8"))
    s = meta["summary"]
    want = (int(s["n_episodes"]), int(s["n_frames"]), int(s["n_agent_boxes"]))

    cls = collections.Counter()
    clips, frames, boxes, no_cls = set(), 0, 0, 0
    with lzma.open(JOIN, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            frames += 1
            clips.add(str(d.get("clip_id", "")))
            for a in d.get("agents") or []:
                boxes += 1
                c = a.get("cls")
                if c is None:
                    no_cls += 1
                else:
                    cls[str(c)] += 1

    got = (len(clips), frames, boxes)
    ok = got == want
    print(f"meta says  episodes/frames/boxes = {want}")
    print(f"I counted                        = {got}   MATCH={ok}")
    assert ok, ("ZZABORT the census does not reproduce the join's own summary; a partial read "
                "would give a plausible but wrong weight vector")
    assert no_cls == 0, f"ZZABORT {no_cls} boxes carry no cls"

    counts = {c: int(cls.get(c, 0)) for c in AGENT_CLASSES}
    unknown = {k: v for k, v in cls.items() if k not in AGENT_CLASSES}
    w = inv_freq(counts)
    wa, wb = inv_freq(EVAL_A), inv_freq(EVAL_B)

    tot = sum(counts.values())
    maj = max(counts, key=counts.get)
    rare = min((c for c in counts if counts[c]), key=lambda c: counts[c])
    res = {"_what": "the TRAIN class census and the inverse-frequency weight vector for H-BOXCLS-1",
           "_evidence_class": "MEASURED (ours), CPU, read-only over the canonical train join",
           "_source": str(JOIN), "_source_built": "2026-09-05",
           "_control": {"meta_summary": want, "counted": got, "match": ok,
                        "_why": "a partial read would produce a plausible weight vector from a "
                                "partial corpus"},
           "n_episodes": len(clips), "n_frames": frames, "n_boxes": tot,
           "boxes_without_cls": no_cls, "unknown_classes": unknown,
           "counts": counts,
           "share": {c: round(counts[c] / tot, 6) for c in AGENT_CLASSES},
           "imbalance_majority_to_rarest": round(counts[maj] / counts[rare], 1),
           "majority": maj, "rarest": rare,
           "WEIGHT_VECTOR_inv_freq_mean1": {c: round(w[c], 6) for c in sorted(w)},
           "classes_absent_in_train": [c for c in AGENT_CLASSES if counts[c] == 0]}

    shared = sorted(set(w) & set(wa) & set(wb))
    rat = {c: {"eval_halfA/train": round(wa[c] / w[c], 3),
               "eval_halfB/train": round(wb[c] / w[c], 3)} for c in shared}
    worst = max(shared, key=lambda c: max(abs(math.log(wa[c] / w[c])),
                                          abs(math.log(wb[c] / w[c]))))
    res["eval_vs_train"] = {
        "_question": "would a held-out EVAL split have been an acceptable proxy after all?",
        "ratios": rat, "worst_class": worst,
        "max_ratio": round(max(max(wa[c] / w[c], w[c] / wa[c], wb[c] / w[c], w[c] / wb[c])
                               for c in shared), 3)}
    res["_VERDICT"] = (
        f"TRAIN weight vector MEASURED on {tot:,} boxes over {len(clips):,} episodes; imbalance "
        f"{res['imbalance_majority_to_rarest']}:1 ({maj} : {rare}). Eval-vs-train disagreement "
        f"reaches {res['eval_vs_train']['max_ratio']}x on {worst} \u2014 reported, not assumed.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("train_cls_census.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
