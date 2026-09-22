"""Is a class-frequency weight vector STABLE across clip samples, or does it need the TRAIN join?

⛔ THE BLOCKER THIS ANSWERS. `H-BOXCLS-1` needs an inverse-frequency weight over the 10 agent
classes, and `58ee9b5` landed the capability while stating plainly that the vector must come from
the **TRAIN** split. Two probes say no train agent join exists on any machine reachable from here
(dev box: only `b1eval_agents_3d.jsonl.xz`; Thor: nothing). Building one is a POD-SIDE step over the
2,376-episode parity corpus (`build_obstacle_join.py`), i.e. compute the PI provisions.

⛔ AND THE EVAL CENSUS IS NOT A SUBSTITUTE BY DEFAULT. Setting a TRAINING hyper-parameter from the
split the arm is SCORED on is the leakage the programme's own probe discipline forbids
(*"fit every hyper-parameter on the FIT split only; the scored split is scored, never tuned on"*).

⭐ BUT THE QUESTION UNDERNEATH IS MEASURABLE FOR FREE, AND IT DECIDES WHETHER THE BLOCKER IS REAL.
The eval corpus is already cut into two DISJOINT, clip-level halves — halfA (62 clips) and halfB
(62) — by `a0_clean124_split.py`, alternating on sorted sha12. Computing the class census on each
half separately asks:

  the two halves' weight vectors AGREE   -> a class distribution is stable under a 62-clip draw, so
        a held-out split is a defensible PROXY and the disagreement BOUNDS the error it costs
  the two halves DISAGREE materially     -> the vector is sampling-sensitive, a proxy would be a
        number with no evidence class, and the TRAIN join is genuinely required

⛔ Either way this is reported as a measurement about SAMPLING STABILITY, never as the train
distribution. halfA and halfB are both EVAL; agreement between them says a 62-clip sample is
sufficient, NOT that eval resembles train. Those are different claims and only the first is tested.

⚠️ The comparison is on the WEIGHT VECTOR, not the raw counts, because that is what the arm would
consume: inverse frequency, normalised to mean 1 so a uniform shift cannot masquerade as agreement.
⛔ CPU only, read-only, no GPU.
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
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
from tanitad.models.agent_slots import AGENT_CLASSES          # noqa: E402
from s1_pass import sha12                                     # noqa: E402

JOIN = ("D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/"
        "b1eval_agents_3d.jsonl.xz")
STAMP = "C:/Users/Admin/qland/a0_clean124_stamp.json"


def inv_freq(counts: dict[str, int]) -> dict[str, float]:
    """Inverse frequency over the classes PRESENT, normalised to mean 1.

    ⛔ Normalised on purpose: an unnormalised vector's scale is absorbed by the loss's own
    denominator (`58ee9b5`), so comparing unnormalised vectors would report agreement that is
    really just a shared constant.
    """
    present = {c: n for c, n in counts.items() if n > 0}
    if not present:
        return {}
    raw = {c: 1.0 / n for c, n in present.items()}
    m = sum(raw.values()) / len(raw)
    return {c: v / m for c, v in raw.items()}


def main() -> int:
    st = json.load(open(STAMP, encoding="utf-8"))
    halves = {h: set(st["halves_sha12"][h]) for h in ("halfA", "halfB")}
    assert len(halves["halfA"]) == 62 and len(halves["halfB"]) == 62
    assert not (halves["halfA"] & halves["halfB"]), "the halves are not disjoint"

    counts = {h: collections.Counter() for h in halves}
    counts["_unassigned"] = collections.Counter()
    n_lines = n_agents = 0
    seen_clips = set()
    with lzma.open(JOIN, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n_lines += 1
            # ⛔ THE JOIN CARRIES A FULL UUID; THE SPLIT IS KEYED ON sha12.
            # MEASURED: a first pass compared them directly and left ALL 905,512
            # agents unassigned -- caught only because `n_agents_unassigned_to_a_half`
            # is reported, not because anything looked wrong. The hash is the
            # programme's OWN `s1_pass.sha12`, never a re-implementation.
            cid = sha12(str(d.get("clip_id", "")))
            seen_clips.add(cid)
            which = ("halfA" if cid in halves["halfA"] else
                     "halfB" if cid in halves["halfB"] else "_unassigned")
            for a in d.get("agents") or []:
                n_agents += 1
                counts[which][str(a.get("cls"))] += 1

    # ⛔ THE JOIN HAS 139 CLIPS; THE CLEAN SPLIT HAS 124. The 15 outside it are the
    # stamp's own exclusions ("139 B1 eval clips - 11 inside parity TRAIN - 4 with no SAM3
    # map"), so agents outside a half are EXPECTED and are counted separately rather than
    # silently dropped. ⚠️ The guard therefore asserts the STRUCTURE (exactly 15 clips
    # outside, 139 seen) rather than zero, which still catches a wrong clip key -- the first
    # pass left ALL 905,512 unassigned and only this counter revealed it.
    n_out = len(seen_clips - halves["halfA"] - halves["halfB"])
    assert len(seen_clips) == 139 and n_out == 15, (
        f"ZZABORT {len(seen_clips)} clips seen, {n_out} outside the 124-clip split - "
        f"expected 139 and 15; the clip key is wrong and every number below is meaningless")
    res = {"_what": "is an inverse-frequency class weight STABLE under a 62-clip draw?",
           "_evidence_class": "MEASURED (ours), CPU, read-only over the EVAL 3-D join",
           "_scope": ("halfA and halfB are BOTH eval. Agreement says a 62-clip sample suffices; "
                      "it does NOT say eval resembles TRAIN. Only the first is tested here."),
           "_blocker": ("no TRAIN agent join exists on the dev box or on Thor (two probes); "
                        "building one is a pod-side step over the 2,376-episode parity corpus"),
           "n_lines": n_lines, "n_agents": n_agents,
           "n_clips_seen": len(seen_clips),
           "n_clips_outside_the_124_split": n_out,
           "n_agents_outside_the_124_split": sum(counts["_unassigned"].values()),
           "_outside_is_expected": ("the stamp excludes 11 clips inside parity TRAIN and 4 with "
                                    "no SAM3 map; they are counted apart, never folded in"),
           "counts": {h: {c: int(counts[h][c]) for c in AGENT_CLASSES} for h in halves}}

    wa, wb = inv_freq(res["counts"]["halfA"]), inv_freq(res["counts"]["halfB"])
    shared = sorted(set(wa) & set(wb))
    res["weight_vectors"] = {"halfA": {c: round(wa[c], 5) for c in sorted(wa)},
                             "halfB": {c: round(wb[c], 5) for c in sorted(wb)}}
    res["classes_present_in_only_one_half"] = sorted(set(wa) ^ set(wb))
    if shared:
        ratios = {c: wa[c] / wb[c] for c in shared}
        worst = max(ratios, key=lambda c: abs(math.log(ratios[c])))
        res["agreement"] = {
            "_statistic": "per-class ratio of the two halves' normalised weights (1.0 = identical)",
            "ratios": {c: round(ratios[c], 4) for c in shared},
            "max_abs_log_ratio_class": worst,
            "max_ratio": round(max(max(ratios.values()), 1 / min(ratios.values())), 4)}
        stable = res["agreement"]["max_ratio"] <= 1.5
        res["_VERDICT"] = (
            f"⭐ STABLE UNDER A 62-CLIP DRAW — the two halves' weight vectors agree to within "
            f"{res['agreement']['max_ratio']}x (worst class: {worst}). A held-out split is a "
            f"defensible PROXY and this ratio BOUNDS what it costs. ⚠️ Still says nothing about "
            f"whether EVAL resembles TRAIN."
            if stable else
            f"⛔ SAMPLING-SENSITIVE — the two halves disagree by up to "
            f"{res['agreement']['max_ratio']}x (worst class: {worst}) on a 62-clip draw. A proxy "
            f"vector would be a number with no evidence class ⇒ the TRAIN join is genuinely "
            f"REQUIRED, and that is a pod-side build.")
    else:
        res["_VERDICT"] = "⛔ no class present in both halves — the split or the parse is wrong"
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("cls_freq_split.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
