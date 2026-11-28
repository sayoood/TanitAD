"""EGOWM, the IA-0 half of the card's read (PREREG_V8F_EGOWM.md, frozen sha256 ddd6fc9b...f446e7). Master Mind, 2026-11-10 ~23:05, written
before any speed arm's read existed. Tier T0.

Card rules applied here, and nothing else:
  * deciding per level: IA-0 skill against Copy-Last at the band end (L1 2.0 s, L2 6.0 s, L3 12 s), by IMAG1B's two-part rule (the read's
    ``passes[level]``: skill, 95 % episode-cluster CI, beats / worse, healthy);
  * the floor per level = the largest |skill difference| among E-KIN at seeds 0, 1 and 2 (E-KIN_rep_s2);
  * an effect counts only if (arm skill - R35H skill, same seed) exceeds the floor of its level at BOTH seeds 0 and 1;
  * IA-0 half of the outcomes: skill effect at L1 above the floor at both seeds -> the planner read decides SPEED-HELPS vs SKILL-ONLY;
    otherwise NO-EFFECT. The planner half is not read here.
Without E-KIN_rep_s2 the floor does not exist and every effect reads PENDING-FLOOR (the card: no effect is read without it).
Usage: python verdict_egowm_ia0.py [--out VERDICT_EGOWM_IA0.json]
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = os.path.normpath(os.path.join(HERE, "..", "..", "ladder_read", "results"))
LV = ("1", "2", "3")
BAND_S = {"1": 2.0, "2": 6.0, "3": 12.0}


def load(name):
    p = os.path.join(LADDER if name.startswith("R35H") else os.path.join(HERE, "results"), f"arm_{name}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    assert not d.get("smoke"), f"{name}: a smoke read is not admissible"
    return d


def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "VERDICT_EGOWM_IA0.json")
    runs = {n: load(n) for n in ("R35H_s0", "R35H_s1", "E-KIN_s0", "E-KIN_s1", "E-KIN_rep_s2", "E-NOW_s0", "E-NOW_s1")}
    missing = [n for n, r in runs.items() if r is None]
    sk = {n: {lv: r["passes"][lv]["skill"] for lv in LV} for n, r in runs.items() if r is not None}
    kin = [n for n in ("E-KIN_s0", "E-KIN_s1", "E-KIN_rep_s2") if n in sk]
    floor = None
    if len(kin) == 3:
        floor = {lv: max(abs(sk[a][lv] - sk[b][lv]) for a, b in itertools.combinations(kin, 2)) for lv in LV}
    res = dict(card="PREREG_V8F_EGOWM.md", card_sha256_prefix="ddd6fc9b", tier="T0", missing=missing, floor=floor,
               floor_rule="max |skill difference| among E-KIN s0, s1, rep_s2, per level", per_run={}, arms={})
    for n, r in runs.items():
        if r is not None:
            res["per_run"][n] = {lv: dict(r["passes"][lv], band_end_s=BAND_S[lv]) for lv in LV}
    for arm in ("E-KIN", "E-NOW"):
        a = {}
        for lv in LV:
            d = {s: (sk[f"{arm}_s{s}"][lv] - sk[f"R35H_s{s}"][lv]) if (f"{arm}_s{s}" in sk and f"R35H_s{s}" in sk) else None for s in (0, 1)}
            if floor is None or None in d.values():
                state = "PENDING-FLOOR" if floor is None else "PENDING-RUN"
            else:
                above = [abs(x) > floor[lv] for x in d.values()]
                same_sign = (d[0] > 0) == (d[1] > 0)
                state = ("GAIN" if d[0] > 0 else "LOSS") if (all(above) and same_sign) else "WITHIN-FLOOR" if not any(above) else "SEEDS-DISAGREE"
            a[lv] = dict(delta_vs_R35H={str(s): x for s, x in d.items()}, floor=None if floor is None else floor[lv], state=state)
        l1 = a["1"]["state"]
        a["ia0_half"] = ("SKILL-RISES (planner read decides SPEED-HELPS vs SKILL-ONLY)" if l1 == "GAIN" else
                         "PENDING" if l1.startswith("PENDING") else "NO-EFFECT")
        res["arms"][arm] = a
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    for arm, a in res["arms"].items():
        print(arm, a["ia0_half"], {lv: (a[lv]["delta_vs_R35H"], a[lv]["state"]) for lv in LV})
    print("floor", floor, "missing", missing)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
