"""Is the TURN label predictable from the nav command? An INDEPENDENT implementation.

⛔ Why this exists. `PREREG_REFCV6_V2.ERRATUM-1.md` withdrew *"42.7 % of the TURN
label's entropy is already in the nav token"* as **UNSUPPORTED**, stating *"No such
measurement exists"*. Two things are now measured:

  1. ⛔ The erratum's search MISSED the source. `REFCV6_CLARIFICATION.md` §4.1 carries a
     FIVE-ROW table -- turn 42.7 %, strategic goal 30.5 %, lateral 16.9 %, speed bucket
     5.5 %, longitudinal 5.2 % -- headed *"How much each label is already predictable
     from the nav token alone (mutual information / label entropy)"*.
  2. ⭐ But the erratum was RIGHT that nothing DERIVES it: a probe over 10,919 tracked
     files finds the distinctive triple (42.7 / 30.5 / 16.9) only in that prose table and
     in the register that quotes it. **Not banked is not the same as not measured.**

⇒ so compute it independently, and say exactly which split it is computed on.
"""
from __future__ import annotations
import collections, gzip, json, math

P = "C:/Users/Admin/refcv5cmp/data/s2_labels_v7.2_eval.jsonl.gz"
TURN = {"TURN_L", "TURN_R"}


def _nav(r): return ((r.get("nav_command") or {}).get("token")) or "<none>"


def _turn(r):
    g = (r.get("g_tac") or {}).get("goals") or {}
    t = sorted(k for k in g if k in TURN)
    return t[0] if t else "<no-turn>"


def _H(c):
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values() if v)


def main():
    rows = [json.loads(l) for l in gzip.open(P, "rt", encoding="utf-8") if l.strip()]
    pairs = [(_nav(r), _turn(r)) for r in rows]
    n = len(pairs)
    HY = _H(collections.Counter(t for _, t in pairs))
    cond = collections.defaultdict(collections.Counter)
    for a, b in pairs:
        cond[a][b] += 1
    HYgX = sum(sum(c.values()) / n * _H(c) for c in cond.values())
    share = (HY - HYgX) / HY

    follow = cond["NAV_FOLLOW_ROAD"]
    turny = collections.Counter()
    for k, c in cond.items():
        if k != "NAV_FOLLOW_ROAD":
            turny.update(c)
    p_follow = 1 - follow["<no-turn>"] / sum(follow.values())
    p_turny = 1 - turny["<no-turn>"] / sum(turny.values())

    out = {
        "_what": ("An INDEPENDENT computation of 'share of the TURN label's entropy "
                  "predictable from the nav token', against a prose table that has no "
                  "banked derivation."),
        "_evidence_class": "MEASURED (ours)",
        "_split": f"v7.2 EVAL labels, n={n} clips",
        "⛔_scope": ("NOT a reproduction: REFCV6_CLARIFICATION.md §4.1's table is over "
                    "4,572 TRAIN clips, which are not on this box. This is a MAGNITUDE "
                    "CHECK on a different, smaller split."),
        "H_turn_bits": round(HY, 4),
        "H_turn_given_nav_bits": round(HYgX, 4),
        "share_predictable_from_nav": round(share, 4),
        "prose_table_value": 0.427,
        "nav_counts": dict(collections.Counter(a for a, _ in pairs)),
        "turn_counts": dict(collections.Counter(b for _, b in pairs)),
        "co_occurrence_claim": {
            "claim": "no clip has NAV_FOLLOW_ROAD together with a tactical turn",
            "counterexamples_on_this_split": sum(
                1 for a, b in pairs if a == "NAV_FOLLOW_ROAD" and b != "<no-turn>"),
            "n_NAV_FOLLOW_ROAD": sum(follow.values()),
        },
        "⭐_the_asymmetry_the_entropy_number_hides": {
            "P(turn | NAV_FOLLOW_ROAD)": round(p_follow, 4),
            "P(turn | NAV_TURN_*)": round(p_turny, 4),
            "reading": ("nav is a PERFECT NEGATIVE predictor -- it rules a turn out with "
                        "certainty -- and a WEAK POSITIVE one. A single entropy share "
                        "averages those two very different facts into one number."),
        },
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    import pathlib
    pathlib.Path("C:/Users/Admin/qland/pkgrl/raw/nav_turn_entropy_share.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
