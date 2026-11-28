"""P1a - LANE-CHANGE PREVALENCE IN THE v7.2 LABEL CORPORA (zero GPU).

The question the brief asks first: is LC_L/LC_R *absent from the labels*?
Counted over BOTH blobs the programme actually uses, by explicit md5, with the
denominator named.  Percentages are printed only beside their <n>/<total>.

EVERY vocabulary is printed IN FULL, never just the rows of interest: the
non-zero rows are the same-breath control proving the key was really read.
(The first version of this script counted `g_tac.tokens`, a key that does not
exist, and produced an all-zero goal table that looked exactly like a finding.)
"""
import collections
import gzip
import hashlib
import json
import sys

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BLOBS = {
    "v7.2 EVAL  (the blob refcv4b's published T1 eval used)":
        r"C:\Users\Admin\navcomp\data\s2_labels_v7.2_eval.jsonl.gz",
    "v7.2 TRAIN (the blob the RL/fan work used)":
        r"C:\Users\Admin\rlgate\s2_labels_v7.2_train.jsonl.gz",
}

LC_FAMILY = ("LANE_CHANGE_L", "LANE_CHANGE_R", "MERGE", "OVERTAKE_VEHICLE",
             "GAP_TARGET", "TAKE_EXIT_L", "TAKE_EXIT_R")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def table(title, vocab, cnt, n, flag_set=()):
    print("\n  " + title)
    for t in vocab:
        c = cnt.get(t, 0)
        flag = "   <== LANE-CHANGE FAMILY" if t in flag_set else ""
        print(f"    {t:<30} {c:>6} / {n}   {100.0 * c / n:6.2f} %{flag}")
    off = {k: v for k, v in cnt.items() if k not in vocab}
    if off:
        print(f"    (OFF-VOCABULARY keys seen: {off})")
    nz = sum(1 for t in vocab if cnt.get(t, 0) > 0)
    print(f"    [read-control] {nz} of {len(vocab)} rows are NON-ZERO -> the "
          f"field was really read")


def main():
    from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                         TACTICAL_LON_ACTIONS_V7,
                                         TACTICAL_GOAL_TOKENS_V7,
                                         STRATEGIC_GOAL_TOKENS_V7,
                                         NAV_COMMAND_TOKENS)
    out = {}
    for name, path in BLOBS.items():
        lat, lon = collections.Counter(), collections.Counter()
        goals, strat, nav = (collections.Counter(), collections.Counter(),
                             collections.Counter())
        n = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                n += 1
                lat[(r.get("a_tac") or {}).get("lat")] += 1
                lon[(r.get("a_tac") or {}).get("lon")] += 1
                gt = (r.get("g_tac") or {}).get("goals") or {}
                for g in (gt if isinstance(gt, dict) else list(gt)):
                    goals[g] += 1
                strat[(r.get("g_str") or {}).get("token")] += 1
                nav[(r.get("nav_command") or {}).get("token")] += 1
        print("=" * 78)
        print(name)
        print(f"  file  {path}")
        print(f"  md5   {md5(path)}")
        print(f"  n_records (CLIPS) = {n}")
        table("TACTICAL LATERAL ACTION  a_tac.lat  (the 8-way v7 head's label)",
              TACTICAL_LAT_ACTIONS_V7, lat, n,
              {"LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC"})
        table("TACTICAL LONGITUDINAL ACTION  a_tac.lon",
              TACTICAL_LON_ACTIONS_V7, lon, n)
        table("TACTICAL GOAL SET  g_tac.goals", TACTICAL_GOAL_TOKENS_V7,
              goals, n, set(LC_FAMILY))
        table("STRATEGIC GOAL  g_str.token", STRATEGIC_GOAL_TOKENS_V7,
              strat, n, {"LANE_CHANGE_L_FOLLOW_ROUTE",
                         "LANE_CHANGE_R_FOLLOW_ROUTE",
                         "EXIT_LEFT_FOLLOW_ROUTE", "EXIT_RIGHT_FOLLOW_ROUTE"})
        table("NAV COMMAND INPUT  nav_command.token", NAV_COMMAND_TOKENS,
              nav, n)
        out[name] = {"file": path, "md5": md5(path), "n_clips": n,
                     "a_tac_lat": dict(lat), "a_tac_lon": dict(lon),
                     "g_tac_goals": dict(goals), "g_str": dict(strat),
                     "nav_command": dict(nav)}
    with open("out_p1a_label_prevalence.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote out_p1a_label_prevalence.json")


if __name__ == "__main__":
    main()
