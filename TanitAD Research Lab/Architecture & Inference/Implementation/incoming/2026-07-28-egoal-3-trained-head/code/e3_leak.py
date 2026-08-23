#!/usr/bin/env python3
"""E-GOAL-3 S0 -- THE LEAK / OVERLAP CHECK, BY CONTENT, ON THE PATHS I READ.

⚠️ This matters MORE here than in E-GOAL-2, because E-GOAL-3 actually TRAINS on
the parity train corpus. A leaked head is worse than no head.

⭐ THE FINGERPRINTS ARE NOT A SECOND READ. `e3_features.py` computes the sha256
of `poses[T,4]`'s raw float32 bytes FROM THE SAME IN-MEMORY TENSOR it derives
the features from, so this checks the bytes actually used -- not a different
copy of nominally the same cache. (E-GOAL-2's own reported gap was that a
sibling's audit covered `_epcache` while the fan dump read `/root/valdata` on
the same pod: an audit of a different copy is not an audit of this one.)

WHAT IS CHECKED
  A  train2376 (my T-TRAIN training set)  x  val600 (my scoring set)  -> 0
  B  filename overlap, the SAME comparison -- reported as the contrast that
     shows the fingerprint is load-bearing (E-GOAL-2 measured 600/600 filename
     overlap against 0/600 real overlap)
  C  internal collisions within each corpus
  D  ⭐ independent cross-check against E-GOAL-2's own `e2_leak.json`
     fingerprints: a DIFFERENT script, on a different day, must produce the same
     hashes for the same episodes. This validates my read path as well as the
     overlap.
  E  the path the fan dump reads (`refc_rerank.VAL`) == the path I read

Run (dev box; consumes the JSONs produced on pod2):
    python e3_leak.py --val <e3_features_val.json> --train <e3_features_train.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAM = HERE.parent
EG2 = STREAM.parent / "2026-07-28-egoal-2-power"

#: `refc_rerank.VAL` -- the path the committed fan dump reads, verbatim
FAN_DUMP_VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default=str(STREAM / "raw" / "e3_leak.json"))
    a = ap.parse_args()

    V = json.loads(Path(a.val).read_text())
    T = json.loads(Path(a.train).read_text())
    vfp = V["pose_sha256_per_episode"]
    tfp = T["pose_sha256_per_episode"]
    vs = {k: v["sha256"] for k, v in vfp.items()}
    ts = {k: v["sha256"] for k, v in tfp.items()}

    content_overlap = sorted(set(vs.values()) & set(ts.values()))
    name_overlap = sorted(set(vs) & set(ts))

    res = {
        "_stream": "2026-07-28-egoal-3-trained-head", "_stage": "S0 leak/overlap",
        "_method": ("sha256 over the raw poses[T,4] float32 bytes, computed by "
                    "e3_features.py ON THE SAME IN-MEMORY TENSOR the features "
                    "are derived from. episode_id is NOT a key (2376 train "
                    "episodes, 2342 unique ids); filenames are NOT a key."),
        "_paths_checked": {
            "scoring_set_val600": V["_root"],
            "training_set_train2376": T["_root"],
            "path_the_committed_fan_dump_reads (refc_rerank.VAL)": FAN_DUMP_VAL},
        "E_path_identity": {
            "_what": ("the val path my feature builder reads must be the path "
                      "the fan dump reads, or the join is across two builds"),
            "mine": V["_root"], "fan_dump": FAN_DUMP_VAL,
            "identical": bool(V["_root"] == FAN_DUMP_VAL)},
        "A_train_x_val_by_CONTENT": {
            "_required": "overlap == 0",
            "n_val": len(vs), "n_train": len(ts),
            "overlap_n": len(content_overlap),
            "overlap_frac_of_val": round(len(content_overlap) / max(len(vs), 1), 6),
            "overlapping_sha256": content_overlap[:20],
            "passes": bool(len(content_overlap) == 0)},
        "B_train_x_val_by_FILENAME": {
            "_what": ("the check the fingerprint replaces -- reported so the "
                      "fingerprint is visibly load-bearing, not decorative"),
            "overlap_n": len(name_overlap),
            "overlap_frac_of_val": round(len(name_overlap) / max(len(vs), 1), 6)},
        "C_internal_collisions": {
            "val600_unique_sha_of_n": [len(set(vs.values())), len(vs)],
            "train2376_unique_sha_of_n": [len(set(ts.values())), len(ts)]},
    }

    # ---- D: independent cross-check against E-GOAL-2's own fingerprints -----
    e2p = EG2 / "raw" / "e2_leak.json"
    if e2p.exists():
        E2 = json.loads(e2p.read_text())

        def find_map(node, want_n):
            """E-GOAL-2 stores its per-episode fingerprints under a few possible
            keys; locate any dict of {filename: {sha256: ...}} of size want_n."""
            if isinstance(node, dict):
                vals = list(node.values())
                if (len(node) == want_n and vals
                        and isinstance(vals[0], dict) and "sha256" in vals[0]):
                    return {k: v["sha256"] for k, v in node.items()}
                if (len(node) == want_n and vals
                        and isinstance(vals[0], str) and len(vals[0]) == 64):
                    return dict(node)
                for v in node.values():
                    r = find_map(v, want_n)
                    if r:
                        return r
            return None

        m600 = find_map(E2, 600)
        m2376 = find_map(E2, 2376)
        d = {}
        for label, mine, theirs in (("val600", vs, m600),
                                    ("train2376", ts, m2376)):
            if not theirs:
                d[label] = {"available": False}
                continue
            common = sorted(set(mine) & set(theirs))
            agree = sum(1 for k in common if mine[k] == theirs[k])
            d[label] = {"available": True, "n_common_filenames": len(common),
                        "n_sha256_agree": agree,
                        "all_agree": bool(agree == len(common) > 0)}
        # E-GOAL-2 stored per-episode fingerprints only for the val side; its
        # train side is an aggregate. Both halves are compared.
        agg = E2.get("A_TRAIN_OVERLAP", {})
        res["D_independent_crosscheck_vs_EGOAL_2"] = {
            "_source": str(e2p),
            "_what": ("a DIFFERENT script (e2_leak.py) on a different run must "
                      "produce the SAME pose fingerprints for the same "
                      "episodes. Validates the read path, not just the overlap."),
            **d,
            "aggregate": {
                "e2": {k: agg.get(k) for k in ("val600_n", "val600_unique_sha",
                                               "train_n", "train_unique_sha",
                                               "overlap_n")},
                "mine": {"val600_n": len(vs),
                         "val600_unique_sha": len(set(vs.values())),
                         "train_n": len(ts),
                         "train_unique_sha": len(set(ts.values())),
                         "overlap_n": len(content_overlap)},
                "identical": bool(
                    agg.get("val600_n") == len(vs)
                    and agg.get("val600_unique_sha") == len(set(vs.values()))
                    and agg.get("train_n") == len(ts)
                    and agg.get("train_unique_sha") == len(set(ts.values()))
                    and agg.get("overlap_n") == len(content_overlap))}}

    res["VERDICT"] = ("CLEAN -- 0 episodes shared by content"
                      if res["A_train_x_val_by_CONTENT"]["passes"]
                      else "LEAK -- REFUSE to quote any number")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if not k.startswith("_")}, indent=1)[:3000], flush=True)
    print(f"-> {a.out}", flush=True)
    if not res["A_train_x_val_by_CONTENT"]["passes"]:
        raise SystemExit("LEAK DETECTED — no number may be quoted")


if __name__ == "__main__":
    main()
