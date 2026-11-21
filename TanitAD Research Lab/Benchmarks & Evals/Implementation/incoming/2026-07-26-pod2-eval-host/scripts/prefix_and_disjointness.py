"""Byte-level disjointness + ORDER-PRESERVING PREFIX check, from scratch.

Independent of the S3 agent's run: the manifests here were produced on pod2's
own copy of the parity train (not pod3's) and on the eval pod's published 40.

Two verdicts:
  A. DISJOINTNESS -- |sha256(poses) of val 600  n  sha256(poses) of parity train|
     must be 0. episode_id is reported alongside ONLY to show it over-reports.
  B. PREFIX -- the published 40 must equal val600[:40] elementwise IN ORDER.
     Set equality is NOT enough: an order-preserving prefix is what makes
     "40 -> 600 adds episodes and re-selects none" true.
"""
import json
import sys
from pathlib import Path

SC = Path(sys.argv[1])


def load(fn):
    return json.loads((SC / fn).read_text())


val = load("verify_val600_pod2.json")
tra = load("verify_train_pod2.json")
pub = load("verify_evalpod40.json")

v_h = [e["poses_sha256"] for e in val["episodes"]]
t_h = [e["poses_sha256"] for e in tra["episodes"]]
p_h = [e["poses_sha256"] for e in pub["episodes"]]
v_e = [e["episode_id"] for e in val["episodes"]]
t_e = [e["episode_id"] for e in tra["episodes"]]

t_hset = set(t_h)
byte_ovl = [h for h in v_h if h in t_hset]
eid_ovl = [e for e in v_e if e in set(t_e)]

# ---- B. prefix, ORDER-PRESERVING ----
first40 = v_h[:40]
prefix_exact = (p_h == first40)
per_index = [{"i": i, "published_file": pub["episodes"][i]["file"],
              "val600_file": val["episodes"][i]["file"],
              "sha_match": p_h[i] == first40[i]} for i in range(40)]
n_match = sum(r["sha_match"] for r in per_index)
set_equal = set(p_h) == set(first40)
# is the published set a subset of the 600 at all, and where does each land?
pos = {h: i for i, h in enumerate(v_h)}
positions = [pos.get(h, -1) for h in p_h]
monotone = positions == sorted(positions) and -1 not in positions

out = {
  "method": ("sha256 over the raw poses[T,4] float32 bytes, read via "
             "torch.load(mmap=True). episode_id is NOT a key -- it collides."),
  "hosts": {"val600": val["host"], "parity_train": tra["host"],
            "published40": pub["host"]},
  "sources": {"val600": val["source_dir"], "parity_train": tra["source_dir"],
              "published40": pub["source_dir"]},
  "counts": {
    "val600_n": val["n_episodes_read"],
    "val600_unique_poses_sha256": val["n_unique_poses_sha256"],
    "val600_unique_episode_id": val["n_unique_episode_id"],
    "val600_T_range": [val["T_min"], val["T_max"]],
    "parity_train_n": tra["n_episodes_read"],
    "parity_train_unique_poses_sha256": tra["n_unique_poses_sha256"],
    "parity_train_unique_episode_id": tra["n_unique_episode_id"],
    "parity_train_T_range": [tra["T_min"], tra["T_max"]],
    "published40_n": pub["n_episodes_read"],
    "published40_T_range": [pub["T_min"], pub["T_max"]],
  },
  "A_DISJOINTNESS": {
    "byte_overlap_n": len(byte_ovl),
    "byte_overlap_pct": round(100.0 * len(byte_ovl) / len(v_h), 4),
    "episode_id_overlap_n": len(eid_ovl),
    "episode_id_overlap_pct": round(100.0 * len(eid_ovl) / len(v_e), 4),
    "verdict": ("CLEAN -- 0 byte overlap" if not byte_ovl
                else f"CONTAMINATED -- {len(byte_ovl)} episodes"),
    "note": ("episode_id over-reports: the parity train holds "
             f"{tra['n_episodes_read']} episodes but only "
             f"{tra['n_unique_episode_id']} unique episode_ids "
             f"({tra['n_episodes_read'] - tra['n_unique_episode_id']} collisions)."),
  },
  "B_PREFIX": {
    "published40_equals_val600_first40_IN_ORDER": prefix_exact,
    "n_positions_matching": n_match,
    "set_equality_only": set_equal,
    "published_positions_within_val600": positions,
    "positions_are_0_to_39_in_order": positions == list(range(40)),
    "monotone_and_all_present": monotone,
    "verdict": ("ORDER-PRESERVING PREFIX CONFIRMED: the published 40 are "
                "val600[0:40] element-for-element, so 40 -> 600 ADDS episodes "
                "and RE-SELECTS none. Parity holds."
                if prefix_exact else
                "PREFIX FAILED -- moving to 600 would re-select episodes. REFUSE."),
    "per_index": per_index,
  },
}
(SC / "prefix_disjointness_result.json").write_text(json.dumps(out, indent=1))

print("A. DISJOINTNESS  val600 vs parity train (pod2's own copy)")
print(f"   byte overlap      : {len(byte_ovl)}/600  ({out['A_DISJOINTNESS']['byte_overlap_pct']} %)")
print(f"   episode_id overlap: {len(eid_ovl)}/600  ({out['A_DISJOINTNESS']['episode_id_overlap_pct']} %)  <- COLLIDING KEY, not leakage")
print(f"   train: n={tra['n_episodes_read']} uniq_sha={tra['n_unique_poses_sha256']} uniq_eid={tra['n_unique_episode_id']}")
print(f"   VERDICT: {out['A_DISJOINTNESS']['verdict']}")
print()
print("B. PREFIX  published 40 (eval pod) vs val600[:40] (pod2)")
print(f"   exact, in order   : {prefix_exact}   ({n_match}/40 positions match)")
print(f"   set equality only : {set_equal}")
print(f"   positions in 600  : {positions[:8]} ... {positions[-3:]}")
print(f"   VERDICT: {out['B_PREFIX']['verdict']}")
