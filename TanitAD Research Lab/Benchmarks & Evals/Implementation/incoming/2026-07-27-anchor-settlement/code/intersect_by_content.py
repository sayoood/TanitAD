"""Intersect two comma2k19 episode caches BY CONTENT and settle the anchor's
train/eval overlap.

Consumes the two artefacts produced by `fingerprint_comma_cache.py`:
  * `fp_61c46fca8f7f.json`  — dev box, 90 eps: `idm_head_v1` ("A0") trained on
    `cm_[0:40]`, held out `cm_[40:70]`, never used `cm_[70:90]`
  * `fp_76b6e94a97a1.json`  — tanitad-eval, 64 eps: the IDM-v3 substrate the
    `+0.0114 -> +0.3308` ANCHOR lives on, split every-3rd-episode into
    val (`cm_00000, cm_00003, …, cm_00063`, 22 eps) and train (42 eps)

PRE-REGISTRATION (from the brief, fixed BEFORE the probe ran):
  overlap == 0  =>  the anchor is clean and +0.3308 stands ON ITS SUBSTRATE
  overlap >  0  =>  the anchor is CONTAMINATED and every claim resting on it —
                    including the lifted disqualification — must be withdrawn or
                    re-measured.  ⛔ A contaminated result is NOT softened into
                    "partially valid".

⛔ Names are never the evidence.  Five independent content hash families are
intersected and must agree; `episode_id` (a sha1 OF A PATH) and the `ep_XXXXX`
filenames are reported as cross-checks that could have disagreed.

Also runs the checks that this program has been burned by not running:
  * WITHIN-cache duplicates (the same segment cached twice under two names)
  * A0 train x A0 heldout self-overlap inside 61c (is the C42 re-score's
    "episode_id-disjoint" claim true BY CONTENT?)
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

HASHES = ("poses_sha256", "poses_xy_sha256", "poses_yaw_sha256",
          "poses_v_sha256", "actions_sha256", "frames_sha256")
PRIMARY = "poses_sha256"          # the brief's named primary
SENSOR = "frames_sha256"          # raw pixels — independent of every label protocol


def idx(tag: str) -> int:
    return int(tag.split("_")[1])


def roles_61c(tags):
    return {"A0_TRAIN_cm_0_40": [t for t in tags if idx(t) < 40],
            "A0_HELDOUT_cm_40_70": [t for t in tags if 40 <= idx(t) < 70],
            "A0_UNUSED_cm_70_90": [t for t in tags if idx(t) >= 70]}


def roles_76b(tags, val_every=3):
    return {"V3_VAL_the_anchor": [t for t in tags if idx(t) % val_every == 0],
            "V3_TRAIN": [t for t in tags if idx(t) % val_every != 0]}


def dup_check(eps: dict, h: str) -> dict:
    seen: dict[str, list[str]] = {}
    for k, v in eps.items():
        seen.setdefault(v[h], []).append(k)
    dups = {kk: v for kk, v in seen.items() if len(v) > 1}
    return {"n_distinct": len(seen), "n_episodes": len(eps),
            "duplicate_groups": [sorted(v) for v in dups.values()]}


def match(ea, eb, ka, kb, h) -> list[tuple[str, str]]:
    ma = {ea[k][h]: k for k in ka}
    mb = {eb[k][h]: k for k in kb}
    return sorted((ma[x], mb[x]) for x in (set(ma) & set(mb)))


def poses(e) -> np.ndarray:
    return np.frombuffer(base64.b64decode(e["poses_b64"]), np.float32).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="fingerprints of 61c46fca8f7f")
    ap.add_argument("--b", required=True, help="fingerprints of 76b6e94a97a1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = json.loads(Path(args.a).read_text(encoding="utf-8"))
    B = json.loads(Path(args.b).read_text(encoding="utf-8"))
    ea, eb = A["episodes"], B["episodes"]
    ra, rb = roles_61c(sorted(ea)), roles_76b(sorted(eb))

    res = {
        "what": "content-based overlap between the ANCHOR's evaluation set and "
                "idm_head_v1's comma TRAINING set",
        "date": "2026-07-27",
        "agent": "anchor-settlement",
        "evidence_class": "MEASURED (ours; sha256 of raw bytes on both hosts)",
        "tier": "decision-grade — the comparison is over raw bytes, so it has no "
                "dependence on any naming convention or tag index",
        "pre_registration": {
            "fixed_before_the_probe_ran": True,
            "source": "the brief, verbatim",
            "overlap_eq_0": "the anchor is clean and the +0.3308 stands ON ITS "
                            "SUBSTRATE",
            "overlap_gt_0": "the anchor is CONTAMINATED and every claim resting "
                            "on it — including the lifted disqualification — must "
                            "be withdrawn or re-measured. NOT softened into "
                            "'partially valid'.",
        },
        "method": {
            "primary": "sha256 of the raw `poses` [T,4] float32 bytes",
            "corroborating": list(HASHES),
            "sensor_level": "sha256 of the raw `frames_u8` [T,9,256,256] uint8 "
                            "bytes — independent of every label protocol",
            "name_derived_CROSS_CHECK_ONLY": [
                "episode_id (= sha1 of '<route>/<segment>', a hash OF A PATH)",
                "the ep_XXXXX.pt filename",
                "the cm_XXXXX tag index"],
            "why": "600/600 filename overlap with 0/600 real overlap has been "
                   "measured in this program; and 4 of 36 val episodes would "
                   "have leaked in a cache with no episode_id.",
        },
        "sets": {
            "A": {"cache": A["cache_key"], "path": A["cache"], "host": A["host"],
                  "n": A["n_fingerprinted"],
                  "roles": {k: len(v) for k, v in ra.items()}},
            "B": {"cache": B["cache_key"], "path": B["cache"], "host": B["host"],
                  "n": B["n_fingerprinted"],
                  "roles": {k: len(v) for k, v in rb.items()}},
        },
        "hash_family_agreement": {},
        "within_cache_duplicates": {
            A["cache_key"]: dup_check(ea, PRIMARY),
            B["cache_key"]: dup_check(eb, PRIMARY),
        },
        "cross_tab": {},
        "THE_ANSWER": {},
    }

    # every hash family must give the same global intersection, or the probe is
    # not trustworthy and says so
    for h in HASHES:
        sa = {ea[k][h] for k in ea}; sb = {eb[k][h] for k in eb}
        res["hash_family_agreement"][h] = len(sa & sb)
    ida = {ea[k]["episode_id"] for k in ea}; idb = {eb[k]["episode_id"] for k in eb}
    res["hash_family_agreement"]["episode_id_NAME_DERIVED"] = len(ida & idb)
    agree = {v for k, v in res["hash_family_agreement"].items()}
    res["hash_family_agreement"]["all_families_agree"] = (len(agree) == 1)

    # the full cross-tabulation, by content
    for na, ka in ra.items():
        for nb, kb in rb.items():
            m = match(ea, eb, ka, kb, PRIMARY)
            m_sensor = match(ea, eb, ka, kb, SENSOR)
            res["cross_tab"][f"{na} x {nb}"] = {
                "n_by_poses_sha256": len(m),
                "n_by_frames_sha256": len(m_sensor),
                "pairs": [{"a": x, "b": y,
                           "a_path": f"{A['cache']}/{x}.pt",
                           "b_path": f"{B['cache']}/{y}.pt",
                           "poses_sha256": ea[x][PRIMARY],
                           "frames_sha256": ea[x][SENSOR],
                           "episode_id": ea[x]["episode_id"],
                           "poses_bitwise_equal": bool(np.array_equal(
                               poses(ea[x]), poses(eb[y]))),
                           "v_max": ea[x]["v_max"],
                           "path_len_m": ea[x]["path_len_m"]} for x, y in m],
            }

    key = "A0_TRAIN_cm_0_40 x V3_VAL_the_anchor"
    leak = res["cross_tab"][key]
    n_leak = leak["n_by_poses_sha256"]
    res["THE_ANSWER"] = {
        "question": "how many of the ANCHOR's 22 comma evaluation episodes are, "
                    "BY CONTENT, also in idm_head_v1's 40 comma TRAINING episodes?",
        "n_overlap": n_leak,
        "n_anchor_val_episodes": len(rb["V3_VAL_the_anchor"]),
        "frac_of_anchor_val_episodes": n_leak / max(len(rb["V3_VAL_the_anchor"]), 1),
        "contaminated_anchor_tags": [p["b"] for p in leak["pairs"]],
        "their_training_counterparts": [p["a"] for p in leak["pairs"]],
        "verdict": ("CONTAMINATED — pre-registered outcome 2" if n_leak
                    else "CLEAN — pre-registered outcome 1"),
    }

    # --- self-overlap INSIDE 61c: is the C42 re-score's substrate really --- #
    # --- disjoint from A0's training clips, BY CONTENT?                   --- #
    res["self_overlap_61c"] = {
        "what": "A0_TRAIN x A0_HELDOUT inside the SAME cache, by content — the "
                "C42 re-score asserted `episode_id`-disjointness, which is a "
                "NAME-derived claim; this is the content one",
        "n_by_poses_sha256": len(match(ea, ea, ra["A0_TRAIN_cm_0_40"],
                                       ra["A0_HELDOUT_cm_40_70"], PRIMARY)),
        "n_by_frames_sha256": len(match(ea, ea, ra["A0_TRAIN_cm_0_40"],
                                        ra["A0_HELDOUT_cm_40_70"], SENSOR)),
    }

    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("THE_ANSWER", "hash_family_agreement",
                       "within_cache_duplicates", "self_overlap_61c")}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
