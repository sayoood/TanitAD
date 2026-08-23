"""Intersect a PhysicalAI VAL cache against the parity TRAIN cache BY CONTENT.

THE QUESTION
------------
`physicalai-val-0c5f7dac3b11` is the substrate under EVERY open-loop and
closed-loop number this program has published.  `MODEL_REGISTRY.md` calls it
"episode-disjoint from train" — an `episode_id` CLAIM, never a content check.

PRE-REGISTRATION (fixed BEFORE any intersection was computed):
  overlap == 0  =>  CLEAN.  The parity split is sound BY CONTENT and the
                    registry's claim is upgraded from `episode_id` to
                    content-verified.
  overlap >  0  =>  LEAKED.  Report the count, the identities, and the FRACTION
                    OF EACH AFFECTED METRIC'S MASS the leaked episodes carry.
                    ⛔ NOT softened into "minor".  A leak in val invalidates
                    GENERALISATION claims, not TRAINING facts, and it affects
                    arms differently depending on which episodes they trained on.

⛔ Names are never the evidence.  Seven independent content hash families are
intersected and must agree; `episode_id` (an integer written by the cache
builder) and the `ep_XXXXX.pt` filename are reported as cross-checks that could
have disagreed — this program has MEASURED 600/600 filename overlap with 0/600
real overlap.

CONTROLS BUILT IN (a leak check that cannot find a leak is worse than none):
  * SELF   — intersect the val set with ITSELF; must return n_val on every
             family.  Proves the matcher can match.
  * SPIKE  — inject K real val records into a COPY of the train set in memory
             and re-run the identical code path; must return exactly K with the
             right identities.  Proves end-to-end detection at the real scale.
  * MUTANT — intersect the val set against a copy of itself with ONE byte of
             the pose tensor flipped; must return 0.  Proves the instrument is
             not trivially colliding.
  * DUPS   — within-cache duplicate scan on both caches.
  * PARTIAL— frame-level (sha1[:16] per frame) containment, which catches the
             leak mode a whole-tensor hash cannot see: a val episode that is a
             SUB-WINDOW or shifted copy of a train clip.

Usage:
    python3 intersect_pai.py --val <val.jsonl> --train <train.jsonl> \
        --out <out.json> [--label ...]
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

FAMILIES_FULL = ("poses_sha256", "poses_xy_sha256", "poses_yaw_sha256",
                 "poses_v_sha256", "actions_sha256", "maneuvers_sha256",
                 "frames_sha256")
FAMILIES_POSES = ("poses_sha256", "poses_xy_sha256", "poses_yaw_sha256",
                  "poses_v_sha256")
PRIMARY = "poses_sha256"
SENSOR = "frames_sha256"


def load(p: str):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    hdr = rows[0]
    assert hdr.get("_header"), f"{p}: first line is not a header"
    eps = {r["file"].replace(".pt", ""): r for r in rows[1:]}
    return hdr, eps


def dup_check(eps: dict, fams) -> dict:
    out = {}
    for h in fams:
        seen: dict[str, list[str]] = {}
        for k, v in eps.items():
            if h in v and v[h] is not None:
                seen.setdefault(v[h], []).append(k)
        dups = {kk: v for kk, v in seen.items() if len(v) > 1}
        out[h] = {"n_distinct": len(seen), "n_episodes": len(eps),
                  "n_duplicate_groups": len(dups),
                  "duplicate_groups": [sorted(v) for v in list(dups.values())[:50]]}
    return out


def match(ea: dict, eb: dict, h: str):
    """All (a_tag, b_tag) pairs sharing hash family `h`. Many-to-many safe."""
    mb: dict[str, list[str]] = {}
    for k, v in eb.items():
        if h in v and v[h] is not None:
            mb.setdefault(v[h], []).append(k)
    pairs = []
    for k, v in ea.items():
        if h in v and v[h] is not None and v[h] in mb:
            for kb in sorted(mb[v[h]]):
                pairs.append((k, kb))
    return sorted(pairs)


def n_matched_a(pairs):
    return len({a for a, _ in pairs})


def poses_arr(e) -> np.ndarray:
    return np.frombuffer(base64.b64decode(e["poses_b64"]), np.float32).reshape(-1, 4)


def partial_frame_overlap(ea: dict, eb: dict) -> dict:
    """Frame-level containment: does any val episode SHARE FRAMES with train
    without being a whole-tensor match (sub-window / shifted copy)?"""
    if not any("frame_digests" in v for v in ea.values()):
        return {"available": False, "why": "val fingerprints carry no frame_digests"}
    if not any("frame_digests" in v for v in eb.values()):
        return {"available": False, "why": "train fingerprints carry no frame_digests"}
    index: dict[str, set] = {}
    for k, v in eb.items():
        for d in v.get("frame_digests", []):
            index.setdefault(d, set()).add(k)
    rows = []
    for k, v in ea.items():
        fd = v.get("frame_digests", [])
        hits = [index.get(d, ()) for d in fd]
        n_any = sum(1 for h in hits if h)
        per_ep: dict[str, int] = {}
        for h in hits:
            for t in h:
                per_ep[t] = per_ep.get(t, 0) + 1
        best = max(per_ep.items(), key=lambda x: x[1]) if per_ep else (None, 0)
        rows.append({"val": k, "T": len(fd),
                     "n_frames_present_in_train_anywhere": n_any,
                     "frac_frames_in_train": n_any / max(len(fd), 1),
                     "best_train_episode": best[0],
                     "best_shared_frames": best[1],
                     "best_frac": best[1] / max(len(fd), 1)})
    rows.sort(key=lambda r: -r["frac_frames_in_train"])
    return {"available": True,
            "n_val_with_any_shared_frame": sum(1 for r in rows if r["n_frames_present_in_train_anywhere"]),
            "max_frac_frames_in_train": max((r["frac_frames_in_train"] for r in rows), default=0.0),
            "top20": rows[:20]}


def run_controls(ea: dict, eb: dict, fams) -> dict:
    ctl = {}
    # --- SELF ------------------------------------------------------------- #
    ctl["SELF"] = {
        "what": "val x val through the identical matcher; MUST equal n_val",
        "n_val": len(ea),
        "per_family": {h: n_matched_a(match(ea, ea, h)) for h in fams},
    }
    ctl["SELF"]["passes"] = all(v == len(ea) for v in ctl["SELF"]["per_family"].values())

    # --- SPIKE ------------------------------------------------------------- #
    k = min(3, len(ea))
    spiked = dict(eb)
    picked = sorted(ea)[:k]
    for i, tag in enumerate(picked):
        spiked[f"__SPIKE_{i}"] = copy.deepcopy(ea[tag])
    pairs = match(ea, spiked, PRIMARY)
    pairs_s = match(ea, spiked, SENSOR) if SENSOR in fams else []
    ctl["SPIKE"] = {
        "what": f"{k} REAL val episodes injected into a COPY of the train set "
                f"in memory (no cache touched); the same matcher MUST find "
                f"exactly {k}, with the right identities",
        "k_injected": k,
        "injected_val_tags": picked,
        "n_found_by_poses_sha256": n_matched_a(pairs),
        "n_found_by_frames_sha256": n_matched_a(pairs_s) if pairs_s or SENSOR in fams else None,
        "identities_recovered": sorted({a for a, _ in pairs}),
    }
    ctl["SPIKE"]["passes"] = (
        n_matched_a(pairs) == k and sorted({a for a, _ in pairs}) == picked
        and (SENSOR not in fams or n_matched_a(pairs_s) == k))

    # --- MUTANT ------------------------------------------------------------ #
    mut = {}
    for tag, v in ea.items():
        w = copy.deepcopy(v)
        raw = bytearray(base64.b64decode(w["poses_b64"]))
        raw[0] ^= 0x01                       # flip ONE bit of ONE float
        w["poses_b64"] = base64.b64encode(bytes(raw)).decode()
        p = np.frombuffer(bytes(raw), np.float32).reshape(-1, 4)
        w["poses_sha256"] = hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
        w["poses_xy_sha256"] = hashlib.sha256(np.ascontiguousarray(p[:, :2]).tobytes()).hexdigest()
        mut["MUT_" + tag] = w
    ctl["MUTANT"] = {
        "what": "val x (val with ONE BIT of the pose tensor flipped); MUST be 0 "
                "— proves the instrument is not matching on something trivial",
        "n_matched_poses_sha256": n_matched_a(match(ea, mut, "poses_sha256")),
        "n_matched_poses_xy_sha256": n_matched_a(match(ea, mut, "poses_xy_sha256")),
    }
    ctl["MUTANT"]["passes"] = (ctl["MUTANT"]["n_matched_poses_sha256"] == 0)
    ctl["ALL_CONTROLS_PASS"] = all(ctl[c].get("passes") for c in ("SELF", "SPIKE", "MUTANT"))
    return ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ha, ea = load(args.val)
    hb, eb = load(args.train)
    full = (ha.get("mode") == "full" and hb.get("mode") == "full")
    fams = FAMILIES_FULL if full else FAMILIES_POSES

    res = {
        "what": "content-based train/val overlap for a PhysicalAI parity split",
        "label": args.label or f"{ha['cache_key']} x {hb['cache_key']}",
        "date": time.strftime("%Y-%m-%d"),
        "agent": "parity-leak-check",
        "evidence_class": "MEASURED (ours; sha256 of raw bytes)",
        "pre_registration": {
            "fixed_before_the_probe_ran": True,
            "overlap_eq_0": "CLEAN — the parity split is sound BY CONTENT; the "
                            "registry's episode-disjoint claim is upgraded from "
                            "an episode_id claim to content-verified.",
            "overlap_gt_0": "LEAKED — report the count, the identities, and the "
                            "fraction of each affected metric's mass the leaked "
                            "episodes carry. NOT softened into 'minor'. Invalidates "
                            "GENERALISATION claims, not TRAINING facts, and hits "
                            "arms differently by which episodes they trained on.",
        },
        "method": {
            "primary": "sha256 of the raw `poses` [T,4] float32 bytes",
            "sensor_level": "sha256 of the raw `frames_u8` [T,9,256,256] uint8 bytes "
                            "— independent of every label protocol",
            "families": list(fams),
            "name_derived_CROSS_CHECK_ONLY": ["episode_id", "ep_XXXXX.pt filename"],
            "why": "600/600 filename overlap with 0/600 real overlap has been "
                   "MEASURED in this program; and 2 of 22 comma eval episodes "
                   "were bit-identical to training episodes, found only by bytes.",
            "mode": {"val": ha.get("mode"), "train": hb.get("mode")},
        },
        "sets": {
            "VAL": {"cache": hb and ha["cache_key"], "path": ha["cache"],
                    "host": ha["host"], "n": len(ea), "mode": ha.get("mode"),
                    "n_in_cache": ha.get("n_episodes_in_cache")},
            "TRAIN": {"cache": hb["cache_key"], "path": hb["cache"],
                      "host": hb["host"], "n": len(eb), "mode": hb.get("mode"),
                      "n_in_cache": hb.get("n_episodes_in_cache")},
        },
    }

    # ---- controls FIRST: an instrument that cannot detect is worse than none #
    res["controls"] = run_controls(ea, eb, fams)

    # ---- hash family agreement ------------------------------------------- #
    agree = {}
    for h in fams:
        agree[h] = n_matched_a(match(ea, eb, h))
    res["hash_family_agreement"] = dict(agree)
    res["hash_family_agreement"]["all_content_families_agree"] = (len(set(agree.values())) == 1)

    # ---- name-derived cross-checks (could have disagreed) ------------------ #
    ida = {v["episode_id"] for v in ea.values() if v.get("episode_id") is not None}
    idb = {v["episode_id"] for v in eb.values() if v.get("episode_id") is not None}
    fa, fb = set(ea), set(eb)
    res["name_derived_cross_checks"] = {
        "episode_id_overlap": len({v["episode_id"] for v in ea.values()
                                   if v.get("episode_id") in idb}),
        "n_val_episode_ids_distinct": len(ida),
        "n_train_episode_ids_distinct": len(idb),
        "filename_overlap": len(fa & fb),
        "n_val_files": len(fa), "n_train_files": len(fb),
        "note": "NOT evidence. Reported so a name/content DISAGREEMENT is visible.",
    }

    # ---- within-cache duplicates ------------------------------------------ #
    res["within_cache_duplicates"] = {
        ha["cache_key"] + " (val)": dup_check(ea, fams),
        hb["cache_key"] + " (train)": dup_check(eb, fams),
    }

    # ---- THE ANSWER -------------------------------------------------------- #
    pairs = match(ea, eb, PRIMARY)
    pairs_sensor = match(ea, eb, SENSOR) if full else []
    leaked = sorted({a for a, _ in pairs})
    detail = []
    for a in leaked:
        bs = sorted({b for x, b in pairs if x == a})
        bit = None
        if "poses_b64" in ea[a] and all("poses_b64" in eb[b] for b in bs):
            bit = bool(np.array_equal(poses_arr(ea[a]), poses_arr(eb[bs[0]])))
        detail.append({
            "val_tag": a, "train_tags": bs,
            "poses_sha256": ea[a]["poses_sha256"],
            "frames_sha256": ea[a].get("frames_sha256"),
            "poses_bitwise_equal": bit,
            "episode_id_val": ea[a].get("episode_id"),
            "episode_id_train": [eb[b].get("episode_id") for b in bs],
            "episode_id_agrees": [ea[a].get("episode_id") == eb[b].get("episode_id") for b in bs],
            "T": ea[a]["T"], "v_max": ea[a]["v_max"], "path_len_m": ea[a]["path_len_m"],
        })

    n = len(leaked)
    res["THE_ANSWER"] = {
        "question": f"how many of {ha['cache_key']}'s {len(ea)} evaluation episodes are, "
                    f"BY CONTENT, also in the parity TRAIN corpus {hb['cache_key']} "
                    f"({len(eb)} episodes)?",
        "n_val_episodes": len(ea),
        "n_train_episodes": len(eb),
        "n_overlap_by_poses_sha256": n,
        "n_overlap_by_frames_sha256": n_matched_a(pairs_sensor) if full else None,
        "frac_of_val_episodes": n / max(len(ea), 1),
        "leaked_val_tags": leaked,
        "detail": detail,
        "verdict": "LEAKED — pre-registered outcome 2" if n else "CLEAN — pre-registered outcome 1",
    }

    # ---- partial / shifted containment ------------------------------------- #
    res["partial_frame_containment"] = partial_frame_overlap(ea, eb)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("THE_ANSWER", "controls", "hash_family_agreement",
                       "name_derived_cross_checks")}, indent=1)[:6000])
    pf = res["partial_frame_containment"]
    print("\npartial_frame_containment:", json.dumps(
        {k: pf[k] for k in pf if k != "top20"}, indent=1))
    if pf.get("available"):
        print("top5:", json.dumps(pf["top20"][:5], indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
