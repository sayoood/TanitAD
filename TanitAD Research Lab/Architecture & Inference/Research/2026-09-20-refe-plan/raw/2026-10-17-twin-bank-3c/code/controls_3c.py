#!/usr/bin/env python3
"""Identity controls I1-I4 of PREREG_3C.md, computed from the control run's files.  Writes controls.json.   Any failure => 3C-INVALID (no number of the full run is read).

  I1  per-twin f(t) (+ d_twin(t), v0, hold_misses) and the commit set == the probe's `held_h6` rows (DE-DUPLICATED), bit-for-bit
  I2  each kept row's 4 s traj / traj_world / goal == the probe's 4 s re-roll `rr_h6` row, bit-for-bit; the whole row equals rr_h6's except aug += {commit_f10, reroll}
  I3  merged bank: every non-lane-rank row byte-identical and in order; lines_out == lines_in - dropped; kept rows in input order; dropped twins absent
      (+ MUTATION arms on the I3 checker itself: a flipped byte / a deleted row / a swapped pair / a re-inserted dropped twin must each be REJECTED)
  I4  (mutation) the no-hold arm's commit set DIFFERS from the held arm's (the check can see the hold)

    python controls_3c.py --held-shards 'D/held/shards/s*' --legacy-shards 'D/legacy/shards/s*' --merged D/merged --bank-i3 D/bank_i3
                          --probe-dedup 'P/held_h6_s*.jsonl' --rr-h6 'R/s*/targets_reroll.jsonl' --out controls.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reroll_bank_3c as B  # noqa: E402

PROBE_ERR_MAP = {"scenario too short for 100 steps": "scenario_too_short_10s", "missing scenario or parts": "missing_"}


def jl(path):
    return [json.loads(l) for l in Path(path).read_bytes().split(b"\n") if l.strip()]


def audit_of(pattern):
    out = {}
    for sd in sorted(glob.glob(pattern)):
        for r in jl(Path(sd) / "audit.jsonl"):
            k = B.key_of_audit(r)
            assert k not in out, ("duplicate audit record", k)
            out[k] = r
    return out


def rows_of(pattern):
    out = {}
    for sd in sorted(glob.glob(pattern)):
        p = Path(sd) / "rows.jsonl"
        if p.exists():
            for r in jl(p):
                out[B.key_of(r)] = r
    return out


def probe_rows(pattern):
    out = {}
    for fp in sorted(glob.glob(pattern)):
        for r in jl(fp):
            k = (r["log_name"], r["token"], int(r["k"]))
            assert k not in out, ("probe dedup file still holds a duplicate", k)
            out[k] = r
    return out


def check_i1(mine, probe):
    res = {"n_probe": len(probe), "n_mine": len(mine), "keys_only_in_probe": sorted(set(probe) - set(mine))[:5], "keys_only_in_mine": sorted(set(mine) - set(probe))[:5],
           "n_compared": 0, "n_with_f": 0, "f_mismatch": [], "d_twin_mismatch": [], "commit_mismatch": [], "v0_mismatch": [], "hold_misses_mismatch": [], "error_mismatch": []}
    for k in sorted(set(mine) & set(probe)):
        m, p = mine[k], probe[k]
        res["n_compared"] += 1
        if "f_at_s" in p:
            res["n_with_f"] += 1
            if m["f_at_s"] != p["f_at_s"]:
                res["f_mismatch"].append(k)
            if m["d_twin_at_s"] != p["d_twin_at_s"]:
                res["d_twin_mismatch"].append(k)
            if m["commit"] != (p["f_at_s"][-1] >= B.KEEP_F10):
                res["commit_mismatch"].append(k)
            if m["v0"] != p["v0"]:
                res["v0_mismatch"].append(k)
            if m["hold_misses"] != p["hold_misses"]:
                res["hold_misses_mismatch"].append(k)
        else:
            want = PROBE_ERR_MAP.get(p.get("error", ""), None)
            if want is None or not str(m["drop_reason"]).startswith(want):
                res["error_mismatch"].append((k, p.get("error"), m["drop_reason"]))
    res["pass"] = (not res["keys_only_in_probe"] and not res["keys_only_in_mine"] and res["n_compared"] == len(probe) and not any(
        res[f] for f in ("f_mismatch", "d_twin_mismatch", "commit_mismatch", "v0_mismatch", "hold_misses_mismatch", "error_mismatch")))
    for f in ("f_mismatch", "d_twin_mismatch", "commit_mismatch", "v0_mismatch", "hold_misses_mismatch"):
        res[f] = [list(x) for x in res[f][:5]] if res[f] else []
    res["commit_set_size_probe"] = sum(1 for p in probe.values() if "f_at_s" in p and p["f_at_s"][-1] >= B.KEEP_F10)
    res["commit_set_size_mine"] = sum(1 for m in mine.values() if m["commit"])
    return res


def check_i2(kept_rows, rr, probe):
    res = {"n_kept": len(kept_rows), "n_rr": len(rr), "kept_not_in_rr": [], "traj_mismatch": [], "traj_world_mismatch": [], "goal_mismatch": [], "row_mismatch": [],
           "probe_committed_but_not_kept": []}
    for k, row in sorted(kept_rows.items()):
        ref = rr.get(k)
        if ref is None:
            res["kept_not_in_rr"].append(k); continue
        if row["traj"] != ref["traj"]:
            res["traj_mismatch"].append(k)
        if row["traj_world"] != ref["traj_world"]:
            res["traj_world_mismatch"].append(k)
        if row["goal"] != ref["goal"]:
            res["goal_mismatch"].append(k)
        exp = json.loads(json.dumps(ref))
        exp["aug"]["commit_f10"] = row["aug"]["commit_f10"]
        exp["aug"]["reroll"] = "3c"
        if exp != row:
            res["row_mismatch"].append(k)
    for k, p in probe.items():
        if "f_at_s" in p and p["f_at_s"][-1] >= B.KEEP_F10 and k not in kept_rows:
            res["probe_committed_but_not_kept"].append(k)
    res["pass"] = res["n_kept"] > 0 and not any(res[f] for f in ("kept_not_in_rr", "traj_mismatch", "traj_world_mismatch", "goal_mismatch", "row_mismatch",
                                                                   "probe_committed_but_not_kept"))
    for f in ("kept_not_in_rr", "traj_mismatch", "traj_world_mismatch", "goal_mismatch", "row_mismatch", "probe_committed_but_not_kept"):
        res[f] = [list(x) for x in res[f][:5]] if res[f] else []
    return res


def i3_verdict(in_lines, out_lines, audit):
    """-> (ok, why).  in_lines/out_lines: lists of bytes lines WITH terminators.  audit: key -> record (for the dropped set)."""
    why = []
    lane = lambda l: b'"lane_rank"' in l and B.is_lane_row(json.loads(l))
    in_other = [l for l in in_lines if not lane(l)]
    out_other = [l for l in out_lines if not lane(l)]
    if in_other != out_other:
        why.append("non-lane-rank rows are not byte-identical / in order")
    in_lane = [B.key_of(json.loads(l)) for l in in_lines if lane(l)]
    out_lane = [B.key_of(json.loads(l)) for l in out_lines if lane(l)]
    dropped = [k for k in in_lane if not (audit[k]["commit"] is True and audit[k]["drop_reason"] is None)]
    if len(out_lines) != len(in_lines) - len(dropped):
        why.append(f"line count {len(out_lines)} != input {len(in_lines)} - dropped {len(dropped)}")
    if set(out_lane) & set(dropped):
        why.append("a dropped twin is present in the output")
    if out_lane != [k for k in in_lane if k not in set(dropped)]:
        why.append("kept twins are not exactly the input's non-dropped twins in input order")
    return (not why), "; ".join(why) or "ok"


def check_i3(bank_dir, merged_dir, audit):
    in_lines = Path(bank_dir, "targets_rank1.jsonl").read_bytes().splitlines(keepends=True)
    out_lines = Path(merged_dir, "targets_rank1.jsonl").read_bytes().splitlines(keepends=True)
    ok, why = i3_verdict(in_lines, out_lines, audit)
    res = {"lines_in": len(in_lines), "lines_out": len(out_lines), "non_lane_rows": sum(1 for l in in_lines if b'"lane_rank"' not in l), "verdict": why, "pass": ok}
    # ---- mutation arms on the checker: each broken output must be REJECTED
    other_idx = [i for i, l in enumerate(out_lines) if b'"lane_rank"' not in l]
    dropped_lines = [l for l in in_lines if b'"lane_rank"' in l and not (audit[B.key_of(json.loads(l))]["commit"] and audit[B.key_of(json.loads(l))]["drop_reason"] is None)]
    muts = {}
    m = list(out_lines); i = other_idx[len(other_idx) // 2]; m[i] = m[i].replace(b"{", b"{ ", 1)                      # one byte inserted in a non-lane row
    muts["flip_byte_in_non_lane_row"] = m
    m = list(out_lines); del m[other_idx[0]]                                                                      # a non-lane row deleted
    muts["delete_non_lane_row"] = m
    m = list(out_lines); m[other_idx[0]], m[other_idx[1]] = m[other_idx[1]], m[other_idx[0]]                      # two non-lane rows swapped
    muts["swap_two_non_lane_rows"] = m
    if dropped_lines:
        m = list(out_lines) + [dropped_lines[0]]                                                                  # a dropped twin re-inserted
        muts["reinsert_dropped_twin"] = m
    res["mutation_arms"] = {}
    for name, lines in muts.items():
        mok, mwhy = i3_verdict(in_lines, lines, audit)
        res["mutation_arms"][name] = {"rejected": (not mok), "why": mwhy}
    res["mutations_all_rejected"] = all(v["rejected"] for v in res["mutation_arms"].values()) and len(res["mutation_arms"]) >= 3
    res["pass"] = res["pass"] and res["mutations_all_rejected"]
    return res


def check_i4(held, legacy):
    ks = sorted(set(held) | set(legacy))
    cs_h = {k for k in ks if held.get(k, {}).get("commit")}
    cs_l = {k for k in ks if legacy.get(k, {}).get("commit")}
    f_diff = [k for k in ks if k in held and k in legacy and held[k]["f_at_s"] != legacy[k]["f_at_s"]]
    return {"n_held": len(held), "n_legacy": len(legacy), "commit_set_held": len(cs_h), "commit_set_legacy": len(cs_l), "symmetric_difference": len(cs_h ^ cs_l),
            "only_held": len(cs_h - cs_l), "only_legacy": len(cs_l - cs_h), "n_twins_f_differs": len(f_diff), "legacy_hold_misses_none": all(
                legacy[k]["hold_misses"] is None for k in legacy if legacy[k]["f_at_s"] is not None),
            "pass": len(cs_h ^ cs_l) > 0}


def selftest(held, probe, out_rows, rr):
    """MUTATION arms on the I1 / I2 checkers: each deliberately broken input must be REJECTED (a pass cannot come from a checker that compares nothing)."""
    import copy
    res = {}
    k = next(k for k, v in held.items() if v["f_at_s"] is not None)
    m = copy.deepcopy(held); m[k]["f_at_s"][3] = round(m[k]["f_at_s"][3] + 0.001, 3)
    res["I1_f_value_off_by_0.001_rejected"] = not check_i1(m, probe)["pass"]
    m = copy.deepcopy(held); kc = next(k for k, v in m.items() if v["commit"]); m[kc]["commit"] = False
    res["I1_commit_flipped_rejected"] = not check_i1(m, probe)["pass"]
    m = copy.deepcopy(held); del m[k]
    res["I1_missing_twin_rejected"] = not check_i1(m, probe)["pass"]
    kk = next(iter(out_rows))
    r2 = copy.deepcopy(out_rows); r2[kk]["traj"][5][0] = float.fromhex(float(r2[kk]["traj"][5][0]).hex()) + 2 ** -40            # one tiny change in one float
    res["I2_traj_float_perturbed_rejected"] = not check_i2(r2, rr, probe)["pass"]
    r2 = copy.deepcopy(out_rows); r2[kk]["goal"][0] += 1e-9
    res["I2_goal_perturbed_rejected"] = not check_i2(r2, rr, probe)["pass"]
    r2 = copy.deepcopy(out_rows); r2[kk]["aug"]["divergence_m"] += 0.0001
    res["I2_divergence_perturbed_rejected"] = not check_i2(r2, rr, probe)["pass"]
    r2 = copy.deepcopy(out_rows); r2.pop(kk)
    res["I2_kept_row_missing_rejected"] = not check_i2(r2, rr, probe)["pass"]
    res["all_rejected"] = all(res.values())
    return res


def sha(p):
    return B.sha256_file(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("held-shards", "legacy-shards", "merged", "bank-i3", "probe-dedup", "rr-h6", "out"):
        ap.add_argument("--" + f, required=True)
    a = ap.parse_args(argv)
    held, legacy = audit_of(a.held_shards), audit_of(a.legacy_shards)
    probe = probe_rows(a.probe_dedup)
    rr = {}
    for fp in sorted(glob.glob(a.rr_h6)):
        for r in jl(fp):
            rr[B.key_of(r)] = r
    kept = rows_of(a.held_shards)
    kept = {k: v for k, v in kept.items() if held[k]["commit"] and held[k]["drop_reason"] is None}
    out_rows = {B.key_of(json.loads(l)): json.loads(l) for l in Path(a.merged, "targets_rank1.jsonl").read_bytes().splitlines() if b'"lane_rank"' in l
                and B.is_lane_row(json.loads(l))}
    res = {"prereg_sha256": B.PREREG_SHA, "I1": check_i1(held, probe), "I2": check_i2(out_rows, rr, probe), "I3": check_i3(a.bank_i3, a.merged, held),
           "I4": check_i4(held, legacy)}
    res["I2"]["merged_rows_equal_shard_rows"] = out_rows == kept
    res["I2"]["pass"] = res["I2"]["pass"] and res["I2"]["merged_rows_equal_shard_rows"]
    res["selftest_mutation_arms"] = selftest(held, probe, out_rows, rr)
    res["all_pass"] = all(res[c]["pass"] for c in ("I1", "I2", "I3", "I4")) and res["selftest_mutation_arms"]["all_rejected"]
    res["verdict"] = "I1-I4 PASS" if res["all_pass"] else "3C-INVALID (a control failed): no number of the full run may be read"
    res["inputs"] = {"bank_i3_sha256": sha(Path(a.bank_i3, "targets_rank1.jsonl")), "merged_sha256": sha(Path(a.merged, "targets_rank1.jsonl")),
                     "held_audit_n": len(held), "legacy_audit_n": len(legacy), "probe_dedup_n": len(probe), "rr_h6_n": len(rr)}
    Path(a.out).write_text(json.dumps(res, indent=1, default=list), encoding="utf-8")
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in res.items() if k in ("I1", "I2", "I3", "I4", "all_pass", "verdict")}))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
