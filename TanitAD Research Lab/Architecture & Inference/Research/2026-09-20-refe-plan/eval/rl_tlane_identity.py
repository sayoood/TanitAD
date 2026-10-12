#!/usr/bin/env python3
"""RL-S1 TL-1: is `refe/rl_tlane_label.py`'s label the SAME label as SFT-4's?  Three independent readings, each a subcommand, each writes a JSON.

  props    python eval/rl_tlane_identity.py props   --queue <dir of props_*.jsonl*> [--old-sets <dir of onpolicy_*.jsonl>] --out props.json
             The JOIN: every real QUEUE row (what `onpolicy_dump.py` wrote and SFT-4's labeller read) is turned into a lean record by the REAL
             `rl_fan_dump.set_record`, then back into a queue row by `rl_tlane_label.queue_row`; the result must equal the original queue row, field for field,
             FLOAT FOR FLOAT (props, teacher, key, ckpt_step, aug).  --old-sets additionally checks the labelled `onpolicy_set` lines of the same samples.
  teacher  python eval/rl_tlane_identity.py teacher --queue <dir> --bank <dir with targets_rank*.jsonl> --out teacher.json
             The queue's `teacher` field (onpolicy_dump.py:141 = TargetBank.rows[i]["traj"]) must equal the BANK row's `traj` for the same (log, token, step, rank).
             This is what lets the labeller read the teacher from the bank instead of from a queue.
  labels   python eval/rl_tlane_identity.py labels  --mine <dir of teacher_lane_r*.jsonl> --banked <teacher_lane xz / jsonl / dir> --out labels.json
             The LABELS: for every key both hold (key + ckpt_step), all 9 leaves of the teacher and of the 64 proposals, EXACT and as a max |delta|, and the
             quantity the trainer actually reads (the soft lane target clip(1 - (d - D0) / (D1 - D0), 0, 1) from center_line.CenterLine.info).
What this does NOT prove (stated in each JSON): that the dump's raw fp32 outputs equal the old dumper's (`eval/rl_fan_identity.py`, needs the GPU), nor that the
simulator is bit-reproducible across machines (it is not: `onpolicy_label.py` selftest comment, centre-line distances differ in ~1e-7 across thread counts).
"""
from __future__ import annotations

import argparse
import glob
import json
import lzma
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "refe")
sys.path.insert(0, REFE)

LEAVES = ("center_line.CenterLine.reward", "center_line.CenterLine.info", "off_road.CrossLane.reward", "off_road.CrossLane.info",
          "off_road.CrossLane.lane_change_info", "off_road.CrossLane.solid_lane_change_info", "off_road.OffRoad.info", "ddc.violation", "dac.violation")


def iter_jsonl(paths):
    for fp in paths:
        op = lzma.open if fp.endswith(".xz") else open
        with op(fp, "rt", encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    try:
                        yield json.loads(ln)
                    except json.JSONDecodeError:
                        continue


def queue_files(d):
    return sorted(f for f in glob.glob(os.path.join(d, "**", "props_r*"), recursive=True) if os.path.isfile(f) and "status" not in os.path.basename(f))


def cmd_props(a) -> int:
    import rl_fan_dump as RFD
    import rl_tlane_label as RTL
    n = n_exact = 0
    bad = []
    by_rank = {}
    for q in iter_jsonl(queue_files(a.queue)):
        if q.get("kind") != "onpolicy_props":
            continue
        n += 1
        rec = RFD.set_record({"log_name": q["log_name"], "token": q["token"], "step": q["step"], "aug": q.get("aug")}, q["rank"], q["ckpt_step"], q["props"])
        back = RTL.queue_row(json.loads(json.dumps(rec)), q["teacher"])          # through JSON, as the file is read
        ok = back == q
        n_exact += ok
        by_rank.setdefault(int(q["rank"]), [0, 0])
        by_rank[int(q["rank"])][0] += 1
        by_rank[int(q["rank"])][1] += ok
        if not ok and len(bad) < 5:
            bad.append({"key": [q["log_name"], q["token"], q["step"], q["rank"]], "fields": [k for k in set(q) | set(back) if q.get(k) != back.get(k)]})
    res = {"tool": "rl_tlane_identity/props", "queue_rows": n, "rebuilt_equal": n_exact, "by_rank": {str(k): v for k, v in sorted(by_rank.items())}, "mismatches_first5": bad,
           "scope": "the JOIN only (real rl_fan_dump.set_record -> rl_tlane_label.queue_row); not the dumper's raw fp32 outputs (eval/rl_fan_identity.py, GPU)"}
    if a.old_sets:
        qk = {}
        for q in iter_jsonl(queue_files(a.queue)):
            qk[(q["log_name"], q["token"], int(q["step"]), int(q["rank"]), int(q["ckpt_step"]))] = q["props"]
        m = e = 0
        for r in iter_jsonl(sorted(glob.glob(os.path.join(a.old_sets, "**", "onpolicy_*.jsonl"), recursive=True))):
            k = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]), int(r["ckpt_step"]))
            if k in qk:
                m += 1
                e += RTL.props_from_set(r) == qk[k]
        res["old_labelled_sets"] = {"matched_to_queue": m, "props_from_set_equals_queue_props": e}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if n and n_exact == n and (not a.old_sets or res["old_labelled_sets"]["matched_to_queue"] == res["old_labelled_sets"]["props_from_set_equals_queue_props"]) else 1


def cmd_teacher(a) -> int:
    want = {}
    for q in iter_jsonl(queue_files(a.queue)):
        if q.get("kind") == "onpolicy_props" and int(q["rank"]) == 0:
            want[(q["log_name"], q["token"], int(q["step"]), 0)] = q["teacher"]
    toks = {k[1] for k in want}
    seen = {}
    dup = 0
    for fp in sorted(glob.glob(os.path.join(a.bank, "targets_rank0.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                r = json.loads(ln)
                if r.get("token") in toks:
                    k = (r["log_name"], r["token"], int(r.get("step", 0)), int(r.get("rank", 0)))
                    if k in want:
                        dup += k in seen
                        seen[k] = r["traj"]
    eq = sum(1 for k, v in seen.items() if v == want[k])
    worst = 0.0
    for k, v in seen.items():
        if v != want[k]:
            worst = max(worst, max(abs(x - y) for p, q2 in zip(v, want[k]) for x, y in zip(p, q2)))
    res = {"tool": "rl_tlane_identity/teacher", "queue_rank0_rows": len(want), "found_in_bank": len(seen), "bank_traj_equals_queue_teacher": eq,
           "not_in_bank": len(want) - len(seen), "bank_duplicates": dup, "max_abs_delta_when_unequal": worst,
           "scope": "exact float equality of the queue's `teacher` and the bank row's `traj` (json round trip as TargetBank reads it)"}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if want and len(seen) == len(want) and eq == len(seen) else 1


def _files(p):
    return sorted(glob.glob(os.path.join(p, "teacher_lane_r*.jsonl*"))) if os.path.isdir(p) else [p]


def cmd_labels(a) -> int:
    import train as T
    d0, d1 = T.OnPolicyBank.LANE_D0, T.OnPolicyBank.LANE_D1
    mine = {(tuple(d["key"]), int(d["ckpt_step"])): d for d in iter_jsonl(_files(a.mine))}
    banked = {(tuple(d["key"]), int(d["ckpt_step"])): d for d in iter_jsonl(_files(a.banked))}
    both = sorted(set(mine) & set(banked))
    n_leaf = n_leaf_eq = n_sets_all_eq = n_sets_info_eq = 0
    max_d = {lf: 0.0 for lf in LEAVES}
    max_soft = 0.0
    n_null_mismatch = 0
    repair_equal = 0
    sets_with_diff = []
    for k in both:
        m, b = mine[k], banked[k]
        repair_equal += m.get("repair") == b.get("repair")
        pairs = [(m["teacher"], b["teacher"])] + list(zip(m["props"], b["props"]))
        all_eq = info_eq = True
        for x, y in pairs:
            for lf in LEAVES:
                vx, vy = x.get(lf), y.get(lf)
                n_leaf += 1
                if vx is None or vy is None:
                    n_null_mismatch += (vx is None) != (vy is None)
                    eq = vx is None and vy is None
                else:
                    eq = vx == vy
                    if not eq:
                        max_d[lf] = max(max_d[lf], abs(float(vx) - float(vy)))
                n_leaf_eq += eq
                if not eq:
                    all_eq = False
                    if lf == "center_line.CenterLine.info":
                        info_eq = False
        n_sets_all_eq += all_eq
        n_sets_info_eq += info_eq
        if not all_eq and len(sets_with_diff) < 5:
            sets_with_diff.append(list(k[0]))
        sx = [min(1.0, max(0.0, 1.0 - (float(p["center_line.CenterLine.info"]) - d0) / (d1 - d0))) for p in m["props"]]
        sy = [min(1.0, max(0.0, 1.0 - (float(p["center_line.CenterLine.info"]) - d0) / (d1 - d0))) for p in b["props"]]
        max_soft = max(max_soft, max(abs(u - v) for u, v in zip(sx, sy)))
    res = {"tool": "rl_tlane_identity/labels", "mine": len(mine), "banked": len(banked), "keys_in_both": len(both), "repair_field_equal": repair_equal,
           "leaf_values_compared": n_leaf, "leaf_values_exactly_equal": n_leaf_eq, "sets_all_65x9_leaves_exactly_equal": n_sets_all_eq,
           "sets_whose_64_centre_line_distances_are_exactly_equal": n_sets_info_eq, "null_vs_value_mismatches": n_null_mismatch,
           "max_abs_delta_by_leaf_when_unequal": max_d, "max_abs_delta_of_the_trainers_soft_target": max_soft, "lane_d0_d1": [d0, d1],
           "first_sets_with_any_difference": sets_with_diff,
           "scope": "same keys, same ckpt_step; the trainer reads ONLY center_line.CenterLine.info (via the soft target). Cross-machine runs are not bit-reproducible (stated in onpolicy_label.py selftest)."}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if both else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("props"); p.add_argument("--queue", required=True); p.add_argument("--old-sets", default=""); p.add_argument("--out", default="")
    t = sub.add_parser("teacher"); t.add_argument("--queue", required=True); t.add_argument("--bank", required=True); t.add_argument("--out", default="")
    l = sub.add_parser("labels"); l.add_argument("--mine", required=True); l.add_argument("--banked", required=True); l.add_argument("--out", default="")
    a = ap.parse_args()
    return {"props": cmd_props, "teacher": cmd_teacher, "labels": cmd_labels}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
