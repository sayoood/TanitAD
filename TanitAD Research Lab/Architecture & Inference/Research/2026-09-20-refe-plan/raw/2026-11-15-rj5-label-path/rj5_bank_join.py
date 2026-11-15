#!/usr/bin/env python3
"""RJ5 -> P7-bank join (REFe, 2026-11-15). Pure stdlib, deterministic, no DB, no numpy, CPU only.

Maps every RJ5-roll row (arm RJ5, from several row dirs, each given a SOURCE LABEL) onto the P7 bank's rank-0 key exactly as
`refe/traj_label_table.py` expects it:  key = "log|token|0|sha256(bank traj as little-endian float64, row-major)".
The RJ5 row's own traj sha is kept as `rj5_traj_sha256` (the sha the dual labeller will record); the bank key stays the bank's.

  python rj5_bank_join.py join --bank bank_p7_rank0_slim.jsonl.gz \
        --rows local=<dir> --rows A=<dir> --rows B=<dir> [--rows C=<dir>] --out-dir <dir> [--min-coverage 0.0]
  python rj5_bank_join.py rekey --join <rj5_bank_join.jsonl.gz> --labels <labels_s*of*.jsonl ...> --out <build_labels.jsonl>
        (labeller records keyed by the RJ5 traj sha -> traj_label_table `build --labels` jsonl keyed to the BANK row; refuses any sha disagreement)

join writes   <out-dir>/rj5_bank_join.jsonl.gz   one line per bank rank-0 key (sorted by log, token):
                 log_name, token, rank 0, bank_key, bank_traj_sha256, status in {rj5, missing, conflict}, source, rj5_traj_sha256,
                 dmax_vs_bank, dxy_vs_bank (rj5 rows only), rj5_traj (rj5 rows only; the traj that the labeller must be fed)
              <out-dir>/rj5_bank_join_manifest.json   coverage by source, duplicates, conflicts, unmapped rows, control R0, knobs, sha256s
Exit 0 iff coverage >= --min-coverage and no conflict and no knob mismatch.  The R0 control is READ (reported, bar 1e-6 m, TF-N2 C-R0), not hidden.
Conflicting duplicates (same key, different traj, within or across sources) are NEVER resolved silently: the key gets status "conflict" and no traj.
"""
import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import struct
import sys
from array import array

SCHEMA = "rj5-bank-join-v1"
JERK = {"RJ5": 5.0, "R0": 1.0}
CTRL_BAR_M = 1e-6


def traj_sha256(traj):
    """identical to refe/traj_label_table.traj_sha256 (little-endian float64, row-major)."""
    h = hashlib.sha256()
    for pose in traj:
        for v in pose:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError:
                    yield {"__torn__": True}            # a torn last line of a killed shard: counted, never silent


def flat(traj):
    return array("d", [float(v) for p in traj for v in p])


def dmax(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def dxy(a, b):
    return max(math.hypot(a[i] - b[i], a[i + 1] - b[i + 1]) for i in range(0, len(a), 3))


def unflat(a):
    return [[a[i], a[i + 1], a[i + 2]] for i in range(0, len(a), 3)]


def load_bank(path):
    bank, dup, n_rank_other = {}, 0, 0
    for r in read_jsonl(path):
        if int(r.get("rank", 0)) != 0:
            n_rank_other += 1
            continue
        k = (r["log_name"], r["token"])
        if k in bank:
            dup += 1
            continue
        bank[k] = (flat(r["traj"]), traj_sha256(r["traj"]))
    return bank, dup, n_rank_other


def join(bank_path, rows_specs, min_coverage=0.0):
    bank, bank_dup, n_other = load_bank(bank_path)
    rj5 = {}                                              # key -> (source, flat traj, sha)
    conflicts = {}                                        # key -> sorted sources involved
    per_src = {}
    ctl = {"n": 0, "n_key_in_bank": 0, "n_not_in_bank": 0, "n_sha_equal_bank": 0, "max_dxy_m": 0.0, "max_abs": 0.0, "n_over_bar": 0,
           "n_recorded_bank_dmax_disagrees": 0, "n_exact_zero": 0}
    knob_bad = []
    unmapped = {}
    files = {}
    seen_ctl = {}
    for src, d in rows_specs:
        s = per_src.setdefault(src, {"files": 0, "rows": 0, "rj5_ok": 0, "r0_ok": 0, "error_rows": 0, "torn_lines": 0, "dup_identical": 0,
                                     "dup_conflicting": 0, "not_in_bank": 0, "mapped_keys": 0, "won_keys": 0})
        for fp in sorted(glob.glob(os.path.join(d, "rj5_*.jsonl"))):
            files[f"{src}/{os.path.basename(fp)}"] = sha256_file(fp)
            s["files"] += 1
            for r in read_jsonl(fp):
                if r.get("__torn__"):
                    s["torn_lines"] += 1
                    continue
                s["rows"] += 1
                if "traj" not in r:
                    s["error_rows"] += 1
                    continue
                arm = r.get("arm")
                k = (r["log_name"], r["token"])
                if arm not in JERK:
                    continue
                if "knobs" in r:
                    got = r["knobs"].get("positive_jerk_limit_when_nonnegative_acc")
                    if got is None or abs(float(got) - JERK[arm]) > 1e-9:
                        knob_bad.append([src, arm, k[0], k[1], got])
                if arm == "R0":
                    s["r0_ok"] += 1
                    if k in seen_ctl:
                        continue
                    ctl["n"] += 1
                    if k not in bank:
                        ctl["n_not_in_bank"] += 1
                        continue
                    ctl["n_key_in_bank"] += 1
                    seen_ctl[k] = 1
                    ft = flat(r["traj"])
                    b_tr, b_sha = bank[k]
                    dm, dx = dmax(ft, b_tr), dxy(ft, b_tr)
                    ctl["max_abs"], ctl["max_dxy_m"] = max(ctl["max_abs"], dm), max(ctl["max_dxy_m"], dx)
                    ctl["n_over_bar"] += dx > CTRL_BAR_M
                    ctl["n_exact_zero"] += dm == 0.0
                    ctl["n_sha_equal_bank"] += traj_sha256(r["traj"]) == b_sha
                    if "bank_dmax" in r and abs(float(r["bank_dmax"]) - dm) > 1e-12:
                        ctl["n_recorded_bank_dmax_disagrees"] += 1
                    continue
                s["rj5_ok"] += 1
                if k not in bank:
                    s["not_in_bank"] += 1
                    unmapped[k] = src
                    continue
                s["mapped_keys"] += 1
                ft = flat(r["traj"])
                if k in rj5:
                    if rj5[k][1] == ft:
                        s["dup_identical"] += 1
                    else:
                        s["dup_conflicting"] += 1
                        conflicts.setdefault(k, {rj5[k][0]}).add(src)
                    continue
                rj5[k] = (src, ft, traj_sha256(r["traj"]))
                s["won_keys"] += 1
    return bank, rj5, conflicts, per_src, ctl, knob_bad, unmapped, files, bank_dup, n_other


def write_join(out_dir, bank, rj5, conflicts):
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "rj5_bank_join.jsonl.gz")
    by_status = {"rj5": 0, "missing": 0, "conflict": 0}
    by_source = {}
    n_moved = 0
    with gzip.GzipFile(outp, "wb", mtime=0) as g:
        for k in sorted(bank):
            b_tr, b_sha = bank[k]
            o = {"log_name": k[0], "token": k[1], "rank": 0, "bank_key": f"{k[0]}|{k[1]}|0|{b_sha}", "bank_traj_sha256": b_sha}
            if k in conflicts:
                o.update(status="conflict", source=None)
            elif k in rj5:
                src, ft, sha = rj5[k]
                o.update(status="rj5", source=src, rj5_traj_sha256=sha, dmax_vs_bank=dmax(ft, b_tr), dxy_vs_bank=dxy(ft, b_tr), rj5_traj=unflat(ft))
                n_moved += o["dxy_vs_bank"] > 1e-6
                by_source[src] = by_source.get(src, 0) + 1
            else:
                o.update(status="missing", source=None)
            by_status[o["status"]] += 1
            g.write((json.dumps(o) + "\n").encode("utf-8"))
    return outp, by_status, by_source, n_moved


def cmd_join(a):
    specs = []
    for s in a.rows:
        if "=" not in s:
            print(f"ZZJOIN_REFUSE --rows wants label=dir, got {s!r}")
            return 2
        lab, d = s.split("=", 1)
        specs.append((lab, d))
    bank, rj5, conflicts, per_src, ctl, knob_bad, unmapped, files, bank_dup, n_other = join(a.bank, specs, a.min_coverage)
    outp, by_status, by_source, n_moved = write_join(a.out_dir, bank, rj5, conflicts)
    cov = by_status["rj5"] / max(1, len(bank))
    rep = {"schema": SCHEMA, "bank_file": os.path.basename(a.bank), "bank_sha256": sha256_file(a.bank), "n_bank_rank0_keys": len(bank),
           "n_bank_duplicate_keys": bank_dup, "n_bank_rows_other_rank_skipped": n_other, "status_counts": by_status, "coverage": cov,
           "coverage_by_source": by_source, "n_missing": by_status["missing"], "n_conflict_keys": len(conflicts),
           "conflict_examples": [[k[0], k[1], sorted(v)] for k, v in sorted(conflicts.items())[:5]],
           "n_rj5_rows_not_in_bank": len(unmapped), "not_in_bank_examples": [[k[0], k[1], v] for k, v in sorted(unmapped.items())[:5]],
           "per_source": per_src, "rj5_moved_gt_1e-6_m": {"n": n_moved, "share": n_moved / max(1, by_status["rj5"])},
           "control_R0": dict(ctl, bar_m=CTRL_BAR_M, passes_bar=(ctl["n_over_bar"] == 0 and ctl["n"] > 0),
                              note="R0 = registered teacher re-roll; its traj must equal the bank traj (same key, same sha) within the bar"),
           "knob_mismatches": {"n": len(knob_bad), "examples": knob_bad[:5]},
           "input_row_files_sha256": files, "output_file": os.path.basename(outp), "output_sha256": sha256_file(outp),
           "output_bytes": os.path.getsize(outp)}
    with open(os.path.join(a.out_dir, "rj5_bank_join_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(json.dumps({k: rep[k] for k in ("n_bank_rank0_keys", "status_counts", "coverage", "coverage_by_source", "n_conflict_keys",
                                          "n_rj5_rows_not_in_bank", "control_R0", "output_sha256", "output_bytes")}))
    ok = cov >= a.min_coverage and not conflicts and not knob_bad
    print("ZZJOIN_OK" if ok else "ZZJOIN_REFUSED")
    return 0 if ok else 3


def cmd_rekey(a):
    jn = {}
    for r in read_jsonl(a.join):
        if r.get("status") == "rj5":
            jn[(r["log_name"], r["token"])] = r
    n = n_err = 0
    bad = []
    with open(a.out, "w", encoding="utf-8", newline="\n") as out:
        for fp in a.labels:
            for r in read_jsonl(fp):
                lab = r.get("p5r") if a.field == "p5r" else r.get("tc20s")
                if lab is None:
                    n_err += 1
                    continue
                k = (r["log"], r["token"])
                j = jn.get(k)
                if j is None:
                    bad.append([k[0], k[1], "no rj5 join row"])
                    continue
                if r.get("traj_sha256") != j["rj5_traj_sha256"]:
                    bad.append([k[0], k[1], "label traj sha != join rj5 sha"])
                    continue
                out.write(json.dumps({"log_name": k[0], "token": k[1], "rank": 0, "label": lab, "traj_sha256": j["bank_traj_sha256"],
                                      "rj5_traj_sha256": j["rj5_traj_sha256"], "rj5_source": j["source"]}) + "\n")
                n += 1
    print(json.dumps({"written": n, "label_error_records_skipped": n_err, "refused": len(bad), "examples": bad[:5]}))
    ok = not bad
    print("ZZREKEY_OK" if ok else "ZZREKEY_REFUSED")
    return 0 if ok else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    j = sp.add_parser("join")
    j.add_argument("--bank", required=True)
    j.add_argument("--rows", action="append", required=True, help="label=dir (dir holds rj5_*.jsonl)")
    j.add_argument("--out-dir", required=True)
    j.add_argument("--min-coverage", type=float, default=0.0)
    k = sp.add_parser("rekey")
    k.add_argument("--join", required=True)
    k.add_argument("--labels", nargs="+", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--field", default="p5r", choices=["p5r", "tc20s"])
    a = ap.parse_args(argv)
    return cmd_join(a) if a.cmd == "join" else cmd_rekey(a)


if __name__ == "__main__":
    sys.exit(main())
