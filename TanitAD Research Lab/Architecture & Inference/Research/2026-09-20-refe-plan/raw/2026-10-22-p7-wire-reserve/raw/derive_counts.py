#!/usr/bin/env python3
"""Derived counts for P7-WIRE-RESERVE, MEASURED by the module itself from the token table (CPU, stdlib only, no network).

  python derive_counts.py --refe <dir holding session_reserve.py and fixtures/> --table <navtrain_token_table.tsv.gz> --out <raw dir>

Writes derived_counts.json (rank-0, per session, steps) and rank0_filter_report.json (the FILTER run over a one-row-per-attributed-token pseudo-bank).
The pseudo-bank has the 3-C bank's rank-0 SHAPE (real tokens, real logs) and none of its features: its `targets_sha256` is NOT a pin for the real bank, only the counts are the result.
Rank-1 is NOT measured here (it needs the 3-C bank, which is on the pod / HF): see measure_rank1_3c.sh.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.refe)
    import session_reserve as SR
    fx = Path(a.refe) / "fixtures" / "p7_reserve_sessions.json"
    res = SR.load_reserve(fx)
    c = SR.count_from_token_table(a.table, res)
    tl = SR.token_log_map(a.table)
    with tempfile.TemporaryDirectory() as tmp:
        bank = Path(tmp) / "bank"
        bank.mkdir()
        with open(bank / "targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for t, lg in tl.items():
                f.write(json.dumps({"log_name": lg, "token": t, "step": 0, "rank": 0}) + "\n")
        rep = SR.filter_bank_dir(bank, Path(tmp) / "out", res, token_log=tl, reserve_info=SR._reserve_sha_info(fx))
    scenes_left = c["n_remaining"]
    steps = 25 * math.ceil(scenes_left / 256)
    p6_steps = 25 * math.ceil(SR.BANK_RANK0_SCENES / 256)
    out = {
        "evidence": {"rank0_exclusion": "MEASURED (session_reserve.count_from_token_table and filter_bank over the token table; artifact: this file)",
                     "steps": "arithmetic (trainer's definition: epochs x ceil(epoch_len / (batch x accum)), SPEC_P6 0.2, launch_gate profile refe_p6 expect_schedule)",
                     "rank1": "PENDING: needs the 3-C bank (HF Sayood/tanitad-refe-navtrain-banks refe_train_grow_cmd_3c.tar.zst); command in measure_rank1_3c.sh",
                     "rank1_estimate": "ESTIMATED, arithmetic only (SESSION_RESERVE.md section 3): if rank-1 twins fall uniformly over scenes"},
        "reserve_file_sha256": SR.RESERVE_FILE_SHA256,
        "token_table": {"rows": c["table_rows"], "attributed": c["attributed"], "logless": c["logless"]},
        "rank0": {"n_in": rep["n_in"], "n_out": rep["n_out"], "n_excluded": rep["n_excluded"], "n_unresolved": rep["n_unresolved"], "token_log_checked": rep["token_log_checked"],
                  "sessions_hit": c["sessions_hit"], "reserve_scenes_pinned": SR.RESERVE_SCENES, "matches_pin": rep["n_excluded"] == SR.RESERVE_SCENES == c["n_excluded"]},
        "steps": {"p7_scenes": scenes_left, "p7_steps": steps, "p6_steps": p6_steps, "saved_steps": p6_steps - steps},
        "rank1": {"status": "PENDING", "bank_rank1_rows": 64263, "estimate_if_uniform_excluded": round(64263 * c["n_excluded"] / c["attributed"]),
                  "estimate_if_uniform_remaining": 64263 - round(64263 * c["n_excluded"] / c["attributed"]),
                  "estimate_total_tuples": scenes_left + 64263 - round(64263 * c["n_excluded"] / c["attributed"]), "class": "ESTIMATED"},
        "per_session_rank0": c["per_session"],
    }
    outdir = Path(a.out)
    (outdir / "derived_counts.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    keep = {k: v for k, v in rep.items() if k not in ("files",)}
    keep["note"] = "pseudo-bank: one rank-0 row per attributed token (real tokens, real logs); targets_sha256 is NOT the real bank's"
    (outdir / "rank0_filter_report.json").write_text(json.dumps(keep, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"rank0": out["rank0"], "steps": out["steps"], "rank1_estimate": out["rank1"]}, sort_keys=True))
    return 0 if out["rank0"]["matches_pin"] else 1


if __name__ == "__main__":
    sys.exit(main())
