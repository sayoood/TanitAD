#!/usr/bin/env python3
"""bank_audit.py -- run dag_read.py's BANK-STAGE controls on one bank directory (as v8f_dagger_bank.py wrote it) and print PASS / FAIL / MISSING per control.

    python bank_audit.py --report <bank dir>/PROVENANCE.json --m2b-sha <64-hex of the M2b checkpoint> --eval-grid <eval_grid_windows.json | JSON list> --train-clips <JSON list | .npz with 'clip'>
                         [--bank-file bank.json] [--seed 0] [--out audit.json]

Read-only. It runs the SAME functions the reader runs before any arm file is opened (BANK-FILES, BANK-CONFIG, K-B0, K-B1, K-B2, K-B3.1, K-B3.3, K-B4, K-B5, BANK-ACCOUNTING) and the COVERAGE number.
The controls that need the trainer's records (K-B3.2, TRAIN-LIST, BANK-DECLARED) are listed as NOT RUN. A failing control prints its reason; "needs the bank generator to add" marks a field the generator
does not write (an older generator's bank lacks `controls` K-B4 / K-B5). Exit code: 0 every control that ran passed, 1 some failed.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_reader():
    spec = importlib.util.spec_from_file_location("dag_read_audit", os.path.join(HERE, "dag_read.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["dag_read_audit"] = m
    spec.loader.exec_module(m)
    return m


def clip_list(path, np):
    if path.endswith(".npz"):
        return [str(x) for x in np.load(path, allow_pickle=False)["clip"].tolist()]
    o = json.load(open(path, encoding="utf-8"))
    return [str(x) for x in (o["order"] if isinstance(o, dict) else o)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", required=True)
    ap.add_argument("--bank-file")
    ap.add_argument("--m2b-sha", required=True)
    ap.add_argument("--eval-grid", required=True)
    ap.add_argument("--train-clips", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    dag = load_reader()
    np = dag.np
    B = dag.load_bank(a.report, a.bank_file, what=f"bank seed {a.seed} report")
    ctl = dag.Ctl()
    dag.bank_stage(ctl, a.seed, B, a.m2b_sha.lower(), clip_list(a.eval_grid, np), clip_list(a.train_clips, np))
    res = {"report": os.path.abspath(a.report), "bank_sha256": B["man_sha256"], "controls": ctl.items, "NOT_RUN": {
        f"K-B3.2:s{a.seed}": "needs the trainer's summary.json on DAG and DAG-0 (`train_window_list_sha256`, recorded on every run by v8f_mini_train.py)",
        f"TRAIN-LIST:s{a.seed}": "needs the trainer's summary.json (train_sha256) on DAG and DAG-0", f"BANK-DECLARED:DAG_s{a.seed}": "needs the DAG arm's argv / built record"}}
    try:
        cv = dag.coverage_of(B)
        res["coverage"] = cv
        res["COVERAGE"] = ("BLOCKED" if cv["eligible_fraction"] < dag.COVERAGE_MIN else "launches") + f" (eligible fraction {cv['eligible_fraction']:.4f}, min {dag.COVERAGE_MIN})"
    except dag.Refusal as e:
        res["COVERAGE"] = "UNREADABLE: " + "; ".join(e.msgs)
    res["header"] = dag.header_summary(B)
    for cid, v in ctl.items.items():
        det = v["detail"]
        tag = "PASS" if v["PASS"] else ("MISSING" if "needs the bank generator to add" in str(det) else "FAIL")
        print(f"{tag:8s} {cid:22s} {json.dumps(det, default=str)[:600] if not v['PASS'] else json.dumps(det, default=str)[:260]}")
    for cid, why in res["NOT_RUN"].items():
        print(f"NOT RUN  {cid:22s} {why}")
    print("COVERAGE", res["COVERAGE"])
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, default=str)
    return 0 if not ctl.failed() else 1


if __name__ == "__main__":
    sys.exit(main())
