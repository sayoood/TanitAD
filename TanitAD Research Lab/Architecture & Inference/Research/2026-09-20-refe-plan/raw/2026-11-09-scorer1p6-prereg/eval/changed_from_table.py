#!/usr/bin/env python3
"""changed_from_table.py (dev box, CPU): the scores of a scorer-only arm's changed picks on E3, looked up in the full navtest truth table instead of re-run through the harness.

A scorer-only arm keeps P6's 64 proposals bit for bit (read_scorer1p6 gate G0-E3 checks it), so the harness score of the arm's pick on a token is the table's
column for that proposal (selection_quality table_p6final, all 64 proposals x 12,146 tokens, harness PDMS v1). Writes the reader's e3_changed_scores.csv format for
every token where the arm's pick (navsim_v1 rule, torch float32, read_scorer1p6.pick_logits) differs from P6's.

  changed_from_table.py --table table_p6final_c.npz --arm-logits <arm>/e3_logits.npz --out <arm>/e3_changed_scores.csv [--control-csv <harness csv>]

--control-csv: every row of a harness-scored csv must appear in the output with the same valid flag, the same six sub-scores and the same score, exactly (float equality).
Prints ZZCHANGED_OK <n rows> [control <n> rows equal] or ZZCHANGED_FAIL <reason>."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_scorer1p6 as R  # noqa: E402

SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")


class Refused(Exception):
    pass


def build(table, arm_logits):
    with np.load(table, allow_pickle=False) as T:
        tok, pdms, sub, valid, tpick, tlog = [str(t) for t in T["token"]], T["pdms"], T["sub"], T["valid"], T["pick"], T["logits"]
    with np.load(arm_logits, allow_pickle=False) as A:
        atok, L = [str(t) for t in A["token"]], A["logits"]
    pos = {t: i for i, t in enumerate(tok)}
    if len(pos) != len(tok) or set(atok) != set(tok) or len(atok) != len(tok):
        raise Refused(f"token sets differ: table {len(tok)}, arm {len(atok)}, common {len(set(tok) & set(atok))}")
    ix = np.array([pos[t] for t in atok])
    p6 = R.pick_logits(tlog)
    if int((p6 != tpick).sum()) > 2:            # 12,144 of 12,146 equal when measured (2 float ties); more means the table is not P6's
        raise Refused(f"the table's stored pick differs from P6's rule on its own logits on {int((p6 != tpick).sum())} tokens")
    ap = R.pick_logits(L)
    rows = []
    for j, i in enumerate(ix):
        k = int(ap[j])
        if k == int(p6[i]):
            continue
        if not np.isfinite(pdms[i, k]):
            raise Refused(f"token {atok[j]} proposal {k}: no table value")
        rows.append((atok[j], bool(valid[i, k]), [float(x) for x in sub[i, k]], float(pdms[i, k])))
    return rows


def write(rows, out):
    tmp = out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "token", "valid", *SUB, "score"])
        for n, (t, v, s, sc) in enumerate(rows):
            w.writerow([n, t, str(v), *[repr(x) for x in s], repr(sc)])
    os.replace(tmp, out)


def control(rows, ctl):
    got = {t: (v, s, sc) for t, v, s, sc in rows}
    n = 0
    with open(ctl, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = r["token"]
            if t not in got:
                raise Refused(f"control: token {t} of the harness csv is not a changed pick in the lookup")
            v, s, sc = got[t]
            hv, hs, hsc = (r["valid"] == "True"), [float(r[c]) for c in SUB], float(r["score"])
            if v != hv or s != hs or sc != hsc:
                raise Refused(f"control: token {t} differs: lookup {(v, s, sc)} harness {(hv, hs, hsc)}")
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--arm-logits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control-csv", default=None)
    a = ap.parse_args(argv)
    try:
        rows = build(a.table, a.arm_logits)
        nc = control(rows, a.control_csv) if a.control_csv else None
        write(rows, a.out)
    except Refused as e:
        print(f"ZZCHANGED_FAIL {e}")
        return 1
    print(f"ZZCHANGED_OK {len(rows)} rows" + (f" control {nc} rows equal" if nc is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
