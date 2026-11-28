#!/usr/bin/env python3
"""EXACT-reproduction comparator: a Thor scorer frame vs the dev-box banked frame, per token, per column.

    python compare_frames.py --ref <dev-box csv> --got <Thor csv> [--key token] [--ignore index ""] \
        [--subset <json with expected_tokens>] --out <diff.json>

Rows are aligned by ``--key`` (default ``token``); summary rows (``extended_pdm_score_*`` /
``average``) are compared too when both sides hold them, but are reported separately because a
subset run's summary is over a different token set by construction. For every shared column:
``max_abs_diff`` over numeric cells (NaN==NaN counts as equal, NaN vs number counts as a
mismatch), ``n_cells_text_differ`` (the CSV text itself: pandas writes the shortest round-trip
repr, so equal text <=> bit-identical float64), and ``n_cells``. The BAR is ``max_abs_diff == 0.0``
AND ``n_cells_text_differ == 0`` on EVERY per-token column; ``verdict`` says which.
Columns present on one side only, token-set differences and duplicate keys are failures, not notes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math

SUMMARY_PREFIXES = ("extended_pdm_score", "average")


def load(path: str, key: str):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = list(rows[0].keys()) if rows else []
    by = {}
    dups = 0
    for r in rows:
        k = r[key]
        if k in by:
            dups += 1
        by[k] = r
    return cols, by, dups


def num(s: str):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def cmp_rows(keys, ref, got, cols):
    out = {}
    for c in cols:
        mx, n_text, n_num_mismatch, n = 0.0, 0, 0, 0
        worst = None
        for k in keys:
            a, b = ref[k].get(c, ""), got[k].get(c, "")
            n += 1
            if a != b:
                n_text += 1
            fa, fb = num(a), num(b)
            if fa is None and fb is None:
                continue
            if (fa is None) != (fb is None):
                n_num_mismatch += 1
                continue
            if math.isnan(fa) and math.isnan(fb):
                continue
            if math.isnan(fa) != math.isnan(fb):
                n_num_mismatch += 1
                continue
            d = abs(fa - fb)
            if d > mx:
                mx, worst = d, {"ref": a, "got": b}
        out[c] = {"max_abs_diff": mx, "n_cells_text_differ": n_text,
                  "n_cells_number_vs_blank_or_nan": n_num_mismatch, "n_cells": n,
                  "worst_example": worst}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--got", required=True)
    ap.add_argument("--key", default="token")
    ap.add_argument("--ignore", nargs="*", default=["", "index"],
                    help="row-position columns (pandas index) -- differ by construction on a subset")
    ap.add_argument("--subset", default="", help="JSON with expected_tokens: Thor must hold exactly these")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rc, ref, rdup = load(a.ref, a.key)
    gc, got, gdup = load(a.got, a.key)
    fails = []
    if rdup or gdup:
        fails.append(f"duplicate keys: ref {rdup}, got {gdup}")
    is_sum = lambda k: k.startswith(SUMMARY_PREFIXES)                          # noqa: E731
    gtok = {k for k in got if not is_sum(k)}
    rtok = {k for k in ref if not is_sum(k)}
    missing_in_ref = sorted(gtok - rtok)
    if missing_in_ref:
        fails.append(f"{len(missing_in_ref)} Thor tokens absent from the reference")
    if a.subset:
        want = set(json.load(open(a.subset, encoding="utf-8"))["expected_tokens"])
        if gtok != want:
            fails.append(f"Thor token set != subset (missing {len(want - gtok)}, extra {len(gtok - want)})")
    keys = sorted(gtok & rtok)
    only_r = [c for c in rc if c not in gc and c not in a.ignore]
    only_g = [c for c in gc if c not in rc and c not in a.ignore]
    if only_r or only_g:
        fails.append(f"column sets differ: ref-only {only_r}, thor-only {only_g}")
    cols = [c for c in rc if c in gc and c != a.key and c not in a.ignore]
    per_col = cmp_rows(keys, ref, got, cols)
    bad = {c: v for c, v in per_col.items()
           if v["max_abs_diff"] != 0.0 or v["n_cells_text_differ"] or v["n_cells_number_vs_blank_or_nan"]}
    if bad:
        fails.append(f"{len(bad)} column(s) differ: {sorted(bad)}")
    skeys = sorted(k for k in got if is_sum(k) and k in ref)
    summ = cmp_rows(skeys, ref, got, cols) if skeys else {}
    rep = {"ref": a.ref, "got": a.got, "key": a.key, "n_tokens_compared": len(keys),
           "n_thor_tokens": len(gtok), "n_ref_tokens": len(rtok), "columns_compared": cols,
           "ignored_columns": a.ignore, "per_column": per_col,
           "max_abs_diff_over_all_columns": max((v["max_abs_diff"] for v in per_col.values()), default=None),
           "n_cells_text_differ_total": sum(v["n_cells_text_differ"] for v in per_col.values()),
           "summary_rows_compared__informational_only": skeys,
           "summary_rows_diff__informational_only": {c: v["max_abs_diff"] for c, v in summ.items()},
           "failures": fails, "verdict": "EXACT" if not fails else "NOT_EXACT"}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("n_tokens_compared", "max_abs_diff_over_all_columns",
                                          "n_cells_text_differ_total", "verdict", "failures")}))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
