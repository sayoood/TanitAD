"""Score the stratum-D sweep — PREPARED before the labels exist, run only after they land.

⛔ THE SAMPLING UNIT IS THE CLIP. The sweep drew one window per clip, so its n is **55 clusters**,
not 58 sheets. The script asserts that (one sha12 per D sheet) and refuses to report otherwise.

⛔ THE THREE MIXED-IN CANDIDATES ARE A DIFFERENT POPULATION — two route-1 shock events and one
route-2 widest-margin window. They were mixed in so the labeller could not tell them apart; they
are reported SEPARATELY here, never folded into the 55, because folding them in would undo the
clip-first draw.

⚠️ The adjudicator's tie-break leans toward on-surface: judge whenever the surface is visible in
ANY panel, reserve cannot-tell for when it is visible in none. A low cannot-tell count is a
property of that rule, not of the sheets. Carried into the output verbatim.

    python score_sweep.py --labels <csv> --key <off-repo json> --raw <raw dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

LABELS = ("on-surface", "over-boundary", "cannot-tell")
D_SOURCE = "D sweep (no rule fires)"
TIE_BREAK = ("the adjudicator judged whenever the surface was visible in ANY panel and reserved "
             "cannot-tell for when it was visible in NONE; a low cannot-tell count is a property "
             "of that rule, not of the sheets")


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided upper bound on the rate: the p where P(X <= k | n, p) = alpha."""
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def read_labels(path: Path) -> tuple[dict, set]:
    lab, borderline = {}, set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip() or line.startswith("blind_id"):
            continue
        r = next(csv.reader([line]))
        lab[r[0].strip()] = r[1].strip()
        if any("BORDERLINE" in f.upper() for f in r[1:]):      # any field: notes carry commas
            borderline.add(r[0].strip())
    return lab, borderline


def score(labels_path: Path, key_path: Path) -> dict:
    key = json.loads(key_path.read_bytes())
    rows = {r["blind_id"]: r for r in key["rows"]}
    lab, borderline = read_labels(labels_path)
    missing, extra = set(rows) - set(lab), set(lab) - set(rows)
    if missing or extra:
        raise SystemExit(f"⛔ labels do not match the key: {len(missing)} unlabelled, "
                         f"{len(extra)} unknown ids")
    bad = {i: l for i, l in lab.items() if l not in LABELS}
    if bad:
        raise SystemExit(f"⛔ labels outside the rubric: {bad}")

    d_ids = [i for i in lab if rows[i]["source"] == D_SOURCE]
    mixed = [i for i in lab if rows[i]["source"] != D_SOURCE]
    clips = {rows[i]["sha12"] for i in d_ids}
    if len(clips) != len(d_ids):
        raise SystemExit(f"⛔ the D sweep is {len(d_ids)} sheets over {len(clips)} clips — the "
                         f"clip-first draw is broken and n cannot be reported as clusters")
    cells_d = Counter(lab[i] for i in d_ids)
    k = cells_d["over-boundary"]
    n = len(d_ids)
    out = {
        "_what": "stratum-D sweep scored; the unit is the CLIP",
        "_evidence_class": "MEASURED (ours)",
        "key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
        "tie_break_caveat": TIE_BREAK,
        "sweep": {
            "n_clusters (clips)": n, "sheets": len(d_ids),
            "cells": {l: cells_d.get(l, 0) for l in LABELS},
            "over_boundary_rate_per_clip": round(k / n, 4) if n else None,
            "upper_bound_95_one_sided": round(clopper_pearson_upper(k, n), 4) if n else None,
            "reading": ("no departure found in any clip; the bound is what the sweep establishes"
                        if k == 0 else
                        "⛔ a departure was found where NO rule fires: a FALSE-PASS for V0, P1 and "
                        "P2 alike, and the first evidence this corpus can give about MISSES"),
        },
        "mixed_in_reported_separately": {
            "n": len(mixed),
            "rows": [{"blind_id": i, "source": rows[i]["source"], "label": lab[i],
                      "sha12": rows[i]["sha12"], "t0": rows[i]["t0"]} for i in sorted(mixed)],
            "note": "a different population — two route-1 shock events and one route-2 "
                    "widest-margin window. NEVER folded into the 55; an over-boundary here would "
                    "mean route 1 or 2 found a real event after all, and would revive the ranking "
                    "question the ruling closed on this corpus",
        },
        "borderline": {"n": len(borderline), "ids": sorted(borderline)},
    }
    return out


def _selftest(tmp: Path, key_path: Path) -> dict:
    """⛔ A scorer that has never run is a liability at the moment the labels land. Four synthetic
    label files, each exercising one branch; the real labels are not involved."""
    key = json.loads(key_path.read_bytes())
    ids = [r["blind_id"] for r in key["rows"]]
    d_ids = [r["blind_id"] for r in key["rows"] if r["source"] == D_SOURCE]
    mixed = [r["blind_id"] for r in key["rows"] if r["source"] != D_SOURCE]

    def write(name, mapping):
        p = tmp / name
        p.write_text("blind_id,label,note\n" + "\n".join(f"{i},{l}," for i, l in mapping.items())
                     + "\n", encoding="utf-8", newline="\n")
        return p

    res = {}
    all_on = {i: "on-surface" for i in ids}
    r = score(write("all_on.csv", all_on), key_path)
    res["all on-surface"] = {"n_clusters": r["sweep"]["n_clusters (clips)"],
                             "k": r["sweep"]["cells"]["over-boundary"],
                             "upper_bound": r["sweep"]["upper_bound_95_one_sided"],
                             "mixed_reported_separately": r["mixed_in_reported_separately"]["n"]}
    one = dict(all_on, **{d_ids[0]: "over-boundary"})
    r = score(write("one_d.csv", one), key_path)
    res["one over-boundary in the sweep"] = {
        "k": r["sweep"]["cells"]["over-boundary"],
        "rate_per_clip": r["sweep"]["over_boundary_rate_per_clip"],
        "upper_bound": r["sweep"]["upper_bound_95_one_sided"],
        "reading_starts": r["sweep"]["reading"][:40]}
    onemix = dict(all_on, **{mixed[0]: "over-boundary"})
    r = score(write("one_mixed.csv", onemix), key_path)
    res["one over-boundary in the MIXED-IN"] = {
        "sweep_k_unchanged": r["sweep"]["cells"]["over-boundary"],
        "mixed_labels": [m["label"] for m in r["mixed_in_reported_separately"]["rows"]].count("over-boundary")}
    ct = dict(all_on, **{d_ids[1]: "cannot-tell"})
    r = score(write("one_ct.csv", ct), key_path)
    res["one cannot-tell"] = {"cells": r["sweep"]["cells"]}
    for name, mapping, why in (("short.csv", {i: "on-surface" for i in ids[:-1]}, "a missing label"),
                               ("badlabel.csv", dict(all_on, **{ids[0]: "probably fine"}),
                                "a label outside the rubric")):
        try:
            score(write(name, mapping), key_path)
            res[f"REFUSES {why}"] = "⛔ NOT REFUSED"
        except SystemExit as e:
            res[f"refuses {why}"] = str(e)[:60]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", help="omit to run the self-test only")
    ap.add_argument("--key", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--selftest-out", default=None)
    a = ap.parse_args()
    key_path = Path(a.key)
    if a.selftest_out:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            res = _selftest(Path(td), key_path)
        Path(a.selftest_out).write_text(json.dumps(
            {"_what": "score_sweep.py exercised on SYNTHETIC labels before the real ones exist",
             "_evidence_class": "MEASURED (ours)", "key_sha256":
                 hashlib.sha256(key_path.read_bytes()).hexdigest(), "branches": res}, indent=1)
            + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(res, indent=1))
    if not a.labels:
        return 0
    out = score(Path(a.labels), key_path)
    Path(a.raw, "sweep_scoring.json").write_text(json.dumps(out, indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
