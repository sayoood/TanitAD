"""Score the blind adjudication labels against the key — per stratum, three label cells apart.

⛔ The three labels are counted as THREE cells. `cannot-tell` is never folded into agreement or
disagreement, because the adjudicator's own tie-break decides how often it appears: on-surface /
over-boundary whenever the surface is visible in ANY panel, `cannot-tell` only when it is visible
in none. A zero there is a property of that rule, not of the images.

Two readings are produced and BOTH are reported, in this order:

  1. THE PRE-COMMITTED ONE — §5.4's bar over the 60 adjudicated windows, exactly as written.
  2. ⚠️ A POST-HOC one — the same labels reweighted by each stratum's POPULATION size, because
     §5.1's sample is deliberately disagreement-enriched. It is computed here because the
     committed reading cannot separate "the rule is wrong" from "the sample was built to find
     where rules fire" — and it is labelled post-hoc everywhere it appears. ⛔ It adopts nothing.

    python score_labels.py --labels <csv> --key <off-repo json> --raw <raw dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATES = ("V0", "P1", "P2")
LABELS = ("on-surface", "over-boundary", "cannot-tell")
#: stratum -> population size, from the anatomy's own pools (they sum to the 736 held-out windows)
POP = {"A V0 fires, P1 does not": 264, "B P1 fires, P2 does not": 31,
       "C all three fire": 33, "D none fires": 408}
BAR_AGREE, BAR_ERR = 0.95, 0.05


def upper_bound_zero_of_n(n: int, alpha: float = 0.05) -> float:
    """One-sided 95 % upper bound on a share when 0 of n were observed (1 - alpha^(1/n))."""
    return 1.0 - alpha ** (1.0 / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--raw", required=True)
    a = ap.parse_args()
    key_b = Path(a.key).read_bytes()
    key = json.loads(key_b)
    rows = {r["blind_id"]: r for r in key["rows"]}
    lab: dict = {}
    borderline = set()
    for line in Path(a.labels).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip() or line.startswith("blind_id"):
            continue
        r = next(csv.reader([line]))
        lab[r[0]] = r[1].strip()
        if len(r) > 2 and "BORDERLINE" in r[2].upper():
            borderline.add(r[0])
    missing = set(rows) - set(lab)
    if missing:
        raise SystemExit(f"⛔ {len(missing)} sheets carry no label: {sorted(missing)[:5]}")
    if set(lab) - set(rows):
        raise SystemExit(f"⛔ labels for ids that are not in the key: {sorted(set(lab) - set(rows))[:5]}")
    bad = {i: l for i, l in lab.items() if l not in LABELS}
    if bad:
        raise SystemExit(f"⛔ labels outside the rubric: {bad}")

    # ── the three label cells, per stratum ────────────────────────────────────────────────
    cells = defaultdict(Counter)
    for i, l in lab.items():
        cells[rows[i]["stratum"]][l] += 1
    cells_total = Counter(lab.values())

    # ── per candidate: agree / false-fire / false-pass / cannot-tell, per stratum ─────────
    def fires(i: str, c: str) -> bool:
        return bool(rows[i][f"{c}_fires"])

    per_cand = {}
    for c in CANDIDATES:
        by_s = defaultdict(Counter)
        for i, l in lab.items():
            s = rows[i]["stratum"]
            f = fires(i, c)
            if l == "cannot-tell":
                by_s[s]["cannot-tell"] += 1
            elif (l == "over-boundary") == f:
                by_s[s]["agree"] += 1
            elif f:
                by_s[s]["false-fire (rule fires, adjudicator says on-surface)"] += 1
            else:
                by_s[s]["false-pass (rule silent, adjudicator says over-boundary)"] += 1
        tot = Counter()
        for s in by_s:
            tot.update(by_s[s])
        scored = sum(v for k, v in tot.items() if k != "cannot-tell")
        agree = tot["agree"] / scored if scored else 0.0
        ff = tot["false-fire (rule fires, adjudicator says on-surface)"] / scored if scored else 0.0
        fp = tot["false-pass (rule silent, adjudicator says over-boundary)"] / scored if scored else 0.0
        # the POST-HOC population reweighting: within a stratum a candidate's verdict is constant
        # by construction, so its corpus agreement is the population share of the strata where it
        # agrees with the adjudicated labels of that stratum.
        w_agree, w_lb = 0.0, 0.0
        for s, n in POP.items():
            n_s = sum(by_s[s].values())
            off_share_ub = upper_bound_zero_of_n(n_s) if by_s[s]["agree"] + \
                by_s[s]["false-fire (rule fires, adjudicator says on-surface)"] == n_s else None
            f_s = any(fires(i, c) for i in lab if rows[i]["stratum"] == s)
            a_s = (by_s[s]["agree"] / max(n_s - by_s[s]["cannot-tell"], 1))
            w_agree += n * a_s
            # worst case: every unsampled window of this stratum goes the other way, bounded by
            # the one-sided 95 % limit for 0 observed in n_s
            lb_s = 0.0 if f_s else 1.0 - (off_share_ub if off_share_ub is not None else 1.0)
            w_lb += n * lb_s
        per_cand[c] = {
            "by_stratum": {s: dict(by_s[s]) for s in sorted(by_s)},
            "totals": dict(tot), "scored_windows": scored,
            "committed_reading (§5.4, over the 60)": {
                "agreement": round(agree, 4),
                "false_fire": round(ff, 4), "false_pass": round(fp, 4),
                "clears_bar": bool(agree >= BAR_AGREE and ff <= BAR_ERR and fp <= BAR_ERR)},
            "post_hoc_population_reweighted": {
                "agreement_point": round(w_agree / sum(POP.values()), 4),
                "agreement_lower_bound_95": round(w_lb / sum(POP.values()), 4),
                "note": "point assumes each stratum's sampled labels hold for its whole "
                        "population; the bound lets every unsampled window go the other way, "
                        "at the one-sided 95 % limit for 0 observed in n"},
        }

    out = {
        "_what": "H-DAC-DEF-1 §5 scoring: the Master Mind's 60 blind labels against the key",
        "_evidence_class": "MEASURED (ours; CPU, read-only)",
        "labels_source": a.labels, "key_sha256": hashlib.sha256(key_b).hexdigest(),
        "key_seed": key["seed"],
        "label_cells_total": {l: cells_total.get(l, 0) for l in LABELS},
        "label_cells_by_stratum": {s: {l: cells[s].get(l, 0) for l in LABELS}
                                   for s in sorted(cells)},
        "borderline_rows": {"n": len(borderline),
                            "by_stratum": dict(Counter(rows[i]["stratum"] for i in borderline)),
                            "ids": sorted(borderline)},
        "candidates": per_cand,
        "population_sizes": POP,
        "bar": {"agreement": BAR_AGREE, "either_error_direction": BAR_ERR},
    }
    Path(a.raw, "adjudication_scoring.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("label_cells_total", "label_cells_by_stratum",
                                          "borderline_rows")}, indent=1))
    for c in CANDIDATES:
        v = per_cand[c]
        print(f"\n{c}: {v['totals']}")
        print("   committed (over the 60):", v["committed_reading (§5.4, over the 60)"])
        print("   post-hoc reweighted:    ", v["post_hoc_population_reweighted"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
