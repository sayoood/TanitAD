"""THE JOB-2 TABLE — file · arm · target · old K1 · new K1 · guard · change.

Pairs every banked INCUMBENT ladder row (``ll_<tag>.json``, ``fit_mode: pc6``)
against its re-read under the module's repair + the C97 guard
(``llR_<tag>.json``, ``fit_mode: unpen``).

⛔ ENUMERATED BY OPENING BOTH ARTIFACTS (C91). Nothing is copied from a report,
and a missing counterpart is printed as MISSING rather than silently dropped —
an inventory that quietly shrinks is the failure C91 records.

⚠️ ROWS ARE PAIRED ON THE SAME WINDOWS. ``n_eval`` is asserted equal between the
banked and re-read row before any delta is reported: the repair changes the
SOLVE, never the window set, so an n that moved would mean the caches or the
split moved and the comparison would be invalid.

⛔ T0-DIAGNOSTIC. A frozen-latent readout is a world-model diagnostic, never
driving performance.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path


def verdict(rec) -> str:
    """The audit's own predicate, verbatim, so old and new are commensurable."""
    if rec["K1_PASSES"]:
        return "PASS"
    return "FAIL-separated" if rec["K1_separated"] else "not-separated"


def guarded_verdict(rec) -> str:
    """The verdict a reader is allowed to quote, after the C97 guard."""
    v = verdict(rec)
    g = rec.get("k1_guard")
    if g is None:                                            # pragma: no cover
        return v + " (UNGUARDED)"
    if v == "not-separated":
        return v
    return v if g["K1_quotable_as_latent_evidence"] else f"{v} → {g['guard_verdict']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banked", required=True)
    ap.add_argument("--reread", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows, missing = [], []
    for p in sorted(glob.glob(os.path.join(a.banked, "ll_*.json"))):
        f = os.path.basename(p)
        if f.startswith("ll_rep_"):
            continue                       # already on a repaired solve
        B = json.loads(Path(p).read_text("utf-8"))
        if B.get("fit_mode") != "pc6" or "targets" not in B:
            continue
        tag = f[len("ll_"):-len(".json")]
        rp = os.path.join(a.reread, f"llR_{tag}.json")
        if not os.path.exists(rp):
            missing.append(f)
            continue
        R = json.loads(Path(rp).read_text("utf-8"))
        for tgt, tb in B["targets"].items():
            tr = R["targets"].get(tgt)
            if tr is None:                                   # pragma: no cover
                missing.append(f"{f}:{tgt}")
                continue
            assert tb["n_eval"] == tr["n_eval"], (
                f"⛔ window set moved for {f}:{tgt} — {tb['n_eval']} vs "
                f"{tr['n_eval']}; the repair changes the SOLVE, not the windows")
            rb, rr = tb["per_seed"]["0"], tr["per_seed"]["0"]
            g = rr["k1_guard"]
            vo, vn = verdict(rb), verdict(rr)
            rows.append({
                "file": f, "arm": B["arm"], "target": tgt,
                "rung": tb.get("rung"), "unit": tb.get("unit"),
                "n_eval": tb["n_eval"], "n_eval_clusters": tb["n_eval_clusters"],
                "old_K1": rb["K1_delta"],
                "old_K1_ci": [rb["K1_lo"], rb["K1_hi"]],
                "old_verdict": vo, "old_alpha": rb["alpha_chosen"],
                "old_err": rb["err"], "old_pred_sd": rb["pred_sd"],
                "new_K1": rr["K1_delta"],
                "new_K1_ci": [rr["K1_lo"], rr["K1_hi"]],
                "new_verdict": vn, "new_alpha": rr["alpha_chosen"],
                "new_alpha_at_grid_edge": rr.get("alpha_at_grid_edge"),
                "new_err": rr["err"], "new_pred_sd": rr["pred_sd"],
                "c_const_err": rr["c_const_err"], "gt_sd": rr["gt_sd"],
                "guard_verdict": g["guard_verdict"],
                "K1B": g["K1B_delta"], "K1B_ci": [g["K1B_lo"], g["K1B_hi"]],
                "K1B_separated": g["K1B_separated"], "K1C": g["K1C_delta"],
                "sd_ratio": g["sd_ratio"], "flat_line": g["flat_line"],
                "k1_exceeds_own_spread": g["k1_exceeds_own_spread"],
                "mean_minus_median_const_gap":
                    g.get("mean_minus_median_const_gap"),
                "quotable": g["K1_quotable_as_latent_evidence"],
                "guarded_verdict": guarded_verdict(rr),
                "verdict_changed": vo != vn,
                "survives_as_finding": bool(vn != "not-separated"
                                            and g["K1_quotable_as_latent_evidence"]),
            })

    if missing:
        print(f"⛔ MISSING re-reads ({len(missing)}): {missing}")

    hdr = (f"{'file':22} {'target':14} {'old K1':>9} {'new K1':>9} "
           f"{'old verdict':15} {'new verdict':15} {'guard':21} change")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ch = "—" if not r["verdict_changed"] else f"{r['old_verdict']}→{r['new_verdict']}"
        print(f"{r['file'][:22]:22} {r['target'][:14]:14} {r['old_K1']:+9.3f} "
              f"{r['new_K1']:+9.3f} {r['old_verdict']:15} {r['new_verdict']:15} "
              f"{r['guard_verdict']:21} {ch}")

    old_c, new_c = Counter(r["old_verdict"] for r in rows), \
        Counter(r["new_verdict"] for r in rows)
    gc = Counter(r["guard_verdict"] for r in rows)
    old_sepfail = [r for r in rows if r["old_verdict"] == "FAIL-separated"]
    survive = [r for r in old_sepfail if r["new_verdict"] == "FAIL-separated"
               and r["quotable"]]
    died_ci = [r for r in old_sepfail if r["new_verdict"] == "not-separated"]
    died_guard = [r for r in old_sepfail if r["new_verdict"] == "FAIL-separated"
                  and not r["quotable"]]
    flipped = [r for r in old_sepfail if r["new_verdict"] == "PASS"]
    new_pass_quotable = [r for r in rows if r["new_verdict"] == "PASS"
                         and r["quotable"]]
    new_pass_caught = [r for r in rows if r["new_verdict"] == "PASS"
                       and not r["quotable"]]

    print(f"\nrows re-read                          : {len(rows)}")
    print(f"old verdicts                          : {dict(old_c)}")
    print(f"new verdicts (repaired solve)         : {dict(new_c)}")
    print(f"guard verdicts                        : {dict(gc)}")
    print(f"verdict changed by the repair         : "
          f"{sum(r['verdict_changed'] for r in rows)}")
    print(f"\nOF THE {len(old_sepfail)} BANKED SEPARATED-FAILs:")
    print(f"  survive repair AND guard  (findings): {len(survive)}")
    print(f"  die at the repair (CI now spans 0)  : {len(died_ci)}")
    print(f"  survive repair, KILLED BY THE GUARD : {len(died_guard)}")
    print(f"  flip to PASS                        : {len(flipped)}")
    print(f"\nPASSes on the repaired solve          : "
          f"{len(new_pass_quotable) + len(new_pass_caught)}")
    print(f"  quotable (guard OK)                 : {len(new_pass_quotable)}")
    print(f"  ⛔ CAUGHT BY THE GUARD               : {len(new_pass_caught)}")

    payload = {
        "_evidence_class": "MEASURED (ours; both sides opened from JSON on "
                           "disk, C91; paired on identical window sets)",
        "eval_tier": "T0-DIAGNOSTIC",
        "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
        "forbidden": "overlapping_holdout_se",
        "repair": "ridge_fit(..., intercept_col=-1) via ll1 --fit-mode unpen",
        "guard": "taniteval.degeneracy.k1_guard",
        "n_rows": len(rows), "n_files": len(set(r["file"] for r in rows)),
        "missing": missing,
        "old_verdict_counts": dict(old_c), "new_verdict_counts": dict(new_c),
        "guard_verdict_counts": dict(gc),
        "n_verdict_changed": sum(r["verdict_changed"] for r in rows),
        "banked_separated_FAILs": len(old_sepfail),
        "separated_FAILs_surviving_repair_and_guard": len(survive),
        "separated_FAILs_dying_at_repair": len(died_ci),
        "separated_FAILs_killed_by_guard": len(died_guard),
        "separated_FAILs_flipping_to_PASS": len(flipped),
        "repaired_PASS_quotable": len(new_pass_quotable),
        "repaired_PASS_caught_by_guard": len(new_pass_caught),
        "surviving_findings": [
            {k: r[k] for k in ("file", "arm", "target", "rung", "new_K1",
                               "new_K1_ci", "K1B", "K1B_ci", "sd_ratio")}
            for r in survive + new_pass_quotable],
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
