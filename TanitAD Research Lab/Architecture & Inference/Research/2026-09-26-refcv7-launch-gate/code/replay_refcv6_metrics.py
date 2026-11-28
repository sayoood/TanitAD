"""G-LIVE's declared-term rule, applied to the REAL record of refcv6-r101-s0.

    python replay_refcv6_metrics.py --tree <code tree> --config <run config.json> \
        --metrics <run metrics.jsonl> --json <out.json>

The declared terms are DERIVED exactly as the gate derives them -- `launch_gate.declared_terms`
over the trainer's own registries (`REFC_WEIGHT_GATES`, `DiffusionFlags`) and the run's OWN argv
(its config.json) -- never listed by hand. Each is then counted in the run's training rows before
and after the A16 switch (step 34,500, where 82c2331 put F3's cascade into the loss).

⭐ What this answers: would the gate's rule have refused the refcv6 LAUNCH? The first rows of the
real run are the rows its 30-step smoke would have produced, on the real config and data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SWITCH_STEP = 34500


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--first-rows", type=int, default=30)
    a = ap.parse_args(argv)
    tree = Path(a.tree).resolve()
    for p in (tree / "taniteval", tree / "stack" / "scripts", tree / "stack"):
        sys.path.insert(0, str(p))
    import tanitad
    assert str(Path(tanitad.__file__).resolve()).startswith(str(tree / "stack")), tanitad.__file__
    import launch_gate as LG
    spec = importlib.util.spec_from_file_location("refc_v3_train_replay",
                                                  str(tree / "stack/scripts/refc_v3_train.py"))
    T = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_replay"] = T
    spec.loader.exec_module(T)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    args = T.build_parser().parse_args(cfg["argv"])            # parse only: no file is read
    terms, problems = LG.declared_terms(T, args)
    rows = LG.train_rows(LG.read_metrics(Path(a.metrics)))
    before = [r for r in rows if int(r["step"]) <= SWITCH_STEP]
    after = [r for r in rows if int(r["step"]) > SWITCH_STEP]
    table = []
    for t in terms:
        for k in t["keys"]:
            table.append({"rule": t["id"], "key": k, "why": t["why"],
                          "rows_before_switch_with_key": sum(1 for r in before if k in r),
                          "rows_before_switch": len(before),
                          "rows_after_switch_with_key": sum(1 for r in after if k in r),
                          "rows_after_switch": len(after)})
    # the rule, over the run's first training rows (what its smoke would have logged)
    first = rows[: a.first_rows]
    steps = [{"keys": sorted(k for k in r if k not in ("step",)), "nonfinite": [],
              "vals": {k: v for k, v in r.items() if isinstance(v, (int, float))}} for r in first]
    rule_reasons = []
    for t in terms:
        for k in t["keys"]:
            n = sum(1 for s in steps if k in s["keys"])
            if n == 0:
                rule_reasons.append(f"declared term `{k}` ({t['why']}) is ABSENT from all "
                                    f"{len(steps)} of the run's first training rows")
    out = {"what": "G-LIVE's declared-term rule on refcv6-r101-s0's real metrics.jsonl",
           "evidence_class": "MEASURED (ours, dev box, CPU; the run's own artifacts)",
           "config": a.config, "config_sha256": LG.sha256_file(a.config),
           "metrics": a.metrics, "metrics_sha256": LG.sha256_file(a.metrics),
           "switch_step": SWITCH_STEP, "declared_terms": terms, "coverage_problems": problems,
           "n_train_rows": len(rows), "first_row_step": rows[0]["step"] if rows else None,
           "last_row_step": rows[-1]["step"] if rows else None, "table": table,
           "rule_over_first_rows": {"n_rows": len(steps),
                                    "first_steps": [int(r["step"]) for r in first[:5]],
                                    "verdict": "FAIL" if rule_reasons else "PASS",
                                    "reasons": rule_reasons}}
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"verdict_first_rows": out["rule_over_first_rows"]["verdict"],
                      "table": [(r["key"], r["rows_before_switch_with_key"], r["rows_before_switch"],
                                 r["rows_after_switch_with_key"], r["rows_after_switch"])
                                for r in table]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
