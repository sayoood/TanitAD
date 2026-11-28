"""select_stage2.py -- card PREREG_V8F_PROOF_ADD153 section 2: picks the stage-1 arm with the highest held-out A (record) among the arms whose controls all pass,
writes SELECT.json and the two stage-2 job files. Exit 0 with a pick, 3 when no stage-1 arm passed its controls. Runs on the dev box on the pulled ACC.json files.

  python -I select_stage2.py      (reads rulefc/leadacc/box2_acc_<arm>/ACC.json)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_jobs_acc as MK  # noqa: E402


def pick(results):
    """results: {arm: ACC.json dict or None} -> (arm or None, table)."""
    table = {}
    for arm, r in results.items():
        ok = bool(r and r.get("controls_all_pass") and "error" not in r)
        table[arm] = {"controls_all_pass": ok, "A": (r or {}).get("held_out", {}).get("A"), "S": (r or {}).get("held_out", {}).get("S"), "name": (r or {}).get("name")}
    cands = [(t["A"], arm) for arm, t in table.items() if t["controls_all_pass"] and t["A"] is not None]
    return (max(cands)[1] if cands else None), table


def main():
    res = {}
    for arm in MK.STAGE1:
        p = os.path.join(HERE, "box2_acc_" + arm, "ACC.json")
        res[arm] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    best, table = pick(res)
    out = {"card": "PREREG_V8F_PROOF_ADD153", "rule": "highest held-out A (record) among stage-1 arms with every control passed", "table": table, "selected": best}
    if best:
        out["jobs"] = MK.stage2(best, os.path.join(HERE, "jobs"))
    json.dump(out, open(os.path.join(HERE, "SELECT.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if best else 3


if __name__ == "__main__":
    sys.exit(main())
