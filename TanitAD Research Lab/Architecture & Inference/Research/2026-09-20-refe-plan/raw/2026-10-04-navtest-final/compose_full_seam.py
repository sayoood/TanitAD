"""The full 12,146-token navtest seam of the final model WITH the adopted route fix (goal_fix="pdm_route"), composed from:
  * the 3,386 tokens whose goal the fix changes (census): Amendment 9's 3,045 confirmation rows + the adoption run's 341;
  * the 8,760 tokens whose goal it does not change: the OFF seam's rows (refe_navtest_final.npz). Their model inputs are
    identical by construction (the census: same goal on every one; ego and frames do not depend on the goal path), and the
    seam is deterministic -- checked here, not assumed: every control row in BOTH fix runs (24 + 8) must equal its OFF row
    bit for bit, every fingerprint must match, and the changed set must equal confirm U rest exactly. Any failure refuses.
Writes <data>/seams/refe_navtest_final_routefix.npz + .report.json (provenance + the per-part frame controls).
"""
import json
import os

import numpy as np

DRV = os.environ.get("REFE_DRIVE", "D:")
PK = f"{DRV}/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
A9 = f"{PK}/raw/2026-10-01-goal-trigger/a9"
SEAMS = f"{DRV}/Projects/TanitAD/data/refe_navtest/seams"


def load(p):
    z = np.load(p)
    t, f, P = z["token"], z["fingerprint"], z["poses"]
    return {str(t[i]): (str(f[i]), P[i].copy()) for i in range(len(t))}, z


def main():
    off, zo = load(f"{SEAMS}/refe_navtest_final.npz")
    arm, za = load(f"{A9}/seam_pdm_route.npz")
    rest, zr = load(f"{A9}/seam_adopted_rest.npz")
    T9 = json.load(open(f"{A9}/tokens_pdm_route.json", encoding="utf-8"))
    TR = json.load(open(f"{A9}/tokens_adopted_rest.json", encoding="utf-8"))
    changed = set(json.load(open(f"{PK}/raw/2026-10-01-goal-trigger/goal_fix_census.json", encoding="utf-8"))
                  ["summary"]["pdm_route"]["changed_tokens"])
    fails = []
    if set(T9["confirm"]) | set(TR["rest"]) != changed or set(T9["confirm"]) & set(TR["rest"]):
        fails.append("changed set != confirm U rest (disjoint)")
    for name, part, ctrl in (("A9", arm, T9["controls"]), ("rest", rest, TR["controls"])):
        eq = [np.array_equal(part[t][1], off[t][1]) for t in ctrl]
        if not all(eq):
            fails.append(f"{name}: controls identical {sum(eq)}/{len(eq)}")
        fp = [part[t][0] == off[t][0] for t in part]
        if not all(fp):
            fails.append(f"{name}: fingerprint mismatch on {len(fp) - sum(fp)} rows")
    missing = [t for t in changed if t not in arm and t not in rest]
    if missing:
        fails.append(f"{len(missing)} changed tokens have no fix row")
    if fails:
        raise SystemExit("REFUSED: " + "; ".join(fails))
    toks = [str(t) for t in zo["token"]]
    src = {t: ("A9" if t in set(T9["confirm"]) else "rest" if t in set(TR["rest"]) else "off") for t in toks}
    poses = np.stack([(arm if src[t] == "A9" else rest if src[t] == "rest" else off)[t][1] for t in toks]).astype(np.float32)
    out = f"{SEAMS}/refe_navtest_final_routefix.npz"
    np.savez(out, token=zo["token"], fingerprint=zo["fingerprint"], poses=poses, sampling=zo["sampling"],
             arm=np.array("REFe_final_routefix"))
    rep = {"what": __doc__.split("\n")[0], "rows": len(toks), "misses": 0,
           "rows_from": {k: sum(1 for v in src.values() if v == k) for k in ("A9", "rest", "off")},
           "controls_identical": {"A9": f"{len(T9['controls'])}/{len(T9['controls'])}", "rest": f"{len(TR['controls'])}/{len(TR['controls'])}"},
           "goal_fix": "pdm_route", "rule": "navsim_v1", "repair_last_heading": True, "sanitize_goal": False,
           "frame_control_parts_max_m": {"off": json.load(open(f"{SEAMS}/refe_navtest_final.report.json"))["frame_control"]["max_m"],
                                         "A9": json.load(open(f"{A9}/seam_pdm_route.report.json"))["frame_control"]["max_m"],
                                         "rest": json.load(open(f"{A9}/seam_adopted_rest.report.json"))["frame_control"]["max_m"]},
           "frame_control_note": "36 tokens / 4 logs fail the 1 mm bar in every part that contains them (a nuPlan-DB vs OpenScene pose "
                                 "disagreement, raw/2026-10-04-navtest-final/frame_control_navtest_full.json); declared, not a pairing error",
           "sources": {"off": f"{SEAMS}/refe_navtest_final.npz", "A9": f"{A9}/seam_pdm_route.npz", "rest": f"{A9}/seam_adopted_rest.npz"}}
    json.dump(rep, open(f"{SEAMS}/refe_navtest_final_routefix.report.json", "w", encoding="utf-8"), indent=1)
    print("ZZCOMPOSE_OK", json.dumps(rep["rows_from"]), rep["controls_identical"])


if __name__ == "__main__":
    main()
