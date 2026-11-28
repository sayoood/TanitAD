#!/usr/bin/env python3
"""nhwu_assert.py (dev box) -- assert on the two parse8 summaries (PARSE8_README "assert on the artifacts, not the exit code"):
both exist and parse; label RESULT; ``arms_refused == {}``; every expected arm present and not UNAVAILABLE (warmup: R8_LEGAL,
R8_LEGAL_s1, R8_BLIND, PRIOR_ha0p + the 3 floors; navhard: the same minus R8_BLIND); ``seed_floor_S2_EPDMS_u`` non-null on
both; navhard ``intervals.R8_LEGAL.status == "OK"``.  Pair / interval records that read UNAVAILABLE are LISTED (not hidden).
Writes --out with ``verdict`` and a digest of the per-arm headline numbers EXACTLY as parse8 wrote them (no rounding).
Tier of every number: NavSim v2 open-loop two-stage (T1-family), zero-shot, ONE training seed; Thor CPU backend (model arms)
vs dev-box floors -> MIXED-backend under RULING_BACKEND_POLICY.md.
"""
import argparse
import json
import os

EXP = {"warmup": ["R8_LEGAL", "R8_LEGAL_s1", "R8_BLIND", "PRIOR_ha0p", "CV_official", "STOP_zero", "ECHO_ha0_ext"],
       "navhard": ["R8_LEGAL", "R8_LEGAL_s1", "PRIOR_ha0p", "CV_official", "STOP_zero", "ECHO_ha0_ext"]}


def walk_unavail(o, path, out):
    if isinstance(o, dict):
        if o.get("status") == "UNAVAILABLE":
            out.append(f"{path}: {o.get('reason')}")
        for k, v in o.items():
            walk_unavail(v, f"{path}.{k}", out)


ap = argparse.ArgumentParser()
ap.add_argument("--summaries", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
bad, digest, unavail = [], {}, []
for split in ("warmup", "navhard"):
    p = os.path.join(a.summaries, f"summary_{split}.json")
    if not os.path.isfile(p):
        bad.append(f"{split}: summary missing")
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except ValueError as e:
        bad.append(f"{split}: unparsable {e}")
        continue
    if "RESULT" not in str(d.get("_label")):
        bad.append(f"{split}: label {d.get('_label')!r}")
    if d.get("arms_refused") != {}:
        bad.append(f"{split}: arms_refused {d.get('arms_refused')}")
    for arm in EXP[split]:
        r = d.get("arms", {}).get(arm)
        if not isinstance(r, dict) or r.get("status") == "UNAVAILABLE":
            bad.append(f"{split}: arm {arm} {r if not isinstance(r, dict) else r.get('reason')}")
    if d.get("seed_floor_S2_EPDMS_u") is None:
        bad.append(f"{split}: seed_floor_S2_EPDMS_u null ({d.get('seed_floor_S2_EPDMS_u_status')})")
    if split == "navhard" and (d.get("intervals", {}).get("R8_LEGAL") or {}).get("status") != "OK":
        bad.append(f"navhard: intervals.R8_LEGAL.status {(d.get('intervals', {}).get('R8_LEGAL') or {}).get('status')}")
    walk_unavail({k: d.get(k) for k in ("pairs", "intervals", "paired_intervals")}, split, unavail)
    digest[split] = {"label": d.get("_label"), "seed_floor_S2_EPDMS_u": d.get("seed_floor_S2_EPDMS_u"),
                     "arms": {arm: {k: (lambda v: {kk: v.get(kk) for kk in ("value", "n_groups", "n_scenes")}
                                        if isinstance(v, dict) else v)(r.get(k) if isinstance(r, dict) else None)
                                    for k in ("official_two_stage_EPDMS", "S2_EPDMS_u", "n_stage1", "n_stage2", "stage1_rows_own")}
                              for arm, r in d.get("arms", {}).items()}}
json.dump({"verdict": "FAIL" if bad else "PASS", "bad": bad, "unavailable_records": unavail, "digest": digest,
           "tier": "NavSim v2 open-loop two-stage EPDMS (T1-family), zero-shot, one training seed; model arms Thor CPU backend, "
                   "floors dev-box banks (MIXED-backend, RULING_BACKEND_POLICY.md)"},
          open(a.out, "w", encoding="utf-8"), indent=1)
print(json.dumps({"verdict": "FAIL" if bad else "PASS", "bad": bad[:6], "n_unavailable_records": len(unavail)}))
