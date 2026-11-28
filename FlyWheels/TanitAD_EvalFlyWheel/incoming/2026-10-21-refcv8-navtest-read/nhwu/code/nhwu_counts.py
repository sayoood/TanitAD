#!/usr/bin/env python3
"""nhwu_counts.py (Thor, any python3) -- the artifact assertion after the scoring pool: every warmup arm's
counts.json PASS with 220 valid rows, backend thor and the seam sha256 the job builder staged; every navhard shard
(<ARM>_THOR_k{k}of6) counts.json PASS with the same seam sha256, the shard's token count, and 0 FINGERPRINT_MISMATCH /
UNKNOWN_TOKEN agent calls.  Prints ``QQCNT <json list of failures>`` and writes --out.

    python3 nhwu_counts.py --sw <scores_warmup> --sn <scores_navhard_thor> --warms "A B" --narms "A B" \
        --wu-report jobs_wu_report.json --nh-report jobs_nh_report.json --out counts_check.json
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--sw", required=True)
ap.add_argument("--sn", required=True)
ap.add_argument("--warms", required=True)
ap.add_argument("--narms", required=True)
ap.add_argument("--wu-report", required=True)
ap.add_argument("--nh-report", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
wr = json.load(open(a.wu_report))["arms"]
nr = json.load(open(a.nh_report))["arms"]
bad, rec = [], {}


def calls_ok(c):
    ac = c.get("agent_calls") or {}
    return not ac.get("FINGERPRINT_MISMATCH") and not ac.get("UNKNOWN_TOKEN")


for arm in a.warms.split():
    p = os.path.join(a.sw, f"score_{arm}.counts.json")
    c = json.load(open(p)) if os.path.exists(p) else {}
    rec[f"warmup/{arm}"] = {k: c.get(k) for k in ("status", "csv_valid_rows", "agent_calls", "seam_sha256", "wall_s", "backend")}
    if (c.get("status") != "PASS" or c.get("csv_valid_rows") != 220 or c.get("backend") != "thor"
            or c.get("seam_sha256") != wr[arm]["seam_sha256"] or not calls_ok(c)):
        bad.append(f"warmup {arm}: {rec[f'warmup/{arm}']} failures={c.get('failures')}")
for arm in a.narms.split():
    tot = 0
    for k in range(6):
        d = os.path.join(a.sn, "_shards", f"{arm}_THOR_k{k}of6")
        p = os.path.join(d, f"score_{arm}_THOR__navhard_two_stage.counts.json")
        c = json.load(open(p)) if os.path.exists(p) else {}
        sub = json.load(open(c["subset"])) if c.get("subset") and os.path.exists(c["subset"]) else {}
        n_exp = len(sub.get("expected_tokens", []))
        rec[f"navhard/{arm}/k{k}"] = {kk: c.get(kk) for kk in ("status", "csv_valid_rows", "agent_calls", "seam_sha256", "wall_s")}
        rec[f"navhard/{arm}/k{k}"]["n_expected_subset"] = n_exp
        tot += c.get("csv_valid_rows") or 0
        if (c.get("status") != "PASS" or c.get("seam_sha256") != nr[arm]["seam_sha256"] or not calls_ok(c)
                or not n_exp or c.get("csv_valid_rows") != n_exp):
            bad.append(f"navhard {arm} k{k}: {rec[f'navhard/{arm}/k{k}']} failures={c.get('failures')}")
    rec[f"navhard/{arm}/total_rows"] = tot
    if tot != 5912:
        bad.append(f"navhard {arm}: shard rows sum {tot} != 5912")
json.dump({"verdict": "FAIL" if bad else "PASS", "bad": bad, "records": rec}, open(a.out, "w"), indent=1)
print("QQCNT " + json.dumps(bad))
