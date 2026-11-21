"""wh_gsuite_from_pod2.py -- (the c02bf3a6 run; same code as wh_gsuite_from_pod.py with the commit and trees of that run) turn the pod's FULL G-SUITE run of the weather-head commit into wh_check's G-SUITE result (schema wh-gsuite-result-v1, scope "full").
Why a pod run: the full suite (14,851+ tests) runs in 8 shards in the base composition on the pod, as STRAT1's seeds commit a4357e79 was checked; the dev-box route
(`wh_check.py suite --full`) would run it serially twice.  The comparison is wh_check's OWN code: junit_results() on every shard xml and suite_compare(), with the same
planted-failure regression arm as cmd_suite.  The tree manifest sha256 is wh_check.tree_sha() of the dev-box tree, whose 5 changed files are byte-equal to the pod tree's
(sha256 list wh5b.sha256, checked on the pod by wh_gsuite2.sh, which also checks v8f_t1_eval.py at its pin 180044eb).  The pod's own strat1_check.py comparison (gsuite_wh2.json) is printed beside and must agree on n_regressions.
Usage: python wh_gsuite_from_pod.py --base-dir <junit new6 xmls> --new-dir <junit new8 xmls> --pod-compare gsuite_wh2.json --out <runs>/gate/gsuite_result.json"""
import argparse
import glob
import json
import os
import sys

BUILD = "F:/TanitAD-scratch/strat1/whead/build"
sys.path.insert(0, BUILD)
import wh_check as C  # noqa: E402


def load(d):
    out = {}
    fs = sorted(glob.glob(os.path.join(d, "shard*.xml")))
    assert len(fs) == 8, (d, len(fs))
    for f in fs:
        r = C.junit_results(f)
        dup = set(r) & set(out)
        assert not dup, f"a test id appears in two shards: {sorted(dup)[:3]}"
        out.update(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--pod-compare", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, new = load(a.base_dir), load(a.new_dir)
    r = C.suite_compare(base, new)
    planted = dict(new)
    k = next(t for t, s in base.items() if s == "pass")
    planted[k] = "fail"
    arm = C.suite_compare(base, planted)["n_regressions"] > 0
    r["regression_arm"] = {"read": "FAIL" if arm else "PASS", "must_read": "FAIL", "pass": arm}
    pod = json.load(open(a.pod_compare, encoding="utf-8"))
    agree = pod.get("n_regressions") == r["n_regressions"]
    res = {"schema": "wh-gsuite-result-v1", "written_by": "wh_gsuite_from_pod2.py (pod 8-shard full run in the base composition, compared by wh_check.suite_compare)",
           "base_commit": C.W.BASE_COMMIT, "new_commit": "c02bf3a6e77fdcd316a9cadc04224d3e3154abf7", "scope": "full", "venue": "LoRA pod CPU, trees /root/strat1/new4 (a4357e79) and new8 (c02bf3a6)",
           "tests": "all of stack/tests (761 files, 8 shards)", "tree_manifest_sha256": C.tree_sha(), **r,
           "pod_strat1_check_compare": {k: pod.get(k) for k in ("base", "new", "n_regressions", "n_new_only")}, "pod_compare_agrees": agree,
           "pass": bool(r["n_regressions"] == 0 and arm and r["new_n"] > 0 and r["base_n"] > 0 and agree)}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res[k] for k in ("scope", "base_n", "new_n", "n_regressions", "new_only_failures", "pod_compare_agrees", "pass")}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
