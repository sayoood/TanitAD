#!/usr/bin/env python3
"""mutation_speedsub_pin.py -- the mutation pass over ``v8f_speedsub_t1_pin.py`` (argv renderer, done records, pin refusals).

Each mutant is the pin script with ONE line changed; ``test_v8f_speedsub_t1_pin.py`` is run against it (``SPEEDSUB_PIN=<mutant>``) as a pytest subprocess; KILLED iff pytest exits non-zero.
The unmutated script must pass first. A SURVIVOR is a finding.

    python mutation_speedsub_pin.py --pin v8f_speedsub_t1_pin.py --tests test_v8f_speedsub_t1_pin.py --work <scratch> --out mutation_pin.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MUTANTS = [
    ("render_not_run_unchecked", "if t[2] != \"run\":", "if False:"),
    ("render_ckpt_unchecked", "if seen.get(\"--ckpt\") != CKPT[run]:", "if False:"),
    ("render_expect_unchecked", "if seen.get(k) != v:", "if False:"),
    ("render_dump_heads_unchecked", "if k not in seen:", "if False:"),
    ("render_logged_any_run", "if mode in LOGGED and run not in (\"CTRL_s0\", \"CTRL_s1\"):", "if False:"),
    ("render_logged_limit", "new[\"--limit\"] = \"logged\"", "new[\"--limit\"] = \"causal\""),
    ("render_smoke_n", "res += [\"--max-windows\", str(int(smoke))]", "res += [\"--max-windows\", \"100\"]"),
    ("render_dump_ticks", "\"--dump-ticks\": f\"{d}/ticks\"}", "\"--dump-ticks\": f\"{d}\"}"),
    ("render_smoke_dir", "os.path.join(out_root, \"SMOKE\" if smoke else \"\", run, mode)", "os.path.join(out_root, \"\", run, mode)"),
    ("tag_smoke_prefix", "{'SMOKE_' if smoke else ''}", "{''}"),
    ("done_tag_unchecked", "d.get(\"tag\") == tag and ", ""),
    ("done_before_unchecked", "d.get(\"ckpt_sha256_before\") == ckpt_sha and ", ""),
    ("done_after_unchecked", "d.get(\"ckpt_sha256_after\") == ckpt_sha and ", ""),
    ("done_digest_unchecked", " and d.get(\"tool_digest\") == TOOL_DIGEST)", ")"),
    ("write_done_ckpt_unchecked", "if after != a.sha_before:", "if False:"),
    ("write_done_missing_unchecked", "    if missing:\n        problems.append", "    if False:\n        problems.append"),
    ("write_done_dump_count", "len(dumps) != N_EPISODES", "len(dumps) < 1"),
    ("write_done_smoke_empty", "(a.smoke and not dumps)", "(False)"),
    ("write_done_limit", "\"limit\": \"logged\" if a.mode in LOGGED else \"causal\"", "\"limit\": \"causal\""),
    ("write_done_digest", "\"tool_digest\": TOOL_DIGEST, \"n_dumps\"", "\"tool_digest\": \"0\" * 64, \"n_dumps\""),
    ("pin_prefix_unchecked", "if not sha.startswith(CKPT_PREFIX[r]):", "if False:"),
    ("pin_overwrite", "if old != pins:", "if False:"),
    ("prefix_constant", "\"CTRL_s0\": \"67265ec1\"", "\"CTRL_s0\": \"67265ec2\""),
    ("n_episodes_constant", "N_EPISODES = 141", "N_EPISODES = 140"),
    ("get_key_split", "for k in a.key.split(\".\"):", "for k in [a.key]:"),
]


def apply(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f"mutant anchor occurs {src.count(old)} times (need exactly 1): {old[:90]!r}")
    return src.replace(old, new, 1)


def pytest_run(tests, pin, tmp):
    env = dict(os.environ, SPEEDSUB_PIN=pin, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", tmp, tests], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=900)
    failed = [ln.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "").strip() for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    return {"rc": p.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = open(a.pin, encoding="utf-8").read()
    os.makedirs(os.path.join(a.work, "pytmp"), exist_ok=True)
    res = {"baseline": pytest_run(a.tests, a.pin, os.path.join(a.work, "pytmp", "baseline")), "mutants": {}}
    print("baseline:", res["baseline"], flush=True)
    if res["baseline"]["rc"] != 0:
        json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
        print("ZZ-MUTATION-STOP baseline does not pass")
        return 2
    for name, old, new in MUTANTS:
        d = os.path.join(a.work, "mutants", name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "v8f_speedsub_t1_pin.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(apply(src, old, new))
        r = pytest_run(a.tests, path, os.path.join(a.work, "pytmp", name))
        r["killed"] = r["rc"] != 0
        res["mutants"][name] = r
        print(f"{'KILLED  ' if r['killed'] else 'SURVIVED'} {name}: {r['failed'][:1]} ({r['seconds']}s)", flush=True)
    res["n_mutants"], res["n_killed"] = len(res["mutants"]), sum(r["killed"] for r in res["mutants"].values())
    res["survivors"] = sorted(n for n, r in res["mutants"].items() if not r["killed"])
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"ZZ-MUTATION-DONE {res['n_killed']}/{res['n_mutants']} killed; survivors={res['survivors']}", flush=True)
    return 0 if not res["survivors"] else 1


if __name__ == "__main__":
    sys.exit(main())
