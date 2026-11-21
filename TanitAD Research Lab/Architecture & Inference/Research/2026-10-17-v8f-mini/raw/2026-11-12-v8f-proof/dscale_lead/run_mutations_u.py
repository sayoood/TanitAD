"""run_mutations_u.py -- the mutation runner of card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD). NOT RUN by the builder; the lead runs it (pattern of finegrid_r/run_mutations_gr.py).

Reads the list MUTATIONS_U.json. Each mutation is a text edit of a COPY of the target file (the old text must occur exactly once); the named test file is run against the copy (through the environment variable
that the test file reads) and must turn RED; the unmutated suites are run first and last and must be GREEN. Result: MUTATIONS_U_RESULT.json (the list file is never overwritten).

  python run_mutations_u.py [--targets precheck,reader,tool] [--python PYTHON]
Targets: precheck (numpy only), reader (needs the pilot-reader chain and its paths: run it where dscale_lead_read.py imports), tool (needs torch and DSCALE_TREE: the pod).
Every run folder (copies, pytest basetemp) is new and unique per mutation; nothing is removed by this script.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work_mut_u")
TARGET_FILE = {"tool": "tok_probe_u.py", "precheck": "dscale_precheck.py", "reader": "dscale_lead_read.py", "evalview": "dscale_evalview.py"}
TARGET_ENV = {"tool": "TOK_PROBE_U_FILE", "precheck": "DSCALE_PRECHECK_FILE", "reader": "DSCALE_LEAD_READ_FILE", "evalview": "DSCALE_EVALVIEW_FILE"}
SUITE_OF = {"tool": "test_tok_probe_u.py", "precheck": "test_dscale_precheck.py", "reader": "test_dscale_lead_read.py", "evalview": "test_dscale_evalview.py"}


def pytest(py, test_file, env_extra, k, base):
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(HERE, test_file), "--basetemp", base, "-x", "-k", k]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=3600, cwd=HERE)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="precheck,reader,tool")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    targets = [t for t in a.targets.split(",") if t]
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    run_dir = os.path.join(WORK, stamp)
    os.makedirs(os.path.join(run_dir, "copies"))
    spec = json.load(open(os.path.join(HERE, "MUTATIONS_U.json"), encoding="utf-8"))
    res = {"card": spec["card"], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": targets, "suite_before": {}, "suite_after": {}, "mutations": []}
    for t in targets:
        rc, last = pytest(a.python, SUITE_OF[t], {}, "not zz_none", os.path.join(run_dir, "pt_before_" + t))
        res["suite_before"][t] = {"green": rc == 0, "last_line": last}
        print("before", t, "GREEN" if rc == 0 else "RED", last, flush=True)
    for m in spec["mutations"]:
        if m["target"] not in targets:
            continue
        src = os.path.join(HERE, TARGET_FILE[m["target"]])
        txt = open(src, encoding="utf-8", newline="").read()
        if txt.count(m["old"]) != 1:
            res["mutations"].append({"name": m["name"], "error": "old text occurs %d times" % txt.count(m["old"])})
            print(m["name"], "ERROR old text count", txt.count(m["old"]), flush=True)
            continue
        d = os.path.join(run_dir, "copies", m["name"])
        os.makedirs(d)
        dst = os.path.join(d, TARGET_FILE[m["target"]])
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt.replace(m["old"], m["new"]))
        if m["target"] == "tool":                                                      # the precheck subcommand imports its sibling
            shutil.copyfile(os.path.join(HERE, "dscale_precheck.py"), os.path.join(d, "dscale_precheck.py"))
        env = {TARGET_ENV[m["target"]]: dst}
        rc, last = pytest(a.python, m["test_file"], env, m["k"], os.path.join(run_dir, "pt_" + m["name"]))
        res["mutations"].append({"name": m["name"], "target": TARGET_FILE[m["target"]], "red": rc != 0, "pytest_rc": rc, "last_line": last})
        print(m["name"], "RED" if rc != 0 else "STILL GREEN (a hole)", flush=True)
    for t in targets:
        rc, last = pytest(a.python, SUITE_OF[t], {}, "not zz_none", os.path.join(run_dir, "pt_after_" + t))
        res["suite_after"][t] = {"green": rc == 0, "last_line": last}
        print("after", t, "GREEN" if rc == 0 else "RED", last, flush=True)
    res["n_mutations"] = len(res["mutations"])
    res["n_red"] = sum(1 for m in res["mutations"] if m.get("red"))
    out = os.path.join(HERE, "MUTATIONS_U_RESULT.json")
    if os.path.exists(out):
        out = os.path.join(run_dir, "MUTATIONS_U_RESULT.json")                          # never overwrite an earlier result
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    ok = res["n_red"] == res["n_mutations"] and all(v["green"] for v in res["suite_before"].values()) and all(v["green"] for v in res["suite_after"].values())
    print("ZZ MUTATIONS-U DONE %d of %d red; suites before %s after %s -> %s" % (res["n_red"], res["n_mutations"], res["suite_before"], res["suite_after"], out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
