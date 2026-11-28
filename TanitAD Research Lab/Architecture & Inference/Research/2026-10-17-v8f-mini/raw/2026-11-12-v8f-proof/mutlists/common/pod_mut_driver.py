"""pod_mut_driver.py -- apply deliberate one-line breakages to a COPY of a lever's tree and run the lever's test functions with minipytest (plain python, no pytest).

  python pod_mut_driver.py SPEC.json OUT.json [--only ID[,ID]] [--no-control]

SPEC.json: {"pristine": <dir of the pristine tree copy (contains stack/ and taniteval/)>, "work": <dir of the scratch tree copy>, "test_files": [paths relative to the tree root],
            "mutations": [{"id","file" (relative to the tree root),"old","new","expect":[test names]}]}
Each run (control, and every mutation) is a fresh child python process: the scratch file is mutated (old text must occur exactly once), the tests run, the pristine file is copied back over the scratch file.
Nothing is deleted. The child reads the tree through sys.path (tree/stack, tree/taniteval, tree/stack/scripts)."""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def child(work, test_files, tmp_root, extra_env=None):
    code = (
        "import sys, json; sys.path.insert(0, %r); import minipytest as mp; out={};\n"
        "for tf in %r:\n"
        "    r = mp.run_module(%r + '/' + tf, tmp_root=%r)\n"
        "    out.update({tf.split('/')[-1] + '::' + k: v for k, v in r.items()})\n"
        "print('@@JSON@@' + json.dumps(out))\n"
    ) % (HERE, test_files, work, tmp_root)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([work + "/stack", work + "/taniteval", work + "/stack/scripts"]), **(extra_env or {}))
    r = subprocess.run([PY, "-c", code], cwd=work, env=env, capture_output=True, text=True, timeout=850)
    for line in r.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return r.returncode, json.loads(line[8:]), ""
    return r.returncode, {}, (r.stderr or r.stdout)[-400:]


def main():
    spec = json.load(open(sys.argv[1]))
    outp = sys.argv[2]
    only = []
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    pr, wk, tfs = spec["pristine"], spec["work"], spec["test_files"]
    tmp_root = spec.get("tmp_root", wk + "/_tmp")
    os.makedirs(tmp_root, exist_ok=True)
    res = json.load(open(outp)) if os.path.exists(outp) else {"mutations": []}
    res["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if "--no-control" not in sys.argv and not only and not ("--control-only" in sys.argv and "control" not in res and False):
        t = time.time()
        rc, r, err = child(wk, tfs, tmp_root, spec.get("env"))
        ign = set(spec.get("ignore_failed", []))
        bad = [k for k, v in r.items() if not v["ok"] and k not in ign]
        res_ign = [k for k, v in r.items() if not v["ok"] and k in ign]
        res["control_after" if ("--control-only" in sys.argv and "control" in res) else "control"] = {"rc": rc, "n_tests": len(r), "failed": bad, "ignored_environment_failures": res_ign, "green": bool(r) and not bad, "seconds": round(time.time() - t), "err": err}
        print("CONTROL", {k: v for k, v in res.items() if k.startswith("control")}, flush=True)
    if "--control-only" in sys.argv:
        json.dump(res, open(outp, "w"), indent=1)
        return
    for m in spec["mutations"]:
        if only and m["id"] not in only:
            continue
        t = time.time()
        p = os.path.join(wk, m["file"])
        pp = os.path.join(pr, m["file"])
        shutil.copyfile(pp, p)
        s = open(p, encoding="utf-8", newline="").read()
        n = s.count(m["old"])
        rec = {k: m[k] for k in ("id", "file", "old", "new", "expect")}
        if n != 1:
            rec.update({"observed": "NOT-APPLIED", "why": "old text occurs %d times" % n})
        else:
            open(p, "w", encoding="utf-8", newline="").write(s.replace(m["old"], m["new"]))
            try:
                rc, r, err = child(wk, m.get("test_files", tfs), tmp_root, spec.get("env"))
            finally:
                shutil.copyfile(pp, p)
            failed = [k for k, v in r.items() if not v["ok"] and k not in set(spec.get("ignore_failed", []))]
            rec.update({"rc": rc, "n_tests": len(r), "failed": failed, "observed": "RED" if failed else ("ERROR" if not r else "STILL-GREEN"),
                        "expected_all_red": all(any(e in f for f in failed) for e in m["expect"]), "seconds": round(time.time() - t), "err": err})
        res["mutations"] = [x for x in res["mutations"] if x["id"] != m["id"]] + [rec]
        print("MUT", m["id"], rec["observed"], len(rec.get("failed", [])), flush=True)
        json.dump(res, open(outp, "w"), indent=1)
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
