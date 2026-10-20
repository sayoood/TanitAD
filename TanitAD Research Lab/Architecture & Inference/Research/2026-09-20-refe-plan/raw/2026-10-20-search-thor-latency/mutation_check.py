#!/usr/bin/env python3
"""Mutation check of thor_search_latency.py's guard lines: each mutation disables ONE guard in a scratch copy of this directory, and `test_thor_search_latency.py` MUST then fail.
A guard whose removal leaves the suite green is untested.  The unmutated copy must pass first (the control); a mutation whose `find` text is not found exactly once is reported
MUTATION_NOT_APPLICABLE (a vacuous mutation must not read as 'caught').
    CUDA_VISIBLE_DEVICES=-1 python mutation_check.py --out mutations.json
Exit 0 only if every mutation is caught."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
F = "thor_search_latency.py"
MUT = [
    # --- the answer must not change under timing
    (F, "same = outputs_identical(ref1[a][i], out)", "same = True", "IDENTITY: the timed output is compared with the untimed reference"),
    (F, "x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()", "True", "IDENTITY: array equality is bitwise (dtype, shape, bytes)"),
    (F, "return bool(x == y) and type(x) is type(y)", "return True", "IDENTITY: scalars compare by value and type"),
    (F, '"ok": len(m["nondeterministic_reference_tokens"][a]) == 0, "nondeterministic_tokens"', '"ok": True, "nondeterministic_tokens"', "DETERMINISM: the device floor is a control"),
    (F, 'ctl_ok = all(v["ok"] for v in ctl.values())', "ctl_ok = True", "CONJ: controls_ok is the conjunction of every control"),
    (F, 'return EXIT["ok"] if rec["controls_ok"] else EXIT["controls_red"]', 'return EXIT["ok"]', "EXITCODE: a RED control exits non-zero"),
    (F, '"is_thor_number": bool(target == "thor" and rep["controls_ok"] and (preflight or {}).get("ok"))', '"is_thor_number": bool(target == "thor")', "THORNUMBER: needs green controls and a passed preflight"),
    # --- counts, the work, the floor
    (F, 'if c != exp["calls"] or r != exp["rows"])', "if False)", "COUNTS: scorer calls and rows per token equal the config's formula"),
    (F, 'return {"calls": k + 1 + elites,', 'return {"calls": k + elites,', "COUNTS: the call formula"),
    (F, "self.rows += int(np.shape(S)[0])", "self.rows += 0", "COUNTS: rows are counted"),
    (F, "self.calls += 1\n        self.rows", "self.rows", "COUNTS: calls are counted"),
    (F, 'bool(a_s["p50"] < arms["b"]["ms"]["p50"] < arms["c"]["ms"]["p50"])', "True", "SEESWORK: a < b < c"),
    (F, '"ok": fl <= 0.05 * a_s["p50"],', '"ok": True,', "FLOOR: the no-op timing floor is far below what is measured"),
    (F, "n0.append(bool(", "n0.append(True or bool(", "N0: k_iters = 0 returns the pick bit for bit with one call"),
    (F, "n_warm = min(int(warmup), n)", "n_warm = 0", "WARMUP: warm-up tokens are actually run"),
    (F, "k = (i + r) % len(order)", "k = 0", "ROTATE: arms rotate through every position"),
    (F, 'return {"logits": logits, "pick": int(np.argmax(A.aggregate(logits)))}', 'return {"logits": logits, "pick": int(np.argmin(A.aggregate(logits)))}', "PICKONLY: (a) is argmax of the aggregate"),
    # --- the sync bracket
    (F, "        self.sync()\n        t0 = self.clock_ns()\n", "        t0 = self.clock_ns()\n", "BRACKET: sync before each scorer call is timed"),
    (F, "        out = self.fn(S)\n        self.sync()\n", "        out = self.fn(S)\n", "BRACKET: sync after each scorer call"),
    (F, "    sync()\n    t0 = clock_ns()\n    out = run(counter)\n", "    t0 = clock_ns()\n    out = run(counter)\n", "BRACKET: outer sync before the token is timed"),
    (F, "    out = run(counter)\n    sync()\n", "    out = run(counter)\n", "BRACKET: outer sync after the token"),
    # --- percentile maths
    (F, "return x[lo] + (x[hi] - x[lo]) * (pos - lo)", "return x[lo]", "PERCENTILE: linear interpolation"),
    (F, "pos = (len(x) - 1) * q / 100.0", "pos = len(x) * q / 100.0 - 1", "PERCENTILE: rank position"),
    (F, "if not all(math.isfinite(v) for v in x):", "if False:", "PERCENTILE: non-finite samples are refused"),
    (F, "if not den[k] > 0.0:", "if False:", "RATIO: a non-positive baseline is refused"),
    (F, "out[k] = num[k] / den[k]", "out[k] = den[k] / num[k]", "RATIO: numerator over denominator"),
    # --- device fields, read-only commands
    (F, "    if missing:                                                                                # @GUARD:DEVFIELDS", "    if False:                                                                                # @GUARD:DEVFIELDS", "DEVFIELDS: missing device fields are refused"),
    (F, "if t not in READONLY_ALLOW:", "if False:", "READONLY: only allowlisted commands run"),
    (F, "\n    (\"jetson_clocks\", \"--show\"),", "\n    (\"jetson_clocks\", \"--show\"), (\"jetson_clocks\",),", "READONLY: bare jetson_clocks (which SETS clocks) is not on the list"),
    (F, "\n    (\"nvpmodel\", \"-q\"),", "\n    (\"nvpmodel\", \"-q\"), (\"free\", \"-m\"),", "READONLY: no Thor memory probe on the list"),
    (F, "if age > max_age_s:", "if False:", "STALE: a stale pre-captured status is refused"),
    # --- the Thor-busy preflight
    (F, 'if JOB_RE.search(r["args"]) or (PY_RE.search(exe) and (me is None or r["user"] == me)):', 'if (PY_RE.search(exe) and (me is None or r["user"] == me)):', "BUSY: trainers / supervisors by command line"),
    (F, 'if JOB_RE.search(r["args"]) or (PY_RE.search(exe) and (me is None or r["user"] == me)):', 'if JOB_RE.search(r["args"]):', "BUSY: any other python job of this user"),
    (F, 'if r["pid"] in mine or EXCLUDE_RE.search(r["args"]):', 'if EXCLUDE_RE.search(r["args"]):', "BUSY: this process and its ancestors are excluded"),
    (F, "    if busy:   # @GUARD:BUSY", "    if False:   # @GUARD:BUSY", "BUSY: a busy process refuses the start"),
    (F, "    if rc != 0 or not rows:   # @GUARD:PSREAD", "    if False:   # @GUARD:PSREAD", "PSREAD: an unreadable process table refuses"),
    (F, "max(utils) >= GPU_UTIL_MAX_PCT", "False", "UTIL: GPU utilisation >= 5 % refuses"),
    (F, "        if v is None:   # @GUARD:UTILREAD", "        if False:   # @GUARD:UTILREAD", "UTILREAD: unreadable GPU utilisation refuses"),
    (F, "    if app_lines:   # @GUARD:APPS", "    if False:   # @GUARD:APPS", "APPS: listed compute apps refuse"),
    (F, "elif host != expect_hostname:", "elif False:", "HOST: the box must be the named one"),
    (F, "if not expect_hostname:", "if False:", "NOHOST: a run is bound to a named box"),
    (F, 'if lock.get("held_by_other"):', "if False:", "LOCK: a held Thor GPU flock refuses"),
    (F, 'rec["held_by_other"] = True', 'rec["held_by_other"] = False', "LOCK: a failed flock means held"),
    (F, '    if hold:\n        rec["held_by_us"] = True', '    if False:\n        rec["held_by_us"] = True', "LOCK: a free lock is held for the run"),
    (F, "    if foreign:                                                                                # @GUARD:ENDSCAN", "    if False:                                                                                # @GUARD:ENDSCAN", "ENDSCAN: a job that appeared during the run is contamination"),
    # --- inputs
    (F, "    if bad:                                                                                    # @GUARD:NOLABEL", "    if False:                                                                                    # @GUARD:NOLABEL", "NOLABEL: label arrays are never read"),
    (F, "if list(toks) != sorted(toks):", "if False:", "SORTED: the first N sorted tokens is well defined"),
    (F, "if got[fn] != want:", "if False:", "CODEPIN: the sealed Step B files are the landed ones"),
    (F, "    if got != want:                                                                            # @GUARD:PIN", "    if False:                                                                            # @GUARD:PIN", "PIN: a different file is refused"),
    (F, "if not (1 <= int(n) <= int(total)):", "if False:", "NTOK: the token count is in range"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="run only mutations whose label contains this text (and the control)")
    a = ap.parse_args(argv)
    res, caught = [], 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8")
    base = tempfile.mkdtemp(prefix="thorlatmut_")
    try:
        for k in range(-1, len(MUT)):
            if k >= 0 and a.only and a.only not in MUT[k][3]:
                continue
            tmp = os.path.join(base, f"m{k + 1}")
            shutil.copytree(HERE, tmp, ignore=shutil.ignore_patterns("__pycache__", "*.json", "pt*", ".pytest_cache"))
            label = "CONTROL (unmutated)"
            if k >= 0:
                f, old, new, label = MUT[k]
                p = os.path.join(tmp, f)
                s = open(p, encoding="utf-8", newline="").read()
                if s.count(old) != 1:
                    res.append({"mutation": label, "file": f, "status": "MUTATION_NOT_APPLICABLE", "occurrences": s.count(old)})
                    print("ZZMUT NOT_APPLICABLE", label, s.count(old), flush=True)
                    shutil.rmtree(tmp, ignore_errors=True)
                    continue
                open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", "test_thor_search_latency.py", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={os.path.join(tmp, 'ptmp')}"],
                               cwd=tmp, capture_output=True, text=True, env=env)
            ok = r.returncode == 0
            if k < 0:
                res.append({"mutation": label, "status": "CONTROL_GREEN" if ok else "CONTROL_RED", "tail": r.stdout[-300:]})
                print("ZZMUT_CONTROL", "GREEN" if ok else "RED", flush=True)
                if not ok:
                    print(r.stdout[-1500:])
                    json.dump(res, open(a.out, "w"), indent=1)
                    return 2
            else:
                failed = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")][:2]
                res.append({"mutation": label, "file": MUT[k][0], "status": "CAUGHT" if not ok else "SURVIVED", "first_failing": failed})
                caught += (not ok)
                print("ZZMUT", "CAUGHT  " if not ok else "SURVIVED", "|", label, flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    judged = [x for x in res if x["status"] in ("CAUGHT", "SURVIVED", "MUTATION_NOT_APPLICABLE")]
    bad = [x for x in judged if x["status"] != "CAUGHT"]
    json.dump({"caught": caught, "total": len(judged), "survived_or_inapplicable": bad, "results": res}, open(a.out, "w"), indent=1)
    print(f"ZZMUT_SUMMARY caught {caught} of {len(judged)}; survived/inapplicable: {[x['mutation'] for x in bad]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
