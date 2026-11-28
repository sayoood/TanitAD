"""Behavioural proof of the refcv6 supervisor's error-count token, old vs fixed.

Runs the supervisor's REAL block (cut out of each script between `    n_err=0` and its `echo "ZZ${ARM}`
line by run_block.sh) against four stderr logs and records the exact bytes of the token each
version emits. Expected values are literals.

usage: sup_token_proof.py <old_script> <fixed_script> <out.json>
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASH = r"C:\Program Files\Git\bin\bash.exe"
BENIGN = ("W0924 09:03:45.814000 3346338 torch/_inductor/utils.py:1953] [0/2] Not enough SMs to use "
          "max_autotune_gemm mode\n")
TB = "Trace" "back (most recent call last):\n"
CASES = {  # name: (log content, MODE)
    "empty_stderr": ("", ""),
    "one_benign_line_like_the_live_run": (BENIGN, ""),
    "two_tracebacks": (TB + "RuntimeError: x\n" + TB, ""),
    "grep_cannot_read_the_log_exit2": (BENIGN, "grep_exit2"),
}
EXPECT = {
    "old": {"empty_stderr": "ZZrefcv6-r101-s0-6571-50400-0-1ZZ",
            "one_benign_line_like_the_live_run": "ZZrefcv6-r101-s0-6571-50400-0\n0-1ZZ",
            "two_tracebacks": "ZZrefcv6-r101-s0-6571-50400-2-1ZZ",
            "grep_cannot_read_the_log_exit2": "ZZrefcv6-r101-s0-6571-50400-0-1ZZ"},
    "fixed": {"empty_stderr": "ZZrefcv6-r101-s0-6571-50400-0-1ZZ",
              "one_benign_line_like_the_live_run": "ZZrefcv6-r101-s0-6571-50400-0-1ZZ",
              "two_tracebacks": "ZZrefcv6-r101-s0-6571-50400-2-1ZZ",
              "grep_cannot_read_the_log_exit2": "ZZrefcv6-r101-s0-6571-50400-U-1ZZ"},
}


def token(script, log, mode):
    env = dict(os.environ, MODE=mode)
    r = subprocess.run([BASH, os.path.join(HERE, "run_block.sh"), script, log, "raw"],
                       capture_output=True, env=env)
    return r.returncode, r.stdout.decode("utf-8")


def main(old, fixed, out):
    res = {"_what": "the supervisor's error-count token, old (fe5872f) vs fixed, from its REAL block",
           "_evidence_class": "MEASURED (ours), Git Bash on the dev box", "cases": {}}
    ok = True
    for name, (content, mode) in CASES.items():
        log = os.path.join(HERE, f"case_{name}.log")
        with open(log, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        row = {}
        for ver, script in (("old", old), ("fixed", fixed)):
            rc, tok = token(script, log, mode)
            match = rc == 0 and tok == EXPECT[ver][name]
            ok &= match
            row[ver] = {"token": tok, "expected": EXPECT[ver][name], "match": match}
        res["cases"][name] = row
        os.remove(log)
    res["_VERDICT"] = ("PROVEN -- the old block splits its token on a benign stderr and reports an unread "
                       "log as 0; the fixed block emits one line and U" if ok else "FAILED -- see cases")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(res["_VERDICT"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
