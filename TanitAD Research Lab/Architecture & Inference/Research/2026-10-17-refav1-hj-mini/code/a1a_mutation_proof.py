"""Deletion-mutation proof for the A1a arm (PREREG_RA1_ES Addendum D): delete ONE line of the real implementation, re-run the A1a tests,
require them RED; restore, require them GREEN (the control).

usage: a1a_mutation_proof.py <tree root> <out.json>      (the tree's stack/tanitad/models/ra1_hjepa.py is mutated IN PLACE and restored)"""
import json
import os
import re
import subprocess
import sys

MUTATIONS = [
    ("M1 the permutation is drawn but never APPLIED", "                a_seq = a_seq[aperm]\n", ""),
    ("M2 G-LIVE no longer gates on the A1a check", "    ok = not missing and not nonfinite and not unexpected and idm_ok and shuffle_ok and act_ok\n",
     "    ok = not missing and not nonfinite and not unexpected and idm_ok and shuffle_ok\n"),
    ("M3 the G-DVB shuffle_act row is deleted",
     '    add("shuffle_act", bool(cfg.spec.shuffle_act), bool(getattr(model, "shuffle_act", False)))   # A1a (Addendum D)\n', ""),
    ("M4 the arm is built WITHOUT its flag", '        self.shuffle_act = bool(sp.shuffle_act)      # A1a; forward_batch reads THIS built flag, and G-DVB reads it back\n',
     "        self.shuffle_act = False\n"),
]


def run_tests(stack):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_ra1_hjepa.py", "-k", "a1a or A1a"],
                       cwd=stack, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    failed = sorted(set(re.findall(r"^FAILED tests/test_ra1_hjepa.py::(\S+)", r.stdout, re.M)))
    tail = [l for l in r.stdout.splitlines() if re.search(r"\d+ (passed|failed)", l)]
    return dict(rc=r.returncode, failed=failed, summary=tail[-1] if tail else r.stdout[-300:])


def main(root, out):
    stack = os.path.join(root, "stack")
    path = os.path.join(stack, "tanitad", "models", "ra1_hjepa.py")
    orig = open(path, "rb").read()
    res = dict(schema="ra1-a1a-mutation-proof-v1", file="stack/tanitad/models/ra1_hjepa.py", tests="test_ra1_hjepa.py -k 'a1a or A1a'",
               control_before=run_tests(stack), mutations=[])
    try:
        for name, line, repl in MUTATIONS:
            src = orig.decode("utf-8")
            assert src.count(line) == 1, f"{name}: the line moved"
            open(path, "wb").write(src.replace(line, repl).encode("utf-8"))
            r = run_tests(stack)
            r.update(name=name, red=bool(r["rc"] != 0 and r["failed"]))
            res["mutations"].append(r)
            open(path, "wb").write(orig)
            print(name, "->", "RED" if r["red"] else "GREEN (!)", r["summary"], r["failed"], flush=True)
    finally:
        open(path, "wb").write(orig)
    res["control_after"] = run_tests(stack)
    res["ok"] = bool(res["control_before"]["rc"] == 0 and res["control_after"]["rc"] == 0 and all(m["red"] for m in res["mutations"]))
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print("control before:", res["control_before"]["summary"], "| after:", res["control_after"]["summary"], "| ok =", res["ok"])
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
