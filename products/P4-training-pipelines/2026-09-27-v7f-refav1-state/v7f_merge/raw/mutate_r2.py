"""Mutation check for the R2 nav fixes: every known-bad variant must turn at least one test RED.
Mutates the merge copy IN PLACE, runs the R2 test file, restores the original bytes and verifies by sha256."""
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path("C:/Users/Admin/v7f_merge/stack")
V6 = ROOT / "tanitad/models/v6.py"
TR = ROOT / "scripts/train_v6_staged.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
import os
ENV = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")

MUTANTS = [
    ("M1 synthetic_batch drops the nav keys", V6,
     'if getattr(self, "nav", None) is not None:\r\n            n_tok', 'if False:\r\n            n_tok'),
    ("M2 nav groups removed", V6,
     '        ("nav.embed.", "predictor_op"), ("nav.arg_proj.", "predictor_op"),\r\n', ''),
    ("M3 splat back ABOVE the .get forwarding (the tip order)", TR, None, None),
    ("M4 a duplicate .get line re-added AFTER the splat", TR,
     '                                    dt=float(getattr(a, "dt", 0.1)))))),\r\n        }\r\n',
     '                                    dt=float(getattr(a, "dt", 0.1)))))),\r\n'
     '            "nav_token": b.get("nav_token"),\r\n        }\r\n'),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_tests():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v7f_r2_nav_fixes.py"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


orig = {V6: V6.read_bytes(), TR: TR.read_bytes()}
h0 = {p: sha(p) for p in orig}
out = []
try:
    rc, tail = run_tests()
    out.append({"mutant": "M0 unmutated (must be GREEN)", "rc": rc, "tail": tail})
    for name, path, old, new in MUTANTS:
        text = orig[path].decode("utf-8")
        if name.startswith("M3"):
            # move the whole "(LAST)" splat block back above the `.get` line: the tip's order
            get_line = '            "nav_token": b.get("nav_token"), "nav_args": b.get("nav_args"),\r\n'
            i_get = text.index(get_line)
            i_blk = text.index("            # ⛔ R2 FIX (2026-09-27)", i_get)
            i_end = text.index("        }\r\n", i_blk)
            block = text[i_blk:i_end]
            mutated = text[:i_get] + block + get_line + text[i_end:]
            mutated = mutated.replace(get_line + block, block + get_line, 1) if False else mutated
            text2 = text[:i_get] + block + get_line + text[i_get + len(get_line):i_blk] + text[i_end:]
        else:
            if text.count(old) != 1:
                raise SystemExit(f"{name}: anchor matched {text.count(old)} times")
            text2 = text.replace(old, new)
        path.write_bytes(text2.encode("utf-8"))
        rc, tail = run_tests()
        out.append({"mutant": name, "rc": rc, "tail": tail, "caught": rc != 0})
        path.write_bytes(orig[path])
finally:
    for p, b in orig.items():
        p.write_bytes(b)
restored = all(sha(p) == h0[p] for p in orig)
print(json.dumps({"results": out, "restored_bit_identical": restored}, indent=1))
sys.exit(0 if restored and out[0]["rc"] == 0 and all(r.get("caught") for r in out[1:]) else 1)
