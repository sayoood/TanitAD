"""Mutation check for --strategic-off (R6): every known-bad variant must turn at least one test RED.
Mutates the merge copy IN PLACE, runs the R6 test file, restores the original bytes and verifies by sha256."""
import hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path("C:/Users/Admin/v7f_merge/stack")
V6 = ROOT / "tanitad/models/v6.py"
TR = ROOT / "scripts/train_v6_staged.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ENV = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
CRLF = "\r\n"
MUTANTS = [
    ("M5 the flag is ignored (always the real strategic embedding)", V6,
     "        e_g_str_down = (torch.zeros_like(e_g_str)" + CRLF +
     "                        if getattr(self.cfg, \"strategic_off\", False) else e_g_str)" + CRLF,
     "        e_g_str_down = e_g_str" + CRLF),
    ("M6 the tac_goal_cond refusal removed", V6,
     "        if getattr(self, \"strategic_off\", False) and getattr(self, \"tac_goal_cond\", False):" + CRLF,
     "        if False:" + CRLF),
    ("M7 the trainer no longer maps the flag", TR,
     "        strategic_off=bool(getattr(a, \"strategic_off\", False))," + CRLF, ""),
    ("M8 only the factored heads get the zero (mixed head reads the real embedding)", V6,
     "        g_tac = self.goal_head_tac(z_tac_p, cond=e_g_str_down)" + CRLF,
     "        g_tac = self.goal_head_tac(z_tac_p, cond=e_g_str)" + CRLF),
    ("M9 only the mixed head gets the zero (factored heads read the real embedding)", V6,
     "            g_tac_lat = self.goal_head_tac_lat(z_tac_p, cond=e_g_str_down)" + CRLF +
     "            g_tac_lon = self.goal_head_tac_lon(z_tac_p, cond=e_g_str_down)" + CRLF,
     "            g_tac_lat = self.goal_head_tac_lat(z_tac_p, cond=e_g_str)" + CRLF +
     "            g_tac_lon = self.goal_head_tac_lon(z_tac_p, cond=e_g_str)" + CRLF),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_tests():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v7f_r6_strategic_off.py"],
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
        if text.count(old) != 1:
            raise SystemExit(f"{name}: anchor matched {text.count(old)} times")
        path.write_bytes(text.replace(old, new).encode("utf-8"))
        rc, tail = run_tests()
        out.append({"mutant": name, "rc": rc, "tail": tail, "caught": rc != 0})
        path.write_bytes(orig[path])
finally:
    for p, b in orig.items():
        p.write_bytes(b)
restored = all(sha(p) == h0[p] for p in orig)
print(json.dumps({"results": out, "restored_bit_identical": restored}, indent=1))
sys.exit(0 if restored and out[0]["rc"] == 0 and all(r.get("caught") for r in out[1:]) else 1)
