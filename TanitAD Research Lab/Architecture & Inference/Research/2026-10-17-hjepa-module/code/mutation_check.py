"""Mutation check for stack/tests/test_hjepa.py: every deliberate regression must turn the suite RED.

Usage (from anywhere):  python mutation_check.py [path-to-venv-python]
It temporarily rewrites stack/tanitad/models/hjepa.py, runs the suite per mutation, and ALWAYS
restores the original bytes (try/finally). Do not run it while another process imports hjepa.
Expected: every mutation lists >= 1 FAILED test; the final line prints `restored True`.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]          # repo root (TanitAD)
STACK = REPO / "stack"
SRC = STACK / "tanitad" / "models" / "hjepa.py"
PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable
MUT = {
 "M1  drop IDM term from the loss": ("level_loss = pred + c.beta * sg + c.gamma * idm", "level_loss = pred + c.beta * sg"),
 "M2  SIGReg prefactor n -> 1/n": ("* float(n_total)", "/ float(n_total)"),
 "M3  ignore world size in the prefactor": ("n_total = n_total * int(world)", "n_total = n_total"),
 "M4  all-reduce backward halved": ("        return g, None", "        return g / 2, None"),
 "M5  macro-action window offset": ("lo = c.window - 1\n", "lo = 0\n"),
 "M6  action swap is a no-op": ("swapped[..., ix] = actions[perm][..., ix]", "swapped[..., ix] = actions[..., ix]"),
 "M7  planner never steps": ("            opt.step()\n", "            pass\n"),
 "M8  IDM target not detached": ("(pred - a_target.detach())", "(pred - a_target)"),
 "M9  non-causal attention": ("is_causal=True,", "is_causal=False,"),
 "M10 prediction target not shifted": ("tgt = z[:, s + 1:e + 1]", "tgt = z[:, s:e]"),
 "M12 SIGReg time in the sample axis": ("self.sigreg(z.transpose(0, 1), generator=generator)", "self.sigreg(z, generator=generator)"),
 "M13 action-SIGReg term dropped": ("level_loss = level_loss + c.act_sigreg * asg", "pass"),
 "M14 macro pooling off by one": ("e_sub = e[:, lo:lo + (n - 1) * c.stride]", "e_sub = e[:, lo + 1:lo + 1 + (n - 1) * c.stride]"),
}
env = dict(os.environ, OMP_NUM_THREADS="6", PYTHONPATH=str(STACK))
orig = SRC.read_bytes()
text = orig.decode("utf-8")
crlf = "\r\n" in text
try:
    for name, (old, new) in MUT.items():
        o, n = (old.replace("\n", "\r\n"), new.replace("\n", "\r\n")) if crlf else (old, new)
        assert text.count(o) == 1, (name, text.count(o))
        SRC.write_bytes(text.replace(o, n).encode("utf-8"))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_hjepa.py", "--tb=no", "-q"],
                           capture_output=True, text=True, env=env, cwd=STACK)
        failed = [l.split("::")[-1][:70] for l in r.stdout.splitlines() if l.startswith("FAILED")]
        print(f"{name:42s} -> {'CAUGHT' if failed else 'SURVIVED'} ({len(failed)} failing)")
finally:
    SRC.write_bytes(orig)
    print("restored", SRC.read_bytes() == orig)
