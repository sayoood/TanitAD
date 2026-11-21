"""Source-level mutation check for listen2.py: break ONE thing in a COPY of the module, run test_listen2.py against the copy (env LISTEN2_PY), and require that
at least one test goes RED; the unmutated module must be GREEN. The real listen2.py is never edited.  Run:  python run_mutations2.py  (CPU only).
Writes mutation_log.txt next to this file."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "listen2.py").read_text(encoding="utf-8")
MUT_DIR = HERE / "mut2"

#: (name, exact old text (must occur exactly once), new text)
MUTATIONS = [
    ("N1 E1 injects the donor nav at T instead of S",
     'r = LS.run_chain(model, P, Dn, {"S": ("nav",)}, ages=ages_d)', 'r = LS.run_chain(model, P, Dn, {"T": ("nav",)}, ages=ages_d)'),
    ("N2 E1 injects the donor nav at S AND T",
     'r = LS.run_chain(model, P, Dn, {"S": ("nav",)}, ages=ages_d)', 'r = LS.run_chain(model, P, Dn, {"S": ("nav",), "T": ("nav",)}, ages=ages_d)'),
    ("N3 E3 joint triple drops the T code", 'E3_TRIPLE = ("ego", "ck", "code")', 'E3_TRIPLE = ("ego", "ck")'),
    ("N4 the mean scene ignores the pad mask", "valid = (~pad)[..., None].double()", "valid = torch.ones_like(pad)[..., None].double()"),
    ("N5 every layer's mean token is taken at O's (NOW) tick", 'self._add(("tok", L, nm), LS.donor_token(m, P, L, nm))',
     'self._add(("tok", L, nm), LS.donor_token(m, P, "O", nm))'),
    ("N6 the mean-age replacement is zeros for a_t", "return a[:, 0].contiguous(), a[:, 1].contiguous()",
     "return torch.zeros_like(a[:, 0]), a[:, 1].contiguous()"),
    ("N7 O's mean cell leaves T's code in place", 'cd = src.code(model, P) if "code" in ov else P.code', "cd = P.code"),
    ("N8 the mean scene is entirely padded",
     "return t[None].expand(B, -1, -1).contiguous(), torch.zeros(B, t.shape[0], dtype=torch.bool)",
     "return t[None].expand(B, -1, -1).contiguous(), torch.ones(B, t.shape[0], dtype=torch.bool)"),
    ("N9 the own-value control hands T the checkpoints in O's frame", 'return P.ck_for_t if layer == "T" else P.ck_for_o', "return P.ck_for_o"),
    ("N10 the mean mode uses the window's own value", 'self.src = MeanSrc(means) if mode == "mean" else (OwnSrc() if mode == "own" else None)',
     "self.src = OwnSrc() if mode in ('mean', 'own') else None"),
    ("N11 the mean divides by n + 1", "f = lambda t: (t / self.n).float()", "f = lambda t: (t / (self.n + 1)).float()"),
    ("N12 T's mean checkpoints are taken in O's frame", 'self._add(("ck", "T"), P.ck_for_t)', 'self._add(("ck", "T"), P.ck_for_o)'),
    ("N13 the collector stores the CLEAN trajectory as the E1 arm", 'self._o(E1_NAME, idx, r["traj60"])', 'self._o(E1_NAME, idx, P.out["traj60"])'),
    ("N14 the collector stores a single swap as the E3 joint", "traj60, _ = e3_joint(m, P, Dn)", 'traj60, _ = LS.cell(m, P, Dn, "O", "ego", ages=ages_d)'),
]


def run(module_path: Path):
    env = dict(os.environ, LISTEN2_PY=str(module_path), LISTEN_CODE=str(HERE / "code"), CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1")
    cp = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={HERE / 'pt_mut2'}", "test_listen2.py"],
                        cwd=HERE, env=env, capture_output=True, text=True)
    out = cp.stdout + cp.stderr
    failed = sorted(set(re.findall(r"^FAILED test_listen2\.py::(\S+)", out, flags=re.M)))
    last = [ln for ln in out.strip().splitlines() if ln.strip()][-1] if out.strip() else ""
    return cp.returncode, failed, last


def main():
    MUT_DIR.mkdir(exist_ok=True)
    lines, ok = [], True
    rc, failed, last = run(HERE / "listen2.py")
    lines.append(f"UNMUTATED: rc={rc} failed={failed} | {last}")
    ok &= rc == 0
    for name, old, new in MUTATIONS:
        n = SRC.count(old)
        if n != 1:
            lines.append(f"{name}: SKIPPED, anchor text occurs {n} times (must be exactly 1)")
            ok = False
            continue
        p = MUT_DIR / "listen2.py"
        p.write_text(SRC.replace(old, new), encoding="utf-8")
        rc, failed, last = run(p)
        red = rc != 0 and len(failed) > 0
        ok &= red
        lines.append(f"{name}: {'RED (caught)' if red else 'GREEN (NOT CAUGHT)'} rc={rc} failed={failed}")
    shutil.rmtree(MUT_DIR, ignore_errors=True)
    shutil.rmtree(HERE / "pt_mut2", ignore_errors=True)
    lines.append("ALL MUTATIONS CAUGHT AND THE UNMUTATED MODULE IS GREEN" if ok else "MUTATION CHECK FAILED")
    (HERE / "mutation_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
