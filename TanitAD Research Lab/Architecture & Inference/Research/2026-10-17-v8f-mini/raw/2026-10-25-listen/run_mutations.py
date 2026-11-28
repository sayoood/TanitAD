"""Source-level mutation check for listen.py: break ONE override in a COPY of the module, run test_listen.py against the copy, and require
that at least one test goes RED; the unmutated module must be GREEN. The real listen.py is never edited.

Run:  python run_mutations.py   (CPU only; ~1 min).  Writes mutation_log.txt next to this file."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "listen.py").read_text(encoding="utf-8")
MUT_DIR = HERE / "mut"

#: (name, exact old text (must occur exactly once), new text)
MUTATIONS = [
    ("M1 nav donor value left unused (override never produces the token)",
     "return {_PRODUCER[x]: donor_token(model, Dn, layer, x) for x in ov if x in _PRODUCER}",
     "return {_PRODUCER[x]: donor_token(model, Dn, layer, x) for x in ov if x in _PRODUCER and x != 'nav'}"),
    ("M2 O x ego swaps p10f in the batch (the prior moves with the ego token)",
     "    if \"age\" in ov:\n        if ages is None:",
     "    if \"ego\" in ov:\n        b = dict(b)\n        b['p10f'] = Dn.b['p10f']\n        ov = tuple(x for x in ov if x != 'ego')\n    if \"age\" in ov:\n        if ages is None:"),
    ("M3 the age cell reads the window's OWN cycle position (donor ages = own)",
     "ad.append(donor_cycle_ages(ds, wins, j, ages_mode))",
     "ad.append(donor_cycle_ages(ds, wins, i, ages_mode))"),
    ("M4 O x code reads the window's own code",
     "    if \"code\" in ov:\n        cd = Dn.code",
     "    if \"code\" in ov:\n        cd = P.code"),
    ("M5 scene override keeps the tokens but zeroes the pad mask",
     "        return model._scene(Dn.adapt, b[\"frame_ok\"], i)",
     "        _tok, _pad = model._scene(Dn.adapt, b[\"frame_ok\"], i)\n        return _tok, torch.zeros_like(_pad)"),
    ("M6 T x ck reads the donor checkpoints in the NOW frame instead of T's frame",
     "    if \"ck\" in ov:\n        ck = Dn.ck_for_t",
     "    if \"ck\" in ov:\n        ck = Dn.ck_for_o"),
    ("M7 the chain re-expresses O's checkpoints in T's frame (identity must break)",
     "ck_for_o = M.transform_ckpt(ck_s, P.b[\"pose_s\"], P.b[\"pose_now\"])\n    traj60",
     "ck_for_o = M.transform_ckpt(ck_s, P.b[\"pose_s\"], P.b[\"pose_t\"])\n    traj60"),
    ("M8 donor token read at the NOW frame instead of the donor's own layer tick",
     "    i, b = _tick(Dn, layer), Dn.b\n",
     "    i, b = Dn.last_now, Dn.b\n"),
    ("M9 donor assignment is not a derangement over episodes (the window is its own donor)",
     "    return donor_indices(N, episode_ids=torch.as_tensor(np.asarray(eid)), seed=int(seed)).numpy()",
     "    return np.arange(N)"),
]


def run(module_path: Path):
    env = dict(os.environ, LISTEN_PY=str(module_path), LISTEN_CODE=str(HERE / "code"), CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1")
    cp = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={HERE / 'pt_mut'}", "test_listen.py"],
                        cwd=HERE, env=env, capture_output=True, text=True)
    out = cp.stdout + cp.stderr
    failed = sorted(set(re.findall(r"^FAILED test_listen\.py::(\S+)", out, flags=re.M)))
    last = [ln for ln in out.strip().splitlines() if ln.strip()][-1] if out.strip() else ""
    return cp.returncode, failed, last


def main():
    MUT_DIR.mkdir(exist_ok=True)
    lines, ok = [], True
    rc, failed, last = run(HERE / "listen.py")
    lines.append(f"UNMUTATED: rc={rc} failed={failed} | {last}")
    ok &= rc == 0
    for name, old, new in MUTATIONS:
        n = SRC.count(old)
        if n != 1:
            lines.append(f"{name}: SKIPPED, anchor text occurs {n} times (must be exactly 1)")
            ok = False
            continue
        p = MUT_DIR / "listen.py"
        p.write_text(SRC.replace(old, new), encoding="utf-8")
        rc, failed, last = run(p)
        red = rc != 0 and len(failed) > 0
        ok &= red
        lines.append(f"{name}: {'RED (caught)' if red else 'GREEN (NOT CAUGHT)'} rc={rc} failed={failed}")
    shutil.rmtree(MUT_DIR, ignore_errors=True)
    shutil.rmtree(HERE / "pt_mut", ignore_errors=True)
    lines.append("ALL MUTATIONS CAUGHT AND THE UNMUTATED MODULE IS GREEN" if ok else "MUTATION CHECK FAILED")
    (HERE / "mutation_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
