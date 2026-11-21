#!/usr/bin/env python3
"""build_final.py <tip package root> <out dir> : rebuild the S4R-1 change set (SFT-4R readiness integration) from the PRISTINE branch tip 3b8c39f, byte-exact and fail-closed.

  <tip package root>  .../Research/2026-09-20-refe-plan   (a `git archive` of the tip; LF)
  <out dir>           receives refe/<7 files>, eval/PREREG_SFT4R.md   (the 8 files of the change set)

Order (every patcher asserts its anchor, so a moved tip aborts here, never half-applies):
  launch_gate.py     <- patch_launch_gate.py                                 (the refe_sft4r profile; from the 2026-10-16 readiness package, applies unchanged on the tip)
  gate_checks.py     <- patch_gate_checks.py (REBASED: scorer_dvb_rows(ns, data, profile) since e2d4422), patch_suite_nesting.py (G-SUITE refuses to nest; adds test_suite_nesting.py to SFT4R_SUITE)
  sft4r_chain.sh, sft4r_pod.sh, test_sft4r_gate.py <- post_p0/ (the previous agent's v2 files; no patcher existed), then
                     patch_test_sft4r_gate.py, patch_sft4r_chain_precheck.py            (2026-10-16)
                     patch_sft4r_chain_tokenenv.py                                      (S4R-1: the token-file path is a REQUIRED env var; no default path in source)
                     patch_test_sft4r_gate_hermetic.py                                  (S4R-1: section 6 hermetic about the nesting depth; the pod script under a real POSIX bash)
  test_suite_nesting.py <- post/ (new file, 2026-10-16)
  test_sft4r_chain.py   <- tip + patch_test_sft4r_chain_tokenenv.py                    (S4R-1: pins the required env var, fail closed, with mutation arms)
  test_launch_gate.py   <- items/test_launch_gate.py                                    (SO-A16: bash by absolute path)
  eval/PREREG_SFT4R.md  <- tip + amendment3_append.md                                   (APPEND ONLY: the original bytes stay an exact prefix)
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
pk, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    shutil.rmtree(out)
(out / "refe").mkdir(parents=True)
(out / "eval").mkdir(parents=True)
w = out / "refe"
for f in ("launch_gate.py", "gate_checks.py", "sft4r_chain.sh", "sft4r_pod.sh", "test_sft4r_chain.py"):
    assert b"\r\n" not in (pk / "refe" / f).read_bytes(), f
    shutil.copyfile(pk / "refe" / f, w / f)


def run(script, target):
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "patch" / script), str(target)], capture_output=True, text=True)
    print(f"  {script}: rc {r.returncode} {r.stdout.strip()[-80:]} {r.stderr.strip()[-300:]}")
    if r.returncode:
        sys.exit(4)


run("patch_launch_gate.py", w / "launch_gate.py")
run("patch_gate_checks.py", w / "gate_checks.py")
run("patch_suite_nesting.py", w)
for f in ("sft4r_chain.sh", "sft4r_pod.sh", "test_sft4r_gate.py"):
    shutil.copyfile(HERE / "post_p0" / "refe" / f, w / f)
run("patch_test_sft4r_gate.py", w)
run("patch_sft4r_chain_precheck.py", w)
run("patch_sft4r_chain_tokenenv.py", w)
run("patch_test_sft4r_gate_hermetic.py", w)
shutil.copyfile(HERE / "post" / "refe" / "test_suite_nesting.py", w / "test_suite_nesting.py")
run("patch_test_sft4r_chain_tokenenv.py", w)
shutil.copyfile(HERE / "items" / "test_launch_gate.py", w / "test_launch_gate.py")
orig = (pk / "eval" / "PREREG_SFT4R.md").read_bytes()
app = (HERE / "amendment3_append.md").read_bytes()
assert orig.endswith(b"\n") and b"\r" not in orig and b"\r" not in app
(out / "eval" / "PREREG_SFT4R.md").write_bytes(orig + app)
assert (out / "eval" / "PREREG_SFT4R.md").read_bytes().startswith(orig)
table = {}
for p in sorted(out.rglob("*")):
    if p.is_file():
        b = p.read_bytes()
        assert b"\r" not in b, p
        table[str(p.relative_to(out)).replace("\\", "/")] = hashlib.sha256(b).hexdigest()
        print(table[str(p.relative_to(out)).replace("\\", "/")], p.relative_to(out))
json.dump(table, open(out / "SHA256_BUILD.json", "w"), indent=1)
