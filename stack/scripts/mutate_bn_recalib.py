"""Mutation proof for `--trunk-bn-recalib` (A7, 2026-09-19).

⛔ WHY THIS EXISTS. The 2026-09-10/11 binding rule: *"MUTATE, do not inspect. Delete
the single line that emits the value and require the check to go RED."* An AST census
once read 0 suspects on BOTH the fixed and the broken trainer. So each load-bearing
line of the recalibration is deleted in a SCRATCH COPY of the stack and the test file
is re-run against it; every mutation must turn at least one named test RED, and the
unmutated control must be fully GREEN.

⛔⛔ THE WRONG-DISK TRAP IS GUARDED, NOT HOPED AWAY. On this box an editable install can
serve `tanitad` from a different checkout, and an MSYS-style PYTHONPATH can be silently
dropped -- a GREEN run on the WRONG code. For a mutation harness that failure is fatal
in the worst direction: an unmutated import makes every mutation look "uncaught". So
each scratch copy carries a `conftest.py` that ASSERTS the imported `tanitad` and
`refc_v3_train` live inside the scratch copy, and aborts the session if not.

⭐ M6/M7 (added the same day) cover the W-BOOTSTRAP window-dump fix found while
preparing A7: the dump pass ran in TRAIN mode (M6 reintroduces that) and consumed the
training RNG (M7). Their tests live in `tests/test_eval_window_dump_mode.py`.

Usage:  python stack/scripts/mutate_bn_recalib.py [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STACK = Path(__file__).resolve().parents[1]
TESTS = ("tests/test_bn_recalib.py", "tests/test_eval_window_dump_mode.py")
TRUNK = "tanitad/models/timm_trunk.py"
TRAINER = "scripts/refc_v3_train.py"

# (name, file, exact line to delete-or-replace, replacement, tests that MUST go red)
MUTATIONS = [
    ("M1_call_site_deleted -- the ORIGINAL DEFECT reintroduced: frozen BN left at "
     "the identity", TRAINER,
     '        _bnr_loader, _run_config["trunk_bn_recalib"] = _bn_recalib_start(\n'
     '            model, dl, args, device, out_dir)\n',
     '        pass  # MUTATION M1\n',
     ["test_END_TO_END_the_real_trainer_recalibrates_stamps_and_holds_the_freeze"]),
    ("M2_forward_deleted -- the pass reaches no layer", TRUNK,
     "                self.forward_features(x)\n",
     "                pass  # MUTATION M2\n",
     ["test_recalibration_moves_EVERY_layer_off_the_identity_and_keeps_it_frozen",
      "test_ANALYTIC_the_first_bn_equals_the_exact_moments_of_its_input",
      "test_END_TO_END_the_real_trainer_recalibrates_stamps_and_holds_the_freeze"]),
    ("M3_chunk_bypass_deleted -- per-image BN batches", TRUNK,
     '            self.memory_levers["chunk_ckpt"] = 0\n',
     "            pass  # MUTATION M3\n",
     ["test_chunking_is_BYPASSED_so_a_bn_batch_is_never_one_image"]),
    ("M4_rng_fork_deleted -- recalibration consumes TRAINING rng", TRAINER,
     "        self._fork = torch.random.fork_rng(devices=self.dev)\n"
     "        self._fork.__enter__()\n",
     "        import contextlib as _c; self._fork = _c.nullcontext()  # MUTATION M4\n"
     "        self._fork.__enter__()\n",
     ["test_ON_does_not_shift_the_TRAINING_rng_stream",
      "test_rng_isolation_SAME_BREATH_CONTROL_and_exact_restore"]),
    ("M5_refusal_deleted -- recalibrating an UNFROZEN BN", TRAINER,
     "        if not cfg.core.encoder.trunk_frozen_bn:\n",
     "        if False:  # MUTATION M5\n",
     ["test_REFUSED_without_frozen_bn_BEFORE_anything_is_written"]),
    ("M6_dump_eval_mode_deleted -- the W-BOOTSTRAP pass back in TRAIN mode", TRAINER,
     "                model.eval()\n                try:\n"
     "                    with _RngIsolated(device, None), torch.no_grad():\n",
     "                pass  # MUTATION M6\n                try:\n"
     "                    with _RngIsolated(device, None), torch.no_grad():\n",
     ["test_the_dump_pass_runs_in_EVAL_mode",
      "test_the_dump_rows_REPRODUCE_the_aggregate_eval_row",
      "test_the_dump_pass_does_not_touch_the_tactical_prior",
      "test_the_dump_changes_NOTHING_about_training"]),
    ("M7_dump_rng_fork_deleted -- the dump consumes TRAINING rng", TRAINER,
     "                    with _RngIsolated(device, None), torch.no_grad():\n",
     "                    with torch.no_grad():  # MUTATION M7\n",
     ["test_the_dump_changes_NOTHING_about_training"]),
]

CONFTEST = '''import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
import tanitad.models.timm_trunk as _t
import refc_v3_train as _r
for _m in (_t, _r):
    if not str(Path(_m.__file__).resolve()).startswith(str(_ROOT)):
        raise SystemExit("WRONG-DISK IMPORT: %s from %s, not the scratch copy %s"
                         % (_m.__name__, _m.__file__, _ROOT))
'''


def _copy_stack(dst: Path) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(STACK / "tanitad", dst / "tanitad", ignore=ign)
    shutil.copytree(STACK / "scripts", dst / "scripts", ignore=ign)
    (dst / "tests").mkdir()
    for t in TESTS:
        shutil.copy2(STACK / t, dst / t)
    (dst / "tests" / "conftest.py").write_text(CONFTEST, encoding="utf-8")


def _apply(dst: Path, rel: str, old: str, new: str) -> None:
    p = dst / rel
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    n = s.count(old)
    if n != 1:
        raise SystemExit("mutation anchor in %s found %d times (must be 1): %r"
                         % (rel, n, old[:80]))
    s = s.replace(old, new, 1)
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))


def _run(dst: Path) -> dict:
    pp = os.pathsep.join(x for x in (str(dst), os.environ.get("PYTHONPATH")) if x)
    env = dict(os.environ, PYTHONPATH=pp, CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", *TESTS,
                        "-p", "no:cacheprovider"], cwd=str(dst), env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    out = r.stdout + r.stderr
    if "WRONG-DISK IMPORT" in out:
        raise SystemExit("⛔ the harness imported the WRONG code -- no verdict:\n"
                         + out[-2000:])
    # ⚠️ ERROR as well as FAILED: a mutation that makes the trainer REFUSE inside the
    # module-scoped `e2e` fixture surfaces as a setup ERROR on every dependent test.
    failed = sorted(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out)))
    passed = sorted(set(re.findall(r"PASSED \S+::(\w+)", out)))
    return {"rc": r.returncode, "failed": failed, "n_failed": len(failed),
            "n_passed": len(passed), "tail": out[-600:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    results = {}
    with tempfile.TemporaryDirectory(prefix="bnr_mut_") as tmp:
        base = Path(tmp)
        ctl = base / "M0" / "stack"
        _copy_stack(ctl)
        results["M0_control_unmutated"] = _run(ctl)
        for name, rel, old, new, must_red in MUTATIONS:
            d = base / name.split()[0] / "stack"
            _copy_stack(d)
            _apply(d, rel, old, new)
            r = _run(d)
            r["must_go_red"] = must_red
            r["caught"] = all(t in r["failed"] for t in must_red)
            results[name] = r
    ok = (results["M0_control_unmutated"]["n_failed"] == 0
          and results["M0_control_unmutated"]["rc"] == 0
          and all(v["caught"] for k, v in results.items() if k != "M0_control_unmutated"))
    for k, v in results.items():
        tag = ("GREEN (control)" if k.startswith("M0") and v["n_failed"] == 0 else
               "CAUGHT" if v.get("caught") else
               "⛔ NOT CAUGHT" if "caught" in v else "⛔ CONTROL NOT GREEN")
        print("%-78s %s  (%d passed / %d failed)" % (k[:78], tag, v["n_passed"],
                                                    v["n_failed"]))
        if "caught" in v and not v["caught"]:
            print("    must go red:", v["must_go_red"], "\n    went red:", v["failed"])
    print("\nMUTATION VERDICT:", "ALL CAUGHT, control green" if ok else "⛔ FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps({"ok": ok, "results": results}, indent=1),
                               encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
