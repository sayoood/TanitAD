"""Write raw/gate_plan_batch3.json -- the machine-readable plan for the Master Mind's Thor gate of
BATCH 3 (the refcv7 launch blockers). Same protocol as batches 1-2.

    python write_gate_plan_b3.py <base-commit> [--b2-pending]

TIP3 = <base-commit> (+ the batch-2 overlay while batch 2 is pending); FIX3 = TIP3 + batch 3.
Scope: the FULL suite (stack + taniteval) -- batch 3 touches refc.py (imported by almost every
test), the six config modules and G-DVB, which every train() calls.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

BASE = sys.argv[1]
B2_PENDING = "--b2-pending" in sys.argv
PK = pathlib.Path(__file__).resolve().parents[1]
RAW = PK / "raw"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def blob(p):
    out = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True,
                         text=True).stdout.strip()
    assert len(out) == 40, (p, out)
    return out


def tip_blob(rp):
    r = subprocess.run(GIT + ["rev-parse", "--verify", "--quiet", f"{BASE}:{rp}"],
                       capture_output=True, text=True)
    out = r.stdout.strip()
    return out if (r.returncode == 0 and len(out) == 40) else None


def overlay(layer):
    root = PK / "code" / layer
    return [{"source": f"code/{layer}/{p.relative_to(root).as_posix()}",
             "target": p.relative_to(root).as_posix(), "md5": md5(p), "blob": blob(p)}
            for p in sorted(x for x in root.rglob("*")
                            if x.is_file() and "__pycache__" not in x.parts)]


full = subprocess.run(GIT + ["rev-parse", BASE], capture_output=True, text=True).stdout.strip()
assert len(full) == 40, full
b2 = overlay("fix2") if B2_PENDING else []
b3 = overlay("fix3")
b2_blob = {o["target"]: o["blob"] for o in b2}
for o in b3:
    o["base_blob"] = b2_blob.get(o["target"]) or tip_blob(o["target"]) or "NEW"
    o["base_is"] = ("batch 2" if o["target"] in b2_blob else
                    f"base commit {BASE[:7]}" if o["base_blob"] != "NEW" else "new file")

NEW_TESTS = {
    "stack/tests/test_grad_unreachable_declared.py": {"passed": 12},
    "stack/tests/test_declared_vs_built.py": {"passed": 40, "note": "32 before batch 3 (+8)"},
    "stack/tests/test_config_hygiene.py": {"passed": 35, "note": "26 before batch 3 (+9)"},
}
plan = {
    "what": ("batch-3 gate (declared-vs-built): the refcv7 launch blockers -- (a) the three "
             "zero-gradient modules FROZEN and DECLARED + G-DVB check/probe; (b) strict_fields on "
             "the six open config classes; (c) check_refcv7_required: residual prior ha0_ext_pose "
             "(argv AND built) and --ego-history"),
    "base_commit": full,
    "tip_overlay_batch2": ({"note": "batch 2 is NOT yet landed: TIP3 = base_commit + these files; "
                                    "when it lands its commit replaces base_commit + this overlay",
                            "files": b2} if B2_PENDING else None),
    "tree_build": ("git -c core.autocrlf=false archive <base_commit> stack taniteval tools, then "
                   "tip_overlay_batch2 (TIP3), then `overlay` on top (FIX3); md5-verify every copy"),
    "env": {"PYTHONPATH": "<tree>/stack:<tree>/stack/scripts:<tree>/taniteval",
            "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "", "assert": "tanitad.__file__ and "
            "taniteval.nav_compliance.__file__ resolve inside <tree>"},
    "pytest": "python -m pytest -q -p no:cacheprovider -rfE --junitxml=<one xml per file> <file>",
    "stages": {
        "TIPRED3": {"tree": "TIP3 + ONLY the three test files below (no code change)",
                    "stack_cwd_files": [t.split("stack/", 1)[1] for t in NEW_TESTS]},
        "FIX3": {"tree": "TIP3 + EVERY file of `overlay`", "files": "the FULL suite"},
        "TIP3": {"tree": "TIP3", "files": "the FULL suite"},
    },
    "regression_rule": "a test that FAILS or ERRORS on FIX3 and PASSES on TIP3 (per junit testcase)",
    "overlay": b3,
    "expected_FIX3_new_tests": {**NEW_TESTS, "total": 87},
    "expected_TIPRED3": {
        "stack/tests/test_grad_unreachable_declared.py": {"failed": 12},
        "_note": ("PREDICTED from source: on TIP3 no module is declared and G-DVB has no "
                  "check/probe_grad_unreachable, GRAD_UNREACHABLE_RULES or REFCV7_RESIDUAL_PRIOR; "
                  "the six classes are not strict and config_hygiene has no "
                  "config_dataclass_instances -- so every batch-3 test that reads those goes RED. "
                  "Tests of unchanged mechanics inside the two EXTENDED files stay GREEN, and so "
                  "may `test_REFCV7_all_three_on_and_built_PASSES` (the old check ignores the "
                  "prior). The BEHAVIOURAL red arms run on FIX3 and must PASS there."),
    },
    "environment_bound": ("as batch 1 (raw/gate_plan.json); test_residual_prior's literal off-mode "
                          "digest is recorded on win32 / torch 2.11 and SKIPS on Thor"),
    "invariants_to_watch": [
        "test_residual_prior.py::test_c_off_is_BIT_IDENTICAL... (dev box only): the freeze adds no "
        "gradient and removes none -- the three tensors took `p.grad is None` before and after",
        "test_built_heads_receive_gradient.py: its census is per top-level child; no verdict moves",
        "test_u0_control_head_reachability.py: a sampler decoder WITHOUT F3 -- control_head stays "
        "trainable; offset_head is frozen there and its `out['offset']` control still reads 0",
    ],
}
(RAW / "gate_plan_batch3.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
print(f"gate_plan_batch3.json: base {full[:12]}{' + batch-2 overlay' if B2_PENDING else ''}; "
      f"{len(b3)} batch-3 files")
for o in b3:
    print(f"  {o['target']}  md5 {o['md5']}  base {o['base_blob'][:12]} ({o['base_is']})")
