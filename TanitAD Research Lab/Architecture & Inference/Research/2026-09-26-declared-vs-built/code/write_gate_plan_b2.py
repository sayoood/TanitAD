"""Write raw/gate_plan_batch2.json -- the machine-readable plan for the Master Mind's Thor gate of
BATCH 2, REBASED onto NEW-1 (Master Mind 2026-09-26: NEW-1 lands first). Same protocol as batch 1
(raw/gate_plan.json): one pytest process per file, a junit per file, FIX vs TIP regression rule.

    python write_gate_plan_b2.py <tip-commit> <NEW-1 package root>

The TIP of this gate is "tip + NEW-1's overlay" until NEW-1 lands; then it is NEW-1's landed commit,
and the base blobs are re-asserted against it (a moved blob refuses the landing, not the plan).
"""
import hashlib
import json
import pathlib
import subprocess
import sys

TIP = sys.argv[1]
N1 = pathlib.Path(sys.argv[2])
PK = pathlib.Path(__file__).resolve().parents[1]
RAW = PK / "raw"
FIX = PK / "code" / "fix2"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]
N1_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-residual-prior"
NEW_TESTS = ["tests/test_grad_reach_logged.py", "tests/test_navc_tau_file.py"]


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def blob(p):
    out = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True,
                         text=True).stdout.strip()
    assert len(out) == 40, (p, out)
    return out


def lst(name):
    return [x for x in (RAW / name).read_text(encoding="utf-8").split() if x.strip()]


full = subprocess.run(GIT + ["rev-parse", TIP], capture_output=True, text=True).stdout.strip()
assert len(full) == 40, full
n1_overlay = []
for p in sorted(x for x in (N1 / "code" / "fix").rglob("*")
                if x.is_file() and x.suffix == ".py" and "__pycache__" not in x.parts):
    rp = p.relative_to(N1 / "code" / "fix").as_posix()
    n1_overlay.append({"source": f"{N1_REL}/code/fix/{rp}", "target": rp, "md5": md5(p),
                       "blob": blob(p)})
n1_blob = {o["target"]: o["blob"] for o in n1_overlay}


def tip_blob(rp):
    """40-char blob at the tip, or None -- `rev-parse` ECHOES an unresolvable argument on stdout
    (exit 128), so the exit code and the length are both asserted, never the text alone."""
    r = subprocess.run(GIT + ["rev-parse", "--verify", "--quiet", f"{TIP}:{rp}"],
                       capture_output=True, text=True)
    out = r.stdout.strip()
    return out if (r.returncode == 0 and len(out) == 40) else None


overlay = []
for p in sorted(x for x in FIX.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
    rp = p.relative_to(FIX).as_posix()
    b = n1_blob.get(rp) or tip_blob(rp)
    overlay.append({"source": f"code/fix2/{rp}", "target": rp, "md5": md5(p), "blob": blob(p),
                    "base_blob": b or "NEW",
                    "base_is": ("NEW-1" if rp in n1_blob else
                                f"tip {TIP[:7]}" if b else "absent at tip and in NEW-1")})

tip_stack, fix_stack = lst("related_stack_files_B2N1_TIP.txt"), lst("related_stack_files_B2N1_FIX.txt")
te = lst("related_taniteval_files_B2N1.txt")
assert sorted(set(fix_stack) - set(tip_stack)) == NEW_TESTS, sorted(set(fix_stack) - set(tip_stack))

plan = {
    "what": ("batch-2 gate (declared-vs-built), REBASED onto NEW-1: D3 (ga_* rows reach "
             "metrics.jsonl; config.json declares grad_reach_logging; first-row refusal; "
             "check_logged_rows) + --nav-compliance-tau-file (SPEC_REFCV7 §7: verified against "
             "the float, sha256 stamped; G-HYG fields; G-DVB entry, registry 203 -> 204)"),
    "base_commit": full,
    "base_overlay_NEW1": {"note": ("the TIP of this gate = base_commit + these NEW-1 files "
                                   "(NEW-1 lands first). When it lands, its commit replaces "
                                   "base_commit + this overlay; re-assert base blobs then."),
                          "files": n1_overlay},
    "tree_build": ("git -c core.autocrlf=false archive <base_commit> stack taniteval tools, then "
                   "base_overlay_NEW1 (TIP2), then `overlay` on top (FIX2); md5-verify every copy"),
    "env": {"PYTHONPATH": "<tree>/stack:<tree>/stack/scripts:<tree>/taniteval",
            "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "", "assert": "tanitad.__file__ and "
            "taniteval.nav_compliance.__file__ resolve inside <tree>"},
    "pytest": "python -m pytest -q -p no:cacheprovider -rfE --junitxml=<one xml per file> <file>",
    "stages": {
        "TIPRED2": {"tree": "TIP2 + ONLY the two new test files (no trainer / G-DVB / refc_v3 change)",
                    "overlay": [f"stack/{t}" for t in NEW_TESTS],
                    "stack_cwd_files": NEW_TESTS, "taniteval_cwd_files": []},
        "FIX2": {"tree": "TIP2 + EVERY file of `overlay`",
                 "stack_cwd_files": fix_stack, "taniteval_cwd_files": te},
        "TIP2": {"tree": "base_commit + base_overlay_NEW1",
                 "stack_cwd_files": tip_stack, "taniteval_cwd_files": te},
    },
    "regression_rule": "a test that FAILS or ERRORS on FIX2 and PASSES on TIP2 (per junit testcase)",
    "why_this_file_set": ("the D3 first-row check runs inside EVERY train() call and the τ-file "
                          "hook sits in _pin_trainer_cfg/train()/_seam_stamp, RefCV3Config gains "
                          "two fields: every test importing refc_v3_train, declared_vs_built or "
                          "refc_v3 (132 on ab436ee + 2 refc_v3-only importers + NEW-1's "
                          "test_residual_prior = 135; taniteval 9), plus the 2 new files"),
    "overlay": overlay,
    "environment_bound_tests": "as raw/gate_plan.json (batch 1): compare to TIP2, do not read alone",
    "expected_FIX2_new_tests": {
        "stack/tests/test_grad_reach_logged.py": {"passed": 6},
        "stack/tests/test_navc_tau_file.py": {"passed": 14},
        "stack/tests/test_declared_vs_built.py": {"note": "NEW-1's file with the count 204: "
                                                          "every test as on TIP2, green"},
        "total_new": 20,
    },
    "expected_TIPRED2": {
        "stack/tests/test_grad_reach_logged.py": {"failed": 6},
        "stack/tests/test_navc_tau_file.py": {"failed": 14},
        "_note": ("PREDICTED from source: STRUCTURAL red on TIP2 -- `_logged_after`, "
                  "`_grad_reach_declared`, `check_logged_rows`, `_verify_navc_tau_file`, the "
                  "--nav-compliance-tau-file flag and the two RefCV3Config fields are absent. "
                  "The BEHAVIOURAL red arms run on FIX2 and must PASS there: the historical ga_* "
                  "condition verbatim (zero keys on cadence rows; refusal at the first log row), "
                  "a mismatched float refused in the REAL train() before config.json, a missing "
                  "file, and G-DVB naming a banked τ that differs from the built one."),
    },
    "not_in_this_gate": ("the two resnet34 real-checkpoint FIX-5 tests (SKIP on Thor): they run on "
                         "the dev box -> raw/gate_r34_ab436ee/"),
}
(RAW / "gate_plan_batch2.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
print(f"gate_plan_batch2.json: base {full[:12]} + NEW-1 ({len(n1_overlay)} files), "
      f"{len(overlay)} batch-2 files, FIX2 {len(fix_stack)}+{len(te)}, TIP2 {len(tip_stack)}+{len(te)}")
for o in overlay:
    print(f"  {o['target']}  md5 {o['md5']}  base {o['base_blob'][:12]} ({o['base_is']})")
