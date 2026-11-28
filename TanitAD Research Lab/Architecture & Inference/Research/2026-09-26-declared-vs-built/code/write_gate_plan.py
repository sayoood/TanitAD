"""Write raw/gate_plan.json -- the exact, machine-readable plan for a SECOND runner (Thor) of the
batch-1 gate: the three stages, their ordered pytest files, the overlay manifest with md5s, the
base commit, environment-bound tests, and the expected literal results of the new test files.

    python write_gate_plan.py <tip-commit>
"""
import hashlib
import json
import pathlib
import subprocess
import sys

TIP = sys.argv[1]
PK = pathlib.Path(__file__).resolve().parents[1]
RAW = PK / "raw"
FIX = PK / "code" / "fix"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def blob_at(rev_path):
    r = subprocess.run(GIT + ["rev-parse", rev_path], capture_output=True, text=True)
    out = r.stdout.strip()
    return out if (r.returncode == 0 and len(out) == 40) else None


def lst(name):
    return [x for x in (RAW / name).read_text(encoding="utf-8").split() if x.strip()]


overlay = []
for p in sorted(x for x in FIX.rglob("*") if x.is_file()):
    rp = p.relative_to(FIX).as_posix()
    overlay.append({"source": f"code/fix/{rp}", "target": rp, "md5": md5(p),
                    "base_blob_at_tip": blob_at(f"{TIP}:{rp}") or "NEW"})
new_modules = ["stack/tanitad/train/config_hygiene.py", "stack/tanitad/train/declared_vs_built.py"]
new_tests_stack = lst("new_stack_files.txt")
new_tests_te = lst("tipred_taniteval_files.txt")
tipred_overlay = [o for o in overlay if o["target"] in new_modules
                  or o["target"] in [f"stack/{t}" for t in new_tests_stack]
                  or o["target"] in [f"taniteval/{t}" for t in new_tests_te]]

plan = {
    "what": "batch-1 gate (declared-vs-built): three stages, each ONE pytest process per file",
    "base_commit": TIP,
    "base_note": ("git archive <base_commit> stack taniteval tools; the six shared files' base blobs "
                  "are identical at 59f0d46, 5de9363, da8400b and this commit (40-char blob "
                  "comparison); since da8400b only steering docs and taniteval/tools/training_watch "
                  "changed, and no test in these stages imports training_watch"),
    "env": {"PYTHONPATH": "<tree>/stack:<tree>/stack/scripts:<tree>/taniteval",
            "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "", "assert": "tanitad.__file__ and "
            "taniteval.nav_compliance.__file__ resolve inside <tree>"},
    "pytest": "python -m pytest -q -p no:cacheprovider -rfE --junitxml=<one xml per file> <file>",
    "stages": {
        "TIPRED": {"tree": "base_commit + ONLY the tipred_overlay files (the two new modules and the "
                           "new tests; NO fix to a shared file) -- which new tests go RED on the "
                           "unfixed code",
                   "overlay": [o["target"] for o in tipred_overlay],
                   "stack_cwd_files": new_tests_stack, "taniteval_cwd_files": new_tests_te},
        "FIX": {"tree": "base_commit + EVERY file of `overlay`",
                "stack_cwd_files": lst("related_stack_files_FIX.txt"),
                "taniteval_cwd_files": lst("related_taniteval_files_FIX.txt")},
        "TIP": {"tree": "base_commit, unmodified",
                "stack_cwd_files": lst("related_stack_files_TIP.txt"),
                "taniteval_cwd_files": lst("related_taniteval_files_TIP.txt")},
    },
    "regression_rule": "a test that FAILS or ERRORS on FIX and PASSES on TIP (per junit testcase)",
    "overlay": overlay,
    "environment_bound_tests": {
        "_reads": "these reference dev-box paths or Windows specifics; a Thor-only failure in them "
                  "is ENVIRONMENT when it fails identically on TIP -- compare, do not read alone",
        "stack/tests/test_anchor_prefilter.py": "sys.platform == 'win32' branch",
        "stack/tests/test_bev_lift.py": "D:/Projects/TanitAD-artifacts/... (bev-lidar-gt, hf-corpus-aug)",
        "stack/tests/test_cot_negative_policy.py": "D:/Projects/TanitAD/... path",
        "stack/tests/test_eval_clips_refused_in_train.py": "D:/Projects/TanitAD-artifacts/v2ep-eval139-*",
        "stack/tests/test_frame_416x1024.py": "D:/Projects/TanitAD-artifacts/v2ep-eval139-*",
        "stack/tests/test_refcv6_perception_supervision_fixes.py": "a D:/x/a40-rescue/... literal",
        "stack/tests/test_guard_mutation_audit.py": "cp1252 console handling",
        "stack/tests/test_p4_p13_p14_wiring.py": "cp1252 console handling",
        "stack/tests/test_refc_v3_lan_preflight.py": "cp1252 console handling",
        "stack/tests/test_tac_goal_trainer_flag.py": "cp1252 console handling",
        "stack/tests/test_v6_effective_weights.py": "cp1252 console handling",
        "taniteval/tests/test_bench_suite_contract.py": "D:/x... literals",
        "taniteval/tests/test_bench_suite_internal_t1.py": "C:/Users/Admin/... and D:/x literals",
        "stack/tests/test_guard_blind_spots_fix5.py (3 tests)": (
            "need timm/resnet34.a1_in1k in the local HF cache (HF_HUB_OFFLINE=1): SKIP if absent "
            "(a skip is not a pass); FAIL if the cached checkpoint differs from the pinned "
            "per-stage fingerprints -- that is E5's real signal, not an environment failure"),
    },
    "expected_FIX_new_tests": {
        "stack/tests/test_declared_vs_built.py": {"passed": 32},
        "stack/tests/test_config_hygiene.py": {"passed": 26},
        "stack/tests/test_label_clock_guard.py": {"passed": 9},
        "stack/tests/test_guard_blind_spots_fix5.py": {
            "passed": 8, "alt_if_resnet34_not_cached": {"passed": 5, "skipped": 3}},
        "stack/tests/test_speed_ceiling_inference_only.py": {"passed": 4},
        "taniteval/tests/test_refcv3_arm_equalize_as_trained.py": {"passed": 3},
        "total": 82,
    },
    "expected_TIPRED_note": ("PREDICTED from source, to be MEASURED by the run: on the unfixed code "
                             "every test that needs a fix goes RED (FIX-3 field/refusal/equalise, the "
                             "real-train G-DVB call and refusal, the 202-entry two-way coverage, the "
                             "FIX-4 declaration, pin refusals and flags, the ack stamp, the legacy "
                             "resolver, the in-training switch, G3, FIX-5, the refcv3_arm rebuild); "
                             "tests of the new modules' own mechanics (G-HYG on a synthetic class, "
                             "G-DVB readers on hand-built models, DrivoR-T, REFCV7_REQUIRED_ON) can "
                             "stay GREEN there"),
}
(RAW / "gate_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
print(f"gate_plan.json: {len(overlay)} overlay files, TIPRED overlay {len(tipred_overlay)}, "
      f"FIX {len(plan['stages']['FIX']['stack_cwd_files'])}+{len(plan['stages']['FIX']['taniteval_cwd_files'])} files, "
      f"TIP {len(plan['stages']['TIP']['stack_cwd_files'])}+{len(plan['stages']['TIP']['taniteval_cwd_files'])} files")
