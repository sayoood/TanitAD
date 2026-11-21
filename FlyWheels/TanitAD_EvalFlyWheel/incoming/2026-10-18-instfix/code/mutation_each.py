"""INSTFIX mutation record: restore EACH removed default / delete EACH gate, one at a time, in a TEMP COPY, and require tests to go RED.

For every mutation: build a temp tree (a copy of the edited tree's stack/tanitad, taniteval/{taniteval,tools,conftest}, the units
test file, test_bench_suite_internal_t1.py, the chain_controls test, and the banked free-point `code/` + the package `code/` they read),
apply exactly ONE replacement, then run the PRIMARY checks:
  * the units test file with -k "test_j or test_k or test_l or test_m"  (HJ_UNITS_NESTED=1 so the temp-copy sub-tests skip, no recursion)
  * taniteval/tests/test_bench_suite_internal_t1.py  (FULL)
  * taniteval/tests/test_chain_controls.py            (FULL)  -- for the chain_controls mutations
A CLEAN control copy is run first with the same selection and must be GREEN.
Writes raw/mutation_each.json and raw/mutation_each_<name>.log. Serial (1 test process at a time).
usage: mutation_each.py <tree> <workdir> <outdir> [name ...]
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TREE = Path(sys.argv[1])
WORK = Path(sys.argv[2])
OUT = Path(sys.argv[3])
ONLY = sys.argv[4:]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SEL_UNITS = "test_j or test_k or test_l or test_m"
FPC = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-17-free-point-navtest/code"

# name -> (file, old, new)   (single replacement; the old text must occur exactly once)
MUTS = {
    "actdiv_default_kappa": ("taniteval/tools/actdiv_anchored.py",
                             'ap.add_argument("--action-units", default=None, choices=ACTION_UNITS_CHOICES,',
                             'ap.add_argument("--action-units", default="kappa", choices=ACTION_UNITS_CHOICES,'),
    "actdiv_gate_deleted": ("taniteval/tools/actdiv_anchored.py",
                            "        if a.action_units is None:\n            ap.error(",
                            "        if False and a.action_units is None:\n            ap.error("),
    "actdiv_apply_none_guard_deleted": ("taniteval/tools/actdiv_anchored.py",
                                        "    if units is None:                           # R46:",
                                        "    if False:                                   # R46:"),
    "refcv3_default_steer": ("taniteval/tools/refcv3_arm.py",
                             'ap.add_argument("--action-units", choices=("kappa", "steer"), default=None,',
                             'ap.add_argument("--action-units", choices=("kappa", "steer"), default="steer",'),
    "refcv3_gate_deleted": ("taniteval/tools/refcv3_arm.py",
                            "    if a.analyze_only is None and a.action_units is None:\n        ap.error(",
                            "    if False and a.analyze_only is None and a.action_units is None:\n        ap.error("),
    "openloop_default_steer": ("taniteval/tools/openloop_suite.py",
                               'roll.add_argument("--action-units", choices=("kappa", "steer"), default=None,',
                               'roll.add_argument("--action-units", choices=("kappa", "steer"), default="steer",'),
    "openloop_gate_deleted": ("taniteval/tools/openloop_suite.py",
                              "    if a.ckpt and a.action_units is None:\n        ap.error(",
                              "    if False and a.ckpt and a.action_units is None:\n        ap.error("),
    "internal_t1_flag_removed": ("taniteval/taniteval/bench/internal_t1.py",
                                 '"--device", device, "--grid", a.grid, "--action-units", "steer"]',
                                 '"--device", device, "--grid", a.grid]'),
    "panel_flag_removed": ("taniteval/tools/run_hierarchy_panel.py",
                           'rc = run(base + ["--action-units", "steer"] + extra, dry)',
                           "rc = run(base + extra, dry)"),
    "doc_flag_removed_openloop_stack_example": ("taniteval/tools/OPENLOOP_SUITE.md",
                                                "\u2026 --action-units steer --stack",
                                                "\u2026 --stack"),
    # ---- fix 2: chain_controls --------------------------------------------------------------------------------------
    "cc_no_producer_check_deleted": ("taniteval/tools/chain_controls.py",
                                     "            if any(_covers(e, ctl) for e in plan.external) or any(_covers(e, ctl) for e in plan.arriving):\n                continue\n",
                                     "            continue\n"),
    "cc_late_producer_counts": ("taniteval/tools/chain_controls.py",
                                "enumerate(plan.stages[:idx]) if any(_covers(p, ctl) for p in s.produces)]\n            if earlier:",
                                "enumerate(plan.stages) if any(_covers(p, ctl) for p in s.produces)]\n            if earlier:"),
    "cc_source_check_disabled": ("taniteval/tools/chain_controls.py",
                                 "                if verify_source:\n                    for _j, s in earlier:",
                                 "                if False:\n                    for _j, s in earlier:"),
    "cc_ledger_check_deleted": ("taniteval/tools/chain_controls.py",
                                "            elif os.path.getsize(fp) == 0:\n                out.append(",
                                "            elif False:\n                out.append("),
}
CC_ONLY = {n for n in MUTS if n.startswith("cc_")}
FIXEDDOCS = {"doc_flag_removed_openloop_stack_example"}


def build(name: str, mut):
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    ign = shutil.ignore_patterns("__pycache__")
    (root / "taniteval" / "tests").mkdir(parents=True)
    shutil.copytree(TREE / "stack" / "tanitad", root / "stack" / "tanitad", ignore=ign)
    shutil.copytree(TREE / "stack" / "scripts", root / "stack" / "scripts", ignore=ign)
    shutil.copytree(TREE / "taniteval" / "taniteval", root / "taniteval" / "taniteval", ignore=ign)
    shutil.copytree(TREE / "taniteval" / "tools", root / "taniteval" / "tools", ignore=ign)
    shutil.copy2(TREE / "taniteval" / "conftest.py", root / "taniteval" / "conftest.py")
    for f in ("test_refav1_units_required.py", "test_bench_suite_internal_t1.py", "test_chain_controls.py"):
        shutil.copy2(TREE / "taniteval" / "tests" / f, root / "taniteval" / "tests" / f)
    (root / FPC).mkdir(parents=True)
    for f in (TREE / FPC).glob("*.py"):
        shutil.copy2(f, root / FPC / f.name)
    pk = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-instfix/code"
    (root / pk).mkdir(parents=True)
    shutil.copy2(TREE / pk / "fp_chain_v2.py", root / pk / "fp_chain_v2.py")
    if (TREE / "stack" / "pyproject.toml").exists():
        shutil.copy2(TREE / "stack" / "pyproject.toml", root / "stack" / "pyproject.toml")
    if mut:
        rel, old, new = mut
        p = root / rel
        s = p.read_bytes().decode("utf-8")
        crlf = "\r\n" in s
        s = s.replace("\r\n", "\n")
        assert s.count(old) == 1, (name, rel, s.count(old))
        s = s.replace(old, new)
        p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
    return root


def run(root: Path, args, cwd):
    env = dict(os.environ, PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", HJ_UNITS_NESTED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(root / "stack"), str(root / "stack" / "scripts")])
    p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf", *args], cwd=str(cwd), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2400)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def one(name, mut):
    root = build(name, mut)
    tani = root / "taniteval"
    rc1, o1 = run(root, ["-k", SEL_UNITS, "tests/test_refav1_units_required.py"], tani)
    rc2, o2 = run(root, ["tests/test_bench_suite_internal_t1.py"], tani)
    rc3, o3 = run(root, ["tests/test_chain_controls.py"], tani)
    allout = o1 + "\n" + o2 + "\n" + o3
    failed = sorted(set(re.findall(r"^FAILED (\S+)", allout, flags=re.M)))
    last = lambda o: [l for l in o.strip().splitlines() if re.search(r"(passed|failed|error)", l)][-1:]
    rec = dict(name=name, mutation=(mut[0] if mut else None), rc_units=rc1, rc_internal_t1=rc2, rc_chain_controls=rc3,
               summary_units=last(o1), summary_internal_t1=last(o2), summary_chain_controls=last(o3),
               n_red=len(failed), red_tests=failed)
    (OUT / f"mutation_each_{name}.log").write_text(
        f"### units (-k {SEL_UNITS})\n{o1[-5000:]}\n### internal_t1 (full)\n{o2[-3000:]}\n### chain_controls (full)\n{o3[-3000:]}\n", encoding="utf-8")
    shutil.rmtree(root, ignore_errors=True)
    return rec


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for name, mut in [("CLEAN_control", None)] + list(MUTS.items()):
        if ONLY and name not in ONLY:
            continue
        r = one(name, mut)
        res.append(r)
        print(name, "| units", r["summary_units"], "| internal_t1", r["summary_internal_t1"], "| chain_controls", r["summary_chain_controls"],
              "| RED:", r["n_red"], flush=True)
    json.dump(res, open(OUT / ("mutation_each_part_%s.json" % ONLY[0] if ONLY else "mutation_each.json"), "w"), indent=1)
