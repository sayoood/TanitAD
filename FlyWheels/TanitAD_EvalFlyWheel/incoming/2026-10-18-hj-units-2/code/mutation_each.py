"""HJ-UNITS-2 mutation record: restore EACH removed default, one at a time, in a TEMP COPY, and require tests to go RED.

For every mutation: build a temp tree (a copy of the edited tree's stack/tanitad, taniteval/{taniteval,tools,conftest}, the units
test file and the interface/conversion tests), apply exactly ONE default-restoring replacement, then run
  * the units test file with the PRIMARY checks selected (-k), and
  * the two pre-existing files that pinned the old defaults (test_steer_curvature_interface.py, test_steer_conversion_complete.py), FULL.
A CLEAN control copy is run first with the same selection and must be GREEN.
Writes raw/mutation_each.json and raw/mutation_each_<name>.log. Serial (1 test process at a time).
"""
import json
import os
import shutil
import subprocess
import sys
import re
from pathlib import Path

TREE = Path("C:/Users/Admin/hjunits2/tree")
WORK = Path("C:/Users/Admin/hjunits2/mut")
OUT = Path("C:/Users/Admin/hjunits2/out/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-hj-units-2/raw")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SEL = "test_a_ or test_a2_ or test_g_omitted or test_g2_unit or test_h_cli"

MUTS = {
    "cli_action_units": ("taniteval/tools/refav1_arm.py",
                         'ap.add_argument("--action-units", choices=("kappa", "steer"), default=None,',
                         'ap.add_argument("--action-units", choices=("kappa", "steer"), default="kappa",'),
    "unicycle_paths": ("stack/tanitad/refs/refa_v1_plan.py",
                       "                   action_units: str,\n                   wheelbase: float = STEER_WHEELBASE_M) -> Tensor:",
                       "                   action_units: str = \"kappa\",\n                   wheelbase: float = STEER_WHEELBASE_M) -> Tensor:"),
    "plan_model_action_units": ("stack/tanitad/refs/refa_v1.py",
                                "cost_chunk: int = 64, model_action_units: str,\n",
                                "cost_chunk: int = 64, model_action_units: str = \"kappa\",\n"),
    "imagined_goal_model_action_units": ("stack/tanitad/refs/refa_v1.py",
                                         "model_action_units: str,\n                      nav_cmd: Tensor | None = None,",
                                         "model_action_units: str = \"kappa\",\n                      nav_cmd: Tensor | None = None,"),
    "imagine_tactical_goal_units": ("stack/tanitad/refs/refa_v1.py",
                                    "                               *, units: str,\n",
                                    "                               *, units: str = \"kappa\",\n"),
    "probe_score_units": ("taniteval/tools/cost_surface_probe.py",
                          "    def score(self, controls, *, units: str, wheelbase: float = 2.9,",
                          "    def score(self, controls, *, units: str = \"kappa\", wheelbase: float = 2.9,"),
}


def build(name: str, mut):
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    ign = shutil.ignore_patterns("__pycache__")
    (root / "taniteval" / "tests").mkdir(parents=True)
    (root / "stack" / "tests").mkdir(parents=True)
    shutil.copytree(TREE / "stack" / "tanitad", root / "stack" / "tanitad", ignore=ign)
    shutil.copytree(TREE / "stack" / "scripts", root / "stack" / "scripts", ignore=ign)
    shutil.copytree(TREE / "taniteval" / "taniteval", root / "taniteval" / "taniteval", ignore=ign)
    shutil.copytree(TREE / "taniteval" / "tools", root / "taniteval" / "tools", ignore=ign)
    shutil.copy2(TREE / "taniteval" / "conftest.py", root / "taniteval" / "conftest.py")
    shutil.copy2(TREE / "taniteval" / "tests" / "test_refav1_units_required.py", root / "taniteval" / "tests")
    for f in ("test_steer_curvature_interface.py", "test_steer_conversion_complete.py"):
        shutil.copy2(TREE / "stack" / "tests" / f, root / "stack" / "tests" / f)
    if (TREE / "stack" / "pyproject.toml").exists():
        shutil.copy2(TREE / "stack" / "pyproject.toml", root / "stack" / "pyproject.toml")
    if mut:
        rel, old, new = mut
        p = root / rel
        s = p.read_text(encoding="utf-8")
        assert s.count(old) == 1, (name, rel, s.count(old))
        p.write_text(s.replace(old, new), encoding="utf-8", newline="\n")
    return root


def run(root: Path, args, cwd):
    env = dict(os.environ, PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", HJ_UNITS_NESTED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(root / "stack"), str(root / "stack" / "scripts")])
    p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf", *args], cwd=str(cwd), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2400)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def one(name, mut):
    root = build(name, mut)
    rc1, o1 = run(root, ["-k", SEL, "tests/test_refav1_units_required.py"], root / "taniteval")
    rc2, o2 = run(root, ["tests/test_steer_curvature_interface.py", "tests/test_steer_conversion_complete.py"], root / "stack")
    failed = sorted(set(re.findall(r"^FAILED (\S+)", o1 + "\n" + o2, flags=re.M)))
    last = lambda o: [l for l in o.strip().splitlines() if re.search(r"(passed|failed|error)", l)][-1:]
    rec = dict(name=name, mutation=(mut[0] if mut else None), rc_units=rc1, rc_pinned=rc2,
               summary_units=last(o1), summary_pinned=last(o2), n_red=len(failed), red_tests=failed)
    (OUT / f"mutation_each_{name}.log").write_text(f"### units (-k {SEL})\n{o1[-6000:]}\n### pinned files (full)\n{o2[-6000:]}\n", encoding="utf-8")
    shutil.rmtree(root, ignore_errors=True)
    return rec


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]
    res = []
    for name, mut in [("CLEAN_control", None)] + list(MUTS.items()):
        if only and name not in only:
            continue
        r = one(name, mut)
        res.append(r)
        print(name, "units rc", r["rc_units"], r["summary_units"], "| pinned rc", r["rc_pinned"], r["summary_pinned"], "| RED:", r["n_red"], flush=True)
    json.dump(res, open(OUT / "mutation_each.json", "w"), indent=1)
