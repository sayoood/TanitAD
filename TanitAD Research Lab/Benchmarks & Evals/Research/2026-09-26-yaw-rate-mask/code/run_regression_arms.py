"""D-YAWMASK-1 -- build a CLEAN tree (git archive of the branch tip + this package's two
files), prove it imports from itself, then run the new test against the FIXED file and
against four deliberate regressions, and (optionally) the related suites FIXED vs TIP.

usage:
  python run_regression_arms.py --tree C:/Users/Admin/ym26/tree --out <pkg>/raw/regression_arms.json
  python run_regression_arms.py --tree ... --out <pkg>/raw/related_suites.json --related

CPU only (CUDA_VISIBLE_DEVICES=""), OMP_NUM_THREADS=4. Never touches D: except to READ the
two package files; never writes the repo. Every verdict is read from the pytest JUnit XML
(the artifact), never from an exit code alone.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

GIT_DIR = "C:/Users/Admin/tanitad-push/.git"
BRANCH = "agent/arch-inf-20260803"
PY = sys.executable
PKG = Path(__file__).resolve().parents[1]
FIX_TOOL = PKG / "code" / "fix" / "taniteval" / "tools" / "refav1_arm.py"
FIX_TEST = PKG / "code" / "fix" / "taniteval" / "tests" / "test_refav1_components_yaw_mask.py"
REL_TOOL = "taniteval/tools/refav1_arm.py"
REL_TEST = "taniteval/tests/test_refav1_components_yaw_mask.py"

FIXED_YAW_LINE = '        "LAT_yaw_rate_mae_radps": yaw,\n'
TIP_YAW_LINE = ('        "LAT_yaw_rate_mae_radps": (Pg["yaw_rate"] - Gg["yaw_rate"])'
                '.abs().mean(1).numpy(),\n')
FIXED_MASK_LINE = '    both_pair = Pg["pair_valid"] & Gg["pair_valid"]\n'

#: arm -> (how the file is made, the replacement pairs applied to the FIXED text)
MUTANTS = {
    "M0_TIP_VERBATIM": ("the branch-tip blob of refav1_arm.py, byte for byte: the real "
                        "historical defect (unmasked cell, no yaw_rate_cell stamp)", None),
    "M0_YAW_LINE_ONLY": ("the FIXED file with only the emitted yaw line reverted to the "
                         "tip's unmasked mean", [(FIXED_YAW_LINE, TIP_YAW_LINE)]),
    "M1_GT_ONLY_MASK": ("mask = gt.pair_valid only",
                        [(FIXED_MASK_LINE, '    both_pair = Gg["pair_valid"]\n')]),
    "M2_PRED_ONLY_MASK": ("mask = pred.pair_valid only",
                          [(FIXED_MASK_LINE, '    both_pair = Pg["pair_valid"]\n')]),
    "M3_SECOND_STEP_ONLY": ("mask = step validity of the pair's SECOND step only",
                            [(FIXED_MASK_LINE,
                              '    both_pair = (Pg["valid"] & Gg["valid"])[:, 1:]\n')]),
}

RELATED = [
    "taniteval/tests/test_refav1_components_yaw_mask.py",
    "stack/tests/test_refav1_arm.py",
    "stack/tests/test_refcv3_arm.py",
    "stack/tests/test_paired_openloop.py",
    "stack/tests/test_openloop_suite.py",
    "stack/tests/test_refav1_kin_contract.py",
    "stack/tests/test_refav1_openloop_report.py",
    "stack/tests/test_refcv3_ablations.py",
    "stack/tests/test_refcv3_ha0_ext_shared.py",
    "stack/tests/test_refav1_lead_block.py",
    "stack/tests/test_refav1_window_list.py",
    "stack/tests/test_render_refav1_arms.py",
    "taniteval/tests/test_bench_suite_internal_t1.py",
    "taniteval/tests/test_four_families_lateral_undefined.py",
    "taniteval/tests/test_four_families_dt.py",
    "taniteval/tests/test_ci.py",
]


def _git(*args, binary=False):
    r = subprocess.run(["git", f"--git-dir={GIT_DIR}", *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed rc={r.returncode}: "
                         f"{r.stderr.decode('utf-8', 'replace')[:400]}")
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def _blob(p: Path) -> str:
    b = _git("hash-object", "--no-filters", str(p))
    if len(b) != 40:
        raise SystemExit(f"INCONCLUSIVE: hash-object returned {b!r} for {p}")
    return b


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_tree(tree: Path) -> dict:
    tip = _git("rev-parse", BRANCH)
    if len(tip) != 40:
        raise SystemExit(f"INCONCLUSIVE: tip sha {tip!r}")
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    # `products/` carries P7-TanitEval/CRITERIA_REGISTRY.json, which the criteria
    # checker (and 7 related tests) read; without it they fail on a MISSING FILE.
    raw = _git("archive", "--format=tar", tip, "stack", "taniteval", "tools", "products",
               ":(exclude)taniteval/results", binary=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        members = [m for m in tf.getmembers()]
        tf.extractall(tree, members=members)
    n_files = sum(1 for m in members if m.isfile())
    n_excl = len([x for x in _git("ls-tree", "-r", "--name-only", tip, "--",
                                   "taniteval/results").splitlines() if x.strip()])
    # overlay the package's two files
    shutil.copyfile(FIX_TOOL, tree / REL_TOOL)
    (tree / REL_TEST).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIX_TEST, tree / REL_TEST)
    tip_tool_blob = _git("rev-parse", f"{tip}:{REL_TOOL}")
    rec = {"tip": tip, "n_files_extracted": n_files,
           "excluded": {"pathspec": ":(exclude)taniteval/results",
                        "n_files_excluded": n_excl,
                        "why": "banked result data only; keeps the tree small"},
           "tip_blob_refav1_arm": tip_tool_blob,
           "fixed_blob_refav1_arm": _blob(tree / REL_TOOL),
           "package_blob_refav1_arm": _blob(FIX_TOOL),
           "test_blob": _blob(tree / REL_TEST),
           "package_test_blob": _blob(FIX_TEST)}
    if rec["fixed_blob_refav1_arm"] != rec["package_blob_refav1_arm"]:
        raise SystemExit("overlay mismatch for refav1_arm.py")
    if rec["test_blob"] != rec["package_test_blob"]:
        raise SystemExit("overlay mismatch for the test")
    if rec["fixed_blob_refav1_arm"] == tip_tool_blob:
        raise SystemExit("the 'fixed' file IS the tip blob -- the fix was not overlaid")
    return rec


def _env(tree: Path) -> dict:
    e = dict(os.environ)
    e.update({"PYTHONPATH": f"{tree.as_posix()}/stack;{tree.as_posix()}/taniteval",
              "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
              "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
              "TANITAD_REPO": tree.as_posix()})
    return e


def import_probe(tree: Path) -> dict:
    code = ("import sys; sys.stdout.reconfigure(encoding='utf-8'); import tanitad, "
            "taniteval.four_families as ff, taniteval.ci as ci; import json; "
            "print(json.dumps({'tanitad': tanitad.__file__, 'four_families': ff.__file__, "
            "'ci': ci.__file__}))")
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=_env(tree),
                       cwd=str(tree), encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"import probe failed: {r.stderr[-800:]}")
    got = json.loads(r.stdout.strip().splitlines()[-1])
    root = str(tree.resolve()).lower().replace("\\", "/")
    ok = {k: str(Path(v).resolve()).lower().replace("\\", "/").startswith(root)
          for k, v in got.items()}
    if not all(ok.values()):
        raise SystemExit(f"IMPORT FROM THE WRONG TREE: {got}")
    return {"resolved": got, "all_under_tree": True}


def run_pytest(tree: Path, targets: list[str], tag: str, xml_dir: Path,
               timeout: int = 3600) -> dict:
    xml = xml_dir / f"junit_{tag}.xml"
    if xml.exists():
        xml.unlink()
    t0 = time.time()
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        f"--junitxml={xml}", *targets],
                       capture_output=True, text=True, env=_env(tree), cwd=str(tree),
                       encoding="utf-8", errors="replace", timeout=timeout)
    out = {"tag": tag, "rc": r.returncode, "wall_s": round(time.time() - t0, 1),
           "stdout_tail": r.stdout[-2500:]}
    if not xml.exists():
        out["verdict"] = "NO_ARTIFACT (junit xml absent -- the run did not complete)"
        return out
    root = ET.parse(xml).getroot()
    cases = {}
    for tc in root.iter("testcase"):
        name = f"{tc.get('classname')}::{tc.get('name')}"
        st = "passed"
        for ch in tc:
            if ch.tag in ("failure", "error"):
                st = ch.tag
            elif ch.tag == "skipped":
                st = "skipped"
        cases[name] = st
    cnt = {s: sum(1 for v in cases.values() if v == s)
           for s in ("passed", "failure", "error", "skipped")}
    out.update({"counts": cnt, "cases": cases,
                "not_passed": sorted(k for k, v in cases.items() if v in ("failure", "error"))})
    return out


def write_tool(tree: Path, text: str | None, raw: bytes | None = None):
    p = tree / REL_TOOL
    if raw is not None:
        p.write_bytes(raw)
    else:
        p.write_bytes(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--related", action="store_true",
                    help="run the related suites on FIXED and on TIP instead of the arms")
    ap.add_argument("--reuse-tree", action="store_true")
    a = ap.parse_args(argv)
    tree = Path(a.tree)
    xml_dir = tree.parent / "junit"
    xml_dir.mkdir(parents=True, exist_ok=True)
    rec = {"tool": "2026-09-26-yaw-rate-mask/code/run_regression_arms.py",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "python": sys.version.split()[0], "cpu_only": True}
    if a.reuse_tree and (tree / REL_TOOL).exists():
        rec["tree"] = {"reused": True, "tip": _git("rev-parse", BRANCH)}
    else:
        rec["tree"] = build_tree(tree)
    rec["import_probe"] = import_probe(tree)
    fixed_raw = FIX_TOOL.read_bytes()
    fixed_text = fixed_raw.decode("utf-8")
    tip_raw = _git("show", f"{BRANCH}:{REL_TOOL}", binary=True)

    if a.related:
        runs = {}
        write_tool(tree, None, fixed_raw)
        runs["FIXED"] = run_pytest(tree, RELATED, "related_FIXED", xml_dir)
        write_tool(tree, None, tip_raw)
        # the new test cannot pass on TIP by design; run the rest for the baseline
        runs["TIP"] = run_pytest(tree, [t for t in RELATED if "yaw_mask" not in t],
                                 "related_TIP", xml_dir)
        write_tool(tree, None, fixed_raw)
        f_np, t_np = set(runs["FIXED"].get("not_passed", [])), set(runs["TIP"].get("not_passed", []))
        rec["related"] = {
            "targets": RELATED, "runs": runs,
            "not_passed_only_with_fix": sorted(f_np - t_np),
            "not_passed_at_tip_too": sorted(f_np & t_np),
            "not_passed_only_at_tip": sorted(t_np - f_np)}
    else:
        arms = {}
        write_tool(tree, None, fixed_raw)
        arms["FIXED"] = {"what": "the package's fixed refav1_arm.py",
                         **run_pytest(tree, [REL_TEST], "arm_FIXED", xml_dir, 900)}
        for nm, (what, repl) in MUTANTS.items():
            if repl is None:
                write_tool(tree, None, tip_raw)
                applied = {"tip_blob_written": _blob(tree / REL_TOOL)}
            else:
                t = fixed_text
                applied = {}
                for old, new in repl:
                    c = t.count(old)
                    if c != 1:
                        raise SystemExit(f"{nm}: the mutation anchor occurs {c}x, not 1x -- "
                                         f"the regression would NOT be committed")
                    t = t.replace(old, new)
                    applied[old.strip()[:60]] = new.strip()[:80]
                write_tool(tree, t)
                if _blob(tree / REL_TOOL) == _blob(FIX_TOOL):
                    raise SystemExit(f"{nm}: mutant is byte-identical to the fix")
            arms[nm] = {"what": what, "applied": applied,
                        **run_pytest(tree, [REL_TEST], f"arm_{nm}", xml_dir, 900)}
        write_tool(tree, None, fixed_raw)
        if _blob(tree / REL_TOOL) != _blob(FIX_TOOL):
            raise SystemExit("tree was not restored to the FIXED file")
        green = arms["FIXED"].get("counts", {}).get("failure", 1) == 0 and \
            arms["FIXED"].get("counts", {}).get("error", 1) == 0 and \
            arms["FIXED"].get("counts", {}).get("passed", 0) > 0
        red = {nm: bool(arms[nm].get("not_passed")) for nm in MUTANTS}
        rec["arms"] = arms
        rec["verdict"] = {"FIXED_green": green, "each_mutant_red": red,
                          "PASS": bool(green and all(red.values()))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in rec if k in ("verdict",)} or
                     {"related": {k: rec["related"][k] for k in rec["related"]
                                  if k.startswith("not_passed")},
                      "FIXED_counts": rec["related"]["runs"]["FIXED"].get("counts"),
                      "TIP_counts": rec["related"]["runs"]["TIP"].get("counts")},
                     indent=1))


if __name__ == "__main__":
    main()
