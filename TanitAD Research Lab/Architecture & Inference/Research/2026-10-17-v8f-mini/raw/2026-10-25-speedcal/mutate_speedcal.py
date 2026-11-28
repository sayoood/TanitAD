"""Mutation check of the PREREG_V8F_SPEEDCAL instruments: every mutant must make `stack/tests/test_v8f_speedcal.py` go RED BY AN ASSERTION.

"Mutate, do not inspect" (CLAUDE.md, reachability rule 3): a test that is green on the right code says nothing until it is shown to go red on the wrong code.
For each mutant ONE edit set (every `old` text found EXACTLY once) is applied to ONE file of a private copy of the tree, the test file runs (`-x`), and the verdict is read from pytest's own summary:

  KILLED    the run failed and the first failure is an assertion (`assert ...` / AssertionError / DID NOT RAISE): the test caught the defect by testing for it
  UNATTRIBUTED  (the `...e` variants, which run the end-to-end tests ALONE) red, but not by the control the brief names
  CRASHED   the run failed but NOT on an assertion (an ImportError, a SystemExit, a KeyError ...): red for the wrong reason, it does not count
  SURVIVED  the run stayed green: the mutation is not pinned by any test (a finding, not a pass)
  BAD       the mutant does not compile, or its `old` text is not found exactly once (a bug in this script)

The unmutated tree must be green before the first mutant and again after the last (every mutated file is restored and its sha256 compared with the source).
Exit 0 only if the baseline is green, EVERY mutant is KILLED and the restored tree is green.

Usage:   python mutate_speedcal.py --work <SHORT dir> --src <repo root> [--only M01,M02] [--out <dir>] [--check] [--clean]
  --check   apply every edit (count must be 1, must compile / pass `bash -n`) and run NO test: the cheap proof that the `old` texts still match
CPU only (CUDA_VISIBLE_DEVICES=-1, 2 threads per run); ~20-35 s per mutant; refuses to run with < 2 GB free; the private copy lives under --work and is removed with --clean.
The pytest basetemp lives under --work, whose directory is created first (pytest does not create the parents of --basetemp)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PKREL = Path("TanitAD Research Lab") / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
DRV = "stack/scripts/v8f_mini_eval_speedcal.py"
RD = (PKREL / "code" / "v8f_speedcal_read.py").as_posix()
RN = (PKREL / "code" / "devbox_ops" / "run_speedcal.sh").as_posix()
TEST = "stack/tests/test_v8f_speedcal.py"
MIN_FREE = 2 * 1024 ** 3
BS = chr(92)                                   # a backslash, for the runner text that carries them


def swap(a: str, b: str, ind: str = "        "):
    """Swap two `out = "<row>"` assignments of `decide` (via a placeholder, so each `old` text is found exactly once)."""
    return [(f'{ind}out = "{a}"\n', f'{ind}out = "@@"\n'), (f'{ind}out = "{b}"\n', f'{ind}out = "{a}"\n'), (f'{ind}out = "@@"\n', f'{ind}out = "{b}"\n')]


CAL_ON_EVAL = ('        elif mode == "cal":\n            mp, msrc = fit["map"], ',
               '        elif mode == "cal":\n            _zc = np.load(os.path.join(a.eval_dir, f"speedcal_cells_{tag_of(\'identity\', a.seed, target)}.npz"))\n'
               '            mp, msrc = ceiling_map_from_cells({k: _zc[k] for k in _zc.files}, int(getattr(a, "min_fit_n", MIN_FIT_N))), ')

# ------------------------------------------------------------------------------------------------ the mutants: (id, file, [(old, new), ...], what it breaks, the test that should catch it)
MUTANTS = [
    # --- the eight the brief names
    ("M01", DRV, [('        elif mode == "cal":\n            mp, msrc = fit["map"], ', CAL_ON_EVAL[1])],
     "FIT ON EVAL INSTEAD OF TRAIN: the `cal` arm applies a map fitted on the EVAL GT (the identity eval's cells), not the training-window fit", "control_2 (the applied map is the fit json's)"),
    ("M02", DRV, [("    return np.asarray(has, bool)[:, None] & np.isfinite(y) & np.isfinite(x)\n", "    return np.isfinite(y) & np.isfinite(x)\n")],
     "DROP THE MASK: windows without a v9 row at T's tick enter the fit", "the mask / fit arrays"),
    ("M03", DRV, [('SPEED_CELLS = SA.SPEED_CELLS\n', 'SPEED_CELLS = SA.SPEED_CELLS[:4]\n')], "WRONG CELL SET: four speed cells, not the PREREG's five", "cell set pin"),
    ("M04", RD, [("    for s in SEEDS:\n        A[s] = {}\n        for m in MODES:\n            E = load_eval(eval_dir, f\"M2b-mixo_s{s}-speedcal-{m}\", need_cells=(m != \"own\"))",
                  "    for s in SEEDS:\n        A[s] = {}\n        for m in MODES:\n            E = load_eval(eval_dir, f\"M2b-mixo_s{1 - s}-speedcal-{m}\", need_cells=(m != \"own\"))"),
                 ("            E[\"block\"] = check_arm(E, s, m, shas[s], allow_smoke)\n", "            E[\"block\"] = check_arm(E, 1 - s, m, shas[1 - s], allow_smoke)\n")],
     "SWAP SEEDS in the reader: seed s reads the OTHER seed's eval dumps AND checks them as that seed (so the reader's own refusals do not fire)", "controls (applied map, KINDX own)"),
    ("M05", DRV, [("    return (a * std.astype(np.float64) + k).astype(std.dtype)\n", "    return (a * std.astype(np.float64) + k + 1e-5).astype(std.dtype)\n")],
     "IDENTITY NOT EXACT: the identity map moves every cell by 1e-5", "control_1"),
    ("M06", RD, [('        r["control_4_permuted_map_is_not_separated_better"]["PASS"] = bool(not better)\n', '        r["control_4_permuted_map_is_not_separated_better"]["PASS"] = True\n')],
     "PERM CONTROL NOT ENFORCED: control 4 is forced to PASS", "control_4 arm"),
    ("M07", RD, swap("HEAD BIAS PAYS", "MAP DOES NOT TRANSFER"), "SWAP OUTCOME ROWS: HEAD BIAS PAYS <-> MAP DOES NOT TRANSFER in `decide`", "decision table"),
    ("M08", RD, swap("MAP DOES NOT TRANSFER", "INFORMATION-LIMITED"), "CEILING ROUTING INVERTED: the two \"no\" rows swapped", "ceiling routing"),
    # --- the fit
    ("M09", DRV, [("    a = float((dx * (y - ym)).sum() / sxx)\n", "    a = float((dx * (y - ym)).sum() / (sxx + 1e-3))\n")], "the fit is not least squares (a ridge-like shrinkage of the slope)", "control_2 (lstsq to 1e-9)"),
    ("M10", DRV, [("        out[idx, c] = y[rng.permutation(idx), c]\n", "        out[idx, c] = y[idx, c]\n")], "permute_targets permutes nothing: control 4's map IS the real fit", "permutation / control_4"),
    ("M11", DRV, [("    Fp = fit_cells(x, y_perm, O[\"has\"], min_n)\n", "    Fp = fit_cells(x, y, O[\"has\"], min_n)\n")], "the perm map is fitted on the REAL targets (control 4 is not a regression)", "control_2 perm checks"),
    ("M12", DRV, [("    y_perm = permute_targets(y, F[\"mask\"], PERM_SEED)\n", "    y_perm = permute_targets(y, F[\"mask\"], PERM_SEED + 1)\n")], "the permutation seed moved off the committed one", "perm seed pin / control_2"),
    # --- the hook and the formula
    ("M13", DRV, [("                src[:, self.idx] = torch.from_numpy(cal).to(device=own.device, dtype=own.dtype)\n",
                   "                src[:, self.idx[1:]] = torch.from_numpy(cal[:, 1:]).to(device=own.device, dtype=own.dtype)\n")], "the hook recalibrates four of the five speed cells (the first is left)", "applied-map guard"),
    ("M14", DRV, [("calibrate_std_np(own[:, self.idx].cpu().numpy(), self.a, self.b, mu, sd)", "calibrate_std_np(own[:, self.idx].cpu().numpy(), self.b, self.a, mu, sd)")],
     "a and b swapped in the hook", "control_1 / control_2"),
    ("M15", DRV, [("    k = ((a - 1.0) * mu + b) / sd\n", "    k = ((a - 1.0) * mu + b)\n")], "WRONG UNITS: the shift is not divided by the cell's sd (identity stays exact)", "applied-map guard"),
    ("M16", DRV, [("    k = ((a - 1.0) * mu + b) / sd\n", "    k = b / sd\n")], "the (a - 1) * mu term dropped (identity stays exact)", "applied-map guard"),
    ("M17", DRV, [("                src = own.clone()\n                src[:, self.idx]", "                src = torch.zeros_like(own)\n                src[:, self.idx]"), ("            new = SA.replace_cells(own, src, self.idx)\n", "            new = src\n")],
     "the hook clobbers the 55 non-speed cells (zeros)", "non-speed cells untouched"),
    ("M18", DRV, [("            mp = ceiling_map_from_cells({k: zc[k] for k in zc.files}, int(getattr(a, \"min_fit_n\", MIN_FIT_N)))\n", "            mp = fit[\"map\"]\n")],
     "the CEILING map is the TRAINING fit (not fitted on the eval GT)", "control_2 (ceiling vs lstsq on eval cells)"),
    ("M19", DRV, [("        if target == \"x\" and fit.get(\"x_sha256\") != got:\n", "        if False:\n")], "the fit's checkpoint sha256 is no longer checked against the evaluated checkpoint", "wrong-fit refusal"),
    ("M20", DRV, [("    if list(a.modes) != order:\n", "    if False:\n")], "the modes may be given out of order (ceiling before identity)", "order refusal"),
    # --- the reader: constants, the table, the predicates
    ("M21", RD, [("F_BAR = 0.1207 ", "F_BAR = 0.1 ")], "F moved after the fact (0.1207 -> 0.1)", "F pin / boundary"),
    ("M22", RD, [("    met = {s: primary_met(prim[s], F) for s in SEEDS}\n", "    met = {s: bool(prim[s][\"point\"] <= -F) for s in SEEDS}\n")], "DROP THE SEPARATION TEST from PRIMARY", "primary / separation"),
    ("M23", RD, [("    met = {s: primary_met(prim[s], F) for s in SEEDS}\n", "    met = {s: bool(prim[s][\"point\"] < -F and prim[s][\"separated\"]) for s in SEEDS}\n")], "PRIMARY boundary: exactly -F no longer meets it", "boundary at exactly -F"),
    ("M24", RD, [("    elif all(negsep.values()) and not all(met.values()):\n", "    elif False:\n")], "PARTIAL BIAS row unreachable", "decision table"),
    ("M25", RD, [("    elif not any(sep.values()) and not all(cle.values()):\n", "    elif not all(sep.values()) and not all(cle.values()):\n")], "INFORMATION-LIMITED needs only one seed to be unseparated", "decision table"),
    ("M26", RD, [("    return bool(g[\"point\"] <= -F)\n", "    return bool(g[\"point\"] < -F)\n")], "CEILING boundary: exactly -F no longer meets it", "ceiling boundary"),
    ("M27", RD, [("    return bool(g[\"point\"] <= -F)\n", "    return bool(g[\"point\"] <= -F and g[\"separated\"])\n")], "CEILING requires a separated interval (the PREREG's test is on the unrounded point)", "ceiling is the point"),
    ("M28", RD, [("    return bool(g[\"point\"] <= -F)\n", "    return bool(g[\"point\"] > -F)\n")], "CEILING ROUTING INVERTED in `ceiling_le`", "ceiling routing"),
    ("M29", RD, [("    if all(met.values()) and all(sec_ok.values()):\n", "    if all(met.values()):\n")], "HEAD BIAS PAYS ignores SECONDARY", "decision table"),
    ("M30", RD, [("    elif not any(sep.values()) and all(cle.values()):\n", "    elif not any(sep.values()) and any(cle.values()):\n")], "MAP DOES NOT TRANSFER needs CEILING at ONE seed, not both", "ceiling routing"),
    ("M31", RD, [("    negsep = {s: primary_negative_separated(prim[s]) for s in SEEDS}\n", "    negsep = {s: bool(prim[s][\"point\"] < 0) for s in SEEDS}\n")], "PARTIAL drops its separation test", "decision table"),
    ("M32", RD, [('TIER = "T1-class open loop (own held outputs; no recorded future action)"', 'TIER = "T0-open-loop (teacher-forced)"')], "the tier stamp is the eval json's mislabelled T0 string", "tier stamp"),
    ("M33", RD, [("encoder adaptation). That needs compute beyond box 2, so a PI go", "encoder adaptation). That needs compute beyond box 2")], "the committed next step of INFORMATION-LIMITED is edited", "table text"),
    ("M34", RD, [('"MAP DOES NOT TRANSFER": ("PRIMARY not separated at either seed, AND CEILING ≤ −F at both seeds",', '"MAP DOES NOT TRANSFER": ("PRIMARY not separated at either seed, AND CEILING ≤ −F at either seed",')],
     "the MAP DOES NOT TRANSFER condition text edited (the committed table is not the PREREG's)", "table text"),
    # --- the controls and the refusals
    ("M35", RD, [("TOL_M = 1e-4 ", "TOL_M = 1.0 ")], "control 3's per-window tolerance loosened 1e-4 m -> 1 m", "control_3 arm"),
    ("M36", RD, [("TOL_LSTSQ = 1e-9 ", "TOL_LSTSQ = 1e-2 ")], "control 2's lstsq tolerance loosened 1e-9 -> 1e-2", "control_2 arm"),
    ("M37", RD, [('c1["PASS"] = bool(d1 == 0.0 and', 'c1["PASS"] = bool(d1 <= 1e-3 and')], "control 1 is no longer EXACTLY 0.0 per window (1e-3 m allowed)", "control_1 exactness"),
    ("M38", RD, [(' and c1["map_is_identity"] and c1["cells_unchanged"])', ' and c1["cells_unchanged"])')], "control 1 no longer requires a = 1 and b = 0", "control_1 arm"),
    ("M39", RD, [('out["fit_clips_disjoint_from_eval_clips"] = bool(clips is not None and len(clips) > 0 and not (set(clips) & set(eval_clips)))\n', 'out["fit_clips_disjoint_from_eval_clips"] = True\n')],
     "the fit clips need not be disjoint from the eval clips", "disjointness"),
    ("M40", RD, [(' and out.get("fit_clips_disjoint_from_eval_clips", True))\n', ')\n')], "disjointness no longer gates control 2", "disjointness"),
    ("M41", RD, [('c2["cal_eval_applied_the_fit_json_map"] = bool(', 'c2["cal_eval_applied_the_fit_json_map"] = True or bool(')], "the cal arm's applied map is not compared with the fit json", "applied-map equality"),
    ("M42", RD, [(' and r["control_3_gt_equals_kindx_speed_oracle"]["PASS"] and r["control_4_permuted_map_is_not_separated_better"]["PASS"]', ' and r["control_3_gt_equals_kindx_speed_oracle"]["PASS"]')],
     "control 4 no longer gates the seed's PASS", "control_4 arm"),
    ("M43", RD, [('r["PASS"] = bool(c1["PASS"] and c2["PASS"] and r["control_3_gt_equals_kindx_speed_oracle"]["PASS"]', 'r["PASS"] = bool(c1["PASS"] and c2["PASS"]')], "control 3 no longer gates the seed's PASS", "control_3 arm"),
    ("M44", RD, [('r["control_3_gt_equals_kindx_speed_oracle"]["PASS"] = bool(d3 <= tol and gtdev == 0.0)\n', 'r["control_3_gt_equals_kindx_speed_oracle"]["PASS"] = bool(d3 <= tol)\n')],
     "control 3 no longer requires the driver's code to equal the eval's oracle code exactly", "control_3 arm"),
    ("M45", RD, [("        better = bool(pp[\"point\"] < 0 and pp[\"separated\"])\n", "        better = bool(pp[\"separated\"])\n")], "control 4 reads 'separated' as 'separated-better' (a worse perm map fails it)", "control_4 arm"),
    ("M46", RD, [('    if b.get("limit") and not allow_smoke:\n', "    if False:\n")], "a LIMITED (smoke) eval is no longer refused", "limited-run refusal"),
    ("M47", RD, [('    if b.get("x_sha256") != sha:\n', "    if False:\n")], "the eval's checkpoint sha256 check removed", "wrong-sha refusal"),
    ("M48", RD, [('    if (b.get("mode"), b.get("seed"), b.get("tag"), j.get("tag"), b.get("target")) != (mode, seed, tag, tag, target):\n', "    if False:\n")], "the mode / seed / tag / target check removed", "mislabelled-mode refusal"),
    ("M49", RD, [('    if fj.get("limit") and not allow_smoke:\n', "    if False:\n")], "a LIMITED (smoke) FIT is no longer refused", "limited-fit refusal"),
    ("M50", RD, [('    if fj.get("x_sha256") != sha:\n', "    if False:\n")], "the fit's checkpoint sha256 check removed", "wrong-fit-sha refusal"),
    ("M51", RD, [("        Fit[s] = load_fit(fit_dir, s, shas[s], allow_smoke)\n", "        Fit[s] = load_fit(fit_dir, 1 - s, shas[1 - s], allow_smoke)\n")], "SWAP SEEDS: seed s is checked against the OTHER seed's fit", "control_2 / refusal"),
    # --- the runner
    ("M52", RN, [('stop() { say "ZZSPEEDCAL-STOP $*";', 'stop() { say "ZZSPEEDCAL-HALT $*";')], "the STOP marker renamed", "runner markers"),
    ("M53", RN, [("MIN_FREE_KB=4194304", "MIN_FREE_KB=0")], "the free-space guard is off", "runner load-bearing lines"),
    ("M54", RN, [("c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d", "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250e")], "X_s0's committed sha256 edited", "runner load-bearing lines"),
    ("M55", RN, [(' || stop "X_s0 and X_s1 are the same file: no second seed"', " || true")], "the runner accepts X_s0 == X_s1 (no second seed)", "runner load-bearing lines"),
    ("M56", RN, [('ck_of()  { if [ "$1" = 0 ]; then echo "$X0"; else echo "$X1"; fi; }', 'ck_of()  { if [ "$1" = 0 ]; then echo "$X1"; else echo "$X0"; fi; }')], "SWAP SEEDS in the runner: seed 0 runs on X_s1", "runner load-bearing lines"),
    ("M57", RN, [('[ "$DRY" = 1 ] || bank_files > /dev/null 2>&1; exit 3; }', 'exit 3; }')], "a STOP no longer banks what exists", "runner banks on stop"),
    ("M58", RN, [('--b "$pb.npz" --b-json "$pb.json" --out', '--b "$pb.npz" --out')], "the substitute control reads the banked npz WITHOUT its banked json", "runner load-bearing lines"),
    ("M59", RN, [("  all)     stage_check; stage_fit 0; stage_fit 1; stage_evals 0;", "  all)     stage_check; stage_evals 0; stage_fit 0; stage_fit 1;")], "the stages run in the wrong order (evals before the fits)", "runner stage order"),
    ("M60", RN, [('--a "$OUT/eval/eval_M2b-mixo_s$s-speedcal-cal.npz" --b "$OUT/eval/eval_M2b-mixo_s$s-speedcal-own.npz"', '--a "$OUT/eval/eval_M2b-mixo_s$s-speedcal-own.npz" --b "$OUT/eval/eval_M2b-mixo_s$s-speedcal-cal.npz"')],
     "the paired tool's A and B swapped (own - cal, not cal - own)", "runner load-bearing lines"),
    ("M61", RN, [('    return ("' + BS * 4 + '?' + BS * 2 + '" + p) if os.name == "nt" and not p.startswith("' + BS * 4 + '?' + BS * 2 + '") else p\n', "    return p\n")], "the bank copies with plain paths again (fails past Windows' MAX_PATH)", "bank long path"),
]


#: the brief's attribution claims, run on the END-TO-END tests alone (`-k "e2e or arm"`: the pure-piece tests are deselected), so it is the CONTROLS that must catch the defect, not a unit test:
#: (id, base mutant, the controls / guard blocks of which at least ONE must read red in the e2e assertion message)
E2E_ONLY = [
    ("M01e", "M01", ["control_2_fitted_map_is_lstsq"]),                                      # a fit on the EVAL GT: control 2 (the applied map is not the training-window fit json's)
    ("M05e", "M05", ["control_1_identity_map_is_identity"]),                                 # the identity moves the cells
    ("M09e", "M09", ["control_2_fitted_map_is_lstsq"]),                                      # a shrunken slope is not lstsq
    ("M13e", "M13", ["guards", "control_2_fitted_map_is_lstsq"]),                            # one cell not recalibrated
    ("M14e", "M14", ["control_1_identity_map_is_identity", "control_2_fitted_map_is_lstsq", "guards"]),
    ("M15e", "M15", ["guards"]),                                                              # wrong units
    ("M17e", "M17", ["control_3_gt_equals_kindx_speed_oracle", "guards"]),                   # the non-speed cells clobbered
    ("M18e", "M18", ["control_2_fitted_map_is_lstsq"]),                                      # the ceiling fitted on train
]
E2E_K = "e2e or arm"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def free_bytes(p: Path) -> int:
    q = p
    while not q.exists():
        q = q.parent
    return shutil.disk_usage(q).free


def build_tree(src: Path, work: Path):
    """A private copy of exactly what the test file needs (no raw/ banks, no caches)."""
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "raw", ".git")
    for rel in ("stack", "taniteval", "Project Steering"):
        if (src / rel).exists():
            shutil.copytree(src / rel, work / rel, ignore=ign, dirs_exist_ok=True)
    pk = src / PKREL
    for f in pk.glob("*"):
        if f.is_file():
            (work / PKREL).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, work / PKREL / f.name)
    if (pk / "code").exists():
        shutil.copytree(pk / "code", work / PKREL / "code", ignore=ign, dirs_exist_ok=True)


def red_controls(e_lines) -> list:
    """The controls / guard blocks whose `PASS` reads false in the e2e test's assertion message (it dumps the controls with `json.dumps(indent=1)`): which control caught the defect.
    Read by RELATIVE indentation (pytest re-indents the message), anchored on the `"seeds": {` line, so the seed-level `PASS` (next to `guards_PASS`) is not mistaken for a control:
    a control opens at seeds + 2 levels (`"control_1_...": {`), its own `PASS` is one level deeper, and `guards_PASS` sits at the control level."""
    cur, red, base = None, [], None
    for ln in e_lines:
        body = ln[4:] if ln.startswith("E   ") else ln[1:]
        ind = len(body) - len(body.lstrip(" "))
        if re.match(r'\s*"seeds": \{', body):
            base = ind
            continue
        if base is None:
            continue
        m = re.match(r'\s*"(control_\d_\w+|guards|cross_seed_same_pair)": \{', body)
        if m and ind == base + 2:
            cur = m.group(1)
        elif re.match(r'\s*"guards_PASS": false', body) and ind == base + 2 and "guards" not in red:
            red.append("guards")
        elif re.match(r'\s*"PASS": false', body) and ind == base + 3 and cur and cur not in red:
            red.append(cur)
    return red


def run_tests(work: Path, timeout: int, k: str | None = None) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join([str(work / "stack"), str(work / "taniteval")]))
    bt = work / "bt" / "run"
    bt.parent.mkdir(parents=True, exist_ok=True)                           # pytest creates --basetemp WITHOUT its parents
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "--tb=short", "-rf", "-p", "no:cacheprovider", f"--basetemp={bt}", TEST]
    if k:
        cmd[cmd.index(TEST):cmd.index(TEST)] = ["-k", k]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(work), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        rc, out = r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = -9, f"TIMEOUT after {timeout}s\n" + ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
    shutil.rmtree(work / "bt", ignore_errors=True)
    failed = re.findall(r"^FAILED (\S+::[^\[\s]+(?:\[.*?\])?)(?: - (.*))?$", out, flags=re.M)
    e_all = [ln for ln in out.splitlines() if ln.startswith("E ")]
    errs = [ln.strip() for ln in e_all][:3]
    m = re.search(r"(\d+) passed", out)
    first = failed[0] if failed else None
    kind = None
    if first is not None:
        msg = first[1] or ""
        kind = "assertion" if (msg.startswith("assert") or msg.startswith("AssertionError")) else "other"
        if kind == "other" and (any(e.startswith("E   assert") or e.startswith("E   AssertionError") for e in e_all) or any("DID NOT RAISE" in e for e in e_all)):
            kind = "assertion"        # (`pytest.raises` -> "DID NOT RAISE" is the test failing by its own expectation)
    elif rc != 0:
        kind = "assertion" if any(e.startswith("E   assert") or e.startswith("E   AssertionError") or "DID NOT RAISE" in e for e in e_all) else "other"   # a FAILED line the regex missed, else a collection error / timeout
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "passed": int(m.group(1)) if m else 0, "first_failed": first[0] if first else None,
            "first_failed_msg": (first[1][:300] if first and first[1] else None), "kind": kind, "E_lines": [e[:300] for e in errs], "red_controls": red_controls(e_all), "tail": out.strip().splitlines()[-4:]}


def edit_text(text: str, rel: str, edits) -> tuple[str | None, str | None]:
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            return None, f"`old` text found {n} times (must be exactly 1): {old[:90]!r}"
        text = text.replace(old, new)
    if rel.endswith(".py"):
        try:
            compile(text, rel, "exec")
        except SyntaxError as e:
            return None, f"the mutant does not compile: {e}"
    return text, None


def apply_mutant(work: Path, rel: str, edits) -> str | None:
    p = work / rel
    text, err = edit_text(p.read_bytes().decode("utf-8"), rel, edits)
    if err:
        return err
    p.write_bytes(text.encode("utf-8"))                                 # bytes: the file's line endings are kept
    return None


def bash_n(work: Path, rel: str, bash: str | None) -> str | None:
    if not rel.endswith(".sh") or not bash:
        return None
    r = subprocess.run([bash, "-n", str(work / rel)], capture_output=True, text=True, timeout=60)
    return None if r.returncode == 0 else f"bash -n: {r.stderr.strip()[:200]}"


def find_bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash")):
        if c and os.path.exists(c):
            try:
                r = subprocess.run([c, "-c", "echo ok; command -v df >/dev/null && echo df"], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.stdout.split() == ["ok", "df"]:
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="a SHORT directory for the private copy (Windows MAX_PATH)")
    ap.add_argument("--src", default=str(HERE.parents[6]), help="the repo root holding the files under test (default: derived from this file's place in the tree)")
    ap.add_argument("--only", default="", help="comma-separated mutant ids")
    ap.add_argument("--out", default=str(HERE.parent))
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--merge", action="store_true", help="re-run only --only ids and REPLACE their rows in the existing mutation_results.json (the other rows are kept; the replaced ids are listed under `reruns`)")
    ap.add_argument("--check", action="store_true", help="apply every edit in memory (count == 1, compiles / bash -n) and run NO test")
    ap.add_argument("--clean", action="store_true", help="remove the private copy at the end")
    a = ap.parse_args()
    src, work, out = Path(a.src), Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {x for x in a.only.split(",") if x}
    muts = [m for m in MUTANTS if not only or m[0] in only]
    bash = find_bash()
    if a.check:
        bad = 0
        for mid, rel, edits, what, hint in MUTANTS:
            p = src / rel
            if not p.exists():
                print(f"{mid} BAD  {rel} is not in {src}")
                bad += 1
                continue
            text, err = edit_text(p.read_bytes().decode("utf-8"), rel, edits)
            if err is None and rel.endswith(".sh") and bash:
                tmp = Path(a.work) / "chk.sh"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(text.encode("utf-8"))
                err = bash_n(Path(a.work), "chk.sh", bash)
            print(f"{mid} {'OK ' if err is None else 'BAD'} {rel.rsplit('/', 1)[-1]:28s} {what[:100]}{'' if err is None else '   <- ' + err}")
            bad += err is not None
        ids = {m[0] for m in MUTANTS}
        for eid, bid, must in E2E_ONLY:
            if bid not in ids:
                print(f"{eid} BAD  base mutant {bid} unknown")
                bad += 1
        print(f"CHECK: {len(MUTANTS)} mutants + {len(E2E_ONLY)} end-to-end-only variants; {bad} bad")
        shutil.rmtree(Path(a.work) / "chk.sh", ignore_errors=True) if (Path(a.work) / "chk.sh").is_dir() else None
        return 1 if bad else 0
    if free_bytes(work) < MIN_FREE:
        print(f"ABORT: < 2 GB free on the drive of {work}")
        return 2
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    build_tree(src, work)
    files = sorted({m[1] for m in MUTANTS})
    for rel in files + [TEST]:
        if not (work / rel).exists():
            print(f"ABORT: {rel} is not in {src}")
            return 2
    res = {"src": str(src), "tests": TEST, "n_mutants": len(muts), "baseline_before": None, "mutants": [], "baseline_after": None}
    prev = None
    if a.merge and (out / "mutation_results.json").exists():
        prev = json.loads((out / "mutation_results.json").read_text(encoding="utf-8"))
    print(f"mutation check of {len(muts)} mutants (+ the end-to-end-only variants); copy under {work}; python {sys.version.split()[0]}", flush=True)

    base = run_tests(work, a.timeout)
    res["baseline_before"] = base
    print(f"BASELINE (unmutated): rc {base['rc']}  {base['passed']} passed  {base['seconds']} s", flush=True)
    if base["rc"] != 0:
        print("ABORT: the unmutated tree is not green", base["tail"])
        (out / "mutation_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 1

    byid = {m[0]: m for m in MUTANTS}
    plan = [(m[0], m[1], m[2], m[3], m[4], None, None) for m in muts]
    plan += [(eid, byid[bid][1], byid[bid][2], byid[bid][3] + "  [END-TO-END TESTS ONLY]", "the controls", E2E_K, must) for eid, bid, must in E2E_ONLY if (not only or eid in only or bid in only)]
    res["n_mutants"] = len(plan)
    for mid, rel, edits, what, hint, kexpr, must in plan:
        if free_bytes(work) < MIN_FREE:
            print("ABORT: < 2 GB free")
            return 2
        orig = (src / rel).read_bytes()
        err = apply_mutant(work, rel, edits) or bash_n(work, rel, bash)
        row = {"id": mid, "file": rel, "breaks": what, "expected_catcher": hint}
        if err:
            row.update(verdict="BAD", detail=err)
        else:
            r = run_tests(work, a.timeout, kexpr)
            row.update(r)
            row["verdict"] = "KILLED" if (r["rc"] != 0 and r["kind"] == "assertion") else ("CRASHED" if r["rc"] != 0 else "SURVIVED")
            if must is not None:
                row["must_be_red_one_of"] = must
                row["attributed"] = bool(set(must) & set(r["red_controls"]))
                if row["verdict"] == "KILLED" and not row["attributed"]:
                    row["verdict"] = "UNATTRIBUTED"                                 # red, but not by the control the brief names
        (work / rel).write_bytes(orig)                                           # restore at once
        assert sha(work / rel) == sha(src / rel)
        res["mutants"].append(row)
        print(f"{mid} {row['verdict']:12s} {row.get('seconds', '-')!s:>6} s  {what}\n      caught by: {row.get('first_failed')}  [{row.get('first_failed_msg')}]  red controls: {row.get('red_controls')}", flush=True)
        (out / "mutation_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")   # bank incrementally

    for rel in files:
        assert sha(work / rel) == sha(src / rel), f"{rel} was not restored"
    if prev is not None:                                                         # --merge: the re-run rows replace the old rows of the same id; everything else is kept
        new_ids = {m["id"] for m in res["mutants"]}
        keep = [m for m in prev["mutants"] if m["id"] not in new_ids]
        res["mutants"] = sorted(keep + res["mutants"], key=lambda m: m["id"])
        res["reruns"] = prev.get("reruns", []) + [sorted(new_ids)]
        res["baseline_first_pass"] = prev.get("baseline_before")
        res["first_pass_summary"] = prev.get("first_pass_summary") or prev.get("summary")
        res["first_pass_script_note"] = "pass 1: the original test file and mutants M01/M04 as first written; the rows re-run below replace their pass-1 rows"
        res["n_mutants"] = len(res["mutants"])
        plan = res["mutants"]
    after = run_tests(work, a.timeout)
    res["baseline_after"] = after
    print(f"BASELINE (restored): rc {after['rc']}  {after['passed']} passed  {after['seconds']} s", flush=True)
    counts = {v: sum(1 for m in res["mutants"] if m["verdict"] == v) for v in ("KILLED", "CRASHED", "SURVIVED", "UNATTRIBUTED", "BAD")}
    res["summary"] = counts
    ok = bool(after["rc"] == 0 and counts["KILLED"] == len(plan))
    res["ALL_KILLED_AND_RESTORED_GREEN"] = ok
    (out / "mutation_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"SUMMARY {counts}  all killed and the restored tree green: {ok}", flush=True)
    if a.clean:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
