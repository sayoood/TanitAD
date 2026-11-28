#!/usr/bin/env python3
"""Mutation proof of the deployed HACC post-map (``stack/tests/test_v8f_hacc_infer.py``). CPU only, ~3 min, edits NOTHING in the tree.

  python mutate_hacc_infer.py [--tree <repo root>] [--out <dir>] [--only M01 ...]

For every mutant: apply the listed (old -> new) replacements to ``stack/tanitad/models/v8f_mini.py`` (each ``old`` must occur EXACTLY once, or the harness stops: a mutant that
did not apply proves nothing), compile it, then run the test file against it in a fresh pytest process through ``_mutplug`` (which swaps ``tanitad.models.v8f_mini`` for the mutant).
A mutant counts as RED-BY-ASSERTION iff at least one test fails and the FIRST ``E`` line of a failing test is an ``assert`` / ``AssertionError`` (a mutant that merely CRASHES the
tests with a TypeError / NameError is reported as CRASH and does not count). The unmutated source is run first through the same harness and must be GREEN with nothing skipped
(the control: it proves the harness itself, the plugin included, breaks nothing).
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
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]            # <repo>/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/<this dir>  ->  <repo>
PY = os.environ.get("TANITAD_PY", "C:/Users/Admin/venvs/tanitad/Scripts/python.exe")

# (name, what it does, [(old, new), ...]) -- the five the brief names first (M01-M05), then the rest
MUTANTS = [
    ("M01_apply_when_absent", "the post-map is applied with a default w = 0.5 when the buffer is ABSENT", [(
        '        if self.hacc_w is None:\n            return out\n        w30 = self.hacc_w.to(device=out["traj60"].device, dtype=out["traj60"].dtype)\n',
        '        w30 = (self.hacc_w if self.hacc_w is not None else torch.full((K30, 2), 0.5)).to(device=out["traj60"].device, dtype=out["traj60"].dtype)\n')]),
    ("M01b_buffer_has_a_default", "the buffer is registered with a default table instead of ABSENT", [(
        'self.register_buffer("hacc_w", None)', 'self.register_buffer("hacc_w", torch.ones(K30, 2))')]),
    ("M02a_swap_axes_in_the_map", "the blend uses the weight columns swapped (along <-> cross)", [(
        'hacc_w_midpoints(w30))', 'hacc_w_midpoints(w30.flip(1)))')]),
    ("M02b_swap_axes_in_the_loader", "the JSON loader stacks [cross, along] instead of [along, cross]", [(
        'for k in HACC_KEYS], 1)', 'for k in HACC_KEYS[::-1]], 1)')]),
    ("M03_even_rows_for_traj30", "traj30 is taken from the EVEN rows of the mapped traj60 (the midpoints), not the 0.2 s knots", [(
        'out["traj30"] = subsample_02(out["traj60"])', 'out["traj30"] = out["traj60"][:, 0::2]')]),
    ("M04a_skip_midpoint_rule_nearest", "the 60 weights are the knot weights repeated (no neighbour mean)", [(
        'hacc_w_midpoints(w30))', 'w30.repeat_interleave(2, 0))')]),
    ("M04b_skip_midpoint_rule_odd_only", "the map is applied to the odd rows only; the midpoint rows keep O's own plan", [(
        '        out["traj60"] = hacc_apply(out["traj60"], upsample_prior(out["p10"]), hacc_w_midpoints(w30))\n',
        '        _m = hacc_apply(out["traj60"], upsample_prior(out["p10"]), hacc_w_midpoints(w30))\n        _t = out["traj60"].clone()\n        _t[:, 1::2] = _m[:, 1::2]\n        out["traj60"] = _t\n')]),
    ("M04c_first_midpoint_is_half", "the first midpoint weight is the mean with an origin weight of 0 (0.5 w[0]) instead of w[0]", [(
        'mid = torch.cat([w30[:1], 0.5', 'mid = torch.cat([0.5 * w30[:1], 0.5')]),
    ("M04d_midpoints_on_the_wrong_parity", "the knot weights sit on the EVEN rows and the midpoints on the odd ones", [(
        'torch.stack([mid, w30], 1)', 'torch.stack([w30, mid], 1)')]),
    ("M05_map_in_the_training_forward", "the post-map moves into _forward (training reads the HACC plan) and out of forward_infer", [(
        '        out["adapt"] = adapt\n        return out\n', '        out["adapt"] = adapt\n        return self._hacc_post_map(out)\n'), (
        '        out["_keys_read"] = set(g.read)\n        return self._hacc_post_map(out)\n\n    def _hacc_post_map',
        '        out["_keys_read"] = set(g.read)\n        return out\n\n    def _hacc_post_map')]),
    ("M06_no_range_check", "the validator accepts entries outside [0, 1]", [(
        'if a.min() < 0.0 or a.max() > 1.0:', 'if False:')]),
    ("M07_no_shape_check", "the validator accepts any shape", [(
        'if a.shape != (K30, 2):', 'if False:')]),
    ("M08_no_state_dict_override", "the _load_from_state_dict override is gone (a model without the buffer cannot load one that has it)", [(
        'def _load_from_state_dict(self, state_dict, prefix,', 'def _load_from_state_dict_disabled(self, state_dict, prefix,')]),
    ("M08b_state_dict_not_validated", "a state dict's hacc_w is not validated on load", [(
        '            hacc_check_weights(state_dict[key])\n', '            pass\n')]),
    ("M09_inverted_blend", "the blend weight is 1 - w (the prior where O should be)", [(
        'torch.lerp(prior, plan, w.expand_as(prior))', 'torch.lerp(prior, plan, 1.0 - w.expand_as(prior))')]),
    ("M10_forward_diag_unmapped", "forward_diag returns O's raw plan while forward_infer returns the HACC plan", [(
        'return self._hacc_post_map(out)                      # the same post-map as forward_infer', 'return out                      # the same post-map as forward_infer')]),
    ("M11_traj30_not_remapped", "traj60 is mapped but traj30 keeps O's plan (the two grids disagree)", [(
        'out["traj30"] = subsample_02(out["traj60"])', 'out["traj30"] = out["traj30_o"]')]),
    ("M12_prior_is_not_p10", "the blend's P is the ZERO plan instead of the deployed prior", [(
        'hacc_apply(out["traj60"], upsample_prior(out["p10"]), hacc_w_midpoints(w30))', 'hacc_apply(out["traj60"], upsample_prior(out["p10"]) * 0.0, hacc_w_midpoints(w30))')]),
]


def apply_mutant(src: str, reps) -> str:
    for old, new in reps:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"HARNESS ERROR: a replacement matched {n} times (need exactly 1): {old[:90]!r}")
        src = src.replace(old, new)
    compile(src, "<mutant>", "exec")
    return src


def run_tests(tree: Path, mut_src: Path, junit: Path, timeout: int = 240) -> dict:
    env = dict(os.environ)
    env.update({"MUT_SRC": str(mut_src), "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join([str(HERE), str(tree / "stack"), str(tree / "taniteval")])})
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "_mutplug", "--tb=short", f"--junitxml={junit}", "-rs",
           str(tree / "stack" / "tests" / "test_v8f_hacc_infer.py")]
    pr = subprocess.run(cmd, cwd=str(tree / "stack"), env=env, capture_output=True, text=True, timeout=timeout)
    root = ET.parse(junit).getroot()
    cases = list(root.iter("testcase"))
    fails, skipped = [], 0
    for tc in cases:
        if tc.find("skipped") is not None:
            skipped += 1
        for tag in ("failure", "error"):
            el = tc.find(tag)
            if el is None:
                continue
            first_e = next((ln[1:].strip() for ln in (el.text or "").splitlines() if ln.startswith("E ")), (el.get("message") or "")[:120])
            is_assert = bool(re.match(r"^(assert\b|AssertionError)", first_e)) and tag == "failure"
            fails.append({"test": tc.get("name"), "kind": tag, "assertion": is_assert, "first_e_line": first_e[:200]})
    return {"returncode": pr.returncode, "n_tests": len(cases), "n_skipped": skipped, "n_failed": len(fails), "failures": fails, "tail": pr.stdout[-400:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=str(ROOT))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    tree, out = Path(a.tree), Path(a.out)
    src_path = tree / "stack" / "tanitad" / "models" / "v8f_mini.py"
    src = src_path.read_text(encoding="utf-8")
    work = Path(tempfile.mkdtemp(prefix="mhi"))          # a SHORT path (Windows MAX_PATH): the repo path of this directory is already 230 characters
    results = {}
    # ---- the control: the UNMUTATED source through the same harness must be green, nothing skipped
    base_py = work / "BASE.py"
    base_py.write_text(src, encoding="utf-8", newline="")
    r = run_tests(tree, base_py, work / "BASE.xml")
    results["BASE_unmutated"] = {**r, "expected": "GREEN, nothing skipped", "ok": r["returncode"] == 0 and r["n_failed"] == 0 and r["n_skipped"] == 0}
    print(f"BASE (unmutated): {r['n_tests']} tests, failed {r['n_failed']}, skipped {r['n_skipped']}  ->  {'GREEN' if results['BASE_unmutated']['ok'] else 'NOT GREEN'}", flush=True)
    if not results["BASE_unmutated"]["ok"]:
        (out / "mutation_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        print("ABORT: the control is not green", flush=True)
        return 2
    for name, what, reps in MUTANTS:
        if a.only and not any(name.startswith(o) for o in a.only):
            continue
        mp = work / f"{name}.py"
        mp.write_text(apply_mutant(src, reps), encoding="utf-8", newline="")
        r = run_tests(tree, mp, work / f"{name}.xml")
        red_assert = any(f["assertion"] for f in r["failures"])
        verdict = "RED_BY_ASSERTION" if red_assert else ("CRASH_ONLY" if r["n_failed"] else "SURVIVED")
        results[name] = {"what": what, "verdict": verdict, "n_failed": r["n_failed"], "n_tests": r["n_tests"],
                         "failing_tests": [f["test"] for f in r["failures"]], "first_assertion_lines": [f["first_e_line"] for f in r["failures"] if f["assertion"]][:3],
                         "non_assertion_failures": [f for f in r["failures"] if not f["assertion"]]}
        print(f"{name:42s} {verdict:17s} failed {r['n_failed']:2d}/{r['n_tests']}  by: {', '.join(sorted({f['test'].split('[')[0] for f in r['failures']}))[:150]}", flush=True)
    n_red = sum(1 for k, v in results.items() if k != "BASE_unmutated" and v["verdict"] == "RED_BY_ASSERTION")
    n_all = sum(1 for k in results if k != "BASE_unmutated")
    results["summary"] = {"mutants": n_all, "red_by_assertion": n_red, "all_red": n_red == n_all, "baseline_green": results["BASE_unmutated"]["ok"]}
    (out / "mutation_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"SUMMARY: baseline GREEN, {n_red} of {n_all} mutants RED by an assertion", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0 if n_red == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
