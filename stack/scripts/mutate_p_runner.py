"""Mutation proof for the P-arm runner's ordering and gate (`stack/scripts/p_runner.py`).

⛔ The Master Mind's required mutation is M1: let a later arm run before `P0-REPLICATE`. If that
does not turn the "P0 is UNSKIPPABLE" test RED, the rule is decoration. M2–M5 break the gate, the
A7 chaining, the STOP-on-INVALID rule and the move-aside promise.

The scratch copy carries only the runner and its test (the runner imports nothing from the stack),
plus a conftest that aborts on a wrong-disk import.

Usage: python stack/scripts/mutate_p_runner.py [--out <json>]
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
MOD = "stack/scripts/p_runner.py"
TEST = "stack/tests/test_p_runner.py"

MUTATIONS = [
    ("M1_a_later_arm_may_run_before_P0", MOD,
     '    if arm_status(out_dir, arms[0]["name"]) != "VALID":\n        return arms[0]\n',
     "    pass  # MUTATION M1: the floor is no longer first\n",
     ["test_P0_is_UNSKIPPABLE_even_if_a_later_arm_already_ran"]),
    ("M2_gate_loosened_to_8GB_of_GPU", MOD,
     "GPU_MAX_MIB = 2500\n",
     "GPU_MAX_MIB = 8000  # MUTATION M2\n",
     ["test_gate_reads_KNOWN_values"]),
    ("M3_A7_chaining_removed", MOD,
     '    if panel_procs > 0:\n        return False, f"A7 panel still running ({panel_procs} process(es))"\n',
     "    pass  # MUTATION M3: a running A7 no longer blocks\n",
     ["test_a_running_A7_panel_BLOCKS_even_when_all_its_arms_are_VALID",
      "test_plan_WAITS_while_A7_runs_even_with_a_clear_gate"]),
    ("M4_INVALID_no_longer_stops_the_panel", MOD,
     '        if arm_status(out_dir, a["name"]) == "INVALID":   # ⛔ RULE 3\n',
     '        if False:  # MUTATION M4\n',
     ["test_plan_STOPS_on_an_INVALID_arm"]),
    ("M6_the_bound_lets_the_NEXT_arm_run", MOD,
     '    if arm_status(out_dir, stop_after) == "VALID":\n'
     '        return None, f"BOUND REACHED: {stop_after} is VALID and the authorisation ends there"\n',
     "    pass  # MUTATION M6: the authorisation no longer ends\n",
     ["test_the_bound_REFUSES_the_next_arm",
      "test_plan_is_DONE_not_RUN_once_the_bound_is_reached"]),
    ("M7_preflight_accepts_an_unauthorised_arm", MOD,
     "    elif arm != authorised:\n"
     "        bad.append(f\"⛔ arm {arm!r} is NOT the authorised arm {authorised!r} — refusing\")\n",
     "    elif False:  # MUTATION M7\n        pass\n",
     ["test_preflight_REFUSES_an_unauthorised_arm"]),
    ("M8_replicate_silently_drops_a_flag", MOD,
     '_REPLICATE_DROP = {"--out": 1, "--seed": 1}\n',
     '_REPLICATE_DROP = {"--out": 1, "--seed": 1, "--w-box3d": 1}  # MUTATION M8\n',
     # ⚠️ ONLY the verbatim test. `test_build_argv_ADDS_NOTHING` cannot see a DROP mutation:
     # its fixture carries no `--w-box3d`, so removing that flag changes nothing there. Listing
     # it would have been an expectation I could not meet — MEASURED 2026-09-20, the proof went
     # ⛔ FAILED on exactly this and the fix was my expectation, never the guard.
     ["test_build_argv_copies_the_base_VERBATIM_except_out_and_seed"]),
    ("M9_unreadable_input_passes_preflight", MOD,
     '        if not p.exists():\n            bad.append(f"{label}: MISSING {path}")\n            continue\n',
     "        if not p.exists():\n            continue  # MUTATION M9: absent reads as fine\n",
     ["test_preflight_REFUSES_a_missing_or_wrong_input"]),
    ("M5_partial_arm_is_DELETED_not_moved", MOD,
     "    dst = d.with_name(f\"{arm}.aborted-{int(time.time())}\")\n    shutil.move(str(d), str(dst))\n    return dst\n",
     "    shutil.rmtree(str(d))  # MUTATION M5\n    return d\n",
     ["test_a_directory_without_a_check_is_PARTIAL_and_is_MOVED_not_deleted"]),
]

CONFTEST = '''import sys
from pathlib import Path
_R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R / "stack"))
sys.path.insert(0, str(_R / "stack" / "scripts"))
import p_runner as _p
if not str(Path(_p.__file__).resolve()).startswith(str(_R)):
    raise SystemExit("WRONG-DISK IMPORT: %s not under %s" % (_p.__file__, _R))
'''


def _copy(dst: Path) -> None:
    (dst / "stack" / "scripts").mkdir(parents=True)
    (dst / "stack" / "tests").mkdir(parents=True)
    shutil.copy2(STACK.parent / MOD, dst / MOD)
    shutil.copy2(STACK.parent / TEST, dst / TEST)
    (dst / "stack" / "tests" / "conftest.py").write_text(CONFTEST, encoding="utf-8")


def _apply(dst: Path, rel: str, old: str, new: str) -> None:
    p = dst / rel
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    if s.count(old) != 1:
        raise SystemExit("anchor count %d in %s: %r" % (s.count(old), rel, old[:70]))
    p.write_bytes(((s.replace(old, new, 1)).replace("\n", "\r\n") if crlf
                   else s.replace(old, new, 1)).encode("utf-8"))


def _run(dst: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(dst / "stack"),
                                                        str(dst / "stack" / "scripts")]),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", TEST, "-p",
                        "no:cacheprovider"], cwd=str(dst), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    if "WRONG-DISK IMPORT" in out:
        raise SystemExit("⛔ wrong-disk import -- no verdict:\n" + out[-1200:])
    failed = sorted(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out)))
    return {"rc": r.returncode, "failed": failed, "n_failed": len(failed),
            "n_passed": len(set(re.findall(r"PASSED \S+::(\w+)", out))), "tail": out[-400:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    with tempfile.TemporaryDirectory(prefix="pr_mut_") as tmp:
        c = Path(tmp) / "M0"
        _copy(c)
        res["M0_control"] = _run(c)
        for name, rel, old, new, red in MUTATIONS:
            d = Path(tmp) / name
            _copy(d)
            _apply(d, rel, old, new)
            r = _run(d)
            r["must_go_red"], r["caught"] = red, all(t in r["failed"] for t in red)
            res[name] = r
    ok = (res["M0_control"]["rc"] == 0 and res["M0_control"]["n_failed"] == 0
          and all(v["caught"] for k, v in res.items() if k != "M0_control"))
    for k, v in res.items():
        tag = ("GREEN (control)" if k == "M0_control" and v["n_failed"] == 0 else
               "CAUGHT" if v.get("caught") else "⛔ NOT CAUGHT")
        print("%-38s %s (%d passed / %d failed)" % (k, tag, v["n_passed"], v["n_failed"]))
        if "caught" in v and not v["caught"]:
            print("    must go red:", v["must_go_red"], "\n    went red:", v["failed"])
    print("\nMUTATION VERDICT:", "ALL CAUGHT, control green" if ok else "⛔ FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps({"ok": ok, "results": res}, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
