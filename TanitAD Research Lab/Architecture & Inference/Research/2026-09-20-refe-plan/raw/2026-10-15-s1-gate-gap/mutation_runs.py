#!/usr/bin/env python3
"""End-to-end mutation proofs for the S1 gate-gap patches.  Each mutation is applied to a COPY of the patched `refe/` (never to the repo), the relevant suite is run on the copy, and the
result is recorded: the suite MUST exit non-zero, and the failing assertion must come from the check that is supposed to catch it (`expect_in_traceback`), not from an unrelated one.

    python mutation_runs.py --pkg <patched package root (refe/ + eval raw code splits)> --work <scratch dir> --python <interpreter> --out mutation_results.json [--only name,name]

A mutation that does NOT make its suite go red, or that goes red in the wrong place, is a defect of the TEST and is reported as such (`caught: false`).
The hashing mutations run `test_launch_gate.py --static --no-banked` (what the pod's G-SUITE runs): the banked generic gate is a SECOND identity oracle on the dev box, and it would
otherwise fail first.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHAIN_BINDING = '    token_ok || refuse "the launch-gate token does not verify for $ARM: scorer_finetune.py is NOT started"\n'
CHAIN_GUARDED = '  [ "$GATE_MINTED_NOW" = 1 ] || token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"'

# name -> (file, old, new, suite args, text that must appear in the failure output)
MUTATIONS = {
    # ---- the chain: the verifies
    "chain_binding_verify_deleted": ("s1aux_pod.sh", CHAIN_BINDING, "", ["test_s1aux_chain.py"], "TRAIN_NO_TOKEN_CHECK"),
    "chain_binding_verify_before_wait": ("s1aux_pod.sh", "    wait_clear\n" + CHAIN_BINDING, CHAIN_BINDING + "    wait_clear\n", ["test_s1aux_chain.py"], "TRAIN_ORDER"),
    "chain_resumed_path_verify_deleted": ("s1aux_pod.sh", CHAIN_GUARDED + "\n", "", ["test_s1aux_chain.py"], "RESUMED_PATH_UNVERIFIED"),
    "chain_redundant_verify_readded": ("s1aux_pod.sh", CHAIN_GUARDED, '  token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"', ["test_s1aux_chain.py"], "GATE_VERIFY_GUARD"),
    "chain_minted_flag_default_on": ("s1aux_pod.sh", "  GATE_MINTED_NOW=0\n  if ! stage_done", "  GATE_MINTED_NOW=1\n  if ! stage_done", ["test_s1aux_chain.py"], "GATE_BLOCK_SHAPE"),
    # ---- the fingerprint: anything that alters a digest
    "fp_relative_path_dropped": ("gate_token.py", 'lines.append(f"{rel_prefix}{fn}\\t{size}\\n")', 'lines.append(f"{fn}\\t{size}\\n")', ["test_launch_gate.py", "--static", "--no-banked"], "fingerprint_identity_static"),
    "fp_size_off_by_one": ("gate_token.py", "size = os.stat(os.path.join(dirpath, fn)).st_size", "size = os.stat(os.path.join(dirpath, fn)).st_size + 1", ["test_launch_gate.py", "--static", "--no-banked"], "fingerprint_identity_static"),
    "fp_skip_names_not_pruned": ("gate_token.py", '_DIR_SKIP_NAMES = frozenset({"_v2manifest.pt", "_v2manifest.pt.tmp", "__pycache__"})', "_DIR_SKIP_NAMES = frozenset()", ["test_launch_gate.py", "--static", "--no-banked"], "fingerprint_identity_static"),
    "fp_lines_not_sorted": ("gate_token.py", "        lines.sort()\n        return {\"kind\": \"dir\"", "        return {\"kind\": \"dir\"", ["test_launch_gate.py", "--static", "--no-banked"], "fingerprint_identity_static"),
    # the next two are also red in OLDER tests of the static suite (which fail first), so the identity block is proven on its own with --fp-identity-only
    "sha256_loses_last_byte": ("gate_token.py", "            h.update(mv[:n])\n", "            h.update(mv[:n - 1] if 1 < n < chunk else mv[:n])\n", ["test_launch_gate.py", "--fp-identity-only"], "fingerprint_identity_static"),
    "sha256_loses_last_byte__whole_static_suite_red_anywhere": ("gate_token.py", "            h.update(mv[:n])\n", "            h.update(mv[:n - 1] if 1 < n < chunk else mv[:n])\n", ["test_launch_gate.py", "--static", "--no-banked"], ""),
    "sha256_chunk_boundary_skipped": ("gate_token.py", "            h.update(mv[:n])\n", "            h.update(mv[:n] if n == chunk or n < 4096 else mv[:n - 4096])\n", ["test_launch_gate.py", "--static", "--no-banked"], "fingerprint_identity_static"),
    "data_manifest_gate_inputs_dropped": ("gate_token.py", "    for key, path in sorted((gate_inputs or {}).items()):\n        jobs.append((key, {\"path\": str(path)}, path))\n", "", ["test_launch_gate.py", "--fp-identity-only"], "fingerprint_identity_static"),
    "data_manifest_gate_inputs_dropped__whole_static_suite_red_anywhere": ("gate_token.py", "    for key, path in sorted((gate_inputs or {}).items()):\n        jobs.append((key, {\"path\": str(path)}, path))\n", "", ["test_launch_gate.py", "--static", "--no-banked"], ""),
    "data_manifest_duplicate_flag_first_wins": ("gate_token.py", "                man[key] = {**extra, **fp}\n                if timings is not None:\n                    timings[key] = round(dt, 2)\n        finally:",
                                                  "                man.setdefault(key, {**extra, **fp})\n                if timings is not None:\n                    timings[key] = round(dt, 2)\n        finally:", ["test_launch_gate.py", "--static", "--no-banked"], "fingerprint_identity_static"),
}


def junction(link: Path, target: Path) -> None:
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        os.symlink(str(target), str(link), target_is_directory=True)


def make_pkg(src: Path, dst: Path) -> None:
    shutil.copytree(src / "refe", dst / "refe", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for d in ("eval", "raw", "code", "splits"):
        junction(dst / d, (src / d).resolve())
    shutil.copy(src / "SHIP_MANIFEST.json", dst / "SHIP_MANIFEST.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    src, work = Path(a.pkg).resolve(), Path(a.work).resolve()
    want = [n for n in MUTATIONS if not a.only or n in a.only.split(",")]
    # NAVTEST of test_s1aux_chain is PKGROOT.parents[3]/FlyWheels/...: put every copy at depth 3 under `work`, with FlyWheels beside
    fly = next((p / "FlyWheels" for p in src.parents if (p / "FlyWheels").is_dir()), None)
    work.mkdir(parents=True, exist_ok=True)
    if fly is not None and not (work / "FlyWheels").exists():
        junction(work / "FlyWheels", fly)
    results = []
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name in want:
        fname, old, new, suite, expect = MUTATIONS[name]
        dst = work / "1" / "2" / "3" / name
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        make_pkg(src, dst)
        p = dst / "refe" / fname
        text = p.read_text(encoding="utf-8")
        if old not in text:
            results.append({"mutation": name, "caught": False, "error": f"the text to mutate is not in {fname} (the patch changed?)"})
            continue
        p.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
        t0 = time.time()
        r = subprocess.run([a.python, str(dst / "refe" / suite[0]), *suite[1:]], cwd=str(dst / "refe"), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=a.timeout)
        out = r.stdout + r.stderr
        tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
        in_right_place = expect in out
        passed_marker = any(ln.startswith(("ZZTEST_PASS", "ZZLG_STATIC_PASS")) for ln in out.splitlines())
        results.append({"mutation": name, "file": fname, "suite": " ".join(suite), "rc": r.returncode, "went_red": r.returncode != 0 and not passed_marker, "expect_in_output": expect,
                        "red_in_the_right_place": in_right_place, "caught": r.returncode != 0 and not passed_marker and in_right_place, "seconds": round(time.time() - t0, 1),
                        "tail": [t[:300] for t in tail]})
        print(json.dumps({k: results[-1][k] for k in ("mutation", "rc", "went_red", "red_in_the_right_place", "caught", "seconds")}), flush=True)
        shutil.rmtree(dst, ignore_errors=True)
    Path(a.out).write_text(json.dumps({"pkg": str(src), "mutations": results, "all_caught": all(x.get("caught") for x in results)}, indent=1), encoding="utf-8")
    print("all caught:", all(x.get("caught") for x in results))
    return 0 if all(x.get("caught") for x in results) else 1


if __name__ == "__main__":
    sys.exit(main())
