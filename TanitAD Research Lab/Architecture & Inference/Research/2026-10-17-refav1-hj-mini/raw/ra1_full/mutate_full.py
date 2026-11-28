"""Mutation check for PREREG_RA1_FULL's new code: every mutation must turn its test RED (CLAUDE.md: mutate, do not inspect)."""
import json
import os
import shutil
import subprocess
import sys

BS, CR = chr(92), "\r\n"
C = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/"
CF = 'if not defined CACHEFULL set "CACHEFULL=D:' + BS + "tanitad" + BS + 'cache_full"' + CR
muts = [
    ("driver encode loses --only-file", C + "ra1_full_box2.py", '"--only-file", str(chunk_file),', "", "driver_dry_run"),
    ("driver writes into the 600-clip root", C + "ra1_full_box2.py", 'CACHE = TAN / "cache_full"', 'CACHE = TAN / "cache"', "driver_dry_run"),
    ("driver final verify allows partial", C + "ra1_full_box2.py", "run(verify_cmd(0, False)", "run(verify_cmd(0, True)", "driver_dry_run"),
    ("driver disk guard off", C + "ra1_full_box2.py", "if free_gb() < MIN_FREE_GB:", "if False:", "driver_dry_run"),
    ("queue full skips fullenv", C + "ra1_box2_queue.cmd", 'call :fullenv || exit /b 2' + CR + 'set "FROOT=%RUNS%_full"' + CR + "for",
     'set "FROOT=%RUNS%_full"' + CR + "for", "queue_full_stages"),
    ("queue full cache set at the top (leak)", C + "ra1_box2_queue.cmd", CF, CF + 'set "CACHE=%CACHEFULL%"' + CR, "queue_full_stages"),
    ("verdict accepts a 600-clip run", C + "ra1_full_verdict.py", "    if n_tr != N_TRAIN_CACHE:", "    if False:", "refuses_a_600"),
    ("verdict tie goes to A1", C + "ra1_full_verdict.py", 'k_arm = "A1" if ha1 < hr0 - TIE_M else "R0"', 'k_arm = "A1" if ha1 <= hr0 else "R0"',
     "refuses_a_600"),
    ("verdict no extension rule", C + "ra1_full_verdict.py", "    if ext and not a.extended:", "    if False:", "refuses_a_600"),
    ("verdict reads K without the gate", C + "ra1_full_verdict.py", "    if not gate:\n        out[\"outcome\"] = \"INSTRUMENT_BROKEN\"",
     "    if False:\n        out[\"outcome\"] = \"INSTRUMENT_BROKEN\"", "behind_the_gate"),
    ("verdict reads any arm's PASS", C + "ra1_full_verdict.py", "    kpass = bool(cand[k_arm][\"PASS\"])",
     "    kpass = any(bool(c.get(\"PASS\")) for c in cand.values())", "behind_the_gate"),
    ("queue full3 drops --wd", C + "ra1_box2_queue.cmd", 'call :run A1_b%~4g%~5_s%%S A1 %%S %~4 %~5 "--wd %~6" || exit /b 1', 'call :run A1_b%~4g%~5_s%%S A1 %%S %~4 %~5 "" || exit /b 1', "full3_stage"),
    ("verdict phase 3 keeps phase 1 K arm", C + "ra1_full_verdict.py", '        k_arm = "A1" if sum(ha) / 2 < hr0 - TIE_M else "R0"', '        pass', "phase3_repicks"),
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH="stack;taniteval")
res = []
for name, f, old, new, k in muts:
    s = open(f, "rb").read().decode("utf-8")
    assert s.count(old) == 1, (name, s.count(old))
    shutil.copy(f, f + ".bak")
    open(f, "wb").write(s.replace(old, new).encode("utf-8"))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "stack/tests/test_ra1_hjepa.py", "-k", k, "-p", "no:cacheprovider"],
                           env=env, capture_output=True, text=True, timeout=600)
    finally:
        shutil.move(f + ".bak", f)
    res.append(dict(mutation=name, test=k, red=r.returncode != 0))
    print("RED  " if r.returncode != 0 else "GREEN", name, flush=True)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "stack/tests/test_ra1_hjepa.py", "-k", "full", "-p", "no:cacheprovider"],
                   env=env, capture_output=True, text=True, timeout=600)
json.dump(dict(mutations=res, all_red=all(x["red"] for x in res), unmutated_green=r.returncode == 0), open(sys.argv[1], "w"), indent=1)
print("all red:", all(x["red"] for x in res), "| unmutated green:", r.returncode == 0)
