"""mutate_check_b3.py -- one source mutation per committed block-3 rule; test_ds_read_b3.py must go RED on every one (and GREEN on the unmutated copy).  Writes out/MUTATIONS_B3.json.  Usage: python mutate_check_b3.py
Before every pytest run it waits for the dev-box commit headroom (wait_commit.py 6 120).  The mutations in SMOKE_MUTS (reader wiring inside main, which unit tests do not reach) are checked with smoke_ds_read_b3.py instead (pass --smoke)."""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
WC = "F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py"
W = HERE + "/mut"
MUTS = [
    ("ds_read_b3.py", "    if delta < -2.0 * f_a:", "    if delta <= -2.0 * f_a:", "the 2F_A bar is not strict on the DATA > EPOCHS side"),
    ("ds_read_b3.py", "    if delta > 2.0 * f_a:", "    if delta >= 2.0 * f_a:", "the 2F_A bar is not strict on the EPOCHS > DATA side"),
    ("ds_read_b3.py", "    if delta < -2.0 * f_a:\n        return \"DATA > EPOCHS\"", "    if delta > 2.0 * f_a:\n        return \"DATA > EPOCHS\"", "the sign of the label is flipped"),
    ("ds_read_b3.py", "    if delta < -2.0 * f_a:", "    if delta < -1.0 * f_a:", "the bar is F_A instead of 2F_A"),
    ("ds_read_b3.py", "or f_a <= 0.0 or", "or", "a zero floor is accepted"),
    ("ds_read_b3.py", "or delta is None or not math.isfinite(delta):", "or delta is None:", "a NaN delta reads NEITHER"),
    ("ds_read_b3.py", "    elif lit != alt:", "    elif False:", "READING-SENSITIVE is never reported"),
    ("ds_read_b3.py", '    if "NOT-YET" in (lit, alt):', "    if False:", "a NOT-YET reading is dropped"),
    ("ds_read_b3.py", '    if blocked:\n        return ["BLOCKED"]', '    if False:\n        return ["BLOCKED"]', "BLOCKED no longer precedes"),
    ("ds_read_b3.py", "    if problems:\n        return", "    if False:\n        return", "a failed control leaves the label quotable"),
    ("ds_read_b3.py", "    if not ade_given or lab is None:", "    if False:", "an unread ADE6 is not NOT-YET"),
    ("ds_read_b3.py", '"U-BL.A": ("U_s0", "BL_s0"),', '"U-BL.A": ("BL_s0", "U_s0"),', "U - BL is read with the wrong sign"),
    ("ds_read_b3.py", '"UE-BL.A": ("UE_s0", "BL_s0"),', '"UE-BL.A": ("UE_s0", "B_s0"),', "UE - BL is read against B_s0"),
    ("ds_read_b3.py", "    if not k.get(\"pass\") or set(k.get(\"items\", {})) != want or not all(v[\"identical\"] for v in k[\"items\"].values()):", "    if False:", "the stage-A K-EVALX record is not checked"),
    ("ds_read_b3.py", "for v in B2.ROUTE_VARIANTS if not os.path.exists(", 'for v in ("real", "donor") if not os.path.exists(', "the zero / realx / donorx stage-A passes are not required"),
    ("ds_read_b3.py", "return {\"pass\": bool(n == 2 and not bad)", "return {\"pass\": bool(n == 2)", "K-RE (ADE6 means) ignores a mismatch"),
    ("ds_read_b3.py", "{f: R.family_status(rows, HEADLINE[f]) for f in FAMILIES}", 'XPOOL = tuple(c for g in FAMILIES for c in HEADLINE[g]); return {f: R.family_status(rows, XPOOL) for f in FAMILIES}', "the families are pooled into one set of cells"),
    ("ds_read_b3.py", "            r.update(strat[\"pairs\"][p])", "            pass", "the STRATEGIC cells are dropped from the family rows"),
    ("ds_lib.py", "    return STEPS_ARM.get(arm, STEPS[stage])", "    return STEPS[stage]", "the block-3 arms are held to 19,164 steps"),
    ("ds_lib.py", 'STEPS_ARM = {"UE_s0": 61568, "BL_s0": 61568}', 'STEPS_ARM = {"UE_s0": 61568, "BL_s0": 19164}', "BL_s0 is held to 19,164 steps"),
    ("ds_lib.py", "if (BLOCK == 3 and arm in STAGE_A_REF) else CK", "if False else CK", "the block-1 arms are looked for in this folder's checkpoints"),
    ("ds_lib.py", "DEFAULT_ITEMS = B3_ITEMS if BLOCK == 3 else", "DEFAULT_ITEMS = B3_ITEMS if BLOCK == 2 else", "the block-3 item default"),
    ("ds_run.py", 'ROUTE_STAGE = "A" if L.BLOCK == 3 else "B"', 'ROUTE_STAGE = "B"', "the route passes stay stage B"),
    ("ds_run.py", "L.ROUTE_ITEMS_B3 if (route and L.BLOCK == 3)", "L.DEFAULT_ITEMS if (route and L.BLOCK == 3)", "the route command defaults to the two block-3 arms"),
    ("sync_b1.py", "            if L.sha256_file(d) == L.sha256_file(s):", "            if True:", "sync_b1 accepts a target with different bytes"),
    ("sync_b1.py", 'for a in ("B_s0", "U_s0", "B_rep", "R_s0"))', 'for a in ("B_s0", "U_s0", "B_rep"))', "sync_b1 skips R_s0"),
    ("launch_dscale10k_b3.sh", '[ "$FREE" -lt 25 ]', '[ "$FREE" -lt 0 ]', "the launcher's F: 25 GB bar"),
    ("launch_dscale10k_b3.sh", 'tail -n +"$S" "$B2LOG"', 'cat "$B2LOG"', "the launcher reads an older DONE / FAIL as the current one"),
    ("launch_dscale10k_b3.sh", "env -u DS_BLOCK TMP", "TMP", "the launcher's pytest inherits DS_BLOCK"),
    ("launch_dscale10k_b3.sh", "ds_pull.py --check-only --items B_s0.A U_s0.A B_rep.A UE_s0.A BL_s0.A", "ds_pull.py --check-only --items UE_s0.A", "the reference checkpoints are not re-checked"),
    ("launch_dscale10k_b3.sh", "grep -q '\"pass\": true' F:/TanitAD-scratch/datascale/read/out/t0route/KEVALX.json || fail", "true || fail", "the launcher skips the block-1 K-EVALX check"),
]
SMOKE_MUTS = [
    ("ds_read_b3.py", 'data_scale_label(ade["pairs"]["U-BL.A"]["point"], ade["pairs"]["UE-BL.A"]["point"], f_a)', 'data_scale_label(ade["pairs"]["U-BL.A"]["point"], ade["pairs"]["UE-B_s0.A"]["point"], f_a)', "the label's second reading takes the wrong pair"),
    ("ds_read_b3.py", 'f_a = float(abs(means["B_rep"] - means["B_s0"]))', 'f_a = float(abs(means["U_s0"] - means["B_s0"]))', "F_A is taken from U_s0 instead of the B_rep replicate"),
    ("ds_read_b3.py", '"UE-U_s0.A": ("UE_s0", "U_s0"),', '"UE-U_s0.A": ("UE_s0", "B_s0"),', "the UE - U_s0 pair is read against B_s0"),
]
FILES = [f for f in os.listdir(HERE) if f.endswith((".py", ".sh", ".ps1", ".cmd")) and f not in ("mutate_check.py", "mutate_check_b3.py", "make_pins.py", "smoke_route_one.py")] + ["READ_PINS.json"]
def prep(n):
    d = f"{W}/{n}"
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in FILES:
        shutil.copy(f"{HERE}/{f}", d)
    shutil.copytree(HERE + "/theory", d + "/theory")
    shutil.copytree(HERE + "/ds_code", d + "/ds_code")
    return d
def run(d, smoke=False):
    subprocess.run([PY, WC, "6", "120"], check=True)
    env = {k: v for k, v in os.environ.items() if k != "DS_BLOCK"}
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", TMP=W, TEMP=W, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    if smoke:
        tree = "F:/TanitAD-scratch/otband_read/tree"
        env["PYTHONPATH"] = f"{tree}/stack;{tree}/taniteval"
        return subprocess.run([PY, "smoke_ds_read_b3.py"], cwd=d, env=env, capture_output=True, encoding="utf-8").returncode
    return subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={d}/bt", "test_ds_read_b3.py"], cwd=d, env=env, capture_output=True, encoding="utf-8").returncode
def mutate(d, f, old, new):
    s = open(f"{d}/{f}", encoding="utf-8", newline="").read()
    assert old in s, (f, old)
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
res = {}
os.makedirs(W, exist_ok=True)
smoke = "--smoke" in sys.argv
ONLY = [int(x) for x in sys.argv[sys.argv.index("--only") + 1:] if x.isdigit()] if "--only" in sys.argv else None      # re-run selected unit mutations (indices into MUTS); the baseline always runs
if not smoke:
    d = prep("base"); res["unmutated"] = {"rc": run(d), "expect": 0}
    for i, (f, old, new, what) in enumerate(MUTS):
        if ONLY is not None and i not in ONLY:
            continue
        d = prep(f"m{i}")
        mutate(d, f, old, new)
        res[f"m{i}: {what}"] = {"rc": run(d), "expect": "nonzero"}
        shutil.rmtree(d)
else:
    for i, (f, old, new, what) in enumerate(SMOKE_MUTS):
        d = prep(f"s{i}")
        mutate(d, f, old, new)
        res[f"s{i}: {what}"] = {"rc": run(d, True), "expect": "nonzero"}
        shutil.rmtree(d)
ok = all(v["rc"] != 0 for k, v in res.items() if k != "unmutated") and (smoke or res["unmutated"]["rc"] == 0)
os.makedirs(HERE + "/out", exist_ok=True)
json.dump({"all_red": ok, "results": res}, open(HERE + "/out/MUTATIONS_B3" + ("_SMOKE" if smoke else "") + (".partial" if ONLY is not None else "") + ".json", "w"), indent=1)
print("MUTATIONS all red:", ok, sum(1 for k, v in res.items() if k != "unmutated" and v["rc"] != 0), "of", len(res) - (0 if smoke else 1))
if ONLY is not None:
    print("(partial re-run of mutations", ONLY, "; the full list is in out/MUTATIONS_B3.json)")
shutil.rmtree(W, ignore_errors=True)
