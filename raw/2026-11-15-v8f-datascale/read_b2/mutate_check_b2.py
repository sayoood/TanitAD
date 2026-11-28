"""mutate_check_b2.py -- one source mutation per committed block-2 rule; test_ds_read_b2.py must go RED on every one (and GREEN on the unmutated copy).  Writes out/MUTATIONS_B2.json.  Usage: python mutate_check_b2.py
Before every pytest run it waits for the dev-box commit headroom (wait_commit.py 6 120)."""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
WC = "F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py"
W = HERE + "/mut"
POOL = 'tuple(c for g in FAMILIES if g != "STRATEGIC" for c in HEADLINE[g])'
MUTS = [
    ("ds_read_b2.py", "    return R.outcome_label(per_seed, True, None)", '    return R.outcome_label({0: per_seed[0], 1: {f: ({**v, "status": "none"} if v["status"] == "cost" else v) for f, v in per_seed[1].items()}}, True, None)', "a seed-1 cost is dropped"),
    ("ds_read_b2.py", "{f: R.family_status(rows, HEADLINE[f]) for f in FAMILIES}", "{f: R.family_status(rows, " + POOL + ") for f in FAMILIES}", "the families are pooled into one set of cells"),
    ("ds_read_b2.py", 'if any(v.get("status") == "NOT-YET" for s in per_seed.values() for v in s.values()):', "if False:", "an unread family is read as no gain"),
    ("ds_read_b2.py", '    if blocked:\n        return ["BLOCKED"]\n    if not reach_read:', '    if False:\n        return ["BLOCKED"]\n    if not reach_read:', "BLOCKED no longer precedes"),
    ("ds_read_b2.py", '    if not reach_ok:\n        return ["NO-REACH"]\n    if set(per_seed)', '    if False:\n        return ["NO-REACH"]\n    if set(per_seed)', "NO-REACH no longer precedes the labels"),
    ("ds_read_b2.py", '    if not reach_read:\n        return ["NOT-YET"]', '    if False:\n        return ["NOT-YET"]', "the unread reach precondition is skipped"),
    ("ds_read_b2.py", '"pass": bool(n > 0 and not bad and not absent)', '"pass": bool(not bad and not absent)', "K-RE (B1) passes with zero rows compared"),
    ("ds_read_b2.py", '"pass": bool(n > 0 and not bad and not absent)', '"pass": bool(n > 0 and not bad)', "K-RE (B1) ignores a banked cell that was not recomputed"),
    ("ds_read_b2.py", "if isinstance(x, float) and math.isnan(x):\n        return None", "if False:\n        return None", "K-RE (B1) NaN handling"),
    ("ds_read_b2.py", 'tuple(rows["U_vs_B"]) + ("floor_mse",)', '("floor_mse",)', "K-RE (mechanism) compares only the floor"),
    ("ds_lib.py", "DEFAULT_ITEMS = B2_ITEMS if BLOCK == 2 else READ_ITEMS", "DEFAULT_ITEMS = B2_ITEMS if BLOCK == 3 else READ_ITEMS", "the block-2 item default"),
    ("ds_lib.py", 'ARMS_B2 = {"B_s1": 1, "U_s1": 1}', 'ARMS_B2 = {"B_s1": 0, "U_s1": 0}', "seed-1 dumps would be read from the seed-0 folder"),
    ("launch_dscale10k_b2.sh", '    *FAIL*) fail "the block-1 read ended in FAIL: $L1";;', "    *FAIL*) ;;", "the launcher ignores a block-1 FAIL"),
    ("launch_dscale10k_b2.sh", 'tail -n +"$S" "$B1LOG"', 'cat "$B1LOG"', "the launcher reads an older FAIL as the current one"),
    ("launch_dscale10k_b2.sh", '[ "$FREE" -lt 25 ]', '[ "$FREE" -lt 0 ]', "the launcher's F: 25 GB bar"),
]
FILES = [f for f in os.listdir(HERE) if f.endswith(".py") and f not in ("mutate_check.py", "mutate_check_b2.py", "make_pins.py")] + ["READ_PINS.json", "launch_dscale10k_b2.sh", "launch_wmi_b2.ps1", "job_dscale10k_b2.cmd"]
def prep(n):
    d = f"{W}/{n}"
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in FILES:
        shutil.copy(f"{HERE}/{f}", d)
    shutil.copytree(HERE + "/theory", d + "/theory")
    return d
def run(d):
    subprocess.run([PY, WC, "6", "120"], check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", TMP=W, TEMP=W)
    return subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={d}/bt", "test_ds_read_b2.py"], cwd=d, env=env, capture_output=True, encoding="utf-8").returncode
res = {}
os.makedirs(W, exist_ok=True)
d = prep("base"); res["unmutated"] = {"rc": run(d), "expect": 0}
for i, (f, old, new, what) in enumerate(MUTS):
    d = prep(f"m{i}")
    s = open(f"{d}/{f}", encoding="utf-8", newline="").read()
    assert old in s, (f, old)
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
    res[f"m{i}: {what}"] = {"rc": run(d), "expect": "nonzero"}
    shutil.rmtree(d)
ok = res["unmutated"]["rc"] == 0 and all(v["rc"] != 0 for k, v in res.items() if k != "unmutated")
os.makedirs(HERE + "/out", exist_ok=True)
json.dump({"all_red": ok, "results": res}, open(HERE + "/out/MUTATIONS_B2.json", "w"), indent=1)
print("MUTATIONS all red:", ok, sum(1 for k, v in res.items() if k != "unmutated" and v["rc"] != 0), "of", len(MUTS))
shutil.rmtree(W)
