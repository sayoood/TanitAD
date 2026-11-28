"""mutate_check_mix.py -- one source mutation per committed rule of the mixture-arm read; test_ds_read_mix.py must go RED on every one (and GREEN on the unmutated copy).  Writes out/MUTATIONS_MIX.json.
Usage: python mutate_check_mix.py            (unit mutations)
       python mutate_check_mix.py --smoke    (reader wiring inside main, which unit tests do not reach: smoke_ds_read_mix.py bl at B 2,000 must go RED; about 6 min each)
       python mutate_check_mix.py --only 3 7 (re-run selected unit mutations)
Before every run it waits for the dev-box commit headroom (wait_commit.py 6 120)."""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
WC = "F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py"
W = HERE + "/mut"
MUTS = [
    # the deciding label
    ("ds_read_mix.py", '        return "DILUTION-CONFIRMED" if um_bl["hi"] >= 0 else "DILUTION-PARTIAL"', '        return "DILUTION-CONFIRMED"', "the UM - BL condition is dropped from DILUTION-CONFIRMED"),
    ("ds_read_mix.py", 'um_bl["hi"] >= 0 else', 'um_bl["hi"] > 0 else', "the UM - BL upper end exactly 0 reads separated worse"),
    ("ds_read_mix.py", 'um_bl["hi"] >= 0 else', 'um_bl["lo"] >= 0 else', "the UM - BL condition reads the lower end"),
    ("ds_read_mix.py", '"UM-UE.A": ("UM_s0", "UE_s0"),', '"UM-UE.A": ("UM_s0", "BL_s0"),', "the UE and BL roles are swapped (deciding pair 1)"),
    ("ds_read_mix.py", '"UM-BL.A": ("UM_s0", "BL_s0"),', '"UM-BL.A": ("UM_s0", "UE_s0"),', "the UE and BL roles are swapped (deciding pair 2)"),
    ("ds_read_mix.py", '"UM-UE.A": "UE_s0", "UM-BL.A": "BL_s0"', '"UM-UE.A": "BL_s0", "UM-BL.A": "UE_s0"', "the mechanism reference frames of the deciding pairs are swapped"),
    ("ds_read_mix.py", 'r.get("floor") is not None and abs(r["delta"]) > r["floor"]', 'r.get("floor") is not None and abs(r["delta"]) >= r["floor"]', "Delta exactly at F_cell reads a gain"),
    ("ds_read_mix.py", ' and r.get("floor") is not None and abs(r["delta"]) > r["floor"]', "", "the F_cell condition is dropped from a gain"),
    ("ds_read_mix.py", 'return row_ok(r) and bool(r["separated"]) and r["delta"] > 0 and r.get("floor")', 'return row_ok(r) and bool(r["separated"]) and r["delta"] < 0 and r.get("floor")', "the gain sign is flipped (TAC-LAT is accuracy)"),
    ("ds_read_mix.py", 'return row_ok(r) and bool(r["separated"]) and r["delta"] > 0\n', 'return row_ok(r) and r["delta"] > 0\n', "separated better ignores separation"),
    ("ds_read_mix.py", '    if row_sep_better(um_ue):\n        return "DILUTION-PARTIAL"', '    if row_sep_better(um_ue):\n        return "NOT-DILUTION"', "a separated gain inside F_cell reads NOT-DILUTION"),
    ("ds_read_mix.py", '    if row_sep_worse(um_ue):\n        return "REBALANCE-HURTS"', '    if False:\n        return "REBALANCE-HURTS"', "REBALANCE-HURTS is never reported"),
    ("ds_read_mix.py", 'return "NOT-DILUTION"\n\n\ndef outcome_mix', 'return "DILUTION-PARTIAL"\n\n\ndef outcome_mix', "an unseparated UM - UE reads DILUTION-PARTIAL"),
    ("ds_read_mix.py", "    if not (row_ok(um_ue) and row_ok(um_bl)):\n        return \"NOT-YET\"", "    if False:\n        return \"NOT-YET\"", "an unreadable row is not NOT-YET"),
    # precedence
    ("ds_read_mix.py", "    if blocked:\n        return [\"BLOCKED\"]", "    if False:\n        return [\"BLOCKED\"]", "BLOCKED no longer precedes"),
    ("ds_read_mix.py", "    if control_problems:\n        return", "    if False:\n        return", "a failed control leaves the label quotable"),
    ("ds_read_mix.py", '    if unread or lab is None or lab == "NOT-YET":', "    if False:", "an unread item is not NOT-YET"),
    ("ds_read_mix.py", '    if blocked:\n        return ["BLOCKED"]\n    if control_problems:\n        return ["CONTROL-FAILED (not quotable)"]\n    if unread', '    if control_problems:\n        return ["CONTROL-FAILED (not quotable)"]\n    if blocked:\n        return ["BLOCKED"]\n    if unread', "CONTROL-FAILED precedes BLOCKED"),
    ("ds_read_mix.py", '    return f"{text} [{A1_TAG}; K-TACSRC as committed: tac_src_empty_batches 10 > 5, BLOCKED as committed]"', '    return text', "the UM-A1 line is not printed beside the label"),
    ("ds_read_mix.py", '; K-TACSRC as committed: tac_src_empty_batches 10 > 5, BLOCKED as committed]"', ']"', "the original as-committed line is not printed"),
    # K-TACSRC under UM-A1
    ("ds_read_mix.py", '    if s.get("tac_src_w") != 0.0:', "    if False:", "K-TACSRC does not check tac_src_w"),
    ("ds_read_mix.py", '    if s.get("tac_src_list_sha256") != LIST_SHA:', "    if False:", "K-TACSRC does not check the list sha256"),
    ("ds_read_mix.py", '    if s.get("tac_src_weight_share_listed") != 1.0:', "    if False:", "K-TACSRC does not check the listed weight share"),
    ("ds_read_mix.py", '    if not (isinstance(ro, (int, float)) and ro > 0):', "    if False:", "K-TACSRC does not check rows_other > 0"),
    ("ds_read_mix.py", 'or s[k_s] != rp[k_r]:', "or False:", "the empty-batch / row counts are not compared with the replay exactly"),
    ("ds_read_mix.py", '(("tac_src_empty_batches", "empty_batches_replayed"), ("tac_src_rows_listed", "rows_listed_replayed"), ("tac_src_rows_other", "rows_other_replayed"))', '(("tac_src_empty_batches", "empty_batches_replayed"),)', "rows listed / other are not compared with the replay"),
    ("ds_read_mix.py", 'ok_committed_bound = isinstance(eb, int) and eb <= EMPTY_BOUND', 'ok_committed_bound = True', "the as-committed bound always passes"),
    ("ds_read_mix.py", "EMPTY_BOUND = 5", "EMPTY_BOUND = 10", "the committed bound is moved to 10"),
    ("ds_read_mix.py", '    return {"pass_under_UM_A1": not problems,', '    return {"pass_under_UM_A1": True,', "K-TACSRC passes whatever the fields read"),
    # gate, pins
    ("ds_read_mix.py", 'or post.get("with_adam_state") != 252 or post.get("problems")', 'or post.get("problems")', "the Adam-state count is not checked"),
    ("ds_read_mix.py", 'post.get("registered") != 252 or ', "", "the registered-parameter count is not checked"),
    ("ds_read_mix.py", '        bad = sorted(k for k, v in checks.items() if not (isinstance(v, dict) and v.get("pass") is True))', "        bad = []", "a failed gate check is not read"),
    ("ds_read_mix.py", '"b2b2c4f0d189359e19d60ff5619ecfe36ab92b5c41d800d9630560ba6ce860d6"', '"b2b2c4f0d189359e19d60ff5619ecfe36ab92b5c41d800d9630560ba6ce860d7"', "the amendment pin is wrong"),
    ("ds_read_mix.py", '        if got != want:\n            return', '        if False:\n            return', "the card / amendment / replay hashes are not checked"),
    # beside labels
    ("ds_read_mix.py", 'return bool(fam_um_ue["status"] != "cost" and fam_um_bl["status"] == "gain")', 'return bool(fam_um_ue["status"] != "cost" and fam_um_bl["status"] != "cost")', "STRAT-KEPT reads a non-cost against BL as a gain"),
    ("ds_read_mix.py", 'return bool(fam_um_ue["status"] != "cost" and', 'return bool(', "STRAT-KEPT drops the UM - UE condition"),
    ("ds_read_mix.py", 'return bool(row and row.get("status") == "OK" and row.get("cost"))', 'return bool(row and row.get("status") == "OK" and row.get("gain"))', "TAC-LON-PRICE reads a gain"),
    ("ds_read_mix.py", 'for f in ("LONGITUDINAL", "LATERAL") if', 'for f in ("LONGITUDINAL",) if', "a LATERAL family cost is not named"),
    # controls and files
    ("ds_read_mix.py", '    if not k.get("pass") or set(k.get("items", {})) != want or not all(v["identical"] for v in k["items"].values()):', "    if False:", "K-EVALX is not checked"),
    ("ds_read_mix.py", "for v in B2.ROUTE_VARIANTS if not os.path.exists", 'for v in ("real", "donor") if not os.path.exists', "the zero / realx / donorx passes are not required"),
    ("ds_read_mix.py", '    return {"pass": bool(n == len(recomputed) and not bad), "rows_compared": n, "mismatches": bad}', '    return {"pass": bool(n == len(recomputed)), "rows_compared": n, "mismatches": bad}', "K-RE of the ADE6 means ignores a mismatch"),
    ("ds_read_mix.py", '    return {"pass": bool(n == len(banked) and not bad), "rows_compared": n, "mismatches": bad[:12]}', '    return {"pass": True, "rows_compared": n, "mismatches": bad[:12]}', "K-RE of the floors always passes"),
    ("ds_read_mix.py", "{f: R.family_status(rows, HEADLINE[f]) for f in FAMILIES}", 'XPOOL = tuple(c for g in FAMILIES for c in HEADLINE[g]); return {f: R.family_status(rows, XPOOL) for f in FAMILIES}', "the families are pooled into one set of cells"),
    ("ds_read_mix.py", '            r.update(strat["pairs"][p])', "            pass", "the STRATEGIC cells are dropped from the family rows"),
    # constants and roots
    ("ds_lib.py", 'STEPS_ARM["UM_s0"] = 61568', 'STEPS_ARM["UM_s0"] = 19164', "UM is held to 19,164 steps"),
    ("ds_lib.py", 'ARMS_MIX = {"UM_s0": 0}', 'ARMS_MIX = {"UM_s0": 1}', "UM is given seed 1"),
    ("ds_lib.py", 'MIX_PULL_ITEMS = tuple((a, "A") for a in ("UM_s0", "UE_s0", "BL_s0"))', 'MIX_PULL_ITEMS = tuple((a, "A") for a in ("UM_s0", "UE_s0"))', "BL is not pulled"),
    ("ds_lib.py", '(BLOCK == "mix" and arm in MIX_REF_B1)', '(BLOCK == "mix" and arm in MIX_REF_B1 + ("UM_s0",))', "UM is looked for in the block-1 checkpoints"),
    ("ds_lib.py", 'MIX_REF_B1 = ("B_s0", "U_s0", "B_rep")', 'MIX_REF_B1 = ("B_s0", "U_s0")', "B_rep is looked for in this folder"),
    ("ds_lib.py", 'DEFAULT_ITEMS = MIX_ITEMS if BLOCK == "mix" else', 'DEFAULT_ITEMS = MIX_ITEMS if BLOCK == "mixx" else', "the mix item default"),
    ("ds_run.py", 'ROUTE_STAGE = "A" if L.BLOCK in (3, "mix") else "B"', 'ROUTE_STAGE = "A" if L.BLOCK in (3,) else "B"', "the mix route passes stay stage B"),
    ("ds_lib.py", "ROUTE_ITEMS_MIX = MIX_ITEMS", "ROUTE_ITEMS_MIX = MIX_PULL_ITEMS", "the mix route default names UE and BL too"),
    ("ds_run.py", 'L.ROUTE_ITEMS_MIX if (route and L.BLOCK == "mix")', 'L.ROUTE_ITEMS_B3 if (route and L.BLOCK == "mix")', "the mix route command defaults to the block-3 five arms"),
    ("ds_pull.py", '(L.MIX_PULL_ITEMS if L.BLOCK == "mix" else L.DEFAULT_ITEMS)', '(L.DEFAULT_ITEMS)', "the pull default skips UE / BL"),
    ("ds_pull.py", 'GIT_BASH = "C:/Program Files/Git/usr/bin/bash.exe"', 'GIT_BASH = "bash"', "the pod wrapper uses the bare bash name (System32 WSL)"),
    # the banker and the chain state
    ("make_banker_mix.py", '    if any(m.group(1) == "DONE" for _, m in sents):', "    if False:", "a DONE chain is accepted"),
    ("make_banker_mix.py", 'post.get("problems") != [ACCEPTED_PROBLEM]', 'ACCEPTED_PROBLEM not in post.get("problems", [])', "an extra problem beside the K-TACSRC refusal is accepted"),
    ("make_banker_mix.py", 'post.get("pass") is not False or ', "", "a passing post check is accepted as the refusal"),
    ("make_banker_mix.py", '    if not pre or not all(\'"pass": true\' in ln for ln in pre):', "    if False:", "a failing pre check is accepted"),
    ("make_banker_mix.py", '    if len(sents) != 1:', "    if False:", "several sentinels are accepted"),
    ("make_banker_mix.py", 'ACCEPTED_PROBLEM = "K-TACSRC tac_src_empty_batches 10 > 5"', 'ACCEPTED_PROBLEM = "K-TACSRC tac_src_empty_batches 11 > 5"', "the accepted problem string is wrong"),
    ("make_banker_mix.py", '    if not rec["sha256"].startswith(CKPT_SHA_PREFIX):', "    if False:", "a different checkpoint sha256 is accepted"),
    ("make_banker_mix.py", "    if hub[ck] != rec[\"sha256\"]:", "    if False:", "the hub LFS sha256 is not compared"),
    ("make_banker_mix.py", "    if not private:", "    if False:", "a public repo is accepted"),
    ("ds_t0.py", "    return q if (os.path.isfile(q) and L.bank_problem(arm, stage.upper(), L.CK) is None) else None", "    return q if os.path.isfile(q) else None", "a re-pulled checkpoint is used without its sha256 check"),
    ("ds_t0.py", 'parts[-1] != "ckpt_last.pt" or parts[-3] != "ckpts" or "." not in parts[-2]', 'False', "the remap accepts any recorded path"),
    # sync
    ("sync_refs.py", "            if L.sha256_file(d) == L.sha256_file(s):", "            if True:", "sync_refs accepts a target with different bytes"),
    ("sync_refs.py", '"R_s0", "UE_s0", "BL_s0"))', '"R_s0", "UE_s0"))', "sync_refs skips BL_s0"),
    # the launcher
    ("launch_dscale_mix.sh", '[ "$FREE" -lt 25 ]', '[ "$FREE" -lt 0 ]', "the launcher F: 25 GB bar"),
    ("launch_dscale_mix.sh", "env -u DS_BLOCK TMP", "TMP", "the launcher pytest inherits DS_BLOCK"),
    ("launch_dscale_mix.sh", "--check-only --items UM_s0.A UE_s0.A BL_s0.A B_s0.A U_s0.A B_rep.A >> \"$LOG\" 2>&1 || fail", "--check-only --items UM_s0.A >> \"$LOG\" 2>&1 || fail", "the reference checkpoints are not re-checked"),
    ("launch_dscale_mix.sh", '$PY -u make_banker_mix.py >> "$LOG" 2>&1 || fail', '$PY -u make_banker_mix.py >> "$LOG" 2>&1 || true; echo', "the banker check is not fail-closed"),
    ("launch_dscale_mix.sh", "grep -q '\"pass\": true' $D/read3/out/t0route/KEVALX.json || fail", "true || fail", "the launcher skips the block-3 K-EVALX check"),
    ("launch_dscale_mix.sh", 'DS_BLOCK=mix', 'DS_BLOCK=3', "the launcher runs the wrong block"),
]
SMOKE_MUTS = [
    ("ds_read_mix.py", "    lab = label_mix(r_ue, r_bl)", "    lab = label_mix(r_bl, r_ue)", "the label is read from the swapped rows"),
    ("ds_read_mix.py", '    r_ue, r_bl = pairs["UM-UE.A"].get(CELL), pairs["UM-BL.A"].get(CELL)', '    r_ue, r_bl = pairs["UM-B_s0.A"].get(CELL), pairs["UM-BL.A"].get(CELL)', "the label takes UM - B_s0 for UM - UE"),
    ("ds_read_mix.py", "(a.dumps3 if arm in ('UE_s0', 'BL_s0') else a.dumps0)", "(a.dumps0 if arm in ('UE_s0', 'BL_s0') else a.dumps3)", "the dump roots of the block-3 and block-1 arms are swapped"),
    ("ds_read_mix.py", "    rep[\"outcome\"] = [tag_um(x) for x in labels]", "    rep[\"outcome\"] = labels", "the outcome is not tagged"),
]
FILES = [f for f in os.listdir(HERE) if f.endswith((".py", ".sh", ".ps1", ".cmd")) and f not in ("mutate_check_mix.py", "make_pins.py", "run_smokes.sh")] + ["READ_PINS.json"]


def prep(n):
    d = f"{W}/{n}"
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in FILES:
        shutil.copy(f"{HERE}/{f}", d)
    for sub in ("theory", "ds_code", "evidence", "banker"):
        shutil.copytree(f"{HERE}/{sub}", f"{d}/{sub}")
    os.makedirs(d + "/out")
    return d


def run(d, smoke=False):
    subprocess.run([PY, WC, "6", "120"], check=True)
    env = {k: v for k, v in os.environ.items() if k != "DS_BLOCK"}
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", TMP=W, TEMP=W, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    if smoke:
        tree = "F:/TanitAD-scratch/otband_read/tree"
        env["PYTHONPATH"] = f"{tree}/stack;{tree}/taniteval"
        return subprocess.run([PY, "smoke_ds_read_mix.py", "bl"], cwd=d, env=env, capture_output=True, encoding="utf-8").returncode
    return subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={d}/bt", "test_ds_read_mix.py"], cwd=d, env=env, capture_output=True, encoding="utf-8").returncode


def mutate(d, f, old, new):
    s = open(f"{d}/{f}", encoding="utf-8", newline="").read()
    assert old in s, (f, old)
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))


res = {}
os.makedirs(W, exist_ok=True)
smoke = "--smoke" in sys.argv
ONLY = [int(x) for x in sys.argv[sys.argv.index("--only") + 1:] if x.isdigit()] if "--only" in sys.argv else None
if not smoke:
    d = prep("base")
    res["unmutated"] = {"rc": run(d), "expect": 0}
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
json.dump({"all_red": ok, "results": res}, open(HERE + "/out/MUTATIONS_MIX" + ("_SMOKE" if smoke else "") + (".partial" if ONLY is not None else "") + ".json", "w"), indent=1)
print("MUTATIONS all red:", ok, sum(1 for k, v in res.items() if k != "unmutated" and v["rc"] != 0), "of", len(res) - (0 if smoke else 1))
shutil.rmtree(W, ignore_errors=True)
