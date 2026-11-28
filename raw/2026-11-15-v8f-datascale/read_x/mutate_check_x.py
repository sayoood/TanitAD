"""mutate_check_x.py -- one source mutation per committed X_s0 read rule; test_ds_read_x.py must go RED on every one (and GREEN on the unmutated copy).  Writes out/MUTATIONS_X.json.  Usage: python mutate_check_x.py
Before every pytest run it waits for the dev-box commit headroom (wait_commit.py 6 120).  The mutations in SMOKE_MUTS (reader wiring inside main, which unit tests do not reach) are checked with smoke_ds_read_x.py instead
(python mutate_check_x.py --smoke): the smoke runs at B 200, where K-RE's interval fields differ from the banked B 2,000 rows by construction, so a mutation counts RED when the SET of failing smoke checks differs from the unmutated copy's at B 200.
Copies live under readx/mut (own temporary data, removed at the end)."""
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
    # ---- R2 label
    ("ds_read_x.py", "    if delta <= -f_cell:", "    if delta < -f_cell:", "X-CONTENT is not inclusive at Delta == -F"),
    ("ds_read_x.py", "    if delta >= f_cell:", "    if delta > f_cell:", "X-GAIN is not inclusive at Delta == +F"),
    ("ds_read_x.py", "    if delta <= -f_cell:\n        return \"X-CONTENT\"", "    if delta >= f_cell:\n        return \"X-CONTENT\"", "the sign of the label is flipped"),
    ("ds_read_x.py", "    if not separated:\n        return \"X-NO-COST\"", "    if False:\n        return \"X-NO-COST\"", "an unseparated Delta is labelled by its size"),
    ("ds_read_x.py", "or f_cell <= 0.0 or delta is None", "or delta is None", "a zero floor is accepted"),
    ("ds_read_x.py", "or delta is None or not math.isfinite(delta) or separated is None:", "or delta is None or separated is None:", "a NaN delta is not NOT-YET"),
    ("ds_read_x.py", "or separated is None:\n        return \"NOT-YET\"", ":\n        return \"NOT-YET\"", "an unread separation is not NOT-YET"),
    ("ds_read_x.py", "    sens = bool(la in LABELS and lb in LABELS and la != lb)", "    sens = False", "STAGE-SENSITIVE is never reported"),
    ("ds_read_x.py", "    sens = bool(la in LABELS and lb in LABELS and la != lb)", "    sens = bool(la != lb)", "a stage-B NOT-YET counts as STAGE-SENSITIVE"),
    ("ds_read_x.py", "    lb = label_x((rowB or {}).get(\"delta\"), (rowB or {}).get(\"separated\"), fB)", "    lb = label_x((rowB or {}).get(\"delta\"), (rowB or {}).get(\"separated\"), fA)", "stage B is read against the stage-A floor"),
    ("ds_read_x.py", "    return {\"label\": la, \"label_stage_A\": la", "    return {\"label\": lb, \"label_stage_A\": la", "stage B decides instead of stage A"),
    ("ds_read_x.py", "    t = f\"{lab['label']} (one seed)\"", "    t = f\"{lab['label']}\"", "the label does not carry (one seed)"),
    ("ds_read_x.py", "    if blocked:\n        return [\"BLOCKED\"]", "    if False:\n        return [\"BLOCKED\"]", "BLOCKED no longer precedes"),
    ("ds_read_x.py", "    if problems:\n        return [f\"{label_text(lab)} (CONTROL FAILED", "    if False:\n        return [f\"{label_text(lab)} (CONTROL FAILED", "a failed control leaves the label quotable"),
    ("ds_read_x.py", "    if not ade_given or lab is None or lab[\"label\"] not in LABELS:", "    if False:", "an unread deciding cell is not NOT-YET"),
    # ---- R4 controls
    ("ds_read_x.py", "    ok_floor, ok_abs = bool(x_vlm <= bar), bool(x_vlm <= abs_bar)", "    ok_floor, ok_abs = bool(x_vlm < bar), bool(x_vlm < abs_bar)", "K-VLM-ZERO is not inclusive"),
    ("ds_read_x.py", "    return {\"pass\": bool(ok_floor or ok_abs), \"x_vlm\"", "    return {\"pass\": bool(ok_floor and ok_abs), \"x_vlm\"", "K-VLM-ZERO needs both bars"),
    ("ds_read_x.py", "    return {\"pass\": bool(ok_floor or ok_abs), \"x_vlm\"", "    return {\"pass\": True, \"x_vlm\"", "K-VLM-ZERO always passes"),
    ("ds_read_x.py", "    bar = b_vlm - floor", "    bar = b_vlm + floor", "K-VLM-ZERO adds the floor"),
    ("ds_read_x.py", "VLM_ABS_BAR = 0.05 ", "VLM_ABS_BAR = 0.10 ", "the absolute bar is 0.10"),
    ("ds_read_x.py", "    if not all(v is not None and math.isfinite(v) for v in (x_vlm, b_vlm, floor)):", "    if False:", "a missing macro-F1 is not a fail"),
    ("ds_read_x.py", "[\"no cell is OK\"] if not ok else []", "[]", "K1 passes when nothing was read (double-quoted form)"),
    ("ds_read_x.py", "if not (v[\"delta\"] == 0.0 and v[\"lo\"] == 0.0 and v[\"hi\"] == 0.0)]", "if not (v[\"delta\"] == 0.0)]", "K1 ignores the interval"),
    ("ds_read_x.py", "or set(k.get(\"items\", {})) != want or not all(", "or not all(", "K-EVALX accepts a different item set"),
    ("ds_read_x.py", "def kevalx_problem_x(path, arms, stages=STAGES):", "def kevalx_problem_x(path, arms, stages=(\"A\",)):", "K-EVALX on X covers stage A only"),
    ("ds_read_x.py", "for v in B2.ROUTE_VARIANTS if not os.path.exists(", "for v in (\"real\", \"donor\") if not os.path.exists(", "the zero / realx / donorx passes are not required"),
    ("ds_read_x.py", "        if why:\n            return f\"K-EVALX: {why} (the zero passes are not admissible)\"", "        if False:\n            return f\"K-EVALX: {why} (the zero passes are not admissible)\"", "route_problem_x ignores K-EVALX"),
    ("ds_read_x.py", "    if c.get(\"pass\") is not True or c.get(\"problems\"):", "    if c.get(\"pass\") is False:", "C-JOIN-EQ accepts a record with problems"),
    ("ds_read_x.py", "    if not c:\n        return \"no C-JOIN-EQ entry in the gate record\"", "    if False:\n        return \"no C-JOIN-EQ entry in the gate record\"", "a missing C-JOIN-EQ entry is not a problem"),
    ("ds_read_x.py", "    return {\"pass\": bool(n == 8 and not bad)", "    return {\"pass\": bool(n == 8)", "K-RE of the goal-F1 tool ignores a mismatch"),
    ("ds_read_x.py", "    return None if got == want else f\"the ruling file", "    return None if True else f\"the ruling file", "the ruling file's sha256 is not checked"),
    ("ds_read_x.py", "RULINGS_SHA256 = \"e6e3f24cc79fb65098af3e6b166ebb9375d2c65ca939e5c4e937e989a58aecca\"", "RULINGS_SHA256 = \"e6e3f24cc79fb65098af3e6b166ebb9375d2c65ca939e5c4e937e989a58aecc0\"", "the committed sha256 changes"),
    ("ds_read_x.py", "PAIRS = {\"X-B_s0\": (\"X_s0\", \"B_s0\"),", "PAIRS = {\"X-B_s0\": (\"B_s0\", \"X_s0\"),", "X - B_s0 is read with the wrong sign"),
    ("ds_read_x.py", "\"X-U_s0\": (\"X_s0\", \"U_s0\")}", "\"X-U_s0\": (\"X_s0\", \"B_s0\")}", "the beside pair is read against B_s0"),
    ("ds_read_x.py", "CELL = \"TAC-LAT\"", "CELL = \"TAC-LON\"", "the deciding cell is TAC-LON"),
    ("ds_read_x.py", "{f: R.family_status(rows, HEADLINE[f]) for f in FAMILIES}", "(lambda XP: {f: R.family_status(rows, XP) for f in FAMILIES})(tuple(c for g in FAMILIES for c in HEADLINE[g]))", "the families are pooled into one set of cells"),
    ("ds_read_x.py", "    r.update(strat_rows or {})", "    pass", "the STRATEGIC cells are dropped from the family rows"),
    # ---- block-X constants, roots, jobs
    ("ds_lib.py", "X_ITEMS = tuple((a, s) for a in ARMS_X for s in (\"A\", \"B\"))", "X_ITEMS = tuple((a, s) for a in ARMS_X for s in (\"A\",))", "X is held to stage A"),
    ("ds_lib.py", "(BLOCK == 4 and arm in REF_ARMS_X)", "(BLOCK == 4 and arm in ())", "the block-1 arms are looked for in this folder's checkpoints"),
    ("ds_lib.py", "DEFAULT_ITEMS = X_ITEMS if BLOCK == 4 else", "DEFAULT_ITEMS = X_ITEMS if BLOCK == 7 else", "the block-X item default"),
    ("ds_lib.py", "REF_ARMS_X = (\"B_s0\", \"U_s0\", \"B_rep\")", "REF_ARMS_X = (\"B_s0\", \"U_s0\")", "B_rep is not a reference arm"),
    ("ds_run.py", "ROUTE_STAGES = (\"A\", \"B\") if L.BLOCK == 4 else", "ROUTE_STAGES = (\"B\",) if L.BLOCK == 4 else", "X is routed at stage B only"),
    ("ds_run.py", "        if stage not in ROUTE_STAGES:\n            continue", "        if stage != \"B\":\n            continue", "K-EVALX for X covers stage B only"),
    ("ds_pull.py", "[GIT_BASH, \"-c\"", "[\"bash\", \"-c\"", "ds_pull calls the bare bash name (the WSL stub)"),
    # ---- make_banker_x
    ("make_banker_x.py", "    if rec.get(\"tag\") != tag or rec.get(\"status\") != \"BANKED\":", "    if False:", "the ledger line's tag and status are not checked"),
    ("make_banker_x.py", "    if rec.get(\"repo\") != REPO or not rec.get(\"revision\"):", "    if False:", "the repo and revision are not checked"),
    ("make_banker_x.py", "or files[ck][1] != rec[\"sha256\"]:", ":", "the ledger's own sha256 consistency is not checked"),
    ("make_banker_x.py", "    for f in (ck, sm, gc):\n        if f not in files:", "    for f in (ck, sm):\n        if f not in files:", "gate/checks.json need not be listed"),
    ("make_banker_x.py", "    if not private:\n        return \"the HF repo is not private\"", "    if False:\n        return \"the HF repo is not private\"", "a public HF repo is accepted"),
    ("make_banker_x.py", "    if hub[ck] != rec[\"sha256\"]:", "    if False:", "the hub's LFS sha256 is not compared"),
    ("make_banker_x.py", "    if ck not in hub or sm not in hub:", "    if ck not in hub:", "summary.json need not be on the hub"),
    ("make_banker_x.py", "    return None if last[0] == \"DONE\" else", "    return None if last else", "a chain FAIL line is accepted"),
    ("make_banker_x.py", "        m = SENTINEL.match(ln.strip())", '        m = re.search(r"ZZ DSCALE10K BLOCKX (DONE|FAIL)\b", ln.strip())', "a sentinel quoted mid-line counts"),
    ("make_banker_x.py", "        if r.get(\"status\") == \"BANKED\" and r.get(\"tag\"):", "        if r.get(\"tag\"):", "a non-BANKED ledger line counts as banked"),
    ("make_banker_x.py", "        if why:\n            print(tag, \"NOT-BANKED:\", why, flush=True)", "        if False:\n            print(tag, \"NOT-BANKED:\", why, flush=True)", "a stage that is not banked still gets a banker record"),
    ("make_banker_x.py", "\"hf_rev\": rec[\"revision\"],", "\"hf_rev\": None,", "the banked revision is not pinned"),
    # ---- launcher
    ("launch_dscale10k_x.sh", '[ "$FREE" -lt 25 ]', '[ "$FREE" -lt 0 ]', "the launcher's F: 25 GB bar"),
    ("launch_dscale10k_x.sh", "env -u DS_BLOCK TMP", "TMP", "the launcher's pytest inherits DS_BLOCK"),
    ("launch_dscale10k_x.sh", "test_ds_read_x.py test_ds_read.py", "test_ds_read_x.py", "the block-1 tests are not run against this copy"),
    ("launch_dscale10k_x.sh", "ds_pull.py --check-only --items X_s0.A X_s0.B B_s0.A B_s0.B U_s0.A U_s0.B B_rep.A B_rep.B", "ds_pull.py --check-only --items X_s0.A", "the reference checkpoints are not re-checked"),
    ("launch_dscale10k_x.sh", "$PY -u make_banker_x.py >> \"$LOG\" 2>&1 || fail", "$PY -u make_banker_x.py >> \"$LOG\" 2>&1 || true", "the launcher does not stop when X_s0 is not banked"),
    ("launch_dscale10k_x.sh", "$PY -u $WC 5.5 30 >> \"$LOG\" 2>&1 || fail \"commit headroom stayed under 5.5 GB for 30 min\"\nenv", "env", "the launcher does not wait for commit headroom before the tests"),
    ("launch_dscale10k_x.sh", "DS_BLOCK=4", "DS_BLOCK=3", "the launcher runs the block-3 defaults"),
    ("launch_dscale10k_x.sh", "|| fail \"the ruling file MM_RULINGS_X_READ.md", "|| true # \"the ruling file MM_RULINGS_X_READ.md", "the ruling-file sha256 gate does not stop the launcher"),
    ("launch_dscale10k_x.sh", "echo \"ZZ DSCALE10K X READ DONE ZZ\"", "echo \"ZZ DSCALE10K X READ FINISHED ZZ\"", "the DONE sentinel changes"),
]
SMOKE_MUTS = [
    ("ds_read_x.py", "lab = stage_labels(rows[\"A\"], rows[\"B\"], F[\"A\"][CELL], F[\"B\"][CELL])", "lab = stage_labels(rows[\"A\"], rows[\"B\"], F[\"A\"][CELL], F[\"A\"][CELL])", "stage B is labelled against the stage-A floor in main"),
    ("ds_read_x.py", "rows = {st: pairs[f\"X-B_s0.{st}\"].get(CELL) for st in STAGES}", "rows = {st: pairs[f\"X-U_s0.{st}\"].get(CELL) for st in STAGES}", "the label is read from X - U_s0"),
    ("ds_read_x.py", "f1[\"floor\"][\"A\"][\"VLM\"])\n        kz[", "f1[\"floor\"][\"A\"][\"GEO\"])\n        kz[", "K-VLM-ZERO takes the GEO floor"),
    ("ds_read_x.py", "kz = k_vlm_zero(f1[\"stage\"][\"A\"][\"X_s0\"][\"VLM\"], f1[\"stage\"][\"A\"][\"B_s0\"][\"VLM\"]", "kz = k_vlm_zero(f1[\"stage\"][\"B\"][\"X_s0\"][\"VLM\"], f1[\"stage\"][\"B\"][\"B_s0\"][\"VLM\"]", "K-VLM-ZERO is read at stage B"),
]
FILES = [f for f in os.listdir(HERE) if f.endswith((".py", ".sh", ".ps1", ".cmd")) and f not in ("mutate_check_x.py",)] + ["READ_PINS.json"]


def prep(n):
    d = f"{W}/{n}"
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in FILES:
        shutil.copy(f"{HERE}/{f}", d)
    shutil.copytree(HERE + "/theory", d + "/theory")
    if os.path.isdir(HERE + "/evidence"):
        shutil.copytree(HERE + "/evidence", d + "/evidence")
    return d


def env_for(smoke):
    env = {k: v for k, v in os.environ.items() if k != "DS_BLOCK"}
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", TMP=W, TEMP=W, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    if smoke:
        tree = "F:/TanitAD-scratch/otband_read/tree"
        env["PYTHONPATH"] = f"{tree}/stack;{tree}/taniteval"
        env["DS_SMOKE_NB"] = "200"
    return env


def run(d, smoke=False):
    subprocess.run([PY, WC, "6", "120"], check=True)
    if smoke:
        r = subprocess.run([PY, "smoke_ds_read_x.py"], cwd=d, env=env_for(True), capture_output=True, encoding="utf-8")
        line = [x for x in r.stdout.splitlines() if x.startswith("SMOKE_CHECKS ")]
        return r.returncode, (json.loads(line[-1][len("SMOKE_CHECKS "):]) if line else {"crashed": True, "tail": (r.stderr or "")[-200:]})
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={d}/bt", "test_ds_read_x.py"], cwd=d, env=env_for(False), capture_output=True, encoding="utf-8")
    return r.returncode, None


def mutate(d, f, old, new):
    s = open(f"{d}/{f}", encoding="utf-8", newline="").read()
    assert old in s, (f, old)
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))


if __name__ == "__main__":
    res = {}
    os.makedirs(W, exist_ok=True)
    smoke = "--smoke" in sys.argv
    ONLY = [int(x) for x in sys.argv[sys.argv.index("--only") + 1:] if x.isdigit()] if "--only" in sys.argv else None
    if not smoke:
        d = prep("base")
        res["unmutated"] = {"rc": run(d)[0], "expect": 0}
        shutil.rmtree(d)
        for i, (f, old, new, what) in enumerate(MUTS):
            if ONLY is not None and i not in ONLY:
                continue
            d = prep(f"m{i}")
            mutate(d, f, old, new)
            res[f"m{i}: {what}"] = {"rc": run(d)[0], "expect": "nonzero"}
            shutil.rmtree(d)
        ok = all(v["rc"] != 0 for k, v in res.items() if k != "unmutated") and res["unmutated"]["rc"] == 0
        n_red = sum(1 for k, v in res.items() if k != "unmutated" and v["rc"] != 0)
        total = len(res) - 1
    else:
        d = prep("sbase")
        rc0, base = run(d, True)
        shutil.rmtree(d)
        res["unmutated (B 200)"] = {"failing_checks": sorted(k for k, v in base.items() if v is False), "n_checks": len(base)}
        for i, (f, old, new, what) in enumerate(SMOKE_MUTS):
            d = prep(f"s{i}")
            mutate(d, f, old, new)
            rc, got = run(d, True)
            fails = sorted(k for k, v in got.items() if v is False) if got else ["crashed"]
            res[f"s{i}: {what}"] = {"failing_checks": fails, "differs_from_unmutated": fails != res["unmutated (B 200)"]["failing_checks"], "rc": rc}
            shutil.rmtree(d)
        ok = bool(base) and not base.get("crashed") and all(v["differs_from_unmutated"] for k, v in res.items() if k.startswith("s"))
        n_red = sum(1 for k, v in res.items() if k.startswith("s") and v["differs_from_unmutated"])
        total = len(SMOKE_MUTS)
    os.makedirs(HERE + "/out", exist_ok=True)
    json.dump({"all_red": ok, "results": res}, open(HERE + "/out/MUTATIONS_X" + ("_SMOKE" if smoke else "") + (".partial" if ONLY is not None else "") + ".json", "w"), indent=1)
    print("MUTATIONS all red:", ok, n_red, "of", total)
    shutil.rmtree(W, ignore_errors=True)
