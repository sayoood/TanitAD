"""Rehearsal of the stage-1 package (AGENT E): the REAL tar (`stage1/t2v8f_stage1.tar`) is extracted into C: scratch, a SHADOW kit stands in for box 2's `D:/t2a/box2_kit`
(the real kit at C:/Users/Admin/t2a_scratch/box2_kit is only READ), and the shipped driver runs the whole control flow on CPU with the stub renderer and stub encoder.

Every `check_*(mut=None)` is one arm. Fault arms apply a deliberate fault and assert the harness goes RED for the right reason; `mut="not_applied"` skips the fault, so the same
assertion must fail (the arm can fail). `code/run_mutations_stage1.py` runs the mutations. Run from t2v8f/: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_stage1.py -q
(about 6 min: the positive run does 28 plumbing cells; set T2V8F_S1_QUICK=1 to skip it for the fault arms only).
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
CODE = f"{T2}/code"
sys.path.insert(0, CODE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
REAL_KIT = "C:/Users/Admin/t2a_scratch/box2_kit"
TAR = f"{T2}/stage1/t2v8f_stage1.tar"
SCRATCH = os.environ.get("T2V8F_S1_SCRATCH", "C:/Users/Admin/t2a_scratch/stage1_rehearsal")
SCENES = ["00040136-e651-4abd-991d-0655ccda9430", "7c72937c-c620-4776-9555-d57222c0081f"]
ARMS = ("C0_s0", "C0_s1", "DAGOACC_s0", "DAGOACC_s1", "P10", "ZERO", "BLIND")
MIB = 1 << 20
_CACHE = {}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def clean_env():
    # TANITEVAL_: importing `taniteval` (the pinned reader's `init()`, any dominance test) exports TANITEVAL_STACK_OVERRIDE into THIS process; a child that inherits it refuses the package's own tanitad tree
    # (StackShadowError). The full suite runs the dominance tests BEFORE the stage tests, so every spawned stage child must start without it (AGENT O, 2026-11-06).
    e = {k: v for k, v in os.environ.items() if not k.startswith(("T2A_", "T2V8F_", "PYTHONPATH", "TANITEVAL_"))}
    e.update(CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return e


# ------------------------------------------------------------------------------------------------ the package and the shadow kit
def tree_ok(pkg):
    """Every file of MANIFEST.json present with its size and sha256 (a changed or missing file -> False)."""
    try:
        m = json.load(open(f"{pkg}/MANIFEST.json", encoding="utf-8"))
        return all(os.path.isfile(f"{pkg}/{e['path']}") and os.path.getsize(f"{pkg}/{e['path']}") == e["size"] and sha256_file(f"{pkg}/{e['path']}") == e["sha256"] for e in m["files"])
    except Exception:                                                               # noqa: BLE001
        return False


def ensure_pkg():
    """The real tar, verified against its `.sha256`, extracted once per tar into scratch. -> pkg dir. Refuses a STALE tar (the shipped code must equal the code in t2v8f/code)."""
    if "pkg" in _CACHE:
        return _CACHE["pkg"]
    want = open(TAR + ".sha256").read().split()[0]
    assert sha256_file(TAR) == want, "the tar does not match its .sha256"
    base = f"{SCRATCH}/{want[:12]}"
    pkg = None
    for k in range(20):
        cand = f"{base}/x{k}/pkg"
        if os.path.isdir(cand) and tree_ok(cand):
            pkg = cand
            break
        if not os.path.exists(f"{base}/x{k}"):
            os.makedirs(f"{base}/x{k}")
            with tarfile.open(TAR) as tf:
                tf.extractall(f"{base}/x{k}", filter="data")
            pkg = cand
            break
    assert pkg and tree_ok(pkg), "could not provide a verified extraction"
    m = json.load(open(f"{pkg}/MANIFEST.json", encoding="utf-8"))
    for e in m["files"]:
        if e["path"].startswith("code/"):
            assert sha256_file(f"{CODE}/{e['path'][5:]}") == e["sha256"], f"STALE TAR: {e['path']} differs from t2v8f/{e['path']}: rebuild with code/make_stage1.py"
    _CACHE["pkg"] = pkg
    return pkg


def shadow_kit():
    """box 2's kit as far as the stub rehearsal reads it: the kit's code, 16-byte placeholders for the two scene files, and its own KIT_MANIFEST.json (md5 of what is there)."""
    if "kit" in _CACHE:
        return _CACHE["kit"]
    pkg = ensure_pkg()
    kit = f"{os.path.dirname(os.path.dirname(pkg))}/shadow_kit"
    if not os.path.exists(f"{kit}/KIT_MANIFEST.json"):
        items = []
        for sub in ("t2a", "tree/stack/experiments/alpasim-gsplat"):
            os.makedirs(f"{kit}/{sub}", exist_ok=True)
            for f in sorted(glob.glob(f"{REAL_KIT}/{sub}/*.py")):
                shutil.copyfile(f, f"{kit}/{sub}/{os.path.basename(f)}")
        dst = "tree/stack/experiments/alpasim-gsplat/closedloop_drive.py"
        items.append({"kind": "overlay", "dst": dst, "size": os.path.getsize(f"{kit}/{dst}"), "md5": hashlib.md5(open(f"{kit}/{dst}", "rb").read()).hexdigest()})
        for sid in SCENES:
            os.makedirs(f"{kit}/scenes/{sid}", exist_ok=True)
            for fn in ("volume.nurec", "rig_trajectories.json"):
                b = b"PLACEHOLDER-0000\n"[:16]
                open(f"{kit}/scenes/{sid}/{fn}", "wb").write(b)
                items.append({"kind": "scene", "dst": f"scenes/{sid}/{fn}", "size": 16, "md5": hashlib.md5(b).hexdigest()})
        json.dump({"note": "SHADOW kit for the stage-1 rehearsal (placeholders); the real kit is never written", "items": items}, open(f"{kit}/KIT_MANIFEST.json", "w"), indent=1)
    os.makedirs(f"{kit}/../hf", exist_ok=True)
    _CACHE["kit"] = kit
    return kit


def run_cli(pkg, kit, pull, timeout=1500):
    """The shipped driver, exactly as the launcher runs it, plus the two rehearsal flags. -> (returncode, seconds)."""
    t0 = time.time()
    p = subprocess.run([PY, "-I", "-u", f"{pkg}/code/t2v8f_stage1.py", "all", "--pkg", pkg, "--kit", kit, "--hf", f"{os.path.dirname(kit)}/hf", "--pull", pull, "--stub", "--matrix-inproc"],
                       env=clean_env(), capture_output=True, text=True, timeout=timeout)
    with open(f"{pull}.stdout.txt", "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n---stderr---\n" + p.stderr)
    return p.returncode, round(time.time() - t0, 1)


def new_pull(tag):
    p = f"{os.path.dirname(os.path.dirname(ensure_pkg()))}/pull_{tag}_{time.strftime('%H%M%S')}"
    assert not os.path.exists(p)
    return p


def result(pull):
    return json.load(open(f"{pull}/STAGE1_RESULT.json", encoding="utf-8"))


def allocated_mib(root):
    """What a tree costs on a 1 MiB-cluster exFAT volume: every file rounds up to whole MiB, every directory takes one."""
    tot = 0
    for r, ds, fs in os.walk(root):
        tot += 1 + sum(max(1, math.ceil(os.path.getsize(os.path.join(r, f)) / MIB)) for f in fs)
    return tot


# ------------------------------------------------------------------------------------------------ arm 1: the positive run, every gate
def positive_run():
    """One full stub run of the shipped driver (cached for the session)."""
    if "pos" in _CACHE:
        return _CACHE["pos"]
    pkg, kit = ensure_pkg(), shadow_kit()
    pull = new_pull("pos")
    rc, secs = run_cli(pkg, kit, pull)
    _CACHE["pos"] = (pull, rc, secs)
    return _CACHE["pos"]


def check_positive(mut=None):
    """Every gate reads correctly on the stub: preflight PASS, install PASS, T5 PASS (plans, executed track and every frame md5 bit-equal), gap REPORTED (9 ticks per scene, finite),
    actor PASS (3 lead ticks drawn by `objects`, the planted `empty` read absent, disabled == off), plumbing PASS (7 arms x {T2, RR} x 2 scenes = 28 cells DONE), markers, STAGE1.done."""
    pull, rc, secs = positive_run()
    r = result(pull)
    assert rc == 0, f"driver rc {rc}"
    assert os.path.exists(f"{pull}/STAGE1.done") and not os.path.exists(f"{pull}/STAGE1.fail") and not os.path.exists(f"{pull}/STAGE1.running")
    for job in ("preflight", "install", "t5", "gap", "actor", "plumbing"):
        assert os.path.exists(f"{pull}/{job}.done") and not os.path.exists(f"{pull}/{job}.fail"), f"marker {job}"
    g = r["gates"]
    assert g == {"preflight": "PASS", "install": "PASS", "t5": "PASS", "gap": "REPORTED", "actor": "PASS", "plumbing": "PASS"}, g
    assert r["overall"] == "ALL_GATES_PASS" and r["stub"] is True and r["purpose"].startswith("PLUMBING ONLY")
    s = r["steps"]
    assert s["preflight"]["failures"] == [] and s["preflight"]["info"]["pkg_files_ok"] == s["preflight"]["info"]["pkg_files"] == len(json.load(open(f"{ensure_pkg()}/MANIFEST.json", encoding="utf-8"))["files"])
    assert s["t5"]["n_frames"] > 0 and s["t5"]["frames_differing"] == 0 and s["t5"]["plans_bit_equal"] and s["t5"]["executed_equal"]
    for sid in SCENES:
        rows = s["gap"]["scenes"][sid]["rows"]
        assert len(rows) == 9 and all(math.isfinite(x["cosine"]) and math.isfinite(x["rel_l2"]) and math.isfinite(x["grad_ncc_pixels"]) for x in rows), sid
    rows = s["actor"]["rows"]
    assert len(rows) == 3 and all(x["present"] and x["planted_absent"] and x["disabled_exact"] for x in rows), rows
    assert all(x["changed_planted_empty_vs_off"] == 0 and x["changed_on_vs_empty"] == x["changed_on_vs_off"] > 0 for x in rows), rows
    assert all(x["beside_clock_past_end_absent"] for x in rows)
    cells = s["plumbing"]["cells"]
    assert sorted(k for k in cells if not k.endswith("matrix_rc")) == sorted(f"{sid[:8]}|{tc}|{a}" for sid in SCENES for tc in ("T2", "RR") for a in ARMS)
    assert all(v == "DONE" for k, v in cells.items() if not k.endswith("matrix_rc")) and all(cells[f"{sid[:8]}|matrix_rc"] == 0 for sid in SCENES)
    recs = [json.loads(line) for f in glob.glob(f"{pull}/plumbing/**/records.jsonl", recursive=True) for line in open(f, encoding="utf-8")]
    assert len(recs) == 28 and all(x["purpose"].startswith("PLUMBING ONLY") and len(x["frame_md5"]) > 0 for x in recs if x["arm"] != "BLIND")
    assert {(x["scene_id"], x["f0"]) for x in recs} == {(SCENES[0], 60), (SCENES[1], 22)}
    return {"seconds": secs, "pull_allocated_mib": allocated_mib(pull), "pull_files": sum(len(f) for _, _, f in os.walk(pull))}


# ------------------------------------------------------------------------------------------------ arms 2 and 3: the preflight refuses
def move_aside_and_run(rel, mutate, fault_applied=True):
    pkg, kit = ensure_pkg(), shadow_kit()
    pull = new_pull(os.path.basename(rel).replace(".", "_"))
    p = f"{pkg}/{rel}"
    keep = p + ".aside_by_test"
    orig = open(p, "rb").read()
    try:
        if fault_applied:
            mutate(p, keep, orig)
        rc, secs = run_cli(pkg, kit, pull, timeout=600)
    finally:
        if os.path.exists(keep):
            os.replace(keep, p)
        else:
            open(p, "wb").write(orig)
    assert tree_ok(pkg), "the extraction was not restored"
    return pull, rc


def check_planted_missing(mut=None):
    """`ckpt/DAGOACC_s1.pt` moved away: STAGE1.fail, MISSING named, the steps are [preflight] only (no later job ran)."""
    pull, rc = move_aside_and_run("ckpt/DAGOACC_s1.pt", lambda p, keep, orig: os.replace(p, keep), fault_applied=mut != "not_applied")
    r = result(pull)
    assert rc == 1 and os.path.exists(f"{pull}/STAGE1.fail") and not os.path.exists(f"{pull}/STAGE1.done"), (rc, os.listdir(pull))
    assert list(r["steps"]) == ["preflight"] and r["steps"]["preflight"]["verdict"] == "REFUSED" and r["overall"] == "INFRASTRUCTURE_FAILURE"
    assert r["steps"]["preflight"]["failures"] == ["MISSING ckpt/DAGOACC_s1.pt"], r["steps"]["preflight"]["failures"]
    assert "MISSING ckpt/DAGOACC_s1.pt" in open(f"{pull}/STAGE1.fail", encoding="utf-8").read() and os.path.exists(f"{pull}/preflight.fail")
    return {"rc": rc, "steps": list(r["steps"])}


def check_changed_byte(mut=None):
    """One byte of `refs/SCREEN_TABLE_2.json` changed (same size): HASH_MISMATCH named, STAGE1.fail, preflight only."""
    def flip(p, keep, orig):
        b = bytearray(orig)
        b[len(b) // 2] ^= 0x01
        open(p, "wb").write(bytes(b))
    pull, rc = move_aside_and_run("refs/SCREEN_TABLE_2.json", flip, fault_applied=mut != "not_applied")
    r = result(pull)
    assert rc == 1 and os.path.exists(f"{pull}/STAGE1.fail") and list(r["steps"]) == ["preflight"], (rc, list(r["steps"]))
    assert r["steps"]["preflight"]["failures"] == ["HASH_MISMATCH refs/SCREEN_TABLE_2.json"], r["steps"]["preflight"]["failures"]
    return {"rc": rc}


# ------------------------------------------------------------------------------------------------ arms 4 and 5: the actor falsifier
def probe_actor(mutation):
    pkg, kit = ensure_pkg(), shadow_kit()
    pull = new_pull("probe_" + mutation)
    p = subprocess.run([PY, "-I", f"{HERE}/stage1_probe.py", "--pkg", pkg, "--kit", kit, "--pull", pull, "--mut", mutation], env=clean_env(), capture_output=True, text=True, timeout=600)
    line = [x for x in p.stdout.splitlines() if x.startswith("PROBE_JSON ")]
    assert line, f"probe printed no result (rc {p.returncode}): {p.stderr[-1500:]}"
    return json.loads(line[-1][len("PROBE_JSON "):])


def check_actor_control(mut=None):
    """The probe with no fault reproduces the positive run's actor gate (PASS, 3 of 3 present, planted absent)."""
    a = probe_actor("none")
    assert a["verdict"] == "PASS" and sum(x["present"] for x in a["rows"]) == 3 and all(x["planted_absent"] for x in a["rows"]), a
    return {"changed_on_vs_empty": [x["changed_on_vs_empty"] for x in a["rows"]]}


def check_roi_shifted(mut=None):
    """The ROI read 400 px away from where the lead is drawn: the actor gate must read FAIL (the lead is not found)."""
    a = probe_actor("none" if mut == "not_applied" else "roi_shift")
    assert a["verdict"] == "FAIL" and sum(x["present"] for x in a["rows"]) == 0, a
    return {"present": [x["present"] for x in a["rows"]]}


def check_empty_draws_lead(mut=None):
    """The `empty` condition draws the lead: the planted no-lead read is NOT absent and `objects` and `empty` do not differ: the actor gate must read FAIL."""
    a = probe_actor("none" if mut == "not_applied" else "empty_draws_lead")
    assert a["verdict"] == "FAIL" and not any(x["planted_absent"] for x in a["rows"]) and not any(x["present"] for x in a["rows"]), a
    return {"empty_vs_off": [x["changed_planted_empty_vs_off"] for x in a["rows"]]}


# ------------------------------------------------------------------------------------------------ arm 6: scoring end to end
def check_score_e2e(mut=None):
    """`matrix` on 3 arms (C0_s0, P10, BLIND) x {T2, RR} x 2 scenes x 2 starts through the dev-box entry, then `score`: every group computes the 19 labels (DK rows read n_valid 0
    on the stub's track-less scenes, reported), and the seed-0 contrasts X1 / X2 / BL return status OK. Mutations: `drop_p10_record` (a start missing from the P10 cell -> the
    pairing check refuses X1), `rows_missing` (the S-head record fields removed from one group -> its S-POS rows are missing, not zero)."""
    import t2v8f_score as S
    out = f"{SCRATCH}/score_e2e_{time.strftime('%H%M%S')}_{mut}"
    env = clean_env()
    env.update(T2A_STUB="1", T2A_KIT=shadow_kit(), T2V8F_V9_DIR="F:/TanitAD-scratch/t2v8f/agentA/v9")
    if "e2e_matrix" not in _CACHE:                                   # the matrix is built once per session; each arm scores its own COPY
        base_out = f"{SCRATCH}/score_e2e_matrix_{time.strftime('%H%M%S')}"
        base = ["--arms", "C0_s0,P10,BLIND", "--t2-conds", "T2,RR", "--agents", "objects", "--starts", "60,80", "--out", base_out, "--inproc", "--t2-encoder", "stub", "--t2-device", "cpu",
                "--t2-ckpt-root", f"{ensure_pkg()}/ckpt", "--t2-blind-mean", f"{ensure_pkg()}/refs/blind_mean_eval141.pt", "--t2-plumbing"]
        for idx, sid in enumerate(("scenaaaa0", "scenbbbb1")):
            p = subprocess.run([PY, "-I", f"{CODE}/t2v8f_run.py", "matrix", "--scene-dir", f"{shadow_kit()}/scenes/{SCENES[idx]}", "--scene-idx", str(idx), "--scene-id", sid] + base,
                               env=env, capture_output=True, text=True, timeout=900)
            assert p.returncode == 0, p.stdout[-600:] + p.stderr[-1200:]
        _CACHE["e2e_matrix"] = base_out
    shutil.copytree(_CACHE["e2e_matrix"], out)
    recs = sorted(glob.glob(f"{out}/**/records.jsonl", recursive=True))
    assert len(recs) == 12, len(recs)
    if mut == "drop_p10_record":                                     # one start of one P10 cell lost
        f = [x for x in recs if "/P10/" in x.replace("\\", "/") and "/T2/" in x.replace("\\", "/") and "scenaaaa" in x.replace("\\", "/")][0]
        lines = open(f, encoding="utf-8").read().splitlines()
        open(f, "w", encoding="utf-8", newline="\n").write(lines[0] + "\n")
    if mut == "rows_missing":                                        # the S head's outputs are gone from every C0_s0 / T2 record
        for f in [x for x in recs if "/C0_s0/" in x.replace("\\", "/") and "/T2/" in x.replace("\\", "/")]:
            rs = [json.loads(line) for line in open(f, encoding="utf-8")]
            for r_ in rs:
                for e in r_["extra"]:                                # `extra` is one dict per S-head tick
                    e.pop("s_ckpt", None)
                    e.pop("s_pose", None)
            open(f, "w", encoding="utf-8", newline="\n").write("".join(json.dumps(r_) + "\n" for r_ in rs))
    sc = f"{out}/SCORE.json"
    p = subprocess.run([PY, "-I", f"{CODE}/t2v8f_run.py", "score", "--root", out, "--out", sc], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0 and os.path.isfile(sc), p.stdout[-600:] + p.stderr[-1500:]
    d = json.load(open(sc, encoding="utf-8"))
    assert len(d["groups"]) == 6, sorted(d["groups"])
    for k, g in d["groups"].items():
        assert set(S.LABELS) == set(g["mean"]) and not g["missing"], f"{k}: labels {sorted(set(S.LABELS) - set(g['mean']))} missing {g['missing']}"
        assert all(g["n_valid"][l] > 0 for l in S.LABELS if not l.startswith("DK")) and all(g["n_valid"][l] == 0 for l in S.LABELS if l.startswith("DK")), k
        assert g["purpose"] == ["PLUMBING ONLY: this rollout is never read for H1 or H2"]
    c = d["contrasts"]["seed0"]
    assert set(c) >= {"X1", "X2", "BL"} and all(c[k]["status"] == "OK" and c[k]["n_episodes"] == 2 for k in ("X1", "X2", "BL")), c
    return {"groups": len(d["groups"]), "x1": c["X1"]}


# ------------------------------------------------------------------------------------------------ pytest
def test_positive():
    out = check_positive()
    os.makedirs(f"{T2}/logs", exist_ok=True)
    json.dump(out, open(f"{T2}/logs/rehearsal_stage1_positive.json", "w"), indent=1)


def test_planted_missing():
    check_planted_missing()


def test_changed_byte():
    check_changed_byte()


def test_actor_control():
    check_actor_control()


def test_roi_shifted():
    check_roi_shifted()


def test_empty_draws_lead():
    check_empty_draws_lead()


def test_score_e2e():
    check_score_e2e()
