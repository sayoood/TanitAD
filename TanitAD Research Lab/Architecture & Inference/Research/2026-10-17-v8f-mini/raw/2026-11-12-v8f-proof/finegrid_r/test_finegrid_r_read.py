"""Tests of finegrid_r_read.py and pod_finegrid_r_run.sh (card PREREG_V8F_PROOF_ADD43). numpy only, synthetic head outputs, tiny worlds (no torch, no real data). Evidence class: MEASURED (synthetic).

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python F:/TanitAD-scratch/v8proof/ops/capmem.py 1000 -m pytest -q -p no:cacheprovider test_finegrid_r_read.py --basetemp F:/TanitAD-scratch/v8proof/finegrid_r/work/pt
The reader under test is finegrid_r_read.py next to this file, or the file named by FINEGRID_R_READ_FILE (run_mutations_gr.py points it at a mutated copy).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RFILE = Path(os.environ.get("FINEGRID_R_READ_FILE", str(HERE / "finegrid_r_read.py")))
spec = importlib.util.spec_from_file_location("finegrid_r_read_under_test", str(RFILE))
R = importlib.util.module_from_spec(spec)
sys.modules["finegrid_r_read_under_test"] = R
spec.loader.exec_module(R)
FR, N, RP, O, S0F = R.FR, R.N, R.RP, R.O, R.S0F
SCRIPT = Path(os.environ.get("POD_R_SCRIPT", str(HERE / "pod_finegrid_r_run.sh")))


def make_ctx(n_ep=40, per=10, seed=0, B=400):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), per)
    n = len(ep)
    G15 = np.full((n, 15), np.nan)
    G15[:, S0F] = rng.uniform(10, 60, n)
    return {"valid": np.ones(n, bool), "c_lab": rng.normal(0, 2.0, n), "G15": G15, "ep10": ep, "n10": n_ep, "C10": RP.boot_counts(n_ep, B, O.SEED)}


@pytest.fixture(scope="module")
def ctx():
    return make_ctx()


def scenario(ctx, tk_sd, f_sd, tk_c=2.0, f_c=2.0, base=100):
    """TK16 and TK64 arms with independent gap noise of sd tk_sd, F16 arms with sd f_sd, for the eight seed indices."""
    y, c = ctx["G15"][:, S0F], ctx["c_lab"]
    outs = {}
    n = len(y)
    for t in R.TYPES:
        for k in R.SEEDS:
            r = np.random.default_rng(base + 1000 * R.TYPES.index(t) + k)
            sd, sc = (f_sd, f_c) if t == "F16" else (tk_sd, tk_c)
            outs[R.arm(t, k)] = {"gap_m": y + r.normal(0, sd, n), "closing_ms": c + r.normal(0, sc, n)}
    return outs


# ------------------------------------------------------------------------------------------------ the two-level estimator
def test_boot_diff_is_the_pilot_readers_paired_row_for_m2_mae_and_m1(ctx):
    o = scenario(ctx, 5.0, 2.0)
    a, b = o["F16_s2"], o["TK16_s2"]
    for measure, theirs in (("M2", FR.pair_r2(ctx, ctx["G15"][:, S0F], a["gap_m"], b["gap_m"])), ("MAE", FR.pair_mae(ctx, a["gap_m"], b["gap_m"])), ("M1", N.pair_m1(ctx, a["closing_ms"], b["closing_ms"]))):
        pt, vec = R.boot_diff(ctx, measure, a, b)
        assert pt == theirs["diff"] and RP.ci(vec) == theirs["ci95"], measure


def test_the_seed_level_is_resampled_the_interval_is_wider_than_episodes_alone(ctx):
    r = np.random.default_rng(1)
    pts = np.array([0.02, 0.10, 0.18, 0.06, 0.14, 0.00, 0.20, 0.08])
    D = pts[:, None] + r.normal(0, 0.003, size=(8, 400))
    tl = R.two_level(pts, D)
    ep_only = RP.ci(D.mean(0))
    assert tl["ci95"][1] - tl["ci95"][0] > 3 * (ep_only[1] - ep_only[0])
    assert tl["mean"] == pts.mean() and tl["n_pos"] == 7 and tl["per_seed"] == [float(x) for x in pts]
    assert R.two_level(pts, D) == tl                                                  # rng 0: the same call gives the same interval


def test_control_r3_planted_gain_every_seed_separated_two_of_eight_not_identical_exactly_zero(ctx):
    c = R.control_r3(ctx)
    assert c["pass"], c
    assert c["gain_at_every_seed"]["ci95"][0] > 0 and c["gain_at_every_seed"]["n_pos"] == 8
    assert not c["gain_at_two_of_eight"]["ci95"][0] > 0 and c["gain_at_two_of_eight"]["n_pos"] == 2
    assert c["identical_pairs"]["mean"] == 0.0 and c["identical_pairs"]["ci95"] == [0.0, 0.0]


def test_a_resample_of_episodes_with_seeds_held_fixed_would_separate_the_two_of_eight_case(ctx):
    """The statement the mutation of the list relies on: with the seed level switched off, 2 of 8 planted gains read as separated, so test_control_r3 above must fail on that mutation."""
    y = ctx["G15"][:, S0F]
    var = float(np.var(y))
    r = np.random.default_rng(5)
    pts, Ds = [], []
    for i in range(8):
        z = r.normal(size=len(y))
        b = {"gap_m": y + np.sqrt(0.5 * var) * z}
        a = {"gap_m": y + np.sqrt(0.4 * var) * z} if i < 2 else b
        p, d = R.boot_diff(ctx, "M2", a, b)
        pts.append(p)
        Ds.append(d)
    assert RP.ci(np.array(Ds).mean(0))[0] > 0


# ------------------------------------------------------------------------------------------------ names (card section 5)
def test_a_planted_better_f16_reads_gap_and_closing(ctx):
    oc = R.compute_outcomes_r(ctx, scenario(ctx, 5.0, 1.5, tk_c=2.0, f_c=0.8))
    n = oc["names"]
    assert n["gap"] == "FINER-GRID-READS-GAP-R" and n["mae_clause_met"] and n["closing"] == "FINER-GRID-READS-CLOSING-R", n
    assert oc["two_level"]["G16"]["M2"]["n_pos"] == 8 and oc["two_level"]["G16w"]["MAE"]["mean"] < 0
    assert set(oc["beside"]["std_over_seeds_of_M2_by_type"]) == {"TK16", "TK64", "F16"}


def test_equal_arms_read_flat_with_every_per_seed_difference_zero(ctx):
    one = scenario(ctx, 4.0, 4.0)["TK16_s2"]
    oc = R.compute_outcomes_r(ctx, {R.arm(t, k): dict(one) for t in R.TYPES for k in R.SEEDS})
    assert oc["names"]["gap"] == "FINER-GRID-FLAT-R" and oc["names"]["closing"].startswith("none")
    assert all(oc["two_level"][p]["M2"]["mean"] == 0.0 and oc["two_level"][p]["M2"]["ci95"] == [0.0, 0.0] for p in R.PAIRS)
    oc2 = R.compute_outcomes_r(ctx, scenario(ctx, 4.0, 4.0))
    assert oc2["names"]["gap"] == "FINER-GRID-FLAT-R"


def test_a_planted_worse_f16_reads_worse(ctx):
    oc = R.compute_outcomes_r(ctx, scenario(ctx, 4.0, 9.0))
    assert oc["names"]["gap"] == "FINER-GRID-WORSE-R" and oc["names"]["closing"].startswith("none")
    assert all(oc["two_level"][p]["M2"]["ci95"][1] < 0 for p in R.PAIRS)


def test_the_mae_clause_and_the_six_of_eight_clause_each_withhold_the_gap_name(ctx):
    tl = {p: {m: {"mean": 0.05, "ci95": [0.01, 0.09], "n_pos": 8, "per_seed": [0.05] * 8, "n_seeds": 8} for m in ("M2", "M1", "MAE")} for p in R.PAIRS}
    ok = R.decide_r(tl, {"TK16": 10.0, "TK64": 9.0, "F16": 8.0})
    assert ok["gap"] == "FINER-GRID-READS-GAP-R"                                       # 8.0 <= 0.9 * 9.0 = 8.1
    assert R.decide_r(tl, {"TK16": 10.0, "TK64": 9.0, "F16": 8.2})["gap"] == "FINER-GRID-FLAT-R"
    tl5 = {p: {m: dict(v) for m, v in d.items()} for p, d in tl.items()}
    tl5["G16w"]["M2"]["n_pos"] = 5
    assert R.decide_r(tl5, {"TK16": 10.0, "TK64": 9.0, "F16": 8.0})["gap"] == "FINER-GRID-FLAT-R"
    tl6 = {p: {m: dict(v) for m, v in d.items()} for p, d in tl.items()}
    tl6["G16"]["M2"]["n_pos"] = 6
    assert R.decide_r(tl6, {"TK16": 10.0, "TK64": 9.0, "F16": 8.0})["gap"] == "FINER-GRID-READS-GAP-R"
    tl0 = {p: {m: dict(v) for m, v in d.items()} for p, d in tl.items()}
    tl0["G16"]["M2"]["ci95"] = [0.0, 0.09]                                             # an interval that touches 0 does not separate
    assert R.decide_r(tl0, {"TK16": 10.0, "TK64": 9.0, "F16": 8.0})["gap"] == "FINER-GRID-FLAT-R"


def test_the_pairing_is_the_same_seed_index_and_f_minus_tk(ctx):
    o = scenario(ctx, 5.0, 2.0)
    t = R.two_level_pair(ctx, o, "G16w", "M2")
    for i, k in enumerate(R.SEEDS):
        want, _ = R.boot_diff(ctx, "M2", o[R.arm("F16", k)], o[R.arm("TK64", k)])
        assert t["per_seed"][i] == want
    assert t["mean"] > 0                                                                # F16 is the better arm in this scenario


# ------------------------------------------------------------------------------------------------ controls
def _wlog(path, arm, total, bce, gap, clos):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": True, "arm": arm, "total": total}) + "\n")
        for i in range(len(bce)):
            step = 50 * (i + 1)
            fh.write(json.dumps({"step": step, "first": False, "lead_sup_mean": bce[i] + gap[i] + clos[i], "lead_sup_bce_mean": bce[i], "lead_sup_gap_mean": gap[i], "lead_sup_clos_mean": clos[i],
                                 "n_gap": 10.0, "n_gap_min_interval": 10.0}) + "\n")


def test_p5_for_a_permuted_label_arm_reads_the_intact_parts_and_reports_the_permuted_part(tmp_path):
    n = 60
    down, up, flat = np.linspace(1.0, 0.5, n), np.linspace(0.5, 1.0, n), np.full(n, 0.7)
    _wlog(tmp_path / "a.jsonl", "TK16SH_s2", 3000, down, down, np.linspace(0.5, 4.0, n))  # lead term as a whole rises (closing part), intact parts fall
    d = R.p5_form(str(tmp_path / "a.jsonl"))
    assert d["pass"] and d["intact_last_mean"] < d["intact_first_mean"] and d["permuted_closing_part_last_mean"] > d["permuted_closing_part_first_mean"]
    _wlog(tmp_path / "b.jsonl", "TK16SH_s2", 3000, up, up, down)
    assert not R.p5_form(str(tmp_path / "b.jsonl"))["pass"]
    _wlog(tmp_path / "c.jsonl", "TK16_s2", 3000, down, down, up)                         # an ordinary arm: the whole lead term of card ADD27
    c = R.p5_form(str(tmp_path / "c.jsonl"))
    assert c["form"].startswith("card ADD27") and c["last_mean"] < c["first_mean"]
    _wlog(tmp_path / "d.jsonl", "F16_s2", 3000, down, down, down)
    assert R.p5_form(str(tmp_path / "d.jsonl"))["pass"]
    _wlog(tmp_path / "e.jsonl", "F16SH_s2", 3000, down[:40], down[:40], up[:40])        # incomplete run
    assert not R.p5_form(str(tmp_path / "e.jsonl"))["pass"]


def test_r1_compare_equal_sequences_pass_a_different_one_and_a_missing_file_fail(tmp_path):
    def w(name, seq, chunk):
        p = tmp_path / name
        p.write_text(json.dumps({"n_batches": len(seq), "idx_hashes": seq, "full_hashes": [s + "f" for s in seq], "sampler_chunk": chunk}), encoding="utf-8")
        return str(p)
    a, b, c = w("a.json", ["x1", "x2", "x3"], 41), w("b.json", ["x1", "x2", "x3"], 41), w("c.json", ["x1", "x2", "y3"], 48)
    assert R.r1_compare(a, b)["pass"] and not R.r1_compare(a, c)["pass"] and not R.r1_compare(a, str(tmp_path / "none.json"))["pass"]
    assert not R.r1_compare(a, w("d.json", ["x1", "x2"], 41))["pass"]


def test_records_g5_and_r2(tmp_path):
    def rec(name, **kw):
        base = {"grid": "pool4x10" if name.startswith("TK") else "pool8x20", "n_tok": 40 if name.startswith("TK") else 160, "n_tok_seen_in_training": 40 if name.startswith("TK") else 160, "W": R.width(name),
                "input_norm": "ln", "step": 19164, "arm": name}
        base["D"] = base["n_tok"] * base["W"]
        base.update(kw)
        (tmp_path / (name + "_eval.npz.json")).write_text(json.dumps(base), encoding="utf-8")
        return {"eval_states": str(tmp_path / (name + "_eval.npz"))}
    for nm in ("TK16_s2", "TK64_s9", "F16_s5"):
        assert R.check_records_r(rec(nm), nm, 19164)["pass"], nm
    assert not R.check_records_r(rec("F16_s5", n_tok_seen_in_training=40), "F16_s5", 19164)["pass"]    # declared against built
    assert not R.check_records_r(rec("TK64_s3", grid="pool8x20"), "TK64_s3", 19164)["pass"]
    assert not R.check_records_r(rec("TK16_s2", step=18000), "TK16_s2", 19164)["pass"]
    assert R.check_records_r(rec("TK16SH_s2", step=2000), "TK16SH_s2", 2000)["pass"]
    assert not R.check_records_r({"eval_states": str(tmp_path / "none.npz")}, "TK16_s2", 19164)["pass"]


def test_the_tool_sha_of_the_devbox_controls_must_equal_the_pod_marker(tmp_path):
    m = tmp_path / "DONE.txt"
    m.write_text("FINEGRID_R_DONE x\ntok_probe_gr.py sha256: abc123\n", encoding="utf-8")
    assert R.tool_sha_control({"notes": {"code_sha256": {"tok_probe_gr.py": "abc123"}}}, str(m))["pass"]
    assert not R.tool_sha_control({"notes": {"code_sha256": {"tok_probe_gr.py": "zzz"}}}, str(m))["pass"]
    assert not R.tool_sha_control({}, str(m))["pass"] and not R.tool_sha_control({"notes": {"code_sha256": {"tok_probe_gr.py": "abc123"}}}, str(tmp_path / "no.txt"))["pass"]


def test_a_failed_control_withholds_names_and_every_outcome_difference(tmp_path):
    res = {"tier": R.TIER_TEXT, "families": R.FAMILY_TEXT, "B": 2000, "evidence_class": "MEASURED once run", "controls_flat": {"r1_same_windows_same_order": False, "g2_grids_belong_together": True},
           "names": None, "outcome_differences": "withheld: a control did not pass (card section 4): r1_same_windows_same_order",
           "arms": {"F16_s2": {"M1": {"r2": 0.1}, "M2": {"r2": 0.5}, "M4": {"auroc": 0.9}}}}
    R.write_verdict(str(tmp_path / "v.md"), res)
    t = (tmp_path / "v.md").read_text(encoding="utf-8")
    assert "No name is given" in t and "withheld" in t and "FINER-GRID" not in t and "G16" not in t and "STRATEGIC: NOT COMPUTED (a token probe reads no strategic quantity; n = 0)" in t and "Tier T0" in t


def test_seeds_and_arm_names():
    assert R.SEEDS == tuple(range(2, 10)) and len(R.main_arms()) == 24 and R.main_arms()[0] == "TK16_s2" and "F16_s9" in R.main_arms() and "F64_s2" not in R.main_arms()
    assert R.width("TK64_s3") == 64 and R.width("F16SH_s2") == 16 and R.family("F16_s2") == "F" and R.family("TK16SH_s2") == "TK"


# ------------------------------------------------------------------------------------------------ the pod script (stub tool, no GPU, no data)
STUB = r'''
import json, os, sys
a = sys.argv[1:]
open(os.environ["CALLS"], "a").write(" ".join(a) + "\n")
def opt(n):
    return a[a.index(n) + 1] if n in a else None
if os.environ.get("FAIL_ON") and os.environ["FAIL_ON"] in " ".join(a):
    sys.exit(7)
c = a[0]
if c == "g2":
    json.dump({"pass": True}, open(opt("--out"), "w"))
elif c == "train":
    out = opt("--out")
    for n in opt("--arms").split(","):
        os.makedirs(os.path.join(out, n), exist_ok=True)
        open(os.path.join(out, n, "ckpt_last.pt"), "w").write(n)
    json.dump({"done": True}, open(os.path.join(out, "TRAIN_DONE.json"), "w"))
    json.dump({"n_batches": 1, "idx_hashes": ["a"], "full_hashes": ["b"]}, open(os.path.join(out, "r1_batches.json"), "w"))
elif c == "extract":
    os.makedirs(opt("--states-dir"), exist_ok=True)
    open(os.path.join(opt("--states-dir"), opt("--arm") + "_eval.npz"), "w").write("x")
elif c == "r1":
    json.dump({"pass": True}, open(opt("--out"), "w"))
'''


def _fake_gpu_bin(tmp_path, n_lines):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    body = "#!/usr/bin/env bash\ncase \"$*\" in\n  *compute-apps*) for i in $(seq 1 %d); do echo $((1000+i)); done;;\n  *) echo 'FakeGPU, 0 MiB, 1000 MiB';;\nesac\nexit 0\n" % n_lines
    (b / "nvidia-smi").write_bytes(body.encode("utf-8"))
    return b


def _fake_cg(tmp_path, free_gb, name="cg"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    maxb = 100 * 2 ** 30
    (d / "memory.max").write_text(str(maxb), encoding="utf-8")
    (d / "memory.stat").write_text("file 1\nanon %d\nkernel 3\n" % (maxb - free_gb * 2 ** 30), encoding="utf-8")
    return d


def _sh(env, timeout=120):
    bash = shutil.which("bash")
    e = dict(os.environ)
    e.update(env)
    return subprocess.run([bash, str(SCRIPT)], env=e, capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def _stub_env(tmp_path, name="o", gpu_jobs=1, free=60, **extra):
    pb = tmp_path / ("pb_" + name)
    pb.mkdir()
    (pb / "tok_probe_gr.py").write_text(STUB, encoding="utf-8")
    tree = tmp_path / "tree"
    (tree / "stack").mkdir(parents=True, exist_ok=True)
    cache = tmp_path / "cache"
    (cache / "train" / "clips").mkdir(parents=True, exist_ok=True)
    (cache / "eval" / "clips").mkdir(parents=True, exist_ok=True)
    cg = _fake_cg(tmp_path, free, "cg_" + name)
    calls = tmp_path / ("calls_" + name + ".txt")
    env = {"OUT": str(tmp_path / ("out_" + name)), "PB": str(pb), "TREE": str(tree), "CACHE": str(cache), "STATES": str(tmp_path / ("st_" + name)), "PY": sys.executable, "CALLS": str(calls),
           "PATH": str(_fake_gpu_bin(tmp_path, gpu_jobs)) + os.pathsep + os.environ["PATH"], "CG_MAX": str(cg / "memory.max"), "CG_STAT": str(cg / "memory.stat"), "PROBE_GAP_S": "0", "POLL_S": "1", "MAX_WAIT_S": "3"}
    env.update(extra)
    return env, calls


def test_pod_script_dry_run_prints_the_32_commands_with_the_right_flags():
    r = _sh({"DRY": "1"})
    assert r.returncode == 0 and r.stdout.count("DRY: ") == 32, r.stdout[-1500:]
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY: ")]
    trains = [ln for ln in lines if " train " in ln]
    assert len(trains) == 4 and sum(" extract " in ln for ln in lines) == 26 and sum(" r1 " in ln for ln in lines) == 1 and sum(" g2 " in ln for ln in lines) == 1
    a, b, p3a, p3b = trains
    assert "--grid pool4x10" in a and "--sampler-chunk-like pool8x20" in a and "--arms TK16_s2,TK64_s2,TK16_s3,TK64_s3" in a and "TK64_s9" in a and "--expect-steps 19164" in a and "--input-norm ln" in a
    assert "--grid pool8x20" in b and "sampler-chunk-like" not in b and "--arms F16_s2,F16_s3" in b and b.count("F16_s") == 8
    assert "--shuffle-closing-labels" in p3a and "--arms TK16SH_s2" in p3a and "--steps 2000" in p3a and "--sampler-chunk-like pool8x20" in p3a
    assert "--shuffle-closing-labels" in p3b and "--arms F16SH_s2" in p3b and "--steps 2000" in p3b and "sampler-chunk-like" not in p3b
    assert all("--splits eval" in ln and "--eval-range 0:141" in ln for ln in lines if " extract " in ln) and not any("train-range" in ln for ln in lines)
    assert "/workspace/v8proof/runs/finegrid_r" in r.stdout and "/workspace/v8proof/finegrid_r/states" in r.stdout and "/workspace/v8proof/finegrid/tok_probe_gr.py" in r.stdout


def test_pod_script_source_rules():
    src = SCRIPT.read_text(encoding="utf-8")
    for needle in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.stat", '$1=="anon"', "MIN_FREE_GB:-28", "MAX_GPU_JOBS:-3", "PROBE_GAP_S:-45", "POLL_S:-60", "MAX_WAIT_S:-36000", "set -o noclobber",
                   "FINEGRID_R_DONE.txt", "FINEGRID_R_STOP.txt", "ZZ FINEGRID-R DONE", "ZZ FINEGRID-R STOP:", "--query-compute-apps=pid --format=csv,noheader", "grep -c ."):
        assert needle in src, needle
    for banned in ("pgrep", "pkill", "ps -", "ps aux", "grep -v grep", "mv ", "rmdir"):
        assert banned not in src, banned
    assert not re.search(r"(^|[;&|(]|\s)rm\s+-?\w*\s", src), "no rm command"


def test_pod_script_runs_one_gpu_step_at_a_time_with_a_slot_wait_before_each_and_writes_the_markers(tmp_path):
    env, calls = _stub_env(tmp_path)
    r = _sh(env)
    out = Path(env["OUT"])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    assert (out / "FINEGRID_R_DONE.txt").exists() and not (out / "FINEGRID_R_STOP.txt").exists()
    log = (out / "finegrid_r.log").read_text(encoding="utf-8")
    assert log.strip().splitlines()[-1].endswith("ZZ FINEGRID-R DONE")
    assert log.count("slot free on two probes") == 30                                    # 4 train steps + 26 extracts, each behind its own slot wait
    seq = [ln.split()[0] for ln in calls.read_text(encoding="utf-8").splitlines()]
    assert seq[0] == "g2" and seq.count("extract") == 26 and seq.count("train") == 4 and seq[-1] == "extract" and seq.count("r1") == 1
    runs = [ln for ln in log.splitlines() if "finegrid_r: run: " in ln]
    waits = [i for i, ln in enumerate(log.splitlines()) if "slot free on two probes" in ln]
    assert len(runs) == 32 and len(waits) == 30
    done = (out / "FINEGRID_R_DONE.txt").read_text(encoding="utf-8")
    assert "tok_probe_gr.py sha256:" in done and done.count("ckpt ") == 26
    assert (Path(env["STATES"]) / "F16_s9_eval.npz").exists() and (Path(env["STATES"]) / "TK64_s2_eval.npz").exists()
    r2 = _sh(env)                                                                         # a finished marker refuses a second run
    assert r2.returncode == 3 and "REFUSED" in r2.stdout


def test_pod_script_second_start_is_refused_by_the_noclobber_marker_and_a_failed_step_stops_with_a_marker(tmp_path):
    env, calls = _stub_env(tmp_path, "b")
    out = Path(env["OUT"])
    out.mkdir()
    (out / "FINEGRID_R_STARTED.txt").write_text("earlier start\n", encoding="utf-8")
    r = _sh(env)
    assert r.returncode == 3 and "second start" in r.stdout and not calls.exists()
    assert (out / "FINEGRID_R_STARTED.txt").read_text(encoding="utf-8") == "earlier start\n"
    env2, calls2 = _stub_env(tmp_path, "c", FAIL_ON="extract --grid pool8x20")
    r = _sh(env2)
    assert r.returncode == 1
    out2 = Path(env2["OUT"])
    assert "extract F16_s2 failed" in (out2 / "FINEGRID_R_STOP.txt").read_text(encoding="utf-8") and not (out2 / "FINEGRID_R_DONE.txt").exists()
    assert (out2 / "finegrid_r.log").read_text(encoding="utf-8").strip().splitlines()[-1].startswith("[20") and "ZZ FINEGRID-R STOP: extract F16_s2 failed" in (out2 / "finegrid_r.log").read_text(encoding="utf-8")


def test_pod_script_waits_on_three_gpu_jobs_and_on_low_memory_and_stops_without_starting(tmp_path):
    env, calls = _stub_env(tmp_path, "d", gpu_jobs=3)
    r = _sh(env)
    out = Path(env["OUT"])
    assert r.returncode == 1 and "fourth GPU job" in (out / "FINEGRID_R_STOP.txt").read_text(encoding="utf-8")
    assert calls.read_text(encoding="utf-8").strip().splitlines() == [l for l in calls.read_text(encoding="utf-8").strip().splitlines() if l.startswith("g2")]   # only g2 (CPU) ran
    env2, calls2 = _stub_env(tmp_path, "e", gpu_jobs=0, free=20)
    r = _sh(env2)
    assert r.returncode == 1 and "free=20 GB" in (Path(env2["OUT"]) / "finegrid_r.log").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ files
BANNED = ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"]
OWN = ("tok_probe_gr.py", "make_tok_probe_gr.py", "finegrid_r_read.py", "test_finegrid_r_read.py", "run_checks_pod.py", "run_mutations_gr.py", "pod_finegrid_r_run.sh", "run_pod_read_r.sh", "HANDOFF.md",
       "FINEGRID_R_DEFS.md", "controls_devbox_r.json")


def test_own_files_have_lf_line_endings_and_none_of_the_banned_words():
    for name in OWN:
        p = HERE / name
        if not p.exists():
            continue
        raw = p.read_bytes()
        assert b"\r" not in raw, name
        low = raw.decode("utf-8").lower()
        for w in BANNED:
            assert w not in low, (name, w)
