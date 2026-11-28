"""Tests of read_dscale2x.py and podread_dscale2x.sh (card PREREG_V8F_PROOF_ADD123). Synthetic data only: tiny worlds, no real state or result file is opened.
Evidence class: MEASURED (synthetic).

What is synthetic and what is injected: the files the controls read (done files, state records, batch logs, train logs, precheck records, the evaluation-id arrays of the state files, the pytest output
file) are written to disk in the formats the banked reader loads. The per-probe read of the banked chain (tokprobe_read.read_arm: head tensors and window states) is replaced by `read_probe`, an
injectable function of the reader that returns the same (R, out, ctrl) triple from synthetic gap outputs; the pack/context build is replaced the same way through `build_ctx`. The banked reader itself is
imported unchanged (it imports on this box).

Run: PYCAP_MB=3000 bash F:/TanitAD-scratch/v8proof/ops/pycap.sh -m pytest test_read_dscale2x.py -q -p no:cacheprovider --basetemp <new folder>
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pytest
import read_dscale2x as R2

D, N, RP, S0F = R2.D, R2.N, R2.RP, R2.S0F
SEEDS = R2.SEEDS
SCRIPT = HERE / "podread_dscale2x.sh"
VENV_PY = "/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
B = 400


def must_fail(fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError("a planted wrong statement was accepted: the check is blind")


def _assert(x):
    assert x


# ------------------------------------------------------------------------------------------------ the synthetic world
def make_ctx(n_ep=40, per=10, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), per)
    n = len(ep)
    G15 = np.full((n, 15), np.nan)
    G15[:, S0F] = rng.uniform(10, 60, n)
    return {"valid": np.ones(n, bool), "c_lab": rng.normal(0, 2.0, n), "G15": G15, "ep10": ep, "n10": n_ep, "C10": RP.boot_counts(n_ep, B, D.O.SEED),
            "cnt": {"scored": N.EXPECT["scored"], "windows": N.EXPECT["windows"]}}


@pytest.fixture(scope="module")
def ctx():
    return make_ctx()


def noise(ctx, label, sd, salt=0):
    r = np.random.default_rng(zlib.crc32((label + str(salt)).encode()))
    return ctx["G15"][:, S0F] + r.normal(0, sd, len(ctx["G15"]))


def make_preds(ctx, variant):
    """label -> gap_m for every probe of the probe table. variant: gap | short | flat | worse."""
    y = ctx["G15"][:, S0F]
    specs = R2.probe_specs()
    P = {}
    sd_c = {"gap": 6.0, "short": 6.0, "flat": 6.0, "worse": 3.0}[variant]
    for label, sp in specs.items():
        if sp["sh"]:
            P[label] = np.random.default_rng(zlib.crc32(label.encode())).uniform(10, 60, len(y))
        elif label.startswith("P16C_") or label.startswith("P16U_"):
            P[label] = noise(ctx, label, sd_c)
    for k in SEEDS:
        c = P["P16C_s%d" % k]
        a, b, z = R2.lab_q(k, 19164), R2.lab_q(k, 28746), R2.lab_q(k, 38328)
        if variant == "gap":
            P[a], P[b], P[z] = noise(ctx, a, 5.0), noise(ctx, b, 2.5), noise(ctx, z, 2.0)
        elif variant == "short":
            P[a], P[b], P[z] = noise(ctx, a, 12.0), noise(ctx, b, 10.0), c.copy()
        elif variant == "flat":
            P[a], P[b], P[z] = c.copy(), c.copy(), c.copy()
        else:
            P[a], P[b], P[z] = noise(ctx, a, 8.0), noise(ctx, b, 8.0), noise(ctx, b, 8.0)
    P["P16U2_s0"] = P[R2.lab_q(0, 38328)].copy()
    return P


def fake_reader(world):
    ctx, preds, auroc = world["ctx"], world["preds"], world["auroc"]
    y = ctx["G15"][:, S0F]

    def read_probe(label, spec, files, c, a, cache):
        p = preds[label]
        head = N.fid_one(y, p, ctx["valid"], ctx["ep10"], ctx["n10"], ctx["C10"])
        return {"read": True, "M2": {"head": head}, "M4": {"auroc": auroc.get(label, 0.95)}}, {"gap_m": p}, {}
    return read_probe


def wjson(p, o):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o), encoding="utf-8")


def wlog(path, total, kept=None, n_gap=200.0, stop_at=None, step1_bce=0.70, step1_gap=0.06, first_extra=None):
    """A train log in the format of the banked reader: a header, the step-1 record (first True), then body records that fall."""
    hdr = {"header": True, "total": total, "w_clos": 0.0, "eval_guard": {"n_shared": 0, "n_eval_ids": 48}}
    if kept is not None:
        hdr["clip_list"] = {"n_kept": kept}
    first = {"step": 1, "first": True, "lead_sup_bce": step1_bce, "lead_sup_gap": step1_gap, "lead_sup_bce_mean": step1_bce, "lead_sup_gap_mean": step1_gap, "lead_sup_clos_mean": 0.0,
             "n_pres": 670.0, "n_gap": 200.0, "n_clos": 0.0}
    first.update(first_extra or {})
    steps = list(range(250, total, 250)) + [total]
    if stop_at is not None:
        steps = [s for s in steps if s < stop_at]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(hdr) + "\n" + json.dumps(first) + "\n")
        for i, s in enumerate(steps):
            lvl = 0.8 - 0.5 * s / total
            f.write(json.dumps({"step": s, "first": False, "lead_sup_bce_mean": 0.9 * lvl, "lead_sup_gap_mean": 0.1 * lvl, "lead_sup_clos_mean": 0.0, "n_pres": 670.0 + (i % 7),
                                "n_gap": n_gap + (i % 11), "n_clos": 0.0}) + "\n")


def good_precheck():
    grp = lambda: {"clips": 200, "clips_absent_from_join": 0, "frames": 1200, "presence_frames": 700, "gap_label_frames": 650, "clips_with_a_gap_label": 190, "pass": True}
    return {"pass": True, "checks": {"counts": {"pass": True}, "d1_no_leak": {"pass": True, "planted": False}, "d2_labels_one_clock": {"pass": True}, "d2j_join_matches_lead_file": {"pass": True},
                                     "d2n_join_matches_lead_file_new_clips": {"pass": True, "sampled": 200, "rows_mismatching": 0},
                                     "d2g_join_labels_per_group": {"pass": True, "corpus": grp(), "new": grp()}}}


def good_clock(plant=0):
    grp = lambda: {"pass": plant == 0, "argmin_shift": -plant, "ratio_shift0_to_nearest": 0.04 if plant == 0 else 22.0, "clips_read": 200, "mean_abs_diff_mps": {"0": 0.005, "1": 0.1, "-1": 0.1}}
    return {"pass": plant == 0, "plant_shift": plant, "n_clips": 200, "groups": {"corpus": grp(), "new": grp()}}


def good_evalset(planted=False):
    return {"pass": not planted, "d2e_eval_set_is_the_study_set": {"pass": not planted, "n_here": 141, "n_study": 141, "same_ids_same_order": not planted, "planted_swap": planted},
            "d3t_tokens_same": {"gates": False}}


EVAL_IDS = ["clip%03d" % i for i in range(30)]


def write_state(in_dir, folder, arm, step, sha, ids=EVAL_IDS):
    base = Path(in_dir) / folder / (arm + "_eval.npz")
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(base), clips=np.array(ids))
    wjson(str(base) + ".json", {"grid": "pool4x10", "n_tok": 40, "n_tok_seen_in_training": 40, "D": 640, "W": 16, "input_norm": "ln", "step": step, "arm": arm, "ckpt_sha256": sha})


def build_world(tmp, variant="gap"):
    ctx = make_ctx()
    w = {"ctx": ctx, "preds": make_preds(ctx, variant), "auroc": {}, "tmp": Path(tmp)}
    root = Path(tmp)
    br, bi, nr, ni = root / "banked_out", root / "banked_out" / "states", root / "new_out", root / "new_out" / "states"
    for d in (br, bi, nr, ni):
        d.mkdir(parents=True, exist_ok=True)
    banked_lines = []
    for label, (folder, arm, steps, sh, corpus) in D.ARM_MAP.items():
        sha = "b" + format(zlib.crc32((folder + arm).encode()), "08x") * 7
        write_state(bi, folder, arm, steps, sha)
        banked_lines.append("ckpt %s/%s %s" % (folder, arm, sha))
        wlog(br / folder / arm / "train_log.jsonl", steps, kept=4512 if corpus else None)
        if sh:
            hdr_path = br / folder / arm / "train_log.jsonl"
            lines = hdr_path.read_text(encoding="utf-8").splitlines()
            h = json.loads(lines[0])
            h["shuffle_gap"] = {"seed": 1}
            hdr_path.write_text("\n".join([json.dumps(h)] + lines[1:]) + "\n", encoding="utf-8")
    (br / "DSCALE_LEAD_DONE.txt").write_text("DSCALE_LEAD_DONE 2026-11-20T00:00:00Z\n" + "\n".join(banked_lines) + "\n", encoding="utf-8")
    wjson(br / "runU2" / "r1_batches.json", {"sampler_chunk": 64, "hashes": {"1": "aa", "2": "bb", "3": "cc"}})
    wjson(nr / "runQ" / "r1_batches.json", {"sampler_chunk": 64, "hashes": {"1": "aa", "2": "bb", "3": "cc"}})
    done = ["DSCALE_2X_DONE 2026-11-23T00:00:00Z"]
    folders = {19164: "runQ_s19164", 28746: "runQ_s28746", 38328: "runQ"}
    for k in SEEDS:
        arm = "TK16_s%d" % k
        for st, folder in folders.items():
            sha = "q%d%08x" % (k, zlib.crc32(("%s%d" % (arm, st)).encode())) * 3
            write_state(ni, folder, arm, st, sha)
            done.append("ckpt_step runQ/%s %d %s" % (arm, st, sha))
            if st == 38328:
                done.append("ckpt runQ/%s %s" % (arm, sha))
        wlog(nr / "runQ" / arm / "train_log.jsonl", 38328)
    (nr / "DSCALE_2X_DONE.txt").write_text("\n".join(done) + "\n", encoding="utf-8")
    wjson(nr / "runQ" / "CK_RECORD.json", {"ck_steps": [19164, 28746], "files": {}, "tok_probe_u_sha256": "x", "tool_sha256": "y"})
    wjson(nr / "runQ" / "TRAIN_DONE.json", {"done": True})
    pre = nr / "precheck"
    wjson(pre / "PRECHECK.json", good_precheck())
    wjson(pre / "PRECHECK_PLANTED.json", {"pass": False, "checks": {"d1_no_leak": {"pass": False, "planted": True}}})
    wjson(pre / "CLOCK.json", good_clock())
    wjson(pre / "CLOCK_PLANTED.json", good_clock(1))
    wjson(pre / "EVALSET.json", good_evalset())
    wjson(pre / "EVALSET_PLANTED.json", good_evalset(True))
    tout = root / "helper_pytest.out"
    tout.write_text("...........\n11 passed in 3.21s\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(tout, (old, old))
    w.update({"banked_run": br, "banked_in": bi, "new_run": nr, "new_in": ni, "test_out": tout, "out": nr / "read"})
    return w


def argv(w, mode, out=None, before=None):
    return [mode, "--banked-run-dir", str(w["banked_run"]), "--banked-in-dir", str(w["banked_in"]), "--new-run-dir", str(w["new_run"]), "--new-in-dir", str(w["new_in"]),
            "--banked-done", str(w["banked_run"] / "DSCALE_LEAD_DONE.txt"), "--new-done", str(w["new_run"] / "DSCALE_2X_DONE.txt"), "--pack-dir", str(w["tmp"]),
            "--out-dir", str(out or w["out"]), "--test-out", str(w["test_out"]), "--test-before", str(before or time.time()), "--B", str(B)]


@pytest.fixture
def run(monkeypatch, capsys):
    def go(w, mode, **kw):
        monkeypatch.setattr(R2, "build_ctx", lambda a: w["ctx"])
        monkeypatch.setattr(R2, "read_probe", fake_reader(w))
        capsys.readouterr()
        rc = R2.main(argv(w, mode, **kw))
        return rc, capsys.readouterr().out
    return go


def controls_json(w):
    return json.loads(Path(R2.latest_path(str(w["out"]), R2.CONTROLS_FILE)).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ tests
def test_probe_table_is_the_card_map_and_the_banked_map():
    sp = R2.probe_specs()
    for label, v in D.ARM_MAP.items():
        assert (sp[label]["folder"], sp[label]["arm"], sp[label]["steps"], sp[label]["sh"]) == v[:4]
    assert sp["P16Q_s2@28746"]["folder"] == "runQ_s28746" and sp["P16Q_s3@38328"]["folder"] == "runQ" and sp["P16Q_s0@19164"]["folder"] == "runQ_s19164"
    assert len(sp) == 11 + 12
    must_fail(lambda: _assert(sp["P16Q_s0@38328"]["folder"] == "runQ_s28746"))


def test_a_pair_with_itself_reads_exactly_zero_and_a_swap_reads_the_negative(ctx):
    o = {"gap_m": noise(ctx, "x", 5.0)}
    o2 = {"gap_m": noise(ctx, "y", 3.0)}
    for m in ("M2", "MAE"):
        p, d = D.boot_diff(ctx, m, o, o)
        assert p == 0.0 and np.all(d == 0.0)
    pts, Ds = [], []
    for _ in SEEDS:
        p, d = D.boot_diff(ctx, "M2", o, o)
        pts.append(p)
        Ds.append(d)
    t = D.two_level(pts, np.array(Ds), 0)
    assert t["mean"] == 0.0 and t["ci95"] == [0.0, 0.0]
    a, b = D.boot_diff(ctx, "M2", o2, o)[0], D.boot_diff(ctx, "M2", o, o2)[0]
    assert a == -b and a != 0.0
    must_fail(lambda: _assert(D.boot_diff(ctx, "M2", o2, o)[0] == 0.0))


def test_clean_world_passes_every_control_and_prints_one_line_each(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "controls")
    assert rc == 0, out
    lines = out.strip().splitlines()
    assert lines[:7] == ["ctl q%d PASS" % i for i in range(7)] and lines[-1] == "ZZ DSCALE2X CONTROLS ALL-PASS" and len(lines) == 8
    cj = controls_json(w)
    assert cj["all_pass"] is True and set(cj["controls"]) == set(R2.CONTROL_NAMES) and all(isinstance(v["pass"], bool) for v in cj["controls"].values())
    assert "beside" in cj["controls"]["q4"] and set(cj["controls"]["q4"]["beside"]) == {"TK16_s%d" % k for k in SEEDS}


def _mut_q0(w):
    p = w["banked_run"] / "DSCALE_LEAD_DONE.txt"
    p.write_text(p.read_text(encoding="utf-8").replace("ckpt runC/TK16_s1 b", "ckpt runC/TK16_s1 c"), encoding="utf-8")


def _mut_q0_precheck(w):
    p = w["new_run"] / "precheck" / "CLOCK_PLANTED.json"
    wjson(p, good_clock(0))


def _mut_q1_batches(w):
    wjson(w["new_run"] / "runQ" / "r1_batches.json", {"sampler_chunk": 64, "hashes": {"1": "aa", "2": "bb", "3": "XX"}})


def _mut_q1_first_record(w):
    wlog(w["new_run"] / "runQ" / "TK16_s0" / "train_log.jsonl", 38328, step1_bce=0.7002)


def _mut_q2(w):
    w["preds"]["P16Q_s1@38328"] = np.random.default_rng(5).uniform(10, 60, len(w["ctx"]["G15"]))


def _mut_q2_auroc(w):
    w["auroc"]["P16C_s2"] = 0.60


def _mut_q3(w):
    w["preds"]["P16USH_s0"] = noise(w["ctx"], "live", 1.0)


def _mut_q3_auroc(w):
    w["auroc"]["P16CSH_s0"] = 0.55


def _mut_q4_log_short(w):
    wlog(w["new_run"] / "runQ" / "TK16_s2" / "train_log.jsonl", 38328, stop_at=30000)


def _mut_q4_full_batch_no_gap(w):
    p = w["new_run"] / "runQ" / "TK16_s3" / "train_log.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    r = json.loads(lines[40])
    r["n_gap"] = 0.0
    lines[40] = json.dumps(r)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_q4_state_step(w):
    p = w["new_in"] / "runQ_s28746" / "TK16_s1_eval.npz.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["step"] = 28000
    wjson(p, r)


def _mut_q4_hash(w):
    p = w["new_in"] / "runQ_s19164" / "TK16_s0_eval.npz.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["ckpt_sha256"] = "00" + r["ckpt_sha256"][2:]
    wjson(p, r)


def _mut_q4_header(w):
    p = w["new_run"] / "runQ" / "TK16_s0" / "train_log.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    h = json.loads(lines[0])
    h["total"] = 19164
    p.write_text("\n".join([json.dumps(h)] + lines[1:]) + "\n", encoding="utf-8")


def _mut_q5_tests_fail(w):
    w["test_out"].write_text("....F....\n1 failed, 10 passed in 3.2s\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(w["test_out"], (old, old))


def _mut_q5_tests_newer(w):
    os.utime(w["test_out"], (time.time() + 1000, time.time() + 1000))


def _mut_q6(w):
    ids = list(EVAL_IDS)
    ids[3], ids[4] = ids[4], ids[3]
    write_state(w["banked_in"], "runU", "TK16_s2", 19164, "b" + format(zlib.crc32(("runU" + "TK16_s2").encode()), "08x") * 7, ids=ids)


MUTATIONS = [("q0", _mut_q0), ("q0", _mut_q0_precheck), ("q1", _mut_q1_batches), ("q1", _mut_q1_first_record), ("q2", _mut_q2), ("q2", _mut_q2_auroc), ("q3", _mut_q3), ("q3", _mut_q3_auroc),
             ("q4", _mut_q4_log_short), ("q4", _mut_q4_full_batch_no_gap), ("q4", _mut_q4_state_step), ("q4", _mut_q4_hash), ("q4", _mut_q4_header), ("q5", _mut_q5_tests_fail),
             ("q5", _mut_q5_tests_newer), ("q6", _mut_q6)]


@pytest.mark.parametrize("name,mut", MUTATIONS, ids=[m[1].__name__ for m in MUTATIONS])
def test_each_control_fails_once_on_one_planted_fault_exit_2_and_names_it(tmp_path, run, name, mut):
    w = build_world(tmp_path)
    mut(w)
    rc, out = run(w, "controls")
    assert rc == 2, out
    assert out.strip().splitlines()[-1] == "ZZ DSCALE2X CONTROLS FAILED " + name, out
    assert "ctl %s FAIL" % name in out and sum(1 for ln in out.splitlines() if ln.endswith("FAIL")) == 1
    assert controls_json(w)["all_pass"] is False
    must_fail(lambda: _assert(rc == 0))


def test_a_full_log_difference_is_printed_and_not_judged(tmp_path, run):
    w = build_world(tmp_path)
    wlog(w["new_run"] / "runQ" / "TK16_s0" / "train_log.jsonl", 38328, step1_bce=0.70005)       # within 1e-4 at step 1, so q1 holds while the later records differ in nothing but this
    rc, out = run(w, "controls")
    assert rc == 0 and controls_json(w)["controls"]["q1"]["printed_not_judged_full_logs_equal"] is False


def test_controls_mode_writes_and_prints_no_s_l_f_value_and_no_name(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "controls")
    assert rc == 0
    text = Path(R2.latest_path(str(w["out"]), R2.CONTROLS_FILE)).read_text(encoding="utf-8")
    assert all(re.match(r"^(ctl q\d (PASS|FAIL)|ZZ DSCALE2X CONTROLS .*)$", ln) for ln in out.strip().splitlines())
    for bad in ("LEAD-READ", "SCALES", "SHORT-OF", '"S"', '"L"', '"F"', "two_level", "abs_diff", "per_seed", "mean_gap_mae", "ci95"):
        assert bad not in text and bad not in out, bad
    ctx = w["ctx"]
    for label in ("P16Q_s0@38328", "P16Q_s1@19164"):
        r2 = D.summ_one(ctx, {"gap_m": w["preds"][label]})["M2"]["r2"]
        mae = D.summ_one(ctx, {"gap_m": w["preds"][label]})["MAE"]["mae_m"]
        assert repr(r2)[:9] not in text and repr(mae)[:9] not in text
    assert not any(f.name.startswith(("DSCALE2X_RESULT", "DSCALE2X_VERDICT")) for f in w["out"].iterdir())
    must_fail(lambda: _assert("ci95" in text))


def test_a_second_controls_call_writes_a_new_file_and_overwrites_nothing(tmp_path, run):
    w = build_world(tmp_path)
    run(w, "controls")
    first = (w["out"] / R2.CONTROLS_FILE).read_bytes()
    run(w, "controls")
    assert (w["out"] / R2.CONTROLS_FILE).read_bytes() == first and (w["out"] / "DSCALE2X_CONTROLS_2.json").exists()


def test_names_refuses_without_a_passing_controls_file_and_after_an_input_changed(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "names")
    assert rc == 3 and "REFUSED" in out and not (w["out"] / R2.RESULT_FILE).exists()
    _mut_q2(w)
    rc, _ = run(w, "controls")
    assert rc == 2
    rc, out = run(w, "names")
    assert rc == 3 and "NAME" not in out and not (w["out"] / R2.RESULT_FILE).exists()
    w2 = build_world(tmp_path / "w2")
    assert run(w2, "controls")[0] == 0
    p = w2["new_in"] / "runQ" / "TK16_s0_eval.npz.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["note"] = "changed after the controls"
    wjson(p, r)
    rc, out = run(w2, "names")
    assert rc == 3 and not (w2["out"] / R2.RESULT_FILE).exists()
    w3 = build_world(tmp_path / "w3")
    assert run(w3, "controls")[0] == 0
    assert run(w3, "names")[0] == 0 and (w3["out"] / R2.RESULT_FILE).exists() and (w3["out"] / R2.VERDICT_FILE).exists()


@pytest.mark.parametrize("variant,expected", [("gap", "LEAD-READ-2X-SCALES-GAP"), ("short", "LEAD-READ-2X-SHORT-OF-TRAINING"), ("flat", "LEAD-READ-2X-FLAT"), ("worse", "LEAD-READ-2X-WORSE")])
def test_each_of_the_four_names_is_reached_by_planted_data(tmp_path, run, variant, expected):
    w = build_world(tmp_path, variant)
    assert run(w, "controls")[0] == 0
    rc, out = run(w, "names")
    assert rc == 0 and out.strip().splitlines()[-1] == "ZZ DSCALE2X NAME " + expected, out
    res = json.loads((w["out"] / R2.RESULT_FILE).read_text(encoding="utf-8"))
    n, M = res["names"], res["measures"]
    assert n["name"] == expected
    assert set(M) >= {"S", "L", "F", "beside"} and set(M["beside"]) == {"gain_19164_to_28746", "Q_minus_banked_U_M2", "Q_minus_banked_U_MAE"}
    md = (w["out"] / R2.VERDICT_FILE).read_text(encoding="utf-8")
    assert expected in md and "T0" in md and "STRATEGIC" in md
    if variant == "gap":
        assert M["S"]["M2"]["ci95"][0] > 0 and M["S"]["M2"]["n_pos"] == 4 and n["gap_error_clause_met"] and n["mean_s_larger_than_f"] and M["F"]["abs_diff_gap_r2"] == 0.0
    if variant == "short":
        assert n["dose_clause_holds"] is False and M["L"]["ci95"][1] > 0.020 and not n["s_above_0_with_n_pos_at_least_3"]
    if variant == "flat":
        assert M["S"]["M2"]["mean"] == 0.0 and M["L"]["ci95"] == [0.0, 0.0] and n["dose_clause_holds"]
    if variant == "worse":
        assert M["S"]["M2"]["ci95"][1] < 0 and M["S"]["M2"]["mean"] < -M["F"]["abs_diff_gap_r2"] and n["dose_clause_holds"]
    must_fail(lambda: _assert(n["name"] == "LEAD-READ-2X-FLAT" and variant != "flat"))


def test_decide_clauses_individually():
    def tl(mean, lo, hi, npos=4):
        return {"mean": mean, "ci95": [lo, hi], "n_pos": npos}
    S, L, F = {"M2": tl(0.05, 0.01, 0.09)}, tl(0.01, -0.01, 0.019), {"abs_diff_gap_r2": 0.02}
    mae = {"Q": 2.0, "C": 3.0}
    assert R2.decide(S, L, F, mae)["name"] == R2.NAMES[0]
    assert R2.decide(S, L, F, {"Q": 2.8, "C": 3.0})["name"] == R2.NAMES[2]                                  # error clause missed, dose holds
    assert R2.decide({"M2": tl(0.05, 0.01, 0.09, 2)}, L, F, mae)["name"] == R2.NAMES[2]                   # two of four seeds
    assert R2.decide({"M2": tl(0.015, 0.01, 0.09)}, L, F, mae)["name"] == R2.NAMES[2]                    # mean S not larger than F
    assert R2.decide(S, tl(0.03, 0.01, 0.0200001), F, {"Q": 2.8, "C": 3.0})["name"] == R2.NAMES[3]          # dose bar missed by a hair: short of training
    assert R2.decide(S, tl(0.03, 0.01, 0.020), F, {"Q": 2.8, "C": 3.0})["name"] == R2.NAMES[2]              # at the bar the clause holds
    assert R2.decide({"M2": tl(-0.05, -0.09, -0.01, 0)}, L, F, mae)["name"] == R2.NAMES[1]
    assert R2.decide({"M2": tl(-0.05, -0.09, -0.01, 0)}, tl(0.03, 0.01, 0.05), F, mae)["name"] == R2.NAMES[3]   # worse but the dose clause fails
    assert R2.decide({"M2": tl(-0.01, -0.09, 0.01, 0)}, L, F, mae)["name"] == R2.NAMES[2]                  # interval touches 0
    must_fail(lambda: _assert(R2.decide(S, L, F, {"Q": 2.8, "C": 3.0})["name"] == R2.NAMES[0]))


def test_the_pytest_output_clause():
    p = Path(os.environ.get("TEMP", ".")) / "a123read_bt_clause.out"
    p.write_text("....\n4 passed in 1.0s\n", encoding="utf-8")
    os.utime(p, (time.time() - 100, time.time() - 100))
    assert R2.tests_clause(str(p), time.time())["pass"]
    assert not R2.tests_clause(str(p), time.time() - 1000)["pass"]
    assert not R2.tests_clause(str(p) + ".none", time.time())["pass"] and not R2.tests_clause(None, time.time())["pass"]
    p.write_text("....s\n4 passed, 1 skipped in 1.0s\n", encoding="utf-8")
    os.utime(p, (time.time() - 100, time.time() - 100))
    assert not R2.tests_clause(str(p), time.time())["pass"]
    p.unlink()


def test_parse_done_and_unique_path(tmp_path):
    p = tmp_path / "done.txt"
    p.write_text("DSCALE_2X_DONE t\nckpt runQ/TK16_s0 aa\nckpt_step runQ/TK16_s0 19164 bb\nnoise\n", encoding="utf-8")
    d = R2.parse_done(str(p))
    assert d["first"].startswith("DSCALE_2X_DONE") and d["ckpt"] == {"runQ/TK16_s0": "aa"} and d["ckpt_step"] == {("runQ/TK16_s0", 19164): "bb"}
    (tmp_path / "x.json").write_text("1", encoding="utf-8")
    assert R2.unique_path(str(tmp_path), "x.json").endswith("x_2.json")


def test_driver_text_never_removes_moves_or_kills_and_refuses_bad_calls(tmp_path):
    txt = SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    for word in ("rm ", "rm\t", "mv ", "kill", "rmdir", "unlink", "ssh", "scp"):
        assert word not in code, word
    must_fail(lambda: _assert("rm " in code))
    bash = shutil.which("bash")
    assert bash
    d = tmp_path / "drv"
    d.mkdir()
    shutil.copy(str(SCRIPT), str(d / "podread_dscale2x.sh"))
    shutil.copy(str(HERE / "read_dscale2x.py"), str(d / "read_dscale2x.py"))
    sp = str(d / "podread_dscale2x.sh")

    def call(*args):
        return subprocess.run([bash, sp, *args], capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PY": VENV_PY})
    r = call("bogus")
    assert r.returncode == 3 and "STOP" in r.stdout
    r = call("controls")
    assert r.returncode == 3 and "NO-READ-ARGS-FILE" in r.stdout
    (d / "read_args.txt").write_text("--new-run-dir\n%s\n--new-done\n%s\n" % ((tmp_path / "none").as_posix(), (tmp_path / "none" / "DSCALE_2X_DONE.txt").as_posix()), encoding="utf-8")
    r = call("controls")
    assert r.returncode == 3 and "NO-DONE-MARKER" in r.stdout
    assert not (tmp_path / "none").exists()                                                      # nothing was created before the done file was found
    assert "--out-dir" in txt and "_${n}.out" in txt


def test_none_of_the_banned_words_occur_in_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("corre", "ction"), ("with", "drawn"), ("no lo", "nger"), ("closed", " loop")]
    for f in ("read_dscale2x.py", "podread_dscale2x.sh", "test_read_dscale2x.py", "DSCALE2X_READ_DEFS.md"):
        p = HERE / f
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8").lower()
        for a, b in parts:
            assert (a + b) not in t, (f, a + b)


def test_parse_done_reads_the_line_forms_of_the_run_script(tmp_path):
    f = tmp_path / "done.txt"
    f.write_text("DSCALE_2X_DONE 2026-01-01T00:00:00Z\n"
                 "ckpt runQ/TK16_s0/ckpt_last " + "a" * 64 + "\n"
                 "ckpt runQ/TK16_s0/ckpt_step19164 " + "b" * 64 + "\n"
                 "ckpt runQ/TK16_s0/ckpt_step28746 " + "c" * 64 + "\n"
                 "ckpt runC/TK16_s1 " + "d" * 64 + "\n"
                 "ckpt_step runQ/TK16_s1 19164 " + "e" * 64 + "\n", encoding="utf-8")
    d = R2.parse_done(str(f))
    assert d["first"].startswith("DSCALE_2X_DONE")
    assert d["ckpt"] == {"runQ/TK16_s0": "a" * 64, "runC/TK16_s1": "d" * 64}
    assert d["ckpt_step"] == {("runQ/TK16_s0", 19164): "b" * 64, ("runQ/TK16_s0", 28746): "c" * 64, ("runQ/TK16_s1", 19164): "e" * 64}
