import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box2_rhacc_queue as Q  # noqa: E402

# the card's section 3 table, written out independently of the driver: letter -> (form, acc gain, every, grid, preroll)
CARD = {
    "a": ("R0", "0", None, "eval_grid_windows.json", None),
    "b": ("R0", "1", None, "eval_grid_windows.json", None),
    "c": ("RLc", "1", "5", "eval_grid_windows.json", None),
    "d": ("RR", "1", "5", "eval_grid_windows.json", None),
    "e": ("RLc", "0", "5", "eval_grid_windows.json", None),
    "f": ("RR", "0", "5", "eval_grid_windows.json", None),
    "g": ("RLc", "1", "5", "grid_p30.json", "30"),
    "h": ("RLc", "1", "5", "grid_p30.json", "0"),
}
ORDER_OF_RECORD = [(m, l) for m in (0, 1, 2) for l in "eabcdfgh"]
NAMES = {"a": "H_R0K", "b": "H_R0A", "c": "H_RLcA", "d": "H_RQ50A", "e": "U_RLcK", "f": "U_RQ50K",
         "g": "H_RLcA_P30", "h": "H_RLcA_P0"}
LEVER_FLAGS = {"--rule-every", "--kappa-max", "--acc-gain", "--path-exec", "--readout", "--preroll-steps",
               "--reader", "--qmap", "--form", "--nav", "--scene", "--schedule", "--limit", "--replan-steps"}


def flag_map(argv):
    """option -> value (None for a bare flag) over the whole argv, ignoring the interpreter and script."""
    out = {}
    i = 2
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[a] = argv[i + 1]
                i += 2
                continue
            out[a] = None
        i += 1
    return out


def check_row(r, letter, m, root=Q.ROOT):
    """Return a list of problems of the run's argv against the card's row."""
    form, accg, every, grid, pre = CARD[letter]
    f = flag_map(r["argv"])
    bad = []
    if f.get("--form") != form:
        bad.append("form")
    if f.get("--readout") != "causal_g":
        bad.append("readout")
    if f.get("--acc-gain") != accg:
        bad.append("acc-gain")
    if "--path-exec" not in f:
        bad.append("path-exec")
    if f.get("--kappa-max") != "0.2":
        bad.append("kappa-max")
    if f.get("--rule-every") != every:
        bad.append("rule-every")
    if os.path.basename(f.get("--eval-grid", "")) != grid:
        bad.append("grid")
    if f.get("--preroll-steps") != pre:
        bad.append("preroll")
    for k, v in (("--replan-steps", "10"), ("--nav", "deadreckon"), ("--scene", "logged"),
                 ("--schedule", "periodic"), ("--limit", "causal"), ("--device", "cuda")):
        if f.get(k) != v:
            bad.append(k)
    for k in ("--dump-heads", "--no-analyze", "--dump-ticks"):
        if k not in f:
            bad.append(k)
    fam = "UE" if letter in "ef" else "RH"
    if os.path.basename(os.path.dirname(f.get("--ckpt", ""))) != "%s_s%d" % (fam, m):
        bad.append("ckpt")
    if (form == "RR") != ("--reader" in f and "--qmap" in f):
        bad.append("reader/qmap")
    if (form == "R0") != ("--rule-log" not in f):
        bad.append("rule-log")
    # no other lever flag
    allowed = {"--form", "--rule-every", "--kappa-max", "--acc-gain", "--path-exec", "--readout", "--readout-log",
               "--preroll-steps", "--reader", "--qmap", "--rule-log", "--tree", "--ckpt", "--tag", "--tracks",
               "--eval-cache", "--eval-grid", "--eval-join", "--out-dir", "--replan-steps", "--nav", "--scene",
               "--schedule", "--limit", "--dump-ticks", "--dump-heads", "--no-analyze", "--device"}
    extra = set(f) - allowed
    if extra:
        bad.append("extra flags %s" % sorted(extra))
    if "run" not in r["argv"]:
        bad.append("run subcommand")
    return bad


def test_count_and_order_of_record():
    runs = Q.build_runs()
    assert len(runs) == 24
    assert [(r["m"], r["letter"]) for r in runs] == ORDER_OF_RECORD
    assert [r["name"] for r in runs[:8]] == [NAMES[l] for l in "eabcdfgh"]
    assert len({r["arm"] for r in runs}) == 24


@pytest.mark.parametrize("m,letter", ORDER_OF_RECORD)
def test_argv_equals_card_row(m, letter):
    runs = {(r["m"], r["letter"]): r for r in Q.build_runs()}
    assert check_row(runs[(m, letter)], letter, m) == []


def test_k3_control_row_e_matches_pod_line_candidate_flags():
    # control k4 partner: the pod's RLcKF5Hb20 = form RLc, every 5, readout causal_g, acc gain 0, path exec, kappa 0.2
    r = {(x["m"], x["letter"]): x for x in Q.build_runs()}[(0, "e")]
    f = flag_map(r["argv"])
    assert (f["--form"], f["--rule-every"], f["--readout"], f["--acc-gain"], f["--kappa-max"]) == (
        "RLc", "5", "causal_g", "0", "0.2")
    assert "--path-exec" in f and "--preroll-steps" not in f and "--reader" not in f


def test_readers_and_maps_per_model_seed():
    runs = {(r["m"], r["letter"]): r for r in Q.build_runs()}
    for m, rs in ((0, 0), (1, 1), (2, 0)):
        f = flag_map(runs[(m, "d")]["argv"])
        assert "TK16_s%d" % rs in f["--reader"]
        assert "QMAP_LR16_s%d_q50.json" % rs in f["--qmap"]


def test_mutation_drop_flag_of_row_b_goes_red():
    runs = {(r["m"], r["letter"]): r for r in Q.build_runs()}
    r = dict(runs[(0, "b")])
    assert check_row(r, "b", 0) == []
    a = list(r["argv"])
    i = a.index("--acc-gain")
    del a[i:i + 2]
    r["argv"] = a
    assert "acc-gain" in check_row(r, "b", 0)
    a2 = list(runs[(0, "b")]["argv"])
    a2.remove("--path-exec")
    r["argv"] = a2
    assert "path-exec" in check_row(r, "b", 0)
    a3 = list(runs[(0, "b")]["argv"]) + ["--w-field", "2.0"]
    r["argv"] = a3
    assert any(x.startswith("extra flags") for x in check_row(r, "b", 0))


def test_ckpt_table_complete_and_full_length():
    assert set(Q.CKPT_SHA) == {"RH_s0", "RH_s1", "RH_s2", "UE_s0", "UE_s1", "UE_s2"}
    assert all(len(v) == 64 for v in Q.CKPT_SHA.values())
    assert Q.CKPT_SHA["RH_s0"].startswith("351bb5f2") and Q.CKPT_SHA["RH_s1"].startswith("c387f4ce")
    assert Q.CKPT_SHA["UE_s2"].startswith("6d434525")


# ---- stop conditions and dry mode, on a scratch root ----------------------------------------------------------------
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = str(tmp_path / "rh").replace("\\", "/")
    P = Q.paths(root)
    os.makedirs(P["runs"])
    os.makedirs(P["logs"])
    os.makedirs(os.path.dirname(P["grid141"]))
    for k in ("grid141", "gridp30"):
        json.dump({"order": ["c%d" % i for i in range(3)], "windows": {}}, open(P[k], "w"))
    monkeypatch.setattr(Q, "free_disk_gb", lambda p: 50.0)
    monkeypatch.setattr(Q, "free_vram_mib", lambda: 6000.0)
    calls = []

    def fake_exec(argv, env, log_path, timeout_s):
        calls.append(argv)
        f = flag_map(argv)
        dd = os.path.join(f["--out-dir"], "dump_" + f["--tag"])
        os.makedirs(dd)
        for i in range(3):
            open(os.path.join(dd, "ep%03d.npz" % i), "w").close()
        if "--preroll-steps" in f and f["--preroll-steps"] != "0":
            arm = os.path.basename(f["--out-dir"])
            open(os.path.join(f["--out-dir"], "ticks", "preroll_log_%s.json" % arm), "w").close()
        return 0

    monkeypatch.setattr(Q, "execute", fake_exec)
    monkeypatch.setattr(Q, "preflight", lambda *a, **k: None)
    return root, P, calls


def test_dry_runs_nothing_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Q, "execute", lambda *a, **k: pytest.fail("executed in dry mode"))
    root = str(tmp_path / "never").replace("\\", "/")
    rc = Q.main(["--dry", "--root", root])
    out = capsys.readouterr().out
    assert rc == 0 and out.count("\nCMD: ") + (1 if out.startswith("CMD: ") else 0) == 24
    assert "24 runs listed" in out
    assert not os.path.exists(root)


def test_full_queue_runs_24_in_order_and_skips_done(sandbox):
    root, P, calls = sandbox
    assert Q.main(["--root", root]) == 0
    assert len(calls) == 24
    arms = [flag_map(c)["--tag"].replace("_m10_causal_heads", "") for c in calls]
    assert arms == [r["arm"] for r in Q.build_runs(root)]
    assert not os.path.exists(P["busy"])
    calls.clear()
    assert Q.main(["--root", root]) == 0 and calls == []  # every done file exists: nothing re-run


def test_argv_json_and_done_written(sandbox):
    root, P, calls = sandbox
    Q.main(["--root", root])
    r0 = Q.build_runs(root)[0]
    a = json.load(open(r0["outdir"] + "/argv.json"))
    assert a["argv"] == r0["argv"]
    d = json.load(open(r0["outdir"] + "/RUN_DONE.json"))
    assert d["rc"] == 0 and d["dump_files"] == 3


def test_stop_file_present_stops_before_first_run(sandbox):
    root, P, calls = sandbox
    open(P["stop"], "w").write("x")
    assert Q.main(["--root", root]) == 3 and calls == []


def test_nonzero_rc_stops_and_writes_stop(sandbox, monkeypatch):
    root, P, calls = sandbox
    n = {"i": 0}
    orig = Q.execute

    def bad(argv, env, log, to):
        n["i"] += 1
        return orig(argv, env, log, to) if n["i"] < 3 else -1073741819

    monkeypatch.setattr(Q, "execute", bad)
    assert Q.main(["--root", root]) == 3
    assert n["i"] == 3 and os.path.exists(P["stop"])
    assert "ROLLOUT-RC" in open(P["stop"]).read()
    assert os.path.exists(P["busy"])  # left in place on a stop


def test_disk_low_stops(sandbox, monkeypatch):
    root, P, calls = sandbox
    monkeypatch.setattr(Q, "free_disk_gb", lambda p: 1.2)
    assert Q.main(["--root", root]) == 3 and calls == []
    assert "DISK-LOW" in open(P["stop"]).read()


def test_vram_low_and_unreadable_stop(sandbox, monkeypatch):
    root, P, calls = sandbox
    monkeypatch.setattr(Q, "free_vram_mib", lambda: 2000.0)
    assert Q.main(["--root", root]) == 3 and calls == []
    assert "VRAM-LOW" in open(P["stop"]).read()
    os.remove(P["stop"])
    monkeypatch.setattr(Q, "free_vram_mib", lambda: None)
    assert Q.main(["--root", root]) == 3 and calls == []
    assert "VRAM-PROBE-UNREADABLE" in open(P["stop"]).read()


def test_output_folder_without_done_is_refused(sandbox):
    root, P, calls = sandbox
    r0 = Q.build_runs(root)[0]
    os.makedirs(r0["outdir"])
    assert Q.main(["--root", root]) == 3 and calls == []
    assert "OUTPUT-EXISTS-WITHOUT-DONE" in open(P["stop"]).read()


def test_dump_count_mismatch_stops(sandbox, monkeypatch):
    root, P, calls = sandbox
    orig = Q.execute

    def short(argv, env, log, to):
        rc = orig(argv, env, log, to)
        f = flag_map(argv)
        dd = os.path.join(f["--out-dir"], "dump_" + f["--tag"])
        os.remove(os.path.join(dd, "ep000.npz"))
        return rc

    monkeypatch.setattr(Q, "execute", short)
    assert Q.main(["--root", root]) == 3
    assert "DUMP-COUNT" in open(P["stop"]).read()


def test_forbidden_words_absent():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "box2_rhacc_queue.py"), encoding="utf-8").read().lower()
    for w in ("vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lon" + "ger", "closed " + "loop"):
        assert w not in src


def test_max_runs_pauses_after_n_and_resumes(sandbox):
    root, P, calls = sandbox
    assert Q.main(["--root", root, "--max-runs", "1"]) == 0
    assert len(calls) == 1
    first = Q.build_runs(root)[0]
    assert flag_map(calls[0])["--tag"].startswith(first["arm"])
    assert first["arm"].startswith("U_RLcK")
    calls.clear()
    assert Q.main(["--root", root]) == 0 and len(calls) == 23


def test_amendment3_wrapper_per_row(tmp_path):
    runs = Q.build_runs(str(tmp_path))
    for r in runs:
        w = r["argv"][1]
        if r["letter"] in ("g", "h"):
            assert w.endswith("/build/v8f_t1_eval_reader_rule_rzgfhq.py") and "--preroll-steps" in r["argv"]
        else:
            assert w.endswith("/build/v8f_t1_eval_reader_rule_rzgfh.py") and "--preroll-steps" not in r["argv"]
    assert Q.PINNED_SHA["wrapper_pre"].startswith("92a8034cc0f7b0d2")
