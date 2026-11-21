"""test_navrbd_chain.py -- tests of the NAVRBD chain pieces: navrbd_t0, navrbd_view, navrbd_strategic, navrbd_pod, navrbd_export and the chain script text.  Run the WHOLE file.  CPU, no model is loaded."""
import hashlib
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import navrbd_t0 as T0  # noqa: E402
import navrbd_view as V  # noqa: E402
import navrbd_strategic as S  # noqa: E402
import navrbd_pod as P  # noqa: E402
import navrbd_export as X  # noqa: E402

CHAIN = open(HERE + "/navrbd_chain.sh", encoding="utf-8").read()


def flags(argv, name):
    i = argv.index(name)
    return argv[i + 1]


# ---------------------------------------------------------------- T0
def test_t0_real_variant_flags():
    a = T0.argv_for("N1D_s0", "real", "OUT")
    assert flags(a, "--nav-rb") == "on" and flags(a, "--nav-rb-feat") == "exitdist" and flags(a, "--nav-rb-variant") == "follow"
    assert "--nav-shuffle" not in a and "--nav-withheld" not in a
    assert flags(a, "--tag") == "NAVRBD-N1D_s0_real" and a[1].endswith("/stack/scripts/v8f_mini_eval_rbd.py")
    assert flags(a, "--nav-rb-sha256") == "298049df71dc469874dd2429781c9019c4727b917bc3bd2233314542f6bbc82a"


def test_t0_donor_holds_the_channel_at_zero_beside_the_donor_nav():
    a = T0.argv_for("N1D_s1", "donor", "OUT")
    assert "--nav-shuffle" in a and flags(a, "--nav-rb-variant") == "zero" and "--nav-withheld" not in a


def test_t0_zero_is_nav_withheld():
    a = T0.argv_for("N1D_s1", "zero", "OUT")
    assert "--nav-withheld" in a and "--nav-shuffle" not in a and "--nav-rb-variant" not in a


def test_t0_arms_are_the_two_n1d_arms_only():
    assert T0.D_ARMS == ("N1D_s0", "N1D_s1") and set(T0.VARIANT_FLAGS) == {"real", "donor", "zero"}


# ---------------------------------------------------------------- view
def test_view_presents_n1d_as_n1_and_keeps_c0():
    p = dict((rel, src) for src, rel in V.plan(t0="T0", t1="T1", banked_t0="B0", banked_t1="B1"))
    assert p["t0/eval_STRAT1-N1_s0_real.npz"] == "T0/eval_NAVRBD-N1D_s0_real.npz"
    assert p["t0/eval_STRAT1-N1_s1_donor.npz"] == "T0/eval_NAVRBD-N1D_s1_donor.npz"
    assert p["t1/ticks/ticks_STRAT1-N1_s1_m10_causal_heads.npz"] == "T1/ticks/ticks_NAVRBD-N1D_s1_m10_causal_heads.npz"
    assert p["t0/eval_STRAT1-C0_rep_zero.npz"] == "B0/eval_STRAT1-C0_rep_zero.npz"
    assert not any("S1" in rel.replace("STRAT1", "") for rel in p), "no S1 arm exists in NAVRBD"
    assert len(V.plan(ticks=False)) == 2 * 3 + 3 * 3


def test_view_build_copies_and_refuses_a_missing_pass(tmp_path):
    t0, bt0 = tmp_path / "t0", tmp_path / "bt0"
    t0.mkdir(), bt0.mkdir()
    for s in (0, 1):
        for v in V.VARS:
            (t0 / f"eval_NAVRBD-N1D_s{s}_{v}.npz").write_bytes(b"d%d%s" % (s, v.encode()))
    for arm in ("C0_s0", "C0_s1", "C0_rep"):
        for v in V.VARS:
            (bt0 / f"eval_STRAT1-{arm}_{v}.npz").write_bytes(b"c")
    n = V.build(str(tmp_path / "view"), t0=str(t0), banked_t0=str(bt0), ticks=False)
    assert n == 15 and (tmp_path / "view/t0/eval_STRAT1-N1_s1_donor.npz").read_bytes() == b"d1donor"
    os.remove(t0 / "eval_NAVRBD-N1D_s1_zero.npz")
    with pytest.raises(FileNotFoundError):
        V.build(str(tmp_path / "view2"), t0=str(t0), banked_t0=str(bt0), ticks=False)


# ---------------------------------------------------------------- strategic
def test_route_cost_is_a_cell_that_is_a_cost_at_both_seeds():
    cells = S.SS.RC_CELLS
    c = cells[0]
    both = {0: {k: {"cost": k == c} for k in cells}, 1: {k: {"cost": k == c} for k in cells}}
    one = {0: {k: {"cost": k == c} for k in cells}, 1: {k: {"cost": False} for k in cells}}
    assert S.route_cost_of(both) == [c] and S.route_cost_of(one) == [] and S.route_cost_of({0: both[0]}) == []


def test_contrast_key_is_the_one_the_read_expects():
    assert S.pair_name() == "N1D-N1R"
    src = open(HERE + "/navrbd_read.py", encoding="utf-8").read()
    assert 'get("route_cost", {}).get("N1D-N1R")' in src
    assert 'rep["route_cost"][pair_name()]' in open(HERE + "/navrbd_strategic.py", encoding="utf-8").read()


def test_strategic_compares_n1d_with_the_navrb2_n1r_pass():
    src = open(HERE + "/navrbd_strategic.py", encoding="utf-8").read()
    assert 'f"NAVRBD-N1D_s{s}"' in src and 'f"NAVRB2-N1R_s{s}"' in src


# ---------------------------------------------------------------- pod
def test_all_end_is_the_only_end():
    main_done = "22:50 x\nZZ RBDPOD DONE ZZ\n"
    assert P.parse_all_end(main_done) is None, "the main DONE line is written before the zero dumps"
    assert P.parse_all_end("ZZ RBDPOD FAIL ZZ step 4\n") is None
    end = main_done + "ZZ RBDPOD ZERO N1D_s0 DONE ZZ\nZZ RBDPOD ALL END ZZ main: ZZ RBDPOD DONE ZZ; zero dumps DONE\n"
    assert P.parse_all_end(end) == ("ZZ RBDPOD DONE ZZ", "DONE")
    f = "ZZ RBDPOD ALL END ZZ main: ZZ RBDPOD FAIL ZZ step 3; zero dumps FAIL\n\n"
    assert P.parse_all_end(f) == ("ZZ RBDPOD FAIL ZZ step 3", "FAIL")
    assert P.parse_all_end("ZZ RBDPOD ALL END ZZ main: x; zero dumps DONE\nlater line\n") is None


def test_wait_end_polls_then_returns_and_caps():
    texts = iter(["a", "ZZ RBDPOD DONE ZZ", "ZZ RBDPOD ALL END ZZ main: m; zero dumps DONE"])
    slept = []
    r = P.wait_end(lambda: next(texts), poll=300, cap_s=10 ** 6, sleep=slept.append, clock=lambda: 0)
    assert r[:2] == ("m", "DONE") and slept == [300, 300]
    t = iter(range(0, 100000, 4000))
    assert P.wait_end(lambda: "ZZ RBDPOD DONE ZZ", poll=1, cap_s=6 * 3600, sleep=lambda s: None, clock=lambda: next(t)) is None


def test_sums_and_verify(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d/a.npz").write_bytes(b"abc")
    good = hashlib.sha256(b"abc").hexdigest()
    sums = P.parse_sums(f"{good}  d/a.npz\n{'0' * 64}  d/b.npz\n")
    assert sums["d/a.npz"] == good and P.verify_tree(str(tmp_path), {"d/a.npz": good}) == []
    assert P.verify_tree(str(tmp_path), {"d/a.npz": "1" * 64}) == ["sha256 differs d/a.npz"]
    assert P.verify_tree(str(tmp_path), sums) == ["missing d/b.npz"]
    assert P.verify_tree(str(tmp_path), {}) == ["no files listed"]


def test_pod_script_is_read_only_and_has_no_address():
    src = open(HERE + "/navrbd_pod.py", encoding="utf-8").read()
    for bad in ("rm -", "kill", "mv ", "> /root", "pkill"):
        assert bad not in src
    assert "/root/strat1/navrbd" in src and "podenv.sh" in src
    import re
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src)


# ---------------------------------------------------------------- export
def test_export_blob_check_catches_crlf_and_long_paths(tmp_path):
    body = b"line1\nline2\n"
    sha = hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()
    (tmp_path / "stack").mkdir()
    (tmp_path / "stack/f.py").write_bytes(body)
    want = X.parse_ls_tree(f"100644 blob {sha}\tstack/f.py\n")
    assert want == {"stack/f.py": sha} and X.problems(want, str(tmp_path)) == []
    (tmp_path / "stack/f.py").write_bytes(body.replace(b"\n", b"\r\n"))
    assert X.problems(want, str(tmp_path)) == ["blob differs stack/f.py"]
    assert X.problems({"x" * 201: sha}, str(tmp_path))[0].startswith("repo path 201")
    assert X.problems({}, str(tmp_path)) == ["the commit lists no files under the exported paths"]


def test_export_paths_hold_what_the_evaluators_load():
    assert set(X.PATHS) == {"stack/scripts", "stack/tanitad", "taniteval/taniteval", "taniteval/tools"}


# ---------------------------------------------------------------- chain script
def test_chain_waits_for_the_final_pod_line_and_the_um_sentinel():
    assert "navrbd_pod.py wait-end" in CHAIN and "ZZ RBDPOD ALL END ZZ" in CHAIN
    assert "RBDPOD DONE" not in CHAIN.replace("ZZ RBDPOD ALL END ZZ main: ...; zero dumps DONE|FAIL", "")
    assert "ZZ DSCALE MIX READ DONE ZZ" in CHAIN and "ZZ DSCALE MIX READ FAIL ZZ" in CHAIN
    assert "NAVRBD_POLL_S=300" in CHAIN and "NAVRBD_CAP_H=6" in CHAIN


def test_chain_order_gate_before_any_pull_and_read_last():
    order = ["# ---- 0 gate", "step pull_ckpts", "step export_tree", "step dumps141", "step t0 ", '$PY -u navrbd_view.py', '$PY -u navrbd_strategic.py', "navrbd_pod.py wait-end", "step pull_dumps", 'env $READENV $PY -u navrbd_read.py --out']
    i = {k: CHAIN.index(k) for k in order}
    assert [i[k] for k in order] == sorted(i[k] for k in order)


def test_chain_sentinels_and_commit_gate():
    assert CHAIN.count("ZZ NAVRBD READ DONE ZZ") >= 1 and "ZZ NAVRBD READ FAIL ZZ" in CHAIN
    assert "wait_commit.py 5.5 60" in CHAIN and "CUDA_VISIBLE_DEVICES=-1" in CHAIN and "OMP_NUM_THREADS=2" in CHAIN
    assert "export PATH=" in CHAIN and "python3" not in CHAIN.replace("python3 -c", "")
    assert "$STEPS/$n.ok" in CHAIN, "re-runnable: finished steps are skipped"
    assert "--contrast" in CHAIN and "NAVRBD_STRATEGIC_contrast" in CHAIN


def test_job_cmd_uses_the_usr_bin_bash():
    j = open(HERE + "/job_navrbd_chain.cmd", encoding="utf-8").read()
    assert "Git\\usr\\bin\\bash.exe" in j and "Git\\bin\\bash.exe" not in j.replace("NOT Git\\bin\\bash.exe", "")
    assert "0x08000200" in open(HERE + "/launch_wmi.ps1", encoding="utf-8").read()
