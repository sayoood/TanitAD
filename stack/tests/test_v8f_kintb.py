"""KINTB (BRIEF_BUILD_KINTB, MM design binding): the kin_v1 blend horizon T_B is a trainer flag, ``--dagger-kin-tb SECONDS`` (default 1.0 = commit 1a8b11a1 bit for bit).

What is pinned here (``mutants_kintb/MUTANTS_RESULT.json`` turns each RED):
  * DEFAULT, GOLD: with the flag absent, ``tb=1.0`` or ``tb=1`` the labeller (``kin_label_bank``, per record ``kin_label``) equals the golden copy of 1a8b11a1's ``v8f_recovery.py`` bit for bit (fixture, planted bank); a cache written
    at the default has the same ARRAYS as 1a8b11a1's and a header that differs only in the two code hashes and the new ``kin_tb``; a cache written by the golden code reads at 1.0 (a header without ``kin_tb`` is T_B = 1.0) and is refused at 2.0;
    the swap and the built levers equal 1a8b11a1's apart from the new ``dagger_kin_tb`` (1.0);
  * 2.0 MOVES ONLY KIN LABELS: a record that is identity_rule or whose reference speed_v1 label was refused keeps its label exactly; every emitted record merges at t >= 2.0 s; a record neither emitted at 1.0 nor at 2.0 keeps its speed_v1 label; at least
    one record's kin label really differs between 1.0 and 2.0 (else the rest is vacuous);
  * REFUSALS: non-default with speed_v1 / human; non-finite / below 0.2 / above 3.0 / not a number; a cache whose ``kin_tb`` differs (both directions); an old cache at 2.0; at ``--dagger-p 0``;
  * REACHABILITY: ``--dagger-kin-tb 2.0`` through the trainer's argv -> the batch's ``gt`` differs from the 1.0 batch on at least one kin row, the built levers read 2.0, the gate passes, the summary carries it.
Run the WHOLE file (module fixtures are shared, and a ``-k`` run leaks them): CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_kintb.py"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_dagger as G  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_recovery as RV  # noqa: E402
from test_v8f_dagger import world, fx10, make_batch  # noqa: E402,F401
from test_v8f_dagger_train import pack, DAG, TR, DG, dvb, _run, _summary, _selfcheck  # noqa: E402,F401
from test_v8f_recovery import eligible_windows_of, _same  # noqa: E402,F401
from test_v8f_recovery_wiring import gate_parts_rl  # noqa: E402,F401
from test_v8f_kinlab import kin_bank  # noqa: E402,F401
from test_v8f_kincache import _fields, _block_labellers, _assert_same_res, _clone, _count_labellers  # noqa: E402,F401

torch.set_num_threads(1)
GOLD_R = HERE / "golden_v8f_recovery_1a8b11a1.py"
GOLD_D = HERE / "golden_v8f_dagger_1a8b11a1.py"
GOLD_R_SHA256 = "795d380cb5bf0a04fb24b84392b6bd5f71df15bbc5ecb78a6267488225dd7440"      # sha256 of `git show 1a8b11a1:stack/tanitad/data/v8f_recovery.py`
GOLD_D_SHA256 = "3a409cedc4e31fe4c0ad0b44ca312b1dda397f5b41c625f0b330886abf218b7e"      # sha256 of `git show 1a8b11a1:stack/tanitad/data/v8f_dagger.py`
KINDS = ("speed_v1", "kin_v1")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def gold():
    assert hashlib.sha256(GOLD_R.read_bytes()).hexdigest() == GOLD_R_SHA256 and hashlib.sha256(GOLD_D.read_bytes()).hexdigest() == GOLD_D_SHA256
    return _load(GOLD_R, "golden_v8f_recovery_1a8b11a1"), _load(GOLD_D, "golden_v8f_dagger_1a8b11a1")


@pytest.fixture(scope="module")
def banks(fx10):
    return {"natural": fx10.bank, "planted": kin_bank(fx10, plant=10**9, kappa=0.02, da=0.6)}


def _kin(f, bank, **kw):
    return RV.kin_label_bank(bank.a, bank.ci, f.J["clip"], f.store.get_kin, **kw)


def _per_record(f, bank, tb):
    """{record index: kin_label result} at blend horizon tb, over the ok records of the bank."""
    out = {}
    for i in np.flatnonzero(bank.reason == G.OK):
        c, r = int(bank.ci[i]), int(bank.a["r"][i])
        grid = np.asarray(f.store.get_kin(str(f.J["clip"][c]))["pose"], np.float64)[:, :3]
        out[int(i)] = RV.kin_label(grid, r, np.asarray(bank.a["p10f"][i, -1, :3], np.float64), tb=tb)
    return out


# ------------------------------------------------------------------------------------------------ the default is 1a8b11a1, bit for bit
@pytest.mark.parametrize("which", ["natural", "planted"])
def test_default_labels_equal_1a8b11a1_bit_for_bit_with_the_flag_absent_1_0_and_the_int_1(fx10, banks, gold, which):
    gr, _ = gold
    bank = banks[which]
    ref = gr.kin_label_bank(bank.a, bank.ci, fx10.J["clip"], fx10.store.get_kin)
    for kw in ({}, {"tb": 1.0}, {"tb": 1}, {"tb": RV.T_B}):
        _assert_same_res(_kin(fx10, bank, **kw), ref, "kin_v1")
    assert RV.T_B == 1.0 and gr.T_B == 1.0
    if which == "planted":
        assert int(ref["is_kin"].sum()) > 0, "the planted bank must carry kin_v1 labels, else this is vacuous"
    for i, res in _per_record(fx10, bank, 1.0).items():                              # per record: the status, the merge time and the label
        c, r = int(bank.ci[i]), int(bank.a["r"][i])
        grid = np.asarray(fx10.store.get_kin(str(fx10.J["clip"][c]))["pose"], np.float64)[:, :3]
        g = gr.kin_label(grid, r, np.asarray(bank.a["p10f"][i, -1, :3], np.float64))
        assert res["status"] == g["status"] and res["t_merge"] == g["t_merge"] and res["max_dev"] == g["max_dev"]
        assert (res["label"] is None) == (g["label"] is None) and (res["label"] is None or res["label"].tobytes() == g["label"].tobytes())


@pytest.mark.parametrize("kind", KINDS)
def test_default_cache_has_1a8b11a1s_arrays_and_a_header_that_differs_only_in_the_code_hashes_and_kin_tb(fx10, banks, gold, tmp_path, kind):
    gr, gd = gold
    bank = banks["planted"]
    fn_g = gr.kin_label_bank if kind == "kin_v1" else gr.label_bank
    res_g = fn_g(bank.a, bank.ci, fx10.J["clip"], fx10.store.get_kin)
    res_n = _kin(fx10, bank) if kind == "kin_v1" else RV.label_bank(bank.a, bank.ci, fx10.J["clip"], fx10.store.get_kin)
    id_g = gd.label_cache_identity(bank, fx10.J["clip"], fx10.store.get_kin, label_kind=kind, labeller_path=GOLD_R, dagger_path=GOLD_D)
    id_n = G.label_cache_identity(bank, fx10.J["clip"], fx10.store.get_kin, label_kind=kind, labeller_path=RV.__file__, dagger_path=G.__file__)
    pg, pn = tmp_path / "g.npz", tmp_path / "n.npz"
    gd.write_label_cache(pg, id_g, res_g)
    G.write_label_cache(pn, id_n, res_n)
    with np.load(pg, allow_pickle=False) as zg, np.load(pn, allow_pickle=False) as zn:
        assert set(zg.files) == set(zn.files)
        for k in zg.files:
            if k != "header_json":
                a, b = np.array(zg[k]), np.array(zn[k])
                assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
        hg, hn = json.loads(bytes(zg["header_json"]).decode()), json.loads(bytes(zn["header_json"]).decode())
    assert hg["arrays_sha256"] == hn["arrays_sha256"]
    moved = {k for k in set(hg) | set(hn) if hg.get(k) != hn.get(k)}
    assert moved == {"labeller_sha256", "dagger_sha256"} | ({"kin_tb"} if kind == "kin_v1" else set()), moved   # the code hashes always move (the files were edited); kin_tb is new for kin_v1 only
    assert ("kin_tb" in hn) == (kind == "kin_v1") and (kind != "kin_v1" or hn["kin_tb"] == 1.0)
    assert id_n["labeller_sha256"] == hashlib.sha256(Path(RV.__file__).read_bytes()).hexdigest() != GOLD_R_SHA256


def test_a_cache_written_by_1a8b11a1s_code_reads_at_the_default_and_is_refused_at_2_0(fx10, banks, gold, tmp_path):
    """A header without ``kin_tb`` is a T_B = 1.0 cache. (The code hashes are held equal here by naming the golden files in the identity; with the edited files the two RK caches built at 1a8b11a1 are
    refused for [labeller_sha256, dagger_sha256] exactly as KINCACHE specified for ANY edit of those files -- see FACTS_KINTB.)"""
    gr, gd = gold
    bank = banks["planted"]
    res_g = gr.kin_label_bank(bank.a, bank.ci, fx10.J["clip"], fx10.store.get_kin)
    mk = lambda tb: G.label_cache_identity(bank, fx10.J["clip"], fx10.store.get_kin, label_kind="kin_v1", labeller_path=GOLD_R, dagger_path=GOLD_D, kin_tb=tb)
    path = tmp_path / "old.npz"
    gd.write_label_cache(path, gd.label_cache_identity(bank, fx10.J["clip"], fx10.store.get_kin, label_kind="kin_v1", labeller_path=GOLD_R, dagger_path=GOLD_D), res_g)
    with np.load(path, allow_pickle=False) as z:
        assert "kin_tb" not in json.loads(bytes(z["header_json"]).decode())
    back, _ = G.read_label_cache(path, mk(1.0))
    _assert_same_res(back, res_g, "kin_v1")
    with pytest.raises(G.BankRefused) as e:
        G.read_label_cache(path, mk(2.0))
    assert _fields(str(e.value)) == {"kin_tb"}, str(e.value)
    with pytest.raises(G.BankRefused) as e:                                                 # and with the edited code files at the default: refused for exactly the two code hashes (kin_tb is not among them)
        G.read_label_cache(path, G.label_cache_identity(bank, fx10.J["clip"], fx10.store.get_kin, label_kind="kin_v1", labeller_path=RV.__file__, dagger_path=G.__file__))
    assert _fields(str(e.value)) == {"labeller_sha256", "dagger_sha256"}, str(e.value)


def test_default_swap_and_built_levers_equal_1a8b11a1s_apart_from_the_new_key(fx10, banks, gold):
    _, gd = gold
    for kind in ("human", "speed_v1", "kin_v1"):
        bn = _clone(fx10, banks["planted"])
        bg = gd.DaggerBank({k: np.array(v) for k, v in banks["planted"].a.items()}, banks["planted"].header, clip_names=fx10.J["clip"], manifest=banks["planted"].manifest, sha256=banks["planted"].sha256, path=banks["planted"].path)
        if kind != "human":
            res = RV.label_bank(bn.a, bn.ci, fx10.J["clip"], fx10.store.get_kin) if kind == "speed_v1" else _kin(fx10, bn)
            for b in (bn, bg):
                b.attach_recovery(res["slot"], res["label"], res["status_counts"], **({"kin": res["is_kin"]} if kind == "kin_v1" else {}))
        batch = make_batch(fx10, eligible_windows_of(fx10, bn))
        for step in (0, 5):
            assert _same(G.DaggerSwap(bn, p=1.0, seed=3).apply(batch, step), gd.DaggerSwap(bg, p=1.0, seed=3).apply(batch, step)), (kind, step)
        lv_n, lv_g = G.DaggerSwap(bn, p=1.0, seed=3).built_levers(), gd.DaggerSwap(bg, p=1.0, seed=3).built_levers()
        assert lv_n.pop("dagger_kin_tb") == 1.0 and "dagger_kin_tb" not in lv_g and lv_n == lv_g, kind
    assert G.DaggerSwap().built_levers()["dagger_kin_tb"] == 1.0


def test_the_flag_exists_defaults_to_1_0_and_has_a_declared_vs_built_entry():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.dagger_kin_tb == 1.0 and isinstance(a.dagger_kin_tb, float) and TR.FLAG_CHECKS["dagger_kin_tb"] == "dagger:dagger_kin_tb"
    assert {act.dest for act in TR.build_parser()._actions if act.dest != "help"} == set(TR.FLAG_CHECKS)
    assert TR.build_parser().parse_args(["--out", "x", "--dagger-kin-tb", "2"]).dagger_kin_tb == 2.0


# ------------------------------------------------------------------------------------------------ 2.0 moves only kin labels
def test_t_b_2_0_moves_only_kin_labels_and_never_merges_before_2_0(fx10, banks):
    n_diff = n_emit2 = n_keep = 0
    for which in ("planted", "natural"):
        n = _check_2_0(fx10, banks[which])
        n_diff, n_emit2, n_keep = n_diff + n[0], n_emit2 + n[1], n_keep + n[2]
    assert n_emit2 > 0 and n_keep > 0, "the fixture must carry emitted and unchanged records, else this is vacuous"
    assert n_diff > 0, "at least one kin record's label differs between T_B 1.0 and 2.0"


def _check_2_0(fx10, bank):
    r1, r2 = _per_record(fx10, bank, 1.0), _per_record(fx10, bank, 2.0)
    sp = RV.label_bank(bank.a, bank.ci, fx10.J["clip"], fx10.store.get_kin)
    n_diff = n_emit2 = n_keep = 0
    for i in r1:
        a, b = r1[i], r2[i]
        assert a["speed_status"] == b["speed_status"], i                                  # the reference does not depend on T_B
        if a["status"] == "identity_rule" or a["status"].startswith("speed_"):             # not a kin record at 1.0 by construction: unchanged
            assert b["status"] == a["status"], i
            assert (a["label"] is None and b["label"] is None) or a["label"].tobytes() == b["label"].tobytes(), i
            n_keep += 1
        if b["status"] == "kin_emitted":
            n_emit2 += 1
            assert b["t_merge"] >= 2.0 - 1e-9, (i, b["t_merge"])
            if a["status"] == "kin_emitted":
                n_diff += int(a["label"].tobytes() != b["label"].tobytes())
        if a["status"] == "kin_emitted":
            assert a["t_merge"] >= 1.0 - 1e-9
        if a["status"] != "kin_emitted" and b["status"] != "kin_emitted":                  # neither: both keep the speed_v1 label (None when the reference was refused)
            for x in (a, b):
                assert x["label"] is None or x["status"] == "identity_rule"
            assert (a["speed_label"] is None) == (b["speed_label"] is None) and (a["speed_label"] is None or a["speed_label"].tobytes() == b["speed_label"].tobytes())
    bk1, bk2 = _kin(fx10, bank), _kin(fx10, bank, tb=2.0)                                   # bank level: a record that is not kin_emitted in either keeps the speed_v1 bank label (slot -> the same bytes)
    assert bk1["label"].shape[1:] == bk2["label"].shape[1:] == (30, 2) and bk1["slot"].shape == bk2["slot"].shape
    for i in range(len(bank.reason)):
        if bank.reason[i] == G.OK and not bk1["is_kin"][i] and not bk2["is_kin"][i] and bk1["slot"][i] >= 0 and bk2["slot"][i] >= 0:
            assert bk1["label"][bk1["slot"][i]].tobytes() == bk2["label"][bk2["slot"][i]].tobytes() == sp["label"][sp["slot"][i]].tobytes(), i
    assert bk2["census"]["merge_time_s"]["p50"] is None or bk2["census"]["merge_time_s"]["p50"] >= 2.0 - 1e-9
    assert bk2["label"].tobytes() != bk1["label"].tobytes() or int(bk1["is_kin"].sum()) == 0
    return n_diff, n_emit2, n_keep


# ------------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, 0.19, 0.1999, -1.0, 3.0001, 3.5, 6.0, True, "1.0", [1.0]])
def test_an_out_of_range_or_non_numeric_t_b_is_refused_by_the_labeller(fx10, banks, bad):
    with pytest.raises(ValueError):
        RV.check_kin_tb(bad)
    with pytest.raises(ValueError):
        _kin(fx10, banks["natural"], tb=bad)
    with pytest.raises(ValueError):
        RV.kin_label(np.zeros((40, 3)), 0, (1.0, 0.0, 0.0), tb=bad)


@pytest.mark.parametrize("ok", [0.2, 0.5, 1.0, 1, 2.0, 3.0, np.float64(1.5)])
def test_the_closed_range_0_2_to_3_0_is_accepted(ok):
    assert RV.check_kin_tb(ok) == float(ok) and isinstance(RV.check_kin_tb(ok), float)


def test_none_means_the_module_default_looked_up_at_the_call(fx10, banks, monkeypatch):
    _assert_same_res(_kin(fx10, banks["planted"], tb=None), _kin(fx10, banks["planted"], tb=1.0), "kin_v1")
    with monkeypatch.context() as mp:
        mp.setattr(RV, "T_B", 2.0)
        _assert_same_res(_kin(fx10, banks["planted"]), _kin(fx10, banks["planted"], tb=2.0), "kin_v1")


def test_the_trainer_refuses_a_non_default_t_b_with_another_label_kind_or_out_of_range(pack):
    for kind in ("speed_v1", "human"):
        with pytest.raises(SystemExit) as e:
            gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", kind, "--dagger-kin-tb", "2.0"])
        assert "valid only with --dagger-label kin_v1" in str(e.value) and "ZZDVB-REFUSEDZZ" in str(e.value), (kind, str(e.value))
        gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", kind, "--dagger-kin-tb", "1.0"])      # the default value spelled out is not a lever
    for bad in ("0.1", "3.5", "nan", "inf", "-2"):
        with pytest.raises(SystemExit) as e:
            gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-kin-tb", bad])
        assert "ZZDVB-REFUSEDZZ" in str(e.value) and "--dagger-kin-tb" in str(e.value), (bad, str(e.value))
    for kw in ({"p": 0.0}, ):                                                                           # p = 0: no swap loaded, the lever does nothing: the gate refuses
        r = gate_parts_rl(pack, ["--init-ckpt", str(pack.ck), "--dagger-p", "0", "--dagger-label", "kin_v1", "--dagger-kin-tb", "2.0"])
        msgs = dvb(r[:9])
        assert any("dagger_kin_tb" in m or "dagger-kin-tb" in m for m in msgs), msgs
    a = TR.build_parser().parse_args(["--out", str(pack.init_out), *__import__("test_v8f_dagger_train").GATE_ARGV, "--dagger-kin-tb", "2.0"])     # no swap built at all
    assert any("dagger_kin_tb" in m and "no DAgger swap was built" in m for m in TR.bad_dagger(a, None))
    ok = TR.build_parser().parse_args(["--out", "x", "--dagger-label", "speed_v1", "--dagger-kin-tb", "2.0"])
    assert "valid only with --dagger-label kin_v1" in TR.kin_tb_refusal(ok)


def _ident(f, bank, tb):
    return G.label_cache_identity(bank, f.J["clip"], f.store.get_kin, label_kind="kin_v1", labeller_path=RV.__file__, dagger_path=G.__file__, kin_tb=tb)


def test_a_cache_whose_kin_tb_differs_is_refused_naming_kin_tb_in_both_directions(fx10, banks, tmp_path):
    bank = banks["planted"]
    paths = {}
    for tb in (1.0, 2.0):
        paths[tb] = tmp_path / f"c{tb}.npz"
        G.write_label_cache(paths[tb], _ident(fx10, bank, tb), _kin(fx10, bank, tb=tb))
    assert paths[1.0].read_bytes() != paths[2.0].read_bytes()
    for have, ask in ((1.0, 2.0), (2.0, 1.0), (2.0, 1.5), (1.0, 3.0)):
        with pytest.raises(G.BankRefused) as e:
            G.read_label_cache(paths[have], _ident(fx10, bank, ask))
        assert _fields(str(e.value)) == {"kin_tb"} and f"T_B {float(have)!r}" in str(e.value) and f"{float(ask)!r} s" in str(e.value), str(e.value)
    for tb in (1.0, 2.0):                                                                  # and each reads back at its own value, equal to a fresh compute
        back, _ = G.read_label_cache(paths[tb], _ident(fx10, bank, tb))
        _assert_same_res(back, _kin(fx10, bank, tb=tb), "kin_v1")
    ident = _ident(fx10, bank, 1.0)                                                         # a hand-edited header value is refused too (a non-number, a bool)
    assert ident["kin_tb"] == 1.0 and "kin_tb" not in G.label_cache_identity(bank, fx10.J["clip"], fx10.store.get_kin, label_kind="speed_v1", labeller_path=RV.__file__, dagger_path=G.__file__)


def test_the_trainer_refuses_a_cache_at_another_t_b_and_reads_its_own(pack, tmp_path, monkeypatch):
    c1, c2 = tmp_path / "t1.npz", tmp_path / "t2.npz"
    d1 = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-label-cache", str(c1)])
    d2 = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-kin-tb", "2.0", "--dagger-label-cache", str(c2)])
    assert d1[4]["dagger"]["speed_label"]["label_cache"]["read"] is False and d2[4]["dagger"]["speed_label"]["label_cache"]["read"] is False
    with pytest.raises(SystemExit) as e:
        gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-kin-tb", "2.0", "--dagger-label-cache", str(c1)])       # a 1.0 cache at 2.0
    assert _fields(str(e.value)) == {"kin_tb"}, str(e.value)
    with pytest.raises(SystemExit) as e:
        gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-label-cache", str(c2)])                                   # a 2.0 cache at the default
    assert _fields(str(e.value)) == {"kin_tb"}, str(e.value)
    _block_labellers(monkeypatch)                                                          # each reads back at its own T_B without the labeller
    r1 = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-label-cache", str(c1)])
    r2 = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-kin-tb", "2.0", "--dagger-label-cache", str(c2)])
    monkeypatch.undo()
    assert r1[5].built_levers()["dagger_kin_tb"] == 1.0 and r2[5].built_levers()["dagger_kin_tb"] == 2.0
    assert r1[5].bank.rl["label"].tobytes() == d1[5].bank.rl["label"].tobytes() and r2[5].bank.rl["label"].tobytes() == d2[5].bank.rl["label"].tobytes()
    assert dvb(r1[:9]) == [] and dvb(r2[:9]) == []


# ------------------------------------------------------------------------------------------------ reachability through the trainer's argv
def test_reachability_t_b_2_0_through_the_trainer_argv_changes_the_batch_gt_and_the_built_levers_read_2_0(pack, monkeypatch):
    n = _count_labellers(monkeypatch)
    base = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1"])
    spelled = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-kin-tb", "1.0"])
    two = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-kin-tb", "2.0"])
    monkeypatch.undo()
    assert n == {"kin_label_bank": 3, "label_bank": 0}
    lv0, lv1, lv2 = (x[5].built_levers() for x in (base, spelled, two))
    assert lv0["dagger_kin_tb"] == lv1["dagger_kin_tb"] == 1.0 and lv2["dagger_kin_tb"] == 2.0
    assert dvb(base[:9]) == [] and dvb(spelled[:9]) == [] and dvb(two[:9]) == []
    bad = dvb(two[:9], dag=base[5])                                                         # declared 2.0, built 1.0: the gate refuses, naming the lever (and the other way round)
    assert any("lever dagger_kin_tb: declared 2.0 but built 1.0" in m for m in bad), bad
    bad = dvb(base[:9], dag=two[5])
    assert any("lever dagger_kin_tb: declared 1.0 but built 2.0" in m for m in bad), bad
    for k in ("slot", "label", "has"):                                                      # spelling the default out changes nothing
        assert base[5].bank.rl[k].tobytes() == spelled[5].bank.rl[k].tobytes(), k
    assert two[5].bank.kin_tb == 2.0 and base[5].bank.kin_tb == 1.0
    assert two[5].bank.rl["kin"].any(), "the fixture bank must carry kin_v1 records, else the gt comparison below is vacuous"
    a, cfg, _, _, info, dag2, J, wins, clips, store = two
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    el = [x for x in wins if dag2.bank.candidates([x[0]], [x[1]])[0][0] > 0][:48]
    b = D.collate([ds.item(x[0], x[1], 1, 3) for x in el])
    o1, o2 = base[5].apply(b, 1), dag2.apply(b, 1)
    assert dag2.last["n_label_kin"] > 0 and base[5].last["n_label_kin"] > 0
    assert not torch.equal(o1["gt"], o2["gt"]), "T_B 2.0 reaches the batch: gt differs from the T_B 1.0 batch"
    kin_rows = (o1["gt"] != o2["gt"]).flatten(1).any(1)
    assert int(kin_rows.sum()) >= 1
    ob = base[5].apply(b, 1)                                                                # the 1.0 side is deterministic: the same batch again equals itself
    assert torch.equal(o1["gt"], ob["gt"])
    s1 = spelled[5].apply(b, 1)
    assert _same(o1, s1), "--dagger-kin-tb 1.0 spelled out is the same batch as the flag absent"


def test_the_summary_and_the_selfcheck_carry_the_lever_through_main(pack, tmp_path):
    out = tmp_path / "run2"
    assert TR.main(_run(out, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-kin-tb", "2.0", "--steps", "2"])) == 0
    assert _summary(out)["levers"]["dagger"]["dagger_kin_tb"] == 2.0
    out0 = tmp_path / "run1"
    assert TR.main(_run(out0, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--steps", "2"])) == 0
    assert _summary(out0)["levers"]["dagger"]["dagger_kin_tb"] == 1.0
    with pytest.raises(SystemExit):                                                          # speed_v1 with 2.0: the trainer stops before the gate
        TR.main(_run(tmp_path / "run3", [*DAG(pack, 1.0, 3), "--dagger-label", "speed_v1", "--dagger-kin-tb", "2.0", "--steps", "2"]))
