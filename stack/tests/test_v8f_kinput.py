"""KINPUT (BRIEF_BUILD_KINPUT, the input-side lever of the KINLAB card): ``--dagger-base own|logged`` (default ``own`` = commit fde8b716 bit for bit).

``logged``: in a swapped own sample O's prior path ``base`` is the LOGGED window's ``base`` at the matched row r' (``kdx_base`` of the logged kinematics at r'); the own pose / kin history (p10f, p10_t, kin3, poses, nav) and every
label stay as today. What is pinned here (``mutants_kinput/MUTANTS_RESULT.json`` turns each RED):
  * DEFAULT, GOLD: the flag absent / ``own`` = the golden copy of fde8b716's ``v8f_dagger.py`` (batch, labels, built levers apart from the new key ``dagger_base``), under human, speed_v1 and kin_v1 labels;
  * ``logged`` MOVES ONLY ``base``: on a planted bank the swapped rows' ``base`` equals ``kdx_base`` of the logged kinematics at r' EXACTLY (read from the clip store), every other key (the own p10f / kin3 / p10_t / poses / nav, the
    labels) equals the ``own`` batch, and the rows the swap did not select are the logged batch's;
  * REFUSALS: ``logged`` with ``--dagger-p 0``, with ``--dagger-mode add``, an unknown value;
  * REACHABILITY through the trainer's argv: the batch's ``base`` differs from the ``own`` batch on at least one swapped row, the plan loss differs, the built levers read ``logged``, the gate passes, G-LIVE (``dagger_probe``) passes.
Run the WHOLE file (module fixtures are shared): CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_kinput.py"""
from __future__ import annotations

import hashlib
import importlib.util
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
from test_v8f_dagger_train import pack, DAG, TR, DG, dvb, probe_parts, GATE_ARGV  # noqa: E402,F401
from test_v8f_recovery import eligible_windows_of, _same  # noqa: E402,F401
from test_v8f_recovery_wiring import gate_parts_rl  # noqa: E402,F401
from test_v8f_kinlab import kin_bank  # noqa: E402,F401
from test_v8f_kincache import _clone  # noqa: E402,F401

torch.set_num_threads(1)
GOLD_D = HERE / "golden_v8f_dagger_fde8b716.py"
GOLD_D_SHA256 = "9976718b341f8a1f70a0c1f1451f8472edda970955838c4a09dde899b205f0b2"      # sha256 of `git show fde8b716:stack/tanitad/data/v8f_dagger.py`
KINDS = ("human", "speed_v1", "kin_v1")


@pytest.fixture(scope="module")
def gold():
    assert hashlib.sha256(GOLD_D.read_bytes()).hexdigest() == GOLD_D_SHA256
    spec = importlib.util.spec_from_file_location("golden_v8f_dagger_fde8b716", GOLD_D)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def planted(f, dv=0.8, da=0.5, dk=0.02):
    """The fixture bank with EVERY ok record's own kin3 moved off the logged one (so the own base and the logged base differ), p10f untouched."""
    arrays = {k: np.array(v) for k, v in f.bank.a.items()}
    ok = arrays["reason"] == G.OK
    arrays["kin3"] = np.array(arrays["kin3"], np.float64)
    arrays["kin3"][ok] += np.array([dv, da, dk])
    return G.DaggerBank(arrays, f.bank.header, clip_names=f.J["clip"], manifest=f.bank.manifest, sha256=f.bank.sha256, path=f.bank.path)


def with_label(f, bank, kind):
    bn = _clone(f, bank)
    if kind != "human":
        res = RV.label_bank(bn.a, bn.ci, f.J["clip"], f.store.get_kin) if kind == "speed_v1" else RV.kin_label_bank(bn.a, bn.ci, f.J["clip"], f.store.get_kin)
        bn.attach_recovery(res["slot"], res["label"], res["status_counts"], **({"kin": res["is_kin"]} if kind == "kin_v1" else {}))
    return bn


def mixed_batch(f, bank, n_other=6):
    """Eligible windows (a swap can pick them) AND windows the swap cannot select (logged rows that must come back unchanged)."""
    el = eligible_windows_of(f, bank)
    other = [w for w in f.wins if bank.candidates([w[0]], [w[1]])[0][0] == 0][:n_other]
    assert len(el) >= 4 and len(other) >= 1
    return make_batch(f, el + other), len(el)


# ------------------------------------------------------------------------------------------------ the default is fde8b716, bit for bit
def test_the_flag_exists_defaults_to_own_and_has_a_declared_vs_built_entry():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.dagger_base == "own" == G.DEFAULT_BASE and G.DAGGER_BASES == ("own", "logged") and TR.FLAG_CHECKS["dagger_base"] == "dagger:dagger_base"
    assert {act.dest for act in TR.build_parser()._actions if act.dest != "help"} == set(TR.FLAG_CHECKS)
    assert TR.build_parser().parse_args(["--out", "x", "--dagger-base", "logged"]).dagger_base == "logged"
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--out", "x", "--dagger-base", "mixed"])


@pytest.mark.parametrize("kind", KINDS)
def test_default_swap_and_built_levers_equal_fde8b716_with_the_flag_absent_and_own(fx10, gold, kind):
    bn = with_label(fx10, planted(fx10), kind)
    bg = gold.DaggerBank({k: np.array(v) for k, v in bn.a.items()}, bn.header, clip_names=fx10.J["clip"], manifest=bn.manifest, sha256=bn.sha256, path=bn.path)
    if kind != "human":
        bg.attach_recovery(bn.rl["slot"], bn.rl["label"], bn.rl["status_counts"], **({"kin": bn.rl["kin"]} if kind == "kin_v1" else {}))
        bg.kin_tb = bn.kin_tb
    batch, _ = mixed_batch(fx10, bn)
    for step in (0, 5):
        ref = gold.DaggerSwap(bg, p=0.7, seed=3).apply(batch, step)
        for sw in (G.DaggerSwap(bn, p=0.7, seed=3), G.DaggerSwap(bn, p=0.7, seed=3, base="own")):
            out = sw.apply(batch, step)
            assert _same(out, ref), (kind, step)
    lv_n, lv_g = G.DaggerSwap(bn, p=0.7, seed=3).built_levers(), gold.DaggerSwap(bg, p=0.7, seed=3).built_levers()
    assert lv_n.pop("dagger_base") == "own" and "dagger_base" not in lv_g and lv_n == lv_g
    assert G.DaggerSwap().built_levers()["dagger_base"] == "own" and G.DaggerSwap(None, p=0.0, base="logged").built_levers()["dagger_base"] == "logged"


# ------------------------------------------------------------------------------------------------ logged moves only base
@pytest.mark.parametrize("kind", KINDS)
def test_logged_moves_only_base_and_it_is_kdx_base_of_the_logged_kinematics_at_r_prime(fx10, kind):
    bn = with_label(fx10, planted(fx10), kind)
    batch, n_el = mixed_batch(fx10, bn)
    own_sw, log_sw = G.DaggerSwap(bn, p=0.7, seed=3), G.DaggerSwap(bn, p=0.7, seed=3, base="logged")
    n_sel_total = n_diff_rows = 0
    for step in (0, 1, 2):
        o_own, o_log = own_sw.apply(batch, step), log_sw.apply(batch, step)
        sel = own_sw.last["sel"]
        assert bool((sel == log_sw.last["sel"]).all()) and 0 < int(sel.sum()) < len(sel), "the same rows are selected, some are not"
        assert set(o_own) == set(o_log) == set(batch)
        for k in batch:                                                                       # every key but `base` is the own-mode batch, bit for bit (p10f / kin3 / p10_t / poses / nav / labels included)
            if k != "base":
                assert _same(o_log[k], o_own[k]), (kind, step, k)
        idx = np.asarray(own_sw.last["record_idx"])
        rec = bn.gather(idx)
        rows = sel.nonzero(as_tuple=True)[0]
        want_log = []
        for n, i in enumerate(rows.tolist()):                                                 # the logged kinematics at r' straight from the clip store (float64 kin3 at the matched row)
            clip, r = str(fx10.J["clip"][int(batch["ci"][i])]), int(rec["r"][n])
            assert r == int(batch["t"][i])
            want_log.append(D.kdx_base(fx10.store.get_kin(clip)["kin3"][r:r + 1], D.KIN_UNITS["yaw"])[0])
        want_log = torch.from_numpy(np.stack(want_log))
        want_own = torch.from_numpy(D.kdx_base(rec["kin3"], D.KIN_UNITS["yaw"]))
        assert torch.equal(o_log["base"][rows], want_log), "logged: base = kdx_base(logged kinematics at r') exactly"
        assert torch.equal(o_log["base"][rows], batch["base"][rows]), "and it is the logged window's own base"
        assert torch.equal(o_own["base"][rows], want_own), "own: kdx_base(own kinematics), as today"
        assert torch.equal(o_log["base"][~sel], batch["base"][~sel]) and torch.equal(o_own["base"][~sel], batch["base"][~sel]), "unselected rows are the logged batch's"
        for k in ("p10f", "kin3", "p10_t"):                                                   # the own state is still in place under `logged`
            assert not torch.equal(o_log[k][rows], batch[k][rows]), k
        for k in ("p10f", "kin3", "p10_t", "pose", "pose_t", "pose_s", "nav"):
            assert torch.equal(o_log[k][~sel], batch[k][~sel]), k
        n_diff_rows += int((o_own["base"][rows] != o_log["base"][rows]).flatten(1).any(1).sum())
        n_sel_total += int(sel.sum())
    assert n_diff_rows >= 1 and n_sel_total >= 3, "the planted own state makes the two bases differ on at least one swapped row (else this test is vacuous)"


def test_logged_changes_the_loss_input_only_through_base_and_leaves_the_batch_object_untouched(fx10):
    bn = planted(fx10)
    batch, _ = mixed_batch(fx10, bn)
    snap = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    G.DaggerSwap(bn, p=1.0, seed=3, base="logged").apply(batch, 0)
    for k, v in snap.items():
        assert _same(batch[k], v), k


# ------------------------------------------------------------------------------------------------ refusals
def test_logged_is_refused_with_p_0_with_add_mode_and_an_unknown_value(fx10):
    bn = planted(fx10)
    batch, _ = mixed_batch(fx10, bn)
    sw = G.DaggerSwap(bn, p=0.0, base="logged")
    assert any("dagger_base 'logged' is declared while dagger_p is 0" in m for m in sw.problems()), sw.problems()
    sw = G.DaggerSwap(bn, p=1.0, base="logged", mode="add")
    assert any("dagger_base 'logged' needs --dagger-mode swap" in m for m in sw.problems()), sw.problems()
    with pytest.raises(ValueError, match="needs --dagger-mode swap"):
        sw.apply(batch, 0)
    assert any("dagger_base 'mixed' is not one of" in m for m in G.DaggerSwap(bn, p=1.0, base="mixed").problems())
    with pytest.raises(ValueError):
        G.DaggerSwap(bn, p=1.0, base="mixed").apply(batch, 0)
    with pytest.raises(ValueError):
        G.swap_rows(batch, torch.zeros(1, dtype=torch.long), bn.gather([int(bn._ok_idx[0])]), base="mixed")
    assert G.DaggerSwap(bn, p=1.0, base="own", mode="add").problems() == []                  # the default is not a lever


def test_the_trainer_refuses_logged_with_p_0_and_with_add_mode(pack):
    with pytest.raises(SystemExit) as e:
        gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-mode", "add", "--dagger-base", "logged"])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and "--dagger-base" in str(e.value) and "--dagger-mode swap" in str(e.value), str(e.value)
    gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-mode", "add", "--dagger-base", "own"])             # the default spelled out is not a lever
    r = gate_parts_rl(pack, ["--init-ckpt", str(pack.ck), "--dagger-p", "0", "--dagger-base", "logged"])
    msgs = dvb(r[:9])
    assert any("dagger_base" in m or "dagger-base" in m for m in msgs), msgs
    a = TR.build_parser().parse_args(["--out", str(pack.init_out), *GATE_ARGV, "--dagger-base", "logged"])    # no swap built at all
    assert any("dagger_base" in m and "no DAgger swap was built" in m for m in TR.bad_dagger(a, None))
    a = TR.build_parser().parse_args(["--out", "x", "--dagger-p", "0.5", "--dagger-mode", "add", "--dagger-base", "logged"])
    assert "--dagger-mode swap" in TR.dagger_base_refusal(a)
    assert TR.dagger_base_refusal(TR.build_parser().parse_args(["--out", "x"])) is None


# ------------------------------------------------------------------------------------------------ reachability through the trainer's argv
def test_reachability_logged_through_the_trainer_argv_changes_the_batch_base_and_the_loss_and_the_built_levers_read_logged(pack):
    own = gate_parts_rl(pack, [*DAG(pack, 1.0, 3)])
    spelled = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-base", "own"])
    logged = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-base", "logged"])
    lv0, lv1, lv2 = (x[5].built_levers() for x in (own, spelled, logged))
    assert lv0["dagger_base"] == lv1["dagger_base"] == "own" and lv2["dagger_base"] == "logged"
    assert dvb(own[:9]) == [] and dvb(spelled[:9]) == [] and dvb(logged[:9]) == []
    bad = dvb(logged[:9], dag=own[5])                                                       # declared logged, built own: the gate refuses, naming the lever (and the other way round)
    assert any("lever dagger_base: declared 'logged' but built 'own'" in m for m in bad), bad
    bad = dvb(own[:9], dag=logged[5])
    assert any("lever dagger_base: declared 'own' but built 'logged'" in m for m in bad), bad
    a, cfg, model, _, info, dag_l, J, wins, clips, store = logged
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    el = [x for x in wins if dag_l.bank.candidates([x[0]], [x[1]])[0][0] > 0][:48]
    b = D.collate([ds.item(x[0], x[1], 1, 3) for x in el])
    o_own, o_log = own[5].apply(b, 1), dag_l.apply(b, 1)
    sel = dag_l.last["sel"]
    assert int(sel.sum()) >= 2
    assert _same(o_own, spelled[5].apply(b, 1)), "--dagger-base own spelled out is the same batch as the flag absent"
    diff = (o_own["base"] != o_log["base"]).flatten(1).any(1)
    assert int(diff.sum()) >= 1, "logged reaches the batch: base differs from the own batch on at least one swapped row"
    assert not bool(diff[~sel].any()) and torch.equal(o_log["base"][sel], b["base"][sel])
    for k in b:
        if k != "base":
            assert _same(o_own[k], o_log[k]), k

    def total(bb):
        model.eval()
        torch.manual_seed(20261123)                                                           # the loss draws from the global RNG: seed it so equal batches give equal totals
        with torch.no_grad():
            return float(model.loss(bb, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]["total"])
    assert total(o_own) != total(o_log), "the plan loss differs between the own-base and the logged-base batch"
    assert total(o_log) == total(dict(o_log))                                                # deterministic: the difference above is the lever


def test_g_live_dagger_probe_passes_under_logged_and_reads_the_logged_base(pack):
    model, batch, cfg, dag = probe_parts(pack, p=0.5, n=24)
    res_own = TR.dagger_probe(model, batch, cfg, dag)
    assert res_own["pass"] is True and "base_source" not in res_own
    dag_l = DG.DaggerSwap(dag.bank, p=0.5, seed=1, kin_mode="yaw", base="logged")
    res = TR.dagger_probe(model, batch, cfg, dag_l)
    assert res["pass"] is True and res["base_source"] == "logged" and res["rows_changed"]["base"] == 0 and res["rows_changed"]["p10f"] >= 1
    assert res["loss_diff"] != 0.0 and res["control_p0_loss_diff"] == 0.0


def test_the_summary_and_the_selfcheck_carry_the_lever_through_main(pack, tmp_path):
    from test_v8f_dagger_train import _run, _summary
    out = tmp_path / "run2"
    assert TR.main(_run(out, [*DAG(pack, 1.0, 3), "--dagger-base", "logged", "--steps", "2"])) == 0
    assert _summary(out)["levers"]["dagger"]["dagger_base"] == "logged"
    out0 = tmp_path / "run1"
    assert TR.main(_run(out0, [*DAG(pack, 1.0, 3), "--steps", "2"])) == 0
    assert _summary(out0)["levers"]["dagger"]["dagger_base"] == "own"
    with pytest.raises(SystemExit):                                                          # add + logged: the trainer stops before the gate
        TR.main(_run(tmp_path / "run3", [*DAG(pack, 1.0, 3), "--dagger-mode", "add", "--dagger-base", "logged", "--steps", "2"]))
