"""PREREG_V8F_ROLLADD, the arm RA: ``--dagger-mode {swap,add}``. ``add`` keeps the logged batch whole and APPENDS the N swapped rows. CPU only, on the synthetic world of the dagger tests.

Run the WHOLE file (module fixtures are shared): CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_rolladd.py

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): every claim has a test that goes RED when the single line that implements it is deleted or bent (``mutants/run_mutants.py``).

  gold          the default (``swap``) batch equals the code of record cf27dbb7 (``golden_v8f_dagger_cf27dbb7.py`` = that commit's blob, sha256 pinned): every key, bit for bit
  add           the first B rows are the logged batch bit for bit; rows B .. B+N-1 equal the swap mode's rows for the same selection; ``n_appended`` == ``n_selected``, counted from the returned batch
  refusal       ``add`` at --dagger-p 0 and beside every stage that assumes B rows is refused by the gate (G-DVB)
  reachability  argv -> DaggerSwap(mode) -> the batch the LOSS receives has B + N rows -> train_log; the probe shows a NON-ZERO gradient difference against the logged batch and EXACTLY 0.0 at p = 0
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_dagger import world, fx10, make_batch, eligible_windows, ineligible_windows  # noqa: E402,F401
from test_v8f_dagger_train import pack, probe_parts, gate_parts, DAG, _run, _log, _summary, COMMON  # noqa: E402,F401

GOLD = HERE / "golden_v8f_dagger_cf27dbb7.py"
GOLD_BLOB_SHA256 = "6a9d50be211082221633b89391d1356154e41913d483b048220d271dbcc14553"      # sha256 of `git show cf27dbb7:stack/tanitad/data/v8f_dagger.py`
torch.set_num_threads(1)


def _load_gold():
    spec = importlib.util.spec_from_file_location("golden_v8f_dagger_cf27", GOLD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _same(a, b):
    if torch.is_tensor(a):
        return a is b or (a.shape == b.shape and a.dtype == b.dtype and bool(((a == b) | (torch.isnan(a) & torch.isnan(b)) if a.is_floating_point() else (a == b)).all()))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(np.all(a == b)) if isinstance(a, np.ndarray) else a == b


def _rows(x, lo, hi=None):
    return {k: v[lo:hi] for k, v in x.items()} if isinstance(x, dict) else x[lo:hi]


def _sel_rows(x, rows):
    return {k: v[rows] for k, v in x.items()} if isinstance(x, dict) else x[rows]


# ------------------------------------------------------------------------------------------------ the flag, and gold
def test_the_flag_exists_defaults_to_swap_and_has_a_declared_vs_built_entry():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.dagger_mode == "swap" == G.DEFAULT_MODE and G.DAGGER_MODES == ("swap", "add")
    assert TR.FLAG_CHECKS["dagger_mode"] == "dagger:dagger_mode"
    assert {act.dest for act in TR.build_parser()._actions if act.dest != "help"} == set(TR.FLAG_CHECKS)
    assert G.DaggerSwap().mode == "swap" and G.DaggerSwap().built_levers()["dagger_mode"] == "swap"
    assert G.DaggerSwap(mode="add").built_levers()["dagger_mode"] == "add"
    assert "mode" not in G.constants() and "mode" not in G.DAGGER_DEFAULTS, "the bank header's constants_sha256 must not move"


def test_the_gold_copy_is_the_commits_blob():
    assert hashlib.sha256(GOLD.read_bytes()).hexdigest() == GOLD_BLOB_SHA256


@pytest.mark.parametrize("p,seed,step", [(1.0, 5, 3), (0.5, 1, 4), (0.3, 7, 11)])
def test_gold_default_swap_equals_the_code_of_record_bit_for_bit(fx10, p, seed, step):
    gold = _load_gold()
    wins = eligible_windows(fx10, 24)
    b = make_batch(fx10, wins)
    gbank = gold.DaggerBank(fx10.bank.a, fx10.bank.header, clip_names=fx10.J["clip"], manifest=fx10.bank.manifest, sha256=fx10.bank.sha256, path=fx10.bank.path)
    old = gold.DaggerSwap(gbank, p=p, seed=seed).apply(b, step)
    for new in (G.DaggerSwap(fx10.bank, p=p, seed=seed).apply(b, step), G.DaggerSwap(fx10.bank, p=p, seed=seed, mode="swap").apply(b, step)):
        assert set(new) == set(old) == set(b)
        for k in b:
            assert _same(new[k], old[k]), f"{k} differs from the code of record under the default mode"
            assert new[k].shape[0] == b[k].shape[0] if torch.is_tensor(new[k]) else True
    sw = G.DaggerSwap(fx10.bank, p=p, seed=seed)
    sw.apply(b, step)
    assert sw.last["n_appended"] == 0 and sw.last["n_rows_out"] == len(wins) and sw.last["n_selected"] > 0, "swap: exactly 0 rows appended, the row count is B"


# ------------------------------------------------------------------------------------------------ add on a real batch
@pytest.mark.parametrize("p,seed,step", [(1.0, 5, 3), (0.5, 1, 4), (0.3, 7, 11)])
def test_add_keeps_the_logged_rows_bit_for_bit_and_appends_the_swap_modes_rows(fx10, p, seed, step):
    wins = eligible_windows(fx10, 24)
    b = make_batch(fx10, wins)
    B = len(wins)
    sw = G.DaggerSwap(fx10.bank, p=p, seed=seed, mode="swap")
    ref = sw.apply(b, step)
    rows = sw.last["sel"].nonzero(as_tuple=True)[0]
    N = int(rows.numel())
    assert N >= 1
    ad = G.DaggerSwap(fx10.bank, p=p, seed=seed, mode="add")
    out = ad.apply(b, step)
    assert set(out) == set(b)
    assert ad.last["n_selected"] == sw.last["n_selected"] == N and ad.last["n_appended"] == N and ad.last["n_rows_out"] == B + N and ad.last["n_rows"] == B
    assert torch.equal(ad.last["sel"], sw.last["sel"]), "the SAME selection"
    for k, v in b.items():
        assert _same(_rows(out[k], 0, B), v), f"the first {B} rows of {k} are not the logged batch"
        assert _same(_rows(out[k], B), _sel_rows(ref[k], rows)), f"the appended rows of {k} are not the swap mode's rows"
    for nm in ("lab", "lab_t", "lab_s"):
        assert all(out[nm][kk].shape[0] == B + N for kk in out[nm]), f"{nm}: an entry was not appended"
    assert bool((out["kin3"][B:] != b["kin3"][rows]).any()) and bool((out["p10f"][B:] != b["p10f"][rows]).any()), "the appended rows hold the OWN state"
    assert torch.equal(out["gt"][B:], b["gt"][rows]) and torch.equal(out["ci"][B:], b["ci"][rows]) and torch.equal(out["t"][B:], b["t"][rows]), "targets and ids are the window's logged ones"


def test_add_with_no_selected_row_returns_the_logged_rows_and_counts_zero(fx10):
    wins = ineligible_windows(fx10, 12)
    b = make_batch(fx10, wins)
    ad = G.DaggerSwap(fx10.bank, p=1.0, seed=3, mode="add")
    out = ad.apply(b, 0)
    assert ad.last["n_selected"] == 0 and ad.last["n_appended"] == 0 and ad.last["n_rows_out"] == len(wins)
    assert all(_same(out[k], b[k]) for k in b)
    off = G.DaggerSwap(fx10.bank, p=0.0, seed=3, mode="add")
    assert off.apply(b, 0) is b and off.last["n_appended"] == 0


# ------------------------------------------------------------------------------------------------ refusals (G-DVB)
def _parts(pack, extra=()):
    return gate_parts(pack, DAG(pack, 0.1, 7) + list(extra))


def test_add_alone_passes_the_gate_and_is_read_back(pack):
    a, cfg, model, opt, info, dag = _parts(pack, ["--dagger-mode", "add"])[:6]
    assert dag.mode == "add" and dag.built_levers()["dagger_mode"] == "add"
    assert TR.bad_dagger(a, dag, cfg, info) == [], "the control: add with the RH-shaped levers passes"
    a2 = copy.copy(a)
    a2.dagger_mode = "swap"
    assert any("lever dagger_mode: declared 'swap' but" in m for m in TR.bad_dagger(a2, dag, cfg, info)), "declared swap, built add: read back and refused"
    a3, cfg3, _, _, info3, dag3 = _parts(pack)[:6]
    assert TR.bad_dagger(a3, dag3, cfg3, info3) == [] and dag3.mode == "swap"
    a4 = copy.copy(a3)
    a4.dagger_mode = "add"
    assert any("lever dagger_mode: declared 'add' but" in m for m in TR.bad_dagger(a4, dag3, cfg3, info3)), "declared add, built swap: refused"


def test_add_at_p0_is_refused_a_lever_that_does_nothing(pack):
    a = TR.build_parser().parse_args(["--out", "x", *["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"], "--dagger-mode", "add"])
    dag = G.DaggerSwap(None, p=0.0, mode="add")
    bad = TR.bad_dagger(a, dag, None, None)
    assert any("dagger-mode" in m and "does nothing" in m for m in bad), bad
    assert any("dagger_mode 'add' is declared while dagger_p is 0" in m for m in G.DaggerSwap(None, p=0.0, mode="add").problems())
    assert G.DaggerSwap(None, p=0.0, mode="swap").problems() == []
    assert any("not one of" in m for m in G.DaggerSwap(None, p=0.5, mode="sideways").law_problems())


@pytest.mark.parametrize("extra,needle", [
    (["--launch-intent-w", "1.0"], "--launch-intent-w"), (["--level-head", "on", "--w-level", "0.1"], "--level-head on"), (["--lead-head", "on", "--w-lead", "0.1"], "--lead-head on"),
    (["--ego-perturb-p", "0.3"], "--ego-perturb-p"), (["--recover-p", "0.3"], "--recover-p"), (["--t2-bank-p", "0.3"], "--t2-bank-p"),
    (["--wm-windows", "wmonly"], "--wm-windows wmonly")])
def test_add_is_refused_beside_a_stage_that_assumes_b_rows(pack, extra, needle):
    a, cfg, model, opt, info, dag = _parts(pack, ["--dagger-mode", "add"])[:6]
    b = copy.copy(a)
    for k, v in vars(TR.build_parser().parse_args(["--out", "x", *extra])).items():
        if k not in ("out",) and v != TR.build_parser().get_default(k):
            setattr(b, k, v)
    bad = TR.bad_dagger(b, dag, cfg, info)
    assert any("--dagger-mode add together with " + needle in m for m in bad), bad


def test_add_is_refused_beside_compose_lora_and_wm_only(pack):
    a, cfg, model, opt, info, dag = _parts(pack, ["--dagger-mode", "add"])[:6]
    for attr, val, needle in (("ego_perturb_with_dagger", True, "--ego-perturb-with-dagger"), ("encoder", "lora", "--encoder lora"), ("wm_only", True, "--wm-only")):
        b = copy.copy(a)
        setattr(b, attr, val)
        assert any("--dagger-mode add together with " + needle in m for m in TR.bad_dagger(b, dag, cfg, info)), attr


# ------------------------------------------------------------------------------------------------ reachability and the gradient
def test_the_add_probe_passes_shows_a_nonzero_gradient_difference_and_exact_zero_controls(pack):
    model, b, cfg, dag = probe_parts(pack)
    dag = G.DaggerSwap(dag.bank, p=dag.p, seed=dag.seed, kin_mode=dag.kin_mode, mode="add")
    res = TR.dagger_probe(model, b, cfg, dag)
    assert res["pass"] is True and res["mode"] == "add" and res["rows_appended"] == res["n_selected_at_p1"] == res["n_eligible"] > 1
    assert res["rows_out"] == res["rows_logged"] + res["rows_appended"]
    assert res["grad_diff_all"] > 0.0 and res["loss_diff"] != 0.0, "the appended rows reach the loss the trainer backpropagates"
    assert res["control_repeat_loss_diff"] == 0.0 and res["control_repeat_grad_diff"] == 0.0
    assert res["control_p0_same_object"] is True and res["control_p0_loss_diff"] == 0.0 and res["control_p0_grad_diff"] == 0.0
    assert res["control_prior_grad_diff"] == 0.0 and res["live_added"]["pass"] is True and res["live_added"]["control_prior_grad_sum"] == 0.0
    assert res["appended_differs_from_logged"] == {"p10f": True, "kin3": True}
    swap = TR.dagger_probe(model, b, cfg, G.DaggerSwap(dag.bank, p=dag.p, seed=dag.seed, kin_mode=dag.kin_mode))
    assert swap["pass"] is True and "mode" not in swap, "the swap probe is the untouched one"


def test_the_add_probe_refuses_a_batch_with_no_appended_row(pack, monkeypatch):
    model, b, cfg, dag = probe_parts(pack)
    dag = G.DaggerSwap(dag.bank, p=dag.p, seed=dag.seed, kin_mode=dag.kin_mode, mode="add")
    monkeypatch.setattr(G, "append_rows", lambda batch, rows, upd: dict(batch))
    with pytest.raises(AssertionError, match="the add batch is not B \\+ N rows"):
        TR.dagger_probe(model, b, cfg, dag)


def test_the_add_probe_refuses_appended_logged_rows_instead_of_the_swapped_ones(pack, monkeypatch):
    model, b, cfg, dag = probe_parts(pack)
    dag = G.DaggerSwap(dag.bank, p=dag.p, seed=dag.seed, kin_mode=dag.kin_mode, mode="add")
    real = G.append_rows
    monkeypatch.setattr(G, "append_rows", lambda batch, rows, upd: real(batch, rows, {}))            # upd empty: every appended row is the LOGGED one
    with pytest.raises(AssertionError, match="appended rows|appended own states equal the logged"):
        TR.dagger_probe(model, b, cfg, dag)


def _train_with_spy(pack, tmp_path, extra, monkeypatch):
    seen = []
    orig = M.V8FMini.loss

    def spy(self, batch, *a, **kw):
        seen.append(int(batch["feats"].shape[0]))
        return orig(self, batch, *a, **kw)
    monkeypatch.setattr(M.V8FMini, "loss", spy)
    out = tmp_path / "run"
    assert TR.main(_run(out, [*DAG(pack, 0.5, 3), *extra, "--steps", "3"])) == 0
    return out, seen


def test_end_to_end_the_loss_receives_b_plus_n_rows_and_the_log_counts_them(pack, tmp_path, monkeypatch):
    out, seen = _train_with_spy(pack, tmp_path, ["--dagger-mode", "add"], monkeypatch)
    log = [r for r in _log(out) if "dagger_appended" in r]
    assert len(log) >= 3 and all(r["dagger_appended"] == r["dagger_n"] for r in log), "every log line: appended == selected"
    assert sum(r["dagger_appended"] for r in log) > 0
    assert sum(r["dagger_rows_out"] for r in log) == sum(r["dagger_rows"] for r in log) + sum(r["dagger_appended"] for r in log)
    loss_calls = seen[-len(log):]                                                    # the training loop's own loss calls (the selfcheck/probe calls come before)
    assert [r["dagger_rows_out"] for r in log] == loss_calls and max(loss_calls) > 16, (loss_calls, "the loss received B + N rows (B = 16)")
    s = _summary(out)
    assert s["dagger_appended"] == sum(r["dagger_appended"] for r in log) > 0 and s["dagger_rows_out"] == sum(loss_calls)
    assert s["levers"]["dagger"]["dagger_mode"] == "add"


def test_end_to_end_the_default_trains_on_b_rows_and_logs_no_add_keys(pack, tmp_path, monkeypatch):
    out, seen = _train_with_spy(pack, tmp_path, [], monkeypatch)
    log = _log(out)
    assert all("dagger_appended" not in r for r in log) and "dagger_appended" not in _summary(out), "a swap run keeps today's key set"
    assert set(seen[-3:]) == {16} and _summary(out)["levers"]["dagger"]["dagger_mode"] == "swap" and _summary(out)["dagger_n"] > 0


# ------------------------------------------------------------------------------------------------ MM request: the size of the SECOND difference (SIGReg x n over near-duplicate windows). Measured, no bar, the model's loss is untouched.
def test_the_wm_and_sigreg_terms_under_swap_and_add_are_recorded(pack, tmp_path):
    """Records, per batch, the ``wm`` term, its SIGReg part (sum over levels of beta x sigreg/L) and their share of ``total`` on (a) the logged batch (B rows), (b) the swap batch (B rows), (c) the add batch (B + N rows), the
    same selection, the same seeds. SYNTHETIC rig and an UNTRAINED tiny model: this shows the mechanism and its size at init, not the real-batch number (that needs the pod's GPU and a trained checkpoint; not run).
    Output: $RA_CENSUS_WM_OUT (a json path) when set. The only asserts are sanity ones: everything finite, the add batch has more rows, and the logged batch equals the add batch's first B rows."""
    import os
    rows = []
    for seed in range(6):
        model, b, cfg, dag = probe_parts(pack, p=0.16, n=64, seed=seed)
        model.train()
        sw = G.DaggerSwap(dag.bank, p=0.16, seed=1, kin_mode="yaw", mode="swap")
        ad = G.DaggerSwap(dag.bank, p=0.16, seed=1, kin_mode="yaw", mode="add")
        bs, ba = sw.apply(b, 0), ad.apply(b, 0)
        N = ad.last["n_appended"]

        def measure(batch):
            torch.manual_seed(1234)
            with torch.no_grad():
                out = model.forward(batch, p_gt=0.5, gen=torch.Generator().manual_seed(13))
                torch.manual_seed(77)
                wm = model.wm(out["adapt"], batch["act"])
                torch.manual_seed(1234)
                T = model.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
            beta = cfg.wm_beta
            sg = float(sum(beta * wm[k] for k in wm if k.startswith("sigreg/")))
            return {"rows": int(batch["feats"].shape[0]), "wm": float(T["wm"]), "wm_weighted": float(cfg.w_wm * T["wm"]), "sigreg_weighted_sum": float(cfg.w_wm * sg), "total": float(T["total"]),
                    "wm_share_of_total": float(cfg.w_wm * T["wm"] / T["total"]), "sigreg_share_of_total": float(cfg.w_wm * sg / T["total"])}
        r = {"seed": seed, "N": int(N), "logged": measure(b), "swap": measure(bs), "add": measure(ba)}
        assert all(np.isfinite(v) for m in ("logged", "swap", "add") for v in r[m].values()) and r["add"]["rows"] == 64 + N and r["swap"]["rows"] == 64
        rows.append(r)
    summ = {k: {m: float(np.mean([r[m][k] for r in rows])) for m in ("logged", "swap", "add")} for k in ("wm", "sigreg_weighted_sum", "total", "wm_share_of_total", "sigreg_share_of_total")}
    out = {"rig": "SYNTHETIC world, UNTRAINED tiny model (d 32, hmax 16, wm b), B = 64, p = 0.16, 6 random batches", "per_batch": rows, "mean": summ, "w_wm": float(cfg.w_wm), "wm_beta": float(cfg.wm_beta)}
    p = os.environ.get("RA_CENSUS_WM_OUT")
    if p:
        Path(p).write_text(json.dumps(out, indent=1), encoding="utf-8")
