"""PREREG_RECOVER_FT (H-RECOVER-FT-1), the TRAINER side: ``--recover-p`` / ``--recover-dev-bank`` / ``--recover-dev-bank-sha256`` / ``--recover-dev-law`` / ``--recover-target`` / ``--recover-rejoin-steps`` /
``--recover-seed`` in ``stack/scripts/v8f_mini_train.py``. The DATA library (``tanitad/data/v8f_recover.py``) is tested in ``test_v8f_recover.py``; THIS file proves it is REACHABLE from the caller -- the class
"built, tested, and unreachable from its caller" (CLAUDE.md, 2026-09-10/11) -- hop by hop, each hop with a control that must read exactly zero:

  argv  ->  make_recover (bank load: sha256, K-B3; the deviation table from THIS trainer's store)  ->  the RecoverPerturb  ->  declared_vs_built (bad_recover: G-DVB)
        ->  recover_probe (G-LIVE: inputs AND target reach the loss gradient, p = 0 reads 0.0)  ->  ``b = rcv.apply(b, step)`` in the loop  ->  the batch the LOSS receives  ->  summary.json / selfcheck.json / checkpoint

Every test builds on the trainer's own synthetic world and a bank the repo's own generator (``v8f_dagger_bank.py gen``) wrote from it (the REAL on-disk layout: sharded npz + bank.json manifest); nothing is read from a
real corpus. Each deliberate-regression arm deletes ONE line (or one rule) in a COPY of the source and requires the check to go RED.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_dagger as DG  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_recover as R  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_dagger_train import (COMMON, GATE_ARGV, TINY, SYN_T, _bank_sha, _selfcheck, _summary, _log, clone_bank, rename_clip, pack)  # noqa: E402,F401  (the synthetic world + the generator-written bank fixture)
from test_v8f_ego_perturb import _woken  # noqa: E402

torch.set_num_threads(1)
TRAIN_P = REPO / "stack" / "scripts" / "v8f_mini_train.py"
RECOVER_P = REPO / "stack" / "tanitad" / "data" / "v8f_recover.py"
APRIORI = ["--recover-p", "0.3", "--recover-dev-law", "apriori", "--recover-seed", "3"]


def REC(pk, p=0.3, seed=3, bank=None, sha=None, **kw):
    """The R arm's flags: p, the bank + its sha256, the empirical law (the default), the rejoin target with K = 10 (the defaults), the seed."""
    b = bank or pk.bank
    return ["--recover-p", str(p), "--recover-dev-bank", str(b), "--recover-dev-bank-sha256", sha or _bank_sha(b), "--recover-target", kw.get("target", "rejoin"),
            "--recover-rejoin-steps", str(kw.get("K", 10)), "--recover-seed", str(seed)]


def _run(out, extra=(), common=COMMON):
    return ["--out", str(out), *common, *extra]


def parts(pk, extra=(), out=None):
    """What ``main`` has in hand when it builds the perturber and runs the gate: (a, cfg, model, opt, info, rcv, J, wins, clips, store)."""
    a = TR.build_parser().parse_args(["--out", str(out or pk.init_out), *GATE_ARGV, *extra])
    cfg = TR.make_config(a)
    store, J, wins, clips, info = TR.build_data(a, cfg)
    info["ds_window_start"] = D.effective_window_start(cfg.hist, cfg.window_start)
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(a.seed), mu, sd)
    if a.init_ckpt:
        info["init_ckpt"] = TR.load_init_ckpt(a.init_ckpt, model, cfg, apply=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    rcv = TR.make_recover(a, J, wins, store, clips, info)
    return a, cfg, model, opt, info, rcv, J, wins, clips, store


def dvb(pt, **kw):
    a, cfg, model, opt, info, rcv = pt[:6]
    return TR.declared_vs_built(a, cfg, model, opt, info, None, None, TR.LRSchedule(a.lr, a.warmup, 10), None, None, kw.get("rec", rcv))


def probe_parts(pk, n=24, wm="b", seed=0):
    """(model, a real collated batch of the trainer's own dataset, cfg) -- hmax 16 and the world-model auxiliary, so the exact-zero WM control is live."""
    J = pk.J
    store = D.ClipStore(pk.world, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(pk.world / "native_tracks.pt"))
    wins = D.train_windows(store, J, D.select_train_clips(list(map(str, J["clip"])), 1.0))
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    batch["lab_s"]["rc_ok"] = torch.ones_like(batch["lab_s"]["rc_ok"])
    cfg = M.V8FMiniConfig(**{**TINY, "hmax": 16, "wm": wm})
    mu, sd = M.cons_stats(J)
    return _woken(M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)), batch, cfg


def small_table(n=64, seed=0):
    r = np.random.default_rng(seed)
    return R.RecoverDevTable({"dv": -r.random(n) * 3, "da": r.normal(0, 0.5, n), "dom": r.normal(0, 0.03, n), "dy": r.normal(0, 0.5, n), "dpsi": r.normal(0, 0.05, n)}, source_sha256="ab" * 32)


def table_perturb(p=0.3, seed=1, **kw):
    kw.setdefault("table", small_table())
    kw.setdefault("bank_path", "bank_x")
    kw.setdefault("bank_sha256", "ab" * 32)
    return R.RecoverPerturb(p=p, seed=seed, **kw)


# ------------------------------------------------------------------------------------------------ flags: entries, defaults, G-DVB
FLAGS = {"recover_p": ["--recover-p", "0.3"], "recover_dev_law": ["--recover-dev-law", "apriori"], "recover_target": ["--recover-target", "logged"],
         "recover_rejoin_steps": ["--recover-rejoin-steps", "5"], "recover_seed": ["--recover-seed", "7"]}
ALL = ("recover_p", "recover_dev_bank", "recover_dev_bank_sha256", "recover_dev_law", "recover_target", "recover_rejoin_steps", "recover_seed")


def test_every_recover_flag_has_a_declared_vs_built_entry_and_the_defaults_are_off():
    a = TR.build_parser().parse_args(["--out", "x"])
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    for k in ALL:
        assert TR.FLAG_CHECKS[k] == f"rec:{k}", k
    assert dests == set(TR.FLAG_CHECKS), "a flag without an entry, or an entry without a flag, is a launch blocker"
    assert (a.recover_p, a.recover_dev_bank, a.recover_dev_bank_sha256, a.recover_dev_law, a.recover_target, a.recover_rejoin_steps, a.recover_seed) == (0.0, "", "", "empirical", "rejoin", 10, 0)
    assert tuple(TR.RECOVER_DESTS) == ALL


@pytest.mark.parametrize("flag", ALL)
def test_a_flag_without_an_entry_is_refused(pack, monkeypatch, flag):
    pt = parts(pack)
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != flag})
    bad = dvb(pt)
    assert any("flags without a declared-vs-built entry" in m and flag in m for m in bad), bad


@pytest.mark.parametrize("flag", list(FLAGS))
def test_g_dvb_passes_what_was_built_and_refuses_a_mismatch_for_each_value_flag(pack, flag):
    pt = parts(pack, APRIORI if flag == "recover_p" else APRIORI + FLAGS[flag])
    assert dvb(pt) == [], "the control: what argv declared is what was built"
    a, rcv = pt[0], pt[5]
    kw = dict(p=a.recover_p, dev_law=a.recover_dev_law, target=a.recover_target, rejoin_steps=a.recover_rejoin_steps, seed=a.recover_seed, kin_mode=a.kin_mode)
    dflt = {"recover_p": ("p", 0.5), "recover_dev_law": ("dev_law", "empirical"), "recover_target": ("target", "rejoin"), "recover_rejoin_steps": ("rejoin_steps", 10), "recover_seed": ("seed", 0)}[flag]
    kw[dflt[0]] = dflt[1]
    if kw["dev_law"] == "empirical":
        kw["table"] = small_table()
        kw["bank_sha256"] = "ab" * 32
    bad = dvb(pt, rec=R.RecoverPerturb(**kw))
    assert any(f"lever {flag}:" in m and "declared" in m and "built" in m for m in bad), bad


@pytest.mark.parametrize("flag", list(FLAGS))
def test_a_declared_flag_with_no_perturber_built_is_refused(pack, flag):
    pt = parts(pack, APRIORI if flag == "recover_p" else APRIORI + FLAGS[flag])
    bad = dvb(pt, rec=None)
    assert any("no recovery perturber was built" in m and flag in m for m in bad), bad
    pt0 = parts(pack, [])
    assert dvb(pt0, rec=None) == [], "the control: all defaults, no perturber, passes (every existing caller)"


@pytest.mark.parametrize("extra, what", [
    (["--recover-p", "0.3", "--recover-dev-law", "apriori", "--ego-perturb-p", "0.3"], "--ego-perturb-p"),
    (["--recover-p", "0.3", "--recover-dev-law", "apriori", "--dagger-p", "0.1"], "--dagger-p"),
    (["--recover-p", "0.3", "--recover-dev-law", "apriori", "--ego-perturb-p", "0.3", "--dagger-p", "0.1", "--ego-perturb-with-dagger"], "exclusive"),
    (["--recover-target", "logged"], "does nothing"),                                           # a lever set while p is 0
    (["--recover-rejoin-steps", "5"], "does nothing"),
    (["--recover-seed", "3"], "does nothing"),
    (["--recover-dev-law", "apriori"], "does nothing"),
    (["--recover-dev-bank", "somewhere"], "does nothing"),
    (["--recover-dev-bank-sha256", "ab" * 32], "does nothing"),
    (["--recover-p", "0.3", "--recover-dev-law", "apriori", "--recover-dev-bank-sha256", "ab" * 32], "does nothing"),   # a bank declared with the a-priori law
    (["--recover-p", "0.3"], "--recover-dev-bank is required"),                                  # the empirical law with no bank
    (["--recover-p", "0.3", "--recover-dev-bank", "somewhere"], "--recover-dev-bank-sha256 is required"),
    (["--recover-p", "1.5", "--recover-dev-law", "apriori"], "not in [0, 1]"),
    (["--recover-p", "0.3", "--recover-dev-law", "apriori", "--recover-rejoin-steps", "0"], "recover_rejoin_steps"),
    (["--recover-p", "0.3", "--recover-dev-law", "apriori", "--recover-rejoin-steps", "31"], "recover_rejoin_steps"),
    (["--recover-p", "0.3", "--recover-dev-law", "apriori", "--kin-mode", "steer"], "needs --kin-mode yaw"),
])
def test_g_dvb_refuses_an_illegal_empty_or_exclusive_declaration(pack, extra, what):
    a = TR.build_parser().parse_args(["--out", "x", *GATE_ARGV, *extra])
    # the perturber as make_recover builds it when NO table was loaded (p = 0, the a-priori law, or a declared bank that was not read): the gate must refuse by its own message
    rcv = R.RecoverPerturb(p=a.recover_p, dev_law=a.recover_dev_law, table=None, target=a.recover_target, rejoin_steps=a.recover_rejoin_steps, seed=a.recover_seed, kin_mode=a.kin_mode,
                           bank_path=a.recover_dev_bank or None, bank_sha256=a.recover_dev_bank_sha256 or None)
    bad = TR.bad_recover(a, rcv)
    assert any(what in m for m in bad), bad


def test_make_recover_reads_no_bank_and_builds_no_table_at_p0_or_with_the_apriori_law(pack):
    for extra in ([], ["--recover-p", "0"], APRIORI):
        a = TR.build_parser().parse_args(["--out", "x", *GATE_ARGV, *extra])
        info = {}
        rcv = TR.make_recover(a, None, None, None, None, info)                                      # J / wins / store / clips are None: any bank read or table build would raise
        assert rcv.table is None and "recover" not in info and rcv.active == (a.recover_p > 0)
        assert TR.bad_recover(a, rcv) == []


def test_a_truncated_bank_sha256_is_refused(pack):
    a = TR.build_parser().parse_args(["--out", "x", *GATE_ARGV, "--recover-p", "0.3", "--recover-dev-bank", str(pack.bank), "--recover-dev-bank-sha256", _bank_sha(pack.bank)[:12]])
    assert any("not a sha256" in m for m in TR.bad_recover(a, table_perturb(bank_sha256=_bank_sha(pack.bank)[:12])))


# ------------------------------------------------------------------------------------------------ the bank, K-B3 (and the absence of K-B2), the table
def test_the_bank_loads_and_the_table_is_built_from_THIS_trainers_store_and_matches_a_naive_recomputation(pack):
    pt = parts(pack, REC(pack))
    a, cfg, model, opt, info, rcv, J, wins, clips, store = pt
    assert rcv.active and rcv.table is not None and len(rcv.table) > 0
    assert dvb(pt) == []
    lv = rcv.built_levers()
    assert lv["recover_dev_bank_sha256"] == _bank_sha(pack.bank) and os.path.normcase(os.path.abspath(lv["recover_dev_bank"])) == os.path.normcase(os.path.abspath(Path(pack.bank) / "bank.json"))
    ctl = info["recover"]
    assert ctl["kb3"]["overlap_with_eval"] == 0 and ctl["kb3"]["window_list_sha256"] == info["train_window_list_sha256"] and ctl["kb2_read"] is False
    assert ctl["table"]["content_sha256"] == rcv.table.content_sha256() and ctl["table"]["n"] == len(rcv.table) == ctl["table"]["counts"]["kept"]
    # an independent, per-record recomputation of the card's five numbers (section 3) from the bank's own arrays and the clip store
    bank = DG.DaggerBank.load(pack.bank, clip_names=J["clip"])
    ok = np.nonzero(bank.reason == 0)[0]
    names = J["clip"].tolist()
    rows, logged_v = [], []

    def wrap(x):
        return (x + math.pi) % (2 * math.pi) - math.pi
    for i in ok:
        d = store.get(str(names[int(bank.ci[i])]))
        j = int(bank.a["j"][i])
        if j < 2 or j >= len(d["pose"]) or not np.isfinite(np.asarray(d["kin3"])[j]).all():
            continue
        logged_v.append(float(d["kin3"][j][0]))
        P = np.asarray(bank.a["pose"][i], np.float64)
        lx, ly, lpsi = d["pose"][j][:3]                                                           # the pose grid carries more than (x, y, yaw); the trainer reads the first three
        dx, dyw = P[-1, 0] - lx, P[-1, 1] - ly
        rows.append([bank.a["kin3"][i][0] - d["kin3"][j][0], bank.a["kin3"][i][1] - d["kin3"][j][1],
                     wrap(P[-1, 2] - P[-2, 2]) / D.DT - wrap(d["pose"][j][2] - d["pose"][j - 1][2]) / D.DT,
                     -math.sin(lpsi) * dx + math.cos(lpsi) * dyw, wrap(P[-1, 2] - lpsi)])
    ref = np.asarray(rows)
    assert ref.shape == (len(rcv.table), 5) and ref.shape[0] > 20
    got = np.stack([rcv.table.cols[k] for k in R.TABLE_FIELDS], 1)
    assert np.abs(got - ref).max() < 1e-9, "the vectorised table is the card's definition, record for record"
    q = rcv.table.quantiles()
    assert set(q) == set(R.TABLE_FIELDS) and q["dv"]["signed"]["q50"] == float(np.quantile(ref[:, 0], 0.5))
    lv_ = np.asarray(logged_v)
    bc = info["recover"]["table_band_3_8"]
    assert bc["n"] == len(rcv.table) == len(lv_) and bc["n_in_band"] == int(((lv_ >= 3.0) & (lv_ <= 8.0)).sum()) and abs(bc["share_in_band"] - float(((lv_ >= 3.0) & (lv_ <= 8.0)).mean())) < 1e-12
    assert info["recover"]["table"]["share_logged_v_3_8"] == bc["share_in_band"]


def test_k_b3_an_eval_clip_a_partial_bank_a_foreign_window_list_and_a_wrong_sha_are_each_refused(pack, tmp_path):
    ev = json.loads((pack.world / "eval_grid_windows.json").read_text())["order"][0]
    cases = {
        "eval clip": (clone_bank(pack.bank, tmp_path / "ev", shard_fn=rename_clip(None, ev)), "K-B3"),
        "partial": (clone_bank(pack.bank, tmp_path / "pa", partial=True), "PARTIAL"),
        "window list": (clone_bank(pack.bank, tmp_path / "wl", common_fn=lambda c: {**c, "window_list_sha256": "0" * 64}), "window-list"),
    }
    for name, (bank, what) in cases.items():
        a = TR.build_parser().parse_args(["--out", str(pack.init_out), *GATE_ARGV, *REC(pack, bank=bank)])
        store, J, wins, clips, info = TR.build_data(a, TR.make_config(a))
        with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ recover-dev-bank") as e:
            TR.make_recover(a, J, wins, store, clips, info)
        assert what in str(e.value), (name, str(e.value)[:300])
    a = TR.build_parser().parse_args(["--out", str(pack.init_out), *GATE_ARGV, *REC(pack, sha="0" * 64)])
    store, J, wins, clips, info = TR.build_data(a, TR.make_config(a))
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ recover-dev-bank.*sha256"):
        TR.make_recover(a, J, wins, store, clips, info)


def test_k_b2_is_NOT_read_a_bank_generated_by_another_checkpoint_than_the_init_is_accepted_and_disclosed(pack):
    """The card draws deviations from the model family's own states (bank2 = DAG's, the init is DAG-OACC): the DAgger swap's K-B2 would refuse this pairing, the recovery lever must not."""
    pt = parts(pack, ["--init-ckpt", str(pack.ck_other), *REC(pack)])
    info = pt[4]
    assert dvb(pt) == []
    assert info["recover"]["bank_ckpt_sha256"] and info["recover"]["init_ckpt_sha256"] and info["recover"]["bank_ckpt_sha256"] != info["recover"]["init_ckpt_sha256"]
    pd = parts(pack, ["--init-ckpt", str(pack.ck_other), "--dagger-p", "0.1", "--dagger-bank", str(pack.bank), "--dagger-bank-sha256", _bank_sha(pack.bank)])
    a, J, wins, clips, info2 = pd[0], pd[6], pd[7], pd[8], pd[4]
    with pytest.raises(SystemExit, match="K-B2"):
        TR.make_dagger(a, J, wins, clips, info2)                                                  # the control: the DAgger path DOES refuse this pairing


def test_g_dvb_refuses_a_perturber_built_from_another_bank_and_a_stale_record(pack, tmp_path):
    pt = parts(pack, REC(pack))
    a, rcv, info = pt[0], pt[5], pt[4]
    other = clone_bank(pack.bank, tmp_path / "other", common_fn=lambda c: {**c, "ckpt_sha256": "1" * 64})
    pt2 = parts(pack, REC(pack, bank=other))
    assert _bank_sha(other) != _bank_sha(pack.bank)
    bad = TR.declared_vs_built(a, pt[1], pt[2], pt[3], info, None, None, None, None, None, pt2[5])           # argv declares bank A, the perturber was built from bank B
    assert any("recover_dev_bank" in m and "declared" in m for m in bad) and any("recover_dev_bank_sha256" in m for m in bad), bad
    bad = TR.declared_vs_built(a, pt[1], pt[2], pt[3], {k: v for k, v in info.items() if k != "recover"}, None, None, None, None, None, rcv)
    assert any("no K-B3 / table record was read" in m for m in bad), bad
    stale = {**info, "recover": {**info["recover"], "table": {**info["recover"]["table"], "content_sha256": "0" * 64}}}
    assert any("not the BUILT table's" in m for m in TR.declared_vs_built(a, pt[1], pt[2], pt[3], stale, None, None, None, None, None, rcv))
    kb3 = {**info["recover"]["kb3"], "window_list_sha256": "0" * 64}
    assert any("K-B3 was not read" in m for m in TR.declared_vs_built(a, pt[1], pt[2], pt[3], {**info, "recover": {**info["recover"], "kb3": kb3}}, None, None, None, None, None, rcv))
    ed = clone_bank(pack.bank, tmp_path / "ed")
    with open(ed / "bank.json", "a", encoding="utf-8") as fh:                                        # the file on disk differs from the one the table was built from: the gate re-hashes it
        fh.write(" \n")
    a2 = TR.build_parser().parse_args(["--out", str(pack.init_out), *GATE_ARGV, *REC(pack, bank=ed, sha=_bank_sha(pack.bank))])
    assert any("recover_dev_bank_sha256" in m for m in TR.bad_recover(a2, rcv, info))


# ------------------------------------------------------------------------------------------------ G-LIVE: recover_probe and its deliberate-regression arms
def test_the_probe_passes_and_records_the_controls_for_the_empirical_and_the_apriori_law(pack):
    model, b, cfg = probe_parts(pack)
    for rcv in (table_perturb(p=0.3, target="rejoin"), table_perturb(p=0.3, target="logged"), R.RecoverPerturb(p=0.3, dev_law="apriori", seed=2)):
        res = TR.recover_probe(model, b, cfg, rcv)
        assert res["pass"] is True and res["n_selected_at_p1"] == res["n_eligible"] > 0
        assert res["control_repeat_grad_diff"] == 0.0 and res["control_p0_grad_diff"] == 0.0 and res["control_p0_loss_diff"] == 0.0 and res["control_p0_same_object"] is True
        assert res["control_prior_grad_diff"] == 0.0 and res["control_wm_grad_diff"] == 0.0
        assert res["grad_diff_inputs_only"] > 0.0 and res["grad_diff_target_only"] > 0.0 and res["loss_diff_inputs_only"] != 0.0 and res["loss_diff_target_only"] != 0.0
        assert res["grad_wrt_p10f_selected_abs_sum"] > 0.0 and res["gt_max_abs_err_vs_rejoin_target"] < 1e-5 and res["live_recovered"]["pass"] is True
        assert res["live_recovered"]["control_prior_grad_sum"] == 0.0
    assert TR.recover_probe(model, b, cfg, R.RecoverPerturb(p=0.0))["pass"] is True, "off: nothing to prove, nothing runs"


def test_MUTATION_deleting_the_apply_line_in_the_library_turns_the_probe_RED(pack):
    model, b, cfg = probe_parts(pack)
    src = RECOVER_P.read_text(encoding="utf-8")
    line = [ln for ln in src.splitlines() if "the ONE line that applies the perturbation" in ln]
    assert len(line) == 1
    ns: dict = {"__name__": "mutant_recover"}
    exec(compile(src.replace(line[0], "            pass"), "mutant_v8f_recover.py", "exec"), ns)
    mutant = ns["RecoverPerturb"](p=0.3, table=small_table(), bank_path="bank_x", bank_sha256="ab" * 32, seed=1)
    with pytest.raises(AssertionError, match="unchanged .* selected rows: the rebuild did not reach it"):
        TR.recover_probe(model, b, cfg, mutant)
    assert TR.recover_probe(model, b, cfg, table_perturb())["pass"] is True, "the control: the unmutated lever passes"


def test_the_probe_refuses_a_rebuild_that_changes_nothing_or_leaves_one_half_unperturbed(pack, monkeypatch):
    model, b, cfg = probe_parts(pack)
    orig = R.RecoverPerturb._rebuilt
    monkeypatch.setattr(R.RecoverPerturb, "_rebuilt", lambda self, batch, rows, *a: {})
    with pytest.raises(AssertionError, match="unchanged .* selected rows: the rebuild did not reach it"):
        TR.recover_probe(model, b, cfg, table_perturb())

    def keep(key):
        def fn(self, batch, rows, *a):
            r = orig(self, batch, rows, *a)
            r[key] = batch[key]
            return r
        return fn
    for key, what in (("gt", "gt is unchanged on every selected row: the target was not re-expressed"), ("p10f", "p10f is unchanged"), ("kin3", "kin3 is unchanged"), ("base", "base is unchanged")):
        monkeypatch.setattr(R.RecoverPerturb, "_rebuilt", keep(key))
        with pytest.raises(AssertionError, match=what):
            TR.recover_probe(model, b, cfg, table_perturb())
    monkeypatch.setattr(R.RecoverPerturb, "_rebuilt", orig)
    assert TR.recover_probe(model, b, cfg, table_perturb())["pass"] is True


def test_the_probe_refuses_a_target_that_is_not_the_registered_one_and_an_rc_that_was_not_re_expressed(pack, monkeypatch):
    model, b, cfg = probe_parts(pack)
    orig = R.RecoverPerturb._rebuilt

    def wrong_target(self, batch, rows, *a):
        r = orig(self, batch, rows, *a)
        r["gt"] = r["gt"].clone()
        r["gt"][rows] = r["gt"][rows] * 1.01
        return r
    monkeypatch.setattr(R.RecoverPerturb, "_rebuilt", wrong_target)
    with pytest.raises(AssertionError, match="the target is not the registered one"):
        TR.recover_probe(model, b, cfg, table_perturb())

    def no_rc(self, batch, rows, *a):
        r = orig(self, batch, rows, *a)
        r["lab_s"] = batch["lab_s"]
        return r
    monkeypatch.setattr(R.RecoverPerturb, "_rebuilt", no_rc)
    with pytest.raises(AssertionError, match=r"lab_s\.rc differs from its frame change"):
        TR.recover_probe(model, b, cfg, table_perturb())


def test_the_probe_refuses_a_p0_that_is_not_the_identity_and_a_nondeterministic_probe(pack, monkeypatch):
    model, b, cfg = probe_parts(pack)
    orig = R.RecoverPerturb.apply

    def leaky(self, batch, step, *, p=None):
        out = orig(self, batch, step, p=p)
        if p == 0.0:
            out = dict(batch)
            out["gt"] = batch["gt"] + 1e-3
        return out
    monkeypatch.setattr(R.RecoverPerturb, "apply", leaky)
    with pytest.raises(AssertionError, match="p = 0 is not the identity"):
        TR.recover_probe(model, b, cfg, table_perturb())


# ------------------------------------------------------------------------------------------------ the whole trainer: selfcheck, the loop, the batch the LOSS receives, the run record
def test_selfcheck_only_passes_for_R_and_for_C0_and_records_the_levers_the_table_and_the_probe(pack, tmp_path):
    out_r, out_c = tmp_path / "R", tmp_path / "C0"
    assert TR.main(_run(out_r, [*REC(pack), "--selfcheck-only"])) == 0
    sc = _selfcheck(out_r)
    assert sc["declared_vs_built"] == "PASS" and sc["recover_probe"]["pass"] is True and sc["levers"]["recover"]["recover_p"] == 0.3
    assert sc["levers"]["recover"]["recover_dev_bank_sha256"] == _bank_sha(pack.bank) == sc["info"]["recover"]["bank_sha256"]
    t = sc["levers"]["recover_table"]
    assert t["n"] > 0 and t["content_sha256"] and "share_logged_v_3_8" in t and sc["info"]["recover"]["table_band_3_8"]["n"] == t["n"]
    assert sc["info"]["recover"]["kb3"]["overlap_with_eval"] == 0
    assert TR.main(_run(out_c, ["--recover-p", "0", "--selfcheck-only"])) == 0                    # C0 names ONLY --recover-p 0
    sc0 = _selfcheck(out_c)
    assert sc0["recover_probe"]["recover"] == "off" and sc0["levers"]["recover"]["recover_p"] == 0.0 and sc0["levers"]["recover_table"] is None and "recover" not in sc0["info"]


def test_C0_and_a_run_that_never_names_the_flags_are_bit_identical_and_R_is_not(pack, tmp_path):
    outs = {}
    for name, extra in (("none", []), ("c0", ["--recover-p", "0"]), ("r", REC(pack, p=1.0)), ("apr", [*APRIORI])):
        outs[name] = tmp_path / name
        assert TR.main(_run(outs[name], ["--steps", "4", *extra])) == 0
    sd = {k: torch.load(v / "ckpt_last.pt", map_location="cpu", weights_only=False) for k, v in outs.items()}
    same = lambda x, y: all(torch.equal(x["model"][n], y["model"][n]) for n in x["model"])          # noqa: E731
    assert same(sd["none"], sd["c0"]), "p = 0: the trainer is bit-identical to a run that never named the flags"
    strip = lambda L: [{k: v for k, v in r.items() if k not in ("sec",)} for r in L]                 # noqa: E731
    assert strip(_log(outs["none"])) == strip(_log(outs["c0"]))
    assert not same(sd["none"], sd["r"]) and not same(sd["none"], sd["apr"]), "the lever on: it trained on something else"
    s0, s1 = _summary(outs["c0"]), _summary(outs["r"])
    assert (s0["recover_p"], s0["recover_n"], s0["recover_eligible"], s0["recover_rows"], s0["recover_changed"], s0["recover_share"]) == (0.0, 0, 0, 0, 0, None)
    assert s1["recover_n"] == s1["recover_eligible"] > 0 and s1["recover_rows"] == 16 * 4 and s1["recover_changed"] == s1["recover_n"], "p = 1: every eligible row, and the batch the loss holds differs on exactly those"
    assert "recover" not in sd["none"] and sd["r"]["recover"]["recover_p"] == 1.0 and sd["r"]["recover_totals"]["n"] == s1["recover_n"]


def test_the_batch_the_LOSS_receives_is_the_recovered_one_on_the_selected_rows_and_only_there(pack, tmp_path, monkeypatch):
    seen = {}
    orig = M.V8FMini.loss

    def spy(self, b, *a, **k):
        seen.setdefault(tag[0], []).append({k_: (v.detach().clone() if torch.is_tensor(v) else v) for k_, v in b.items() if k_ in ("kin3", "gt", "p10f", "pose_now", "t")})
        return orig(self, b, *a, **k)
    monkeypatch.setattr(M.V8FMini, "loss", spy)
    tag = ["c0"]
    for t, extra in (("c0", ["--recover-p", "0"]), ("r", [*APRIORI])):
        tag[0] = t
        assert TR.main(_run(tmp_path / f"t_{t}", ["--steps", "3", *extra])) == 0
    c0 = [x for x in seen["c0"]][-3:]
    r = [x for x in seen["r"]][-3:]
    moved = 0
    for x0, x1 in zip(c0, r):
        ch = (x0["kin3"] != x1["kin3"]).any(1)
        moved += int(ch.sum())
        assert torch.equal(x0["t"], x1["t"]), "C0 and R see the same windows in the same order (the private generator never touches the global RNG)"
        assert not bool((x0["p10f"] != x1["p10f"]).reshape(len(ch), -1).any(1)[~ch].any()) and not bool((x0["gt"] != x1["gt"]).reshape(len(ch), -1).any(1)[~ch].any())
    assert moved > 0


def test_MUTATION_deleting_the_loop_line_stops_the_run_at_its_first_step(pack, tmp_path):
    src = TRAIN_P.read_text(encoding="utf-8")
    anchor = [ln for ln in src.splitlines() if "the ONE loop line that applies the recovery perturbation" in ln]
    assert len(anchor) == 1 and anchor[0].strip().startswith("b = rcv.apply(b, step)")
    mutated = src.replace(anchor[0], anchor[0][: len(anchor[0]) - len(anchor[0].lstrip())] + "pass")
    ns: dict = {"__name__": "mutant_trainer", "__file__": str(TRAIN_P)}
    exec(compile(mutated, str(TRAIN_P), "exec"), ns)
    with pytest.raises(SystemExit, match="ZZRECOVER-NOT-APPLIEDZZ step 0"):
        ns["main"](_run(tmp_path / "mut", ["--steps", "3", *APRIORI]))
    assert TR.main(_run(tmp_path / "ok", ["--steps", "3", *APRIORI])) == 0, "the control: the unmutated trainer runs"


def test_a_resumed_run_draws_what_an_uninterrupted_run_drew_and_keeps_counting(pack, tmp_path, monkeypatch):
    full, part = tmp_path / "full", tmp_path / "part"
    assert TR.main(_run(full, ["--steps", "6", "--ckpt-every", "3", *APRIORI])) == 0
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "4")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(_run(part, ["--steps", "6", "--ckpt-every", "3", *APRIORI]))
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert TR.main(_run(part, ["--steps", "6", "--ckpt-every", "3", *APRIORI])) == 0
    a, b = _summary(full), _summary(part)
    assert (a["recover_n"], a["recover_eligible"], a["recover_rows"], a["recover_changed"]) == (b["recover_n"], b["recover_eligible"], b["recover_rows"], b["recover_changed"]) and a["recover_n"] > 0
    ca, cb = (torch.load(x / "ckpt_last.pt", map_location="cpu", weights_only=False) for x in (full, part))
    assert all(torch.equal(ca["model"][n], cb["model"][n]) for n in ca["model"]), "the same run"


def test_the_log_lines_carry_the_per_interval_counts_only_when_the_lever_is_on(pack, tmp_path):
    assert TR.main(_run(tmp_path / "off", ["--steps", "3", "--recover-p", "0"])) == 0
    assert TR.main(_run(tmp_path / "on", ["--steps", "3", *APRIORI])) == 0
    off, on = _log(tmp_path / "off"), _log(tmp_path / "on")
    assert all("recover_n" not in r for r in off) and all("recover_n" in r and r["recover_n"] <= r["recover_eligible"] <= r["recover_rows"] == 16 for r in on)
    assert sum(r["recover_n"] for r in on) == _summary(tmp_path / "on")["recover_n"]


# ------------------------------------------------------------------------------------------------ mutation anchors
def test_the_mutation_anchors_exist_exactly_once():
    s = TRAIN_P.read_text(encoding="utf-8")
    assert s.count("the ONE loop line that applies the recovery perturbation") == 1
    assert s.count("                b = rcv.apply(b, step) ") == 1
    assert RECOVER_P.read_text(encoding="utf-8").count("the ONE line that applies the perturbation") == 1
