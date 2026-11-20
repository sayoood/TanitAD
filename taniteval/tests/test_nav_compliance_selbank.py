"""``from_refcv3_dump`` on a dump whose ``sel_bank_*`` is RAGGED.

MEASURED 2026-11-19: every refcv9 E2 dump read ``nav_compliance did not run: ValueError:
operands could not be broadcast together with shapes (855,) (2397,)`` -- the whole
STRATEGIC family lost. ``sel_bank_{cond}`` is written as ``bank[si:si + 1]`` with ``si``
indexing the REFINED candidate set, which is wider than the bank, so a window whose
selection lies beyond the bank has an EMPTY slice. The old flatten kept an episode's array
``if v.shape[0] == n`` and so concatenated only the episodes that happened to match.

The whole file runs without scipy.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)              # <repo>/taniteval
if _PKG_PARENT not in sys.path:
    sys.path.append(_PKG_PARENT)
_STACK = os.path.join(os.path.dirname(_PKG_PARENT), "stack")
if os.path.isdir(_STACK) and _STACK not in sys.path:
    sys.path.append(_STACK)

from taniteval import nav_compliance as nc   # noqa: E402

N_BOOT = 100
B_TRUE = 100              # the synthetic bank size
N_EP, N_WIN, T_FR = 12, 12, 200
CONDS = ("nav_true", "nav_shuffled", "nav_zero")


def _th(side):
    """Terminal heading of a follower fed ``side`` (+1 left, -1 right, 0 follow)."""
    return 0.6 * np.asarray(side, dtype=np.float64)


def _path(th):
    """[8, 2] ego-frame path whose LAST segment points at ``th``."""
    p = np.zeros((8, 2))
    p[:, 0] = 2.0 * np.arange(8)
    p[7] = p[6] + 2.0 * np.array([np.cos(th), np.sin(th)])
    return p


def _side_of(nav):
    return {nc.NAV_LEFT: 1, nc.NAV_RIGHT: -1}.get(int(nav), 0)


def make_dump(root, *, full_bank=False, bad_partition=False, ragged_extra=False,
              seed=0):
    """A synthetic refcv3 dump + labels blob. Returns ``(dump_dir, labels_path, truth)``.

    Even episodes carry a commanded turn at 4-8 s (every window informative); odd ones
    are FOLLOW. The model is a perfect follower of whatever it is FED. ``sel_idx`` lies in
    [0, 90) for windows whose bank row exists and in [110, 149) for those beyond it, so
    the true bank size ``B_TRUE`` = 100 sits inside the data-bounded interval.
    """
    rng = np.random.default_rng(seed)
    dd = os.path.join(str(root), "dump")
    os.makedirs(os.path.join(dd, "decisions"), exist_ok=True)
    nav_all = []
    for e in range(N_EP):
        turn = e % 2 == 0
        side = 1 if (e // 2) % 2 == 0 else -1
        nav_all += [(nc.NAV_LEFT if side > 0 else nc.NAV_RIGHT) if turn else nc.NAV_FOLLOW] * N_WIN
    nav_all = np.asarray(nav_all, dtype=np.int64)
    shuf_all = np.roll(nav_all, 7)
    fed_all = {"nav_true": nav_all, "nav_shuffled": shuf_all,
               "nav_zero": np.zeros_like(nav_all)}
    episodes, labels = [], []
    truth = {"beyond": {c: np.zeros(N_EP * N_WIN, dtype=bool) for c in CONDS},
             "turn_window": np.zeros(N_EP * N_WIN, dtype=bool), "N": N_EP * N_WIN}
    for e in range(N_EP):
        sl = slice(e * N_WIN, (e + 1) * N_WIN)
        nav, shuf = nav_all[sl], shuf_all[sl]
        turn = e % 2 == 0
        side = _side_of(nav[0])
        truth["turn_window"][sl] = turn
        yaw = np.zeros(T_FR)
        if turn:
            yaw[40:81] = np.linspace(0.0, np.deg2rad(60.0) * side, 41)
            yaw[81:] = np.deg2rad(60.0) * side
        ep_poses = np.stack([np.zeros(T_FR), np.zeros(T_FR), yaw, np.zeros(T_FR)], 1)
        gt = np.zeros((N_WIN, 60, 4))
        gt[:, :, 0] = 1.5 * (1 + np.arange(60))
        gt[:, :, 2] = side * 0.5 if turn else 0.0
        gt[:, :, 1] = side * 0.3 * np.arange(60) if turn else 0.0
        d = {"ws": (5 * np.arange(N_WIN)).astype(np.int64), "ep_poses": ep_poses,
             "gt_future_ext": gt, "pose_last": np.tile([0.0, 0.0, 0.0, 10.0], (N_WIN, 1)),
             "ego_t0": np.tile([10.0, 0.0, 0.0, 0.0], (N_WIN, 1)),
             "nav_cmd": nav, "nav_cmd_shuf": shuf, "nav_valid": np.ones(N_WIN, bool)}
        for c in CONDS:
            fed = fed_all[c][sl]
            th = _th([_side_of(x) for x in fed])
            d[f"plan_full_{c}"] = np.stack([_path(t) for t in th])
            d[f"gstr_{c}"] = np.stack([np.cos(th), np.sin(th), np.zeros(N_WIN)], 1)
            beyond = (np.zeros(N_WIN, bool) if full_bank
                      else rng.random(N_WIN) < 0.15)
            if e == 0 and not full_bank:
                beyond[0] = True
            si = np.where(beyond, rng.integers(110, 149, N_WIN),
                          rng.integers(0, 90, N_WIN)).astype(np.int64)
            keep = ~beyond
            bank = np.stack([_path(t) for t in th[keep]]) if keep.any() \
                else np.zeros((0, 8, 2))
            d[f"sel_idx_{c}"] = si
            d[f"sel_bank_{c}"] = bank
            truth["beyond"][c][sl] = beyond
            if bad_partition and c == "nav_true" and e in (0, 1):
                si[0] = 140 if e == 0 else 111
                d["sel_idx_nav_true"] = si
                truth["beyond"][c][sl] = si >= 100
                keep = si < 100
                if e == 0:       # ep0 holds a row for EVERY window: needs B > 140
                    bank = np.stack([_path(t) for t in th])
                else:            # ep1 needs B <= 111
                    bank = np.stack([_path(t) for t in th[keep]])
                d["sel_bank_nav_true"] = bank
        if ragged_extra:
            d["aux_ragged"] = np.zeros((N_WIN - (1 if e == 0 else 0), 3))
            if e != 3:
                d["aux_partial"] = np.zeros((N_WIN, 2))
        episodes.append(d)
        np.savez(os.path.join(dd, "decisions", f"ep{e:03d}.npz"), **d)
        cid = f"clip{e:03d}"
        tok = "NAV_FOLLOW_ROAD" if not turn else ("NAV_TURN_L" if side > 0 else "NAV_TURN_R")
        labels.append({"clip_id": cid, "nav_command": {"token": tok}, "t0_s": 0.0,
                       "manoeuvre_sequence": ([{"t_start_s": 4.0, "t_end_s": 8.0,
                                                "dyaw_deg": 60.0 * side, "is_turn": True}]
                                              if turn else [])})
    man = {"episodes": [{"file_index": e, "clip_id": f"clip{e:03d}"} for e in range(N_EP)],
           "corpus": {"frames": {"provider_to_raw_frame_offset": 0}},
           "model": {"ckpt": "synthetic", "step": 0}}
    with open(os.path.join(dd, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh)
    lp = os.path.join(str(root), "labels.jsonl.gz")
    with gzip.open(lp, "wt", encoding="utf-8") as fh:
        for r in labels:
            fh.write(json.dumps(r) + "\n")
    truth["episodes"] = episodes
    return dd, lp, truth


def _read(dd, lp, mod=nc):
    return mod.from_refcv3_dump(dd, labels_path=lp, n_boot=N_BOOT, seed=0)


def _check_a(rep, truth):
    """(a) + (b): no refusal, plan/gstr over all N, anchor n = informative minus beyond."""
    assert rep.get("status") != "UNAVAILABLE", rep.get("reason")
    N = truth["N"]
    assert rep["n_windows"] == N
    n_plan = rep["informative"]["plan"]["n_windows"]
    assert n_plan == int(truth["turn_window"].sum()) > 0
    for name in ("plan", "gstr"):
        blk = rep["readouts"][name]
        assert "n_excluded_nonfinite_signal" not in blk
        assert blk["conditionings"]["nav_true"]["n"] == rep["informative"][name]["n_windows"]
        assert blk["conditionings"]["nav_true"]["compliance_with_TRUE_command"]["mean"] == 1.0
    a = rep["readouts"]["anchor"]
    assert a.get("status") != "UNAVAILABLE", a
    any_beyond = np.zeros(N, bool)
    for c in CONDS:
        any_beyond |= truth["beyond"][c]
    expect_n = int((truth["turn_window"] & ~any_beyond).sum())
    assert a["n_excluded_nonfinite_signal"] == int((truth["turn_window"] & any_beyond).sum())
    assert a["n_informative_windows"] == expect_n < n_plan
    for c in CONDS:
        assert a["conditionings"][c]["n"] == expect_n
    return a


def test_a_b_short_sel_bank_no_exception_plan_gstr_full_anchor_drops_beyond_bank(tmp_path):
    dd, lp, truth = make_dump(tmp_path)
    rep = _read(dd, lp)
    a = _check_a(rep, truth)
    # a perfect follower: the anchor rate over the SURVIVING windows is exactly 1.0 -- a NaN
    # row counted as a miss would pull it below
    assert a["conditionings"]["nav_true"]["compliance_with_TRUE_command"]["mean"] == 1.0
    sc = rep["sel_bank_scatter"]
    for c in CONDS:
        assert sc[c]["status"] == "OK"
        assert sc[c]["bank_size_min"] <= B_TRUE <= sc[c]["bank_size_max"]
        assert sc[c]["n_windows_selected_beyond_bank"] == int(truth["beyond"][c].sum())
        assert sc[c]["n_windows"] == truth["N"]


def test_b_scatter_is_nan_exactly_on_the_beyond_bank_windows(tmp_path):
    _dd, _lp, truth = make_dump(tmp_path)
    eps = truth["episodes"]
    arr, info = nc.scatter_selbank([e["sel_idx_nav_true"] for e in eps],
                                   [e["sel_bank_nav_true"] for e in eps])
    assert info["status"] == "OK"
    nan_rows = np.isnan(arr).all(axis=(1, 2))
    assert np.array_equal(nan_rows, truth["beyond"]["nav_true"])
    assert not np.isnan(arr[~nan_rows]).any()
    # window order: the surviving rows are the episode's bank rows, in order
    th = nc.terminal_heading(arr[~nan_rows])
    fed = np.concatenate([e["nav_cmd"] for e in eps])[~nan_rows]
    assert np.allclose(th, _th([_side_of(x) for x in fed]))


def test_c_other_ragged_keys_are_excluded_and_listed_never_partial(tmp_path):
    dd, lp, truth = make_dump(tmp_path, ragged_extra=True)
    rep = _read(dd, lp)
    ex = rep["_ragged_keys"]["excluded"]
    assert set(ex) == {"aux_ragged", "aux_partial"}      # sel_bank_* has its own path
    assert ex["aux_ragged"]["n_episodes_one_row_per_window"] == N_EP - 1
    assert ex["aux_partial"]["n_episodes_one_row_per_window"] == N_EP - 1
    _check_a(rep, truth)                                  # and nothing else is disturbed


def test_d_inconsistent_partition_makes_anchor_unavailable_plan_gstr_intact(tmp_path):
    dd, lp, truth = make_dump(tmp_path, bad_partition=True)
    rep = _read(dd, lp)
    assert rep.get("status") != "UNAVAILABLE", rep.get("reason")
    a = rep["readouts"]["anchor"]
    assert a["status"] == "UNAVAILABLE" and "nav_true" in a["reason"]
    st = a["conditionings"]["nav_true"]["compliance_with_TRUE_command"]
    assert st["status"] == "UNAVAILABLE" and st["reason"]
    assert "anchor" not in rep["verdicts"]
    for name in ("plan", "gstr"):
        blk = rep["readouts"][name]
        assert blk["conditionings"]["nav_true"]["compliance_with_TRUE_command"]["mean"] == 1.0
        assert blk["n_informative_windows"] == int(truth["turn_window"].sum())


def test_scatter_selbank_refuses_malformed_input_with_a_reason():
    s = [np.array([1, 2, 3])]
    arr, info = nc.scatter_selbank(s, [np.zeros((2, 8, 3))])
    assert arr is None and info["status"] == "UNAVAILABLE"
    arr, info = nc.scatter_selbank([np.array([-1, 2])], [np.zeros((1, 8, 2))])
    assert arr is None and "negative" in info["reason"]
    arr, info = nc.scatter_selbank(s, [np.zeros((5, 8, 2))])
    assert arr is None and "more sel_bank rows" in info["reason"]


# --------------------------------------------------------------------------- #
# (e) + (f): the old flatten, reproduced by a textual MUTATION of the source   #
# --------------------------------------------------------------------------- #
def _mutated_module(tmp_path):
    """``nav_compliance`` with the pre-fix behaviour restored: the per-episode
    ``if v.shape[0] == n`` flatten (partial concatenation) and the anchor read straight
    off the flattened ``A["sel_bank_*"]``. Each replacement must hit exactly once."""
    src = open(nc.__file__, encoding="utf-8").read()
    swaps = [
        ("        if all(ok_eps):\n            cols[k] = [np.asarray(e[k]) for e in eps]",
         "        if any(ok_eps):\n"
         "            cols[k] = [np.asarray(e[k]) for e, _ok in zip(eps, ok_eps) if _ok]"),
        ('        if k.startswith("sel_bank_"):\n            continue', "        if False:\n            continue"),
        ("        if any(has_bank):", "        if False:"),
        ('        if f"fan_term_heading_{c}" in A:',
         '        if f"sel_bank_{c}" in A:\n'
         '            readouts["anchor"][c] = terminal_heading(A[f"sel_bank_{c}"].astype(np.float64))\n'
         '        if f"fan_term_heading_{c}" in A:'),
    ]
    for old, new in swaps:
        assert src.count(old) == 1, f"mutation anchor not found exactly once: {old[:50]!r}"
        src = src.replace(old, new)
    path = os.path.join(str(tmp_path), "nav_compliance_mut.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("taniteval.nav_compliance_mut", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_e_no_ragged_keys_block_identical_to_the_old_reader(tmp_path):
    dd, lp, truth = make_dump(tmp_path, full_bank=True)
    mut = _mutated_module(tmp_path)
    new, old = _read(dd, lp), _read(dd, lp, mod=mut)
    assert "_ragged_keys" not in new and "n_excluded_nonfinite_signal" not in new["readouts"]["anchor"]
    new.pop("sel_bank_scatter")                    # the only ADDED key
    dump = lambda r: json.dumps(r, sort_keys=True, default=str)   # noqa: E731
    assert dump(new) == dump(old)
    assert new["readouts"]["anchor"]["n_informative_windows"] == int(truth["turn_window"].sum())


def test_f_MUTATE_old_flatten_turns_test_a_red(tmp_path):
    dd, lp, truth = make_dump(tmp_path)
    _check_a(_read(dd, lp), truth)                                     # GREEN: the fix
    mut = _mutated_module(tmp_path)
    with pytest.raises((ValueError, AssertionError, KeyError)):        # RED: the old code
        _check_a(_read(dd, lp, mod=mut), truth)


# --------------------------------------------------------------------------- #
# (g) real data (skipped off the dev box)                                      #
# --------------------------------------------------------------------------- #
def test_h_nan_in_a_readout_not_named_keeps_the_old_scoring(tmp_path, monkeypatch):
    """MM 2026-11-19: only a readout NAMED in ``exclude_nonfinite`` drops its NaN windows (from_refcv3_dump names the
    scattered ``anchor``). Every other readout scores exactly as before -- a NaN counts as a MISS -- and only gains the
    diagnostic ``n_nonfinite_signal_scored_as_miss``; other consumers (refcv6/r7 panels, nav-tau, preflight) see no change."""
    dd, lp, truth = make_dump(tmp_path, full_bank=True)
    real = nc.nav_compliance_report
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(nc, "nav_compliance_report", spy)
    rep = _read(dd, lp)
    monkeypatch.setattr(nc, "nav_compliance_report", real)
    assert seen, "from_refcv3_dump did not call nav_compliance_report through the module global"
    n_plan = rep["informative"]["plan"]["n_windows"]
    plan = {c: np.array(v, dtype=np.float64, copy=True) for c, v in seen["readouts"]["plan"].items()}
    i = int(np.flatnonzero(truth["turn_window"])[0])
    plan["nav_true"][i] = np.nan
    kw = {**seen, "readouts": {**seen["readouts"], "plan": plan}}
    old = real(**{**kw, "exclude_nonfinite": frozenset()})["readouts"]["plan"]
    assert old["n_nonfinite_signal_scored_as_miss"] == 1
    assert "n_excluded_nonfinite_signal" not in old
    assert old["conditionings"]["nav_true"]["n"] == n_plan                     # the window is still scored ...
    assert old["conditionings"]["nav_true"]["compliance_with_TRUE_command"]["mean"] < 1.0   # ... as a miss
    ex = real(**{**kw, "exclude_nonfinite": frozenset({"plan"})})["readouts"]["plan"]
    assert ex["n_excluded_nonfinite_signal"] == 1
    assert ex["conditionings"]["nav_true"]["n"] == n_plan - 1
    assert ex["conditionings"]["nav_true"]["compliance_with_TRUE_command"]["mean"] == 1.0


_REAL = "F:/refcv9_e3/navcomp_diag/dump_s0"
_LABELS = "E:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/v8/s2_labels_v8_eval.jsonl.gz"
_OUT = "F:/refcv9_e3/navcomp_diag/navcomp_k2dc_s0.json"


@pytest.mark.skipif(not (os.path.isdir(_REAL) and os.path.exists(_LABELS)),
                    reason="the K2DC E2 dump / labels copy are on the dev box only")
def test_g_real_k2dc_dump_reads_all_three_strategic_readouts():
    rep = nc.from_refcv3_dump(_REAL, labels_path=_LABELS, n_boot=2000, seed=0)
    assert rep.get("status") != "UNAVAILABLE", rep.get("reason")
    assert rep["n_windows"] == 2397
    for name in ("plan", "gstr", "anchor"):
        r = rep["readouts"][name]["conditionings"]["nav_true"]["compliance_with_TRUE_command"]
        assert r.get("status") != "UNAVAILABLE", (name, r)
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
