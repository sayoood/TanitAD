"""PREREG_V8F_ROLLHIST, the label lever RL: ``--dagger-label {human,speed_v1}``. The SPEED-consistent label on the swap's own states.  CPU only, on the synthetic world of the dagger tests.

Run the WHOLE file (module fixtures are shared): CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_recovery.py

WHY THIS LABEL (MM, 2026-11-22): ``swap_rows`` re-anchors the own poses onto the LOGGED NOW at r', so the own pose offset is erased from every input; what survives is the own KINEMATICS (p10f / kin3 speed). The label is the
human's polyline from r' sampled at the human arc plus the integrated speed deficit of V1's rate-limited speed profile started from the OWN speed. Matcher-ok records only; refusals keep today's human label.

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): each claim has a test that goes RED when the single line that implements it is deleted or bent (``mutants/run_mutants.py``).

  gold          ``--dagger-label human`` (no labels attached) swaps EXACTLY like the code of record 97fe0a56 (``golden_v8f_dagger_97fe0a56.py`` = that commit's blob, sha256 pinned below): every key, bit for bit
  identity      zero speed deficit -> the human label, ``np.array_equal``
  reference     ``speed_label`` equals an independent loop implementation of the brief's formula
  reachability  under speed_v1 the count of swapped rows whose gt differs from the human label is > 0 and under human EXACTLY 0; the eligible index is EQUAL under both
  refusal       a path past the polyline's end and an infeasible (kinked) path are refused; the unchecked arm emits them
  verbatim      the ported V1 labeller body is the research file's, sha256 pinned
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
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
from test_v8f_dagger import world, fx10, make_batch, eligible_windows  # noqa: E402,F401  (the module's fixtures: the synthetic world and the bank the CLI generated)

GOLD = HERE / "golden_v8f_dagger_97fe0a56.py"
GOLD_BLOB_SHA256 = "796e3e7dd78b4e3bd6e0c3d0f47ff629faad360a4949755ae829221b72a73fd1"      # sha256 of `git show 97fe0a56:stack/tanitad/data/v8f_dagger.py`
LABELLER_BODY_SHA256 = "f7dc07dd0c9ff0765c6ce2e761568ef7aaaf3371e12e781937ba72f27ea38b9c"
torch.set_num_threads(1)


def _load_gold():
    spec = importlib.util.spec_from_file_location("golden_v8f_dagger", GOLD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _labeller_body() -> bytes:
    src = inspect.getsource(RV)
    a = src.index("from __future__ import annotations")
    b = src.index("# ================================================================================================ INTEGRATION")
    return src[a:b].rstrip("\n").encode("utf-8")


def _same(a, b):
    if torch.is_tensor(a):
        return a is b or (a.shape == b.shape and a.dtype == b.dtype and bool(((a == b) | (torch.isnan(a) & torch.isnan(b)) if a.is_floating_point() else (a == b)).all()))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(np.all(a == b)) if isinstance(a, np.ndarray) else a == b


def chord_speed(pg, r):
    """u_h[1]: the human polyline's own step speed at r' (the single speed source of speed_v1)."""
    g = np.asarray(pg, np.float64)
    return float(np.hypot(*(g[r + 1, :2] - g[r, :2])) / 0.2)


def speed_bank(f, plant=0, dv=-1.0):
    """A copy of the fixture's bank, with ``plant`` ok records given an own speed ``dv`` m/s away from the logged speed at r' (the fixture's own states may already differ; planting makes the count certain)."""
    arrays = {k: np.array(v) for k, v in f.bank.a.items()}
    ok = np.flatnonzero(arrays["reason"] == G.OK)
    for i in ok[:plant]:
        c, r = int(f.bank.ci[i]), int(arrays["r"][i])
        pg = f.store.get_kin(str(f.J["clip"][c]))["pose"]
        arrays["p10f"][i, -1, 0] = np.float32(chord_speed(pg, r) + dv)
    return G.DaggerBank(arrays, f.bank.header, clip_names=f.J["clip"], manifest=f.bank.manifest, sha256=f.bank.sha256, path=f.bank.path)


def attach(f, bank, **kw):
    res = RV.label_bank(bank.a, bank.ci, f.J["clip"], f.store.get_kin, **kw)
    bank.attach_recovery(res["slot"], res["label"], res["status_counts"])
    return res


def eligible_windows_of(f, bank):
    return [w for w in f.wins if bank.candidates([w[0]], [w[1]])[0][0] > 0]


def _swap_all(f, bank, p=1.0, seed=3):
    wins = eligible_windows_of(f, bank)
    b = make_batch(f, wins)
    sw = G.DaggerSwap(bank, p=p, seed=seed)
    return wins, b, sw.apply(b, 0), sw


# ------------------------------------------------------------------------------------------------ verbatim + gold
def test_the_ported_labeller_body_is_the_research_files_body():
    """The module docstring names the research file's sha256; the body must be that file with only its docstring replaced."""
    doc = RV.__doc__
    named = doc.split("sha256 ", 1)[1].split()[0].strip(".:,;)")
    assert named == "deb258d14f3e47fb53d267fd3c6e543a74ce520f2857fcc7b408eb6ecd4dac60"
    body_sha = hashlib.sha256(_labeller_body()).hexdigest()
    assert body_sha == LABELLER_BODY_SHA256, body_sha


def test_the_gold_copy_is_the_commits_blob():
    assert hashlib.sha256(GOLD.read_bytes()).hexdigest() == GOLD_BLOB_SHA256


@pytest.mark.parametrize("p,seed,step", [(1.0, 5, 3), (0.5, 1, 4), (0.3, 7, 11)])
def test_gold_human_label_swap_equals_the_code_of_record_bit_for_bit(fx10, p, seed, step):
    gold = _load_gold()
    wins = eligible_windows(fx10, 24)
    b = make_batch(fx10, wins)
    new = G.DaggerSwap(fx10.bank, p=p, seed=seed).apply(b, step)
    gbank = gold.DaggerBank(fx10.bank.a, fx10.bank.header, clip_names=fx10.J["clip"], manifest=fx10.bank.manifest, sha256=fx10.bank.sha256, path=fx10.bank.path)
    old = gold.DaggerSwap(gbank, p=p, seed=seed).apply(b, step)
    assert set(new) == set(old) == set(b)
    for k in b:
        assert _same(new[k], old[k]), f"{k} differs from the code of record under --dagger-label human"
    for k in ("gt", "s_h", "gt_ok", "lead_gap_h"):
        assert new[k] is b[k]                                          # the targets are the SAME objects, as in the code of record
    assert fx10.bank.rl is None and fx10.bank.label_kind == "human" and "rl_has" not in fx10.bank.gather([0])
    assert np.array_equal(fx10.bank._ok_idx, gbank._ok_idx)


# ------------------------------------------------------------------------------------------------ the formula, on synthetic paths
def _path(T=140, v0=8.0, a=0.0, kappa=0.0):
    i = np.arange(T)
    t = 0.2 * i
    v = v0 + a * t
    s = v0 * t + 0.5 * a * t * t
    th = kappa * s
    if kappa:
        x, y = np.sin(th) / kappa, (1 - np.cos(th)) / kappa
    else:
        x, y = s, np.zeros(T)
    c0, s0 = np.cos(0.3), np.sin(0.3)                                        # a rigidly rotated + shifted world pose (the label is in the frame at r'), heading consistent with the path
    return np.stack([c0 * x - s0 * y + 3.0, s0 * x + c0 * y - 2.0, th + 0.3], 1), v


def _reference(grid, r, v_own, A=3.0):
    """An independent loop implementation of the brief's formula (no use of the labeller's helpers): human step speeds from the polyline's chords, label speeds rate-limited at A, label arc, points at the label arc."""
    g = np.asarray(grid, float)
    c, s_ = np.cos(-g[r, 2]), np.sin(-g[r, 2])
    P = [((g[j, 0] - g[r, 0]) * c - (g[j, 1] - g[r, 1]) * s_, (g[j, 0] - g[r, 0]) * s_ + (g[j, 1] - g[r, 1]) * c) for j in range(r, len(g))]
    arc = [0.0]
    for j in range(1, len(P)):
        arc.append(arc[-1] + float(np.hypot(P[j][0] - P[j - 1][0], P[j][1] - P[j - 1][1])))
    P, arc = np.array(P), np.array(arc)
    uh = [None] + [(arc[k] - arc[k - 1]) / 0.2 for k in range(1, 31)]
    ul = [float(v_own)]
    merged = False
    for k in range(1, 31):
        d = uh[k] - ul[-1]
        if merged or d == 0 or abs(d) <= A * 0.2:
            ul.append(uh[k])
            merged = True
        else:
            ul.append(ul[-1] + np.sign(d) * A * 0.2)
    L = [0.0]
    for k in range(1, 31):
        L.append(L[-1] + ul[k] * 0.2)
    delta = [L[k] - arc[k] for k in range(31)]
    return np.stack([np.interp(L, arc, P[:, 0]), np.interp(L, arc, P[:, 1])], 1)[1:], delta


@pytest.mark.parametrize("kw,dv", [(dict(), -1.0), (dict(kappa=0.01), -1.5), (dict(a=0.3), 1.2), (dict(a=-0.4, v0=9.0), -2.5), (dict(kappa=0.01), 1.4)])
def test_speed_label_equals_an_independent_implementation_of_the_formula(kw, dv):
    grid, v = _path(**kw)
    r = 20
    vo = chord_speed(grid, r) + dv
    res = RV.speed_label(grid, r, vo)
    assert res["status"] == RV.STATUS_EMITTED, res["status"]
    ref, delta = _reference(grid, r, vo)
    assert np.allclose(res["label"], ref, atol=1e-9, rtol=0)
    assert np.allclose(res["delta"], delta, atol=1e-9, rtol=0) and abs(delta[30]) > 0.05
    assert not np.array_equal(res["label"], RV.human_label(grid, r))
    assert RV.violations(res["label"], vo, 30)["any"] is False


def test_speed_label_is_rate_limited_not_a_jump():
    grid, v = _path()
    r = 20
    vo = chord_speed(grid, r) - 2.5
    res = RV.speed_label(grid, r, vo)
    pts = np.concatenate([[[0.0, 0.0]], res["label"]])
    ul = np.hypot(*np.diff(pts, axis=0).T) / 0.2                              # the emitted label's own step speeds (straight path: chord = arc)
    assert np.abs(np.diff(np.concatenate([[vo], ul]))).max() <= 0.2 * 3.0 + 1e-9, "no step of the label's speed exceeds A_LABEL * dt"
    assert abs(ul[0] - (vo + 0.6)) < 1e-9, "the first label step is the own speed moved toward the human by A * dt, not the human's step speed"
    jump = RV.speed_label(grid, r, vo, clip=False, check=False)
    assert abs(np.hypot(*jump["label"][0]) / 0.2 - chord_speed(grid, r)) < 1e-6 and not np.allclose(jump["label"], res["label"], atol=1e-3), "the unclipped arm jumps (the deliberate regression)"
    assert abs(RV.A_LABEL - 3.0) < 1e-12


def test_one_speed_source_the_logged_speed_column_is_not_read():
    """Two grids that differ only in the 4th column (the logged speed) give the same label: the polyline's own chords are the only human speed."""
    grid, v = _path(kappa=0.01)
    g2 = np.concatenate([grid, v[:, None]], 1)
    g3 = np.concatenate([grid, (v + 7.0)[:, None]], 1)
    a, b = RV.speed_label(g2, 20, 6.5), RV.speed_label(g3, 20, 6.5)
    assert a["status"] == RV.STATUS_EMITTED and np.array_equal(a["label"], b["label"])


def test_a_rate_equal_to_a_max_without_margin_raises_the_refusal_share_on_a_curved_path():
    """A = 4.0 (= A_MAX, no margin) against the checker's 0.99 A_MAX: every rate-limited step violates, so refusals appear where A = 3.0 emits."""
    grid, v = _path(kappa=0.01)
    r = 20
    n3 = n4 = 0
    for dv in (-0.8, -1.2, -1.6, -2.0, 1.0, 1.5):
        vo = chord_speed(grid, r) + dv
        n3 += RV.speed_label(grid, r, vo)["status"] == "refused_infeasible"
        n4 += RV.speed_label(grid, r, vo, a_rate=4.0)["status"] == "refused_infeasible"
    assert n3 == 0 and n4 > n3, (n3, n4)


def test_identity_zero_deficit_gives_exactly_the_human_label():
    for kw in (dict(), dict(kappa=0.01), dict(a=0.3)):
        grid, v = _path(**kw)
        for r in (0, 20, 100):
            for d in (0.0, 0.5, -0.5):                                      # within A * dt = 0.6 m/s of the human's first step speed: merged at k = 1, the label IS the human's
                res = RV.speed_label(grid, r, chord_speed(grid, r) + d)
                assert res["status"] == RV.STATUS_IDENTITY and np.array_equal(res["label"], RV.human_label(grid, r))
                assert np.abs(res["delta"][1:]).max() < 1e-9


def test_a_label_that_would_run_past_the_polyline_end_is_refused_and_so_is_a_kinked_path_unless_the_refusal_is_skipped():
    grid, v = _path()
    r = len(grid) - 31                                                     # exactly 30 future rows: no polyline beyond the horizon
    res = RV.speed_label(grid, r, chord_speed(grid, r) + 3.0)                    # own FASTER than the human: the label's arc runs past the end
    assert res["status"] == "path_exhausted" and res["label"] is None
    assert RV.speed_label(grid, r, chord_speed(grid, r) - 1.5)["status"] == RV.STATUS_EMITTED      # slower: the arc stays inside
    th = np.where(np.arange(140) >= 32, np.pi / 2, 0.0)                    # a 90 deg yaw kink mid-horizon: a polyline the checker cannot pass
    x, y = [0.0], [0.0]
    for i in range(1, 140):
        x.append(x[-1] + 1.6 * np.cos(th[i]))
        y.append(y[-1] + 1.6 * np.sin(th[i]))
    kink, vk = np.stack([x, y, th], 1), np.full(140, 8.0)
    bad = RV.speed_label(kink, 20, 7.0)
    assert bad["status"] == "refused_infeasible" and bad["label"] is None, bad["status"]
    assert RV.speed_label(kink, 20, 7.0, check=False)["status"] == RV.STATUS_EMITTED, "the unchecked arm emits what the checker refuses"


def test_arc_from_label_is_the_path_length_and_matches_human_arc_on_a_human_label(fx10):
    c = int(fx10.wins[0][0])
    pg = fx10.store.get_kin(str(fx10.J["clip"][c]))["pose"]
    t = int(fx10.wins[0][1])
    lab = RV.human_label(pg[:, :3], t)
    assert np.allclose(RV.arc_from_label(lab[None])[0], D.human_arc(pg)[t], atol=1e-4, rtol=0)


# ------------------------------------------------------------------------------------------------ identity + reachability + semantics on the bank
def test_identity_a_zero_deficit_own_state_gets_the_human_label_np_array_equal_in_the_swapped_batch(fx10):
    arrays = {k: np.array(v) for k, v in fx10.bank.a.items()}
    i = int(np.flatnonzero(arrays["reason"] == G.OK)[0])
    c, r = int(fx10.bank.ci[i]), int(arrays["r"][i])
    pg = fx10.store.get_kin(str(fx10.J["clip"][c]))["pose"]
    arrays["p10f"][i, -1, 0] = np.float32(chord_speed(pg, r))                # the own speed IS the human polyline's step speed at r' (the cache is float32)
    bank = G.DaggerBank(arrays, fx10.bank.header, clip_names=fx10.J["clip"], manifest=fx10.bank.manifest, sha256=fx10.bank.sha256, path=fx10.bank.path)
    res = RV.label_bank(bank.a, bank.ci, fx10.J["clip"], fx10.store.get_kin)
    assert res["slot"][i] >= 0
    bank.attach_recovery(res["slot"], res["label"], res["status_counts"])
    rec = bank.gather([i])
    assert bool(rec["rl_has"][0])
    win = (c, r)
    assert win in {(w[0], w[1]) for w in fx10.wins}
    b = make_batch(fx10, [win])
    out = G.DaggerSwap(bank, p=1.0, seed=0).apply(b, 0)
    assert np.array_equal(out["gt"].numpy(), b["gt"].numpy()), "zero deficit: the speed-consistent label is the human label, bit for bit"
    assert np.allclose(out["s_h"].numpy(), b["s_h"].numpy(), atol=1e-4, rtol=0), "s_h re-derived from the same waypoints (float32 resolution)"


def test_reachability_speed_v1_replaces_labels_human_does_not_and_the_eligible_index_is_equal(fx10):
    hb = speed_bank(fx10, plant=8)
    wins_h, b_h, out_h, sw_h = _swap_all(fx10, hb)
    changed_h = int((out_h["gt"] != b_h["gt"]).reshape(len(wins_h), -1).any(1).sum())
    assert changed_h == 0 and sw_h.last["n_label_replaced"] == 0 and out_h["gt"] is b_h["gt"]          # human: exactly 0
    rb = speed_bank(fx10, plant=8)
    res = attach(fx10, rb)
    assert np.array_equal(rb._ok_idx, hb._ok_idx) and np.array_equal(rb.eligible_mask, hb.eligible_mask) and np.array_equal(rb._ok_key, hb._ok_key)      # the eligible index is EQUAL under both
    wins_r, b_r, out_r, sw_r = _swap_all(fx10, rb)
    assert wins_r == wins_h
    changed_r = int((out_r["gt"] != b_r["gt"]).reshape(len(wins_r), -1).any(1).sum())
    assert changed_r > 0, f"speed_v1: swapped rows whose label differs from the human label ({res['status_counts']})"
    assert sw_r.last["n_label_replaced"] > 0 and changed_r <= sw_r.last["n_label_replaced"]
    for k in b_r:                                                          # the ONLY targets replaced are gt and s_h; everything else is the same object
        if k in G.REBUILT or k in ("gt", "s_h", "lab_s"):
            continue
        assert out_r[k] is b_r[k], k
    assert out_r["gt"] is not b_r["gt"] and out_r["s_h"] is not b_r["s_h"]
    ch = (out_r["gt"] != b_r["gt"]).reshape(len(wins_r), -1).any(1)
    assert bool(ch.any()) and np.allclose(out_r["s_h"][ch].numpy(), RV.arc_from_label(out_r["gt"][ch].double().numpy()), atol=1e-4, rtol=0), "s_h of a replaced row is re-derived from the SAME waypoints"
    assert not np.allclose(out_r["s_h"][ch].numpy(), b_r["s_h"][ch].numpy(), atol=1e-4, rtol=0), "and it is not the human arc"
    assert out_r["gt"].dtype == b_r["gt"].dtype and out_r["s_h"].dtype == b_r["s_h"].dtype


def test_the_own_speed_not_the_human_speed_drives_the_label(fx10):
    """Planted own speeds -1.0 and +1.0 m/s from the logged speed give DIFFERENT labels; each equals the independent formula on its own speed."""
    lo, hi = speed_bank(fx10, plant=4, dv=-1.0), speed_bank(fx10, plant=4, dv=+1.0)
    rl = attach(fx10, lo)
    attach(fx10, hi)
    n_diff = 0
    for i in np.flatnonzero(lo.reason == G.OK)[:4]:
        if lo.rl["slot"][i] >= 0 and hi.rl["slot"][i] >= 0:
            a, b = lo.rl["label"][lo.rl["slot"][i]], hi.rl["label"][hi.rl["slot"][i]]
            n_diff += int(not np.array_equal(a, b))
            c, r = int(lo.ci[i]), int(lo.a["r"][i])
            pg = fx10.store.get_kin(str(fx10.J["clip"][c]))["pose"]
            ref, _ = _reference(pg[:, :3], r, float(lo.a["p10f"][i, -1, 0]))
            assert np.allclose(a, ref, atol=1e-9, rtol=0)
    assert n_diff > 0
    assert rl["census"]["relabelled_share_of_ok"] > 0


def test_semantics_only_ok_records_carry_a_label_refused_ones_keep_the_human_label_and_unselected_rows_are_untouched(fx10):
    rb = speed_bank(fx10, plant=8)
    res = attach(fx10, rb)
    has = rb.rl["slot"] >= 0
    assert not has[rb.reason != G.OK].any()                                # off_*, end_edge, nonfinite: never labelled (dropped exactly as today)
    nonok = np.flatnonzero(rb.reason != G.OK)
    if len(nonok):
        bad = rb.rl["slot"].copy()
        bad[nonok[0]] = 0
        with pytest.raises(ValueError):                                    # a label on a non-ok record is refused by the bank
            rb.attach_recovery(bad, rb.rl["label"])
    refused_ok = np.flatnonzero((rb.reason == G.OK) & ~has)
    if len(refused_ok):
        assert not rb.gather(refused_ok[:3])["rl_has"].any()
    wins = eligible_windows_of(fx10, rb)[:24]
    b = make_batch(fx10, wins)
    sw = G.DaggerSwap(rb, p=0.5, seed=2)
    out = sw.apply(b, 6)
    sel = sw.last["sel"]
    assert 0 < int(sel.sum()) < len(wins)
    for k in ("gt", "s_h"):
        assert bool((out[k][~sel] == b[k][~sel]).all())
    assert res["n_labelled"] == int((rb.reason == G.OK).sum())


def test_the_item_path_carries_the_speed_label_like_the_batch_path(fx10):
    rb = speed_bank(fx10, plant=8)
    attach(fx10, rb)
    wins = eligible_windows_of(fx10, rb)
    sw = G.DaggerSwap(rb, p=1.0, seed=4)
    b = make_batch(fx10, wins)
    out = sw.apply(b, 1)
    idx = sw.last["record_idx"]
    k = next(i for i in range(len(wins)) if bool((out["gt"][i] != b["gt"][i]).any()))
    it = fx10.ds.item(wins[k][0], wins[k][1], 1, 3)
    one = G.swap_item(it, rb.gather([int(idx[k])]))
    assert np.array_equal(one["gt"], out["gt"][k].numpy()) and np.array_equal(one["s_h"], out["s_h"][k].numpy())


def test_the_census_adds_up_per_tick(fx10):
    rb = speed_bank(fx10, plant=8)
    res = attach(fx10, rb)
    cen = res["census"]
    ok = rb.reason == G.OK
    tot = 0
    for t, row in cen["per_tick"].items():
        assert row["ok"] == int((ok & (rb.a["tick"] == int(t))).sum())
        assert row["relabelled"] + row["identity"] + sum(row["refused"].values()) == row["ok"]
        assert set(row["refused_by_term"]) == {"accel", "curv", "fric"} and max(row["refused_by_term"].values()) <= row["refused"].get("refused_infeasible", 0)
        tot += row["ok"]
    assert tot == int(ok.sum()) and cen["abs_v_own_minus_u_h1"]["n"] > 0
    assert cen["abs_delta30"]["p95"] >= cen["abs_delta30"]["p50"] >= 0.0
    assert 0.0 < cen["relabelled_share_of_ok"] <= 1.0
