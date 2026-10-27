"""PREREG_V8F_DAGGER (H-V8F-DAGGER-1), the DATA side: the matcher (section 2), the own-state bank (section 1), the swap of an own state into a training item (section 2), and the library-level controls of
section 6 (K-B0, K-B1, K-B2, K-B3, K-B4, K-B5, COVERAGE).  CPU only, on the synthetic world of ``v8f_mini_data.make_synthetic_world``.

Run: CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_dagger.py

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): every rule below has a test that goes RED when the single line that implements it is deleted or bent, and the
mutants are listed in ``raw/2026-10-26-dagger-build/mutate_dagger.py`` (one per rule).

  K-B4   the matcher        the 7 synthetic cases of the probe; the boundaries (3.0 m, 30 deg, T - 31) are inclusive on the kept side; the cut order; the tie set is ARC ties only
  K-B0   tick 0             the tick-0 own state IS the logged item, bit for bit (WindowState -> build_item vs the store's item); a leaky state is refused by the generator
  bank   the record         ticks 1 and 2 only (j = t0 + 10 tick); rows <= t0 are logged, rows after are the T1 rollout's executed motion; a cut record is stored with its reason and is never eligible
  K-B1   replay             a re-run chunk reproduces its rows bit for bit; a stochastic planner makes the check fail
  K-B3   TRAIN only         a bank holding one eval clip is REFUSED (library and generator); the control (a clean bank) reads overlap exactly 0
  K-B2   on policy          the bank's checkpoint sha256 must equal the arm's init sha256;  COVERAGE = the eligible fraction of TRAIN windows
  K-B5   re-anchor          pose_now is the logged pose at r'; the held-frame relative transforms are preserved to <= 1e-4 m
  swap   reachability       the swapped item differs from the logged one EXACTLY in REBUILT (and lab_s.rc); every KEPT / RECORD / target tensor is the same object; unselected and ineligible rows are untouched
  swap   controls           p = 0 returns the SAME batch; the same loss twice and p = 0 read EXACTLY 0.0; the swap moves the loss and its gradient; the frozen prior and the WM (act kept) read EXACTLY 0.0
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_dagger as G  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_dagger_bank as B  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402

torch.set_num_threads(1)
PK = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
THRESHOLDS = PK / "raw" / "2026-10-26-dagger-draft" / "probe" / "THRESHOLDS_FIXED.txt"
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_dag")
    w = D.make_synthetic_world(root, n_train=4, n_eval=3, T=90, seed=3)
    return {**w, "root": Path(root)}


def tiny_model(J, hmax=10, wm="off", seed=0, residual=0.05):
    cfg = M.V8FMiniConfig(**{**TINY, "hmax": hmax, "wm": wm})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    prior = M.P10Prior.synthetic(0)
    with torch.no_grad():                         # a random prior is an UNSTABLE closed loop; keep it tame (as the T1 tests do)
        for q in prior.parameters():
            q.mul_(0.02)
    model = M.V8FMini(cfg, prior, mu, sd).eval()
    with torch.no_grad():                         # head_o is zero-init: give the residual something to do
        torch.nn.init.normal_(model.head_o[2].weight, std=residual)
    return model, cfg


def write_inputs(world, hmax, wm, name):
    root = world["root"]
    J = D.load_join(root / "join_train.npz")
    model, cfg = tiny_model(J, hmax=hmax, wm=wm)
    ck = root / f"ck_{name}.pt"
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}, ck)
    av = root / "argv.json"
    json.dump({"argv": ["--arm", "M2b", "--seed", "0", "--out", "x", "--kin-mode", "yaw", "--batch", "8", "--d", "32", "--n-q", "2"]}, open(av, "w"))
    return ck, av, J, model, cfg


def gen_args(world, ck, av, out, *extra):
    r = world["root"]
    return ["gen", "--ckpt", str(ck), "--m2b-argv", str(av), "--cache-dir", str(r), "--tracks", str(r / "native_tracks.pt"), "--join-train", str(r / "join_train.npz"),
            "--eval-grid", str(r / "eval_grid_windows.json"), "--out-dir", str(out), "--batch", "8", *extra]


class BankFix:
    pass


@pytest.fixture(scope="module")
def fx10(world, tmp_path_factory):
    """hmax 10 (T / S read nothing older), the bank generated by the CLI itself (replay on 20 % of the chunks)."""
    f = BankFix()
    f.ck, f.av, f.J, f.model, f.cfg = write_inputs(world, 10, "off", "h10")
    f.out = tmp_path_factory.mktemp("bank10")
    assert B.main(gen_args(world, f.ck, f.av, f.out, "--replay-frac", "0.2")) == 0
    f.bank = G.DaggerBank.load(f.out, clip_names=f.J["clip"])
    f.tracks = D.load_native_tracks(world["root"] / "native_tracks.pt")
    f.store = D.ClipStore(world["root"], "train", lru=8, kin_mode="yaw", tracks=f.tracks)
    f.wins = D.train_windows(f.store, f.J, D.select_train_clips(world["names"]["train"], 1.0))        # the trainer's TRAIN order (sha12)
    f.ds = D.V8FMiniWindows(f.store, f.J, f.wins, hmax=10)
    return f


@pytest.fixture(scope="module")
def fx16(world, tmp_path_factory):
    """hmax 16 + the world-model auxiliary (M2b's shape): for the gradient controls (act is kept, so the WM reads exactly 0.0)."""
    f = BankFix()
    f.ck, f.av, f.J, f.model, f.cfg = write_inputs(world, 16, "b", "h16")
    f.out = tmp_path_factory.mktemp("bank16")
    assert B.main(gen_args(world, f.ck, f.av, f.out, "--replay-frac", "0")) == 0
    f.bank = G.DaggerBank.load(f.out, clip_names=f.J["clip"])
    f.tracks = D.load_native_tracks(world["root"] / "native_tracks.pt")
    f.store = D.ClipStore(world["root"], "train", lru=8, kin_mode="yaw", tracks=f.tracks)
    f.wins = D.train_windows(f.store, f.J, D.select_train_clips(world["names"]["train"], 1.0))        # the trainer's TRAIN order (sha12)
    f.ds = D.V8FMiniWindows(f.store, f.J, f.wins, hmax=16)
    return f


def eligible_windows(f, n=None):
    w = [x for x in f.wins if f.bank.candidates([x[0]], [x[1]])[0][0] > 0]
    return w if n is None else w[:n]


def ineligible_windows(f, n):
    return [x for x in f.wins if f.bank.candidates([x[0]], [x[1]])[0][0] == 0][:n]


def make_batch(f, wins, a_t=1, a_s=3):
    return D.collate([f.ds.item(w[0], w[1], a_t, a_s) for w in wins])


def _same_obj_or_equal(a, b):
    if torch.is_tensor(a):
        return a is b or (a.shape == b.shape and bool(((a == b) | (torch.isnan(a) & torch.isnan(b)) if a.is_floating_point() else (a == b)).all()))
    return a == b


# ------------------------------------------------------------------------------------------------ constants
def test_constants_are_the_preregs_and_p_is_nowhere_hard_coded():
    assert (G.OFF_ROUTE_LAT_M, G.OFF_ROUTE_HEAD_DEG, G.ARC_TIE_M) == (3.0, 30.0, 0.05)
    assert G.END_EDGE_ROWS == 31 and G.TICKS == (1, 2) and G.REPLAN_STEPS == 10 and G.COVERAGE_MIN == 0.5
    assert G.ROLLOUT == {"replan_steps": 10, "limit": "causal", "nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "post_map": "none (PLAIN)"}
    if THRESHOLDS.exists():
        txt = THRESHOLDS.read_text(encoding="utf-8")
        for name, val in (("OFF_ROUTE_LAT_M", G.OFF_ROUTE_LAT_M), ("OFF_ROUTE_HEAD_DEG", G.OFF_ROUTE_HEAD_DEG), ("ARC_TIE_M", G.ARC_TIE_M)):
            assert float(re.search(rf"^{name}\s*=\s*([0-9.]+)", txt, re.M).group(1)) == val, name
        assert G.sha256_file(THRESHOLDS).startswith("438366d5"), "THRESHOLDS_FIXED.txt is the file the PREREG cites"
    # p is the trainer's value: this module offers no default other than OFF, and names neither 0.3 nor 0.1 as an arm constant
    assert G.DAGGER_DEFAULTS["p"] == 0.0 and G.DaggerSwap().p == 0.0 and not G.DaggerSwap().active
    assert not [n for n in dir(G) if n.upper().startswith("P_DAG")]
    src = (REPO / "stack" / "tanitad" / "data" / "v8f_dagger.py").read_text(encoding="utf-8")
    assert not re.search(r"\bp\s*[:=]\s*0\.[1-9]", src.replace("p: float = DAGGER_DEFAULTS", "")), "a non-zero literal default of p"


# ------------------------------------------------------------------------------------------------ K-B4: the matcher
def _straight(T=80):
    pg = np.zeros((T, 4))
    pg[:, 0] = np.arange(T) * 2.0
    return pg


def test_kb4_the_seven_probe_cases():
    """The six synthetic cases of PREREG section 6 K-B4 ARE ``G.kb4_cases()``: the function the bank generator runs at generation time and writes to PROVENANCE.json ``controls`` (the unit test and the record are one
    piece of code, so they cannot disagree)."""
    assert G.KB4_CASES == ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")
    res = G.kb4_cases()
    assert tuple(res) == G.KB4_CASES and res == {c: True for c in G.KB4_CASES}, res
    assert G.kb4_record() == {"PASS": True, "cases": res}
    assert G.kb4_combine([G.kb4_record(), G.kb4_record()])["PASS"] is True


# one bend per case: (an anchor in ``match_route``'s source that occurs exactly once, its replacement). The bent matcher must turn ITS case False -- the deliberate-regression arm of K-B4.
BENDS = {
    "straight": ("    return Match(r, e_lat, dpsi, OK)\n", "    return Match(r + 1, e_lat, dpsi, OK)\n"),
    "stationary_tie": ("    r = int(rows[np.lexsort((rows, np.abs(rows - int(j))))[0]])", "    r = int(rows[0])"),
    "lateral_cut": ("    if e_lat > OFF_ROUTE_LAT_M:\n", "    if False:\n"),
    "heading_cut": ("    if dpsi > OFF_ROUTE_HEAD_DEG:\n", "    if False:\n"),
    "end_edge": ("    if r > T - END_EDGE_ROWS:\n", "    if False:\n"),
    "loop": ("    i0 = int(np.argmin(np.abs(arc - s_star)))", "    i0 = int(np.argmax(np.abs(arc - s_star)))"),          # the row FARTHEST in arc (the loop case has no gentler bend: its own state lies exactly on the path)
}


def bent_match_route(case):
    """``match_route`` with the single edit ``BENDS[case]`` (compiled from its own source, in a copy of the module's namespace)."""
    import inspect
    old, new = BENDS[case]
    src = inspect.getsource(G.match_route)
    assert src.count(old) == 1, f"the bend anchor of {case} must occur exactly once"
    ns = dict(vars(G))
    exec(compile(src.replace(old, new), f"<match_route bent for {case}>", "exec"), ns)
    return ns["match_route"]


@pytest.mark.parametrize("case", G.KB4_CASES)
def test_kb4_each_case_goes_false_when_the_matcher_is_bent(case, monkeypatch):
    monkeypatch.setattr(G, "match_route", bent_match_route(case))
    res = G.kb4_cases()
    assert res[case] is False, f"the bent matcher must fail its case: {res}"
    rec = G.kb4_record()
    assert rec["PASS"] is False and rec["cases"][case] is False, "PASS is derived from the cases"
    comb = G.kb4_combine([{"PASS": True, "cases": {c: True for c in G.KB4_CASES}}])
    assert comb["PASS"] is False and comb["cases"][case] is False, "the merge runs the cases AGAIN: shards written before the matcher was bent do not pass for it"


def test_kb4_a_matcher_that_raises_fails_every_case_and_the_combine_needs_every_shard_to_have_every_case(monkeypatch):
    with monkeypatch.context() as mp:
        mp.setattr(G, "match_route", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert G.kb4_cases() == {c: False for c in G.KB4_CASES}, "a case that raises did not pass"
    ok = G.kb4_record()
    assert G.kb4_combine([ok])["PASS"] is True
    assert G.kb4_combine([])["PASS"] is False, "no shard record = nothing was checked at generation"
    bad = {"PASS": True, "cases": {**ok["cases"], "loop": False}}
    comb = G.kb4_combine([ok, bad])
    assert comb["PASS"] is False and comb["cases"]["loop"] is False and comb["cases"]["straight"] is True, "one shard that recorded a case False fails the bank (its own flag is not trusted)"
    comb = G.kb4_combine([ok, {"PASS": True}])
    assert comb["PASS"] is False, "a shard record with no cases fails"
    comb = G.kb4_combine([ok, {"PASS": True, "cases": {k: v for k, v in ok["cases"].items() if k != "loop"}}])
    assert comb["PASS"] is False and comb["cases"]["loop"] is False, "an absent case is not a True case"


def test_kb4_cut_boundaries_cut_order_and_the_candidate_set():
    pg = _straight()
    assert G.match_route(pg, 5, (40.0, 3.0, 0.0), 30).reason == G.OK, "e_lat == 3.0 m is kept (the cut is strictly greater)"
    assert G.match_route(pg, 5, (40.0, 3.0001, 0.0), 30).reason == G.OFF_LAT
    assert G.match_route(pg, 5, (40.0, 0.0, math.radians(29.9)), 30).reason == G.OK
    assert G.match_route(pg, 5, (40.0, 0.0, math.radians(30.1)), 30).reason == G.OFF_HEAD
    assert G.match_route(pg, 5, (2.0 * 49, 0.0, 0.0), 30).reason == G.OK, "r' == T - 31 has a complete 6 s future"
    assert G.match_route(pg, 5, (2.0 * 50, 0.0, 0.0), 30).reason == G.END_EDGE, "r' == T - 30 has not"
    assert G.match_route(pg, 5, (2.0 * 60, 9.0, math.radians(60)), 70).reason == G.OFF_LAT, "the cut order is lateral, heading, end edge"
    assert G.match_route(pg, 5, (2.0 * 60, 0.0, math.radians(60)), 70).reason == G.OFF_HEAD
    m = G.match_route(pg, 10, (2.0 * 2, 0.0, 0.0), 12)                                             # an own pose NEAR row 2, but the candidates start at t0 = 10
    assert m.r == 10 and m.e_lat == pytest.approx(16.0), "rows before t0 are not candidates"
    m = G.match_route(pg, 5, (float("nan"), 0.0, 0.0), 30)
    assert m.reason == G.NONFINITE and m.r == -1, "a non-finite own pose is counted under its own reason, never matched"
    bad = pg.copy()
    bad[30, 0] = np.nan
    assert G.match_route(bad, 5, (40.0, 0.0, 0.0), 30).reason == G.NONFINITE


def test_kb4_tie_set_is_arc_ties_only_and_ties_break_by_time_then_the_earlier_row():
    pg = _straight()
    # exactly between rows 1 and 2 (arc 2 and 4 from t0 = 0; s* = 3): equally near, NOT an arc tie (2 m apart) -> the earlier row, whatever j says
    assert G.match_route(pg, 0, (3.0, 0.0, 0.0), 2).r == 1 and G.match_route(pg, 0, (3.0, 0.0, 0.0), 40).r == 1
    # 2 cm past the midpoint: row 2 is nearer, and is the answer even when j sits at row 1 (a tie set of "within 0.05 m of the minimum DISTANCE" would let j pull it back)
    assert G.match_route(pg, 0, (3.02, 0.0, 0.0), 1).r == 2
    pg2 = pg.copy()
    pg2[20:30, 0] = 40.0
    pg2[30:, 0] = 40.0 + np.arange(80 - 30) * 2.0                                                  # rows 20..30 stationary at x = 40
    assert G.match_route(pg2, 5, (40.0, 0.0, 0.0), 25).r == 25
    assert G.match_route(pg2, 5, (40.0, 0.0, 0.0), 60).r == 30, "j beyond the stretch: its last row (nearest in time)"
    assert G.match_route(pg2, 5, (40.0, 0.0, 0.0), 10).r == 20, "j before the stretch: its first row"
    assert G.match_route(pg2, 5, (40.0, 0.0, 0.0), 24.5).r == 24, "j = 24.5 truncates to 24 (``int(j)``): the nearest row, NOT a time tie (a time tie cannot occur at an integer j: see the next tests)"
    pg3 = pg.copy()                                                                              # 4 cm creep per row inside the stretch is still ONE tie set
    pg3[20:30, 0] = 40.0 + 0.01 * np.arange(10)
    pg3[30:, 0] = 40.09 + np.arange(80 - 30) * 2.0
    assert G.match_route(pg3, 5, (40.0, 0.0, 0.0), 27).r == 25, "rows 20..25 are within 0.05 m of the nearest arc; the nearest in time to j = 27 is 25"


def test_kb4_a_heading_error_EXACTLY_at_the_cut_is_kept_the_cut_is_strictly_greater():
    """PREREG section 2 item 5: ``|wrap(psi - psi_logged[r'])| > 30 deg`` is cut, so a state AT the cut is KEPT (the heading twin of the 3.0 m lateral case above). Two probes, because 30 deg itself is not
    reachable through ``wrap()`` in binary floating point (``wrap`` adds pi, and the sum lies on a 2**-51 rad grid that does not contain radians(30): the nearest reachable errors are 29.999999999999993 and
    30.000000000000018 deg), so ``> 30.0`` and ``>= 30.0`` never differ at the literal 30.0 -- they differ at the float the matcher actually computes:
    (1) a sweep of ulp-neighbours of radians(30): kept iff the COMPUTED error is <= the cut, and the sweep must straddle the cut;
    (2) the cut moved onto the computed error itself (the constant is pinned to 30.0 by ``test_constants_...``): equality is kept, one ulp below it is cut."""
    pg = _straight()
    y, ys = math.radians(30.0), []
    lo, hi = y, y
    for _ in range(8):
        ys += [lo, hi]
        lo, hi = float(np.nextafter(lo, 0.0)), float(np.nextafter(hi, 1.0))
    kept = cut = 0
    for yaw in sorted(set(ys)):
        m = G.match_route(pg, 5, (40.0, 0.0, yaw), 30)
        assert m.reason in (G.OK, G.OFF_HEAD)
        assert (m.reason == G.OK) == (m.dpsi_deg <= G.OFF_ROUTE_HEAD_DEG), f"yaw {yaw!r}: dpsi {m.dpsi_deg!r} deg -> reason {m.reason}"
        kept, cut = kept + int(m.reason == G.OK), cut + int(m.reason == G.OFF_HEAD)
    assert kept > 0 and cut > 0, "the sweep must straddle the 30 deg cut, or it proves nothing"
    m0 = G.match_route(pg, 5, (40.0, 0.0, math.radians(30.0)), 30)
    assert m0.reason == G.OK and m0.dpsi_deg == pytest.approx(30.0, abs=1e-9)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "OFF_ROUTE_HEAD_DEG", m0.dpsi_deg)                                           # the cut EQUALS this state's error
        assert G.match_route(pg, 5, (40.0, 0.0, math.radians(30.0)), 30).reason == G.OK, "a heading error equal to the cut is kept (strictly greater is cut)"
        mp.setattr(G, "OFF_ROUTE_HEAD_DEG", float(np.nextafter(m0.dpsi_deg, 0.0)))                 # one ulp below: the same state is now over the cut
        assert G.match_route(pg, 5, (40.0, 0.0, math.radians(30.0)), 30).reason == G.OFF_HEAD
    # the lateral twin, with the same device (3.0 m is exactly representable, so the literal case is already in ``cut_boundaries``)
    ml = G.match_route(pg, 5, (40.0, 1.37, 0.0), 30)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "OFF_ROUTE_LAT_M", ml.e_lat)
        assert G.match_route(pg, 5, (40.0, 1.37, 0.0), 30).reason == G.OK
        mp.setattr(G, "OFF_ROUTE_LAT_M", float(np.nextafter(ml.e_lat, 0.0)))
        assert G.match_route(pg, 5, (40.0, 1.37, 0.0), 30).reason == G.OFF_LAT


def test_kb4_a_time_tie_cannot_occur_at_an_integer_clock_row_so_the_nearest_in_time_row_is_the_clamp():
    """PREREG section 2 item 4: among the tied rows, the row nearest in time to j, THEN the earlier row. The tied rows are an arc stretch (arc is non-decreasing), i.e. a CONTIGUOUS run of rows [a, b];
    for an integer j the nearest row is ``clip(j, a, b)`` and it is unique (two rows at equal distance on both sides of j would put j inside the run, at distance 0). So the secondary clause ("then the
    earlier row") never fires through ``match_route``: the mutant "the LATER row on equal time distance" is behaviourally EQUIVALENT, and is pinned by the forced-tie test below instead.
    This test is the proof by exhaustion: every integer j, several stretch shapes (a creep inside the 0.05 m tolerance included)."""
    for a, b, creep in [(20, 40, 0.0), (20, 21, 0.0), (20, 29, 0.005), (7, 30, 0.0)]:
        pg = _straight()
        n = b - a + 1
        pg[a:b + 1, 0] = 40.0 + creep * np.arange(n)
        pg[b + 1:, 0] = 40.0 + creep * (n - 1) + 2.0 * np.arange(1, len(pg) - b)
        for j in range(5, 80):
            m = G.match_route(pg, 5, (40.0, 0.0, 0.0), j)
            assert m.reason == G.OK, (a, b, j, m)
            assert m.r == min(max(j, a), b), f"stretch [{a}, {b}] creep {creep}, j = {j}: the nearest row in time is the clamp, got {m.r}"


def test_kb4_the_secondary_key_of_the_tie_break_is_the_earlier_row_pinned_through_a_forced_time_tie():
    """The clause "then the earlier row" is unreachable at an integer j (see above), so it is pinned through a SEAM, not through reachable behaviour: ``np.lexsort`` is wrapped so that its primary key (the time
    distance) is all zeros -- every tied row equally near in time -- and the answer must be the EARLIER row. White-box by construction: if the matcher stops using ``np.lexsort`` the spy sees no call and this
    test fails loudly (fix the seam, do not delete the test). Mutants it kills: 'the LATER row on equal time distance' (``-rows``)."""
    pg = _straight()
    pg[20:41, 0] = 40.0
    pg[41:, 0] = 40.0 + 2.0 * np.arange(1, len(pg) - 40)                                           # rows 20..40 stationary at x = 40
    assert G.match_route(pg, 5, (40.0, 0.0, 0.0), 35).r == 35                                      # unforced: the nearest in time
    real, seen = np.lexsort, []

    def forced(keys, *a, **k):
        keys = list(keys)
        seen.append([np.asarray(x).copy() for x in keys])
        keys[-1] = np.zeros_like(keys[-1])                                                         # lexsort's LAST key is the primary one: the time distance -> a total tie in time
        return real(tuple(keys), *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np, "lexsort", forced)
        r_forced = G.match_route(pg, 5, (40.0, 0.0, 0.0), 35).r
    assert len(seen) == 1 and len(seen[0]) == 2, "the matcher must break the tie with exactly (secondary, time distance) through np.lexsort: the seam moved"
    assert np.array_equal(seen[0][1], np.abs(seen[0][0] - 35)), "the primary key is the time distance to j"
    assert r_forced == 20, "equally near in time -> the EARLIER row (20), never the later (40)"


# ------------------------------------------------------------------------------------------------ the bank format
def test_shard_roundtrip_is_deterministic_and_the_manifest_refuses_a_tampered_shard(fx10, tmp_path):
    arrays, header = G.load_shard(next(Path(fx10.out).glob("shard_*.npz")))
    for k in G.RECORD_FIELDS:
        assert arrays[k].dtype == np.dtype(G.RECORD_FIELDS[k]), k
    G.save_shard(tmp_path / "a.npz", arrays, header)
    G.save_shard(tmp_path / "b.npz", arrays, header)
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes(), "a fixed zip timestamp: the same arrays are the same bytes"
    assert G.sha256_file(tmp_path / "a.npz") == G.sha256_file(next(Path(fx10.out).glob("shard_*.npz"))), "and they are the bytes the generator wrote"
    a2, h2 = G.load_shard(tmp_path / "a.npz")
    assert h2 == header and all(np.array_equal(a2[k], arrays[k], equal_nan=True) for k in a2)
    assert G.content_sha256(a2) == G.content_sha256(arrays)
    # tamper: same file name, other bytes
    import shutil
    shutil.copytree(fx10.out, tmp_path / "t")
    shard = next((tmp_path / "t").glob("shard_*.npz"))
    a3 = dict(arrays)
    a3["e_lat"] = arrays["e_lat"] + 1.0
    G.save_shard(shard, a3, header)
    with pytest.raises(G.BankRefused, match="sha256 differs"):
        G.DaggerBank.load(tmp_path / "t")
    with pytest.raises(G.BankRefused, match="declared"):
        G.DaggerBank.load(fx10.out, expect_sha256="0" * 64)
    assert G.DaggerBank.load(fx10.out, expect_sha256=fx10.bank.sha256).n == fx10.bank.n, "the control: the declared sha256 loads"


def test_bank_holds_ticks_1_and_2_only_with_the_prereg_record_and_counts_every_cut(fx10):
    a, h = fx10.bank.a, fx10.bank.header
    assert set(np.unique(a["tick"]).tolist()) == {1, 2}, "tick 0 is the logged state and is never an own state"
    assert np.array_equal(a["j"], a["t0"] + 10 * a["tick"].astype(np.int32)), "j = t0 + 10 tick"
    assert a["p10f"].shape[1:] == (10, 8) and a["pose"].shape[1:] == (10, 3) and a["arc"].shape[1] == 10 and a["kin3"].shape[1:] == (3,)
    n_win = len(fx10.wins)
    assert fx10.bank.n == 2 * n_win, "two own states per window, cut or not: nothing is dropped silently"
    assert sum(fx10.bank.reason_counts().values()) == fx10.bank.n
    assert set(fx10.bank.reason_counts()) == set(G.REASONS) and fx10.bank.reason_counts()["ok"] > 0 and fx10.bank.reason_counts()["end_edge"] > 0
    assert h["partial"] is False and h["constants_sha256"] == G.constants_sha256() and h["rollout"] == G.ROLLOUT
    assert h["train"]["n_windows"] == len(fx10.wins) and h["window_list_sha256"] == G.windows_sha256(fx10.J["clip"], fx10.wins)
    # each record against the logged grid: re-run the matcher on the stored own pose
    for i in range(0, fx10.bank.n, 7):
        pg = fx10.store.get(str(fx10.J["clip"][fx10.bank.ci[i]]))["pose"]
        m = G.match_route(pg, int(a["t0"][i]), a["pose"][i, -1], int(a["j"][i]))
        assert (m.r, m.reason) == (int(a["r"][i]), int(a["reason"][i])) and m.e_lat == a["e_lat"][i] and m.dpsi_deg == a["dpsi"][i]
        assert a["r"][i] >= a["t0"][i], "candidates are rows >= t0"
    ok = a["reason"] == G.OK
    assert bool((a["e_lat"][ok] <= 3.0).all() and (a["dpsi"][ok] <= 30.0).all() and (a["r"][ok] <= 90 - 31).all())


@pytest.mark.parametrize("which", ["fx10", "fx16"])
def test_bank_rows_are_the_t1_rollouts_own_states_and_no_logged_future_leaks(which, request):
    """Independent of the generator's snapshot: a full ``v8f_t1_eval.rollout`` (3 ticks, no hook) of the SAME chunk gives the executed path in the t0 ego frame; the bank's world poses at the rows after t0,
    mapped into that frame, must equal it. Rows <= t0 (hmax 16 reaches back past t0) are the LOGGED rows exactly; the rows after t0 are not the logged ones."""
    f = request.getfixturevalue(which)
    hmax = f.cfg.hmax
    chunk = [w for w in f.wins if w[0] == 0][:8]
    states = T1.make_states(f.store, f.tracks, f.J, chunk, "deadreckon")
    proxy = T1._OneClipStore()
    ds_own = D.V8FMiniWindows(proxy, f.J, [], hist=f.cfg.hist, hmax=hmax, max_age_t=f.cfg.max_age_t, max_age_s=f.cfg.max_age_s, window_start=f.cfg.window_start)
    _, exe = T1.rollout(T1.model_planner(f.model, "cpu"), states, ds_own, proxy, m=10, device="cpu", limit="causal")
    recs, _ = B.rollout_chunk(chunk, planner=T1.model_planner(f.model, "cpu"), store=f.store, tracks=f.tracks, J=f.J, ds_log=f.ds, ds_own=ds_own, proxy=proxy, hmax=hmax)
    assert len(recs) == 2 * len(chunk)
    n_past = 0
    for r in recs:
        w = [i for i, c in enumerate(chunk) if c[1] == r["t0"]][0]
        j, t0 = r["j"], r["t0"]
        lg = f.store.get(str(f.J["clip"][r["ci"]]))["pose"]
        rows = np.arange(j - hmax + 1, j + 1)
        after, past = rows > t0, rows <= t0
        now = lg[t0, :3]
        c, s = math.cos(-now[2]), math.sin(-now[2])
        d = r["pose"][after, :2] - now[None, :2]
        ego = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)
        np.testing.assert_allclose(ego, exe[w][rows[after] - t0 - 1], atol=1e-9)
        real_past = past & (rows >= 0)
        assert np.array_equal(r["pose"][real_past], lg[rows[real_past], :3]), "rows <= t0 are the logged rows, exactly"
        assert np.isnan(r["pose"][rows < 0]).all() and np.isnan(r["arc"][rows < 0]).all(), "rows before the clip start are NaN, as in the store"
        n_past += int(real_past.sum())
        assert not np.allclose(r["pose"][after], lg[rows[after], :3]), "rows after t0 are the executed motion, not the logged one"
        assert np.isfinite(r["p10f"][rows >= 1]).all() and np.isnan(r["p10f"][rows < 0]).all() and r["tick"] in (1, 2)
    assert (n_past > 0) == (hmax > 10), "hmax 16 reaches back past t0 at tick 1 (and must then hold logged rows); hmax 10 does not"


def test_kb0_tick0_identity_and_a_leaky_state_is_refused(fx10):
    proxy = T1._OneClipStore()
    ds_own = D.V8FMiniWindows(proxy, fx10.J, [], hist=fx10.cfg.hist, hmax=10, max_age_t=fx10.cfg.max_age_t, max_age_s=fx10.cfg.max_age_s, window_start=fx10.cfg.window_start)
    chunk = fx10.wins[20:28]
    for st in T1.make_states(fx10.store, fx10.tracks, fx10.J, chunk, "deadreckon"):
        own = T1.build_item(ds_own, proxy, st, 0, 0, 0, "logged", "logged")
        log = fx10.ds.item(st.ci, st.t0, 0, 0)
        for k in G.REBUILT + ("pose_now",):
            assert B._same(own[k], log[k]), f"tick-0 {k} differs from the logged item"
    kw = dict(planner=T1.model_planner(fx10.model, "cpu"), store=fx10.store, tracks=fx10.tracks, J=fx10.J, ds_log=fx10.ds, ds_own=ds_own, proxy=proxy, hmax=10)
    recs, k0 = B.rollout_chunk(chunk, **kw)
    assert k0 == {"checked": 8, "mismatch": 0}, "the control: a clean state passes the K-B0 check on every window"

    class Leaky(T1.WindowState):
        """A state whose tick-0 kinematics are not the logged ones (one feature row bent)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.d["p10f"][self.t0] = self.d["p10f"][self.t0] + 1e-3

    with pytest.raises(G.BankRefused, match="K-B0"):
        B.rollout_chunk(chunk, state_cls=Leaky, **kw)


def test_kb1_replay_is_bit_exact_and_a_stochastic_planner_fails_it(fx10):
    prov = json.loads(next(Path(fx10.out).glob("shard_*.provenance.json")).read_text(encoding="utf-8"))
    assert prov["kb1_replay"]["chunks"] >= 1 and prov["kb1_replay"]["mismatch_chunks"] == 0, prov["kb1_replay"]
    proxy = T1._OneClipStore()
    ds_own = D.V8FMiniWindows(proxy, fx10.J, [], hist=fx10.cfg.hist, hmax=10, max_age_t=fx10.cfg.max_age_t, max_age_s=fx10.cfg.max_age_s, window_start=fx10.cfg.window_start)
    kw = dict(wins=fx10.wins[:24], store=fx10.store, tracks=fx10.tracks, J=fx10.J, ds_log=fx10.ds, ds_own=ds_own, proxy=proxy, hmax=10, batch=8, replay_chunks={0, 1, 2})
    _, _, rep = B.run_windows(planner_factory=lambda: T1.model_planner(fx10.model, "cpu"), **kw)
    assert rep == {"chunks": 3, "windows": 24, "mismatch_chunks": 0}

    class Noisy:
        keys_read = set()

        def __init__(self):
            self.inner = T1.model_planner(fx10.model, "cpu")

        def __call__(self, batch):
            return self.inner(batch) + np.random.normal(0.0, 1e-3, (int(batch["p10f"].shape[0]), 30, 2)).astype(np.float32)

    _, _, rep = B.run_windows(planner_factory=Noisy, **kw)
    assert rep["mismatch_chunks"] == 3, "the control: a planner that differs run to run is caught on every replayed chunk"


def test_the_generator_is_deterministic_and_reuses_the_t1_rollout(fx10, world, tmp_path):
    out2 = tmp_path / "again"
    assert B.main(gen_args(world, fx10.ck, fx10.av, out2, "--replay-frac", "0.2")) == 0
    assert G.sha256_file(out2 / "bank.json") == G.sha256_file(Path(fx10.out) / "bank.json"), "the same inputs give the same bank, byte for byte (same sha256: the arm's --dagger-bank-sha256)"
    src = (REPO / "stack" / "scripts" / "v8f_dagger_bank.py").read_text(encoding="utf-8")
    assert "T1.rollout(" in src and "T1.make_states(" in src and "T1.build_item(" in src and "T1.model_planner(" in src, "the rollout is v8f_t1_eval's"
    assert not re.search(r"def (rollout|commit_plan|hermite_frames|build_item|envelope)\b", src), "and is not forked"


# ------------------------------------------------------------------------------------------------ K-B3 / K-B2 / COVERAGE
def test_kb3_a_bank_holding_an_eval_clip_is_refused_and_the_clean_bank_reads_overlap_exactly_zero(fx10, world):
    train = world["names"]["train"]
    ev = list(json.load(open(world["root"] / "eval_grid_windows.json"))["order"])
    sha = fx10.bank.header["window_list_sha256"]
    ok = fx10.bank.check_train_only(train_clips=train, eval_clips=ev, train_window_list_sha256=sha)
    assert ok["overlap_with_eval"] == 0 and ok["n_bank_clips"] == 4 and ok["n_eval_clips"] == 3, "the control"
    with pytest.raises(G.BankRefused, match="eval clip"):                              # deliberate regression: one eval clip in the bank
        G.check_train_only(bank_clips=train + ev[:1], train_clips=train + ev[:1], eval_clips=ev, bank_window_list_sha256=sha, train_window_list_sha256=sha)
    bad = copy.deepcopy(fx10.bank)
    bad.header["clips"] = {**bad.header["clips"], "0": ev[0]}                           # the same bank with its first clip renamed to an eval clip
    with pytest.raises(G.BankRefused, match="eval clip"):
        bad.check_train_only(train_clips=train, eval_clips=ev, train_window_list_sha256=sha)
    with pytest.raises(G.BankRefused, match="not in the arm's TRAIN"):
        fx10.bank.check_train_only(train_clips=train[1:], eval_clips=ev, train_window_list_sha256=sha)
    with pytest.raises(G.BankRefused, match="window-list sha256"):
        fx10.bank.check_train_only(train_clips=train, eval_clips=ev, train_window_list_sha256="f" * 64)


def test_kb3_the_generator_refuses_an_eval_clip_in_the_train_list_before_any_rollout(fx10, world, tmp_path):
    grid = json.load(open(world["root"] / "eval_grid_windows.json"))
    grid["order"] = grid["order"] + world["names"]["train"][:1]                        # an eval grid that contains a TRAIN clip
    gp = tmp_path / "grid_bad.json"
    json.dump(grid, open(gp, "w"))
    args = gen_args(world, fx10.ck, fx10.av, tmp_path / "out_bad")
    args[args.index("--eval-grid") + 1] = str(gp)
    with pytest.raises(SystemExit, match="REFUSED.*eval clip"):
        B.main(args)
    assert not (tmp_path / "out_bad").exists() or not list((tmp_path / "out_bad").iterdir()), "nothing was written"
    with pytest.raises(SystemExit, match="REFUSED.*windows"):
        B.main(gen_args(world, fx10.ck, fx10.av, tmp_path / "o2", "--expect-windows", "12345"))


def test_kb2_on_policy_and_coverage(fx10):
    sha = fx10.bank.header["ckpt_sha256"]
    assert sha == G.sha256_file(fx10.ck) and fx10.bank.check_on_policy(sha.upper()) == {"ckpt_sha256": sha}
    with pytest.raises(G.BankRefused, match="K-B2"):
        fx10.bank.check_on_policy("0" * 64)
    cov = fx10.bank.coverage(fx10.wins)
    ok = fx10.bank.a["reason"] == G.OK
    have = {(int(c), int(r)) for c, r in zip(fx10.bank.ci[ok], fx10.bank.a["r"][ok])}
    brute = sum((w[0], w[1]) in have for w in fx10.wins)
    assert cov["n_windows"] == len(fx10.wins) and cov["n_eligible"] == brute and cov["fraction"] == brute / len(fx10.wins) and cov["min"] == 0.5
    assert cov["blocked"] is (cov["fraction"] < 0.5)
    prov = json.loads(next(Path(fx10.out).glob("PROVENANCE.json")).read_text(encoding="utf-8"))
    assert prov["coverage"]["n_eligible"] == brute and prov["bank_sha256"] == fx10.bank.sha256
    thin = copy.copy(fx10.bank)
    thin.__dict__ = {**fx10.bank.__dict__}
    thin.__init__({**fx10.bank.a, "reason": np.full(fx10.bank.n, G.END_EDGE, np.int8)}, fx10.bank.header, clip_names=fx10.J["clip"])
    assert thin.coverage(fx10.wins) == {"n_windows": len(fx10.wins), "n_eligible": 0, "fraction": 0.0, "min": 0.5, "blocked": True}, "the control: every record cut -> BLOCKED"


def test_dry_run_writes_nothing_limit_clips_marks_partial_and_merge_checks_the_shards(fx10, world, tmp_path, capsys):
    assert B.main(gen_args(world, fx10.ck, fx10.av, tmp_path / "dry", "--dry-run")) == 0
    assert '"mode": "dry-run"' in capsys.readouterr().out and not (tmp_path / "dry").exists(), "--dry-run prints the plan and writes nothing"
    assert B.main(gen_args(world, fx10.ck, fx10.av, tmp_path / "part", "--limit-clips", "2", "--replay-frac", "0")) == 0
    part = G.DaggerBank.load(tmp_path / "part")
    assert part.header["partial"] is True and part.header["limit_clips"] == 2 and len(part.header["clips"]) == 2
    with pytest.raises(G.BankRefused, match="PARTIAL"):                              # a partial bank can never satisfy the arm's gate
        part.check_train_only(train_clips=world["names"]["train"], eval_clips=[], train_window_list_sha256=fx10.bank.header["window_list_sha256"])
    assert part.check_train_only(train_clips=world["names"]["train"], eval_clips=[], allow_partial=True)["n_bank_clips"] == 2
    # two shards: merge refuses until both are there
    sh = tmp_path / "sh"
    assert B.main(gen_args(world, fx10.ck, fx10.av, sh, "--shard", "0", "--n-shards", "2", "--replay-frac", "0")) == 0
    with pytest.raises(SystemExit, match="complete bank"):
        B.main(["merge", "--out-dir", str(sh)])
    assert B.main(gen_args(world, fx10.ck, fx10.av, sh, "--shard", "1", "--n-shards", "2", "--replay-frac", "0")) == 0
    assert B.main(["merge", "--out-dir", str(sh)]) == 0
    merged = G.DaggerBank.load(sh, clip_names=fx10.J["clip"])
    assert merged.n == fx10.bank.n and set(zip(merged.ci.tolist(), merged.a["t0"].tolist(), merged.a["tick"].tolist())) == set(zip(fx10.bank.ci.tolist(), fx10.bank.a["t0"].tolist(), fx10.bank.a["tick"].tolist()))
    assert merged.header["partial"] is False and merged.header["window_list_sha256"] == fx10.bank.header["window_list_sha256"]
    # a shard from another configuration is refused
    other = tmp_path / "other"
    assert B.main(gen_args(world, fx10.ck, fx10.av, other, "--shard", "0", "--n-shards", "2", "--replay-frac", "0", "--batch", "8", "--limit-clips", "4")) == 0
    a1, h1 = G.load_shard(next(sh.glob("shard_001of002.npz")))
    G.save_shard(other / "shard_001of002.npz", a1, h1)
    (other / "shard_001of002.provenance.json").write_text((sh / "shard_001of002.provenance.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit, match="different configuration"):
        B.main(["merge", "--out-dir", str(other)])


def test_an_off_route_own_state_is_stored_with_its_cut_counted_and_is_never_swapped(fx10):
    proxy = T1._OneClipStore()
    ds_own = D.V8FMiniWindows(proxy, fx10.J, [], hist=fx10.cfg.hist, hmax=10, max_age_t=fx10.cfg.max_age_t, max_age_s=fx10.cfg.max_age_s, window_start=fx10.cfg.window_start)

    class Drift:
        """Plans a path 8 m to the left of where the car is: the executed state leaves the route by far more than 3 m."""
        keys_read = set()

        def __call__(self, batch):
            B_ = int(batch["p10f"].shape[0])
            plan = np.zeros((B_, 30, 2), np.float32)
            plan[:, :, 0] = np.arange(1, 31)[None] * 1.0
            plan[:, :, 1] = 8.0
            return plan

    chunk = [w for w in fx10.wins if w[0] == 1][:8]
    recs, _ = B.rollout_chunk(chunk, planner=Drift(), store=fx10.store, tracks=fx10.tracks, J=fx10.J, ds_log=fx10.ds, ds_own=ds_own, proxy=proxy, hmax=10)
    arr = B.records_to_arrays(recs)
    assert len(recs) == 16 and int((arr["reason"] == G.OFF_LAT).sum()) >= 8, np.bincount(arr["reason"], minlength=5)
    assert bool((arr["e_lat"][arr["reason"] == G.OFF_LAT] > 3.0).all())
    bank = G.DaggerBank(arr, {"clips": {str(i): str(c) for i, c in enumerate(fx10.J["clip"])}})
    only_cut = np.nonzero(arr["reason"] != G.OK)[0]
    cut_windows = [(int(arr["ci"][i]), int(arr["r"][i])) for i in only_cut]
    ineligible = [(c, t) for c, t in cut_windows if bank.candidates([c], [t])[0][0] == 0]
    assert ineligible, "the drifting planner left windows whose only own states are cut"
    b = D.collate([fx10.ds.item(c, t, 1, 3) for c, t in ineligible[:6]])
    sw = G.DaggerSwap(bank, p=1.0)
    out = sw.apply(b, 0)
    assert sw.last["n_eligible"] == 0 and sw.last["n_selected"] == 0
    assert all(out[k] is b[k] for k in G.REBUILT), "a cut state is never swapped in"


# ------------------------------------------------------------------------------------------------ the swap
def test_the_partition_is_orcs_and_covers_every_inference_key():
    assert G.REBUILT == D.EGO_PERTURB_REBUILT == ("p10f", "base", "kin3", "p10_t", "pose", "pose_t", "pose_s", "nav")
    assert G.KEPT == D.EGO_PERTURB_KEPT == ("feats", "frame_ok", "limit", "limit_ok", "a_t", "a_s", "pose_now") and G.RECORD == ("act",)
    assert set(M.INFERENCE_INPUT_KEYS) <= set(G.REBUILT) | set(G.KEPT) | set(G.RECORD)
    assert G.LABELS_RE_EXPRESSED == ("lab_s.rc",)


# ------------------------------------------------------------------------------------------------ K-B5, the INDEPENDENT oracles (no bank, no generator, no kb5_measure)
# WHERE THESE SIT IS PART OF THEM.  ``mutate_dagger.py`` runs each mutant's selector with ``-x`` and counts only FAILED lines as a kill.  A test that takes ``fx10`` / ``fx16`` builds its bank through the generator, and since
# the generator now REFUSES a bank whose K-B5 control fails (exit 6, no bank.json), a mutant that breaks the re-anchoring turns those tests into SETUP ERRORS, not failures; if the first such test in file order runs before any
# test that FAILS in its body, the run stops on an ERROR and the mutant is booked as a survivor (MEASURED 2026-10-26: 'no re-anchoring: the own poses stay in the world frame' -> 41 deselected, 1 error; with -x off, 3 tests
# FAIL and 9 ERROR).  So the oracles below take no bank fixture and stay ABOVE every test that does; ``test_kb5_the_oracles_are_collected_before_every_bank_fixture_test`` pins the order.
def _hand_rows():
    """Three rows of own poses [N=3, F=4, 3] and the poses they must have after the rigid motion that puts the own NOW on the logged NOW -- every number below worked out by hand (heading = angle from +x, counter-
    clockwise; 'ahead' = along the heading, 'left' = 90 deg counter-clockwise of it), so nothing here calls cos / sin or the library. Frames, per row: behind, left, ahead, NOW (NOW last, as in a history)."""
    h = math.pi / 2
    own = np.array([
        [[10.0, 1.0, h], [8.0, 5.0, h + 0.25], [10.0, 8.0, h], [10.0, 5.0, h]],                                    # row 0: own NOW (10, 5) facing +y. behind 4 m, left 2 m (yaw + 0.25), ahead 3 m
        [[1.0, 7.0, math.pi], [-4.0, 6.0, math.pi + 0.1], [-9.0, 7.0, math.pi], [-4.0, 7.0, math.pi]],             # row 1: own NOW (-4, 7) facing -x. behind 5 m, left 1 m (yaw + 0.1), ahead 5 m
        [[1.0, 2.0, 3.0], [-4.0, 6.0, 0.0], [0.5, 0.25, -1.0], [7.0, -3.0, 0.5]],                                  # row 2: own NOW == logged NOW (7, -3, 0.5): the identity, whatever the other frames
    ])
    own_now = own[:, -1].copy()
    log_now = np.array([[100.0, -20.0, 0.0], [50.0, 60.0, -h], [7.0, -3.0, 0.5]])                                  # row 0 faces +x at (100, -20); row 1 faces -y at (50, 60)
    want = np.array([
        [[96.0, -20.0, 0.0], [100.0, -18.0, 0.25], [103.0, -20.0, 0.0], [100.0, -20.0, 0.0]],                      # behind 4 -> x 96; left 2 of a +x heading -> y -18; ahead 3 -> x 103
        [[50.0, 65.0, -h], [51.0, 60.0, -h + 0.1], [50.0, 55.0, -h], [50.0, 60.0, -h]],                            # facing -y: behind 5 -> y 65; left of -y is +x -> x 51; ahead 5 -> y 55
        own[2],
    ])
    return own, own_now, log_now, want


def test_kb5_reanchor_rows_is_a_rigid_motion_that_puts_the_own_now_on_the_logged_pose_checked_against_hand_computed_poses():
    own, own_now, log_now, want = _hand_rows()
    got = G.reanchor_rows(own, own_now, log_now)
    assert got.shape == own.shape and got.dtype == np.float64
    assert np.abs(got - want).max() <= 1e-9, f"the re-anchored poses are not the hand-computed ones:\n{got}\nwant\n{want}"
    assert np.abs(own[:2] - want[:2]).max() > 50.0, "the fixture is not vacuous: the own history is far from where the logged frame puts it"
    assert np.array_equal(got[2], own[2]), "own NOW == logged NOW (float32): the row is returned UNCHANGED, bit for bit"
    assert np.abs(got[:2, -1] - log_now[:2]).max() <= 1e-9, "NOW lands on the logged pose at r' (x, y, heading)"
    for n in range(2):                                                                                              # a single state through the one-state wrapper: the same numbers
        assert np.abs(G.reanchor(own[n], own_now[n], log_now[n]) - want[n]).max() <= 1e-9
    assert np.abs(G.reanchor_rows(own[:1], own_now[:1], log_now[:1]) - want[:1]).max() <= 1e-9, "a row does not depend on the other rows of the call"
    assert np.abs(G.reanchor_rows(own[1:2], own_now[1:2], log_now[1:2]) - want[1:2]).max() <= 1e-9


@pytest.fixture(scope="module")
def ds_nobank(world):
    """The logged windows of the synthetic world WITHOUT a bank (so a mutant of the bank generator cannot turn a test that uses this into a setup error)."""
    f = BankFix()
    f.J = D.load_join(world["root"] / "join_train.npz")
    f.tracks = D.load_native_tracks(world["root"] / "native_tracks.pt")
    f.store = D.ClipStore(world["root"], "train", lru=8, kin_mode="yaw", tracks=f.tracks)
    f.wins = D.train_windows(f.store, f.J, D.select_train_clips(world["names"]["train"], 1.0))
    f.ds = D.V8FMiniWindows(f.store, f.J, f.wins, hmax=10)
    return f


def test_kb5_a_swapped_batch_puts_the_rigidly_moved_own_history_back_on_the_logged_pose_and_keeps_pose_now_without_any_bank(ds_nobank):
    """A bank-free swap: the OWN poses are the logged history carried to another world frame by a rigid motion written out here (k * 90 deg about the origin, a translation of ~1 km: exact cos / sin, no library call).
    The swap must bring them back to the logged frame: pose_now is the logged pose at r' (kept, and the own NOW lands on it exactly), the whole history, pose_t, pose_s and lab_s.rc are the logged ones."""
    f = ds_nobank
    wins = [w for w in f.wins if w[1] >= 20][::9][:6]
    n = len(wins)
    assert n == 6 and len({w[0] for w in wins}) >= 2
    b = D.collate([f.ds.item(w[0], w[1], 1, 3) for w in wins])
    F = int(b["pose"].shape[1])
    logged = b["pose"].double().numpy()                                                                             # [n, F, 3]: the logged poses at r' - (F - 1) .. r'
    cs = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]                                                         # (cos, sin) of k * 90 degrees, exactly
    own = np.empty_like(logged)
    for r in range(n):
        c, s = cs[r % 4]
        tx, ty = 1000.0 + 37.0 * r, -500.0 - 11.0 * r
        own[r, :, 0] = c * logged[r, :, 0] - s * logged[r, :, 1] + tx
        own[r, :, 1] = s * logged[r, :, 0] + c * logged[r, :, 1] + ty
        own[r, :, 2] = logged[r, :, 2] + (r % 4) * (math.pi / 2)
    assert np.abs(own[..., :2] - logged[..., :2]).min() > 400.0, "the own history is NOT the logged one: every pose is hundreds of metres away"
    rec = {"p10f": b["p10f"].numpy(), "j": b["t"].numpy(), "kin3": b["kin3"].numpy(), "pose": own, "arc": np.zeros((n, F))}
    out = G.swap_rows(b, torch.arange(n), rec)
    assert torch.equal(out["pose"][:, -1], b["pose_now"]), "the own NOW is the logged pose at r' (bit for bit)"
    assert float((out["pose"].double() - b["pose"].double()).abs().max()) <= 1e-4, "the whole own history is back on the logged poses (a rigid motion, float32 tensors)"
    assert float((out["pose"].double()[..., :2] - torch.from_numpy(own)[..., :2]).abs().min()) > 400.0, "and it is not the world-frame history the record holds"
    assert float((out["pose_t"].double() - b["pose_t"].double()).abs().max()) <= 1e-4 and float((out["pose_s"].double() - b["pose_s"].double()).abs().max()) <= 1e-4
    assert float((out["lab_s"]["rc"] - b["lab_s"]["rc"]).abs().max()) <= 1e-3, "the route checkpoints are re-expressed from the logged S frame into the (same) swapped one: they do not move"


def test_kb5_the_oracles_are_collected_before_every_bank_fixture_test_the_survivor_selector_picks(request):
    """Pins the property the comment above relies on: in the order pytest will run them, the oracles come before every test that builds a bank through the generator AND is picked by the -k selector of the mutant
    'no re-anchoring' in mutate_dagger.py (``kb5 or swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap``)."""
    picked = ("kb5", "swap_reachability", "swap_item", "rc_is", "rebuild_is", "nav_progress", "tick0_swap")
    items = request.session.items
    mine = [i for i, it in enumerate(items) if it.name.startswith(("test_kb5_reanchor_rows_is_a_rigid_motion", "test_kb5_a_swapped_batch_puts_the_rigidly_moved"))]
    bank = [i for i, it in enumerate(items) if {"fx10", "fx16"} & set(it.fixturenames) and any(k in it.name.lower() for k in picked)]
    if len(mine) < 2:
        pytest.skip("an oracle is deselected by -k: the order is moot")
    assert not bank or max(mine) < min(bank), "an oracle sits below a bank-fixture test: the mutation run's -x would stop on a setup ERROR before it"


def test_swap_reachability_differs_exactly_in_the_ego_tensors(fx10):
    wins = eligible_windows(fx10, 24) + ineligible_windows(fx10, 8)
    b = make_batch(fx10, wins)
    sw = G.DaggerSwap(fx10.bank, p=1.0, seed=5)
    out = sw.apply(b, 3)
    el = sw.last["sel"]
    assert int(el.sum()) == 24 and sw.last["n_eligible"] == 24 and sw.last["n_selected"] == 24 and sw.last["n_rows"] == 32
    assert not bool(el[24:].any()), "the ineligible rows are not selected"
    valid_now = b["nav"][:, -1, 4] > 0
    assert int(valid_now[:24].sum()) > 4 and int((~valid_now[:24]).sum()) > 4, "the fixture holds selected rows with and without a route"
    for k in G.REBUILT:                                                              # replaced: every selected row moves, every other row is bit-identical
        assert out[k].shape == b[k].shape and out[k].dtype == b[k].dtype, k
        moved = ((out[k] != b[k]) & ~(torch.isnan(out[k]) & torch.isnan(b[k]))).reshape(32, -1).any(1)
        assert not bool(moved[24:].any()), f"{k}: an unselected row changed"
        if k == "nav":
            same_tok = D.EgoPerturb._nav_masks(b, torch.arange(32))[0]                      # real frames sharing NOW's token and validity
            want = (same_tok[:, :-1] & (b["nav"][:, :-1, 4] > 0)).any(1)                    # a frame BEFORE NOW that can be rebuilt (NOW itself is the anchor: back = 0)
            assert torch.equal(moved[:24], want[:24]), "nav: rebuilt exactly where there is a valid-route frame sharing NOW's token behind NOW; the logged 0.0 stays where there is no route"
            assert int(want[:24].sum()) > 10 and not bool(moved[:24][~valid_now[:24]].any())
        else:
            assert bool(moved[:24].all()), f"{k}: every selected row must change"
    nav_ch = ((out["nav"] != b["nav"]) & ~torch.isnan(b["nav"])).any(0).any(0)
    assert nav_ch.tolist() == [False, False, False, True, False], "nav: only the progress channel is rebuilt (token and validity are the route at r')"
    for k in G.KEPT + G.RECORD + ("gt", "gt_ok", "s_h", "lead_gap_h", "t", "ci"):
        assert out[k] is b[k], f"{k} must be the SAME object"
    for nm in ("lab", "lab_t"):
        assert out[nm] is b[nm], nm
    assert out["lab_s"] is not b["lab_s"] and all(out["lab_s"][k] is b["lab_s"][k] for k in b["lab_s"] if k != "rc"), "lab_s: only rc is replaced"
    rc_moved = (out["lab_s"]["rc"] != b["lab_s"]["rc"]).reshape(32, -1).any(1)
    assert bool(rc_moved[:24].any()) and not bool(rc_moved[24:].any())
    assert set(out) == set(b)
    # a random half: unselected eligible rows are untouched
    half = G.DaggerSwap(fx10.bank, p=0.5, seed=1)
    o2 = half.apply(b, 4)
    s2 = half.last["sel"]
    assert 0 < int(s2.sum()) < 24
    for k in G.REBUILT:
        x, y = o2[k][~s2], b[k][~s2]
        assert bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all()), f"{k}: an unselected row changed"
    assert all(bool((o2[k][s2] != b[k][s2]).reshape(int(s2.sum()), -1).any(1).any()) for k in ("p10f", "kin3", "pose")), "and the selected ones moved"


def test_swap_item_is_the_batch_swap_on_a_batch_of_one(fx10):
    w = eligible_windows(fx10, 6)
    b = make_batch(fx10, w)
    sw = G.DaggerSwap(fx10.bank, p=1.0, seed=2)
    out = sw.apply(b, 9)
    for r in range(6):
        it = fx10.ds.item(w[r][0], w[r][1], 1, 3)
        it2 = G.swap_item(it, fx10.bank.gather([int(sw.last["record_idx"][r])]))
        for k in G.REBUILT:
            assert np.array_equal(it2[k], out[k][r].numpy(), equal_nan=True), k
        assert np.array_equal(it2["lab_s"]["rc"], out["lab_s"]["rc"][r].numpy())
        assert it2["feats"] is it["feats"] and it2["gt"] is it["gt"] and it2["lab"] is it["lab"]


def test_kb5_re_anchoring_keeps_pose_now_and_the_held_frame_relative_transforms():
    rng = np.random.default_rng(0)
    P = np.cumsum(rng.normal(0, 1.0, (4, 10, 3)), axis=1) + np.array([500.0, -300.0, 0.0])           # four own histories far from the origin
    own_now, log_now = P[:, -1].copy(), np.stack([rng.normal(0, 800, 4), rng.normal(0, 800, 4), rng.uniform(-3, 3, 4)], 1).astype(np.float32).astype(np.float64)
    R = G.reanchor_rows(P, own_now, log_now)
    assert np.abs(R[:, -1, :2] - log_now[:, :2]).max() == 0.0, "the own NOW lands on the logged pose at r' exactly"
    assert np.array_equal(R[:, -1, 2], log_now[:, 2])

    def rel(Q, a, b):                                                                    # pose b in the frame of pose a
        c, s = math.cos(-Q[a, 2]), math.sin(-Q[a, 2])
        d = Q[b, :2] - Q[a, :2]
        return np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c, Q[b, 2] - Q[a, 2]])

    for n in range(4):
        for a_, b_ in ((9, 8), (9, 6), (7, 4), (9, 0)):
            assert np.abs(rel(R[n], a_, b_) - rel(P[n], a_, b_)).max() <= 1e-9, "a rigid motion in float64"
    f32 = R.astype(np.float32).astype(np.float64)
    worst = max(np.abs(rel(f32[n], a_, b_) - rel(P[n].astype(np.float32).astype(np.float64), a_, b_))[:2].max() for n in range(4) for a_, b_ in ((9, 8), (9, 6), (7, 4)))
    assert worst <= 1e-4, f"the float32 tensors preserve the relative transforms to {worst:.2e} m"
    same = G.reanchor_rows(P, P[:, -1], P[:, -1])                                        # own NOW == logged NOW: the identity, bit for bit
    assert np.array_equal(same, P)
    m = G.kb5_measure(P, log_now)                                                        # the SAME measurement the generator records in PROVENANCE.json controls.K-B5, on these histories (logged poses up to ~1.5 km out)
    assert m["max_rel_transform_err_float64_m"] <= 1e-9 and m["max_reanchor_vs_pose_now_m"] == 0.0, "the re-anchoring's own arithmetic is exact; what the float32 tensors add at ~1 km is the representation floor, which the GATED numbers include"


def test_kb5_through_the_swap_pose_now_is_kept_and_pose_t_pose_s_are_the_re_anchored_history(fx10):
    w = eligible_windows(fx10, 12)
    b = make_batch(fx10, w)
    sw = G.DaggerSwap(fx10.bank, p=1.0, seed=8)
    out = sw.apply(b, 1)
    rec = fx10.bank.gather(sw.last["record_idx"])
    assert out["pose_now"] is b["pose_now"]
    assert torch.equal(out["pose"][:, -1], b["pose_now"]), "the own NOW is the logged pose at r' (bit for bit)"
    F = 10
    for r in range(12):
        at, as_ = int(b["a_t"][r]), int(b["a_s"][r])
        assert torch.equal(out["pose_t"][r], out["pose"][r, F - 1 - at]) and torch.equal(out["pose_s"][r], out["pose"][r, F - 1 - as_])
        own = rec["pose"][r]
        for ix in (F - 1 - at, F - 1 - as_):
            c, s = math.cos(-own[-1, 2]), math.sin(-own[-1, 2])
            d = own[ix, :2] - own[-1, :2]
            want = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])
            o = out["pose"][r].double().numpy()
            c2, s2 = math.cos(-o[-1, 2]), math.sin(-o[-1, 2])
            d2 = o[ix, :2] - o[-1, :2]
            got = np.array([d2[0] * c2 - d2[1] * s2, d2[0] * s2 + d2[1] * c2])
            assert np.abs(got - want).max() <= 1e-4


def _kb5_histories(n=6, F=10, seed=0):
    """n own world-pose histories far from the origin (like the real clips') and n logged poses at r' (hundreds of metres away, float64)."""
    rng = np.random.default_rng(seed)
    P = np.cumsum(rng.normal(0, 1.0, (n, F, 3)), axis=1) + np.array([500.0, -300.0, 0.0])
    L = np.stack([rng.normal(0, 300, n), rng.normal(0, 300, n), rng.uniform(-3, 3, n)], 1)
    return P, L


def test_kb5_measure_reads_a_correct_rigid_motion_inside_the_tolerance_and_the_re_anchoring_alone_exactly_zero():
    assert G.KB5_TOL_M == 1e-4
    P, L = _kb5_histories()
    P[0, :3] = np.nan                                                                  # a record whose first three rows are before the clip
    m = G.kb5_measure(P, L)
    assert m["n"] == 6 and m["n_pairs"] == 5 * 100 + 7 * 7, "every pair of REAL held frames (a NaN frame is not a held frame)"
    assert m["max_reanchor_vs_pose_now_m"] == 0.0, "the re-anchoring alone puts the own NOW exactly on the sample's float32 pose_now"
    assert 0.0 < m["max_pose_now_err_m"] <= G.KB5_TOL_M, "against the float64 logged pose the float32 quantisation of a pose hundreds of metres out is small and not zero"
    assert 0.0 < m["max_rel_transform_err_m"] <= G.KB5_TOL_M
    assert m["max_pose_now_yaw_err_rad"] <= G.KB5_TOL_M and m["max_rel_transform_yaw_err_rad"] <= G.KB5_TOL_M
    assert all(math.isfinite(v) for v in m.values())
    rec = {**{k: v for k, v in m.items() if k.startswith("max_")}, "n_records": m["n"]}
    assert G.kb5_pass(rec) is True
    # the SAME code, on the same histories as the unit test of the re-anchoring below, through the accumulator the generator uses
    acc = G.Kb5Accumulator()
    acc.update([{"pose": P[i], "log_now": L[i], "reason": G.OK} for i in range(6)] + [{"pose": P[0], "log_now": L[0], "reason": G.NONFINITE}])
    r = acc.record()
    assert r["PASS"] is True and r["n_records"] == 6 and r["n_skipped_nonfinite"] == 1 and r["n_pairs"] == m["n_pairs"]
    assert r["max_rel_transform_err_m"] == m["max_rel_transform_err_m"] and r["tol_m"] == 1e-4


def test_kb5_measure_sees_a_re_anchoring_that_drops_the_heading_and_a_nan_that_appears(monkeypatch):
    P, L = _kb5_histories()
    real = G.reanchor_rows

    def drop_heading(P_, own_now, log_now):                                            # positions rotated, headings not: the held-frame relative transforms are wrong
        out = np.array(real(P_, own_now, log_now))
        out[..., 2] = np.asarray(P_, np.float64)[..., 2]
        return out
    monkeypatch.setattr(G, "reanchor_rows", drop_heading)
    m = G.kb5_measure(P, L)
    assert m["max_rel_transform_err_m"] > 1.0 and m["max_pose_now_yaw_err_rad"] > 0.1
    acc = G.Kb5Accumulator()
    acc.update([{"pose": P[i], "log_now": L[i], "reason": G.OK} for i in range(6)])
    assert acc.record()["PASS"] is False
    monkeypatch.setattr(G, "reanchor_rows", lambda P_, o, l: np.full(np.shape(P_), np.nan))      # a NaN where the record was finite must propagate, never be skipped
    acc = G.Kb5Accumulator()
    acc.update([{"pose": P[i], "log_now": L[i], "reason": G.OK} for i in range(6)])
    r = acc.record()
    assert r["PASS"] is False and not math.isfinite(r["max_rel_transform_err_m"]), "a NaN is a failure"


def test_kb5_pass_is_computed_from_the_numbers_and_needs_a_measured_record():
    ok = {"n_records": 5, "max_pose_now_err_m": 5e-5, "max_rel_transform_err_m": 1e-4, "max_pose_now_yaw_err_rad": 0.0, "max_rel_transform_err_float64_m": 1e-13}
    assert G.kb5_pass(ok) is True, "the tolerance is inclusive"
    assert G.kb5_pass({**ok, "n_records": 0}) is False, "nothing measured is not a pass"
    assert G.kb5_pass({**ok, "max_rel_transform_err_m": 1.0001e-4}) is False
    assert G.kb5_pass({**ok, "max_pose_now_err_m": 2e-4}) is False
    assert G.kb5_pass({**ok, "max_pose_now_yaw_err_rad": 2e-4}) is False
    assert G.kb5_pass({**ok, "max_rel_transform_err_m": float("nan")}) is False
    assert G.kb5_pass({k: v for k, v in ok.items() if k != "max_rel_transform_err_m"}) is False
    assert G.kb5_pass({**ok, "PASS": True, "max_rel_transform_err_m": 1e-3}) is False, "a stored flag is not read"
    a = {**ok, "max_rel_transform_err_m": 2e-5, "max_reanchor_vs_pose_now_m": 0.0, "max_rel_transform_yaw_err_rad": 1e-7, "n_records": 3, "n_skipped_nonfinite": 1, "n_pairs": 30}
    b = {**ok, "max_rel_transform_err_m": 9e-5, "max_reanchor_vs_pose_now_m": 0.0, "max_rel_transform_yaw_err_rad": 3e-7, "n_records": 4, "n_skipped_nonfinite": 2, "n_pairs": 40, "PASS": False}
    c = G.kb5_combine([a, b])
    assert c["PASS"] is True and c["n_records"] == 7 and c["n_skipped_nonfinite"] == 3 and c["n_pairs"] == 70 and c["n_shards"] == 2
    assert c["max_rel_transform_err_m"] == 9e-5 and c["max_rel_transform_yaw_err_rad"] == 3e-7, "the MAX over shards"
    assert G.kb5_combine([a, {**b, "max_rel_transform_err_m": 2e-4}])["PASS"] is False, "one shard over the tolerance fails the bank, whatever its own flag says"
    assert G.kb5_combine([])["PASS"] is False


@pytest.fixture(scope="module")
def kbctl_in(world):
    """A checkpoint + M2b argv for the in-body generations of the control-record tests (no bank is generated here: each test runs ``B.main`` itself, so a generator that breaks FAILS the test)."""
    ck, av, J, model, cfg = write_inputs(world, 10, "off", "kbctl")
    tracks = D.load_native_tracks(world["root"] / "native_tracks.pt")
    return {"ck": ck, "av": av, "J": J, "store": D.ClipStore(world["root"], "train", lru=8, kin_mode="yaw", tracks=tracks)}


def _gen(world, k, out, *extra):
    return B.main(gen_args(world, k["ck"], k["av"], out, "--replay-frac", "0", *extra))


def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _independent_kb5(arrays, J, store):
    """K-B5 recomputed from the stored bank records and the store's LOGGED poses with explicit rotation matrices (no code of the library): (n, worst pose_now error, worst relative-transform error [m],
    sha256 of the logged poses compared, in record order)."""
    n, w_now, w_rel, h = 0, 0.0, 0.0, hashlib.sha256()
    for i in range(len(arrays["ci"])):
        if int(arrays["reason"][i]) == G.NONFINITE:
            continue
        lg = np.asarray(store.get(str(J["clip"][int(arrays["ci"][i])]))["pose"][int(arrays["r"][i])][:3], np.float64)
        h.update(lg.tobytes())
        l32 = lg.astype(np.float32).astype(np.float64)
        P = np.asarray(arrays["pose"][i], np.float64)
        d = l32[2] - P[-1, 2]
        c, s = math.cos(d), math.sin(d)
        dx, dy = P[:, 0] - P[-1, 0], P[:, 1] - P[-1, 1]
        R = np.stack([l32[0] + c * dx - s * dy, l32[1] + s * dx + c * dy, P[:, 2] + d], 1).astype(np.float32).astype(np.float64)
        n += 1
        w_now = max(w_now, float(np.hypot(*(R[-1, :2] - lg[:2]))))
        real = np.nonzero(np.isfinite(P).all(1))[0]
        for a in real:
            for b in real:
                def rel(Q):
                    cc, ss = math.cos(-Q[a, 2]), math.sin(-Q[a, 2])
                    e = Q[b, :2] - Q[a, :2]
                    return np.array([e[0] * cc - e[1] * ss, e[0] * ss + e[1] * cc])
                w_rel = max(w_rel, float(np.hypot(*(rel(R) - rel(P)))))
    return n, w_now, w_rel, h.hexdigest()


def test_kbctl_gen_writes_k_b4_and_k_b5_into_the_shard_and_the_bank_provenance(world, kbctl_in, tmp_path):
    """THE RECORDS THE READER (dag_read.py g_kb4 / g_kb5) REFUSES WITHOUT: ``PROVENANCE.json`` ``controls``. The generator is run IN THE BODY (a generator that fails to write them fails THIS test, not a fixture)."""
    out = tmp_path / "bank"
    assert _gen(world, kbctl_in, out) == 0
    prov, sp = _read(out / "PROVENANCE.json"), _read(out / "shard_000of001.provenance.json")
    c = prov["controls"]
    assert c["K-B4"]["PASS"] is True and c["K-B4"]["cases"] == {k: True for k in ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")}, "the six cases the reader requires, each True"
    k5 = c["K-B5"]
    assert k5["PASS"] is True and k5["tol_m"] == 1e-4
    for k in ("max_pose_now_err_m", "max_rel_transform_err_m"):
        assert isinstance(k5[k], float) and math.isfinite(k5[k]) and 0.0 <= k5[k] <= 1e-4, (k, k5[k])
    assert k5["max_rel_transform_err_float64_m"] <= 1e-9 and k5["max_reanchor_vs_pose_now_m"] == 0.0 and k5["n_shards"] == 1 and k5["n_pairs"] > 0
    arrays, _ = G.load_shard(out / "shard_000of001.npz")
    n_fin = int((arrays["reason"] != G.NONFINITE).sum())
    assert k5["n_records"] == n_fin > 0 and k5["n_records"] + k5["n_skipped_nonfinite"] == prov["n_records"] == len(arrays["ci"]), "measured on the bank's OWN records, all of them but the non-finite"
    assert sp["controls"]["K-B4"] == {"PASS": True, "cases": c["K-B4"]["cases"]} and sp["controls"]["K-B5"]["PASS"] is True and sp["controls"]["K-B5"]["n_records"] == k5["n_records"]
    assert (out / "bank.json").is_file() and prov["bank_sha256"] == G.sha256_file(out / "bank.json") and prov["manifest"] == "bank.json" and prov.get("complete", True) is True
    n, w_now, w_rel, dig = _independent_kb5(arrays, kbctl_in["J"], kbctl_in["store"])
    assert n == k5["n_records"]
    assert sp["controls"]["K-B5"]["log_pose_sha256"] == dig and k5["log_pose_sha256_by_shard"] == [dig], "the K-B5 record pins WHICH logged poses (row r' of each record) it was measured against"
    assert w_now <= 1e-4 and w_rel <= 1e-4 and abs(w_now - k5["max_pose_now_err_m"]) <= 3.5e-5 and abs(w_rel - k5["max_rel_transform_err_m"]) <= 3.5e-5, \
        f"an independent recomputation agrees (to a float32 ulp): recorded {k5['max_pose_now_err_m']:.3e} / {k5['max_rel_transform_err_m']:.3e}, recomputed {w_now:.3e} / {w_rel:.3e}"


def test_kbctl_gen_a_bent_matcher_writes_pass_false_exits_non_zero_and_writes_no_bank(world, kbctl_in, tmp_path, monkeypatch):
    """The deliberate-regression arm of K-B4 at the generator: a matcher bent in one rule makes the generator write PASS false, exit non-zero, and write NO bank.json."""
    monkeypatch.setattr(G, "match_route", bent_match_route("stationary_tie"))
    out = tmp_path / "bank"
    rc = _gen(world, kbctl_in, out)
    assert rc != 0
    prov, sp = _read(out / "PROVENANCE.json"), _read(out / "shard_000of001.provenance.json")
    assert prov["controls"]["K-B4"]["PASS"] is False and prov["controls"]["K-B4"]["cases"]["stationary_tie"] is False and prov["controls"]["K-B4"]["cases"]["straight"] is True
    assert sp["controls"]["K-B4"]["PASS"] is False and prov["controls"]["K-B5"]["PASS"] is True, "only the failed control is False"
    assert prov["complete"] is False and prov["failed_controls"] == ["K-B4"] and prov["bank_sha256"] is None and prov["manifest"] is None
    assert not (out / "bank.json").exists(), "a bank with a failed control must not look complete"
    out2 = tmp_path / "bank2"                                                           # several shards: the shard itself exits non-zero (there is nothing to merge yet)
    assert _gen(world, kbctl_in, out2, "--n-shards", "2", "--shard", "0") != 0
    assert _read(out2 / "shard_000of002.provenance.json")["controls"]["K-B4"]["PASS"] is False


def test_kbctl_gen_a_bent_re_anchoring_fails_k_b5_the_same_way(world, kbctl_in, tmp_path, monkeypatch):
    real = G.reanchor_rows

    def drop_heading(P, own_now, log_now):
        out = np.array(real(P, own_now, log_now))
        out[..., 2] = np.asarray(P, np.float64)[..., 2]
        return out
    monkeypatch.setattr(G, "reanchor_rows", drop_heading)
    out = tmp_path / "bank"
    assert _gen(world, kbctl_in, out) != 0
    prov = _read(out / "PROVENANCE.json")
    k5 = prov["controls"]["K-B5"]
    assert k5["PASS"] is False and k5["max_rel_transform_err_m"] > 1e-4 and prov["controls"]["K-B4"]["PASS"] is True
    assert prov["failed_controls"] == ["K-B5"] and not (out / "bank.json").exists()


def test_kbctl_merge_recomputes_pass_from_the_numbers_takes_the_max_and_refuses_a_shard_without_controls(world, kbctl_in, tmp_path):
    out = tmp_path / "bank2"
    for k in (0, 1):
        assert _gen(world, kbctl_in, out, "--n-shards", "2", "--shard", str(k)) == 0
    assert B.main(["merge", "--out-dir", str(out)]) == 0
    sha0 = G.sha256_file(out / "bank.json")
    p0, p1 = out / "shard_000of002.provenance.json", out / "shard_001of002.provenance.json"
    s0, s1 = _read(p0), _read(p1)
    c = _read(out / "PROVENANCE.json")["controls"]["K-B5"]
    assert c["PASS"] is True and c["n_shards"] == 2 and c["n_records"] == s0["controls"]["K-B5"]["n_records"] + s1["controls"]["K-B5"]["n_records"]
    for k in ("max_pose_now_err_m", "max_rel_transform_err_m"):
        assert c[k] == max(s0["controls"]["K-B5"][k], s1["controls"]["K-B5"][k]), "the MAX over the shards"
    keep = p1.read_text(encoding="utf-8")

    def put(d):
        p1.write_text(json.dumps(d), encoding="utf-8")
        (out / "bank.json").unlink(missing_ok=True)

    d = json.loads(keep)                                                                # a shard that CLAIMS pass but whose number is over the tolerance
    d["controls"]["K-B5"].update(PASS=True, max_rel_transform_err_m=2e-4)
    put(d)
    assert B.main(["merge", "--out-dir", str(out)]) != 0
    prov = _read(out / "PROVENANCE.json")
    assert prov["controls"]["K-B5"]["PASS"] is False and prov["controls"]["K-B5"]["max_rel_transform_err_m"] == 2e-4 and prov["failed_controls"] == ["K-B5"], "PASS is recomputed from the numbers; the max wins"
    assert not (out / "bank.json").exists()
    d = json.loads(keep)                                                                # a shard that recorded a K-B4 case False
    d["controls"]["K-B4"]["cases"]["loop"] = False
    put(d)
    assert B.main(["merge", "--out-dir", str(out)]) != 0
    prov = _read(out / "PROVENANCE.json")
    assert prov["controls"]["K-B4"]["cases"]["loop"] is False and prov["controls"]["K-B4"]["PASS"] is False and prov["failed_controls"] == ["K-B4"]
    d = json.loads(keep)                                                                # a shard from a generator that did not run the controls
    del d["controls"]
    put(d)
    with pytest.raises(SystemExit, match="controls.K-B4"):
        B.main(["merge", "--out-dir", str(out)])
    p1.write_text(keep, encoding="utf-8")                                               # restored: the merge passes again, and the bank is the same bytes (the controls are not in bank.json)
    assert B.main(["merge", "--out-dir", str(out)]) == 0
    assert G.sha256_file(out / "bank.json") == sha0 and _read(out / "PROVENANCE.json")["controls"]["K-B5"]["PASS"] is True


def test_the_rebuild_is_the_stored_own_state_and_agrees_with_orcs_rebuild_at_the_identity(fx10):
    w = eligible_windows(fx10, 10)
    b = make_batch(fx10, w)
    sw = G.DaggerSwap(fx10.bank, p=1.0, seed=3)
    out = sw.apply(b, 0)
    rec = fx10.bank.gather(sw.last["record_idx"])
    assert np.array_equal(out["kin3"].numpy(), rec["kin3"].astype(np.float32))
    assert np.array_equal(out["base"].numpy(), D.kdx_base(rec["kin3"], "kappa")), "base = kdx_base(own kin3), yaw units: ORC's spelling"
    assert np.array_equal(out["p10_t"].numpy(), rec["p10f"][:, -1]) and np.array_equal(out["p10f"][:, -1].numpy(), rec["p10f"][:, -1])
    # the real frames carry the stored own rows; an index < 1 (not a kinematic row) keeps the logged value
    F = 10
    for r in range(10):
        j = int(rec["j"][r])
        for f in range(F):
            if j - (F - 1) + f >= 1:
                assert np.array_equal(out["p10f"][r, f].numpy(), rec["p10f"][r, f])
    # cross-check with ORC's own rebuild at the identity (s = 1, d = 0, n = 0) on a tick-0 own state: they agree to float32 rounding
    proxy = T1._OneClipStore()
    sts = T1.make_states(fx10.store, fx10.tracks, fx10.J, w, "deadreckon")
    recs = [B.snapshot(st, 0, 10, fx10.store.get(str(fx10.J["clip"][st.ci]))["pose"]) for st in sts]
    bank0 = G.DaggerBank(B.records_to_arrays(recs), {"clips": {str(i): str(c) for i, c in enumerate(fx10.J["clip"])}})
    o0 = G.DaggerSwap(bank0, p=1.0).apply(b, 0)
    rows = torch.arange(10)
    orc = D.EgoPerturb._rebuilt(b, rows, torch.ones(10, dtype=torch.float64), torch.zeros(10, dtype=torch.float64), torch.zeros(10, dtype=torch.float64))
    for k in ("p10f", "kin3", "base", "p10_t"):
        assert torch.equal(o0[k], b[k]), f"{k}: a tick-0 own state swapped into the logged item is the logged tensor, bit for bit"
        assert float((orc[k] - o0[k]).abs().max()) <= 1e-6, f"{k}: ORC's rebuild at the identity differs from the stored state by float32 rounding only"


def test_nav_progress_follows_the_own_arc_and_leaves_the_token_alone(fx10):
    w = eligible_windows(fx10, 16)
    b = make_batch(fx10, w)
    sw = G.DaggerSwap(fx10.bank, p=1.0, seed=4)
    out = sw.apply(b, 2)
    rec = fx10.bank.gather(sw.last["record_idx"])
    F, n_changed_frames, n_kept = 10, 0, 0
    for r in range(16):
        nav, t = b["nav"][r].numpy(), int(b["t"][r])
        j = int(rec["j"][r])
        arc = rec["arc"][r]
        for f in range(F):
            real = (t - (F - 1) + f >= 1) and bool(b["frame_ok"][r, f]) and (j - (F - 1) + f >= 1)
            same = (nav[f, :3].argmax() == nav[-1, :3].argmax()) and nav[f, 4] == nav[-1, 4]
            if real and same and nav[f, 4] > 0:
                want = np.clip(float(nav[-1, 3]) + (arc[-1] - arc[f]) / 100.0, 0.0, 3.0)
                assert out["nav"][r, f, 3].item() == pytest.approx(want, abs=1e-6)
                n_changed_frames += 1
            else:
                assert out["nav"][r, f, 3].item() == nav[f, 3], "a frame across a manoeuvre boundary / not real / without a route keeps its logged value"
                n_kept += int(bool(b["frame_ok"][r, f]) and (t - (F - 1) + f >= 1) and not same)
        assert np.array_equal(out["nav"][r][:, [0, 1, 2, 4]].numpy(), nav[:, [0, 1, 2, 4]])
    assert n_changed_frames > 0
    assert sw.last["dagger_nav_kept"] >= n_kept > 0, "frames across a boundary are counted"
    big = b["nav"].clone()
    big[:, :, 3] = 2.99                                                                  # the clamp: 2.99 + arc / 100 cannot exceed 3
    bb = dict(b)
    bb["nav"] = big
    o2 = G.DaggerSwap(fx10.bank, p=1.0, seed=4).apply(bb, 2)
    assert float(o2["nav"][..., 3].max()) <= 3.0 and float(o2["nav"][..., 3].min()) >= 0.0


@pytest.mark.parametrize("case", ["token", "validity"])
def test_nav_progress_is_rebuilt_only_on_frames_sharing_NOWs_token_and_validity(fx10, case):
    """PREREG section 2 (the table): channel 3 is rebuilt "on frames sharing NOW's token and validity (ORC's rule)". The natural batch rarely holds a REAL, route-valid frame with ANOTHER token, so the
    test builds one: every frame carries token 0 with a route and logged progress 2.0, NOW's progress is 0.5 (so a rebuild of ANY frame moves it, even where the own arc behind NOW is 0 m).
    * ``token``: frames 3..5 carry token 1 (a manoeuvre boundary inside the history, route still valid) -> kept at exactly 2.0; the other real frames follow the own arc.
    * ``validity``: NOW has NO route (channel 4 = 0) while every earlier frame has one -> every earlier frame is across the validity boundary, none is rebuilt, the whole nav is the logged tensor.
    Mutant it kills: the token / validity test dropped (``nav_ok = own_ok & (valid > 0)``: a frame across the boundary would be rebuilt)."""
    w = eligible_windows(fx10, 10)
    b = make_batch(fx10, w)
    F, n = int(b["nav"].shape[1]), len(w)
    nav = b["nav"].clone()
    nav[..., :3], nav[..., 0], nav[..., 3], nav[..., 4] = 0.0, 1.0, 2.0, 1.0
    nav[:, -1, 3] = 0.5
    other = [3, 4, 5]
    if case == "token":
        nav[:, other, 0], nav[:, other, 1] = 0.0, 1.0
    else:
        nav[:, -1, 4] = 0.0
    bb = dict(b)
    bb["nav"] = nav
    sw = G.DaggerSwap(fx10.bank, p=1.0, seed=4)
    out = sw.apply(bb, 2)
    assert sw.last["n_selected"] == n, "every row is swapped (p = 1, all eligible)"
    rec = fx10.bank.gather(sw.last["record_idx"])
    got = out["nav"]
    assert np.array_equal(got[..., [0, 1, 2, 4]].numpy(), nav[..., [0, 1, 2, 4]].numpy()), "token and validity are never touched"
    n_boundary, n_rebuilt = 0, 0
    for r in range(n):
        t, j, arc = int(b["t"][r]), int(rec["j"][r]), rec["arc"][r]
        for f in range(F):
            real = (t - (F - 1) + f >= 1) and bool(b["frame_ok"][r, f]) and (j - (F - 1) + f >= 1)
            across = (case == "token" and f in other) or (case == "validity" and f < F - 1)
            if real and not across and f != F - 1 and case == "token":
                want = float(np.clip(0.5 + (arc[-1] - arc[f]) / 100.0, 0.0, 3.0))
                assert got[r, f, 3].item() == pytest.approx(want, abs=1e-6), f"row {r} frame {f}: a frame sharing NOW's token and validity follows the own arc"
                n_rebuilt += 1
            elif real and across:
                assert got[r, f, 3].item() == 2.0, f"row {r} frame {f}: a REAL frame across a {case} boundary keeps its logged progress"
                n_boundary += 1
            else:
                assert got[r, f, 3].item() == nav[r, f, 3].item(), f"row {r} frame {f}: not a real own frame / NOW itself: as logged"
    assert n_boundary > 0, "the case must contain real frames across the boundary, or the test is vacuous"
    assert (n_rebuilt > 0) == (case == "token")
    assert sw.last["dagger_nav_kept"] >= n_boundary, "frames across a boundary are counted"
    if case == "validity":
        assert torch.equal(got[..., 3], nav[..., 3]), "no route at NOW: nav progress is exactly the logged tensor on every frame"


def test_rc_is_re_expressed_not_changed(fx10):
    w = eligible_windows(fx10, 20)
    b = make_batch(fx10, w, a_t=1, a_s=3)
    sw = G.DaggerSwap(fx10.bank, p=1.0, seed=6)
    out = sw.apply(b, 5)
    ls, lo = b["lab_s"], out["lab_s"]
    src, dst = b["pose_s"].double(), out["pose_s"].double()
    for r in range(20):
        for m in range(3):
            if not bool(ls["rc_ok"][r, m]):
                assert torch.equal(lo["rc"][r, m], ls["rc"][r, m]), "a placeholder (rc_ok False) is not a world point: untouched"
                continue
            x, y, psi = ls["rc"][r, m].double().tolist()
            wxy_old = src[r, :2] + torch.tensor([math.cos(src[r, 2]) * x - math.sin(src[r, 2]) * y, math.sin(src[r, 2]) * x + math.cos(src[r, 2]) * y], dtype=torch.float64)
            x2, y2, psi2 = lo["rc"][r, m].double().tolist()
            wxy_new = dst[r, :2] + torch.tensor([math.cos(dst[r, 2]) * x2 - math.sin(dst[r, 2]) * y2, math.sin(dst[r, 2]) * x2 + math.cos(dst[r, 2]) * y2], dtype=torch.float64)
            assert float((wxy_old - wxy_new).abs().max()) <= 5e-3, "the SAME world point (float32 coordinates of ~1e3 m)"
            assert abs(math.remainder((psi + float(src[r, 2])) - (psi2 + float(dst[r, 2])), 2 * math.pi)) <= 1e-4, "and the same world heading"
    assert float((lo["rc"] - ls["rc"]).abs().max()) > 0.0, "the frame differs from the logged one at a_s = 3 (the lever does something)"
    b0 = make_batch(fx10, w, a_t=0, a_s=0)
    out0 = G.DaggerSwap(fx10.bank, p=1.0, seed=6).apply(b0, 5)
    assert torch.equal(out0["lab_s"]["rc"], b0["lab_s"]["rc"]), "a_s = 0: the S frame IS the logged frame at r' -- the identity, bit for bit"


def test_draws_are_a_function_of_seed_and_step_and_never_touch_the_global_rng(fx10):
    b = make_batch(fx10, eligible_windows(fx10, 32))
    s = lambda seed, step, p=0.5: G.DaggerSwap(fx10.bank, p=p, seed=seed)
    sw = s(7, 0)
    sw.apply(b, 11)
    a = (sw.last["sel"].clone(), sw.last["record_idx"].copy())
    torch.manual_seed(123)
    st0 = torch.get_rng_state().clone()
    np_state = np.random.get_state()[1].copy()
    sw2 = s(7, 0)
    sw2.apply(b, 11)
    assert torch.equal(torch.get_rng_state(), st0) and np.array_equal(np.random.get_state()[1], np_state), "the global RNGs are untouched"
    assert torch.equal(sw2.last["sel"], a[0]) and np.array_equal(sw2.last["record_idx"], a[1]), "same (seed, step): the same draws"
    sw3 = s(7, 0)
    sw3.apply(b, 12)
    sw4 = s(8, 0)
    sw4.apply(b, 11)
    assert not torch.equal(sw3.last["sel"], a[0]) and not torch.equal(sw4.last["sel"], a[0]), "another step or another seed draws something else"
    torch.manual_seed(999)
    sw5 = s(7, 0)
    sw5.apply(b, 11)
    assert torch.equal(sw5.last["sel"], a[0]), "and a different global RNG state changes nothing"


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.9])
def test_p_is_a_parameter_the_selected_fraction_follows_it(fx10, p):
    b = make_batch(fx10, eligible_windows(fx10, 32))
    sw = G.DaggerSwap(fx10.bank, p=p, seed=1)
    n_sel = n_el = 0
    for step in range(200):
        sw.apply(b, step)
        n_sel += sw.last["n_selected"]
        n_el += sw.last["n_eligible"]
    sd = math.sqrt(p * (1 - p) / n_el)
    assert abs(n_sel / n_el - p) <= 4 * sd, (p, n_sel / n_el)


def test_p_zero_returns_the_same_batch_p_one_swaps_every_eligible_row(fx10):
    b = make_batch(fx10, eligible_windows(fx10, 8) + ineligible_windows(fx10, 4))
    sw = G.DaggerSwap(fx10.bank, p=0.0)
    assert sw.apply(b, 0) is b and sw.last["n_selected"] == 0 and sw.last["n_eligible"] == 8 and sw.last["n_rows"] == 12
    assert G.DaggerSwap(fx10.bank, p=0.4).apply(b, 0, p=0.0) is b, "the gate's p override reaches the off path"
    one = G.DaggerSwap(fx10.bank, p=0.4)
    one.apply(b, 0, p=1.0)
    assert one.last["n_selected"] == 8
    with pytest.raises(ValueError, match="REFUSED"):
        G.DaggerSwap(None, p=0.5).apply(b, 0)
    with pytest.raises(ValueError, match="REFUSED"):
        G.DaggerSwap(fx10.bank, p=1.5).apply(b, 0)


def test_the_draw_among_a_windows_eligible_own_states_is_uniform_and_only_eligible(fx10):
    bank = fx10.bank
    cnt, _ = bank.candidates([w[0] for w in fx10.wins], [w[1] for w in fx10.wins])
    multi = [(w, int(c)) for w, c in zip(fx10.wins, cnt) if c >= 2]
    assert multi, "the fixture holds windows with several eligible own states"
    (ci, t, _, _), c = multi[0]
    ids = [int(bank.pick([ci], [t], [u])[0]) for u in np.linspace(0.0, 0.999, 60)]
    assert len(set(ids)) == c and ids == sorted(ids, key=ids.index), "u sweeps every candidate, in bank order"
    for i in set(ids):
        assert bank.a["reason"][i] == G.OK and bank.ci[i] == ci and bank.a["r"][i] == t, "and every candidate is an eligible state of exactly that window"
    b = make_batch(fx10, [(ci, t, 0, 0)] * 64)
    sw = G.DaggerSwap(bank, p=1.0, seed=2)
    sw.apply(b, 0)
    seen = np.bincount(np.searchsorted(sorted(set(ids)), sw.last["record_idx"]), minlength=c)
    assert len(seen) == c and (seen > 0).all() and seen.max() < 64, f"64 draws reach all {c} candidates: {seen.tolist()}"
    assert bank.pick([ci], [10**6], [0.5])[0] == -1, "a window without an eligible state has no draw"


def test_declared_and_inert_levers_are_problems(fx10):
    assert G.DaggerSwap().problems() == [], "DAG-0: p = 0 and nothing else named"
    assert G.DaggerSwap(fx10.bank, p=0.3, seed=1).problems() == []
    assert any("declared and does nothing" in x for x in G.DaggerSwap(fx10.bank, p=0.0).problems()), "a bank named at p = 0 does nothing"
    assert any("declared and does nothing" in x for x in G.DaggerSwap(None, p=0.0, seed=4).problems())
    assert any("no bank" in x for x in G.DaggerSwap(None, p=0.3).problems())
    assert any("kin-mode yaw" in x for x in G.DaggerSwap(fx10.bank, p=0.3, kin_mode="steer").problems())
    assert any("not in [0, 1]" in x for x in G.DaggerSwap(fx10.bank, p=2.0).problems())
    lv = G.DaggerSwap(fx10.bank, p=0.1, seed=3).built_levers()
    assert lv["dagger_p"] == 0.1 and lv["dagger_seed"] == 3 and lv["dagger_bank_sha256"] == fx10.bank.sha256 and lv["rebuilt"] == list(G.REBUILT)
    with pytest.raises(ValueError, match="another hmax"):                              # a bank of another history length is refused
        G.swap_rows(make_batch(fx10, eligible_windows(fx10, 2)) | {}, torch.tensor([0, 1]), {**fx10.bank.gather([0, 1]), "p10f": np.zeros((2, 12, 8), np.float32)})


# ------------------------------------------------------------------------------------------------ the loss
def _grads(model, b, seed=20261026):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for _, p in named:
        p.grad = None
    for p in model.prior.parameters():
        p.grad = None
    torch.manual_seed(seed)
    T = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
    T["total"].backward()
    g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
    gp = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in model.prior.parameters()])
    return float(T["total"].detach()), g, gp


def _gdiff(ga, gb, pre=""):
    return float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))


@pytest.mark.parametrize("which", ["fx10", "fx16"])
def test_the_swap_reaches_the_loss_gradient_and_the_zero_controls_read_exactly_zero(which, request):
    f = request.getfixturevalue(which)
    wm = which == "fx16"
    b = make_batch(f, eligible_windows(f, 32))
    model = f.model
    model.train()
    sw = G.DaggerSwap(f.bank, p=1.0, seed=1)
    t0, g0, gp0 = _grads(model, b)
    t0b, g0b, _ = _grads(model, b)
    assert t0 == t0b and _gdiff(g0, g0b) == 0.0, "the control: the same loss twice differs by exactly 0.0"
    sw_b = sw.apply(b, 1)
    t1, g1, gp1 = _grads(model, sw_b)
    assert t1 != t0 and _gdiff(g1, g0) > 0.0, "the swapped batch moves the loss and its gradient"
    for grp in ("e_ego", "read_o", "head_o"):
        assert _gdiff(g1, g0, grp) > 0.0, f"the swap reaches {grp}"
    assert float((gp1 - gp0).abs().sum()) == 0.0, "the frozen P10 prior's gradient difference is exactly 0.0"
    if wm:
        assert _gdiff(g1, g0, "wm.") == 0.0, "act (the WM's transition record) is kept: the WM auxiliary reads exactly 0.0"
    zero = sw.apply(b, 1, p=0.0)
    assert zero is b
    t2, g2, _ = _grads(model, zero)
    assert t2 == t0 and _gdiff(g2, g0) == 0.0, "p = 0: loss and gradient differences are exactly 0.0"
    # replacing p10f alone moves the loss (the ego token reaches the head)
    only = dict(b)
    only["p10f"] = sw_b["p10f"]
    t3, _, _ = _grads(model, only)
    assert t3 != t0


def test_a_tick0_swap_is_the_identity_on_the_whole_batch_and_the_loss_difference_is_exactly_zero(fx10):
    w = fx10.wins[24:56]
    b = make_batch(fx10, w)
    sts = T1.make_states(fx10.store, fx10.tracks, fx10.J, w, "deadreckon")
    recs = [B.snapshot(st, 0, 10, fx10.store.get(str(fx10.J["clip"][st.ci]))["pose"]) for st in sts]
    arr = B.records_to_arrays(recs)
    bank0 = G.DaggerBank(arr, {"clips": {str(i): str(c) for i, c in enumerate(fx10.J["clip"])}})
    assert bool((arr["r"] == arr["t0"]).all() and (arr["reason"] == G.OK).all()), "at tick 0 the matched row is t0 itself"
    # make the logged nav progress consistent with the logged arc (the synthetic join draws it at random), so that the identity can be asserted on nav as well
    F = 10
    arc = torch.from_numpy(arr["arc"])
    back = arc[:, -1:] - arc
    nav = b["nav"].clone()
    same = (nav[..., :3].argmax(-1) == nav[:, -1:, :3].argmax(-1)) & (nav[..., 4] == nav[:, -1:, 4])
    real = D.EgoPerturb._hist_valid(b, torch.arange(len(w)))
    nav3 = (nav[:, -1:, 3].double() + back / 100.0).clamp(0.0, 3.0).float()
    nav[..., 3] = torch.where(same & real & (nav[..., 4] > 0), nav3, nav[..., 3])
    bc = dict(b)
    bc["nav"] = nav
    out = G.DaggerSwap(bank0, p=1.0).apply(bc, 0)
    for k in G.REBUILT:
        assert torch.equal(out[k], bc[k]), f"{k}: own state == logged state -> the swap is the identity, bit for bit"
    assert torch.equal(out["lab_s"]["rc"], bc["lab_s"]["rc"])
    model = fx10.model
    model.train()
    ta, ga, _ = _grads(model, bc)
    tb, gb, _ = _grads(model, out)
    assert ta == tb and _gdiff(ga, gb) == 0.0, "the control that must read EXACTLY equal: loss and gradient"
