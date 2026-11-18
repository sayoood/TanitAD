"""PREREG_V8F_NAVRBD section 2 / 3 / 6 (G-CLOCK-RBD): the 15-column roundabout channel table builder.  CPU, synthetic fixtures, no data files.

Run the WHOLE file (module fixtures): ``CUDA_VISIBLE_DEVICES=-1 pytest stack/tests/test_navtext_nav_rbd.py``.  The mutation driver points ``NAVRBD_BUILDER`` at a mutated copy of the builder
(next to a copy of build_navtext_nav.py); the same tests then must go RED (table in the build report).
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[2]
SCRIPTS = TREE / "stack" / "scripts"
BUILDER = Path(os.environ.get("NAVRBD_BUILDER", str(SCRIPTS / "build_navtext_nav_rbd.py")))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load(BUILDER, "build_navtext_nav_rbd_under_test")
RB2 = _load(SCRIPTS / "build_navtext_nav_rb.py", "build_navtext_nav_rb_ref")
CN = _load(SCRIPTS / "build_navtext_nav.py", "build_navtext_nav_ref_t")

NavTok = namedtuple("NavTok", "cls dist_m exit_n")
VOCAB = np.array(["CONTINUE", "TURN_L", "TURN_R", "RB_ENTER", "RB_TAKE_EXIT", "RB_EXIT", "ROTARY_EXIT", "UNPARSED"])
TEXTS = ["Continue", "Turn left in 50m", "Enter roundabout", "At roundabout, take exit 2", "At roundabout, take exit 2 in 80m",
         "At roundabout, take exit 9 in 30m", "Exit roundabout", "At roundabout, take exit 1 in 120m", "Turn right in 400m", "At roundabout, take exit 3 in 5m"]
_P = {"Continue": NavTok("CONTINUE", None, None), "Turn left in 50m": NavTok("TURN_L", 50.0, None), "Enter roundabout": NavTok("RB_ENTER", None, None),
      "At roundabout, take exit 2": NavTok("RB_TAKE_EXIT", None, 2), "At roundabout, take exit 2 in 80m": NavTok("RB_TAKE_EXIT", 80.0, 2),
      "At roundabout, take exit 9 in 30m": NavTok("RB_TAKE_EXIT", 30.0, 9), "Exit roundabout": NavTok("RB_EXIT", None, None),
      "At roundabout, take exit 1 in 120m": NavTok("RB_TAKE_EXIT", 120.0, 1), "Turn right in 400m": NavTok("TURN_R", 400.0, None),
      "At roundabout, take exit 3 in 5m": NavTok("RB_TAKE_EXIT", 5.0, 3)}


def parse_full(t):
    return _P[t]


class Tab(dict):
    files = property(lambda self: list(self.keys()))


def make_world(seed=0, n_clips=3, n=60):
    """Two splits' worth of one synthetic world: table rows per step, a 10 Hz track per clip (first two rows NaN), now_s per step."""
    rng = np.random.RandomState(seed)
    cls_id_t = np.array([int(np.flatnonzero(VOCAB == _P[t].cls)[0]) for t in TEXTS], np.int16)
    dist_t = np.array([np.nan if _P[t].dist_m is None else _P[t].dist_m for t in TEXTS], np.float32)
    N = n_clips * n
    cls = np.full(N, -1, np.int16)
    flag = np.zeros(N, np.uint8)
    dist = np.full(N, np.nan, np.float32)
    ti = np.full(N, -1, np.int32)
    age = np.full(N, np.nan, np.float32)
    now = np.zeros(N)
    tracks, clips = {}, []
    for c in range(n_clips):
        o = c * n
        clips.append(f"clip{c}")
        t_now = 3.0 + 0.2 * np.arange(n)
        now[o:o + n] = t_now
        speed = 4.0 + 3.0 * rng.rand()
        x = np.cumsum(np.r_[0.0, np.full(2 * n - 1, speed * 0.1)])
        tr = np.stack([x, 0.1 * x, np.zeros(2 * n), np.full(2 * n, speed)], 1)
        tr[:2] = np.nan
        tracks[f"clip{c}"] = tr
        flag[o:o + 2] = 1                                                    # steps before the clip's first row
        j, cur, start = 2, int(rng.randint(len(TEXTS))), t_now[2] - 0.05
        while j < n:
            run = int(rng.randint(3, 9))
            for k in range(j, min(n, j + run)):
                ti[o + k] = cur
                cls[o + k], dist[o + k] = cls_id_t[cur], dist_t[cur]
                age[o + k] = t_now[k] - start
            j += run
            cur, start = int(rng.randint(len(TEXTS))), t_now[min(j, n - 1)] - 0.05
        flag[o + 5] = 3                                                      # a step without a v9 row
        cls[o + 5], dist[o + 5], ti[o + 5], age[o + 5] = -1, np.nan, -1, np.nan
    tab = Tab({"vocab": VOCAB, "eval_clip": np.array(clips), "eval_off": np.arange(n_clips) * n, "eval_n": np.full(n_clips, n), "eval_cls_id": cls, "eval_flag": flag,
               "eval_dist_m": dist, "eval_text_i": ti, "eval_age_s": age})
    J = {"clip": np.array(clips), "off": np.arange(n_clips) * n, "n": np.full(n_clips, n), "now_s": now}
    return tab, J, tracks


@pytest.fixture(scope="module")
def world():
    tab, J, tracks = make_world()
    ex_t = B.exit_array(TEXTS, parse_full)
    return tab, J, tracks, ex_t


def cols_of(w, shift=None):
    tab, J, tracks, ex_t = w
    return B.build_columns(tab, "eval", ex_t, J, tracks, B.LIVE_SHIFT if shift is None else shift)[0]


# ------------------------------------------------------------------ imports and admissibility
def test_tanitad_is_the_study_tree():
    sys.path.insert(0, str(TREE / "stack"))
    import tanitad
    assert Path(tanitad.__file__).resolve().is_relative_to(TREE), tanitad.__file__


def test_ast_guard_route_input_only():
    """Section 3: no situation module, no join label, no v9 nav, no future track: nothing of the kind is imported or read by key."""
    src = BUILDER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = set()
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Import):
            names |= {a.name for a in nd.names}
        elif isinstance(nd, ast.ImportFrom):
            names.add(nd.module or "")
    assert not [x for x in names if "situation" in x.lower() or "tanitad" in x.lower()], names
    keys = {nd.value for nd in ast.walk(tree) if isinstance(nd, ast.Constant) and isinstance(nd.value, str) and len(nd.value) < 40}
    assert not (keys & {"lat_cls", "has", "nav", "nav_d", "lon_cls", "sit_cls"}), keys & {"lat_cls", "has", "nav", "nav_d", "lon_cls", "sit_cls"}
    assert "J[\"lat_cls\"]" not in src and "J['nav']" not in src


# ------------------------------------------------------------------ hand-computed columns on a tiny fixture
def test_hand_computed_columns():
    vocab = VOCAB
    # one clip, 6 steps. step: text, flag.  now_s = 10 + 0.2 j; the track moves 1.0 m per 10 Hz row (5 m/s): 2 rows per step = 1 m per step
    ti = np.array([-1, 2, 3, 4, 6, 5], np.int32)                 # -, Enter, exit2, exit2 in 80m, Exit roundabout, exit9 in 30m
    cls = np.array([-1, 3, 4, 4, 5, 4], np.int16)
    flag = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    dist = np.array([np.nan, np.nan, np.nan, 80.0, np.nan, 30.0], np.float32)
    age = np.array([np.nan, 0.1, 0.3, 0.2, 0.2, 0.1], np.float32)
    tab = Tab({"vocab": vocab, "eval_clip": np.array(["c"]), "eval_off": np.array([0]), "eval_n": np.array([6]), "eval_cls_id": cls, "eval_flag": flag, "eval_dist_m": dist, "eval_text_i": ti, "eval_age_s": age})
    J = {"clip": np.array(["c"]), "off": np.array([0]), "n": np.array([6]), "now_s": 10.0 + 0.2 * np.arange(6)}
    tr = np.stack([np.arange(12) * 1.0, np.zeros(12), np.zeros(12), np.full(12, 5.0)], 1)
    tr[:2] = np.nan
    ex_t = B.exit_array(TEXTS, parse_full)
    out = B.build_columns(tab, "eval", ex_t, J, {"c": tr}, B.LIVE_SHIFT)[0]
    exp = np.zeros((6, 15), np.float32)
    exp[1:, 0] = 1                                                # RB at steps 1..5 (flag 0, RB class; step 4 is RB_EXIT, also a roundabout class)
    exp[2, 1 + 1] = 1; exp[3, 1 + 1] = 1; exp[5, 1 + 5] = 1       # exit_cur: exit 2 (bin 1) at 2, 3; exit 9 -> 6+ (bin 5) at 5; none at 1, 4
    exp[2, 7 + 1] = 1; exp[3, 7 + 1] = 1; exp[4, 7 + 1] = 1; exp[5, 7 + 5] = 1      # exit_last: backward only (step 1 none; step 4 keeps exit 2)
    exp[3, 14] = 1; exp[5, 14] = 1                                 # dist_valid
    # the track moves 1.0 m per 10 Hz row; step 3 reads a row whose text began 0.2 s (2 rows) earlier: d = 80 - 2.0 ; step 5: 0.1 s (1 row): d = 30 - 1.0
    c13 = out[:, 13].copy()
    out[:, 13] = 0
    assert np.array_equal(out, exp), (out - exp)
    np.testing.assert_allclose(c13, [0, 0, 0, 0.78, 0, 0.29], rtol=0, atol=1e-6)


# ------------------------------------------------------------------ the independent recompute (G-CLOCK-RBD a)
def test_columns_equal_independent_recompute(world):
    tab, J, tracks, ex_t = world
    ref = B.reference_columns(tab, "eval", ex_t, J, tracks)
    out = cols_of(world)
    assert out.shape == (len(ref), 15) and out.dtype == np.float32
    assert ref[:, 0].sum() > 20                                  # the fixture has roundabout steps
    assert np.array_equal(out, ref)


def test_c0_equals_navrb2_rule(world):
    tab, J, tracks, ex_t = world
    ref, _ = RB2.state(tab["eval_cls_id"], tab["eval_flag"], tab["vocab"], J["off"], J["n"], 0)
    assert np.array_equal(cols_of(world)[:, 0], ref[:, 0])


def test_c13_equals_clip_nav_rule(world):
    """c13 on a roundabout step equals channel 3 of the pinned ``clip_nav`` (sign -1), with every roundabout class mapped to a side."""
    tab, J, tracks, ex_t = world
    side = np.array([1 if str(v).startswith(("RB_", "ROTARY")) else 0 for v in VOCAB], np.int8)
    o, k = 0, int(J["n"][0])
    nav, _, _, _ = CN.clip_nav(tab["eval_cls_id"][o:o + k], tab["eval_dist_m"][o:o + k], tab["eval_flag"][o:o + k], tab["eval_age_s"][o:o + k], J["now_s"][o:o + k], tracks["clip0"], side, -1)
    out = cols_of(world)[o:o + k]
    has = np.isfinite(tab["eval_dist_m"][o:o + k]) & (out[:, 0] > 0)
    assert has.sum() >= 3
    assert np.array_equal(out[has, 13], nav[has, 3])
    # and the countdown is really subtracting something: strictly below the raw text distance / 100 where the car moved
    raw = np.clip(tab["eval_dist_m"][o:o + k].astype(np.float64) / 100.0, 0, 3)
    assert (out[has, 13] < raw[has] - 1e-6).any()


def test_c13_dead_reckoning_arithmetic():
    """Constant 10 m/s (1 m per 10 Hz row, 2 rows per step): a text 80 m ahead, its row at t = 10, read at t = 10.0 / 10.2 / 10.4 reads 80 - 2 j m."""
    ti = np.array([4, 4, 4], np.int32)
    tab = Tab({"vocab": VOCAB, "eval_clip": np.array(["c"]), "eval_off": np.array([0]), "eval_n": np.array([3]), "eval_cls_id": np.array([4, 4, 4], np.int16), "eval_flag": np.zeros(3, np.uint8),
               "eval_dist_m": np.full(3, 80.0, np.float32), "eval_text_i": ti, "eval_age_s": np.array([0.0, 0.2, 0.4], np.float32)})
    J = {"clip": np.array(["c"]), "off": np.array([0]), "n": np.array([3]), "now_s": 10.0 + 0.2 * np.arange(3)}
    tr = np.stack([np.arange(6) * 1.0, np.zeros(6), np.zeros(6), np.full(6, 10.0)], 1)
    out = B.build_columns(tab, "eval", B.exit_array(TEXTS, parse_full), J, {"c": tr}, B.LIVE_SHIFT)[0]
    np.testing.assert_allclose(out[:, 13], [0.80, 0.78, 0.76], rtol=0, atol=1e-6)


# ------------------------------------------------------------------ admissibility: no future row, backward-only exit-last
def test_future_rows_garbage_leaves_columns_unchanged(world):
    tab, J, tracks, ex_t = world
    base = cols_of(world)
    rng = np.random.RandomState(1)
    for cut in (7, 20, 33, 52):
        t2 = Tab({k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in tab.items()})
        tr2 = {k: v.copy() for k, v in tracks.items()}
        for c in range(3):
            o, k = int(J["off"][c]), int(J["n"][c])
            sl = slice(o + cut + 1, o + k)
            m = k - cut - 1
            t2["eval_cls_id"][sl] = rng.randint(0, len(VOCAB), m)
            t2["eval_flag"][sl] = rng.randint(0, 2, m)
            t2["eval_dist_m"][sl] = rng.rand(m) * 500
            t2["eval_text_i"][sl] = rng.randint(0, len(TEXTS), m)
            t2["eval_age_s"][sl] = rng.rand(m)
            tr2[f"clip{c}"][2 * (cut + 1):] = rng.rand(len(tr2[f"clip{c}"]) - 2 * (cut + 1), 4) * 1000        # the track after the cut step's row 2 cut
        got = B.build_columns(t2, "eval", ex_t, J, tr2, B.LIVE_SHIFT)[0]
        for c in range(3):
            o = int(J["off"][c])
            assert np.array_equal(got[o:o + cut + 1], base[o:o + cut + 1]), (cut, c)


def test_exit_last_is_backward_only():
    ex = np.array([-1, 2, -1, -1, 4, -1])
    flag = np.zeros(6, np.uint8)
    flag[4] = 0
    last = B.last_exit_rows((ex >= 1) & (flag == 0))
    assert last.tolist() == [-1, 1, 1, 1, 4, 4]                  # a row never sees a LATER exit number
    # a flag != 0 row carries no exit number into the future
    flag2 = np.array([0, 1, 0, 0, 0, 0], np.uint8)
    assert B.last_exit_rows((ex >= 1) & (flag2 == 0)).tolist() == [-1, -1, -1, -1, 4, 4]


def test_exit_last_in_table_matches_backward_scan(world):
    tab, J, tracks, ex_t = world
    out = cols_of(world)
    ti, fl = tab["eval_text_i"], tab["eval_flag"]
    exv = np.where((fl == 0) & (ti >= 0), ex_t[np.clip(ti, 0, None)], -1)
    for c in range(3):
        o, k = int(J["off"][c]), int(J["n"][c])
        for j in range(k):
            if out[o + j, 0] == 0:
                continue
            past = [exv[o + q] for q in range(j + 1) if exv[o + q] >= 1]
            want = np.zeros(6, np.float32)
            if past:
                want[min(past[-1], 6) - 1] = 1
            assert np.array_equal(out[o + j, 7:13], want), (c, j)


# ------------------------------------------------------------------ structure
def test_every_column_zero_where_c0_zero(world):
    out = cols_of(world)
    off = out[:, 0] == 0
    assert off.sum() > 20 and (out[:, 0] > 0).sum() > 20
    assert not (out[off] != 0).any()


def test_one_hot_blocks_at_most_one(world):
    out = cols_of(world)
    assert (out[:, 1:7].sum(1) <= 1).all() and (out[:, 7:13].sum(1) <= 1).all()
    assert np.isin(out[:, :13], (0.0, 1.0)).all() and np.isin(out[:, 14], (0.0, 1.0)).all()
    assert (out[:, 13] >= 0).all() and (out[:, 13] <= 3).all()
    p, ok = B.structure_problems(out)
    assert ok, p


def test_exit_bins():
    assert B.exit_bin(np.array([-1, 0, 1, 2, 5, 6, 7, 99])).tolist() == [-1, -1, 0, 1, 4, 5, 5, 5]


# ------------------------------------------------------------------ G-CLOCK-RBD end to end, with the regression arm
def test_gclock_rbd_passes_and_regression_arm_fails(world):
    tab, J, tracks, ex_t = world
    out, rep, ok = B.build(tab, ex_t, {"eval": J}, tracks)
    q, qr = rep["eval"]["G-CLOCK-RBD"], rep["eval"]["G-CLOCK-RBD_regression_arm_shift_5"]
    assert q["pass"] and q["n_differ_from_recompute"] == 0 and q["n_age_negative"] == 0
    assert not qr["pass"] and qr["n_differ_from_recompute"] > 0 and qr["n_age_negative"] > 0
    assert ok and rep["eval"]["G-CLOCK-RBD_pass"]
    assert out["eval_nav_rb"].shape == (180, 15)


def test_age_nonnegative_on_every_row_used(world):
    tab, J, tracks, ex_t = world
    out, src, srcl, _ = B.build_columns(tab, "eval", ex_t, J, tracks, B.LIVE_SHIFT)
    a_now, a_last = B.ages_used(tab, "eval", src, srcl)
    on = out[:, 0] > 0
    assert (a_now[on] >= 0).all() and (a_last[on][np.isfinite(a_last[on])] >= 0).all()


def test_live_read_is_the_step_itself(world):
    tab, J, tracks, ex_t = world
    _, src, _, _ = B.build_columns(tab, "eval", ex_t, J, tracks, B.LIVE_SHIFT)
    assert B.LIVE_SHIFT == 0 and np.array_equal(src, np.arange(len(src)))


def test_misaligned_join_refused(world):
    tab, J, tracks, ex_t = world
    J2 = dict(J)
    J2["n"] = J["n"] - 1
    with pytest.raises(SystemExit):
        B.check_aligned(tab, J2, "eval", "the join")


@pytest.mark.skipif(not os.path.exists("F:/TanitAD-scratch/datascale/navtok/navtok.py"), reason="frozen navtok grammar not on this box")
def test_real_navtok_exit_numbers():
    NT = B.load_navtok("F:/TanitAD-scratch/datascale/navtok")
    ex = B.exit_array(["At roundabout, take exit 3 in 100m", "At roundabout, take exit 2", "Turn left in 50m", "Enter roundabout"], NT.parse_full)
    assert ex.tolist() == [3, 2, -1, -1]
