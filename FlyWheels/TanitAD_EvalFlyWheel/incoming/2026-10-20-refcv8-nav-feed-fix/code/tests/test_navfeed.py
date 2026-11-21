"""CPU tests for the nav-feed overlay (EvalFlyWheel 2026-10-20).  LITERAL expectations; no GPU, no model build.

    cd <hold>/code/tests
    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='<hold>/code/overlay;C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider test_navfeed.py

What is real here: the launch tree's refcv3_arm.py SOURCE (the patches are applied to it and the patched TEXT is executed), the v9 EVAL release
(md5-checked) read through the trainer's own ``load_v9_join`` / ``R8LabelJoin.item`` path, ``refcv8_train.scale_nav``, and the archived
``step40000ema_unc/dump_s0`` decisions.  What is stubbed: the dataset OBJECT (episodes / window / clock), because building the real one needs
the model; the stub's ``_raw_offset`` is the trainer's formula and its ``_now_s`` returns the release's own clock for the row (so the 1e-6 s
clock refusal passes trivially -- the clock is not what is under test).  A missing input is a FAILURE, never a skip.

(a) clip default reproduces the banked ``decisions.nav_cmd``      (b) perframe_token reproduces the v9 per-frame token where it differs
(c) perframe_full's r8_nav == training's scale_nav               (d) MUTATION: the clip token under the perframe flag turns (b) RED
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import navfeed_overlay as NO

HOLD = Path(__file__).resolve().parents[2]
ARCHIVE = Path("C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/dump_s0")
V9_EVAL = "E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz"
V9_MD5 = "6b5c7f207cffc3b7eb3cd527fd433599"
W = 8

# ---- LITERALS: (episode_index, ws, clip token the battery FED, v9 per-frame token).  Read from the archive + the release at authoring time
# (2026-10-20) and re-asserted against the archive below; the (12, 9) and (13, 8) rows are the OPPOSITE-TURN windows (clip RIGHT, per-frame LEFT).
MISMATCH = [(0, 7, 2, 0), (0, 177, 2, 0), (6, 11, 2, 0), (12, 9, 2, 1), (12, 169, 2, 0), (13, 8, 2, 1), (13, 148, 2, 0),
            (16, 131, 1, 0), (16, 151, 1, 0)]
# (episode_index, ws, token both agree on)
MATCHED = [(0, 62, 2), (6, 31, 2), (16, 11, 1)]
# NAV_TRACE.md sec 3 (MEASURED by the trace agent): confusion[v9 row][fed col] on the 4,754 banked windows
TRACE_CONFUSION = [[2909, 280, 739], [26, 158, 29], [76, 7, 530]]


# =============================================================================================== fixtures
@pytest.fixture(scope="module")
def archive():
    man = json.load(open(ARCHIVE / "manifest.json", encoding="utf-8"))
    eps = {}
    for e in man["episodes"]:
        fi = int(e["file_index"])
        dec = np.load(ARCHIVE / "decisions" / f"ep{fi:03d}.npz", allow_pickle=False)
        eps[int(e["episode_index"])] = {"episode_id": int(e["episode_id"]), "ws": dec["ws"].astype(np.int64),
                                        "nav_cmd": dec["nav_cmd"].astype(np.int64)}
    return eps


@pytest.fixture(scope="module")
def join():
    from tanitad.train import refcv8_train as r8
    return r8.load_v9_join(V9_EVAL, expect_md5=V9_MD5, rc_variant="A50", lat_variant="a")


@pytest.fixture(scope="module")
def trainer_fns():
    from tanitad.train import refcv8_train as r8
    return {"scale_nav": r8.scale_nav, "nav_map": {0: 0, 1: 1, 2: 2}, "NAV_D_CLIP_M": r8.NAV_D_CLIP_M}


def make_stub_ds(join, archive, clip_by_sid=None):
    """A stand-in for the eval V3Dataset exposing exactly what navfeed_overlay.perframe_nav reads."""
    eps_by_idx = {}

    class Ep:
        def __init__(self, sid):
            self.episode_id = sid
            self.frames = np.zeros((1, 9, 1, 1), dtype=np.uint8)       # n_stack 3 -> frames.shape[1]//3 - 1 == 2 (refc_v3_train.py:4290-4294)

    for idx, rec in archive.items():
        eps_by_idx[idx] = Ep(rec["episode_id"])

    class DS:
        window = W
        r8_nav_from_v9 = True
        nav_from_v7 = True

        def __init__(self):
            self.r8_join = join
            self.episodes = eps_by_idx
            self._nav_by_sid = clip_by_sid or {}

        @staticmethod
        def _raw_offset(ep):
            return max(int(ep.frames.shape[1]) // 3 - 1, 0)

        def _now_s(self, ep, t):
            sid = int(ep.episode_id)
            k = int(t) + W - 1 + self._raw_offset(ep)
            return float(join.rel.rows["now_s"][join.rel.clips[sid][0] + k - join.rel.clips[sid][2]])
    return DS()


def make_sel(windows):
    """refcv3_arm's ``sel``: (window_index, episode_index, t) with t = ws - (W - 1)."""
    return [(i, e, int(ws) - (W - 1)) for i, (e, ws, *_rest) in enumerate(windows)]


def clip_nav_from_archive(archive):
    """The clip-level token per sid, as the battery fed it (constant within every episode: NAV_TRACE sec 3)."""
    out = {}
    for rec in archive.values():
        u = np.unique(rec["nav_cmd"])
        assert len(u) == 1
        out[rec["episode_id"]] = int(u[0])
    return out


# =============================================================================================== patcher
def _src():
    return NO.refcv3_arm_path().read_text(encoding="utf-8")


def test_each_anchor_occurs_exactly_once_in_the_launch_tree_source():
    src = _src()
    counts = {p["id"]: src.count(p["anchor"]) for p in NO.PATCHES}
    assert counts == {"P1-argparse": 1, "P2-nav_true": 1, "P3-item-assert": 1, "P4-model-call": 1, "P5-sidecar": 1, "P6-manifest": 1}


def test_patched_source_compiles_and_differs_from_the_launch_tree_file():
    src = _src()
    new = NO.patched_source(src)
    compile(new, "refcv3_arm_patched.py", "exec")
    assert new != src and len(new) > len(src)


def test_the_only_removed_line_of_the_whole_diff_is_the_model_call_tail():
    """The default path can differ from today's ONLY by that one trailing parenthesis becoming ``,`` + a guarded kwarg splat."""
    src = _src()
    diff = NO.unified_diff(src, NO.patched_source(src)).splitlines()
    removed = [ln[1:] for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    assert removed == ["                            **_geom(int(fr_b.shape[0])))"]


def test_anchor_duplicated_or_missing_refuses():
    src = _src()
    bad_dup = src + "\n" + NO.PATCHES[0]["anchor"]
    with pytest.raises(SystemExit):
        NO.patched_source(bad_dup)
    with pytest.raises(SystemExit):
        NO.patched_source(src.replace(NO.PATCHES[1]["anchor"], "# gone\n"))


def test_per_frame_override_precedes_every_consumer_of_nav_true():
    p = NO.patched_source(_src())
    at = p.index('        nav_true, nav_valid, pf_args = _navfeed.perframe_nav(')
    for consumer in ('nav_shuf, shuf_stats = ra.shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)',
                     'nav_flip = nav_true.copy()',
                     'nav_vals = {"nav_true": int(nav_true[i]),',
                     'dec.setdefault("nav_cmd", []).append(int(nav_true[i]))'):
        assert p.index(consumer) > at, consumer


# =============================================================================================== harness: execute the PATCHED text
def run_nav_block(patches, ds, windows, nav_feed, *, with_flip=True):
    """Execute the patched refcv3_arm.py text from ``nav_on = ...`` through the nav-flip block, in a namespace of stubs, and return it.
    The text under test is the launch tree's real source; only the names it reads (a, ds, sel, ra, tr, refb_labels, ...) are stubs."""
    src = NO.patched_source(_src(), patches)
    s = src.index('    nav_on = bool(getattr(ds, "nav_from_v7", False)) or nav_src == "v72"\n')
    sh = '    nav_shuf, shuf_stats = ra.shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)\n'
    e = src.index(sh) + len(sh)
    block = src[s:e]
    if with_flip:
        fs = src.index('    nav_flip = None\n')
        fe_marker = 'windows — the sharpest intervention (optional arm)")\n'
        fe = src.index(fe_marker, fs) + len(fe_marker)
        block += src[fs:fe]
    seen = {}

    def shuffle_nav(nav, valid, seed):                      # recorder: what did the shuffle receive?
        seen["nav"] = np.asarray(nav).copy()
        seen["valid"] = np.asarray(valid).copy()
        return np.asarray(nav).copy(), {"seed": seed, "stub": True}
    from tanitad.train import refcv8_train as r8
    ns = {"np": np, "ds": ds, "sel": make_sel(windows), "nav_src": "v72",
          "a": SimpleNamespace(nav_feed=nav_feed, nav_shuffle_seed=0, with_navflip=with_flip, with_navzero=False),
          "ra": SimpleNamespace(shuffle_nav=shuffle_nav),
          "tr": SimpleNamespace(R8_V9_NAV_TO_NAV_CMD={0: 0, 1: 1, 2: 2}, r8train=SimpleNamespace(scale_nav=r8.scale_nav)),
          "refb_labels": SimpleNamespace(NAV_LEFT=1, NAV_RIGHT=2), "conds_fed": ["nav_true", "nav_shuffled"], "arms": [],
          "_p": lambda *x, **k: None, "navfeed_overlay": NO}
    if nav_feed is None:
        del ns["a"].nav_feed                                # an argv that predates the flag: getattr default must be 'clip'
    exec(textwrap.dedent(block), ns)
    ns["_shuffle_seen"] = seen
    return ns


def literal_ds(join, archive):
    return make_stub_ds(join, archive, clip_by_sid=clip_nav_from_archive(archive))


# =============================================================================================== (a) clip default
def test_a_archive_literals_still_read_what_the_tests_assume(archive):
    for e, ws, fed, _v9 in MISMATCH:
        i = int(np.where(archive[e]["ws"] == ws)[0][0])
        assert int(archive[e]["nav_cmd"][i]) == fed
    for e, ws, tok in MATCHED:
        i = int(np.where(archive[e]["ws"] == ws)[0][0])
        assert int(archive[e]["nav_cmd"][i]) == tok


@pytest.mark.parametrize("flag", ["clip", None])
def test_a_default_clip_reproduces_the_banked_nav_cmd(join, archive, flag):
    """The ORIGINAL nav_true loop of refcv3_arm.py, then the patched block, with the flag at its default ('clip' or absent entirely):
    ``nav_true`` is exactly the clip token the battery banked, ``pf_args`` is None, and the shuffle was handed that same vector."""
    ds = literal_ds(join, archive)
    wins = MISMATCH + MATCHED
    ns = run_nav_block(NO.PATCHES, ds, wins, flag)
    got = [int(x) for x in ns["nav_true"]]
    assert got == [fed for _e, _w, fed, _v in MISMATCH] + [tok for _e, _w, tok in MATCHED]          # literal, from the archive
    assert ns["pf_args"] is None and ns["nav_feed"] == "clip"
    assert bool(ns["nav_valid"].all())
    assert [int(x) for x in ns["_shuffle_seen"]["nav"]] == got                                       # the shuffle sees the clip token
    assert [int(x) for x in ns["nav_flip"]] == [{1: 2, 2: 1}.get(x, x) for x in got]                # and so does the flip


def test_a_default_clip_equals_the_unpatched_block_on_every_banked_window(join, archive):
    """Whole-archive version: patched-with-default == unpatched, element for element, on all 4,754 banked windows."""
    ds = literal_ds(join, archive)
    wins = [(e, int(w), 0) for e in sorted(archive) for w in archive[e]["ws"]]
    assert len(wins) == 4754
    a = run_nav_block(NO.PATCHES, ds, wins, "clip")["nav_true"]
    b = run_nav_block([], ds, wins, "clip")["nav_true"]
    banked = np.concatenate([archive[e]["nav_cmd"] for e in sorted(archive)])
    assert np.array_equal(a, b) and np.array_equal(a, banked)


# =============================================================================================== (b) perframe_token
def check_b(run, join, archive):
    """The (b) assertion, written once: the token the block leaves in ``nav_true`` is the v9 per-frame token on the literal windows."""
    ds = literal_ds(join, archive)
    ns = run(ds, MISMATCH, "perframe_token")
    got = [int(x) for x in ns["nav_true"]]
    want = [v9 for _e, _w, _f, v9 in MISMATCH]
    assert got == want, f"per-frame token {got} != v9 literal {want}"
    assert all(g != fed for g, (_e, _w, fed, _v) in zip(got, MISMATCH)), "token still equals the clip token"
    return ns


def _patches_under_test():
    """NF_MUTATE=flag_dropped | clip_under_flag runs THIS file's (b) against a mutant (the literal 'mutation arm'); unset = the real patches."""
    which = os.environ.get("NF_MUTATE", "")
    if not which:
        return NO.PATCHES
    return {"flag_dropped": mutant_patches, "clip_under_flag": mutant_p2_clip_under_flag}[which]()


def test_b_perframe_token_reproduces_the_v9_token_where_it_differs(join, archive):
    ns = check_b(lambda ds, wins, flag: run_nav_block(_patches_under_test(), ds, wins, flag), join, archive)
    assert ns["nav_feed"] == "perframe_token"
    assert ns["pf_args"]["mode"] == "perframe_token"
    got = [int(x) for x in ns["nav_true"]]
    assert [int(x) for x in ns["nav_clip"]] == [fed for _e, _w, fed, _v in MISMATCH]                # the clip token is kept beside it
    assert [int(x) for x in ns["_shuffle_seen"]["nav"]] == got                                       # shuffle derived from the PER-FRAME token
    assert [int(x) for x in ns["nav_flip"]] == [{1: 2, 2: 1}.get(x, x) for x in got]                # flip derived from the PER-FRAME token
    assert bool(ns["nav_valid"].all())                                                               # training: nav_valid = True (refc_v3_train.py:4668)
    assert {(MISMATCH[3][0], MISMATCH[3][1]): int(ns["nav_true"][3]), (MISMATCH[5][0], MISMATCH[5][1]): int(ns["nav_true"][5])} \
        == {(12, 9): 1, (13, 8): 1}                                                                  # the OPPOSITE-TURN windows: clip RIGHT -> LEFT


def test_b_perframe_token_agrees_on_matched_windows(join, archive):
    ds = literal_ds(join, archive)
    ns = run_nav_block(NO.PATCHES, ds, MATCHED, "perframe_token")
    assert [int(x) for x in ns["nav_true"]] == [tok for _e, _w, tok in MATCHED]
    assert np.array_equal(ns["nav_true"], ns["nav_clip"])


def test_b_whole_archive_reproduces_the_trace_agent_confusion_matrix(join, archive, trainer_fns):
    """4,754 windows through the real join item path == the trace agent's independent npz-array join (NAV_TRACE sec 3): 1,157 differ."""
    ds = literal_ds(join, archive)
    wins = [(e, int(w), 0) for e in sorted(archive) for w in archive[e]["ws"]]
    nav_true, nav_valid, pf = NO.perframe_nav(ds, make_sel(wins), "perframe_full", True, nav_map=trainer_fns["nav_map"],
                                              scale_nav=trainer_fns["scale_nav"])
    fed = np.concatenate([archive[e]["nav_cmd"] for e in sorted(archive)])
    conf = np.zeros((3, 3), dtype=int)
    for v, f in zip(nav_true, fed):
        conf[int(v), int(f)] += 1
    assert conf.tolist() == TRACE_CONFUSION
    assert int((nav_true != fed).sum()) == 1157
    assert int(pf["known"].sum()) == 2802 and len(nav_true) == 4754          # NAV_TRACE sec 2: args valid on 2,802/4,754 (58.9 %)


def test_b_refuses_instead_of_falling_back_to_the_clip_token(join, archive, trainer_fns):
    ds = literal_ds(join, archive)
    sel = make_sel(MISMATCH)
    kw = dict(nav_map=trainer_fns["nav_map"], scale_nav=trainer_fns["scale_nav"])
    ds.r8_nav_from_v9 = False
    with pytest.raises(SystemExit):
        NO.perframe_nav(ds, sel, "perframe_token", True, **kw)
    ds.r8_nav_from_v9 = True
    ds.r8_join = None
    with pytest.raises(SystemExit):
        NO.perframe_nav(ds, sel, "perframe_token", True, **kw)
    ds.r8_join = join
    with pytest.raises(SystemExit):
        NO.perframe_nav(ds, sel, "perframe_token", False, **kw)               # nav source 'none'
    with pytest.raises(SystemExit):
        NO.perframe_nav(ds, sel, "clip", True, **kw)                          # clip is not a per-frame mode


# =============================================================================================== (c) perframe_full r8_nav == scale_nav
def _hand_row(d_next, d_end, dyaw, side, look):
    lg = lambda d: math.log1p(min(max(d, 0.0), 1000.0) / 50.0)               # independent of torch / of the code under test
    return [lg(d_next), lg(d_end), dyaw / 90.0, max(-1.0, min(1.0, side)), lg(look), 1.0]


def test_c_scale_nav_hand_built_example_known_row(trainer_fns):
    raw = torch.tensor([[120.0, 340.0, 85.0, -1.0, 500.0]], dtype=torch.float32)
    got = trainer_fns["scale_nav"](raw, torch.tensor([True]))[0].tolist()
    want = _hand_row(120.0, 340.0, 85.0, -1.0, 500.0)
    assert got == pytest.approx(want, abs=1e-6)
    assert got[5] == 1.0
    assert want[0] == pytest.approx(1.2237754316221157, abs=1e-12)           # log1p(2.4), literal


def test_c_unknown_row_is_exactly_zeros_like_the_training_dropout_path(trainer_fns):
    """refcv8_train.py:560-567 + :572-575: ``known &= ~dropout`` then ``scale_nav(nav, known)``; an unknown row is zeros next to a 0."""
    raw = torch.tensor([[120.0, 340.0, 85.0, -1.0, 500.0], [0.0, 0.0, 0.0, 0.0, 12.0]], dtype=torch.float32)
    got = trainer_fns["scale_nav"](raw, torch.tensor([False, False]))
    assert got.tolist() == [[0.0] * 6, [0.0] * 6]
    assert torch.equal(got, torch.zeros(2, 6))


def test_c_model_kw_builds_the_trainer_row_on_the_nav_true_slot_only(trainer_fns):
    raw = np.array([[120.0, 340.0, 85.0, -1.0, 500.0], [0.0, 0.0, 0.0, 0.0, 12.0]], dtype=np.float32)
    known = np.array([True, False])
    fed_row = trainer_fns["scale_nav"](torch.from_numpy(raw), torch.from_numpy(known)).numpy().astype(np.float32)
    pf = {"mode": "perframe_full", "raw": raw, "known": known, "fed_row": fed_row, "token": np.array([1, 0])}
    conds = ["nav_true", "nav_shuffled", "nav_flipped"]
    kw = NO.model_kw("perframe_full", pf, 0, conds, "cpu")
    assert set(kw) == {"r8_nav"} and tuple(kw["r8_nav"].shape) == (3, 6)
    assert kw["r8_nav"][0].tolist() == pytest.approx(_hand_row(120.0, 340.0, 85.0, -1.0, 500.0), abs=1e-6)
    assert kw["r8_nav"][1].tolist() == [0.0] * 6 and kw["r8_nav"][2].tolist() == [0.0] * 6      # perturbed rows: the UNKNOWN row
    kw1 = NO.model_kw("perframe_full", pf, 1, ["nav_true"], "cpu")
    assert kw1["r8_nav"].tolist() == [[0.0] * 6]                                                  # args not valid for this window
    assert NO.model_kw("perframe_token", pf, 0, conds, "cpu") == {}
    assert NO.model_kw("clip", pf, 0, conds, "cpu") == {}


def test_c_fed_row_of_every_real_window_equals_the_trainer_call_on_the_collated_item(join, archive, trainer_fns):
    """Emulate the trainer's batch: collate the real items' r8_nav_raw / r8_nav_known and call scale_nav on the whole batch."""
    ds = literal_ds(join, archive)
    wins = MISMATCH + MATCHED
    nav_true, _v, pf = NO.perframe_nav(ds, make_sel(wins), "perframe_full", True, nav_map=trainer_fns["nav_map"],
                                       scale_nav=trainer_fns["scale_nav"])
    raws, knowns = [], []
    for (e, ws, *_r) in wins:
        ep = ds.episodes[e]
        j8 = join.item(int(ep.episode_id), int(ws) + 2, ds._now_s(ep, ws - (W - 1)))
        raws.append(j8["r8_nav_raw"])
        knowns.append(j8["r8_nav_known"])
    batch_raw, batch_known = torch.stack(raws), torch.stack(knowns)
    want = trainer_fns["scale_nav"](batch_raw.to(torch.float32), batch_known.to(torch.bool).reshape(len(wins)))
    assert torch.equal(torch.from_numpy(pf["fed_row"]), want)
    assert pf["fed_row"].shape == (len(wins), 6) and np.isfinite(pf["fed_row"]).all()
    assert np.all(pf["fed_row"][~pf["known"]] == 0.0)


def test_c_assert_item_matches_accepts_the_real_item_and_rejects_a_drifted_one(join, archive, trainer_fns):
    ds = literal_ds(join, archive)
    nav_true, _v, pf = NO.perframe_nav(ds, make_sel(MISMATCH), "perframe_full", True, nav_map=trainer_fns["nav_map"],
                                       scale_nav=trainer_fns["scale_nav"])
    e, ws = MISMATCH[3][0], MISMATCH[3][1]
    ep = ds.episodes[e]
    j8 = join.item(int(ep.episode_id), int(ws) + 2, ds._now_s(ep, ws - (W - 1)))
    item = {"nav_cmd": torch.tensor(int(nav_true[3])), "r8_nav_raw": j8["r8_nav_raw"], "r8_nav_known": j8["r8_nav_known"]}
    NO.assert_item_matches(item, 3, nav_true, pf)                                                   # no raise
    for bad in ({**item, "nav_cmd": torch.tensor(2)},                                              # the clip token sneaks in
                {**item, "r8_nav_raw": j8["r8_nav_raw"] + 1.0},
                {**item, "r8_nav_known": ~j8["r8_nav_known"]}):
        with pytest.raises(SystemExit):
            NO.assert_item_matches(bad, 3, nav_true, pf)


# =============================================================================================== sidecar + manifest snippets (P5, P6)
def test_p5_p6_default_adds_nothing_and_perframe_adds_the_keys(join, archive, trainer_fns):
    ds = literal_ds(join, archive)
    ns = run_nav_block(NO.PATCHES, ds, MISMATCH, "perframe_full")
    # P5 executed for window 3
    p5 = [p for p in NO.PATCHES if p["id"] == "P5-sidecar"][0]["replacement"]
    dec = {}
    env = {"dec": dec, "i": 3, "nav_true": ns["nav_true"], "nav_clip": ns["nav_clip"], "pf_args": ns["pf_args"], "nav_feed": "perframe_full", "np": np}
    exec(textwrap.dedent(p5), env)
    assert dec["nav_cmd"] == [1] and dec["nav_cmd_clip"] == [2] and dec["r8_nav_known"] == [bool(ns["pf_args"]["known"][3])]
    assert dec["r8_nav_fed"][0].shape == (1, 6)
    dec2 = {}
    exec(textwrap.dedent(p5), {**env, "dec": dec2, "pf_args": None, "nav_feed": "clip"})
    assert dec2 == {"nav_cmd": [1]}                                                                  # default: only the original key
    # P6: the manifest expression
    p6 = [p for p in NO.PATCHES if p["id"] == "P6-manifest"][0]["replacement"].strip().rstrip(",")
    man = eval("{" + p6 + "}", {"shuf_stats": {}, "pf_args": ns["pf_args"], "_navfeed": NO, "nav_feed": "perframe_full",
                                "nav_clip": ns["nav_clip"], "nav_true": ns["nav_true"]})
    assert man["nav_feed"]["mode"] == "perframe_full" and man["nav_feed"]["n_token_differs_from_clip"] == 9
    man0 = eval("{" + p6 + "}", {"shuf_stats": {}, "pf_args": None, "_navfeed": NO})
    assert man0 == {"nav_shuffle": {}}


def test_p4_model_call_default_is_the_original_call(join, archive):
    p4 = [p for p in NO.PATCHES if p["id"] == "P4-model-call"][0]
    calls = []

    def model(*a, **k):
        calls.append(k)
        return "out"
    geom = lambda n: {}
    for pf_args, want_keys in ((None, set()), ({"mode": "perframe_full", "fed_row": np.ones((1, 6), np.float32)}, {"r8_nav"})):
        env = {"model": model, "fr_b": torch.zeros(1, 1), "nav_t": 1, "nav_on": True, "_v0_fed": None, "steps": 2, "ego_b": None,
               "_geom": geom, "pf_args": pf_args, "_navfeed": NO, "nav_feed": "perframe_full", "i": 0, "conds_fed": ["nav_true"], "dev": "cpu"}
        exec(textwrap.dedent(p4["replacement"]), env)
        assert set(calls[-1]) - {"nav_cmd", "v0", "steps", "ego_state"} == want_keys


# =============================================================================================== (d) MUTATION: clip token under the perframe flag
def mutant_patches():
    """The flag is accepted but the per-frame override is DROPPED: nav_true stays the clip token (== feeding the clip token under the flag)."""
    return [p for p in NO.PATCHES if p["id"] != "P2-nav_true"]


def mutant_p2_clip_under_flag():
    """P2 present, but every per-frame token is replaced by the window's clip token before it is returned."""
    out = []
    for p in NO.PATCHES:
        if p["id"] == "P2-nav_true":
            p = dict(p)
            p["replacement"] = p["replacement"].replace(
                "    nav_shuf, shuf_stats = ra.shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)\n",
                "        nav_true = nav_clip.copy()\n    nav_shuf, shuf_stats = ra.shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)\n")
        out.append(p)
    return out


@pytest.mark.parametrize("which", ["flag_dropped", "clip_under_flag"])
def test_d_mutation_arm_turns_b_red(join, archive, which):
    patches = mutant_patches() if which == "flag_dropped" else mutant_p2_clip_under_flag()
    with pytest.raises(AssertionError):                                  # (b)'s assertion function goes RED on the mutant
        check_b(lambda ds, wins, flag: run_nav_block(patches, ds, wins, flag), join, archive)
    # and the unmutated patches are GREEN under the very same function (the control that makes RED mean something)
    check_b(lambda ds, wins, flag: run_nav_block(NO.PATCHES, ds, wins, flag), join, archive)


def test_d_mutation_at_the_join_level_also_turns_b_red(join, archive):
    """A join that returns the clip token for r8_nav_token (the clip token fed through the per-frame path) fails the same assertion."""
    clip = clip_nav_from_archive(archive)

    class ClipJoin:
        def __init__(self, real):
            self.real, self.rel = real, real.rel

        def item(self, sid, k, now_s, **kw):
            out = dict(self.real.item(sid, k, now_s, **kw))
            out["r8_nav_token"] = torch.tensor(clip[int(sid)], dtype=torch.long)
            return out

    def run(ds, wins, flag):
        ds.r8_join = ClipJoin(join)
        return run_nav_block(NO.PATCHES, ds, wins, flag)
    with pytest.raises(AssertionError):
        check_b(run, join, archive)


# =============================================================================================== window selection (PREREG sec 3)
def test_window_selection_matches_trace_literals_and_the_clip_id_rule():
    p = HOLD / "raw" / "windows_navfeed.json"
    txt = p.read_text(encoding="utf-8")
    d = json.loads(txt)
    c = d["counts"]
    assert c["n_mismatch"] == 1157 and c["n_mismatch_episodes"] == 59 and c["n_control"] == 300
    assert c["control_by_token"] == {"FOLLOW": 100, "LEFT": 100, "RIGHT": 100}
    assert c["confusion_v9_row_fed_col"] == TRACE_CONFUSION and c["mismatch_control_overlap"] == 0
    assert all(d["matches_nav_trace_literals"].values())
    assert d["control_seed"] == 20261020
    # the clip-id rule: no uuid-shaped string, no 8+-hex run followed by '-', and the only 12-hex strings are the sha12 fields
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{3,}", txt)
    sha = {e["sha12"] for e in d["episodes"]}
    for m in re.findall(r"\b[0-9a-f]{12}\b", txt):
        assert m in sha or m in {d["inputs"]["v9_eval_md5"][:12]}
    # every window is listed once, mismatch and control disjoint
    mism = {(e["episode_index"], w) for e in d["episodes"] for w in e["mismatch_ws"]}
    ctrl = {(e["episode_index"], w) for e in d["episodes"] for w in e["control_ws"]}
    assert len(mism) == 1157 and len(ctrl) == 300 and not (mism & ctrl)
    for e in d["episodes"]:
        assert len(e["mismatch_ws"]) == len(e["mismatch_fed"]) == len(e["mismatch_v9"])
        assert all(f != v for f, v in zip(e["mismatch_fed"], e["mismatch_v9"]))
