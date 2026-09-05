"""``obstacle.offline`` -> the batch: the E-AGT-HEAD label wiring.

⛔ WHY THIS FILE EXISTS. The refcv5 build stream shipped `refc_agents.py`, the
DETR head, the set loss and the trainer flags, and then had to escalate that
``E-AGT-HEAD`` **could not train**: ``V3Dataset`` emitted no ``agent_box``, so
``--agents head`` refused. The escalation also said the train-corpus join *did
not exist* -- it did (built 2026-08-17, banked to HF; md5
``24cbdca8c3b23aafc2fb17e6bf99cf76``), which is the "absence found at ONE
location is not absence" class in its most expensive form.

These tests pin the wiring that closes it, and in particular the four
distinctions that are silent when wrong:

1. **NO_LABEL is not LABELLED-CLEAR.** An absent ``(clip, frame)`` line means
   the join says nothing; an EMPTY agent list means the road really was clear.
   Collapsing them trains the presence head to answer "empty" on frames full of
   cars -- the ``"no agents"`` defect the join doc names.
2. **A missing RATE is masked, never zero-filled** -- zero is a legitimate
   value (a stationary car).
3. **A join that covers none of the corpus is REFUSED**, not quietly loaded:
   a run that stamps ``w_agent > 0`` and supervises nothing reads as "the agent
   head does not help".
4. **The LEGACY 16-bit id must be asked for by name.** It is the first 4 BYTES
   of the clip id and COLLIDES, so an episode absent from the join can match a
   different clip that is present -- a label corruption no downstream metric
   could attribute.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refc_v3_train as t                                     # noqa: E402
from train_p8_occupancy import (JoinFileReader,               # noqa: E402
                                episode_uid_of_clip,
                                legacy_episode_id_of_clip)

DT = 0.1007          # the join's own grid spacing, NOT 0.1 (join doc CLOCK)
CLIPS = ("aaaaaaaa-0000-4000-8000-000000000001",
         "bbbbbbbb-0000-4000-8000-000000000002",
         "cccccccc-0000-4000-8000-000000000003")


def _rec(clip, fi, agents):
    return {"clip_id": clip, "frame_idx": fi, "t_s": round(fi * DT, 6),
            "agents": agents}


def _agent(cx, cy=0.0, yaw=0.0, l=4.5, w=1.9, occ=0, tid="t0", cls="automobile"):
    return {"cx": cx, "cy": cy, "yaw": yaw, "l": l, "w": w, "occ": occ,
            "track_id": tid, "cls": cls}


def _write(tmp_path, records, name="join.jsonl"):
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def _simple_join(tmp_path, n_frames=30, clips=CLIPS):
    """One track driving away at +1 m per frame, plus a second agent that
    appears in ONE frame only (so its rate cannot be differenced)."""
    recs = []
    for ci, c in enumerate(clips):
        for f in range(n_frames):
            ag = [_agent(10.0 + f * 1.0, cy=0.5 * ci, tid="t0")]
            if f == 5:
                ag.append(_agent(7.0, cy=-2.0, tid="ONCE", cls="person"))
            recs.append(_rec(c, f, ag))
    return _write(tmp_path, recs)


# ===========================================================================
# the reader
# ===========================================================================
def test_reader_defaults_are_unchanged(tmp_path):
    """Every new argument defaults to the historical behaviour -- the reader
    `train_p8_occupancy` already depends on must not move."""
    r = JoinFileReader(_simple_join(tmp_path))
    assert r.n_records == 90 and r.n_clips == 3
    assert r.has_classes and r.has_occlusion_flags
    assert r.episode_ids is None and r.with_rates is False
    assert r.lookup_rates(episode_uid_of_clip(CLIPS[0]), 3) is None


def test_reader_episode_filter_keeps_only_the_named_clips(tmp_path):
    keep = {episode_uid_of_clip(CLIPS[1])}
    r = JoinFileReader(_simple_join(tmp_path), episode_ids=keep)
    assert r.n_clips == 1 and r.n_records == 30
    assert r.n_records_filtered_out == 60
    assert r.covers_episode(episode_uid_of_clip(CLIPS[1]))
    assert not r.covers_episode(episode_uid_of_clip(CLIPS[0]))


def test_reader_accepts_the_legacy_id_in_the_filter_too(tmp_path):
    keep = {legacy_episode_id_of_clip(CLIPS[2])}
    r = JoinFileReader(_simple_join(tmp_path), episode_ids=keep)
    assert r.n_clips == 1 and r.n_records == 30


def test_reader_REFUSES_a_join_that_covers_none_of_the_corpus(tmp_path):
    """⛔ The wrong join is a REFUSAL, not an empty load. A run that stamps
    `w_agent > 0` while supervising nothing reads as a refutation."""
    with pytest.raises(ValueError, match="WRONG JOIN"):
        JoinFileReader(_simple_join(tmp_path), episode_ids={123456789})


def test_reader_max_agents_per_frame_is_measured(tmp_path):
    r = JoinFileReader(_simple_join(tmp_path))
    assert r.max_agents_per_frame == 2        # the ONCE frame


def test_reader_rates_are_a_finite_difference_over_the_joins_own_t_s(tmp_path):
    """⚠️ The denominator is READ from the records (~0.1007 s), never assumed
    to be 0.1 -- a 0.7 % error in every closing speed otherwise."""
    r = JoinFileReader(_simple_join(tmp_path), with_rates=True)
    eid = episode_uid_of_clip(CLIPS[0])
    rates, mask = r.lookup_rates(eid, 10)     # interior frame -> central diff
    assert mask[0] and rates.shape == (1, 3)
    assert rates[0, 0] == pytest.approx(1.0 / DT, rel=1e-4)
    assert rates[0, 1] == pytest.approx(0.0, abs=1e-6)
    # the clip's FIRST and LAST frames get a ONE-SIDED difference, not None
    for f in (0, 29):
        rr, mm = r.lookup_rates(eid, f)
        assert mm[0] and rr[0, 0] == pytest.approx(1.0 / DT, rel=1e-4)


def test_reader_masks_a_rate_it_could_not_observe(tmp_path):
    """⛔ A missing rate is MASKED, never zero-filled: zero is a legitimate
    value (a stationary car), and filling it would teach the head that unseen
    means still."""
    r = JoinFileReader(_simple_join(tmp_path), with_rates=True)
    rates, mask = r.lookup_rates(episode_uid_of_clip(CLIPS[0]), 5)
    assert mask.shape == (2,)
    assert bool(mask[0]) is True             # the persistent track
    assert bool(mask[1]) is False            # seen in ONE frame only
    assert float(rates[1].sum()) == 0.0      # and its row is zero AND masked


def test_reader_REFUSES_out_of_order_records_when_rates_are_asked_for(
        tmp_path):
    """A silently mis-ordered file yields rates that look entirely plausible
    and are differences between unrelated frames."""
    recs = [_rec(CLIPS[0], 0, [_agent(10.0)]),
            _rec(CLIPS[0], 2, [_agent(12.0)]),
            _rec(CLIPS[0], 1, [_agent(11.0)])]
    with pytest.raises(ValueError, match="does not follow"):
        JoinFileReader(_write(tmp_path, recs, "bad.jsonl"), with_rates=True)


def test_reader_reads_an_xz_join_without_expanding_it(tmp_path):
    import lzma
    src = _simple_join(tmp_path)
    dst = str(tmp_path / "join.jsonl.xz")
    with open(src, "rb") as fi, lzma.open(dst, "wb") as fo:
        fo.write(fi.read())
    r = JoinFileReader(dst)
    assert r.n_records == 90 and r.n_clips == 3


# ===========================================================================
# the dataset seam
# ===========================================================================
def _episodes(n=2, clips=CLIPS, stable=True):
    """`_synth_episodes` with the episode ids the join is keyed on. The synth
    ids are STRINGS (`synth-000`), so they are replaced rather than parsed."""
    cfg = t.v3.refc_v3_smoke_config(True)
    eps = t._synth_episodes(n, cfg.core, seed=0, min_frames=40)
    for e, c in zip(eps, clips):
        e.episode_id = (episode_uid_of_clip(c) if stable
                        else legacy_episode_id_of_clip(c))
    return eps, cfg


def _wired(tmp_path, n=2, stable=True, with_rates=True, **kw):
    eps, cfg = _episodes(n, stable=stable)
    ds = t.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    r = JoinFileReader(_simple_join(tmp_path),
                       episode_ids={int(e.episode_id) for e in eps},
                       with_rates=with_rates)
    stats = ds.enable_agent_join(r, **kw)
    return ds, r, stats


def test_dataset_emits_the_EXACT_keys_the_loss_consumes(tmp_path):
    """The contract is READ OFF `compute_losses_v3`, never invented here: it
    builds `tgt_ag` from these keys and hands it to `agent_losses`."""
    ds, r, stats = _wired(tmp_path)
    item = ds[0]
    n = r.max_agents_per_frame
    assert item["agent_box"].shape == (n, 4)
    assert item["agent_box"].dtype is torch.float32
    assert item["agent_yaw"].shape == (n,)
    assert item["agent_cls"].shape == (n,) and item["agent_cls"].dtype is \
        torch.int64
    assert item["agent_valid"].shape == (n,) and item["agent_valid"].dtype is \
        torch.bool
    assert item["agent_occ"].shape == (n,)
    assert item["agent_rates"].shape == (n, 3)
    assert item["agent_rates_mask"].shape == (n,)
    assert item["agent_label"].dtype is torch.bool
    assert stats["agent_pad"] == n


def test_the_box_columns_are_cx_cy_l_w_in_METRES(tmp_path):
    """`targets_from_join` maps the join's ``[A, 6] = (cx, cy, yaw, l, w, occ)``
    to ``(cx, cy, l, w)`` -- the column order the decode and `slot_set_loss`
    assume. A transposed l/w is invisible in every scalar the loss reports."""
    ds, _, _ = _wired(tmp_path)
    # window 0's NOW frame is t + w - 1 = 3 -> the track sits at 10 + 3 = 13 m
    item = ds[0]
    v = item["agent_valid"]
    assert bool(v[0])
    assert float(item["agent_box"][0, 0]) == pytest.approx(13.0)
    assert float(item["agent_box"][0, 2]) == pytest.approx(4.5)   # l
    assert float(item["agent_box"][0, 3]) == pytest.approx(1.9)   # w


def test_the_NOW_frame_is_t_plus_w_minus_1(tmp_path):
    """The detector and the v7.2 tactical heads must be supervised at ONE
    instant. `__getitem__` reads both at `t + w - 1`, the last OBSERVED frame."""
    ds, _, _ = _wired(tmp_path)
    w = ds.window
    for i in (0, 3, 7):
        e_i, tt = ds.index[i]
        want = 10.0 + (tt + w - 1) * 1.0
        assert float(ds[i]["agent_box"][0, 0]) == pytest.approx(want)


def test_NO_LABEL_is_NOT_labelled_clear(tmp_path):
    """⛔⛔ The distinction the whole wiring turns on. Both states carry
    `valid` all-False; only `agent_label` separates them, and scoring a
    NO_LABEL frame as labelled-clear teaches the presence head that an
    unlabelled frame is an empty road."""
    eps, cfg = _episodes(2)
    ds = t.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    recs = []
    for c in CLIPS[:2]:
        recs.append(_rec(c, 3, []))                    # LABELLED CLEAR
        recs.append(_rec(c, 4, [_agent(11.0)]))
    r = JoinFileReader(_write(tmp_path, recs, "sparse.jsonl"),
                       episode_ids={int(e.episode_id) for e in eps})
    ds.enable_agent_join(r)
    by_now = {}
    for i in range(len(ds)):
        e_i, tt = ds.index[i]
        by_now.setdefault(tt + ds.window - 1, i)
    clear = ds[by_now[3]]
    assert bool(clear["agent_label"]) is True          # a LABEL: road clear
    assert int(clear["agent_valid"].sum()) == 0
    nolab = ds[by_now[10]]
    assert bool(nolab["agent_label"]) is False         # NO_LABEL
    assert int(nolab["agent_valid"].sum()) == 0


def test_enable_REFUSES_a_join_that_covers_no_window(tmp_path):
    eps, cfg = _episodes(2)
    ds = t.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    # labelled ONLY at frame indices no window's NOW can reach
    recs = [_rec(c, 9999, [_agent(11.0)]) for c in CLIPS[:2]]
    r = JoinFileReader(_write(tmp_path, recs, "far.jsonl"),
                       episode_ids={int(e.episode_id) for e in eps})
    with pytest.raises(SystemExit, match="ZERO of"):
        ds.enable_agent_join(r)


def test_enable_REFUSES_a_legacy_only_join_unless_asked_by_name(tmp_path):
    """⛔ The legacy key is the first 4 BYTES of the clip id and COLLIDES, so
    an episode ABSENT from the join can match a different clip that IS
    present. That must be accepted by name and stamped, never by default."""
    with pytest.raises(SystemExit, match="LEGACY 16-bit"):
        _wired(tmp_path, stable=False)
    ds, _, stats = _wired(tmp_path, stable=False, allow_legacy_ids=True)
    assert stats["id_space"] == "legacy-16bit-COLLIDING"
    assert stats["allow_legacy_ids"] is True


def test_stats_report_supervision_coverage_not_just_clip_coverage(tmp_path):
    """Clip coverage is not supervision coverage: the join's label span ends
    ~20 s in, so a window whose NOW frame is past it carries NO label. A run
    that reports only the clip count cannot say whether its detector saw
    100 % or 3 % of its windows -- and those are different experiments."""
    ds, _, stats = _wired(tmp_path)
    assert stats["n_windows"] == len(ds)
    assert 0 < stats["n_windows_labelled"] <= stats["n_windows"]
    assert stats["frac_windows_labelled"] == pytest.approx(
        stats["n_windows_labelled"] / stats["n_windows"], abs=5e-5)
    assert stats["n_target_boxes_prefilter"] > 0
    assert stats["reader_with_rates"] is True


def test_the_pad_is_the_measured_max_so_nothing_is_truncated(tmp_path):
    ds, r, _ = _wired(tmp_path)
    tot = 0
    for i in range(len(ds)):
        tot += int(ds[i]["agent_n_truncated"])
    assert tot == 0
    # and an explicitly TOO-SMALL pad truncates VISIBLY, never silently
    ds.agent_pad = 1
    seen = sum(int(ds[i]["agent_n_truncated"]) for i in range(len(ds)))
    assert seen > 0


# ===========================================================================
# the loss end to end -- and the deliberate-regression control
# ===========================================================================
def test_the_emitted_block_produces_a_REAL_moving_detection_loss(tmp_path):
    """⛔ The failure this closes: an arm trained, converged, wrote a
    checkpoint and stamped `w_agent: 1.0` while the detector was NEVER
    supervised. A wiring that cannot produce a non-zero loss with a non-zero
    matched n is not a wiring."""
    from torch.utils.data import default_collate
    from tanitad.refs import refc_agents as ra
    ds, _, _ = _wired(tmp_path)
    batch = default_collate([ds[i] for i in range(min(16, len(ds)))])
    sel = batch["agent_label"].nonzero(as_tuple=False).flatten()
    assert sel.numel() > 0
    tgt = {k: batch["agent_" + k].index_select(0, sel)
           for k in ("box", "yaw", "cls", "valid", "occ", "rates",
                     "rates_mask")}
    cfg = ra.AgentSeamConfig(enable=True, queries=8, enforce_band=False)
    torch.manual_seed(0)
    head = ra.build_agent_head(cfg, d_memory=32, n_memory=8)
    mem = torch.randn(int(sel.numel()), 8, 32)
    out = ra.agent_losses(head(mem), tgt, cfg)
    assert torch.isfinite(out["total"]) and float(out["total"]) > 0.0
    assert out["n"]["matched"] > 0
    assert out["n"]["rates"] > 0            # the LONGITUDINAL term is LIVE
    # and it MOVES: the loss must fall under optimisation on a FIXED batch.
    # (The live 600-step tiny-rig run reads centre 19.31 -> 3.98 m, presence
    # 0.426 -> 0.141, cls 2.034 -> 0.631 -- this is that in miniature.)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    first_total = float(out["total"])
    first_centre = float(out["loss_centre"])
    for _ in range(150):
        opt.zero_grad()
        parts = ra.agent_losses(head(mem), tgt, cfg)
        parts["total"].backward()
        opt.step()
    assert float(parts["total"]) < first_total
    assert float(parts["loss_centre"]) < first_centre


def test_agent_head_without_a_JOIN_refuses_BEFORE_the_gpu():
    """⛔ DELIBERATE REGRESSION: with the labels withheld the arm must REFUSE,
    not quietly score. It fires in `_pin_refcv5_seams` -- before a GPU day --
    rather than at the first batch."""
    import argparse
    cfg = t.v3.refc_v3_sized_config("tiny", hier=True)
    a = argparse.Namespace(agents="head", w_agent=1.0, agent_join=None,
                           sampler="anchored", w_u0=0.0, agent_queries=32,
                           agent_sigma_range=0.0, agent_miss_rate=0.0,
                           agent_w_project=0.0, agent_w_ground=0.0,
                           agent_presence_hard=False)
    with pytest.raises(SystemExit, match="NO LABELS"):
        t._pin_refcv5_seams(cfg, a)
    # ...and the same arm WITH a join passes the preflight
    a.agent_join = "some/join.jsonl"
    t._pin_refcv5_seams(cfg, a)
    assert cfg.core.agents.enable is True
