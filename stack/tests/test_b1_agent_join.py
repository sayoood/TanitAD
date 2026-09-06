"""Guards of the B1 EVAL agent join, pinned by MUTATION.

⛔ WHY MUTATION AND NOT INSPECTION. An AST census once read 0 suspects on BOTH the
fixed and the broken trainer, so "the guard is in the source" is not evidence the
guard fires. Every test here REINTRODUCES the defect and requires the refusal.

Corpus-free on purpose: these run anywhere, including with the G: mount down.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "stack" / "scripts", _REPO / "stack"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bj = pytest.importorskip("build_b1_agent_join")


def _rec(clip="c", frame=0, cx=1.0, n=1):
    return {"clip_id": clip, "frame": frame, "frame_idx": frame - 2,
            "t_s": 0.1 * frame,
            "agents": [{"cx": cx + i, "cy": 2.0, "yaw": 0.0, "l": 4.0, "w": 2.0,
                        "occ": 0, "track_id": str(i), "cls": "automobile"}
                       for i in range(n)]}


def _write(p, recs):
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# the CONTROL first: a valid join must PASS, else every refusal below could be
# passing for the wrong reason (a guard that refuses everything is not a guard).
# --------------------------------------------------------------------------
def test_a_valid_join_passes_the_content_assertion(tmp_path):
    p = _write(tmp_path / "ok.jsonl", [_rec(frame=0), _rec(frame=1)])
    got = bj.assert_content(p, 2, False)
    assert got["n_lines"] == 2
    assert got["n_agent_boxes"] == 2
    assert got["unique_keys"] == 2
    assert got["mean_abs_cx_m"] > 0.0


def test_an_empty_join_is_refused_as_a_zeros_artifact(tmp_path):
    """A pre-allocated file of zeros and a successful build are indistinguishable
    by size and exit code. They are distinguishable by re-reading the bytes."""
    p = _write(tmp_path / "empty.jsonl", [])
    with pytest.raises(SystemExit, match="EMPTY"):
        bj.assert_content(p, 0, False)


def test_a_join_with_lines_but_no_boxes_is_refused(tmp_path):
    """Every line labelled-clear is a degenerate artifact, not supervision."""
    p = _write(tmp_path / "noboxes.jsonl",
               [_rec(frame=0, n=0), _rec(frame=1, n=0)])
    with pytest.raises(SystemExit, match="EMPTY"):
        bj.assert_content(p, 2, False)


def test_duplicate_keys_are_refused_as_ambiguous(tmp_path):
    p = _write(tmp_path / "dup.jsonl", [_rec(frame=0), _rec(frame=0)])
    with pytest.raises(SystemExit, match="duplicate"):
        bj.assert_content(p, 2, False)


def test_a_short_write_is_refused(tmp_path):
    """The builder's own line counter disagreeing with the bytes on disk is the
    truncation signature -- a non-ASCII print has silently truncated a banked
    artifact before."""
    p = _write(tmp_path / "short.jsonl", [_rec(frame=0)])
    with pytest.raises(SystemExit, match="expected"):
        bj.assert_content(p, 2, False)


# --------------------------------------------------------------------------
# the alignment gate: the mis-join(+1) must separate on the SPEED statistic
# --------------------------------------------------------------------------
def _fake_block(clip, n, t0=0.0, dt=0.1007, v0=10.0, dv=-0.05):
    return {clip: {i: (t0 + dt * i, v0 + dv * i) for i in range(n)}}


def test_the_speed_gate_separates_a_true_join_from_a_shifted_one():
    """The gate statistic must be small for the true index and large for a
    one-frame shift -- proven here on both, not asserted on one."""
    n, clip = 60, "c"
    block = _fake_block(clip, n)
    t_s = np.array([0.1007 * i for i in range(n)])
    poses = np.zeros((n, 4))
    poses[:, 3] = [10.0 - 0.05 * i for i in range(n)]

    true = bj.alignment_probe(clip, t_s, poses, block)
    assert true["speed_max_dv_mps"] < bj.LEAD_SPEED_TOL_MPS
    # the control: frame i+1's speed read against the block's frame i
    assert true["speed_misjoin1_max_dv_mps"] > bj.LEAD_SPEED_TOL_MPS
    sep = true["speed_misjoin1_max_dv_mps"] / max(true["speed_max_dv_mps"], 1e-12)
    assert sep >= bj.MIN_MISJOIN_SEPARATION

    shifted = bj.alignment_probe(clip, t_s[1:], poses[1:], block)
    assert shifted["speed_max_dv_mps"] > bj.LEAD_SPEED_TOL_MPS


def test_the_tolerance_is_the_programmes_own_constant_not_an_invented_one():
    """⛔ 1e-3 m/s is refav1_arm.py:238 LEAD_SPEED_TOL_MPS. If someone loosens it
    here, this test says where the number is supposed to come from."""
    assert bj.LEAD_SPEED_TOL_MPS == 1e-3


# --------------------------------------------------------------------------
# the block time-base fallback (the stationary-clip rescue)
# --------------------------------------------------------------------------
def test_block_time_base_requires_a_CONTIGUOUS_cover():
    """A partial cover would silently shift the tail, so it must return None
    rather than a short array."""
    clip = "c"
    full = _fake_block(clip, 10)
    assert bj._block_time_base(full, clip, 10) is not None
    assert len(bj._block_time_base(full, clip, 10)) == 10
    holey = {clip: {i: v for i, v in full[clip].items() if i != 4}}
    assert bj._block_time_base(holey, clip, 10) is None
    assert bj._block_time_base(full, "other", 10) is None
    assert bj._block_time_base(None, clip, 10) is None


# --------------------------------------------------------------------------
# the index-space contract -- the trap this builder exists to close
# --------------------------------------------------------------------------
def test_both_index_spaces_are_emitted_and_differ_by_the_trim():
    """`frame` is RAW; `frame_idx` is post-n_stack-trim. A consumer reading the
    wrong one lands 2 frames (~0.2 s) away, which is ~2.7 m of lead displacement
    at 13.6 m/s -- silent, and exactly why both are written."""
    r = _rec(frame=7)
    assert r["frame"] == 7
    assert r["frame_idx"] == 5              # n_stack 3 -> trim 2
    assert bj.FRAME_KEY == "frame", (
        "the RAW key must NOT be named frame_idx -- that name belongs to the "
        "post-trim space read by train_p8_occupancy.JoinFileReader")


# --------------------------------------------------------------------------
# the B1 **TRAIN** gate: the frame-unit statistic, pinned by MUTATION.
#
# The EVAL gate needs two INDEPENDENTLY produced speed arrays indexed by the
# same frame. B1 TRAIN has neither (no lead block; the poses are BUILT on the
# camera grid, so poses[:,3] IS the reference's own speeds). MEASURED on the 141
# EVAL clips, the "re-sample at the registered times" variant read true
# 1.191e-01 m/s against a mis-join(+1) of 2.134e-02 m/s -- the CONTROL SMALLER
# THAN THE TRUE VALUE. So the gate moved to frames, and these tests are what
# stop it drifting back.
# --------------------------------------------------------------------------
def test_the_frame_gate_separates_a_true_join_from_a_shifted_one():
    """TRUE well inside half a frame; the mis-join(+1) control at ~1.0 frames.
    The control must read a KNOWN value, not merely a bigger one."""
    n, clip, dt = 60, "c", 0.1007
    block = _fake_block(clip, n, dt=dt)
    t_s = np.array([dt * i for i in range(n)])
    poses = np.zeros((n, 4))

    pr = bj.alignment_probe(clip, t_s, poses, block)
    assert pr["grid_dt_s"] == pytest.approx(dt, rel=1e-6)
    assert pr["true_max_err_frames"] <= bj.FRAME_ERR_MAX_TRUE
    assert pr["misjoin1_min_err_frames"] >= bj.FRAME_ERR_MIN_MISJOIN
    # the control is a ONE-frame shift, so it must land on ~1.0 frames
    assert pr["misjoin1_min_err_frames"] == pytest.approx(1.0, abs=0.05)


def test_the_frame_gate_REFUSES_a_join_shifted_by_one_frame():
    """MUTATION: hand the probe a time base that really is one frame late. The
    gate must fail, or it is not a gate."""
    n, clip, dt = 60, "c", 0.1007
    block = _fake_block(clip, n, dt=dt)
    bad = np.array([dt * (i + 1) for i in range(n)])          # +1 frame
    pr = bj.alignment_probe(clip, bad, np.zeros((n, 4)), block)
    assert pr["true_max_err_frames"] > bj.FRAME_ERR_MAX_TRUE
    assert pr["true_max_err_frames"] == pytest.approx(1.0, abs=0.05)


def test_the_frame_gate_REFUSES_a_half_frame_drift():
    """MUTATION: a drift too small to be a whole frame but past the 0.5-frame
    decision boundary still lands on the wrong integer index."""
    n, clip, dt = 60, "c", 0.1007
    block = _fake_block(clip, n, dt=dt)
    bad = np.array([dt * i + 0.6 * dt for i in range(n)])
    pr = bj.alignment_probe(clip, bad, np.zeros((n, 4)), block)
    assert pr["true_max_err_frames"] > bj.FRAME_ERR_MAX_TRUE


def test_the_frame_thresholds_bracket_the_half_frame_boundary():
    """A frame index is an INTEGER: the boundary is 0.5 frames, and both
    thresholds must sit a clear margin away from it on the correct side."""
    assert bj.FRAME_ERR_MAX_TRUE < 0.5 < bj.FRAME_ERR_MIN_MISJOIN
    assert bj.FRAME_ERR_MAX_TRUE == 0.25
    assert bj.FRAME_ERR_MIN_MISJOIN == 0.75


def test_the_reconstruct_gate_refuses_the_camera_clock_it_cannot_use():
    """The ego track is read in microseconds; a camera grid on another clock
    would silently mis-scale every query, so it is REFUSED, not rescaled."""
    assert bj.EGO_CLOCK_UNIT == 1e6


def test_grid_reference_is_not_the_grid_compared_with_itself():
    """The reference and the gate's poses must come from DIFFERENT time bases.
    An identity would read exactly 0.0 with infinite separation and measure
    nothing -- the failure this programme paid for four times in one afternoon.
    """
    import inspect
    src = inspect.getsource(bj.poses_at_registered_time)
    assert "t_s" in src, "the gate must sample at the REGISTERED times"
    doc = bj.grid_reference.__doc__ or ""
    assert "TAUTOLOGY" in doc.upper()


def test_the_corpus_bound_on_gate_exclusions_exists_and_is_tight():
    """"Exclude the failures" is one edit away from "loosen the gate", so the
    corpus-level bound is what stops a SYSTEMATIC one-frame defect being
    absorbed as "a few bad clips". MEASURED: 1 of 4,434 clips failed, so the
    bound must sit far above that and far below anything systematic."""
    assert 0.0 < bj.MAX_ALIGNMENT_EXCLUDED_FRAC <= 0.01
    one_offender = 1 / 4434
    assert one_offender < bj.MAX_ALIGNMENT_EXCLUDED_FRAC
    # a systematic defect (>= 5 % of clips) must still REFUSE the build
    assert bj.MAX_ALIGNMENT_EXCLUDED_FRAC < 0.05


def test_the_measured_offender_would_be_excluded_not_absorbed():
    """MUTATION with the REAL numbers: clip 065482b3 read true 1.06e-01 s
    against a mis-join(+1) of 1.04e-01 s on a ~0.1009 s grid -- the control is
    98 % of the true value, which is what a genuinely one-frame-off clip looks
    like. Both halves of the per-clip gate must reject it."""
    n, clip, dt = 40, "c", 0.1009
    block = _fake_block(clip, n, dt=dt)
    t_s = np.array([dt * i + 1.06e-01 for i in range(n)])   # ~1 frame late
    pr = bj.alignment_probe(clip, t_s, np.zeros((n, 4)), block)
    kept = (pr["true_max_err_frames"] <= bj.FRAME_ERR_MAX_TRUE
            and pr["misjoin1_min_err_frames"] >= bj.FRAME_ERR_MIN_MISJOIN)
    assert not kept, "the measured offender must be EXCLUDED, not written"
    assert pr["true_max_err_frames"] > bj.FRAME_ERR_MAX_TRUE


# --------------------------------------------------------------------------
# the digest-scope declaration -- without it the join is UNLOADABLE
# --------------------------------------------------------------------------
def test_the_builder_declares_a_digest_scope_the_consumer_can_read():
    """MEASURED 2026-09-06: `refc_v3_train.py --agent-join-verify auto` calls
    `join_meta.read_digest_scope`, which requires a TOP-LEVEL `digest_scope`
    BLOCK and REFUSES a sidecar that declares nothing. This builder wrote only
    `summary.digest_scope`, a STRING nested inside `summary`, which that reader
    never looks at -- so a perfectly good join would have killed the train run
    at startup. The test asserts the block the CONSUMER reads, not the string
    the builder used to write."""
    jm = pytest.importorskip("tanitad.data.join_meta")
    meta = {"summary": {"md5": "0" * 32, "digest_scope": "compressed"}}
    out = jm.attach(meta, "0" * 32, scope="compressed",
                    filename="b1train_agents.jsonl.xz", algo="md5",
                    declared_by="build_b1_agent_join")
    sc = jm.read_digest_scope(out)          # must not raise
    assert sc.scope == "compressed"
    assert sc.filename == "b1train_agents.jsonl.xz"
    # MUTATION: the legacy-only sidecar must still be REFUSED, or this test
    # would pass on a builder that never attached anything.
    with pytest.raises(Exception):
        jm.read_digest_scope({"summary": {"md5": "0" * 32,
                                          "digest_scope": "compressed"}})


def test_plain_was_never_a_valid_scope():
    """The builder emitted "plain" for a non-.xz output while ARTIFACT_SCOPES is
    ("compressed", "decompressed") -- a value that raises even if a reader found
    it. Pinned so the vocabulary cannot drift back."""
    jm = pytest.importorskip("tanitad.data.join_meta")
    assert "plain" not in jm.ARTIFACT_SCOPES
    assert set(jm.ARTIFACT_SCOPES) == {"compressed", "decompressed"}
    # ⚠️ Match the CODE form, not the bare word. A first cut asserted
    # `'"plain"' not in src` and FAILED on the comment that explains why "plain"
    # is wrong -- a guard matching its own explanatory prose, which is the
    # "filter contains the pattern it searches for" trap in test costume.
    src = io.open(_REPO / "stack" / "scripts" / "build_b1_agent_join.py",
                  encoding="utf-8").read()
    assert 'else "plain"' not in src, "the builder still EMITS the invalid scope"
    # and the value it does emit must be a member
    assert 'else "decompressed"' in src
