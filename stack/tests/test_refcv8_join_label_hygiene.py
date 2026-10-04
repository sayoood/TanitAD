"""refcv8 WP-C fixes 3 + 4: the EGO-as-agent box mask and the track-id-switch RATE mask (OPT-IN, frame lists as data).

Source: D3 ``TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D3_raw_plausibility`` (DEFECT D-2;
MINOR "Boxes" row: 0.028 % track jumps, 91 % single-track id switches, up to 276 m).

Every expectation below is a LITERAL (a hand-computed rate, a hand-listed row count) -- never an expression over the code
under test. Each fix carries a MUTATION that must go RED:

* ego mask: a mask that ignores the frame list (strips the footprint everywhere) fails the "unlisted frame keeps its row" arm;
* rate mask: a mask that covers only record f (not f-1) leaves a 255 m/s target alive -- the arm that proves the TWO-record
  rule is load-bearing;
* default: ``defect_masks=None`` reads the SAME bytes as the unmodified tip reader (golden digest computed from the TIP's
  ``train_p8_occupancy.py`` before the change).

The mined lists themselves (``refcv8_join_label_defects.json``) are pinned against D3's banked aggregates.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data import join_label_hygiene as H           # noqa: E402
from train_p8_occupancy import JoinFileReader               # noqa: E402

CFG = Path(H.__file__).resolve().parent.parent / "configs" / "refcv8_join_label_defects.json"
DT = 0.1
CLIP = "synthetic-clip-0001"
CLIP_B = "synthetic-clip-0002"


def _agent(cx, cy=0.0, tid="t0", cls="automobile", l=4.5, w=1.9):
    return {"cx": cx, "cy": cy, "yaw": 0.0, "l": l, "w": w, "occ": 0, "track_id": tid, "cls": cls}


def _records():
    """One clip, frames 0..5. t_car: +1 m / frame (a relative speed of exactly 10 m/s at DT 0.1). t_jump: the same, but
    it jumps +49 m on top of its +1 m between frame 2 and 3 (an id switch). t_ego: the phantom ego box at (1.4, 0) in frames 2, 3 (LISTED) and 4
    (NOT listed). t_far: a legitimate box at (30, 3) in the listed frames."""
    out = []
    for f in range(6):
        ag = [_agent(20.0 + f, tid="t_car")]
        ag.append(_agent(30.0 + f + (49.0 if f >= 3 else 0.0), 2.0, tid="t_jump"))
        if f in (2, 3, 4):
            ag.append(_agent(1.4, 0.0, tid="t_ego", l=4.6, w=2.0))
        if f in (2, 3):
            ag.append(_agent(30.0, 3.0, tid="t_far"))
        out.append({"clip_id": CLIP, "frame_idx": f, "t_s": round(f * DT, 6), "agents": ag})
    out.append({"clip_id": CLIP_B, "frame_idx": 2, "t_s": 0.2, "agents": [_agent(1.4, 0.0, tid="t_ego")]})   # other clip
    return out


def _write(tmp_path) -> Path:
    p = tmp_path / "join.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _records()) + "\n", encoding="utf-8")
    return p


def _masks(rate_records=None) -> H.JoinDefectMasks:
    m = H.JoinDefectMasks({H.clip_key(CLIP): {2, 3}}, {H.clip_key(CLIP): {(3, H.track_key("t_jump"))}})
    if rate_records is not None:                      # the MUTATION hook: replace the contaminated-record table
        m._rate_records = rate_records
    return m


def _reader(path, masks=None):
    return JoinFileReader(path, with_rates=True, with_track_ids=True, defect_masks=masks)


EID_OF = {}


def _eid(rd, clip=CLIP):
    from train_p8_occupancy import episode_uid_of_clip
    return episode_uid_of_clip(clip)


def _digest(rd, clip=CLIP) -> str:
    """sha256 over every array a consumer can read for frames 0..5 (rows, classes, track ids, rates, rate masks)."""
    h = hashlib.sha256()
    e = _eid(rd, clip)
    for f in range(6):
        ag = rd.lookup(e, f)
        h.update(repr((f, None if ag is None else ag.shape)).encode())
        if ag is None:
            continue
        h.update(np.ascontiguousarray(ag).tobytes())
        h.update(repr(list(rd.lookup_classes(e, f))).encode())
        h.update(repr(list(rd.lookup_track_ids(e, f))).encode())
        rt, rm = rd.lookup_rates(e, f)
        h.update(np.ascontiguousarray(rt).tobytes() + np.ascontiguousarray(rm).tobytes())
    return h.hexdigest()


#: computed from the TIP's UNMODIFIED train_p8_occupancy.JoinFileReader (blob 6faaf509) on the same synthetic file, BEFORE
#: the change: the default read must reproduce it bit for bit.
GOLDEN_DEFAULT_DIGEST = "cee2f6cc611ed61531982461bb200866ffc9e5d13245f44e965489b00f079a94"


# =========================================================================== #
# 0. the rule predicates                                                       #
# =========================================================================== #
def test_ego_footprint_predicate_is_the_d3_rule_with_strict_bounds():
    f = H.is_ego_footprint
    assert f(1.4, 0.0) is True
    assert f(0.69, -0.06) and f(3.99, 0.99) and f(-0.99, -0.99)
    assert not f(4.0, 0.0) and not f(-1.0, 0.0) and not f(1.4, 1.0) and not f(1.4, -1.0)   # open interval / strict
    assert not f(30.0, 3.0) and not f(5.0, 0.0)
    assert list(H.is_ego_footprint(np.array([1.4, 5.0, 3.0]), np.array([0.0, 0.0, 2.0]))) == [True, False, False]


def test_class_kind_and_jump_thresholds_are_d3s_literals():
    assert H.jump_threshold_m("automobile") == 5.0 and H.jump_threshold_m("bus") == 5.0
    assert H.jump_threshold_m("person") == 2.0 and H.jump_threshold_m("stroller") == 2.0
    assert H.jump_threshold_m("rider") == 3.0 and H.jump_threshold_m("animal") == 3.0
    assert H.POSE_GLITCH_FRAC == 0.5


def test_sha12_keys_are_sha256_prefixes_and_never_the_raw_id():
    assert H.clip_key("abc") == "ba7816bf8f01"                 # sha256("abc") = ba7816bf 8f01cfea ...
    assert H.track_key("abc") == "ba7816bf8f01"
    assert len(H.clip_key(CLIP)) == 12 and CLIP not in H.clip_key(CLIP)


# =========================================================================== #
# 1. DEFAULT UNCHANGED                                                         #
# =========================================================================== #
def test_default_reader_is_bit_identical_to_the_unmodified_tip_reader(tmp_path):
    p = _write(tmp_path)
    assert _digest(_reader(p, None)) == GOLDEN_DEFAULT_DIGEST
    # and the default carries no mask state
    assert _reader(p, None).defect_stats() is None


# =========================================================================== #
# 2. fix 3 -- the ego box                                                      #
# =========================================================================== #
def test_ego_rows_are_removed_in_listed_frames_only_and_every_aligned_array_loses_the_same_row(tmp_path):
    rd = _reader(_write(tmp_path), _masks())
    e = _eid(rd)
    # frames 2 and 3 are LISTED: t_ego is gone, t_far (a legitimate box in the same frame) stays
    for f, want in ((2, ["t_car", "t_jump", "t_far"]), (3, ["t_car", "t_jump", "t_far"])):
        assert [str(t) for t in rd.lookup_track_ids(e, f)] == want
        assert rd.lookup(e, f).shape == (3, 6)
        assert len(rd.lookup_classes(e, f)) == 3 and rd.lookup_rates(e, f)[0].shape == (3, 3)
        assert rd.lookup_rates(e, f)[1].shape == (3,)
    # frame 4 carries the same phantom but is NOT in the list: untouched (the list is authoritative)
    assert [str(t) for t in rd.lookup_track_ids(e, 4)] == ["t_car", "t_jump", "t_ego"]
    # the OTHER clip is not listed either
    assert rd.lookup(_eid(rd, CLIP_B), 2).shape == (1, 6)
    st = rd.defect_stats()
    assert st["n_ego_rows_removed"] == 2 and st["n_ego_frames_hit"] == 2


def test_MUTATION_a_mask_that_ignores_the_frame_list_strips_the_unlisted_frame_and_goes_red(tmp_path):
    class Everywhere(H.JoinDefectMasks):                 # the deliberate regression: geometry only, no list
        def has_ego_frame(self, clip_id, frame_idx):
            return True
    m = Everywhere({}, {})
    rd = _reader(_write(tmp_path), m)
    e = _eid(rd)
    assert [str(t) for t in rd.lookup_track_ids(e, 4)] == ["t_car", "t_jump"]          # the row is gone ...
    # ... so the assertion of the real test (frame 4 keeps its phantom row) FAILS for this mutant
    assert [str(t) for t in rd.lookup_track_ids(e, 4)] != ["t_car", "t_jump", "t_ego"]


def test_a_synthetic_ego_footprint_box_in_a_listed_frame_is_removed_and_its_neighbours_are_untouched():
    m = H.JoinDefectMasks({H.clip_key("c"): {7}}, {})
    ag = [_agent(1.4, 0.0, tid="ego", l=4.6, w=2.0), _agent(12.0, -1.5, tid="car"), _agent(3.0, 1.2, tid="near")]
    out, n = m.strip_ego_boxes("c", 7, ag)
    assert n == 1 and [a["track_id"] for a in out] == ["car", "near"]                  # (3.0, 1.2): |y| >= 1 -> kept
    same, n0 = m.strip_ego_boxes("c", 8, ag)                                          # an unlisted frame: the SAME list
    assert n0 == 0 and same is ag


# =========================================================================== #
# 3. fix 4 -- the track-id switch                                              #
# =========================================================================== #
def test_unmasked_rates_carry_the_jump_as_a_255_m_per_s_target(tmp_path):
    """The defect, measured on the synthetic join: records 2 and 3 of t_jump read 255 m/s (51 m over 0.2 s); the
    neighbours read the true 10 m/s."""
    rd = _reader(_write(tmp_path), None)
    e = _eid(rd)
    vx = {f: float(rd.lookup_rates(e, f)[0][1, 0]) for f in range(1, 5)}               # row 1 = t_jump
    assert vx[1] == pytest.approx(10.0, abs=1e-6) and vx[4] == pytest.approx(10.0, abs=1e-6)
    assert vx[2] == pytest.approx(255.0, abs=1e-6) and vx[3] == pytest.approx(255.0, abs=1e-6)


def _violations(rd) -> list:
    e = _eid(rd)
    bad = []
    for f in (2, 3):                                           # the two contaminated records: masked AND zeroed
        ids = [str(t) for t in rd.lookup_track_ids(e, f)]
        i = ids.index("t_jump")
        rt, rm = rd.lookup_rates(e, f)
        if bool(rm[i]) or not np.all(rt[i] == 0.0):
            bad.append(("not masked", f))
    for f in (1, 4):                                           # the clean neighbours keep their TRUE rate
        ids = [str(t) for t in rd.lookup_track_ids(e, f)]
        i = ids.index("t_jump")
        rt, rm = rd.lookup_rates(e, f)
        if (not bool(rm[i])) or abs(float(rt[i, 0]) - 10.0) > 1e-6:
            bad.append(("clean row damaged", f))
    for f in (1, 2, 3, 4):                                     # t_car is never touched
        ids = [str(t) for t in rd.lookup_track_ids(e, f)]
        i = ids.index("t_car")
        rt, rm = rd.lookup_rates(e, f)
        if (not bool(rm[i])) or abs(float(rt[i, 0]) - 10.0) > 1e-6:
            bad.append(("t_car touched", f))
    return bad


def test_the_jump_is_masked_at_exactly_records_f_minus_1_and_f(tmp_path):
    rd = _reader(_write(tmp_path), _masks())
    assert _violations(rd) == []
    st = rd.defect_stats()
    assert st["n_rate_rows_masked"] == 2 and st["n_rate_records_hit"] == 2


def test_MUTATION_masking_only_record_f_leaves_a_255_m_per_s_target_and_goes_red(tmp_path):
    only_f = {(H.clip_key(CLIP), 3): frozenset({H.track_key("t_jump")})}
    rd = _reader(_write(tmp_path), _masks(rate_records=only_f))
    bad = _violations(rd)
    assert ("not masked", 2) in bad                                # record f-1 still carries the jump
    assert float(rd.lookup_rates(_eid(rd), 2)[0][1, 0]) == pytest.approx(255.0, abs=1e-6)


def test_MUTATION_masking_three_records_damages_a_clean_row_and_goes_red(tmp_path):
    wide = {(H.clip_key(CLIP), f): frozenset({H.track_key("t_jump")}) for f in (1, 2, 3)}
    bad = _violations(_reader(_write(tmp_path), _masks(rate_records=wide)))
    assert ("clean row damaged", 1) in bad


def test_the_masked_row_stays_aligned_with_its_track_in_every_array(tmp_path):
    rd = _reader(_write(tmp_path), _masks())
    e = _eid(rd)
    ids = [str(t) for t in rd.lookup_track_ids(e, 2)]
    assert ids == ["t_car", "t_jump", "t_far"]
    rt, rm = rd.lookup_rates(e, 2)
    assert list(rm) == [True, False, True]                        # t_car observed, t_jump MASKED, t_far one-sided (0 m/s, kept)
    assert rt[2].tolist() == [0.0, 0.0, 0.0] and rt[1].tolist() == [0.0, 0.0, 0.0]
    assert rt.dtype == np.float32 and rm.dtype == bool


def test_apply_rate_mask_never_touches_its_inputs_and_keeps_unobserved_rows_unobserved():
    m = H.JoinDefectMasks({}, {H.clip_key("c"): {(5, H.track_key("x"))}})
    ag = [{"track_id": "x"}, {"track_id": "y"}]
    rt = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rm = np.array([True, True])
    r2, m2 = m.apply_rate_mask("c", 5, ag, rt, rm)
    assert r2.tolist() == [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]] and m2.tolist() == [False, True]
    assert rt.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] and rm.tolist() == [True, True]      # inputs intact
    r3, m3 = m.apply_rate_mask("c", 9, ag, rt, rm)                                                  # an unlisted record
    assert r3 is rt and m3 is rm


# =========================================================================== #
# 4. the loader's refusals                                                     #
# =========================================================================== #
def test_load_refuses_a_wrong_schema_a_missing_section_and_a_raw_looking_key(tmp_path):
    good = {"schema": H.SCHEMA, "ego_footprint": {"frames": {"0123456789ab": [1]}},
            "track_jumps": {"events": {"0123456789ab": [[1, "fedcba987654"]]}}}
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(good))
    m = H.JoinDefectMasks.load(p)
    assert m.stamp["n_ego_frames"] == 1 and m.stamp["n_track_events"] == 1 and len(m.stamp["sha256"]) == 64
    for name, mut in (("schema", lambda d: d.update(schema="x/1")),
                      ("section", lambda d: d.pop("track_jumps")),
                      ("rawkey", lambda d: d["ego_footprint"]["frames"].update({"aaaaaaaa-0000-4000-8000-000000000001": [1]}))):
        d = json.loads(json.dumps(good))
        mut(d)
        q = tmp_path / f"{name}.json"
        q.write_text(json.dumps(d))
        with pytest.raises(ValueError):
            H.JoinDefectMasks.load(q)
    with pytest.raises(ValueError):
        H.JoinDefectMasks.load(tmp_path / "missing.json")


# =========================================================================== #
# 5. the MINED lists == D3's banked aggregates                                 #
# =========================================================================== #
#: D3 raw/c3b_agents_detail.json (ego_footprint / jump_events), copied
D3_EGO_PER_CLIP = [121, 112, 109, 76, 75, 51, 35, 26, 24, 14, 9, 9, 9, 6, 6, 5, 4, 2]
D3_EGO_CLIPS = ["0b0b4d51fcae", "12d32539099d", "13cf9a018b88", "25f5497ad4b4", "26a495963798", "2dbf6145c79a",
                "45e5ab168091", "541a78acd052", "54b166d5f5b5", "664255bfb077", "6ce1028bce56", "94c89eaf4036",
                "9fde498c6bde", "b9627d84cd4b", "d1817e0898bf", "dd6c33d5d52f", "eb8c04a9b386", "f0695071221a"]


def test_the_shipped_lists_reproduce_d3s_ego_aggregates_and_jump_counts():
    d = json.loads(CFG.read_text(encoding="utf-8"))
    assert d["schema"] == H.SCHEMA
    fr = d["ego_footprint"]["frames"]
    assert sorted(fr) == D3_EGO_CLIPS
    assert sorted((len(v) for v in fr.values()), reverse=True) == D3_EGO_PER_CLIP
    assert sum(len(v) for v in fr.values()) == 693 == d["ego_footprint"]["n_boxes"]
    tj = d["track_jumps"]
    assert tj["n_event_frames_all"] == 6410 and tj["n_pose_glitch_like_excluded"] == 853
    assert sum(len(v) for v in tj["events"].values()) == tj["n_track_events_listed"] == 6227
    ag = d["provenance"]["d3_aggregates"]
    assert ag["jump_events"] == {"n_event_frames": 6410, "n_clips": 887, "pose_glitch_like": 853, "single_track_like": 5822}
    assert ag["ego_footprint"]["frac_within_0.5m_of_(1.4,0)"] == pytest.approx(0.8427128427128427, abs=1e-15)
    # no raw clip id anywhere: every key is a sha12
    for k in list(fr) + list(tj["events"]):
        assert len(k) == 12 and set(k) <= set("0123456789abcdef")


def test_the_shipped_file_loads_and_expands_to_the_two_record_rule():
    m = H.JoinDefectMasks.load(CFG)
    assert m.stamp["n_ego_frames"] == 693 and m.stamp["n_ego_clips"] == 18
    # every listed event contaminates exactly records f-1 and f
    n_pairs = len({(ck, f - 1) for ck, ev in m.track_events.items() for f, _t in ev}
                  | {(ck, f) for ck, ev in m.track_events.items() for f, _t in ev})
    assert len(m._rate_records) == n_pairs
    ck = next(iter(m.track_events))
    f, t = sorted(m.track_events[ck])[0]
    assert t in m._rate_records[(ck, f)] and t in m._rate_records[(ck, f - 1)]
