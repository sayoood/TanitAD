"""refcv8 join-defect masks for the 4,572-clip train view: pins of ``refcv8_join_label_defects_train4572.json``.

The file is the OLD ``refcv8_join_label_defects.json`` (4,369 train + eval139, mined over the 3-D join md5 401c9c31...)
PLUS the entries of the 58 ADD clips that sit in that join. Every expectation is a LITERAL taken from the miner's own
re-run over the old manifest (which reproduced the old file byte for byte except ``wall_s``), never an expression over
the code under test:

* the OLD file's 18 ego clips / 693 frames / 828 jump clips / 6,227 events are all present and identical;
* the only new keys are the five ADD sha12s below, with the exact frame counts;
* a deliberate-regression arm drops one old key / one old frame and MUST be detected by the same comparison.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tanitad.data import join_label_hygiene as H

CONF = Path(__file__).resolve().parents[1] / "tanitad" / "configs"
OLD = CONF / "refcv8_join_label_defects.json"
NEW = CONF / "refcv8_join_label_defects_train4572.json"

NEW_EGO = {"5ed957d47a41": 52, "70928506fcc6": 127, "9dee20474b44": 196}          # ADD clips: ego-as-agent boxes
NEW_JUMPS = {"11c906128f92": 1, "4b3773213797": 1, "9dee20474b44": 3}             # ADD clips: track-id-switch events


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _old_entries_survive(old: dict, new: dict) -> bool:
    oe, ov = old["ego_footprint"]["frames"], old["track_jumps"]["events"]
    ne, nv = new["ego_footprint"]["frames"], new["track_jumps"]["events"]
    return all(ne.get(k) == v for k, v in oe.items()) and all(nv.get(k) == v for k, v in ov.items())


def test_old_file_is_unchanged_anchor():
    m = H.JoinDefectMasks.load(OLD)
    s = m.stats()
    assert (s["n_ego_frames"], s["n_ego_clips"], s["n_track_events"], s["n_track_event_clips"]) == (693, 18, 6227, 828)


def test_new_file_totals():
    m = H.JoinDefectMasks.load(NEW)                       # refuses a wrong schema / a raw-looking clip id
    s = m.stats()
    assert (s["n_ego_frames"], s["n_ego_clips"], s["n_track_events"], s["n_track_event_clips"]) == (1068, 21, 6232, 831)
    d = _load(NEW)
    assert d["ego_footprint"]["n_boxes"] == 1068 and d["ego_footprint"]["n_clips"] == 21
    assert d["track_jumps"]["n_event_frames_all"] == 6415 and d["track_jumps"]["n_pose_glitch_like_excluded"] == 853
    assert d["track_jumps"]["n_track_events_listed"] == 6232
    assert d["provenance"]["join_md5"] == "401c9c3119a50632dafc35c595d9e051"
    assert [x["md5"] for x in d["provenance"]["manifests"]] == ["77dafb50af0aedd1b940cf12c12518ef",
                                                                 "3443323229279a18253691ff36ebffd3"]
    assert d["provenance"]["n_records"] == 875657        # every line of the join: 864,524 old + 11,133 ADD
    assert _load(OLD)["ego_footprint"]["rule"] == d["ego_footprint"]["rule"]
    assert _load(OLD)["track_jumps"]["rule"] == d["track_jumps"]["rule"]


def test_every_old_entry_survives_and_new_keys_are_exactly_the_add_clips():
    old, new = _load(OLD), _load(NEW)
    assert _old_entries_survive(old, new)
    ne, nv = new["ego_footprint"]["frames"], new["track_jumps"]["events"]
    oe, ov = old["ego_footprint"]["frames"], old["track_jumps"]["events"]
    assert {k: len(ne[k]) for k in set(ne) - set(oe)} == NEW_EGO
    assert {k: len(nv[k]) for k in set(nv) - set(ov)} == NEW_JUMPS
    assert all(len(k) == 12 for k in list(ne) + list(nv))


def test_deliberate_regression_is_detected():
    """The comparison above must go RED when an old key or an old frame is lost (it is not vacuous)."""
    old, new = _load(OLD), _load(NEW)
    k = sorted(old["ego_footprint"]["frames"])[0]
    lost_key = copy.deepcopy(new)
    del lost_key["ego_footprint"]["frames"][k]
    assert not _old_entries_survive(old, lost_key)
    k2 = sorted(old["track_jumps"]["events"])[0]
    lost_event = copy.deepcopy(new)
    lost_event["track_jumps"]["events"][k2] = lost_event["track_jumps"]["events"][k2][1:]
    assert not _old_entries_survive(old, lost_event)
