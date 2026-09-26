"""On TRAIN, the blacklist-blind sample-count formula is WRONG — so it must refuse to stand in there.

⛔ MEASURED 2026-09-26 on the real nuScenes trainval metadata (Master Mind: "assert the blacklist
intersection on train explicitly, so the formula can never silently stand in for the measurement").
``expected_sample_counts`` is arithmetic and ignores ST-P3's scene blacklist. On val that is harmless (no
val scene is blacklisted — pinned in test_nuscenes_planning.py). On train it is not:

* the train list is 700 scenes by TWO independent routes (the devkit's ``train_detect ∪ train_track`` and
  ``scene.json − val``), which agree exactly;
* 16 of the 22 blacklisted scenes are in train; the other 6 (0309–0314) are not in trainval at all;
* ST-P3's real train count is **22,020**; the formula says **22,530** — +510, silently.

Every expectation is a MEASURED LITERAL.
"""
from __future__ import annotations

import ast
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_TE)
for _p in (os.path.join(_REPO, "stack"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import nuscenes_planning as NP              # noqa: E402

SPLITS = os.path.join(_REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming",
                      "2026-09-19-nuscenes-planning-harness", "raw", "devkit_splits_b40adc4.py.txt")
META_ROOT = "D:/Archive/devbox-C/nuscenes/data"
SCENE_JSON = os.path.join(META_ROOT, "v1.0-trainval", "scene.json")

TRAIN_BLACKLISTED = ["scene-0161", "scene-0162", "scene-0163", "scene-0164", "scene-0165", "scene-0166",
                     "scene-0167", "scene-0168", "scene-0170", "scene-0171", "scene-0172", "scene-0173",
                     "scene-0174", "scene-0175", "scene-0176", "scene-0419"]


def _devkit_train() -> list[str]:
    """``train = sorted(set(train_detect + train_track))`` in the devkit — not a literal, so it is rebuilt
    from its two literal halves (a plain ``literal_eval`` pass silently skips it)."""
    src = open(SPLITS, encoding="utf-8").read()
    lists = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                lists[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return sorted(set(lists["train_detect"] + lists["train_track"]))


def test_train_is_700_scenes_and_16_of_them_are_blacklisted():
    train = _devkit_train()
    assert len(train) == 700
    assert sorted(n for n in train if int(n[-4:]) in NP.STP3_SCENE_BLACKLIST) == TRAIN_BLACKLISTED


def test_six_blacklist_entries_are_not_in_trainval_at_all():
    names = set(_devkit_train()) | set(NP.DEVKIT_VAL_SCENES)
    assert sorted(NP.STP3_SCENE_BLACKLIST - {int(n[-4:]) for n in names}) == [309, 310, 311, 312, 313, 314]


def test_the_formula_REFUSES_on_train():
    """⛔ The pin: given train's scene names, the formula must refuse rather than return a wrong count."""
    with pytest.raises(NP.RefusedInput, match="16 of these scenes"):
        NP.expected_sample_counts(28130, 700, scene_names=_devkit_train())


def test_the_formula_still_answers_on_val():
    """Control: where its assumption holds, the guard must not get in the way."""
    assert NP.expected_sample_counts(6019, 150, scene_names=NP.DEVKIT_VAL_SCENES) == \
        {"uniad": 6019, "vad": 5119, "stp3": 4819}


@pytest.mark.skipif(not os.path.exists(SCENE_JSON), reason="nuScenes metadata not on this box")
def test_devkit_train_equals_scene_json_minus_val():
    """Two independent routes to the train list must agree exactly."""
    names = {s["name"] for s in json.load(open(SCENE_JSON, encoding="utf-8"))}
    assert _devkit_train() == sorted(names - set(NP.DEVKIT_VAL_SCENES))


@pytest.mark.skipif(not os.path.exists(SCENE_JSON), reason="nuScenes metadata not on this box")
def test_MUTATION_on_train_the_formula_would_be_wrong_by_510():
    """⭐ Why the refusal exists: the MEASURED ST-P3 train count is 22,020; the blacklist-blind formula
    gives 22,530. If these ever agree, the blacklist has stopped doing anything."""
    meta = NP.Meta.load(META_ROOT, "v1.0-trainval")
    rows = NP.sample_sets(meta, _devkit_train())
    measured = sum(1 for r in rows if r["stp3"])
    naive = NP.expected_sample_counts(len(rows), 700)["stp3"]      # no scene_names -> the old, blind path
    assert (len(rows), measured, naive) == (28130, 22020, 22530)
