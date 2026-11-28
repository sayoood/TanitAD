"""refcv7 NEW-2 -- SPEC_REFCV7 §23 (A18, the PI 2026-09-27 "Budget to 3,000 steps"): the
REGISTERED G-MAP-OVERFIT spec ``raw/gmo_spec_A18.json`` AS WRITTEN (md5 pinned), read through the
harness's own ``load_spec`` and ``lr_multiplier`` (NEW-2 R5): 3,000 steps, the A17.1 decay moved
to steps 2,700-3,000, A15's map path, every bar and must-fail unchanged -- and nothing else
changed against the A17.1 spec it amends.

A file of its own (not ``test_map_hires_overfit.py``) so the MAP-LIFT landing does not edit R5's
test file; the launch gate's own pin of the same spec (by sha256) is in ``test_launch_gate.py``.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import map_hires_overfit as O                                   # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7    # noqa: E402

RAW = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" / "Research"
       / "2026-09-26-refcv7-map-hires" / "raw")
A18_SPEC = RAW / "gmo_spec_A18.json"
A171_SPEC = RAW / "gmo_spec_A171.json"
#: md5 of the A18 spec AS WRITTEN (registered by SPEC_REFCV7 §23, 37086c3)
A18_SPEC_MD5 = "4eda0636f1a41a9b60b59ccb6a4afca3"


@pytest.mark.skipif(not A18_SPEC.is_file(), reason="the A18 spec is not in this checkout")
def test_A18_the_spec_loads_as_written_and_amends_only_what_A18_says():
    assert hashlib.md5(A18_SPEC.read_bytes()).hexdigest() == A18_SPEC_MD5
    s = O.load_spec(A18_SPEC, decision_rule="prior_corrected", class_weights="w.json",
                    band_keys=EXTENT_REFCV7.band_keys)
    assert s["steps"] == 3000 and s["eval_every"] == 100                        # A18
    assert s["lr_decay"] == {"kind": "cosine_to_zero", "start_step": 2700}
    m = O.lr_multiplier(s)
    assert m(1) == 1.0 and m(2700) == 1.0                                         # held
    assert m(2850) == pytest.approx(0.5, abs=1e-12)                               # half-way
    assert m(3000) == pytest.approx(0.0, abs=1e-12)                               # zero
    assert s["near_lift_m"] == 20.0 and s["near_refine_blocks"] == 1            # A15's config
    assert "class_weights_definition" not in s                                   # sqrt_mf (A8)
    assert s["must_fail"] == {"lane_w0": ["lane"],
                              "s8_zeros": ["lane", "crosswalk", "arrow", "edge", "hatched"],
                              "near_block_zeros": ["edge"]}
    assert s["must_fail_all"] == {"s8_zeros": True}
    assert len(s["frames"]) == 16 and s["batch"] == 4 and s["lr"] == 0.001 and s["seed"] == 0
    assert s["thresholds"]["iou"] == {"nocls": 0.85, "drivable": 0.85, "sidewalk": 0.85,
                                      "lane": 0.5, "crosswalk": 0.5, "arrow": 0.5,
                                      "edge": 0.5, "hatched": 0.5}           # bars unchanged
    assert s["amends"]["spec_md5"] == "5abd5b907738fc735a0def0c1e7fbb64"      # A17.1, as landed
    if A171_SPEC.is_file():
        base = O.load_spec(A171_SPEC, decision_rule="prior_corrected",
                           class_weights="w.json", band_keys=EXTENT_REFCV7.band_keys)
        diff = sorted(k for k in set(base) | set(s) if base.get(k) != s.get(k))
        assert diff == ["amends", "lr_decay", "registered", "steps"], diff
