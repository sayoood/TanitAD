"""⛔ A `--join3d` that was ASKED FOR must exist and cover the corpus -- or the run REFUSES.

⭐ THE DEFECT, MEASURED 2026-09-23 by reading the call site before refcv6's launch:
``open_join3d`` returns ``None`` for a path that does not exist (correct for its default
caller, which may run masked), and ``train()`` handed that ``None`` straight to
``enable_join3d``. A mistyped path -- or a build that had not finished writing -- trained the
2-D rung (``zh_mask`` all-False, ``box3d_n_z 0``) for the whole run while ``argv`` named a 3-D
join. A join that exists but covers none of the corpus's clips did the same.

Pinned against the REAL reader (``AgentJoin3D.open`` on a file in the builder's line format)
and the REAL guard ``train()`` calls, with a GREEN control so a guard that refuses everything
cannot pass. CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T                                    # noqa: E402
from tanitad.data import agent_cuboid_gt as C                # noqa: E402


def _join3d(tmp_path, clip="clip-A") -> Path:
    """Two lines in the 3-D builder's format: the 2-D line plus `cz`/`h` per agent."""
    p = tmp_path / "j3d.jsonl"
    rows = [{"clip_id": clip, "frame": f, "frame_idx": f - 2, "t_s": 0.1 * f,
             "agents": [{"cx": 5.0, "cy": 0.0, "yaw": 0.0, "l": 4.5, "w": 1.9, "occ": 0,
                         "track_id": "t1", "cls": "automobile", "cz": 0.75, "h": 1.5}]}
            for f in (2, 3)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_a_MISSING_path_that_was_asked_for_REFUSES(tmp_path):
    missing = tmp_path / "not_built_yet.jsonl.xz"
    j = C.open_join3d(str(missing), clips={"clip-A"})
    assert j is None                                   # the reader's documented default
    with pytest.raises(SystemExit) as e:
        T.require_join3d(j, str(missing), 1)
    assert "does not exist" in str(e.value)


def test_a_join_covering_ZERO_of_the_corpus_REFUSES_on_train(tmp_path):
    j = C.open_join3d(str(_join3d(tmp_path)), clips={"clip-B"})   # the WRONG corpus
    assert j is not None and j.n_lines == 0
    with pytest.raises(SystemExit) as e:
        T.require_join3d(j, "j3d.jsonl", 1)
    assert "ZERO" in str(e.value)


def test_GREEN_a_covering_join_passes_unchanged(tmp_path):
    """⭐ The control: without it the two refusals above could be a brick."""
    j = C.open_join3d(str(_join3d(tmp_path)), clips={"clip-A"})
    assert j.n_lines == 2
    assert T.require_join3d(j, "j3d.jsonl", 1) is j


def test_zero_coverage_on_EVAL_warns_and_passes(tmp_path, capsys):
    j = C.open_join3d(str(_join3d(tmp_path)), clips={"clip-B"})
    assert T.require_join3d(j, "j3d.jsonl", 1, split="eval") is j
    assert "n = 0" in capsys.readouterr().out


def test_train_CALLS_the_guard_at_BOTH_join3d_sites():
    """The call sites: the historical defect lived there, not in the reader."""
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    assert src.count("ds.enable_join3d(require_join3d(") == 2
    assert "split=\"eval\"))" in src
