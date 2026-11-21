"""smoke_check.load() must slice ONLY per-window keys.  MEASURED 2026-10-20 06:35: the first real smoke died with
`IndexError: index 4 is out of bounds for axis 0 with size 4` because `decisions/*.npz` carries `sel_bank_nav_true`
with shape (4, 8, 2) -- not per window -- and the old loader indexed every key by the window position."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smoke_check as SC  # noqa: E402


def _dump(tmp_path: Path) -> Path:
    d = tmp_path / "dump"
    (d / "decisions").mkdir(parents=True)
    json.dump({"episodes": [{"file_index": 0, "episode_index": 0}]}, open(d / "manifest.json", "w"))
    np.savez(d / "decisions" / "ep000.npz", ws=np.array([7, 12, 17, 62, 82, 102]), nav_cmd=np.array([2, 2, 2, 0, 0, 0]),
             sel_bank_nav_true=np.zeros((4, 8, 2)), ep_poses=np.zeros((199, 4)))
    return d


def test_load_keeps_only_per_window_keys(tmp_path):
    man, out = SC.load(_dump(tmp_path))
    assert sorted(w for (_e, w) in out) == [7, 12, 17, 62, 82, 102]
    assert set(out[(0, 82)]) == {"ws", "nav_cmd"}            # literal: the (4,8,2) bank and ep_poses are dropped
    assert int(out[(0, 82)]["nav_cmd"]) == 0


def test_mutation_old_loader_raises(tmp_path):
    d = _dump(tmp_path)
    dec = np.load(d / "decisions" / "ep000.npz")
    with pytest.raises(IndexError):                           # the 06:35 failure, reproduced
        {k: dec[k][4] for k in dec.files if k != "ep_poses"}
