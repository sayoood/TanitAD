"""ClipWins.item (collect_latents_train.py) == V8FMiniWindows.item on eval windows: feats, act, frame_ok, t compared array for array. CPU."""
import numpy as np
import wmact_lib as W
from collect_latents_train import ClipWins


def test_shim_equals_real_item():
    ds, wins, eid = W.open_windows(16)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(wins), 60, replace=False)
    bad, padded = 0, 0
    for i in pick:
        ci, t, _, _ = wins[i]
        real = ds.item(ci, t)
        shim = ClipWins(ds.store.get(str(ds.J["clip"][ci])), [t], 16).item(0, t)
        ok = (np.array_equal(real["feats"], shim["feats"]) and np.array_equal(real["act"], shim["act"])
              and np.array_equal(real["frame_ok"], shim["frame_ok"]) and real["t"] == shim["t"])
        bad += (not ok)
        padded += (not real["frame_ok"].all())
    print("windows compared", len(pick), "mismatches", bad, "with a padded frame", padded)
    assert bad == 0
