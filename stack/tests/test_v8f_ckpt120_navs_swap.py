"""PREREG_V8F_CKPT120, MM ruling on part 1's deviation 3: on DAgger-swapped rows, ``nav_s``'s countdown (channel 3) is rebuilt from the OWN travelled arc, by the same rule as the swap's ``nav`` channel 3
(real frames sharing NOW's token and validity; NOW is the anchor; ``clip(nav_s[NOW, 3] + own arc behind NOW / 100, 0, 3)``; channels 0-2 and 4 untouched). The evaluator wrappers dead-reckon ``nav_s`` like the
tactical channel, so training and evaluation must match. Every check here is run on a synthetic bank record built from the batch itself (no model, no bank file): the swap function is the real one.
Mutation: the one line that writes the rebuilt ``nav_s`` channel is replaced by the logged value; the same check must go RED."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8f_ckpt120 import REPO, make_batch, mod_from_text, world  # noqa: E402,F401  (world: the module fixture of part 1)

from tanitad.data import v8f_dagger as G  # noqa: E402

DAG_P = REPO / "stack" / "tanitad" / "data" / "v8f_dagger.py"
NAVS_LINE = 'new["nav_s"][rows, :, 3] = torch.where(ns_ok, ns3.to(ns_r.dtype), ns_r[..., 3])'


def synth_rec(b, rows, step_m=4.0, shift=7.0):
    """A bank record for ``rows`` taken from the batch itself: own history rows = the logged cache steps (j = t), an own arc growing ``step_m`` metres per frame, own poses = the logged poses moved by ``shift`` m."""
    F = int(b["p10f"].shape[1])
    n = int(rows.numel())
    pose = b["pose"][rows].double().cpu().numpy().copy()
    pose[..., 0] += shift
    arc = np.tile(np.arange(F, dtype=np.float64) * step_m, (n, 1))
    return {"p10f": b["p10f"][rows].float().cpu().numpy(), "j": b["t"][rows].cpu().numpy().astype(np.int32), "kin3": b["kin3"][rows].double().cpu().numpy(), "pose": pose, "arc": arc}


def usable_rows(b):
    """Rows whose NOW frame is a real kinematic row (the swap refuses any other)."""
    return torch.nonzero((b["t"] >= 1) & torch.isfinite(b["p10f"][:, -1]).all(-1)).flatten()


def check_navs_rebuilt(out, b, rows, rec, mask_nav_s=True):
    """The rule, spelt out per frame: rebuilt where the frame is real (logged and own), shares NOW's token and validity and has a route; as logged elsewhere. Returns (n_rebuilt, n_boundary_kept)."""
    ns, F = b["nav_s"], int(b["nav_s"].shape[1])
    n_reb = n_kept = 0
    for k, r in enumerate(rows.tolist()):
        t, j, arc = int(b["t"][r]), int(rec["j"][k]), rec["arc"][k]
        x = ns[r].numpy()
        for f in range(F):
            real = (t - (F - 1) + f >= 1) and bool(b["frame_ok"][r, f]) and (j - (F - 1) + f >= 1) and bool(np.isfinite(rec["p10f"][k, f]).all())
            same = (x[f, :3].argmax() == x[-1, :3].argmax()) and x[f, 4] == x[-1, 4]
            got = out["nav_s"][r, f, 3].item()
            if real and same and x[f, 4] > 0:
                want = float(np.clip(float(x[-1, 3]) + (arc[-1] - arc[f]) / 100.0, 0.0, 3.0))
                assert got == pytest.approx(want, abs=1e-6), f"row {r} frame {f}: nav_s countdown must follow the own arc"
                n_reb += int(abs(want - float(x[f, 3])) > 1e-9)
            else:
                assert got == x[f, 3], f"row {r} frame {f}: a frame across a boundary / not real / without a route keeps its logged nav_s"
                n_kept += int(real and not same)
        assert np.array_equal(out["nav_s"][r][:, [0, 1, 2, 4]].numpy(), x[:, [0, 1, 2, 4]]), "token and validity of nav_s are never touched"
    return n_reb, n_kept


def test_swap_rebuilds_nav_s_countdown_from_the_own_arc_like_nav(world):
    b, J = make_batch(world, n=16)
    rows = usable_rows(b)
    assert rows.numel() >= 8
    rec = synth_rec(b, rows)
    out = G.swap_rows(b, rows, rec)
    n_reb, n_kept = check_navs_rebuilt(out, b, rows, rec)
    assert n_reb > 0, "the check must contain rebuilt frames, or it is vacuous"
    assert n_kept > 0, "and real frames across a token boundary (random token per row in the fixture), or the boundary rule is untested"
    keep = [r for r in range(int(b["nav_s"].shape[0])) if r not in set(rows.tolist())]
    assert torch.equal(out["nav_s"][keep], b["nav_s"][keep]), "rows the swap did not select keep nav_s bit for bit"
    assert float(out["nav_s"][..., 3].min()) >= 0.0 and float(out["nav_s"][..., 3].max()) <= 3.0
    b0 = {k: v for k, v in b.items() if k != "nav_s"}
    out0 = G.swap_rows(b0, rows, rec)
    assert "nav_s" not in out0, "a batch with no nav_s (every pre-CKPT120 arm) gets no nav_s key"
    assert torch.equal(out0["nav"], out["nav"]) and torch.equal(out0["pose_s"], out["pose_s"]), "the tactical nav and the frame are the same with or without nav_s"


def test_swap_item_carries_the_rebuilt_nav_s(world):
    from pathlib import Path as P
    from tanitad.data import v8f_mini_data as D
    from test_v8f_ckpt120 import joined
    J = joined(world)
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    it = next(x for x in (ds.item(w[0], w[1], 1, 3) for w in wins) if x["t"] >= 9)
    b = D.collate([it])
    rec = synth_rec(b, torch.zeros(1, dtype=torch.long))
    out = G.swap_item(it, rec)
    want = G.swap_rows(b, torch.zeros(1, dtype=torch.long), rec)["nav_s"][0].numpy()
    assert np.array_equal(out["nav_s"], want) and not np.array_equal(out["nav_s"], it["nav_s"])
    assert np.array_equal(out["nav_s"][:, [0, 1, 2, 4]], it["nav_s"][:, [0, 1, 2, 4]])
    it0 = {k: v for k, v in it.items() if k != "nav_s"}
    assert "nav_s" not in G.swap_item(it0, rec), "an item with no nav_s (every pre-CKPT120 arm) gets none"


def test_mutation_leaving_nav_s_as_logged_turns_the_rebuild_check_red(world):
    b, J = make_batch(world, n=16)
    rows = usable_rows(b)
    rec = synth_rec(b, rows)
    text = DAG_P.read_text(encoding="utf-8")
    assert text.count(NAVS_LINE) == 1, text.count(NAVS_LINE)
    mut = mod_from_text(text.replace(NAVS_LINE, "pass"), "v8f_dagger_mut_navs", DAG_P)
    out_m = mut.swap_rows(b, rows, rec)
    assert torch.equal(out_m["nav_s"], b["nav_s"]), "the mutant leaves nav_s as logged"
    with pytest.raises(AssertionError):
        check_navs_rebuilt(out_m, b, rows, rec)
    check_navs_rebuilt(G.swap_rows(b, rows, rec), b, rows, rec)      # the real function passes the same check
