"""K-step rollout lever (train.rollout_k) — mechanism tests.

The bake-off lever `kstep_rollout` was 'planned' until the mechanism landed;
these tests pin (a) default-off behavior, (b) the future-frame index extension,
(c) a runnable smoke train at K=2, (d) the lever's OFAT isolation.
"""

import torch

from tanitad.config import smoke_config
from tanitad.data.toy_driving import ToyDrivingDataset
from tanitad.eval.bakeoff import default_levers, lever_diff
from tanitad.models.fourbrain import WorldModel
from tanitad.train.train_worldmodel import (_needed_future_indices,
                                            _rollout_loss)


def test_default_off_and_indices_extended():
    cfg = smoke_config()
    assert cfg.train.rollout_k == 1
    base_needed, _ = _needed_future_indices(cfg)
    cfg.train.rollout_k = 4
    needed, idx_of = _needed_future_indices(cfg)
    assert set(range(4)) <= set(needed)          # rollout targets 0..K-1
    assert set(base_needed) <= set(needed)       # horizon targets kept
    assert all(i in idx_of for i in needed)


def test_rollout_loss_runs_and_is_finite():
    cfg = smoke_config()
    cfg.train.rollout_k = 2
    model = WorldModel(cfg)
    needed, idx_of = _needed_future_indices(cfg)
    ds = ToyDrivingDataset([0], window=cfg.predictor.window,
                           max_horizon=max(needed) + 1,
                           size=cfg.encoder.image_size, steps=30)
    b = torch.utils.data.default_collate([ds[0], ds[1]])
    assert "future_actions" in b                 # new contract key
    states = model.encode_window(b["frames"].float())
    fut = model.encode_window(b["future_frames"].float()[:, needed])
    loss = _rollout_loss(model, states, b["actions"], fut,
                         b["future_actions"], idx_of, K=2)
    assert torch.isfinite(loss) and loss.item() >= 0
    # zero-order-hold fallback when future actions are unavailable
    loss2 = _rollout_loss(model, states, b["actions"], fut, None, idx_of, K=2)
    assert torch.isfinite(loss2)


def test_lever_is_runnable_and_one_factor():
    levers = {l.name: l for l in default_levers()}
    assert "kstep_rollout" in levers
    lever = levers["kstep_rollout"]
    base = smoke_config()
    variant = lever.apply(smoke_config())
    changed = lever_diff(base, variant)
    assert set(changed) == {"train.rollout_k"}   # OFAT isolation holds
    assert variant.train.rollout_k == 4


def test_cached_data_mode(tmp_path):
    import torch as _t

    from tanitad.data.mixing import save_episode
    from tanitad.data.toy_driving import ToyEpisode
    from tanitad.train.train_worldmodel import _build_datasets

    cfg = smoke_config()
    for split in ("train", "val"):
        d = tmp_path / f"toy-{split}-abc123"
        d.mkdir()
        for i in range(2):
            ep = ToyEpisode(
                frames=_t.zeros(30, cfg.encoder.in_channels,
                                cfg.encoder.image_size, cfg.encoder.image_size,
                                dtype=_t.uint8),
                actions=_t.zeros(30, 2), poses=_t.zeros(30, 4), episode_id=i)
            save_episode(ep, str(d / f"ep_{i:05d}.pt"))
    tr, va = _build_datasets(cfg, 2, "cached", str(tmp_path))
    assert len(tr) > 0 and len(va) > 0
    item = tr[0]
    assert "future_actions" in item
