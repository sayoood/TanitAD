"""refcv9 S1 -- the step-time levers behind ``--r9-perc-rig``. Every expectation is a LITERAL; each guard carries a RED arm.

MEASURED on Thor (smoke4, 2026-11-02): 50 steps, steady state ~4.4 s/step; the rig's loader decoded 32 raw 416x1024 PNGs per sample
(window 8 + max_horizon 20 rows, n_stack 3) and shipped 1.718 GB per collated batch, while ``compute_losses_perc_rig`` reads
``batch["frames"][:, -1:]`` and never ``future_frames``. ``V3Dataset.newest_row_only`` (set by ``train()`` only under
``--r9-perc-rig`` + ``--u8-batches``) makes the loader build exactly the one row the rig reads.

CPU. The rig loss is checked on the tiny REAL ``RefCV3Model`` of ``test_r9_perc_rig`` (resnet18 @ 64x64, the 10 cm branch).
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
tvio = pytest.importorskip("torchvision.io")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.data import v2_dataset as V2  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402

S, NSTACK = 16, 3
WINDOW, HORIZON = 4, 3


# --------------------------------------------------------------------------- dataset ---------------------------------------
def _write_v2ep(path: Path, n_raw: int, eid: int, seed: int) -> None:
    """One synthetic PNG clip in the exact ``build_compressed`` payload format; pixel value = 7 * raw frame index."""
    bufs, lens = [], []
    for i in range(n_raw):
        b = tvio.encode_png(torch.full((3, S, S), fill_value=(i * 7) % 256, dtype=torch.uint8))
        bufs.append(b)
        lens.append(int(b.numel()))
    g = torch.Generator().manual_seed(seed)
    poses = torch.cumsum(torch.randn(n_raw, 4, generator=g) * 0.1, 0)
    poses[:, 3] = poses[:, 3].abs() + 3.0
    torch.save({"jpeg_buf": torch.cat(bufs), "jpeg_len": torch.tensor(lens, dtype=torch.int64),
                "actions": torch.randn(n_raw, 2, generator=g), "poses": poses, "n_stack": NSTACK, "image_size": S,
                "episode_id": eid, "clip_id": f"clip{eid:08d}", "quality": 90, "codec": "png"}, str(path))


def _lazy_dataset(tmp_path, n_clips=2, n_raw=30):
    for e in range(n_clips):
        _write_v2ep(tmp_path / f"clip{e:08d}.v2ep.pt", n_raw=n_raw, eid=e, seed=3 + e)
    eps = V2.build_v2_providers(tmp_path, lru_size=2, verbose=False)
    ds = T.V3Dataset(eps, window=WINDOW, max_horizon=HORIZON, channels=3 * NSTACK)
    ds.u8_frames = True
    return ds


def _item_keys_equal(a: dict, b: dict, skip=()):
    assert set(a) == set(b)
    for k in a:
        if k in skip:
            continue
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def test_1_newest_row_only_is_row_minus_1_of_the_full_window_and_every_other_key_is_untouched(tmp_path):
    ds = _lazy_dataset(tmp_path)
    full = ds[5]
    assert tuple(full["frames"].shape) == (WINDOW, 9, S, S) and tuple(full["future_frames"].shape) == (HORIZON, 9, S, S)
    ds.newest_row_only = True
    new = ds[5]
    assert tuple(new["frames"].shape) == (1, 9, S, S) and tuple(new["future_frames"].shape) == (0, 9, S, S)
    assert torch.equal(new["frames"][0], full["frames"][-1])                     # the rig's `[:, -1:]`, bit for bit
    _item_keys_equal(new, full, skip=("frames", "future_frames"))                # every label / pose / join key identical


def test_2_RED_flag_off_or_float_path_is_the_tips_full_window(tmp_path):
    ds = _lazy_dataset(tmp_path)
    assert ds.newest_row_only is False                                           # the class default: every other arm
    assert tuple(ds[3]["frames"].shape) == (WINDOW, 9, S, S) and tuple(ds[3]["future_frames"].shape) == (HORIZON, 9, S, S)
    from refb_train import FailLoudWindowDataset
    _item_keys_equal(ds._window_u8(3), FailLoudWindowDataset._window_u8(ds, 3))   # the override delegates, byte for byte
    ds.newest_row_only = True
    ds.u8_frames = False                                                         # the float path is the shared base contract
    assert tuple(ds[3]["frames"].shape) == (WINDOW, 9, S, S)                     # NOT overridden: the flag needs --u8-batches
    assert ds[3]["frames"].dtype == torch.float32


def test_3_the_loader_decodes_3_raw_pngs_per_sample_not_11(tmp_path, monkeypatch):
    """Counts the real PNG decodes of ONE ``ds[i]``. Full window: rows [t, t+4) need raw [t, t+6) = 6, future rows [t+4, t+7)
    need raw [t+4, t+9) = 5 -> 11. Newest row: row t+3 needs raw [t+3, t+6) = 3 (its n_stack frames)."""
    ds = _lazy_dataset(tmp_path)
    n = {"c": 0}
    real = V2.tvio.decode_png

    def counting(*a, **k):
        n["c"] += 1
        return real(*a, **k)
    monkeypatch.setattr(V2.tvio, "decode_png", counting)
    ds[2]
    assert n["c"] == 11
    ds.newest_row_only = True
    n["c"] = 0
    ds[2]
    assert n["c"] == 3


def test_4_a_worker_collated_batch_carries_one_row_and_no_future_frames(tmp_path):
    """End to end through a real DataLoader worker process (spawn on Windows / macOS, fork on Linux): the empty ``future_frames``
    must cross the shared-memory boundary and collate."""
    ds = _lazy_dataset(tmp_path)
    ds.newest_row_only = True
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=1, drop_last=True)
    b = next(iter(dl))
    assert tuple(b["frames"].shape) == (2, 1, 9, S, S) and b["frames"].dtype == torch.uint8
    assert tuple(b["future_frames"].shape) == (2, 0, 9, S, S)
    assert b["frames"].numel() * b["frames"].element_size() == 2 * 1 * 9 * S * S
    ref = ds[0]["frames"]
    assert torch.equal(b["frames"][0], ref)


# --------------------------------------------------------------------------- the loss ------------------------------------
def _rig(monkeypatch):
    orig = v3.RefCV3Model
    monkeypatch.setattr(v3, "RefCV3Model", lambda cfg: orig(cfg, perc_rig=True))
    return R._build()


def _loss(model, batch):
    torch.manual_seed(1234)
    return T.compute_losses_v3(model, batch, "cpu", log_metrics=False)


def test_5_the_rig_loss_is_bit_identical_on_the_newest_row_only_batch(monkeypatch):
    model, cfg = _rig(monkeypatch)
    full = R._batch(cfg)
    assert full["frames"].shape[1] == cfg.core.window and full["future_frames"].shape[1] == 20
    new = dict(full)
    new["frames"] = full["frames"][:, -1:].clone()
    new["future_frames"] = full["future_frames"][:, :0].clone()
    a, b = _loss(model, full), _loss(model, new)
    assert set(a) == set(b) and "map_hires" in a
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k


def test_6_RED_an_unused_row_changes_nothing_and_the_newest_row_does(monkeypatch):
    """The control that makes test 5 mean something: rows 0..W-2 and every future frame have NO influence on the rig loss,
    while the newest row has (the probe can detect a change)."""
    model, cfg = _rig(monkeypatch)
    full = R._batch(cfg)
    base = _loss(model, full)["map_hires"]
    junk = dict(full)
    junk["frames"] = full["frames"].clone()
    junk["frames"][:, :-1] = 255 - junk["frames"][:, :-1]
    junk["future_frames"] = 255 - full["future_frames"]
    assert torch.equal(_loss(model, junk)["map_hires"], base)                       # unused rows: exactly nothing
    hit = dict(full)
    hit["frames"] = full["frames"].clone()
    hit["frames"][:, -1] = 255 - hit["frames"][:, -1]
    assert not torch.equal(_loss(model, hit)["map_hires"], base)                   # the newest row: it moves


# --------------------------------------------------------------------------- reachability ------------------------------------
def _flag_sites(src: str) -> list:
    """The names X of every ``X.newest_row_only = <expr naming _r9rig_on and u8>`` in ``train()``."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute) \
                and node.targets[0].attr == "newest_row_only" and isinstance(node.targets[0].value, ast.Name):
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if {"_r9rig_on", "u8"} <= names:
                out.append(node.targets[0].value.id)
    return sorted(out)


def test_7_train_sets_the_flag_on_BOTH_the_train_and_the_eval_dataset_and_only_from_the_rig_flag():
    src = inspect.getsource(T.train)
    assert _flag_sites(src) == ["ds", "e_ds"]
    # RED: delete the train line (anchored: `e_ds.` must survive) -> the checker must see only the eval site; then none
    no_train = re.sub(r"(?m)^( *)ds\.newest_row_only = .*$", r"\1pass", src)
    assert _flag_sites(no_train) == ["e_ds"]
    none = re.sub(r"(?m)^( *)(e_)?ds\.newest_row_only = .*$", r"\1pass", src)
    assert _flag_sites(none) == []
    # the flag is bound to the rig flag, never to a bare True
    assert "newest_row_only = True" not in src


def test_8_the_config_stamp_estimates_the_bytes_the_loader_really_ships():
    src = inspect.getsource(T.train)
    assert "(1 if ds.newest_row_only else cfg.core.window + kw[\"max_horizon\"])" in src
