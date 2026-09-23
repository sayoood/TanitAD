"""⭐ `--trunk-dedup-frames`: each DISTINCT frame of overlapping D-015 stacks computed ONCE.

MEASURED 2026-09-23 (Thor, `--arm hier`): the refcv6 trunk makes 8 window rows x 3 frames = 24
backbone passes per sample, but a D-015 row stacks raw frames (j, j+1, j+2) oldest -> newest
(`tanitad/data/v2_dataset.py::_decode_stacked`), so row i+1 repeats row i's newest two frames and
a window holds only W + K - 1 = 10 distinct frames. With BatchNorm frozen, a frame's features
depend on that frame alone, so computing it once and gathering it into each slot is the same
function -- and the gather's backward SUMS each frame's gradient over its slots, which is what
the separate passes' weight gradients summed to.

What is pinned, each OBSERVED rather than inferred:
* slot for slot, the deduplicated backbone returns what the plain backbone returns, and the
  whole `forward_features` (fusion randomised, so EVERY stack position matters) agrees, with the
  same parameter gradients;
* it computes exactly B x (W + K - 1) frames -- a hook on the stem counts the rows it is fed --
  and reports the same pair in `last_dedup`;
* a batch with NO overlap computes every slot (the overlap is checked, never assumed), and ONE
  corrupted overlapping frame is not reused;
* refusals: without frozen BN, in "inflate" mode, with K = 1; the trainer flag needs
  --trunk-frozen-bn and --trunk timm;
* the flag reaches the BUILT trunk and the stamp, and a real `train()` on D-015-structured
  synthetic windows logs `trunk_frames_computed` < `trunk_frame_slots`.
CPU only; random-init resnet18, no download.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.models import timm_trunk as TT  # noqa: E402

K = 3


def _trunk(**kw):
    torch.manual_seed(0)
    t = TT.build_timm_trunk(in_channels=3 * K, image_hw=(64, 128),
                            model_name="resnet18.a1_in1k", pretrained=False, **kw)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():       # every stack position must matter, not only the newest
        for m in (t.fuse16, t.fuse32):
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return t


def _windows(b=2, w=4, seed=1):
    """[B*W, 3K, H, W] in the D-015 layout: row w = raw frames (w, w+1, w+2) along channels."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(b, w + K - 1, 3, 64, 128, generator=g)
    rows = torch.stack([torch.cat([raw[:, i + j] for j in range(K)], dim=1)
                        for i in range(w)], dim=1)          # [B, W, 3K, H, W]
    return rows.reshape(b * w, 3 * K, 64, 128)


def _count_stem_rows(t, fn):
    seen = []
    conv = next(m for m in t.net.modules() if isinstance(m, torch.nn.Conv2d))
    h = conv.register_forward_hook(lambda m, i, o: seen.append(int(i[0].shape[0])))
    try:
        out = fn()
    finally:
        h.remove()
    return sum(seen), out


def _rel(a, b):
    return float(((a - b).norm() / b.norm().clamp_min(1e-30)).detach())


def test_every_SLOT_gets_the_features_the_plain_backbone_computes():
    plain = _trunk(frozen_bn=True)
    dd = _trunk(frozen_bn=True, dedup_frames=True)
    x = plain.normalise(_windows())
    per = x.reshape(x.shape[0], K, 3, *x.shape[2:])
    with torch.no_grad():
        want = plain._backbone(per.reshape(-1, 3, *x.shape[2:]))
        got = dd._backbone_dedup(per)
    for a, b in zip(got, want):
        assert a.shape == b.shape
        assert _rel(a, b) < 1e-6


def test_forward_features_and_GRADIENTS_are_the_same_function():
    plain = _trunk(frozen_bn=True)
    dd = _trunk(frozen_bn=True, dedup_frames=True)
    x = _windows()
    oa, ob = plain.forward_features(x), dd.forward_features(x)
    for a, b in zip(ob, oa):
        assert _rel(a, b) < 1e-6
    sum(o.pow(2).mean() for o in oa).backward()
    sum(o.pow(2).mean() for o in ob).backward()
    ga = dict(plain.named_parameters())
    gb = dict(dd.named_parameters())
    n = 0
    for k, p in ga.items():
        if p.grad is None:
            continue
        n += 1
        assert _rel(gb[k].grad, p.grad) < 1e-5, k
    assert n > 20


def test_it_COMPUTES_only_the_distinct_frames_and_says_so():
    dd = _trunk(frozen_bn=True, dedup_frames=True)
    x = _windows(b=2, w=4)
    with torch.no_grad():
        n_rows, _ = _count_stem_rows(dd, lambda: dd.forward_features(x))
    assert n_rows == 2 * (4 + K - 1)                       # 12 frames, not 24
    assert dd.last_dedup == (2 * 4 * K, 12)
    plain = _trunk(frozen_bn=True)
    with torch.no_grad():
        n_plain, _ = _count_stem_rows(plain, lambda: plain.forward_features(x))
    assert n_plain == 2 * 4 * K
    assert plain.last_dedup is None


def test_NO_overlap_computes_every_slot_and_still_matches():
    g = torch.Generator().manual_seed(3)
    x = torch.rand(6, 3 * K, 64, 128, generator=g)            # independent stacks
    plain = _trunk(frozen_bn=True)
    dd = _trunk(frozen_bn=True, dedup_frames=True)
    with torch.no_grad():
        n_rows, ob = _count_stem_rows(dd, lambda: dd.forward_features(x))
        oa = plain.forward_features(x)
    assert n_rows == 6 * K and dd.last_dedup == (6 * K, 6 * K)
    for a, b in zip(ob, oa):
        assert _rel(a, b) < 1e-6


def test_ONE_corrupted_overlapping_frame_is_NOT_reused():
    x = _windows(b=1, w=4)
    x[2, 0:3] += 0.01          # row 2's OLDEST frame no longer equals row 1's middle frame
    plain = _trunk(frozen_bn=True)
    dd = _trunk(frozen_bn=True, dedup_frames=True)
    with torch.no_grad():
        oa = plain.forward_features(x)
        ob = dd.forward_features(x)
    assert dd.last_dedup == (4 * K, 3 + 1 + 3 + 1)            # row 2 recomputes all K
    for a, b in zip(ob, oa):
        assert _rel(a, b) < 1e-6


def test_the_lever_REFUSES_where_it_would_change_the_function():
    with pytest.raises(ValueError, match="without frozen_bn"):
        _trunk(dedup_frames=True)
    with pytest.raises(ValueError, match="mode 'shared'"):
        TT.build_timm_trunk(in_channels=9, image_hw=(64, 128), model_name="resnet18.a1_in1k",
                            pretrained=False, mode="inflate", frozen_bn=True,
                            dedup_frames=True)
    with pytest.raises(ValueError, match="mode 'shared'"):
        TT.build_timm_trunk(in_channels=3, image_hw=(64, 128), model_name="resnet18.a1_in1k",
                            pretrained=False, frozen_bn=True, dedup_frames=True)


def test_composes_with_chunked_checkpointing_bf16_nhwc_and_fold():
    plain = _trunk(frozen_bn=True, fold_bn=True, bf16=True, channels_last=True, chunk_ckpt=4)
    dd = _trunk(frozen_bn=True, fold_bn=True, bf16=True, channels_last=True, chunk_ckpt=4,
                dedup_frames=True)
    x = _windows()
    oa, ob = plain.forward_features(x), dd.forward_features(x)
    for a, b in zip(ob, oa):
        assert _rel(a, b) < 2e-2              # both bf16; the dedup itself adds nothing
    sum(o.pow(2).mean() for o in ob).backward()
    grads = [p.grad for p in dd.net.parameters() if p.requires_grad]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_trainer_flag_reaches_the_BUILT_trunk_is_stamped_and_refuses():
    import dataclasses as dc

    import refc_v3_train as T
    from tanitad.refs import refc
    from tanitad.refs import refc_v3 as v3
    P = T.build_parser()
    args = P.parse_args(["--arm", "hier", "--out", "x", "--trunk", "timm", "--trunk-name",
                         "resnet18.a1_in1k", "--trunk-in-channels", "9",
                         "--trunk-frozen-bn", "--trunk-dedup-frames"])
    cfg = v3.RefCV3Config(hier=True)
    T._pin_trainer_cfg(cfg, args)
    assert cfg.core.encoder.trunk_dedup_frames is True
    built = refc.build_encoder(dc.replace(cfg.core.encoder, trunk_pretrained=False))
    assert built.memory_levers["dedup_frames"] is True
    assert T._seam_stamp(cfg, args)["trunk_dedup_frames"] is True
    bad = P.parse_args(["--arm", "hier", "--out", "x", "--trunk", "timm",
                        "--trunk-dedup-frames"])
    with pytest.raises(SystemExit, match="without --trunk-frozen-bn"):
        T._pin_trainer_cfg(v3.RefCV3Config(hier=True), bad)
    refc_trunk = P.parse_args(["--arm", "hier", "--out", "x", "--trunk", "refc",
                               "--trunk-frozen-bn", "--trunk-dedup-frames"])
    with pytest.raises(SystemExit, match=r"--trunk-dedup-frames .*need --trunk timm"):
        T._pin_trainer_cfg(v3.RefCV3Config(hier=True), refc_trunk)


def test_a_real_train_LOGS_that_it_computed_fewer_frames_than_slots(tmp_path, monkeypatch):
    """The CI corpus draws each timestep's stack independently, so there is nothing to share;
    here its episodes are rebuilt in the D-015 layout (row t = raw frames t, t+1, t+2) and the
    REAL training loop must log fewer computed frames than slots -- and the same loss as the
    plain trunk on the same windows."""
    import refc_v3_train as T
    real = T._synth_episodes

    def _d015(n, cfg, seed=0, min_frames=40, clip_ids=None):
        eps = real(n, cfg, seed=seed, min_frames=min_frames, clip_ids=clip_ids)
        for ep in eps:
            raw = ep.frames[:, 0:3]                               # [T, 3, h, w] u8
            t_n = raw.shape[0] - (K - 1)
            ep.frames = torch.stack([torch.cat([raw[t + j] for j in range(K)], dim=0)
                                     for t in range(t_n)])
            ep.poses, ep.actions = ep.poses[:t_n], ep.actions[:t_n]
        return eps

    monkeypatch.setattr(T, "_synth_episodes", _d015)
    base = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "2",
            "--batch", "2", "--device", "cpu", "--log-every", "1", "--save-every", "100",
            "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k", "--no-trunk-pretrained",
            "--trunk-in-channels", "9", "--trunk-frozen-bn"]
    T.train(T.build_parser().parse_args(base + ["--out", str(tmp_path / "d"),
                                                "--trunk-dedup-frames"]))
    T.train(T.build_parser().parse_args(base + ["--out", str(tmp_path / "p")]))

    def rows(d):
        out = []
        for ln in (tmp_path / d / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(ln) if ln.strip() else {}
            if isinstance(r.get("step"), int) and "loss" in r:
                out.append(r)
        return out

    rd, rp = rows("d"), rows("p")
    assert rd and all(r["trunk_frames_computed"] < r["trunk_frame_slots"] for r in rd)
    # counted PER ROW: every step makes the same calls, so an unreset counter would grow
    assert len({(r["trunk_frame_slots"], r["trunk_frames_computed"]) for r in rd}) == 1
    assert all("trunk_frames_computed" not in r for r in rp)
    for a, b in zip(rd, rp):
        assert abs(a["loss"] - b["loss"]) <= 1e-4 * abs(b["loss"]), (a["loss"], b["loss"])
    cfg = json.loads((tmp_path / "d" / "config.json").read_text(encoding="utf-8"))
    assert cfg["seams"]["trunk_dedup_frames"] is True
    assert cfg["trunk_memory_levers"]["dedup_frames"] is True
