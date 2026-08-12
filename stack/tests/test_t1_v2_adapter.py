"""E1.4 — the T1 adapter that makes ``taniteval/tools/t1_eval.py`` runnable on
the v5f / v5.8f stack (v2 compressed caches + the checkpoint's OWN
``grounding.step['op']`` decoder). CPU-only: no GPU, no checkpoint, no corpus.

WHAT IS PINNED, and why each group exists:

  (1) THE LEGACY PATH IS UNTOUCHED. ``--corpus`` + ``--head`` still routes to
      :func:`t1_eval.run_rollout` — the roll the §1.12 byte-close gate
      validated. Pinned three ways: the dispatch predicate never selects the
      adapter for legacy args; the legacy roll functions carry no adapter
      token; and a SOURCE HASH over ``run_rollout`` + ``roll_closed`` +
      ``decode_open`` fails loudly if any of them is edited.
  (2) THE ADAPTER CONTRACT. A v2-style episode -> ``V2RawEp`` yields exactly the
      ``(feats, poses, actions)`` shapes/dtypes the roll consumes, and the
      slicing the roll performs works on the lazy frames proxy. With
      torchvision present the REAL ``build_v2_providers`` path is exercised
      end to end on a synthetic ``*.v2ep.pt``.
  (3) THE DECODE CALL CONVENTION. ``decode_open_grounding`` is
      ``metric_dynamics.decode_transitions`` — the IMPORTED function, compared
      against it and against ``rollout_decode`` (the canary), never a copy.
      ``w7_roll_rerank`` is checked to import the same two symbols from the
      same module.
  (4) THE ONE PIECE OF NEW MATH. ``implied_controls`` against hand numbers
      (hypot speed, finite-difference accel, the 0.3 m/s curvature clamp, the
      atan(2.9*kappa) corpus steer encoding), and ``roll_closed_grounding``
      proved to actually CLOSE the loop: the action the predictor sees at step
      j+1 is the control implied by the Δpose decoded at step j.
  (5) THE ROLL END TO END on a stub trunk: a real dump in the documented npz
      schema, consumed by the untouched ``analyze`` with T0/T1 tier stamps.
  (6) THE CLI REFUSALS: two corpus formats, or two decoders, are a refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[2]
TOOL = _REPO / "taniteval" / "tools" / "t1_eval.py"
SUMMARY_TOOL = _REPO / "taniteval" / "tools" / "t1_summary.py"
sys.path.insert(0, str(_REPO / "stack" / "scripts"))

_spec = importlib.util.spec_from_file_location("t1_eval_adapter_test", TOOL)
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)

from tanitad.models.metric_dynamics import (  # noqa: E402
    HierarchicalGrounding, StepDisplacementReadout, accumulate_se2,
    decode_transitions, rollout_decode, rollout_transitions)

DT, WB = t1.DT, t1.WHEELBASE
S, T_EP, W, K = 8, 20, 4, 3
C, H, WD = 3, 6, 8


# =========================================================================== #
# (1) the legacy §1.12 path is untouched                                      #
# =========================================================================== #
#: md5 of the three functions that make up the byte-close-validated roll.
#: ⛔ IF THIS TEST FAILS you edited the §1.12 roll. That is allowed ONLY with a
#: re-run of the pod-side byte-close gate (``--analyze-only`` over the original
#: ``dump_cl`` reproducing ``closed_loop_analysis.json``); update the constant
#: in the SAME change and say so in the commit. It is not a formatting nit.
LEGACY_ROLL_MD5 = "326b40f747c2c1c663f7e52a444b05df"


def _legacy_source() -> str:
    return "".join(inspect.getsource(f) for f in
                   (t1.decode_open, t1.roll_closed, t1.run_rollout))


def test_legacy_roll_source_is_byte_identical():
    got = hashlib.md5(_legacy_source().encode("utf-8")).hexdigest()
    assert got == LEGACY_ROLL_MD5, (
        "decode_open / roll_closed / run_rollout changed. These three ARE the "
        "§1.12 byte-close path (E1.4 validated it against dump_cl). Re-run the "
        "pod-side byte-close gate and update LEGACY_ROLL_MD5 in the same "
        f"change; do not just paste {got}.")


def test_legacy_roll_carries_no_adapter_token():
    """The adapter must not have leaked into the validated functions."""
    src = _legacy_source()
    for tok in ("v2_val_cache", "grounding_readout", "window_stride",
                "run_rollout_ext", "implied_controls", "V2RawEp"):
        assert tok not in src, tok
    assert "load_frames([f])[0]" in src            # its own episode source
    assert "UnicycleStepReadout(" in src           # its own decoder
    assert "list(range(0, T - w - k))" in src      # the stride-1 §1.12 grid


def test_public_signatures_unchanged():
    assert str(inspect.signature(t1.run_rollout)) == "(a)"
    assert str(inspect.signature(t1.roll_closed)) == (
        "(model, head, states, awE, v0, ego, k=20, dt=0.1, wheelbase=2.9)")
    assert str(inspect.signature(t1.decode_open)) == "(head, trans, v0, dt=0.1)"
    assert "byte_check" in inspect.signature(t1.analyze).parameters


@pytest.mark.parametrize("kw,want", [
    ({"corpus": "/x", "head": "/h"}, False),                   # the legacy path
    ({"corpus": "/x", "grounding_readout": True}, True),
    ({"v2_val_cache": ["/x"], "head": "/h"}, True),
    ({"v2_val_cache": ["/x"], "grounding_readout": True}, True),
])
def test_dispatch_predicate_never_diverts_the_legacy_combination(kw, want):
    a = argparse.Namespace(corpus=None, v2_val_cache=None, head=None,
                           grounding_readout=False, **{})
    for k, v in kw.items():
        setattr(a, k, v)
    assert t1.uses_ext_rollout(a) is want


def test_window_stride_default_is_the_1_12_grid():
    """A stride knob that defaulted to anything but 1 would silently re-grid
    every future T1 run."""
    src = TOOL.read_text(encoding="utf-8")
    i = src.index('"--window-stride"')
    assert "default=1" in src[i:i + 200]


# =========================================================================== #
# (2) the adapter contract: (feats, poses, actions)                           #
# =========================================================================== #
class _FramesProxy:
    """Minimal stand-in for ``v2_dataset._V2FramesProxy`` (dim-0 slice ->
    decoded uint8), so the contract test does not need torchvision."""

    def __init__(self, frames):
        self._f = frames

    @property
    def shape(self):
        return self._f.shape

    @property
    def dtype(self):
        return torch.uint8

    def __len__(self):
        return int(self._f.shape[0])

    def __getitem__(self, idx):
        return self._f[idx].clone()


class _FakeProvider:
    """``LazyV2Episode``-shaped: .frames / .poses / .actions / .episode_id."""

    def __init__(self, t=T_EP, eid=1234567):
        g = torch.Generator().manual_seed(0)
        self.frames = _FramesProxy(
            (torch.rand(t, C, H, WD, generator=g) * 255).to(torch.uint8))
        self.poses = torch.randn(t, 4, generator=g)
        self.poses[:, 3] = self.poses[:, 3].abs() * 5.0
        self.actions = torch.randn(t, 2, generator=g)
        self.episode_id = eid


def test_v2rawep_yields_the_shapes_and_dtypes_the_roll_consumes():
    p = _FakeProvider()
    ep = t1.V2RawEp(p, 3)
    assert ep.feats.shape == (T_EP, C, H, WD) and ep.feats.dtype == torch.uint8
    assert ep.poses.shape == (T_EP, 4) and ep.poses.dtype == torch.float32
    assert ep.actions.shape == (T_EP, 2) and ep.actions.dtype == torch.float32
    assert ep.episode_id == 1234567 and ep.clip_index == 3
    # the exact operations run_rollout/run_rollout_ext perform on an episode
    s, w, k = 2, W, K
    fw = torch.stack([torch.as_tensor(ep.feats[s:s + w])]).float().div_(255.0)
    assert fw.shape == (1, w, C, H, WD) and float(fw.max()) <= 1.0
    aw = torch.stack([ep.actions[s:s + w]]).float()
    assert aw.shape == (1, w, 2)
    fa = torch.stack([ep.actions[s + w:s + w + k]]).float()
    assert fa.shape == (1, k, 2)
    last = torch.tensor([s + w - 1])
    assert ep.poses[last].shape == (1, 4)
    fp = torch.stack([ep.poses[s + w:s + w + k]])
    assert fp.shape == (1, k, 4)
    T = min(int(ep.feats.shape[0]), int(ep.poses.shape[0]),
            int(ep.actions.shape[0]))
    assert T == T_EP


def test_v2rawep_mirrors_taniteval_rawep_field_for_field():
    """The adapter is a RENAME of the provider surface onto ``RawEp``'s — if
    ``RawEp`` grows a field the roll reads, this catches the divergence."""
    from taniteval.data import RawEp
    raw_fields = {"feats", "actions", "poses", "episode_id"}
    assert raw_fields <= set(t1.V2RawEp.__slots__)
    src = inspect.getsource(RawEp)
    assert "self.feats = ep.frames" in src        # the field the rename maps


def test_real_v2_provider_path(tmp_path):
    """The REAL ``build_v2_providers`` -> ``V2RawEp`` path on a synthetic
    ``*.v2ep.pt`` (needs torchvision, as the v2 loader decodes JPEG/PNG)."""
    tvio = pytest.importorskip("torchvision.io")
    from tanitad.data.v2_dataset import build_v2_providers, decode_full_episode
    n_raw, n_stack, size = 12, 3, 16
    g = torch.Generator().manual_seed(7)
    vid = (torch.rand(n_raw, 3, size, size, generator=g) * 255).to(torch.uint8)
    jpegs = [tvio.encode_jpeg(vid[i].contiguous(), quality=95)
             for i in range(n_raw)]
    poses = torch.randn(n_raw, 4, generator=g)
    poses[:, 3] = poses[:, 3].abs() * 5.0
    torch.save({"jpeg_buf": torch.cat(jpegs),
                "jpeg_len": torch.tensor([int(j.numel()) for j in jpegs],
                                         dtype=torch.int64),
                "actions": torch.randn(n_raw, 2, generator=g), "poses": poses,
                "n_stack": n_stack, "image_size": size, "episode_id": 42,
                "clip_id": "clip-e1p4", "quality": 95},
               str(tmp_path / "clip-e1p4.v2ep.pt"))
    provs = build_v2_providers([str(tmp_path)], lru_size=2, verbose=False)
    assert len(provs) == 1
    ep = t1.V2RawEp(provs[0], 0)
    t_out = n_raw - (n_stack - 1)
    assert tuple(ep.feats.shape) == (t_out, 3 * n_stack, size, size)
    assert ep.poses.shape == (t_out, 4) and ep.actions.shape == (t_out, 2)
    assert ep.episode_id != 0
    ref = decode_full_episode(str(tmp_path / "clip-e1p4.v2ep.pt"))
    sl = torch.as_tensor(ep.feats[1:4])
    assert sl.dtype == torch.uint8
    assert torch.equal(sl, ref.frames[1:4])           # partial == full decode
    assert torch.allclose(ep.poses, ref.poses)
    assert torch.allclose(ep.actions, ref.actions)


# =========================================================================== #
# (3) the decode call convention == w7_roll_rerank's                          #
# =========================================================================== #
class StubPredictor(nn.Module):
    """z_hat depends on the LAST action, so a changed feedback is observable."""

    def __init__(self, s=S):
        super().__init__()
        self.lin = nn.Linear(s + 3, s)
        self.seen: list[torch.Tensor] = []

    def forward(self, win_s, win_a):
        self.seen.append(win_a[:, -1].detach().clone())
        z = self.lin(torch.cat([win_s[:, -1], win_a[:, -1]], dim=-1))
        return (z, z)


def _roll_inputs(b=2, s=S, w=W, k=K, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = torch.randn(b, w, s, generator=g)
    aw = torch.randn(b, w, 3, generator=g)
    fa = torch.randn(b, k, 3, generator=g)
    return states, aw, fa


def test_decode_open_grounding_is_the_imported_decode_transitions():
    torch.manual_seed(0)
    sr = StepDisplacementReadout(S, hidden=16).eval()
    pred = StubPredictor().eval()
    states, aw, fa = _roll_inputs()
    with torch.no_grad():
        trans = rollout_transitions(pred, states, aw, fa, K)
        mine = t1.decode_open_grounding(sr, trans, K)
        theirs = decode_transitions(sr, trans, K)[0]
    assert torch.equal(mine, theirs)                       # the same function
    assert mine.shape == (2, K, 2)


def test_grounding_t0_arm_reproduces_the_canary_rollout_decode():
    """The adapter's T0 arm = ``decode_transitions(rollout_transitions(...))``,
    which ``metric_dynamics`` pins as reproducing ``rollout_decode`` — the
    canary's decode (train_flagship_v4.py:584-586). Compared here against the
    imported ``rollout_decode`` itself, not a restatement of it."""
    torch.manual_seed(1)
    sr = StepDisplacementReadout(S, hidden=16).eval()
    pred = StubPredictor().eval()
    states, aw, fa = _roll_inputs(seed=3)
    with torch.no_grad():
        mine = t1.decode_open_grounding(
            sr, rollout_transitions(pred, states, aw, fa, K), K)
        canary, _ = rollout_decode(pred, states, aw, fa, sr, K)
    assert torch.equal(mine, canary)


def test_w7_uses_the_same_two_symbols_from_the_same_module():
    src = (_REPO / "stack" / "scripts" / "w7_roll_rerank.py").read_text(
        encoding="utf-8")
    assert ("from tanitad.models.metric_dynamics import (decode_transitions,\n"
            in src)
    assert 'step_readout = grounding.step["op"]' in src
    assert "decode_transitions(step_readout, trans, roll_k)" in src
    # and the adapter reaches the same readout on the same object
    ada = inspect.getsource(t1.run_rollout_ext)
    assert 'grounding.step["op"]' in ada


def test_hold_action_arm_equals_the_zero_order_hold_roll():
    """``faH = aw[:, -1:].expand(k)`` is byte-identical to
    ``rollout_transitions(..., None, k)``'s hold branch — the same identity
    ``stage_a_probes`` pins for its HOLD channel."""
    torch.manual_seed(2)
    sr = StepDisplacementReadout(S, hidden=16).eval()
    pred = StubPredictor().eval()
    states, aw, _ = _roll_inputs(seed=5)
    with torch.no_grad():
        faH = aw[:, -1:, :].expand(-1, K, -1)
        a = t1.decode_open_grounding(
            sr, rollout_transitions(pred, states, aw, faH, K), K)
        b = t1.decode_open_grounding(
            sr, rollout_transitions(pred, states, aw, None, K), K)
    assert torch.equal(a, b)


# =========================================================================== #
# (4) the new math: implied_controls + a genuinely closed loop                #
# =========================================================================== #
def test_implied_controls_hand_cases():
    # straight, 0.5 m in dt=0.1 -> v = 5 m/s, no accel (v_prev 5), no yaw
    d = torch.tensor([[0.5, 0.0, 0.0]])
    v, acc, yr, steer = t1.implied_controls(d, torch.tensor([5.0]))
    assert float(v) == pytest.approx(5.0)
    assert float(acc) == pytest.approx(0.0, abs=1e-5)
    assert float(yr) == pytest.approx(0.0) and float(steer) == pytest.approx(0.)
    # SPEED IS THE HYPOT, not dx: (0.3, 0.4) -> 0.5 m -> 5 m/s
    v2, _, _, _ = t1.implied_controls(torch.tensor([[0.3, 0.4, 0.0]]),
                                      torch.tensor([5.0]))
    assert float(v2) == pytest.approx(5.0)
    # accel is the finite difference against the CARRIED speed
    _, acc2, _, _ = t1.implied_controls(torch.tensor([[0.5, 0.0, 0.0]]),
                                        torch.tensor([4.0]))
    assert float(acc2) == pytest.approx(10.0)          # (5 - 4) / 0.1
    # yaw rate -> curvature -> the corpus steer encoding atan(2.9 * kappa)
    _, _, yr3, steer3 = t1.implied_controls(torch.tensor([[0.5, 0.0, 0.02]]),
                                            torch.tensor([5.0]))
    assert float(yr3) == pytest.approx(0.2)
    assert float(steer3) == pytest.approx(float(np.arctan(WB * (0.2 / 5.0))))


def test_implied_controls_uses_the_same_low_speed_clamp_as_roll_closed():
    """``kappa = yaw_rate / max(v, 0.3)`` — the clamp roll_closed already uses.
    At standstill the curvature must be bounded, not infinite."""
    assert "clamp_min(0.3)" in inspect.getsource(t1.roll_closed)
    assert "clamp_min(0.3)" in inspect.getsource(t1.implied_controls)
    _, _, _, steer = t1.implied_controls(torch.tensor([[0.0, 0.0, 0.02]]),
                                         torch.tensor([0.0]))
    assert float(steer) == pytest.approx(float(np.arctan(WB * (0.2 / 0.3))))
    assert np.isfinite(float(steer))


def test_implied_controls_matches_the_steer_encoding_stage_a_uses():
    from stage_a_probes import WHEELBASE as SA_WB, steer_of_kappa
    assert SA_WB == WB
    yr, v = 0.2, 5.0
    _, _, _, steer = t1.implied_controls(torch.tensor([[0.5, 0.0, yr * DT]]),
                                         torch.tensor([v]))
    assert float(steer) == pytest.approx(
        float(steer_of_kappa(torch.tensor(yr / v))), abs=1e-6)


class ConstReadout(nn.Module):
    """Emits a fixed Δpose per step, so the fed-back action is hand-computable."""

    def __init__(self, dpose):
        super().__init__()
        self.register_buffer("d", torch.as_tensor(dpose, dtype=torch.float32))

    def forward(self, z_t, z_next):
        return self.d.expand(z_t.shape[0], 3).clone()


def test_roll_closed_grounding_feeds_back_the_implied_action():
    """THE property that makes this T1: at step j+1 the predictor sees the
    action implied by the Δpose the readout emitted at step j — not a recorded
    action, not the previous one."""
    torch.manual_seed(3)
    pred = StubPredictor().eval()
    model = argparse.Namespace(predictor=pred)
    dpose = [0.5, 0.0, 0.02]
    sr = ConstReadout(dpose).eval()
    b, k = 2, 4
    states = torch.randn(b, W, S)
    v0 = torch.full((b,), 5.0)
    ego = (v0 / 10.0)[:, None]
    awE = torch.randn(b, W, 3)
    with torch.no_grad():
        wp = t1.roll_closed_grounding(model, sr, states, awE, v0, ego, k=k)
    assert wp.shape == (b, k, 2)
    # the trajectory is accumulate_se2 of the emitted Δposes — the same
    # geometry decode_transitions uses, so nothing downstream can tell them apart
    want = accumulate_se2(torch.tensor(dpose).view(1, 1, 3)
                          .expand(b, k, 3).contiguous())
    assert torch.allclose(wp, want, atol=1e-5)
    # step 0 saw the RECORDED last window action; steps 1.. see the implied one
    assert torch.allclose(pred.seen[0], awE[:, -1])
    v, acc, _yr, steer = t1.implied_controls(
        torch.tensor(dpose).view(1, 3).expand(b, 3), v0)
    for j in (1, 2, 3):
        got = pred.seen[j]
        assert torch.allclose(got[:, 0], steer, atol=1e-5), j
        assert torch.allclose(got[:, 2], ego[:, 0], atol=1e-6), j   # v0 held
    # after step 0 the speed is constant at 5 m/s -> accel 0 from step 2 on
    assert torch.allclose(pred.seen[1][:, 1], acc, atol=1e-5)
    assert torch.allclose(pred.seen[2][:, 1], torch.zeros(b), atol=1e-5)


def test_roll_closed_grounding_is_not_teacher_forced():
    """Changing the RECORDED future actions cannot move a T1 roll (it never
    reads them); changing the decoder's emission must."""
    torch.manual_seed(4)
    pred = StubPredictor().eval()
    model = argparse.Namespace(predictor=pred)
    states = torch.randn(1, W, S)
    v0 = torch.tensor([5.0])
    ego = (v0 / 10.0)[:, None]
    awE = torch.randn(1, W, 3)
    with torch.no_grad():
        a = t1.roll_closed_grounding(model, ConstReadout([0.5, 0.0, 0.02]),
                                     states, awE, v0, ego, k=4)
        b = t1.roll_closed_grounding(model, ConstReadout([0.5, 0.0, -0.02]),
                                     states, awE, v0, ego, k=4)
    assert not torch.allclose(a, b)          # the loop is genuinely closed


# =========================================================================== #
# (5) the roll end to end on a stub trunk -> a real dump -> analyze           #
# =========================================================================== #
class StubWorld(nn.Module):
    def __init__(self, s=S):
        super().__init__()
        self.predictor = StubPredictor(s)
        self.enc = nn.Linear(C * H * WD, s)
        self.state_dim = s

    def encode_window(self, fw):
        b, w = fw.shape[:2]
        return self.enc(fw.reshape(b, w, -1))


class _Frame:
    height, width = H, WD

    def to_dict(self):
        return {"height": H, "width": WD}


def test_run_rollout_ext_writes_the_documented_dump_and_analyze_reads_it(
        tmp_path, monkeypatch):
    torch.manual_seed(5)
    world = StubWorld().eval()
    grounding = HierarchicalGrounding(S, hidden=16).eval()
    eps = [_FakeProvider(eid=100 + i) for i in range(2)]

    monkeypatch.setattr(t1, "resolve_ext_frames",
                        lambda a: (None, _Frame(), _Frame()))
    monkeypatch.setattr(t1, "load_ext_trunk",
                        lambda a, frame: (world, grounding, 30000))
    monkeypatch.setattr(
        t1, "ext_episode_sources",
        lambda a, **kw: ([(lambda p=p, i=i: t1.V2RawEp(p, i))
                          for i, p in enumerate(eps)],
                         {"corpus_format": "v2-compressed", "dirs": ["/fake"],
                          "n_episodes": len(eps), "n_available": len(eps)}))

    dump = tmp_path / "dump"
    a = argparse.Namespace(
        ckpt="/fake/ckpt.pt", device="cpu", grounding_readout=True, head=None,
        head_hidden=512, head_state_dim=S, head_speed_input=False,
        head_predict_delta=False, no_amp=True, window=W, horizon_k=K, dt=DT,
        window_stride=1, chunk=4, wheelbase=WB, with_t0_open_loop=True,
        with_hold_action=True, dump_dir=str(dump), episodes=0,
        v2_val_cache=["/fake"], corpus=None, v2_subframe="176x624")
    prov = t1.run_rollout_ext(a)

    assert prov["grid"]["n_windows"] == 2 * len(range(0, T_EP - W - K))
    assert prov["decoder"]["kind"].startswith("grounding.step")
    assert prov["arms"] == ["cl", "ol", "ha"]
    assert prov["canary_equivalent_arm"]           # the cross-check is stated
    files = sorted(str(f) for f in dump.glob("ep*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as d:
        assert set(d.files) == {"g", "cl", "ol", "ha", "ws"}
        n = d["g"].shape[0]
        assert d["g"].shape == (n, K, 2) and d["g"].dtype == np.float32
        for arm in ("cl", "ol", "ha"):
            assert d[arm].shape == (n, K, 2)
        assert d["ws"].shape == (n,)
        # T1 and T0 are different rolls, not the same numbers twice
        assert not np.allclose(d["cl"], d["ol"])

    res = t1.analyze(files, n_boot=25, dt=DT)
    assert res["arms"]["cl"]["tier"] == "T1"
    assert res["arms"]["ol"]["tier"] == "T0"
    assert res["arms"]["ha"]["tier"] == "T1"
    assert res["n_episodes"] == 2 and res["n_windows"] == prov["grid"][
        "n_windows"]
    dec = res["paired_decision_grade"]["paired_closed_minus_open"]
    assert dec["estimator"] == "paired_episode_cluster_bootstrap"
    # STRATEGIC stays UNAVAILABLE — PhysicalAI-AV carries no map, lane graph or
    # route signal, so no rescore can close it (the VLM PH2 pipeline is the
    # programme's answer). TACTICAL is now derived from the driven trajectory,
    # which at T1 nothing but the arm's own actions produced.
    assert res["arms"]["cl"]["four_families"]["strategic"]["status"] == \
        "UNAVAILABLE"
    assert res["arms"]["cl"]["four_families"]["tactical"]["status"] == "OK"


def test_run_rollout_ext_head_mode_calls_the_legacy_roll_closed(monkeypatch,
                                                                tmp_path):
    """``--v2-val-cache`` with a ``--head`` ckpt must reuse the VALIDATED
    ``roll_closed``/``decode_open``, not a second unicycle implementation."""
    src = inspect.getsource(t1.run_rollout_ext)
    assert "roll_closed(model, head, states, awE, v0, ego," in src
    assert "decode_open(head, tr, v0, dt=dt)" in src
    assert "decode_open(head, trH, v0, dt=dt)" in src


# =========================================================================== #
# (6) the CLI: refusals, and the byte-check mode still reachable              #
# =========================================================================== #
def _cli(*args, timeout=300):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, timeout=timeout)


def test_cli_refuses_two_corpus_formats(tmp_path):
    r = _cli("--arm", "x", "--out", str(tmp_path / "o.json"), "--ckpt", "/c",
             "--dump-dir", str(tmp_path / "d"), "--corpus", "/raw",
             "--v2-val-cache", "/v2", "--grounding-readout")
    assert r.returncode != 0
    assert "two CORPUS FORMATS" in (r.stdout + r.stderr)


def test_cli_refuses_two_decoders(tmp_path):
    r = _cli("--arm", "x", "--out", str(tmp_path / "o.json"), "--ckpt", "/c",
             "--dump-dir", str(tmp_path / "d"), "--v2-val-cache", "/v2",
             "--head", "/h", "--grounding-readout")
    assert r.returncode != 0
    assert "two DECODERS" in (r.stdout + r.stderr)


def test_cli_still_requires_a_corpus_and_a_decoder(tmp_path):
    r = _cli("--arm", "x", "--out", str(tmp_path / "o.json"), "--ckpt", "/c",
             "--dump-dir", str(tmp_path / "d"))
    assert r.returncode != 0
    assert "--corpus or --v2-val-cache" in (r.stdout + r.stderr)


def test_cli_help_lists_both_paths():
    r = _cli("--help")
    assert r.returncode == 0, r.stderr[-800:]
    for flag in ("--corpus", "--head", "--v2-val-cache", "--grounding-readout",
                 "--v2-subframe", "--frame-hfov", "--projection",
                 "--byte-check-dump", "--window-stride"):
        assert flag in r.stdout, flag


def test_byte_check_dump_mode_still_works(tmp_path):
    """E1.4 validated the legacy path THROUGH this mode; it must stay live."""
    p = tmp_path / "dump"
    p.mkdir()
    ref = tmp_path / "ref"
    ref.mkdir()
    g = np.random.default_rng(0).normal(size=(6, 20, 2))
    cl = g + 0.25
    np.savez(p / "ep000.npz", g=g, cl=cl, ws=np.arange(6))
    np.savez(ref / "ep000.npz", b=g)
    out = tmp_path / "t1.json"
    r = _cli("--arm", "bc", "--analyze-only", str(p), "--out", str(out),
             "--n-boot", "25", "--byte-check-dump", str(ref),
             "--byte-check-arm", "cl", "--byte-check-key", "b")
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    rec = json.loads(out.read_text())
    assert rec["byte_check"]["max_abs"] == pytest.approx(0.25, abs=1e-6)
    # ... and the record says honestly that NO roll ran in this invocation
    assert "analyze-only" in rec["rollout_path"]
    assert rec["rollout_provenance"] is None


# =========================================================================== #
# t1_summary: the cross-arm paired join, and its refusal                      #
# =========================================================================== #
def _mini_dump(dirp: Path, shift: float, n_ep=2, n_w=5, seed=0):
    dirp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for e in range(n_ep):
        g = np.cumsum(np.tile(np.array([[0.5, 0.0]]), (n_w, 20, 1)), axis=1)
        cl = g + np.array([shift, 0.0]) + rng.normal(scale=1e-3, size=g.shape)
        np.savez(dirp / f"ep{e:03d}.npz", g=g, cl=cl, ws=np.arange(n_w))
    return sorted(str(f) for f in dirp.glob("ep*.npz"))


def _summary_module():
    spec = importlib.util.spec_from_file_location("t1_summary_test",
                                                  SUMMARY_TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_t1_summary_pairs_two_arms_on_a_matched_grid(tmp_path):
    ts = _summary_module()
    recs, paths = {}, {}
    for name, shift in (("A", 0.10), ("B", 0.30)):
        d = tmp_path / name
        files = _mini_dump(d, shift)
        rec = t1.analyze(files, n_boot=25, dt=DT)
        rec["dump_dir"] = str(d)
        rec["ckpt"] = f"/fake/{name}.pt"
        recs[name] = rec
        paths[name] = f"/fake/{name}.json"
    out = ts.build_summary(recs, paths, pairs=[("B", "A")], n_boot=50, dt=DT)
    row = out["cross_arm_paired"]["B_minus_A"]["cl"]
    assert row["ade_dense_m"]["estimator"] == "paired_episode_cluster_bootstrap"
    # B is 0.2 m further from GT than A on every window
    assert row["ade_dense_m"]["delta"] == pytest.approx(0.2, abs=0.01)
    assert row["ade_dense_m"]["separated"] is True
    assert row["n_episodes"] == 2 and row["n_windows"] == 10
    assert out["arms"]["A"]["cl"]["tier"] == "T1"
    assert "PRIMARY" in out["_tier_doctrine"]
    assert "overlapping_holdout_se is used nowhere" in out["_estimator"]


def test_t1_summary_refuses_a_cross_grid_join(tmp_path):
    """Different grids must REFUSE, not truncate — a positional join across
    grids scores one arm against another's traffic."""
    ts = _summary_module()
    recs, paths = {}, {}
    for name, n_w in (("A", 5), ("B", 4)):
        d = tmp_path / name
        files = _mini_dump(d, 0.1, n_w=n_w)
        rec = t1.analyze(files, n_boot=25, dt=DT)
        rec["dump_dir"] = str(d)
        recs[name] = rec
        paths[name] = f"/fake/{name}.json"
    out = ts.build_summary(recs, paths, pairs=[("B", "A")], n_boot=25, dt=DT)
    row = out["cross_arm_paired"]["B_minus_A"]["cl"]
    assert row["status"] == "REFUSED" and "same grid" in row["reason"]
