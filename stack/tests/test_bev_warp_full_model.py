"""refcv9 T1-FULL -- the temporal BEV warp (``--r9-bev-warp``) REACHES the FULL model.  CPU, the tiny full model of ``test_map_only_ft._build()``.

THE CLASS ("built, tested, and UNREACHABLE FROM ITS CALLER", CLAUDE.md 2026-09-10/11).  Before this change T1 ran in ONE place: ``bev_warp_flag.pin_args``
REFUSED ``--r9-bev-warp`` without ``--r9-perc-rig`` and the only ``_bw.stage(model, batch, device)`` call sat in ``compute_losses_perc_rig``.  A T1 winner in E1
could not be carried into refcv9-full: ``compute_losses_v3`` never staged the history, so the hook on ``model._map_hires.lift`` would have refused every
forward.  This file proves the path on the FULL model (planner + tactical + perception + 10 cm map; NO rig), hop by hop:

  HOPS        argv -> ``args.r9_bev_warp`` -> ``pin_args`` (no rig needed) -> ``attach`` -> ``attach_dataset`` (train + eval) -> V3Dataset item (``bw_*``) -> default
              collate -> ``compute_losses_v3`` -> ``_bw.stage`` (BEFORE the one model forward) -> the lift hook -> the FUSED BEV -> the loss -> ``.backward()``
  LIVE        gate 0: the loss equals the OFF loss EXACTLY; a non-zero gate moves it ONLY through the fuse (fuse weight 0 or no history => equals OFF);
              the gate AND the fuse weight get a NON-ZERO gradient from ``loss``; CONTROLS read EXACTLY 0.0 (fuse at gate 0; both with no history)
  MUTATION    the ``_bw.stage`` line deleted from ``compute_losses_v3`` (source-level, in-process, no file touched) => the reach check goes RED
  REFUSAL     a batch without ``bw_*`` while built REFUSES; a forward with nothing staged REFUSES (no silent un-warped forward); ``--ablate-frames`` REFUSES
  OFF         flag off => loss dict and EVERY parameter gradient bit-identical (torch.equal) to ``compute_losses_v3`` without the two T1 statements
  ADMISSIBLE  perturbing every batch key that is not (frames, map_ep, bw_*) -- future poses, labels, nav, goals -- changes the fused BEV by EXACTLY 0.0; RED arm
  ENTRY POINTS every eval / inference entry point that builds the model through the loader is classified: it stages through ``compute_losses_v3``, stages itself,
              or cannot hold a T1 model; a NEW one fails the census
"""
from __future__ import annotations

import argparse
import copy
import inspect
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402
from tanitad.train import bev_warp_flag as BF  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402

TRAINER_SRC = (ROOT / "scripts" / "refc_v3_train.py").read_bytes().decode("utf-8").replace("\r\n", "\n")


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """CPU reduction order is thread-count dependent; bit-identity is asserted, so pin ONE thread for this module."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ===================================================================================================================== #
# helpers                                                                                                               #
# ===================================================================================================================== #
def _args(**over):
    a = argparse.Namespace(r9_bev_warp=True, r9_bev_warp_lags="0.5,1.0", r9_bev_warp_shuffle=False, r9_perc_rig=False, map_hires="on",
                           trunk="timm", trunk_mode="shared", r8_train_only_map=False, r9_train_only=None, ablate_frames=False)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _full(on: bool, **over):
    """The tiny FULL model (no rig: planner + tactical + perception + 10 cm map), T1 built when ``on``.  -> (model, cfg, args)."""
    model, cfg = R._build()
    assert not getattr(model.core, "_perc_rig", False) and hasattr(model.core, "decoder"), "this must be the FULL model, not the perception rig"
    args = _args()
    if on:
        BF.pin_args(args)                    # the rig is NOT required any more
        BF.attach(model, args)
    return model, cfg, args


def _batch(cfg, model, args, idxs=(12, 14), attach=True):
    """``test_map_only_ft._batch``'s body at windows whose NOW row has both lags (t >= 3), on a dataset the flag is attached to (when ``attach``)."""
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    if attach:
        BF.attach_dataset(ds, model, args)
    batch = torch.utils.data.default_collate([ds[i] for i in idxs])
    v7l = T.v7l
    n = len(idxs)
    batch["lat_v7"] = torch.tensor([0] + [v7l.IGNORE_ID] * (n - 1), dtype=torch.long)
    batch["lon_v7"] = torch.tensor([len(v7l.HEADS["tac_lon"]) - 1] + [v7l.IGNORE_ID] * (n - 1), dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2][:n], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True] * n)
    batch["tac_goal_y"] = torch.zeros(n, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(n, len(v7l.TAC_GOAL_TOKENS))
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 8, (n, 600, 320), generator=g, dtype=torch.uint8)
    codes[:, :40] = 255
    batch.update({"map_fine": codes, "map_fine_label": torch.tensor([True] * n),
                  "map_ep": torch.tensor([int(stable_episode_id(R.CLIP))] * n)})
    return batch


def _loss(model, batch, seed=5, fn=None, **kw):
    model.train()
    torch.manual_seed(seed)
    return (fn or T.compute_losses_v3)(model, batch, "cpu", log_metrics=False, **kw)


def _tensors(out) -> dict:
    return {k: v for k, v in out.items() if torch.is_tensor(v)}


def _dict_equal(a, b) -> bool:
    ta, tb = _tensors(a), _tensors(b)
    return set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)


def _grads(model) -> dict:
    return {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}


def _grads_equal(ga, gb) -> bool:
    return set(ga) == set(gb) and all((ga[n] is None and gb[n] is None) or (ga[n] is not None and gb[n] is not None and torch.equal(ga[n], gb[n])) for n in ga)


def _backward(model, out):
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    return _grads(model)


def _open_gate(model, v=0.3):
    fz = model._map_hires.bev_warp
    fz.gate.data.fill_(v)
    return fz


def _fresh_on(gate=0.0, zero_fuse=False):
    """A NEW identically-initialised ON model per exactness comparison: ``core.r8_gen`` (a dedicated generator) advances on every forward, so two forwards
    of ONE model differ in ``cls`` / ``sel_v3`` even with T1 off (MEASURED here); two fresh models called once each do not."""
    model, cfg, args = _full(True)
    fz = model._map_hires.bev_warp
    fz.gate.data.fill_(gate)
    if zero_fuse:
        fz.fuse.weight.data.zero_()
    return model, cfg, args


def _nz(g) -> int:
    return 0 if g is None else int(torch.count_nonzero(g))


def _exec_variant(drop=(), replace=()):
    """``compute_losses_v3`` re-compiled from ITS OWN SOURCE with statements deleted / replaced (an in-process source mutation; no file is touched).
    ``drop``: regexes whose single match (MULTILINE) is deleted.  ``replace``: (regex, text) pairs, a single match each."""
    src = inspect.getsource(T.compute_losses_v3)
    for pat in drop:
        src, n = re.subn(pat, "", src, flags=re.M)
        assert n == 1, ("the statement to mutate is not unique / not found in compute_losses_v3", pat, n)
    for pat, txt in replace:
        src, n = re.subn(pat, lambda m: txt, src, flags=re.M)
        assert n == 1, ("the statement to mutate is not unique / not found in compute_losses_v3", pat, n)
    g = dict(T.__dict__)
    exec(compile(src, "<compute_losses_v3 variant>", "exec"), g)
    return g["compute_losses_v3"]


STAGE_RE = r"^    _bw\.stage\(model, batch, device\)[^\n]*\n"
ABLATE_RE = r"^    if ablate_frames:\n        _bw\.refuse_ablate\(model\)[^\n]*\n"


def _reach_problems(fn) -> list:
    """The REACH check, parameterised on the loss function: T1 built, gate open, history attached -> a finite loss and a NON-ZERO gradient on the gate and
    the fuse weight.  [] iff the flag reaches the loss and the parameters through ``fn``."""
    on, cfg, args = _full(True)
    fz = _open_gate(on, 0.2)
    batch = _batch(cfg, on, args)
    try:
        out = _loss(on, batch, fn=fn)
        g = _backward(on, out)
    except (RuntimeError, SystemExit) as e:
        return ["the forward / backward raised %s: %s" % (type(e).__name__, str(e)[:100])]
    probs = []
    if not torch.isfinite(out["loss"]):
        probs.append("non-finite loss")
    if not _nz(g["_map_hires.bev_warp.gate"]):
        probs.append("zero gradient on the gate")
    if not _nz(g["_map_hires.bev_warp.fuse.weight"]):
        probs.append("zero gradient on the fuse weight")
    return probs


# ===================================================================================================================== #
# HOPS: argv -> ... -> the loss                                                                                         #
# ===================================================================================================================== #
def test_the_argv_hop_and_pin_args_no_longer_require_the_rig():
    p = T.build_parser()
    assert {a.dest: a.default for a in p._actions}["r9_bev_warp"] is False
    ns = p.parse_args(["--arm", "hier", "--out", "x", "--r9-bev-warp"])                        # no --r9-perc-rig: the FULL model's argv
    assert ns.r9_perc_rig is False and ns.r9_bev_warp is True and ns.r9_bev_warp_lags == "0.5,1.0" and ns.r9_bev_warp_shuffle is False
    BF.pin_args(_args(r9_perc_rig=False))                                    # the FULL model's argv: accepted
    BF.pin_args(_args(r9_perc_rig=True))                                     # the rig's argv: still accepted
    for over, match in ((dict(map_hires="off"), "map-hires"), (dict(trunk="resnet"), "trunk"), (dict(trunk_mode="inflate"), "trunk-mode"),
                        (dict(r8_train_only_map=True), "freeze"), (dict(r9_bev_warp_lags="0.55"), "whole number"),
                        (dict(r9_bev_warp_lags="1.0,0.5"), "increasing"), (dict(ablate_frames=True), "ablate-frames")):
        for rig_on in (False, True):
            with pytest.raises(SystemExit, match=match):                      # EVERY other refusal stands, with or without the rig
                BF.pin_args(_args(r9_perc_rig=rig_on, **over))
    for over in (dict(r9_bev_warp=False, r9_bev_warp_lags="0.5,2.0"), dict(r9_bev_warp=False, r9_bev_warp_shuffle=True)):
        with pytest.raises(SystemExit, match="dead flag"):
            BF.pin_args(_args(**over))
    assert D.REGISTRY["r9_bev_warp"].kind == "built"


def test_the_trainers_hops_are_named_at_their_source_lines():
    full = TRAINER_SRC[TRAINER_SRC.index("def compute_losses_v3("):TRAINER_SRC.index("def compute_losses_perc_rig(")]
    rigfn = TRAINER_SRC[TRAINER_SRC.index("def compute_losses_perc_rig("):]
    assert full.count("_bw.stage(model, batch, device)") == 1, "the FULL model's loss stages the history exactly once"
    assert full.index("_bw.stage(model, batch, device)") < full.index("out = _fwd(frames"), "BEFORE the model forward"
    assert full.index("_bw.refuse_ablate(model)") < full.index("_bw.stage(model, batch, device)")
    assert rigfn.index("_bw.stage(model, batch, device)") < rigfn.index("out = model(frames, perception_grid=_pgrid")      # the rig's, unchanged
    tr = inspect.getsource(T.train)
    for needle in ("_bw.attach(model, args)", "_bw.attach_dataset(ds, model, args)", "_bw.attach_dataset(e_ds, model, args)", "_bw.stamp(model, args)"):
        assert tr.count(needle) == 1, needle                                   # none of them is gated on the rig
    assert "_bw.pin_args(args)" in inspect.getsource(T._pin_refcv8)
    assert "_bw.item_fields(self.episodes, e_i, t + w - 1, self._r9_bw)" in inspect.getsource(T.V3Dataset.__getitem__)


def test_the_item_hop_the_batch_carries_the_history_and_the_flag_reaches_the_dataset():
    on, cfg, args = _full(True)
    batch = _batch(cfg, on, args)
    assert tuple(batch["bw_frames"].shape[:2]) == (2, 2) and batch["bw_frames"].dtype == torch.uint8
    assert tuple(batch["bw_has"].shape) == (2, 2) and bool(batch["bw_has"].all()), "both lags exist at t = 12 / 14"
    assert tuple(batch["bw_pose_now"].shape) == (2, 4) and tuple(batch["bw_pose_past"].shape) == (2, 2, 4)
    off, _, oargs = _full(False)
    ob = _batch(cfg, off, oargs, attach=False)
    assert not any(k.startswith("bw_") for k in ob), "OFF: the dataset item is the tip's"


# ===================================================================================================================== #
# LIVE: the full model's loss with T1 on                                                                                #
# ===================================================================================================================== #
def test_gate_zero_loss_equals_the_OFF_loss_exactly_on_the_full_model():
    off, cfg, _ = _full(False)
    on, _, args = _full(True)
    assert all(torch.equal(off.state_dict()[k], v) for k, v in on.state_dict().items() if not k.startswith(BF.KEY_PREFIX)), "the fuse's init moved a shared init"
    batch = _batch(cfg, on, args)
    out_off, out_on = _loss(off, batch), _loss(on, batch)
    assert float(on._map_hires.bev_warp.gate.abs().max()) == 0.0
    assert len(_tensors(out_off)) >= 15 and "traj" in out_on and "cls" in out_on, "this is the FULL model's loss dict (planner terms present)"
    assert _dict_equal(out_off, out_on), "gate 0 must reproduce the OFF loss for EVERY returned tensor"


def test_a_non_zero_gate_moves_the_loss_ONLY_through_the_fuse():
    off, cfg, _ = _full(False)
    probe, _, args = _fresh_on()
    batch = _batch(cfg, probe, args)
    out_off = _loss(off, batch)
    out_hist = _loss(_fresh_on(gate=0.3)[0], batch)
    moved = sorted(k for k, v in _tensors(out_off).items() if not torch.equal(v, _tensors(out_hist)[k]))
    assert "loss" in moved and "map_hires" in moved, moved                                         # the history changed the BEV the 10 cm map reads
    # (a) the fuse's weights zeroed, gate open: no contribution => EXACTLY the OFF loss
    assert _dict_equal(out_off, _loss(_fresh_on(gate=0.3, zero_fuse=True)[0], batch)), "gate open + fuse weight 0 must equal OFF exactly"
    # (b) no history (every window before its clip start): the fuse output is exactly 0 for ANY gate => EXACTLY the OFF loss
    nohist = dict(batch)
    nohist["bw_has"] = torch.zeros_like(batch["bw_has"])
    assert _dict_equal(out_off, _loss(_fresh_on(gate=0.3)[0], nohist)), "no history must equal OFF exactly, any gate"
    # (c) the history CONTENT is read: other pixels, same poses => a different loss
    other = dict(batch)
    other["bw_frames"] = (255 - batch["bw_frames"].to(torch.int32)).to(torch.uint8)
    assert not _dict_equal(out_hist, _loss(_fresh_on(gate=0.3)[0], other))


def _leaves(d, pre=""):
    for k, v in d.items():
        if torch.is_tensor(v):
            yield pre + str(k), v
        elif isinstance(v, dict):
            yield from _leaves(v, pre + str(k) + ".")


def _forward_leaves(model, batch) -> dict:
    """Every tensor the FULL model's forward returns (nested keys joined by '.'), stopped at the forward's return by a hook."""
    cap = {}

    def stop(_m, _i, o):
        cap["out"] = o
        raise _Stop()
    h = model.register_forward_hook(stop)
    try:
        _loss(model, batch)
    except _Stop:
        pass
    finally:
        h.remove()
    return dict(_leaves(cap["out"]))


def test_the_fused_BEV_reaches_the_BEV_consumers_of_the_full_model_forward():
    """What the tiny full model's forward returns: only the BEV consumers move with the gate (``perception.bev_feats`` = the planner's pooled BEV,
    ``perception.map_hires_logits`` = the 10 cm map); the planner's own outputs do NOT move in THIS build (its decoder reads no BEV tokens) -- the
    tactical / box read of the fused BEV is the next test, on a build whose decoder does."""
    off, cfg, _ = _full(False)
    probe, _, args = _fresh_on()
    batch = _batch(cfg, probe, args)
    a = _forward_leaves(off, batch)
    b = _forward_leaves(_fresh_on(gate=0.3)[0], batch)
    moved = sorted(k for k in a if k in b and a[k].shape == b[k].shape and not torch.equal(a[k], b[k]))
    assert {"perception.bev_feats", "perception.map_hires_logits"} <= set(moved), moved
    c = _forward_leaves(_fresh_on(gate=0.0)[0], batch)
    assert set(a) == set(c) and all(torch.equal(a[k], c[k]) for k in a), "gate 0: EVERY forward output equals the OFF forward"


def test_a_tactical_decoder_that_reads_the_BEV_reads_the_FUSED_BEV_and_trains_the_fuse():
    """The refcv9 full model's tactical decoder reads BEV tokens (``d_bev > 0``).  Build that tiny full model (test_plan_wires_r9_model._rig: tac decoder v6
    over (agent, bev), the pooled BEV, banded 10 cm head) and put T1 in it: the tactical logits, the box slots and the BEV tokens move with the gate;
    gate 0 reproduces every output; the FULL loss trains the gate and (once open) the fuse."""
    PW = pytest.importorskip("test_plan_wires_r9_model")

    def build(on, gate=0.0):
        cfg, model, _ = PW._rig(perc="tiny", attach=False)
        assert model.tac_decoder_v6.cfg.d_bev > 0 and not getattr(model.core, "_perc_rig", False)
        args = _args()
        if on:
            BF.pin_args(args)
            BF.attach(model, args)
            model._map_hires.bev_warp.gate.data.fill_(gate)
        return cfg, model, args

    def mk(cfg, model, args, idxs=(12, 14)):
        eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
        ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
        ds.ego_history = True
        if BF.is_built(model):
            BF.attach_dataset(ds, model, args)
        batch = torch.utils.data.default_collate([ds[i] for i in idxs])
        for k, v in PW._batch(cfg).items():
            if k not in batch or k in ("lat_v7", "lon_v7", "nav_cmd", "nav_valid", "tac_goal_y", "tac_goal_w", "map_fine", "map_fine_label", "map_ep"):
                batch[k] = v
        return batch

    cfg, off, _ = build(False)
    _, probe, args = build(True)
    batch = mk(cfg, probe, args)
    a = _forward_leaves(off, batch)
    z = _forward_leaves(build(True, 0.0)[1], batch)
    assert set(a) == set(z) and all(torch.equal(a[k], z[k]) for k in a), "gate 0: EVERY output of the full model equals the OFF forward"
    b = _forward_leaves(build(True, 0.3)[1], batch)
    moved = sorted(k for k in a if k in b and a[k].shape == b[k].shape and not torch.equal(a[k], b[k]))
    assert {"perception.bev_tokens", "tacv6_lat_logits", "tacv6_lon_logits"} <= set(moved), moved
    assert any(k.startswith("perception.box_slots.") for k in moved), moved
    m = build(True, 0.0)[1]
    gate, fuse = "_map_hires.bev_warp.gate", "_map_hires.bev_warp.fuse.weight"
    g0 = _backward(m, _loss(m, batch))
    assert _nz(g0[gate]) > 0 and _nz(g0[fuse]) == 0, "the gate learns from the full loss; CONTROL: the fuse reads EXACTLY 0.0 at gate 0"
    m2 = build(True, 0.05)[1]
    g1 = _backward(m2, _loss(m2, batch))
    assert _nz(g1[gate]) > 0 and _nz(g1[fuse]) > 0


def test_the_gradient_reaches_the_gate_and_the_fuse_and_the_controls_read_exactly_zero():
    on, cfg, args = _full(True)
    batch = _batch(cfg, on, args)
    g0 = _backward(on, _loss(on, batch))                                                           # gate 0
    gate, fuse = "_map_hires.bev_warp.gate", "_map_hires.bev_warp.fuse.weight"
    assert _nz(g0[gate]) > 0, "the gate learns from the FULL loss at init"
    assert _nz(g0[fuse]) == 0 and float(g0[fuse].abs().sum()) == 0.0, "CONTROL: at gate 0 the fuse's gradient is EXACTLY 0.0 (the ReZero shape)"
    _open_gate(on, 0.05)
    g1 = _backward(on, _loss(on, batch))                                                           # gate open: both wake
    assert _nz(g1[gate]) > 0 and _nz(g1[fuse]) > 0
    assert float(g1[fuse].abs().sum()) > 0.0
    nohist = dict(batch)
    nohist["bw_has"] = torch.zeros_like(batch["bw_has"])
    g2 = _backward(on, _loss(on, nohist))                                                          # CONTROL: no history => EXACTLY 0.0 on BOTH
    assert _nz(g2[fuse]) == 0 and _nz(g2[gate]) == 0 and float(g2[gate].abs().sum()) == 0.0
    # the CURRENT frame's lift still trains (the past path is detached) and the non-T1 parameters are untouched by the control
    assert any(g1[n] is not None and float(g1[n].abs().sum()) > 0 for n in g1 if n.startswith("_map_hires.lift."))
    # the G-DVB check agrees on the FULL model: declared == built, the fuse hooked on the shared lift
    assert BF.dvb_check(on, _args()) == [] and BF.dvb_check_lags(on, _args()) == [] and BF.dvb_check_shuffle(on, _args()) == []
    assert BF.dvb_check(on, _args(r9_bev_warp=False)), "built but not declared must be caught"


def test_the_EMA_copy_of_a_full_model_stages_into_its_own_fuse():
    on, cfg, args = _full(True)
    _open_gate(on, 0.2)
    batch = _batch(cfg, on, args)
    cp = copy.deepcopy(on)
    ref = _loss(on, batch)
    out = _loss(cp, batch)
    assert _dict_equal(ref, out), "the EMA copy computes the same loss: its own hook, its own staged context"
    assert on._map_hires.bev_warp._ctx is None and cp._map_hires.bev_warp._ctx is None, "ONE-SHOT: nothing stale is left staged"
    hooks = list(cp._map_hires.lift._forward_hooks.values())
    assert len(hooks) == 1 and hooks[0].__self__ is cp._map_hires.bev_warp


# ===================================================================================================================== #
# MUTATION: delete the stage line => RED                                                                                #
# ===================================================================================================================== #
def test_the_reach_check_is_green_on_the_real_loss_and_RED_when_the_stage_call_is_deleted():
    assert _reach_problems(None) == []                                                            # the real compute_losses_v3
    mutated = _exec_variant(drop=(STAGE_RE,))
    probs = _reach_problems(mutated)
    assert probs and "no history was staged" in probs[0], probs                                    # the hook REFUSES: a history-less forward never runs silently


def test_the_stage_call_is_a_real_statement_of_the_full_loss_and_not_dead_code():
    """Control for the mutation above: the variant machinery changes NOTHING when no statement is dropped (a variant == the real function)."""
    ident = _exec_variant()
    a, cfg, args = _fresh_on(gate=0.2)
    batch = _batch(cfg, a, args)
    assert _dict_equal(_loss(a, batch), _loss(_fresh_on(gate=0.2)[0], batch, fn=ident))


def test_a_forward_with_nothing_staged_REFUSES_it_is_never_silently_unwarped(monkeypatch):
    on, cfg, args = _full(True)
    batch = _batch(cfg, on, args)
    monkeypatch.setattr(BF, "stage", lambda *a, **k: None)                                         # an entry point that forgot to stage
    with pytest.raises(RuntimeError, match="no history was staged"):
        _loss(on, batch)


def test_a_batch_without_bw_keys_while_built_REFUSES():
    on, cfg, args = _full(True)
    nodata = _batch(cfg, on, args, attach=False)                                                    # a dataset the flag was never attached to
    assert not any(k.startswith("bw_") for k in nodata)
    with pytest.raises(SystemExit, match="carries no"):
        _loss(on, nodata)
    partial = _batch(cfg, on, args)
    partial.pop("bw_has")
    with pytest.raises(SystemExit, match="bw_has"):
        _loss(on, partial)


def test_ablate_frames_REFUSES_with_a_history_built_and_its_RED_arm_does_not():
    on, cfg, args = _full(True)
    batch = _batch(cfg, on, args)
    with pytest.raises(SystemExit, match="ablate-frames"):
        _loss(on, batch, ablate_frames=True)
    off, ocfg, oargs = _full(False)
    ob = _batch(ocfg, off, oargs, attach=False)
    assert torch.isfinite(_loss(off, ob, ablate_frames=True)["loss"]), "CONTROL: OFF + --ablate-frames is the tip's lever, untouched"
    # RED arm: the refusal statement replaced by `pass` => the information-free control would run with the scene still in the history
    leaky = _exec_variant(replace=((ABLATE_RE, "    if ablate_frames:\n        pass\n"),))
    assert torch.isfinite(_loss(on, batch, fn=leaky, ablate_frames=True)["loss"]), "without the refusal the leaky run goes through: the test above is a real check"


# ===================================================================================================================== #
# OFF: bit-identical to the loss function without the T1 statements                                                     #
# ===================================================================================================================== #
def test_OFF_the_loss_and_every_gradient_are_bit_identical_to_the_function_without_the_two_t1_statements():
    base = _exec_variant(drop=(STAGE_RE, ABLATE_RE))
    off, cfg, args = _full(False)
    off2, _, _ = _full(False)                                                                       # a SECOND identical build: one forward each
    batch = _batch(cfg, off, args, attach=False)
    out_real = _loss(off, batch)
    g_real = _backward(off, out_real)
    out_base = _loss(off2, batch, fn=base)
    g_base = _backward(off2, out_base)
    n_grad = sum(g is not None for g in g_real.values())
    assert n_grad >= 100 and len(_tensors(out_real)) >= 15, (n_grad, len(_tensors(out_real)))      # non-vacuous: the whole full model
    assert _dict_equal(out_real, out_base) and _grads_equal(g_real, g_base)
    # positive control: a ONE-ULP tamper on one gradient is caught by the comparison
    name = next(n for n, g in g_base.items() if g is not None and g.numel() > 0 and float(g.abs().max()) > 0)
    tampered = dict(g_base)
    flat = g_base[name].clone().view(-1)
    flat[0] = torch.nextafter(flat[0], torch.tensor(float("inf")))
    tampered[name] = flat.view_as(g_base[name])
    assert not _grads_equal(g_real, tampered)
    # the OFF model carries no fuse, and `stage` / `refuse_ablate` return at once
    assert not BF.is_built(off) and BF.stage(off, batch, "cpu") is None and BF.refuse_ablate(off) is None
    assert BF.stamp(off, args) is None and BF.warm_start_prefixes(off) == ()


# ===================================================================================================================== #
# ADMISSIBILITY: T1 reads past frames and past poses, nothing else                                                      #
# ===================================================================================================================== #
class _Stop(Exception):
    pass


def _bev_and_ctx(model, batch):
    """Run the FULL loss function up to the model forward's return; -> (the FUSED BEV the lift hook produced, the staged history)."""
    cap = {}
    fz = model._map_hires.bev_warp
    orig_stage = fz.stage

    def rec_stage(s8, T_, has):
        cap["ctx"] = (s8.detach().clone(), T_.detach().clone(), has.detach().clone())
        orig_stage(s8, T_, has)
    fz.stage = rec_stage
    h_lift = model._map_hires.lift.register_forward_hook(lambda m, i, o: cap.__setitem__("bev", o.detach().clone()))     # AFTER the fuse hook: it sees the FUSED BEV

    def stop(_m, _i, _o):
        raise _Stop()
    h_stop = model.register_forward_hook(stop)
    try:
        try:
            _loss(model, batch)
        except _Stop:
            pass
    finally:
        h_lift.remove()
        h_stop.remove()
        del fz.stage
    assert "bev" in cap and "ctx" in cap, "the forward did not reach the lift hook / the stage did not run"
    return cap["bev"], cap["ctx"]


INPUT_KEYS = ("frames", "map_ep")                       # + every bw_* : the ONLY batch entries the fused BEV may depend on


def _perturb_everything_but_the_inputs(batch) -> tuple[dict, int]:
    pert, n = dict(batch), 0
    for k, v in batch.items():
        if k in INPUT_KEYS or k.startswith("bw_") or not torch.is_tensor(v):
            continue
        if v.dtype.is_floating_point:
            pert[k] = v + 123.0
        elif v.dtype == torch.uint8:
            pert[k] = 255 - v if v.dim() > 2 and k.startswith("future_") else torch.roll(v, 1, 0)
        else:
            pert[k] = torch.roll(v, 1, 0)
        n += 1
    return pert, n


def _future_problems(model, batch, stage_fn=None, monkeypatch=None) -> list:
    if stage_fn is not None:
        monkeypatch.setattr(BF, "stage", stage_fn)
    bev_a, ctx_a = _bev_and_ctx(model, batch)
    pert, n = _perturb_everything_but_the_inputs(batch)
    assert n >= 10 and any(k.startswith("future_") for k in pert), "the batch carries too few non-input keys to make this test mean something"
    bev_b, ctx_b = _bev_and_ctx(model, pert)
    probs = []
    if not torch.equal(bev_a, bev_b):
        probs.append("the FUSED BEV moved under a perturbation of the non-input keys (max |d| %g)" % float((bev_a - bev_b).abs().max()))
    if not all(torch.equal(x, y) for x, y in zip(ctx_a, ctx_b)):
        probs.append("the staged history moved")
    return probs


def test_future_pose_and_label_perturbation_changes_the_fused_BEV_by_exactly_zero_and_the_RED_arm_does_not(monkeypatch):
    on, cfg, args = _full(True)
    _open_gate(on)
    batch = _batch(cfg, on, args)
    assert _future_problems(on, batch) == []                                                       # exactly 0.0, not a tolerance
    # positive controls: the instrument SEES a change in a real input
    bev0, _ = _bev_and_ctx(on, batch)
    cur = dict(batch)
    cur["frames"] = (255 - batch["frames"].to(torch.int32)).to(torch.uint8) if batch["frames"].dtype == torch.uint8 else 1.0 - batch["frames"]
    assert not torch.equal(bev0, _bev_and_ctx(on, cur)[0]), "perturbing the CURRENT frame must move the BEV"
    hist = dict(batch)
    hist["bw_frames"] = (255 - batch["bw_frames"].to(torch.int32)).to(torch.uint8)
    assert not torch.equal(bev0, _bev_and_ctx(on, hist)[0]), "perturbing the PAST frames must move the BEV"
    pose = dict(batch)
    pose["bw_pose_past"] = batch["bw_pose_past"] + torch.tensor([1.0, 0.5, 0.2, 0.0])
    assert not torch.equal(bev0, _bev_and_ctx(on, pose)[0]), "perturbing the PAST poses must move the BEV"
    real = BF.stage

    def reads_future(model, b, device):
        b = dict(b)
        b["bw_pose_now"] = b["bw_pose_now"] + 0.01 * b["future_poses"][:, 0, :]                    # RED arm: a stage that mixes the first FUTURE pose in
        return real(model, b, device)
    assert _future_problems(on, batch, reads_future, monkeypatch), "a stage that reads future_poses must FAIL the full-model future test"


def test_stage_reads_ONLY_the_bw_keys_on_the_full_model_batch():
    on, cfg, args = _full(True)
    _open_gate(on)
    batch = _batch(cfg, on, args)
    only = {k: batch[k] for k in BF.BATCH_KEYS}                                                    # NOTHING else in the dict: no future, no label, no nav
    fz = on._map_hires.bev_warp
    BF.stage(on, only, "cpu")
    a = tuple(t.clone() for t in fz._ctx)
    BF.stage(on, batch, "cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, fz._ctx)), "staging from the bw_* keys alone == staging from the whole batch"
    for k in BF.BATCH_KEYS:
        short = {kk: vv for kk, vv in only.items() if kk != k}
        with pytest.raises(SystemExit, match=k):
            BF.stage(on, short, "cpu")


# ===================================================================================================================== #
# EVAL / INFERENCE ENTRY POINTS                                                                                         #
# ===================================================================================================================== #
def test_the_eval_loader_attaches_the_history_to_the_eval_dataset_or_refuses():
    from tanitad.eval import refcv7_loader as L
    on, cfg, args = _full(True)
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    rec = L.attach_t1_history(T, ds, on, args)
    assert rec["lag_rows"] == [5, 10] and rec["shuffle"] is False and rec["n_episodes"] == 2 and ds._r9_bw is not None
    assert "bw_frames" in ds[12], "the loader-built eval dataset's items carry the history"
    ds2 = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    with pytest.raises(SystemExit, match="no model was passed"):
        L.attach_t1_history(T, ds2, None, args)                                                    # an entry point with no model cannot build a T1 eval set
    off, _, oargs = _full(False)
    assert L.attach_t1_history(T, ds2, off, _args(r9_bev_warp=False)) is None and getattr(ds2, "_r9_bw", None) is None      # OFF: nothing touched
    with pytest.raises(SystemExit, match="no temporal BEV warp"):
        L.attach_t1_history(T, ds2, off, args)                                                      # argv says T1, the model has none
    src = inspect.getsource(L.build_eval_dataset)
    assert src.count("attach_t1_history(tr, e_ds, model, args)") == 1 and src.index("attach_t1_history(") < src.index("return e_ds, e_eps, rec")
    assert 'rec["bev_warp"]' in src
    # mutation: the call deleted from build_eval_dataset => the pin above is RED (the pin is a real check)
    assert "attach_t1_history(" not in src.replace("attach_t1_history(tr, e_ds, model, args)", "")
    mutated = src.replace("    _t1 = attach_t1_history(tr, e_ds, model, args)", "    _t1 = None")
    assert mutated != src and mutated.count("attach_t1_history(tr, e_ds, model, args)") == 0
    # the loader builds the model with the fuse in it (the strict load reads the trained fuse into the same build)
    assert "tr._bw.attach(model, args)" in inspect.getsource(L)


def test_G_EVAL_forward_of_the_launch_gate_stages_the_history_on_the_full_model():
    LG = pytest.importorskip("launch_gate")
    on, cfg, args = _full(True)
    _open_gate(on, 0.2)
    batch = _batch(cfg, on, args)
    out = LG._forward_out(T, on, batch, args, 1234)                                                 # G-EVAL's forward: compute_losses_v3 stopped by a hook
    assert "map_hires_logits" in (out.get("perception") or {})
    nodata = _batch(cfg, on, args, attach=False)
    with pytest.raises(SystemExit, match="carries no"):
        LG._forward_out(T, on, nodata, args, 1234)                                                  # an eval dataset the loader did not attach REFUSES, loudly


def test_ddv2_forward_deployed_stages_before_its_forward():
    src = (ROOT / "scripts" / "ddv2_rl_refcv7.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
    body = src[src.index("def forward_deployed("):src.index("    def capture(")]
    assert body.count("tr._bw.stage(model, batch, dev)") == 1 and body.index("tr._bw.stage(model, batch, dev)") < body.index("return model(frames")


#: every file that builds a model through a loader's ``build_model`` and what it does with a T1-built model.
#:   "loss"   : the forward goes through ``refc_v3_train.compute_losses_v3`` (which stages) on a loader-built eval dataset (``attach_t1_history``)
#:   "stage"  : the file stages itself before its own bare ``model(...)``
#:   "refcv6" : the refcv6 loader never attaches the fuse -- a T1 checkpoint does not strict-load there, so a T1-built model cannot reach its forward
ENTRY_POINTS = {
    "stack/scripts/r9_e1_dump.py": "loss",
    "stack/scripts/launch_gate.py": "loss",
    "stack/scripts/ddv2_rl_refcv7.py": "stage+loss",
    "taniteval/tools/render_refcv7_replay.py": "loss",
    "taniteval/tools/render_refcv7_video.py": "loss",
    "taniteval/tools/render_refcv6_map_video.py": "refcv6",
}
_REPO = ROOT.parent


def _loader_build_model_users() -> set:
    users = set()
    for base in ("stack/scripts", "stack/tanitad", "taniteval/tools", "taniteval/taniteval"):
        for f in (_REPO / base).rglob("*.py"):
            if "tests" in f.parts or f.name == "refcv7_loader.py" or f.name == "refcv6_loader.py":
                continue
            txt = f.read_bytes().decode("utf-8", "replace")
            if re.search(r"\bL\.build_model\(", txt):
                users.add(f.relative_to(_REPO).as_posix())
    return users


def test_every_loader_entry_point_is_classified_and_stages_or_cannot_hold_a_T1_model():
    assert _loader_build_model_users() == set(ENTRY_POINTS), (
        "a file that builds a model through the loader is not in the census (or one left): classify it -- does its forward stage the T1 history? %s"
        % sorted(_loader_build_model_users() ^ set(ENTRY_POINTS)))
    for rel, how in ENTRY_POINTS.items():
        txt = (_REPO / rel).read_bytes().decode("utf-8", "replace")
        if "loss" in how:
            assert "compute_losses_v3(" in txt, rel
        if "stage" in how:
            assert "_bw.stage(" in txt, rel
        if how == "refcv6":
            assert "import refcv6_loader as L" in txt, rel
    loader6 = (ROOT / "tanitad" / "eval" / "refcv6_loader.py").read_bytes().decode("utf-8", "replace")
    assert "_bw.attach" not in loader6, "the refcv6 loader attaches no fuse: if this changes, its forward needs staging"
