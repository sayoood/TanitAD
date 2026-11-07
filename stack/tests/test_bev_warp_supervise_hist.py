"""refcv9 E1 arm T1h -- ``--r9-bev-warp-supervise-hist``: the 10 cm map loss supervises ``lift_valid_now OR hist_valid``.  CPU, the tiny FULL model of
``test_map_only_ft._build()`` (via ``test_bev_warp_full_model``'s helpers).

WHY THE LEVER EXISTS (MEASURED by the MM, refcv9 E1 arm T1): the temporal BEV warp never opened its gate (|g| 0.0077).  The map loss supervises only the cells the camera reaches
NOW (``--map-lift-valid-mask`` default True); at 0-5 m 94.7 % of cells are unobserved now and 42 % are covered ONLY by warped history -- exactly the cells history could fix, and the
loss never sees them.  ``hist_valid`` = the union over lags of the warp-valid mask (``bev_warp.hist_valid_of``): GEOMETRY ONLY (poses, the camera's own ``valid``, ``has``).

THE CLASS ("built, tested, and UNREACHABLE FROM ITS CALLER", CLAUDE.md 2026-09-10/11).  Hop by hop, each one proved here:

  argv ``--r9-bev-warp-supervise-hist`` -> ``pin_args`` -> ``attach`` (``fuse.emit_hist`` + ``BevWarpState.supervise_hist``) -> ``BevWarpFuse.lift_hook``
  -> ``MapHiresBranch.forward`` (``map_hires_hist_valid``) -> ``RefCV3Model._bev_hook`` (``pout``) -> ``refc_v3_train._map_hires_loss`` (``_bw.supervised_lift_valid``) -> the CE -> ``.backward()``

  LIVE        a cell that is lift-INVALID now and hist-VALID gives the GATE a NON-ZERO gradient (and the fuse weight, gate open); under the tip's mask the SAME cell gives EXACTLY 0.0
              (the defect, reproduced); a cell neither lift- nor hist-valid gives EXACTLY 0.0 with the flag ON (the zero control: the probe works, and the OR is not "supervise everything")
  MUTATION    the OR line deleted from ``_map_hires_loss`` (source-level, in-process) => the LIVE check goes RED; the emission dropped between the forward and the loss => the loss REFUSES
  OFF         flag off => the loss dict, EVERY parameter gradient and the ``pout`` key set are bit-identical to ``_map_hires_loss`` WITHOUT the T1h lines (an in-process source variant)
  SHUFFLE     ``--r9-bev-warp-shuffle`` has the SAME supervision mask, while the history CONTENT differs (the control is a control)
  GATE        a G-DVB entry (kind ``built``); the flag without ``--r9-bev-warp`` / with ``--no-map-lift-valid-mask`` is refused (K2: the banded head is ADMITTED -- tests/test_r9_k2_bandhist.py); the four-flag T1 set is ONE fact across the
              gate, both harnesses, the emitter and the trainer; refcv9-e1rig ADMITS the T1h argv, the full family REFUSES it; G-EVAL reads the loss-mask state off both builds
"""
from __future__ import annotations

import inspect
import json
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
import test_bev_warp_full_model as FM  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.train import bev_warp_flag as BF  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
TRAINER_SRC = FM.TRAINER_SRC
FINE_HW = (600, 320)
#: the trainer's REAL ``_map_hires_loss`` captured at import: the recorder below patches ``T._map_hires_loss`` per test, and a later ``inspect.getsource`` / default must never see a stale wrapper
_REAL = T._map_hires_loss


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ===================================================================================================================== #
# helpers                                                                                                               #
# ===================================================================================================================== #
def _built(sup: bool, shuffle: bool = False, gate: float = 0.0):
    """A NEW identically-initialised tiny FULL model with T1 built (T1h iff ``sup``).  -> (model, cfg, args)."""
    model, cfg = R._build()
    args = FM._args(r9_bev_warp_shuffle=shuffle, r9_bev_warp_supervise_hist=sup)
    BF.pin_args(args)
    BF.attach(model, args)
    model._map_hires.bev_warp.gate.data.fill_(gate)
    return model, cfg, args


def _plain():
    """The tiny FULL model with T1 NOT built at all."""
    model, cfg = R._build()
    return model, cfg, FM._args(r9_bev_warp=False)


def _install(monkeypatch, fn=None):
    """Patch ``T._map_hires_loss`` with a recorder around ``fn`` (default: the real one).  -> the dict the recorder fills ({pout, extra})."""
    cap: dict = {}
    real = fn or _REAL

    def wrap(model, out, batch, device, extra, with_metrics=True):
        cap["pout"], cap["extra"] = out["perception"], extra
        return real(model, out, batch, device, extra, with_metrics)
    monkeypatch.setattr(T, "_map_hires_loss", wrap)
    return cap


def _variant(drop=(), replace=(), name="_map_hires_loss"):
    """``T.<name>`` re-compiled from ITS OWN SOURCE with regex-matched blocks deleted / replaced (an in-process mutation; no file is touched)."""
    src = inspect.getsource(_REAL if name == "_map_hires_loss" else getattr(T, name))
    for pat in drop:
        src, n = re.subn(pat, "", src, flags=re.M)
        assert n == 1, ("the block to mutate is not unique / not found", pat, n)
    for pat, txt in replace:
        src, n = re.subn(pat, lambda m: txt, src, flags=re.M)
        assert n == 1, ("the block to mutate is not unique / not found", pat, n)
    g = dict(T.__dict__)
    exec(compile(src, "<%s variant>" % name, "exec"), g)
    return g[name]


T1H_BLOCK_RE = (r"^    _lv_now = lv\n    lv = _bw\.supervised_lift_valid\(model, pout, lv, sel\)[^\n]*\n"
                r"    if lv is not _lv_now:[^\n]*\n        extra\[\"n_map_hires_hist_cells025\"\][^\n]*\n")
OR_LINE_RE = r"^    lv = _bw\.supervised_lift_valid\(model, pout, lv, sel\)[^\n]*\n"


def _tip_loss_fn():
    """``_map_hires_loss`` as the TIP had it: the T1h block deleted."""
    return _variant(drop=(T1H_BLOCK_RE,))


def _batch(cfg, model, args):
    return FM._batch(cfg, model, args)


def _fine(mask_025):
    return H.lift_valid_to_fine(mask_025, FINE_HW)


def _codes_on(cells_fine: torch.Tensor, seed: int = 3) -> torch.Tensor:
    """uint8 ``[n, 600, 320]``: NOT_SEEN (255) everywhere except the chosen fine cells, which carry classes 1..7."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.full(tuple(cells_fine.shape), 255, dtype=torch.uint8)
    rnd = torch.randint(1, 8, tuple(cells_fine.shape), generator=g, dtype=torch.uint8)
    return torch.where(cells_fine, rnd, codes)


def _geometry(monkeypatch):
    """The per-window geometry masks of the tiny rig (they depend on poses + the camera bank, NOT on weights): (lv, hv) [n, 240, 128] bool, from a T1h forward."""
    model, cfg, args = _built(True)
    batch = _batch(cfg, model, args)
    cap = _install(monkeypatch)
    FM._loss(model, batch)
    pout = cap["pout"]
    return pout["map_hires_lift_valid"].clone(), pout[BF.HIST_KEY].clone()


def _gate_grads(monkeypatch, sup: bool, cell_class: str, *, fn=None, gate: float = 0.0, shuffle: bool = False):
    """ONE fresh model, ONE batch whose 10 cm labels sit ONLY on ``cell_class`` cells (``hist_only`` = lift-invalid now & hist-valid; ``neither``; ``now`` = lift-valid now),
    backward of the unweighted ``map_hires`` term alone.  -> (gate grad, fuse weight grad, n supervised cells, n 0.25 m cells the history added or None)."""
    lv, hv = _geometry(monkeypatch)
    sel = {"hist_only": hv & ~lv, "neither": ~hv & ~lv, "now": lv}[cell_class]
    assert int(sel.sum()) > 0, cell_class
    model, cfg, args = _built(sup, shuffle=shuffle, gate=gate)
    batch = _batch(cfg, model, args)
    batch["map_fine"] = _codes_on(_fine(sel))
    cap = _install(monkeypatch, fn)
    out = FM._loss(model, batch)
    model.zero_grad(set_to_none=True)
    out["map_hires"].backward()
    g = dict(model.named_parameters())
    gg, gf = g["_map_hires.bev_warp.gate"].grad, g["_map_hires.bev_warp.fuse.weight"].grad
    return gg, gf, float(cap["extra"]["n_map_hires_cells"]), cap["extra"].get("n_map_hires_hist_cells025")


def _nz(g) -> int:
    return 0 if g is None else int(torch.count_nonzero(g))


# ===================================================================================================================== #
# the geometry the rest of the file stands on                                                                           #
# ===================================================================================================================== #
def test_the_tiny_rig_has_history_only_and_neither_cells_so_the_probes_below_are_not_vacuous(monkeypatch):
    lv, hv = _geometry(monkeypatch)
    assert lv.dtype == hv.dtype == torch.bool and tuple(lv.shape) == tuple(hv.shape) == (2, 240, 128)
    assert int((hv & ~lv).sum()) > 0 and int((~hv & ~lv).sum()) > 0 and int((hv & lv).sum()) > 0
    assert int((hv & ~lv).sum()) < int((~lv).sum()), "history covers SOME blind cells, not all (a real mask, not a switch)"


# ===================================================================================================================== #
# LIVE: the gate's gradient from a history-only cell, and the zero controls                                             #
# ===================================================================================================================== #
def test_LIVE_a_history_only_cell_gives_the_gate_a_NON_ZERO_gradient_with_the_flag_ON(monkeypatch):
    gg, gf, n, added = _gate_grads(monkeypatch, True, "hist_only")
    assert n > 0 and added is not None and added > 0, "the history widened the supervision by cells the camera does not reach now"
    assert _nz(gg) > 0 and float(gg.abs().max()) > 0.0 and bool(torch.isfinite(gg).all())
    assert _nz(gf) == 0, "ReZero: at gate 0 the fuse weight's gradient is EXACTLY 0.0 by construction"
    _, gf_open, n2, _ = _gate_grads(monkeypatch, True, "hist_only", gate=0.3)
    assert n2 == n and _nz(gf_open) > 0, "gate open: the fuse weight gets a non-zero gradient from the history-only cells too"


def test_LIVE_the_SAME_cell_gives_EXACTLY_zero_under_the_tips_mask_which_is_the_defect(monkeypatch):
    gg, gf, n, added = _gate_grads(monkeypatch, False, "hist_only")
    assert n == 0.0 and added is None, "flag OFF: a lift-invalid cell is not supervised"
    assert _nz(gg) == 0 and float(gg.abs().max()) == 0.0 and _nz(gf) == 0
    gg_open, gf_open, n2, _ = _gate_grads(monkeypatch, False, "hist_only", gate=0.3)
    assert n2 == 0.0 and _nz(gg_open) == 0 and _nz(gf_open) == 0


def test_LIVE_CONTROL_a_cell_that_is_neither_lift_valid_nor_hist_valid_reads_EXACTLY_zero_with_the_flag_ON(monkeypatch):
    for gate in (0.0, 0.3):
        gg, gf, n, added = _gate_grads(monkeypatch, True, "neither", gate=gate)
        assert n == 0.0 and added > 0, "labels only where neither mask reaches: nothing supervised (the counter is the MASK widening, label-independent, > 0)"
        assert _nz(gg) == 0 and float(gg.abs().max()) == 0.0 and _nz(gf) == 0


def test_LIVE_a_currently_visible_cell_is_supervised_identically_with_the_flag_on_or_off(monkeypatch):
    g_on, _, n_on, added_on = _gate_grads(monkeypatch, True, "now")
    g_off, _, n_off, _ = _gate_grads(monkeypatch, False, "now")
    assert n_on == n_off > 0 and added_on > 0, "labels on lift-valid cells: the supervised LABELLED set is the tip's (the OR widened the mask elsewhere, where no label sits)"
    assert _nz(g_on) > 0 and _nz(g_off) > 0, "and the gate hears the currently visible cells either way"


# ===================================================================================================================== #
# MUTATION                                                                                                              #
# ===================================================================================================================== #
def test_MUTATION_deleting_the_OR_line_turns_the_LIVE_check_RED(monkeypatch):
    """The LIVE check of the first test, run against ``_map_hires_loss`` with ONLY the OR line deleted: it must FAIL (zero gradient, zero history cells)."""
    mutant = _variant(drop=(OR_LINE_RE,))
    assert inspect.getsource(_REAL).count("_bw.supervised_lift_valid(") == 1, "the real function carries the OR line; the mutant does not (its regex matched exactly once)"
    assert "supervised_lift_valid" in _REAL.__code__.co_names and "supervised_lift_valid" not in mutant.__code__.co_names
    gg, _, n, added = _gate_grads(monkeypatch, True, "hist_only", fn=mutant)
    problems = []
    if not (n > 0):
        problems.append("no history-only cell supervised")
    if not _nz(gg):
        problems.append("zero gradient on the gate")
    assert problems == ["no history-only cell supervised", "zero gradient on the gate"], problems       # RED: both readings fail, exactly as the OFF arm
    assert added is None, "(the counter line sits behind the same identity test, so it is silent too)"
    # and the SAME check on the real function is GREEN (the previous tests), so the check discriminates
    gg_real, _, n_real, _ = _gate_grads(monkeypatch, True, "hist_only")
    assert n_real > 0 and _nz(gg_real) > 0


def test_MUTATION_a_forward_that_drops_the_emitted_key_REFUSES_it_never_silently_trains_on_the_narrow_mask(monkeypatch):
    model, cfg, args = _built(True)
    batch = _batch(cfg, model, args)
    real = _REAL

    def strip(model_, out, batch_, device, extra, with_metrics=True):
        out["perception"].pop(BF.HIST_KEY)                     # the hop between MapHiresBranch.forward and the loss is gone
        return real(model_, out, batch_, device, extra, with_metrics)
    monkeypatch.setattr(T, "_map_hires_loss", strip)
    with pytest.raises(SystemExit, match="carries no `map_hires_hist_valid`"):
        FM._loss(model, batch)


def test_MUTATION_an_attach_that_forgets_emit_hist_is_refused_by_the_loss_AND_by_G_DVB():
    """``attach`` without ``fuse.emit_hist = sup`` (simulated on the built model): the state says T1h, the fuse emits nothing -> the loss REFUSES, G-DVB names it."""
    model, cfg, args = _built(True)
    model._map_hires.bev_warp.emit_hist = False
    with pytest.raises(SystemExit, match="carries no `map_hires_hist_valid`"):
        FM._loss(model, _batch(cfg, model, args))
    assert [m.read_from for m in D.REGISTRY["r9_bev_warp_supervise_hist"].check(model, args)] == ["model._map_hires.bev_warp.emit_hist"]


def test_MUTATION_a_fuse_that_does_not_record_hist_valid_is_refused_by_the_branch(monkeypatch):
    orig = BF.BW.BevWarpFuse.lift_hook

    def no_record(self, lift, inputs, out):
        r = orig(self, lift, inputs, out)
        self._hist_valid = None                                 # the hook "forgot" to record it
        return r
    monkeypatch.setattr(BF.BW.BevWarpFuse, "lift_hook", no_record)      # BEFORE the build: ``attach`` registers the bound method it finds
    model, cfg, args = _built(True)
    with pytest.raises(RuntimeError, match="recorded no hist_valid"):
        FM._loss(model, _batch(cfg, model, args))


# ===================================================================================================================== #
# OFF: bit-identical                                                                                                    #
# ===================================================================================================================== #
def _loss_and_grads(model, batch, fn, monkeypatch):
    cap = _install(monkeypatch, fn)
    out = FM._loss(model, batch)
    return out, FM._backward(model, out), cap


@pytest.mark.parametrize("t1_built", [True, False])
def test_OFF_the_loss_every_gradient_and_the_pout_keys_are_bit_identical_to_the_tip(monkeypatch, t1_built):
    """T1 built with the flag OFF (the existing T1 arms), and T1 not built at all (every other arm): ``_map_hires_loss`` with the T1h block == without it, bit for bit."""
    make = (lambda: _built(False)) if t1_built else _plain
    m_new, cfg, a_new = make()
    m_tip, _, a_tip = make()
    assert all(torch.equal(m_new.state_dict()[k], v) for k, v in m_tip.state_dict().items()), "two identical builds"
    b_new, b_tip = _batch(cfg, m_new, a_new), _batch(cfg, m_tip, a_tip)
    out_new, g_new, cap_new = _loss_and_grads(m_new, b_new, None, monkeypatch)
    out_tip, g_tip, cap_tip = _loss_and_grads(m_tip, b_tip, _tip_loss_fn(), monkeypatch)
    assert FM._dict_equal(out_new, out_tip), "every returned tensor equal (torch.equal)"
    assert FM._grads_equal(g_new, g_tip), "every parameter gradient equal (torch.equal)"
    assert set(cap_new["pout"]) == set(cap_tip["pout"]) and BF.HIST_KEY not in cap_new["pout"], "OFF: the key is ABSENT from the forward's output"
    assert "n_map_hires_hist_cells025" not in cap_new["extra"] and set(cap_new["extra"]) == set(cap_tip["extra"])
    assert float(out_new["map_hires"]) == float(out_tip["map_hires"]) and out_new["map_hires"].requires_grad


def test_OFF_the_flag_adds_no_parameter_and_no_state_key_and_changes_no_shared_init():
    on, _, _ = _built(True)
    off, _, _ = _built(False)
    assert list(on.state_dict()) == list(off.state_dict()), "the lever adds no parameter / buffer"
    assert all(torch.equal(v, off.state_dict()[k]) for k, v in on.state_dict().items())
    assert on._map_hires.bev_warp.emit_hist is True and off._map_hires.bev_warp.emit_hist is False
    assert on._r9_bev_warp.supervise_hist is True and off._r9_bev_warp.supervise_hist is False
    assert BF.supervise_hist_on(on) and not BF.supervise_hist_on(off) and not BF.supervise_hist_on(_plain()[0])


def test_OFF_supervised_lift_valid_returns_the_SAME_object_when_off_or_when_the_lift_mask_is_off():
    lv = torch.ones(2, 4, 4, dtype=torch.bool)
    sel = torch.tensor([0, 1])
    off, _, _ = _built(False)
    assert BF.supervised_lift_valid(off, {}, lv, sel) is lv
    on, _, _ = _built(True)
    assert BF.supervised_lift_valid(on, {BF.HIST_KEY: lv}, None, sel) is None, "--no-map-lift-valid-mask: nothing to widen"
    hv = torch.zeros(3, 4, 4, dtype=torch.bool)
    hv[2, 1, 1] = True
    lv2 = torch.zeros(2, 4, 4, dtype=torch.bool)
    out = BF.supervised_lift_valid(on, {BF.HIST_KEY: hv}, lv2, torch.tensor([2, 0]))       # `sel` indexes the history exactly as it indexed `lv`
    assert out.dtype == torch.bool and bool(out[0, 1, 1]) and int(out.sum()) == 1 and not bool(out[1].any())


# ===================================================================================================================== #
# SHUFFLE: the same supervision set, a different history CONTENT                                                         #
# ===================================================================================================================== #
def test_SHUFFLE_the_supervision_mask_is_identical_to_the_unshuffled_arms_while_the_history_content_differs(monkeypatch):
    m_a, cfg, a_a = _built(True, shuffle=False, gate=0.3)
    m_s, _, a_s = _built(True, shuffle=True, gate=0.3)
    b_a, b_s = _batch(cfg, m_a, a_a), _batch(cfg, m_s, a_s)
    assert not torch.equal(b_a["bw_frames"], b_s["bw_frames"]), "the control's history FRAMES come from the donor episode"
    assert torch.equal(b_a["bw_pose_past"], b_s["bw_pose_past"]) and torch.equal(b_a["bw_has"], b_s["bw_has"]), "...with the item's OWN poses and has"
    cap_a = _install(monkeypatch)
    out_a = FM._loss(m_a, b_a)
    cap_s = _install(monkeypatch)
    out_s = FM._loss(m_s, b_s)
    pa, ps = cap_a["pout"], cap_s["pout"]
    assert torch.equal(pa[BF.HIST_KEY], ps[BF.HIST_KEY]), "hist_valid is GEOMETRY ONLY: identical under the shuffle"
    assert torch.equal(pa["map_hires_lift_valid"], ps["map_hires_lift_valid"])
    assert cap_a["extra"]["n_map_hires_cells"] == cap_s["extra"]["n_map_hires_cells"] and cap_a["extra"]["n_map_hires_hist_cells025"] == cap_s["extra"]["n_map_hires_hist_cells025"] > 0
    assert not torch.equal(out_a["map_hires"], out_s["map_hires"]), "the history CONTENT differs, so the fused BEV and the loss differ: the control is a control"


# ===================================================================================================================== #
# the hops, at their source lines                                                                                       #
# ===================================================================================================================== #
def test_the_hops_are_named_at_their_source_lines():
    loss_src = inspect.getsource(_REAL)
    assert loss_src.count("_bw.supervised_lift_valid(model, pout, lv, sel)") == 1
    assert loss_src.index("_bw.supervised_lift_valid(model, pout, lv, sel)") < loss_src.index("_mhr.map_hires_loss_row(") and \
        loss_src.index('if bool(getattr(model, "_map_lift_valid_mask", True))') < loss_src.index("_bw.supervised_lift_valid(")
    assert "_bw.supervise_hist_on(model)" not in loss_src and loss_src.count("return _map_banded_loss(") == 1, "K2: the banded dispatch no longer refuses T1h"
    assert inspect.getsource(T._map_banded_loss).count("_bw.supervised_banded_unobs(") == 1, "K2: the banded loss takes the OR itself (tests/test_r9_k2_bandhist.py proves it reaches the gradient)"
    mh_src = inspect.getsource(H.MapHiresBranch.forward)
    assert mh_src.count("take_hist_valid()") == 1 and mh_src.count('out["map_hires_hist_valid"] = hist') == 1
    assert mh_src.index("bev = self.lift(") < mh_src.index("take_hist_valid()") < mh_src.index("feats = self._run(self.encoder, bev)")
    rc3 = (ROOT / "tanitad" / "refs" / "refc_v3.py").read_bytes().decode("utf-8")
    assert rc3.count('pout["map_hires_hist_valid"] = hi["map_hires_hist_valid"]') == 1
    hook_src = inspect.getsource(BF.BW.BevWarpFuse.lift_hook)
    assert hook_src.count("self._hist_valid = hist_valid_of(masks) if self.emit_hist else None") == 1
    assert hook_src.index("self._hist_valid = hist_valid_of(masks)") < hook_src.index("return self.mix(out, warped, masks)")
    assert inspect.getsource(BF.attach).count("fuse.emit_hist = sup") == 1
    assert "--r9-bev-warp-supervise-hist" in {s for a in T.build_parser()._actions for s in a.option_strings}
    p = {a.dest: a.default for a in T.build_parser()._actions}
    assert p["r9_bev_warp_supervise_hist"] is False, "default OFF: the flag must be NAMED in the arm's argv"
    # the eval loader rebuilds the arm through the SAME attach (it reads argv, never a copy of the state)
    assert "tr._bw.attach(model, args)" in (ROOT / "tanitad" / "eval" / "refcv7_loader.py").read_bytes().decode("utf-8")


def test_the_other_lift_valid_mask_sites_have_nothing_to_widen_under_map_hires(monkeypatch):
    """Sites 6809 / 7359 (the legacy 0.5 m `map_loss_row`): their mask is `pout["map_valid"]`, emitted ONLY by a perception branch that owns its own s16 lift.  With a 10 cm branch built the
    branch is `map_hires_pool` (a lift + a 10 cm branch is refused at the forward), so no mask exists there to OR into -- and T1h REQUIRES the 10 cm branch."""
    model, cfg, args = _built(True)
    cap = _install(monkeypatch)
    FM._loss(model, _batch(cfg, model, args))
    assert "map_valid" not in cap["pout"], "no legacy mask under --map-hires"
    pb = (ROOT / "tanitad" / "models" / "refcv6_perception_branch.py").read_bytes().decode("utf-8")
    assert pb.count('out["map_valid"] = map_valid_from_lift(valid)') == 1 and "if self.lift is not None:" in pb
    assert model._perception is None or model._perception.lift is None


# ===================================================================================================================== #
# the flag's refusals, G-DVB, the stamp                                                                                  #
# ===================================================================================================================== #
def test_pin_args_REFUSES_the_flag_without_r9_bev_warp_and_the_dead_combinations():
    with pytest.raises(SystemExit, match=r"--r9-bev-warp-supervise-hist without --r9-bev-warp: nothing would read it \(a dead flag\)"):
        BF.pin_args(FM._args(r9_bev_warp=False, r9_bev_warp_supervise_hist=True))
    with pytest.raises(SystemExit, match="dead flag"):
        BF.pin_args(FM._args(r9_bev_warp=False, r9_bev_warp_shuffle=True, r9_bev_warp_supervise_hist=True))
    BF.pin_args(FM._args(r9_bev_warp_supervise_hist=True, r9_map_banded=True))       # K2: accepted -- the banded loss ORs hist_valid into its `unobs` (tests/test_r9_k2_bandhist.py)
    with pytest.raises(SystemExit, match="--no-map-lift-valid-mask"):
        BF.pin_args(FM._args(r9_bev_warp_supervise_hist=True, map_lift_valid_mask=False))
    BF.pin_args(FM._args(r9_bev_warp_supervise_hist=True))                           # accepted: T1 + the flag
    BF.pin_args(FM._args(r9_bev_warp_supervise_hist=True, r9_bev_warp_shuffle=True))  # accepted: the content control
    BF.pin_args(FM._args(r9_bev_warp_supervise_hist=False, r9_map_banded=True, map_lift_valid_mask=False))     # the flag OFF: the other refusals are not this flag's business
    # through the REAL parser: the argv the PI would launch
    ns = T.build_parser().parse_args(["--arm", "hier", "--out", "x", "--r9-bev-warp-supervise-hist"])
    assert ns.r9_bev_warp_supervise_hist is True and ns.r9_bev_warp is False
    with pytest.raises(SystemExit, match="dead flag"):
        BF.pin_args(ns)


def test_the_G_DVB_entry_exists_is_kind_built_and_catches_both_halves():
    lever = D.REGISTRY["r9_bev_warp_supervise_hist"]
    assert lever.kind == "built" and lever.check is BF.dvb_check_supervise_hist
    on, _, a_on = _built(True)
    off, _, a_off = _built(False)
    assert lever.check(on, a_on) == [] and lever.check(off, a_off) == []
    mism = lever.check(off, a_on)                                                   # argv declares T1h, the build has none
    assert [m.lever for m in mism] == ["r9_bev_warp_supervise_hist"] and mism[0].declared is True and mism[0].built is False
    assert [m.lever for m in lever.check(on, a_off)] == ["r9_bev_warp_supervise_hist"]   # the reverse
    on._map_hires.bev_warp.emit_hist = False                                        # the loss mask is widened but the fuse emits nothing: the second half
    half = lever.check(on, a_on)
    assert half and half[0].read_from == "model._map_hires.bev_warp.emit_hist"
    assert lever.check(_plain()[0], FM._args(r9_bev_warp=False)) == [], "T1 not built, flag not declared: inert"


def test_the_stamp_records_the_flag_when_on_and_is_unchanged_when_off():
    on, _, a_on = _built(True)
    off, _, a_off = _built(False)
    s_on, s_off = BF.stamp(on, a_on), BF.stamp(off, a_off)
    assert "supervise_hist" in s_on and s_on["supervise_hist"]["flag"] == "--r9-bev-warp-supervise-hist" and BF.HIST_KEY in s_on["supervise_hist"]["emitted_as"]
    assert "supervise_hist" not in s_off, "OFF: the T1 stamp (config.json r9_bev_warp) is key-for-key what it was"
    assert set(s_on) - {"supervise_hist"} == set(s_off)
    assert BF.stamp(_plain()[0], FM._args(r9_bev_warp=False)) is None


# ===================================================================================================================== #
# the launch gate, the harnesses, the emitter, the argv files                                                            #
# ===================================================================================================================== #
LG = pytest.importorskip("launch_gate")
G = pytest.importorskip("g_box_overfit")
O = pytest.importorskip("map_hires_overfit")
MK = pytest.importorskip("r9_e1_make_argv")


def _argv(name):
    return json.loads((RUNS / f"r9e1_{name}.argv.json").read_text(encoding="utf-8"))["argv"]


C0, T1, T1S, T1H, T1HS = (_argv(n) for n in ("C0", "T1", "T1s", "T1h", "T1hs"))
SUP = "--r9-bev-warp-supervise-hist"
FUSE = "--r9-bev-warp-fuse-init"
FILL = "--r9-bev-warp-fill-only"


def test_the_six_flag_T1_set_is_ONE_fact_across_gate_harnesses_emitter_and_trainer():
    assert LG.E1_T1_FLAGS == ("--r9-bev-warp", "--r9-bev-warp-lags", "--r9-bev-warp-shuffle", SUP, FUSE, FILL) == tuple(G.T1_FLAGS) == tuple(O.T1_FLAGS)
    assert (BF.FLAG, BF.LAGS_FLAG, BF.SHUF_FLAG, BF.SUP_FLAG, BF.FUSE_INIT_FLAG, BF.FILL_FLAG) == LG.E1_T1_FLAGS
    assert {t for t in MK.T1_ARMS["T1h"] + MK.T1_ARMS["T1hs"] if t.startswith("--")} <= set(LG.E1_T1_FLAGS)
    assert set(G.T1_VALUE_FLAGS) == {"--r9-bev-warp-lags", FUSE} <= set(LG.E1_T1_FLAGS)
    assert set(LG.E1_T1_FLAGS) <= set(T.build_parser()._option_string_actions)


def test_the_T1h_argvs_are_T1_and_T1s_plus_the_three_levers_and_nothing_else():
    def pairs(argv):
        return [(f, v) for f, v in LG.flag_pairs(argv) if f not in ("--out",)]
    for new, base in ((T1H, T1), (T1HS, T1S)):
        assert [p for p in pairs(new) if p[0] not in (SUP, FUSE, FILL)] == pairs(base), "every other flag AND value equal, same order"
        assert dict(LG.flag_pairs(new))[FILL] == [] and FILL not in dict(LG.flag_pairs(base)) and sum(1 for f, _ in LG.flag_pairs(new) if f == FILL) == 1
        assert dict(LG.flag_pairs(new))[SUP] == [] and SUP not in dict(LG.flag_pairs(base))
        assert dict(LG.flag_pairs(new))[FUSE] == ["identity"] and FUSE not in dict(LG.flag_pairs(base))
        assert sum(1 for f, _ in LG.flag_pairs(new) if f == SUP) == 1 and sum(1 for f, _ in LG.flag_pairs(new) if f == FUSE) == 1
    assert [p for p in pairs(T1HS) if p[0] != "--r9-bev-warp-shuffle"] == pairs(T1H), "T1hs = T1h + the shuffle, nothing else"
    assert dict(LG.flag_pairs(T1H))["--out"] == ["/home/nvidia/r8ns/r9e1/runs/r9e1-T1h"] and dict(LG.flag_pairs(T1HS))["--out"] == ["/home/nvidia/r8ns/r9e1/runs/r9e1-T1hs"]
    assert "--r9-bev-warp-shuffle" in dict(LG.flag_pairs(T1HS)) and "--r9-bev-warp-shuffle" not in dict(LG.flag_pairs(T1H))
    for arm, argv in (("T1h", T1H), ("T1hs", T1HS)):
        raw = (RUNS / f"r9e1_{arm}.argv.json").read_bytes()
        assert b"\r" not in raw and raw.endswith(b"}\n")
        rec = json.loads(raw)
        assert rec["arm"] == f"r9e1_{arm}" and rec["seed"] == 0 and rec["derived_from"] == "r9e1_C0" and rec["steps"] == 4000 and rec["argv"] == argv
        assert SUP in rec["lever"]["flags"] and FUSE in rec["lever"]["flags"] and FILL in rec["lever"]["flags"] and "identity" not in rec["lever"]["flags"]
        assert "supervise_hist" in rec["lever"] and "fuse_init" in rec["lever"] and "fill_only" in rec["lever"] and rec["lever"]["shuffle_control"] == (arm == "T1hs")
    assert len({LG.argv_sha256(a) for a in (C0, T1, T1S, T1H, T1HS)}) == 5
    # the emitter reproduces the committed files and its pins are real
    for arm, argv in (("T1h", T1H), ("T1hs", T1HS)):
        assert MK.t1_arm_argv(C0, arm) == argv
        MK.check_t1_arm_pins(argv, arm)
    with pytest.raises(SystemExit, match="parsed as"):
        MK.check_t1_arm_pins(T1H, "T1")                                  # the wrong arm name for the argv (the first lever that disagrees is named)
    with pytest.raises(SystemExit, match="fuse-init parsed as"):
        MK.check_t1_arm_pins(T1, "T1h")
    with pytest.raises(SystemExit, match="fill-only parsed as"):
        MK.check_t1_arm_pins([t for t in T1H if t != FILL], "T1h")      # the fill-only lever is part of the arm
    with pytest.raises(SystemExit, match="supervise-hist parsed as"):
        MK.check_t1_arm_pins([t for t in T1H if t != SUP], "T1h")
    with pytest.raises(SystemExit, match="shuffle control parsed as"):
        MK.check_t1_arm_pins(T1H, "T1hs")
    with pytest.raises(SystemExit, match="fuse-init parsed as"):
        MK.check_t1_arm_pins(LG.set_flag(T1H, FUSE, ["rezero"]), "T1h")                   # the init lever is part of the arm
    # the argv parses to ONE built state: T1 on, the OR on, the shuffle exactly on the control
    for arm, argv, shuf in (("T1h", T1H, False), ("T1hs", T1HS, True)):
        ns = T.build_parser().parse_args(argv)
        assert ns.r9_bev_warp and ns.r9_bev_warp_supervise_hist and ns.r9_bev_warp_shuffle is shuf and ns.map_lift_valid_mask is True and ns.r9_map_banded is False
        assert ns.r9_bev_warp_fuse_init == "identity" and ns.r9_bev_warp_fill_only is True
        BF.pin_args(ns)


def test_the_argv_builds_the_arm_the_eval_loader_path_attach_gives_the_same_state():
    ns = T.build_parser().parse_args(T1H)
    model, _ = R._build()
    BF.attach(model, ns)                                                 # exactly what refcv7_loader.py:441 and train() call
    assert model._r9_bev_warp.supervise_hist is True and model._map_hires.bev_warp.emit_hist is True and model._map_hires.bev_warp.init_mode == "identity"
    assert D.REGISTRY["r9_bev_warp_supervise_hist"].check(model, ns) == [] and D.REGISTRY["r9_bev_warp_fuse_init"].check(model, ns) == []
    t1 = T.build_parser().parse_args(T1)
    model_t1, _ = R._build()
    BF.attach(model_t1, t1)
    assert model_t1._r9_bev_warp.supervise_hist is False and model_t1._map_hires.bev_warp.emit_hist is False and model_t1._map_hires.bev_warp.init_mode == "rezero"


def test_refcv9_e1rig_ADMITS_T1h_and_the_full_family_admits_it_ONLY_AS_THE_BUNDLE():
    RIG = LG.PROFILES["refcv9-e1rig"]
    for argv in (T1H, T1HS, LG.set_flag(C0, SUP, []), LG.set_flag(C0, FUSE, ["identity"]), LG.set_flag(C0, FILL, [])):
        assert LG.open_item_reasons(RIG, argv) == [], "no own-record item: the flag changes no head and no inference input"
        assert LG.own_record_row(RIG, argv, "G-MAP-OVERFIT") is None and LG.own_record_row(RIG, argv, "G-BOX-OVERFIT") is None
        assert LG.forbidden_lever_reasons(RIG, argv)[0] == LG.forbidden_lever_reasons(RIG, C0)[0] == []
    for name in ("refcv9-e2", "refcv9-full", "refcv9-full-d1"):
        # THE FLIP (MM 2026-11-07, H-R9-E1-T1H KEPT): T1h's four flags are ADMITTED in the full family as ONE UNIT; the shuffle control (T1hs) and the lags value stay refused
        p = LG.PROFILES[name]
        nr = lambda a: LG.set_flag(a, "--r9-perc-rig", None)                      # the E1 arm argvs carry the rig flag, which the full family refuses on its own (refcv8's entry)  # noqa: E731
        bw = lambda rs: [x for x in rs if "bev-warp" in x]                        # only the T1 reasons: an E1 argv differs from a full-model argv in many other values  # noqa: E731
        assert LG.forbidden_lever_reasons(p, nr(T1H))[0] == [] and bw(LG.required_on_reasons(p, nr(T1H))[0]) == [], name
        reasons, det = LG.forbidden_lever_reasons(p, nr(T1HS))
        assert det["argv_hits"] == ["--r9-bev-warp-shuffle"] and any("--r9-bev-warp-shuffle is passed, but it is T1hs's CONTENT control" in r for r in reasons), (name, reasons)
        for lone in (LG.set_flag(C0, SUP, []), LG.set_flag(C0, FUSE, ["identity"]), LG.set_flag(C0, FILL, [])):       # a bundle flag without the warp: refused by the bundle rule
            assert bw(LG.forbidden_lever_reasons(p, nr(lone))[0]) == [] and any(x.startswith("--r9-bev-warp (a switch: no value) is REQUIRED") for x in LG.required_on_reasons(p, nr(lone))[0]), name
        assert not any("bev-warp" in r for r in LG.forbidden_lever_reasons(p, C0)[0])
    for typo in ("--r9-bev-warp-supervise-hist2", "--r9-bev-warp-supervise"):          # the exception is EXACT
        ids = [x.split(":")[0] for x in LG.open_item_reasons(RIG, LG.set_flag(T1H, typo, []))]
        assert ids == ["OPEN ITEM E1-BOX-OWN-RECORD"], (typo, ids)
    for name, p in LG.PROFILES.items():                                                  # every other profile is untouched by the flag
        if name in ("refcv9-e1rig", "refcv9-e2", "refcv9-full", "refcv9-full-d1"):
            continue
        assert LG.open_item_reasons(p, T1H) == LG.open_item_reasons(p, T1), name
        assert LG.forbidden_lever_reasons(p, T1H)[0] == LG.forbidden_lever_reasons(p, T1)[0], name


def test_the_harnesses_do_not_build_T1h_the_build_argv_drops_the_flag_and_stamps_it():
    for argv in (T1H, T1HS):
        build, present = G.strip_t1(argv)
        assert SUP not in build and present == [f for f in G.T1_FLAGS if f in argv]
        assert build == [t for t in argv if t not in present and t != "identity"], "nothing else is touched (the flag AND the value token of --r9-bev-warp-fuse-init are dropped)"
        assert "identity" not in build and FUSE not in build and FILL not in build
    base_t1, _ = G.strip_t1(T1)
    base_t1h, _ = G.strip_t1(T1H)
    assert [f for f, _ in LG.flag_pairs(base_t1h)] == [f for f, _ in LG.flag_pairs(base_t1)], \
        "T1h's effective BUILD argv has the same flags as T1's: the E1-REUSE identity (`--reuse-from T1`) holds up to --out and the per-arm stamps"
    assert G.b3_record_fields(T1H)[G.T1_STAMP] == ["--r9-bev-warp", SUP, FUSE, FILL]
    assert G.b3_record_fields(T1HS)[G.T1_STAMP] == ["--r9-bev-warp", "--r9-bev-warp-shuffle", SUP, FUSE, FILL]


def test_G_EVAL_reads_the_loss_mask_state_off_both_builds_and_names_a_disagreement():
    on, _, a_on = _built(True)
    on2, _, a_on2 = _built(True)
    off, _, a_off = _built(False)
    r, d = LG.rig_eval_t1_build_reasons(a_on, on, a_on2, on2)
    assert r == [] and d["trainer"]["supervise_hist"] is True and d["trainer"]["fuse_emit_hist"] is True and d["trainer"] == d["loader"]
    r, _ = LG.rig_eval_t1_build_reasons(a_on, on, a_off, off)                           # trainer T1h, loader T1 (the loader forgot the flag)
    assert [x.split(" differs")[0] for x in r] == ["T1 (--r9-bev-warp): supervise_hist", "T1 (--r9-bev-warp): fuse_emit_hist"], r
    r, _ = LG.rig_eval_t1_build_reasons(a_off, off, a_off, off)
    assert r == [], "T1 without the flag: both builds say False/False, inert for the existing arms"
