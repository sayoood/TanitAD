"""refcv9 K2 -- T1h's history supervision INSIDE the BANDED map loss (``--r9-bev-warp-supervise-hist`` WITH ``--r9-map-banded``).  CPU, the tiny REAL FULL model.

WHAT THE BANDED MASK IS (read from source, file:line at the tree commit e0cc3261 -- see HANDOFF.md):
  ``unobs`` = ``~lift_valid`` (``MapHiresBranch._forward_banded``: ``lv = valid.any(dim=1)``; ``map_banded.unobs_masks(lv, geom)``) resampled to each band: near = the 0.25 m cell its 10 cm centre falls in
  (nearest-exact), mid = the lift cell itself, far = observed only when ALL FOUR of its 0.25 m cells are.  The loss supervises ``seen & ~unobs`` (``map_banded.supervised_mask``) and pulls ``unobs`` cells to
  p = 0.5 (max-entropy).  ``--map-lift-valid-mask`` never reaches this path (``_map_banded_loss`` reads ``unobs`` only; ``--no-map-lift-valid-mask`` is REFUSED with the banded head,
  ``map_banded_flag.pin_args``).  ``hist_valid`` is at the SAME 0.25 m lift grid (``[B, X4, Y4]``, ``bev_warp.hist_valid_of``), NOT at 10 cm.

K2 = ``unobs_loss = unobs_masks(lift_valid_now | hist_valid)``: the OR at the lift grid (where T1h defines it), then the head's OWN resampling (``bev_warp_flag.supervised_banded_unobs``).

THE CLASS ("built, tested, and UNREACHABLE FROM ITS CALLER", CLAUDE.md 2026-09-10/11).  Hop by hop, each one proved below:
  argv (the T1h bundle + ``--r9-map-banded`` [+ ``--r9-map-banded-near-m 20 --map-hires-near-lift-m 20``]) -> the trainer's own parser -> ``pin_args`` (no banded refusal) -> ``attach`` (``emit_hist``,
  ``BevWarpState.supervise_hist``) -> ``lift_hook`` -> ``MapHiresBranch.forward`` (``map_hires_hist_valid`` beside ``map_banded``) -> ``_bev_hook`` (``pout``) -> ``_map_banded_loss``
  (``_bw.supervised_banded_unobs``) -> ``banded_map_loss`` (``seen & ~unobs``) -> ``.backward()``.

  REACH       a cell that is hist-valid and NOT lift-valid-now gets a NON-ZERO gradient through the banded loss in EACH band (near, mid, far) at near 10 and near 20; the SAME cell with the flag
              OFF reads EXACTLY 0.0 (the control); a cell neither lift- nor hist-valid reads EXACTLY 0.0 with the flag ON (the OR is not "supervise everything")
  UNSET       the banded argv WITHOUT the flag: ``supervised_banded_unobs`` returns the emitted list itself; loss, every gradient and the extra keys are bit-identical to the tree's
              ``_map_banded_loss``; T1h on the 10 cm head: bit-identical to the tree's ``_map_hires_loss``
  MUTATION    the OR dropped; the mid band reading another band's mask; the near band reading the mid band's; the refusal re-added; hist_valid applied with the flag OFF -- each goes RED
"""
from __future__ import annotations

import dataclasses as dc
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
import test_bev_warp_full_model as FM  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_V2  # noqa: E402
from tanitad.models import map_banded as MB  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.train import bev_warp_flag as BF  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_banded_flag as MF  # noqa: E402

_REAL_BANDED = T._map_banded_loss              # the trainer's REAL banded loss, captured at import (recorders below patch the module attribute per test)
_REAL_HIRES = T._map_hires_loss
FINE_HW = (600, 320)
BANDS = MB.BAND_NAMES

#: the T1h bundle exactly as r9_e1_make_argv.py spells it (fuse-init identity + fill-only included), on the banded head
T1H_BUNDLE = ["--r9-bev-warp", "--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "identity", "--r9-bev-warp-fill-only"]
BASE_ARGV = ["--arm", "hier", "--out", "x", "--trunk", "timm", "--trunk-mode", "shared", "--map-hires", "on", "--w-map-hires", "1.0"]


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ===================================================================================================================== #
# helpers                                                                                                               #
# ===================================================================================================================== #
def _argv(*, banded=True, sup=True, near=10, extra=()):
    a = list(BASE_ARGV)
    if banded:
        a += ["--r9-map-banded"] + (["--r9-map-banded-near-m", str(near)] if near != 10 else [])
        a += ["--map-hires-near-lift-m", str(near)]
    a += [f for f in T1H_BUNDLE if sup or f != "--r9-bev-warp-supervise-hist"]
    return a + list(extra)


def _parse(**kw):
    """The TRAINER'S OWN parser on the arm's argv -> the namespace every later hop reads."""
    return T.build_parser().parse_args(_argv(**kw))


def _build(args):
    """A NEW tiny REAL FULL model BUILT FROM the parsed namespace's declarations (near end, banded or not), the T1 fuse attached through the flag's own ``attach``.  -> (model, cfg)."""
    near = MF.near_m(args) if MF.is_on(args) else 20.0
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k", trunk_pretrained=False, in_channels=9, image_size=64, image_width=None, trunk_frozen_bn=True)
    model = v3.RefCV3Model(cfg)
    model.core.encoder.enable_s8_tap()
    kw = dict(banded=True, banded_near_m=near) if MF.is_on(args) else {}
    hcfg = H.MapHiresConfig(w_map_hires=float(args.w_map_hires), x_max_m=float(EXTENT_V2.x_max_m), y_half_m=float(EXTENT_V2.y_half_m), d_lift=16, d_model=16, d_up=16, dilations=(1, 2),
                            grad_ckpt=True, near_lift_x_m=near, near_refine_blocks=1, **kw)
    model._map_hires = H.build_map_hires_branch(model, hcfg)
    model._w_map_hires = float(args.w_map_hires)
    if MF.is_on(args):
        setattr(model, MF.MODEL_ATTR, True)
    h, wd = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank({R.CLIP: RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=hcfg)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = bool(args.map_lift_valid_mask)
    model._w_map, model._w_box3d = 0.0, 0.0
    model._perception = PB.build_perception_branch(model, PB.PerceptionBranchConfig(w_map=0.0, w_box3d=0.0, bev_source="map_hires_pool"))
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():            # a TRAINED near lift / refine (zero-initialised at build: their grads would read 0)
        for n, p in model.named_parameters():
            if n.startswith(("_map_hires.near.", "_map_hires.near_refine.")):
                p.add_(0.02 * torch.randn(p.shape, generator=g))
    BF.pin_args(args)
    BF.attach(model, args)
    if BF.is_on(args):
        model._map_hires.bev_warp.gate.data.fill_(0.0)
    return model, cfg


def _batch(cfg, model, args):
    return FM._batch(cfg, model, args)


class _Rec:
    """Patch ``T._map_banded_loss`` / ``T._map_hires_loss`` with recorders around a chosen implementation; ``pout`` and ``extra`` are what the loss saw."""

    def __init__(self, monkeypatch, banded_fn=None, hires_fn=None):
        self.cap = {}
        real_b, real_h = banded_fn or _REAL_BANDED, hires_fn or _REAL_HIRES

        def wb(model, out, batch, device, extra, with_metrics=True):
            self.cap["pout"], self.cap["extra"] = out["perception"], extra
            return real_b(model, out, batch, device, extra, with_metrics)

        def wh(model, out, batch, device, extra, with_metrics=True):
            self.cap["pout"], self.cap["extra"] = out["perception"], extra
            return real_h(model, out, batch, device, extra, with_metrics)
        monkeypatch.setattr(T, "_map_banded_loss", wb)
        monkeypatch.setattr(T, "_map_hires_loss", wh)

    @property
    def pout(self):
        return self.cap["pout"]

    @property
    def extra(self):
        return self.cap["extra"]


def _variant(name, drop=(), replace=()):
    """``T.<name>`` re-compiled from its own source with regex-matched blocks deleted / replaced (an in-process mutation; no file is touched)."""
    src = inspect.getsource({"_map_banded_loss": _REAL_BANDED, "_map_hires_loss": _REAL_HIRES}[name]).replace("\r\n", "\n")
    for pat in drop:
        src, n = re.subn(pat, "", src, flags=re.M)
        assert n == 1, ("the block to mutate is not unique / not found", pat, n)
    for pat, txt in replace:
        src, n = re.subn(pat, lambda m: txt, src, flags=re.M)
        assert n == 1, ("the block to mutate is not unique / not found", pat, n)
    g = dict(T.__dict__)
    exec(compile(src, "<%s variant>" % name, "exec"), g)
    return g[name]


#: the camera is made BLIND on these x-ranges (m) so history-only cells exist in EVERY band of the tiny rig (its synthetic camera sees all of 10-60 m; MEASURED: hv & ~lv is empty beyond ~12 m
#: without a stripe).  Only ``valid`` is changed; the history, the hook, the emission and the loss are the real code.
STRIPES = {10: ((20.0, 28.0), (46.0, 54.0)), 20: ((12.0, 18.0), (26.0, 34.0), (46.0, 54.0))}
GROUPS = {"near": ("_map_hires.banded.head_near",), "mid": ("_map_hires.banded.mid.",), "far": ("_map_hires.banded.far.",)}
GATE = ("_map_hires.bev_warp.gate",)


def _blind(model, stripes):
    bank = model._lift_bank_hires
    real = bank.for_episodes
    keep = torch.ones(240, dtype=torch.bool)
    for a, b in stripes:
        for r in range(240):
            if a <= (r + 0.5) * 0.25 < b:
                keep[r] = False

    def fe(*aa, **kk):
        grid, valid = real(*aa, **kk)
        return grid, valid & keep.view(*([1] * (valid.dim() - 2)), 240, 1)
    bank.for_episodes = fe


def _codes_for(cells: dict, near: int):
    """uint8 [B, 600, 320], NOT_SEEN (255) except the fine cell at the CENTRE SAMPLE of each chosen band cell (``build_banded_targets``' rule), so ``seen`` of band b == cells[b] exactly."""
    geom = MB.band_geometry(float(EXTENT_V2.x_max_m), float(EXTENT_V2.y_half_m), float(near))
    nb = next(iter(cells.values())).shape[0]
    codes = torch.full((nb,) + FINE_HW, 255, dtype=torch.uint8)
    for g in geom:
        c = cells.get(g.name)
        if c is None:
            continue
        r0, r1 = g.fine_rows
        ratio = g.cell / 0.1
        ri = torch.floor((torch.arange(g.rows).float() + 0.5) * ratio).long().clamp(max=r1 - r0 - 1) + r0
        cj = torch.floor((torch.arange(g.cols).float() + 0.5) * ratio).long().clamp(max=FINE_HW[1] - 1)
        b_i, i_i, j_i = c.nonzero(as_tuple=True)
        codes[b_i, ri[i_i], cj[j_i]] = (1 + (i_i + j_i) % 7).to(torch.uint8)
    return codes


def _noT1_args(near):
    return T.build_parser().parse_args([a for a in _argv(near=near, sup=False) if a not in T1H_BUNDLE and a != "identity"])


def _run(mp, near, *, sup=True, bw=True, cells=None, banded_fn=None, hires_fn=None, maxent0=True):
    args = _parse(near=near, sup=sup) if bw else _noT1_args(near)
    model, cfg = _build(args)
    _blind(model, STRIPES[near])
    batch = _batch(cfg, model, args)
    if cells is not None:
        batch["map_fine"] = _codes_for(cells, near)
    if maxent0:
        mp.setattr(MB, "MAXENT_W", 0.0)                      # isolate the SUPERVISED term: with the max-entropy pull on, an OFF run has gradient from the unobserved cells
    rec = _Rec(mp, banded_fn, hires_fn)
    out = FM._loss(model, batch)
    model.zero_grad(set_to_none=True)
    out["map_hires"].backward()
    return model, rec, out


def _gnz(model, prefixes) -> float:
    return sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith(prefixes) and p.grad is not None)


_GEOM_CACHE: dict = {}


def _geometry(near):
    """(gain, neither) per band of the blind-stripe rig, from ONE real T1h-banded forward; gain = old_unobs & ~new_unobs (supervised only with the OR), neither = new_unobs (unobserved even with history)."""
    if near not in _GEOM_CACHE:
        with pytest.MonkeyPatch.context() as mp:
            model, rec, _ = _run(mp, near)
            lv, hv = rec.pout["map_hires_lift_valid"], rec.pout[BF.HIST_KEY]
            geom = model._map_hires.banded.geom
            old, new = MB.unobs_masks(lv, geom), MB.unobs_masks(lv | hv, geom)
            assert all(torch.equal(a, b) for a, b in zip(old, rec.pout["map_banded"]["unobs"])), "the head's emitted unobs is unobs_masks(lift_valid)"
        _GEOM_CACHE[near] = ({g.name: (o & ~n) for g, o, n in zip(geom, old, new)}, {g.name: n for g, n in zip(geom, new)})
    return _GEOM_CACHE[near]


def _reach_problems(near, band, kind, *, sup=True, banded_fn=None, hires_fn=None, impl=None, patch=None) -> list:
    """The REACH probe: labels ONLY on ``kind`` cells of ``band`` (``gain`` = history-only; ``neither``), backward of the unweighted banded term (max-entropy off).  -> the list of what is WRONG."""
    gain, neither = _geometry(near)
    cells = {band: (gain if kind == "gain" else neither)[band]}
    assert int(cells[band].sum()) > 0, (near, band, kind, "the probe would be vacuous")
    with pytest.MonkeyPatch.context() as mp:
        if impl is not None:
            mp.setattr(BF, "supervised_banded_unobs", impl)
        if patch is not None:
            patch(mp)
        try:
            model, rec, out = _run(mp, near, sup=sup, cells=cells, banded_fn=banded_fn, hires_fn=hires_fn)
        except SystemExit as e:
            return ["REFUSED: %s" % str(e)[:120]]
        except ValueError as e:
            return ["RAISED: %s" % str(e)[:120]]
    bad = []
    ex = rec.extra
    n = {b: ex["map_banded_n_%s" % b] for b in BANDS}
    want = float(cells[band].sum()) if (kind == "gain" and sup) else 0.0
    if n[band] != want:
        bad.append("n supervised in %s = %s, want %s" % (band, n[band], want))
    if any(n[b] != 0.0 for b in BANDS if b != band):
        bad.append("another band is supervised: %s" % n)
    g_own, g_gate = _gnz(model, GROUPS[band]), _gnz(model, GATE)
    other = sum(_gnz(model, GROUPS[b]) for b in BANDS if b != band)
    if want > 0:
        if not (g_own > 0.0 and g_gate > 0.0):
            bad.append("gradient not reached: band head %.3g gate %.3g" % (g_own, g_gate))
        if not ex.get("n_map_banded_hist_cells_%s" % band, 0.0) > 0.0:
            bad.append("hist-cell counter not logged")
    elif g_own != 0.0 or g_gate != 0.0:
        bad.append("expected EXACTLY 0.0, got band head %.3g gate %.3g" % (g_own, g_gate))
    if other != 0.0:
        bad.append("the other bands' heads received gradient %.3g (must be exactly 0.0)" % other)
    return bad


def _hist_absent(rec) -> bool:
    return BF.HIST_KEY not in rec.pout


# ===================================================================================================================== #
# the mask, independently                                                                                               #
# ===================================================================================================================== #
def _ref_unobs(v, geom):
    """An INDEPENDENT unobs reference: plain integer index arithmetic, no interpolate / pooling.  v = the lift validity [B, X4, Y4] bool."""
    out = []
    for g in geom:
        i, j = torch.arange(g.rows), torch.arange(g.cols)
        if g.name == "near":                                   # a 0.10 m cell -> the 0.25 m cell holding its centre: floor((i + .5) * .1 / .25) = (2i + 1) // 5
            out.append(~v[:, (2 * i + 1) // 5][:, :, (2 * j + 1) // 5])
        elif g.name == "mid":                                  # the lift cell itself, rows offset by the band start
            out.append(~v[:, int(round(g.x0 / 0.25)) + i][:, :, j])
        else:                                                  # far: observed only when ALL FOUR of its 0.25 m cells are
            r2 = int(round(g.x0 / 0.25))
            a, b = r2 + 2 * i, r2 + 2 * i + 1
            out.append(~(v[:, a][:, :, 2 * j] & v[:, a][:, :, 2 * j + 1] & v[:, b][:, :, 2 * j] & v[:, b][:, :, 2 * j + 1]))
    return out


class _Stub:
    """A model that carries only what ``supervised_banded_unobs`` reads: the T1 state (supervise_hist on / off)."""

    def __init__(self, sup):
        setattr(self, BF.MODEL_ATTR, BF.BevWarpState((0.5, 1.0), (5, 10), False, bool(sup)))


@pytest.mark.parametrize("near", [10, 20])
def test_the_OR_equals_an_independent_reference_on_the_production_geometry_and_only_ever_adds_cells(near):
    geom = MB.band_geometry(100.0, 30.0, float(near))
    g = torch.Generator().manual_seed(5)
    lv = torch.rand(3, 400, 240, generator=g) > 0.5
    hv = torch.rand(3, 400, 240, generator=g) > 0.5
    sel = torch.tensor([0, 2])
    emitted = [m.index_select(0, sel) for m in MB.unobs_masks(lv, geom)]
    pout = {"map_hires_lift_valid": lv, BF.HIST_KEY: hv}
    got = BF.supervised_banded_unobs(_Stub(True), pout, emitted, sel, geom)
    ref = _ref_unobs((lv | hv).index_select(0, sel), geom)
    for gg, a, b, e in zip(geom, got, ref, emitted):
        assert tuple(a.shape[1:]) == (gg.rows, gg.cols) and torch.equal(a, b), gg.name                 # the OR, resampled the head's way
        assert not bool((a & ~e).any()), "the OR only ADDS observed cells: new unobs is a subset of the emitted unobs"
        assert int(a.sum()) < int(e.sum()), "and it added some"
    for gg, e, r0 in zip(geom, emitted, _ref_unobs(lv.index_select(0, sel), geom)):
        assert torch.equal(e, r0), "the reference reproduces the head's own emitted unobs (so the comparison above is like for like)"
    assert BF.supervised_banded_unobs(_Stub(False), pout, emitted, sel, geom) is emitted               # OFF: the SAME list object (bit-identical)
    for bad in ({"map_hires_lift_valid": lv}, {BF.HIST_KEY: hv}):                                         # ON with a missing emission REFUSES
        with pytest.raises(SystemExit, match="silently supervise only"):
            BF.supervised_banded_unobs(_Stub(True), bad, emitted, sel, geom)
    with pytest.raises(SystemExit, match="not unobs_masks"):                                              # the drift guard: a head whose emitted unobs is not unobs_masks(lift_valid)
        BF.supervised_banded_unobs(_Stub(True), pout, [~m for m in emitted], sel, geom)


def test_the_FAR_band_rule_is_ALL_four_cells_seen_now_or_in_history_never_ANY():
    geom = MB.band_geometry(100.0, 30.0, 10.0)
    r2 = int(round(geom[2].x0 / 0.25))                         # the far band's first lift row (160)
    lv = torch.zeros(1, 400, 240, dtype=torch.bool)
    hv = torch.zeros(1, 400, 240, dtype=torch.bool)
    lv[0, r2, 0], lv[0, r2, 1], lv[0, r2 + 1, 0] = True, True, True
    hv[0, r2 + 1, 1] = True                                    # far cell (0,0): 3 seen now + 1 by history -> OBSERVED
    lv[0, r2, 2] = True
    hv[0, r2, 3] = True                                        # far cell (0,1): 1 seen now + 1 by history, 2 by neither -> still UNOBSERVED (an ANY rule would supervise it)
    sel = torch.tensor([0])
    emitted = MB.unobs_masks(lv, geom)
    got = BF.supervised_banded_unobs(_Stub(True), {"map_hires_lift_valid": lv, BF.HIST_KEY: hv}, emitted, sel, geom)[2]
    assert bool(emitted[2][0, 0, 0]) and not bool(got[0, 0, 0]), "3 now + 1 history: observed with the OR, unobserved without"
    assert bool(emitted[2][0, 0, 1]) and bool(got[0, 0, 1]), "1 now + 1 history of 4: NOT observed (the head's conservative ALL rule)"


# ===================================================================================================================== #
# the hops, from the argv                                                                                               #
# ===================================================================================================================== #
@pytest.mark.parametrize("near", [10, 20])
def test_ARGV_the_T1h_bundle_plus_the_banded_head_parses_pins_builds_and_agrees_with_G_DVB(near):
    args = _parse(near=near)
    assert args.r9_bev_warp and args.r9_bev_warp_supervise_hist and args.r9_map_banded and args.r9_bev_warp_fill_only and args.r9_bev_warp_fuse_init == "identity"
    assert MF.near_m(args) == float(near) and float(args.map_hires_near_lift_m) == float(near) and args.map_lift_valid_mask is True
    BF.pin_args(args)                                          # K2: no banded refusal (the refusal at bev_warp_flag.py:178-179 of the tree is gone)
    MF.pin_args(args)
    model, _ = _build(args)
    st = getattr(model, BF.MODEL_ATTR)
    assert st.supervise_hist and model._map_hires.bev_warp.emit_hist and model._map_hires.banded is not None and model._map_hires.banded.geom[0].rows == 100 * (near // 10)
    assert BF.dvb_check_supervise_hist(model, args) == [] and MF.dvb_check(model, args) == [] and MF.dvb_near_m(model, args) == []
    assert BF.stamp(model, args)["supervise_hist"]["banded"].startswith("K2:")
    for kw, pat in ((dict(r9_bev_warp=False), "dead flag"), (dict(map_lift_valid_mask=False), "--no-map-lift-valid-mask"), (dict(ablate_frames=True), "ablate")):
        bad = _parse(near=near)
        for k, v in kw.items():
            setattr(bad, k, v)
        with pytest.raises(SystemExit, match=pat):
            BF.pin_args(bad)                                   # every OTHER refusal stands


# ===================================================================================================================== #
# REACH: each band, near 10 and near 20, with the zero controls                                                         #
# ===================================================================================================================== #
@pytest.mark.parametrize("near", [10, 20])
@pytest.mark.parametrize("band", BANDS)
def test_REACH_a_history_only_cell_gives_a_NON_ZERO_gradient_in_every_band_and_EXACTLY_0_with_the_flag_OFF(near, band):
    assert _reach_problems(near, band, "gain") == [], "flag ON: the hist-only cells are supervised and the band head AND the gate hear them"
    assert _reach_problems(near, band, "gain", sup=False) == [], "CONTROL: the SAME labels with the flag OFF read EXACTLY 0.0"
    assert _reach_problems(near, band, "neither") == [], "CONTROL: cells neither seen now nor in history read EXACTLY 0.0 with the flag ON"


def test_the_emitted_unobs_is_not_widened_and_the_LOSSs_is_and_the_counters_say_so():
    with pytest.MonkeyPatch.context() as mp:
        model, rec, _ = _run(mp, 10, maxent0=False)
    lv, hv = rec.pout["map_hires_lift_valid"], rec.pout[BF.HIST_KEY]
    assert tuple(lv.shape) == tuple(hv.shape) == (2, 240, 128) and lv.dtype == hv.dtype == torch.bool, "hist_valid is at the 0.25 m LIFT grid, not at 10 cm"
    geom = model._map_hires.banded.geom
    old, new = _ref_unobs(lv, geom), _ref_unobs(lv | hv, geom)
    assert all(torch.equal(a, b) for a, b in zip(old, rec.pout["map_banded"]["unobs"])), "the EMITTED unobs (plan wires / dumps read it) is NOT widened"
    for g, o, nn in zip(geom, old, new):
        assert rec.extra["n_map_banded_hist_cells_%s" % g.name] == float((o & ~nn).sum())
        assert abs(rec.extra["map_banded_unobs_frac_%s" % g.name] - float(nn.float().mean())) < 1e-6, "the LOSS's unobs (and so its max-entropy set) is the widened one"
    with pytest.MonkeyPatch.context() as mp:
        _, rec_off, _ = _run(mp, 10, sup=False, maxent0=False)
    assert _hist_absent(rec_off) and not any(k.startswith("n_map_banded_hist_cells") for k in rec_off.extra), "OFF: no key is emitted, no counter is logged"
    assert all(abs(rec_off.extra["map_banded_unobs_frac_%s" % g.name] - float(o.float().mean())) < 1e-6 for g, o in zip(geom, old))


# ===================================================================================================================== #
# UNSET identity                                                                                                        #
# ===================================================================================================================== #
K2_BLOCK_RE = (r'^    _un_now = sub\["unobs"\]\n    sub\["unobs"\] = _bw\.supervised_banded_unobs\([^\n]*\n    if sub\["unobs"\] is not _un_now:[^\n]*\n'
               r'        for _bn, _a, _b in zip[^\n]*\n            extra\["n_map_banded_hist_cells_%s" % _bn\][^\n]*\n')
DISPATCH_RE = r'^    if bool\(getattr\(model, _mbf\.MODEL_ATTR, False\)\):[^\n]*\n'
OLD_REFUSAL = '    if bool(getattr(model, _mbf.MODEL_ATTR, False)):\n        if _bw.supervise_hist_on(model):\n            raise SystemExit("[v3] banded+T1h refused (the tree)")\n'


def _tensors(out) -> dict:
    return {k: v for k, v in out.items() if torch.is_tensor(v)}


def _identity_problems(near, *, banded=True, sup=False, patch=None) -> list:
    """Two FRESH identical models, one loss each (core.r8_gen advances per forward): the REAL trainer vs the TREE's loss (the K2 lines deleted / the old refusal restored).
    Loss dict, every gradient and the extra key set must be bit-identical."""
    res = []
    for which in ("new", "tip"):
        with pytest.MonkeyPatch.context() as mp:
            if patch is not None and which == "new":
                patch(mp)
            fn_b = _variant("_map_banded_loss", drop=(K2_BLOCK_RE,)) if which == "tip" else None
            fn_h = _variant("_map_hires_loss", replace=((DISPATCH_RE, OLD_REFUSAL),)) if which == "tip" else None
            args = _parse(near=near, sup=sup) if banded else T.build_parser().parse_args(BASE_ARGV + T1H_BUNDLE + ["--map-hires-near-lift-m", "20"])
            model, cfg = _build(args)
            batch = _batch(cfg, model, args)
            rec = _Rec(mp, fn_b, fn_h)
            out = FM._loss(model, batch)
            model.zero_grad(set_to_none=True)
            out["loss"].backward()
            res.append((_tensors(out), {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}, set(rec.extra)))
    (ta, ga, ea), (tb, gb, eb) = res
    bad = []
    if set(ta) != set(tb) or not all(torch.equal(ta[k], tb[k]) for k in ta):
        bad.append("loss dict differs from the tree's")
    if set(ga) != set(gb) or not all((ga[n] is None and gb[n] is None) or (ga[n] is not None and gb[n] is not None and torch.equal(ga[n], gb[n])) for n in ga):
        bad.append("a gradient differs from the tree's")
    if ea != eb:
        bad.append("extra keys differ: %s" % sorted(ea ^ eb))
    return bad


@pytest.mark.parametrize("near", [10, 20])
def test_UNSET_a_banded_argv_WITHOUT_supervise_hist_is_bit_identical_to_the_trees_banded_loss(near):
    assert _identity_problems(near, sup=False) == []


def test_UNSET_a_banded_argv_with_NO_T1_at_all_emits_no_hist_and_builds_no_T1_state():
    with pytest.MonkeyPatch.context() as mp:
        model, rec, _ = _run(mp, 10, bw=False, maxent0=False)
    assert _hist_absent(rec) and getattr(model, BF.MODEL_ATTR, None) is None and not any(k.startswith("n_map_banded_hist_cells") for k in rec.extra)


def test_UNSET_T1h_on_the_10cm_head_with_no_banded_is_bit_identical_to_the_trees_loss():
    assert _identity_problems(20, banded=False, sup=True) == []


# ===================================================================================================================== #
# MUTATIONS: each must turn a guard RED                                                                                 #
# ===================================================================================================================== #
_REAL_SBU = getattr(BF, "supervised_banded_unobs", None)      # None on the TREE (the evidence script imports this module there too)


def _resize(m, like):
    return torch.nn.functional.interpolate(m[:, None].float(), size=tuple(like.shape[1:]), mode="nearest")[:, 0] > 0.5


def test_MUT_the_OR_dropped_from_the_banded_loss_turns_REACH_RED():
    assert _reach_problems(10, "near", "gain") == [], "the unmutated loss is GREEN on this very probe"
    for band in BANDS:
        bad = _reach_problems(10, band, "gain", impl=lambda model, pout, unobs, sel, geom: unobs)
        assert bad and "n supervised" in bad[0], (band, bad)
    tip = _variant("_map_banded_loss", drop=(K2_BLOCK_RE,))
    assert "supervised_banded_unobs" in _REAL_BANDED.__code__.co_names and "supervised_banded_unobs" not in tip.__code__.co_names
    assert _reach_problems(10, "mid", "gain", banded_fn=tip), "the OR block deleted from the real loss's SOURCE"


def test_MUT_the_mid_band_reading_another_bands_mask_and_the_near_band_reading_the_mids_turn_REACH_RED():
    def mid_from_near(model, pout, unobs, sel, geom):             # the mid band takes the 10 cm near mask, crudely resized instead of its own lift cell
        r = _REAL_SBU(model, pout, unobs, sel, geom)
        return [r[0], _resize(r[0], r[1]), r[2]]

    def near_from_mid(model, pout, unobs, sel, geom):             # the near band takes the mid band's mask
        r = _REAL_SBU(model, pout, unobs, sel, geom)
        return [_resize(r[1], r[0]), r[1], r[2]]

    def far_any(model, pout, unobs, sel, geom):                   # the far band under an ANY rule (a different semantic from the head's ALL)
        r = _REAL_SBU(model, pout, unobs, sel, geom)
        v = (pout["map_hires_lift_valid"] | pout[BF.HIST_KEY]).index_select(0, sel)
        r2 = int(round(geom[2].x0 / 0.25))
        return [r[0], r[1], ~(torch.nn.functional.max_pool2d(v[:, r2:][:, None].float(), 2)[:, 0] > 0.5)]
    assert _reach_problems(10, "mid", "gain", impl=mid_from_near), "mid reads the near mask"
    assert _reach_problems(10, "near", "gain", impl=near_from_mid), "near reads the mid mask"
    # far under ANY instead of ALL: the REACH probe labels exactly the ALL-gain cells, so a SUPERSET mask is invisible to it -- the rule is pinned by the reference / hand-built far-cell tests instead, and THIS mutant breaks both
    geom = MB.band_geometry(100.0, 30.0, 10.0)
    r2 = int(round(geom[2].x0 / 0.25))
    lv, hv = torch.zeros(1, 400, 240, dtype=torch.bool), torch.zeros(1, 400, 240, dtype=torch.bool)
    lv[0, r2, 2], hv[0, r2, 3] = True, True                                  # far cell (0,1): 1 now + 1 history of 4
    sel = torch.tensor([0])
    pout = {"map_hires_lift_valid": lv, BF.HIST_KEY: hv}
    emitted = MB.unobs_masks(lv, geom)
    assert bool(_REAL_SBU(_Stub(True), pout, emitted, sel, geom)[2][0, 0, 1]), "ALL: still unobserved"
    assert not bool(far_any(_Stub(True), pout, emitted, sel, geom)[2][0, 0, 1]), "the ANY mutant supervises it: the hand-built far-cell test would go RED"
    g = torch.Generator().manual_seed(5)
    lv, hv = torch.rand(2, 400, 240, generator=g) > 0.5, torch.rand(2, 400, 240, generator=g) > 0.5
    pout = {"map_hires_lift_valid": lv, BF.HIST_KEY: hv}
    ref = _ref_unobs(lv | hv, geom)[2]
    assert torch.equal(_REAL_SBU(_Stub(True), pout, MB.unobs_masks(lv, geom), torch.arange(2), geom)[2], ref)
    assert not torch.equal(far_any(_Stub(True), pout, MB.unobs_masks(lv, geom), torch.arange(2), geom)[2], ref), "and the independent-reference test goes RED"


def test_MUT_the_refusal_re_added_at_any_of_its_three_sites_turns_REACH_RED():
    refuse_loss = _variant("_map_hires_loss", replace=((DISPATCH_RE, OLD_REFUSAL),))
    assert _reach_problems(10, "near", "gain", hires_fn=refuse_loss), "the trainer's dispatch refusal (refc_v3_train.py:5286-5287 of the tree)"
    real_pin = BF.pin_args

    def pin_with_old_refusal(a):
        if getattr(a, "r9_bev_warp_supervise_hist", False) and getattr(a, "r9_map_banded", False):
            raise SystemExit("[bev-warp] --r9-bev-warp-supervise-hist with --r9-map-banded: refused")
        return real_pin(a)
    assert _reach_problems(10, "near", "gain", patch=lambda mp: mp.setattr(BF, "pin_args", pin_with_old_refusal)), "pin_args's refusal (bev_warp_flag.py:178-179 of the tree)"
    real_fwd = H.MapHiresBranch.forward

    def fwd_with_old_refusal(self, *a, **k):
        o = real_fwd(self, *a, **k)
        if BF.HIST_KEY in o and self.banded is not None:
            raise ValueError("[map-hires] --r9-bev-warp-supervise-hist is not implemented for the banded head")
        return o
    assert _reach_problems(10, "near", "gain", patch=lambda mp: mp.setattr(H.MapHiresBranch, "forward", fwd_with_old_refusal)), "MapHiresBranch.forward's refusal (map_head_hires.py:967-968 of the tree)"


def test_MUT_hist_valid_applied_with_supervise_hist_OFF_turns_the_UNSET_identity_RED():
    assert _identity_problems(10, sup=False) == [], "the unmutated OFF run is GREEN on this very check"

    def always_on(mp):
        mp.setattr(BF, "supervise_hist_on", lambda m: True)                 # the OR applies whatever the model carries ...
        real_attach = BF.attach

        def attach_emit(model, args):
            real_attach(model, args)
            model._map_hires.bev_warp.emit_hist = True                       # ... and the fuse emits hist_valid although the flag is OFF
        mp.setattr(BF, "attach", attach_emit)
    assert _identity_problems(10, sup=False, patch=always_on), "history-widened supervision with the flag OFF must break bit-identity"
