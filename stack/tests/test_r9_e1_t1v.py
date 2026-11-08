"""refcv9 E1 arm T1v -- ``--r9-hist-supervise-geom``: T1h's SUPERVISION SET from GEOMETRY ALONE (REFCV9_E1_T1V_PREREG.md).  CPU, the tiny REAL model of ``test_map_only_ft._build()`` /
``test_r9_k2_bandhist._build()`` (the trainer's OWN parser on the arm's argv -> the trainer's own loss functions).

WHY THE LEVER EXISTS (MEASURED by the MM, T1hs CONTROL-READ): T1h's +0.295 gain is the supervision of history-reachable camera-blind cells, the mask is geometry only
(``bev_warp.hist_valid_of``), and the past-frame encode + lift + warp + fuse T1h also runs costs +0.6 s/step for content that moved nothing.  T1v = C0 + ONE flag that builds the SAME
mask with none of that: the 10 cm map loss's (and the banded loss's) lift-valid mask is ``lift_valid_now | hist_valid_geom``.

THE CLASS ("built, tested, and UNREACHABLE FROM ITS CALLER", CLAUDE.md 2026-09-10/11).  Hop by hop, each one proved here:

  argv ``--r9-hist-supervise-geom`` (``T.build_parser``) -> ``pin_args`` -> ``attach`` (``BW.HistGeom`` on ``model._map_hires.hist_geom`` + ``model._r9_hist_geom``) -> ``attach_dataset``
  (``ItemSource(with_frames=False)``) -> ``V3Dataset.__getitem__`` (``item_fields``: poses + has) -> ``_bw.stage`` (``HistGeom.stage``) -> ``MapHiresBranch.forward``
  (``HistGeom.take_hist_valid`` -> ``map_hires_hist_valid``) -> ``_map_hires_loss`` (``supervised_lift_valid``) / ``_map_banded_loss`` (``supervised_banded_unobs``) -> the CE -> ``.backward()``

  IDENTITY   on the SAME batch, T1v's ``hist_valid`` == T1h's BIT FOR BIT (full model AND perception rig; the 10 cm head AND the banded head at near 10 and near 20), and so is what the two
             losses actually feed the CE (``supervised_lift_valid`` / ``supervised_banded_unobs`` outputs)
  REACH      a cell that is hist-valid and NOT lift-valid-now gives the map head a NON-ZERO gradient under T1v, EXACTLY 0.0 under C0 (and EXACTLY 0.0 on a cell neither mask reaches)
  NOT BUILT  no ``bev_warp`` parameter, ``state_dict`` keys == C0's, no past-frame encode / lift / warp_bev (CALL COUNTS against C0 and against T1h as the positive control), the loader reads no past image
  GATE       a G-DVB entry (``built``) with its deliberate-regression arms; the refusals
  MUTATION   each of the above goes RED when its own line is broken (the mask from another T / other lags, a past frame encoded, the OR dropped, the G-DVB entry removed, the banded path
             ignoring the mask)
"""
from __future__ import annotations

import dataclasses as dc
import sys
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_bev_warp as TB  # noqa: E402
import test_bev_warp_full_model as FM  # noqa: E402
import test_bev_warp_supervise_hist as SH  # noqa: E402
import test_map_only_ft as R  # noqa: E402
import test_r9_k2_bandhist as K  # noqa: E402
from tanitad.models import bev_warp as BW  # noqa: E402
from tanitad.models import map_banded as MB  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.train import bev_warp_flag as BF  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402

T1H = list(K.T1H_BUNDLE)                    # the T1h bundle exactly as r9_e1_make_argv.py spells it
T1V = ["--r9-hist-supervise-geom"]
HIST = BF.HIST_KEY
BANDS = MB.BAND_NAMES
GEOM_DEST = "r9_hist_supervise_geom"


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ===================================================================================================================== #
# helpers                                                                                                               #
# ===================================================================================================================== #
def _parse(flags, near=10, banded=True, drop=()):
    """The TRAINER'S OWN parser on the arm's argv (C0 = ``K.BASE_ARGV``; the banded head optional) -> the namespace every later hop reads."""
    a = [x for x in K.BASE_ARGV]
    if banded:
        a += ["--r9-map-banded"] + (["--r9-map-banded-near-m", str(near)] if near != 10 else [])
        a += ["--map-hires-near-lift-m", str(near)]
    a += list(flags)
    for d in drop:
        a = [x for x in a if x != d]
    return T.build_parser().parse_args(a)


def _model(mp, args, rig=False):
    """A NEW tiny REAL model built FROM the parsed namespace, the flag's own ``pin_args`` + ``attach`` run.  ``rig`` = the perception RIG model (E1's real arms are rig models)."""
    if rig:
        mp.setattr(v3, "RefCV3Model", lambda cfg: TB._ORIG_MODEL(cfg, perc_rig=True))
        model, cfg = R._build()
        BF.pin_args(args)
        BF.attach(model, args)
        if BF.is_on(args):
            model._map_hires.bev_warp.gate.data.fill_(0.0)
        return model, cfg
    return K._build(args)


class Run:
    """One forward + (optionally) backward of the arm; what the losses saw."""


def _run(mp, flags, *, near=10, banded=True, rig=False, cells=None, blind=True, backward=True, maxent0=True, pre=None, drop=()):
    r = Run()
    r.args = _parse(flags, near, banded, drop)
    r.model, cfg = _model(mp, r.args, rig)
    if blind and not rig:
        K._blind(r.model, K.STRIPES[near])
    if pre is not None:
        pre(r.model)
    batch = FM._batch(cfg, r.model, r.args)
    r.batch = batch
    if cells is not None:
        batch["map_fine"] = cells(near) if callable(cells) else cells
    if maxent0:
        mp.setattr(MB, "MAXENT_W", 0.0)
    r.rec = K._Rec(mp)
    r.lv_sup, r.unobs_sup = [], []
    real_lv, real_un = BF.supervised_lift_valid, BF.supervised_banded_unobs

    def spy_lv(model, pout, lv, sel):
        out = real_lv(model, pout, lv, sel)
        r.lv_sup.append(None if out is None else out.clone())
        return out

    def spy_un(model, pout, unobs, sel, geom):
        out = real_un(model, pout, unobs, sel, geom)
        r.unobs_sup.append([x.clone() for x in out])
        return out
    mp.setattr(BF, "supervised_lift_valid", spy_lv)
    mp.setattr(BF, "supervised_banded_unobs", spy_un)
    r.out = FM._loss(r.model, batch)
    if backward:
        r.model.zero_grad(set_to_none=True)
        (r.out["map_hires"] if "map_hires" in r.out else r.out["loss"]).backward()
    return r


def _gnz(model, prefixes) -> float:
    return sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith(prefixes) and p.grad is not None)


def _same(a, b) -> bool:
    return a is not None and b is not None and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a, b)


# ===================================================================================================================== #
# 1. IDENTITY with T1h's mask                                                                                           #
# ===================================================================================================================== #
def _identity_problems(near, banded, rig=False, patch=None) -> list:
    with pytest.MonkeyPatch.context() as mp:
        h = _run(mp, T1H, near=near, banded=banded, rig=rig)
    with pytest.MonkeyPatch.context() as mp:
        if patch is not None:
            patch(mp)
        v = _run(mp, T1V, near=near, banded=banded, rig=rig)
    bad = []
    hh, hv = h.rec.pout[HIST], v.rec.pout.get(HIST)
    if hv is None:
        return ["T1v emitted no %s" % HIST]
    if not _same(hh, hv):
        bad.append("hist_valid differs: %d vs %d cells set" % (int(hh.sum()), int(hv.sum())))
    if not _same(h.rec.pout["map_hires_lift_valid"], v.rec.pout["map_hires_lift_valid"]):
        bad.append("lift_valid_now differs")
    lv = h.rec.pout["map_hires_lift_valid"]
    if not (int((hh & ~lv).sum()) > 0 and int((~hh & ~lv).sum()) > 0 and int((hh & lv).sum()) > 0):
        bad.append("VACUOUS: the batch has no history-only / neither / both cells")
    if not banded:
        if not (len(h.lv_sup) == len(v.lv_sup) == 1 and _same(h.lv_sup[0], v.lv_sup[0])):
            bad.append("the mask the 10 cm loss supervises (supervised_lift_valid) differs")
        if h.rec.extra.get("n_map_hires_hist_cells025") != v.rec.extra.get("n_map_hires_hist_cells025"):
            bad.append("n_map_hires_hist_cells025 differs: %s vs %s" % (h.rec.extra.get("n_map_hires_hist_cells025"), v.rec.extra.get("n_map_hires_hist_cells025")))
    else:
        if not (len(h.unobs_sup) == len(v.unobs_sup) == 1 and all(_same(a, b) for a, b in zip(h.unobs_sup[0], v.unobs_sup[0]))):
            bad.append("the per-band mask the banded loss supervises (supervised_banded_unobs) differs")
        for b in BANDS:
            k = "n_map_banded_hist_cells_%s" % b
            if h.rec.extra.get(k) != v.rec.extra.get(k):
                bad.append("%s differs: %s vs %s" % (k, h.rec.extra.get(k), v.rec.extra.get(k)))
    return bad


@pytest.mark.parametrize("rig", [False, True], ids=["full", "rig"])
def test_IDENTITY_the_10cm_loss_supervises_T1hs_hist_valid_bit_for_bit(rig):
    assert _identity_problems(10, False, rig=rig) == []


@pytest.mark.parametrize("near", [10, 20])
def test_IDENTITY_the_banded_loss_supervises_T1hs_per_band_mask_bit_for_bit_at_near_10_and_20(near):
    assert _identity_problems(near, True) == []


def test_IDENTITY_the_T1v_batch_carries_no_past_image_and_T1h_does():
    with pytest.MonkeyPatch.context() as mp:
        v = _run(mp, T1V, banded=False, backward=False)
    with pytest.MonkeyPatch.context() as mp:
        h = _run(mp, T1H, banded=False, backward=False)
    assert "bw_frames" not in v.batch and set(BF.GEOM_BATCH_KEYS) <= set(v.batch)
    assert "bw_frames" in h.batch
    for k in BF.GEOM_BATCH_KEYS:
        assert torch.equal(v.batch[k], h.batch[k]), "the SAME poses and has (T1v and T1h) -- only the images are dropped"


# ===================================================================================================================== #
# 2. REACH + the zero controls                                                                                          #
# ===================================================================================================================== #
_GEO: dict = {}


def _geometry(near):
    """(gain, neither) per band of the blind-stripe rig, from ONE real T1v-banded forward: gain = old_unobs & ~new_unobs (supervised only with the OR), neither = new_unobs."""
    if near not in _GEO:
        with pytest.MonkeyPatch.context() as mp:
            r = _run(mp, T1V, near=near, backward=False)
        lv, hv = r.rec.pout["map_hires_lift_valid"], r.rec.pout[HIST]
        geom = r.model._map_hires.banded.geom
        old, new = MB.unobs_masks(lv, geom), MB.unobs_masks(lv | hv, geom)
        _GEO[near] = ({g.name: (o & ~n) for g, o, n in zip(geom, old, new)}, {g.name: n for g, n in zip(geom, new)})
    return _GEO[near]


def _banded_reach_problems(arm, near, band, kind, patch=None) -> list:
    """Labels ONLY on ``kind`` cells of ``band`` (``gain`` = history-only; ``neither``), backward of the unweighted banded term.  ``arm`` = ``t1v`` | ``c0``.  -> what is WRONG."""
    gain, neither = _geometry(near)
    cells = {band: (gain if kind == "gain" else neither)[band]}
    assert int(cells[band].sum()) > 0, (near, band, kind, "the probe would be vacuous")
    with pytest.MonkeyPatch.context() as mp:
        if patch is not None:
            patch(mp)
        try:
            r = _run(mp, T1V if arm == "t1v" else [], near=near, cells=lambda n: K._codes_for(cells, n))
        except SystemExit as e:
            return ["REFUSED: %s" % str(e)[:120]]
    ex, model = r.rec.extra, r.model
    n = {b: ex["map_banded_n_%s" % b] for b in BANDS}
    want = float(cells[band].sum()) if (kind == "gain" and arm == "t1v") else 0.0
    bad = []
    if n[band] != want:
        bad.append("n supervised in %s = %s, want %s" % (band, n[band], want))
    if any(n[b] != 0.0 for b in BANDS if b != band):
        bad.append("another band is supervised: %s" % n)
    g_own = _gnz(model, K.GROUPS[band])
    other = sum(_gnz(model, K.GROUPS[b]) for b in BANDS if b != band)
    if want > 0:
        if not g_own > 0.0:
            bad.append("gradient not reached: band head %.3g" % g_own)
        if not ex.get("n_map_banded_hist_cells_%s" % band, 0.0) > 0.0:
            bad.append("hist-cell counter not logged")
    elif g_own != 0.0:
        bad.append("expected EXACTLY 0.0, got band head %.3g" % g_own)
    if other != 0.0:
        bad.append("the other bands' heads received gradient %.3g (must be exactly 0.0)" % other)
    return bad


@pytest.mark.parametrize("near", [10, 20])
@pytest.mark.parametrize("band", list(BANDS))
def test_REACH_banded_a_history_only_cell_gets_a_NON_ZERO_gradient_under_T1v_and_EXACTLY_zero_under_C0(near, band):
    assert _banded_reach_problems("t1v", near, band, "gain") == []
    assert _banded_reach_problems("c0", near, band, "gain") == []


@pytest.mark.parametrize("near", [10, 20])
@pytest.mark.parametrize("band", list(BANDS))
def test_REACH_banded_CONTROL_a_cell_neither_mask_reaches_reads_EXACTLY_zero_under_T1v(near, band):
    assert _banded_reach_problems("t1v", near, band, "neither") == []


def _hires_reach(arm, cell_class, rig=False, patch=None):
    """Non-banded 10 cm head: labels ONLY on ``cell_class`` cells (``hist_only`` | ``neither`` | ``now``), backward of the unweighted ``map_hires`` term.  -> (grad abs-sum of the 10 cm head, n supervised cells)."""
    with pytest.MonkeyPatch.context() as mp:
        g = _run(mp, T1V, banded=False, rig=rig, backward=False)
    lv, hv = g.rec.pout["map_hires_lift_valid"], g.rec.pout[HIST]
    sel = {"hist_only": hv & ~lv, "neither": ~hv & ~lv, "now": lv}[cell_class]
    assert int(sel.sum()) > 0, cell_class
    codes = SH._codes_on(SH._fine(sel))
    with pytest.MonkeyPatch.context() as mp:
        if patch is not None:
            patch(mp)
        r = _run(mp, T1V if arm == "t1v" else [], banded=False, rig=rig, cells=codes)
    return _gnz(r.model, "_map_hires."), float(r.rec.extra["n_map_hires_cells"]), r


@pytest.mark.parametrize("rig", [False, True], ids=["full", "rig"])
def test_REACH_10cm_a_history_only_cell_gets_a_NON_ZERO_gradient_under_T1v_and_EXACTLY_zero_under_C0(rig):
    g_v, n_v, _ = _hires_reach("t1v", "hist_only", rig)
    g_c, n_c, _ = _hires_reach("c0", "hist_only", rig)
    assert n_v > 0 and g_v > 0.0, "T1v: the history-only cell is supervised and the map head hears it"
    assert n_c == 0.0 and g_c == 0.0, "C0 (the control): the SAME cell is not supervised, EXACTLY 0.0"


@pytest.mark.parametrize("rig", [False, True], ids=["full", "rig"])
def test_REACH_10cm_CONTROL_neither_cell_reads_EXACTLY_zero_and_a_visible_cell_is_the_same_under_T1v_and_C0(rig):
    g_n, n_n, _ = _hires_reach("t1v", "neither", rig)
    assert n_n == 0.0 and g_n == 0.0, "labels where neither mask reaches: nothing supervised (the OR is not 'supervise everything')"
    g_v, n_v, _ = _hires_reach("t1v", "now", rig)
    g_c, n_c, _ = _hires_reach("c0", "now", rig)
    assert n_v == n_c > 0 and g_v > 0.0 and g_c > 0.0, "a currently visible cell: the supervised LABELLED set is C0's"


# ===================================================================================================================== #
# 3. NOT BUILT: no bev_warp parameter, no past-frame encode / lift / warp, no past image                                #
# ===================================================================================================================== #
def test_NOT_BUILT_T1v_adds_no_parameter_and_no_state_key_and_changes_no_shared_init():
    with pytest.MonkeyPatch.context() as mp:
        c0_args = _parse([])
        c0, _ = _model(mp, c0_args)
        v_args = _parse(T1V)
        v, _ = _model(mp, v_args)
        h_args = _parse(T1H)
        h, _ = _model(mp, h_args)
    assert list(v.state_dict()) == list(c0.state_dict()), "T1v's state_dict KEYS are C0's"
    assert all(torch.equal(x, c0.state_dict()[k]) for k, x in v.state_dict().items()), "and so is every value (no shared init moved)"
    assert not any("bev_warp" in n for n in dict(v.named_parameters()) | dict(v.named_buffers()))
    assert any("bev_warp" in k for k in h.state_dict()), "the positive control: T1h DOES build bev_warp parameters"
    assert BF.is_built(h) and not BF.is_built(v) and BF._fuse(v) is None
    assert not isinstance(v._map_hires.hist_geom, torch.nn.Module) and "hist_geom" not in v._map_hires._modules
    assert getattr(v, "_r9_bev_warp", None) is None and getattr(v, BF.GEOM_ATTR) == BF.HistGeomState((0.5, 1.0), BW.lags_to_rows((0.5, 1.0)))
    assert sum(p.numel() for p in v.parameters()) == sum(p.numel() for p in c0.parameters())
    assert BF.supervise_hist_on(v) and not BF.supervise_hist_on(c0)
    assert BF.warm_start_prefixes(v) == () and BF.stamp(v, v_args) is None and BF.geom_stamp(c0, c0_args) is None
    st = BF.geom_stamp(v, v_args)
    assert st["flag"] == "--r9-hist-supervise-geom" and st["n_params"] == 0 and st["lag_rows"] == list(BW.lags_to_rows((0.5, 1.0)))


def _count(mp, model, rig_calls: Counter):
    """Wrap the past-frame path's entry points with counters.  The positive control (T1h) must move them, T1v must read C0's."""
    enc = model.core.encoder
    lift = model._map_hires.lift
    real_s8, real_lift = enc.s8_from_normalised, type(lift).forward
    real_warp, real_vb, real_grid = BW.warp_bev, BW.warp_valid_bool, BW.sample_grid

    def s8(*a, **k):
        rig_calls["s8_from_normalised"] += 1
        return real_s8(*a, **k)

    def lf(self, *a, **k):
        rig_calls["lift_forward"] += 1
        return real_lift(self, *a, **k)

    def wb(*a, **k):
        rig_calls["warp_bev"] += 1
        return real_warp(*a, **k)

    def vb(*a, **k):
        rig_calls["warp_valid_bool"] += 1
        return real_vb(*a, **k)

    def sg(*a, **k):
        rig_calls["sample_grid"] += 1
        return real_grid(*a, **k)
    enc.s8_from_normalised = s8
    mp.setattr(type(lift), "forward", lf)
    mp.setattr(BW, "warp_bev", wb)
    mp.setattr(BW, "warp_valid_bool", vb)
    mp.setattr(BW, "sample_grid", sg)


def _calls(flags, patch=None, rig=False):
    c = Counter()
    with pytest.MonkeyPatch.context() as mp:
        if patch is not None:
            patch(mp)
        _run(mp, flags, banded=False, rig=rig, backward=False, pre=lambda m: _count(mp, m, c))
    return c


def _no_past_encode_problems(patch=None, rig=False) -> list:
    c0, v, h = _calls([], rig=rig), _calls(T1V, patch, rig), _calls(T1H, rig=rig)
    bad = []
    if not (h["s8_from_normalised"] >= 2 and h["warp_bev"] >= 2 and h["lift_forward"] > c0["lift_forward"]):
        bad.append("the POSITIVE CONTROL (T1h) did not move the counters: %s vs C0 %s" % (dict(h), dict(c0)))
    for k in ("s8_from_normalised", "lift_forward", "warp_bev"):
        if v[k] != c0[k]:
            bad.append("%s: T1v %d vs C0 %d (a past frame was %s)" % (k, v[k], c0[k], "encoded / lifted / warped"))
    if v["warp_valid_bool"] != 2 or v["sample_grid"] != 2 or h["warp_valid_bool"] != 2:
        bad.append("the mask function: T1v %d, T1h %d calls (want 2 = one per lag, the SAME function)" % (v["warp_valid_bool"], h["warp_valid_bool"]))
    return bad


@pytest.mark.parametrize("rig", [False, True], ids=["full", "rig"])
def test_NOT_BUILT_T1v_never_encodes_lifts_or_warps_a_past_frame_CALL_COUNTS(rig):
    assert _no_past_encode_problems(rig=rig) == []


class _Frames:
    """A recorder proxy around an episode's frame tensor: logs every INTEGER index (a single past frame); slices (the window) and attribute reads pass through."""

    def __init__(self, t, log):
        self._t, self._log = t, log

    def __getitem__(self, k):
        if isinstance(k, int):
            self._log.append(k)
        return self._t[k]

    def __getattr__(self, name):
        return getattr(self._t, name)

    def __len__(self):
        return len(self._t)


def _loader_int_reads(flags):
    with pytest.MonkeyPatch.context() as mp:
        args = _parse(flags, banded=False)
        model, cfg = _model(mp, args)
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
    log = []
    for e in eps:
        e.frames = _Frames(e.frames, log)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    BF.attach_dataset(ds, model, args)
    items = [ds[i] for i in (12, 14)]
    return log, items


def test_NOT_BUILT_the_loader_reads_no_past_image_for_T1v_and_does_for_T1h():
    c0, _ = _loader_int_reads([])
    v, items = _loader_int_reads(T1V)
    h, h_items = _loader_int_reads(T1H)
    assert c0 == [] and v == [], "C0 and T1v read no single past frame (integer index) off the episode"
    assert len(h) == 2 * 2 and all(k["bw_frames"].shape[0] == 2 for k in h_items), "the POSITIVE CONTROL: T1h reads one past frame per lag per item"
    assert all("bw_frames" not in it and set(BF.GEOM_BATCH_KEYS) <= set(it) for it in items)


def test_NOT_BUILT_item_fields_with_frames_False_never_touches_the_episodes_frames_at_all():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("the T1v worker touched ep.frames.%s" % name)

        def __getitem__(self, k):
            raise AssertionError("the T1v worker indexed ep.frames[%r]" % (k,))

    ep = type("Ep", (), {})()
    ep.frames, ep.poses = Boom(), torch.arange(40 * 4, dtype=torch.float32).view(40, 4)
    src = BF.ItemSource(BW.lags_to_rows((0.5, 1.0)), False, 1, with_frames=False)
    out = BF.item_fields([ep], 0, 30, src)
    assert set(out) == set(BF.GEOM_BATCH_KEYS) and out["bw_has"].tolist() == [True, True]
    with pytest.raises(AssertionError, match="touched|indexed"):
        BF.item_fields([ep], 0, 30, BF.ItemSource(BW.lags_to_rows((0.5, 1.0)), False, 1, with_frames=True))      # the control: with frames it DOES read them


# ===================================================================================================================== #
# 4. GATE: the G-DVB entry, the refusals                                                                                #
# ===================================================================================================================== #
def _gdvb_problems() -> list:
    bad = []
    if GEOM_DEST in D.coverage(T.build_parser()):
        bad.append("D.coverage: the trainer flag has no G-DVB registry entry (the gate itself refuses an unregistered flag)")
    ent = D.REGISTRY.get(GEOM_DEST)
    if ent is None:
        return bad + ["no G-DVB entry for %s" % GEOM_DEST]
    if ent.kind != "built":
        bad.append("kind %r, want built" % ent.kind)
    with pytest.MonkeyPatch.context() as mp:
        args = _parse(T1V)
        model, _ = _model(mp, args)
        if ent.check(model, args):
            bad.append("a correct T1v build was flagged: %s" % ent.check(model, args))
        off_args = _parse([])
        off, _ = _model(mp, off_args)
        if ent.check(off, off_args):
            bad.append("C0 was flagged")
        # deliberate-regression arms: each must produce >= 1 Mismatch
        if not ent.check(off, args):
            bad.append("argv declares T1v, the build has none: NOT caught")
        if not ent.check(model, off_args):
            bad.append("the build carries T1v, argv does not declare it: NOT caught")
        m2, _ = _model(mp, _parse(T1V))
        del m2._map_hires.hist_geom
        if not ent.check(m2, args):
            bad.append("state says T1v but the 10 cm branch holds no HistGeom: NOT caught")
        m3, _ = _model(mp, _parse(T1V))
        m3._map_hires.hist_geom.lag_rows = (1, 2)
        if not ent.check(m3, args):
            bad.append("wrong lag rows: NOT caught")
        m4, _ = _model(mp, _parse(T1V))
        m4._map_hires.add_module(BF.HEAD_ATTR, BW.BevWarpFuse(16, 2, 60.0, 16.0, BW.lags_to_rows((0.5, 1.0))))
        if not ent.check(m4, args):
            bad.append("a bev_warp fuse built beside T1v: NOT caught")
    return bad


def test_GATE_G_DVB_has_the_entry_and_its_deliberate_regression_arms_are_caught():
    assert _gdvb_problems() == []


def test_GATE_the_refusals():
    def refused(flags, drop=(), banded=False, match=""):
        with pytest.raises(SystemExit, match=match):
            BF.pin_args(_parse(flags, banded=banded, drop=drop))
    refused(T1V + ["--r9-bev-warp"], match="ALTERNATIVE")                     # not an add-on to the bundle
    refused(T1V + ["--r9-bev-warp", "--r9-bev-warp-supervise-hist"], match="ALTERNATIVE")
    refused(T1V + ["--no-map-lift-valid-mask"], match="dead flag")
    refused(T1V, drop=("--map-hires", "on"), match="map-hires is not")
    refused(T1V + ["--r9-bev-warp-lags", "0.5,2.0"], match="dead flag")        # non-default lags: the existing dead-flag rule (it needs --r9-bev-warp)
    BF.pin_args(_parse(T1V))                                                   # and the plain T1v argv is ADMITTED, on both heads
    BF.pin_args(_parse(T1V, banded=False))
    with pytest.raises(SystemExit, match="alternative"):
        a = _parse(T1V + ["--r9-bev-warp"])
        BF.attach(R._build()[0], a)


# ===================================================================================================================== #
# 5. MUTATIONS: each check above goes RED when its own line is broken (the checks DISCRIMINATE)                         #
# ===================================================================================================================== #
def _shifted_T(mp):
    real = BF._relative_T

    def shifted(batch, device):
        t = real(batch, device).clone()
        t[..., 0] += 0.7                                     # T1v's mask from ANOTHER relative transform than T1h's
        return t
    mp.setattr(BF, "_relative_T", shifted)


def _other_lags(mp):
    mp.setattr(BW, "LAGS_S_DEFAULT", (0.5, 2.0))             # T1v built with other lags than T1h's default pair


def _encode_one_past_frame(mp):
    real = BF.stage

    def st(model, batch, device):
        real(model, batch, device)
        enc = model.core.encoder
        k = int(enc._mean.shape[1]) // 3
        x = torch.zeros(int(batch["bw_pose_now"].shape[0]), 3, 64, 64)
        with torch.no_grad():
            enc.s8_from_normalised(enc.normalise(x.repeat(1, k, 1, 1)), torch.arange(x.shape[0]))      # T1v "still encodes a past frame"
    mp.setattr(BF, "stage", st)


def _or_dropped(mp):
    mp.setattr(BF, "supervised_lift_valid", lambda model, pout, lv, sel: lv)                           # T1v == C0 on the 10 cm loss
    mp.setattr(BF, "supervised_banded_unobs", lambda model, pout, unobs, sel, geom: unobs)             # ... and on the banded loss


def _banded_ignores_mask(mp):
    mp.setattr(BF, "supervised_banded_unobs", lambda model, pout, unobs, sel, geom: unobs)             # ONLY the banded path drops the geometry mask


def test_MUTATION_the_mask_from_another_T_or_other_lags_turns_the_IDENTITY_RED():
    for patch in (_shifted_T, _other_lags):
        for banded in (False, True):
            bad = _identity_problems(10, banded, patch=patch)
            assert bad and any("differs" in p for p in bad), (patch.__name__, banded, bad)


def test_MUTATION_a_T1v_that_still_encodes_a_past_frame_turns_the_CALL_COUNT_RED():
    bad = _no_past_encode_problems(_encode_one_past_frame)
    assert any("s8_from_normalised" in p for p in bad), bad


def test_MUTATION_the_OR_dropped_makes_T1v_equal_C0_and_turns_the_REACH_RED():
    assert _banded_reach_problems("t1v", 10, "near", "gain", patch=_or_dropped) != []
    g, n, _ = _hires_reach("t1v", "hist_only", patch=_or_dropped)
    assert n == 0.0 and g == 0.0, "the mutant supervises nothing on the history-only cell: the check `n > 0 and g > 0` would FAIL (RED)"
    bad = _identity_problems(10, False, patch=_or_dropped)
    assert bad and any("supervised_lift_valid" in p for p in bad), bad


def test_MUTATION_a_banded_path_that_ignores_the_geometry_mask_turns_ONLY_the_banded_checks_RED():
    assert _banded_reach_problems("t1v", 20, "mid", "gain", patch=_banded_ignores_mask) != []
    bad = _identity_problems(10, True, patch=_banded_ignores_mask)
    assert bad and any("supervised_banded_unobs" in p for p in bad), bad
    assert _identity_problems(10, False, patch=_banded_ignores_mask) == [], "the 10 cm path is untouched by that mutation (the checks are per path)"


def test_MUTATION_the_G_DVB_entry_removed_turns_the_gate_check_RED(monkeypatch):
    monkeypatch.delitem(D.REGISTRY, GEOM_DEST)
    bad = _gdvb_problems()
    assert bad and any("no G-DVB" in p or "coverage" in p for p in bad), bad


def test_MUTATION_a_forward_with_no_staged_geometry_or_no_emitted_key_is_REFUSED_never_trained_on_the_narrow_mask():
    with pytest.MonkeyPatch.context() as mp:                                       # the stage hop is gone
        mp.setattr(BF, "stage", lambda model, batch, device: None)
        with pytest.raises(SystemExit, match="carries no `map_hires_hist_valid`"):
            _run(mp, T1V, banded=False, backward=False)
    with pytest.MonkeyPatch.context() as mp:                                       # the emission hop is gone
        mp.setattr(BW.HistGeom, "take_hist_valid", lambda self, src_valid, dtype: None)
        with pytest.raises(SystemExit, match="--r9-hist-supervise-geom is built with the banded head but the forward's perception output carries no"):
            _run(mp, T1V, banded=True, backward=False)
    with pytest.MonkeyPatch.context() as mp:                                       # the dataset was not attached: the batch has no poses
        args = _parse(T1V, banded=False)
        model, cfg = _model(mp, args)
        batch = FM._batch(cfg, model, args, attach=False)
        with pytest.raises(SystemExit, match="batch carries no"):
            FM._loss(model, batch)
