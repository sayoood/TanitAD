"""``--nav-args`` (E13b, D-GSTR-1 P3) — the nav command's RANGE and TIME.

MEASURED (`.../Research/2026-09-06-nav-vs-routehead/RESULT.md`, commit
a07d733d7): `vocab_v7.NAV_ARG_SLOTS = ("distance_m", "time_s")` is written by
`s2_geom_emit_v7.nav_command()` on EVERY turn token — train TURN_L median
27.3 m / 7.2 s, TURN_R 36.6 m / 7.4 s — and THREE consumers disagree: v6/v7f
FEED both through `NavConditioner.arg_proj`, refav1 DISCARDS them, refcv3 /
refcv4b NEVER READ them. So the deployed refcv4b sees a bare 3-way categorical:
exactly the stripped BEARING that is separated WORSE by +2.3632 m, while a goal
carrying range recovers 57.1 % of the longitudinal selection ceiling.

⛔ EVERY GUARD HERE IS PROVEN BY MUTATION, NOT BY INSPECTION. An AST census
once read "0 suspects" on BOTH the fixed and the broken trainer; a guard that
has never been shown to FIRE is not a guard. So each test reintroduces the
specific defect it protects against and asserts the refusal.

The disciplines pinned, one test each:
(a) OFF IDENTITY  — default off; a model built with the seam and fed args is
    BIT-IDENTICAL at step 0 to one fed none (the zero-init nav output
    projections), so any later delta is attributable to training.
(b) BOTH REFUSALS — args without the seam, and the seam without args, each
    RAISE. A silently-dropped channel reads as "the nav args do not help".
(c) THE VALIDITY BIT IS LOAD-BEARING — `NAV_FOLLOW_ROAD` carries `args: {}` on
    2,897/2,897 train records, and `NavEmitter._args_for_window` defaults a
    missing slot to 0.0. The model RE-APPLIES the bit, so an invalid row with
    garbage values is indistinguishable from an invalid row with zeros; and an
    invalid row is NOT the same input as a valid row whose distance is 0 m.
(d) THE CHANNEL IS NOT INERT — once the zero-init gate is opened, changing the
    ARGS ALONE moves `g_str`. Without this the seam could be wired and dead.
(e) THE LOADER — validity comes from the KEY'S PRESENCE, never from the value;
    the census reports the fraction of WINDOWS carrying a real distance; the
    eval dataset takes the TRAIN split's normaliser and never fits its own.
(f) UNITS AND CLI — the raw units are declared, `--nav-args` defaults False and
    is refused without `--nav-from-v7`.
CPU-only, synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T                                    # noqa: E402
from tanitad.data import v7_labels as v7l                    # noqa: E402
from tanitad.models.nav_conditioning import NavArgStats      # noqa: E402
from tanitad.models.vocab_v7 import (NAV_ARG_SLOTS,          # noqa: E402
                                     NAV_COMMAND_TOKENS)
from tanitad.refs import refc_v3 as v3                       # noqa: E402


# ---------------------------------------------------------------- the rig
def _cfg(nav_args=False):
    cfg = v3.refc_v3_smoke_config(True)
    cfg.nav_args_inject = bool(nav_args)
    return cfg


def _episodes(n, *, min_frames=60, seed=0, ids=None):
    cfg = v3.refc_v3_smoke_config(True)
    eps = T._synth_episodes(n, cfg.core, seed=seed, min_frames=min_frames)
    if ids is not None:
        for ep, sid in zip(eps, ids):
            ep.episode_id = sid
    return cfg, eps


def _dataset(cfg, eps):
    return T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                       channels=cfg.core.encoder.in_channels)


def _label(clip_id, nav_token, *, args=None):
    """A v7.2-shaped record. ``args=None`` reproduces the REAL corpus's
    ``NAV_FOLLOW_ROAD`` shape: an EMPTY dict, not a missing key."""
    nav = {"token": nav_token, "provenance": "ego-future", "oracle": True,
           "args": ({} if args is None else dict(args))}
    return v7l.V7Label(
        clip_id=clip_id,
        tac_lat=v7l.HEADS["tac_lat"][0], tac_lon=v7l.HEADS["tac_lon"][0],
        str_action=v7l.HEADS["str_action"][0],
        str_goal=v7l.HEADS["str_goal"][0],
        tac_anchor=None, bands={"tactical_s": [0.0, 4.0]}, t0_s=2.0,
        horizon={}, audit={}, _oracle={"nav_command": nav})


def _manifest(n, allow=True):
    return v7l.LabelManifest(path="synthetic", md5="0" * 32, n_records=n,
                             schema_version=v7l.EXPECTED_SCHEMA,
                             vocab=v7l.EXPECTED_VOCAB, allow_oracle_nav=allow)


def _model(nav_args=False, seed=0):
    torch.manual_seed(seed)
    return v3.RefCV3Model(_cfg(nav_args)).eval()


def _batch(b=2, cfg=None):
    cfg = cfg or v3.refc_v3_smoke_config(True)
    c, (h, w) = cfg.core.encoder.in_channels, cfg.core.encoder.image_hw()
    torch.manual_seed(7)
    return {"frames": torch.rand(b, cfg.core.window, c, h, w),
            "nav_cmd": torch.tensor([1, 2][:b], dtype=torch.long),
            "v0": torch.full((b,), 8.0)}


# --------------------------------------------------- (a) the OFF identity
def test_default_is_off_and_the_seam_is_absent():
    assert v3.RefCV3Config().nav_args_inject is False
    m = _model(False)
    assert getattr(m, "nav_arg_proj", "missing") is None
    assert v3.NAV_ARG_DIMS == 3
    assert len(NAV_ARG_SLOTS) == v3.NAV_ARG_DIMS - 1, (
        "the block is the two vocab_v7 slots PLUS one validity bit")


def test_step_zero_is_bit_identical_with_and_without_the_args():
    """⭐ ATTRIBUTION: `nav_to_tac` / `nav_to_str` are zero-init, so at step 0
    the whole nav term is exactly 0 whatever the args say. A later delta is
    therefore training, never initialisation."""
    bt = _batch()
    m = _model(True)
    with torch.no_grad():
        a = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"],
              nav_args=torch.tensor([[3.0, -2.0, 1.0], [0.0, 0.0, 0.0]]))
        b = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"],
              nav_args=torch.tensor([[-9.0, 9.0, 1.0], [5.0, 5.0, 1.0]]))
    assert torch.equal(a["g_str"], b["g_str"])
    assert torch.equal(a["traj"], b["traj"])


# ------------------------------------------------------- (b) BOTH refusals
def test_args_supplied_to_a_build_without_the_seam_RAISES():
    """MUTATION: hand the tensor to a seam-less build. It must refuse, not
    drop — a dropped channel reads as 'the nav args do not help'."""
    m = _model(False)
    bt = _batch()
    with pytest.raises(ValueError, match="SILENTLY"):
        m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"],
          nav_args=torch.zeros(2, 3))


def test_seam_without_args_RAISES_rather_than_defaulting():
    """MUTATION: build the seam, then forget the tensor. A silent zero would
    say 'the turn is here, now' on every window."""
    m = _model(True)
    bt = _batch()
    with pytest.raises(ValueError, match="no nav_args reached"):
        m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"])


def test_wrong_width_RAISES():
    """MUTATION: the 2-wide form — i.e. the validity bit dropped."""
    m = _model(True)
    bt = _batch()
    with pytest.raises(ValueError, match="validity slot is NOT optional"):
        m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"],
          nav_args=torch.zeros(2, 2))


# ------------------------------------------- (c)+(d) the bit, and non-inertness
def _open_the_gate(m, scale=0.35):
    """Zero-init is what makes step 0 inert; open it so the channel's effect
    is observable at all. Without this the next two tests would 'pass' on a
    completely disconnected seam."""
    with torch.no_grad():
        for lin in (m.nav_to_tac, m.nav_to_str):
            lin.weight.normal_(0.0, scale)


def test_the_validity_bit_gates_the_values_inside_the_model():
    """⛔ X15's rule: the CONSUMER re-applies the flag, so a caller that
    forgot to zero an invalid row cannot leak a phantom range."""
    m = _model(True)
    _open_the_gate(m)
    bt = _batch()
    junk = torch.tensor([[17.0, -4.0, 0.0], [-3.0, 8.0, 0.0]])
    zero = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with torch.no_grad():
        a = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"], nav_args=junk)
        b = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"], nav_args=zero)
    assert torch.equal(a["g_str"], b["g_str"]), (
        "an INVALID row's values reached the model — the bit is decorative")


def test_invalid_is_a_DIFFERENT_input_from_a_valid_zero_distance():
    """'no range known' must not collapse onto 'the turn is 0 m away'. This is
    the whole reason the block is 3-wide and not 2-wide."""
    m = _model(True)
    _open_the_gate(m)
    bt = _batch()
    with torch.no_grad():
        inval = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"],
                  nav_args=torch.zeros(2, 3))["g_str"]
        val0 = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"],
                 nav_args=torch.tensor([[0.0, 0.0, 1.0],
                                        [0.0, 0.0, 1.0]]))["g_str"]
    assert not torch.equal(inval, val0)


def test_the_ARGS_ALONE_move_g_str_once_the_gate_is_open():
    """⛔ THE NON-INERTNESS GUARD. Same token, same frames, same v0 — only the
    RANGE changes. If this passes on a disconnected seam the seam is dead."""
    m = _model(True)
    _open_the_gate(m)
    bt = _batch()
    with torch.no_grad():
        near = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"],
                 nav_args=torch.tensor([[-1.2, -1.1, 1.0],
                                        [-1.2, -1.1, 1.0]]))["g_str"]
        far = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"],
                nav_args=torch.tensor([[2.4, 2.2, 1.0],
                                       [2.4, 2.2, 1.0]]))["g_str"]
    assert not torch.equal(near, far)
    assert float((near - far).abs().max()) > 1e-6


def test_the_ledger_accounts_the_new_parameters():
    on, off = _model(True), _model(False)
    bo = v3.param_breakdown_v3(on)
    bf = v3.param_breakdown_v3(off)
    extra = v3.NAV_ARG_DIMS * on.cfg.d_nav + on.cfg.d_nav
    assert bo["nav_inject"] - bf["nav_inject"] == extra
    assert bo["total"] - bf["total"] == extra


# ------------------------------------------------------------ (e) the loader
def _loader_rig(with_args_on_follow=False):
    ids = [2001, 2002, 2003]
    cfg, eps = _episodes(3, min_frames=60, seed=3, ids=ids)
    ds = _dataset(cfg, eps)
    turn = {"distance_m": 27.3, "time_s": 7.2}
    ds.v7_by_sid = {
        2001: _label("clip-a", "NAV_FOLLOW_ROAD",
                     args=(turn if with_args_on_follow else None)),
        2002: _label("clip-b", "NAV_TURN_L", args=turn),
        2003: _label("clip-c", "NAV_TURN_R",
                     args={"distance_m": 36.6, "time_s": 7.4}),
    }
    ds.enable_nav_from_v7(_manifest(3))
    return cfg, eps, ds


def test_validity_comes_from_the_KEY_not_the_value():
    """⛔ THE DEFECT THIS PREVENTS: `distance_m == 0.0` is AMBIGUOUS because
    `_args_for_window` defaults a missing slot to 0.0. Only the key's presence
    separates 'the turn is 0 m away' from 'there is no turn'."""
    _cfgx, _eps, ds = _loader_rig()
    rep = ds.enable_nav_args()
    assert ds._nav_args_by_sid[2001][2] == 0.0, "empty args read as VALID"
    assert ds._nav_args_by_sid[2002][2] == 1.0
    assert ds._nav_args_by_sid[2002][:2] == (27.3, 7.2)
    assert rep["n_clips_with_real_args"] == 2 and rep["n_clips"] == 3
    # MUTATION: give FOLLOW_ROAD a real zero distance -> it becomes VALID.
    _c2, _e2, ds2 = _loader_rig(with_args_on_follow=True)
    ds2.enable_nav_args()
    assert ds2._nav_args_by_sid[2001][2] == 1.0


def test_the_census_reports_the_WINDOW_fraction_and_the_units():
    _cfgx, _eps, ds = _loader_rig()
    rep = ds.enable_nav_args()
    assert 0.0 < rep["window_real_distance_frac"] < 1.0
    assert rep["n_windows_with_real_args"] < rep["n_windows"]
    assert rep["raw_units"] == ["distance_m:metres", "time_s:seconds"]
    assert list(v3.NAV_ARG_UNITS) == rep["raw_units"]
    assert rep["slots"] == ["distance_norm", "time_norm", "args_valid"]
    assert rep["normaliser"]["n_fit"] == 2
    assert rep["normaliser_fitted_here"] is True


def test_items_carry_the_block_and_invalid_rows_are_explicitly_zero():
    _cfgx, _eps, ds = _loader_rig()
    ds.enable_nav_args()
    seen = {}
    for i, (e_i, _t) in enumerate(ds.index):
        item = ds[i]
        assert "nav_args" in item and tuple(item["nav_args"].shape) == (3,)
        assert item["nav_args"].dtype == torch.float32
        seen[int(e_i)] = item["nav_args"]
    assert torch.equal(seen[0], torch.zeros(3))          # FOLLOW_ROAD
    for e in (1, 2):
        assert float(seen[e][2]) == 1.0
    assert not torch.equal(seen[1][:2], seen[2][:2]), (
        "two different distances normalised to the same value")


def test_the_eval_dataset_takes_the_TRAIN_normaliser_and_never_fits_its_own():
    """⛔ A statistic fitted on the split it will later score is the
    2026-08-22 ridge failure in a loader costume."""
    _cfgx, _eps, tr_ds = _loader_rig()
    tr_ds.enable_nav_args()
    _c2, _e2, ev_ds = _loader_rig()
    rep = ev_ds.enable_nav_args(stats=tr_ds.nav_arg_stats)
    assert rep["normaliser_fitted_here"] is False
    assert ev_ds.nav_arg_stats is tr_ds.nav_arg_stats
    assert rep["normaliser"] == tr_ds.nav_args_report["normaliser"]


def test_nav_args_without_nav_from_v7_is_REFUSED():
    cfg, eps = _episodes(2, min_frames=60, seed=4, ids=[3001, 3002])
    ds = _dataset(cfg, eps)
    with pytest.raises(ValueError, match="requires --nav-from-v7|needs --nav-from-v7"):
        ds.enable_nav_args()


def test_a_split_with_no_real_args_anywhere_is_REFUSED():
    """MUTATION: every clip is FOLLOW_ROAD. The channel would be a constant
    pad and the arm would measure it as noise."""
    ids = [4001, 4002]
    cfg, eps = _episodes(2, min_frames=60, seed=5, ids=ids)
    ds = _dataset(cfg, eps)
    ds.v7_by_sid = {i: _label(f"c{i}", "NAV_FOLLOW_ROAD") for i in ids}
    ds.enable_nav_from_v7(_manifest(2))
    with pytest.raises(ValueError, match="NOT ONE clip"):
        ds.enable_nav_args()


def test_the_normaliser_is_fitted_on_VALID_rows_only():
    """The FOLLOW_ROAD pad is the MAJORITY on the real corpus (2,897/4,572);
    folding its zeros into the mean would fit mostly on padding."""
    _cfgx, _eps, ds = _loader_rig()
    ds.enable_nav_args()
    hand = NavArgStats.from_fit_split([27.3, 36.6], [7.2, 7.4])
    assert ds.nav_arg_stats.distance_mean == pytest.approx(hand.distance_mean)
    assert ds.nav_arg_stats.n_fit == 2


# --------------------------------------------------------------- (f) the CLI
def _pin(*extra):
    """Through the REAL parser, so the test exercises the shipped flag rather
    than a stub whose defaults could drift from argparse's."""
    a = T.build_parser().parse_args(["--out", "x", "--arm", "hier", *extra])
    return T._pin_trainer_cfg(v3.refc_v3_smoke_config(hier=True), a), a


def test_the_flag_EXISTS_and_defaults_off():
    opts = {o for a in T.build_parser()._actions for o in a.option_strings}
    assert "--nav-args" in opts
    cfg, a = _pin()
    assert a.nav_args is False
    assert cfg.nav_args_inject is False


def test_pin_trainer_cfg_turns_the_seam_on_with_its_supplier():
    cfg, _a = _pin("--nav-args", "--nav-from-v7")
    assert cfg.nav_args_inject is True
    assert cfg.nav_inject is True


def test_pin_trainer_cfg_refuses_nav_args_without_its_supplier():
    """MUTATION: ask for the args with no supplier for them."""
    with pytest.raises(SystemExit, match="requires --nav-from-v7"):
        _pin("--nav-args")


def test_nav_args_with_the_nav_path_switched_off_is_REFUSED():
    """MUTATION: --goal-point-inject sets nav_inject=False by
    pre-registration, so the args would be a silently inert flag."""
    with pytest.raises(SystemExit, match="silently inert"):
        _pin("--nav-args", "--nav-from-v7", "--goal-point-inject")


def test_the_three_nav_tokens_are_the_ones_the_args_ride_on():
    assert tuple(NAV_COMMAND_TOKENS) == ("NAV_FOLLOW_ROAD", "NAV_TURN_L",
                                         "NAV_TURN_R")
    assert tuple(NAV_ARG_SLOTS) == ("distance_m", "time_s")
