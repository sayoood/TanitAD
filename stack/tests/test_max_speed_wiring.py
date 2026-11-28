"""E16 wiring — the ``--max-speed-input`` edge inside ``RefCV3Model`` and the trainer.

⛔ WHY THIS FILE EXISTS SEPARATELY FROM ``test_max_speed_input.py``. That file tests
the CHANNEL (the ladder, the units refusal, the conditioner). This one tests the
WIRING — the thing the filed ``WIRING_DIFF.md`` explicitly could NOT provide, because
it can only be written against the patched file: that the flag exists, that it is
REFUSED when dead, that the OFF build did not move, and that the ON build is
ROLLABLE.

⛔⛔ THE HAZARD THIS FILE IS AIMED AT. Building a head on the vocabulary alone made
refcv4b, three refcv3 checkpoints and a LIVE refcv5 run UNROLLABLE: the head's params
were absent from every recorded ``param_breakdown`` and ``cross_check_config``
correctly refused them. So the acceptance bar here is ROLLABILITY, not "the tests
pass": one construction site, gated on the flag, with a ledger line that reads the
BUILT OBJECT.

⚠️ WHAT THE OFF-PARITY TEST DOES AND DOES NOT CLAIM. With the flag OFF **nothing is
constructed and nothing is fed**, so the identity is STRUCTURAL, not a zero-init
cancellation. The zero-init claim belongs to the ON path (bit-inert EMISSION at step
0) and is tested separately below. Conflating the two would be a false strength claim.
"""
from __future__ import annotations

import ast
import importlib.util
import io
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.data import v7_labels as v7l                   # noqa: E402
from tanitad.refs import max_speed_input as msi             # noqa: E402
from tanitad.refs import refc_v3 as v3                      # noqa: E402


# ==========================================================================
# rig
# ==========================================================================
def _smoke(on: bool, mode: str = msi.DEFAULT_MODE, seed: int = 0):
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.max_speed_input = on
    if on:
        cfg.max_speed_cfg = msi.MaxSpeedConfig(enabled=True, mode=mode)
    torch.manual_seed(seed)
    return cfg, v3.RefCV3Model(cfg).eval()


def _frames(cfg, b: int = 2, seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.core.encoder.image_hw()
    return torch.rand(b, cfg.core.window, cfg.core.encoder.in_channels, h, w,
                      generator=g)


def _fwd(m, cfg, **kw):
    with torch.no_grad():
        return m(_frames(cfg), nav_cmd=torch.tensor([0, 1]),
                 v0=torch.tensor([3.0, 7.0]), steps=2, **kw)


V_MAX = torch.tensor([11.19, 30.0])
V_OK = torch.tensor([1.0, 1.0])


def _trainer():
    """``refc_v3_train.py`` by PATH — it is a script, and ``build_parser`` +
    ``_pin_trainer_cfg`` are the only way a run's config is recoverable. This is
    ``refcv3_arm``'s own convention and ``test_refc_v3_rollability``'s, reused."""
    p = os.path.join(os.path.dirname(_STACK), "stack", "scripts",
                     "refc_v3_train.py")
    if not os.path.exists(p):
        pytest.skip(f"trainer not present at {p}")
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_msi", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_msi"] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(tr, *extra, arm="hier"):
    return tr.build_parser().parse_args(
        ["--arm", arm, "--size", "small", "--out", "/x", "--v2-cache", "/x/c"]
        + list(extra))


def _pin(tr, a):
    base = v3.refc_v3_sized_config(a.size, hier=(a.arm == "hier"))
    return tr._pin_trainer_cfg(base, a)


# ==========================================================================
# 1 — the OFF contract: the live run resumes through this file
# ==========================================================================
def test_the_flag_defaults_OFF_so_the_live_build_did_not_move():
    assert v3.RefCV3Config().max_speed_input is False
    assert v3.refc_v3_hier_config().max_speed_input is False
    assert v3.refc_v3_flat_config().max_speed_input is False
    assert v3.RefCV3Config().max_speed_cfg.enabled is False
    # ⭐ and the DEFAULT mode is the quantized one (PI: "good idea with
    # quantization"), so an operator who passes only `--max-speed-input` gets
    # the ladder, not the raw ego-derived float.
    assert v3.RefCV3Config().max_speed_cfg.mode == "quantized"
    assert msi.DEFAULT_MODE == "quantized"


def test_v3_parity_when_off_no_max_speed_parameters_and_no_hook_key():
    """⛔ THE PARITY ASSERTION THAT MATTERS FOR A LIVE RUN: with the flag OFF there
    is no ``max_speed_*`` parameter, no ``max_speed_*`` state_dict key, no ledger
    line, and the hook reports the edge as not injected.

    ⚠️ This identity is STRUCTURAL — nothing is built and nothing is fed. It is not
    a zero-init cancellation, and must not be quoted as one."""
    cfg, m = _smoke(False)
    assert m.max_speed_cond is None
    assert not [k for k in m.state_dict() if "max_speed" in k]
    assert "max_speed_inject" not in v3.param_breakdown_v3(m)
    out = _fwd(m, cfg)
    assert out["max_speed_injected"] is False
    # ⭐ SAME-BREATH POSITIVE CONTROL. Without it a passing OFF assertion is
    # indistinguishable from a misspelled attribute name.
    _, m2 = _smoke(True)
    assert [k for k in m2.state_dict() if "max_speed" in k]
    assert "max_speed_inject" in v3.param_breakdown_v3(m2)


def test_the_OFF_build_is_bit_identical_to_the_same_build_before_the_flag():
    """⭐ The strongest form available inside the test suite: two OFF builds on the
    SAME seed agree tensor-for-tensor, and the ON build does NOT — so the
    comparison can fail. (The full pre-patch comparison, against the file at git
    HEAD, lives in the research package's ``off_parity_proof.py``.)"""
    cfg_a, a = _smoke(False, seed=7)
    cfg_b, b = _smoke(False, seed=7)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert torch.equal(_fwd(a, cfg_a)["traj"], _fwd(b, cfg_b)["traj"])
    # CONTROL: the ON build on the SAME seed differs — the equality has teeth.
    cfg_c, c = _smoke(True, seed=7)
    assert set(c.state_dict()) != set(sa)


# ==========================================================================
# 2 — the ON contract: inert at init, REACHABLE once trained, VALUE-sensitive
# ==========================================================================
def test_the_edge_is_BIT_INERT_at_init():
    """Zero-init on both output projections, exactly like E13 (nav) and E11'
    (ego), so an ON-vs-OFF comparison is attributable to training and not to a
    perturbed step 0."""
    cfg, m = _smoke(True)
    out = _fwd(m, cfg, v_max_ms=V_MAX, v_max_valid=V_OK)
    assert torch.equal(out["traj"], out["traj_base"])
    assert torch.equal(out["sel_idx"], out["sel_idx_base"])
    assert out["max_speed_injected"] is True


def test_the_edge_is_REACHABLE_once_the_projections_are_trained():
    """⛔ THE HALF A ZERO-INIT CHECK CANNOT GIVE YOU. An edge that is bit-inert at
    init AND unreachable after training is a DEAD WIRE that reads as a clean
    ablation."""
    cfg, m = _smoke(True)
    before = _fwd(m, cfg, v_max_ms=V_MAX, v_max_valid=V_OK)
    with torch.no_grad():
        torch.manual_seed(1)
        m.max_speed_cond.to_tac.weight.normal_(std=0.5)
        m.max_speed_cond.to_str.weight.normal_(std=0.5)
    after = _fwd(m, cfg, v_max_ms=V_MAX, v_max_valid=V_OK)
    assert not torch.equal(before["z_tac"], after["z_tac"])
    assert not torch.equal(before["g_str"], after["g_str"])


def test_the_CEILING_VALUE_reaches_the_state_not_only_its_presence():
    """⛔⛔ THE E13 FAILURE MODE, TESTED FOR DIRECTLY. refcv4b's nav edge moved
    ``g_str`` by 0.0103 when its VALUE changed and by 0.1902 when it was REMOVED —
    18.6x, i.e. a PRESENCE-GATED BIAS. A ceiling channel that only says "a limit
    exists" would be that defect again, so two DIFFERENT valid ceilings must give
    two different states."""
    cfg, m = _smoke(True)
    with torch.no_grad():
        torch.manual_seed(2)
        m.max_speed_cond.to_tac.weight.normal_(std=0.5)
        m.max_speed_cond.to_str.weight.normal_(std=0.5)
    lo = _fwd(m, cfg, v_max_ms=torch.tensor([8.0, 8.0]), v_max_valid=V_OK)
    hi = _fwd(m, cfg, v_max_ms=torch.tensor([33.0, 33.0]), v_max_valid=V_OK)
    assert not torch.equal(lo["z_tac"], hi["z_tac"])
    assert not torch.equal(lo["g_str"], hi["g_str"])


def test_the_WITHHOLD_control_is_exactly_the_no_ceiling_state():
    """⛔ THE CONTROL THIS EDGE OWES EVERY RESULT. ``valid = 0`` must produce
    EXACTLY the un-conditioned state — not a bias-shifted leftover — or the
    withhold arm measures "a different bias" instead of "no ceiling"."""
    cfg, m = _smoke(True)
    with torch.no_grad():
        torch.manual_seed(3)
        m.max_speed_cond.to_tac.weight.normal_(std=0.5)
        m.max_speed_cond.to_str.weight.normal_(std=0.5)
        m.max_speed_cond.proj.bias.normal_(std=0.5)      # the leftover's source
    off = torch.tensor([0.0, 0.0])
    a = _fwd(m, cfg, v_max_ms=V_MAX, v_max_valid=off)
    b = _fwd(m, cfg, v_max_ms=torch.tensor([0.0, 0.0]), v_max_valid=off)
    assert torch.equal(a["z_tac"], b["z_tac"]), \
        "a withheld row leaked its VALUE — the X15 defect"
    # CONTROL in the same breath: with valid = 1 the SAME weights DO move the
    # state, so the equality above is a gate and not a dead edge.
    c = _fwd(m, cfg, v_max_ms=V_MAX, v_max_valid=V_OK)
    assert not torch.equal(a["z_tac"], c["z_tac"])


def test_the_SHUFFLE_control_moves_the_state():
    """The other obligation: serving another clip's ceiling must change the
    prediction, or the shuffle control cannot separate 'the arm used the channel'
    from 'the arm gained a parameter'."""
    cfg, m = _smoke(True)
    with torch.no_grad():
        torch.manual_seed(4)
        m.max_speed_cond.to_tac.weight.normal_(std=0.5)
    real = _fwd(m, cfg, v_max_ms=V_MAX, v_max_valid=V_OK)
    shuf = _fwd(m, cfg, v_max_ms=V_MAX.flip(0), v_max_valid=V_OK)
    assert not torch.equal(real["z_tac"], shuf["z_tac"])


# ==========================================================================
# 3 — ROLLABILITY: the ledger, and ONE gated construction site
# ==========================================================================
@pytest.mark.parametrize("flag", [False, True])
def test_ledger_line_tracks_the_built_object(flag):
    """⭐⭐ THE ANTI-DRIFT PIN, and the D-ROLL-1 contract. ``param_breakdown_v3``
    reports this line by reading the BUILT OBJECT, never by re-deriving
    ``cfg.max_speed_input`` — a second copy of a construction condition is exactly
    how a recorded ledger drifts from a rebuild and makes a checkpoint
    unrollable."""
    _, m = _smoke(flag)
    bd = v3.param_breakdown_v3(m)
    assert ("max_speed_inject" in bd) == (m.max_speed_cond is not None)
    if m.max_speed_cond is not None:
        assert bd["max_speed_inject"] == sum(
            p.numel() for p in m.max_speed_cond.parameters())
    # the ledger sums EXACTLY in BOTH states — an unaccounted module is a
    # capacity confound inside the very claim the arm exists to test.
    assert sum(v for k, v in bd.items() if k != "total") == bd["total"]


def test_the_optin_costs_exactly_the_conditioner_and_nothing_else():
    off = v3.param_breakdown_v3(_smoke(False)[1])
    on = v3.param_breakdown_v3(_smoke(True)[1])
    assert on["total"] - off["total"] == on["max_speed_inject"]


def test_the_conditioner_has_exactly_one_construction_site_and_it_is_gated():
    """⭐ STRUCTURAL REACHABILITY, the ``preflight``-only lesson generalised: a gate
    only covers the paths that go through it, so assert there is no other path."""
    root = os.path.join(_STACK, "tanitad")
    sites, scanned, unreadable = [], 0, []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                src = io.open(p, encoding="utf-8").read()
            except OSError:                     # the G: mount flaps
                unreadable.append(p)
                continue
            scanned += 1
            if "MaxSpeedConditioner" not in src:
                continue
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.Call) and (
                        getattr(node.func, "attr", None) == "MaxSpeedConditioner"
                        or getattr(node.func, "id", None) == "MaxSpeedConditioner"):
                    sites.append((os.path.relpath(p, root), node.lineno))
    # ⛔ 0 hits is a claim about the SEARCH unless the search is asserted to have
    # READ its files. A same-breath non-zero control settles it.
    assert scanned > 50, f"scan read only {scanned} files ({unreadable[:5]})"
    assert not unreadable, f"unreadable — INCONCLUSIVE, not absent: {unreadable[:5]}"
    assert len(sites) == 1, f"expected ONE construction site, found {sites}"
    src = io.open(os.path.join(root, "refs", "refc_v3.py"), encoding="utf-8").read()
    assert "        if cfg.max_speed_input:\n" in src, \
        "the gate's exact form changed — re-derive this test, do not relax it"


def test_both_trainer_entry_points_reach_the_gate():
    """⭐⭐ EMPIRICAL REACHABILITY: not *"the gate works when called"* but *"the gate
    is reached on the configs the real entry points produce"*. ``preflight`` and
    ``train`` both build through ``_pin_trainer_cfg``; a counter on the
    conditioner's ``__init__`` measures whether it ran."""
    tr = _trainer()
    real_init, calls = msi.MaxSpeedConditioner.__init__, []

    def counting(self, *a, **k):
        calls.append(1)
        return real_init(self, *a, **k)

    msi.MaxSpeedConditioner.__init__ = counting
    try:
        cfg = _pin(tr, _args(tr))                       # no flag
        m = v3.RefCV3Model(cfg)
        n_default = len(calls)
        del m
        cfg2 = _pin(tr, _args(tr, "--max-speed-input"))  # the SAME helper
        m = v3.RefCV3Model(cfg2)
        n_optin = len(calls) - n_default
        del m
    finally:
        msi.MaxSpeedConditioner.__init__ = real_init
    assert n_default == 0, "the gate did NOT hold on a trainer-produced config"
    assert n_optin == 1, f"the gate is not REACHED — built {n_optin}, not 1"


# ==========================================================================
# 4 — the TRAINER: the flag exists, and a dead flag is REFUSED
# ==========================================================================
def test_the_flag_exists_on_the_parser():
    """⛔ THE DEFECT THIS PINS. ``--tac-goal-tok-head`` silently gated the ENTIRE
    tactical vocabulary because the head existed, ``refc_v3.py`` gated on a config
    field, and NO CLI FLAG could ever set it. A channel with no flag is a channel
    no run can use."""
    tr = _trainer()
    dests = {a.dest for a in tr.build_parser()._actions}
    assert "max_speed_input" in dests
    assert "max_speed_mode" in dests
    a = _args(tr)
    assert a.max_speed_input is False and a.max_speed_mode == msi.DEFAULT_MODE


def test_the_pin_puts_the_flag_and_the_mode_on_the_CONFIG():
    """⛔ NOT ONLY ON ``args``. ``refcv3_arm.rebuild_config`` rebuilds through
    ``_pin_trainer_cfg``, so a mode that lived only on the Namespace would be lost
    on every roll and the checkpoint would rebuild with the wrong encoding."""
    tr = _trainer()
    cfg = _pin(tr, _args(tr, "--max-speed-input"))
    assert cfg.max_speed_input is True
    assert cfg.max_speed_cfg.enabled is True
    assert cfg.max_speed_cfg.mode == "quantized"
    cfg_raw = _pin(tr, _args(tr, "--max-speed-input", "--max-speed-mode", "raw"))
    assert cfg_raw.max_speed_cfg.mode == "raw"
    # CONTROL: without the flag the config is untouched.
    assert _pin(tr, _args(tr)).max_speed_input is False


def test_a_dead_flag_is_REFUSED_not_ignored():
    """The four refusals this trainer already implements, extended by this one:
    ``--w-agent`` under ``--agents off``, ``--nav-args`` without ``--nav-from-v7``,
    both halves of the P14 selection split, ``--tac-goal-tok-head`` under kin3."""
    tr = _trainer()
    with pytest.raises(SystemExit, match="v7-labels"):
        tr._check_max_speed_args(_args(tr, "--max-speed-input"))
    with pytest.raises(SystemExit, match="INERT"):
        _pin(tr, _args(tr, "--max-speed-mode", "raw"))
    with pytest.raises(SystemExit, match="FLAT"):
        _pin(tr, _args(tr, "--max-speed-input", arm="flat"))
    # CONTROLS, same breath: none of the three fires when the condition is gone.
    tr._check_max_speed_args(_args(tr))
    _pin(tr, _args(tr, "--max-speed-input", "--max-speed-mode", "raw"))


def test_preflight_refuses_EARLY_and_says_it_is_not_the_flag_failing():
    """⛔ THE LATE-FAILURE TRAP. `preflight` builds its corpus with
    `_synth_episodes` (CI-only), and a synthetic clip carries no
    `speed_max_input` — so with the flag on, the run used to reach the loss step
    and die there, AFTER the gates had printed PASS. An operator preflighting
    their real launch line would read that as "the flag is broken" and drop the
    channel: a true error that produces a wrong next action. The refusal now
    fires at the top and says explicitly that it is not the flag failing."""
    tr = _trainer()
    a = _args(tr, "--max-speed-input", "--v7-labels", "/x/labels.jsonl.gz")
    a.preflight = True
    with pytest.raises(SystemExit) as ei:
        tr._check_max_speed_args(a)
    msg = str(ei.value)
    assert "NOT the flag failing" in msg and "SYNTHETIC" in msg
    # CONTROLS, same breath: the SAME args without `preflight` are allowed, and
    # a preflight WITHOUT the flag is untouched.
    a.preflight = False
    tr._check_max_speed_args(a)
    b = _args(tr)
    b.preflight = True
    tr._check_max_speed_args(b)


def test_the_model_refuses_a_silent_drop_in_BOTH_directions():
    cfg, m = _smoke(False)
    with pytest.raises(ValueError, match="SILENTLY DROPPED"):
        _fwd(m, cfg, v_max_ms=V_MAX)
    cfg2, m2 = _smoke(True)
    with pytest.raises(ValueError, match="no v_max_ms reached"):
        _fwd(m2, cfg2)
    # CONTROL: supplied to a build that HAS the seam, it goes through.
    assert _fwd(m2, cfg2, v_max_ms=V_MAX, v_max_valid=V_OK)["max_speed_injected"]


def test_a_flat_build_refuses_the_flag_rather_than_dropping_it():
    cfg = v3.refc_v3_flat_config()
    cfg.max_speed_input = True
    with pytest.raises(ValueError, match="FLAT"):
        v3.RefCV3Model(cfg)


# ==========================================================================
# 5 — the LOADER: the shipped value, the units, the ladder
# ==========================================================================
def _label(clip_id, smi):
    lab = object.__new__(v7l.V7Label)
    object.__setattr__(lab, "clip_id", clip_id)
    object.__setattr__(lab, "_oracle", {"speed_max_input": smi})
    return lab


def _manifest(allow=True):
    return v7l.LabelManifest(path="/fake", md5="deadbeef", n_records=1,
                             schema_version="x", vocab="v7.0",
                             allow_oracle_nav=allow)


def _ds(labels):
    import types
    d = types.SimpleNamespace()
    d.v7_by_sid = {i: lab for i, lab in enumerate(labels)}
    d.episodes = [types.SimpleNamespace(episode_id=i)
                  for i in range(len(labels))]
    d.index = [(i, 0) for i in range(len(labels))]
    return d


def _enable(labels, mode=msi.DEFAULT_MODE):
    tr = _trainer()
    return tr.V3Dataset.enable_max_speed(_ds(labels), _manifest(), mode)


GOOD = {"v_max_ms": 11.19, "units": "m/s", "v_max_bucket_kmh": 50,
        "v_max_bucket_ms": 13.8889,
        "bucket_steps_kmh": list(msi.POSTED_LIMIT_STEPS_KMH)}


def test_the_field_is_behind_the_ORACLE_GATE_like_nav_command():
    """The record declares ``oracle: true`` / ``provenance: ego-future``, so the
    read goes through the SAME manifest permission ``nav_command`` does. An arm
    that used it is identifiable from its own artifacts."""
    with pytest.raises(v7l.OracleNavRefused, match="ORACLE"):
        v7l.oracle_max_speed(_label("c", GOOD), _manifest(allow=False))
    assert v7l.oracle_max_speed(_label("c", GOOD), _manifest())["v_max_ms"] == 11.19


def test_the_loader_ships_the_RAW_value_and_the_ladder_is_applied_ONCE():
    """⛔⛔ THE DOUBLE-QUANTIZATION HAZARD, MEASURED, NOT ARGUED. The record's
    ``v_max_bucket_ms`` is rounded to 4 dp (13.8889) while the ladder's 50 km/h
    step is 13.888888…, so the SHIPPED bucket is strictly GREATER than the step it
    names and snaps UP to the next one. MEASURED on the v8 train blob: feeding the
    shipped bucket back through the ladder moves **2,631 of 4,572 clips (57.5 %)
    one step up**. The loader therefore ships the RAW ``v_max_ms`` and the model
    quantizes exactly once."""
    assert 13.8889 > msi.POSTED_LIMIT_STEPS_MS[2], \
        "the 4 dp shipped bucket is no longer above its own step — re-derive"
    q, _ = msi.quantize_up(13.8889)
    assert round(q * 3.6) == 70, "the hazard's mechanism changed"
    rep = _enable([_label("c", GOOD)])
    ds = _ds([_label("c", GOOD)])
    _trainer().V3Dataset.enable_max_speed(ds, _manifest(), msi.DEFAULT_MODE)
    assert ds._max_speed_by_sid[0] == (11.19, 1.0), \
        "the loader must ship the RAW shipped value, not a pre-quantized one"
    assert rep["n_windows_with_ceiling"] == 1


def test_the_shipped_bucket_and_the_pinned_ladder_are_CROSS_CHECKED():
    with pytest.raises(SystemExit, match="two experiments"):
        _enable([_label("c", dict(GOOD, bucket_steps_kmh=[30, 50, 70, 100]))])
    with pytest.raises(SystemExit, match="shipped"):
        _enable([_label("c", dict(GOOD, v_max_bucket_kmh=70))])
    # CONTROL: the agreeing record passes.
    assert _enable([_label("c", GOOD)])["n_clips_with_block"] == 1


def test_undeclared_units_are_REFUSED_and_the_error_NAMES_THE_FIELD():
    """m/s vs km/h vs mph is a 1.61x spread and this programme published a 396 g
    anchor table from exactly this class."""
    bad = {k: v for k, v in GOOD.items() if k != "units"}
    with pytest.raises(SystemExit) as ei:
        _enable([_label("clip-A", bad)])
    msg = str(ei.value)
    assert "speed_max_input" in msg and "clip-A" in msg and "1.61" in msg


def test_an_empty_channel_is_REFUSED_because_the_field_is_a_v8_ADDITION():
    with pytest.raises(SystemExit, match="v8 addition"):
        _enable([_label("a", None), _label("b", None)])


def test_the_census_carries_the_number_that_decides_the_claim():
    """⛔ ``window_ceiling_frac`` is what a reader must quote before saying the
    channel carried information — the ``nav_args`` census rule verbatim — and the
    provenance caveat travels WITH it."""
    rep = _enable([_label("a", GOOD), _label("b", None)])
    assert rep["window_ceiling_frac"] == 0.5
    assert rep["n_clips_with_block"] == 1 and rep["n_clips"] == 2
    assert rep["provenance"].startswith("ego-future"), rep["provenance"]
    assert rep["control_units"] == "m_s"
    assert rep["mode"] == "quantized"
