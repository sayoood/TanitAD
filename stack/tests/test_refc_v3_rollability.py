"""D-ROLL-1 — a banked checkpoint must rebuild with the parameter set it was
trained with, and the gate that guarantees it must be REACHED on every path.

⭐⭐ THE REGRESSION THIS PINS (2026-09-06). ``RefCV3Model.__init__`` built
``tac_goal_tok_head`` whenever ``tac_vocab_version != "kin3"``. Every v7.0
checkpoint in the programme was trained BEFORE that head existed, so a rebuild
carried 11,286 params (22 tokens x (d_tac + 1)) that the recorded
``param_breakdown`` does not name, and ``refcv3_arm.cross_check_config``
correctly REFUSED. In one commit that made ``refcv4b@40284``, all three local
refcv3 checkpoints AND the live refcv5 A40 run unrollable — and refcv5
unresumable, since a resume rebuilds through this same file.

⛔ THE GUARD WAS RIGHT; THE CONSTRUCTION WAS WRONG. Nothing here loosens a
cross-check. The head is now OPT-IN (``RefCV3Config.tac_goal_tok_head``,
default OFF), which is the rule this config class already states twice —
``nav_args_inject`` (*"no banked arm's recipe changes"*) and the v4 pins
(*"a default that moved would silently change a training in flight"*).

⭐ WHAT MAKES THIS A REACHABILITY TEST, NOT ONLY A CORRECTNESS TEST. MEASURED
the same night: ``refc_v3_train.main`` runs ``preflight`` only under
``--preflight`` and otherwise calls ``train()`` directly, so a guard wired into
``preflight`` alone covers ONE launch path of two. Correctness and wiring are
different claims. :func:`test_the_head_has_exactly_one_construction_site` and
:func:`test_every_config_production_path_reaches_the_gate` assert the gate is
ACTUALLY REACHED — the first structurally (there is no second place the head can
be built), the second empirically (a counter fires through the trainer's own
config-production helper, on the configs both entry points produce).

⛔ TWO-SIDED. Each behavioural test carries the opposite build in the SAME
BREATH, so a gate stuck in either position fails. The SOURCE-level mutation —
restore the unconditional construction, watch the real instrument refuse the
real refcv4b checkpoint — lives in
``Research/2026-09-06-rollability/raw/run_mutation_proof.py``; a test that can
only pass measures nothing.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import io
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.refs import refc_v3 as v3                      # noqa: E402

#: 22 tokens x (d_tac + 1). ⛔ DERIVED, never typed — the same rule the sibling
#: rung test states: a literal here would silently re-date the head's cost.
TACGOAL = 22 * (v3.RefCV3Config().d_tac + 1)


def _smoke(vocab: str, *, goal_head: bool):
    """A cheap hier build at the requested (vocabulary, flag)."""
    from tanitad.refs import refc
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = refc.CNNEncoderConfig(
        in_channels=cfg.core.encoder.in_channels, image_size=64,
        base_width=8, blocks=(1, 1, 1, 1))
    cfg.tac_vocab_version = vocab
    cfg.tac_goal_tok_head = goal_head
    return v3.RefCV3Model(cfg), cfg


# ==========================================================================
# 1 — the rollability contract itself, both sides in one breath
# ==========================================================================
def test_v7_default_build_carries_no_goal_token_head():
    """⛔ THE CONTRACT: the DEFAULT v7.0 build is what a banked checkpoint
    rebuilds through, so it must carry neither the head nor a ledger line."""
    off, _ = _smoke("v7.0", goal_head=False)
    bd_off = v3.param_breakdown_v3(off)
    assert off.tac_goal_tok_head is None, \
        "D-ROLL-1 FAILED: the goal-token head was built without being asked " \
        "for — every banked v7.0 checkpoint is unrollable again"
    assert "tac_goal_tok_head" not in bd_off, \
        "the ledger names a head that was not built"

    # ⭐ SAME-BREATH CONTROL, and it is what makes the assertion above a
    # measurement rather than a build that is broken in a different way: asked
    # for, the head EXISTS, at its derived width, on its own ledger line.
    on, cfg_on = _smoke("v7.0", goal_head=True)
    bd_on = v3.param_breakdown_v3(on)
    assert on.tac_goal_tok_head is not None
    # ⚠️ DERIVED AT THE RUNG BEING BUILT, never the default's constant. The
    # smoke rung's `d_tac` is not 512, so the shared-`d_tac` cost differs there;
    # a literal here would be a true number quoted outside its scope.
    want = 22 * (cfg_on.d_tac + 1)
    assert bd_on["tac_goal_tok_head"] == want, (bd_on["tac_goal_tok_head"], want)
    assert bd_on["total"] - bd_off["total"] == want, \
        "the opt-in must cost EXACTLY the head and nothing else"
    # ...and at the DEFAULT `d_tac` the cost is the programme's banked 11,286 —
    # the exact figure absent from every recorded `param_breakdown`, which is
    # what made every v7.0 checkpoint unrollable.
    assert TACGOAL == 11_286, TACGOAL


def test_kin3_refuses_the_head_even_when_it_is_explicitly_asked_for():
    """⛔ The vocabulary is NECESSARY, and the flag does not override it.
    ``kin3`` has no tactical goal vocabulary, so 22 logits there could never be
    supervised — the defect ``effective_mask`` exists to prevent."""
    m, _ = _smoke("kin3", goal_head=True)
    assert m.tac_goal_tok_head is None
    assert "tac_goal_tok_head" not in v3.param_breakdown_v3(m)
    # CONTROL, same breath: the build is otherwise healthy — the kin3 ACTION
    # heads are there — so the absence above is scoped to the goal head and is
    # not a broken model.
    assert m.lat_head_tac.out_features == 3


# ==========================================================================
# 2 — ONE PREDICATE, ONE CONSUMER (the stale-consumer class)
# ==========================================================================
@pytest.mark.parametrize("vocab", ["kin3", "v7.0"])
@pytest.mark.parametrize("flag", [False, True])
def test_ledger_line_tracks_the_built_object(vocab, flag):
    """⭐⭐ THE ANTI-DRIFT PIN. ``param_breakdown_v3`` reports this line by
    reading the BUILT OBJECT, never by re-deriving the construction condition.

    ⚠️ WHY IT IS PINNED. The defect corrected the same night in ``refcv3_arm``
    was exactly a second copy of a convention going stale: a comment claiming
    ``a_star`` was computed *"exactly as the trainer computes it"* — true until
    the trainer was corrected on 2026-09-04, never updated after. A ledger that
    re-derived ``_vv != "kin3" and cfg.tac_goal_tok_head`` would be that same
    second copy, free to drift from the constructor. This asserts there is no
    second copy: the ledger line exists if and only if the module does.
    """
    m, _ = _smoke(vocab, goal_head=flag)
    bd = v3.param_breakdown_v3(m)
    assert ("tac_goal_tok_head" in bd) == (m.tac_goal_tok_head is not None)
    if m.tac_goal_tok_head is not None:
        assert bd["tac_goal_tok_head"] == sum(
            p.numel() for p in m.tac_goal_tok_head.parameters())
    # and the ledger still sums exactly in BOTH states — an unaccounted module
    # is a capacity confound inside the very claim the arm exists to test.
    assert sum(v for k, v in bd.items() if k != "total") == bd["total"]


# ==========================================================================
# 3 — THE BAR, against REAL recorded run records (no checkpoint needed)
# ==========================================================================
#: MEASURED 2026-09-06, copied from each run's own ``config.json``. These are
#: the records ``refcv3_arm.cross_check_config`` compares a rebuild against, so
#: reproducing them IS rollability. ⛔ None carries a ``tac_goal_tok_head``
#: line — that is the fact the regression contradicted.
BANKED = {
    "refcv4b@40284": {
        "argv": ["--arm", "hier", "--size", "base", "--v2-cache", "/x/train",
                 "--v7-labels", "/x/train.jsonl.gz", "--image-hw", "256", "640",
                 "--n-anchors", "117", "--anchor-v0-conditioned",
                 "--anchor-control-units", "alat", "--sel-accel-max", "2.0",
                 "--goal-str", "--ego-state-inject", "--ego-dropout", "0.5",
                 "--nav-from-v7", "--out", "/x/out"],
        "param_breakdown": {"core": 104879584, "phi_tac": 1757440,
                            "str_goal_head": 771, "gstr_cond": 66816,
                            "tac_heads": 14364, "tac_latent_proj": 262656,
                            "scorer": 1145, "ego_inject": 25536,
                            "nav_inject": 50176, "total": 107058488}},
    "refcv3-b1-v72@30k": {
        "argv": ["--arm", "hier", "--size", "base", "--v2-cache", "/x/train",
                 "--v7-labels", "/x/train.jsonl.gz", "--image-hw", "256", "640",
                 "--nav-from-v7", "--out", "/x/out"],
        "param_breakdown": {"core": 104879522, "phi_tac": 1757440,
                            "str_goal_head": 771, "gstr_cond": 66816,
                            "tac_heads": 14364, "tac_latent_proj": 262656,
                            "scorer": 1156, "nav_inject": 50176,
                            "total": 107032901}},
    "refcv5-ddim@live": {
        "argv": ["--arm", "hier", "--size", "base", "--v2-cache", "/x/train",
                 "--v7-labels", "/x/train.jsonl.gz", "--image-hw", "256", "640",
                 "--n-anchors", "117", "--anchor-v0-conditioned",
                 "--anchor-control-units", "alat", "--sel-accel-max", "2.0",
                 "--goal-str", "--ego-state-inject", "--ego-dropout", "0.5",
                 "--sampler", "ddim", "--w-u0", "0.5", "--agents", "off",
                 "--nav-from-v7", "--out", "/x/out"],
        "param_breakdown": {"core": 106067312, "phi_tac": 1757440,
                            "str_goal_head": 771, "gstr_cond": 66816,
                            "tac_heads": 14364, "tac_latent_proj": 262656,
                            "scorer": 1145, "ego_inject": 25536,
                            "nav_inject": 50176, "total": 108246216}},
}


def _trainer():
    """``refc_v3_train.py`` by PATH — it is a script, and ``build_parser`` +
    ``_pin_trainer_cfg`` are the ONLY way a run's config is recoverable. This
    is ``refcv3_arm``'s own convention, reused rather than copied."""
    p = os.path.join(os.path.dirname(_STACK), "stack", "scripts",
                     "refc_v3_train.py")
    if not os.path.exists(p):
        pytest.skip(f"trainer not present at {p}")
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_roll", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_roll"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rebuild(tr, argv):
    args = tr.build_parser().parse_args(list(argv))
    base = v3.refc_v3_sized_config(args.size, hier=(args.arm == "hier"))
    return tr._pin_trainer_cfg(base, args), args


@pytest.mark.parametrize("name", sorted(BANKED))
def test_banked_records_rebuild_to_their_recorded_param_breakdown(name):
    """⛔⛔ THE BAR. A rebuild through the trainer's OWN parser must reproduce
    the recorded ledger key-for-key. This is precisely what
    ``refcv3_arm.cross_check_config`` compares, so a failure here IS an
    unrollable checkpoint — measured without needing the 1.3 GB weights."""
    rec = BANKED[name]
    tr = _trainer()
    cfg, args = _rebuild(tr, rec["argv"])
    assert cfg.tac_vocab_version == "v7.0", "fixture must exercise the v7 path"
    assert cfg.tac_goal_tok_head is False, \
        "a recorded run that never had the head must not rebuild with it"
    model = v3.RefCV3Model(cfg)
    try:
        bd = {k: int(v) for k, v in v3.param_breakdown_v3(model).items()}
    finally:
        del model
    assert bd == rec["param_breakdown"], (
        f"{name}: rebuilt ledger != recorded ledger. "
        f"only-in-rebuilt={sorted(set(bd) - set(rec['param_breakdown']))} "
        f"only-in-record={sorted(set(rec['param_breakdown']) - set(bd))} "
        f"total {bd['total']} vs {rec['param_breakdown']['total']}")


# ==========================================================================
# 4 — REACHABILITY: the gate cannot be bypassed, and it IS reached
# ==========================================================================
def test_the_head_has_exactly_one_construction_site_and_it_is_gated():
    """⭐ STRUCTURAL REACHABILITY. A gate only covers the paths that go through
    it, so assert there is no other path: ``TacGoalTokenHead`` is instantiated
    in exactly ONE place in ``tanitad/``, inside ``RefCV3Model.__init__``.

    ⚠️ This is the ``preflight``-only lesson generalised. A guard placed on one
    of two launch paths is not a guard; the way to stop that recurring is to
    leave only one path."""
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
            if "TacGoalTokenHead" not in src:
                continue
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "TacGoalTokenHead"):
                    sites.append((os.path.relpath(p, root), node.lineno))
    # ⛔ 0 hits is a claim about the SEARCH unless the search is asserted to
    # have READ its files. A same-breath non-zero control settles it.
    assert scanned > 50, f"scan read only {scanned} files (unreadable: {unreadable[:5]})"
    assert not unreadable, f"files unreadable — INCONCLUSIVE, not absent: {unreadable[:5]}"
    assert len(sites) == 1, f"expected ONE construction site, found {sites}"

    # ...and it is inside RefCV3Model.__init__, under a gate that reads the cfg.
    src = io.open(os.path.join(root, "refs", "refc_v3.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    holder = None
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "TacGoalTokenHead"):
                    holder = (cls.name, fn.name)
    assert holder == ("RefCV3Model", "__init__"), holder
    assert 'if _vv != "kin3" and bool(getattr(cfg, "tac_goal_tok_head", False)):' \
        in src, "the gate's exact form changed — re-derive this test, do not " \
                "relax it"


def test_every_config_production_path_reaches_the_gate():
    """⭐⭐ EMPIRICAL REACHABILITY. Not *"the gate works when called"* but
    *"the gate is actually reached on the configs the real entry points
    produce"*.

    Both trainer entry points — ``preflight`` and ``train`` — build their
    config through ``_pin_trainer_cfg`` (asserted by AST below), so a config
    that helper produces is what every launch and every roll actually
    constructs from. A counter on the head's ``__init__`` then measures whether
    the gate ran, rather than assuming it did."""
    tr = _trainer()

    # (a) AST: both entry points reach ONE config-production helper.
    src = io.open(os.path.join(os.path.dirname(_STACK), "stack", "scripts",
                               "refc_v3_train.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    builds, pins = set(), set()
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                nm = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if nm == "RefCV3Model":
                    builds.add(fn.name)
                if nm == "_pin_trainer_cfg":
                    pins.add(fn.name)
    assert {"preflight", "train"} <= builds, builds     # control: both are real
    assert {"preflight", "train"} <= pins, (
        f"a launch path builds a model without the trainer's config helper — "
        f"this is the `preflight`-only wiring defect. builds={sorted(builds)} "
        f"pins={sorted(pins)}")

    # (b) EMPIRICAL: count the head's construction on a config that helper made.
    from tanitad.refs import tac_goal_head as tg
    real_init, calls = tg.TacGoalTokenHead.__init__, []

    def counting(self, *a, **k):
        calls.append(1)
        return real_init(self, *a, **k)

    argv = BANKED["refcv3-b1-v72@30k"]["argv"]
    tg.TacGoalTokenHead.__init__ = counting
    try:
        cfg, _ = _rebuild(tr, argv)
        assert cfg.tac_vocab_version == "v7.0"
        m = v3.RefCV3Model(cfg)
        n_default = len(calls)
        del m
        # SAME-BREATH CONTROL: flip only the flag on the SAME helper-produced
        # config. A counter that never increments would report "not reached"
        # for a gate that is simply never exercised — that is the guard that
        # cannot fail, and it measures nothing.
        cfg.tac_goal_tok_head = True
        m = v3.RefCV3Model(cfg)
        n_optin = len(calls) - n_default
        del m
    finally:
        tg.TacGoalTokenHead.__init__ = real_init
    assert n_default == 0, \
        "the gate did NOT hold on a config the trainer's own helper produced"
    assert n_optin == 1, \
        f"the gate is not REACHED — opting in built {n_optin} heads, not 1"
