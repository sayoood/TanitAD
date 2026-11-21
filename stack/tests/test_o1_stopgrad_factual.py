"""O1's LIT-3 stop-gradient lever — the CALLEE half, and the guard that stops
a caller/callee split from being written again.

⛔ WHY THIS FILE EXISTS (MEASURED 2026-09-17). ``train_v6_staged.py`` passed
``stopgrad_factual=`` into :func:`train_stage_a.stage_a_losses` while the
parameter DID NOT EXIST, so every O1 path — any run with ``w_o1_ctrl``,
``w_o1_fact`` or ``w_o1_scene`` nonzero — died with ``TypeError:
stage_a_losses() got an unexpected keyword argument 'stopgrad_factual'``.

The two halves did not disagree; one of them was REVERTED UNDER THE OTHER:

  * ``f1deba9`` (2026-09-02 00:37) landed the callee half, blob ``6f73897``.
  * ``0fcb5ff`` (2026-09-02 01:06 — the SAME NIGHT, the commit whose own
    message documents the ``git ls-tree -r`` silent-truncation trap and says
    it "reverted a good commit and reconstructed content that was never
    lost") restored ``train_stage_a.py`` to ``97e295f``, the PRE-flag blob.
    It is the ONLY commit that has touched the file since, and it left the
    caller half in ``train_v6_staged.py`` untouched.

So the lesson is not "write both halves" — both halves WERE written. It is
that a signature agreement spanning two files had no mechanical check, and a
revert could therefore delete one side silently.
:func:`test_every_kwarg_the_TRAINER_passes_is_ACCEPTED_by_the_callee` is that
check, and it is proven by MUTATION against the pre-fix blob, not by
inspection.

WHAT THE LEVER MEANS (LIT-3 / PhyLatent CASC — ``GOALS_AND_CLAIMS.md`` row
LIT-3, PUBLISHED): *"The factual prediction is treated as a stop-gradient
reference, so this loss separates the counterfactual branch without moving
the factual branch."* Only O1's SEPARATION term (response-form ``l_ctrl``)
stops seeing the factual waypoints. ``l_fact`` still trains them — that is
its job, and the module docstring's own attribution rule ("the absolute
factual position is L_factual's job, so attribution between the two losses
stays clean") is why the scope is exactly this narrow and no wider.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "stack" / "scripts"))
sys.path.insert(1, str(_ROOT / "stack"))

from train_stage_a import stage_a_losses          # noqa: E402

_REL = "stack/scripts/train_stage_a.py"
_MARKER = "stopgrad_factual"


# =========================================================================== #
# 0. the incumbent module, resolved BY CONTENT (never by HEAD)
# =========================================================================== #
@functools.lru_cache(maxsize=1)
def _pre_change_module():
    """Import the newest revision of ``train_stage_a.py`` that does NOT carry
    ``stopgrad_factual``. Resolved by CONTENT, not by ``HEAD`` (C75): HEAD
    moves under us — a sibling's whole-index commit sweeps in-progress files
    into it — and a HEAD-relative identity test then compares a module with
    itself and passes forever. Returns ``None`` -> the caller SKIPS; a skip is
    honest, a self-comparison dressed as a test is not.
    """
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--", _REL],
                             cwd=_ROOT, capture_output=True, timeout=300)
        if log.returncode != 0:
            return None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", sha + ":" + _REL], cwd=_ROOT,
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not r.stdout:
                continue
            if _MARKER.encode() in r.stdout:
                continue                      # already carries the change
            src, ref = r.stdout, sha
            break
        else:
            return None
    except Exception:
        return None
    tmp = Path(tempfile.mkdtemp()) / "train_stage_a_pre_stopgrad.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location(tmp.stem, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tmp.stem] = mod
    spec.loader.exec_module(mod)
    mod._ref = ref
    return mod


# =========================================================================== #
# 1. fixtures — every perturbed element is one the function actually CONSUMES
# =========================================================================== #
_S, _B, _W, _K = 32, 3, 4, 3


class _MockPredictor(nn.Module):
    """1-step contract of ``OperativePredictor`` (the shape
    ``test_stage_a_train`` uses): residual linear in the state AND the last
    action, so the rollout has something to differentiate."""

    def __init__(self, s=_S, a=3):
        super().__init__()
        self.lin_a = nn.Linear(a, s)
        self.lin_s = nn.Linear(s, s)

    def forward(self, win_s, win_a, intent=None):
        return {1: win_s[:, -1] + 0.1 * self.lin_s(win_s[:, -1])
                + 0.1 * self.lin_a(win_a[:, -1])}


def _fixtures(perturb=False):
    torch.manual_seed(0)
    from tanitad.models.metric_dynamics import StepDisplacementReadout
    pred = _MockPredictor()
    sr = StepDisplacementReadout(_S, hidden=32)
    states = torch.randn(_B, _W, _S)
    aw2 = torch.stack([torch.full((_W,), 0.02), torch.full((_W,), 0.5)],
                      dim=-1).unsqueeze(0).repeat(_B, 1, 1)
    out = dict(pred=pred, sr=sr, states=states, aw2=aw2, fa2=aw2.clone(),
               v0=torch.full((_B,), 8.0), gt=torch.randn(_B, _K, 2) * 0.1,
               z_true=torch.randn(_B, _S),
               dk=torch.tensor([0.01, -0.02, 0.03]),
               da=torch.tensor([0.5, -1.0, 1.5]))
    if perturb:
        # ⚠️ 1-ulp nudge at the LAST window step. The predictor reads
        # ``win_s[:, -1]`` ONLY, so a nudge at ``states[0, 0, 0]`` is
        # INVISIBLE and the control silently proves nothing. The first draft
        # of this control did exactly that and dutifully reported "no
        # difference" — which is why the index is spelled out here.
        s = states.clone()
        s[0, -1, 0] = torch.nextafter(s[0, -1, 0], torch.tensor(1e9))
        out["states"] = s
    return out


def _call(fn, flag=None, perturb=False, **over):
    f = _fixtures(perturb)
    kw = dict(dkappa=0.05, daccel=3.0, rand_dk=f["dk"], rand_da=f["da"],
              w_ctrl=1.0, w_fact=1.0, w_scene=0.3, ctrl_form="response")
    kw.update(over)
    if flag is not None:
        kw[_MARKER] = flag
    torch.manual_seed(1234)
    L = fn(f["pred"], f["sr"], f["states"], f["aw2"], f["fa2"], f["v0"],
           f["gt"], f["z_true"], _K, **kw)
    ps = [("pred." + n, p) for n, p in f["pred"].named_parameters()]
    ps += [("sr." + n, p) for n, p in f["sr"].named_parameters()]
    return L, ps, torch.get_rng_state()


def _bits(t):
    """RAW BYTES — not ``allclose``, not ``torch.equal``. Bit-identity is the
    claim, so the comparison is on the bit pattern itself."""
    return t.detach().contiguous().cpu().numpy().tobytes()


_SCALARS = ("loss", "l_ctrl", "l_fact", "l_scene", "l_scene_cf",
            "l_scene_true", "factual_ade")


def _fwd(L):
    d = {k: _bits(L[k]) for k in _SCALARS if k in L}
    d.update({"arm:" + a_: _bits(v) for a_, v in L["l_ctrl_arms"].items()})
    d["basis_dims"] = repr(L["basis_dims"])
    return d


def _grads(L, ps, key="loss"):
    gs = torch.autograd.grad(L[key], [p for _, p in ps], retain_graph=True,
                             allow_unused=True)
    return {n: (None if g is None else _bits(g)) for (n, _), g in zip(ps, gs)}


def _diff(d1, d2):
    return sorted(k for k in set(d1) | set(d2) if d1.get(k) != d2.get(k))


# =========================================================================== #
# 2. the API surface — additive, and defaulting to the incumbent
# =========================================================================== #
def test_the_parameter_EXISTS_and_defaults_to_the_incumbent():
    """The bug in one line: the trainer's keyword must be in the signature.
    A lever that defaults ON is a silent behaviour change on a live run, so
    the default is asserted too, and keyword-only so no positional call can
    reach it by accident."""
    p = inspect.signature(stage_a_losses).parameters
    assert _MARKER in p, (
        "train_v6_staged.py:4296 passes stopgrad_factual= ; without this "
        "parameter every O1 path raises TypeError")
    assert p[_MARKER].default is False
    assert p[_MARKER].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_signature_change_is_ADDITIVE_ONLY():
    """Every banked arm must stay launchable bit-identically, which requires
    that nothing was removed or reordered — only appended."""
    old = _pre_change_module()
    if old is None:
        pytest.skip("git could not produce a pre-stopgrad_factual module")
    a = list(inspect.signature(old.stage_a_losses).parameters)
    b = list(inspect.signature(stage_a_losses).parameters)
    assert b[:len(a)] == a, "a parameter was removed or reordered"
    assert b[len(a):] == [_MARKER], b[len(a):]


# =========================================================================== #
# 3. ⛔ THE CLASS GUARD — caller/callee signature agreement, by MUTATION
# =========================================================================== #
def _kwargs_passed_to(src: bytes, callee: str):
    """Every keyword name any call to ``callee`` passes, per call site.
    ``**splat`` call sites are reported separately: they cannot be checked
    statically and must not be silently counted as clean."""
    tree = ast.parse(src)
    sites, splats = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = (f.id if isinstance(f, ast.Name)
                else f.attr if isinstance(f, ast.Attribute) else None)
        if name != callee:
            continue
        if any(k.arg is None for k in node.keywords):
            splats += 1
        sites.append((node.lineno,
                      sorted(k.arg for k in node.keywords if k.arg)))
    return sites, splats


def _files_calling(callee: str):
    out = []
    for d in ("scripts", "tests"):
        for p in sorted((_ROOT / "stack" / d).glob("*.py")):
            b = p.read_bytes()
            if (callee + "(").encode() in b:
                out.append((p, b))
    return out


def test_every_kwarg_the_TRAINER_passes_is_ACCEPTED_by_the_callee():
    """⛔ THE GUARD THIS FILE EXISTS FOR. A cross-file signature agreement
    with no check is one revert away from a TypeError that only fires on the
    paths nobody runs in CI — which is exactly how this one survived 15 days.
    Checked over EVERY call site in ``stack/scripts`` and ``stack/tests``,
    not just the one that broke."""
    accepted = set(inspect.signature(stage_a_losses).parameters)
    files = _files_calling("stage_a_losses")
    assert files, "found no call sites — the scan itself is broken"
    bad = []
    for path, src in files:
        sites, _ = _kwargs_passed_to(src, "stage_a_losses")
        for lineno, kws in sites:
            for kw in kws:
                if kw not in accepted:
                    bad.append(path.name + ":" + str(lineno) + " -> " + kw)
    assert not bad, ("call sites pass keywords stage_a_losses does not "
                     "accept: " + repr(bad))


def test_the_class_guard_FIRES_on_the_exact_defect_it_was_written_for():
    """⛔ MUTATION, NOT INSPECTION. A guard that has never been seen to fail
    is not evidence. Re-run the SAME check against the PRE-FIX module — the
    real historical state, resolved by content — and it must name
    ``stopgrad_factual`` at the trainer's call site."""
    old = _pre_change_module()
    if old is None:
        pytest.skip("git could not produce a pre-stopgrad_factual module")
    accepted_then = set(inspect.signature(old.stage_a_losses).parameters)
    assert _MARKER not in accepted_then, (
        "the pre-change module already accepts it — the reference resolution "
        "returned the FIXED file, so this control is vacuous")
    trainer = (_ROOT / "stack" / "scripts" / "train_v6_staged.py").read_bytes()
    sites, _ = _kwargs_passed_to(trainer, "stage_a_losses")
    assert sites, "no stage_a_losses call site found in train_v6_staged.py"
    fired = [(ln, kw) for ln, kws in sites for kw in kws
             if kw not in accepted_then]
    assert fired, ("the guard did NOT fire against the pre-fix signature — "
                   "it cannot detect the defect it was written for")
    assert any(kw == _MARKER for _, kw in fired), fired


def test_the_trainer_really_does_pass_it():
    """Pins the caller half too. If a future edit drops the keyword from the
    trainer, the callee parameter becomes dead code and the guard above would
    pass vacuously — so assert the call site is still there."""
    trainer = (_ROOT / "stack" / "scripts" / "train_v6_staged.py").read_bytes()
    sites, _ = _kwargs_passed_to(trainer, "stage_a_losses")
    assert any(_MARKER in kws for _, kws in sites), \
        "train_v6_staged.py no longer passes stopgrad_factual"


# =========================================================================== #
# 4. bit-identity on the default path — PROVEN, with a control that bites
# =========================================================================== #
def test_default_path_is_BIT_IDENTICAL_to_the_pre_change_module():
    """⛔ THE ONE THAT PROTECTS EVERY BANKED ARM. Every landed ``config.json``
    in the Research Lab carries ``"o1_stopgrad_factual": false``, so the
    False path is the only path any measured number came from: it must be
    bit-identical, forward AND backward, to the module that produced them.

    Backward is the half that matters here — a ``detach`` cannot move a
    forward value, so a forward-only comparison would pass even if the
    default path had been silently stop-gradded."""
    old = _pre_change_module()
    if old is None:
        pytest.skip("git could not produce a pre-stopgrad_factual module")
    Lo, Po, Ro = _call(old.stage_a_losses)            # no such kwarg there
    Ln, Pn, Rn = _call(stage_a_losses, flag=False)
    assert _diff(_fwd(Lo), _fwd(Ln)) == [], "the DEFAULT forward MOVED"
    assert _diff(_grads(Lo, Po), _grads(Ln, Pn)) == [], \
        "the DEFAULT gradients MOVED"
    for key in ("l_ctrl", "l_fact", "l_scene", "l_scene_cf", "l_scene_true"):
        assert _diff(_grads(Lo, Po, key), _grads(Ln, Pn, key)) == [], key
    assert torch.equal(Ro, Rn), "the RNG draw-count changed"


def test_NEGATIVE_CONTROL_the_bit_identity_comparator_CAN_fail():
    """Prove the comparison above BITES. Without this, "no differences" is
    indistinguishable from a comparator that never looks."""
    La, Pa, _ = _call(stage_a_losses, flag=False)
    Lb, Pb, _ = _call(stage_a_losses, flag=False, perturb=True)
    assert _diff(_fwd(La), _fwd(Lb)) != [], \
        "a 1-ulp nudge on a CONSUMED input went undetected"
    assert _diff(_grads(La, Pa), _grads(Lb, Pb)) != []


# =========================================================================== #
# 5. the True branch: ARMED, and SCOPED to the separation term
# =========================================================================== #
def test_the_flag_is_ARMED_and_is_a_BACKWARD_only_change():
    """A stop-gradient changes the graph, never the value. So True vs False
    must be identical in the forward and DIFFERENT in the gradient — and if
    the forward moved, something other than a detach was implemented."""
    Lf, Pf, _ = _call(stage_a_losses, flag=False)
    Lt, Pt, _ = _call(stage_a_losses, flag=True)
    assert _diff(_fwd(Lf), _fwd(Lt)) == [], \
        "a detach moved a forward value — that is not a stop-gradient"
    assert _diff(_grads(Lf, Pf, "l_ctrl"), _grads(Lt, Pt, "l_ctrl")) != [], \
        "the flag is INERT: l_ctrl gradients did not change"


def test_under_True_the_separation_term_STILL_reaches_the_predictor():
    """Distinguishes "the FACTUAL reference was frozen" from "the
    COUNTERFACTUAL branch was frozen". Detaching ``wp_c`` would leave
    ``l_ctrl`` with no path to the predictor at all (``an_c``/``an_f`` are
    computed under ``no_grad``), so a nonzero gradient is the discriminator."""
    Lt, Pt, _ = _call(stage_a_losses, flag=True)
    g = _grads(Lt, Pt, "l_ctrl")
    live = [n for n, b in g.items() if b is not None
            and torch.frombuffer(bytearray(b), dtype=torch.float32
                                 ).abs().sum() > 0]
    assert live, "l_ctrl no longer reaches the predictor under the flag"


def test_the_flag_is_SCOPED_to_the_separation_term_and_nothing_else():
    """LIT-3 stop-grads the factual prediction in the SEPARATION term only.
    ``l_fact`` keeps training the factual branch, and ``z_f`` inside
    ``l_scene``'s counterfactual deltas is NOT detached — a wider detach
    would be a different loss wearing the same flag."""
    Lf, Pf, _ = _call(stage_a_losses, flag=False)
    Lt, Pt, _ = _call(stage_a_losses, flag=True)
    for key in ("l_fact", "l_scene", "l_scene_cf", "l_scene_true"):
        assert _diff(_grads(Lf, Pf, key), _grads(Lt, Pt, key)) == [], \
            key + " gradients moved — the detach is wider than LIT-3 specifies"
    assert _diff(_grads(Lf, Pf, "l_ctrl"), _grads(Lt, Pt, "l_ctrl")) != []


def test_the_implemented_detach_IS_the_SPECIFIED_detach():
    """⭐ The semantic proof, independent of the implementation: re-derive the
    response-form separation term here, with an EXPLICIT ``wp_f.detach()``,
    and require the gradients to match bitwise. Pins WHICH tensor is frozen —
    not merely that something changed."""
    from tanitad.models.metric_dynamics import (decode_transitions,
                                                rollout_transitions)
    from train_p8_occupancy import lift_actions3
    import train_stage_a as M

    def manual(detach_factual):
        f = _fixtures()
        torch.manual_seed(1234)
        aw3, fa3 = lift_actions3(f["aw2"], f["fa2"], f["v0"])
        trans_f = rollout_transitions(f["pred"], f["states"], aw3, fa3, _K)
        wp_f, _ = decode_transitions(f["sr"], trans_f, _K)
        wp_f = wp_f.float()
        with torch.no_grad():
            an_f = M.analytic_endpoints(f["aw2"], f["fa2"], f["v0"],
                                        _K).float()
        ref = wp_f.detach() if detach_factual else wp_f
        arms = {}
        for arm in M.TRAIN_ARMS:
            aw_c, fa_c = M.build_cf_actions(f["aw2"], f["fa2"], arm,
                                            dkappa=0.05, daccel=3.0,
                                            dk=f["dk"], da=f["da"])
            aw3c, fa3c = lift_actions3(aw_c, fa_c, f["v0"])
            trans_c = rollout_transitions(f["pred"], f["states"], aw3c, fa3c,
                                          _K)
            wp_c, _ = decode_transitions(f["sr"], trans_c, _K)
            with torch.no_grad():
                an_c = M.analytic_endpoints(aw_c, fa_c, f["v0"], _K).float()
            arms[arm] = ((wp_c.float() - ref) - (an_c - an_f)).abs().mean()
        ps = [("pred." + n, p) for n, p in f["pred"].named_parameters()]
        ps += [("sr." + n, p) for n, p in f["sr"].named_parameters()]
        return {"l_ctrl": torch.stack(list(arms.values())).mean()}, ps

    Md, Pd = manual(True)
    Ml, Pl = manual(False)
    assert _diff(_grads(Md, Pd, "l_ctrl"), _grads(Ml, Pl, "l_ctrl")) != [], \
        "the two re-derivations are identical — this control is vacuous"
    Lt, Pt, _ = _call(stage_a_losses, flag=True)
    Lf, Pf, _ = _call(stage_a_losses, flag=False)
    assert _diff(_grads(Lt, Pt, "l_ctrl"), _grads(Md, Pd, "l_ctrl")) == [], \
        "True does not match an explicit wp_f.detach()"
    assert _diff(_grads(Lf, Pf, "l_ctrl"), _grads(Ml, Pl, "l_ctrl")) == [], \
        "False does not match the un-detached form"


def test_absolute_form_has_no_factual_reference_so_the_flag_is_a_NOOP():
    """``ctrl_form="absolute"`` scores ``wp_c`` against the analytic endpoint
    directly — ``wp_f`` never enters it. The flag must therefore change
    nothing at all there, forward or backward; if it does, the detach escaped
    the response branch."""
    Lf, Pf, _ = _call(stage_a_losses, flag=False, ctrl_form="absolute")
    Lt, Pt, _ = _call(stage_a_losses, flag=True, ctrl_form="absolute")
    assert _diff(_fwd(Lf), _fwd(Lt)) == []
    assert _diff(_grads(Lf, Pf), _grads(Lt, Pt)) == []
