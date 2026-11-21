"""The launch gate's ``v8f`` profile (v8F rung alpha, arm alpha-1: the refcv8 recipe with ONE lever changed -- the residual
prior is the frozen learned kinematic floor P10) and its G-DVB / G-LIVE / G-CKPT / G-EVAL / G-CLOCK clauses.

TWO profiles, one set of clauses (every test below runs on both unless it says otherwise):
  ``v8f``        arm alpha-1 AS LAUNCHED: ``--residual-prior p10_pose`` -- the POSE-ONLY P10. alpha-0's prior ``ha0_ext_pose``
                 reads past poses only, so the input set is unchanged and only what is LEARNED on top differs.
  ``v8f_steer``  the optional arm alpha-1s: ``--residual-prior p10`` -- the steer-reading P10, whose input (the recorded steer
                 at inference) has NO PI ruling (SPEC_REFCV7 A5). It is built and tested, and REFUSED by name under ``v8f``.

CPU only, no training data. Every expectation is a LITERAL authored here from PLAN_V8F section 6 and the brief, never
computed from the code under test. Every guard carries a DELIBERATE-REGRESSION arm that must go RED, and the arm's
regex in ``launch_gate_arms.ARMS`` must MATCH the reason the guard gives (otherwise the real gate CLI would grade the
arm MISCREDITED, not CAUGHT):

  profile      the refcv8 recipe + exactly the P10 levers; every other prior is REFUSED by name; the geometry is OPEN
  G-DVB        the BUILT P10: unfrozen, zeroed MLP, lost module, wrong sha, mode flip, wrong VARIANT
  G-LIVE       the emitted per-tick prior == the gate's INDEPENDENT recompute; P10 CALLED on every training forward;
               frozen (no Parameter, exactly zero gradient, bytes unchanged)
  G-CKPT       ckpt.pt carries the P10 identity
  G-EVAL       the eval loader builds the SAME P10 (non-persistent buffers: the state_dict comparison cannot see them)
  G-CLOCK      the in-decoder P10 equals the banked predictions <= 1e-4 m; ONE FRAME EARLY must FAIL
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]                  # .../stack
REPO = ROOT.parent
for _p in (str(ROOT), str(ROOT / "scripts"), str(Path(__file__).resolve().parent), str(REPO / "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import launch_gate_arms as ARMS  # noqa: E402
from tanitad.models import kinematic_prior as KP  # noqa: E402
from tanitad.models import p10_prior as P10  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
# the tiny-model helpers and fixtures of the P10 module tests (one definition of "the tiny P10 arm")
from test_p10_prior import (P10_FILE, P10_SHA256, T, _arm, _batch)  # noqa: E402,F401
from test_p10_pose_prior import POSE_FILE, POSE_FIXTURE, POSE_SHA256  # noqa: E402

R8 = LG.PROFILES["refcv8"]
LAUNCH_JSON = ROOT / "ops" / "runs.d" / "refcv8-r101-s0.argv.json"
#: where the launch tree's copy of each artifact sits on Thor (the join-defect-masks convention); the path is the launch
#: tree's, the IDENTITY is the sha256
THOR_DIR = "/home/nvidia/v8f_run/launch_tree/stack/tanitad/configs/v8f_p10/"
THOR_P10 = THOR_DIR + "p10_prior.pt"
THOR_P10_POSE = THOR_DIR + "p10_pose_prior.pt"
FULL_SHA = P10_SHA256                                       # the STEER artifact (v8F-mini)
GEOM = {"--steps", "--batch", "--eval-batches", "--eval-every", "--b1-guard-step", "--b1-guard-ref-ap2m"}


class Var:
    """One P10 variant's literals."""

    def __init__(self, variant, mode, prof, file, sha, fixture, thor, n_fit, open_ids):
        self.variant, self.mode, self.prof, self.file, self.sha = variant, mode, prof, file, sha
        self.fixture, self.thor, self.n_fit, self.open_ids = fixture, thor, n_fit, open_ids
        self.other_mode = "p10" if variant == "pose" else "p10_pose"
        self.sig_file = file.name

    def argv_obs(self):
        return ["--residual-prior", self.mode, "--p10-prior", str(self.file), "--p10-prior-sha256", self.sha]


VARS = {
    "pose": Var("pose", "p10_pose", LG.PROFILES["v8f"], POSE_FILE, POSE_SHA256, POSE_FIXTURE, THOR_P10_POSE, 4572,
                ["V8F-RUNG-GEOMETRY"]),
    "steer": Var("steer", "p10", LG.PROFILES["v8f_steer"], P10_FILE, P10_SHA256, POSE_FIXTURE.parent / "p10_fixture.npz",
                 THOR_P10, 600, ["V8F-RUNG-GEOMETRY", "V8F-P10-ACTION-INPUT"]),
}


@pytest.fixture(params=["pose", "steer"])
def V(request):
    return VARS[request.param]


@pytest.fixture
def fxv(V):
    z = np.load(V.fixture, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _arm_v(T_, V_, **kw):
    kw.setdefault("path", V_.file)
    kw.setdefault("sha", V_.sha)
    return _arm(T_, V_.mode, **kw)


# ----------------------------------------------------------------------------------------------- argv builders
def _pairs(a):
    out, i = [], 0
    while i < len(a):
        j = i + 1
        while j < len(a) and not a[j].startswith("--"):
            j += 1
        out.append((a[i], list(a[i + 1:j])))
        i = j
    return out


def _with(argv, flag, vals):
    ps, done = [], False
    for f, v in _pairs(argv):
        if f == flag:
            if not done and vals is not None:
                ps.append((f, [str(x) for x in vals]))
            done = True
        else:
            ps.append((f, v))
    if not done and vals is not None:
        ps.append((flag, [str(x) for x in vals]))
    return [t for f, v in ps for t in [f, *v]]


def alpha_pair(variant="pose"):
    """(alpha0, alpha1) from the committed refcv8 launch argv: alpha-0 = it, with a distinct --out; alpha-1 = alpha-0 with
    the residual prior swapped to P10 (``pose``: alpha-1 as launched; ``steer``: the optional alpha-1s). ONLY the prior
    flags (and --out) differ."""
    v = VARS[variant]
    d = json.loads(LAUNCH_JSON.read_text(encoding="utf-8"))
    base = list(d["argv"])
    a0 = _with(base, "--out", ["/home/nvidia/v8f_run/runs/v8f-a0-s0"])
    out1 = "/home/nvidia/v8f_run/runs/v8f-a1-s0" if variant == "pose" else "/home/nvidia/v8f_run/runs/v8f-a1s-s0"
    a1 = _with(_with(_with(_with(a0, "--out", [out1]), "--residual-prior", [v.mode]), "--p10-prior", [v.thor]),
               "--p10-prior-sha256", [v.sha])
    return a0, a1


_OPTS: dict = {}


@pytest.fixture(scope="module")
def _tau(tmp_path_factory):
    _a0, a1 = alpha_pair("pose")
    rec = tmp_path_factory.mktemp("tau") / "nav_tau.json"
    rec.write_text(json.dumps({"status": "OK", "tau": float(LG.flag_values(a1, "--nav-compliance-tau-rad")[0]),
                               "split_v2_cache": LG.flag_values(a1, "--v2-cache")[0],
                               "labels": LG.flag_values(a1, "--v7-labels")[0]}), encoding="utf-8")
    _OPTS["nav_tau_record"] = str(rec)
    return str(rec)


@pytest.fixture
def pair(V, _tau):
    return alpha_pair(V.variant)


def _rules(argv, prof):
    return LG.profile_argv_rules(prof, argv, _OPTS)


# =========================================================================== #
# the profile                                                                 #
# =========================================================================== #
def test_the_profile_is_refcv8_plus_exactly_the_P10_levers_and_refcv8_is_untouched(V):
    PROF = V.prof
    assert PROF["name"] == ("v8f" if V.variant == "pose" else "v8f_steer")
    assert PROF["required_values"]["--residual-prior"] == V.mode
    assert PROF["required_values"]["--p10-prior-sha256"] == V.sha
    assert LG.V8F_P10_SHA256 == POSE_SHA256 and LG.V8F_P10_STEER_SHA256 == FULL_SHA          # the two literals
    assert PROF["dvb_residual_prior"] == V.mode
    # the ONE lever: every other refcv8 literal that is not geometry is inherited VERBATIM
    for f, v in R8["required_values"].items():
        if f in GEOM or f == "--residual-prior":
            continue
        assert PROF["required_values"][f] == v, f
    assert set(PROF["required_values"]) == (set(R8["required_values"]) - GEOM) | {"--p10-prior-sha256"}
    assert not GEOM & set(PROF["required_values"])               # the tiny rung's geometry is NOT forced to the full run's
    assert PROF["required_flags"] == tuple(R8["required_flags"]) + ("--p10-prior", "--p10-prior-sha256")
    assert PROF["model_input_flags"] == tuple(R8["model_input_flags"]) + ("--p10-prior",)
    assert PROF["forbidden_values"]["--residual-prior"] == ("off", "ha0_ext", "ha0_ext_pose", "cv_yawrate", V.other_mode)
    # refcv8 is byte-for-byte what it was: the full-run geometry and its own prior rules
    assert R8["required_values"]["--steps"] == "48852" and R8["required_values"]["--batch"] == "16"
    assert R8["required_values"]["--residual-prior"] == "ha0_ext_pose"
    assert R8["forbidden_values"]["--residual-prior"] == ("off", "ha0_ext", "cv_yawrate")
    assert "dvb_residual_prior" not in R8 and "p10_prior" not in R8
    spec = PROF["p10_prior"]
    assert spec["variant"] == V.variant and spec["sha256"] == V.sha and spec["known_value_tol_m"] == 1e-4
    assert spec["ade6_bar_m"] == 0.05 and spec["fit_clips"] == V.n_fit
    assert spec["file_bytes"] == V.file.stat().st_size


def test_alpha1_as_launched_is_the_POSE_variant_and_the_steer_variant_is_refused_by_name_with_its_reason(_tau):
    pose, steer = VARS["pose"], VARS["steer"]
    _a0, a1 = alpha_pair("pose")
    _s0, a1s = alpha_pair("steer")
    r, _ = _rules(a1s, pose.prof)                                # the steer arm's argv under the alpha-1 profile
    text = " || ".join(r)
    assert any("--residual-prior" in x and "REFUSED" in x for x in r), r
    assert "THIS value: steer at inference: REJECTED by the PI 2026-10-17 (the model gets measured ego motion and its history, incl. yaw rate)" in text, r
    r2, _ = _rules(a1, steer.prof)                               # and the reverse
    assert any("--residual-prior" in x and "REFUSED" in x and "pose-only sibling" in x for x in r2), r2
    # `ha0_ext` (the steer ECHO) carries the same reason under alpha-1
    r3, _ = _rules(_with(a1, "--residual-prior", ["ha0_ext"]), pose.prof)
    assert "steer at inference: REJECTED by the PI 2026-10-17 (the model gets measured ego motion and its history, incl. yaw rate)" in " || ".join(r3)


def test_the_open_items_keep_the_token_from_PASSING_and_alpha1_has_exactly_one(V):
    PROF = V.prof
    ids = [it["id"] for it in PROF["open_items"]]
    assert ids == V.open_ids
    rs = LG.open_item_reasons(PROF)
    assert len(rs) == len(V.open_ids) and all(r.startswith("OPEN ITEM ") for r in rs)
    assert LG.open_item_reasons(R8) == []                        # refcv8's are all closed: the mechanism is per profile
    owners = {it["id"]: it["owner"] for it in PROF["open_items"]}
    assert owners["V8F-RUNG-GEOMETRY"].startswith("the Master Mind")
    if V.variant == "steer":
        assert owners["V8F-P10-ACTION-INPUT"].startswith("PI")
    else:
        assert "V8F-P10-ACTION-INPUT" not in owners and "V8F-P10-ARTIFACT" not in owners   # closed: no steer, refit banked


def test_the_alpha1_launch_argv_passes_every_argv_rule(V, pair):
    _a0, a1 = pair
    refusals, pending = _rules(a1, V.prof)
    assert refusals == [], refusals
    assert pending == []


def test_the_pair_differs_ONLY_in_the_prior_flags_and_the_run_dir(V, pair):
    a0, a1 = pair
    p0, p1 = dict((f, tuple(v)) for f, v in _pairs(a0)), dict((f, tuple(v)) for f, v in _pairs(a1))
    only1 = {f for f in p1 if f not in p0}
    changed = {f for f in p0 if f in p1 and p0[f] != p1[f]}
    assert only1 == {"--p10-prior", "--p10-prior-sha256"}
    assert changed == {"--residual-prior", "--out"}
    assert p0["--residual-prior"] == ("ha0_ext_pose",) and p1["--residual-prior"] == (V.mode,)
    assert p0["--out"] != p1["--out"]
    assert {f for f in p0 if f not in p1} == set()
    # alpha-0 is gated by the refcv8 profile (its recipe), alpha-1 by v8f: each is REFUSED by the other
    assert _rules(a0, R8)[0] == []
    assert any("--residual-prior" in r for r in _rules(a0, V.prof)[0])
    assert any("--residual-prior" in r for r in _rules(a1, R8)[0])


def test_DELIBERATE_REGRESSION_argv_each_wrong_prior_or_identity_is_REFUSED(V, pair):
    _a0, a1 = pair
    assert _rules(a1, V.prof)[0] == []                                          # the control: the unmutated argv is clean
    for mut, needle in [(("--residual-prior", ["ha0_ext_pose"]), "--residual-prior"),     # the baseline arm's prior
                        (("--residual-prior", ["off"]), "--residual-prior"),               # the refcv6 decoder
                        (("--residual-prior", ["ha0_ext"]), "--residual-prior"),           # the steer ECHO
                        (("--residual-prior", ["cv_yawrate"]), "--residual-prior"),        # no acceleration
                        (("--residual-prior", [V.other_mode]), "--residual-prior"),        # the OTHER P10 variant
                        (("--p10-prior-sha256", ["0" * 64]), "--p10-prior-sha256"),        # a different file than the banked one
                        (("--p10-prior-sha256", [P10_SHA256 if V.variant == "pose" else POSE_SHA256]), "--p10-prior-sha256"),
                        (("--p10-prior-sha256", None), "--p10-prior-sha256"),              # the identity not stated
                        (("--p10-prior", None), "--p10-prior")]:                           # the file not named
        flag, vals = mut
        refusals, _ = _rules(_with(a1, flag, vals), V.prof)
        assert any(needle in r for r in refusals), (mut, refusals)


def test_the_P10_file_is_a_MODEL_input_and_a_bound_data_input(V, pair, tmp_path):
    import types
    _a0, a1 = pair
    ctx = types.SimpleNamespace(argv=a1, path_map=[], prof=V.prof)
    out, rec = LG._sentinel_argv(ctx, V.prof["model_input_flags"], tmp_path)
    assert rec["kept"]["--p10-prior"] == V.thor                                  # the model job reads it for real
    assert "--p10-prior" not in rec["sentinel"]
    out_r8, rec_r8 = LG._sentinel_argv(types.SimpleNamespace(argv=a1, path_map=[], prof=R8), R8["model_input_flags"], tmp_path)
    assert "--p10-prior" in rec_r8["sentinel"]                                   # refcv8 would send it to the sentinel
    assert ("--p10-prior", V.thor) in LG.data_inputs(a1, V.prof)                 # and the token's data manifest binds it


# =========================================================================== #
# G-DVB: the declared-vs-built P10                                            #
# =========================================================================== #
def _ctx(V_, arm):
    return type("C", (), {"arm": arm, "argv": V_.argv_obs()})()


def _dvb(T_, V_, mode_arm=None, **kw):
    """-> (model, [mismatches that concern the prior]) for the tiny P10 arm built under an optional gate arm."""
    pat = LG._Patcher()
    try:
        if mode_arm:
            LG.apply_p10_arm(_ctx(V_, mode_arm), pat, T_)
        args, cfg, model = _arm_v(T_, V_, **kw)
        mm = D.check(model, args, T_.build_parser(), forbid_kinds=())
        base = [m for m in mm if "p10" in str(m).lower() or "residual" in str(m).lower()]
        req = [m for m in D.check_refcv7_required(model, args, residual_prior=V_.prof["dvb_residual_prior"])
               if "--residual-prior" in str(m)]
        return model, base, req
    finally:
        pat.restore()


def test_G_DVB_the_control_reads_clean_and_the_literal_is_the_alpha1_mode(T, V):
    assert D.V8F_RESIDUAL_PRIOR == "p10_pose"                                    # alpha-1 as launched
    _m, base, req = _dvb(T, V)
    assert base == [] and req == []
    # refcv7's own helper (no residual_prior=) still demands ha0_ext_pose: unchanged for every other profile
    args, _c, model = _arm_v(T, V)
    assert any("--residual-prior" in str(m) for m in D.check_refcv7_required(model, args))
    # the DVB helper compares to the profile's own literal: the pose literal against a steer build is a mismatch
    if V.variant == "steer":
        assert D.check_refcv7_required(model, args, residual_prior=D.V8F_RESIDUAL_PRIOR)


@pytest.mark.parametrize("arm,sig", [("p10_zero_weights", r"core\.decoder\.p10 live buffers"),
                                     ("p10_unfrozen", r"Parameter count"),
                                     ("p10_variant_flipped", r"core\.decoder\.p10\.variant")])
def test_G_DVB_DELIBERATE_REGRESSION_the_build_arms_go_RED_with_the_arms_signature(T, V, arm, sig):
    _m, base, _req = _dvb(T, V, arm)
    assert base, arm
    assert any(re.search(sig, str(m)) for m in base), [str(m) for m in base]
    assert re.search(ARMS.ARMS[arm]["sig"], " || ".join(str(m) for m in base)), arm


def test_G_DVB_DELIBERATE_REGRESSION_a_wrong_sha_a_missing_file_and_the_OTHER_variants_file_REFUSE_at_the_build(T, V):
    with pytest.raises(P10.P10Error, match="sha256"):
        _arm_v(T, V, sha="0" * 64)
    with pytest.raises(P10.P10Error, match="does not exist"):
        _arm_v(T, V, path=V.file.parent / "__nope__.pt")
    other = VARS["steer" if V.variant == "pose" else "pose"]
    with pytest.raises(P10.P10Error, match="variant"):                          # the other artifact WITH ITS OWN right sha
        _arm_v(T, V, path=other.file, sha=other.sha)
    # the gate turns that build refusal into a STATED refusal of the argv (a graded arm, not a crashed job)
    src = (ROOT / "scripts" / "launch_gate.py").read_text(encoding="utf-8")
    assert 'type(e_).__name__ != "P10Error"' in src and 'raise SystemExit(f"P10Error: {e_}")' in src
    assert re.search(ARMS.ARMS["p10_wrong_sha"]["sig"], "P10Error: P10 file 'x' has sha256 aa, the record names bb")
    assert re.search(ARMS.ARMS["p10_missing_file"]["sig"], "P10Error: P10 file 'x' does not exist")


# =========================================================================== #
# G-LIVE: the emitted prior is the declared P10, P10 is called, P10 is frozen //
# =========================================================================== #
def _live(T_, V_, fx_, arm=None, seed=3, ego_dropout=0.0):
    """One TRAINING step on real fixture windows with the gate's G-LIVE observer attached -> (record, judge reasons, det).
    ``ego_dropout`` is the model's own per-row v0 dropout (0.5 in the launch); 0.0 keeps every row for the control."""
    pat = LG._Patcher()
    try:
        if arm:
            LG.apply_p10_arm(_ctx(V_, arm), pat, T_)
        _a, cfg, model = _arm_v(T_, V_)
        model.core.cfg.ego_dropout = float(ego_dropout)
        argv = V_.argv_obs()
        ind, rec = LG.p10_new_record(argv)
        b = _batch(T_, cfg, fx_)

        def hook(m, i, o):
            if m.training and isinstance(o, dict):
                LG.p10_live_observe(rec, ind, m, o, b)
        model.register_forward_hook(hook)
        p10 = model.core.decoder.p10
        p10.register_forward_hook(lambda m, i, o: rec.__setitem__("p10_calls", rec["p10_calls"] + 1))
        model.train()
        torch.manual_seed(seed)
        loss = T_.compute_losses_v3(model, b, "cpu", mode="diffusion")
        loss["loss"].backward()
        rec["frozen"] = LG.p10_frozen_report(model)
        grads = {n: {"nz_steps": int(p.grad is not None and float(p.grad.abs().sum()) > 0.0)}
                 for n, p in model.named_parameters() if p.requires_grad}
        reasons, det = LG.judge_p10_live(V_.prof, argv, rec, list(grads), grads, fwd_train_calls=rec["forwards_seen"])
        return rec, reasons, det
    finally:
        pat.restore()


def test_G_LIVE_the_control_PASSES_the_emitted_prior_is_the_independent_P10_recompute(T, V, fxv):
    rec, reasons, det = _live(T, V, fxv)
    assert reasons == [], reasons
    assert rec["variant"] == V.variant and rec["frozen"]["variant"] == V.variant
    assert rec["forwards_checked"] == rec["forwards_seen"] >= 1
    assert rec["rows_checked"] >= 3 and rec["max_abs_diff_a"] <= 1e-4 and rec["max_abs_diff_k"] <= 1e-4
    assert rec["fallback_control_max_abs_diff"] > 1e-3          # the comparison CAN see a fallback to a constant prior
    assert rec["p10_calls"] >= 1
    fr = rec["frozen"]
    assert (fr["n_parameters"], fr["n_buffers_requires_grad"], fr["n_tensors_with_grad"], fr["n_in_state_dict"]) == (0, 0, 0, 0)
    assert fr["content_sha256"] == fr["content_sha_at_load"] == rec["content_sha_first"] \
        == rec["independent_load"]["content_sha256"]
    assert det["p10_parameters_in_optimizer"] == [] and det["p10_parameters_with_nonzero_grad"] == []


def test_G_LIVE_DELIBERATE_REGRESSION_each_arm_goes_RED_and_its_regex_matches_the_reason(T, V, fxv):
    ctl_rec, ctl_reasons, _ = _live(T, V, fxv)
    assert ctl_reasons == []
    arms = ["p10_fallback_ha0_ext", "p10_clock_shift", "p10_unfrozen", "p10_zero_weights"]
    if V.variant == "pose":
        arms.append("p10_pose_a0_stride2")
    for arm in arms:
        _rec, reasons, _det = _live(T, V, fxv, arm)
        assert reasons, f"{arm} ESCAPED G-LIVE"
        sig = ARMS.ARMS[arm]["sig"]
        joined = " || ".join(reasons)
        if arm in ("p10_zero_weights", "p10_clock_shift", "p10_pose_a0_stride2"):
            # arms GRADED on another check (G-DVB / G-CLOCK): G-LIVE must ALSO see a prior that is not the file's P10
            assert "INDEPENDENT P10 recompute" in joined or "content hash" in joined, (arm, reasons)
        else:
            assert re.search(sig, joined), (arm, sig, reasons)
            assert not re.search(sig, " || ".join(ctl_reasons))


def test_G_LIVE_withheld_rows_are_read_off_the_models_own_speed_and_are_NOT_false_alarms(T, V, fxv):
    """The launch trains with ego_dropout 0.5: about half the rows carry a ZERO prior by design. The observer must
    neither flag those nor miss the kept ones -- and a withheld row that carries a prior is a LEAK."""
    seen = []
    for seed in range(40):
        rec, reasons, _ = _live(T, V, fxv, seed=seed, ego_dropout=0.5)
        assert reasons == [] or rec["rows_checked"] == 0, (seed, reasons)
        if rec["rows_kept"] and rec["rows_withheld"]:
            seen.append(seed)
            assert reasons == [], reasons
            assert rec["rows_withheld_nonzero"] == 0 and rec["rows_checked"] == rec["rows_kept"]
        if len(seen) >= 2:
            break
    assert len(seen) >= 2, "no seed produced a mixed kept / withheld batch"
    # every row withheld: P10 was never read -> RED, with the reason that says so (not 'a mismatch')
    rec, reasons, _ = _live(T, V, fxv, ego_dropout=1.0)
    assert rec["rows_withheld"] == rec["rows"] and rec["rows_checked"] == 0
    assert any("ZERO rows were comparable" in r for r in reasons)


def test_G_LIVE_a_withheld_row_that_carries_a_prior_is_a_LEAK_and_goes_RED(T, V, fxv):
    real = KP.withhold
    pat = LG._Patcher()
    try:
        pat.set(KP, "withhold", lambda a0, k0, keep: (a0, k0))         # the zeroing is lost: the window leaks past dropout
        rec, reasons, _ = _live(T, V, fxv, seed=1, ego_dropout=1.0)
    finally:
        pat.restore()
    assert KP.withhold is real
    assert rec["rows_withheld_nonzero"] == rec["rows"] > 0
    assert any("leaked past the ego dropout" in r for r in reasons)


def test_G_LIVE_the_fallback_is_caught_TWICE_the_recompute_disagrees_and_P10_was_never_called(T, V, fxv):
    rec, reasons, _ = _live(T, V, fxv, "p10_fallback_ha0_ext")
    assert rec["p10_calls"] == 0
    assert any("CALLED 0 times" in r for r in reasons)
    assert any("INDEPENDENT P10 recompute" in r for r in reasons)
    assert rec["max_abs_diff_a"] > 1e-2


def test_G_LIVE_the_pose_observer_uses_NO_action_channel(T):
    """alpha-1 reads the pose track only: the gate's recompute must not need `actions`, so a batch without them is fine."""
    V_ = VARS["pose"]
    z = np.load(V_.fixture, allow_pickle=False)
    _a, cfg, model = _arm_v(T, V_)
    b = {k: v for k, v in _batch(T, cfg, {k: z[k] for k in z.files}).items() if k != "actions"}
    ind, rec = LG.p10_new_record(V_.argv_obs())
    with torch.no_grad():
        model.eval()
        model.core.set_ego_window(b["pose_hist"], int(b["pose_hist"].shape[1]), actions=None)
        o = model(T.frames_to_device(b["frames"], "cpu"), nav_cmd=b["nav_cmd"], v0=b["pose_last"][:, 3],
                  steps=cfg.core.decoder.diffusion_steps)
    LG.p10_live_observe(rec, ind, model, o, b)
    assert rec["problems"] == [] and rec["rows_checked"] == 4 and rec["max_abs_diff_a"] <= 1e-4


def test_G_LIVE_judge_refuses_an_empty_unreadable_or_wrong_variant_observation(V):
    r, _ = LG.judge_p10_live(V.prof, V.argv_obs(), None, [], {}, 5)
    assert r and "NOTHING" in r[0]
    ind, rec = LG.p10_new_record(["--residual-prior", V.mode, "--p10-prior", "/nonexistent/p.pt",
                                  "--p10-prior-sha256", V.sha])
    assert ind is None and not rec["independent_load"]["ok"]
    r, _ = LG.judge_p10_live(V.prof, V.argv_obs(), rec, [], {}, 5)
    assert any("independent read" in x for x in r)
    ind2, rec2 = LG.p10_new_record(["--residual-prior", V.mode, "--p10-prior", str(V.file),
                                    "--p10-prior-sha256", "0" * 64])
    assert ind2 is None and "hashes to" in rec2["independent_load"]["error"]          # a wrong-sha file is not read at all
    other = VARS["steer" if V.variant == "pose" else "pose"]
    ind3, rec3 = LG.p10_new_record(["--residual-prior", V.mode, "--p10-prior", str(other.file),
                                    "--p10-prior-sha256", other.sha])
    assert ind3 is None and "variant" in rec3["independent_load"]["error"]             # the other input channel's file


def test_G_LIVE_config_stamp_must_name_the_declared_P10(T, V, fxv):
    rec, _r, _d = _live(T, V, fxv)
    good = {"seams": {"residual_prior": {"residual_prior": V.mode, "p10": {
        "file_sha256": V.sha, "content_sha256": rec["independent_load"]["content_sha256"]}}}}
    grads = {}
    r, _ = LG.judge_p10_live(V.prof, V.argv_obs(), {**rec}, [], grads, rec["forwards_seen"], good)
    assert r == [], r
    bad = json.loads(json.dumps(good))
    bad["seams"]["residual_prior"]["p10"]["file_sha256"] = "0" * 64
    r, _ = LG.judge_p10_live(V.prof, V.argv_obs(), {**rec}, [], grads, rec["forwards_seen"], bad)
    assert any("config.json stamps P10 file sha256" in x for x in r)
    r, _ = LG.judge_p10_live(V.prof, V.argv_obs(), {**rec}, [], grads, rec["forwards_seen"], {"seams": {}})
    assert any("carries no `p10` identity" in x for x in r)
    # the built module's variant must be the mode's
    wrong = {**rec, "frozen": {**rec["frozen"], "variant": V.other_mode and ("steer" if V.variant == "pose" else "pose")}}
    r, _ = LG.judge_p10_live(V.prof, V.argv_obs(), wrong, [], grads, rec["forwards_seen"], good)
    assert any("variant" in x for x in r), r


# =========================================================================== #
# G-CKPT                                                                      #
# =========================================================================== #
def test_G_CKPT_ckpt_carries_the_P10_identity_and_DROPPING_it_goes_RED(T, V, fxv):
    rec, _r, _d = _live(T, V, fxv)
    _a, _c, model = _arm_v(T, V)
    ck = {"model": {}, "p10_prior": T._p10_ckpt_extra(model)["p10_prior"]}
    r, det = LG.judge_p10_ckpt(V.prof, V.argv_obs(), ck, rec)
    assert r == [], r
    assert det["ckpt_p10_prior"]["file_sha256"] == V.sha
    # the arm: the trainer saves without the identity
    pat = LG._Patcher()
    try:
        LG.apply_p10_arm(_ctx(V, "p10_ckpt_stamp_dropped"), pat, T)
        dropped = {"model": {}, **T._p10_ckpt_extra(model)}
    finally:
        pat.restore()
    r, _ = LG.judge_p10_ckpt(V.prof, V.argv_obs(), dropped, rec)
    assert r and re.search(ARMS.ARMS["p10_ckpt_stamp_dropped"]["sig"], r[0])
    # a different P10's identity, and no checkpoint at all
    r, _ = LG.judge_p10_ckpt(V.prof, V.argv_obs(), {"p10_prior": {"file_sha256": "0" * 64, "content_sha256": "1" * 64}}, rec)
    assert len(r) == 2
    assert LG.judge_p10_ckpt(V.prof, V.argv_obs(), None, rec)[0]


# =========================================================================== #
# G-EVAL                                                                      #
# =========================================================================== #
def test_G_EVAL_the_loader_must_build_the_SAME_P10_and_a_different_one_goes_RED(T, V):
    _a, _c, mt = _arm_v(T, V)
    _a2, _c2, ml = _arm_v(T, V)
    r, det = LG.judge_p10_eval(V.prof, V.argv_obs(), mt, ml)
    assert r == [], r
    assert det["trainer"]["content_sha256"] == det["loader"]["content_sha256"]
    assert det["trainer"]["variant"] == det["loader"]["variant"] == V.variant
    # the state_dict comparison is BLIND to it (non-persistent buffers) -- which is why this clause exists
    assert not [k for k in mt.state_dict() if ".p10." in k]
    pat = LG._Patcher()
    try:
        LG.apply_p10_arm(_ctx(V, "p10_loader_other_p10"), pat)
        _a3, _c3, other = _arm_v(T, V)
    finally:
        pat.restore()
    r, _ = LG.judge_p10_eval(V.prof, V.argv_obs(), mt, other)
    assert r and re.search(ARMS.ARMS["p10_loader_other_p10"]["sig"], " || ".join(r)), r
    # a loader whose P10 is the other INPUT CHANNEL's variant
    pat = LG._Patcher()
    try:
        LG.apply_p10_arm(_ctx(V, "p10_variant_flipped"), pat)
        _a4, _c4, flipped = _arm_v(T, V)
    finally:
        pat.restore()
    assert any("variant" in x for x in LG.judge_p10_eval(V.prof, V.argv_obs(), mt, flipped)[0])
    # a loader whose model lost the module
    ml.core.decoder.p10 = None
    assert any("has no P10" in x for x in LG.judge_p10_eval(V.prof, V.argv_obs(), mt, ml)[0])


def test_G_EVAL_the_eval_loader_remaps_each_P10_file_to_the_tree_under_test_by_basename(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from tanitad.eval import refcv7_loader as L\n"
            "try:\n"
            "    out, rec = L.remap_argv(['--p10-prior', sys.argv[2], '--p10-prior-sha256', 'ab'])\n"
            "    print(json.dumps({'out': out, 'ok': rec['remapped']}))\n"
            "except SystemExit as e:\n"
            "    print(json.dumps({'refused': str(e)}))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REFCV6_", "REFCV7_"))}
    env.update(REFCV6_KIT=str(kit), PYTHONIOENCODING="utf-8")

    def run(p):
        r = subprocess.run([sys.executable, "-c", code, str(ROOT), p], env=env, capture_output=True, text=True,
                           encoding="utf-8", timeout=300)
        assert r.returncode == 0, r.stderr[-1500:]
        return json.loads(r.stdout.strip().splitlines()[-1])
    for thor, sha, fname in ((THOR_P10_POSE, POSE_SHA256, "p10_pose_prior.pt"), (THOR_P10, FULL_SHA, "p10_prior.pt")):
        got = run(thor)
        local = got["out"][1]
        assert Path(local) == ROOT / "tanitad" / "configs" / "v8f_p10" / fname and Path(local).is_file()
        assert LG.sha256_file(local) == sha                     # the remapped file IS the declared artifact
        assert got["ok"]["--p10-prior"]["source"] == "kit default"
    bad = run(THOR_DIR + "p10_prior_OTHER.pt")
    assert "refused" in bad and "--p10-prior: the argv names 'p10_prior_OTHER.pt'" in bad["refused"]


# =========================================================================== #
# G-CLOCK                                                                     #
# =========================================================================== #
def test_G_CLOCK_the_in_decoder_P10_equals_the_banked_predictions_and_one_frame_early_FAILS(V):
    r, det = LG.p10_clock_check(V.prof, V.argv_obs(), [], REPO)
    assert r == [], r
    assert det["variant"] == V.variant
    assert det["n_windows"] >= 500 and det["in_decoder_vs_banked_max_m"] <= 1e-4
    assert det["one_frame_early_control"]["frac_beyond_tol"] > 0.95
    assert det["ade6_P10u_vs_P10_m"]["mean"] <= 0.05 and det["delta_zero_vs_P10u_max_m"] <= 1e-4
    assert det["disjoint"]["eval141_clip_ids"]["overlap"] == det["disjoint"]["eval139_clip_ids"]["overlap"] == 0
    assert det["disjoint"]["eval139_clip_ids"]["n_fit"] == V.n_fit


def test_G_CLOCK_DELIBERATE_REGRESSION_a_ONE_FRAME_SHIFT_inside_the_decoders_features_goes_RED(V):
    pat = LG._Patcher()
    try:
        LG.apply_p10_arm(_ctx(V, "p10_clock_shift"), pat)
        r, det = LG.p10_clock_check(V.prof, V.argv_obs(), [], REPO)
    finally:
        pat.restore()
    assert r, "the shifted clock ESCAPED G-CLOCK"
    assert re.search(ARMS.ARMS["p10_clock_shift"]["sig"], " || ".join(r)), r
    assert det["in_decoder_vs_banked_max_m"] > 1e-2
    # restored: the control reads clean again
    assert LG.p10_clock_check(V.prof, V.argv_obs(), [], REPO)[0] == []


def test_G_CLOCK_DELIBERATE_REGRESSION_the_pose_features_with_the_steer_variants_a0_convention_go_RED():
    V_ = VARS["pose"]
    pat = LG._Patcher()
    try:
        LG.apply_p10_arm(_ctx(V_, "p10_pose_a0_stride2"), pat)
        r, det = LG.p10_clock_check(V_.prof, V_.argv_obs(), [], REPO)
    finally:
        pat.restore()
    assert r and re.search(ARMS.ARMS["p10_pose_a0_stride2"]["sig"], " || ".join(r)), r
    assert det["in_decoder_vs_banked_max_m"] > 1e-4
    assert LG.p10_clock_check(V_.prof, V_.argv_obs(), [], REPO)[0] == []


def test_G_CLOCK_a_P10_of_the_other_variant_is_unreadable_to_the_clock_check(V):
    other = VARS["steer" if V.variant == "pose" else "pose"]
    argv = ["--residual-prior", V.mode, "--p10-prior", str(other.file), "--p10-prior-sha256", other.sha]
    r, _ = LG.p10_clock_check(V.prof, argv, [], REPO)
    assert r and "variant" in r[0], r


def test_G_CLOCK_overlapping_fit_and_eval_clips_are_REFUSED(V, tmp_path):
    z = dict(np.load(REPO / V.prof["p10_prior"]["fixture"], allow_pickle=False))
    z["eval139_clip_ids"] = np.concatenate([z["eval139_clip_ids"][:5], z["fit_clip_ids"][:1]])
    d = tmp_path / "stack" / "tanitad" / "configs" / "v8f_p10"
    d.mkdir(parents=True)
    np.savez_compressed(d / Path(V.prof["p10_prior"]["fixture"]).name, **z)
    r, _ = LG.p10_clock_check(V.prof, V.argv_obs(), [], tmp_path)
    assert any("overlap" in x for x in r), r


# =========================================================================== #
# the arms runner                                                             #
# =========================================================================== #
def test_the_ten_P10_arms_are_registered_for_the_v8f_profiles_and_each_names_its_check():
    names = ["p10_wrong_sha", "p10_missing_file", "p10_zero_weights", "p10_unfrozen", "p10_fallback_ha0_ext",
             "p10_clock_shift", "p10_ckpt_stamp_dropped", "p10_loader_other_p10", "p10_pose_a0_stride2",
             "p10_variant_flipped"]
    assert [k for k in ARMS.ARMS if k.startswith("p10_")] == names
    checks = {n: ARMS.ARMS[n]["checks"] for n in names}
    assert checks == {"p10_wrong_sha": ["G-DVB"], "p10_missing_file": ["G-DVB"], "p10_zero_weights": ["G-DVB"],
                      "p10_unfrozen": ["G-DVB", "G-LIVE"], "p10_fallback_ha0_ext": ["G-LIVE"],
                      "p10_clock_shift": ["G-CLOCK"], "p10_ckpt_stamp_dropped": ["G-CKPT"],
                      "p10_loader_other_p10": ["G-EVAL"], "p10_pose_a0_stride2": ["G-CLOCK"],
                      "p10_variant_flipped": ["G-DVB"]}
    assert all(ARMS.ARMS[n]["profile"] == "v8f" and ARMS.ARMS[n]["requires_flag"].startswith("--p10-prior")
               for n in names)
    assert ARMS.ARMS["p10_pose_a0_stride2"]["only_residual_prior"] == "p10_pose"      # meaningless for the steer variant
    assert ARMS.ARMS["p10_wrong_sha"]["argv_set"] == ("--p10-prior-sha256", ["0" * 64])
    # every arm the apply function knows is one the runner lists, and an unknown name applies nothing
    pat = LG._Patcher()
    try:
        assert LG.apply_p10_arm(type("C", (), {"arm": "loader_skips_pin"})(), pat) is None
        assert LG.apply_p10_arm(type("C", (), {"arm": "p10_wrong_sha"})(), pat) is None     # an ARGV arm: nothing to patch
    finally:
        pat.restore()


def test_the_arms_runner_applies_the_p10_arms_to_the_steer_profile_too_and_skips_the_ones_that_mean_nothing_there():
    src = (ROOT / "scripts" / "launch_gate_arms.py").read_text(encoding="utf-8")
    assert 'if prof == "v8f" and a.profile == "v8f_steer":' in src and "only_residual_prior" in src


def test_the_arm_patches_are_RESTORED_so_one_arm_cannot_leak_into_the_next(T, V):
    def snap():
        return (P10.load_frozen_p10, P10.kin3_from_window, P10.FrozenP10.__init__, P10.FrozenP10.forward,
                KP.p10_tick_prior, KP.prior_controls, KP.pose_kin3)
    before = snap()
    arms = ["p10_zero_weights", "p10_unfrozen", "p10_clock_shift", "p10_loader_other_p10", "p10_fallback_ha0_ext",
            "p10_variant_flipped"] + (["p10_pose_a0_stride2"] if V.variant == "pose" else [])
    for arm in arms:
        pat = LG._Patcher()
        try:
            assert LG.apply_p10_arm(_ctx(V, arm), pat, T)
            assert any(a is not b for a, b in zip(before, snap())), arm       # the arm DID patch something
        finally:
            pat.restore()
        assert all(a is b for a, b in zip(before, snap())), arm
