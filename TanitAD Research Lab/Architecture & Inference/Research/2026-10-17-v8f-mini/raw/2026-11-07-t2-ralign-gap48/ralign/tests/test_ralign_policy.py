"""AlignPolicy gates on CPU with the REAL C0 checkpoints behind the stub encoder: G-IDMAP (identity map -> the wrapped composite is the unwrapped composite, bit for bit), G-P10 (P10 under the wrapper
is P10), the pass-through of ego and nav (only `feats` may differ), and the mutants that must be caught. Synthetic frames and tokens only.
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest ralign/tests/test_ralign_policy.py -q --basetemp F:/TanitAD-scratch/ralign_pytest
Every `check_*(mut=None)` is one arm; `mut=<name>` makes the SAME assertion run against a deliberately broken wrapper so the arm can be shown RED (the tests call both).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import ralign_common as K  # noqa: E402
import ralign_lib as L  # noqa: E402
import ralign_policy as RP  # noqa: E402

SEEDS = [s for s in (0, 1) if os.path.isfile(K.CKPT[s])]


def wrapped(seed, tok_map, **kw):
    return RP.AlignPolicy(K.build_model_policy(seed), tok_map, scene_id="sc", scene_fold="A", **kw)


# ============================================================================================================================ lib: the signed zero, the map arithmetic
def test_identity_pair_is_bit_exact_on_signed_zero_and_fp8_range():
    a, b = L.identity_pair((2, 8))
    assert bool(torch.signbit(b).all()) and bool((b == 0).all()) and bool((a == 1).all())
    x = torch.tensor([[-0.0, 0.0, 1.5, -448.0, 448.0, 2.0 ** -9, -2.0 ** -9, 0.25]] * 2).to(torch.float8_e4m3fn)
    m = L.TokenMap(*L.identity_pair((2, 8)))
    y = m.apply(x)
    assert y.view(torch.int16).tolist() == x.to(torch.float16).view(torch.int16).tolist(), "identity map changed a token bit (signed zero?)"
    plus = L.TokenMap(a, torch.zeros_like(b))                       # b = +0.0 would turn a stored -0.0 into +0.0: why the pair stores -0.0
    assert plus.apply(x).view(torch.int16).tolist() != x.to(torch.float16).view(torch.int16).tolist()


# ============================================================================================================================ G-IDMAP
def check_idmap(seed, mut=None):
    """The wrapped composite with a := 1, b := 0 reproduces the unwrapped composite bit for bit over 3 ticks (plan, every `extra` key, and the model batch)."""
    ref = K.run_scenario(K.build_model_policy(seed))
    m = K.identity_map()
    if mut == "perturbed_b":
        m.b = m.b + 1e-3                                           # a perturbed identity: must FAIL the gate
    if mut == "perturbed_a":
        m.a = m.a * (1.0 + 1e-3)
    got = K.run_scenario(wrapped(seed, m))
    assert K.bit_equal_runs(ref, got), "G-IDMAP: the wrapped composite differs from the unwrapped one"
    for r, g in zip(ref, got):
        assert all(torch.equal(r[2][k], g[2][k]) for k in r[2]), "the model batch differs"
    assert got[-1][1]["ralign"]["n_token_sets_mapped"] > 0, "the map never ran: an identity that proves nothing"


@pytest.mark.parametrize("seed", SEEDS)
def test_g_idmap_passes_on_the_real_checkpoint(seed):
    check_idmap(seed)


@pytest.mark.parametrize("mut", ["perturbed_a", "perturbed_b"])
def test_g_idmap_red_on_a_perturbed_identity(mut):
    with pytest.raises(AssertionError, match="G-IDMAP|batch"):
        check_idmap(SEEDS[0], mut)


# ============================================================================================================================ G-P10
def check_p10(seed, mut=None):
    """P10 (arm 'p10': the prior of the same forward, never the image) under a strongly non-identity map equals P10 exactly; the model arm's prior (`p10_prior`) does too. The model arm's plan
    must DIFFER under the map (otherwise the check could not see a change)."""
    m = K.rand_map(3, 0.5)
    ref = K.run_scenario(K.build_model_policy(seed, arm="p10"))
    got = K.run_scenario(RP.AlignPolicy(K.build_model_policy(seed, arm="p10"), m))
    assert all(a[0].tobytes() == b[0].tobytes() and a[1]["p10_prior"] == b[1]["p10_prior"] for a, b in zip(ref, got)), "G-P10: P10 under the wrapper differs from P10"
    moved = {k for a, b in zip(ref, got) for k in K.strip(a[1]) if a[1][k] != b[1][k]}
    assert moved <= {"s_ckpt", "tau_o"}, f"G-P10: unexpected extra keys moved under the map: {sorted(moved)}"          # the model heads of the SAME forward read the tokens; P10's plan is the prior, which does not
    mref = K.run_scenario(K.build_model_policy(seed, composite=False))
    mgot = K.run_scenario(RP.AlignPolicy(K.build_model_policy(seed, composite=False), m))
    assert all(a[1]["p10_prior"] == b[1]["p10_prior"] for a, b in zip(mref, mgot)), "the P10 prior of the model arm moved under the map"
    if mut == "map_not_applied":
        mgot = mref
    assert any(a[0].tobytes() != b[0].tobytes() for a, b in zip(mref, mgot)), "the model plan did not move under a strong map: the check is blind"


@pytest.mark.parametrize("seed", SEEDS)
def test_g_p10_passes_on_the_real_checkpoint(seed):
    check_p10(seed)


def test_g_p10_check_is_red_when_the_map_is_not_applied():
    with pytest.raises(AssertionError, match="blind"):
        check_p10(SEEDS[0], "map_not_applied")


# ============================================================================================================================ pass-through: only the scene tokens change
def check_pass_through(factory, seed=0):
    """Every key of the model batch except `feats` is bit-identical between the unwrapped policy and the wrapper under a non-identity map; `feats` equals the map applied to the unwrapped
    policy's rows (so the change is exactly a * r + b, nothing else)."""
    ref = K.run_scenario(K.build_model_policy(seed))
    got = K.run_scenario(factory(seed))
    for r, g in zip(ref, got):
        for k in r[2]:
            if k == "feats":
                continue
            assert torch.equal(r[2][k], g[2][k]), f"PASS_THROUGH: batch key {k!r} differs (ego / nav / limit must not be touched)"
        want = torch.stack([m_apply(r[2]["feats"][0, i]) for i in range(r[2]["feats"].shape[1])])[None]
        ok = r[2]["frame_ok"][0]
        assert torch.equal(want[0][ok], g[2]["feats"][0][ok]), "PASS_THROUGH: feats is not a * r + b of the unwrapped rows"
        assert not torch.equal(r[2]["feats"][0][ok], g[2]["feats"][0][ok]), "the map changed no row"
    return True


M_STRONG = K.rand_map(5, 0.4)


def m_apply(row):
    return M_STRONG.apply(row)


def test_pass_through_holds_for_the_wrapper():
    assert check_pass_through(lambda s: RP.AlignPolicy(K.build_model_policy(s), M_STRONG), SEEDS[0])


class ScalesEgo(RP.AlignPolicy):
    """MUTANT: also scales the ego speed column of the policy's own track (the wrapper must not touch ego). The track is put back after the plan so the run itself stays valid."""

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        b = self._base
        orig = b.track
        b.track = orig.copy()
        b.track[:, 3] = b.track[:, 3] * 1.01
        try:
            return super().plan(frames, intr, b.track[-1, 3], nav_cmd, nav_known)
        finally:
            b.track = orig


class ScalesNav(RP.AlignPolicy):
    """MUTANT: also scales the nav row (the countdown channel) by 0.9 for the duration of the plan."""

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        b = self._base
        orig = b.nav0
        b.nav0 = np.array([1, 0, 0, 0.9, 1], np.float32)
        try:
            return super().plan(frames, intr, v0, nav_cmd, nav_known)
        finally:
            b.nav0 = orig


class DropsMap(RP.AlignPolicy):
    """MUTANT: the proxy is installed and then put back (the map silently not applied)."""

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        self._base.encoder = self._orig_encoder
        return self.inner.plan(frames, intr, v0, nav_cmd, nav_known)


@pytest.mark.parametrize("cls", [ScalesEgo, ScalesNav])
def test_mutant_that_touches_ego_or_nav_is_caught(cls):
    with pytest.raises(AssertionError, match="PASS_THROUGH"):
        check_pass_through(lambda s: cls(K.build_model_policy(s), M_STRONG), SEEDS[0])


def test_mutant_that_drops_the_map_is_caught_by_the_runtime_guard_and_by_the_check():
    w = wrapped(SEEDS[0], M_STRONG)                              # something else puts the original encoder back after the wrap: the wrapper's guard refuses to plan
    K.base_of(w).encoder = w._orig_encoder
    with pytest.raises(RuntimeError, match="RALIGN_ENCODER_REPLACED"):
        K.run_scenario(w)
    with pytest.raises(AssertionError, match="map changed no row|PASS_THROUGH"):           # a wrapper WITHOUT the guard that silently drops the map is caught by the batch comparison
        check_pass_through(lambda s: DropsMap(K.build_model_policy(s), M_STRONG), SEEDS[0])


# ============================================================================================================================ refusals and wiring
def test_wrapper_refuses_a_policy_without_an_encoder_and_a_double_wrap():
    import t2a_v8f_policy as P
    bare = P.V8FMiniT2Policy(K.CKPT[SEEDS[0]], encoder=None, arm="model", device="cpu")
    with pytest.raises(ValueError, match="RALIGN_NO_ENCODER"):
        RP.AlignPolicy(bare, K.identity_map())
    w = wrapped(SEEDS[0], K.identity_map())
    with pytest.raises(ValueError, match="RALIGN_ALREADY_WRAPPED"):
        RP.AlignPolicy(w, K.identity_map())
    assert w.unwrap() is not None and not isinstance(K.base_of(w.inner).encoder, RP.MappedEncoder)


def test_encoder_attributes_pass_through_the_proxy():
    w = wrapped(SEEDS[0], K.identity_map())
    assert w.encoder.canon == {"mode": "stub"} and w.hist == 4 and w.window == w.inner.window


def make_R():
    import types

    class R:                                                     # the two attributes install_align touches on t2v8f_run
        @staticmethod
        def build_arm(args):
            return types.SimpleNamespace(inner=K.build_model_policy(0))
    return R


def test_install_align_picks_the_other_folds_map_and_refuses_a_wrong_table(tmp_path):
    import types
    ids = K.scene_ids(1, 1)
    folds = {s: L.fold_of(s) for s in ids}
    paths = {}
    for f, m in (("A", K.rand_map(1)), ("B", K.rand_map(2))):
        paths[f] = L.save_map(str(tmp_path / f"MAP_{f}.pt"), m.a, m.b, {"fold": f, "fold_scenes": [], "n_pairs": 0})
    block = {"fold_digit": 0, "maps": {f: {"file": f"MAP_{f}.pt", **paths[f]} for f in "AB"}, "scene_fold": folds}
    R = make_R()
    RP.install_align(R, block, str(tmp_path), ("X_s0",))
    for sid in ids:
        pol = R.build_arm(types.SimpleNamespace(arm="X_s0", t2_scene_id=sid))
        assert isinstance(pol.inner, RP.AlignPolicy) and pol.inner.tok_map.fold == L.other_fold(folds[sid]) and pol.inner.scene_fold == folds[sid]
    assert not isinstance(R.build_arm(types.SimpleNamespace(arm="P10", t2_scene_id=ids[0])).inner, RP.AlignPolicy)
    with pytest.raises(SystemExit, match="RALIGN_NO_SCENE_ID"):
        R.build_arm(types.SimpleNamespace(arm="X_s0", t2_scene_id=None))
    R2 = make_R()
    RP.install_align(R2, dict(block, scene_fold={ids[0]: L.other_fold(folds[ids[0]]), ids[1]: folds[ids[1]]}), str(tmp_path), ("X_s0",))
    with pytest.raises(SystemExit, match="RALIGN_FOLD_MISMATCH"):
        R2.build_arm(types.SimpleNamespace(arm="X_s0", t2_scene_id=ids[0]))
    bad = dict(paths["A"], tensor_sha256="0" * 64)
    with pytest.raises(ValueError, match="RALIGN_MAP_TENSOR_MISMATCH"):
        RP.install_align(make_R(), dict(block, maps={"A": {"file": "MAP_A.pt", **bad}, "B": block["maps"]["B"]}), str(tmp_path), ("Y_s0",))
