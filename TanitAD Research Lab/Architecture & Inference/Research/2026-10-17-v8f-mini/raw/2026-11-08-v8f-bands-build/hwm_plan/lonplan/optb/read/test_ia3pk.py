"""Tests of the IA-3P-K reader (synthetic worlds, CPU) with deliberate-regression arms that must go RED (the planted gain is NOT recovered by a broken reader).
Run: python -m pytest test_ia3pk.py -q --basetemp <dir under read/tmp>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest
import torch
import ia3pk_read as R

D, K, NCL, NPER = 12, 10, 70, 40


def synth(b_scale=1.0, seed=0, noise=0.6):
    """clips with actions that vary smoothly; real transition z_{t+K} - z_t = b * abar + noise (b = the real per-unit-action response)."""
    rng = np.random.RandomState(seed)
    b = rng.randn(D); b /= np.linalg.norm(b)
    EP, FR, ZT, ZK, AS = [], [], [], [], []
    for c in range(NCL):
        for f in range(NPER):
            a = (rng.randn() * 0.8 + 0.3 * rng.randn(K)).astype(np.float32)
            aseq = np.stack([a, rng.randn(K).astype(np.float32) * 0.2], 1)
            z0 = rng.randn(D)
            zk = z0 + b_scale * b * aseq[:, 0].mean() * K * 0.5 + noise * rng.randn(D)
            EP.append(c); FR.append(f); ZT.append(z0); ZK.append(zk); AS.append(aseq)
    P = {"ep": np.array(EP), "fr": np.array(FR), "zt": torch.tensor(np.array(ZT), dtype=torch.float32), "zk": torch.tensor(np.array(ZK), dtype=torch.float32),
         "aseq": torch.tensor(np.array(AS)), "clips": [f"clip_{c:03d}" for c in range(NCL)]}
    P["abar"] = P["aseq"].mean(1).numpy().astype(np.float64)
    P["part"] = R.pick_partners(P["ep"], P["fr"])
    return P, torch.tensor(b, dtype=torch.float32)


def world(b, g):
    """a world whose response to a mean acceleration command is g times the real one."""
    return lambda z0, a: z0 + g * b * a[:, :, 0].mean(1, keepdim=True) * K * 0.5


W = R.boot_weights(NCL, n_boot=400)


@pytest.mark.parametrize("g", [1.0, 0.5, 2.0])
def test_planted_gain_is_recovered(g):
    P, b = synth()
    res, _ = R.read_core(P, world(b, g), W=W)
    G = res["ia3pk_gain_L1_2s_lon"]
    assert abs(G["point"] - g) < 0.15 * g + 0.05, G
    assert G["ci95"][0] < G["point"] < G["ci95"][1]
    assert res["c10p_s_swap_ci_above_0"]


def test_zero_response_world_reads_zero():
    P, b = synth()
    res, _ = R.read_core(P, lambda z0, a: z0 + 0.0, W=W)                # the world ignores the action: the shuffled / no-response regression arm
    G = res["ia3pk_gain_L1_2s_lon"]
    assert abs(G["point"]) < 1e-9 and G["ci95"][0] <= 0 <= G["ci95"][1] + 1e-12


def test_wrong_channel_world_reads_zero():
    P, b = synth()
    res, _ = R.read_core(P, lambda z0, a: z0 + b * a[:, :, 1].mean(1, keepdim=True), W=W)   # responds to yaw only
    assert abs(res["ia3pk_gain_L1_2s_lon"]["point"]) < 1e-6


def test_c10p_goes_red_when_actions_do_not_move_the_transition():
    P, b = synth(b_scale=0.0)                                            # real transitions carry no action signal
    res, _ = R.read_core(P, world(b, 1.0), W=W)
    assert res["s_swap"]["ci95"][0] <= 0, res["s_swap"]
    assert res["c10p_s_swap_ci_above_0"] is False


def test_reader_is_the_calibration_not_a_constant():
    """doubling the real transition response halves nothing in G: G stays ~1 when the world follows the real response (a fixed constant s would not)."""
    for bs in (0.5, 2.0):
        P, b = synth(b_scale=bs)
        res, _ = R.read_core(P, world(b, bs), W=W)
        assert abs(res["ia3pk_gain_L1_2s_lon"]["point"] - 1.0) < 0.2, (bs, res["ia3pk_gain_L1_2s_lon"])


def test_shuffled_calibration_breaks_the_recovery():
    """deliberate regression: shuffle the real actions across anchors -> s_swap ~ 0 (CI includes 0), so the calibrated gain is no longer trustworthy and C10' reads FAIL."""
    P, b = synth()
    rng = np.random.RandomState(1)
    perm = rng.permutation(len(P["ep"]))
    P2 = dict(P); P2["abar"] = P["abar"][perm]
    res, _ = R.read_core(P2, world(b, 1.0), W=W)
    assert res["c10p_s_swap_ci_above_0"] is False


def test_folds_are_ia3s():
    assert [R.fold_of(c) for c in ("a", "b", "clip_000")] == [int(__import__("hashlib").sha256(c.encode()).hexdigest()[:8], 16) % 2 for c in ("a", "b", "clip_000")]
    lr = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/ladder_read"
    if os.path.isdir(lr):                                                # compare with the code of record by source text (lr_lib imports the full tree, not loaded here)
        src = open(os.path.join(lr, "lr_lib.py"), encoding="utf-8").read()
        assert "int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2" in src


def test_boot_weights_match_imag1b_draw():
    imag = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/imag1b/imag1b_lib.py"
    if not os.path.exists(imag):
        pytest.skip("imag1b_lib not present")
    src = open(imag, encoding="utf-8").read()
    assert "rng = np.random.default_rng(seed)" in src and "pick = rng.choice(uniq, size=n, replace=True)" in src
    w = R.boot_weights(5, n_boot=3)
    assert w.shape == (3, 5) and (w.sum(1) == 5).all()


def mk(tag_pts):
    return {t: {"ia3pk_gain_L1_2s_lon": {"point": p, "ci95": [p - 0.05, p + 0.05]}} for t, p in tag_pts.items()}


def good():
    return {"FTNULL_s0": 1.0, "FTNULL_s2": 1.02, "FTNULL_s4": 0.99, "FTRSP_s0": 1.3, "FTRSP_s2": 1.32,
            "FTNULL_s1": 0.9, "FTNULL_s3": 0.92, "FTNULL_s5": 0.91, "FTRSP_s1": 1.2, "FTRSP_s3": 1.22}


def test_mech_moved_pass():
    assert R.mech_moved(mk(good()))["label"] == "MECH-MOVED"


def test_mech_moved_goes_red():
    g = good(); g["FTRSP_s3"] = 0.93                                     # gap below the floor / CIs overlap at one lever seed
    assert R.mech_moved(mk(g))["label"] == "NOT-MECH-MOVED"
    g = good(); g["FTRSP_s0"] = 1.0                                      # gain no higher than its pair
    assert R.mech_moved(mk(g))["label"] == "NOT-MECH-MOVED"
    g = good(); g["FTRSP_s2"] = 1.1                                      # CIs disjoint (1.05 > 1.07? no: lo 1.05 vs hi 1.07) -> overlap
    assert R.mech_moved(mk(g))["label"] == "NOT-MECH-MOVED"


def test_mech_moved_floor_binds():
    g = good(); g["FTNULL_s4"] = 0.5                                     # a wide training floor (0.52) swallows a 0.3 gap
    assert R.mech_moved(mk(g))["label"] == "NOT-MECH-MOVED"


def test_mech_moved_missing_arm_is_not_computable():
    g = good(); del g["FTNULL_s5"]
    assert R.mech_moved(mk(g))["label"] == "NOT-COMPUTABLE"
