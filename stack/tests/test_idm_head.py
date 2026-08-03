"""CPU smoke for the supervised non-causal IDM head (scripts/idm_head.py).

Guards the pipeline that produces the IDM/YouTube go/no-go number: window
construction (target alignment), forward shapes, a finite/differentiable loss,
and that a tiny fit runs and yields finite metrics. Fast (CPU, synthetic).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import idm_head as ih  # noqa: E402


def test_build_windows_target_alignment():
    # A hand-built episode: speed/steer/accel are exact functions of the index so
    # the derived targets at the window CENTER must match the source rows exactly.
    T, D = 60, 16
    z = torch.randn(T, D)
    yaw = 0.01 * torch.arange(T).float()                # constant yaw-rate 0.1 rad/s
    v = 5.0 + 0.0 * torch.arange(T).float()
    x = torch.cumsum(v * torch.cos(yaw) * ih.DT, 0)
    y = torch.cumsum(v * torch.sin(yaw) * ih.DT, 0)
    poses = torch.stack([x, y, yaw, v], 1).float()
    actions = torch.stack([0.02 * torch.arange(T).float(),
                           0.5 * torch.ones(T)], 1).float()
    Z, scal, traj = ih.build_windows(z, poses, actions, k=4, stride=1)
    assert Z.shape[1] == 9 and Z.shape[2] == D
    assert scal.shape[1] == 4 and traj.shape[1:] == (4, 2)
    # first valid center is t=4 (needs 4 past frames); check its targets
    t0 = 4
    assert torch.allclose(scal[0, 0], poses[t0, 3])                 # speed
    assert torch.allclose(scal[0, 2], actions[t0, 0])              # steer
    assert torch.allclose(scal[0, 3], actions[t0, 1])              # accel
    # centered yaw-rate ~ 0.1 rad/s (constant-rate construction)
    assert abs(float(scal[0, 1]) - 0.1) < 1e-3
    # trajectory forward at 2 s ~ v*2 = 10 m ahead (ego +x), lateral small
    assert traj[0, -1, 0] > 8.0


def test_forward_shapes_and_differentiable():
    # the REAL (default) head on the flagship state_dim is "a few M" params
    assert 1e6 < ih.count_params(ih.IDMHead(state_dim=2048)) < 1e7
    head = ih.IDMHead(state_dim=32, d_model=64, depth=2, n_heads=4)
    z = torch.randn(7, 9, 32)
    out = head(z)
    assert out["scalars"].shape == (7, 4) and out["traj"].shape == (7, 4, 2)
    scal = torch.randn(7, 4)
    traj = torch.randn(7, 4, 2)
    std = ih.Standardizer.fit(scal)
    ld = ih.idm_loss(out, scal, traj, std)
    assert torch.isfinite(ld["loss"])
    ld["loss"].backward()
    g = sum(float(p.grad.norm()) for p in head.parameters() if p.grad is not None)
    assert g > 0 and g == g                                        # finite, nonzero


def test_train_head_runs_and_metrics_finite():
    D = 32
    def cat(lst):
        return (torch.cat([a[0] for a in lst]), torch.cat([a[1] for a in lst]),
                torch.cat([a[2] for a in lst]))
    tr = [ih.build_windows(*ih._synthetic_episode(80, D, s), k=4) for s in range(6)]
    va = [ih.build_windows(*ih._synthetic_episode(80, D, s), k=4) for s in (6, 7)]
    res = ih.train_head(cat(tr), {"val": cat(va)}, state_dim=D, epochs=2,
                        batch=64, log=lambda *_: None)
    m = res["val"]["val"]
    assert m["n"] > 0
    for name in ih.SCALAR_NAMES:
        assert m["r2"][name] == m["r2"][name]                     # not NaN
    assert m["ade_2s"] == m["ade_2s"] and m["ade_2s"] >= 0.0


# --------------------------------------------------------------------------- #
# the speed-sequence readout and the DERIVED long_accel                       #
#                                                                             #
# WHY: on the banked held-out read the head's `long_accel` R2 is NEGATIVE on   #
# every seed and both domains (-0.15 .. -0.42, idm5_ensemble.json) while its   #
# `speed` R2 is 0.86 — and the CAN accel channel is recoverable from the TRUE  #
# speed track by a centred difference at R2 0.902 (MEASURED, comma val cache,  #
# 8,940 windows). The head was regressing a quantity it could have derived.    #
# --------------------------------------------------------------------------- #
def test_speed_seq_targets_are_the_window_speeds():
    T, D, k = 40, 8, 4
    poses = torch.zeros(T, 4)
    poses[:, 3] = torch.arange(T).float()          # speed == index, so alignment shows
    t = torch.tensor([10, 20])
    q = ih.speed_seq_targets_at(poses, t, k)
    assert q.shape == (2, 2 * k + 1)
    assert torch.allclose(q[0], torch.arange(6, 15).float())
    assert torch.allclose(q[1], torch.arange(16, 25).float())


def test_derive_long_accel_is_the_centred_difference():
    seq = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
    # centre is index 4; (seq[5] - seq[3]) / (2*DT) = (3-1)/0.2 = 10
    assert float(ih.derive_long_accel(seq, center=4)) == 10.0


def test_derived_accel_recovers_the_finite_difference_of_true_speed():
    """The identity the fix rests on: fed the TRUE window speeds, the derived
    channel reproduces the centred-difference acceleration exactly."""
    T, k = 60, 4
    poses = torch.zeros(T, 4)
    poses[:, 3] = 12.0 + 3.0 * torch.sin(0.2 * torch.arange(T).float())
    t = torch.arange(k, T - k)
    q = ih.speed_seq_targets_at(poses, t, k)
    got = ih.derive_long_accel(q, center=k)
    want = (poses[t + 1, 3] - poses[t - 1, 3]) / (2 * ih.DT)
    assert torch.allclose(got, want, atol=1e-5)


def test_derive_long_accel_rejects_a_window_with_no_neighbour():
    with pytest.raises(ValueError):
        ih.derive_long_accel(torch.zeros(2, 3), center=2)


def test_speed_seq_head_shapes_and_deployed_swap():
    head = ih.IDMHead(state_dim=16, d_model=32, depth=1, n_heads=2, speed_seq=True)
    out = head(torch.randn(5, 9, 16))
    assert out["speed_seq"].shape == (5, 9)
    assert out["long_accel"].shape == (5,)
    dep = head.deployed_scalars(out)
    j = ih.SCALAR_NAMES.index("long_accel")
    assert dep.shape == (5, 4)
    assert torch.allclose(dep[:, j], out["long_accel"])          # the swap happened
    keep = [c for c in range(4) if c != j]
    assert torch.allclose(dep[:, keep], out["scalars"][:, keep])  # nothing else moved


def test_legacy_head_is_untouched_by_the_option():
    """Every banked checkpoint predates the readout; the default path must be
    bit-identical to before."""
    head = ih.IDMHead(state_dim=16, d_model=32, depth=1, n_heads=2)
    out = head(torch.randn(3, 9, 16))
    assert "speed_seq" not in out and "long_accel" not in out
    assert head.speed_seq_head is None
    assert torch.equal(head.deployed_scalars(out), out["scalars"])


def test_loss_drops_the_direct_accel_column_when_the_sequence_supervises_it():
    """Keeping both would let the head satisfy long_accel the broken way, and the
    arm would no longer isolate the fix."""
    head = ih.IDMHead(state_dim=16, d_model=32, depth=1, n_heads=2, speed_seq=True)
    z = torch.randn(6, 9, 16)
    scal = torch.randn(6, 4)
    traj = torch.randn(6, 4, 2)
    q = torch.randn(6, 9)
    std = ih.Standardizer.fit(scal)
    out = head(z)
    base = ih.idm_loss(out, scal, traj, std, speed_seq=q)
    moved = scal.clone()
    moved[:, ih.SCALAR_NAMES.index("long_accel")] += 5.0
    other = ih.idm_loss(out, moved, traj, std, speed_seq=q)
    assert float(base["scalar_loss"]) == float(other["scalar_loss"])
    assert "speed_seq_loss" in base
    # ... while a LEGACY loss on the same tensors DOES move
    legacy = ih.IDMHead(state_dim=16, d_model=32, depth=1, n_heads=2)
    lo = legacy(z)
    assert float(ih.idm_loss(lo, scal, traj, std)["scalar_loss"]) != \
        float(ih.idm_loss(lo, moved, traj, std)["scalar_loss"])


def test_train_head_demands_the_sequence_target_when_the_option_is_on():
    z = torch.randn(4, 9, 8)
    with pytest.raises(ValueError, match="fourth element"):
        ih.train_head((z, torch.randn(4, 4), torch.randn(4, 4, 2)), {},
                      state_dim=8, speed_seq=True, epochs=1)


def test_fit_head_returns_the_module_and_train_head_stays_json_safe():
    """`train_head`'s result is written straight to disk by run_idm_proof.py:387,
    so it must never carry an nn.Module."""
    import json

    parts = [ih.build_windows(*ih._synthetic_episode(60, 8, s), k=4)
             for s in range(3)]
    tr = tuple(torch.cat([p[i] for p in parts]) for i in range(3))
    head, std, meta = ih.fit_head(tr, state_dim=8, epochs=1, log=lambda *_: None)
    assert isinstance(head, ih.IDMHead) and isinstance(std, ih.Standardizer)
    res = ih.train_head(tr, {"v": tr}, state_dim=8, epochs=1, log=lambda *_: None)
    json.dumps(res)                                   # must not raise
    assert meta["params"] == res["params"]


def test_deriving_accel_beats_regressing_it_on_a_corpus_where_a_equals_dv_dt():
    """END-TO-END CONTRACT, on a corpus built to the identity the real one
    MEASURES (R2 0.902): the latent carries the per-frame speed and long_accel IS
    the centred difference of speed. The derived channel must beat the
    independent readout -- and by MORE when the latent is noisier, which is the
    regime the real head is actually in (its direct long_accel R2 is NEGATIVE).
    """
    k, D = 4, 12

    def episode(seed, noise):
        gg = torch.Generator().manual_seed(seed)
        T = 110
        t = torch.arange(T).float()
        p1, p2 = torch.rand(2, generator=gg) * 6.28
        v = 12.0 + 4.0 * torch.sin(0.15 * t + p1)
        yaw = 0.40 * torch.sin(0.08 * t + p2)
        x = torch.cumsum(v * torch.cos(yaw) * ih.DT, 0)
        y = torch.cumsum(v * torch.sin(yaw) * ih.DT, 0)
        poses = torch.stack([x, y, yaw, v], 1)
        a = torch.zeros(T)
        a[1:-1] = (v[2:] - v[:-2]) / (2 * ih.DT)          # the identity
        yr = torch.zeros(T)
        yr[1:-1] = (yaw[2:] - yaw[:-2]) / (2 * ih.DT)
        actions = torch.stack([2.0 * yr, a], 1)
        z = torch.randn(T, D, generator=gg) * noise
        z[:, 0] = (v - 12.0) / 4.0
        z[:, 1] = yaw / 0.4
        return z, poses, actions

    def build(seeds, noise):
        Z, S, Tj, Q = [], [], [], []
        for s in seeds:
            z, poses, actions = episode(s, noise)
            zz, sc, tj = ih.build_windows(z, poses, actions, k=k, stride=1)
            t = ih.valid_centers(z.shape[0], k, ih.DEFAULT_HORIZONS, 1)
            Z.append(zz); S.append(sc); Tj.append(tj)
            Q.append(ih.speed_seq_targets_at(poses, t, k))
        return tuple(torch.cat(x) for x in (Z, S, Tj, Q))

    common = dict(state_dim=D, epochs=30, batch=128, seed=0, log=lambda *_: None,
                  head_kw=dict(d_model=64, depth=2, n_heads=4))
    tr, va = build(range(40), 0.30), build(range(100, 108), 0.30)
    base = ih.train_head(tr[:3], {"v": va[:3]}, **common)["val"]["v"]
    fix = ih.train_head(tr, {"v": va}, speed_seq=True, **common)["val"]["v"]
    assert fix["r2"]["long_accel"] > base["r2"]["long_accel"] + 0.2, (
        base["r2"], fix["r2"])
    assert fix["r2"]["long_accel"] > 0.5, fix["r2"]
    # the DERIVED channel is what carries it: the untouched direct readout of the
    # SAME head sits at chance, so this is not just "more capacity helped".
    assert abs(fix["r2_direct"]["long_accel"]) < 0.2, fix["r2_direct"]
    # and the other channels are not sacrificed for it
    assert fix["r2"]["yaw_rate"] > 0.9 and fix["r2"]["steer"] > 0.9, fix["r2"]


def test_derived_accel_needs_its_own_supervision_not_just_a_speed_sequence():
    """REGRESSION for a real design error caught by measurement. Differencing two
    window positions multiplies the speed sequence's error by 1/(2*dt) = 5x, so a
    head trained ONLY on absolute speeds derives a WORSE accel than a direct
    readout. The loss must therefore also supervise the DERIVED channel on its own
    target -- this asserts that term exists and is what it claims."""
    head = ih.IDMHead(state_dim=8, d_model=32, depth=1, n_heads=2, speed_seq=True)
    z = torch.randn(6, 9, 8)
    scal = torch.randn(6, 4)
    traj = torch.randn(6, 4, 2)
    q = torch.randn(6, 9)
    std = ih.Standardizer.fit(torch.randn(64, 4))
    ld = ih.idm_loss(head(z), scal, traj, std, speed_seq=q)
    assert "derived_accel_loss" in ld and "speed_seq_loss" in ld
    # moving ONLY the long_accel target must move the loss -- via the derived term
    moved = scal.clone()
    moved[:, ih.SCALAR_NAMES.index("long_accel")] += 5.0
    ld2 = ih.idm_loss(head(z), moved, traj, std, speed_seq=q)
    assert float(ld2["derived_accel_loss"]) != float(ld["derived_accel_loss"])
    assert float(ld2["scalar_loss"]) == float(ld["scalar_loss"])
