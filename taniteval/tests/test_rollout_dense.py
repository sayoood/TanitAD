"""``rollout.collect`` must PERSIST the dense 10 Hz path, not discard 16/20 steps.

THE RESIDUAL THIS CLOSES (open since 2026-07-09)
------------------------------------------------
``rollout_decode`` computes the full ``[b, 20, 2]`` path; ``rollout.collect``
kept only the 4 waypoints at ``WP_STEPS`` and threw the rest away at what
``driving.py``'s docstring called "rollout.py:94". Everything needing 10 Hz
derivatives — jerk, the adopted comfort bounds, a real curvature *profile*,
decel-onset lead time (``tanitad.eval.metrics.compute_lal_v2``), plan stability
— was therefore blocked on that ONE line rather than on any new science
(TANITEVAL_V2_METRIC_SUITE §7 E2). The whole behavioural / comfort axis cost
~1 MB per arm and could not be computed at all.

WHAT IS PINNED
  * the dense keys exist and carry every step;
  * the SPARSE keys are UNCHANGED — ``pred``/``gt`` are exactly the dense path
    sampled at WP_STEPS, so no existing consumer's number can move;
  * ``gt_dense`` uses the SAME ego-frame convention as the trusted sparse
    ``gt`` (the real risk in densifying a geometry helper);
  * ``dense_speed_profile``'s origin-prepend convention, which every derived
    jerk / decel-onset index depends on;
  * the storage delta, MEASURED here rather than estimated;
  * backward compatibility: pre-2026-07-25 dumps have no dense keys and must
    still load and still score.

CPU-only, no GPU, no pod, no checkpoint: ``rollout_decode`` is stubbed so the
test exercises THIS module's persistence + geometry, not the predictor.
Standalone: ``python taniteval/tests/test_rollout_dense.py``.
"""
import sys
import types
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # taniteval/
sys.path.insert(0, "/root/taniteval")              # pod layout

from taniteval import rollout as R  # noqa: E402

T_FRAMES = 60
N_EPISODES = 3
WINDOW, STRIDE = 8, 8
WP_IDX = [k - 1 for k in R.WP_STEPS]                # [4, 9, 14, 19]

_CACHE = {}


def _episode(eid, seed):
    """A synthetic episode on a smooth curved trajectory (real geometry)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T_FRAMES, dtype=torch.float32)
    yaw = 0.02 * t + 0.1 * torch.sin(0.15 * t)      # a real, turning heading
    v = 8.0 + 2.0 * torch.sin(0.09 * t)             # a real, varying speed
    dx, dy = v * R.DT * yaw.cos(), v * R.DT * yaw.sin()
    poses = torch.stack([dx.cumsum(0), dy.cumsum(0), yaw, v], dim=1)  # [T,4]
    return types.SimpleNamespace(
        episode_id=eid, poses=poses,
        actions=torch.randn(T_FRAMES, 2, generator=g) * 0.01,
        feats=torch.randn(T_FRAMES, 4, generator=g))


def _fake_rollout_decode(predictor, states, aw, fa, step_readout, k):
    """Deterministic, per-window-distinct [b,k,2] — enough to prove the sparse
    view is a strict sub-view of what is now persisted."""
    b = states.shape[0]
    step = torch.arange(1, k + 1, dtype=torch.float32)[None, :]
    base = torch.arange(b, dtype=torch.float32)[:, None]
    return torch.stack([base * 3.0 + step * 0.7,
                        base * 0.1 + (0.3 * step).sin()], dim=-1), None


class _FakeModel:
    predictor = None

    def encode_window(self, fw):
        return fw


def _win():
    if "win" not in _CACHE:
        real = R.rollout_decode
        R.rollout_decode = _fake_rollout_decode
        try:
            _CACHE["win"] = R.collect(
                _FakeModel(), None,
                [_episode(i, 100 + i) for i in range(N_EPISODES)],
                device="cpu", window=WINDOW, stride=STRIDE, batch=2)
        finally:
            R.rollout_decode = real
    return _CACHE["win"]


# --------------------------------------------------------------------------- #
# 1. the dense path is persisted                                              #
# --------------------------------------------------------------------------- #
def test_dense_keys_exist_with_every_step():
    w = _win()
    n = w["pred"].shape[0]
    assert n > 0
    for k in ("pred_dense", "gt_dense"):
        assert k in w, k
        assert w[k].shape == (n, R.K_MAX, 2), (k, w[k].shape)
    assert w["dense_steps"] == list(range(1, R.K_MAX + 1))
    assert w["dt_s"] == R.DT
    # 20 of 20 steps kept, not 4
    assert len(w["dense_steps"]) == R.K_MAX == 20
    assert len(w["wp_steps"]) == 4


def test_sparse_view_is_unchanged_and_is_a_strict_subview_of_the_dense():
    """The backward-compatibility contract: adding the dense field must not
    change what ``pred``/``gt`` mean, or every published ADE moves."""
    w = _win()
    assert torch.equal(w["pred"], w["pred_dense"][:, WP_IDX])
    assert torch.equal(w["gt"], w["gt_dense"][:, WP_IDX])


def test_gt_dense_uses_the_same_ego_frame_convention_as_the_trusted_sparse_gt():
    """``gt_dense`` calls ``gt_ego_waypoints`` with 20 steps instead of 4. If
    that densification silently changed frame, origin or sign, the sparse
    sub-view would stop matching the independently-computed sparse ``gt``."""
    from driving_diagnostic import gt_ego_waypoints
    ep = _episode(0, 100)
    last = torch.tensor([WINDOW - 1, WINDOW - 1 + STRIDE])
    sparse = gt_ego_waypoints(ep.poses, last)
    dense = gt_ego_waypoints(ep.poses, last,
                             wp_steps=tuple(range(1, R.K_MAX + 1)))
    assert dense.shape == (2, R.K_MAX, 2)
    assert torch.equal(sparse, dense[:, WP_IDX])
    # the path must actually advance (a degenerate all-zero dense path would
    # satisfy the equality above only if the sparse one were degenerate too)
    assert float(dense[:, -1].norm(dim=-1).min()) > 1.0


# --------------------------------------------------------------------------- #
# 2. the speed-profile convention the behavioural metrics consume             #
# --------------------------------------------------------------------------- #
def test_dense_speed_profile_prepends_the_origin():
    """A constant-velocity straight path must give a CONSTANT speed at EVERY
    step including the first. Differencing without prepending the ego origin
    drops step 0 and shifts every derived decel-onset index by one sample."""
    v, dt, k = 7.5, R.DT, R.K_MAX
    steps = torch.arange(1, k + 1, dtype=torch.float32)
    path = torch.stack([v * dt * steps, torch.zeros(k)], dim=-1)[None]  # [1,k,2]
    spd = R.dense_speed_profile(path, dt)
    assert spd.shape == (1, k)
    assert torch.allclose(spd, torch.full((1, k), v), atol=1e-4), spd


def test_dense_speed_profile_on_the_real_collected_gt():
    """Sanity on real geometry: the synthetic episodes cruise at 8 ± 2 m/s, so
    the GT dense profile must land in that band — proving the persisted dense
    GT is metric and directly usable as ``compute_lal_v2``'s ``ego_v``."""
    spd = R.dense_speed_profile(_win()["gt_dense"])
    assert spd.shape == (_win()["gt"].shape[0], R.K_MAX)
    assert 5.0 < float(spd.mean()) < 11.0, float(spd.mean())
    assert float(spd.min()) > 0.0


def test_dense_path_actually_unblocks_a_jerk_style_derivative():
    """The axis this fix exists for: 4 samples 0.5 s apart cannot give a 10 Hz
    jerk; 20 samples can. Pin that the derivative chain is now computable and
    finite end-to-end."""
    spd = R.dense_speed_profile(_win()["gt_dense"])
    accel = spd.diff(dim=1) / R.DT
    jerk = accel.diff(dim=1) / R.DT
    assert jerk.shape[1] == R.K_MAX - 2 == 18       # vs 2 from the sparse view
    assert bool(torch.isfinite(jerk).all())


# --------------------------------------------------------------------------- #
# 3. what it costs, and what still works without it                           #
# --------------------------------------------------------------------------- #
def test_storage_delta_is_measured_and_within_budget(tmp_path):
    """MEASURED, not estimated: two [N,20,2] float32 tensors. The axis was
    costed at ~1 MB/arm; at the real 881-window scale this is +275 KB."""
    w = _win()
    n = w["pred"].shape[0]
    dense_only = tmp_path / "with.pt"
    sparse_only = tmp_path / "without.pt"
    R.save_windows(w, dense_only)
    R.save_windows({k: v for k, v in w.items()
                    if k not in ("pred_dense", "gt_dense", "dense_steps", "dt_s")},
                   sparse_only)
    delta = dense_only.stat().st_size - sparse_only.stat().st_size
    per_window = 2 * R.K_MAX * 2 * 4                # 2 tensors x 20 x 2 x fp32
    assert delta >= n * per_window * 0.95, (delta, n * per_window)
    assert delta <= n * per_window * 1.10 + 4096, (delta, n * per_window)
    # extrapolated to the real 881-window dump, still far under the 1 MB budget
    assert 881 * per_window < 1_000_000


def test_pre_dense_dumps_still_load_and_still_score():
    """Dumps written before 2026-07-25 (and every refb_eval / refc_eval dump)
    carry NO dense keys. They must load, must not raise, and the driving panel
    must report their dense surface as absent rather than pretend."""
    pin = _HERE.parents[1] / "results" / "windows_flagship-30k.pt"
    if not pin.exists():
        print("  (committed dump absent — backward-compat check UNVERIFIED)")
        return
    old = R.load_windows(pin)
    assert old.get("pred_dense") is None
    assert old["pred"].shape[1] == 4
    from taniteval import driving as D
    block = D.tier0(old, n_boot=50, arm="flagship-30k")
    assert block["dense_surface_available"] is False
    assert block["headline"]["ade_0_2s"]["estimator"] == "episode_cluster_bootstrap"


def test_new_dumps_advertise_the_dense_surface():
    from taniteval import driving as D
    block = D.tier0(_win(), n_boot=50, arm="synthetic")
    assert block["dense_surface_available"] is True
    assert "IS persisted" in block["surface"]


if __name__ == "__main__":
    import tempfile
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"PASS {name}")
        except Exception as e:                                    # noqa: BLE001
            bad += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"==== {len(fns) - bad}/{len(fns)} passed ====")
    sys.exit(1 if bad else 0)
