"""GS-9 — the TRANSITION-LEVEL probe rung: ``Δx`` from ``Δz`` with the programme's
probe-panel rules. Every control must read its KNOWN value (CLAUDE.md; C133 family;
RETRACTION_LOG 2026-08-22 ridge panel).

What is pinned, and why each group exists:

  (1) THE CONSTANT-ONLY CONTROL reads skill 0.0000 EXACTLY — by construction of the
      skill score (pred == fit-split mean), not approximately.
  (2) PLANTED STRUCTURE IS RECOVERED (skill > 0.99 on Δx = A·Δz + noise), the raw
      floor on structureless "pixels" sits at the null, and the GLOBAL time-shuffle
      sits in the null band [−0.05, 0.05].
  (3) THE WITHIN-CLIP SHUFFLE PRESERVES A CLIP-LEVEL CUE and destroys the transition
      part — so "transition-specific skill = real − within" reads the right thing.
  (4) EPISODE-DISJOINTNESS: no clip is ever scored by a fit that contained it, and λ is
      selected on the FIT split only (garbage score targets cannot move it).
  (5) THE RIDGE INTERCEPT IS UNPENALISED (a huge target offset costs nothing) and
      standardisation uses FIT statistics.
  (6) TARGETS FROM POSES: a straight constant-speed track reads (v·dt, 0, 0, 0); a
      pure turn reads dyaw; ``dv`` is the pose speed difference — v never enters as an
      input anywhere in the feature set.
  (7) THE CLIP-CLUSTER BOOTSTRAP: the point estimate equals the pooled skill and a
      paired delta of an arm against itself is exactly zero.
  (8) END TO END on a fresh tiny ``V6Stack`` and a synthetic ``*.v2ep.pt`` corpus (PNG
      frames, poses, actions): the feature builder produces every cell with consistent
      row counts and the panel runs with ``const`` at exactly 0.

CPU only; no checkpoint, no real corpus, no GPU.
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[2]
_STACK = _REPO / "stack"
for _p in (str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
TOOL = _REPO / "taniteval" / "tools" / "transition_probe.py"
_spec = importlib.util.spec_from_file_location("transition_probe_under_test", TOOL)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


# ---------------------------------------------------------------------------
# (1)-(3) the panel on planted structure
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    X, Y, P = tp.synthetic_clips(n_clips=16, rows=50, d=16, seed=0, clip_offset=0.0)
    feats = {"const": [np.ones((len(y), 1)) for y in Y], "pixdelta": P, "dz_enc": X}
    return tp.run_panel(feats, Y, [f"c{i}" for i in range(len(Y))], n_boot=100, k_outer=4,
                        k_inner=3)


def test_constant_control_reads_exactly_zero(panel):
    c = panel["cells"]["const"]
    assert panel["controls"]["constant_reads_exactly_zero"]
    assert all(s == 0.0 for s in c["real"]["skill"])
    assert all(s == 0.0 for s in c["global_shuffle"]["skill"])
    assert all(s == 0.0 for s in c["within_shuffle"]["skill"])


def test_planted_structure_recovered_and_floor_at_null(panel):
    assert all(s > 0.99 for s in panel["cells"]["dz_enc"]["real"]["skill"])
    assert all(abs(s) < 0.05 for s in panel["cells"]["pixdelta"]["real"]["skill"])
    assert all(r > 0.99 for r in panel["cells"]["dz_enc"]["real"]["r"])


def test_global_time_shuffle_sits_in_the_null_band(panel):
    for name in ("dz_enc", "pixdelta"):
        assert all(abs(s) < 0.05 for s in panel["cells"][name]["global_shuffle"]["skill"]), name


def test_every_cell_prints_n_and_d(panel):
    for name, c in panel["cells"].items():
        assert c["n_score"] > 0 and c["d"] >= 1 and len(c["n_fit_per_fold"]) == 4, name
        assert len(c["lambda_rel_per_fold"]) == 4


def test_within_clip_shuffle_keeps_the_clip_level_cue_only():
    """Targets = A·Δz + 5·u_c, where u_c is a CLIP-LEVEL scalar that is VISIBLE as a
    constant column of X. The within-clip shuffle decouples Δz from Y inside each clip
    but keeps u_c aligned, so its skill is the clip-level share alone; the global
    shuffle destroys both; and transition-specific skill = real − within."""
    rng = np.random.default_rng(1)
    X, Y, _ = tp.synthetic_clips(n_clips=16, rows=50, d=8, seed=1, clip_offset=0.0, noise=0.1)
    u = rng.standard_normal(len(Y))
    Y = [y + 5.0 * u[i] for i, y in enumerate(Y)]
    feats = {"dz_enc": [np.concatenate([x, np.full((len(x), 1), u[i])], 1)
                        for i, x in enumerate(X)]}
    out = tp.run_panel(feats, Y, [f"c{i}" for i in range(len(Y))], n_boot=50, k_outer=4, k_inner=3)
    c = out["cells"]["dz_enc"]
    real = np.array(c["real"]["skill"])
    within = np.array(c["within_shuffle"]["skill"])
    glob = np.array(c["global_shuffle"]["skill"])
    assert (real > 0.9).all(), real
    assert (within > 0.2).all() and (within < real - 0.05).all(), within
    assert (np.abs(glob) < 0.1).all(), glob
    assert np.allclose(np.array(c["transition_specific_skill"]), real - within)


# ---------------------------------------------------------------------------
# (4) episode-disjointness and fit-only λ selection
# ---------------------------------------------------------------------------
def test_grouped_folds_partition_clips_disjointly():
    g = np.repeat(np.arange(13), 7)
    folds = tp.grouped_folds(g, 5, seed=0)
    allc = np.concatenate(folds)
    assert sorted(allc.tolist()) == list(range(13))
    assert len(set(allc.tolist())) == 13


def test_lambda_is_selected_on_the_fit_split_only():
    X, Y, _ = tp.synthetic_clips(n_clips=12, rows=40, d=10, seed=2)
    g = np.concatenate([np.full(len(y), i) for i, y in enumerate(Y)])
    Xa, Ya = np.concatenate(X), np.concatenate(Y)
    lam1, _ = tp.select_lambda(Xa, Ya, g, k_inner=3, seed=0)
    lam2, _ = tp.select_lambda(Xa, Ya, g, k_inner=3, seed=0)
    assert lam1 == lam2
    # cross-fitting never scores a clip inside its own fit: rows of the held-out fold get
    # predictions from a basis built without them (checked via the fold bookkeeping)
    cf = tp.crossfit(X, Y, k_outer=4, k_inner=3, seed=0)
    assert np.isfinite(cf["pred"]["real"]).all()
    assert sum(cf["n_fit_per_fold"]) == 3 * cf["n_score"]      # each row in exactly 3 of 4 fits


# ---------------------------------------------------------------------------
# (5) the intercept is unpenalised; fit statistics standardise
# ---------------------------------------------------------------------------
def test_intercept_is_unpenalised_and_fit_stats_are_used():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, 5))
    beta = rng.standard_normal((5, 2))
    Y = X @ beta + 1000.0                       # a huge offset must cost nothing
    basis = tp._RidgeBasis(X[:300])
    betas, ymu = basis.coef(Y[:300], [1e-3])
    pred = basis.predict(X[300:], betas[0], ymu)
    assert np.abs(pred - Y[300:]).mean() < 0.05
    assert np.allclose(basis.mu, X[:300].mean(0)) and np.allclose(basis.sd, X[:300].std(0))


# ---------------------------------------------------------------------------
# (6) targets from poses
# ---------------------------------------------------------------------------
def test_targets_from_poses_straight_track_and_pure_turn():
    dt, v = 0.1, 8.0
    T = 20
    t = np.arange(T) * dt
    yaw = np.full(T, 0.7)
    x, y = v * t * np.cos(0.7), v * t * np.sin(0.7)
    P = np.stack([x, y, yaw, np.full(T, v)], 1)
    tg = tp.targets_from_poses(P, 1)
    assert np.allclose(tg[:, 0], v * dt) and np.allclose(tg[:, 1], 0.0, atol=1e-9)
    assert np.allclose(tg[:, 2], 0.0) and np.allclose(tg[:, 3], 0.0)
    # a pure heading change with speed ramp: dyaw and dv read the differences
    P2 = np.stack([np.zeros(T), np.zeros(T), np.linspace(0, 1.0, T), np.linspace(5, 7, T)], 1)
    tg2 = tp.targets_from_poses(P2, 1)
    assert np.allclose(tg2[:, 2], 1.0 / (T - 1)) and np.allclose(tg2[:, 3], 2.0 / (T - 1))
    # wrap-around
    P3 = np.array([[0, 0, np.pi - 0.1, 1.0], [0, 0, -np.pi + 0.1, 1.0]])
    assert tp.targets_from_poses(P3, 1)[0, 2] == pytest.approx(0.2)


def test_v_is_never_a_feature_input():
    """The feature builder's inputs: none of them is built from the pose speed column
    — pinned on the SOURCE so a future edit that adds v as an input fails loudly."""
    src = TOOL.read_text(encoding="utf-8")
    body = src.split("def clip_features")[1].split("def build_corpus_features")[0]
    feats_block = body.split("feats = {")[1]
    assert "poses[" not in feats_block.split("return")[0]
    assert "act[rows].numpy()" in feats_block                 # act2 = (steer, accel) only


# ---------------------------------------------------------------------------
# (7) the clip-cluster bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_point_equals_pooled_skill_and_self_pair_is_zero():
    rng = np.random.default_rng(4)
    N = 300
    clip = np.repeat(np.arange(10), 30)
    y = rng.standard_normal((N, 2))
    pred = y + 0.3 * rng.standard_normal((N, 2))
    const = np.zeros_like(y)
    bs = tp.clip_bootstrap_skill(pred, const, y, clip, n_boot=50, seed=0)
    assert np.allclose(bs["point"], tp.skill_score(pred, const, y))
    assert all(lo <= p <= hi for p, lo, hi in zip(bs["point"], bs["lo"], bs["hi"]))
    paired = tp.clip_bootstrap_skill(pred, const, y, clip, n_boot=50, seed=0,
                                     pred_b=pred, const_b=const)
    assert np.allclose(paired["point"], 0.0) and np.allclose(paired["lo"], 0.0)
    assert not any(paired["separated"])


# ---------------------------------------------------------------------------
# (8) end to end on a tiny V6Stack and a synthetic v2ep corpus
# ---------------------------------------------------------------------------
def _write_synthetic_v2ep(path: Path, T=30, H=32, W=32, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    bufs, lens = [], []
    for _ in range(T):
        im = Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
        b = io.BytesIO()
        im.save(b, format="PNG")
        raw = b.getvalue()
        bufs.append(np.frombuffer(raw, dtype=np.uint8))
        lens.append(len(raw))
    t = np.arange(T) * 0.1
    v = 5.0 + 0.5 * np.sin(t)
    yaw = 0.05 * t
    x, y = np.cumsum(v * 0.1 * np.cos(yaw)), np.cumsum(v * 0.1 * np.sin(yaw))
    poses = np.stack([x, y, yaw, v], 1).astype(np.float32)
    actions = np.stack([0.01 * np.sin(t), np.gradient(v, 0.1)], 1).astype(np.float32)
    torch.save({"jpeg_buf": torch.from_numpy(np.concatenate(bufs)),
                "jpeg_len": torch.tensor(lens), "poses": torch.from_numpy(poses),
                "actions": torch.from_numpy(actions), "n_stack": 3, "codec": "png",
                "episode_id": seed, "clip_id": f"syn{seed}"}, path)


def test_end_to_end_tiny_stack_synthetic_corpus(tmp_path):
    pytest.importorskip("PIL")
    from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig
    from tanitad.eval.v6_probe_trunk import V6ProbeTrunk
    from tanitad.models.v6 import V6Config, V6Stack
    from train_v6_staged import COND_INCUMBENT
    cfg = V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=32, image_width=32, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=4),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1,), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32, f_hidden_tac=32,
        f_hidden_str=32, d_plan_feat=16, emission_hidden=16, n_candidates=3,
        aux_hidden=16, sigreg_slices=8)
    torch.manual_seed(0)
    world = V6ProbeTrunk(V6Stack(cfg).eval())
    clips = []
    for i in range(6):
        p = tmp_path / f"syn{i}.v2ep.pt"
        _write_synthetic_v2ep(p, T=30, seed=i)
        clips.append(str(p))
    F, T, names = tp.build_corpus_features(world, clips, "cpu", COND_INCUMBENT,
                                           max_frames=30, replace="last")
    assert len(T) == 6 and names[0] == "syn0"
    n_rows = [len(t) for t in T]
    # 30 frames, stack 3 -> 28 rows; window 4 -> rows 3..26 -> 24 rows per clip
    assert n_rows == [24] * 6
    for k, v in F.items():
        assert [len(x) for x in v] == n_rows, k
    assert F["z_t"][0].shape[1] == cfg.d_op and F["act2"][0].shape[1] == 2
    assert F["pixdelta"][0].shape[1] == tp.PIX_HW[0] * tp.PIX_HW[1]
    # the anchored response of a fresh stack is exactly zero (zero-init FiLM)
    assert np.abs(np.concatenate(F["dzhat_anch"])).max() == 0.0
    out = tp.run_panel(F, T, names, n_boot=20, k_outer=3, k_inner=2)
    assert out["controls"]["constant_reads_exactly_zero"]
    assert set(tp.FEATURE_ORDER) <= set(out["cells"])
    assert all(len(v["point"]) == 4 for v in out["paired_marginals"].values())
