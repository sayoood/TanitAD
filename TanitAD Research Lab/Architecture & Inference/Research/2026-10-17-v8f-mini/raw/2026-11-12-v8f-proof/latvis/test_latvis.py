"""Fast synthetic CPU tests for latvis.py. Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider test_latvis.py"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import latvis as LV  # noqa: E402

SMALL = (16, 16, 16, 8, 8)


def test_decoder_shapes_and_range():
    for kind, x in (("Z", torch.randn(3, 128)), ("G", torch.randn(3, 128, 4, 10))):
        m = LV.make_decoder(kind, SMALL)
        m.set_norm(x)
        o = m(x)
        assert tuple(o.shape) == (3, 3, 64, 160)
        assert bool(torch.isfinite(o).all()) and float(o.detach().min()) >= 0.0 and float(o.detach().max()) <= 1.0


def test_decoder_full_width_param_count_runs():
    m = LV.make_decoder("G")
    assert tuple(m(torch.randn(2, 128, 4, 10)).shape) == (2, 3, 64, 160)


def test_pool4x10_index_map():
    tok = np.zeros((1, 640, 128), np.float32)
    r, c = np.divmod(np.arange(640), 40)               # token = r * 40 + c, 16 rows x 40 cols
    tok[0, :, 0] = (r // 4) * 10 + (c // 4)            # constant inside each 4x4 block: the block id
    tok[0, :, 1] = r                                    # varies along rows
    tok[0, :, 2] = c                                    # varies along columns
    g = LV.pool4x10(tok)
    assert g.shape == (1, 128, 4, 10)
    exp = (np.arange(4)[:, None] * 10 + np.arange(10)[None, :]).astype(np.float32)
    assert np.allclose(g[0, 0], exp)
    assert np.allclose(g[0, 1], (4 * np.arange(4) + 1.5)[:, None] * np.ones((1, 10)))
    assert np.allclose(g[0, 2], np.ones((4, 1)) * (4 * np.arange(10) + 1.5)[None, :])


def test_crossfit_never_decodes_an_episode_with_a_decoder_trained_on_it():
    ep = np.repeat(np.arange(20), 7)
    fold = LV.RP.make_folds(ep, k=5, seed=0)
    assert all(len(set(ep[fold == f]) & set(ep[fold != f])) == 0 for f in range(5))     # the unit is the episode
    seen = []

    def fit(tr):
        return {"train_eps": set(ep[tr])}

    def apply(model, te):
        seen.append((set(ep[te]), model["train_eps"]))
        return None

    out = LV.crossfit(fold, fit, apply)
    assert len(out) == 5 and sum(len(te) for te, _ in out.values()) == len(ep)
    assert all(not (te_eps & tr_eps) for te_eps, tr_eps in seen)


def _synthetic(n_ep=24, per=5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_ep * per
    ep = np.repeat(np.arange(n_ep), per)
    X = np.zeros((n, 128), np.float32)
    X[:, :2] = rng.normal(size=(n, 2))                  # two informative dimensions, the others constant
    yy, xx = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 160), indexing="ij")
    P1, P2 = np.sin(3 * xx) * yy, np.cos(2 * yy) * xx
    a, b = np.tanh(X[:, 0]), np.tanh(X[:, 1])
    img = 0.5 + 0.35 * (a[:, None, None] * P1[None] + b[:, None, None] * P2[None])
    IMG = np.repeat(np.round(img * 255).clip(0, 255).astype(np.uint8)[:, None], 3, axis=1)
    return X, IMG, ep


def _skill(perm):
    X, IMG, ep = _synthetic()
    Xt, It = torch.from_numpy(X), torch.from_numpy(IMG)
    fold = LV.RP.make_folds(ep, k=3, seed=0)
    tr, te = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    model, info = LV.fit_decoder("Z", Xt, It, tr, ep[tr], seed=0, device=torch.device("cpu"), max_epochs=50, bs=16, patience=100, lr=3e-3, widths=SMALL, perm=perm)
    te_t = torch.as_tensor(te)
    sse = LV.mse_in(model, Xt[te_t], It, te_t)
    mean_img = It[torch.as_tensor(tr)].float().mean(0) / 255.0
    sse0 = ((It[te_t].float() / 255.0 - mean_img) ** 2).mean(dim=(1, 2, 3)).double().numpy()
    return LV.skill_pooled(sse, sse0)


def test_skill_formula_planted_arithmetic():
    sse0 = np.array([1.0, 2.0, 3.0])
    assert LV.skill_pooled(sse0, sse0) == pytest.approx(0.0)
    assert LV.skill_pooled(0.5 * sse0, sse0) == pytest.approx(0.5)
    assert LV.skill_pooled(0 * sse0, sse0) == pytest.approx(1.0)


def test_skill_planted_decoder_input_determines_image_vs_permuted():
    s_true, s_perm = _skill(False), _skill(True)
    assert s_true > 0.5, s_true
    assert s_perm < 0.1, s_perm


def test_panel_rule_one_per_episode_and_ordering():
    clip = np.array(["a", "a", "b", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    g0 = np.arange(12)
    v0 = np.zeros(12)
    v10 = np.array([9, 1, 8, 2, 7, 6, 5, 4, 3, 0.5, 0.1, np.nan])
    gap = np.array([30, 5, 20, 4, 3, 40, 2, 1, 50, 60, 70, 80.0])
    ch = LV.select_panel(clip, g0, v0, v10, gap)
    A = [c for c in ch if c[1] == "speed_change"]
    B = [c for c in ch if c[1] == "lead_gap"]
    assert len(A) == 6 and len(B) == 6
    assert [c[0] for c in A] == [0, 2, 4, 5, 6, 7]                    # |dv| descending, one per episode (windows 1 and 3 share an episode with 0 and 2)
    assert [c[2] for c in A] == sorted([c[2] for c in A], reverse=True)
    assert len({clip[c[0]] for c in A}) == 6 and len({clip[c[0]] for c in B}) == 6
    assert [c[2] for c in B] == sorted(c[2] for c in B)               # gap ascending
    assert not ({c[0] for c in A} & {c[0] for c in B})                # no window twice
    assert [c[0] for c in B] == [3, 1, 8, 9, 10, 11]                  # windows 7, 6, 4 (smallest gaps) are already in A, so they are skipped


def test_bootstrap_excludes_zero_for_a_planted_positive_difference():
    n_ep = 30
    C = LV.RP.boot_counts(n_ep, B=500, seed=0)
    s = np.full(n_ep, 1.0)
    lo, hi = LV.boot_ci(lambda C_: (C_ @ s) / (C_ @ np.ones(n_ep)), C)
    assert lo > 0 and hi > 0
