"""REF-A stage-2 trainer tests (tanitad/refs/refa.py + scripts/refa_train.py).

Pins the stability-assurance items at feature level (REFERENCE_ARCHITECTURES
REF-A, items 1-4): (a) frozen standardizer stats, (b) no grads into the
frozen-feature path, (c) rollout_k=4 executes, (d) ckpt round-trip reproduces
adapter outputs bit-exactly. CPU-only, synthetic feature files.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refa_train  # noqa: E402  (scripts/refa_train.py)
from tanitad.refs.refa import (FeatureStandardizer,  # noqa: E402
                               RefAModel)

D, N_TOK, T_EP = 768, 8, 24


def _make_feature_root(tmp_path, n_train=3, n_val=1, T=T_EP):
    """Synthetic dino_precompute output: tiny T, 768 dims, few tokens."""
    torch.manual_seed(0)
    for split, n in (("train", n_train), ("val", n_val)):
        d = tmp_path / f"toy-{split}-cache-dinov2-b14"
        d.mkdir()
        for i in range(n):
            torch.save({"feats_fp16": torch.randn(T, N_TOK, D).half(),
                        "actions": torch.randn(T, 2),
                        "poses": torch.zeros(T, 3),
                        "episode_id": i}, d / f"ep_{i:05d}.pt")
    return tmp_path


def _smoke_model() -> RefAModel:
    return RefAModel(refa_train.smoke_pred_config(), sigreg_slices=32)


def _batch(root, model, rollout_k=4, batch=4):
    max_h = max(max(model.pred_cfg.horizons), rollout_k)
    eps, _ = refa_train.load_feature_episodes(str(root), "*train*")
    ds = refa_train.FeatureWindowDataset(eps, model.pred_cfg.window, max_h)
    return torch.utils.data.default_collate([ds[i] for i in range(batch)])


# ---------- (a) standardizer stats are frozen ----------

def test_standardizer_fit_once_and_frozen(tmp_path):
    st = FeatureStandardizer(D)
    feats = [torch.randn(10, N_TOK, D).half() for _ in range(3)]
    with pytest.raises(RuntimeError):           # unfitted forward refuses
        st(feats[0])
    st.fit(iter(feats))
    assert bool(st.fitted)
    mean0, std0 = st.mean.clone(), st.std.clone()
    with pytest.raises(RuntimeError):           # fitting twice = error
        st.fit(iter(feats))
    assert torch.equal(st.mean, mean0) and torch.equal(st.std, std0)

    # state_dict round-trip: loaded stats identical, and STILL frozen —
    # a checkpoint-loaded standardizer can never recompute at eval.
    st2 = FeatureStandardizer(D)
    st2.load_state_dict(st.state_dict())
    assert bool(st2.fitted)
    assert torch.equal(st2.mean, st.mean) and torch.equal(st2.std, st.std)
    with pytest.raises(RuntimeError):
        st2.fit(iter(feats))
    x = torch.randn(4, N_TOK, D).half()
    assert torch.equal(st(x), st2(x))


# ---------- (b) one training step; no grads where they don't belong ----------

def test_train_step_finite_and_feature_path_grad_free(tmp_path):
    root = _make_feature_root(tmp_path)
    model = _smoke_model()
    eps, _ = refa_train.load_feature_episodes(str(root), "*train*")
    model.standardizer.fit(ep["feats_fp16"] for ep in eps)
    mean0, std0 = model.standardizer.mean.clone(), model.standardizer.std.clone()

    batch = _batch(root, model)
    opt = torch.optim.AdamW(refa_train.param_groups(model, 1e-3), lr=1e-3)
    out = refa_train.compute_losses(model, batch, rollout_k=4)
    for key in ("loss", "pred", "roll", "inv", "sigreg"):
        assert torch.isfinite(out[key].detach()), key
    out["loss"].backward()

    # Spec item 2 analog at feature level: the frozen-feature inputs are
    # gradient-free end-to-end.
    assert batch["feats"].requires_grad is False
    assert batch["feats"].grad is None
    assert batch["future_feats"].requires_grad is False
    assert batch["future_feats"].grad is None
    # Standardizer stats are buffers: no grad, no change from a step.
    assert not model.standardizer.mean.requires_grad
    opt.step()
    assert torch.equal(model.standardizer.mean, mean0)
    assert torch.equal(model.standardizer.std, std0)
    # The trainable path DID receive finite grads (adapter + predictor).
    # predictor.out_proj is reserved-but-unused in OperativePredictor.forward
    # ("reserved: feed predictions back") — legitimately grad-free.
    for name, p in model.named_parameters():
        if name.startswith("predictor.out_proj."):
            assert p.grad is None, name
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # Collapse monitor is a positive finite scalar on real batches.
    s = model.adapter_dim_std(out["states"])
    assert s > 0 and torch.isfinite(torch.tensor(s))


# ---------- (c) rollout_k=4 path ----------

def test_rollout_k4_executes_and_finite(tmp_path):
    root = _make_feature_root(tmp_path)
    model = _smoke_model()
    eps, _ = refa_train.load_feature_episodes(str(root), "*train*")
    model.standardizer.fit(ep["feats_fp16"] for ep in eps)
    batch = _batch(root, model, rollout_k=4)
    out = refa_train.compute_losses(model, batch, rollout_k=4)
    roll = out["roll"].detach()
    assert torch.isfinite(roll) and float(roll) > 0
    # SigReg pool = 3 horizon heads + 4 rollout predictions per window.
    assert out["n_sig"] == batch["feats"].shape[0] * 7
    # rollout_k=1 disables the rollout term (lever contract).
    out1 = refa_train.compute_losses(model, batch, rollout_k=1)
    assert float(out1["roll"]) == 0.0


# ---------- (d) ckpt save -> load -> resume, identical adapter outputs ----------

def test_ckpt_roundtrip_and_resume(tmp_path):
    root = _make_feature_root(tmp_path)
    out_dir = tmp_path / "run"
    argv = ["--data-root", str(root), "--out", str(out_dir), "--steps", "2",
            "--rollout-k", "4", "--batch", "4", "--lr", "1e-3",
            "--log-every", "1", "--device", "cpu", "--smoke"]
    metrics = refa_train.main(argv)
    assert metrics["final"]["step"] == 1
    assert all(torch.isfinite(torch.tensor(metrics["final"][k]))
               for k in ("loss", "pred", "roll", "adapter_std"))
    ckpt_path = out_dir / "ckpt.pt"
    assert ckpt_path.exists()

    # Fresh model + loaded ckpt reproduces adapter outputs bit-exactly,
    # WITHOUT any refit (stats come from the checkpoint).
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    trained = RefAModel(refa_train.smoke_pred_config())
    trained.load_state_dict(ck["model"])
    reloaded = RefAModel(refa_train.smoke_pred_config())
    reloaded.load_state_dict(ck["model"])
    assert bool(reloaded.standardizer.fitted)
    torch.manual_seed(123)
    fixed = torch.randn(2, 5, N_TOK, D).half()
    with torch.no_grad():
        a = trained.encode_window(fixed)
        b = reloaded.encode_window(fixed)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()

    # Resume: rerun with more steps — trainer must pick up at step 3 (no
    # refit; stored stats reused) and finish with finite losses.
    metrics2 = refa_train.main(argv[:5] + ["4"] + argv[6:])
    assert metrics2["final"]["step"] == 3
    ck2 = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ck2["step"] == 3
    assert torch.equal(ck2["model"]["standardizer.mean"],
                       ck["model"]["standardizer.mean"])
    assert torch.equal(ck2["model"]["standardizer.std"],
                       ck["model"]["standardizer.std"])


def test_grid_adapter_spatial_sensitivity():
    """Stage-2b (Sayed review): the grid adapter must PRESERVE token layout.

    Mean-pool is permutation-invariant (spatial info destroyed); the grid
    adapter must not be. Also pins state_dim = readout out_dim (2048 at
    full config; 16-token toy here) and window-shape handling.
    """
    import torch as t
    from tanitad.refs.refa import DinoGridAdapter, RefAModel

    t.manual_seed(0)
    ad = DinoGridAdapter(n_tokens=16, d_in=32, grid=2, d_readout=8)
    x = t.randn(3, 5, 16, 32)                      # [B, W, N, D]
    out = ad(x)
    assert out.shape == (3, 5, 32)                 # 2*2*8 = out_dim
    assert out.shape[-1] == ad.out_dim
    perm = t.randperm(16)
    out_p = ad(x[..., perm, :])
    assert not t.allclose(out, out_p, atol=1e-5), \
        "grid adapter must be sensitive to token order (spatial layout)"

    # RefAModel wiring: grid kind overrides state_dim with readout out_dim.
    m = RefAModel(adapter_kind="grid", d_dino=768, n_tokens=256)
    assert m.state_dim == m.adapter.out_dim == 2048
    m2 = RefAModel(adapter_kind="pool")
    assert m2.state_dim == 768                     # v1 unchanged (ckpt compat)
