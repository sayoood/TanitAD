"""CPU tests of latvis_gw.py. Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider test_latvis_gw.py   (cwd = this folder; no PYTHONPATH needed)."""
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("LATVIS_NOLOCK", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import latvis_gw as G  # noqa: E402

CK = "F:/TanitAD-scratch/v8proof/spatial/ckpt"


def _have(arm):
    return os.path.exists(f"{CK}/{arm}/ckpt_last.pt") and os.path.exists(f"{CK}/{arm}/ckpt_last.sha256")


def test_tree_hashes_match_pins():
    h = G.assert_tree()
    assert all(v["matches_pin"] for v in h.values())


@pytest.mark.skipif(not _have("G_s0"), reason="G_s0 checkpoint absent")
def test_build_grid_strict_and_asserts():
    model, info = G.load_arm_model(f"{CK}/G_s0/ckpt_last.pt", 640, "grid")
    assert info["wm_state"] == "grid" and info["built_probe"]["location_sensitive"]
    with pytest.raises(AssertionError):
        G.load_arm_model(f"{CK}/G_s0/ckpt_last.pt", 640, "mean")       # kind assert
    with pytest.raises(AssertionError):
        G.load_arm_model(f"{CK}/G_s0/ckpt_last.pt", 128, "grid")       # width assert


@pytest.mark.skipif(not _have("W_s0"), reason="W_s0 checkpoint absent")
def test_build_mean_strict_and_asserts():
    model, info = G.load_arm_model(f"{CK}/W_s0/ckpt_last.pt", 640, "mean")
    assert info["wm_state"] == "mean" and not info["built_probe"]["location_sensitive"]
    with pytest.raises(AssertionError):
        G.load_arm_model(f"{CK}/W_s0/ckpt_last.pt", 640, "grid")


@pytest.mark.skipif(not _have("W_s0"), reason="W_s0 checkpoint absent")
def test_strict_loading_refuses_a_missing_key(tmp_path):
    import torch
    ck = torch.load(f"{CK}/W_s0/ckpt_last.pt", map_location="cpu", weights_only=False)
    k = next(iter(k for k in ck["model"] if k.startswith("wm.proj")))
    del ck["model"][k]
    p = str(tmp_path / "broken.pt")
    torch.save(ck, p)
    with pytest.raises(RuntimeError):
        G.load_arm_model(p, 640, "mean")


def test_sha_check_refuses_wrong_hash(tmp_path):
    f = tmp_path / "ckpt_last.pt"
    f.write_bytes(b"abc" * 1000)
    good = G.sha256_file(str(f))
    (tmp_path / "ckpt_last.sha256").write_text(good + "  ckpt_last.pt\n")
    assert G.check_sha(str(f)) == good
    (tmp_path / "ckpt_last.sha256").write_text("0" * 64 + "  ckpt_last.pt\n")
    with pytest.raises(ValueError):
        G.check_sha(str(f))
    os.remove(tmp_path / "ckpt_last.sha256")
    with pytest.raises(FileNotFoundError):
        G.check_sha(str(f))


@pytest.mark.skipif(not _have("G_s0"), reason="G_s0 checkpoint absent")
def test_real_sha_files_match():
    for arm in ("G_s0", "G_s1", "W_s0", "W_s1"):
        if _have(arm):
            G.check_sha(f"{CK}/{arm}/ckpt_last.pt")


def test_to_input_cell_layout():
    import torch
    n, cd = 3, 16
    X = torch.arange(n * 40 * cd, dtype=torch.float32).reshape(n, 40 * cd)
    g = G.to_input("grid", X)
    assert tuple(g.shape) == (n, cd, 4, 10)
    for cell in (0, 9, 10, 39):                      # cell = r * 10 + c, values cell-major
        r, c = divmod(cell, 10)
        assert torch.equal(g[1, :, r, c], X[1, cell * cd:(cell + 1) * cd])
    assert G.to_input("mean", X) is X


def test_decoders_tiny_end_to_end():
    import torch
    rng = np.random.default_rng(0)
    n = 60
    ep = np.repeat(np.arange(12), 5)
    IMG = torch.from_numpy(rng.integers(0, 255, (n, 3, 64, 160), dtype=np.uint8))
    for kind, X in (("Z640", torch.randn(n, 640)), ("G16", torch.randn(n, 16, 4, 10)), ("Z128", torch.randn(n, 128)), ("G128", torch.randn(n, 128, 4, 10))):
        m, info = G.LV.fit_decoder(kind, X, IMG, np.arange(n), ep, seed=0, device=torch.device("cpu"), max_epochs=1, widths=(32, 24, 16, 16, 8))
        o = m(X[:4])
        assert tuple(o.shape) == (4, 3, 64, 160) and bool(torch.isfinite(o).all()), kind
        e = G.LV.mse_in(m, X[:8], IMG, torch.arange(8))
        assert e.shape == (8,) and np.isfinite(e).all()


def test_decoder_matches_baseline_at_baseline_width():
    b = G.LV.make_decoder_orig("Z")
    g = G.make_decoder_gw("Z128")
    assert {k: tuple(v.shape) for k, v in b.state_dict().items()} == {k: tuple(v.shape) for k, v in g.state_dict().items()}
    b = G.LV.make_decoder_orig("G")
    g = G.make_decoder_gw("G128")
    assert {k: tuple(v.shape) for k, v in b.state_dict().items()} == {k: tuple(v.shape) for k, v in g.state_dict().items()}


def test_outcome_rules():
    assert G.q1_name(0.60, 0.65, None)[0] == "Z-NEAR-G"                        # 0.923 of the grid
    assert G.q1_name(0.50, 0.65, [0.01, 0.03])[0] == "Z-ABOVE-MEAN"
    assert G.q1_name(0.50, 0.65, [-0.01, 0.03])[0] == "Z-FLAT"
    assert G.q1_name(0.586, 0.65, None)[0] == "Z-NEAR-G"
    assert G.q1_name(0.584, 0.65, None)[0] == "Z-FLAT"
    assert G.both_seeds(["Z-NEAR-G", "Z-NEAR-G"], "Z-FLAT") == "Z-NEAR-G"
    assert G.both_seeds(["Z-NEAR-G", "Z-FLAT"], "Z-FLAT") == "Z-FLAT"
    pos = {"headroom_b_minus_c": {"positive_excl_0": True}, "learned": {"label": "IMAGINE-PIXELS"}}
    neg = {"headroom_b_minus_c": {"positive_excl_0": False}, "learned": {"label": "IMAGINE-PIXELS"}}
    flat = {"headroom_b_minus_c": {"positive_excl_0": True}, "learned": {"label": "IMAGINE-FLAT"}}
    assert G.pixel_label([pos, pos]) == "IMAGINE-PIXELS"
    assert G.pixel_label([pos, flat]) == "IMAGINE-FLAT"
    assert G.pixel_label([pos, neg]) == "NO-HEADROOM"
    assert G.px_label_one(True, (-0.2, -0.1)) == "IMAGINE-PIXELS" and G.px_label_one(True, (0.1, 0.2)) == "IMAGINE-BELOW-COPY"
    assert G.px_label_one(True, (-0.1, 0.2)) == "IMAGINE-FLAT" and G.px_label_one(False, (-0.2, -0.1)) == "NO-HEADROOM"
    assert G.attribution(True, [(0.01, 0.02), (0.005, 0.03)], +1) == "LOCATION"
    assert G.attribution(True, [(-0.01, 0.02), (-0.005, 0.03)], +1) == "WIDTH"
    assert G.attribution(True, [(0.01, 0.02), (-0.005, 0.03)], +1) == "MIXED"
    assert G.attribution(True, [(-0.02, -0.01), (-0.03, -0.005)], -1) == "LOCATION"
    assert G.attribution(False, [(0.01, 0.02)], +1) == "NO-EFFECT-TO-ATTRIBUTE"


def test_participation_ratio():
    rng = np.random.default_rng(0)
    assert G.participation_ratio(rng.normal(size=(2000, 64))) > 55
    x = np.outer(rng.normal(size=2000), rng.normal(size=64)) + 1e-6 * rng.normal(size=(2000, 64))
    assert G.participation_ratio(x) < 1.1


def test_smoke_outputs_if_present():
    d = f"{HERE}/smoke"
    if not os.path.exists(f"{d}/RESULT_LATVIS_GW.json"):
        pytest.skip("smoke not run")
    import json
    r = json.load(open(f"{d}/RESULT_LATVIS_GW.json", encoding="utf-8"))
    for e in r["arms"].values():
        assert np.isfinite(e["V1"]["heldout_skill"]) and np.isfinite(e["c4_participation_ratio"])
    z = np.load(f"{d}/LATENTS_GW_G_s0.npz")
    assert {"z0", "pred_2s", "true_2s", "pred_0p8s", "true_0p8s", "eid", "g0", "t", "w_fold", "act_scaled"} <= set(z.files)
    assert z["z0"].shape[1] == 640 and np.isfinite(z["pred_2s"].astype(np.float32)).all()
