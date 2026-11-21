"""Local tests of the arm S data-side kit (synthetic data, no pod, no GPU, no network).

Run the whole file:  PYCAP_MB=3000 bash /f/TanitAD-scratch/v8proof/ops/pycap.sh -m pytest test_steer_enc.py -q -p no:cacheprovider --basetemp <new dir>
Needs the stack tree (TANITAD_STACK, default E:/Projects/TanitAD/stack); the geometry tests skip when it is absent.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "enc"))
import enc_common as E   # noqa: E402
import steer_encode as SE   # noqa: E402
import ctl_s3 as S3   # noqa: E402
import ctl_s8 as S8   # noqa: E402

torch = pytest.importorskip("torch")
HAVE_STACK = os.path.isdir(os.path.join(E.STACK, "tanitad"))
needs_stack = pytest.mark.skipif(not HAVE_STACK, reason="stack tree not found")


def synth_intr():
    from tanitad.data.calib import FThetaIntrinsics
    return FThetaIntrinsics(poly=(0.0, 927.5032, 23.1353, -58.5012, 16.5067), cx=958.0, cy=700.0, width=1920, height=1080, per_clip=True)


def noise_video(T=6, h=270, w=480, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(T, 3, h, w, generator=g) * 255).to(torch.uint8)


# ------------------------------------------------------------------------------------------------ 1 centre reproduces a fixed window
@needs_stack
def test_steering_at_a_fixed_centre_reproduces_the_fixed_window():
    intr = synth_intr()
    vid = noise_video()
    h, w = vid.shape[-2:]
    frames = [E.window_frame(s) for s in E.SIZES_DEG]
    T = vid.shape[0]
    for az0, el0 in ((0.0, -6.0), (0.0, -2.0), (3.5, -1.25)):
        wins = SE.windows_from_video(vid, intr, frames, np.full(T, az0), np.full(T, el0))
        for i in range(2):
            g, m = E.window_grid(intr, h, w, frames[i], az0, el0)                       # the data kit's scalar route (arm D)
            ref = E.window_sample(vid.float(), g.expand(T, -1, -1, -1), m.expand(T, -1, -1)).numpy()
            d = np.abs(wins[:, i].astype(np.int32) - ref.astype(np.int32))
            assert d.max() <= 1, (az0, el0, i, int(d.max()))                             # same code; batched matmul may flip one level
            assert (d == 0).mean() > 0.999


@needs_stack
def test_steering_zero_zero_is_the_unrotated_window():
    intr = synth_intr()
    vid = noise_video(T=2)
    h, w = vid.shape[-2:]
    frames = [E.window_frame(s) for s in E.SIZES_DEG]
    wins = SE.windows_from_video(vid, intr, frames, np.zeros(2), np.zeros(2))
    g, m = E.window_grid(intr, h, w, frames[0], 0.0, 0.0)
    ref = E.window_sample(vid.float(), g.expand(2, -1, -1, -1), m.expand(2, -1, -1)).numpy()
    assert np.array_equal(wins[:, 0], ref)


# ------------------------------------------------------------------------------------------------ 2 refusals
def test_refuses_wrong_length_nan_inf_dtype_shape(tmp_path):
    ok = np.zeros((10, 2), np.float32)
    assert SE.check_steering(ok, 10) is ok
    with pytest.raises(ValueError, match="frames"):
        SE.check_steering(np.zeros((9, 2), np.float32), 10)
    with pytest.raises(ValueError, match="frames"):
        SE.check_steering(np.zeros((11, 2), np.float32), 10)
    bad = ok.copy()
    bad[4, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SE.check_steering(bad, 10)
    bad[4, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        SE.check_steering(bad, 10)
    with pytest.raises(ValueError, match="float32"):
        SE.check_steering(np.zeros((10, 2), np.float64), 10)
    with pytest.raises(ValueError, match="shape"):
        SE.check_steering(np.zeros((10, 3), np.float32), 10)
    p = tmp_path / "c.npy"
    np.save(p, np.zeros((9, 2), np.float32))
    with pytest.raises(ValueError):
        SE.read_steer(str(p), 10)
    with pytest.raises(FileNotFoundError):
        SE.read_steer(str(tmp_path / "none.npy"), 10)


# ------------------------------------------------------------------------------------------------ 3 s3 comparison logic
def fake_tokens(vid, intr, frames, az, el):
    """Stand-in for the encoder: the centred window pixels, flattened (a frame-sensitive 'token')."""
    w = SE.windows_from_video(vid, intr, frames, az, el).astype(np.float32) - 127.5
    return w.reshape(w.shape[0], -1)


@needs_stack
def test_s3_logic_passes_on_the_same_route_and_catches_a_one_frame_shift():
    intr = synth_intr()
    vid = noise_video(T=8)
    frames = [E.window_frame(s) for s in E.SIZES_DEG]
    T = vid.shape[0]
    az = np.linspace(-4.0, 6.0, T)
    el = np.linspace(-3.0, 1.0, T)
    has = np.ones(T, bool)
    has[[2, 5]] = False
    tokB = fake_tokens(vid, intr, frames, az, el)
    steer = np.stack([az, el], 1).astype(np.float32)
    tokS = fake_tokens(vid, intr, frames, steer[:, 0], steer[:, 1])             # float32 steering, same route
    r = S3.compare_s3(tokS, tokB, has)
    assert r["pass"] and r["n_lead"] == 6 and r["min"] >= S3.THRESH
    shifted = np.roll(steer, 1, axis=0)                                           # steering one frame late against the video
    tokS2 = fake_tokens(vid, intr, frames, shifted[:, 0], shifted[:, 1])
    r2 = S3.compare_s3(tokS2, tokB, has)
    assert not r2["pass"] and r2["n_below"] >= 5 and r2["min"] < 0.99
    shifted_up = np.roll(steer, -1, axis=0)                                       # and one frame early
    r3 = S3.compare_s3(fake_tokens(vid, intr, frames, shifted_up[:, 0], shifted_up[:, 1]), tokB, has)
    assert not r3["pass"]


def test_s3_compare_fails_on_shape_mismatch_and_no_lead():
    a = np.ones((4, 2, 3), np.float16)
    assert not S3.compare_s3(a, a[:3], np.ones(4, bool))["pass"]
    assert not S3.compare_s3(a, a, np.zeros(4, bool))["pass"]
    assert S3.compare_s3(a, a, np.ones(4, bool))["pass"]


def test_cosine_per_frame_values():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    c = S3.cosine_per_frame(a, b)
    assert np.allclose(c, [1.0, 2 ** -0.5, 0.0])


# ------------------------------------------------------------------------------------------------ s8 / manifest logic
def test_s8_check_clip_flags_missing_sha_and_mismatch(tmp_path):
    s_root = str(tmp_path)
    os.makedirs(os.path.join(s_root, "steer"))
    os.makedirs(os.path.join(s_root, "tok", "S"))
    c = "abc"
    st = np.zeros((5, 2), np.float32)
    np.save(SE.steer_path(s_root, c), st)
    tok = np.zeros((5, 2, 65, 4), np.float16)
    SE.write_npy(SE.tok_path(s_root, c), tok)
    tsha = E.sha256_file(SE.tok_path(s_root, c))
    ssha = E.sha256_file(SE.steer_path(s_root, c))
    man = {"clips": {c: {"n_frames": 5, "bytes": os.path.getsize(SE.tok_path(s_root, c)), "sha256": tsha, "steer_sha256": ssha}}}
    why, ok = S8.check_clip(c, {c: 5}, s_root, man)
    assert why == [] and ok
    why, _ = S8.check_clip(c, {c: 6}, s_root, man)
    assert any("steering file refused" in w for w in why)
    man["clips"][c]["steer_sha256"] = ""
    why, ok = S8.check_clip(c, {c: 5}, s_root, man)
    assert "manifest has no steering sha256" in why and not ok
    man["clips"][c]["steer_sha256"] = "0" * 64
    why, ok = S8.check_clip(c, {c: 5}, s_root, man)
    assert any("differs from the file on disk" in w for w in why) and not ok
    man["clips"][c]["steer_sha256"] = ssha
    man["clips"][c]["sha256"] = "1" * 64
    why, _ = S8.check_clip(c, {c: 5}, s_root, man)
    assert any("token sha256 differs" in w for w in why)


def test_box_steering_uses_box_direction_on_lead_frames(tmp_path):
    az = np.array([1.0, np.nan, 3.0])
    el = np.array([0.5, 0.0, -0.25])
    np.savez(tmp_path / "c.npz", has_lead=np.array([True, True, False]), ctr_az=az, ctr_el=el)
    st, has = SE.box_steering(str(tmp_path), "c")
    assert st.dtype == np.float32 and has.tolist() == [True, False, False]
    assert st[0].tolist() == [1.0, 0.5]
    assert st[1].tolist() == list(SE.NO_LEAD_FILL_DEG) and st[2].tolist() == list(SE.NO_LEAD_FILL_DEG)
    assert np.isfinite(st).all()
