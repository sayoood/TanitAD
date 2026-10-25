"""HiCAP audio bridge tests (CPU, synthetic). They check mechanisms and controls -- shapes, that the bridge CAN align to a text space on
a toy paired set, Doppler sign, the bearing-flip control, audio-null bit-exactness -- not audio competence on real recordings."""
import math
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from tanitad.hicap import audio as A
from tanitad.hicap.knowledge import hashed_text_embedding
from tanitad.hicap.model import GatedAudio, HiCAP, HiCAPConfig, LEVELS

TONE = [dict(fc=1000.0, fd=0.0, T=1.0, amp=1.0, gate=None, phase0=0.0)]


def _peak_hz(x, sr=16000):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), 1 << 20))
    return float(np.fft.rfftfreq(1 << 20, 1 / sr)[np.argmax(X)])


# ------------------------------------------------------------------ shapes
def test_frontend_shapes_mono_and_array():
    cfg = A.AudioConfig()
    w1, _ = A.render_synthetic_event("horn", A.linear_track([20, 5], [-5, 0]), seed=0)
    w4, m4 = A.render_synthetic_event("siren_wail", A.linear_track([30, 20], [-10, 0]), mic_xy=A.ROOF4, seed=1)
    assert w1.shape == (1, 32000) and w4.shape == (4, 32000) and w4.dtype == np.float32
    f1 = A.AudioFrontend(cfg, 1)(torch.from_numpy(w1)[None])
    assert f1["logmel"].shape == (1, 1, 128, 201) and f1["mono_logmel"].shape == (1, 128, 201) and f1["spatial"] is None
    fe4 = A.AudioFrontend(cfg, 4)
    f4 = fe4(torch.from_numpy(w4)[None])
    assert f4["spatial"].shape == (1, 50, fe4.d_spatial) and fe4.d_spatial == 3 * (3 * 16 + 2 * 64 + 1)
    assert f4["gcc"].shape == (1, 3, 129)
    assert m4["bearing_valid"] and 0 <= m4["sector"] < 8 and m4["dist_bin"] == 1


def test_window_stream_is_2s_at_half_second_hop():
    cfg = A.AudioConfig()
    x = torch.randn(2, 3, 16000 * 5)
    w = A.window_stream(x, cfg)
    assert w.shape == (2, 7, 3, 32000)                           # (5 - 2) / 0.5 + 1 windows
    assert torch.equal(w[:, 1, :, :8000], x[:, :, 8000:16000])


@pytest.mark.parametrize("d_v", [64, 2048])
def test_bridge_tokens_shape_and_rms(d_v):
    cfg = A.AudioConfig(n_mels=64); fe = A.AudioFrontend(cfg, 4); enc = A.StandInEncoder(64, 96)
    br = A.AudioBridge(A.BridgeConfig(d_enc=96, d_v=d_v, d_spatial=fe.d_spatial, width=64, heads=4, enc_hz=enc.frame_hz))
    path = A.AudioPath(fe, enc, br)
    w = torch.stack([torch.from_numpy(A.render_synthetic_event("horn", A.linear_track([15, -5], [0, 0]), mic_xy=A.ROOF4, seed=s)[0]) for s in range(3)])
    tok = path(w)
    assert tok.shape == (3, 8, d_v)
    assert torch.allclose(tok.pow(2).mean(-1).sqrt(), torch.ones(3, 8), atol=1e-4)
    assert A.n_params(enc) == 0 and isinstance(enc, A.FrozenAudioEncoder)                 # the tower is frozen
    tok.sum().backward()
    assert all(p.grad is None for p in enc.parameters()) and br.latents.grad is not None
    br.set_output_rms(0.02)                                       # mode (b): match the backbone's input-embedding RMS
    with torch.no_grad(): assert torch.allclose(path(w).pow(2).mean(-1).sqrt(), torch.full((3, 8), 0.02), atol=1e-5)


def test_event_heads_shapes_and_masked_loss():
    h = A.EventHeads(d_v=64)
    out = h(torch.randn(5, 8, 64))
    assert out["cls"].shape == (5, len(A.EVENT_CLASSES)) and out["bearing"].shape == (5, 8) and out["dist"].shape == (5, 4) and out["approach"].shape == (5,)
    tgt = dict(cls=torch.tensor([0, 1, 3, 0, 5]), bearing=torch.tensor([0, 2, 4, 0, 6]), dist=torch.tensor([0, 1, 2, 0, 3]),
               approach=torch.tensor([0., 1., 0., 0., 1.]), bearing_valid=torch.tensor([True, True, False, True, True]))
    l = A.EventHeads.loss(out, tgt)
    assert torch.isfinite(l["loss"]) and l["bearing"] > 0
    none = A.EventHeads.loss(out, dict(tgt, cls=torch.zeros(5, dtype=torch.long)))
    assert none["bearing"] == 0 and none["dist"] == 0 and none["approach"] == 0          # no event -> only the class term


def test_splice_soft_tokens_mode_b():
    emb = torch.zeros(2, 12, 16); tok = torch.randn(2, 8, 16)
    mask = torch.zeros(2, 12, dtype=torch.bool); mask[:, 3:11] = True
    out = A.splice_soft_tokens(emb, mask, tok)
    assert torch.equal(out[:, 3:11], tok) and torch.equal(out[:, :3], emb[:, :3])
    with pytest.raises(ValueError): A.splice_soft_tokens(emb, mask[:, :11].clone().resize_(2, 12).fill_(False), tok)


def test_caption_uses_metadata_only_and_respects_mono():
    _, m4 = A.render_synthetic_event("siren_wail", A.linear_track([-30, 10], [12, 0]), mic_xy=A.ROOF4, seed=2)
    _, m1 = A.render_synthetic_event("siren_wail", A.linear_track([-30, 10], [12, 0]), seed=2)
    c4, c1 = A.caption_for_event(m4), A.caption_for_event(m1)
    assert c4.startswith("a wailing siren approaching from the rear-left") and "m away" in c4
    assert "from the" not in c1                                   # mono carries no bearing -> the caption must not claim one
    assert A.caption_for_event({"cls": "none"}).startswith("no salient")


# ------------------------------------------------------------------ alignment
def test_bridge_alignment_learns_toy_pairs_and_retrieval_beats_chance():
    torch.manual_seed(0)
    CL, D = A.EVENT_CLASSES, 64
    fe = A.AudioFrontend(A.AudioConfig(n_mels=64), 1); enc = A.StandInEncoder(64, 96, seed=0)

    def make(n_per, seed0):
        X, y = [], []; rng = np.random.default_rng(seed0)
        for k, c in enumerate(CL):
            for _ in range(n_per):
                r, th, v = rng.uniform(8, 60), rng.uniform(-np.pi, np.pi), rng.uniform(-15, 15)
                p0 = np.array([r * np.cos(th), r * np.sin(th)])
                w, _ = A.render_synthetic_event(c, A.linear_track(p0, -v * p0 / r), ego_speed=rng.uniform(0, 15), seed=int(rng.integers(1 << 30)))
                X.append(w); y.append(k)
        with torch.no_grad(): return enc(fe(torch.from_numpy(np.stack(X)))["mono_logmel"]), torch.tensor(y)

    Etr, ytr = make(12, 1); Ete, yte = make(6, 2)                  # disjoint renders (different emitters, noise, seeds)
    bank = torch.from_numpy(np.stack([hashed_text_embedding(A.caption_for_event({"cls": c}, detail="class"), D) for c in CL]))
    br = A.AudioBridge(A.BridgeConfig(d_enc=96, d_v=D, width=64, heads=4, enc_hz=enc.frame_hz)); lossf = A.BridgeAlignLoss()
    with torch.no_grad(): base = (A.retrieve(br.pooled(br(Ete)), bank) == yte).float().mean().item()
    opt = torch.optim.Adam(list(br.parameters()) + list(lossf.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(0); hist = []
    for step in range(150):
        idx = torch.randperm(len(ytr), generator=g)[:32]
        out = lossf(br.pooled(br(Etr[idx])), bank[ytr[idx]], ytr[idx])
        opt.zero_grad(); out["loss"].backward(); opt.step(); hist.append(out["loss"].item())
    with torch.no_grad(): acc = (A.retrieve(br.pooled(br(Ete)), bank) == yte).float().mean().item()
    assert np.mean(hist[-10:]) < 0.6 * np.mean(hist[:5]), (hist[:5], hist[-10:])
    assert acc >= 0.5 and acc > base, (acc, base, 1 / len(CL))   # chance = 1/7


def test_align_loss_multi_positive_and_perfect_case():
    lf = A.BridgeAlignLoss(lam_cos=1.0)
    t = torch.eye(4)[[0, 1, 1, 2]]
    out = lf(t.clone(), t, torch.tensor([0, 1, 1, 2]))
    assert out["top1"] == 1.0 and out["cos"].abs() < 1e-6
    assert out["nce"] > 0                                         # duplicate captions share mass: floor log(2) on those rows, not inf


# ------------------------------------------------------------------ physics controls
def test_doppler_sign_and_magnitude():
    wa, ma = A.render_synthetic_event("horn", A.linear_track([80, 0], [-20, 0]), components=TONE, noise=False)
    wr, mr = A.render_synthetic_event("horn", A.linear_track([40, 0], [20, 0]), components=TONE, noise=False)
    fa, fr = _peak_hz(wa[0]), _peak_hz(wr[0])
    assert fa > 1000.0 > fr                                       # approaching -> higher pitch
    assert abs(fa - 1000 * A.C_SOUND / (A.C_SOUND - 20)) < 1.0 and abs(fr - 1000 * A.C_SOUND / (A.C_SOUND + 20)) < 1.0
    assert ma["approaching"] and not mr["approaching"] and ma["v_radial"] == pytest.approx(20, abs=0.1)


def test_gain_falls_as_one_over_r():
    near, _ = A.render_synthetic_event("horn", A.linear_track([10, 0], [0, 0]), components=TONE, noise=False)
    far, _ = A.render_synthetic_event("horn", A.linear_track([40, 0], [0, 0]), components=TONE, noise=False)
    ratio_db = 20 * np.log10(np.sqrt((near ** 2).mean()) / np.sqrt((far ** 2).mean()))
    exp_db = 20 * np.log10(4) + float(A.air_absorption_db_per_m(np.array([1000.0]))[0]) * 30
    assert ratio_db == pytest.approx(exp_db, abs=0.05)


@pytest.mark.parametrize("mics,p0", [(A.STEREO_1M, [10, 15]), (A.ROOF4, [-20, -12])])
def test_bearing_flip_control(mics, p0):
    w, m = A.render_synthetic_event("siren_wail", A.linear_track(p0, [3, 0]), mic_xy=mics, ego_speed=10, seed=3)
    truth = "left" if m["bearing_deg"] > 0 else "right"
    side, tau = A.rule_bearing_side(w, mics)
    side_m, tau_m = A.rule_bearing_side(A.mirror_channels(w, mics), mics)
    assert side == truth and side_m != side and tau_m == pytest.approx(-tau)
    fe = A.AudioFrontend(A.AudioConfig(), len(mics))
    f, fm = fe(torch.from_numpy(w)[None]), fe(torch.from_numpy(A.mirror_channels(w, mics))[None])
    if len(mics) == 2:                                            # the pair's level / delay features flip sign under the mirror
        K, L = fe.cfg.n_spatial_bands, fe.cfg.max_lag
        assert torch.sign(f["spatial"][..., :K].mean()) == -torch.sign(fm["spatial"][..., :K].mean())
        assert torch.allclose(fm["gcc"], f["gcc"].flip(-1), atol=1e-6)     # mirroring time-reverses the cross-correlation exactly


def test_track_from_agent_rig_frame():
    t = np.linspace(0, 2, 21)
    ego = np.stack([5 * t, np.zeros_like(t)], 1)                 # ego drives +x at 5 m/s, yaw 90 deg (facing world +y) -> rotate
    tr = A.track_from_agent(t, ego + np.array([0.0, 10.0]), t, ego, np.full_like(t, np.pi / 2))
    assert np.allclose(tr.xy, [[10.0, 0.0]] * 21)                # agent 10 m along world +y = straight ahead in the rig frame


# ------------------------------------------------------------------ audio-null parity through model.GatedAudio (test T1)
SMALL = dict(d=64, layers=2, heads=4, d_v=48, n_vlm_tokens=8, n_motion_tokens=16, d_text=32, d_audio=48, rerank_layers=1)


def _batch(B=3, c=None, seed=0):
    g = torch.Generator().manual_seed(seed); c = c or HiCAPConfig(**SMALL)
    return dict(vlm=torch.randn(B, 3, c.n_vlm_tokens, c.d_v, generator=g), last=torch.randn(B, c.d_v, generator=g),
                age_s=torch.rand(B, generator=g), frames=torch.randn(B, 3, 3, 64, 160, generator=g),
                ego=torch.randn(B, 3, generator=g), nav=torch.randint(0, 3, (B,), generator=g), nav_args=torch.randn(B, 4, generator=g),
                route=torch.randn(B, 4, generator=g))


def _bridge_tokens(d_v, B=3):
    fe = A.AudioFrontend(A.AudioConfig(n_mels=64), 2); enc = A.StandInEncoder(64, 96)
    br = A.AudioBridge(A.BridgeConfig(d_enc=96, d_v=d_v, d_spatial=fe.d_spatial, width=64, heads=4, enc_hz=enc.frame_hz))
    w = torch.stack([torch.from_numpy(A.render_synthetic_event("siren_yelp", A.linear_track([20, 8 * s - 8], [-8, 0]), mic_xy=A.STEREO_1M, seed=s)[0]) for s in range(B)])
    with torch.no_grad(): return A.AudioPath(fe, enc, br)(w)


def test_gated_audio_null_bit_exact_none_and_gate_zero():
    c = HiCAPConfig(**SMALL); ga = GatedAudio(c).eval()
    q = torch.randn(3, len(LEVELS), c.d); tok = _bridge_tokens(c.d_v)
    assert tok.shape == (3, 8, c.d_audio)                         # mode (a): d_audio = d_v
    assert ga(q, A.hicap_audio_input(None)) is q                  # audio=None -> branch skipped
    assert A.hicap_audio_input(tok, torch.zeros(3, dtype=torch.bool)) is None
    with torch.no_grad(): out0 = ga(q, A.hicap_audio_input(tok))
    assert torch.equal(out0.view(torch.int32), q.view(torch.int32))   # gate = 0 -> identical BITS
    with torch.no_grad(): ga.g.fill_(0.5); out1 = ga(q, tok)
    assert not torch.equal(out1, q)                               # the path is live once the gate opens


def test_hicap_encode_audio_null_bit_exact_with_bridge_tokens():
    m = HiCAP(HiCAPConfig(**SMALL)).eval(); b = _batch()
    with torch.no_grad():
        s0, g0, gp0, sc0 = m.encode(b)
        b2 = dict(b, audio=_bridge_tokens(48))
        s1, g1, gp1, sc1 = m.encode(b2)
    for k in s0: assert torch.equal(s0[k].view(torch.int32), s1[k].view(torch.int32))
    assert torch.equal(g0, g1) and torch.equal(gp0, gp1) and torch.equal(sc0, sc1)


def test_mirror_permutation_requires_symmetric_array():
    assert list(A.mirror_permutation(A.ROOF4)) == [1, 0, 3, 2]
    with pytest.raises(ValueError): A.mirror_permutation(np.array([[0.0, 0.3], [0.0, -0.1]]))
