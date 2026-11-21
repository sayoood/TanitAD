"""PREREG_V8F_IMAG1 control 4: the synthetic planted-effect test. A world whose latents are a linear embedding of (speed, accel, yaw rate, nuisance); four 'models':
  future  = the predicted latent is the true future plus noise        -> IA-2 IMAGINES (speed rows at the longer horizons)
  present = the predicted latent is the present (Copy-Last)           -> COPIES
  gain1   = the contrasted rollouts move the decoded speed / yaw rate by exactly the physical effect (gain 1.0) -> ANSWERS
  gain2   = the same with gain 2.0                                    -> OVERSHOOTS
The labels come from the SAME functions the real run uses (imag1_lib.run_draw, imag1_analyse.analyse). References: R = 19 permuted-label draws (smoke size; the real run uses 199).
Run: pytest -q test_imag1_controls.py   (CPU)."""
import os, sys, hashlib
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from types import SimpleNamespace
import numpy as np
import pytest
import imag1_probes as PB
import imag1_lib as IL
import imag1_analyse as AN

PB.MLP_CFG["max_epochs"] = 8
IL.R_REF = 19
NE, NWIN, D = 30, 6, 24


def make_synth(scn, gain=1.0):
    rng = np.random.default_rng(123)                          # the world is the same for every scenario (so the permutation references can be shared)
    M = rng.normal(size=(6, D))
    nw = NE * NWIN
    ep = np.repeat([f"ep{e:03d}" for e in range(NE)], NWIN)
    v0, aw, ww = rng.normal(8, 3, nw), rng.normal(0, 1.5, nw), rng.normal(0, 0.1, nw)
    g = rng.normal(size=(nw, 3))
    f = np.arange(16)[None]
    v = v0[:, None] + aw[:, None] * 0.2 * f
    a = np.repeat(aw[:, None], 16, 1)
    w = np.repeat(ww[:, None], 16, 1)
    nuis = g[:, None, :] + 0.1 * rng.normal(size=(nw, 16, 3))

    def emb(v_, a_, w_, n_):
        s = np.concatenate([(v_ / 3)[..., None], a_[..., None], (w_ / 0.1)[..., None], n_], -1)
        return s @ M + 0.05 * rng2.normal(size=s.shape[:-1] + (D,))

    rng2 = np.random.default_rng(124)
    Zall = emb(v, a, w, nuis).astype(np.float32)               # [nw, 16, D]
    srng = np.random.default_rng(7 + (1 if scn == "present" else 0))
    c = SimpleNamespace()
    c.name, c.tags, c.rand_tags = f"synth_{scn}", list(IL.TAGS), []
    c.beid = np.repeat(ep, 16).astype("U36")
    c.fold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in c.beid])
    n1, n2, n3 = nuis[..., 0], nuis[..., 1], nuis[..., 2]
    defined = n3 > 0.3
    Yb = {"speed": v, "accel": a, "yaw_rate": w, "curvature": w / np.maximum(v, 2.0),
          "lead_gap": np.where(defined, 15 + 4 * n1, np.nan), "lead_closing": np.where(defined, 2 * n2 + 0.5 * a, np.nan),
          "lead_present": (n1 > 0).astype(float), "lon7": np.clip(np.floor(a * 1.5 + 3), 0, 6), "agents30": np.round(5 + 2 * n2)}
    Yb["ttc"] = np.where(np.isfinite(Yb["lead_closing"]), np.clip(10 - Yb["lead_closing"], 0, 10), np.nan)
    c.Y = {q: Yb[q].reshape(-1) for q in IL.TARGETS}
    c.qe = ep.astype("U36")
    c.Yw = {q: Yb[q] for q in IL.TARGETS}
    bz = Zall.reshape(-1, D)
    c.BZ = {t: {l: bz for l in IL.LEVELS} for t in c.tags}
    c.Z0 = {t: {l: Zall[:, 0] for l in IL.LEVELS} for t in c.tags}
    c.ORA, c.PRED, c.OFF, c.CON, c.bankrank, c.actmean = {}, {}, {}, {}, {}, {}
    for t in c.tags:
        c.ORA[t], c.PRED[t], c.OFF[t], c.CON[t], c.bankrank[t] = {}, {}, {}, {}, {}
        for l in IL.LEVELS:
            for k in IL.KS:
                fk = k * IL.STRIDE[l]
                h = k * IL.DT[l]
                c.ORA[t][(l, k)] = Zall[:, fk]
                c.OFF[t][(l, k)] = Zall[:, fk + 1]
                c.PRED[t][(l, k)] = Zall[:, 0] if scn == "present" else Zall[:, fk] + 0.05 * srng.normal(size=(nw, D)).astype(np.float32)
                gn = {"present": 0.0, "future": 1.0, "gain1": 1.0, "gain2": 2.0}[scn]
                for cn, (ch, dl) in {"lonp": (0, 1.0), "lonm": (0, -1.0), "latp": (1, 0.25), "latm": (1, -0.25)}.items():
                    vv, ww_ = v[:, fk].copy(), w[:, fk].copy()
                    if ch == 0:
                        vv = vv + gn * dl * h
                    else:
                        ww_ = ww_ + gn * dl
                    z = emb(vv, a[:, fk], ww_, nuis[:, fk]).astype(np.float32) if scn != "present" else Zall[:, 0]
                    c.CON[t][(cn, l, k)] = z
                c.bankrank[t][(l, k)] = np.full(nw, 8.0)
                c.actmean[(l, k)] = np.stack([a[:, 0], w[:, 0]], 1)
    c.chance_bank = float((1.0 / np.arange(1, 17)).sum() / 16)
    c.win_mean_yaw = ww
    return IL.finish_ctx(c)


@pytest.fixture(scope="module")
def refs():
    c = make_synth("future")
    out = [AN.compress(c, IL.run_draw(c, s, "ref")) for s in range(1, IL.R_REF + 1)]
    cal = [AN.compress(c, IL.run_draw(c, s, "cal")) for s in (1001, 1002)]
    return out, cal


@pytest.fixture(scope="module")
def results(refs):
    r, cal = refs
    res = {}
    for scn in ("future", "present", "gain1", "gain2"):
        c = make_synth(scn)
        res[scn] = AN.analyse(c, IL.run_draw(c, 0, "real"), r, cal)
    return res


def test_future_is_imagined(results):
    lab = results["future"]["ia2_labels"]
    for (l, k) in ((2, 3), (3, 2), (3, 3)):
        assert lab[AN.key(l, k, "speed")]["ridge"] == "IMAGINES", lab[AN.key(l, k, "speed")]
        assert lab[AN.key(l, k, "speed")]["decision"] == "IMAGINES", lab[AN.key(l, k, "speed")]


def test_present_is_copied(results):
    lab = results["present"]["ia2_labels"]
    for (l, k) in ((1, 1), (1, 3), (2, 3), (3, 2), (3, 3)):
        assert lab[AN.key(l, k, "speed")]["decision"] == "COPIES", lab[AN.key(l, k, "speed")]
    # the predicted latent IS the copy: the paired difference is exactly 0
    e = results["present"]["ia2"][AN.key("ridge", "M2b_s0", 3, 3, "speed")]["pred_minus_copy"]
    assert e["point"] == 0.0 and e["lo"] == 0.0 and e["hi"] == 0.0


def test_gain1_answers(results):
    lab = results["gain1"]["ia3_labels"]
    for l in IL.LEVELS:
        for k in IL.KS:
            for ch in ("lon", "lat"):
                assert lab[AN.key(l, k, ch)]["ridge"] == "ANSWERS", (l, k, ch, lab[AN.key(l, k, ch)], results["gain1"]["ia3"][AN.key("ridge", "M2b_s0", l, k)][ch])
    g = results["gain1"]["ia3"][AN.key("ridge", "M2b_s0", 2, 2)]["lon"]["gain"]["point"]
    assert 0.9 < g < 1.1, g


def test_gain2_overshoots(results):
    lab = results["gain2"]["ia3_labels"]
    for l in IL.LEVELS:
        for k in IL.KS:
            for ch in ("lon", "lat"):
                assert lab[AN.key(l, k, ch)]["ridge"] == "OVERSHOOTS", (l, k, ch, lab[AN.key(l, k, ch)], results["gain2"]["ia3"][AN.key("ridge", "M2b_s0", l, k)][ch])
    g = results["gain2"]["ia3"][AN.key("ridge", "M2b_s0", 2, 2)]["lon"]["gain"]["point"]
    assert 1.8 < g < 2.2, g


def test_present_is_inert(results):
    lab = results["present"]["ia3_labels"]
    for ch in ("lon", "lat"):
        assert lab[AN.key(1, 1, ch)]["ridge"] == "INERT", lab[AN.key(1, 1, ch)]


def test_offset_is_not_the_future(results):
    c2 = results["future"]["control2"]
    assert c2["n_cells"] > 0 and not c2["offset_beats_oracle_cells"]


def test_json_roundtrip(results):
    import json
    from run_imag1 import jsonable
    json.dumps(jsonable(results["future"]))
