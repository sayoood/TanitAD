"""PREREG_V8F_IMAG1B control 4 (extended): the synthetic planted-effect test, through the SAME functions the real run uses (imag1b_lib.run_draw, imag1b_analyse.compress / analyse).
A world whose latents are a linear embedding of (speed, accel, yaw rate, nuisance); the 'models':
  future  = the predicted latent is the true future plus noise        -> IA-2 IMAGINES
  present = the predicted latent is the present (Copy-Last)           -> COPIES
  gain1 / gain2 = the contrasted rollouts move the decoded speed / yaw rate by gain 1.0 / 2.0 -> ANSWERS / OVERSHOOTS
  shift   = the present plus a FIXED direction unrelated to every label (IMAG1B's added arm)  -> must NOT be IMAGINES; calibration under the refit reference must hold (<= 5 %),
            while IMAG1's paired sign-flip rule on the same draws must exceed 5 % (it reproduces the failure that invalidated IMAG1).
Control 2: pass in the 'future' world; a deliberate-regression arm (the not-future replaced by the Oracle's own latent) must FAIL it.
References: R = 39 refit draws and 12 calibration draws (smoke sizes; the real run uses 199 and 20). Run: pytest -q test_imag1b_controls.py   (CPU)."""
import os, sys, hashlib
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from types import SimpleNamespace
import numpy as np
import pytest
import imag1b_probes as PB
import imag1b_lib as IL
import imag1b_analyse as AN

PB.MLP_CFG["max_epochs"] = 8
IL.R_REF = 39
N_CAL = 12
NE, NWIN, D = 30, 6, 24


def make_synth(scn, gain=1.0):
    rng = np.random.default_rng(123)                          # the world is the same for every scenario
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
    u = np.random.default_rng(99).normal(size=D)
    u = (3.0 * u / np.linalg.norm(u)).astype(np.float32)       # the fixed direction of the 'shift' world, drawn independently of every label
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
    c.ORA, c.PRED, c.CON, c.bankrank, c.actmean = {}, {}, {}, {}, {}
    c.offidx = {}
    for t in c.tags:
        c.ORA[t], c.PRED[t], c.CON[t], c.bankrank[t] = {}, {}, {}, {}
        for l in IL.LEVELS:
            for k in IL.KS:
                fk = k * IL.STRIDE[l]
                h = k * IL.DT[l]
                c.ORA[t][(l, k)] = Zall[:, fk]
                if scn == "present":
                    c.PRED[t][(l, k)] = Zall[:, 0]
                elif scn == "shift":
                    c.PRED[t][(l, k)] = Zall[:, 0] + u
                else:
                    c.PRED[t][(l, k)] = Zall[:, fk] + 0.05 * srng.normal(size=(nw, D)).astype(np.float32)
                gn = {"present": 0.0, "shift": 0.0, "future": 1.0, "gain1": 1.0, "gain2": 2.0}[scn]
                for cn, (ch, dl) in {"lonp": (0, 1.0), "lonm": (0, -1.0), "latp": (1, 0.25), "latm": (1, -0.25)}.items():
                    vv, ww_ = v[:, fk].copy(), w[:, fk].copy()
                    if ch == 0:
                        vv = vv + gn * dl * h
                    else:
                        ww_ = ww_ + gn * dl
                    z = emb(vv, a[:, fk], ww_, nuis[:, fk]).astype(np.float32) if scn not in ("present", "shift") else (Zall[:, 0] if scn == "present" else Zall[:, 0] + u)
                    c.CON[t][(cn, l, k)] = z
                c.bankrank[t][(l, k)] = np.full(nw, 8.0)
                c.actmean[(l, k)] = np.stack([a[:, 0], w[:, 0]], 1)
    for l in IL.LEVELS:
        for k in IL.KS:
            for d in IL.OFFD:
                fk = k * IL.STRIDE[l]
                c.offidx[(l, k, d)] = np.array([i * 16 + fk + d if fk + d <= 15 else -1 for i in range(nw)], np.int64)
    c.chance_bank = float((1.0 / np.arange(1, 17)).sum() / 16)
    c.win_mean_yaw = ww
    return IL.finish_ctx(c)


def run_refs(c, n_ref=None, n_cal=N_CAL, keep_cal_res=False):
    refs = [AN.compress(c, IL.run_draw(c, s, "ref")) for s in range(1, (n_ref or IL.R_REF) + 1)]
    cal, cal_res = [], []
    for s in range(1001, 1001 + n_cal):
        r = IL.run_draw(c, s, "cal")
        cal.append(AN.compress(c, r))
        if keep_cal_res:
            cal_res.append(r)
    return refs, cal, cal_res


@pytest.fixture(scope="module")
def world_future():
    c = make_synth("future")
    refs, cal, _ = run_refs(c)
    return c, refs, cal


@pytest.fixture(scope="module")
def results(world_future):
    c0, refs, cal = world_future
    res = {}
    for scn in ("future", "present", "gain1", "gain2"):
        c = make_synth(scn)
        res[scn] = AN.analyse(c, IL.run_draw(c, 0, "real"), refs, cal)
    return res


@pytest.fixture(scope="module")
def shift_world():
    c = make_synth("shift")
    refs, cal, cal_res = run_refs(c, keep_cal_res=True)
    real = IL.run_draw(c, 0, "real")
    return c, refs, cal, cal_res, AN.analyse(c, real, refs, cal)


def test_future_is_imagined(results):
    lab = results["future"]["ia2_labels"]
    for (l, k) in ((2, 3), (3, 2), (3, 3)):
        assert lab[AN.key(l, k, "speed")]["ridge"] == "IMAGINES", lab[AN.key(l, k, "speed")]
        assert lab[AN.key(l, k, "speed")]["decision"] == "IMAGINES", lab[AN.key(l, k, "speed")]


def test_present_is_copied(results):
    lab = results["present"]["ia2_labels"]
    for (l, k) in ((1, 1), (1, 3), (2, 3), (3, 2), (3, 3)):
        assert lab[AN.key(l, k, "speed")]["decision"] == "COPIES", lab[AN.key(l, k, "speed")]
    e = results["present"]["ia2"][AN.key("ridge", "M2b_s0", 3, 3, "speed")]["pred_minus_copy"]
    assert e["point"] == 0.0 and e["lo"] == 0.0 and e["hi"] == 0.0           # the predicted latent IS the copy: the paired difference is exactly 0


def test_gain1_answers(results):
    lab = results["gain1"]["ia3_labels"]
    for l in IL.LEVELS:
        for k in IL.KS:
            for ch in ("lon", "lat"):
                assert lab[AN.key(l, k, ch)]["ridge"] == "ANSWERS", (l, k, ch, lab[AN.key(l, k, ch)], results["gain1"]["ia3"][AN.key("ridge", "M2b_s0", l, k)][ch])
    g = results["gain1"]["ia3"][AN.key("ridge", "M2b_s0", 2, 2)]["lon"]["gain"]["point"]
    assert 0.9 < g < 1.1, g
    e = results["gain1"]["ia3"][AN.key("ridge", "M2b_s0", 2, 2)]
    assert e["cos_gt_0.7_and_separated"], e["cos"]


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


def test_nf_is_a_different_episode():
    qe = np.repeat([f"ep{e:03d}" for e in range(NE)], NWIN).astype("U36")
    a, b = IL.nf_indices(qe), IL.nf_indices(qe)
    assert (a == b).all() and (qe[a] != qe).all() and a.min() >= 0 and a.max() < len(qe)
    assert len(np.unique(a)) > len(qe) // 2                                   # a spread draw, not one fixed window


def test_control2_passes_and_regression_arm_fails(results, world_future):
    c2 = results["future"]["control2"]
    assert c2["n_cells"] > 0 and c2["pass"], [x for x in c2["cells"] if not x["separated"]][:4]
    # deliberate regression: the not-future is the Oracle's own latent -> the gap is exactly 0 in every draw -> control 2 must FAIL
    c = make_synth("future")
    c.nf_idx = np.arange(len(c.qe))
    refs = [AN.compress(c, IL.run_draw(c, s, "ref")) for s in range(1, 6)]
    r = AN.analyse(c, IL.run_draw(c, 0, "real"), refs, [AN.compress(c, IL.run_draw(c, 1001, "cal"))])
    assert r["control2"]["n_cells"] > 0 and not r["control2"]["pass"] and r["control2"]["n_not_separated"] == r["control2"]["n_cells"]


def test_decorrelation_curve_is_reported(results):
    dc = results["future"]["decorrelation"]["cells"]
    assert dc and all(set(v) >= {"nf", "off1", "off4", "off8", "targets"} for v in dc.values())


def test_shift_is_not_imagined(shift_world):
    R = shift_world[4]
    labs = [v["decision"] for v in R["ia2_labels"].values()]
    assert "IMAGINES" not in labs, [k for k, v in R["ia2_labels"].items() if v["decision"] == "IMAGINES"]


def test_shift_calibration_new_rule_holds_old_rule_fails(shift_world):
    """At the smoke size R = 39 a 97.5th-percentile threshold is exceeded at a nominal 5 % per side (about 3 % at the real R = 199), and a family such as SCENE has few, strongly correlated rows.
    So the smoke asks: the new rule's share of predicted-vs-Copy rows is at most 5 % overall and at most 10 % (twice nominal) in every family, and IMAG1's paired sign-flip rule on the SAME draws
    exceeds 10 % and is at least 3 times larger (it reproduces the failure that invalidated IMAG1). The real run applies the card's 5 % bar per family AND per row kind at R = 199."""
    c, refs, cal, cal_res, R = shift_world
    fam = R["control3"]["families"]
    n_new = s_new = 0
    for f in AN.FAMS:
        rows, sep = fam[f]["by_kind"]["pred-vs-Copy"]
        assert sep / rows <= 0.10, (f, rows, sep, fam[f]["split"])
        n_new += rows
        s_new += sep
    assert s_new / n_new <= 0.05, (s_new, n_new)
    n = s = 0
    for res in cal_res:
        for tag, t in res["tags"].items():
            for P in IL.PROBES:
                for (l, k, q, qt), m in t["ia2"][P].items():
                    if qt != "pred":
                        continue
                    pt = IL.paired_test(m.diff(t["ia2"][P][(l, k, q, "copy")]), c.W, c.S)
                    n += 2
                    s += int(pt["above"]) + int(pt["below"])
    assert n == n_new, (n, n_new)
    assert s / n > 0.10 and s / n >= 3 * (s_new / n_new), (s, n, s_new)


def test_json_roundtrip(results):
    import json
    from run_imag1b import jsonable
    json.dumps(jsonable(results["future"]))


def _write_synth_result(R, path, c3_pass=True):
    import json, copy
    from run_imag1b import jsonable
    D = jsonable(copy.deepcopy(R))
    D.update({"status": "RUN", "wall_s": 1.0, "draw_time_s": {"mean": 1.0, "n": 1, "pool": 1}})
    D["control2"]["pass"] = True                      # the report/figure CODE is exercised on the synthetic world; the controls themselves are tested above
    D["control3"]["pass"] = c3_pass
    json.dump({"card": "x", "card_prefix_sha256": "0" * 64, "impl_sha256": "0" * 64, "tier": "T0", "datasets": {"eval141": D}}, open(path, "w", encoding="utf-8"))


def test_report_and_figs_run_on_synthetic_result(results, tmp_path):
    import subprocess
    env = dict(os.environ, IMAG1B_SUFFIX="_synth", IMAG1B_SYNTH_REPORT_TEST="1", IMAG1B_FIGDIR=str(tmp_path / "fig"))
    res_json, res_md = os.path.join(HERE, "RESULT_IMAG1B_synth.json"), os.path.join(HERE, "RESULT_IMAG1B_synth.md")
    try:
        # valid read: the per-family tables, the triggers and the figures are written
        _write_synth_result(results["future"], res_json)
        for cmd in (["report_imag1b.py"], ["figs_imag1b.py", "eval141"], ["report_imag1b.py"]):
            r = subprocess.run([sys.executable] + cmd, cwd=HERE, env=env, capture_output=True, text=True)
            assert r.returncode == 0, (cmd, r.stderr[-1500:])
        md = open(res_md, encoding="utf-8").read()
        for needle in ("VALID", "LONGITUDINAL (LON)", "LATERAL (LAT)", "TACTICAL (TAC)", "SCENE (SCENE)", "STRATEGIC", "decorrelation curve", "Triggered levers", "Turning vs straight"):
            assert needle in md, needle
        assert len([p for p in os.listdir(tmp_path / "fig") if p.endswith(".png")]) == 6
        # invalid read (control 3 fails): ONLY the control rows are written, no label table
        _write_synth_result(results["future"], res_json, c3_pass=False)
        r = subprocess.run([sys.executable, "report_imag1b.py"], cwd=HERE, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        md = open(res_md, encoding="utf-8").read()
        assert "INVALID as committed" in md and "## IA-2" not in md and "Decision labels" not in md and "LONGITUDINAL (LON)" not in md and "Triggered levers" not in md
    finally:
        for p in (res_json, res_md):
            if os.path.exists(p):
                os.remove(p)
