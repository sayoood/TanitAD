"""test_imaghead.py: tests and deliberate-regression (mutation) runs of the tool `imaghead` (card PREREG_V8F_PROOF_ADD36, controls c1 to c5).
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_imaghead.py --basetemp F:/TanitAD-scratch/v8proof/imaghead/work/pt/<name>
Every control has a mutation test: the control must go RED when the thing it guards is broken (a passing control alone is not evidence that it reads the right thing)."""
import json
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = "F:/TanitAD-scratch/v8proof/moved_from_c/leadsup/tree_patch"
READ_DIR = "F:/TanitAD-scratch/v8proof/spatial/leadsup/read"
for p in (HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)
import imag_dump as ID  # noqa: E402
import imag_read as IR  # noqa: E402

torch, M, _tree = ID.import_stack(TREE)
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).replace(os.sep, "/").lower().startswith(TREE.lower() + "/stack/"), tanitad.__file__


# ------------------------------------------------------------------------------------------------ the tiny real model (random head, randomised weights so the action matters)
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    sys.path.insert(0, READ_DIR)
    import make_tiny_ckpt as MT
    d = tmp_path_factory.mktemp("tiny")
    ck = str(d / "tiny.pt")
    MT.build(ck, seed=0, tree=TREE)
    cfg, model, st, _ = ID.load_model(torch, M, ck)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in list(st.parameters()) + list(model.wm.lead_head.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return {"ckpt": ck, "cfg": cfg, "model": model, "st": st, "dir": d}


def windows_data(n, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, 16, D, generator=g) * 0.5
    A = torch.randn(n, 10, 2, generator=g) * 0.5
    return Z, A


# ------------------------------------------------------------------------------------------------ c1 and c4 on the real (tiny) model
def test_c1_offset_zero_is_exactly_zero(tiny):
    Z, A = windows_data(6, 640)
    r = ID.compute_reads(torch, M, tiny["model"], tiny["st"], Z, A, chunk=4)
    i0 = list(ID.OFFSETS).index(0.0)
    assert float(np.abs(r["roll_off"][:, i0] - r["roll_logged"]).max()) == 0.0
    assert float(np.abs(r["roll_off"][:, 1] - r["roll_logged"]).max()) > 1e-6            # the control reads something: a real offset moves the read


def test_c1_mutation_inexact_offset_goes_red(tiny, monkeypatch):
    Z, A = windows_data(6, 640)
    orig = ID.offset_actions
    monkeypatch.setattr(ID, "offset_actions", lambda A_, d, s: orig(A_, d, s) + 1e-3)       # the offset path adds a constant even at d = 0
    r = ID.compute_reads(torch, M, tiny["model"], tiny["st"], Z, A, chunk=4)
    assert float(np.abs(r["roll_off"][:, list(ID.OFFSETS).index(0.0)] - r["roll_logged"]).max()) > 0.0       # RED: the c1 check (== 0.0) would fail


def test_c4_zeroed_action_input_gives_exactly_zero(tiny):
    Z, A = windows_data(6, 640)
    zm = ID.zero_action_input(torch, tiny["model"])
    r = ID.compute_reads(torch, M, tiny["model"], tiny["st"], Z, A, chunk=4, zero_model=zm)
    for i in range(len(ID.OFFSETS)):
        assert float(np.abs(r["z_roll_off"][:, i] - r["z_roll_logged"]).max()) == 0.0
    assert float(np.abs(r["roll_off"][:, 1] - r["roll_logged"]).max()) > 1e-6
    # the original model is untouched by the copy
    assert float(sum(p.abs().sum() for p in tiny["st"].act_embed.parameters())) > 0.0


def test_c4_mutation_unzeroed_copy_goes_red(tiny, monkeypatch):
    Z, A = windows_data(6, 640)
    import copy
    monkeypatch.setattr(ID, "zero_action_input", lambda torch_, m: copy.deepcopy(m).eval())           # the copy keeps its action input
    zm = ID.zero_action_input(torch, tiny["model"])
    r = ID.compute_reads(torch, M, tiny["model"], tiny["st"], Z, A, chunk=4, zero_model=zm)
    assert max(float(np.abs(r["z_roll_off"][:, i] - r["z_roll_logged"]).max()) for i in range(len(ID.OFFSETS))) > 1e-6        # RED


# ------------------------------------------------------------------------------------------------ c2: two code paths on the same model
def test_c2_tool_rollout_equals_trainer_rollout_and_head(tiny):
    Z, A = windows_data(8, 640, seed=3)
    c2 = ID.control_c2(torch, M, tiny["model"], tiny["st"], Z, A)
    assert c2["pass"], c2
    assert c2["rollout_vs_roll_pairs_max_abs"] <= 1e-5 and c2["head_vs_sequence_max_abs"] <= 1e-5


def test_c2_mutation_shifted_actions_goes_red(tiny, monkeypatch):
    Z, A = windows_data(8, 640, seed=3)
    orig = ID.roll_states
    monkeypatch.setattr(ID, "roll_states", lambda t, st, z0, A_, ctx: orig(t, st, z0, A_ + 0.05, ctx))   # the tool feeds other actions than the trainer's path
    c2 = ID.control_c2(torch, M, tiny["model"], tiny["st"], Z, A)
    assert not c2["pass"] and c2["rollout_vs_roll_pairs_max_abs"] > 1e-5


def test_c2_mutation_head_window_off_by_one_goes_red(tiny, monkeypatch):
    Z, A = windows_data(8, 640, seed=3)
    orig = ID.head_raw
    monkeypatch.setattr(ID, "head_raw", lambda model, Z5: orig(model, torch.roll(Z5, 1, dims=1)))
    c2 = ID.control_c2(torch, M, tiny["model"], tiny["st"], Z, A)
    assert not c2["pass"] and c2["head_vs_sequence_max_abs"] > 1e-5


def test_facts_of_the_tree_are_on_the_stated_lines():
    f = ID.facts_check(TREE)
    assert all(v["ok"] for v in f.values()), f


def test_tree_pins_match_pins_file():
    import hashlib
    bad = []
    for ln in open("F:/TanitAD-scratch/v8proof/spatial/leadsup/PINS.txt", encoding="utf-8"):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        rel, want = t[1], (t[-1] if t[0] != "same" else t[2])
        got = hashlib.sha256(open(os.path.join(TREE, rel), "rb").read()).hexdigest()
        if got != want:
            bad.append(rel)
    assert not bad, bad


# ------------------------------------------------------------------------------------------------ c3: planted predictor and planted head through the tool's response code
class PlantedStack:
    """level-1 'predictor': coordinate 0 = displacement of the gap [m] since the start, coordinate 1 = ego speed change [m/s]. The ego accelerates by the PHYSICAL acceleration
    a = scaled channel x WM_ACC_SCALE. Exact constant-acceleration kinematics per 0.2 s step: gap -= v dt + 0.5 a dt^2; v += a dt."""
    dt = 0.2

    def __init__(self, acc_scale):
        self.predictors = [types.SimpleNamespace(context=3)]
        self.acc_scale = acc_scale

    def rollout(self, level, z0, A, context=1):
        z = z0.clone().double()
        outs = []
        for k in range(A.shape[1]):
            a = A[:, k, 0].double() * self.acc_scale
            z = z.clone()
            z[:, 0] = z[:, 0] - (z[:, 1] * self.dt + 0.5 * a * self.dt ** 2)
            z[:, 1] = z[:, 1] + a * self.dt
            outs.append(z)
        return torch.stack(outs, 1)


class PlantedModel:
    def __init__(self):
        head = lambda Z5: torch.stack([torch.zeros_like(Z5[:, -1, 0]), Z5[:, -1, 0] / 50.0, torch.zeros_like(Z5[:, -1, 0])], -1)[:, None]    # gap output = coordinate 0 (metres) / 50
        self.wm = types.SimpleNamespace(lead_head=head)


def planted_dump(n=120, n_ep=20, seed=1, acc_scale=2.0):
    Mfake = types.SimpleNamespace(WM_ACC_SCALE=acc_scale, LEAD_SUP_GAP_SCALE_M=50.0, LEAD_SUP_CLOS_SCALE_MS=10.0)
    g = torch.Generator().manual_seed(seed)
    Z = torch.zeros(n, 16, 4)
    A = torch.randn(n, 10, 2, generator=g) * 0.3          # |physical a| around 0.6 m/s2
    r = ID.compute_reads(torch, Mfake, PlantedModel(), PlantedStack(acc_scale), Z, A, chunk=50)
    rng = np.random.default_rng(seed)
    eid = np.array(["e%02d" % i for i in rng.integers(0, n_ep, n)])
    d = dict(r, eid=eid, g0=np.arange(n), offsets=np.asarray(ID.OFFSETS, np.float64), gap_t=rng.uniform(6, 49, n), v_t=rng.uniform(6, 20, n))
    return d, n_ep


def slope_of(d, T, n_ep, mutate=None):
    r, e = IR.h2_arrays(d, T)
    if mutate == "wrong_sign":
        e = -e
    if mutate == "wrong_time":
        e = e * 0.25
    ueid = sorted(set(d["eid"]))
    epi = np.array([ueid.index(x) for x in d["eid"]])
    sre, see, nw = IR.slope_stats(r, e, epi, n_ep, IR.h2_window_mask(d))
    return float(sre.sum() / see.sum()), nw


def test_c3_planted_predictor_and_head_read_slope_one():
    d, n_ep = planted_dump()
    b2, nw = slope_of(d, 2.0, n_ep)
    b1, _ = slope_of(d, 1.0, n_ep)
    assert nw > 100
    assert abs(b2 - 1.0) <= 1e-6, b2
    assert abs(b1 - 1.0) <= 1e-6, b1


def test_c3_planted_reader_level_c3_passes():
    assert IR.c3_planted()["pass"]


@pytest.mark.parametrize("mut", ["wrong_sign", "wrong_time"])
def test_c3_mutations_go_red(mut):
    d, n_ep = planted_dump()
    b, _ = slope_of(d, 2.0, n_ep, mutate=mut)
    assert abs(b - 1.0) > 1e-3, (mut, b)


def test_c3_mutation_offset_in_wrong_units_goes_red():
    """the offset added to the scaled channel without dividing by WM_ACC_SCALE (a unit error) moves the slope away from 1.0."""
    Mfake = types.SimpleNamespace(WM_ACC_SCALE=2.0, LEAD_SUP_GAP_SCALE_M=50.0, LEAD_SUP_CLOS_SCALE_MS=10.0)
    g = torch.Generator().manual_seed(1)
    Z, A = torch.zeros(120, 16, 4), torch.randn(120, 10, 2, generator=g) * 0.3
    orig = ID.offset_actions
    ID.offset_actions = lambda A_, d, s: orig(A_, d, 1.0)
    try:
        r = ID.compute_reads(torch, Mfake, PlantedModel(), PlantedStack(2.0), Z, A, chunk=50)
    finally:
        ID.offset_actions = orig
    rng = np.random.default_rng(1)
    d = dict(r, eid=np.array(["e%02d" % i for i in rng.integers(0, 20, 120)]), g0=np.arange(120), offsets=np.asarray(ID.OFFSETS), gap_t=rng.uniform(6, 49, 120), v_t=rng.uniform(6, 20, 120))
    b, _ = slope_of(d, 2.0, 20)
    assert abs(b - 2.0) < 1e-5            # the unit error doubles the response: slope 2.0, far from 1.0


# ------------------------------------------------------------------------------------------------ synthetic dumps for the reader
N_EP, WPE = 40, 8


def synth(tmp, seed, b0=1.0, mode="lead", name="X", present_only=False):
    """mode: 'lead' (rollout adds lead info beyond ego floor), 'ego' (adds ego-only info the floor also has), 'flat' (rollout read = present read). Returns (dump path, floor path)."""
    rng = np.random.default_rng(1000 + seed)
    eids = ["ep%03d" % i for i in range(N_EP)]
    n = N_EP * WPE
    eid = np.repeat(eids, WPE)
    g0 = np.tile(np.arange(4, 4 + 4 * WPE, 4), N_EP)
    gap_t = rng.uniform(8, 48, n)
    d_ego, d_lead = rng.normal(0, 4, n), rng.normal(0, 4, n)
    gap_t2 = gap_t + d_ego + d_lead
    gap_t2[rng.random(n) < 0.1] = np.nan
    v_t = rng.uniform(4, 20, n)
    ii = gap_t + rng.normal(0, 1.0, n)
    iii = np.where(np.isfinite(gap_t2), gap_t2, gap_t) + rng.normal(0, 1.0, n)
    if mode == "lead":
        i_read = ii + d_ego + d_lead + rng.normal(0, 1.0, n)
    elif mode == "ego":
        i_read = ii + d_ego + rng.normal(0, 1.0, n)
    else:
        i_read = ii.copy()
    fl = gap_t + d_ego + rng.normal(0, 1.0, n)            # the ego-only floor knows the ego part
    offs = np.asarray(ID.OFFSETS)
    roll_logged = np.zeros((n, 2, 3), np.float32)
    roll_logged[:, 1, 1] = i_read
    roll_logged[:, 0, 1] = ii
    roll_off = np.repeat(roll_logged[:, None], 4, 1).astype(np.float32)
    for j, d in enumerate(offs):
        for ti, T in ((0, 1.0), (1, 2.0)):
            roll_off[:, j, ti, 1] += (-0.5 * d * T * T * b0 + rng.normal(0, 0.3, n) * (d != 0)).astype(np.float32)
    present = np.zeros((n, 3), np.float32)
    present[:, 1] = ii
    true2 = np.zeros((n, 3), np.float32)
    true2[:, 1] = iii
    dmp = dict(eid=eid, g0=g0, row=np.arange(n), t=g0, fold=np.zeros(n, int), v_t=v_t, gap_t=gap_t, gap_t1=gap_t, gap_t2=gap_t2, pres_t=np.ones(n), offsets=offs, present=present,
               true1=present, true2=true2, roll_logged=roll_logged, roll_off=roll_off, z_roll_logged=np.zeros((n, 2, 3), np.float32), z_roll_off=np.zeros((n, 4, 2, 3), np.float32),
               ckpt_sha256=np.array("sha%d" % seed))
    p = os.path.join(str(tmp), "DUMP_%s_s%d.npz" % (name, seed))
    np.savez_compressed(p, **dmp)
    meta = {"ckpt_sha256": "sha%d_%s" % (seed, name), "n_windows_dumped": n, "controls_at_dump": {"c1_offset0_max_abs_response": 0.0, "c2": {"pass": True, "rollout_vs_roll_pairs_max_abs": 1e-7, "head_vs_sequence_max_abs": 0.0},
            "c4": {"max_abs_response_zero_model": 0.0, "max_abs_response_real_model_nonzero_offsets": 1.0}}}
    json.dump(meta, open(p + ".json", "w", encoding="utf-8"))
    yk = np.where(np.isfinite(gap_t2), gap_t2, np.nan)
    fp = os.path.join(str(tmp), "OOF_RB_G_s%d.npz" % seed)
    np.savez(fp, w_eid=eid, w_g0=g0, yk=yk, w_epi=np.repeat(np.arange(N_EP), WPE), uep=np.array(eids), p_iv=fl + 1.0, p_ivb=fl)
    return p, fp


def read_args(tmp, dumps, floors, arm, B=200):
    out = os.path.join(str(tmp), "out_" + arm)
    return ["read", "--dump", *dumps, "--arm", arm, "--seeds", "0,1", "--out-dir", out, "--floor-oof", *floors, "--B", str(B)], out


@pytest.mark.parametrize("mode,want", [("lead", "IMAG-HEAD-ADDS + IMAG-HEAD-ADDS-LEAD"), ("ego", "IMAG-HEAD-ADDS + IMAG-HEAD-ADDS-EGO-ONLY"), ("flat", "IMAG-HEAD-FLAT")])
def test_names_h1_branches(tmp_path, mode, want):
    ps = [synth(tmp_path, s, mode=mode, name=mode) for s in (0, 1)]
    argv, out = read_args(tmp_path, [p[0] for p in ps], [p[1] for p in ps], "H1" + mode)
    rc = IR.main(argv)
    res = json.load(open(os.path.join(out, "RESULT_IMAG_H1%s.json" % mode), encoding="utf-8"))
    assert res["controls"]["c1"]["pass"] and res["controls"]["c2"]["pass"] and res["controls"]["c3"]["pass"] and res["controls"]["c4"]["pass"], res["controls"]
    assert res["controls"]["c5"]["pass"], res["controls"]["c5"]
    assert rc == 0 and res["names"]["H1"] == want, res["names"]


@pytest.mark.parametrize("b0,want,over", [(1.0, "IMAG-RESPONDS", []), (0.3, "IMAG-RESPONDS-WEAKLY", []), (0.0, "IMAG-DEAF", []), (2.0, "IMAG-RESPONDS", [0, 1])])
def test_names_h2(tmp_path, b0, want, over):
    ps = [synth(tmp_path, s, b0=b0, mode="lead", name="b%g" % b0) for s in (0, 1)]
    argv, out = read_args(tmp_path, [p[0] for p in ps], [p[1] for p in ps], "H2_%g" % b0)
    IR.main(argv)
    res = json.load(open(os.path.join(out, "RESULT_IMAG_H2_%g.json" % b0), encoding="utf-8"))
    assert res["names"]["H2"] == want and res["names"]["H2_overshoot_seeds"] == over, (res["names"], res["H2"]["b_T2.0"])
    assert abs(res["H2"]["b_T2.0"]["pooled"] - b0) < 0.1
    verdict = open(os.path.join(out, "VERDICT_IMAG_H2_%g.md" % b0), encoding="utf-8").read()
    assert "STRATEGIC: PAUSED (PI item 25)" in verdict and "T0" in verdict and "driving performance, never" not in verdict.lower().replace("never driving performance", "") and ("closed" + " loop") not in verdict.lower()
    assert verdict.index("## Controls") < verdict.index("## H1")


def test_no_name_when_a_control_fails(tmp_path):
    ps = [synth(tmp_path, s, b0=1.0, name="cf") for s in (0, 1)]
    d = dict(np.load(ps[0][0], allow_pickle=True))
    d["z_roll_off"] = d["z_roll_off"].copy()
    d["z_roll_off"][0, 1, 1, 1] = 1e-3                       # the zero-action-input copy responds: c4 must fail
    np.savez_compressed(ps[0][0], **d)
    argv, out = read_args(tmp_path, [p[0] for p in ps], [p[1] for p in ps], "cf")
    rc = IR.main(argv)
    res = json.load(open(os.path.join(out, "RESULT_IMAG_cf.json"), encoding="utf-8"))
    assert rc == 2 and not res["controls"]["c4"]["pass"] and res["names"] == "NOT GIVEN: a control failed"


def test_c1_reader_mutation_inexact_offset0_fails(tmp_path):
    ps = [synth(tmp_path, s, name="c1m") for s in (0, 1)]
    d = dict(np.load(ps[1][0], allow_pickle=True))
    d["roll_off"] = d["roll_off"].copy()
    d["roll_off"][3, 0, 1, 1] += 1e-3
    np.savez_compressed(ps[1][0], **d)
    argv, out = read_args(tmp_path, [p[0] for p in ps], [p[1] for p in ps], "c1m")
    assert IR.main(argv) == 2
    assert not json.load(open(os.path.join(out, "RESULT_IMAG_c1m.json"), encoding="utf-8"))["controls"]["c1"]["pass"]


def test_c2_reader_mutation_label_mismatch_fails(tmp_path):
    ps = [synth(tmp_path, s, name="c2m") for s in (0, 1)]
    f = dict(np.load(ps[0][1], allow_pickle=True))
    f["yk"] = f["yk"].copy()
    f["yk"][5] += 0.5
    np.savez(ps[0][1], **f)
    argv, out = read_args(tmp_path, [p[0] for p in ps], [p[1] for p in ps], "c2m")
    assert IR.main(argv) == 2
    assert not json.load(open(os.path.join(out, "RESULT_IMAG_c2m.json"), encoding="utf-8"))["controls"]["c2"]["pass"]


# ------------------------------------------------------------------------------------------------ c5 on synthetic data (+ mutations)
def load_synth(tmp, mode="lead", name="c5"):
    ps = [synth(tmp, s, mode=mode, name=name) for s in (0, 1)]
    dumps = [IR.load_dump(p[0]) for p in ps]
    floors = [IR.load_floor(p[1]) for p in ps]
    return dumps, floors


def test_c5_synthetic_controls_pass(tmp_path):
    dumps, floors = load_synth(tmp_path)
    c = IR.c5_calibrated(dumps, floors, N_EP, 200)
    assert c["pass"], c
    assert max(abs(v) for q in ("H1", "H1e") for v in c["present_as_rollout"][q]) <= 1e-9
    assert all(v > 0 for v in c["true_future_as_rollout"]["H1"])
    assert all(c["permuted"][q]["ci"][0] <= 0 for q in ("H1", "H1e"))


def test_c5_mutation_identity_permutation_goes_red(tmp_path):
    dumps, floors = load_synth(tmp_path)
    c = IR.c5_calibrated(dumps, floors, N_EP, 200, mutate="identity_perm")
    assert not c["parts"]["permuted_not_above_0"] and not c["pass"]


def test_c5_mutation_present_read_not_substituted_goes_red(tmp_path):
    dumps, floors = load_synth(tmp_path)
    c = IR.c5_calibrated(dumps, floors, N_EP, 200, mutate="no_present_as_roll")
    assert not c["parts"]["present_as_rollout_is_0"] and not c["pass"]


def test_c5_mutation_flat_true_future_goes_red(tmp_path):
    """if the true-future read carried no information the third part must fail: replace iii by the present read."""
    dumps, floors = load_synth(tmp_path)
    for d in dumps:
        d["true2"] = d["present"].copy()
    d2 = [IR.c5_calibrated(dumps, floors, N_EP, 200)]
    # with iii == ii the H1 with the true future is exactly 0 (not above 0)
    assert not d2[0]["parts"]["true_future_as_rollout_above_0"]


def test_pair_names(tmp_path):
    outs = {}
    for arm, b0, mode in (("RL", 1.0, "lead"), ("LS", 0.4, "flat")):
        sub = tmp_path / arm
        sub.mkdir()
        ps = [synth(sub, s, b0=b0, mode=mode, name=arm) for s in (0, 1)]
        argv, out = read_args(sub, [p[0] for p in ps], [p[1] for p in ps], arm)
        IR.main(argv)
        outs[arm] = out
    IR.main(["pair", "--a-dir", outs["RL"], "--b-dir", outs["LS"], "--out-dir", str(tmp_path / "pair")])
    res = json.load(open(str(tmp_path / "pair" / "PAIR_IMAG_RL_minus_LS.json"), encoding="utf-8"))
    assert res["names"] == ["RL-IMAGINES-BETTER", "RL-RESPONDS-BETTER"], res
    IR.main(["pair", "--a-dir", outs["LS"], "--b-dir", outs["RL"], "--out-dir", str(tmp_path / "pair2"), "--a-arm", "LS", "--b-arm", "RL"])
    res2 = json.load(open(str(tmp_path / "pair2" / "PAIR_IMAG_LS_minus_RL.json"), encoding="utf-8"))
    assert res2["names"] == "nothing is named for the pair"


# ------------------------------------------------------------------------------------------------ cross-implementation equality with pool_rb.py
def test_statistics_equal_pool_rb():
    sys.path.insert(0, "F:/TanitAD-scratch/v8proof/spatial/q2/pool4")
    sys.path.insert(0, "F:/TanitAD-scratch/v8proof/spatial/q2")
    try:
        import pool_rb as PR
    except Exception as ex:                                             # noqa: BLE001
        pytest.skip("pool_rb not importable: %r" % (ex,))
    rng = np.random.default_rng(3)
    n, n_ep = 400, 30
    epi = rng.integers(0, n_ep, n)
    y = rng.normal(20, 8, n)
    y[rng.random(n) < 0.2] = np.nan
    P = {"i": y + rng.normal(0, 3, n), "ii": y + rng.normal(0, 4, n), "iii": y + rng.normal(0, 2, n), "fl": y + rng.normal(0, 5, n)}
    cal_mine, cal_ref = IR.calibrate(y, epi, P), PR.calibrate(y, epi, P)
    for k in IR.CAL:
        assert np.allclose(cal_mine[k], cal_ref[k], equal_nan=True, atol=1e-12), k
    S_ref = PR.RP.ep_stats_r2(epi, y, P["i"], n_ep)
    assert np.allclose(IR.ep_stats_r2(epi, y, P["i"], n_ep), S_ref, atol=1e-12)
    assert abs(IR.r2_pooled(y, P["i"]) - PR.RP.r2_pooled(y, P["i"])) < 1e-12


# ------------------------------------------------------------------------------------------------ end to end: tiny checkpoint, synthetic state file, dump, read
def test_end_to_end_smoke(tiny, tmp_path):
    n_ep, wpe = 10, 6
    eids = ["ep%02d" % i for i in range(n_ep)]
    n = n_ep * wpe
    eid = np.repeat(eids, wpe)
    g0 = np.tile(np.arange(4, 4 + 4 * wpe, 4), n_ep)
    rng = np.random.default_rng(0)
    # state file: every clip holds frames 0 .. 4*wpe + 16
    T = 4 * wpe + 16
    Zs = (rng.normal(0, 0.5, (n_ep * T, 640))).astype(np.float16)
    np.savez(tmp_path / "states.npz", Z=Zs, clip_idx=np.repeat(np.arange(n_ep), T).astype(np.int32), t=np.tile(np.arange(T), n_ep).astype(np.int16), clips=np.array(eids))
    np.savez(tmp_path / "windows.npz", eid=eid, g0=g0, t=g0, w_fold=np.zeros(n, int), act_scaled=rng.normal(0, 0.5, (n, 10, 2)).astype(np.float32))
    G15 = rng.uniform(8, 48, (n, 15))
    G15[::7, 4] = np.nan
    np.savez(tmp_path / "pack.npz", eid=eid, g0=g0, fold=np.zeros(n, int), V=rng.uniform(4, 20, (n, 16)), G15=G15, pres0=np.ones(n))
    out = str(tmp_path / "dump_s0.npz")
    rc = ID.main(["--ckpt", tiny["ckpt"], "--tree", TREE, "--windows", str(tmp_path / "windows.npz"), "--pack", str(tmp_path / "pack.npz"), "--states", str(tmp_path / "states.npz"),
                  "--out", out, "--chunk", "16", "--threads", "2"])
    assert rc == 0
    meta = json.load(open(out + ".json", encoding="utf-8"))
    assert meta["controls_at_dump"]["c1_offset0_max_abs_response"] == 0.0 and meta["controls_at_dump"]["c2"]["pass"] and meta["controls_at_dump"]["c4"]["max_abs_response_zero_model"] == 0.0
    assert all(f["ok"] for f in meta["facts"].values()) and "/stack/" in meta["tanitad_file"] and meta["n_windows_dumped"] == int(np.isfinite(G15[:, 4]).sum())
    d = IR.load_dump(out)
    assert d["roll_off"].shape == (meta["n_windows_dumped"], 4, 2, 3) and np.isfinite(d["roll_off"]).all()
    # the dump equals an independent evaluation of the same model on the same assembled windows (the head on frames 0..4 at the present)
    sel = d["row"]
    Z = torch.as_tensor(Zs.astype(np.float32).reshape(n_ep, T, 640))[torch.as_tensor(sel // wpe)]
    Zw = torch.stack([Z[i, g0[sel[i]] - 4:g0[sel[i]] - 4 + 16] for i in range(len(sel))])
    ref = ID.head_phys(ID.load_model(torch, M, tiny["ckpt"])[1], M, Zw[:, 0:5])
    assert float(np.abs(ref.detach().numpy() - d["present"]).max()) <= 1e-5
    # reader: runs; the floor file is a synthetic one over the same windows (no banked file exists for a tiny model)
    yk = np.where(np.isfinite(G15[:, 4]) & np.isfinite(G15[:, 14]), G15[:, 14], np.nan)
    np.savez(tmp_path / "OOF_RB_G_s0.npz", w_eid=eid, w_g0=g0, yk=yk, w_epi=np.repeat(np.arange(n_ep), wpe), uep=np.array(eids), p_iv=yk + rng.normal(0, 3, n), p_ivb=yk + rng.normal(0, 3, n))
    argv = ["read", "--dump", out, "--arm", "SMOKE", "--seeds", "0", "--out-dir", str(tmp_path / "smoke_out"), "--floor-oof", str(tmp_path / "OOF_RB_G_s0.npz"), "--B", "50"]
    IR.main(argv)
    res = json.load(open(str(tmp_path / "smoke_out" / "RESULT_IMAG_SMOKE.json"), encoding="utf-8"))
    assert res["controls"]["c1"]["pass"] and res["controls"]["c4"]["pass"] and res["controls"]["c2"]["pass"], res["controls"]
    assert os.path.exists(str(tmp_path / "smoke_out" / "VERDICT_IMAG_SMOKE.md"))
