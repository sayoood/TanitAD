"""test_latvis2.py: tests and deliberate-regression (mutation) runs of the tool `latvis2` (card PREREG_V8F_PROOF_ADD38, controls k1 to k6 and the naming rules).
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_latvis2.py --basetemp F:/TanitAD-scratch/v8proof/latvis2/work/pt/<new name>
Every control has a mutation test: the control must go RED when the thing it guards is broken (a passing control alone is not evidence that it reads the right thing).
All data here is synthetic or from a tiny random model; no number of a test is a result."""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lv2_pod_tokens as PT  # noqa: E402
import lv2_read as RD  # noqa: E402

TREE = "F:/TanitAD-scratch/v8proof/moved_from_c/leadsup/tree_patch"
IMAG = "F:/TanitAD-scratch/v8proof/imaghead"
READ_DIR = "F:/TanitAD-scratch/v8proof/spatial/leadsup/read"
import itertools
_CNT = itertools.count()
NF, NC = 24, 6          # synthetic token width and components


# ------------------------------------------------------------------------------------------------ a synthetic world with the real layout
def make_world(tmp, n_ep=12, T=40, seed=0, flat=False, n_fold=3, noise=0.8):
    """Episodes with a latent random walk u_s (8 dims); tokens [T,40,NF] = u B_c + noise; state [T,640] = u W + noise. flat=True: u constant, no state noise -> state and tokens of t and t + 2.0 s alike."""
    rng = np.random.default_rng(seed)
    Bc = rng.normal(size=(NC + 2, NCELL := 40, NF)) / np.sqrt(NC + 2)
    Wz = rng.normal(size=(NC + 2, 640)) / np.sqrt(NC + 2)
    tok, st = {}, {}
    for e in range(n_ep):
        name = f"ep{e:02d}"
        u = np.cumsum(rng.normal(size=(T, NC + 2)) * 0.6, 0) if not flat else np.tile(rng.normal(size=(1, NC + 2)), (T, 1))
        tok[name] = (np.einsum("tk,kcf->tcf", u, Bc) + rng.normal(size=(T, NCELL, NF)) * noise).astype(np.float16)
        st[name] = (u @ Wz + (0 if flat else rng.normal(size=(T, 640)) * noise)).astype(np.float32)
    eid, g0, fold = [], [], []
    for e in range(n_ep):
        for g in range(4, T - 11, 4):
            eid.append(f"ep{e:02d}"); g0.append(g); fold.append(e % n_fold)
    eid, g0, fold = np.array(eid), np.array(g0, np.int64), np.array(fold, np.int64)
    keys, fidx = PT.unique_frames(eid, g0)
    mmp = os.path.join(str(tmp), f"tok_{seed}_{int(flat)}_{next(_CNT)}.npy")
    mm = np.lib.format.open_memmap(mmp, mode="w+", dtype=np.float16, shape=(len(keys), 40, NF))
    for r, (e, s) in enumerate(keys):
        mm[r] = tok[e][s]
    mm.flush()
    mm = np.load(mmp, mmap_mode="r")
    Zw = np.stack([[st[e][g - 4 + f] for f in range(16)] for e, g in zip(eid, g0)]).astype(np.float32)
    return dict(eid=eid, g0=g0, fold=fold, keys=keys, fidx=fidx, mm=mm, Zw=Zw, tok=tok, st=st, A=np.zeros((len(eid), 10, 2), np.float32))


def run_compute(w, alpha=0.6, mutate=None, panel=(), nc=NC):
    Zw = w["Zw"]
    imagine = lambda z0: (z0 + alpha * (Zw[:, 14] - z0)).astype(np.float32)           # an imagination that moves a fraction alpha toward the true future state
    return PT.compute(Zw, w["A"], w["eid"], w["g0"], w["fold"], w["mm"], w["keys"], w["fidx"], imagine, nc=nc, lam_rel=1e-2, panel_windows=panel, chunk=64, mutate=mutate)


def make_dump(w, out, k5=0.0, seed_tag=0, ckpt="x" * 64):
    res, recs, pan, Zim, cd = out
    d = dict(eid=w["eid"], g0=w["g0"], fold=w["fold"], **res, **pan,
             skill_train=np.array([r["skill_train"] for r in recs]), skill_heldout=np.array([r["skill_heldout"] for r in recs]),
             skill_heldout_permuted=np.array([r["skill_heldout_permuted"] for r in recs]), skill_train_present=np.array([r["skill_train_present"] for r in recs]),
             skill_heldout_present=np.array([r["skill_heldout_present"] for r in recs]), fold_ids=np.array([r["fold"] for r in recs]), k5_max_abs=np.float64(k5))
    d["meta"] = {"ckpt_sha256": ckpt, "k5": {"file": "banked.npz"}}
    d["path"] = f"mem://{out_id(seed_tag)}"
    return d


def out_id(i):
    return f"dump{i}"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return make_world(tmp_path_factory.mktemp("w"))


@pytest.fixture(scope="module")
def good(world):
    return run_compute(world, 0.6)


def ref_file(w, tmp_path, order=None):
    p = str(tmp_path / "ref_R1.npz")
    idx = np.arange(len(w["eid"])) if order is None else order
    np.savez(p, eid=w["eid"][idx], g0=w["g0"][idx], w_fold=w["fold"][idx])
    return p


def two_dumps(world, alpha=0.6, mutate=None, k5=0.0):
    ds = []
    for s in range(2):
        w = make_world(world["mm"].filename and os.path.dirname(world["mm"].filename), seed=0, noise=0.05) if False else world
        ds.append(make_dump(w, run_compute(w, alpha, mutate), k5=k5, seed_tag=s))
    return ds


# ------------------------------------------------------------------------------------------------ layout and basics
def test_cell_layout_is_cell_major_blocks_of_16(world, good):
    """level L3: cell c = state[c*16:(c+1)*16] (v8f_mini.py:880). A change confined to the block of cell 7 is seen in cell 7 only; and the stored dist_pres is that function of the states."""
    a = np.zeros((1, 640)); b = np.zeros((1, 640)); b[0, 7 * 16:8 * 16] = 1.0
    d = PT.cell_sq(a, b)[0]
    assert d[7] == 16.0 and np.delete(d, 7).sum() == 0.0
    ref = PT.cell_sq(world["Zw"][:, 4], world["Zw"][:, 14])
    assert np.allclose(good[0]["dist_pres"], ref, rtol=1e-4, atol=1e-4)


def test_cell_layout_mutation_strided_blocks_go_red(world, good):
    Z4, Z14 = world["Zw"][:, 4], world["Zw"][:, 14]
    strided = ((Z4 - Z14) ** 2).reshape(-1, 16, 40).sum(1)                     # the wrong layout (16 x 40 instead of 40 x 16)
    assert not np.allclose(good[0]["dist_pres"], strided, rtol=1e-3, atol=1e-3)


def test_facts_of_the_tree_are_on_the_stated_lines():
    f = PT.facts_check(TREE)
    assert all(v["ok"] for v in f.values()), f


def test_boot_plan_equals_imaghead():
    sys.path.insert(0, IMAG)
    import imag_read as IR
    a, b = RD.boot_plan(2, 31, 50, 0), IR.boot_plan(2, 31, 50, 0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_unique_frames_and_copies_agree(world):
    keys, fidx = world["keys"], world["fidx"]
    assert len(keys) == len(set(keys))
    Xf, diff = PT.frame_states(world["Zw"], fidx, len(keys))
    assert diff == 0.0


def test_token_map_has_skill_and_perm_does_not(good):
    res, recs, pan, Zim, cd = good
    for r in recs:
        assert r["skill_train"] > r["skill_heldout"] > 0.3
        assert r["skill_heldout_present"] > 0.3
        assert r["skill_heldout_permuted"] <= RD.K1_BAR


def test_D_positive_for_an_informative_imagination(good, world):
    d = make_dump(world, good)
    st = RD.seed_stats(d)
    assert RD.point(st, "D_all") > 0 and RD.point(st, "D_mov") > 0 and RD.point(st, "H_mov") > RD.point(st, "D_mov") * 0.9


def test_D_statistic_matches_a_direct_loop(good, world):
    d = make_dump(world, good)
    st = RD.seed_stats(d)
    mov = RD.moving_mask(d["tok_change"])
    tot = 0.0; n = 0
    for i in range(len(d["eid"])):
        tot += float((d["err_pres"][i][mov[i]].astype(np.float64) - d["err_imag"][i][mov[i]]).mean()); n += 1
    assert abs(tot / n - RD.point(st, "D_mov")) < 1e-6 * abs(tot / n)
    assert (mov.sum(1) == RD.NMOV).all()


# ------------------------------------------------------------------------------------------------ k1 .. k6, each with its deliberate regression
def test_k1_passes_and_identity_permutation_goes_red(world):
    ok = RD.control_k1([make_dump(world, run_compute(world, 0.6), seed_tag=0)])
    bad = RD.control_k1([make_dump(world, run_compute(world, 0.6, mutate="identity_perm"), seed_tag=0)])
    assert ok["pass"] and not bad["pass"], (ok, bad)


def test_k2_passes_and_in_sample_heldout_skill_goes_red(world):
    ok = RD.control_k2([make_dump(world, run_compute(world, 0.6))])
    bad = RD.control_k2([make_dump(world, run_compute(world, 0.6, mutate="skill_in_sample"))])
    assert ok["pass"] and not bad["pass"] and not any(bad["train_gt_heldout"])


def test_k2_present_skill_must_be_above_zero(world, good):
    d = make_dump(world, good)
    d["skill_heldout_present"] = np.array([0.0, -0.1, 0.2])
    assert not RD.control_k2([d])["pass"]


def test_k3_passes_and_unsubstituted_goes_red(world, good):
    d = make_dump(world, good)
    assert RD.control_k3([d])["pass"]
    assert not RD.control_k3([d], substitute=False)["pass"]


def test_k4_known_positive_passes_and_flat_future_goes_red(world, good, tmp_path):
    B = 300
    plan = RD.boot_plan(1, len(np.unique(world["eid"])), B, 0)
    assert RD.control_k4([make_dump(world, good)], plan, B)["pass"]
    wf = make_world(tmp_path, seed=3, flat=True)
    bad = RD.control_k4([make_dump(wf, run_compute(wf, 0.6))], RD.boot_plan(1, len(np.unique(wf["eid"])), B, 0), B)
    assert not bad["pass"] and bad["D_equals_headroom"] == [True]          # the equality holds by construction; the interval above 0 is what fails


def test_k5_equal_rollout_passes_shifted_rollout_and_missing_go_red(world, good):
    assert RD.control_k5([make_dump(world, good, k5=2e-5)])["pass"]
    assert not RD.control_k5([make_dump(world, good, k5=3e-2)])["pass"]
    assert not RD.control_k5([make_dump(world, good, k5=float("nan"))])["pass"]


def test_k6_same_windows_pass_other_order_and_missing_go_red(world, good, tmp_path):
    d = make_dump(world, good)
    assert RD.control_k6([d], ref_file(world, tmp_path))["pass"]
    n = len(world["eid"])
    assert not RD.control_k6([d], ref_file(world, tmp_path, order=np.arange(n)[::-1]))["pass"]       # a window list of another order
    assert not RD.control_k6([d], ref_file(world, tmp_path, order=np.arange(n - 1)))["pass"]          # another count
    assert not RD.control_k6([d], None)["pass"]
    d2 = dict(d, fold=(d["fold"] + 1) % 3)
    assert not RD.control_k6([d2], ref_file(world, tmp_path))["pass"]                                 # other folds


# ------------------------------------------------------------------------------------------------ names, both outcomes
def read_pair_of_seeds(worlds, alpha, tmp_path, **kw):
    ds = [make_dump(w, run_compute(w, alpha), seed_tag=i) for i, w in enumerate(worlds)]
    for d in ds:
        d["k5_max_abs"] = np.float64(1e-6)
    return ds


@pytest.fixture(scope="module")
def worlds2(tmp_path_factory):
    t = tmp_path_factory.mktemp("w2")
    return [make_world(t, seed=0), make_world(t, seed=1)]


def test_name_tokens_imagined_when_informative_and_flat_when_not(worlds2, tmp_path):
    ref = ref_file(worlds2[0], tmp_path)
    for alpha, want in ((0.6, "TOKENS-IMAGINED"), (0.0, "TOKENS-FLAT")):
        ds = read_pair_of_seeds(worlds2, alpha, tmp_path)
        res, arrs = RD.read_dumps(ds, "G", [0, 1], B=300, ref_windows=ref)
        assert res["controls_pass"], [c for c in res["controls"] if not c["pass"]]
        assert res["name"] == want, (alpha, res["stats"]["D_mov"])


def test_no_name_when_a_control_fails_or_one_seed_only(worlds2, tmp_path):
    ds = read_pair_of_seeds(worlds2, 0.6, tmp_path)
    res, _ = RD.read_dumps(ds, "G", [0, 1], B=200, ref_windows=None)             # k6 not run
    assert not res["controls_pass"] and res["name"] is None and "k6" in res["name_withheld_reason"]
    ref = ref_file(worlds2[0], tmp_path)
    res1, _ = RD.read_dumps(ds[:1], "G", [0], B=200, ref_windows=ref)
    assert res1["name"] is None and "both seeds" in res1["name_rule"]
    res3, _ = RD.read_dumps(ds, "G", [0, 1], B=200, ref_windows=ref, mutate="k3_not_substituted")      # the deliberate regression of k3 blocks the name
    assert not res3["controls_pass"] and res3["name"] is None


def test_pair_names_both_outcomes_and_refuses_other_windows(worlds2, tmp_path):
    ref = ref_file(worlds2[0], tmp_path)
    good_ds = read_pair_of_seeds(worlds2, 0.7, tmp_path)
    flat_ds = read_pair_of_seeds(worlds2, 0.0, tmp_path)
    rg, ag = RD.read_dumps(flat_ds, "G", [0, 1], B=300, ref_windows=ref)
    rr, ar = RD.read_dumps(good_ds, "RR", [0, 1], B=300, ref_windows=ref)
    sa, sb = _npz(tmp_path, "a", ar), _npz(tmp_path, "b", ag)
    p = RD.pair(sa, sb, rr, rg, "RR", "G", B=300)
    assert p["name"] == "RR-IMAGINES-TOKENS-BETTER" and p["diff"]["D_mov"]["pooled_ci"][0] > 0
    q = RD.pair(sb, sa, rg, rr, "G", "RR", B=300)                              # the other way round: nothing named
    assert q["name"] is None and q["diff"]["D_mov"]["pooled"] < 0
    r_same = RD.pair(_npz(tmp_path, "c", ag), _npz(tmp_path, "d", ag), rg, rg, "RR", "G", B=300)
    assert r_same["name"] is None                                              # no difference, nothing named
    ar_bad = dict(ar); ar_bad["windows_sha"] = np.array("other")
    with pytest.raises(SystemExit):
        RD.pair(_npz(tmp_path, "e", ar_bad), sb, rr, rg, "RR", "G", B=100)
    rg_bad = dict(rg, controls_pass=False)
    assert RD.pair(sa, sb, rr, rg_bad, "RR", "G", B=100)["name"] is None


def _npz(tmp_path, nm, arrs):
    p = str(tmp_path / f"{nm}.npz")
    np.savez(p, **arrs)
    return np.load(p, allow_pickle=True)


def test_verdict_text_rules(worlds2, tmp_path):
    ds = read_pair_of_seeds(worlds2, 0.6, tmp_path)
    res, _ = RD.read_dumps(ds, "G", [0, 1], B=200, ref_windows=ref_file(worlds2[0], tmp_path))
    txt = RD.verdict(res)
    assert txt.index("## Controls") < txt.index("## Statistic D")
    assert "STRATEGIC: PAUSED (PI item 25)" in txt and "T0" in txt
    assert txt.count("driving performance") == txt.count("never driving performance")


# ================================================================================================ window selection (card section 2)
import inspect  # noqa: E402
import lv2_select as SEL  # noqa: E402


def fake_pack(n_ep=14, per=3, seed=0):
    rng = np.random.default_rng(seed)
    eid, g0, V, G = [], [], [], []
    for e in range(n_ep):
        for k in range(per):
            eid.append(f"e{e:02d}"); g0.append(4 + 4 * k)
            V.append(np.full(16, 8.0))
            g = np.full(15, 20.0 + rng.normal() * 5)
            g[14] = g[4] + rng.normal() * 12
            G.append(g)
    return np.array(eid), np.array(g0), np.array(V), np.array(G)


def test_lead_rule_three_and_three_one_per_episode_ties_by_index():
    eid, g0, V, G = fake_pack()
    G[:, 4] = 20.0
    G[:, 14] = 20.0                       # everything ties at 0 change
    G[7, 14] = 5.0; G[8, 14] = 6.0; G[9, 14] = 4.0                      # windows 7, 8 lie in episode 2, window 9 in episode 3
    G[30, 14] = 40.0; G[31, 14] = 41.0; G[40, 14] = 39.0
    lw = SEL.lead_windows(eid, g0, V, G)
    assert [w["group"] for w in lw] == ["lead_gap_closing"] * 3 + ["lead_gap_opening"] * 3
    assert len({w["eid"] for w in lw}) == 6                                      # at most one per episode
    assert lw[0]["window"] == 9 and lw[0]["change_m"] == -16.0               # the most negative first
    assert lw[1]["window"] == 7 and lw[1]["change_m"] == -15.0               # window 8 (same episode as 7, less negative) is skipped
    assert lw[2]["window"] == 0                                              # the remaining ties: the lower window index
    G2 = G.copy(); G2[:, 14] = 20.0
    lw2 = SEL.lead_windows(eid, g0, V, G2)
    assert [w["window"] for w in lw2[:3]] == [0, 3, 6]                        # all tied: lowest indices, one per episode


def test_lead_rule_filters_gap_speed_and_future_label():
    eid, g0, V, G = fake_pack()
    G[:, 4] = 20.0; G[:, 14] = 20.0
    G[0, 14] = 0.0; G[0, 4] = 60.0       # the best closing window, but its gap at t is above 50 m: out
    G[1, 14] = 0.0; V[1, 4] = 4.0        # ego speed below 5 m/s: out
    G[2, 14] = np.nan                    # no label at t + 2.0 s: out
    G[3, 14] = 1.0                       # eligible
    lw = SEL.lead_windows(eid, g0, V, G)
    assert lw[0]["window"] == 3


def assert_labels_only(fn):
    allowed = {"eid", "g0", "V", "G15", "panel"}
    extra = set(inspect.signature(fn).parameters) - allowed
    assert not extra, f"selection takes more than labels: {extra}"


def test_selection_takes_labels_only():
    assert_labels_only(SEL.select_windows)
    assert list(inspect.signature(SEL.select_windows).parameters) == ["eid", "g0", "V", "G15", "panel"]
    src = open(os.path.join(HERE, "lv2_select.py"), encoding="utf-8").read()
    for bad in ("err_imag", "err_pres", "roll_logged", "pred_2s", "imag_dump", "lv2_pod_tokens"):
        assert bad not in src, bad
    ap_src = inspect.getsource(SEL.main)
    assert "--dump" not in ap_src and "--tok" not in ap_src


def test_selection_mutation_a_model_input_goes_red():
    def leaky(eid, g0, V, G15, panel, roll_logged):          # a selection that is given a model output
        return []
    with pytest.raises(AssertionError):
        assert_labels_only(leaky)


def test_selection_on_the_real_pack_matches_the_card_and_the_file():
    pack = "F:/TanitAD-scratch/v8proof/spatial/leadsup/read/eval_pack.npz"
    pw = "F:/TanitAD-scratch/v8proof/latvis/PANEL_WINDOWS.json"
    if not (os.path.exists(pack) and os.path.exists(pw)):
        pytest.skip("real pack not on this machine")
    P = np.load(pack, allow_pickle=True)
    r = SEL.select_windows(P["eid"].astype(str), P["g0"], P["V"], P["G15"], json.load(open(pw, encoding="utf-8")))
    assert len(r["windows"]) == 18 and len(set(r["window_indices"])) == 18
    lead = [w for w in r["windows"] if w["group"].startswith("lead_gap_")]
    assert len(lead) == 6 and all(5 <= w["gap_t_m"] <= 50 and w["speed_t_mps"] >= 5 for w in lead)
    assert len({w["eid"] for w in lead}) == 6
    assert sum(w["group"] == "lead_gap_closing" for w in lead) == 3 and all(w["change_m"] < 0 for w in lead if w["group"] == "lead_gap_closing")
    wf = os.path.join(HERE, "WINDOWS_LV2.json")
    if os.path.exists(wf):
        assert json.load(open(wf, encoding="utf-8"))["window_indices"] == r["window_indices"]


# ================================================================================================ sheets
import lv2_sheet as SH  # noqa: E402


def synthetic_imag_dump(eid, g0, n=3, seed=0):
    rng = np.random.default_rng(seed)
    return dict(eid=np.array(eid[:n]), g0=np.array(g0[:n]), gap_t=rng.uniform(10, 40, n), gap_t1=rng.uniform(10, 40, n), gap_t2=rng.uniform(10, 40, n),
                present=rng.uniform(5, 40, (n, 3)).astype(np.float32), true1=rng.uniform(5, 40, (n, 3)).astype(np.float32), true2=rng.uniform(5, 40, (n, 3)).astype(np.float32),
                roll_logged=rng.uniform(5, 40, (n, 2, 3)).astype(np.float32), roll_off=rng.uniform(5, 40, (n, 4, 2, 3)).astype(np.float32), offsets=np.array([0.0, -2.0, -1.0, 1.0]))


def test_sheet_renders_all_levels_and_public_form_ignores_frames(world, tmp_path):
    res, recs, pan, Zim, cd = run_compute(world, 0.6, panel=(0, 1))
    tok = dict(eid=world["eid"], **res, **pan)
    imag = synthetic_imag_dump(world["eid"], world["g0"])
    fr = str(tmp_path / "frames_0.npz")
    np.savez(fr, frames=np.random.default_rng(1).integers(0, 255, (2, 64, 160, 3), dtype=np.uint8))
    win = {"window": 0, "group": "speed_change", "eid": str(world["eid"][0]), "g0": int(world["g0"][0])}
    tk = {"G_s0": tok, "RR_s0": tok}
    pub0 = SH.render_sheet(win, tk, {"LS_s0": imag}, None, str(tmp_path / "pub0.png"), False)
    pub1 = SH.render_sheet(win, tk, {"LS_s0": imag}, fr, str(tmp_path / "pub1.png"), False)
    priv_f = SH.render_sheet(win, tk, {"LS_s0": imag}, fr, str(tmp_path / "priv_f.png"), True)
    priv_n = SH.render_sheet(win, tk, {"LS_s0": imag}, None, str(tmp_path / "priv_n.png"), True)
    b = lambda p: open(p, "rb").read()
    assert b(pub0) == b(pub1)                                    # no camera content can reach the public form
    assert b(priv_f) != b(priv_n)                                # the check can go red: the private form does change with frames
    from PIL import Image
    assert Image.open(pub0).width == 1600 and Image.open(priv_f).height > Image.open(pub0).height


def test_sheet_missing_levels_are_empty_tiles_not_errors(world, tmp_path):
    win = {"window": 5, "group": "lead_gap_closing", "eid": "nope", "g0": 4}
    SH.render_sheet(win, {}, {}, None, str(tmp_path / "e1.png"), True)                                   # nothing given at all
    res, recs, pan, Zim, cd = run_compute(world, 0.6)
    tok = dict(eid=world["eid"], **res, **pan)                                                          # no colours stored (empty panel)
    SH.render_sheet(win, {"G_s0": tok}, {"LS_s0": synthetic_imag_dump(world["eid"], world["g0"])}, None, str(tmp_path / "e2.png"), False)
    assert os.path.getsize(str(tmp_path / "e2.png")) > 5000


def test_contact_sheet(tmp_path):
    from PIL import Image
    ps = []
    for i in range(4):
        p = str(tmp_path / f"s{i}.png"); Image.new("RGB", (1600, 1800), (200, 200, 200 - i * 10)).save(p); ps.append(p)
    out = SH.contact_sheet(ps, str(tmp_path / "c.png"), ncol=2, width=200)
    assert Image.open(out).size == (400, 450)


# ================================================================================================ end to end: pod script -> reader -> sheet, on a tiny random model
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    sys.path.insert(0, IMAG)
    import imag_dump as ID
    torch_, M, _ = ID.import_stack(TREE)
    sys.path.insert(0, READ_DIR)
    import make_tiny_ckpt as MT
    MT.TINY["d_feat"] = NF
    d = tmp_path_factory.mktemp("e2e")
    ck = MT.build(str(d / "tiny.pt"), seed=0, lead_sup="off", tree=TREE)
    cfg, model, st, _ = ID.load_model(torch_, M, ck, require_head=False)
    rng = np.random.default_rng(0)
    T, n_ep = 36, 9
    os.makedirs(d / "cache" / "eval" / "clips")
    names = [f"clip{e:02d}" for e in range(n_ep)]
    with open(d / "cache" / "eval" / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for nm in names:
            fh.write(json.dumps({"clip": nm, "n_frames": T}) + "\n")
    Zs, Ci, Ts = [], [], []
    for j, nm in enumerate(names):
        u = np.cumsum(rng.normal(size=(T, 4)) * 0.5, 0)
        tok = (np.einsum("tk,kcf->tcf", u, rng.normal(size=(4, 40, NF))) + rng.normal(size=(T, 40, NF)) * 0.5).astype(np.float32)
        x8 = torch_.as_tensor(tok).to(torch_.float8_e4m3fn)
        torch_.save({"pool4x10": x8}, str(d / "cache" / "eval" / "clips" / (nm + ".pt")))
        with torch_.no_grad():
            z = model.wm.z1(model._encode({"feats": x8.float()[None]}))[0]
        Zs.append(z.to(torch_.float16).numpy()); Ci.append(np.full(T, j, np.int32)); Ts.append(np.arange(T, dtype=np.int16))
    np.savez(d / "states.npz", Z=np.concatenate(Zs), clip_idx=np.concatenate(Ci), t=np.concatenate(Ts), clips=np.array(names))
    eid, g0, fold = [], [], []
    for e, nm in enumerate(names):
        for g in range(4, T - 11, 4):
            eid.append(nm); g0.append(g); fold.append(e % 3)
    n = len(eid)
    A = rng.normal(size=(n, 10, 2)).astype(np.float32) * 0.3
    np.savez(d / "windows_R1.npz", eid=np.array(eid), g0=np.array(g0), t=np.array(g0) + 11, w_fold=np.array(fold), act_scaled=A)
    np.savez(d / "pack.npz", eid=np.array(eid), g0=np.array(g0), fold=np.array(fold), V=np.full((n, 16), 8.0), G15=np.full((n, 15), np.nan), pres0=np.zeros(n))
    Z = ID.assemble_states(torch_, str(d / "states.npz"), np.array(eid), np.array(g0))
    R = ID.roll_states(torch_, st, Z[:, 4], torch_.as_tensor(A), ID.predictor_context(st))[:, 9].numpy()
    np.savez(d / "banked_ok.npz", eid=np.array(eid), g0=np.array(g0), pred_2s=R, z0=Z[:, 4].numpy(), true_2s=Z[:, 14].numpy())
    np.savez(d / "banked_shift.npz", eid=np.array(eid), g0=np.array(g0), pred_2s=R + 1e-2, z0=Z[:, 4].numpy(), true_2s=Z[:, 14].numpy())
    return dict(dir=d, ckpt=ck, n=n)


def run_pod(e2e, tag, banked):
    d = e2e["dir"]
    json.dump({"window_indices": [0, 4]}, open(d / "panel.json", "w", encoding="utf-8"))
    argv = ["--ckpt", e2e["ckpt"], "--tree", TREE, "--windows", str(d / "windows_R1.npz"), "--pack", str(d / "pack.npz"), "--states", str(d / "states.npz"), "--cache", str(d / "cache"),
            "--banked-roll", str(d / banked), "--panel-windows", str(d / "panel.json"), "--out", str(d / f"DUMP_LV2_{tag}.npz"), "--workdir", str(d / f"wd_{tag}"), "--ncomp", "6", "--threads", "2", "--chunk", "32"]
    assert PT.main(argv) == 0
    return str(d / f"DUMP_LV2_{tag}.npz")


def test_end_to_end_pod_read_sheet(e2e):
    d = e2e["dir"]
    p_ok = run_pod(e2e, "ok", "banked_ok.npz")
    dump = RD.load_dump(p_ok)
    assert dump["k5_max_abs"] <= RD.TOL_K5 and dump["meta"]["k5"]["max_abs_z0"] == 0.0 and dump["meta"]["k5"]["max_abs_true_2s"] == 0.0     # k5, and the state assembly equals the banked states
    assert all(v["ok"] for v in dump["meta"]["facts"].values())
    assert len(dump["eid"]) == e2e["n"] and np.isfinite(dump["err_imag"]).all() and dump["col_true"].shape == (2, 2, 40, 3)
    out = d / "read_ok"
    assert RD.main(["read", "--dump", p_ok, p_ok, "--arm", "G", "--seeds", "0,1", "--out-dir", str(out), "--ref-windows", str(d / "windows_R1.npz"), "--B", "200"]) == 0
    r = json.load(open(out / "RESULT_LV2_G.json", encoding="utf-8"))
    byname = {c["name"]: c["pass"] for c in r["controls"]}
    assert byname["k5"] and byname["k6"] and byname["k3"]
    assert (r["name"] is None) == (not r["controls_pass"])
    assert (out / "VERDICT_LV2_G.md").exists() and (out / "STATS_LV2_G.npz").exists()
    # the shifted banked rollout (k5 deliberate regression, end to end) goes red and withholds the name
    p_bad = run_pod(e2e, "shift", "banked_shift.npz")
    RD.main(["read", "--dump", p_bad, p_bad, "--arm", "GX", "--seeds", "0,1", "--out-dir", str(out), "--ref-windows", str(d / "windows_R1.npz"), "--B", "100"])
    rb = json.load(open(out / "RESULT_LV2_GX.json", encoding="utf-8"))
    assert not {c["name"]: c["pass"] for c in rb["controls"]}["k5"] and rb["name"] is None
    # a window list of another order (k6 end to end)
    W = np.load(d / "windows_R1.npz")
    perm = np.arange(e2e["n"])[::-1]
    np.savez(d / "windows_rev.npz", eid=W["eid"][perm], g0=W["g0"][perm], w_fold=W["w_fold"][perm])
    RD.main(["read", "--dump", p_ok, p_ok, "--arm", "GY", "--seeds", "0,1", "--out-dir", str(out), "--ref-windows", str(d / "windows_rev.npz"), "--B", "100"])
    assert not {c["name"]: c["pass"] for c in json.load(open(out / "RESULT_LV2_GY.json", encoding="utf-8"))["controls"]}["k6"]
    # the sheet from these dumps
    wl = {"windows": [{"window": 0, "group": "speed_change", "eid": str(W["eid"][0]), "g0": int(W["g0"][0])}, {"window": 4, "group": "lead_gap_closing", "eid": str(W["eid"][4]), "g0": int(W["g0"][4])}]}
    json.dump(wl, open(d / "W.json", "w", encoding="utf-8"))
    assert SH.main(["--windows", str(d / "W.json"), "--out-dir", str(d / "sheetout"), "--tok", f"G_s0={p_ok}", "--tok", f"RR_s0={p_ok}", "--ncol", "2"]) == 0
    assert (d / "sheetout" / "sheets_public" / "sheet_0.png").exists() and (d / "sheetout" / "private" / "sheets" / "sheet_4.png").exists()
    assert (d / "sheetout" / "private" / "PRIVATE_DO_NOT_LAND.txt").exists() and (d / "sheetout" / "sheets_public" / "contact_sheet.png").exists()
