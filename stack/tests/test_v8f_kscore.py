"""PREREG_V8F_KSCORE: the instruments (stack/scripts/v8f_kscore.py + chain_v8f_kscore.sh), on a TINY random-init v8F-mini and synthetic windows. CPU only, small memory.

Run:  CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 python -I -m pytest -q stack/tests/test_v8f_kscore.py --basetemp=<OUT>/pt_tmp -p no:cacheprovider

What is pinned, and the mutation that must make it go RED:
  * the spec           the constants (F, B's banked ADE6, lr / epochs / batch, 4,512 clips) are the PREREG's own text; the functions copied from khyp.py ARE khyp's (ast-equal) and the
                       hypotheses / features reproduce khyp's ``enumerate_batch`` bit for bit
  * the scorer         0.2 M parameters (PREREG 0.1-0.3 M); inputs are vision-only (keys read subset of INFERENCE_INPUT_KEYS, no situation-classifier token in the code);
                       DELIBERATE REGRESSION: with the scene input zeroed the scene path is DEAD (gradient exactly 0.0, logits independent of its weights), alive with the real scene
  * CONTROL 1          the scorer forced to T's argmax IS SEL-ARGMAX: same picks, bit-identical plans, paired delta exactly 0.0 [0, 0].   MUTATION: the pick becomes argmin -> RED
  * CONTROL 2          a scorer trained on PERMUTED k* does not separate from SEL-CONST while the real scorer does (a world with signal).  MUTATION: identity permutation -> RED
  * SEL-CONST          chosen on the TRAIN windows, per seed.                                                                          MUTATION: chosen on the eval windows -> RED
  * the launch gate    G-DVB (every flag has an entry; hid / lr / PREREG pins), G-LIVE (CE finite, scorer gradient non-zero, B's parameters EXACTLY 0.0 with a positive control that
                       reads non-zero, the scene-zeroed control EXACTLY 0.0), G-INPUTS; fit refuses a missing / tampered token.
                       MUTATIONS: delete the scene input -> gate FAILS;  unfreeze B (a grad-enabled featuriser) -> gate FAILS
  * the read           the PREREG outcome table over ALL 256 (seed x bar) states against an independent oracle; the states it does not map are refused (rc 2).
                       MUTATION: two outcomes swapped -> RED
  * the chain          bash -n, the --dry-run prints the exact command of every stage in order, the guards and markers are in the text
  * end to end         hyps (2 seeds, resumable shards) -> gate -> fit (+ permuted) -> eval -> read on a synthetic world.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):      # BEFORE numpy: OpenBLAS reserves commit per logical CPU at import
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import v8f_kscore as K  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

torch.set_num_threads(1)
PK = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
KHYP_P = PK / "raw" / "2026-10-25-khyp" / "khyp.py"
DKR_P = PK / "code" / "v8f_dk_read.py"
CHAIN = REPO / "stack" / "scripts" / "chain_v8f_kscore.sh"
PREREG = REPO / "Project Steering" / "PREREG_V8F_KSCORE.md"
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
FITARGS = ["--allow-nonprereg", "--allow-any-ckpt", "--commit", "c0ffee1", "--epochs", "2", "--batch", "32", "--hid", "16", "--probe-n", "16", "--log-every", "1"]
needs_khyp = pytest.mark.skipif(not KHYP_P.exists(), reason="khyp.py (PREREG_V8F_KHYP's harness) is not in this tree")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ a synthetic world, tiny models, helpers
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    w = D.make_synthetic_world(root, n_train=5, n_eval=3, T=60, seed=0)
    return {**w, "root": Path(root)}


def make_model(world, seed=0, scale=0.3):
    J = D.load_join(world["root"] / "join_train.npz")
    cfg = M.V8FMiniConfig(**TINY)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():                                         # code-sensitive O (zero-init head_o would start at P10), and a T with spread logits
        for p in list(model.head_o.parameters()) + list(model.e_code.parameters()) + list(model.h_lon.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return model.eval(), cfg


def write_ckpt(world, path, seed):
    model, cfg = make_model(world, seed=seed)
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}, path)
    return model


def common(world, run_dir):
    r = world["root"]
    return ["--run-dir", str(run_dir), "--train-cache", str(r), "--eval-cache", str(r), "--train-join", str(r / "join_train.npz"), "--eval-join", str(r / "join_eval.npz"),
            "--eval-grid", str(r / "eval_grid_windows.json"), "--eval-kin", str(r / "eval_kin.npz"), "--tracks", str(r / "native_tracks.pt"), "--kin-mode", "steer",
            "--exclude", "", "--threads", "1"]


def batches(world, split="train", n=40, bs=20):
    J = D.load_join(world["root"] / f"join_{split}.npz")
    store = D.ClipStore(world["root"], split, lru=4)
    wins = D.train_windows(store, J, world["names"][split]) if split == "train" else D.eval_grid_windows(world["root"] / "eval_grid_windows.json", J)
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    out = []
    for b0 in range(0, min(n, len(wins)), bs):
        out.append(D.collate(K.batch_items(ds, wins, b0, min(b0 + bs, len(wins), n))))
    return out


def run_hyps(world, run_dir, ck, seed, splits=("eval", "train")):
    for split in splits:
        argv = ["hyps", "--seed", str(seed), "--split", split, "--ckpt", str(ck), "--allow-any-ckpt", "--batch", "8", "--shard", "16", "--lru", "2"] + common(world, run_dir)
        assert K.main(argv) == 0


def fit_argv(world, run_dir, ck, seed, extra=()):
    return ["fit", "--seed", str(seed), "--ckpt", str(ck)] + FITARGS + list(extra) + common(world, run_dir)


def make_lead_block(world, path):
    r = world["root"]
    grid = json.load(open(r / "eval_grid_windows.json"))
    J = D.load_join(r / "join_eval.npz")
    X = D.eval_context(grid, r / "eval_kin.npz", J, kin_mode="steer")
    order = grid["order"]
    n = len(X["gt"])
    v0 = np.asarray(X["kin"][:, 0], np.float64)
    ts = 0.2 * np.arange(1, 31)
    leads = np.zeros((n, 30, 2))
    leads[:, :, 0] = 20.0 + 0.8 * v0[:, None] * ts[None]
    state = np.where(np.arange(n) % 3 != 0, "LEAD", "NO_LEAD")
    np.savez(path, clip_id=np.array([order[e] for e in X["eid"]]), frame=2 * np.asarray(X["t"], np.int64), ts_rel_s=ts, leads=leads, lead_lens=np.full(n, 4.5), speeds=v0,
             state=state, dt_s=np.array([0.2]))
    return path


# ------------------------------------------------------------------------------------------------ the spec: constants are the PREREG's text; copied code IS khyp's
def test_constants_are_the_prereg_text():
    t = PREREG.read_text(encoding="utf-8") if PREREG.exists() else ""
    assert t, "PREREG_V8F_KSCORE.md must be in the tree"
    for s in ("0.1207", "2.1753 / 2.0546", "lr 1e-3", "3 epochs", "batch 256", "4,512 clips", "0.1–0.3 M", "SELECTION PAYS", "BIAS, NOT SELECTION", "NOT PREDICTABLE", "COSTS SAFETY", "COSTS DYNAMICS", "Amendment 1"):
        assert s in t, s
    assert K.F_PREREG == 0.1207 and K.BANKED_B_ADE6 == {0: 2.1753, 1: 2.0546}
    assert (K.FIT["lr"], K.FIT["epochs"], K.FIT["batch"]) == (1e-3, 3, 256) and K.PINNED["n_clips"] == 4512 and K.PINNED["n_windows"] == 306566
    assert (K.O_PAYS, K.O_BIAS, K.O_NOTPRED, K.O_COSTS, K.O_COSTS_DYN) == ("SELECTION PAYS", "BIAS, NOT SELECTION", "NOT PREDICTABLE", "COSTS SAFETY", "COSTS DYNAMICS")


def _norm(fn):
    """The function's AST without its docstring."""
    t = ast.parse(inspect.getsource(fn).strip() if not inspect.getsource(fn).startswith(" ") else inspect.cleandoc(inspect.getsource(fn)))
    f = t.body[0]
    if f.body and isinstance(f.body[0], ast.Expr) and isinstance(getattr(f.body[0], "value", None), ast.Constant) and isinstance(f.body[0].value.value, str):
        f.body = f.body[1:]
    return ast.dump(f)


@needs_khyp
def test_copied_functions_are_khyps_and_hypotheses_reproduce_khyp_bitwise(world):
    kh = _load("khyp_ref_mod", KHYP_P)
    for name in ("lon_onehot_code", "ade6_per_hyp", "pick", "permute_labels", "hypotheses_distinct", "arm_pw", "prow", "single"):
        assert _norm(getattr(K, name)) == _norm(getattr(kh, name)), f"{name} is not khyp's"
    assert (K.K, K.LON0) == (kh.K, kh.LON0)
    model, _ = make_model(world)
    b = batches(world, "train", 20, 20)[0]
    with torch.no_grad():
        ref = kh.enumerate_batch(model, b)
    mine = K.featurize_batch(model, b)
    for kk_mine, kk_ref in (("tau", "tau"), ("own", "own"), ("p30", "p30"), ("p_lon", "p_lon"), ("own_diag_maxabs", "own_diag_maxabs")):
        assert np.array_equal(K._np(mine[kk_mine]), np.asarray(ref[kk_ref])), kk_mine
    assert float(K._np(mine["own_diag_maxabs"]).max()) == 0.0, "forward_diag(T's own code) must equal forward_infer EXACTLY"
    assert K.hypotheses_distinct(K._np(mine["tau"])), "a code-sensitive model must give 7 distinct plans, or the scorer picks among copies"
    assert mine["scene"].shape == (20, 4, 32) and mine["ego"].shape == (20, 32)


def test_inputs_are_vision_and_ego_only(world):
    model, _ = make_model(world)
    b = batches(world, "train", 20, 20)[0]
    f = K.featurize_batch(model, b)
    assert set(f["keys_read"]) <= set(M.INFERENCE_INPUT_KEYS) and not any(k.startswith("lab") for k in f["keys_read"])
    assert K.M.scan_source_for_forbidden(Path(K.__file__).read_text(encoding="utf-8")) == [], "no situation-classifier token in the scorer's code"
    # a label must not change a single scorer input: poison every label and the gt, the features are identical
    b2 = {k: (v if not isinstance(v, dict) else {kk: (vv * 0 + 1 if vv.dtype != torch.bool else ~vv) for kk, vv in v.items()}) for k, v in b.items()}
    b2["gt"] = b["gt"] + 100.0
    f2 = K.featurize_batch(model, b2)
    for kk in ("tau", "scene", "ego", "p_lon"):
        assert torch.equal(f[kk], f2[kk]), f"{kk} depends on a label"


# ------------------------------------------------------------------------------------------------ the scorer and its deliberate regression
def test_scorer_size_is_in_the_prereg_band():
    s = K.build_scorer(128, "learned")
    n = s.built()["n_params"]
    assert n == 205_825 and 100_000 <= n <= 300_000
    assert s.scene_proj.bias is None and s.ego_proj.bias is None


def _raw(n=24, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, K.K, K.N_KNOT, 2, generator=g) * 5.0, torch.softmax(torch.randn(n, K.K, generator=g), -1), torch.randn(n, 4, d, generator=g), torch.randn(n, d, generator=g))


def test_scene_path_is_dead_when_the_scene_is_zeroed_and_alive_otherwise():
    torch.manual_seed(0)
    s = K.build_scorer(16, "learned", d=16, hist=4)
    tau, p, sc, ego = _raw()
    y = torch.randint(0, K.K, (24,))
    tau_s, scene_s, ego_s = s.standardise(tau, sc, ego)
    # alive: the real scene reaches the loss
    s.zero_grad()
    torch.nn.functional.cross_entropy(s.forward_std(tau_s, p, scene_s, ego_s), y).backward()
    assert float(s.scene_proj.weight.grad.abs().sum()) > 0.0
    # DEAD: the scene zeroed -> exactly 0.0 gradient on the scene path, every other path alive
    s.zero_grad()
    torch.nn.functional.cross_entropy(s.forward_std(tau_s, p, torch.zeros_like(scene_s), ego_s), y).backward()
    assert float(s.scene_proj.weight.grad.abs().sum()) == 0.0
    assert all(float(q.grad.abs().sum()) > 0.0 for n, q in s.named_parameters() if n != "scene_proj.weight")
    # and the logits do not depend on the scene weights at all
    with torch.no_grad():
        a0 = s.forward_std(tau_s, p, torch.zeros_like(scene_s), ego_s)
        s.scene_proj.weight.add_(torch.randn_like(s.scene_proj.weight))
        a1 = s.forward_std(tau_s, p, torch.zeros_like(scene_s), ego_s)
        a2 = s.forward_std(tau_s, p, scene_s, ego_s)
    assert torch.equal(a0, a1) and not torch.equal(a1, a2)


# ------------------------------------------------------------------------------------------------ a world with signal: controls 1 and 2, SEL-CONST
def signal_world(n_ep=40, per=40, d=16, seed=0, hid=16):
    """Hypothesis k's plan sits (k - 5) * 4 m to the left of a base path; the GT is the best hypothesis' plan; the best k is a function of the SCENE (T's softmax is uninformative)."""
    rng = np.random.default_rng(seed)
    N = n_ep * per
    scene = rng.normal(0, 1, (N, 4, d)).astype(np.float32)
    ego = rng.normal(0, 1, (N, d)).astype(np.float32)
    w = rng.normal(0, 1, (4 * d, K.K))
    bias = np.zeros(K.K)
    bias[5] = 3.6                                                           # a skewed k* like the real one (a majority class): a collapsed scorer is then SEL-CONST
    k_true = (0.5 * scene.reshape(N, -1) @ w + 0.8 * rng.normal(0, 1, (N, K.K)) + bias).argmax(1)
    base = np.stack([np.arange(1, 31) * 1.0, np.zeros(30)], 1)
    tau = np.repeat(base[None, None], N, 0).repeat(K.K, 1).astype(np.float32)
    tau[:, :, :, 1] += ((np.arange(K.K) - 5) * 4.0)[None, :, None]                          # class 5 (the majority) is the CENTRAL hypothesis: the best constant is the majority class
    gt = tau[np.arange(N), k_true] + rng.normal(0, 0.05, (N, 30, 2)).astype(np.float32)
    p_lon = rng.dirichlet(np.ones(K.K), N).astype(np.float32)
    eid = np.repeat(np.arange(n_ep), per)
    return dict(tau=tau, p_lon=p_lon, scene=scene.reshape(N, -1), ego=ego, gt=gt, eid=eid, k_true=k_true)


def fit_arrays(arrs, labels, seed, tmp_path, d=16, hid=16, epochs=3):
    """The training loop on arrays (the token is a stand-in: the loop itself needs a VERIFIED token; the real token path is tested end to end below)."""
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    a = K.build_parser().parse_args(["fit", "--run-dir", str(tmp_path), "--seed", str(seed), "--allow-nonprereg", "--epochs", str(epochs), "--batch", "64", "--hid", str(hid), "--log-every", "1000"])
    tok = tmp_path / "tok.json"
    K.jdump({"gate": "PASS"}, tok)
    a.token = str(tok)
    torch.manual_seed(seed)
    s = K.build_scorer(hid, "learned", d=d, hist=4)
    st = K.Stats(4, d)
    st.add(arrs["tau"], arrs["scene"], arrs["ego"])
    s.set_stats(st.result())
    opt = torch.optim.AdamW(s.parameters(), lr=a.lr, weight_decay=a.wd)
    N = len(labels)
    plan = {"epochs": epochs, "batch": 64, "steps_per_epoch": -(-N // 64), "steps_total": epochs * -(-N // 64)}
    out = tmp_path / f"fit{seed}"
    out.mkdir(parents=True, exist_ok=True)
    kst = K.kstar_of(K.ade6_per_hyp(arrs["tau"], arrs["gt"]))
    K.train_loop(a, s, opt, {k: arrs[k] for k in ("tau", "p_lon", "scene", "ego")}, labels, kst, plan, out, {"gate": "PASS"}, 0.0, torch.device("cpu"))
    return s.eval()


def paired(a_plan, b_plan, X_gt, eid, nb=300):
    from taniteval import v8f_beta_eval as BE
    return BE.paired_row(K.arm_pw(a_plan, X_gt, BE)["ade6"], K.arm_pw(b_plan, X_gt, BE)["ade6"], eid, nb)


def read_world(tmp_path_factory):
    arrs = signal_world()
    ktr = K.kstar_of(K.ade6_per_hyp(arrs["tau"], arrs["gt"]))
    tmp = tmp_path_factory.mktemp("sig")
    real = fit_arrays(arrs, ktr, 0, tmp)
    perm = fit_arrays(arrs, K.permute_labels(ktr, 0), 0, tmp / "p")
    return arrs, ktr, real, perm


@pytest.fixture(scope="module")
def sig(tmp_path_factory):
    # TRAIN on the first 30 episodes, EVALUATE on the last 10 (disjoint episodes)
    arrs = signal_world(n_ep=100, per=40, seed=1)
    tr = {k: v[:2400] for k, v in arrs.items()}
    ev = {k: v[2400:] for k, v in arrs.items()}
    ktr = K.kstar_of(K.ade6_per_hyp(tr["tau"], tr["gt"]))
    tmp = tmp_path_factory.mktemp("sig")
    return {"tr": tr, "ev": ev, "ktr": ktr, "tmp": tmp, "real": fit_arrays(tr, ktr, 0, tmp), "perm": fit_arrays(tr, K.permute_labels(ktr, 0), 0, tmp / "p")}


def _plans(sig, perm_scorer=None):
    ev = sig["ev"]
    raw = {k: ev[k] for k in ("tau", "p_lon", "scene", "ego")}
    k_sc = K.picks_from_logits(K.scorer_logits(sig["real"], raw))
    k_pm = K.picks_from_logits(K.scorer_logits(perm_scorer or sig["perm"], raw))
    forced = K.build_scorer(16, "forced_argmax", d=16, hist=4).eval()
    k_fo = K.picks_from_logits(K.scorer_logits(forced, raw))
    const = int(np.argmin(K.ade6_per_hyp(sig["tr"]["tau"], sig["tr"]["gt"]).mean(0)))
    return {"KSCORE": K.pick(ev["tau"], k_sc), "PERM": K.pick(ev["tau"], k_pm), "FORCED": K.pick(ev["tau"], k_fo), "ARGMAX": K.pick(ev["tau"], ev["p_lon"].argmax(1)),
            "CONST": K.pick(ev["tau"], np.full(len(ev["tau"]), const)), "k_fo": k_fo}


def control1(sig):
    P = _plans(sig)
    gt, eid = sig["ev"]["gt"].astype(np.float64), sig["ev"]["eid"]
    r = paired(P["FORCED"], P["ARGMAX"], gt, eid)
    return bool(np.array_equal(P["FORCED"], P["ARGMAX"]) and r["mean"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0)


def control2(sig, perm_scorer=None):
    P = _plans(sig, perm_scorer)
    r = paired(P["PERM"], P["CONST"], sig["ev"]["gt"].astype(np.float64), sig["ev"]["eid"])
    return not r["separated"]


def test_control1_forced_scorer_is_sel_argmax_exactly(sig, monkeypatch):
    assert control1(sig) is True
    monkeypatch.setattr(K, "picks_from_logits", lambda lg: np.asarray(lg).argmin(1))       # MUTATION: the pick becomes argmin -> the forced scorer is no longer SEL-ARGMAX
    P = _plans(sig)                                                                     # (SEL-ARGMAX itself is an independent expression, so it does not move)
    assert not np.array_equal(P["FORCED"], P["ARGMAX"]), "MUTATION must be visible: the forced scorer picks the LEAST likely class"
    assert control1(sig) is False


def test_the_scorer_reads_the_scene_and_the_permuted_scorer_does_not(sig):
    P = _plans(sig)
    gt, eid = sig["ev"]["gt"].astype(np.float64), sig["ev"]["eid"]
    sel = paired(P["KSCORE"], P["CONST"], gt, eid)
    assert sel["mean"] < 0 and sel["separated"], f"the signal world has a scene-readable best hypothesis: {sel}"           # selection beats the best constant
    arg = paired(P["KSCORE"], P["ARGMAX"], gt, eid)
    assert arg["mean"] < 0 and arg["separated"], "T's softmax is uninformative here, the scene is the signal"
    assert control2(sig) is True, "CONTROL 2: a scorer trained on permuted k* must not separate from SEL-CONST"


def test_mutation_identity_permutation_turns_control2_red(sig, tmp_path, monkeypatch):
    monkeypatch.setattr(K, "permute_labels", lambda y, seed=0: np.asarray(y))              # MUTATION: the 'permutation' is the identity, so the control scorer sees the TRUE labels
    leaky = fit_arrays(sig["tr"], K.permute_labels(sig["ktr"], 0), 0, tmp_path / "id")
    assert control2(sig, leaky) is False, "MUTATION must be visible: a scorer trained on the true labels separates from SEL-CONST"


def test_sel_const_is_chosen_on_the_train_windows(sig):
    tr_means = K.ade6_per_hyp(sig["tr"]["tau"], sig["tr"]["gt"]).mean(0)
    ev_means = K.ade6_per_hyp(sig["ev"]["tau"], sig["ev"]["gt"]).mean(0)
    chosen = int(np.argmin(tr_means))
    P = _plans(sig)
    assert np.array_equal(P["CONST"], K.pick(sig["ev"]["tau"], np.full(len(sig["ev"]["tau"]), chosen)))
    # the mutation: choosing the class on the EVAL windows can only look better (it reads the answer) -- that is the leak the rule forbids
    mutated = K.pick(sig["ev"]["tau"], np.full(len(sig["ev"]["tau"]), int(np.argmin(ev_means))))
    d_ok = np.linalg.norm(P["CONST"] - sig["ev"]["gt"], axis=-1).mean()
    d_leak = np.linalg.norm(mutated - sig["ev"]["gt"], axis=-1).mean()
    assert d_leak <= d_ok + 1e-12


# ------------------------------------------------------------------------------------------------ the outcome table over every state
def _oracle(pb0, pb1, ps0, ps1, lon0, lon1, cl0, cl1):
    """The PREREG table, written out independently of K.decide, with LITERAL labels (a swapped constant in K cannot move this)."""
    PB, PS, LON, CL = pb0 and pb1, ps0 and ps1, lon0 and lon1, cl0 and cl1
    if not PB:
        return "NOT PREDICTABLE"                                  # PRIMARY-B fails
    if not PS:
        return "BIAS, NOT SELECTION"                              # PRIMARY-B passes, PRIMARY-SEL fails
    if not CL:
        return "COSTS SAFETY"                                     # PRIMARY passes, CONSISTENCY or LEAD-VIOL separated-worse at either seed: takes PRECEDENCE (Amendment 1)
    if not LON:
        return "COSTS DYNAMICS"                                   # Amendment 1: PRIMARY passes, SECONDARY separated-worse on speed or accel MAE ONLY
    return "SELECTION PAYS"                                       # PRIMARY-B, PRIMARY-SEL and SECONDARY at both seeds


def mapping_mismatches():
    bad, n_unmapped = [], 0
    for m in range(256):
        bits = [bool(m >> i & 1) for i in range(8)]
        per = {0: {"pb": bits[0], "ps": bits[2], "sec_lon_ok": bits[4], "sec_cl_ok": bits[6]}, 1: {"pb": bits[1], "ps": bits[3], "sec_lon_ok": bits[5], "sec_cl_ok": bits[7]}}
        want = _oracle(*bits)
        try:
            got = K.decide(per)["outcome"]
        except K.Unmapped:
            got = None
        n_unmapped += got is None
        if got != want:
            bad.append((bits, got, want))
    return bad, n_unmapped


def test_decide_maps_all_256_states_to_exactly_one_outcome_with_the_amended_precedence():
    bad, n_unmapped = mapping_mismatches()
    assert bad == [], bad[:3]
    assert n_unmapped == 0, "Amendment 1 closed the one gap: every boolean state has a row"
    # exactly one outcome per state, and the census is the arithmetic of the table: PB fails 192; PB passes, PS fails 48; both pass (16 states over lon/cl): cl fails 12
    # (COSTS SAFETY, precedence), cl ok and lon fails 3 (COSTS DYNAMICS), all ok 1
    census = {}
    for m in range(256):
        bits = [bool(m >> i & 1) for i in range(8)]
        per = {0: {"pb": bits[0], "ps": bits[2], "sec_lon_ok": bits[4], "sec_cl_ok": bits[6]}, 1: {"pb": bits[1], "ps": bits[3], "sec_lon_ok": bits[5], "sec_cl_ok": bits[7]}}
        o = K.decide(per)["outcome"]
        census[o] = census.get(o, 0) + 1
    assert census == {"NOT PREDICTABLE": 192, "BIAS, NOT SELECTION": 48, "COSTS SAFETY": 12, "COSTS DYNAMICS": 3, "SELECTION PAYS": 1}, census
    # the precedence in one state: dynamics fails at seed 0 AND lead-viol fails at seed 1 -> COSTS SAFETY
    ok = {"pb": True, "ps": True, "sec_lon_ok": True, "sec_cl_ok": True}
    both = K.decide({0: dict(ok, sec_lon_ok=False), 1: dict(ok, sec_cl_ok=False)})
    assert both["outcome"] == "COSTS SAFETY" and K.decide({0: dict(ok, sec_lon_ok=False), 1: dict(ok)})["outcome"] == "COSTS DYNAMICS"
    assert K.decide({0: dict(ok), 1: dict(ok)})["committed_next_step_quoted"] == K.NEXT["SELECTION PAYS"]
    for r in ({"pb": None, "ps": True, "sec_lon_ok": True, "sec_cl_ok": True}, {"pb": True, "ps": True, "sec_lon_ok": True, "sec_cl_ok": None}):
        with pytest.raises(K.Unmapped):
            K.decide({0: r, 1: dict(r)})
    with pytest.raises(K.Unmapped):
        K.decide({0: {"pb": True, "ps": True, "sec_lon_ok": True, "sec_cl_ok": True}})                      # one seed only: every bar is 'at BOTH seeds'


def test_mutation_two_outcomes_swapped_goes_red(monkeypatch):
    monkeypatch.setattr(K, "O_BIAS", "NOT PREDICTABLE")
    monkeypatch.setattr(K, "O_NOTPRED", "BIAS, NOT SELECTION")
    bad, _ = mapping_mismatches()
    assert bad, "MUTATION must be visible: the table maps PRIMARY-B failing to BIAS"
    monkeypatch.undo()
    assert mapping_mismatches()[0] == []


def test_mutation_costs_safety_precedence_swapped_goes_red(monkeypatch):
    """Swap the two COSTS branches in the REAL source of ``decide`` (dynamics tested before safety): a state with both failing must then read COSTS DYNAMICS, and the 256-state oracle must see it."""
    import inspect
    src = inspect.getsource(K.decide)
    a = "    elif not cl:\n        o = O_COSTS "
    b = "    elif not lon:\n        o = O_COSTS_DYN "
    assert a in src and b in src and src.index(a) < src.index(b), "the precedence lines must be where the mutation expects them"
    i, j = src.index(a), src.index(b)
    k = src.index("    else:\n        o = O_PAYS")
    block_a, block_b = src[i:j], src[j:k]
    mutated = src[:i] + block_b + block_a + src[k:]
    assert mutated != src
    ns = dict(vars(K))
    exec(compile(mutated, "<mutated decide>", "exec"), ns)
    monkeypatch.setattr(K, "decide", ns["decide"])
    bad, _ = mapping_mismatches()
    assert bad, "MUTATION must be visible: COSTS SAFETY no longer takes precedence over COSTS DYNAMICS"
    assert all(w == "COSTS SAFETY" and g == "COSTS DYNAMICS" for _, g, w in bad), "only the both-fail states move"
    monkeypatch.undo()
    assert mapping_mismatches()[0] == []
    monkeypatch.setattr(K, "O_COSTS", "COSTS DYNAMICS")                      # swapping the two LABELS is caught as well
    monkeypatch.setattr(K, "O_COSTS_DYN", "COSTS SAFETY")
    assert mapping_mismatches()[0]


def test_every_outcome_quotes_the_prereg_next_step():
    t = PREREG.read_text(encoding="utf-8")
    for o, nxt in K.NEXT.items():
        assert o in t and nxt in t, (o, nxt)                                  # the committed next step is the PREREG's own text, verbatim
    assert "joins `forward_infer` as a K-hypothesis planner" in t and "class-balanced loss" in t and "OACC's line" in t and "reported; not deployed" in t
    assert "re-read on top of OACC's base once OACC exists" in t and "precedence: **COSTS SAFETY**" in t


# ------------------------------------------------------------------------------------------------ the launch gate, end to end on the tiny world
@pytest.fixture(scope="module")
def pipe(world, tmp_path_factory):
    base = tmp_path_factory.mktemp("kscore")
    cks = {s: base / f"ck{s}.pt" for s in (0, 1)}
    for s in (0, 1):
        write_ckpt(world, cks[s], s)
    run = base / "run"
    for s in (0, 1):
        run_hyps(world, run, cks[s], s)
    out = {"run": run, "cks": cks, "base": base}
    for s in (0, 1):
        for perm in (False, True):
            extra = ["--perm-labels"] if perm else []
            assert K.main(fit_argv(world, run, cks[s], s, extra + ["--gate-only"])) == 0          # the gate writes the token and exits
            assert K.main(fit_argv(world, run, cks[s], s, extra + ["--require-token"])) == 0       # training only against the token
    assert K.main(["eval", "--run-dir", str(run), "--seeds", "0", "1"]) == 0
    lb = make_lead_block(world, base / "lead_block.npz")
    out["rc_read"] = K.main(["read", "--smoke", "--no-banked", "--n-boot", "40", "--lead-block", str(lb), "--dk-reader", str(DKR_P)] + common(world, run))
    out["lead_block"] = lb
    return out


def test_shards_are_resumable_and_complete(world, pipe, tmp_path):
    run = pipe["run"]
    meta = json.load(open(run / "hyps" / "M2b_s0_train.meta.json"))
    sdir = run / "hyps" / "M2b_s0_train_shards"
    shards = sorted(sdir.glob("shard_*.npz"))
    assert meta["n_windows"] == 135 and len(shards) == meta["n_shards"] >= 2 and meta["own_diag_maxabs_over_windows"] == 0.0
    before = K.load_split(run, 0, "train")[0]
    shards[-1].unlink()                                                                    # a killed run: the last shard is gone, the meta is stale
    assert K.main(["hyps", "--seed", "0", "--split", "train", "--ckpt", str(pipe["cks"][0]), "--allow-any-ckpt", "--batch", "8", "--shard", "16", "--lru", "2"] + common(world, run)) == 0
    after = K.load_split(run, 0, "train")[0]
    assert all(np.array_equal(before[k], after[k]) for k in before)
    z = np.load(sdir / "shard_00000.npz")
    assert z["tau"].shape[1:] == (7, 30, 2) and z["scene"].dtype == np.float16 and z["scene"].shape[1:] == (4, 32) and z["ego"].shape[1:] == (32,)


def test_gate_passed_with_every_probe_alive(pipe):
    for d in ("s0", "s1", "s0_perm", "s1_perm"):
        sc = json.load(open(pipe["run"] / "fit" / d / "selfcheck.json"))
        tok = json.load(open(pipe["run"] / "fit" / d / "gate_token.json"))
        assert sc["gate"] == "PASS" and sc["declared_vs_built"] == "PASS" and sc["live"]["pass"] is True and sc["inputs_identity"]["pass"] is True
        lv = sc["live"]
        assert lv["B_grad_abs_sum"] == 0.0, "B is frozen: EXACTLY 0.0"
        assert lv["B_grad_positive_control_abs_sum"] > 0.0, "the probe can see a leak: a grad-enabled featuriser reaches B's parameters"
        assert lv["scene_zero_control_scene_proj_grad"] == 0.0 and lv["scene_real_scene_proj_grad"] > 0.0 and lv["B_state_unchanged"] is True
        assert lv["scorer_grad_on_B_inputs"]["scene_proj"] > 0.0 and not lv["dead_params"] and all(np.isfinite(lv["ce"]))
        assert sc["own_diag_maxabs"] == 0.0 and set(sc["keys_read"]) <= set(M.INFERENCE_INPUT_KEYS)
        assert tok["gate"] == "PASS" and tok["commit"] == "c0ffee1" and tok["selfcheck_sha256"] == K.sha256_file(pipe["run"] / "fit" / d / "selfcheck.json")
        assert tok["prereg_pinned"] is False and tok["perm_labels"] == d.endswith("perm") and tok["inputs"]["n_windows"] == 135
        sm = json.load(open(pipe["run"] / "fit" / d / "summary.json"))
        assert sm["done"] is True and sm["gate"] == "PASS" and sm["steps"] == 2 * -(-135 // 32) and sm["token_sha256"] == K.sha256_file(pipe["run"] / "fit" / d / "gate_token.json")


GUARD_P = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code" / "box2_ops" / "v8f_guard.sh"


@pytest.mark.skipif(not GUARD_P.exists(), reason="v8f_guard.sh (the box-2 guard) is not in this tree")
def test_the_box2_guard_can_read_our_fit_logs(pipe, tmp_path):
    """v8f_guard.sh's TIME rule parses ``train_log.jsonl`` ({step, sec}) and ``selfcheck.json`` ({steps_total}) of the run named in $V8F_CURRENT. Run ITS embedded reader on OUR
    files (our log must carry {step, sec}; a synthetic 21-record tail at step >= 1000 arms the rule, which waits for that): a rule that cannot parse our files is silently inert (it exits 0 on any exception)."""
    g = GUARD_P.read_text(encoding="utf-8")
    snip = g.split("<<'EOF'\n", 1)[1].split("\nEOF\n", 1)[0]
    fit = pipe["run"] / "fit" / "s0"
    rows = [json.loads(x) for x in open(fit / "train_log.jsonl", encoding="utf-8") if x.strip()]
    assert rows and all("step" in r and "sec" in r for r in rows), "the guard's reader needs step and sec in every record"
    sc = json.load(open(fit / "selfcheck.json"))
    assert sc["steps_total"] == sc["plan"]["steps_total"] > 0, "the guard reads steps_total at the TOP level"
    synth = tmp_path / "tail.jsonl"
    synth.write_text("".join(json.dumps({"step": 1000 + 50 * i, "sec": 100.0 + 5.0 * i}) + "\n" for i in range(21)), encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", "-c", snip, str(synth), str(fit / "selfcheck.json")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    hours, rate, step, total = r.stdout.split()
    assert float(rate) == pytest.approx(0.1) and int(step) == 2000 and int(total) == sc["steps_total"], r.stdout
    c = CHAIN.read_text(encoding="utf-8")
    assert 'echo "../v8f_kscore/fit/s$s$p" > $CUR' in c, "V8F_CURRENT must resolve (under D:/tanitad/runs/v8f_mini_full/) to our fit dir, or the guard's TIME rule reads nothing"


def test_fit_refuses_a_missing_or_tampered_token(world, pipe, tmp_path):
    run2 = tmp_path / "run2"
    run_hyps(world, run2, pipe["cks"][0], 0, ("train",))
    out = run2 / "fit" / "s0"
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0, ["--require-token"])) == 3          # no token: fit will not train
    assert not (out / "ckpt_last.pt").exists() and not (out / "summary.json").exists()
    with pytest.raises(K.Refused):
        K.verify_token(out / "nope.json", {"commit": "x"})
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0, ["--gate-only"])) == 0               # issue one
    tok = json.load(open(out / "gate_token.json"))
    for k, v in (("commit", "deadbeef"), ("argv_sha", "0" * 16), ("gate", "FAIL")):             # a token for another commit / argv / a FAILED gate
        bad = dict(tok, **{k: v})
        K.jdump(bad, out / "gate_token.json")
        assert K.main(fit_argv(world, run2, pipe["cks"][0], 0, ["--require-token"])) == 3, k
        assert not (out / "ckpt_last.pt").exists()
    K.jdump(tok, out / "gate_token.json")
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0, ["--require-token", "--lr", "0.002"])) == 3     # the same token under another lever (argv sha) is refused
    with pytest.raises(K.Refused):
        K.train_loop(K.build_parser().parse_args(["fit", "--run-dir", str(tmp_path), "--seed", "0"]), None, None, {}, np.zeros(1), np.zeros(1), {}, tmp_path, None, 0.0, torch.device("cpu"))


def test_mutation_delete_the_scene_input_makes_the_gate_fail(world, pipe, tmp_path, monkeypatch):
    run2 = tmp_path / "run2"
    run_hyps(world, run2, pipe["cks"][0], 0, ("train",))
    monkeypatch.setattr(K, "scene_at_now", lambda model, adapt, frame_ok: torch.zeros(adapt.shape[0], model.cfg.hist, model.cfg.d))
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0)) == 3
    sc = json.load(open(run2 / "fit" / "s0" / "selfcheck.json"))
    assert sc["gate"] == "FAIL" and not (run2 / "fit" / "s0" / "gate_token.json").exists() and not (run2 / "fit" / "s0" / "ckpt_last.pt").exists()
    assert sc["inputs_identity"]["pass"] is False or sc["live"]["scorer_grad_on_B_inputs"]["scene_proj"] == 0.0


def test_mutation_unfreeze_B_makes_the_gate_fail(world, pipe, tmp_path, monkeypatch):
    run2 = tmp_path / "run2"
    run_hyps(world, run2, pipe["cks"][0], 0, ("train",))
    monkeypatch.setattr(K, "featurize_batch", lambda model, b: K.featurize_t(model, b))          # MUTATION: the featuriser keeps B's autograd graph
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0)) == 3
    sc = json.load(open(run2 / "fit" / "s0" / "selfcheck.json"))
    assert sc["gate"] == "FAIL" and sc["live"]["pass"] is False and sc["live"]["B_grad_abs_sum"] > 0.0, "B's parameters received a gradient: the gate must read it"
    assert not (run2 / "fit" / "s0" / "gate_token.json").exists()


def test_gate_refuses_declared_vs_built_mismatches(world, pipe, tmp_path, monkeypatch):
    run2 = tmp_path / "run2"
    run_hyps(world, run2, pipe["cks"][0], 0, ("train",))
    real_build = K.build_scorer
    monkeypatch.setattr(K, "build_scorer", lambda hid, mode, d=256, hist=4: real_build(hid * 2, mode, d=d, hist=hist))      # built hid != declared hid
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0)) == 3
    assert any("hid: declared 16" in b for b in json.load(open(run2 / "fit" / "s0" / "selfcheck.json"))["declared_vs_built"])
    monkeypatch.undo()
    real_perm = K.permute_labels
    monkeypatch.setattr(K, "permute_labels", lambda y, seed=0: np.asarray(y))                      # --perm-labels declared, labels not permuted
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0, ["--perm-labels"])) == 3
    assert any("perm_labels declared" in b for b in json.load(open(run2 / "fit" / "s0_perm" / "selfcheck.json"))["declared_vs_built"])
    monkeypatch.undo()
    monkeypatch.setitem(K.FLAG_CHECKS, "zz_new_lever", "run")                                      # a check for a flag that does not exist
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0)) == 3
    monkeypatch.undo()
    monkeypatch.delitem(K.FLAG_CHECKS, "hid")                                                      # a flag with no declared-vs-built entry is refused
    assert K.main(fit_argv(world, run2, pipe["cks"][0], 0)) == 3
    assert any("flags without a declared-vs-built entry" in b and "hid" in b for b in json.load(open(run2 / "fit" / "s0" / "selfcheck.json"))["declared_vs_built"])
    assert real_perm is K.permute_labels or True


def test_prereg_pins_are_enforced_at_real_size():
    a = K.build_parser().parse_args(["fit", "--run-dir", "x", "--seed", "0"])
    s = K.build_scorer(128, "learned")
    opt = torch.optim.AdamW(s.parameters(), lr=a.lr, weight_decay=a.wd)
    y = np.arange(306566) % 7
    info = {"n_windows": 306566, "n_clips": 4512, "n_excluded": 60, "n_excluded_declared": 60, "excluded_in_train": 0, "kin_mode": "yaw", "ckpt_sha256": K.CKPT_SHA[0],
            "ckpt_file_sha256": K.CKPT_SHA[0], "n_probe": 256}
    plan = {"epochs": 3, "batch": 256, "steps_per_epoch": 1198, "steps_total": 3594}
    a.expect_clips, a.expect_windows = 4512, 306566
    assert K.declared_vs_built(a, s, opt, plan, info, y, y) == []
    a2 = K.build_parser().parse_args(["fit", "--run-dir", "x", "--seed", "0", "--lr", "0.002", "--expect-clips", "4512", "--expect-windows", "306566"])
    opt2 = torch.optim.AdamW(s.parameters(), lr=a2.lr, weight_decay=a2.wd)
    assert any("PREREG pin: lr" in b for b in K.declared_vs_built(a2, s, opt2, plan, info, y, y))
    assert any("PREREG pin: clips" in b for b in K.declared_vs_built(a, s, opt, plan, dict(info, n_clips=4000), y, y))
    assert any("PREREG pin: the checkpoint" in b for b in K.declared_vs_built(a, s, opt, plan, dict(info, ckpt_sha256="0" * 64, ckpt_file_sha256="0" * 64), y, y))
    assert any("optimiser holds" in b for b in K.declared_vs_built(a, s, torch.optim.AdamW(list(s.parameters())[:3], lr=a.lr, weight_decay=a.wd), plan, info, y, y))


# ------------------------------------------------------------------------------------------------ eval + read, end to end
def test_eval_picks_and_sel_const_come_from_the_train_windows(pipe):
    for s in (0, 1):
        sel = json.load(open(pipe["run"] / "fit" / f"s{s}" / "sel_const.json"))
        z = np.load(pipe["run"] / "eval" / f"kscore_s{s}.npz")
        tr, _ = K.load_split(pipe["run"], s, "train", ("tau", "gt"))
        assert sel["chosen_on"] == "train" and sel["n_windows"] == 135
        assert sel["chosen_class"] == int(np.argmin(K.ade6_per_hyp(tr["tau"], tr["gt"]).mean(0)))
        assert (z["k_const"] == sel["chosen_class"]).all() and z["logits"].shape == (21, 7)
        ev, _ = K.load_split(pipe["run"], s, "eval")
        assert np.array_equal(K.picks_from_logits(z["logits_forced"]), ev["p_lon"].argmax(1)), "the forced scorer IS T's argmax"


def test_read_end_to_end_applies_the_table_and_controls_pass(pipe):
    assert pipe["rc_read"] == 0
    R = json.load(open(pipe["run"] / "READ.json"))
    assert R["decision"]["outcome"] in K.NEXT
    assert R["decision"]["committed_next_step_quoted"] == K.NEXT[R["decision"]["outcome"]]
    for s in ("s0", "s1"):
        ps = R["per_seed"][s]
        c = ps["controls"]
        assert c["control1_forced_is_sel_argmax"]["pass"] is True and c["control4_oracle_le_vertex_selectors"]["pass"] is True and c["harness_identity_maxabs_m"] == 0.0
        assert c["hypotheses_distinct"] is True and "control2_perm_scorer_vs_sel_const" in c
        for k in ("pb", "ps", "sec_lon_ok", "sec_cl_ok"):
            assert ps["seed_row"][k] in (True, False)
        assert np.isfinite(ps["PRIMARY_B"]["point_unrounded"]) and np.isfinite(ps["PRIMARY_SEL"]["point_unrounded"])
        assert ps["SECONDARY"]["lead_viol_at_risk"]["status"] == "OK" and ps["stack_HS"]["status"] == "NOT-INSTRUMENTED"
        assert ps["selector_accuracy"]["n_eval"] == 21 and sum(ps["selector_accuracy"]["kstar_eval_counts"]) == 21
    md = (pipe["run"] / "KSCORE_RESULT.md").read_text(encoding="utf-8")
    assert md.startswith("# KSCORE_RESULT") and "PAUSED" in md and "Outcome (the PREREG table applied verbatim)" in md


def test_read_refuses_with_rc2_on_a_stop_control_a_missing_block_and_an_unmapped_state(world, pipe, tmp_path, monkeypatch):
    run = pipe["run"]
    argv = ["read", "--smoke", "--no-banked", "--n-boot", "20", "--dk-reader", str(DKR_P)] + common(world, run)
    assert K.main(argv + ["--lead-block", str(tmp_path / "missing.npz")]) == 2                    # LEAD-VIOL cannot be read: the table maps no state without it
    assert json.load(open(run / "READ.json"))["status"] == "STOPPED"
    lb = str(pipe["lead_block"])
    monkeypatch.setattr(K, "decide", lambda per: (_ for _ in ()).throw(K.Unmapped("test: a state with no row")))
    assert K.main(argv + ["--lead-block", lb]) == 2
    assert json.load(open(run / "READ.json"))["status"] == "UNMAPPED"
    monkeypatch.undo()
    # control 3: B's own-code ADE6 must round to the PREREG's banked 4 dp -> on a random-init B it does not
    assert K.main(["read", "--smoke", "--n-boot", "20", "--lead-block", lb, "--dk-reader", str(DKR_P)] + common(world, run)) == 2
    assert "CONTROL 3" in json.dumps(json.load(open(run / "READ.json"))["stops"])
    # control 1 stop: tamper the stored forced logits (the forced scorer no longer reproduces SEL-ARGMAX)
    z = dict(np.load(run / "eval" / "kscore_s0.npz"))
    good = z["logits_forced"].copy()
    z["logits_forced"] = -good
    np.savez(run / "eval" / "kscore_s0.npz", **z)
    assert K.main(argv + ["--lead-block", lb]) == 2
    assert "CONTROL 1" in json.dumps(json.load(open(run / "READ.json"))["stops"])
    z["logits_forced"] = good
    np.savez(run / "eval" / "kscore_s0.npz", **z)
    assert K.main(argv + ["--lead-block", lb]) == 0


# ------------------------------------------------------------------------------------------------ the box-2 chain
def _bash():
    """A GNU bash that can run the chain. On Windows a bare ``bash`` may be System32's WSL launcher (rc != 0, no distro), so ask ``shutil.which`` and the Git-for-Windows
    install first and accept only a binary whose ``--version`` says GNU bash."""
    import shutil
    cands = [os.environ.get("KSCORE_BASH"), shutil.which("bash"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe", "bash"]
    for c in cands:
        if not c:
            continue
        try:
            r = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and "GNU bash" in r.stdout:
                return c
        except Exception:
            pass
    return None


def test_chain_syntax_and_dry_run_prints_the_exact_stage_commands(tmp_path):
    bash = _bash()
    if bash is None:
        pytest.skip("bash is needed to run the chain's --dry-run")
    r = subprocess.run([bash, "-n", str(CHAIN)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, KSCORE_PYX=sys.executable, KSCORE_COMMIT_DRY="abc123")
    r = subprocess.run([bash, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    stages = ["hyps_eval_0", "hyps_train_0", "hyps_eval_1", "hyps_train_1", "gate_s0", "gate_s0_perm", "gate_s1", "gate_s1_perm", "fit_s0", "fit_s0_perm", "fit_s1", "fit_s1_perm", "eval", "read"]
    heads = [x.split(" ")[1] for x in out if x.startswith("== ") and x.split(" ")[1] in stages]
    assert heads == stages, heads
    cmd = {}
    for i, x in enumerate(out):
        if x.startswith("== ") and x.split(" ")[1] in stages:
            cmd[x.split(" ")[1]] = out[i + 1]
    want = {"hyps_train_0": ["hyps", "--seed", "0", "--split", "train"], "gate_s1_perm": ["fit", "--seed", "1", "--perm-labels", "--gate-only", "--commit", "abc123"],
            "fit_s0": ["fit", "--seed", "0", "--require-token", "--commit", "abc123"], "eval": ["eval", "--seeds", "0", "1"], "read": ["read"]}
    for st, args in want.items():
        assert K.box2_cmdline(args) == cmd[st], st                     # the printed line IS the line the job embeds
    for st in stages:
        assert cmd[st].startswith('"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\v8f_kscore.py"')
    assert "--gate-only" in cmd["gate_s0"] and "--require-token" not in cmd["gate_s0"] and "--require-token" in cmd["fit_s0"] and "--gate-only" not in cmd["fit_s0"]
    assert "--device cuda" in cmd["hyps_train_1"] and "--exclude 540:600" in cmd["hyps_train_1"] and "--expect-windows 306566" in cmd["fit_s1"] and "--expect-clips 4512" in cmd["gate_s0"]
    assert "--allow-nonprereg" not in r.stdout and "--allow-any-ckpt" not in r.stdout, "the chain never passes a TEST-ONLY flag"
    assert "wait for <logs>/chain_v8f_flat.out: ^ZZV8FMINI-DONE" in r.stdout
    for s in (0, 1):
        assert f"M2b_s{s} D:/tanitad/runs/v8f_mini_full/M2b_s{s}/ckpt_last.pt {K.CKPT_SHA[s]}" in r.stdout
    pos = {x.split(" ")[1]: n for n, x in enumerate(out) if x.startswith("== ") and x.split(" ")[1] in stages}
    assert pos["hyps_train_1"] < pos["gate_s0"] and all(pos[f"gate_{t}"] < pos[f"fit_{t}"] for t in ("s0", "s0_perm", "s1", "s1_perm")), "all hypotheses exist before the first gate; every gate before its fit"
    assert r.stdout.count("assert-gate") >= 4


def test_chain_text_has_the_load_bearing_lines():
    c = CHAIN.read_text(encoding="utf-8")
    for s in ("( set -C; echo", "BOX2_BUSY.txt", 'grep -a -q "^$AFTER_DONE"', "ZZV8FMINI-STOP", "ZZV8FMINI-DONE", "chain_v8f_kscore.out", "v8f_guard.sh", "assert-gate", "TIP_V8FKSCORE.txt",
              "does not know v8f_kscore", "ckptsha_b", "git -C $MIRROR show \"$T:stack/scripts/v8f_kscore.py\" | grep -q '^FLAG_CHECKS = {'", "waitjob.sh", "launch.ps1", "stop \"gate s$s$p assertions\""):
        assert s in c, s
    assert "rm -rf" not in c and "--allow-nonprereg" not in c and "--allow-any-ckpt" not in c
    assert c.index("kjob \"gate_s$s$p\"") < c.index("kjob \"fit_s$s$p\""), "no fit before its gate"
    assert c.index("for st in hyps_eval_0") < c.index("kjob \"gate_s$s$p\"")
    # the guard the chain depends on must know our python (WIRING.patch): the chain refuses to run against a guard that cannot stop it
    assert "v8f_kscore" in c.split('stop "guard stop"')[0]


# =============================================================================================================================================
# THE PREDECESSOR WAIT (PREREG "Compute and order"): the predecessor is FLAT's chain. A predecessor that ended on its OWN verdict (a run-level rc != 0, the reader refusing, the launch gate
# refusing) must not leave box 2 idle for hours -- an unconditional "any ZZV8FMINI-STOP stops me" idled a GPU for 8.3 h -- and a predecessor that left the BOX unhealthy (a thermal / time guard
# stop, a reboot, an unreachable box, a timeout, a busy marker, ANY text this predicate has never seen) must not be handed a 10-hour job. ONE function (after_decision) decides, for the wait
# loop and for `--classify-predecessor`, which runs HERE on synthetic logs.   MUTATIONS that must make a check RED: an unknown text PROCEEDs; the box-level evidence is ignored; the REFRESH
# job's rc is let through; the run-level regex is widened to any job name.
# =============================================================================================================================================
FLAT_CHAIN = Path(os.environ["KSCORE_FLAT_CHAIN"]) if os.environ.get("KSCORE_FLAT_CHAIN") else PK / "code" / "box2_ops" / "chain_v8f_flat.sh"
PROCEED, HOLDING, WAIT = 0, 1, 2
P, H = PROCEED, HOLDING
JOB = "v8fflat10261904_"                  # a FLAT job name: v8fflat<MMDDHHMM>_<stage>, as queue() in chain_v8f_flat.sh builds it
WAIT_MSG = "(2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
DONE_LINE = "ZZV8FMINI-DONE 10-26 04:10 tip 18c62306: PREREG_V8F_FLAT HIERARCHY BEATS FLAT"

# EVERY text FLAT's chain can print after ZZV8FMINI-STOP, as the source spells it (the 42 distinct `stop "<msg>"` texts), rendered with plausible values, each with the decision this chain
# commits for it. PROCEED = a gate / train / eval job's rc, FLAT's reader refusing, FLAT's launch gate refusing, and the 21 PRE-FLIGHT texts (FLAT stopped on the dev box before box 2
# ran anything; MM 2026-10-25). Everything else is box-level and HOLDS.
FLAT_STOPS = {
    "$1 rc=$rc": [(f"{JOB}gateflat rc=1", P), (f"{JOB}t_M2b_s0_flat rc=1", P), (f"{JOB}t_M2b_s1_flat rc=2", P), (f"{JOB}e_M2b_s0_flat rc=1", P), (f"{JOB}e_M2b_s1_flat rc=137", P),
                  (f"{JOB}refresh rc=1", H), (f"{JOB}refresh rc=3", H),                                  # the refresh job runs no model code: a tar into box 2's repo failed (the box's problem)
                  (f"{JOB}t_M2b_s0_flat rc=", H), (f"{JOB}t_M2b_s0_flat rc=0", H), (f"{JOB}t_M2b_s0_flat rc=-1073741819", H)],   # nothing parsed / not a failure / a native crash (signed)
    "flat read": [("flat read", P)],
    "gate FLAT assertions": [("gate FLAT assertions", P)],
    "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)": [(f"{JOB}t_M2b_s0_flat wait=2 {WAIT_MSG}", H), (f"{JOB}e_M2b_s1_flat wait=3 {WAIT_MSG}", H), (f"{JOB}gateflat wait=4 {WAIT_MSG}", H),
                                                                    (f"{JOB}refresh wait=4 {WAIT_MSG}", H)],
    "box 2 unreachable": [("box 2 unreachable", H)],
    "guard stop": [("guard stop", H)],
    "no fresh telemetry ($t1 / $t2): the temperature guard would be blind": [("no fresh telemetry (a / b): the temperature guard would be blind", H)],
    "ship $j": [(f"ship {JOB}gateflat", H)],
    "launch $j": [(f"launch {JOB}t_M2b_s0_flat", H)],
    "pull $R/$1": [("pull D:/tanitad/runs/v8f_mini_full/M2b_s0_flat/ckpt_last.pt", H)],
    "ship the code archive": [("ship the code archive", H)],
    "ship refresh job": [("ship refresh job", H)],
    "launch refresh": [("launch refresh", H)],
    "archive": [("archive", P)],
    "extract the reader": [("extract the reader", P)],
    "write the gate token": [("write the gate token", H)],
    "remote tip $T is not in the mirror": [("remote tip 18c62306 is not in the mirror", P)],
    "box 2 $f != tip $T": [("box 2 stack/scripts/v8f_mini_train.py != tip 18c62306", H)],
    "box 2 v9 ${kv%%=*} md5 != ${kv#*=}": [("box 2 v9 v9_labels_train.npz md5 != 83e30bd3791f0767707eade5f4671759", H)],
    "$1: cannot hash box 2's M2b_s$s checkpoint": [("before the gate: cannot hash box 2's M2b_s0 checkpoint", H)],
    "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}": [("before the gate: box 2's M2b_s0 checkpoint sha256 aa != the banked run's bb", H)],
    "cannot hash box 2's M2b_s${1}_flat checkpoint ($2)": [("cannot hash box 2's M2b_s0_flat checkpoint (trained)", H)],
    "M2b_s${1}_flat's checkpoint changed during the evals: $g != $g0": [("M2b_s0_flat's checkpoint changed during the evals: a != b", H)],
    "the reader cannot use the M2b side (check-inputs)": [("the reader cannot use the M2b side (check-inputs)", P)],
    "stage the banked M2b_s$s/$f": [("stage the banked M2b_s0/summary.json", P)],
    "stage eval_M2b_s$s": [("stage eval_M2b_s0", P)],
    "stage $f": [("stage eval_M2b_s0-tacabs.npz", P)],
    "stage floors_perwindow.npz (the P10 reproduction guard)": [("stage floors_perwindow.npz (the P10 reproduction guard)", P)],
    "v8f_guard.sh does not know this chain's markers (anchored): it would never exit": [("v8f_guard.sh does not know this chain's markers (anchored): it would never exit", H)],
    "the tip's queue has no $st stage": [("the tip's queue has no gateflat stage", P)],
    "the tip's queue prints no gateflat line": [("the tip's queue prints no gateflat line", P)],
    "the tip has no $f": [("the tip has no v8f_flat_read.py", P)],
    "the tip has no per-window strata file": [("the tip has no per-window strata file", P)],
    "the tip has no B1 lead block": [("the tip has no B1 lead block", P)],
    "the tip's built_levers does not record o_interface": [("the tip's built_levers does not record o_interface", P)],
    "the tip's built_levers does not record p_gt_start": [("the tip's built_levers does not record p_gt_start", P)],
    "the tip's built_levers does not record p_gt_end": [("the tip's built_levers does not record p_gt_end", P)],
    "the tip's v8f_mini.py has no O_INTERFACE_MODES": [("the tip's v8f_mini.py has no O_INTERFACE_MODES", P)],
    "the tip's v8f_mini.py has no cut line": [("the tip's v8f_mini.py has no cut line", P)],
    "the tip's trainer has no --o-interface flag": [("the tip's trainer has no --o-interface flag", P)],
    "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --o-interface": [("the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --o-interface", P)],
    "the tip's trainer does not run the o_interface probe in its in-process gate": [("the tip's trainer does not run the o_interface probe in its in-process gate", P)],
}
# the direct `echo ZZV8FMINI-STOP <msg>` lines of FLAT's chain (not through stop()); the last is printed by `--dry-run` only
FLAT_DIRECT = {
    "the predecessor's end does not clear the box: $d": [("the predecessor's end does not clear the box: HOLD box-level evidence in the predecessor's logs: ZZGUARD-STOP TEMP (its stop: guard stop)", H)],
    "busy marker present: $(cat $B/BOX2_BUSY.txt)": [("busy marker present: v8F-mini PREREG_V8F_MIXO (X = MIXO) since 10-25 12:00", H)],
    "busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)": [("busy marker taken while arming: v8F-mini PREREG_V8F_KIND", H), ("busy marker taken while arming: ", H)],
    "no queue at the tip": [("no queue at the tip", H)],
}
# texts that are NOT FLAT's (another chain's, a stranger's, a malformed one): the regexes are FLAT's, so every one of them HOLDS (fail closed)
NOT_FLATS = [
    ("v8fmixo10261904_t_M2b_s0_mixo rc=1", H), ("gatemixo rc=1", H), ("mixo read final", H), ("mixo read", H), ("gate X assertions", H), ("gate s0 assertions", H), ("gate s1_perm assertions", H),
    ("t_M2b_s0_flat rc=1", H), ("t_x rc=1", H), ("rc=1", H), (f"{JOB}t_M2b_s2_flat rc=1", H), ("v8fflat1026190_t_M2b_s0_flat rc=1", H), (f"{JOB}telemetry_long rc=1", H),
    (f"{JOB}t_M2b_s0_flat rc=1 (the read's rc 2 = a failed stop-control)", H), ("flat read final", H), ("flat read s0", H), ("gate FLAT assertions (extra)", H),
    ("the predecessor stopped: ZZV8FMINI-STOP t_x rc=1", H), ("a stop text this predicate has never seen", H), ("", H),
]
UNKNOWN = [m for m, w in NOT_FLATS if m in ("a stop text this predicate has never seen", "", "gate X assertions", "flat read final")]
PREFLIGHT_KEYS = ['archive', 'extract the reader', 'remote tip $T is not in the mirror', 'the reader cannot use the M2b side (check-inputs)', 'stage the banked M2b_s$s/$f', 'stage eval_M2b_s$s', 'stage $f', 'stage floors_perwindow.npz (the P10 reproduction guard)', "the tip's queue has no $st stage", "the tip's queue prints no gateflat line", 'the tip has no $f', 'the tip has no per-window strata file', 'the tip has no B1 lead block', "the tip's built_levers does not record o_interface", "the tip's built_levers does not record p_gt_start", "the tip's built_levers does not record p_gt_end", "the tip's v8f_mini.py has no O_INTERFACE_MODES", "the tip's v8f_mini.py has no cut line", "the tip's trainer has no --o-interface flag", "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --o-interface", "the tip's trainer does not run the o_interface probe in its in-process gate"]
STOP_TABLE = [x for d in (FLAT_STOPS, FLAT_DIRECT) for rows in d.values() for x in rows] + NOT_FLATS
EVIDENCE = ["ZZGUARD-STOP TEMP 10-25 13:00 USB HDD: 12 of the last 20 samples >= 61 C, max 63 C",
            "ZZGUARD-STOP TIME 10-25 13:00 M2b_s0_flat projects 11.2 h (> 10 h) at 2.1 s/step",
            "ZZCRASH box 2 rebooted: 2026-10-24 07:00:00 -> 2026-10-25 13:01:00", "ZZUNREACHABLE 12 min", "ZZTIMEOUT 1320 polls"]    # the exact shapes v8f_guard.sh / waitjob.sh print, at column 0
PROCEEDING_STOPS = [f"{JOB}t_M2b_s0_flat rc=1", "flat read", "gate FLAT assertions", "archive", "the tip has no B1 lead block"]


def _classify(tmp_path, out_lines, guard_lines=None, name="c", chain=None, raw=None):
    bash = _bash()
    if bash is None:
        pytest.skip("bash is needed to run the chain's predicate")
    o = tmp_path / f"{name}.out"
    o.write_bytes(raw if raw is not None else (("\n".join(out_lines) + "\n").encode("utf-8") if out_lines is not None else b""))
    args = [bash, str(chain or CHAIN), "--classify-predecessor", o.as_posix()]
    if guard_lines is not None:
        g = tmp_path / f"{name}.guard"
        g.write_bytes(("\n".join(guard_lines) + "\n").encode("utf-8"))
        args.append(g.as_posix())
    r = subprocess.run(args, capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout.strip(), r.stderr


def _stop(msg):
    return ["10-25 12:00 boot0=2026-10-24 07:00:00", f"ZZV8FMINI-STOP {msg}".rstrip()]


def _row_violations(tmp_path, chain, rows, tag):
    """The rows whose decision the chain at `chain` does NOT make: [] on a correct chain, non-empty on a mutant."""
    bad = []
    for i, (msg, want) in enumerate(rows):
        rc, out, err = _classify(tmp_path, _stop(msg), name=f"{tag}{i}", chain=chain)
        if rc != want or out.split()[:1] != [("PROCEED" if want == P else "HOLD")]:
            bad.append((msg, want, rc, out, err[:200]))
    return bad


def _evidence_violations(tmp_path, chain, tag, evs=EVIDENCE):
    """Every proceeding stop, with box-level evidence in the chain log or in the guard log, must HOLD."""
    bad = []
    for e, ev in enumerate(evs):
        for s, stop in enumerate(PROCEEDING_STOPS):
            for where in ("chain", "guard"):
                rows = _stop(stop)
                lines, guard = (rows[:1] + [ev] + rows[1:], None) if where == "chain" else (rows, [ev])
                rc, out, _ = _classify(tmp_path, lines, guard, name=f"{tag}{e}{s}{where}", chain=chain)
                if rc != H or not out.startswith("HOLD box-level evidence"):
                    bad.append((ev, stop, where, rc, out))
    return bad


def _mutant(tmp_path, old, new, name):
    src = CHAIN.read_text(encoding="utf-8")
    assert src.count(old) == 1, f"the mutation target is not unique / not there: {old!r}"
    p = tmp_path / name
    p.write_bytes(src.replace(old, new).encode("utf-8"))
    return p


def test_predecessor_done_proceeds_and_no_end_marker_waits(tmp_path):
    rc, out, err = _classify(tmp_path, ["10-25 12:00 boot0=2026-10-24 07:00:00", f"10-25 12:30 {JOB}e_M2b_s1_flat rc=0", DONE_LINE])
    assert rc == PROCEED and out.startswith("PROCEED the predecessor ended: ZZV8FMINI-DONE"), (rc, out, err)
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=2026-10-24 07:00:00", f"10-25 12:30 {JOB}t_M2b_s0_flat rc=0"], name="wait")
    assert rc == WAIT and out.startswith("WAIT no ZZV8FMINI-DONE"), (rc, out)
    rc, out, _ = _classify(tmp_path, None, name="empty")
    assert rc == WAIT, "an empty log: the predecessor has not started"
    rc, out, _ = _classify(tmp_path, None, name="crlf", raw=(DONE_LINE + "\r\n").encode())
    assert rc == PROCEED, "a CRLF log (the chain's output file) is read the same"
    rc, out, _ = _classify(tmp_path, None, name="crlfstop", raw=(f"ZZV8FMINI-STOP {JOB}t_M2b_s0_flat rc=1\r\n").encode())
    assert rc == PROCEED and out.endswith("rc=1"), "a CRLF stop line is read the same (the CR is not part of the message)"
    rc, out, _ = _classify(tmp_path, [f"ZZV8FMINI-STOP {JOB}t_M2b_s0_flat rc=1", DONE_LINE], name="both")
    assert rc == PROCEED, "a DONE anywhere in the log wins (a stop followed by a done is a finished chain)"
    rc, out, _ = _classify(tmp_path, ["ZZV8FMINI-STOP guard stop", f"ZZV8FMINI-STOP {JOB}t_M2b_s0_flat rc=1"], name="last")
    assert rc == PROCEED and "rc=1" in out, "the LAST stop line decides"
    rc, out, _ = _classify(tmp_path, [f"ZZV8FMINI-STOP {JOB}t_M2b_s0_flat rc=1", "ZZV8FMINI-STOP guard stop"], name="last2")
    assert rc == HOLDING, "the LAST stop line decides (a later box-level stop holds)"


@pytest.mark.parametrize("msg, want", STOP_TABLE, ids=[f"{i:02d}" for i in range(len(STOP_TABLE))])
def test_predecessor_stop_is_classified_run_verdict_gate_proceed_and_everything_else_holds(tmp_path, msg, want):
    rc, out, err = _classify(tmp_path, _stop(msg))
    assert rc == want, (msg, rc, out, err)
    assert out.split()[0] == ("PROCEED" if want == PROCEED else "HOLD")


def test_the_stop_table_covers_every_text_of_flats_chain_and_exactly_three_proceed():
    """The table is a LIST of FLAT's texts (literal, so it is checked even where FLAT's chain is not in the tree) and, where FLAT's chain IS in the tree, it is checked against its source."""
    assert len(FLAT_STOPS) == 42 and len(FLAT_DIRECT) == 4
    proceeding = sorted(k for d in (FLAT_STOPS, FLAT_DIRECT) for k, rows in d.items() if any(w == P for _, w in rows))
    assert proceeding == sorted(["$1 rc=$rc", "flat read", "gate FLAT assertions"] + PREFLIGHT_KEYS), "the proceeding classes are exactly run / verdict / gate + the 21 pre-flight texts"
    assert all(w == P for k in PREFLIGHT_KEYS for _, w in FLAT_STOPS[k])
    assert all(w == H for _, w in NOT_FLATS), "no text outside FLAT's chain may proceed"
    refresh = [(m, w) for m, w in FLAT_STOPS["$1 rc=$rc"] if "refresh" in m]
    assert refresh and all(w == H for _, w in refresh), "the code-refresh job's rc HOLDS: it ran no model code (documented deviation from the MIXO-era predicate)"


@pytest.mark.skipif(not FLAT_CHAIN.exists(), reason="chain_v8f_flat.sh is not in this tree (FLAT has not landed): the table is checked against its literal list only. Set KSCORE_FLAT_CHAIN to check it against a copy")
def test_every_stop_text_of_flats_chain_source_is_in_the_table_so_a_new_one_cannot_slip_through_unclassified():
    """A stop text added to chain_v8f_flat.sh later is not in the table: this test names it. (The predicate holds on anything it does not recognise, so the default is the safe one, but
    a NEW text deserves a decision made on purpose: it may be one that should proceed, and holding it re-creates the idle box.)"""
    src = FLAT_CHAIN.read_text(encoding="utf-8")
    code = "\n".join(x for x in src.splitlines() if not x.lstrip().startswith("#"))
    stops = set(re.findall(r'stop "([^"]*)"', code))
    direct = set(re.findall(r'echo "ZZV8FMINI-STOP ([^"]*)"', code)) - {"$*"}
    assert sorted(stops - set(FLAT_STOPS)) == [] and sorted(set(FLAT_STOPS) - stops) == [], "FLAT's `stop \"<msg>\"` texts and the table differ"
    assert sorted(direct - set(FLAT_DIRECT)) == [] and sorted(set(FLAT_DIRECT) - direct) == [], "FLAT's direct ZZV8FMINI-STOP lines and the table differ"
    # what RUN_LEVEL_RE is built from: FLAT's own job names, tag format and the stages that run its code
    assert "TAG=$(date '+%m%d%H%M')" in src and 'local j="v8fflat${TAG}_$1"' in src, "FLAT's job name is v8fflat<MMDDHHMM>_<stage>"
    assert "queue gateflat 120 gateflat" in src and 'queue "t_$g" $TRAIN_POLLS trainflat $s' in src and 'queue "e_$g" $EVAL_POLLS evalflat $s' in src
    assert 'g="M2b_s${1}_flat"' in src and "for s in 0 1; do" in src and "runone 0\n" in src and "runone 1\n" in src and "runone 2" not in src, "FLAT trains seeds 0 and 1 only"
    assert "launch v8fflat${TAG}_refresh" in src and "waitrc v8fflat${TAG}_refresh" in src, "FLAT's refresh job is v8fflat<tag>_refresh and goes through waitrc (its rc stop is `<job> rc=<n>`)"
    assert 'stop "$1 rc=$rc"' in src and "|| stop \"flat read\"" in src and '"gate FLAT assertions"' in src
    # the logs this chain waits on are FLAT's own
    assert "V8F_CHAIN_OUT=$L/chain_v8f_flat.out" in src and "> $L/v8fflat_guard.out" in src
    mine = CHAIN.read_text(encoding="utf-8")
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_flat.out}" in mine and "AFTER_GUARD=${AFTER_GUARD:-$L/v8fflat_guard.out}" in mine
    # the job-name part of RUN_LEVEL_RE renders exactly the jobs FLAT queues
    m = re.search(r"^RUN_LEVEL_RE='\^v8fflat\[0-9\]\{8\}_\((.*)\) rc=\[1-9\]\[0-9\]\*\$'$", mine, re.M)
    assert m and m.group(1) == "gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat", "RUN_LEVEL_RE must name exactly FLAT's gate / train / eval jobs"


@pytest.mark.parametrize("ev", EVIDENCE)
def test_predecessor_stop_with_box_level_evidence_in_its_logs_holds(tmp_path, ev):
    """The evidence overrides the message: a run-level `rc=1` that came with a thermal / time stop, a reboot, an unreachable box or a timeout is the box's failure, not the run's."""
    bad = _evidence_violations(tmp_path, None, "ev", [ev])
    assert bad == [], bad
    rc, out, _ = _classify(tmp_path, _stop(f"{JOB}t_M2b_s0_flat rc=1"), ["10-25 13:00 chain ended; guard off (max USB 52 C)"], name="clean")
    assert rc == PROCEED, "a guard log that merely ended is no evidence"
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=x", ev, DONE_LINE], name="done_wins")
    assert rc == PROCEED, "ZZV8FMINI-DONE proceeds whatever the log says before it: the predecessor FINISHED"


def test_hold_texts_name_their_reason(tmp_path):
    rc, out, _ = _classify(tmp_path, _stop(f"{JOB}refresh rc=1"))
    assert rc == HOLDING and out.startswith("HOLD box-level: FLAT's code-refresh job"), out
    rc, out, _ = _classify(tmp_path, _stop("something new"))
    assert rc == HOLDING and out.startswith("HOLD a stop that is not recognised") and out.endswith("something new"), out
    rc, out, _ = _classify(tmp_path, _stop(f"{JOB}gateflat rc=1"), ["ZZGUARD-STOP TEMP 10-25 13:00 x"])
    assert rc == HOLDING and "its stop: " + f"{JOB}gateflat rc=1" in out, out


def test_mutation_an_unknown_stop_text_that_proceeds_turns_the_unknown_rows_red(tmp_path):
    rows = [(m, H) for m in UNKNOWN]
    assert _row_violations(tmp_path, None, rows, "u0") == [], "the real predicate holds on every unknown text"
    mutant = _mutant(tmp_path, '(fail closed): $msg"; return 1', '(fail closed): $msg"; return 0', "mut_unknown_proceeds.sh")
    bad = _row_violations(tmp_path, mutant, rows, "u1")
    assert len(bad) == len(rows) and all(b[2] == PROCEED for b in bad), f"the mutation (an unrecognised stop PROCEEDs) was not detected: {bad}"


def test_mutation_ignoring_the_box_level_evidence_turns_the_evidence_check_red(tmp_path):
    evs = [EVIDENCE[0], EVIDENCE[2]]
    assert _evidence_violations(tmp_path, None, "e0", evs) == []
    mutant = _mutant(tmp_path, '[ -z "$ev" ] || { echo "HOLD box-level evidence in the predecessor\'s logs: $ev (its stop: $msg)"; return 1; }', ":", "mut_no_evidence.sh")
    bad = _evidence_violations(tmp_path, mutant, "e1", evs)
    assert len(bad) == len(evs) * len(PROCEEDING_STOPS) * 2, f"the mutation (box-level evidence ignored) was not detected on every case: {len(bad)}"


def test_mutation_letting_the_refresh_job_rc_through_turns_its_rows_red(tmp_path):
    rows = [(m, w) for m, w in FLAT_STOPS["$1 rc=$rc"] if "refresh" in m]
    assert len(rows) == 2 and _row_violations(tmp_path, None, rows, "r0") == []
    mutant = _mutant(tmp_path, "(gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat)", "(gateflat|refresh|t_M2b_s[01]_flat|e_M2b_s[01]_flat)", "mut_refresh.sh")
    assert len(_row_violations(tmp_path, mutant, rows, "r1")) == len(rows), "the mutation (the refresh job's rc proceeds) was not detected"


def test_mutation_widening_the_run_level_regex_to_any_job_name_turns_the_wrong_chain_rows_red(tmp_path):
    rows = [(m, w) for m, w in NOT_FLATS if re.fullmatch(r"[A-Za-z0-9_.-]+ rc=[1-9][0-9]*", m)]       # what the MIXO-era, job-name-agnostic regex would let through
    assert len(rows) >= 4 and _row_violations(tmp_path, None, rows, "w0") == []
    mutant = _mutant(tmp_path, "RUN_LEVEL_RE='^v8fflat[0-9]{8}_(gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat) rc=[1-9][0-9]*$'", "RUN_LEVEL_RE='^[A-Za-z0-9_.-]+ rc=[1-9][0-9]*$'", "mut_anyjob.sh")
    bad = _row_violations(tmp_path, mutant, rows, "w1")
    assert len(bad) == len(rows), f"the mutation (RUN_LEVEL_RE matches any job name) was not detected on every wrong-chain row: {bad}"


def test_the_wait_loop_calls_the_same_predicate_and_the_chain_stops_on_a_hold():
    raw = CHAIN.read_bytes()
    assert b"\r\n" not in raw, "a bash script with CRLF endings does not run"
    c = raw.decode("utf-8")
    assert 'd=$(after_decision "$AFTER_OUT" "$AFTER_GUARD"); r=$?' in c and 'if [ $r = 1 ]; then echo "ZZV8FMINI-STOP the predecessor\'s end does not clear the box: $d"; exit 1; fi' in c
    assert 'if [ $r = 0 ]; then echo "$(date \'+%m-%d %H:%M\') predecessor: $d"; break; fi' in c
    assert 'if [ "${1:-}" = "--classify-predecessor" ]; then after_decision' in c
    # the predicate and the classify mode come before EVERY statement that touches box state (the mkdir of $E, the ssh / scp definitions) and before the dry-run: classify has no side effect
    assert c.index("after_decision() {") < c.index('if [ "${1:-}" = "--classify-predecessor" ]') < c.index('if [ "${1:-}" = "--dry-run" ]') < c.index("mkdir -p $E/fit $E/eval") < c.index("waiting for the predecessor chain")
    assert c.index("AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}") < c.index("after_decision() {")
    assert "sleep 300" in c and "until grep -q" not in c and "the predecessor stopped:" not in c, "the old 'any STOP stops me' loop is gone"
    assert "RUN_LEVEL_RE='^v8fflat[0-9]{8}_(gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat) rc=[1-9][0-9]*$'" in c and "VERDICT_LEVEL_RE='^flat read$'" in c and "GATE_LEVEL_RE='^gate FLAT assertions$'" in c
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_flat.out}" in c and "AFTER_GUARD=${AFTER_GUARD:-$L/v8fflat_guard.out}" in c


def test_the_wait_loop_runs_end_to_end_proceeds_on_flats_own_stops_and_holds_on_a_box_level_stop(tmp_path):
    """Run the chain itself against a synthetic predecessor log with a stubbed box: DONE / a run-level / a verdict-level / a gate-level stop pass the wait (and the chain then dies at its
    first box-2 step: the stubbed `ssh` fails, the busy marker is taken and released); a thermal stop, a timeout and an unknown text never get past it, and take no marker; a log with
    no end marker WAITs, and the loop proceeds the moment the marker appears."""
    bash = _bash()
    if bash is None:
        pytest.skip("bash is needed to run the chain")
    box = tmp_path / "box2_ship"
    (box / "logs").mkdir(parents=True)
    src = CHAIN.read_text(encoding="utf-8")
    # THE COPY CANNOT REACH ANYTHING REAL: the box-2 directory is this tmp dir, ssh / scp are `false` (so `boot` finds no box and the chain stops there, before it touches git or ships a file),
    # and the mirror is a path that does not exist. Each substitution is asserted: a chain whose text moved must not fall through to the real box or the real mirror.
    for old, new in (("B=/c/Users/Admin/box2_ship", f"B={box.as_posix()}"), ('SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"', 'SSH="false"'),
                     ('SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"', 'SCP="false"'), ("MIRROR=/c/Users/Admin/tanitad-push;", f"MIRROR={(tmp_path / 'no_mirror').as_posix()};")):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    code = "\n".join(x for x in src.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(?<![\w$])(ssh|scp)\s", code), "no literal ssh / scp command is left: every call goes through $SSH / $SCP, which are `false` here"
    script = tmp_path / "chain_under_test.sh"
    script.write_bytes(src.encode("utf-8"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("AFTER_")}              # a leaked AFTER_* override would point the wait elsewhere
    env["TMPDIR"] = tmp_path.as_posix()

    def run(lines, guard=None, script_path=script):
        pred = box / "logs" / "chain_v8f_flat.out"
        pred.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        gp = box / "logs" / "v8fflat_guard.out"
        gp.write_bytes(("\n".join(guard) + "\n").encode("utf-8") if guard else b"")
        (box / "BOX2_BUSY.txt").unlink(missing_ok=True)
        r = subprocess.run([bash, str(script_path)], env=env, capture_output=True, text=True, timeout=120, encoding="utf-8", stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout

    def proceeds(out, how):
        return f"predecessor: PROCEED {how}" in out and "ZZV8FMINI-STOP box 2 unreachable" in out

    rc, out = run([DONE_LINE])
    assert rc == 1 and proceeds(out, "the predecessor ended"), out
    assert not (box / "BOX2_BUSY.txt").exists(), "a chain that stops releases the marker it took"
    rc, out = run([f"ZZV8FMINI-STOP {JOB}t_M2b_s0_flat rc=1"])
    assert rc == 1 and proceeds(out, f"run-level stop: {JOB}t_M2b_s0_flat rc=1"), out
    rc, out = run(["ZZV8FMINI-STOP flat read"])
    assert rc == 1 and proceeds(out, "verdict-level stop: flat read"), out
    rc, out = run(["ZZV8FMINI-STOP gate FLAT assertions"])
    assert rc == 1 and proceeds(out, "gate-level stop"), out
    rc, out = run(["ZZV8FMINI-STOP guard stop"], ["ZZGUARD-STOP TEMP 10-25 13:00 USB HDD: x"])
    assert rc == 1 and "ZZV8FMINI-STOP the predecessor's end does not clear the box: HOLD" in out and "box 2 unreachable" not in out, out
    assert not (box / "BOX2_BUSY.txt").exists(), "a held chain never took the marker"
    rc, out = run([f"ZZV8FMINI-STOP {JOB}t_M2b_s0_flat wait=4 {WAIT_MSG}", "ZZTIMEOUT 1320 polls"])
    assert rc == 1 and "does not clear the box: HOLD" in out and "box 2 unreachable" not in out, out
    rc, out = run([f"ZZV8FMINI-STOP {JOB}t_M2b_s0_flat rc=1"], ["ZZGUARD-STOP TIME 10-25 13:00 M2b_s0_flat projects 11.2 h (> 10 h) at 2.1 s/step"])
    assert rc == 1 and "does not clear the box: HOLD box-level evidence" in out and "box 2 unreachable" not in out, "a run-level rc=1 that the guard caused HOLDS"
    rc, out = run(["ZZV8FMINI-STOP a stop text nobody has seen"])
    assert rc == 1 and "does not clear the box: HOLD a stop that is not recognised" in out and "box 2 unreachable" not in out, out
    # WAIT -> PROCEED: no end marker yet, so the loop sleeps; the stubbed sleep appends FLAT's DONE to the log, and the next pass proceeds
    waits = src.replace("sleep 300", 'echo "ZZV8FMINI-DONE 10-26 05:00 tip x: appended while the loop slept" >> "$AFTER_OUT"')
    assert waits != src
    sleeper = tmp_path / "chain_waits.sh"
    sleeper.write_bytes(waits.encode("utf-8"))
    rc, out = run(["10-25 12:00 boot0=x"], script_path=sleeper)
    assert rc == 1 and "waiting for the predecessor chain" in out and proceeds(out, "the predecessor ended: ZZV8FMINI-DONE 10-26 05:00"), out


def test_mkjob_text_is_a_crlf_safe_mode_box2_job():
    t = K.mkjob_text("j1", ["fit", "--seed", "0", "--gate-only", "--commit", "abc"])
    assert "\r\n" in t and "\r\r\n" not in t and t.endswith("\r\n")
    assert t.count("%RUNSAFE%") == 1 and "call D:\\tanitad\\env.cmd" in t and 'set "REPO=D:\\tanitad\\repo"' in t and "ZZJOB-END %DATE% %TIME% rc=!RC!" in t
    assert "/" not in re.sub(r"^echo .*$", "", t, flags=re.M).replace("%DATE%", ""), "box 2's cmd needs real backslashes (a forward-slash interpreter path silently ends a batch file in a pipe)"
    r = subprocess.run([sys.executable, "-I", str(CHAIN.parent / "v8f_kscore.py"), "mkjob", "jx", "eval", "--seeds", "0", "1"], capture_output=True, timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and b"\r\r\n" not in r.stdout and r.stdout.count(b"\r\n") == len(r.stdout.splitlines())


def test_hyps_stage_peak_memory_at_real_size_is_under_2_gb(tmp_path):
    """The hyps stage streams: peak PRIVATE commit of a real-size (d = 256, hist 4, 40 tokens x 1024) random-init B over a batch of 16 windows must stay far under 2 GB."""
    try:
        import psutil  # noqa: F401
    except Exception:
        pytest.skip("psutil is needed for the peak-commit probe")
    code = (
        "import os,sys,json\n"
        "os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')\n"
        f"sys.path[:0]=[{str(REPO / 'stack' / 'scripts')!r},{str(REPO / 'taniteval')!r},{str(REPO / 'stack')!r}]\n"
        "import torch,numpy as np,psutil\n"
        "import v8f_kscore as K\n"
        "from tanitad.data import v8f_mini_data as D\n"
        "from tanitad.models import v8f_mini as M\n"
        "torch.set_num_threads(4)\n"
        f"w=D.make_synthetic_world({str(tmp_path / 'w')!r},n_train=2,n_eval=1,T=48,seed=0)\n"
        "J=D.load_join(os.path.join(w['root'],'join_train.npz'))\n"
        "cfg=M.V8FMiniConfig(hmax=10)\n"
        "mu,sd=M.cons_stats(J)\n"
        "m=M.V8FMini(cfg,M.P10Prior.synthetic(0),mu,sd).eval()\n"
        "st=D.ClipStore(w['root'],'train',lru=2)\n"
        "wins=D.train_windows(st,J,w['names']['train'])\n"
        "ds=D.V8FMiniWindows(st,J,wins,hmax=10)\n"
        "o=K.enumerate_window_batch(m,ds,wins,0,16,torch.device('cpu'),np.float16)\n"
        "pm=psutil.Process().memory_info()\n"
        "print(json.dumps({'peak_commit_gb':getattr(pm,'peak_pagefile',pm.rss)/2**30,'peak_wset_gb':getattr(pm,'peak_wset',pm.rss)/2**30,'params':sum(p.numel() for p in m.parameters())}))\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=600, env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"), stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr[-2000:]
    j = json.loads(r.stdout.strip().splitlines()[-1])
    print("HYPS_PEAK", j)
    assert j["peak_commit_gb"] < 2.0, j


def test_preflight_stops_proceed_and_a_mutant_that_holds_them_is_detected(tmp_path):
    """MM 2026-10-25: FLAT's 21 pre-flight stops happen on the dev box before box 2 runs anything, and KSCORE re-checks its own tip, so they PROCEED (holding idles box 2 for a night)."""
    rows = [x for k in PREFLIGHT_KEYS for x in FLAT_STOPS[k]]
    assert len(rows) == 21 and _row_violations(tmp_path, None, rows, "pf0") == []
    mutant = _mutant(tmp_path, '(on the dev box, before box 2 ran anything; this chain re-checks its own tip): $msg"; return 0', '(on the dev box, before box 2 ran anything; this chain re-checks its own tip): $msg"; return 1', "mut_pf_holds.sh")
    bad = _row_violations(tmp_path, mutant, rows, "pf1")
    assert len(bad) == 21, f"the mutation (pre-flight HOLDS) was not detected on every pre-flight row: {len(bad)} of 21"
    near = [("the tip has no v8f_other_read.py", H), ("stage eval_M2b_s2", H), ("stage something else", H), ("archive the logs", H), ("remote tip xyz is not in the mirror", H)]
    assert _row_violations(tmp_path, None, near, "pf2") == [], "the pre-flight regex is anchored to FLAT's own texts"
