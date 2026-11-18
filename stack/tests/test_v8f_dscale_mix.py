"""PREREG_V8F_DSCALE_MIX (claim row H-V8F-DSCALE-MIX-1, arm UM_s0): the per-source T-block row weight. Card section 3.3 (the change) and section 3.4 (the reachability proof, 7 checks) plus the MM ruling 2 (identity 1: bit for bit
unchanged when the batch key ``tac_w`` is ABSENT).

  argv  ->  TR.make_tac_src (refusals)  ->  TR.TacSrc  ->  declared_vs_built (bad_tac_src: G-DVB)  ->  tac_src_probe (G-LIVE, checks 1-5)  ->  ``tac_w = tac_src_stage(b, tsrc)`` / ``b["tac_w"] = tac_w`` in the loop (check 6)
        ->  V8FMini.loss (``rw = batch.get("tac_w")``)  ->  T["goal"] / T["lat"] / T["lon"] / T["cons"]  ->  T["total"].backward()  ->  summary.json (K-TACSRC)

Every check has a RED arm: the MODEL mutations load a mutated COPY of ``v8f_mini.py`` under another module name, the TRAINER mutations run a mutated COPY of ``v8f_mini_train.py`` laid out as ``<tmp>/stack/scripts/``. The tree is never
mutated. The synthetic world's trainer tests come BEFORE nothing imported here is a module-scoped fixture of another file.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
COMMON = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
BASE_MODEL = Path(os.environ.get("TANITAD_MIX_BASE_MODEL", "F:/TanitAD-scratch/datascale/mix_impl/base/v8f_mini.py.base"))   # the pristine cdbff971 model file (the MM's pod run has none: that one test skips)


# ------------------------------------------------------------------------------------------------ rig
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixsyn")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    return {"root": root, **w}


class Rig:
    pass


def _list_file(tmp, ids, name="bset_clips.txt"):
    path = Path(tmp) / name
    path.write_bytes("\n".join(ids).encode("utf-8"))                              # newline-joined, no trailing newline (the card's file format)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", *extra])


@pytest.fixture(scope="module")
def rig(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mixrig")
    k = Rig()
    k.names = world["names"]["train"]
    k.listed = sorted(k.names, key=D.sha12)[:3]                                     # 3 of the 6 train clips play the 4,512 corpus clips
    k.path, k.sha = _list_file(tmp, k.listed)
    k.a = _args(["--tac-src-w", "0.0", "--tac-src-clips", k.path, "--tac-src-clips-sha256", k.sha])
    k.J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    k.wins = D.train_windows(store, k.J, k.names)
    k.ds = D.V8FMiniWindows(store, k.J, k.wins, hmax=10)
    k.info = {}
    k.src = TR.make_tac_src(k.a, k.J, k.names, k.wins, k.info)
    assert k.src is not None
    ci_l = [w for w in k.wins if bool(k.src.full_ci[w[0]])]
    ci_o = [w for w in k.wins if not bool(k.src.full_ci[w[0]])]
    rng = np.random.default_rng(0)
    pick = lambda ws, n: [ws[i] for i in rng.choice(len(ws), n, replace=False)]      # noqa: E731
    mk = lambda ws: D.collate([k.ds.item(ci, t, 1, 3) for ci, t, _, _ in ws])        # noqa: E731
    wl, wo = pick(ci_l, 16), pick(ci_o, 16)
    k.b_listed, k.b_other, k.b_mixed = mk(wl), mk(wo), mk(wl[:8] + wo[:8])
    k.cfg = M.V8FMiniConfig(**TINY)
    return k


def _model(J, mod=M, **kw):
    cfg = mod.V8FMiniConfig(**{**TINY, **kw})
    torch.manual_seed(0)
    mu, sd = mod.cons_stats(J)
    return mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd), cfg


def _probe(rig, mod=M, checks=(1, 2, 3, 4, 5)):
    model, cfg = _model(rig.J, mod)
    return TR.tac_src_probe(model, cfg, rig.src, rig.b_mixed, rig.b_listed, rig.b_other, checks=checks)


_MUT_N = [0]


def _mutated_model_module(tmp, old, new):
    """A COPY of v8f_mini.py with ``old`` (exactly one occurrence) replaced by ``new``, loaded under its own module name; the tree is untouched."""
    src = Path(M.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, (src.count(old), old)
    _MUT_N[0] += 1
    name = f"v8f_mini_mut{_MUT_N[0]}"
    path = Path(tmp) / f"{name}.py"
    path.write_text(src.replace(old, new), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ the sources really differ (a probe blind to the lever would pass everything)
def test_rig_has_both_sources_with_labelled_rows(rig):
    for b, want in ((rig.b_listed, 1), (rig.b_other, 0), (rig.b_mixed, None)):
        full = rig.src.full_ci[b["ci"]]
        if want == 1:
            assert bool(full.all())
        elif want == 0:
            assert not bool(full.any())
        else:
            assert 0 < int(full.sum()) < full.shape[0]
        lt = b["lab_t"]
        assert bool(lt["has"].any()) and bool((lt["lat_al"].any(-1) & lt["has"]).any()) and bool((lt["lon_al"].any(-1) & lt["has"]).any())


def test_stage_values_are_one_on_listed_and_lambda_elsewhere(rig):
    tw = TR.tac_src_stage(rig.b_mixed, rig.src)
    full = rig.src.full_ci[rig.b_mixed["ci"]]
    assert tw.dtype == torch.float32 and tw.shape == full.shape
    assert bool((tw[full] == 1.0).all()) and bool((tw[~full] == 0.0).all())
    assert rig.src.n_listed_clips == 3 and rig.src.n_other_clips == 3
    assert rig.src.n_listed_windows + rig.src.n_other_windows == len(rig.wins)
    assert rig.info["tac_src"]["list_sha256"] == rig.sha


# ------------------------------------------------------------------------------------------------ identity 1 (MM ruling 2): ABSENT = the tip, bit for bit
def _orig_T_block(m, b, p_gt=0.5, seed=11):
    """The ORIGINAL (cdbff971) T-block lines, re-spelled from the model's own outputs: the oracle for identity 1."""
    out = m.forward(b, p_gt, torch.Generator().manual_seed(seed))
    lt = b["lab_t"]
    w = lt["goal_w"] * lt["has"][:, None].float()
    bce = F.binary_cross_entropy_with_logits(out["goal_logits"], lt["goal_y"], reduction="none")
    T = {"goal": (bce * w).sum() / w.sum().clamp_min(1.0)}

    def pce(logits, allowed, has):
        valid = allowed.any(-1) & has
        lse = torch.logsumexp(logits, -1)
        lsa = torch.logsumexp(logits.masked_fill(~allowed, float("-inf")), -1)
        l = torch.where(valid, lse - lsa, torch.zeros_like(lse))
        return l.sum() / valid.float().sum().clamp_min(1.0)

    T["lat"], T["lon"] = pce(out["lat_logits"], lt["lat_al"], lt["has"]), pce(out["lon_logits"], lt["lon_al"], lt["has"])
    tgt = (lt["cons"] - m.cons_mu) / m.cons_sd
    if m.cfg.cons_absent == "zero":
        mk = lt["has"][:, None].expand_as(tgt)
        tgt = torch.where(torch.isfinite(tgt), tgt, torch.zeros_like(tgt))
    else:
        mk = torch.isfinite(tgt) & lt["has"][:, None]
    hub = F.huber_loss(out["cons_std"], torch.nan_to_num(tgt), reduction="none", delta=1.0)
    T["cons"] = (hub * mk).sum() / mk.sum().clamp_min(1.0)
    return T


def _loss_absent(model, b):
    torch.manual_seed(20261118)
    return model.loss({k: v for k, v in b.items() if k != "tac_w"}, p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]


def _identity1(model, b):
    ref = _orig_T_block(model, b)
    got = _loss_absent(model, b)
    return {k: bool(torch.equal(ref[k], got[k])) for k in ref}


@pytest.mark.parametrize("which", ["mixed", "listed", "other"])
def test_identity1_absent_key_reproduces_the_original_T_block_bit_for_bit(rig, which):
    model, _ = _model(rig.J)
    b = {"mixed": rig.b_mixed, "listed": rig.b_listed, "other": rig.b_other}[which]
    assert all(_identity1(model, b).values()), _identity1(model, b)


def test_identity1_absent_key_equals_the_pristine_cdbff971_model_terms_and_gradients(rig, tmp_path):
    """The whole loss (every term, the total, every parameter gradient) of the changed model with ``tac_w`` absent equals the PRISTINE cdbff971 model file's, bit for bit, on the same weights."""
    if not BASE_MODEL.exists():
        pytest.skip(f"the pristine cdbff971 model file {BASE_MODEL} is not on this machine")
    name = "v8f_mini_pristine"
    spec = importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, str(BASE_MODEL)))
    base = importlib.util.module_from_spec(spec)
    sys.modules[name] = base
    spec.loader.exec_module(base)
    res = []
    for mod in (M, base):
        model, _ = _model(rig.J, mod)
        T = _loss_absent(model, rig.b_mixed)
        T["total"].backward()
        res.append(({k: v.detach().clone() for k, v in T.items()}, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    assert res[0][0].keys() == res[1][0].keys()
    assert all(torch.equal(res[0][0][k], res[1][0][k]) for k in res[0][0])
    assert res[0][1].keys() == res[1][1].keys() and all(torch.equal(res[0][1][n], res[1][1][n]) for n in res[0][1])


def test_identity1_RED_when_an_absent_key_gets_a_default_weight(rig, tmp_path):
    mod = _mutated_model_module(tmp_path, 'rw = batch.get("tac_w")  ', 'rw = batch.get("tac_w", torch.full((B,), 0.999))  ')
    model, _ = _model(rig.J, mod)
    assert not all(_identity1(model, rig.b_mixed).values())
    assert all(_identity1(_model(rig.J)[0], rig.b_mixed).values())               # control: the unmutated model is GREEN on the same batch


# ------------------------------------------------------------------------------------------------ 3.4 checks 1-5: GREEN on the change
def test_probe_green_all_five_checks_with_the_committed_values(rig):
    r = _probe(rig)
    assert r["pass"] is True
    assert r["control_repeat_term_diff"] == 0.0 and r["control_repeat_grad_diff"] == 0.0
    assert r["c1_term_diff"] == 0.0 and r["c1_grad_diff"] == 0.0                                      # check 1
    assert r["c2_block"] == 0.0 and r["c2_grad_abs_sum"] == 0.0                                       # check 2
    assert r["c2_control_block"] > 0.0 and all(r["c2_control_heads"][h] > 0.0 for h in ("h_goal", "h_lat", "h_lon", "h_cons"))
    for nm in ("lat", "lon"):                                                                         # check 3
        c = r[f"c3_{nm}"]
        assert math.isclose(c["got"], c["analytic"], rel_tol=1e-6)
        assert abs(c["got"] - c["lambda0"]) > 1e-5 * abs(c["got"]) and abs(c["got"] - c["lambda1"]) > 1e-5 * abs(c["got"])
    assert r["c4_nonT_term_diff"] == 0.0 and r["c4_nonT_grad_diff"] == 0.0 and r["c4_T_term_diff_control"] > 0.0 and r["c4_n_nonT_terms"] > 5      # check 4
    assert r["c5_term_diff"] == 0.0 and r["c5_grad_diff"] == 0.0                                      # check 5


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_each_check_is_green_alone(rig, n):
    assert _probe(rig, checks=(n,)).get("pass") is True


def test_the_T_block_at_lambda_zero_reads_exactly_zero_on_other_rows_in_the_real_loss(rig):
    """Independent of the probe: the model's own total on an other-only batch at lambda 0, and its gradient on the T-only parameters, with the control."""
    model, cfg = _model(rig.J)
    tw = TR.tac_src_stage(rig.b_other, rig.src)
    b = dict(rig.b_other)
    b["tac_w"] = tw
    torch.manual_seed(3)
    T = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(1))["terms"]
    assert float(T["goal"]) == 0.0 and float(T["lat"]) == 0.0 and float(T["lon"]) == 0.0 and float(T["cons"]) == 0.0
    # on the MIXED batch lambda 0 is NOT the unweighted mean (the lever moves the term)
    bm = dict(rig.b_mixed)
    bm["tac_w"] = TR.tac_src_stage(rig.b_mixed, rig.src)
    torch.manual_seed(3)
    Tm = model.loss(bm, p_gt=0.5, gen=torch.Generator().manual_seed(1))["terms"]
    assert Tm["lat"].item() > 0.0
    Tabs = _loss_absent(model, rig.b_mixed)
    assert abs(float(Tm["lat"]) - float(Tabs["lat"])) > 1e-6


# ------------------------------------------------------------------------------------------------ 3.4 RED arms: MODEL mutations (a mutated COPY of v8f_mini.py)
MODEL_MUTATIONS = [
    # (id, check that must fire, old, new)
    ("c1_upper_identity_epsilon", 1, 'w = w * rw[:, None]', 'w = w * (rw[:, None] + 1e-3 * torch.arange(rw.shape[0], device=rw.device)[:, None])'),
    ("c2_lat_call_loses_rw", 2, 'T["lat"] = self._partial_ce(out["lat_logits"], lt["lat_al"], lt["has"], rw)', 'T["lat"] = self._partial_ce(out["lat_logits"], lt["lat_al"], lt["has"])'),
    ("c2_lon_call_loses_rw", 2, 'T["lon"] = self._partial_ce(out["lon_logits"], lt["lon_al"], lt["has"], rw)', 'T["lon"] = self._partial_ce(out["lon_logits"], lt["lon_al"], lt["has"])'),
    ("c2_goal_loses_rw", 2, 'w = w * rw[:, None]', 'w = w'),
    ("c2_cons_loses_rw", 2, 'mf = m.float() * rw[:, None]', 'mf = m.float()'),
    ("c3_switch_not_weight", 3, 'rw = batch.get("tac_w")  ', 'rw = batch.get("tac_w")\n        rw = None if rw is None else (rw > 0.5).float()  '),
    ("c4_rw_reaches_S_mask", 4, 'ok = (ls["rc_ok"] & ls["has"][:, None]).float()', 'ok = (ls["rc_ok"] & ls["has"][:, None]).float() * (1.0 if batch.get("tac_w") is None else batch["tac_w"][:, None])'),
]


@pytest.mark.parametrize("mid,check,old,new", MODEL_MUTATIONS, ids=[m[0] for m in MODEL_MUTATIONS])
def test_model_mutation_turns_its_check_RED(rig, tmp_path, mid, check, old, new):
    mod = _mutated_model_module(tmp_path, old, new)
    with pytest.raises(AssertionError, match=f"check {check}"):
        _probe(rig, mod, checks=(check,))
    assert _probe(rig, M, checks=(check,)).get("pass") is True                    # control: the unmutated model is GREEN on the same check, same batches


def test_check5_mutation_inverted_source_turns_RED(rig, monkeypatch):
    def inverted(b, src):
        full = src.full_ci[b["ci"].cpu()]
        return torch.where(full, torch.full(full.shape, float(src.lam)), torch.ones(full.shape))        # torch.where(full, lambda, 1.0): the source is inverted
    monkeypatch.setattr(TR, "tac_src_stage", inverted)
    with pytest.raises(AssertionError, match="check 5"):
        _probe(rig, M, checks=(5,))
    with pytest.raises(AssertionError, match="check 2"):                           # and the lower identity sees it too
        _probe(rig, M, checks=(2,))
    monkeypatch.undo()
    assert _probe(rig, M, checks=(5,)).get("pass") is True


def test_the_probe_refuses_a_lever_with_no_T_weight(rig):
    model, cfg = _model(rig.J, w_tac=0.0)
    with pytest.raises(AssertionError, match="w_tac"):
        TR.tac_src_probe(model, cfg, rig.src, rig.b_mixed, rig.b_listed, rig.b_other)


# ------------------------------------------------------------------------------------------------ make_tac_src: the refusals (G-RM4, G-RM5, G-RM6 and the rest of the card's list)
def _refuses(rig, tmp_path, argv_extra, ids=None, msg="ZZDVB-REFUSEDZZ"):
    a = _args(argv_extra)
    with pytest.raises(SystemExit) as e:
        TR.make_tac_src(a, rig.J, rig.names, rig.wins, {})
    assert msg in str(e.value), str(e.value)


def test_default_argv_builds_no_stage(rig):
    a = _args([])
    assert (a.tac_src_w, a.tac_src_clips, a.tac_src_clips_sha256) == (1.0, "", "")
    assert TR.make_tac_src(a, rig.J, rig.names, rig.wins, {}) is None


def test_refusals(rig, tmp_path):
    p, sha = rig.path, rig.sha
    f = lambda *x: list(x)                                                           # noqa: E731
    _refuses(rig, tmp_path, f("--tac-src-w", "0.0"))                                                                # G-RM5: lambda 0 with no list
    _refuses(rig, tmp_path, f("--tac-src-w", "1.0", "--tac-src-clips", p, "--tac-src-clips-sha256", sha))           # G-RM6: a list that does nothing
    _refuses(rig, tmp_path, f("--tac-src-clips", p, "--tac-src-clips-sha256", sha, "--tac-src-w", "1.5"))
    _refuses(rig, tmp_path, f("--tac-src-clips", p, "--tac-src-clips-sha256", sha, "--tac-src-w", "-0.1"))
    _refuses(rig, tmp_path, f("--tac-src-clips", p, "--tac-src-clips-sha256", sha, "--tac-src-w", "nan"))
    _refuses(rig, tmp_path, f("--tac-src-clips", p, "--tac-src-clips-sha256", sha[:-1] + ("0" if sha[-1] != "0" else "1"), "--tac-src-w", "0.0"))     # sha differs
    _refuses(rig, tmp_path, f("--tac-src-clips", p, "--tac-src-w", "0.0"))                                          # a list with no declared sha
    _refuses(rig, tmp_path, f("--tac-src-clips", p + ".nope", "--tac-src-clips-sha256", sha, "--tac-src-w", "0.0"))
    one_less, sha1 = _list_file(tmp_path, rig.listed[:-1], "one_less.txt")                                          # G-RM4: a list with one clip removed (its own sha is declared: it parses, so it is the sha rule of the ARGV that refuses the original)
    _refuses(rig, tmp_path, f("--tac-src-clips", one_less, "--tac-src-clips-sha256", sha, "--tac-src-w", "0.0"))
    foreign, shaf = _list_file(tmp_path, rig.listed[:2] + ["not-a-train-clip"], "foreign.txt")
    _refuses(rig, tmp_path, f("--tac-src-clips", foreign, "--tac-src-clips-sha256", shaf, "--tac-src-w", "0.0"), msg="not in the train clips")
    every, shae = _list_file(tmp_path, rig.names, "all.txt")
    _refuses(rig, tmp_path, f("--tac-src-clips", every, "--tac-src-clips-sha256", shae, "--tac-src-w", "0.0"), msg="some and not all")
    none, shan = _list_file(tmp_path, [], "none.txt")
    _refuses(rig, tmp_path, f("--tac-src-clips", none, "--tac-src-clips-sha256", shan, "--tac-src-w", "0.0"), msg="some and not all")
    dup, shad = _list_file(tmp_path, rig.listed + rig.listed[:1], "dup.txt")
    _refuses(rig, tmp_path, f("--tac-src-clips", dup, "--tac-src-clips-sha256", shad, "--tac-src-w", "0.0"), msg="more than once")
    ok = TR.make_tac_src(_args(["--tac-src-w", "0.0", "--tac-src-clips", one_less, "--tac-src-clips-sha256", sha1]), rig.J, rig.names, rig.wins, {})      # control: a well-formed shorter list builds
    assert ok.n_listed_clips == 2


def test_gdvb_reads_the_flags_back_from_the_built_stage(rig):
    assert TR.bad_tac_src(rig.a, rig.src, rig.info) == []
    assert {"tac_src_w", "tac_src_clips", "tac_src_clips_sha256"} <= set(TR.FLAG_CHECKS)
    for flag, val in (("tac_src_w", 0.25), ("tac_src_clips", rig.path + "x"), ("tac_src_clips_sha256", "0" * 64)):
        a = _args(["--tac-src-w", "0.0", "--tac-src-clips", rig.path, "--tac-src-clips-sha256", rig.sha])
        setattr(a, flag, val)
        assert TR.bad_tac_src(a, rig.src, rig.info), flag
    a = _args(["--tac-src-w", "0.0"])                                                # declared, nothing built
    assert TR.bad_tac_src(a, None, {})
    assert TR.bad_tac_src(_args([]), None, {}) == []
    for extra in (["--wm-only"], ["--encoder", "lora"]):
        a = _args(["--tac-src-w", "0.0", "--tac-src-clips", rig.path, "--tac-src-clips-sha256", rig.sha, *extra])
        assert TR.bad_tac_src(a, rig.src, rig.info), extra


# ------------------------------------------------------------------------------------------------ check 6 and 7, check G-CKPT: the TRAINER on the synthetic world
def _syn_list(tmp, seed=5, n_listed=3):
    names = D.make_synthetic_world(Path(tmp) / "namesw", n_train=6, n_eval=2, T=60, seed=seed)["names"]["train"]     # the names the trainer's own synthetic world (same seed) will carry
    return _list_file(tmp, sorted(names, key=D.sha12)[:n_listed])


def _train(out, extra=(), steps=6, mod=TR):
    return mod.main(["--out", str(out), *COMMON, "--steps", str(steps), *extra])


def _summary(out):
    return json.load(open(Path(out) / "summary.json", encoding="utf-8"))


def test_check6_loop_reach_summary_carries_the_K_TACSRC_fields(tmp_path):
    lst, sha = _syn_list(tmp_path)
    out = tmp_path / "on"
    assert _train(out, ["--tac-src-w", "0.0", "--tac-src-clips", lst, "--tac-src-clips-sha256", sha]) == 0
    s = _summary(out)
    assert s["tac_src_w"] == 0.0 and s["tac_src_list_sha256"] == sha
    assert s["tac_src_rows_other"] > 0 and s["tac_src_rows_listed"] > 0
    assert s["tac_src_rows_listed"] + s["tac_src_rows_other"] == 6 * 16                      # every row of every step is counted once
    assert s["tac_src_weight_share_listed"] == 1.0                                          # exactly 1.0 at lambda 0
    assert s["tac_src_empty_batches"] <= 5
    sc = json.load(open(out / "selfcheck.json", encoding="utf-8"))
    assert sc["tac_src_probe"]["pass"] is True and sc["tac_src"]["list_sha256"] == sha


def _mutated_trainer(tmp, old, new, name):
    """A COPY of v8f_mini_train.py laid out as <tmp>/stack/scripts/ (its REPO resolves to <tmp>), with ``old`` (exactly one occurrence) replaced by ``new``; loaded under its own name."""
    root = Path(tmp) / f"mut_{name}"
    (root / "stack" / "scripts").mkdir(parents=True)
    src = Path(TR.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, (src.count(old), old)
    path = root / "stack" / "scripts" / "v8f_mini_train.py"
    path.write_text(src.replace(old, new), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"v8f_mini_train_mut_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_check6_RED_when_the_attach_line_is_deleted_the_loop_guard_exits(tmp_path):
    lst, sha = _syn_list(tmp_path)
    mod = _mutated_trainer(tmp_path, '                    b["tac_w"] = tac_w\n', '                    pass\n', "attach")
    with pytest.raises(SystemExit, match="ZZTACSRC-NOT-APPLIEDZZ"):
        _train(tmp_path / "attach", ["--tac-src-w", "0.0", "--tac-src-clips", lst, "--tac-src-clips-sha256", sha], mod=mod)
    assert _train(tmp_path / "attach_ok", ["--tac-src-w", "0.0", "--tac-src-clips", lst, "--tac-src-clips-sha256", sha]) == 0        # control: the unmutated trainer runs the same argv


def test_check6_RED_when_the_stage_is_not_called_the_counts_read_zero_or_the_run_dies(tmp_path):
    lst, sha = _syn_list(tmp_path)
    mod = _mutated_trainer(tmp_path, '                    tac_src_count(tac_tot, b, tsrc, tac_w)\n', '                    pass\n', "count")
    assert _train(tmp_path / "nocount", ["--tac-src-w", "0.0", "--tac-src-clips", lst, "--tac-src-clips-sha256", sha], mod=mod) == 0
    assert _summary(tmp_path / "nocount")["tac_src_rows_other"] == 0                          # the K-TACSRC reading `tac_src_rows_other > 0` would refuse this arm
    assert _summary(tmp_path / "nocount")["tac_src_weight_share_listed"] is None


def test_check7_default_off_builds_no_stage_and_the_summary_keeps_its_key_set(tmp_path):
    out = tmp_path / "off"
    assert _train(out) == 0
    s = _summary(out)
    assert not any(k.startswith("tac_src") for k in s)
    sc = json.load(open(out / "selfcheck.json", encoding="utf-8"))
    assert "tac_src_probe" not in sc and "tac_src" not in sc
    blob = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert "tac_src" not in blob and "tac_src_totals" not in blob


def test_check7_RED_when_the_default_of_tac_src_w_is_zero_the_trainer_refuses(tmp_path):
    mod = _mutated_trainer(tmp_path, 'ap.add_argument("--tac-src-w", dest="tac_src_w", type=float, default=1.0,', 'ap.add_argument("--tac-src-w", dest="tac_src_w", type=float, default=0.0,', "default")
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ"):
        _train(tmp_path / "def0", mod=mod)


def test_gckpt_resume_is_bit_equal_and_the_counts_are_carried(tmp_path, monkeypatch):
    lst, sha = _syn_list(tmp_path)
    extra = ["--tac-src-w", "0.0", "--tac-src-clips", lst, "--tac-src-clips-sha256", sha, "--ckpt-every", "3"]
    a_out, b_out = tmp_path / "unint", tmp_path / "crash"
    assert _train(a_out, extra) == 0
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "4")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        _train(b_out, extra)
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert _train(b_out, extra) == 0
    sa, sb = _summary(a_out), _summary(b_out)
    for k in ("tac_src_rows_listed", "tac_src_rows_other", "tac_src_weight_share_listed", "tac_src_empty_batches"):
        assert sa[k] == sb[k], k
    ca = torch.load(a_out / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    cb = torch.load(b_out / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    assert ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)
