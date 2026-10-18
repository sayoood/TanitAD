"""RA-1 (refav1-HJ mini): arms, launch gate, instrument controls, resume, eval + verdict end to end.  CPU only (asserted), < 3 min.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_ra1_hjepa.py

What is pinned, and the control that must go red for each:
  * G-HYG: a flag whose default disables its mechanism must be named (A1 w/o --gamma, A0 with --gamma, R0 with --beta, --hist-drop
    outside A1h, a window length that does not end on a level boundary).
  * EVERY arm (R0 A0 A1 A2 A1x A1h) trains 3 steps on CPU through the real trainer, passes its own launch gate, and an eval of the
    result passes every instrument control and the post-training audit (no parameter without Adam state).
  * the IDM term: present with a NON-ZERO gradient into the adapter in A1, EXACTLY 0.0 (value and gradient) in A0.
  * MUTATION: deleting the IDM term (gamma := 0 after the build) turns the G-LIVE report and the G-DVB reader RED.
  * A1x: the frame permutation is non-identity in EVERY window, is what the world-model loss actually receives, and leaves the
    readout path untouched.
  * A1h: the history REALLY is dropped (p = 1: position i is independent of every earlier latent/action and equals the context-1
    prediction; p = 0: bit-identical to the plain predictor and DEPENDENT on history; eval: identity), adds no parameter, and the
    flag changes the loss of a model that is otherwise identical.
  * SENS: the self-swap control reads EXACTLY 0.0; a real donor reads > 0; per_window means equal the float and the default call is unchanged.
  * units: the P10 trajectory the trainer uses is the STEER-unit kd_x (it differs from the kappa-unit one).
  * sampling is a pure function of (seed, step); resume after a crash == the uninterrupted run, bit for bit.
  * HOLDOUT / SELECTION (PREREG_HJ_STAGE0 Amendment 1): the 60-of-600 train holdout is a function of the manifest only, identical for
    every arm and n_clips, disjoint from what every arm trains on (spied at ClipBank.load / sample_batch -- the REAL caller -- and a
    split that does not exclude it turns that spy RED); the trainer REFUSES to run without --holdout-n (and with 0); the evaluator
    re-derives the holdout and refuses a drifted checkpoint; `verdict` selects the A1 cell on the HOLDOUT ADE@6s, REFUSES a cell with
    no holdout numbers (no fallback to eval-141), and a source-level MUTATION back to eval-141 selection turns the synthetic
    holdout-winner != eval-winner case RED; the box-2 queue names --holdout-n 60 in every trainer argv.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

# CPU-only by design (the dev-box GPU belongs to other jobs). A module-level SKIP, not an assert: an assert breaks
# collection of the whole suite on any machine where a GPU is visible (the lander's import probe caught it).
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1",
    reason="RA-1 tests are CPU only: run with CUDA_VISIBLE_DEVICES=-1")

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import ra1_pooled as RD  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import ra1_hjepa as R  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402
import ra1_hjepa_eval as EV  # noqa: E402
import ra1_hjepa_train as TR  # noqa: E402

torch.set_num_threads(2)
# (CPU-only is enforced by pytestmark above; an import-time assert would break suite collection where a GPU is visible.)

TINY_FLAGS = ["--d-tok", "4", "--adapter-hidden", "16", "--d1", "16", "--d-up", "16", "--macro-dim", "8", "--enc-hidden", "16",
              "--pred-width", "16", "--pred-depth", "1", "--pred-heads", "2", "--readout-hidden", "16", "--sigreg-proj", "8",
              "--batch", "4", "--warmup", "1", "--log-every", "1", "--ckpt-every", "3"]
ARM_FLAGS = {"R0": [], "A0": ["--beta", "0.08"], "A1": ["--beta", "0.08", "--gamma", "100"], "A2": ["--beta", "0.08", "--gamma", "100"],
             "A1x": ["--beta", "0.08", "--gamma", "100"], "A1h": ["--beta", "0.08", "--gamma", "100", "--hist-drop", "0.5"]}
# PREREG Amendment 1: --holdout-n is REQUIRED by the trainer. The synthetic world has 6 train clips, so the pool shrinks to 6 and the
# holdout to 2 (the real queue passes --holdout-n 60 against the 600-clip pool: pinned by test_queue_*).
HOLD_FLAGS = ["--holdout-n", "2", "--holdout-pool", "6"]
TINY_CFG = dict(d_feat=32, n_tok=40, d_tok=4, adapter_hidden=16, d1=16, d_up=16, macro_dim=8, enc_hidden=16, pred_width=16, pred_depth=1,
                pred_heads=2, readout_hidden=16, sigreg_proj=8)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ra1syn")
    w = D.make_synthetic_world(root, n_train=6, n_eval=3, T=70, seed=0, d_feat=32)
    prior = M.P10Prior.synthetic(0)
    M.save_p10(prior, root / "p10_prior.pt", {"synthetic": True})
    w["prior_path"] = str(root / "p10_prior.pt")
    w["prior"] = M.load_p10(w["prior_path"])
    w["grid_json"] = str(root / "eval_grid_windows.json")
    return w


@pytest.fixture(scope="module")
def bank(world):
    b = RD.ClipBank(world["root"], "train", world["prior"], "4x10")
    b.load(world["names"]["train"])
    return b


def cfg_of(arm, **kw):
    base = dict(TINY_CFG, arm=arm)
    if arm != "R0":
        base["beta"] = 0.08
    if arm in ("A1", "A2", "A1x", "A1h"):
        base["gamma"] = 100.0
    if arm == "A1h":
        base["hist_drop"] = 0.5
    base.update(kw)
    return R.RA1Config(**base)


def model_of(arm, **kw):
    torch.manual_seed(0)
    return R.build_model(cfg_of(arm, **kw), torch.zeros(3), torch.ones(3))


def a_batch(bank, world, step=0, b=4):
    return RD.sample_batch(bank, world["names"]["train"], 0, step, b, 19)


def run_train(world, arm, out, seed=0, steps=3, extra=()):
    # these tests exercise the LEGACY steer representation explicitly (PI ruling D-PI-YAWRATE-1: the trainer's default is now `yaw`);
    # the compliant mode is covered by test_yaw_rate_ruling.py
    argv = ["--arm", arm, "--seed", str(seed), "--cache", world["root"], "--p10-prior", world["prior_path"], "--out", str(out),
            "--steps", str(steps), "--kin-mode", "steer"] + TINY_FLAGS + ARM_FLAGS[arm] + HOLD_FLAGS + list(extra)
    return TR.main(argv)


# ------------------------------------------------------------------------------------------------ G-HYG
@pytest.mark.parametrize("arm,kw,frag", [
    ("A1", dict(gamma=None), "gamma"), ("A2", dict(gamma=0.0), "gamma"), ("A0", dict(gamma=100.0), "SIGReg-only"),
    ("A0", dict(beta=None), "beta"), ("R0", dict(beta=0.08), "R0 has no world model"), ("R0", dict(gamma=100.0), "R0 has no world model"),
    ("A1", dict(hist_drop=0.5), "hist-drop"), ("A1h", dict(hist_drop=0.0), "hist-drop"), ("A1", dict(t1=18), "t1"),
    ("A2", dict(t1=13), "level-3 window boundary"), ("A1", dict(beta=0.0), "beta")])
def test_config_hygiene_refuses_unnamed_or_unreachable_flags(arm, kw, frag):
    base = {"A0": dict(beta=0.08), "A1": dict(beta=0.08, gamma=100.0), "A2": dict(beta=0.08, gamma=100.0),
            "A1h": dict(beta=0.08, gamma=100.0, hist_drop=0.5), "R0": {}}[arm]
    base = dict(TINY_CFG, arm=arm, **{**base, **kw})
    with pytest.raises(ValueError) as e:
        R.validate_config(R.RA1Config(**base))
    assert frag in str(e.value)


# ------------------------------------------------------------------------------------------------ every arm, the real trainer
@pytest.mark.parametrize("arm", list(R.ARMS))
def test_every_arm_trains_three_steps_on_cpu_and_passes_its_gate(world, tmp_path, arm):
    out = tmp_path / arm
    assert run_train(world, arm, out) == 0
    log = [json.loads(l) for l in open(out / "train_log.jsonl")]
    assert len(log) == 3 and all(np.isfinite(r["total"]) and np.isfinite(r["traj"]) for r in log)
    gate = json.load(open(out / "launch_check.json"))
    assert gate["ok"] and gate["G_LIVE"]["ok"] and all(r["ok"] for r in gate["G_DVB"])
    assert gate["vram_estimate"]["estimate_gb"] > 0
    assert (out / "final.pt").exists() and (out / "DONE").exists()
    if arm == "A1h":
        assert 0.0 < json.load(open(out / "DONE"))["hist_drop_fraction"] < 1.0


# ------------------------------------------------------------------------------------------------ the IDM term
def test_idm_term_is_live_in_A1_and_exactly_zero_in_A0(bank, world):
    batch = a_batch(bank, world)
    r1 = R.live_report(model_of("A1"), batch)
    assert r1["ok"] and r1["idm_term_grad_into_adapter"] > 0.0
    assert r1["terms"]["idm_term/L1"] > 0.0 and r1["terms"]["idm_term/L2"] > 0.0 and r1["unexpected_dead"] == []
    m0 = model_of("A0")
    r0 = R.live_report(m0, batch)
    assert r0["ok"], r0
    assert r0["idm_term_grad_into_adapter"] == 0.0 and r0["terms"]["idm_term/L1"] == 0.0 and r0["terms"]["idm_term/L2"] == 0.0
    assert r0["terms"]["idm/L1"] > 0.0, "the (unweighted) head output is still computed in A0; only its WEIGHT is zero"
    assert all(n.startswith("stack.idm_heads.") for n in r0["expected_dead"]) and r0["expected_dead"]


def test_mutation_deleting_the_idm_term_goes_red(bank, world):
    batch = a_batch(bank, world)
    m = model_of("A1")
    assert R.live_report(m, batch)["ok"] and all(r["ok"] for r in R.declared_vs_built(m.cfg, m))
    for lv in m.stack.cfg:                                  # DELETE the IDM term from the objective
        lv.gamma = 0.0
    live = R.live_report(m, batch)
    assert not live["ok"] and not live["idm_ok"] and live["idm_term_grad_into_adapter"] == 0.0
    dvb = R.declared_vs_built(m.cfg, m)
    assert any((not r["ok"]) and r["lever"].startswith("gamma_L") for r in dvb)


# ------------------------------------------------------------------------------------------------ A1x
def test_a1x_permutation_is_nonidentity_in_every_window_and_reaches_the_wm_loss(bank, world):
    for t in (2, 3, 4, 19):
        p = R.time_shuffle_perm(256, t, torch.Generator().manual_seed(t))
        assert bool((p != torch.arange(t)[None]).any(1).all()), "an identity row would be an A1 window"
        assert bool((p.sort(1).values == torch.arange(t)[None]).all()), "rows must be permutations"
    batch = a_batch(bank, world)
    m1, mx = model_of("A1"), model_of("A1x")
    mx.load_state_dict(m1.state_dict())
    seen = {}
    for nm, m in (("A1", m1), ("A1x", mx)):
        orig = m.stack.loss
        m.stack.loss = (lambda z, a, _o=orig, _n=nm, **kw: (seen.__setitem__(_n, z.detach().clone()), _o(z, a, **kw))[1])
    z1 = m1.adapter(batch["tokens"].float()).detach()
    o1 = m1.forward_batch(batch, generator=torch.Generator().manual_seed(0), shuffle_generator=torch.Generator().manual_seed(0))
    ox = mx.forward_batch(batch, generator=torch.Generator().manual_seed(0), shuffle_generator=torch.Generator().manual_seed(0))
    assert torch.allclose(seen["A1"], z1)
    assert not torch.allclose(seen["A1x"], z1)
    assert torch.allclose(seen["A1x"].sort(1).values, z1.sort(1).values, atol=1e-6), "same frames, shuffled order"
    assert ox["shuffle_nonidentity_frac"] > 0.5
    assert torch.equal(o1["traj"], ox["traj"]), "the readout path must be untouched by the shuffle"
    assert not torch.allclose(o1["pred/L1"], ox["pred/L1"]) and not torch.equal(o1["idm/L1"], ox["idm/L1"])


# ------------------------------------------------------------------------------------------------ A1h
def _tiny_pred():
    torch.manual_seed(1)
    return H.CausalPredictor(8, 3, width=16, depth=2, heads=2, context=3).eval()


def test_a1h_history_is_really_dropped():
    inner = _tiny_pred()
    z, a = torch.randn(5, 3, 8), torch.randn(5, 3, 3)
    plain = inner(z, a)
    w = R.HistoryDropPredictor(inner, 1.0)
    w.train()
    w.generator = torch.Generator().manual_seed(0)
    full_drop = w(z, a)
    for i in range(3):
        solo = inner(z[:, i:i + 1], a[:, i:i + 1])[:, 0]
        assert torch.allclose(full_drop[:, i], solo, atol=1e-6), "p=1: every position equals its context-1 prediction"
    z2, a2 = z.clone(), a.clone()
    z2[:, :2] += 3.0
    a2[:, :2] -= 2.0
    assert torch.allclose(w(z2, a2)[:, 2], full_drop[:, 2], atol=1e-6), "p=1: position 2 is blind to positions 0-1"
    assert not torch.allclose(plain[:, 2], inner(z2, a2)[:, 2], atol=1e-4), "control: the plain predictor DOES use its history"
    w0 = R.HistoryDropPredictor(inner, 0.0).train()
    assert torch.equal(w0(z, a), plain), "p=0 is the plain predictor bit for bit"
    w.eval()
    assert torch.equal(w(z, a), plain), "eval is the identity (nothing is dropped at inference)"


def test_a1h_adds_no_parameter_and_the_flag_changes_the_loss(bank, world):
    m1, mh = model_of("A1"), model_of("A1h", hist_drop=1.0)
    assert sum(p.numel() for p in m1.parameters()) == sum(p.numel() for p in mh.parameters())
    assert len([p for p in mh.stack.predictors if isinstance(p, R.HistoryDropPredictor)]) == mh.stack.n_levels
    sd = {k.replace("predictors.0.", "predictors.0.inner.").replace("predictors.1.", "predictors.1.inner."): v
          for k, v in m1.state_dict().items()}
    mh.load_state_dict(sd)
    batch = a_batch(bank, world)
    mh.set_generators(torch.Generator().manual_seed(0))
    oh = mh.forward_batch(batch, generator=torch.Generator().manual_seed(0), shuffle_generator=torch.Generator().manual_seed(0))
    o1 = m1.forward_batch(batch, generator=torch.Generator().manual_seed(0), shuffle_generator=torch.Generator().manual_seed(0))
    assert torch.equal(o1["traj"], oh["traj"]) and not torch.allclose(o1["pred/L1"], oh["pred/L1"])
    mp = model_of("A1h", hist_drop=0.5)
    mp.set_generators(torch.Generator().manual_seed(0))
    for s in range(6):
        mp.forward_batch(a_batch(bank, world, s), generator=torch.Generator().manual_seed(s))
    n_d = sum(p.n_dropped for p in mp.stack.predictors)
    n_p = sum(p.n_positions for p in mp.stack.predictors)
    assert 0.3 < n_d / n_p < 0.7
    # the dropped fraction is what the DONE file / gate report: p = 0.5 draws, positions with L == 1 are skipped


# ------------------------------------------------------------------------------------------------ SENS controls
def test_self_swap_reads_exactly_zero_and_a_real_donor_reads_nonzero(bank, world):
    m = model_of("A1")
    n = 12
    z = torch.randn(n, 16)
    fut = torch.randn(n, 10, 3) * 3.0 + 1.0
    eid = torch.arange(n) // 3
    sw = R.sensitivity_per_window(m, z, fut, eid, perm=torch.arange(n))
    assert all(v == 0.0 for k, v in sw.items() if not k.startswith("win/")), "self-swap control"
    assert all(float(np.abs(v).max()) == 0.0 for k, v in sw.items() if k.startswith("win/"))
    real = R.sensitivity_per_window(m, z, fut, eid)
    assert real["sens/all/k10"] > 0.0 and real["sens/steer/k10"] > 0.0 and real["sens/v/k10"] > 0.0
    for k, v in real.items():
        if k.startswith("win/"):
            assert abs(float(v.mean()) - real["sens/" + k[4:]]) < 1e-5, "per-window mean == the float"
    plain = m.stack.action_sensitivity(z, m.std_actions(fut), level=1, ks=(1, 5, 10), episode_ids=eid)
    assert not any(k.startswith("win/") for k in plain), "default call unchanged"


def test_identity_predictor_reads_zero_through_the_ra1_probe():
    m = model_of("A1")
    ident = torch.nn.Module()
    ident.context = 3
    ident.forward = lambda z, a: z
    m.stack.predictors[0] = ident
    n = 8
    r = R.sensitivity_per_window(m, torch.randn(n, 16), torch.randn(n, 10, 3), torch.arange(n) // 2)
    assert all(v == 0.0 for k, v in r.items() if k.startswith("sens/"))


# ------------------------------------------------------------------------------------------------ COST known values
def test_cost_windows_known_values():
    T, D_ = 80, 4
    t = np.arange(0, 40, 5)
    ramp = torch.zeros(T, D_)
    ramp[:, 0] = torch.arange(T).float()      # the latent walks steadily: distance to the +30 goal falls strictly with time
    c = R.cost_windows([ramp, ramp.clone(), ramp.clone()], t)
    for k in ("L1_native", "L2_native", "L3_native", "L1_matched_s3"):
        assert np.allclose(c[k], 1.0), k
    # control: a latent with NO progress structure (iid noise) reads ~0, so a high value is not a property of the probe
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(T, D_, generator=g)
    cn = R.cost_windows([noise, noise], t)
    assert abs(float(cn["L1_native"].mean())) < 0.5 and (cn["L1_native"] < 1.0).all()


# ------------------------------------------------------------------------------------------------ data
def test_steer_unit_prior_trajectory_differs_from_kappa_unit(world):
    c = world["names"]["train"][0]
    d = RD.load_clip(Path(world["root"]) / "train" / "clips" / f"{c}.pt", world["prior"], "4x10")
    o = torch.load(Path(world["root"]) / "train" / "clips" / f"{c}.pt", weights_only=True)
    kin3 = o["kin3"].double().numpy()
    base_steer = D.kdx_base(kin3)
    ha0 = fk.ha0_paths(np.where(np.isfinite(kin3), kin3, 0)[:, 0], D.DT, D.HORIZON)
    kd_kappa = fk.kd_x(ha0, fk.ha0_ext_paths(np.where(np.isfinite(kin3), kin3, 0), D.DT, D.HORIZON, action_units="kappa")).astype(np.float32)
    ref = world["prior"](torch.nan_to_num(o["p10"].float()), torch.from_numpy(base_steer))
    assert torch.allclose(d["p10_traj"], ref)
    assert float(np.abs(base_steer[5:40] - kd_kappa[5:40]).max()) > 1e-2, "the two units must give different paths on a steering clip"
    v = d["valid"]
    assert not bool(v[:3].any()) and not bool(v[-30:].any()) and bool(v[3:d["T"] - 30].all())


def test_sampling_is_a_pure_function_of_seed_and_step(bank, world):
    names = world["names"]["train"]
    a = RD.sample_batch(bank, names, 3, 11, 4, 19)
    b = RD.sample_batch(bank, names, 3, 11, 4, 19)
    c = RD.sample_batch(bank, names, 3, 12, 4, 19)
    assert all(torch.equal(a[k].float(), b[k].float()) for k in a)
    assert not torch.equal(a["tokens"].float(), c["tokens"].float())
    assert a["tokens"].dtype == torch.float8_e4m3fn and a["tokens"].shape == (4, 19, 40, 32)
    assert bool(a["valid"].any(1).all()) and bool(torch.isfinite(a["gt"][a["valid"]]).all())
    assert RD.chunk_clips(names, 0, 7, 3, 5) == RD.chunk_clips(names, 0, 9, 3, 5), "same chunk"
    assert len(RD.chunk_clips(names, 0, 7, 3, 5)) == 3 and RD.chunk_clips(names, 0, 7, 0, 5) == list(names)


def test_strata_labels_known_values():
    pose = torch.zeros(60, 4)
    pose[:, 3] = 10.0
    assert not any(RD.strata_labels(pose, 5).values())
    pose2 = pose.clone()
    pose2[5 + 30:, 2] = 0.5
    assert RD.strata_labels(pose2, 5)["turn"]
    pose3 = pose.clone()
    pose3[5 + 20:, 3] = 0.0
    assert RD.strata_labels(pose3, 5)["stop"]


# ------------------------------------------------------------------------------------------------ resume
def test_resume_after_a_crash_equals_the_uninterrupted_run(world, tmp_path, monkeypatch):
    run_train(world, "A1h", tmp_path / "full", steps=6)
    real = RD.sample_batch

    def boom(bank, names, seed, step, *a, **k):
        if step == 4:
            raise RuntimeError("simulated crash")
        return real(bank, names, seed, step, *a, **k)

    monkeypatch.setattr(RD, "sample_batch", boom)
    with pytest.raises(RuntimeError):
        run_train(world, "A1h", tmp_path / "crash", steps=6)
    monkeypatch.setattr(RD, "sample_batch", real)
    assert torch.load(tmp_path / "crash" / "latest.pt", weights_only=False)["step"] == 3
    run_train(world, "A1h", tmp_path / "crash", steps=6)
    a = torch.load(tmp_path / "full" / "final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "crash" / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a), "resume must reproduce the uninterrupted run exactly"
    with pytest.raises(SystemExit, match="different configuration"):  # a changed configuration is refused
        TR.main(["--arm", "A1h", "--seed", "5", "--cache", world["root"], "--p10-prior", world["prior_path"], "--out",
                 str(tmp_path / "crash"), "--steps", "6", "--kin-mode", "steer"] + TINY_FLAGS + ARM_FLAGS["A1h"] + HOLD_FLAGS)


def test_post_training_audit_names_a_parameter_that_never_got_a_gradient(world, tmp_path):
    run_train(world, "A1", tmp_path / "a", steps=3)
    ck = torch.load(tmp_path / "a" / "final.pt", weights_only=False)
    assert EV.untrained_params(ck) == []
    victim = ck["param_names"][5]
    del ck["opt"]["state"][5]                       # control: a parameter whose Adam state was never allocated
    assert EV.untrained_params(ck) == [victim]


def test_launch_gate_refuses_a_silent_a0(world, tmp_path):
    assert TR.main(["--arm", "A1", "--cache", world["root"], "--p10-prior", world["prior_path"], "--out", str(tmp_path / "x"),
                    "--steps", "2", "--beta", "0.08", "--kin-mode", "steer"] + TINY_FLAGS + HOLD_FLAGS) == 2      # --gamma unnamed => REFUSED, no training
    assert not (tmp_path / "x" / "final.pt").exists()


# ------------------------------------------------------------------------------------------------ eval + verdict, end to end
def test_eval_and_verdict_end_to_end(world, tmp_path):
    plan = [("R0", 0), ("R0", 1), ("A0", 0), ("A0", 1), ("A1", 0), ("A1", 1), ("A2", 0), ("A2", 1), ("A1x", 0), ("A1h", 0)]
    cell = {"R0": "", "A0": "b0.08", "A1": "b0.08g100", "A2": "b0.08g100", "A1x": "b0.08g100", "A1h": "b0.08g100"}
    runs = []
    for arm, s in plan:
        run_train(world, arm, tmp_path / f"{arm}_{s}", seed=s)
        ev = tmp_path / f"{arm}_{s}" / "eval"
        rc = EV.main(["eval", "--ckpt", str(tmp_path / f"{arm}_{s}" / "final.pt"), "--eval-cache", world["root"], "--p10-prior",
                      world["prior_path"], "--grid-json", world["grid_json"], "--train-cache", world["root"], "--out", str(ev),
                      "--n-boot", "40", "--fam-boot", "20"])
        assert rc == 0, f"{arm}@{s}: an instrument control failed"
        r = json.load(open(ev / "eval.json"))
        assert r["instrument_ok"] and r["controls"]["p10_minus_p10_ok"] and r["controls"]["gt_ade_max"] == 0.0
        assert r["param_audit"]["untrained"] == [], f"{arm}: parameters that never received a gradient"
        assert r["K"]["ade6_arm_minus_p10"]["estimator"].startswith("paired_")
        assert r["holdout"]["status"] == "OK" and r["holdout"]["disjoint_from_train"] and r["holdout"]["disjoint_from_eval141"]
        assert r["holdout"]["n_clips"] == 2 and r["holdout"]["n_windows"] > 0 and r["holdout"]["ade6_arm"]["n_episodes"] == 2
        assert r["families"]["arm"]["strategic"]["status"] == "PAUSED" and "longitudinal" in r["families"]["arm"]
        if arm == "R0":
            assert r["SENS"]["status"].startswith("n/a")
        else:
            assert r["SENS"]["self_swap_control_ok"] and r["SENS"]["self_swap_control_max"] == 0.0
            assert r["SENS"]["tier"].startswith("T0") and "L2_minus_L1_native" in r["COST"]
            assert r["SENS"]["channels"]["steer/k10"]["mean_of_ratios"] > 0.0
        if arm == "A2":
            assert "L3_native" in r["COST"]
        runs += ["--run", f"{arm}{':' + cell[arm] if cell[arm] else ''}@{s}={ev}"]
    vout = tmp_path / "verdict.json"
    assert EV.main(["verdict", "--out", str(vout), "--n-boot", "40"] + runs) == 0
    v = json.load(open(vout))
    assert v["A1_best_cell"] == "b0.08g100" and v["branch"] and "A1x_control" in v and "A1h_EXPLORATORY" in v
    assert set(v["SENS"]["per_seed"]) == {"seed0", "seed1"} and v["K"]["best_arm"] in ("R0", "A1", "A2")
    assert v["training_seed_floor_A1"] is not None and v["COST"]["per_run"]
    assert v["selection_basis"].startswith("the withheld TRAIN holdout") and "NOT selected on eval-141" in v["A1_best_rule"]
    assert set(v["A1_cells_seed0_holdout_ade6"]) == set(v["A1_cells_seed0_eval141_ade6_NON_DECIDING"]) == {"b0.08g100"}
    assert v["holdout"]["n_clips"] == 2 and v["holdout"]["n_windows"] > 0 and len(v["holdout"]["ade6_mean_by_run"]) == len(plan)


def test_verdict_degrades_gracefully_on_partial_inputs(world, tmp_path):
    run_train(world, "A1", tmp_path / "A1_0")
    ev = tmp_path / "A1_0" / "eval"
    assert EV.main(["eval", "--ckpt", str(tmp_path / "A1_0" / "final.pt"), "--eval-cache", world["root"], "--p10-prior", world["prior_path"],
                    "--grid-json", world["grid_json"], "--train-cache", world["root"], "--out", str(ev), "--n-boot", "30",
                    "--fam-boot", "10"]) == 0
    assert EV.main(["verdict", "--out", str(tmp_path / "v.json"), "--n-boot", "30", "--run", f"A1:b0.08g100@0={ev}"]) == 0
    v = json.load(open(tmp_path / "v.json"))
    assert v["SENS"]["PASS"] is False and v["K"]["PASS"] is False and "NOT RUN" in v["A1x_control"]["verdict"]
    assert v["branch"].startswith("A1x GATE"), "with the instrument gate open no other branch may be selected"


# ================================================================================================ HOLDOUT / SELECTION (PREREG Amendment 1)
def _ids(n):
    return [f"clip-{i:05d}" for i in range(n)]


def test_holdout_is_the_last_60_of_the_600_in_sha12_order_and_independent_of_everything_else():
    pop = _ids(700)
    order = sorted(pop, key=D.sha12)
    h = RD.holdout_clips(pop, 60)
    assert RD.HOLDOUT_CLIPS == 60 and RD.STAGE_A_CLIPS == 600
    assert len(h) == 60 and h == order[540:600], "the LAST 60 of the first 600 in sha12 order"
    # the 600-clip stage-A cache is the nested sha12 prefix of the full corpus: the SAME 60 clips are withheld at 600 and at 4,572
    assert RD.holdout_clips(order[:600], 60) == h
    assert RD.holdout_clips(list(reversed(pop)), 60) == h, "a function of the clip SET: manifest order is irrelevant"
    for n_clips in (0, 100, 540, 560, 600, 650, 700):
        tr, hold = RD.split_train_holdout(pop, n_clips, 60)
        assert hold == h, f"the holdout must not depend on --n-clips ({n_clips})"
        assert not (set(tr) & set(h)), f"n_clips={n_clips}: a training clip is in the holdout"
        assert tr == [c for c in RD.pick_train_clips(pop, n_clips) if c not in set(h)]
    assert len(RD.split_train_holdout(pop, 0, 60)[0]) == 640 and len(RD.split_train_holdout(pop, 600, 60)[0]) == 540
    assert len(RD.split_train_holdout(pop, 540, 60)[0]) == 540, "a prefix that stops before the holdout loses nothing"


def test_holdout_refusals():
    pop = _ids(700)
    for n, frag in ((0, "holdout_n must be >= 1"), (-3, "holdout_n must be >= 1"), (600, "smaller than the stage-A pool")):
        with pytest.raises(ValueError, match=frag):
            RD.holdout_clips(pop, n)
    with pytest.raises(ValueError, match="fewer than the 600-clip"):
        RD.holdout_clips(_ids(599), 60)
    assert RD.clips_sha256(["a", "b"]) != RD.clips_sha256(["b", "a"]) and len(RD.clips_sha256([])) == 64


def _train_argv(world, arm, out, extra):
    return ["--arm", arm, "--seed", "0", "--cache", world["root"], "--p10-prior", world["prior_path"], "--out", str(out),
            "--steps", "2", "--kin-mode", "steer"] + TINY_FLAGS + ARM_FLAGS[arm] + list(extra)


def test_trainer_refuses_to_run_without_holdout_n_and_with_an_unusable_one(world, tmp_path, capsys):
    out = tmp_path / "nohold"
    with pytest.raises(SystemExit) as e:
        TR.main(_train_argv(world, "A1", out, []))
    assert e.value.code == 2 and "--holdout-n" in capsys.readouterr().err, "argparse must name the missing flag"
    assert not (out / "final.pt").exists() and not (out / "train_log.jsonl").exists() and not (out / "launch_check.json").exists()
    for i, bad in enumerate((["--holdout-n", "0", "--holdout-pool", "6"], ["--holdout-n", "-1", "--holdout-pool", "6"],
                             ["--holdout-n", "6", "--holdout-pool", "6"], ["--holdout-n", "2", "--holdout-pool", "600"])):
        o = tmp_path / f"bad{i}"
        assert TR.main(_train_argv(world, "A1", o, bad)) == 2, bad          # G-HYG: named but unusable => REFUSED, no training
        assert not (o / "final.pt").exists() and not (o / "train_log.jsonl").exists()
    assert "REFUSED" in capsys.readouterr().out


def _spy_training(monkeypatch):
    """Record every clip the REAL trainer loads into its bank and every clip its sampler may draw from (the caller, not the library)."""
    seen = {"loaded": set(), "sampled": set()}
    real_load, real_sample = RD.ClipBank.load, RD.sample_batch

    def load(self, names):
        names = list(names)
        seen["loaded"].update(names)
        return real_load(self, names)

    def sample(bank, names, *a, **k):
        seen["sampled"].update(names)
        return real_sample(bank, names, *a, **k)

    monkeypatch.setattr(RD.ClipBank, "load", load)
    monkeypatch.setattr(RD, "sample_batch", sample)
    return seen


def test_every_arm_withholds_the_same_holdout_and_never_trains_on_it(world, tmp_path, monkeypatch):
    allc = world["names"]["train"]
    want = RD.holdout_clips(allc, 2, 6)
    want_train = [c for c in RD.pick_train_clips(allc, 0) if c not in set(want)]
    seen = _spy_training(monkeypatch)
    shas = set()
    for arm in ("R0", "A0", "A1x"):
        seen["loaded"].clear()
        seen["sampled"].clear()
        out = tmp_path / arm
        assert run_train(world, arm, out) == 0
        assert seen["loaded"] and seen["sampled"]
        assert not (seen["loaded"] & set(want)), f"{arm}: the bank loaded a holdout clip"
        assert not (seen["sampled"] & set(want)), f"{arm}: the sampler could draw a holdout clip"
        assert seen["loaded"] == set(want_train), f"{arm}: every non-holdout clip trains"
        meta = json.load(open(out / "meta.json"))
        assert meta["holdout"]["clips"] == want and meta["holdout"]["sha256"] == RD.clips_sha256(want)
        assert meta["holdout"]["disjoint_from_train"] is True and meta["holdout"]["n"] == 2 and meta["holdout"]["pool"] == 6
        assert meta["train"]["sha256"] == RD.clips_sha256(want_train) and meta["train"]["n"] == len(want_train)
        ck = torch.load(out / "final.pt", weights_only=False)
        assert ck["holdout"]["clips"] == want and ck["holdout"]["sha256"] == meta["holdout"]["sha256"]
        assert json.load(open(out / "DONE"))["holdout_sha256"] == meta["holdout"]["sha256"]
        shas.add(meta["holdout"]["sha256"])
    assert len(shas) == 1, "the holdout is IDENTICAL across arms"


def test_mutation_a_split_that_does_not_exclude_the_holdout_turns_the_spy_red(world, tmp_path, monkeypatch):
    want = RD.holdout_clips(world["names"]["train"], 2, 6)
    seen = _spy_training(monkeypatch)
    monkeypatch.setattr(RD, "split_train_holdout",
                        lambda clips, n, hn, pn: (RD.pick_train_clips(clips, n), RD.holdout_clips(clips, hn, pn)))   # the bug: nothing withheld
    assert run_train(world, "A0", tmp_path / "mut") == 0
    assert seen["loaded"] & set(want) and seen["sampled"] & set(want), "the control must see the holdout being trained on"
    assert json.load(open(tmp_path / "mut" / "meta.json"))["holdout"]["disjoint_from_train"] is False, "and the run's own meta must say so"


def test_resume_refuses_a_changed_holdout(world, tmp_path):
    run_train(world, "A1", tmp_path / "r", steps=3)
    with pytest.raises(SystemExit, match="different configuration"):
        run_train(world, "A1", tmp_path / "r", steps=3, extra=["--holdout-n", "1"])        # the last --holdout-n wins in argparse


def test_eval_rederives_the_holdout_and_refuses_a_drifted_checkpoint(world, tmp_path):
    run_train(world, "A1", tmp_path / "a")
    ck = torch.load(tmp_path / "a" / "final.pt", weights_only=False)

    def ev(ckpt, tag, train_cache=True):
        argv = ["eval", "--ckpt", str(ckpt), "--eval-cache", world["root"], "--p10-prior", world["prior_path"], "--grid-json",
                world["grid_json"], "--out", str(tmp_path / ("e_" + tag)), "--n-boot", "20", "--fam-boot", "10"]
        return EV.main(argv + (["--train-cache", world["root"]] if train_cache else []))

    assert ev(tmp_path / "a" / "final.pt", "ok") == 0
    ok = json.load(open(tmp_path / "e_ok" / "eval.json"))["holdout"]
    assert ok["status"] == "OK" and ok["clips_sha256"] == ck["holdout"]["sha256"] and ok["n_windows"] > 0
    z = np.load(tmp_path / "e_ok" / "perwindow.npz", allow_pickle=True)
    assert set(("hold_eid", "hold_t", "hold_ade6", "hold_p10_ade6")) <= set(z.files) and len(z["hold_ade6"]) == ok["n_windows"]
    assert not np.array_equal(z["ade6"], z["hold_ade6"]), "the holdout ADE is its own array, never the eval-141 one"
    # no --train-cache => the holdout is NOT computed, and says so (verdict will refuse)
    assert ev(tmp_path / "a" / "final.pt", "nocache", train_cache=False) == 0
    assert json.load(open(tmp_path / "e_nocache" / "eval.json"))["holdout"]["status"].startswith("NOT COMPUTED")
    # a checkpoint with a record that no longer matches the manifest: refused loudly
    drift = dict(ck, holdout=dict(ck["holdout"], clips=list(reversed(ck["holdout"]["clips"]))))
    torch.save(drift, tmp_path / "drift.pt")
    with pytest.raises(AssertionError, match="HOLDOUT DRIFT"):
        ev(tmp_path / "drift.pt", "drift")
    wrong_train = dict(ck, holdout=dict(ck["holdout"], train_sha256="0" * 64))
    torch.save(wrong_train, tmp_path / "wt.pt")
    with pytest.raises(AssertionError, match="TRAIN-SPLIT DRIFT"):
        ev(tmp_path / "wt.pt", "wt")
    old = {k: v for k, v in ck.items() if k != "holdout"}                      # trained before Amendment 1
    torch.save(old, tmp_path / "old.pt")
    assert ev(tmp_path / "old.pt", "old") == 0
    assert json.load(open(tmp_path / "e_old" / "eval.json"))["holdout"]["status"].startswith("NOT COMPUTED")


# ---- verdict: SYNTHETIC per-window numbers where the holdout winner differs from the eval-141 winner
CELLS = ["b0.04g50", "b0.04g100", "b0.08g50", "b0.08g100"]
EVAL141 = {"b0.04g50": 2.0, "b0.04g100": 2.4, "b0.08g50": 2.9, "b0.08g100": 2.6}        # eval-141 ADE@6s: b0.04g50 would win
HOLDOUT = {"b0.04g50": 3.1, "b0.04g100": 2.7, "b0.08g50": 2.2, "b0.08g100": 2.9}        # holdout ADE@6s: b0.08g50 wins


def _fake_run(d, eval_ade, hold_ade, *, hold=True, hold_t=(3, 4, 3, 4), sha="a" * 64):
    d.mkdir(parents=True, exist_ok=True)
    arr = dict(eid=np.repeat(np.arange(3), 2), t=np.tile([3, 4], 3), ade6=np.full(6, eval_ade), p10_ade6=np.full(6, 3.0))
    ev = {"instrument_ok": True}
    if hold:
        arr.update(hold_eid=np.repeat(np.arange(2), 2), hold_t=np.asarray(hold_t), hold_ade6=np.full(4, hold_ade), hold_p10_ade6=np.full(4, 3.0))
        ev["holdout"] = {"status": "OK", "clips_sha256": sha, "n_clips": 2, "n": 2, "pool": 6}
    np.savez_compressed(d / "perwindow.npz", **arr)
    json.dump(ev, open(d / "eval.json", "w"))
    return d


def _verdict_args(tmp_path, tag, **kw):
    runs = []
    for c in CELLS:
        d = _fake_run(tmp_path / tag / c, EVAL141[c], HOLDOUT[c], **kw)
        runs += ["--run", f"A1:{c}@0={d}"]
    return runs, tmp_path / tag / "v.json"


def test_verdict_selects_the_A1_cell_on_the_holdout_not_on_eval141(tmp_path):
    eval_winner, hold_winner = min(EVAL141, key=EVAL141.get), min(HOLDOUT, key=HOLDOUT.get)
    assert eval_winner != hold_winner, "the synthetic case must discriminate the two rules"
    runs, vout = _verdict_args(tmp_path, "real")
    assert EV.main(["verdict", "--out", str(vout), "--n-boot", "20"] + runs) == 0
    v = json.load(open(vout))
    assert v["A1_best_cell"] == hold_winner != eval_winner
    assert v["A1_cells_seed0_holdout_ade6"] == pytest.approx(HOLDOUT)
    nd = v["A1_cells_seed0_eval141_ade6_NON_DECIDING"]
    assert {c: nd[c]["ade6"] for c in CELLS} == pytest.approx(EVAL141), "every cell's eval-141 number is still reported ..."
    assert nd[eval_winner]["arm_minus_p10"] == pytest.approx(EVAL141[eval_winner] - 3.0)
    rule = v["A1_best_rule"]                                                        # ... and the rule is written truthfully
    assert "withheld TRAIN holdout" in rule and "NOT selected on eval-141" in rule and "Amendment 1" in rule
    assert "2 clips" in rule and "sha256 " + "a" * 12 in rule and "4 windows" in rule and "NON_DECIDING" in rule
    assert v["holdout"]["n_windows"] == 4 and v["holdout"]["clips_sha256"] == "a" * 64 and v["selection_basis"].startswith("the withheld TRAIN")


def _mutant_eval_module():
    """The evaluator's REAL source with ONE line changed: the A1 cell is selected on eval-141 again (the pre-amendment behaviour)."""
    src = inspect.getsource(EV)
    line = 'best = min(a1c, key=lambda c: float(selection_metric(runs[("A1", c, 0)]["z"]).mean()))'
    assert src.count(line) == 1, "the selection line moved: update this mutation"
    mutant = src.replace(line, 'best = min(a1c, key=lambda c: float(ade(("A1", c, 0)).mean()))')
    m = types.ModuleType("ra1_hjepa_eval_mutant")
    m.__file__ = EV.__file__
    exec(compile(mutant, EV.__file__, "exec"), m.__dict__)
    return m


def test_mutation_selecting_on_eval141_again_turns_the_selection_check_red(tmp_path):
    eval_winner, hold_winner = min(EVAL141, key=EVAL141.get), min(HOLDOUT, key=HOLDOUT.get)
    runs, vout = _verdict_args(tmp_path, "mut")

    def selects_on_holdout(module):
        assert module.main(["verdict", "--out", str(vout), "--n-boot", "20"] + runs) == 0
        return json.load(open(vout))["A1_best_cell"] == hold_winner

    assert selects_on_holdout(EV), "control: the real evaluator passes the check"
    assert not selects_on_holdout(_mutant_eval_module()), "the one-line mutation (eval-141 selection) must make the check RED"
    assert json.load(open(vout))["A1_best_cell"] == eval_winner


def test_verdict_refuses_an_A1_cell_without_holdout_numbers_and_has_no_eval141_fallback(tmp_path):
    runs, vout = _verdict_args(tmp_path, "nohold", hold=False)
    assert EV.main(["verdict", "--out", str(vout), "--n-boot", "20"] + runs) == 4
    v = json.load(open(vout))
    assert v["A1_best_cell"] is None and v["status"].startswith("A1 SELECTION REFUSED") and "branch" not in v
    assert "A1_best_rule" not in v and set(v["A1_cells_seed0_eval141_ade6_NON_DECIDING"]) == set(CELLS)
    # a single cell with no holdout is refused too (it is not "trivially the best")
    d = _fake_run(tmp_path / "one", 2.0, 2.0, hold=False)
    assert EV.main(["verdict", "--out", str(tmp_path / "one.json"), "--n-boot", "20", "--run", f"A1:b0.08g100@0={d}"]) == 4
    # ... and a MIX (one cell lacks it) is refused, naming the cell
    runs2, vout2 = _verdict_args(tmp_path, "mix")
    _fake_run(tmp_path / "mix" / CELLS[1], EVAL141[CELLS[1]], 0.0, hold=False)
    assert EV.main(["verdict", "--out", str(vout2), "--n-boot", "20"] + runs2) == 4
    assert CELLS[1] in json.load(open(vout2))["status"]


def test_verdict_asserts_every_run_scored_the_same_holdout(tmp_path):
    runs, vout = _verdict_args(tmp_path, "same")
    odd = _fake_run(tmp_path / "same" / "odd", 2.0, 2.0, hold_t=(3, 4, 3, 5))
    with pytest.raises(AssertionError, match="holdout windows differ"):
        EV.main(["verdict", "--out", str(vout), "--n-boot", "20"] + runs + ["--run", f"R0@0={odd}"])
    odd2 = _fake_run(tmp_path / "same" / "odd2", 2.0, 2.0, sha="b" * 64)
    with pytest.raises(AssertionError, match="SAME holdout"):
        EV.main(["verdict", "--out", str(vout), "--n-boot", "20"] + runs + ["--run", f"R0@0={odd2}"])


# ---- the Box-2 queue (the REAL caller): every trainer argv names --holdout-n 60, every eval gets the train cache
QUEUE = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "code" / "ra1_box2_queue.cmd"
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the refav1-hj-mini package is not in this checkout")


@needs_queue
def test_queue_text_names_holdout_n_60_in_every_trainer_invocation():
    lines = QUEUE.read_text(encoding="utf-8").splitlines()
    assert 'set "HOLDOUT=--holdout-n 60"' in lines
    arg_lines = [ln for ln in lines if ln.lstrip().startswith('set "ARGS=--arm')]
    assert len(arg_lines) == 2 and all("%HOLDOUT%" in ln for ln in arg_lines), "both :run and :chk build ARGS with %HOLDOUT%"
    train_calls = [ln for ln in lines if '"%TRAIN%"' in ln]
    assert len(train_calls) >= 3 and all("%ARGS%" in ln for ln in train_calls), "every trainer call goes through ARGS"
    eval_calls = [ln for ln in lines if '"%EVAL%" eval' in ln]
    assert len(eval_calls) >= 2 and all('--train-cache "%CACHE%"' in ln for ln in eval_calls), "evals must score the holdout"


@needs_queue
@pytest.mark.skipif(os.name != "nt", reason="the queue is a Windows batch file")
def test_queue_dry_run_prints_holdout_n_60_in_every_trainer_argv(tmp_path):
    import subprocess
    env = dict(os.environ, RA1_DRYRUN="1", REPO=str(REPO), RUNS=str(tmp_path / "runs"))

    def dry(args):
        # cmd /c with an outer pair of quotes: the package path contains an ampersand
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout.splitlines()

    for args, n_train in (("gate", 6), ("stage1", 10), ("stage2 0.08 100", 5)):
        out = dry(args)
        tr = [ln for ln in out if "ra1_hjepa_train.py" in ln]
        assert len(tr) == n_train, (args, len(tr), out)
        assert all("--holdout-n 60" in ln for ln in tr), (args, [ln for ln in tr if "--holdout-n 60" not in ln])
        if args != "gate":
            ev = [ln for ln in out if "ra1_hjepa_eval.py" in ln]
            assert len(ev) == n_train and all("--train-cache" in ln for ln in ev)


# ---- PREREG_RA1_ES: the budget-sweep stages reuse the :run block, so every argv keeps --holdout-n 60, --kin-mode yaw and the budget
@needs_queue
@pytest.mark.skipif(os.name != "nt", reason="the queue is a Windows batch file")
def test_queue_es_stages_dry_run_carry_the_budget_holdout_and_yaw(tmp_path):
    import re
    import subprocess
    runs = tmp_path / "runs"
    env = dict(os.environ, RA1_DRYRUN="1", REPO=str(REPO), RUNS=str(runs))

    def dry(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        return r.returncode, r.stdout.splitlines()

    rc, out = dry('es "250 500"')
    assert rc == 0, out
    tr = [ln for ln in out if "ra1_hjepa_train.py" in ln]
    got = [(re.search(r"--arm (\S+) --seed (\S+)", ln).groups(), re.search(r"--steps (\d+)", ln).group(1)) for ln in tr]
    assert got == [(("R0", "0"), "250"), (("R0", "1"), "250"), (("R0", "0"), "500"), (("R0", "1"), "500")], got
    assert all("--holdout-n 60" in ln and "--kin-mode yaw" in ln for ln in tr)
    assert all(f"runs_es{os.sep}st{s}{os.sep}R0_s" in ln for ln, (_, s) in zip(tr, got)), "each budget gets its own run dir"
    ev = [ln for ln in out if "ra1_hjepa_eval.py" in ln]
    assert len(ev) == 4 and all("--train-cache" in ln for ln in ev), "every eval scores the holdout"

    rc, out = dry("es2 500 0.08 50")
    assert rc == 0, out
    tr = [ln for ln in out if "ra1_hjepa_train.py" in ln]
    arms = [re.search(r"--arm (\S+) --seed (\S+)", ln).groups() for ln in tr]
    assert arms == [("A1", "0"), ("A1", "1"), ("A1x", "0")], arms
    assert all("--steps 500" in ln and "--beta 0.08 --gamma 50" in ln and "--holdout-n 60" in ln and "--kin-mode yaw" in ln for ln in tr)
    vd = [ln for ln in out if "ra1_hjepa_eval.py" in ln and " verdict --out " in ln]
    assert len(vd) == 1 and vd[0].count("--run ") == 5 and "es2_verdict.json" in vd[0], vd
    assert all(f"{k}=" in vd[0] for k in ("R0@0", "R0@1", "A1:b0.08g50@0", "A1:b0.08g50@1", "A1x:b0.08g50@0")), vd

    for bad in ("es", "es2 500"):                              # a missing budget list / cell is refused, never defaulted
        rc, out = dry(bad)
        assert rc == 2, (bad, out)
        assert not any("ra1_hjepa_train.py" in ln for ln in out)


ES_VERDICT = QUEUE.parent / "ra1_es_verdict.py"


def _es_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ra1_es_verdict", ES_VERDICT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _es_eval(d, seed, step, hold_delta, sha="h" * 64, arm="R0", kin="yaw"):
    p = Path(d) / f"R0_s{seed}" / "eval"
    p.mkdir(parents=True)
    dd = lambda x: dict(delta=x, lo=x - 0.05, hi=x + 0.05, separated=abs(x) > 0.05)     # noqa: E731
    ev = dict(arm=arm, kin_mode=kin, instrument_ok=True, ckpt_step=step,
              holdout=dict(clips_sha256=sha, ade6_arm=dict(mean=3.0 + hold_delta), ade6_p10=dict(mean=3.0), ade6_arm_minus_p10=dd(hold_delta)),
              K=dict(ade6_arm=dict(mean=2.6 + hold_delta), ade6_arm_minus_p10=dd(hold_delta)))
    (p / "eval.json").write_text(json.dumps(ev), encoding="utf-8")


@pytest.mark.skipif(not ES_VERDICT.exists(), reason="the refav1-hj-mini package is not in this checkout")
def test_es_verdict_selects_on_the_holdout_with_the_shorter_tie_and_the_null_stop(tmp_path):
    m = _es_mod()

    def build(name, deltas):                                   # deltas: {budget: (seed0, seed1)}, 6000 is the banked root
        root, banked = tmp_path / name / "es", tmp_path / name / "banked"
        for S, (d0, d1) in deltas.items():
            base = banked if S == 6000 else root / f"st{S}"
            _es_eval(base, 0, S, d0)
            _es_eval(base, 1, S, d1)
        a = types.SimpleNamespace(root=str(root), banked=str(banked), budgets=[str(S) for S in deltas if S != 6000])
        return a

    # 500 is the minimum (-0.100); 1000 is within the 0.01 m tie band (-0.095) but LONGER -> 500; 250 is shorter but outside the band
    r = m.select(build("tie", {250: (-0.05, -0.06), 500: (-0.10, -0.10), 1000: (-0.09, -0.10), 2000: (0.1, 0.1), 6000: (0.40, 0.38)}))
    assert r["status"] == "OK" and r["S_star"] == 500 and r["next"] == "es2 500 0.08 50", r
    r = m.select(build("tie2", {250: (-0.095, -0.095), 500: (-0.10, -0.10), 6000: (0.4, 0.4)}))
    assert r["S_star"] == 250, "a budget within the tie band that is SHORTER wins"
    # no budget below 0 -> NULL, ES-2 not run
    r = m.select(build("null", {250: (0.01, 0.0), 500: (0.05, 0.02), 6000: (0.40, 0.38)}))
    assert r["next"] == "NULL" and "NULL" in r["reading"], r
    # a run whose checkpoint step is not its budget is refused
    a = build("step", {250: (-0.1, -0.1), 6000: (0.4, 0.4)})
    ev = Path(a.root) / "st250" / "R0_s1" / "eval" / "eval.json"
    j = json.loads(ev.read_text(encoding="utf-8")); j["ckpt_step"] = 6000; ev.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(SystemExit, match="ckpt_step"):
        m.select(a)
    # a steer-representation run is refused
    a = build("steer", {250: (-0.1, -0.1), 6000: (0.4, 0.4)})
    ev = Path(a.banked) / "R0_s0" / "eval" / "eval.json"
    j = json.loads(ev.read_text(encoding="utf-8")); j["kin_mode"] = "steer"; ev.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(SystemExit, match="kin_mode"):
        m.select(a)
    # a missing run is INCOMPLETE, never a selection over what happens to exist
    a = build("miss", {250: (-0.1, -0.1), 6000: (0.4, 0.4)})
    (Path(a.root) / "st250" / "R0_s1" / "eval" / "eval.json").unlink()
    assert m.select(a)["status"] == "INCOMPLETE"


@pytest.mark.skipif(not ES_VERDICT.exists(), reason="the refav1-hj-mini package is not in this checkout")
def test_es_verdict_read_gates_k_on_the_a1x_instrument(tmp_path):
    m = _es_mod()
    es1 = tmp_path / "es1.json"
    es1.write_text(json.dumps(dict(next="es2 500 0.08 50", S_star=500, reading="r")), encoding="utf-8")

    def verdict(gate_ok, r0_pass, a1_pass):
        cand = dict(R0=dict(mean_ade6=2.40, PASS=r0_pass), A1=dict(mean_ade6=2.45, PASS=a1_pass), A2=dict(mean_ade6=None, PASS=False))
        v = dict(kin_mode="yaw", deciding=True, A1_best_cell="b0.08g50", instrument_ok_all=True,
                 A1x_control=dict(verdict="OK: A1x loses to A1, separated" if gate_ok else "INSTRUMENT BROKEN: A1x does not lose to A1"),
                 K=dict(candidates=cand), training_seed_floor_A1=0.02)
        p = tmp_path / f"v_{gate_ok}_{r0_pass}_{a1_pass}.json"
        p.write_text(json.dumps(v), encoding="utf-8")
        return types.SimpleNamespace(es1=str(es1), es2_verdict=str(p))

    assert m.read(verdict(False, True, True))["outcome"] == "INSTRUMENT_BROKEN", "a broken gate is never overridden by a K pass"
    assert m.read(verdict(True, True, False))["outcome"] == "K_PASS"
    assert m.read(verdict(True, False, True))["outcome"] == "K_FAIL", "K is read on the BEST mean arm (R0 here), not on any passing arm"
    es1.write_text(json.dumps(dict(next="NULL", S_star=250, reading="r")), encoding="utf-8")
    with pytest.raises(SystemExit, match="NULL"):
        m.read(verdict(True, True, True))


# ---- PREREG_RA1_RIDGE: the probe's closed-form ridge and its reading order
def _ridge_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ra1_ridge_probe", REPO / "stack" / "scripts" / "ra1_ridge_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ridge_probe_recovers_a_planted_map_and_tends_to_the_bias():
    m = _ridge_mod()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 12)) * rng.uniform(0.5, 3.0, size=12)
    Wt = rng.normal(size=(12, 6))
    Y = X @ Wt + 0.7 + 0.05 * rng.normal(size=(4000, 6))
    rg = m.Ridge(X, Y)
    pred = rg.predict(X, rg.coef(1e-8))
    assert np.abs(pred - (X @ Wt + 0.7)).max() < 0.05, "a tiny penalty recovers the planted map"
    far = rg.predict(X, rg.coef(1e9))
    assert np.abs(far - Y.mean(0)).max() < 1e-4, "an infinite penalty IS the bias-only control"
    zero = m.Ridge(rng.normal(size=(4000, 12)), Y)                # features carrying nothing
    assert np.abs(zero.coef(1e-6)).max() < 0.1 * np.abs(rg.coef(1e-6)).max(), "features carrying nothing get coefficients at the noise level, an order below the planted map's"


def test_ridge_probe_reading_order():
    m = _ridge_mod()
    sep = lambda d: dict(delta=d, lo=d - 0.01, hi=d + 0.01, separated=abs(d) > 0.01)     # noqa: E731
    o = dict(controls=dict(ok=True), sets={"mean+f8": dict(eval_minus_p10=sep(-0.10))}, vision_minus_f8_eval=sep(-0.05))
    assert m.reading(o).startswith("SIGNAL")
    o["vision_minus_f8_eval"] = sep(0.0)
    assert m.reading(o).startswith("PRIOR CORRECTION ONLY")
    o["sets"]["mean+f8"]["eval_minus_p10"] = sep(0.0)
    assert m.reading(o).startswith("NO LINEAR SIGNAL")
    o["controls"]["ok"] = False
    o["sets"]["mean+f8"]["eval_minus_p10"] = sep(-0.5)
    assert m.reading(o).startswith("INSTRUMENT"), "a failed control is never overridden by a good number"


@needs_queue
@pytest.mark.skipif(os.name != "nt", reason="the queue is a Windows batch file")
def test_queue_es3_and_ridge_stages_dry_run(tmp_path):
    import re
    import subprocess
    env = dict(os.environ, RA1_DRYRUN="1", REPO=str(REPO), RUNS=str(tmp_path / "runs"))

    def dry(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        return r.returncode, r.stdout.splitlines()

    rc, out = dry("es3 250 0.08")
    assert rc == 0, out
    tr = [ln for ln in out if "ra1_hjepa_train.py" in ln]
    assert [re.search(r"--arm (\S+) --seed (\S+)", ln).groups() for ln in tr] == [("A0", "0"), ("A0", "1")]
    assert all("--steps 250" in ln and "--beta 0.08" in ln and "--gamma" not in ln and "--holdout-n 60" in ln and "--kin-mode yaw" in ln
               for ln in tr), tr
    vd = [ln for ln in out if " verdict --out " in ln]
    assert len(vd) == 1 and vd[0].count("--run ") == 7 and "es3_verdict.json" in vd[0], vd
    assert all(f"{k}=" in vd[0] for k in ("R0@0", "R0@1", "A0:b0.08@0", "A0:b0.08@1", "A1:b0.08g50@0", "A1:b0.08g50@1", "A1x:b0.08g50@0"))
    rc, out = dry("ridge")
    assert rc == 0, out
    rp = [ln for ln in out if "ra1_ridge_probe.py" in ln]
    assert len(rp) == 1 and "--holdout-n 60" in rp[0] and "native_tracks.pt" in rp[0] and "p10_yaw540" in rp[0], rp
    assert "--device" not in rp[0], "the ridge probe is CPU only"
    assert dry("es3 250")[0] == 2


# ---- PREREG_RA1_ES Addendum C: --wd is a lever under test, so the gate reads it back from the BUILT optimizer
def test_gate_reads_weight_decay_back_from_the_built_optimizer(world, tmp_path):
    out = tmp_path / "wd"
    assert run_train(world, "A1", out, extra=["--wd", "0.3"]) == 0
    gate = json.load(open(out / "launch_check.json", encoding="utf-8"))
    rows = [r for r in gate["G_DVB"] if r["lever"] == "weight_decay"]
    assert len(rows) == 1 and rows[0]["ok"] and rows[0]["declared"] == 0.3 and rows[0]["built"] == [0.3], rows


def test_gate_refuses_an_optimizer_that_ignores_wd(world, tmp_path, monkeypatch):
    real = TR.R.optimizer_groups

    def ignores_wd(model, lr, upper_lr_mult, weight_decay):          # the deliberate regression: --wd never reaches AdamW
        return real(model, lr, upper_lr_mult, 1e-3)

    monkeypatch.setattr(TR.R, "optimizer_groups", ignores_wd)
    out = tmp_path / "wd_bad"
    assert run_train(world, "A1", out, extra=["--wd", "0.3"]) == 2, "a declared --wd the optimizer does not carry must fail the gate"
    assert not (out / "final.pt").exists()
    monkeypatch.setattr(TR.R, "optimizer_groups", real)
    assert run_train(world, "A1", tmp_path / "wd_ctl", extra=["--wd", "1e-3"]) == 0, "control: a matching wd passes"


@needs_queue
@pytest.mark.skipif(os.name != "nt", reason="the queue is a Windows batch file")
def test_queue_es5_stage_carries_wd_on_every_trainer_argv(tmp_path):
    import re
    import subprocess
    env = dict(os.environ, RA1_DRYRUN="1", REPO=str(REPO), RUNS=str(tmp_path / "runs"))

    def dry(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=120)
        return r.returncode, r.stdout.splitlines()

    rc, out = dry("es5 1000 0.08 50 0.3")
    assert rc == 0, out
    tr = [ln for ln in out if "ra1_hjepa_train.py" in ln]
    assert [re.search(r"--arm (\S+) --seed (\S+)", ln).groups() for ln in tr] == [("A1", "0"), ("A1", "1"), ("A1x", "0")]
    assert all("--wd 0.3" in ln and "--steps 1000" in ln and "--beta 0.08 --gamma 50" in ln and "--holdout-n 60" in ln
               and "--kin-mode yaw" in ln and f"st1000_wd0.3{os.sep}" in ln for ln in tr), tr
    vd = [ln for ln in out if " verdict --out " in ln]
    assert len(vd) == 1 and vd[0].count("--run ") == 5 and "es5_verdict.json" in vd[0], vd
    assert f"st1000{os.sep}R0_s0" in vd[0] and f"st1000_wd0.3{os.sep}A1x_b0.08g50_s0" in vd[0], "R0 is the banked unregularised run"
    assert dry("es5 1000 0.08 50")[0] == 2


# ------------------------------------------------------------------------------------------- PREREG_RA1_FULL (the full train corpus)
FULL_VERDICT = QUEUE.parent / "ra1_full_verdict.py"
FULL_DRIVER = QUEUE.parent / "ra1_full_box2.py"


def _dry_queue(env, args):
    import subprocess
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return r.returncode, r.stdout.splitlines()


@pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
def test_queue_full_stages_read_their_own_cache_root_and_prior_and_nothing_else_does(tmp_path):
    """The trainer takes every clip in its cache's manifest minus the holdout, so the cache ROOT sets the corpus. Only `full` / `full2`
    may point at the full-corpus root and P10-yaw-4512; the 540-clip stages must still read the 600-clip root and P10-yaw-540."""
    import re
    cf = str(tmp_path / "cache_full")
    env = dict(os.environ, RA1_DRYRUN="1", REPO=str(REPO), RUNS=str(tmp_path / "runs"), CACHEFULL=cf)
    rc, out = _dry_queue(env, 'full "500 1000" 0.08 50')
    assert rc == 0, out
    tr = [ln for ln in out if "ra1_hjepa_train.py" in ln]
    got = [(re.search(r"--arm (\S+) --seed (\S+)", ln).groups(), re.search(r"--steps (\d+)", ln).group(1)) for ln in tr]
    assert got == [((a, s), S) for S in ("500", "1000") for a in ("R0", "A1") for s in ("0", "1")], got
    assert all(f'--cache "{cf}"' in ln and f"p10_yaw4512{os.sep}p10_prior.pt" in ln and "--holdout-n 60" in ln and "--kin-mode yaw" in ln
               and f'--tracks "{cf}{os.sep}native_tracks.pt"' in ln for ln in tr), tr
    assert all("--beta 0.08 --gamma 50" in ln for ln in tr if "--arm A1 " in ln)
    assert all(f"runs_full{os.sep}st{S}{os.sep}" in ln for ln, (_, S) in zip(tr, got)), "each budget gets its own run dir"
    ev = [ln for ln in out if "ra1_hjepa_eval.py" in ln]
    assert len(ev) == 8 and all(f'--eval-cache "{cf}" --train-cache "{cf}"' in ln and f"p10_yaw4512{os.sep}p10_eval_pred6.npy" in ln
                                for ln in ev), ev
    rc, out = _dry_queue(env, "full2 1000 500 0.08 50")
    assert rc == 0, out
    tr = [ln for ln in out if "ra1_hjepa_train.py" in ln]
    assert len(tr) == 1 and "--arm A1x --seed 0" in tr[0] and "--steps 1000" in tr[0] and f'--cache "{cf}"' in tr[0], tr
    vd = [ln for ln in out if " verdict --out " in ln]
    assert len(vd) == 1 and vd[0].count("--run ") == 5 and "full2_verdict.json" in vd[0], vd
    assert f"st500{os.sep}R0_s0" in vd[0] and f"st1000{os.sep}A1_b0.08g50_s1" in vd[0] and f"st1000{os.sep}A1x_b0.08g50_s0" in vd[0], vd
    for bad in ("full", 'full "500"', "full2 1000 500 0.08"):          # a missing budget list / cell is refused, never defaulted
        rc, out = _dry_queue(env, bad)
        assert rc == 2 and not any("ra1_hjepa_train.py" in ln for ln in out), (bad, out)
    for stage in ('es "250"', "es5 1000 0.08 50 0.3", "stage1"):       # the override never leaks into a 540-clip stage
        rc, out = _dry_queue(env, stage)
        assert rc == 0, out
        lines = [ln for ln in out if "ra1_hjepa_" in ln]
        assert lines and not any(cf in ln or "p10_yaw4512" in ln for ln in lines), (stage, lines)
        assert all("p10_yaw540" in ln for ln in lines if "--p10-prior" in ln), stage


def _full_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ra1_full_verdict", FULL_VERDICT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _full_eval(root, arm, seed, step, hold, n_train=4572, sha=None, p10=3.0682):
    m = _full_mod()
    p = Path(root) / f"st{step}" / m.ARMS[arm].format(seed=seed) / "eval"
    p.mkdir(parents=True)
    dd = lambda x: dict(delta=x, lo=x - 0.05, hi=x + 0.05, separated=abs(x) > 0.05)     # noqa: E731
    ev = dict(arm=arm, kin_mode="yaw", instrument_ok=True, ckpt_step=step, disjointness=dict(n_train=n_train),
              holdout=dict(clips_sha256=sha or m.HOLDOUT_SHA, train_sha256="t" * 64, ade6_arm=dict(mean=hold), ade6_p10=dict(mean=p10),
                           ade6_arm_minus_p10=dd(hold - p10)),
              K=dict(ade6_arm=dict(mean=hold - 0.5), ade6_arm_minus_p10=dd(hold - p10)))
    (p / "eval.json").write_text(json.dumps(ev), encoding="utf-8")


@pytest.mark.skipif(not FULL_VERDICT.exists(), reason="the refav1-hj-mini package is not in this checkout")
def test_full_verdict_selects_each_arm_on_the_holdout_and_refuses_a_600_clip_run(tmp_path):
    m = _full_mod()
    grid = [500, 1000, 3000, 6000]
    hold = {"R0": {500: 3.10, 1000: 3.05, 3000: 3.045, 6000: 3.20}, "A1": {500: 3.08, 1000: 3.00, 3000: 2.95, 6000: 3.10}}
    for arm, by in hold.items():
        for S, h in by.items():
            for seed in (0, 1):
                _full_eval(tmp_path, arm, seed, S, h + 0.002 * seed)
    a = types.SimpleNamespace(root=str(tmp_path), budgets=[str(S) for S in grid], extended=False)
    r = m.select(a)
    assert r["status"] == "OK" and r["S_star"] == {"R0": 1000, "A1": 3000}, r["S_star"]   # R0: 3000 is within 0.01 m -> the shorter
    assert r["K_arm"] == "A1" and r["next"] == "full2 3000 1000 0.08 50" and r["edge"] == {"R0": None, "A1": None}
    assert abs(r["memorisation_R0_holdout_minus_p10_by_budget"]["6000"] - (3.201 - 3.0682)) < 1e-9
    # an S* on the grid edge names its one extension; --extended reads it there and discloses it
    _full_eval(tmp_path / "e", "R0", 0, 500, 3.0); _full_eval(tmp_path / "e", "R0", 1, 500, 3.0)
    _full_eval(tmp_path / "e", "R0", 0, 1000, 3.1); _full_eval(tmp_path / "e", "R0", 1, 1000, 3.1)
    for S, h in ((500, 3.2), (1000, 3.1)):
        for seed in (0, 1):
            _full_eval(tmp_path / "e", "A1", seed, S, h)
    e = m.select(types.SimpleNamespace(root=str(tmp_path / "e"), budgets=["500", "1000"], extended=False))
    assert e["edge"] == {"R0": "BELOW", "A1": "ABOVE"} and e["extension_budgets"] == {"R0": 250, "A1": 2000} and e["next"].startswith("EXTEND")
    e2 = m.select(types.SimpleNamespace(root=str(tmp_path / "e"), budgets=["500", "1000"], extended=True))
    assert e2["next"].startswith("full2") and "DISCLOSED" in e2["reading"]
    # arms within 0.01 m of each other go to R0, the simpler model: the world model has to earn its place
    for arm, h in (("R0", 3.000), ("A1", 2.995)):
        for seed in (0, 1):
            _full_eval(tmp_path / "tie", arm, seed, 500, h)
    assert m.select(types.SimpleNamespace(root=str(tmp_path / "tie"), budgets=["500"], extended=False))["K_arm"] == "R0"
    # a run from the 600-clip cache, a second holdout or a second prior is refused, never averaged in
    for kw in (dict(n_train=600), dict(sha="x" * 64), dict(p10=3.0529)):
        d = tmp_path / ("bad_" + next(iter(kw)))
        for arm in ("R0", "A1"):
            for seed in (0, 1):
                _full_eval(d, arm, seed, 500, 3.0, **(kw if (arm, seed) == ("A1", 1) else {}))
        with pytest.raises(ValueError):
            m.select(types.SimpleNamespace(root=str(d), budgets=["500"], extended=False))


@pytest.mark.skipif(not FULL_VERDICT.exists(), reason="the refav1-hj-mini package is not in this checkout")
def test_full_verdict_reads_k_only_behind_the_gate_and_for_the_holdout_chosen_arm(tmp_path):
    m = _full_mod()
    f1 = dict(status="OK", next="full2 3000 1000 0.08 50", S_star={"R0": 1000, "A1": 3000}, K_arm="A1")
    (tmp_path / "f1.json").write_text(json.dumps(f1), encoding="utf-8")
    cands = {"R0": {"PASS": True}, "A1": {"PASS": False}}

    def verdict(gate):
        v = dict(kin_mode="yaw", deciding=True, A1_best_cell="b0.08g50", A1x_control=dict(verdict=gate), K=dict(candidates=cands))
        (tmp_path / "v.json").write_text(json.dumps(v), encoding="utf-8")
        return m.read(types.SimpleNamespace(full1=str(tmp_path / "f1.json"), verdict=str(tmp_path / "v.json")))

    assert verdict("UNRESOLVED: A1x is worse but not separated")["outcome"] == "INSTRUMENT_BROKEN"
    r = verdict("OK: A1x loses to A1, separated")
    assert r["outcome"] == "K_FAIL" and r["K_read"]["K_arm"] == "A1", "R0's PASS must not stand in for the holdout-chosen arm"
    cands["A1"]["PASS"] = True
    assert verdict("OK: A1x loses to A1, separated")["outcome"] == "K_PASS"


@pytest.mark.skipif(not FULL_DRIVER.exists(), reason="the refav1-hj-mini package is not in this checkout")
def test_full_box2_driver_dry_run_builds_then_encodes_then_verifies_each_chunk_into_the_full_root(tmp_path):
    """Order is the safety contract: the control first, CPU build and GPU encode never in one step, the encode fed the chunk's own list,
    every chunk verified before the next, and the final verify over ALL clips into the full root -- never the 600-clip `cache`."""
    import subprocess
    st = tmp_path / "data" / "trainall_stage"
    st.mkdir(parents=True)
    rest = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(5)]
    (st / "clips_rest.txt").write_text("\n".join(rest) + "\n", encoding="utf-8")
    (st / "camera_sha256.json").write_text(json.dumps({c: {"sha256": "0" * 64} for c in rest}), encoding="utf-8")
    (st / "ctl_expected.json").write_text(json.dumps({"c" * 8 + "-ctl": "1" * 64}), encoding="utf-8")
    env = dict(os.environ, RA1FULL_ROOT=str(tmp_path), LABELFLAGS="--listing L --train-labels T", RA1FULL_MIN_FREE_GB="0")
    r = subprocess.run([sys.executable, str(FULL_DRIVER), "--dry-run", "--chunk", "2"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    runs = [ln.split(" RUN ", 1)[1] for ln in r.stdout.splitlines() if " RUN " in ln]
    kind = ["build" if "pod_build_b1_epcache.py" in x else "encode" if "pull_encode_pooled.py" in x else "verify" if "verify_cache.py" in x
            else "?" for x in runs]
    assert kind == ["build"] + ["build", "encode", "verify"] * 3 + ["verify"], kind
    assert "_ctl256" in runs[0], "the identity control builds first, into its own dir"
    full = str(tmp_path / "cache_full")
    for i, x in enumerate(runs[1:-1]):
        k = i // 3
        if kind[i + 1] == "encode":
            assert f"chunk_{k:02d}.txt" in x and "--keep-src" in x and full in x and "--listing L --train-labels T" in x, x
        if kind[i + 1] == "verify":
            assert "--allow-partial" in x and full in x, x
    assert "--allow-partial" not in runs[-1] and "--sample 0" in runs[-1] and full in runs[-1]
    assert not any(str(tmp_path / "cache") + " " in x + " " for x in runs), "nothing may write into the 600-clip cache root"
    # the disk guard: a box without the space stops before the first chunk builds, with its own exit code
    env["RA1FULL_MIN_FREE_GB"] = "1e12"
    r = subprocess.run([sys.executable, str(FULL_DRIVER), "--dry-run", "--chunk", "2"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 11 and "ZZDISK-GUARD" in r.stdout and "pull_encode_pooled.py" not in r.stdout, r.stdout
