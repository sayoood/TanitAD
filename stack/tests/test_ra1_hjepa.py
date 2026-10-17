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
"""
from __future__ import annotations

import json
import os
import sys
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
    argv = ["--arm", arm, "--seed", str(seed), "--cache", world["root"], "--p10-prior", world["prior_path"], "--out", str(out),
            "--steps", str(steps)] + TINY_FLAGS + ARM_FLAGS[arm] + list(extra)
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
    with pytest.raises(SystemExit):                                   # a changed configuration is refused
        TR.main(["--arm", "A1h", "--seed", "5", "--cache", world["root"], "--p10-prior", world["prior_path"], "--out",
                 str(tmp_path / "crash"), "--steps", "6"] + TINY_FLAGS + ARM_FLAGS["A1h"])


def test_post_training_audit_names_a_parameter_that_never_got_a_gradient(world, tmp_path):
    run_train(world, "A1", tmp_path / "a", steps=3)
    ck = torch.load(tmp_path / "a" / "final.pt", weights_only=False)
    assert EV.untrained_params(ck) == []
    victim = ck["param_names"][5]
    del ck["opt"]["state"][5]                       # control: a parameter whose Adam state was never allocated
    assert EV.untrained_params(ck) == [victim]


def test_launch_gate_refuses_a_silent_a0(world, tmp_path):
    assert TR.main(["--arm", "A1", "--cache", world["root"], "--p10-prior", world["prior_path"], "--out", str(tmp_path / "x"),
                    "--steps", "2", "--beta", "0.08"] + TINY_FLAGS) == 2      # --gamma unnamed => REFUSED, no training
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


def test_verdict_degrades_gracefully_on_partial_inputs(world, tmp_path):
    run_train(world, "A1", tmp_path / "A1_0")
    ev = tmp_path / "A1_0" / "eval"
    assert EV.main(["eval", "--ckpt", str(tmp_path / "A1_0" / "final.pt"), "--eval-cache", world["root"], "--p10-prior", world["prior_path"],
                    "--grid-json", world["grid_json"], "--out", str(ev), "--n-boot", "30", "--fam-boot", "10"]) == 0
    assert EV.main(["verdict", "--out", str(tmp_path / "v.json"), "--n-boot", "30", "--run", f"A1:b0.08g100@0={ev}"]) == 0
    v = json.load(open(tmp_path / "v.json"))
    assert v["SENS"]["PASS"] is False and v["K"]["PASS"] is False and "NOT RUN" in v["A1x_control"]["verdict"]
    assert v["branch"].startswith("A1x GATE"), "with the instrument gate open no other branch may be selected"
