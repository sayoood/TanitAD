"""PREREG_V8F_HACC_LAT: ``--dump-ticks`` / ``--windows train`` in ``v8f_t1_eval.py`` and the fits, bakes and controls of ``v8f_hlat.py``. CPU only, synthetic world, < 120 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval;<repo>/stack/scripts  pytest -q stack/tests/test_v8f_hlat.py

What is pinned, and the control that must go red for each (no test edits a file on disk: every mutant is an in-process patch):
  * C-T   ``--dump-ticks`` changes no dump array and no JSON value of the tool (the run with the flag equals the run without it, bit for bit); the real-data form against the ORIGINAL tool is ``ct_smoke.sh``.
  * C-T2  the tick-0 samples equal the open-loop batch's tau (``traj30_o``) and P (``p10``) at tick-0 ages (0, 0); the GT in the executed frame equals the store's ``gt[t0]`` at tick 0, equals
          ``gt[t0 + n m]`` under the LOGGED executor at every tick, and follows the EXECUTED pose otherwise. [mutant: the GT taken in the logged frame]
  * the recorder keeps the plan the model issued (post-map) beside tau and P, the tick's measured speed ``p10f[:, -1, 0]`` and the window / episode / clip ids; a clip that ends inside the plan is ``gt_ok`` False.
  * the TRAIN-only guard: a tick file that is not the train split, shares a clip with the eval-141 join, or holds a clip outside ``fit_clips.json`` is REFUSED; a TickBundle cannot be made from the eval split.
  * the fitter is HACC's: each copied function equals ``v8f_hacc.py``'s source by AST; lambda = 0 equals the clipped closed form (C-A2) to 1e-12; a fit recovers a known w; the deliberate regression (GT := P) gives w = 0.
  * arm B: the offline gate equals the MODEL's gated plan; the v* rule picks the smallest qualifying grid value, falls back to 8, and reports the other readings of its two clauses.
  * the bake: a NEW file (never an overwrite), every source tensor carried bit for bit plus exactly the new buffer(s), the baked file's ``forward_infer`` equal to the runtime-set model's, CA1 equal to HACC's shipped
    tensors. [mutants: the gate not set; the weights not set]
  * C-A1 / C-B1 (``compare-dumps``), C-B2 (``check-b2``) and C-T (``check-ct``) readers pass equal inputs and FAIL a 1e-6 perturbation, a missing file, and (C-B2) a gate that also swallows the along axis.
  * REACHABILITY end to end: a baked arm-B checkpoint run through the T1 CLI records plans equal to the formula applied to the recorded tau, P, v0 and the table. [mutants: a bake without the gate; a different table]
"""
from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_hlat as HL  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
SEED = "M2b_s0"


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_hlat")
    w = D.make_synthetic_world(root, n_train=3, n_eval=3, T=90, seed=5)
    return {"root": root, **w}


def setup(world, split="eval"):
    root = Path(world["root"])
    tracks = D.load_native_tracks(root / "native_tracks.pt")
    J = D.load_join(root / f"join_{split}.npz")
    store = D.ClipStore(root, split, lru=8, kin_mode="yaw", tracks=tracks)
    if split == "eval":
        wins = D.eval_grid_windows(root / "eval_grid_windows.json", J)
    else:
        wins = D.train_windows(store, J, world["names"]["train"])
    return tracks, J, store, wins


def tiny_model(J, seed=0, hacc=True, gate=None):
    cfg = M.V8FMiniConfig(**TINY)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    prior = M.P10Prior.synthetic(0)
    with torch.no_grad():
        for q in prior.parameters():
            q.mul_(0.02)
    model = M.V8FMini(cfg, prior, mu, sd).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.head_o[2].weight, std=0.02)
    if hacc:
        model.set_hacc_weights(table(1))
    if gate is not None:
        model.set_hacc_gate(gate)
    return model, cfg


def table(seed=1):
    w = np.random.default_rng(seed).uniform(0.2, 0.9, (30, 2))
    w[:3, 1] = 0.0
    return w


def ds_for(cfg, J):
    proxy = T1._OneClipStore()
    return D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s), proxy


def rollout_with_ticks(model, cfg, world, chunk, m, split="eval", executor="plan", nav="deadreckon"):
    tracks, J, store, wins = setup(world, split)
    ds, proxy = ds_for(cfg, J)
    mp = T1.model_planner(model, "cpu", keep_aux=True)
    rec = T1.TickRecorder(mp, store, J)
    states = T1.make_states(store, tracks, J, chunk, nav)
    logged = {ci: tracks[str(J["clip"][ci])] for ci in {c[0] for c in chunk}}
    plan0, exe = T1.rollout(mp, states, ds, proxy, m=m, device="cpu", hook=rec, executor=executor, logged_tracks=logged)
    return rec, plan0, exe, store, J, states


def write_ckpt(path, model, cfg):
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}, path)


# ------------------------------------------------------------------------------------------------ C-T2 and the recorder
def test_tick0_equals_open_loop_batch_at_tick0_ages(world):
    model, cfg = tiny_model(setup(world)[1])
    tracks, J, store, wins = setup(world)
    chunk = wins[:6] + wins[20:24]
    rec, plan0, exe, store, J, _ = rollout_with_ticks(model, cfg, world, chunk, m=10)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    full = D.collate([ds.item(ci, t, 0, 0) for ci, t, _, _ in chunk])
    with torch.no_grad():
        out = model.forward_infer({k: full[k] for k in T1.MODEL_KEYS})
    z = {k: np.asarray(v) for k, v in rec.rows.items()}
    t0 = np.where(z["tick"] == 0)[0]
    t0 = t0[np.argsort(z["win"][t0])]
    assert len(t0) == len(chunk)
    assert np.array_equal(z["tau"][t0], out["traj30_o"].double().numpy()) and np.array_equal(z["P"][t0], out["p10"].double().numpy())
    assert np.array_equal(z["plan"][t0], out["traj30"].double().numpy()) and np.array_equal(z["v0"][t0], full["p10f"][:, -1, 0].double().numpy())
    assert np.array_equal(z["gt"][t0].astype(np.float32), full["gt"].numpy()) and z["gt_ok"][t0].all()
    assert (z["j"][t0] == z["t0"][t0]).all() and sorted(set(z["tick"].tolist())) == [0, 1, 2] and len(z["tick"]) == 3 * len(chunk)
    assert np.array_equal(plan0, out["traj30"].numpy())                                    # and the tool's own tick-0 plans are the same numbers


def test_gt_follows_the_executed_frame_and_the_logged_executor_reproduces_the_store(world):
    model, cfg = tiny_model(setup(world)[1])
    tracks, J, store, wins = setup(world)
    chunk = [w for w in wins if w[0] == 0][:3]
    rec, *_ = rollout_with_ticks(model, cfg, world, chunk, m=10, executor="logged")
    z = {k: np.asarray(v) for k, v in rec.rows.items()}
    d = store.get(str(J["clip"][0]))
    for i in range(len(z["tick"])):
        j = int(z["j"][i])
        if j + 30 <= len(d["pose"]) - 1:
            assert z["gt_ok"][i] and np.array_equal(z["gt"][i].astype(np.float32), d["gt"][j]), (i, j)       # logged executor: executed frame == logged frame at every tick
    # the plan executor: the executed pose differs, and the GT follows it by the same rotation arithmetic
    rec2, *_ = rollout_with_ticks(model, cfg, world, chunk, m=10, executor="plan")
    z2 = {k: np.asarray(v) for k, v in rec2.rows.items()}
    for i in np.where(z2["tick"] > 0)[0]:
        if z2["gt_ok"][i]:
            g, ok = T1.gt_in_frame(d["pose"], z2["pose"][i], int(z2["j"][i]))
            assert ok and np.array_equal(g, z2["gt"][i])
            assert not np.allclose(z2["pose"][i][:3], d["pose"][int(z2["j"][i])][:3])                           # it is NOT the logged pose
            wrong, _ = T1.gt_in_frame(d["pose"], d["pose"][int(z2["j"][i])], int(z2["j"][i]))                    # the mutant: the GT in the logged frame
            assert not np.allclose(wrong, z2["gt"][i], atol=1e-3)


def test_gt_in_frame_is_nan_when_the_clip_ends_inside_the_plan():
    pg = np.cumsum(np.ones((40, 4)), 0)
    g, ok = T1.gt_in_frame(pg, pg[5], 5)
    assert ok and g.shape == (30, 2) and np.isfinite(g).all()
    g, ok = T1.gt_in_frame(pg, pg[10], 10)                                                  # 10 + 30 > 39
    assert not ok and np.isnan(g).all()
    g, ok = T1.gt_in_frame(pg, pg[9], 9)                                                    # 9 + 30 = 39 = T - 1: the last legal tick
    assert ok


def test_recorder_keeps_the_gated_plan_the_speed_and_the_gate_flags(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J, gate=6.0)
    rec, *_ = rollout_with_ticks(model, cfg, world, wins[:12], m=10)
    z = {k: np.asarray(v) for k, v in rec.rows.items()}
    fired = z["v0"] < 6.0
    assert (z["fired"] == fired).all() and 0 < fired.sum() <= len(fired)
    W = np.broadcast_to(table(1)[None], (len(fired), 30, 2)).copy()
    W[fired, :, 1] = 0.0
    ref = z["P"] + W * (z["tau"] - z["P"])
    assert np.abs(z["plan"] - ref).max() < 1e-5
    assert (z["plan"][fired][..., 1] == z["P"][fired][..., 1]).all()
    assert set(rec.clip) <= set(world["names"]["eval"]) and len(rec.clip) == len(z["tick"])


# ------------------------------------------------------------------------------------------------ C-T through the CLI
def run_cli(world, tmp, tag, extra=(), ck="ck.pt", model=None):
    root = Path(world["root"])
    if model is not None:
        write_ckpt(tmp / ck, *model)
    rc = T1.main(["run", "--ckpt", str(tmp / ck), "--tag", tag, "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"),
                  "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(tmp / tag), "--replan-steps", "10", "--batch", "6", "--max-windows", "12", "--n-boot", "20", *extra])
    assert rc == 0
    return tmp / tag


def test_ct_dump_ticks_changes_no_output_bit(world, tmp_path):
    _, J, _, _ = setup(world)
    model = tiny_model(J)
    a = run_cli(world, tmp_path, "plain", model=model)
    b = run_cli(world, tmp_path, "plain_b", extra=("--dump-ticks", str(tmp_path / "ticks")))
    # same tag => same file names inside; compare dumps bit for bit and the JSON but for the extra key
    for f in sorted((a / "dump_plain").iterdir()):
        za, zb = np.load(f), np.load(b / "dump_plain_b" / f.name)
        assert set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k]) and za[k].dtype == zb[k].dtype for k in za.files), f.name
    ja, jb = json.load(open(a / "t1v8f_plain.json")), json.load(open(b / "t1v8f_plain_b.json"))
    assert jb["t1v8f"].pop("ticks_file").endswith("ticks_plain_b.npz")
    for j in (ja, jb):
        j["t1v8f"].pop("tag")
    assert ja == jb
    assert HL.main(["check-ct", "--a", str(a), "--b", str(a), "--out", str(tmp_path / "ct.json")]) == 0
    z = np.load(tmp_path / "ticks" / "ticks_plain_b.npz")
    assert str(z["split"]) == "eval" and int(z["n_windows"]) == 12 and len(z["tau"]) == 36 and z["tau"].shape[1:] == (30, 2)
    # the tick-0 plans are the dump's `ol`
    d = {(int(c), int(t)): ol for f in (b / "dump_plain_b").iterdir() for c, t, ol in zip(np.load(f)["clip_index"], np.load(f)["ws"], np.load(f)["ol"])}
    for i in np.where(z["tick"] == 0)[0]:
        assert np.array_equal(z["plan"][i], d[(int(z["ci"][i]), int(z["t0"][i]))])


def test_ct_flagless_run_has_no_tick_machinery_in_its_header(world, tmp_path):
    _, J, _, _ = setup(world)
    a = run_cli(world, tmp_path, "nof", model=tiny_model(J))
    h = json.load(open(a / "t1v8f_nof.json"))["t1v8f"]
    assert "ticks_file" not in h and "hacc_gate_v" not in h and not (tmp_path / "ticks").exists()


def test_train_mode_writes_tick_samples_only_and_refuses_without_its_inputs(world, tmp_path):
    root = Path(world["root"])
    _, J, _, _ = setup(world, "train")
    model = tiny_model(J)
    write_ckpt(tmp_path / "ck.pt", *model)
    fc = tmp_path / "fit_clips.json"
    json.dump({"clips": world["names"]["train"]}, open(fc, "w"))
    base = ["run", "--ckpt", str(tmp_path / "ck.pt"), "--tag", "tr", "--tracks", str(root / "native_tracks.pt"), "--windows", "train", "--train-cache", str(root),
            "--train-join", str(root / "join_train.npz"), "--fit-clips", str(fc), "--out-dir", str(tmp_path / "o"), "--replan-steps", "10", "--batch", "7"]
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.main(base)                                                                       # no --dump-ticks
    assert T1.main([*base, "--dump-ticks", str(tmp_path / "tk")]) == 0
    z = np.load(tmp_path / "tk" / "ticks_tr.npz")
    n = int(z["n_windows"])
    assert str(z["split"]) == "train" and len(z["tau"]) == 3 * n and set(map(str, z["clip"])) <= set(world["names"]["train"])
    assert not any(c in set(world["names"]["eval"]) for c in map(str, z["clip"]))
    assert not list((tmp_path / "o").glob("dump_*")) and (tmp_path / "o" / "t1v8f_tr.json").exists()      # no floor rollout, no dumps, no analysis
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.main(["run", "--ckpt", str(tmp_path / "ck.pt"), "--tag", "x", "--tracks", str(root / "native_tracks.pt"), "--out-dir", str(tmp_path / "o2")])    # eval mode without a grid


# ------------------------------------------------------------------------------------------------ the train-only guard
def tick_file(path, world, split="train", clips=None, eval_names=None, n=40, seed=0):
    rng = np.random.default_rng(seed)
    clips = clips if clips is not None else world["names"]["train"]
    P = np.cumsum(np.full((n, 30, 2), 0.5) * np.array([1.0, 0.02]), 1)
    tau = P + rng.normal(0, 0.3, (n, 30, 2))
    y = P + 0.5 * (tau - P) + rng.normal(0, 0.05, (n, 30, 2))
    np.savez(path, tau=tau, P=P, plan=P, gt=y, gt_ok=np.ones(n, bool), v0=rng.uniform(0, 12, n), clip=np.array([clips[i % len(clips)] for i in range(n)]), split=split,
             n_windows=n, tick=np.zeros(n, int), win=np.arange(n), fired=np.zeros(n, bool))
    return path


def hl_inputs(world, tmp_path, fit=None, evalnames=None):
    fc = tmp_path / "fc.json"
    json.dump({"clips": fit if fit is not None else world["names"]["train"]}, open(fc, "w"))
    return dict(eval_join=str(Path(world["root"]) / "join_eval.npz"), fit_clips=str(fc), banked_w=np.full((30, 2), 0.5), expect_windows=None)


def test_tick_bundle_accepts_the_train_split_and_counts(world, tmp_path):
    f = tick_file(tmp_path / "t.npz", world)
    T, info = HL.TickBundle.from_ticks(f, SEED, **hl_inputs(world, tmp_path))
    assert T.split == "train" and len(T.tau) == 40 and info["n_gt_ok"] == 40 and info["n_clips"] == 3 and "disjoint" in info["checked"]


@pytest.mark.parametrize("kind", ["eval_split", "eval_clip", "clip_not_in_fit", "short_windows"])
def test_tick_bundle_refuses_anything_that_could_carry_eval_data(world, tmp_path, kind):
    kw = hl_inputs(world, tmp_path)
    if kind == "eval_split":
        f = tick_file(tmp_path / "t.npz", world, split="eval")
        exc = HL.EvalLeakError
    elif kind == "eval_clip":
        f = tick_file(tmp_path / "t.npz", world, clips=world["names"]["train"] + world["names"]["eval"][:1])
        exc = HL.EvalLeakError
    elif kind == "clip_not_in_fit":
        f = tick_file(tmp_path / "t.npz", world)
        kw = hl_inputs(world, tmp_path, fit=world["names"]["train"][:2])
        exc = HL.EvalLeakError
    else:
        f = tick_file(tmp_path / "t.npz", world)
        kw["expect_windows"] = HL.N_TRAIN_WINDOWS
        exc = ValueError
    with pytest.raises(exc):
        HL.TickBundle.from_ticks(f, SEED, **kw)
    with pytest.raises(ValueError):
        HL.TickBundle(tau=np.zeros((2, 30, 2)), P=np.zeros((2, 30, 2)), y=np.zeros((2, 30, 2)), v0=np.zeros(2), banked_w=np.zeros((30, 2)), split="eval")
    with pytest.raises(TypeError):
        HL.select_lambda({"tau": 1})                                                         # the lambda rule takes a TickBundle alone


def test_a_dropped_gt_not_ok_sample_never_enters_the_bundle(world, tmp_path):
    f = tick_file(tmp_path / "t.npz", world)
    z = dict(np.load(f))
    z["gt_ok"][::4] = False
    z["gt"][::4] = np.nan
    np.savez(f, **z)
    T, info = HL.TickBundle.from_ticks(f, SEED, **hl_inputs(world, tmp_path))
    assert len(T.tau) == 30 and info["n_samples"] == 40 and np.isfinite(T.y).all()


# ------------------------------------------------------------------------------------------------ the fitter is HACC's
def _fn_dumps(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {n.name: ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names}


COPIED = ("d2_matrix", "stack_system", "kkt_rel", "fit_w", "terms_at", "qualifies", "choose_lambda")
HACC_SRC = HL.HACC_RUN / "v8f_hacc.py"
HBLEND_SRC = HL.REPO / HL.PK / "raw" / "2026-10-25-hblend" / "v8f_hblend.py"


@pytest.mark.skipif(not HACC_SRC.exists(), reason="v8f_hacc.py is not in this tree")
def test_copied_functions_equal_haccs_source_by_ast():
    a, b = _fn_dumps(HACC_SRC, COPIED), _fn_dumps(HL.__file__, COPIED)
    assert set(a) == set(b) == set(COPIED)
    assert {k: a[k] == b[k] for k in COPIED} == {k: True for k in COPIED}, [k for k in COPIED if a[k] != b[k]]
    if HBLEND_SRC.exists():
        assert _fn_dumps(HBLEND_SRC, ["apply_w", "arm_pw"])["apply_w"] == _fn_dumps(HL.__file__, ["apply_w"])["apply_w"]
    src = HACC_SRC.read_text(encoding="utf-8")
    assert "RULE_RATIO = 0.98" in src and HL.RULE_RATIO == 0.98 and "KKT_TOL = 1e-6" in src and HL.KKT_TOL == 1e-6
    assert HL.GRID_M_A == (0, 1, 4, 16, 64)                                                  # HACC's grid without 256: the PREREG's arm A grid


def known_world(n=250, seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    P = np.cumsum(np.full((n, 30, 2), 0.5) * np.array([1.0, 0.03]), 1)
    tau = P + rng.normal(0, 0.8, (n, 30, 2))
    w_true = np.stack([np.linspace(0.2, 0.9, 30), np.linspace(0.0, 0.6, 30)], 1)
    y = P + w_true * (tau - P) + rng.normal(0, noise, (n, 30, 2))
    return tau, P, y, w_true


def test_lambda0_fit_recovers_a_known_w_and_equals_the_clipped_closed_form():
    tau, P, y, w_true = known_world()
    f = HL.fit_w(tau, P, y, 0.0)
    cf = HL.clipped_closed_form(tau, P, y)
    assert np.abs(f["w"] - cf).max() <= 1e-12                                                # C-A2's bar
    assert np.abs(f["w"] - w_true).max() < 0.05 and max(f["kkt"]) <= HL.KKT_TOL
    fl = HL.fit_w(tau, P, y, 50.0)
    assert max(fl["kkt"]) <= HL.KKT_TOL and np.abs(fl["w"] - w_true).max() < 0.08


def test_regression_gt_is_p_gives_w_zero_and_a_broken_solver_trips_the_kkt_guard():
    tau, P, _, _ = known_world(120)
    f = HL.fit_w(tau, P, P, 10.0)
    assert np.abs(f["w"]).max() <= 1e-12

    def bad_solver(A, b, bounds, method, tol, max_iter):
        class R:
            x = np.full(A.shape[1], 0.5)
            cost, nit, status = 0.0, 0, 0
        return R()
    tau, P, y, _ = known_world(120)
    with pytest.raises(HL.KKTError):
        HL.fit_w(tau, P, y, 0.0, solver=bad_solver)


def test_select_lambda_applies_the_rule_on_the_tick_samples(world, tmp_path):
    tau, P, y, _ = known_world(160)
    T = HL.TickBundle(tau=tau, P=P, y=y, v0=np.random.default_rng(0).uniform(0, 12, 160), banked_w=np.full((30, 2), 0.5))
    sel = HL.select_lambda(T, grid=(0, 4))
    assert [r["m"] for r in sel["grid"]] == [0, 4] and sel["lambda0"] > 0 and sel["m_star"] in (0, 4)
    first = next((r for r in sel["grid"] if r["qualifies"]), None)
    if first is not None:
        assert sel["m_star"] == first["m"]                                                   # the smallest qualifying grid lambda
    assert sel["w_star"].shape == (30, 2) and ((sel["w_star"] >= 0) & (sel["w_star"] <= 1)).all()
    ref = HL.terms_at(T.tau, T.P, T.y, T.banked_w)
    assert sel["lambda0"] == ref[0] / ref[1]


# ------------------------------------------------------------------------------------------------ arm B
def test_offline_gate_equals_the_models_gated_plan(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J, gate=5.0)
    rec, *_ = rollout_with_ticks(model, cfg, world, wins[:12], m=10)
    z = {k: np.asarray(v) for k, v in rec.rows.items()}
    off = HL.apply_gated(z["P"], z["tau"], table(1), z["v0"], 5.0)
    assert np.abs(off - z["plan"]).max() < 1e-5
    assert np.array_equal(HL.apply_gated(z["P"], z["tau"], table(1), z["v0"], 0.0), HL.apply_w(z["P"], z["tau"], table(1)))      # v* = 0: HACC
    inf = HL.apply_gated(z["P"], z["tau"], table(1), z["v0"], float("inf"))
    assert np.array_equal(inf[..., 1], z["P"][..., 1]) and not np.array_equal(inf[..., 0], z["P"][..., 0])


def vstar_world(n=600, seed=0):
    """Three speed bands. v0 < 2 and 2 <= v0 < 4: O's cross plan carries a small POSITIONAL wobble (0.05 m: invisible to an ADE, a large curvature at walking pace -- the failure the PREREG diagnoses). v0 >= 4:
    O's cross plan carries the true slow lateral signal, so HACC's cross weight HELPS there. y cross = the signal everywhere, P cross = 0."""
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(0.2, 14.0, n)
    s = np.arange(1, 31)[None, :, None] * 0.2
    P = np.concatenate([v0[:, None, None] * s, np.zeros((n, 30, 1))], -1)
    sig = np.sin(np.arange(1, 31)[None, :] * 0.1 + rng.uniform(0, 1, n)[:, None])
    wob = 0.05 * np.sin(np.arange(30)[None, :] * 1.7 + rng.uniform(0, 6, n)[:, None])
    y = P.copy()
    y[..., 1] = sig
    tau = P.copy()
    tau[..., 1] = np.where((v0 < 4.0)[:, None], wob, sig)
    return tau, P, y, v0


def test_vstar_rule_picks_the_smallest_qualifying_value_and_reports_other_readings():
    tau, P, y, v0 = vstar_world()
    T = HL.TickBundle(tau=tau, P=P, y=y, v0=v0, banked_w=np.full((30, 2), 0.5))
    w = np.stack([np.ones(30), np.full(30, 0.8)], 1)
    r = HL.select_vstar(T, w)
    assert r["v_star"] == 4.0, r["grid"]                                                         # 2 leaves the wobbly 2..4 m/s samples ungated (curvature worse than P10); 4 gates every wobbly sample
    quals = [x["v_star"] for x in r["grid"] if x["qualifies"]]
    assert quals and r["v_star"] == min(quals) and r["branch"] == "qualified"
    assert set(r["reading_sensitivity"]) == {"one_sided_ade6", "exact_F_half", "same_as_used"}
    # nothing can qualify when no sample is slow enough to be gated: HACC-V == HACC, whose curvature on the wobbly samples is worse than P10's -> fallback 8
    bad = HL.select_vstar(HL.TickBundle(tau=tau, P=P, y=y, v0=np.full(len(v0), 20.0), banked_w=np.full((30, 2), 0.5)), w)
    assert bad["v_star"] == HL.V_B_FALLBACK and bad["branch"].startswith("fallback")
    with pytest.raises(TypeError):
        HL.select_vstar({"tau": 1}, w)


# ------------------------------------------------------------------------------------------------ the bake
def fake_plain(world, tmp_path):
    _, J, _, _ = setup(world, "train")
    model, cfg = tiny_model(J, hacc=False)
    write_ckpt(tmp_path / "plain.pt", model, cfg)
    return tmp_path / "plain.pt", model


def small_batch(world):
    tracks, J, store, wins = setup(world, "train")
    cfg = M.V8FMiniConfig(**TINY)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    full = D.collate([ds.item(ci, t, 1, 3) for ci, t, _, _ in wins[::7][:10]])
    return {k: full[k] for k in T1.MODEL_KEYS}


@pytest.fixture()
def fit_dir(tmp_path):
    d = tmp_path / "fit"
    d.mkdir()
    HL.write_table(d / f"weights_A_{SEED}.json", table(9))
    json.dump({"v_star": 4.0}, open(d / f"gate_B_{SEED}.json", "w"))
    return d


@pytest.mark.parametrize("arm", ["A", "B", "CB1", "CB2"])
def test_bake_carries_the_source_and_adds_exactly_the_new_buffers(world, tmp_path, fit_dir, arm):
    plain, src = fake_plain(world, tmp_path)
    out = tmp_path / f"{arm}.pt"
    rep = HL.bake(arm, SEED, str(out), fit_dir=str(fit_dir), plain_path=str(plain), verify_batch=small_batch(world))
    assert rep["carried_bit_for_bit"] and rep["baked_vs_runtime"].startswith("bit-identical")
    assert rep["new_keys"] == (["hacc_gate_v", "hacc_w"] if arm in ("B", "CB1", "CB2") else ["hacc_w"])
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert ck["hlat"]["arm"] == arm and ck["hlat"]["prereg"] == "PREREG_V8F_HACC_LAT"
    if arm == "B":
        assert float(ck["model"]["hacc_gate_v"]) == 4.0
    if arm == "CB2":
        assert float(ck["model"]["hacc_gate_v"]) == float("inf") and rep["n_fired"] == rep["n_verify_windows"]
    if arm == "CB1":
        assert rep["n_fired"] == 0
    with pytest.raises(FileExistsError):
        HL.bake(arm, SEED, str(out), fit_dir=str(fit_dir), plain_path=str(plain))


def test_bake_mutants_are_caught(world, tmp_path, fit_dir, monkeypatch):
    plain, _ = fake_plain(world, tmp_path)
    vb = small_batch(world)
    with monkeypatch.context() as mp:                                                        # the gate is not set
        mp.setattr(M.V8FMini, "set_hacc_gate", lambda self, v: None)
        with pytest.raises((AssertionError, KeyError)):
            HL.bake("B", SEED, str(tmp_path / "m1.pt"), fit_dir=str(fit_dir), plain_path=str(plain), verify_batch=vb)
    with monkeypatch.context() as mp:                                                        # the weights are not set
        mp.setattr(M.V8FMini, "set_hacc_weights", lambda self, w: None)
        with pytest.raises((AssertionError, ValueError)):
            HL.bake("A", SEED, str(tmp_path / "m2.pt"), fit_dir=str(fit_dir), plain_path=str(plain), verify_batch=vb)


# ------------------------------------------------------------------------------------------------ the readers of the controls
def fake_dump(d, scale=1.0, drop=False, n_ep=3):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for e in range(n_ep - int(drop)):
        n = 4
        np.savez(d / f"ep{e:04d}.npz", g=rng.normal(size=(n, 30, 2)), ws=np.arange(n), eid=np.full(n, e), clip_index=np.full(n, e), v0=rng.uniform(0, 9, n),
                 cl=rng.normal(size=(n, 30, 2)) * scale, ol=rng.normal(size=(n, 30, 2)), fl=rng.normal(size=(n, 30, 2)))


def test_compare_dumps_reader_passes_equal_and_fails_a_perturbation_or_a_missing_file(tmp_path):
    fake_dump(tmp_path / "a")
    fake_dump(tmp_path / "b")
    args = lambda b, tol="1e-9": ["compare-dumps", "--a", str(tmp_path / "a"), "--b", str(b), "--label", "C-A1", "--tol", tol, "--out", str(tmp_path / "o.json")]
    assert HL.main(args(tmp_path / "b")) == 0
    z = dict(np.load(tmp_path / "b" / "ep0001.npz"))
    z["cl"] = z["cl"] + 1e-6
    np.savez(tmp_path / "b" / "ep0001.npz", **z)
    assert HL.main(args(tmp_path / "b")) == 2                                                # 1e-6 > 1e-9
    assert HL.main(args(tmp_path / "b", "1e-5")) == 0
    fake_dump(tmp_path / "c", drop=True)
    assert HL.main(args(tmp_path / "c")) == 2


def test_check_b2_reader(tmp_path):
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 30, 2))
    plan = P + np.stack([rng.normal(size=(20, 30)), np.zeros((20, 30))], -1)
    base = dict(P=P, plan=plan, split="train", fired=np.ones(20, bool))
    np.savez(tmp_path / "ok.npz", **base)
    assert HL.main(["check-b2", "--ticks", str(tmp_path / "ok.npz"), "--out", str(tmp_path / "o.json")]) == 0
    bad = plan.copy()
    bad[3, 5, 1] += 1e-5
    np.savez(tmp_path / "bad.npz", **{**base, "plan": bad})
    assert HL.main(["check-b2", "--ticks", str(tmp_path / "bad.npz"), "--out", str(tmp_path / "o.json")]) == 2
    np.savez(tmp_path / "along.npz", **{**base, "plan": P.copy()})                           # a gate that swallows the along axis too: no power
    assert HL.main(["check-b2", "--ticks", str(tmp_path / "along.npz"), "--out", str(tmp_path / "o.json")]) == 2
    np.savez(tmp_path / "nofire.npz", **{**base, "fired": np.zeros(20, bool)})
    assert HL.main(["check-b2", "--ticks", str(tmp_path / "nofire.npz"), "--out", str(tmp_path / "o.json")]) == 2


def test_check_ct_reader(tmp_path):
    fake_dump(tmp_path / "a" / "dump_x")
    fake_dump(tmp_path / "b" / "dump_x")
    json.dump({"k": 1}, open(tmp_path / "a" / "t.json", "w"))
    json.dump({"k": 1}, open(tmp_path / "b" / "t.json", "w"))
    args = ["check-ct", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "b"), "--out", str(tmp_path / "o.json")]
    assert HL.main(args) == 0
    json.dump({"k": 2}, open(tmp_path / "b" / "t.json", "w"))
    assert HL.main(args) == 2
    json.dump({"k": 1}, open(tmp_path / "b" / "t.json", "w"))
    z = dict(np.load(tmp_path / "b" / "dump_x" / "ep0000.npz"))
    z["cl"] = np.nextafter(z["cl"], np.inf)                                                  # one ulp
    np.savez(tmp_path / "b" / "dump_x" / "ep0000.npz", **z)
    assert HL.main(args) == 2
    fake_dump(tmp_path / "b" / "dump_x")
    json.dump({"k": 1}, open(tmp_path / "b" / "extra.json", "w"))
    assert HL.main(args) == 2                                                                # an extra file in b


# ------------------------------------------------------------------------------------------------ reachability end to end
def formula_gap(rec_rows, w, vstar):
    z = {k: np.asarray(v) for k, v in rec_rows.items()}
    ref = HL.apply_gated(z["P"], z["tau"], w, z["v0"], vstar)
    return float(np.abs(ref - z["plan"]).max()), int(z["fired"].sum())


def test_a_baked_arm_b_checkpoint_reaches_the_executed_plan_through_the_cli(world, tmp_path, fit_dir):
    plain, _ = fake_plain(world, tmp_path)
    w_hacc = HL.read_table(HL.HACC_W_JSON[SEED])
    json.dump({"v_star": 8.0}, open(fit_dir / f"gate_B_{SEED}.json", "w"))                       # the synthetic drive sits around 8 m/s: both sides of the gate occur
    HL.bake("B", SEED, str(tmp_path / "B.pt"), fit_dir=str(fit_dir), plain_path=str(plain))
    run_cli(world, tmp_path, "armB", extra=("--dump-ticks", str(tmp_path / "tk")), ck="B.pt")
    z = {k: v for k, v in np.load(tmp_path / "tk" / "ticks_armB.npz").items()}
    ref = HL.apply_gated(z["P"], z["tau"], w_hacc, z["v0"], 8.0)
    gap = float(np.abs(ref - z["plan"]).max())
    assert gap < 1e-5 and 0 < z["fired"].sum() < len(z["fired"])
    # the mutants of this chain: a bake without the gate, a different table -> the same formula check fails
    assert float(np.abs(HL.apply_w(z["P"], z["tau"], w_hacc) - z["plan"]).max()) > 1e-3                 # no gate would differ from what ran
    assert float(np.abs(HL.apply_gated(z["P"], z["tau"], table(9), z["v0"], 8.0) - z["plan"]).max()) > 1e-3
    assert float(json.load(open(tmp_path / "armB" / "t1v8f_armB.json"))["t1v8f"]["hacc_gate_v"]) == 8.0
