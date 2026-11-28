"""T1-HEADS: ``--dump-ticks`` and ``--dump-heads`` of ``stack/scripts/v8f_t1_eval.py`` -- the per-tick S / T head outputs the T1-kin dumps lacked. CPU only, synthetic world, 7 small CLI runs (measured 1.5-3 min on a loaded dev box).

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval;<repo>/stack/scripts  CUDA_VISIBLE_DEVICES=-1  pytest -q stack/tests/test_v8f_t1_heads.py

Why it exists: ``RESULT_V8F_T1KIN_HACC.md`` could not compute the STRATEGIC and the model's own TACTICAL family at T1 -- the dumps carried trajectories only. This pins the recorder that gives the
rows the scorer needs, and the controls that must read EXACTLY as stated (no test edits a file on disk: every mutant is an in-process patch or a scratch copy, see ``out/mutation.out``).

What is pinned, and the control that must go red for each:
  * C-OFF   both flags absent => the dumps and the result JSON are byte-identical to the PRE-FLAG tool's on the same input. The reference is the tip blob ``635487e5:stack/scripts/v8f_t1_eval.py``,
            passed as ``V8F_T1_TIP_TOOL=<file>`` and pinned by sha256 (a wrong file is REFUSED); without it this one test SKIPS and the next two still bind the flag-off path.
  * C-ON    ``--dump-ticks`` and ``--dump-ticks --dump-heads`` change no dump array and no JSON value but the two keys they add (``ticks_file``, ``ticks_heads``).
  * SCHEMA  ``--dump-ticks`` alone writes exactly the carried (hlat) columns, no head column; ``--dump-heads`` adds exactly :data:`T1.TickRecorder.HEAD_FIELDS`.
  * TICK-0  the dump's tick-0 head rows equal a DIRECT ``forward_infer`` call on the tick-0 batch (built by the open-loop item builder, not by the T1 machinery) to <= 1e-6. [mutant: delete the T logits line]
  * EVERY TICK  at every tick of a rolled window the recorded rows equal a direct ``forward_infer`` on that tick's batch, ages != 0 included; ``s_pose`` / ``t_pose`` are the executed poses at
            ``j - age_s`` / ``j - age_t``.
  * FRAMES  ``s_ckpt`` (S tick frame), ``s_ckpt_now`` and ``s_ckpt_w0`` denote the same WORLD points (plain rotation + translation, not ``transform_ckpt``), and the check is not vacuous (the frames differ).
  * the flag guard: ``--dump-heads`` without ``--dump-ticks`` is REFUSED; a recorder asked for heads by a planner that keeps none raises; the P10 floor records nothing.
  * the carried ``--dump-ticks`` invariants (hlat C-T2): tick-0 tau / P / plan / v0 / GT equal the open-loop batch's; the GT follows the EXECUTED frame; NaN GT when the clip ends inside the plan.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
import types
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
import v8f_t1_eval as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
#: sha256 of ``git show 635487e5:stack/scripts/v8f_t1_eval.py`` (LF): the tool as it was BEFORE these flags
TIP_SHA256 = "baf16101736d4599791e0b1ef1bd5eed3afc3c65dcc491d0aa663f1be3cbcb7d"
HEAD_PAIRS = (("s_ckpt", "ckpt"), ("t_goal_prob", "goal_prob"), ("t_lat_logits", "lat_logits"), ("t_lon_logits", "lon_logits"), ("t_cons_pred", "cons_pred"))
TICK_ONLY_KEYS = set(T1.TickRecorder.FIELDS) | {"clip", "split", "tag", "ckpt", "replan_steps", "n_windows", "clip_names"}


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_heads")
    w = D.make_synthetic_world(root, n_train=2, n_eval=3, T=90, seed=5)
    return {"root": root, **w}


def setup(world):
    root = Path(world["root"])
    tracks = D.load_native_tracks(root / "native_tracks.pt")
    J = D.load_join(root / "join_eval.npz")
    store = D.ClipStore(root, "eval", lru=8, kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(root / "eval_grid_windows.json", J)
    return tracks, J, store, wins


def tiny_model(J, seed=0, hacc=True):
    cfg = M.V8FMiniConfig(**TINY)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    prior = M.P10Prior.synthetic(0)
    with torch.no_grad():                         # a random prior is an UNSTABLE closed loop; keep it tame
        for q in prior.parameters():
            q.mul_(0.02)
    model = M.V8FMini(cfg, prior, mu, sd).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.head_o[2].weight, std=0.02)
    if hacc:                                      # the post-map on: tau (O before it) != plan, so the carried tau / plan columns are told apart
        model.set_hacc_weights(np.random.default_rng(1).uniform(0.2, 0.9, (30, 2)))
    return model, cfg


def ds_for(cfg, J):
    proxy = T1._OneClipStore()
    return D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s), proxy


def write_ckpt(path, model, cfg):
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}, path)


def load_tip_tool():
    """The pre-flag tool, by SOURCE (so it never touches the repo's copy): ``__file__`` is the repo script's, so its REPO / sys.path arithmetic resolves to this tree."""
    p = os.environ.get("V8F_T1_TIP_TOOL", "")
    if not p or not Path(p).is_file():
        pytest.skip("V8F_T1_TIP_TOOL=<635487e5:stack/scripts/v8f_t1_eval.py> not given")
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == TIP_SHA256, "REFUSED: that file is not the tip blob of v8f_t1_eval.py (sha256 pin)"
    mod = types.ModuleType("v8f_t1_eval_tip")
    mod.__file__ = str(Path(T1.__file__).resolve())
    exec(compile(src.decode("utf-8"), str(p), "exec"), mod.__dict__)
    return mod


def run_cli(world, tmp, label, *extra, tool=None, model=None, tag="syn", m=5):
    """One CLI run into ``tmp/<label>``; the same ``tag`` and ckpt path every time, so the JSONs of two runs can be compared as BYTES."""
    root = Path(world["root"])
    if model is not None:
        write_ckpt(tmp / "ck.pt", *model)
    rc = (tool or T1).main(["run", "--ckpt", str(tmp / "ck.pt"), "--tag", tag, "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root),
                            "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(tmp / label),
                            "--replan-steps", str(m), "--batch", "6", "--max-windows", "12", "--n-boot", "20", "--device", "cpu", *extra])
    assert rc == 0
    return tmp / label


def dump_arrays(out, tag="syn"):
    return {f.name: {k: np.load(f)[k] for k in np.load(f).files} for f in sorted((out / f"dump_{tag}").iterdir())}


def assert_same_dumps(a, b):
    """Array-for-array, bit for bit (np.savez stamps its zip members with the clock, so the container bytes are not comparable; the arrays are)."""
    da, db = dump_arrays(a), dump_arrays(b)
    assert da.keys() == db.keys() and len(da) >= 3
    for f in da:
        assert set(da[f]) == set(db[f]), f
        for k in da[f]:
            x, y = da[f][k], db[f][k]
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (f, k)


def ticks(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ C-OFF / C-ON
def test_c_off_flags_absent_is_byte_identical_to_the_tip_tool(world, tmp_path):
    tip = load_tip_tool()
    _, J, _, _ = setup(world)
    ours = run_cli(world, tmp_path, "ours", model=tiny_model(J))
    ref = run_cli(world, tmp_path, "tip", tool=tip)
    assert_same_dumps(ours, ref)
    a, b = (ours / "t1v8f_syn.json").read_bytes(), (ref / "t1v8f_syn.json").read_bytes()
    assert a == b and len(a) > 1000                                           # the result JSON: EXACTLY equal as bytes


def test_c_off_flags_absent_leaves_no_tick_machinery_in_the_output(world, tmp_path):
    _, J, _, _ = setup(world)
    out = run_cli(world, tmp_path, "off", model=tiny_model(J))
    h = json.load(open(out / "t1v8f_syn.json", encoding="utf-8"))["t1v8f"]
    assert "ticks_file" not in h and "ticks_heads" not in h
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("tick")) == [] and not list(out.glob("ticks_*"))
    mp = T1.model_planner(tiny_model(J)[0], "cpu")
    assert mp.aux is None and mp.heads is None                                # nothing kept unless asked


def test_c_on_the_flags_change_no_trajectory_dump_and_no_json_value(world, tmp_path):
    _, J, _, _ = setup(world)
    off = run_cli(world, tmp_path, "off", model=tiny_model(J))
    t = run_cli(world, tmp_path, "ticks", "--dump-ticks", str(tmp_path / "tk_t"))
    th = run_cli(world, tmp_path, "heads", "--dump-ticks", str(tmp_path / "tk_h"), "--dump-heads")
    j0 = json.load(open(off / "t1v8f_syn.json", encoding="utf-8"))
    for on, flag_heads in ((t, False), (th, True)):
        assert_same_dumps(off, on)
        j1 = json.load(open(on / "t1v8f_syn.json", encoding="utf-8"))
        h = j1["t1v8f"]
        assert h.pop("ticks_file").endswith("ticks_syn.npz") and h.pop("ticks_heads") is flag_heads
        assert j1 == j0                                                       # every other value of the result JSON is the flag-off run's


# ------------------------------------------------------------------------------------------------ the schema
def test_schema_ticks_alone_is_the_carried_one_and_heads_add_exactly_the_head_columns(world, tmp_path):
    _, J, _, _ = setup(world)
    run_cli(world, tmp_path, "t", "--dump-ticks", str(tmp_path / "tk_t"), model=tiny_model(J))
    run_cli(world, tmp_path, "h", "--dump-ticks", str(tmp_path / "tk_h"), "--dump-heads")
    z0, z1 = ticks(tmp_path / "tk_t" / "ticks_syn.npz"), ticks(tmp_path / "tk_h" / "ticks_syn.npz")
    N = 12 * (30 // 5)                                                        # 12 windows x 6 ticks (the model's ticks only: the floor is never recorded)
    assert set(z0) == TICK_ONLY_KEYS and not (set(z0) & set(T1.TickRecorder.HEAD_FIELDS))
    assert set(z1) == TICK_ONLY_KEYS | set(T1.TickRecorder.HEAD_FIELDS)
    for k in z0:                                                              # the carried columns are the same numbers with or without the head columns
        assert z0[k].dtype == z1[k].dtype and z0[k].shape == z1[k].shape and z0[k].tobytes() == z1[k].tobytes(), k
    want = {"s_ckpt": ((N, 3, 3), np.float32), "s_ckpt_now": ((N, 3, 3), np.float32), "s_ckpt_w0": ((N, 3, 3), np.float32), "s_pose": ((N, 3), np.float32),
            "t_goal_prob": ((N, 22), np.float32), "t_lat_logits": ((N, 5), np.float32), "t_lon_logits": ((N, 7), np.float32),
            "t_cons_pred": ((N, len(D.CONS_FIELDS)), np.float32), "t_pose": ((N, 3), np.float32), "pose_w0": ((N, 3), np.float64)}
    for k, (shape, dt) in want.items():
        assert z1[k].shape == shape and z1[k].dtype == dt and np.isfinite(z1[k]).all(), (k, z1[k].shape, z1[k].dtype)
    assert sorted(set(z1["tick"].tolist())) == list(range(6)) and (np.bincount(z1["tick"]) == 12).all()
    assert str(z1["split"]) == "eval" and int(z1["n_windows"]) == 12


# ------------------------------------------------------------------------------------------------ tick 0 vs a direct forward_infer
def test_tick0_head_rows_equal_a_direct_forward_infer_call(world, tmp_path):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    run_cli(world, tmp_path, "h", "--dump-ticks", str(tmp_path / "tk"), "--dump-heads", model=(model, cfg))
    z = ticks(tmp_path / "tk" / "ticks_syn.npz")
    r0 = np.where(z["tick"] == 0)[0]                                          # row order = chunk-major, window order inside a chunk
    assert len(r0) == 12 and (z["win"][r0] == np.arange(12)).all()
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    worst = {k: 0.0 for k, _ in HEAD_PAIRS}
    for c0 in range(0, 12, 6):                                                # the tool's own --batch 6 partition
        rows = r0[c0:c0 + 6]
        full = D.collate([ds.item(int(z["ci"][i]), int(z["t0"][i]), 0, 0) for i in rows])      # the OPEN-LOOP builder: nothing of the T1 machinery
        with torch.no_grad():
            out = model.forward_infer({k: full[k] for k in T1.MODEL_KEYS})
        for kd, ko in HEAD_PAIRS:
            ref = out[ko].float().numpy()
            assert z[kd][rows].shape == ref.shape, (kd, z[kd][rows].shape, ref.shape)
            worst[kd] = max(worst[kd], float(np.abs(z[kd][rows].astype(np.float64) - ref).max()))
        assert np.abs(z["plan"][rows] - out["traj30"].double().numpy()).max() <= 1e-6        # the recorded heads are of the call that made the plan
    assert max(worst.values()) <= 1e-6, worst
    # the comparison is not vacuous: every head output varies across windows and across ticks
    for kd, _ in HEAD_PAIRS:
        assert float(np.ptp(z[kd][r0], axis=0).max()) > 1e-6, f"{kd} is constant across windows at tick 0"
        assert float(np.ptp(z[kd], axis=0).max()) > 1e-6, f"{kd} is constant"
    assert (z["age_t"][r0] == 0).all() and (z["age_s"][r0] == 0).all()
    # at tick 0 the frames coincide: S's tick frame IS the window origin IS NOW (up to the float32 round trip through world coordinates of ~10^2 m: measured 1.5e-5)
    np.testing.assert_allclose(z["s_ckpt_now"][r0], z["s_ckpt"][r0], atol=1e-4)
    np.testing.assert_allclose(z["s_ckpt_w0"][r0], z["s_ckpt"][r0], atol=1e-4)


def rolled(world, model_cfg, chunk, m=5, heads=True, executor="plan"):
    tracks, J, store, wins = setup(world)
    model, cfg = model_cfg
    ds, proxy = ds_for(cfg, J)
    mp = T1.model_planner(model, "cpu", keep_aux=True, keep_heads=heads)
    rec = T1.TickRecorder(mp, store, J, heads=heads)
    states = T1.make_states(store, tracks, J, chunk, "deadreckon")
    return rec, mp, states, ds, proxy, store, J, tracks


def test_every_tick_head_rows_equal_a_direct_forward_infer_on_that_ticks_batch(world):
    _, J, _, wins = setup(world)
    model, cfg = tiny_model(J)
    chunk = wins[:5] + wins[20:24]
    rec, mp, states, ds, proxy, *_ = rolled(world, (model, cfg), chunk)
    seen, worst, ages = [], 0.0, set()

    def hook(n, batch, plan, sts):
        nonlocal worst
        before = len(rec.rows["tick"])
        rec(n, batch, plan, sts)
        with torch.no_grad():
            direct = model.forward_infer(batch)                                # a SECOND call on the very batch the tool's call saw
        for kd, ko in HEAD_PAIRS:
            got = np.stack(rec.rows[kd][before:]).astype(np.float64)
            assert got.shape == tuple(direct[ko].shape), (n, kd)
            worst = max(worst, float(np.abs(got - direct[ko].float().numpy()).max()))
        for b, st in enumerate(sts):                                          # the poses that frame the heads are the EXECUTED poses at the held ticks
            a_t, a_s = int(batch["a_t"][b]), int(batch["a_s"][b])
            ages.add((a_t, a_s))
            assert np.array_equal(rec.rows["s_pose"][before + b], st.d["pose"][st.j - a_s][:3].astype(np.float32)), (n, b, "s_pose")
            assert np.array_equal(rec.rows["t_pose"][before + b], st.d["pose"][st.j - a_t][:3].astype(np.float32)), (n, b, "t_pose")
            assert np.array_equal(rec.rows["pose_w0"][before + b], np.asarray(st.d["pose"][st.t0][:3], np.float64)), (n, b, "pose_w0")
        seen.append(n)

    T1.rollout(mp, states, ds, proxy, m=5, device="cpu", hook=hook)
    assert seen == list(range(6)) and worst <= 1e-6, (seen, worst)
    assert len(ages) >= 4 and any(a_s > 0 for _, a_s in ages) and any(a_t > 0 for a_t, _ in ages)      # the held-age schedule was exercised, not only (0, 0)


def to_world(ck, pose):
    """[N, M, 3] checkpoints in the frame of pose [N, 3] (x, y, yaw) -> world (x, y, psi), by plain rotation + translation."""
    c, s = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
    return np.stack([pose[:, 0:1] + c * ck[..., 0] - s * ck[..., 1], pose[:, 1:2] + s * ck[..., 0] + c * ck[..., 1], ck[..., 2] + pose[:, 2:3]], -1)


def test_ckpt_frames_denote_the_same_world_points(world, tmp_path):
    _, J, _, _ = setup(world)
    run_cli(world, tmp_path, "h", "--dump-ticks", str(tmp_path / "tk"), "--dump-heads", model=tiny_model(J))
    z = {k: v.astype(np.float64) if v.dtype == np.float32 else v for k, v in ticks(tmp_path / "tk" / "ticks_syn.npz").items()}
    w_s = to_world(z["s_ckpt"], z["s_pose"])
    w_now = to_world(z["s_ckpt_now"], z["pose"][:, :3])
    w_w0 = to_world(z["s_ckpt_w0"], z["pose_w0"])
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi                  # noqa: E731
    for w in (w_now, w_w0):
        assert np.abs(w[..., :2] - w_s[..., :2]).max() < 2e-3
        assert np.abs(wrap(w[..., 2] - w_s[..., 2])).max() < 2e-3
    # not vacuous: later ticks (held ages > 0, the car moved) have three DIFFERENT representations
    late = z["tick"] > 0
    assert np.abs(z["s_ckpt_now"] - z["s_ckpt"])[late].max() > 1e-2
    assert np.abs(z["s_ckpt_w0"] - z["s_ckpt_now"])[late].max() > 1.0
    # the executed pose at the tick is z['pose']; at tick 0 it IS the window origin
    r0 = z["tick"] == 0
    np.testing.assert_allclose(z["pose"][r0][:, :3], z["pose_w0"][r0], atol=1e-9)


# ------------------------------------------------------------------------------------------------ the guards
def test_dump_heads_without_dump_ticks_is_refused(world, tmp_path):
    _, J, _, _ = setup(world)
    write_ckpt(tmp_path / "ck.pt", *tiny_model(J))
    with pytest.raises(SystemExit, match="REFUSED"):
        run_cli(world, tmp_path, "x", "--dump-heads")
    assert not (tmp_path / "x").exists()


def test_a_recorder_asked_for_heads_by_a_planner_that_keeps_none_raises(world):
    _, J, _, wins = setup(world)
    model, cfg = tiny_model(J)
    rec, mp, states, ds, proxy, *_ = rolled(world, (model, cfg), wins[:3], heads=False)
    rec.heads = True                                                          # the recorder wants heads, the planner (keep_heads False) kept none
    with pytest.raises(RuntimeError, match="keep_heads"):
        T1.rollout(mp, states, ds, proxy, m=10, device="cpu", hook=rec)


def test_the_floor_planner_has_no_heads_and_is_never_recorded(world):
    _, J, _, _ = setup(world)
    model, cfg = tiny_model(J)
    assert getattr(T1.floor_planner(model, "cpu"), "heads", None) is None and not hasattr(T1.floor_planner(model, "cpu"), "keys_read")
    assert T1.model_planner(model, "cpu", keep_heads=True).heads is None      # before any call; filled by a call, and only on the model planner


def test_heads_ride_the_same_call_that_made_the_plan_not_a_second_one(world):
    _, J, _, wins = setup(world)
    model, cfg = tiny_model(J)
    calls = []
    real = model.forward_infer
    model.forward_infer = lambda b: (calls.append(1), real(b))[1]
    rec, mp, states, ds, proxy, *_ = rolled(world, (model, cfg), wins[:4])
    T1.rollout(mp, states, ds, proxy, m=10, device="cpu", hook=rec)
    assert len(calls) == 3                                                    # 30 / 10 ticks, one forward_infer each: the heads cost no extra model call


# ------------------------------------------------------------------------------------------------ the carried --dump-ticks invariants (hlat C-T2)
def test_carried_tick0_equals_the_open_loop_batch_at_tick0_ages(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    chunk = wins[:6] + wins[20:24]
    rec, mp, states, ds, proxy, *_ = rolled(world, (model, cfg), chunk, heads=False)
    plan0, _ = T1.rollout(mp, states, ds, proxy, m=10, device="cpu", hook=rec)
    ds0 = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    full = D.collate([ds0.item(ci, t, 0, 0) for ci, t, _, _ in chunk])
    with torch.no_grad():
        out = model.forward_infer({k: full[k] for k in T1.MODEL_KEYS})
    z = {k: np.asarray(v) for k, v in rec.rows.items()}
    t0 = np.where(z["tick"] == 0)[0]
    assert len(t0) == len(chunk) and sorted(set(z["tick"].tolist())) == [0, 1, 2] and len(z["tick"]) == 3 * len(chunk)
    assert np.array_equal(z["tau"][t0], out["traj30_o"].double().numpy()) and np.array_equal(z["P"][t0], out["p10"].double().numpy())
    assert np.array_equal(z["plan"][t0], out["traj30"].double().numpy()) and np.array_equal(z["v0"][t0], full["p10f"][:, -1, 0].double().numpy())
    assert not np.array_equal(z["tau"][t0], z["plan"][t0])                    # the post-map is on: tau and plan are different columns
    assert np.array_equal(z["gt"][t0].astype(np.float32), full["gt"].numpy()) and z["gt_ok"][t0].all()
    assert not z["fired"].any()                                               # the tip model has no HACC-LAT gate: the carried column reads False everywhere
    assert np.array_equal(plan0, out["traj30"].numpy())


def test_carried_gt_follows_the_executed_frame(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    chunk = [w for w in wins if w[0] == 0][:3]
    rec, mp, states, ds, proxy, *_ = rolled(world, (model, cfg), chunk, heads=False)
    logged = {ci: tracks[str(J["clip"][ci])] for ci in {c[0] for c in chunk}}
    T1.rollout(mp, states, ds, proxy, m=10, device="cpu", hook=rec, executor="logged", logged_tracks=logged)
    z = {k: np.asarray(v) for k, v in rec.rows.items()}
    d = store.get(str(J["clip"][0]))
    for i in range(len(z["tick"])):                                           # the logged executor: the executed frame IS the logged frame at every tick
        j = int(z["j"][i])
        if j + 30 <= len(d["pose"]) - 1:
            assert z["gt_ok"][i] and np.array_equal(z["gt"][i].astype(np.float32), d["gt"][j]), (i, j)
    rec2, mp2, states2, ds2, proxy2, *_ = rolled(world, (model, cfg), chunk, heads=False)
    T1.rollout(mp2, states2, ds2, proxy2, m=10, device="cpu", hook=rec2)
    z2 = {k: np.asarray(v) for k, v in rec2.rows.items()}
    for i in np.where(z2["tick"] > 0)[0]:                                     # the plan executor: the GT follows the EXECUTED pose
        if z2["gt_ok"][i]:
            g, ok = T1.gt_in_frame(d["pose"], z2["pose"][i], int(z2["j"][i]))
            assert ok and np.array_equal(g, z2["gt"][i])
            wrong, _ = T1.gt_in_frame(d["pose"], d["pose"][int(z2["j"][i])], int(z2["j"][i]))      # the mutant: the GT in the logged frame
            assert not np.allclose(wrong, z2["gt"][i], atol=1e-3)


def test_carried_gt_in_frame_is_nan_when_the_clip_ends_inside_the_plan():
    pg = np.cumsum(np.ones((40, 4)), 0)
    g, ok = T1.gt_in_frame(pg, pg[5], 5)
    assert ok and g.shape == (30, 2) and np.isfinite(g).all()
    g, ok = T1.gt_in_frame(pg, pg[10], 10)                                    # 10 + 30 > 39
    assert not ok and np.isnan(g).all()
    g, ok = T1.gt_in_frame(pg, pg[9], 9)                                      # 9 + 30 = 39 = T - 1: the last legal tick
    assert ok
