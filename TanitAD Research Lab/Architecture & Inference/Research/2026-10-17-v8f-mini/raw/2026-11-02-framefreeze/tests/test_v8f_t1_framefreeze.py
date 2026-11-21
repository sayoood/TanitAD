"""FRAMEFREEZE (the frame-freeze control, DRAFT_RECOVERY_CARD_v2.md section 8.3): ``--scene hold_now`` of ``stack/scripts/v8f_t1_eval_framefreeze.py`` (the HISTSWAP-LF tool 6c66f0a5 + ``code/apply_fz_patch.py``).

Run (from a tree that holds stack/ and taniteval/, the tool in stack/scripts/ beside the LF tool, this file in stack/tests/):
  CUDA_VISIBLE_DEVICES=-1  pytest -q stack/tests/test_v8f_t1_framefreeze.py        (``V8F_T1_LF_TOOL=<the LF tool>`` overrides the default stack/scripts/v8f_t1_eval_histswap_lf.py; both are sha256-pinned; absent => FAIL, never skip)
  * KFZ1     under ``--scene hold_now`` every batch the network is handed has ``feats[:, k] == feats[:, -1]`` for ALL k, max abs deviation exactly 0.0 (also composed with ``--histswap cv_logged``); non-vacuous: under
             ``logged`` the rows differ; the frozen last row is the logged tick-j row, bit for bit; ``frame_ok`` is unchanged; shape / dtype unchanged. The run header carries the same receipt over every item.
  * KFZ2     ``--scene logged`` and the flag absent give dumps / ticks / header byte-identical to the SOURCE (LF) tool, ``--scene frozen`` and ``--histswap cv_logged`` too; and three mutations turn KFZ1 RED:
             the apply line deleted, the apply line freezing to the FIRST row, the apply line leaving one row unfrozen.
  * REACH    the freeze reaches the network (tick-0 plan differs from the logged-frame plan on the tiny model), the plan is still committed to the executed state, ``frame_ok`` is identical.
  * HEADER   ``camera_motion: false`` and the ``frame_freeze`` block only under hold_now (the card ref is recorded; a card ref without hold_now is refused); ``strict.camera`` stays False; the ticks file carries ``scene``.
"""

import dataclasses
import hashlib
import json
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
import v8f_t1_eval_framefreeze as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
LF_TOOL_SHA256 = "6c66f0a527b3b08c640550dd4d065c45f43186d1af5220830f38bd933245719f"
M_STEPS = 15
APPLY = '        it["feats"] = hold_now_feats(it)\n'


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_framefreeze")
    w = D.make_synthetic_world(root, n_train=2, n_eval=3, T=90, seed=5)
    return {"root": root, **w}


def tiny_model(J, seed=0):
    cfg = M.V8FMiniConfig(**TINY)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    prior = M.P10Prior.synthetic(0)
    with torch.no_grad():
        for q in prior.parameters():
            q.mul_(0.02)
    model = M.V8FMini(cfg, prior, mu, sd).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.head_o[2].weight, std=0.3)               # a rough O head: the plan wanders, the prior does not
    return model, cfg


def tool_from_source(src: str, name: str):
    mod = types.ModuleType(name)
    mod.__file__ = str(Path(T1.__file__).resolve())
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


def mutated(old: str, new: str):
    src = Path(T1.__file__).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert src.count(old) == 1, (old, src.count(old))
    return tool_from_source(src.replace(old, new), "v8f_t1_eval_framefreeze_mut")


def load_lf_tool():
    p = os.environ.get("V8F_T1_LF_TOOL", "") or str(Path(T1.__file__).resolve().parent / "v8f_t1_eval_histswap_lf.py")
    assert Path(p).is_file(), f"the SOURCE (HISTSWAP-LF) tool is not at {p}: KFZ2 cannot run (failing, not skipping)"
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == LF_TOOL_SHA256, "REFUSED: that file is not the HISTSWAP-LF tool (sha256 pin)"
    return tool_from_source(src.decode("utf-8"), "v8f_t1_eval_histswap_lf_src")


def run_cli(world, tmp, label, *extra, tool=None, model=None, tag="syn", m=M_STEPS):
    root = Path(world["root"])
    if model is not None:
        torch.save({"cfg": dataclasses.asdict(model[1]), "model": model[0].state_dict(), "grid": "pool4x10"}, tmp / "ck.pt")
    t = tool or T1
    t.CAUSAL_STATS.clear()
    rc = t.main(["run", "--ckpt", str(tmp / "ck.pt"), "--tag", tag, "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root),
                 "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(tmp / label),
                 "--replan-steps", str(m), "--batch", "6", "--max-windows", "12", "--n-boot", "20", "--device", "cpu",
                 "--dump-ticks", str(tmp / label / "ticks"), *extra])
    assert rc == 0
    return tmp / label


def dumps(out, key=None, tag="syn"):
    fs = sorted((out / f"dump_{tag}").iterdir())
    if key is not None:
        return np.concatenate([np.load(f)[key] for f in fs])
    return {f.name: {k: np.load(f)[k] for k in np.load(f).files} for f in fs}


def ticks(out, tag="syn"):
    z = np.load(out / "ticks" / f"ticks_{tag}.npz")
    return {k: z[k] for k in z.files}


def header(out, tag="syn"):
    return json.load(open(out / f"t1v8f_{tag}.json"))["t1v8f"]


def assert_same_run(a, b):
    """Byte-identical dumps and ticks (same keys, same dtypes) and the same header apart from the ticks path; neither carries a freeze block."""
    da, db = dumps(a), dumps(b)
    assert da.keys() == db.keys() and len(da) >= 3
    for f in da:
        assert set(da[f]) == set(db[f]) and all(da[f][k].tobytes() == db[f][k].tobytes() and da[f][k].dtype == db[f][k].dtype for k in da[f]), f
    ta, tb = ticks(a), ticks(b)
    assert ta.keys() == tb.keys() and all(ta[k].tobytes() == tb[k].tobytes() for k in ta)
    ja, jb = header(a), header(b)
    assert "frame_freeze" not in ja and "frame_freeze" not in jb and "camera_motion" not in ja and "camera_motion" not in jb
    assert {k: v for k, v in ja.items() if k != "ticks_file"} == {k: v for k, v in jb.items() if k != "ticks_file"}


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    J = D.load_join(Path(world["root"]) / "join_eval.npz")
    model = tiny_model(J)
    base = ("--smooth", "sg72", "--limit", "causal", "--dump-heads")
    out = {"tmp": tmp, "base": base, "y": run_cli(world, tmp, "y", *base, model=model)}      # the flag absent
    out["logged"] = run_cli(world, tmp, "logged", *base, "--scene", "logged")
    out["frozen"] = run_cli(world, tmp, "frozen", *base, "--scene", "frozen")
    out["hold"] = run_cli(world, tmp, "hold", *base, "--scene", "hold_now", "--freeze-card-ref", "RECOVER-FT test sha line")
    out["hold_stats"] = dict(T1.FREEZE_STATS)
    out["hold_cv"] = run_cli(world, tmp, "hold_cv", *base, "--scene", "hold_now", "--histswap", "cv_logged")
    out["cv_logged"] = run_cli(world, tmp, "cv_logged", *base, "--histswap", "cv_logged")
    return out


# ------------------------------------------------------------------------------------------------ direct rollout (the tool under test passed in)
def capture(tool, world, tmp, scene, hist_swap=False, limit="causal", m=M_STEPS):
    root = Path(world["root"])
    ck = torch.load(tmp / "ck.pt", map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks = D.load_native_tracks(str(root / "native_tracks.pt"))
    J = D.load_join(str(root / "join_eval.npz"))
    store = D.ClipStore(str(root), "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(str(root / "eval_grid_windows.json"), J)[:12]
    proxy = tool._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    states = tool.make_states(store, tracks, J, wins, "deadreckon", hist_swap=hist_swap)
    cap = []

    def hook(n, batch, plan, sts):
        for b, st in enumerate(sts):
            cap.append({"n": n, "j": st.j, "t0": st.t0, "clip": str(J["clip"][st.ci]), "plan": np.array(plan[b], np.float64),
                        "pose_exec": np.array(st.d["pose"][st.j], np.float64),
                        "batch": {k: batch[k][b].detach().cpu().numpy().copy() for k in ("feats", "frame_ok", "p10f")}})

    tool.CAUSAL_STATS.clear()
    planner = tool.model_planner(model, torch.device("cpu"))
    tool.rollout(planner, states, ds, proxy, m=m, schedule="periodic", scene=scene, limit=limit, device=torch.device("cpu"), hook=hook)
    cl = [st.executed_ego() for st in states]
    return {"cap": cap, "cl": np.asarray(cl), "J": J, "ds": ds, "states": states}


def rows_dev(feats) -> float:
    """max abs deviation of every history row from the LAST row (exactly 0.0 iff every row equals the current frame's)."""
    f = np.asarray(feats, np.float64)
    return float(np.abs(f - f[-1:]).max())


def check_kfz1(c, logged):
    """KFZ1 on a capture: at every tick of every window every row of the batch's feats equals the LAST row, max abs deviation exactly 0.0, AND that row is the logged tick-j frame (``logged`` = the same windows
    captured under ``--scene logged``: a freeze to a stale row would pass the first half alone); raises AssertionError otherwise."""
    assert len(c["cap"]) >= 24 and len(c["cap"]) == len(logged["cap"])
    for r, lg in zip(c["cap"], logged["cap"]):
        assert (r["n"], r["clip"], r["t0"]) == (lg["n"], lg["clip"], lg["t0"])
        f = r["batch"]["feats"]
        assert f.shape[0] == c["ds"].hmax, f.shape
        assert rows_dev(f) == 0.0, (r["n"], r["clip"], rows_dev(f))
        assert np.array_equal(f, np.repeat(f[-1:], f.shape[0], axis=0))
        assert np.array_equal(f[-1], lg["batch"]["feats"][-1]), "the frozen row is not the logged tick-j frame"


@pytest.fixture(scope="module")
def caps(world, runs):
    return {"logged": capture(T1, world, runs["tmp"], "logged"), "hold": capture(T1, world, runs["tmp"], "hold_now"),
            "hold_cv": capture(T1, world, runs["tmp"], "hold_now", hist_swap="cv_logged"), "frozen": capture(T1, world, runs["tmp"], "frozen")}


# ------------------------------------------------------------------------------------------------ flag, refusals
def test_flag_parser_default_and_refusals(world, tmp_path):
    base = ["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x"]
    assert T1.build_parser().parse_args(base).scene == "logged"
    assert T1.SCENE_MODES == ("logged", "frozen", "hold_now")
    for m in T1.SCENE_MODES:
        assert T1.build_parser().parse_args(base + ["--scene", m]).scene == m
    with pytest.raises(SystemExit):
        T1.build_parser().parse_args(base + ["--scene", "hold"])
    for sc in ("logged", "frozen"):
        with pytest.raises(SystemExit) as e:
            T1.main(base + ["--scene", sc, "--freeze-card-ref", "x"])
        assert "REFUSED" in str(e.value) and "hold_now" in str(e.value)
    with pytest.raises(RuntimeError) as e:
        T1.hold_now_feats({"frame_ok": True})
    assert "REFUSED" in str(e.value) and "feats" in str(e.value)


# ------------------------------------------------------------------------------------------------ KFZ1
def test_kfz1_every_batch_feats_rows_equal_the_last_row_exactly(caps, runs):
    check_kfz1(caps["hold"], caps["logged"])
    check_kfz1(caps["hold_cv"], caps["logged"])                                             # composed with the oracle history level (FZ-HLVL)
    n_items = 12 * 2 * 2                                                     # 12 windows x 2 ticks x (the model's rollout + the P10 floor's rollout: both build items through build_item)
    assert runs["hold_stats"]["items"] == n_items and runs["hold_stats"]["max_row_dev"] == 0.0 and runs["hold_stats"]["n_unequal_elements"] == 0
    h = header(runs["hold"])["frame_freeze"]
    assert h["items_frozen"] == n_items and h["max_row_dev"] == 0.0 and h["n_unequal_elements"] == 0
    assert header(runs["hold_cv"])["frame_freeze"]["items_frozen"] == n_items


def test_kfz1_is_not_vacuous_and_freezes_to_the_current_frame_not_another(caps):
    lg, hd = caps["logged"]["cap"], caps["hold"]["cap"]
    assert len(lg) == len(hd)
    assert max(rows_dev(r["batch"]["feats"]) for r in lg) > 0.0, "the logged frames must differ across rows (else KFZ1 would pass vacuously)"
    assert sum(rows_dev(r["batch"]["feats"]) > 0.0 for r in lg) >= len(lg) // 2
    for a, b in zip(lg, hd):
        assert (a["n"], a["clip"], a["t0"]) == (b["n"], b["clip"], b["t0"])
        fa, fb = a["batch"]["feats"], b["batch"]["feats"]
        assert fa.dtype == fb.dtype and fa.shape == fb.shape
        assert np.array_equal(fb[-1], fa[-1]), "the frozen row must be the logged tick-j row, bit for bit"
        assert np.array_equal(a["batch"]["frame_ok"], b["batch"]["frame_ok"]), "frame_ok must be unchanged"
    # the tick-0 'frozen' scene (stale tick-0 frames) is NOT the hold: it differs from hold_now at ticks >= 1
    fz = caps["frozen"]["cap"]
    assert any(not np.array_equal(a["batch"]["feats"], b["batch"]["feats"]) for a, b in zip(fz, hd) if a["n"] >= 1)


# ------------------------------------------------------------------------------------------------ KFZ2
def test_kfz2_logged_and_flag_absent_are_bit_identical_to_the_source_tool(world, runs):
    assert_same_run(runs["y"], runs["logged"])
    src = load_lf_tool()
    assert_same_run(runs["y"], run_cli(world, runs["tmp"], "src_y", *runs["base"], tool=src))
    assert_same_run(runs["frozen"], run_cli(world, runs["tmp"], "src_frozen", *runs["base"], "--scene", "frozen", tool=src))
    assert_same_run(runs["cv_logged"], run_cli(world, runs["tmp"], "src_cv", *runs["base"], "--histswap", "cv_logged", tool=src))


def test_kfz2_mutations_turn_kfz1_red(world, runs, caps):
    good = capture(T1, world, runs["tmp"], "hold_now")
    check_kfz1(good, caps["logged"])
    survived = []
    for old, new, why in ((APPLY, "        pass\n", "the apply line deleted"),
                          ('np.repeat(f[-1:], f.shape[0], axis=0)', 'np.repeat(f[:1], f.shape[0], axis=0)', "freezing to the FIRST row"),
                          ('np.repeat(f[-1:], f.shape[0], axis=0)', 'np.concatenate([f[:1], np.repeat(f[-1:], f.shape[0] - 1, axis=0)])', "one row left unfrozen")):
        mc = capture(mutated(old, new), world, runs["tmp"], "hold_now")
        try:
            check_kfz1(mc, caps["logged"])
            survived.append(why)
        except AssertionError:
            pass
    assert not survived, f"mutations that KFZ1 did not turn RED: {survived}"
    # the receipt in the header is a second, independent witness of the same defect
    mt = mutated(APPLY, "        pass\n")
    run_cli(world, runs["tmp"], "mut_hold", *runs["base"], "--scene", "hold_now", tool=mt)
    h = header(runs["tmp"] / "mut_hold")["frame_freeze"]
    assert h["max_row_dev"] > 0.0 and h["n_unequal_elements"] > 0


# ------------------------------------------------------------------------------------------------ REACH
def test_the_freeze_reaches_the_network_and_the_plan_stays_committed_to_the_executed_state(caps, runs):
    lg = {(r["n"], r["clip"], r["t0"]): r for r in caps["logged"]["cap"]}
    hd = {(r["n"], r["clip"], r["t0"]): r for r in caps["hold"]["cap"]}
    dev0 = max(float(np.abs(lg[k]["plan"] - hd[k]["plan"]).max()) for k in lg if k[0] == 0)
    assert dev0 > 1e-3, f"the freeze did not reach the network on the tiny model (tick-0 plan dev {dev0:.2e} m)"
    za, zb = ticks(runs["logged"]), ticks(runs["hold"])
    assert za["tick"].tobytes() == zb["tick"].tobytes() and za["win"].tobytes() == zb["win"].tobytes()
    assert np.array_equal(za["pose"][za["tick"] == 0], zb["pose"][zb["tick"] == 0])
    assert float(np.abs(za["plan_raw"][za["tick"] == 0] - zb["plan_raw"][zb["tick"] == 0]).max()) > 1e-3
    # the P10 floor reads no frames: identical under the freeze
    assert dumps(runs["logged"], "fl").tobytes() == dumps(runs["hold"], "fl").tobytes()
    # composed with the oracle history level: the history reached the network too (v0_in != v0 after tick 0) while the frames stay frozen
    zc = ticks(runs["hold_cv"])
    assert "v0_in" in zc and float(np.abs(zc["v0_in"] - zc["v0"])[zc["tick"] >= 1].max()) > 0.0
    assert str(zc["histswap"]) == "cv_logged" and str(zc["scene"]) == "hold_now"


# ------------------------------------------------------------------------------------------------ HEADER
def test_header_ticks_and_strictness_under_hold_now(runs):
    jh = header(runs["hold"])
    assert jh["scene"] == "hold_now" and jh["camera_motion"] is False
    assert jh["strict"]["camera"] is False
    assert any("CURRENT frame" in c and "no frame motion" in c for c in jh["open_channels"])
    assert jh["frame_freeze"]["card_ref"] == "RECOVER-FT test sha line" and jh["frame_freeze"]["scene"] == "hold_now"
    assert "UNVERIFIED" in header(runs["hold_cv"])["frame_freeze"]["card_ref"]
    assert str(ticks(runs["hold"])["scene"]) == "hold_now"
    for k in ("logged", "frozen", "y"):
        assert "camera_motion" not in header(runs[k]) and "scene" not in ticks(runs[k])
    assert header(runs["frozen"])["strict"]["camera"] is True and header(runs["logged"])["strict"]["camera"] is False
