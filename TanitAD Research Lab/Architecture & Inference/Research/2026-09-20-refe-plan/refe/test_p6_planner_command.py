#!/usr/bin/env python3
"""P6 / D6 planner reachability: a command-mode checkpoint driven through the REAL navtest runner path (CLAUDE.md "built, tested and unreachable from its caller").

Before this test the hop "runner sets planner.driving_command" did not exist for a command-mode checkpoint: `eval/refe_navtest_seam.py` set it ONLY under
`--sanitize-goal` (Amendment 8), so `eval_checkpoint.py -> refe_navtest_seam.py -> REFePlanner.infer` crashed at the first token of a P6 checkpoint
(`planner._command_for` RuntimeError). `eval/refe_navtest_seam.set_driving_command` is that hop now; this file proves it, end to end, on REAL navtest scenarios.

What runs (CPU, ViT-S, 4 cameras, RANDOM-INIT weights -- a wiring test, not a driving test; no claim about quality follows from it):
  * a tiny command-mode checkpoint and a tiny goal-mode checkpoint are written the way train.py writes them (meta["nav_mode"] only when not "goal");
  * `REFePlanner(checkpoint=...)` loads them through the normal path (ckpt_io.config_for_checkpoint -> load_for_inference);
  * 3 real navtest tokens of one log, one per NAVSIM command (LEFT / STRAIGHT / RIGHT), real camera frames from the dev-box frame zips, real nuPlan scenarios;
  * IN-PROCESS: seam.set_driving_command -> planner.driving_command -> _command_for -> the model's forward input (captured by a forward pre-hook) -> trajectories;
  * SUBPROCESS: `refe_navtest_seam.py main()` itself on the same tokens (the runner as the eval invokes it), then its npz is compared with the in-process poses.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_p6_planner_command.py --fast    # checks only
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_p6_planner_command.py           # checks, then every mutation (each must go RED)
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_p6_planner_command.py --check NAME
Needs nuPlan (the driverl-eval venv; the trainer venv has no nuplan), the navtest frame zips, the nuPlan test DBs and the W3 export on the dev box. Nothing skips: if any
is missing the test FAILS, and it must never be read as a pass. Prints `ZZP6PLAN_PASS <n>` only when every check (and, without --fast, every mutation) held.
"""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # CPU: deterministic, and the dev-box GPU stays free for the refcv7 chain
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
HERE = Path(__file__).resolve().parent             # <pkg>/refe
PKG = HERE.parent
SEAM = PKG / "eval" / "refe_navtest_seam.py"
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

DATA = "D:/Projects/TanitAD/data"
FRAMES = f"{DATA}/refe_navtest/frames"
TEST_DB_DIR = f"{DATA}/nuplan/nuplan-v1.1/splits/test"
DZ = "C:/Users/Admin/dz/DriveZero/DriveRL"
LOG = "2021.05.25.14.16.10_veh-35_01100_01664"
# one real navtest token per NAVSIM command (indices into the driving_command one-hot: 0 LEFT, 1 STRAIGHT, 2 RIGHT); asserted against the export below
TOKENS = {"LEFT": "025b657634505df3", "STRAIGHT": "054743e97abd5b9c", "RIGHT": "026e7b1e0e335625"}
CMD_INDEX = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}
POSE_TOL_M = 1e-4          # subprocess vs in-process: same CPU, same thread count; float32 npz
FLIP_MIN_M = 1e-3          # a command flip must move the executed pose by more than this (MEASURED on this rig: ~0.04-0.2 m)
N = 0


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def raises(exc, fn, needle=None, what=""):
    try:
        fn()
    except exc as e:
        ok(needle is None or needle in str(e), f"{what}: wrong message: {e}")
        return
    ok(False, f"{what}: did not raise {exc.__name__}")


def onehot(i):
    v = [0.0] * 4
    v[i] = 1.0
    return v


# ------------------------------------------------------------------------------------------------ the session (built once per process)
_S: dict = {}


def seam_env() -> dict:
    """the DriveRL/nuPlan environment, identical to eval/eval_checkpoint.env_driverl() (the eval's own launcher) + CPU"""
    e = dict(os.environ)
    e.update({"DZ_ROOT": DZ, "NUPLAN_DATA_ROOT": f"{DATA}/nuplan", "NUPLAN_MAPS_ROOT": f"{DATA}/nuplan-maps/nuplan-maps-v1.0",
              "DRIVERL_EVAL_ROOT": DZ, "DRIVERL_EVAL_NUPLAN_ROOT": f"{DZ}/nuplan-devkit",
              "DRIVERL_EVAL_CONFIG_PATH": f"{DZ}/release/configs/driverl_teacher.yaml",
              "DRIVERL_EVAL_CHECKPOINT_PATH": f"{DZ}/release/checkpoints/checkpoint_2400.pt",
              "PYTHONPATH": os.pathsep.join([f"{DZ}/nuplan-devkit", f"{DZ}/src"]),
              "REFE_BACKBONE_ROOT": f"{DATA}/backbones", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "4",
              "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""})
    return e


def session() -> dict:
    if _S:
        return _S
    try:
        import nuplan  # noqa: F401
    except ImportError as ex:                      # FAIL, never skip: a skipped reachability test reads like a pass
        raise AssertionError(f"nuPlan is not importable in {sys.executable} -- run this test with the driverl-eval venv ({ex})")
    os.environ.update({k: v for k, v in seam_env().items() if k.startswith(("NUPLAN_", "DZ_", "DRIVERL_", "REFE_"))})
    for p in (f"{DZ}/nuplan-devkit", f"{DZ}/src"):
        if p not in sys.path:
            sys.path.append(p)
    import ckpt_io
    import navtrain_scenarios as NS
    import nav_command as NC
    import planner as PL
    from model import REFe, REFeConfig
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    spec = importlib.util.spec_from_file_location("seam_under_test", str(SEAM))
    seam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seam)

    td = Path(tempfile.mkdtemp(prefix="p6plan_"))
    ck = {}
    for mode in ("command", "goal"):                               # written the way train.py writes them (train.py:1376-1385)
        cfg = REFeConfig.for_backbone("vits16", nav_mode=mode)
        torch.manual_seed(0)
        m = REFe(cfg)
        meta = {"backbone": "vits16", "weights_dir": "none", "frozen_sha256": None, "per_sample_calib": False,
                "n_cameras": cfg.n_cameras, "cameras": list(cfg.cameras)}
        if cfg.nav_mode != "goal":
            meta["nav_mode"] = cfg.nav_mode
        p = td / f"tiny_{mode}.pt"
        ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": ckpt_io.full_state(m), "meta": meta}, p)
        ck[mode] = str(p)
    exp_all = json.load(gzip.open(seam.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    exp = {t: exp_all[t] for t in TOKENS.values()}
    del exp_all
    for name, t in TOKENS.items():                                 # the token -> command pairing, from the export itself (not from nav_command)
        ok(exp[t]["log_name"] == LOG, f"{name} token {t} is not in {LOG}")
        ok(int(np.argmax(exp[t]["ego_statuses"][-1]["driving_command"])) == CMD_INDEX[name], f"{name}: the export says otherwise")
    scen = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(f"{TEST_DB_DIR}/{LOG}.db", list(TOKENS.values()),
                                                                              history_rows=1, future_rows=80)}
    ok(set(scen) == set(TOKENS.values()), "the scenario builder skipped a token")
    _S.update(td=td, ck=ck, exp=exp, scen=scen, seam=seam, NC=NC, PL=PL, ckpt_io=ckpt_io, REFe=REFe, REFeConfig=REFeConfig, PI=PlannerInitialization, NS=NS)
    return _S


def make_planner(mode, **kw):
    S = session()
    return S["PL"].REFePlanner(checkpoint=S["ck"][mode], images_root=FRAMES, db_dir=TEST_DB_DIR, backbone="vits16", device="cpu", **kw)


def prep(planner, tok):
    """the runner's per-token steps (refe_navtest_seam.main): initialize -> ego -> frames. Returns (ego, img)."""
    S = session()
    sc = S["scen"][tok]
    planner._scenario = sc
    planner.initialize(S["PI"](route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
    ego = sc.get_ego_state_at_iteration(0)
    img = planner._image_for(ego)
    ok(img is not None and tuple(img.shape[:2]) == (1, 4), f"no frames for {tok}: {planner.frames.miss_reason}")
    return ego, img


def poses(planner, traj, k):
    return session()["seam"].to_navsim(planner.executed(traj, k).float().cpu().numpy())


def capture_forward_goal(planner):
    """forward pre-hook: what the MODEL actually receives in the `goal` slot (positional arg 2)"""
    cap: list = []
    h = planner.model.register_forward_pre_hook(lambda _m, args: cap.append(args[2].detach().clone()))
    return cap, h


_CMD_RUN: dict = {}


def command_runs() -> dict:
    """In-process command-mode runs, cached: {(name, command_name): (traj, k, poses)} plus the NAVSIM-command run through the runner's helper."""
    if _CMD_RUN:
        return _CMD_RUN
    S = session()
    pl = make_planner("command")
    ok(pl.model.cfg.nav_mode == "command")
    runs, via_runner, cap_rows = {}, {}, {}
    for name, tok in TOKENS.items():
        ego, img = prep(pl, tok)
        for cn, ci in CMD_INDEX.items():
            pl.driving_command = onehot(ci)
            traj, _s, k = pl.infer(ego, img)
            runs[(name, cn)] = (traj.cpu().numpy().copy(), k, poses(pl, traj, k))
        # the real hop: the runner's helper sets the command from the export, the model's forward input is captured
        pl.driving_command = None
        cap, h = capture_forward_goal(pl)
        did = S["seam"].set_driving_command(pl, tok, S["exp"][tok])
        ok(did is True, "the runner helper did not set a command for a command-mode checkpoint")
        ok(pl.driving_command is not None, "RUNNER DID NOT SET planner.driving_command")
        traj, _s, k = pl.infer(ego, img)
        h.remove()
        via_runner[name] = (traj.cpu().numpy().copy(), k, poses(pl, traj, k), pl.driving_command)
        cap_rows[name] = cap[0].cpu().numpy().copy()
        ok(len(cap) == 1)
    _CMD_RUN.update(planner=pl, runs=runs, via_runner=via_runner, captured=cap_rows)
    return _CMD_RUN


_SEAM_RUN: dict = {}


def seam_run(mode) -> dict:
    """`refe_navtest_seam.py` as a subprocess on the 3 tokens (the eval's runner path), CPU. Cached per mode."""
    if mode in _SEAM_RUN:
        return _SEAM_RUN[mode]
    S = session()
    td = S["td"]
    tf = td / "tokens3.json"
    tf.write_text(json.dumps(list(TOKENS.values())), encoding="utf-8")
    out, rin = td / f"seam_{mode}.npz", td / f"seam_{mode}.inputs.json"
    argv = [sys.executable, str(SEAM), "--ckpt", S["ck"][mode], "--frames", FRAMES, "--tokens", str(tf), "--out", str(out), "--device", "cpu",
            "--backbone", "vits16", "--record-inputs", str(rin)]
    r = subprocess.run(argv, cwd=str(SEAM.parent), env=seam_env(), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    res = {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    if r.returncode == 0:
        z = np.load(out)
        res.update(npz={k: z[k] for k in z.files}, report=json.load(open(str(out.with_suffix("")) + ".report.json", encoding="utf-8")),
                   inputs=json.load(open(rin, encoding="utf-8")))
    _SEAM_RUN[mode] = res
    return res


# ------------------------------------------------------------------------------------------------ the checks
def chk_built_in_command_mode():
    """(a) the planner builds the checkpoint's own nav mode through the normal path: cmd_embed present, goal_freqs absent -- and the reverse for a goal checkpoint."""
    S = session()
    ckpt_io, REFeConfig = S["ckpt_io"], S["REFeConfig"]
    base = REFeConfig.for_backbone("vits16")
    pc = ckpt_io.config_for_checkpoint(base, S["ck"]["command"])
    ok(pc.nav_mode == "command" and base.nav_mode == "goal", "config_for_checkpoint must return a COPY with the checkpoint's nav_mode")
    ok(ckpt_io.config_for_checkpoint(base, S["ck"]["goal"]).nav_mode == "goal")
    pl = make_planner("command")
    ok(pl.model.cfg.nav_mode == "command" and hasattr(pl.model, "cmd_embed") and not hasattr(pl.model, "goal_freqs"),
       "a command-mode checkpoint must build cmd_embed and NOT the goal encoder")
    ok("cmd_embed.weight" in pl.model.state_dict() and "goal_freqs" not in pl.model.state_dict())
    ok(pl.trained and pl.ckpt_format == S["ckpt_io"].FORMAT_FULL)
    pg = make_planner("goal")
    ok(pg.model.cfg.nav_mode == "goal" and hasattr(pg.model, "goal_freqs") and not hasattr(pg.model, "cmd_embed"))


def chk_runner_hop():
    """(b) export -> seam.set_driving_command -> planner.driving_command -> _command_for -> the model's forward input, on 3 real tokens, one per command."""
    S = session()
    NC = S["NC"]
    R = command_runs()
    for name, tok in TOKENS.items():
        navsim = S["exp"][tok]["ego_statuses"][-1]["driving_command"]
        ok(R["via_runner"][name][3] is navsim or list(R["via_runner"][name][3]) == list(navsim), f"{name}: planner.driving_command is not NAVSIM's")
        want = np.asarray([onehot(int(np.argmax(navsim)))], dtype=np.float32)                      # derived from the export alone
        ok(np.array_equal(R["captured"][name], want), f"{name}: the model received {R['captured'][name]} but NAVSIM said {navsim}")
        ok(R["captured"][name].shape == (1, 4) and NC.onehot(NC.from_driving_command(navsim)) == want[0].tolist())
        ok(int(np.argmax(R["captured"][name])) == CMD_INDEX[name], f"{name}: wrong command reached the model")
        # the runner path's trajectory IS the trajectory of NAVSIM's command (bitwise, same process) -- and not of any other
        t_run = R["via_runner"][name][0]
        ok(np.array_equal(t_run, R["runs"][(name, name)][0]), f"{name}: runner-path trajectory != direct NAVSIM-command trajectory")
        for other in CMD_INDEX:
            if other != name:
                ok(float(np.abs(t_run - R["runs"][(name, other)][0]).max()) > FLIP_MIN_M, f"{name}: runner trajectory indistinguishable from {other}'s")
    ok(len({int(np.argmax(R["captured"][n])) for n in TOKENS}) == 3, "the three tokens must carry three DIFFERENT commands")


def chk_flip_changes_output():
    """(c) flipping the command changes the output trajectory; ZERO CONTROL: the same command, however it is spelled, changes it by exactly 0.0."""
    R = command_runs()
    pl = R["planner"]
    for name, tok in TOKENS.items():
        names = list(CMD_INDEX)
        for i in range(3):
            for j in range(i + 1, 3):
                d = float(np.abs(R["runs"][(name, names[i])][0] - R["runs"][(name, names[j])][0]).max())
                ok(d > FLIP_MIN_M, f"{name}: {names[i]} vs {names[j]} moved the proposals by only {d:.2e} m")
        ego, img = prep(pl, tok)
        ci = CMD_INDEX[name]
        base = None
        for spelled in (onehot(ci), [int(x) for x in onehot(ci)], np.asarray(onehot(ci), dtype=np.int64), np.asarray(onehot(ci), dtype=np.float32)):
            pl.driving_command = spelled                                                      # NAVSIM hands an int64 array; the export a list
            t = pl.infer(ego, img)[0].cpu().numpy()
            base = t if base is None else base
            ok(float(np.abs(t - base).max()) == 0.0, f"{name}: the same command spelled differently changed the output")
        ok(np.array_equal(base, R["runs"][(name, name)][0]), "a repeat of the same command must be BITWISE identical (CPU)")
        # an all-zero vector is UNKNOWN (planner rule): a different command, so it must also move the output
        pl.driving_command = [0, 0, 0, 0]
        unk = pl.infer(ego, img)[0].cpu().numpy()
        pl.driving_command = onehot(3)
        ok(np.array_equal(unk, pl.infer(ego, img)[0].cpu().numpy()), "all-zero must equal the UNKNOWN one-hot")
        ok(float(np.abs(unk - base).max()) > FLIP_MIN_M)


def chk_missing_raises():
    """(d) a missing / None / malformed command REFUSES, at the planner and at the runner helper; nothing defaults."""
    S = session()
    seam = S["seam"]
    pl = make_planner("command")
    tok = TOKENS["STRAIGHT"]
    ego, img = prep(pl, tok)
    ok(pl.driving_command is None, "a fresh planner must hold no command")
    raises(RuntimeError, lambda: pl.infer(ego, img), "driving_command", "fresh planner, never given a command")
    pl.driving_command = onehot(1)
    pl.infer(ego, img)
    pl.driving_command = None
    raises(RuntimeError, lambda: pl.infer(ego, img), "driving_command", "command reset to None")
    rec = S["exp"][tok]
    raises(RuntimeError, lambda: seam.set_driving_command(pl, tok, {"ego_statuses": []}), "driving_command", "runner: no ego_statuses")
    raises(RuntimeError, lambda: seam.set_driving_command(pl, tok, {"ego_statuses": [{}]}), "driving_command", "runner: no driving_command key")
    raises(RuntimeError, lambda: seam.set_driving_command(pl, tok, {"ego_statuses": [{"driving_command": None}]}), "driving_command", "runner: None")
    raises(RuntimeError, lambda: seam.set_driving_command(pl, tok, {"ego_statuses": [{"driving_command": [1, 0, 0]}]}), "4-vector", "runner: wrong length")
    raises(RuntimeError, lambda: seam.set_driving_command(pl, tok, {"ego_statuses": [{"driving_command": [float("nan"), 0, 0, 0]}]}), "4-vector", "runner: nan")
    ok(pl.driving_command is None, "a refused command must leave the planner's command untouched")
    # the Amendment-8 path (goal mode + sanitisation) is NOT a command-mode path: there a missing command used to KeyError and must still be loud
    ps = make_planner("goal", sanitize_goal=True, goal_fix="off")
    raises(RuntimeError, lambda: seam.set_driving_command(ps, tok, {"ego_statuses": [{}]}), "driving_command", "Amendment 8 path: no driving_command")
    ok(ps.driving_command is None)
    ok(seam.set_driving_command(pl, tok, rec) is True and list(pl.driving_command) == list(rec["ego_statuses"][-1]["driving_command"]))


def chk_seam_subprocess():
    """(b, runner as the eval invokes it) `refe_navtest_seam.py` itself, 3 tokens, command checkpoint: exits 0 with ZZSEAM_OK, records the NAVSIM command it fed, and its
    poses equal the in-process poses of NAVSIM's command and differ from every other command's."""
    S = session()
    res = seam_run("command")                      # FIRST: the runner on its own, so a broken runner hop is caught here and not only by the in-process runs
    ok(res["rc"] == 0, f"seam exited {res['rc']} on a command-mode checkpoint:\n{res['stderr'][-1500:]}")
    ok(re.search(r"ZZSEAM_OK 3 ", res["stdout"]) is not None, res["stdout"][-600:])
    rep = res["report"]
    ok(rep["nav_mode"] == "command" and rep["driving_command_set"] == 3 and rep["rows"] == 3 and rep["misses"] == 0, rep)
    R = command_runs()
    npz = res["npz"]
    ok({str(t) for t in npz["token"]} == set(TOKENS.values()))
    for name, tok in TOKENS.items():
        i = [str(t) for t in npz["token"]].index(tok)
        navsim = S["exp"][tok]["ego_statuses"][-1]["driving_command"]
        ok(res["inputs"][tok]["goal"] == onehot(int(np.argmax(navsim))), f"{name}: the seam fed {res['inputs'][tok]['goal']} for NAVSIM's {navsim}")
        got = npz["poses"][i].astype(np.float64)
        d_own = float(np.abs(got - R["runs"][(name, name)][2]).max())
        ok(d_own < POSE_TOL_M, f"{name}: seam poses differ from the in-process NAVSIM-command poses by {d_own:.2e}")
        for other in CMD_INDEX:
            if other != name:
                d_o = float(np.abs(got - R["runs"][(name, other)][2]).max())
                ok(d_o > 10 * POSE_TOL_M, f"{name}: seam poses are indistinguishable from the {other}-command poses ({d_o:.2e})")


def chk_goal_mode_unchanged():
    """(e) a goal-mode checkpoint takes the OLD path: the runner helper is a no-op (or Amendment 8's exact old assignment), no command is read, and the seam's poses are the
    in-process goal-mode poses whatever driving_command holds. (The whole-seam pre/post-edit bit-identity was measured once -- eval/P6_PLANNER_COMMAND.md.)"""
    S = session()
    seam = S["seam"]
    pg = make_planner("goal")
    ok(pg.sanitize_goal is False and pg.goal_fix == "pdm_route")
    in_proc = {}
    for name, tok in TOKENS.items():
        ego, img = prep(pg, tok)
        pg.driving_command = None
        ok(seam.set_driving_command(pg, tok, S["exp"][tok]) is False and pg.driving_command is None, "goal mode (no Amendment 8) must not read or set a command")
        outs = []
        for poison in (None, onehot(0), onehot(2), [9, 9, 9, 9]):
            pg.driving_command = poison
            prep(pg, tok)                                                                    # initialize resets the route state; the goal is recomputed
            traj, _s, k = pg.infer(ego, img)
            outs.append((traj.cpu().numpy().copy(), k, poses(pg, traj, k)))
            ok(tuple(pg.last_inputs[1].shape) == (1, 4))
        for o in outs[1:]:
            ok(float(np.abs(o[0] - outs[0][0]).max()) == 0.0 and o[1] == outs[0][1], f"{name}: goal mode read driving_command")
        in_proc[name] = outs[0][2]
    pg.driving_command = None
    # Amendment 8's assignment is preserved verbatim: the SAME object, goal_diag reset
    ps = make_planner("goal", sanitize_goal=True, goal_fix="off")
    rec = S["exp"][TOKENS["LEFT"]]
    ps.goal_diag = {"stale": 1}
    ok(seam.set_driving_command(ps, TOKENS["LEFT"], rec) is True and ps.driving_command is rec["ego_statuses"][-1]["driving_command"] and ps.goal_diag is None)
    res = seam_run("goal")
    ok(res["rc"] == 0 and re.search(r"ZZSEAM_OK 3 ", res["stdout"]) is not None, res["stderr"][-1500:])
    ok(res["report"]["nav_mode"] == "goal" and res["report"]["driving_command_set"] == 0, res["report"])
    for name, tok in TOKENS.items():
        i = [str(t) for t in res["npz"]["token"]].index(tok)
        d = float(np.abs(res["npz"]["poses"][i].astype(np.float64) - in_proc[name]).max())
        ok(d < POSE_TOL_M, f"{name}: goal-mode seam poses differ from the in-process goal-mode poses by {d:.2e}")


CHECKS = {"built_in_command_mode": chk_built_in_command_mode, "runner_hop": chk_runner_hop, "flip_changes_output": chk_flip_changes_output,
          "missing_raises": chk_missing_raises, "seam_subprocess": chk_seam_subprocess, "goal_mode_unchanged": chk_goal_mode_unchanged}

# (file relative to the package, old, new, the check that must go RED). Each neutralises ONE emitting line.
MUTATIONS = [
    ("eval/refe_navtest_seam.py", "    planner.driving_command = cmd\n    planner.goal_diag = None\n", "    planner.goal_diag = None\n", "runner_hop"),
    ("eval/refe_navtest_seam.py", "    planner.driving_command = cmd\n    planner.goal_diag = None\n", "    planner.goal_diag = None\n", "seam_subprocess"),
    ("refe/planner.py", "return torch.tensor([NC.onehot(NC.from_driving_command(cmd))], dtype=torch.float32)", "return torch.tensor([NC.onehot(1)], dtype=torch.float32)",
     "runner_hop"),
    ("refe/planner.py", "return torch.tensor([NC.onehot(NC.from_driving_command(cmd))], dtype=torch.float32)", "return torch.tensor([NC.onehot(1)], dtype=torch.float32)",
     "flip_changes_output"),
    ("refe/ckpt_io.py", "    if nav and nav != c.nav_mode:\n        c.nav_mode = nav\n", "", "built_in_command_mode"),
    ("refe/planner.py", "        if cmd is None:\n            raise RuntimeError(\"this checkpoint is nav_mode='command'", "        if False:\n            raise RuntimeError(\"this checkpoint is nav_mode='command'",
     "missing_raises"),
    ("eval/refe_navtest_seam.py", "    if cmd is None:                                    # Amendment 8 used to KeyError here; keep it loud\n",
     "    if False:                                          # Amendment 8 used to KeyError here; keep it loud\n", "missing_raises"),
    ("eval/refe_navtest_seam.py", "        if v.size != NC.N_COMMANDS or not np.isfinite(v).all():\n", "        if False:\n", "missing_raises"),
]


def run_mutations() -> None:
    global N
    for rel, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="p6planmut_"))
        (td / "refe").mkdir()
        (td / "eval").mkdir()
        for f in HERE.glob("*.py"):
            shutil.copyfile(f, td / "refe" / f.name)
        shutil.copyfile(SEAM, td / "eval" / SEAM.name)
        src = (td / rel).read_text(encoding="utf-8")
        ok(src.count(old) == 1, f"mutation target not found exactly once in {rel}: {old[:70]!r}")
        (td / rel).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(td / "refe" / "test_p6_planner_command.py"), "--check", check], cwd=str(td / "refe"),
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"), capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok(r.returncode != 0, f"MUTATION SURVIVED: {rel}: {old[:60]!r} -> check {check} still passed")
        tail = [l for l in (r.stderr or r.stdout).strip().splitlines() if l.strip()][-1:] or [""]
        print(f"  mutation RED as required: {rel} [{old.strip()[:46]}] -> {check}\n      {tail[0][:200]}", flush=True)
        shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    try:
        if "--check" in sys.argv:
            name = sys.argv[sys.argv.index("--check") + 1]
            CHECKS[name]()
            print(f"CHECK_OK {name} {N}")
            return 0
        for name, fn in CHECKS.items():
            fn()
            print(f"  {name}: ok ({N})", flush=True)
        if "--fast" not in sys.argv:
            run_mutations()
        print(f"ZZP6PLAN_PASS {N}")
        return 0
    finally:
        if _S.get("td"):
            shutil.rmtree(_S["td"], ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
