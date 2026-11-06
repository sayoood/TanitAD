"""Stage-2 overlay (AGENT I): the hold executor gates, the cell order, the cell argv, the resumable scene copy, and the STUB REHEARSAL of the stage-2 driver.

Every `check_*(mut=None)` is one arm. A fault arm applies a deliberate fault and asserts the harness goes RED for the right reason; `mut="<name>"` (or `"not_applied"`) makes the
SAME assertion fail, so the arm can fail. `code/run_mutations_stage2.py` runs the mutations. Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_stage2.py -q

Hold executor (card of the other read: PREREG_EXEC_STANDSTILL section 3; the frozen T2 card runs `base`):
  G-ID    at eps = 1e-3 hold == base bit for bit (commit level and through `run_t2_rollout`).   RED: eps = 0.1 differs on standstill-noise plans
  G-MOV   when every executed step exceeds eps, hold(0.1) == base bit for bit.                  RED: a plan with sub-eps steps differs
  G-PHYS  under hold(0.1) no raw row with tangent <= eps changes the heading, every other row equals base, x / y / v equal base.   RED: hold disabled breaks it
Rehearsal: the REAL stage-1 tar + a FIXTURE overlay (3 scenes: 7c72937c in a shadow kit, 2 copied) in C: scratch, `t2v8f_stage2.py all --stub --inproc`; 42 cells.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
CODE = f"{T2}/code"
for _p in (CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402

import make_overlay2 as MO  # noqa: E402
import make_stage2 as MK  # noqa: E402
import t2v8f_exec as X  # noqa: E402
import t2v8f_exec_hold as H  # noqa: E402
import t2v8f_stage1 as S1  # noqa: E402
import t2v8f_stage2 as D  # noqa: E402
import t2v8f_stage2_cell as C  # noqa: E402
import test_t2v8f_stage1 as T1  # noqa: E402

PY = T1.PY
SCRATCH = os.environ.get("T2V8F_S2_SCRATCH", "C:/Users/Admin/t2a_scratch/stage2_rehearsal")
PLAN = f"{T2}/audit/stage2_plan.json"
_C = {}


# ============================================================================================================================ hold executor gates
def tool():
    if "tool" not in _C:
        _C["tool"] = X.load_tool()
    return _C["tool"]


def history(kind, n_rows=7, seed=0):
    """Raw 10 Hz rows (x, y, yaw, v), len odd (NOW is the last, an even index). moving = 10 m/s along +x; stopped = at rest with 0.1 mm jitter."""
    k = np.arange(n_rows, dtype=np.float64)
    if kind == "moving":
        return np.column_stack([100.0 + k, np.zeros(n_rows), np.zeros(n_rows), np.full(n_rows, 10.0)])
    rng = np.random.default_rng(seed)
    return np.column_stack([5.0 + 1e-4 * rng.standard_normal(n_rows), 2.0 + 1e-4 * rng.standard_normal(n_rows), np.full(n_rows, 0.3), np.zeros(n_rows)])


def plan_of(kind, seed):
    """30 ego-frame knots. random = curvy moving; moving = 2 m / 0.2 s (every executed step >> 0.1 m); noise = 1 cm noise from a stopped car (P10's plan);
    launch = 6 knots of noise, then a ramp to 6 m/s; slow_start = moving, but knots 3 to 8 are sub-eps (all inside the 10 executed knots)."""
    r = np.random.default_rng(1000 + seed)
    if kind == "random":
        st = np.column_stack([r.uniform(0.5, 3.0, 30), r.normal(0.0, 0.3, 30)])
    elif kind == "moving":
        st = np.column_stack([2.0 + 0.05 * r.standard_normal(30), 0.05 * r.standard_normal(30)])
    elif kind == "noise":
        st = 0.01 * r.standard_normal((30, 2))
    elif kind == "launch":
        st = np.vstack([0.01 * r.standard_normal((6, 2)), np.column_stack([np.linspace(0.05, 1.2, 24), 0.02 * r.standard_normal(24)])])
    elif kind == "slow_start":
        st = np.column_stack([2.0 + 0.05 * r.standard_normal(30), 0.05 * r.standard_normal(30)])
        st[3:9] = 0.01 * r.standard_normal((6, 2))                      # inside the 10 knots the executor commits
    else:
        raise ValueError(kind)
    return np.cumsum(st, axis=0)


class PlantedPolicy(D.ReplayPolicy):
    """Returns the planted ego-frame plans, one per replan (the real executor, the real rollout loop, no model)."""

    def __init__(self, gt_T, plans):
        super().__init__(gt_T, X)
        self.plans, self.k = plans, 0

    def plan(self, frames, intr, v0, nav_cmd):
        p = self.plans[self.k]
        self.k += 1
        return p.copy(), {}


def synthetic_scene(kind, n=200, v=10.0):
    """gt_T [n,4,4] and gt_ts_us at 10 Hz: kind moving = straight at v m/s, stopped = at rest with 0.1 mm jitter. Identity rotation, z = 0."""
    rng = np.random.default_rng(7)
    gt = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = v * 0.1 * i if kind == "moving" else 5.0 + 1e-4 * rng.standard_normal()
        T[1, 3] = 0.0 if kind == "moving" else 2.0 + 1e-4 * rng.standard_normal()
        gt.append(T)
    return gt, (1e5 * np.arange(n)).astype(np.int64)


def rollout(executor, eps, scene_kind, plan_kinds, seed=0):
    """`run_t2_rollout` (the real loop) under the base or the hold executor; the module is restored afterwards."""
    gt, ts = synthetic_scene(scene_kind)
    plans = [plan_of(k, seed + i) for i, k in enumerate(plan_kinds)]
    try:
        if executor == "hold":
            H.install(X, eps)
            assert H.active(X) == "hold"
        else:
            H.uninstall(X)
            assert H.active(X) == "base"
        return X.run_t2_rollout(PlantedPolicy(gt, plans), lambda T, t: None, gt, ts, 40, None, nav_cmd=0, condition="T2", tool=tool())
    finally:
        H.uninstall(X)
        assert H.active(X) == "base", "uninstall did not restore the base executor"


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def check_g_id(mut=None):
    """eps = 1e-3: hold equals base bit for bit on random, moving, standstill-noise and launch plans (commit level, 40 cases) and through `run_t2_rollout` (3 replans, the track
    and the T1-window arrays). The one place they can differ is a tangent of EXACTLY eps (base updates, hold holds): measure-zero on real data, pinned by `check_tie_documented`."""
    eps = 0.1 if mut == "eps_0.1" else 1e-3
    cls, T = H.make_hold_class(X.PlaybackExecutor, eps), tool()
    n = 0
    for hk, pk in (("moving", "random"), ("moving", "moving"), ("stopped", "noise"), ("stopped", "launch"), ("moving", "noise")):
        for seed in range(8):
            trk, plan = history(hk, seed=seed), plan_of(pk, seed)
            a, b = X.PlaybackExecutor(trk, T).commit(plan), cls(trk, T).commit(plan)
            assert same_bits(a, b), f"commit differs from base: history {hk}, plan {pk}, seed {seed}, max |d| {np.abs(a - b).max():.3g}"
            n += 1
    for sk, pks in (("moving", ("random", "random", "random")), ("stopped", ("noise", "noise", "launch"))):
        a, b = rollout("base", None, sk, pks), rollout("hold", eps, sk, pks)
        for key in ("track", "cl", "g"):
            assert same_bits(a[key], b[key]), f"rollout field {key} differs from base ({sk}, {pks})"
        assert b["policy_track_equals_executor"]
    return n


def check_g_mov(mut=None):
    """Every executed step above eps: hold(0.1) equals base bit for bit. RED: a plan with 6 sub-eps knots in the middle (hold holds the heading there, base follows the noise)."""
    pks = ("moving", "slow_start", "moving") if mut == "slow" else ("moving", "moving", "moving")
    a, b = rollout("base", None, "moving", pks), rollout("hold", 0.1, "moving", pks)
    assert same_bits(a["track"], b["track"]) and same_bits(a["cl"], b["cl"]), "hold(0.1) differs from base although every executed step exceeds 0.1 m"
    v = a["track"][:, 3] * X.DT if hasattr(X, "DT") else a["track"][:, 3] * 0.2
    assert float(v[len(v) - 30:].min()) > 0.1 or mut == "slow"
    return float(np.abs(a["track"][:, 2]).max())


def check_g_phys(mut=None):
    """Hold(0.1) on a stopped car with 1 cm plan noise (10 seeds): (1) x, y, v equal base; (2) every new raw row whose tangent (= v * 0.2 s) is <= 0.1 keeps the previous heading
    exactly; (3) every other row equals base. Non-vacuous: at least 5 held rows per case. RED: hold disabled (the base class) changes the heading on the held rows."""
    T = tool()
    try:
        H.install(X, 0.1)
        cls = X.PlaybackExecutor
        if mut == "hold_disabled":
            H.uninstall(X)
            cls = X.PlaybackExecutor
        held_total = 0
        for seed in range(10):
            trk, plan = history("stopped", seed=seed), plan_of("noise", seed)
            base, new = X._PLAYBACK_BASE(trk, T).commit(plan), cls(trk, T).commit(plan)
            assert same_bits(base[:, [0, 1, 3]], new[:, [0, 1, 3]]), f"seed {seed}: x / y / v differ from base (only the yaw column may)"
            held = new[:, 3] * 0.2 <= 0.1
            assert held.sum() >= 5, f"seed {seed}: only {held.sum()} held rows: the test would be vacuous"
            prev = np.concatenate([[trk[-1, 2]], new[:-1, 2]])
            assert np.array_equal(new[held, 2], prev[held]), f"seed {seed}: a row with tangent <= 0.1 changed the heading"
            assert np.array_equal(new[~held, 2], base[~held, 2]), f"seed {seed}: a row with tangent > 0.1 differs from base"
            held_total += int(held.sum())
        return held_total
    finally:
        H.uninstall(X)


def check_tie_documented(mut=None):
    """The rules differ at exactly one point: a tangent of EXACTLY eps (base `>=` updates, hold `>` holds). Constructed knots with a tangent of exactly 1e-3: pinned, not hidden."""
    T = tool()
    # K = [k_prev, (0,0), (0,0), (0.002,0), (0.002,0)]: the first interval's midpoint tangent is 2.5e-4 (both hold), the knot tangent is 0.5 * (0.002 - 0) = 1e-3 exactly
    k_prev, knots = np.array([0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0], [0.002, 0.0], [0.002, 0.0]])
    xb, pb, _ = T.hermite_frames(k_prev, knots, 2, 1.0)
    xh, ph, _ = H.hermite_frames_eps(k_prev, knots, 2, 1.0, 1e-3 if mut != "same_rule" else 0.0)
    assert np.array_equal(xb, xh), "positions differ"
    assert pb[0] == ph[0] == 1.0, "the sub-eps midpoint frame must hold in both"
    assert pb[1] == 0.0 and ph[1] == 1.0, f"expected base to update (0.0) and hold to hold (1.0) at a tangent of exactly eps, got {pb[1]}, {ph[1]}"
    return pb.tolist(), ph.tolist()


def check_cell_switch(mut=None):
    """`t2v8f_stage2_cell.split_argv`: default base, flags removed from what goes on, an unknown executor refuses; `hold` installs, `base` restores (reachability of the flag)."""
    ex, eps, rest = C.split_argv(["--arm", "C0_s0", "--executor", "hold", "--hold-eps", "0.25", "--out", "x"])
    assert (ex, eps, rest) == ("hold", 0.25, ["--arm", "C0_s0", "--out", "x"])
    assert C.split_argv(["--arm", "P10"]) == ("base", 0.1, ["--arm", "P10"])
    assert C.split_argv(["--executor=hold", "--hold-eps=0.5"]) == ("hold", 0.5, [])
    try:
        C.split_argv(["--executor", "bogus"])
        raise AssertionError("an unknown executor was accepted")
    except SystemExit:
        pass
    try:
        H.install(X, 0.3)
        assert H.active(X) == "hold" and X.PlaybackExecutor.HOLD_EPS == 0.3
    finally:
        if mut != "no_restore":
            H.uninstall(X)
    assert H.active(X) == "base", "uninstall did not restore"
    H.uninstall(X)
    return True


# ============================================================================================================================ order, argv, resume
def plan_scenes():
    p = json.load(open(PLAN, encoding="utf-8"))
    return [dict(s, idx=i) for i, s in enumerate(p["scenes"])]


def check_order(mut=None):
    """588 cells over the real plan: 420 deciding (5 arms x {T2, RR} x 42 scenes) then 168 beside (DAGOACC x 2 x 42), scene-major inside a pass, every cell unique,
    the X1 / X2 pairs adjacent at the head of each scene. RED: beside first."""
    sc = plan_scenes()
    order = D.cell_order(sc, ("beside", "deciding") if mut == "beside_first" else ("deciding", "beside"))
    assert len(order) == 588 and len({(c["idx"], c["tc"], c["arm"]) for c in order}) == 588
    assert [c["pass"] for c in order] == ["deciding"] * 420 + ["beside"] * 168, "the deciding cells must all run before any beside cell"
    assert {c["arm"] for c in order[:420]} == set(D.DECIDING) and {c["arm"] for c in order[420:]} == set(D.BESIDE)
    assert [c["idx"] for c in order[:420]] == [i for i in range(42) for _ in range(10)] and [c["idx"] for c in order[420:]] == [i for i in range(42) for _ in range(4)]
    head = [(c["tc"], c["arm"]) for c in order[:5]]
    assert head == [("T2", "C0_s0"), ("T2", "P10"), ("RR", "C0_s0"), ("T2", "C0_s1"), ("RR", "C0_s1")], head
    return len(order)


def fake_ctx(kit="K", pkg="P", scenes2="S2"):
    f = types.SimpleNamespace(pkg=pkg, kit=kit, hf="H", encoder="dinov3", device="cuda", scenes2=scenes2)
    f.scene_dir = lambda sid: f"{kit}/scenes/{sid}"
    f.scene_dir2 = lambda sc: D.Ctx2.scene_dir2(f, sc)
    return f


def check_cell_argv(mut=None):
    """The stage-2 cell argv equals stage 1's `entry_args` for the same cell minus `--t2-plumbing`, for all 7 arms x {T2, RR}; for a scenes2 scene only the scene dir differs.
    RED: the plumbing flag kept."""
    f, sc = fake_ctx(), plan_scenes()
    for scene, expect_dir in ((sc[22], "K/scenes/"), (sc[0], "S2/")):
        assert scene["scene"].startswith("7c72937c") == (scene is sc[22])
        s = dict(scene, where="kit" if scene is sc[22] else "scenes2")
        for tc in ("T2", "RR"):
            for arm in D.DECIDING + D.BESIDE:
                a1 = S1.Ctx.entry_args(f, arm, tc, "objects", "OUT", s["scene"], s["idx"], ",".join(str(x) for x in s["starts"]))
                a2 = D.cell_argv(f, s, tc, arm, "OUT")
                want = a1 if mut == "plumbing_kept" else [x for x in a1 if x != "--t2-plumbing"]
                if s["where"] == "scenes2":
                    want = [w.replace(f"K/scenes/{s['scene']}", f"S2/{s['scene']}") for w in want]
                assert a2 == want, f"{arm} {tc} {s['scene'][:8]}: {a2} != {want}"
                assert a2[1] == f"{expect_dir}{s['scene']}"
    return True


def check_cell_footprint(mut=None):
    """The cell directory of the MEASURED stage-1 pull (box 2) costs CELL_MIB on a 1 MiB-cluster volume (dir + DONE + actor_attach + records + rollouts + scene npz)."""
    d = f"{T2}/stage1/pull_box2/pull/plumbing/00040136/objects/T2/P10"
    got = T1.allocated_mib(d)
    assert got == MK.CELL_MIB - (1 if mut == "off_by_one" else 0), f"a stage-1 cell takes {got} MiB, make_stage2.CELL_MIB says {MK.CELL_MIB}"
    return got


def check_resume(mut=None):
    """`make_stage2.resume`: a complete remote file is skipped, a SHORTER one is continued with reput, a missing one is put, a LONGER one is put again; the kit scene never appears."""
    out = f"{SCRATCH}/resume_{time.strftime('%H%M%S')}_{mut}"
    os.makedirs(out, exist_ok=True)
    mk = lambda i, rel, n, w="scenes2": {"idx": i, "scene": f"sc{i:06d}-aaaa", "rel": rel, "bytes": n, "sha256": "0" * 64, "source": f"F:/src/sc{i}/{rel}", "where": w}      # noqa: E731
    src = [mk(1, "volume.nurec", 1000), mk(1, "rig_trajectories.json", 50), mk(2, "volume.nurec", 2000), mk(2, "extracted/sequence_tracks.json", 70), mk(3, "volume.nurec", 3000, "kit")]
    json.dump(src, open(f"{out}/src.json", "w"))
    short = 1000 if mut == "short_is_complete" else 400
    lines = [f"D:\\t2v8f\\scenes2\\sc000001-aaaa\\volume.nurec|{short}", "D:\\t2v8f\\scenes2\\sc000001-aaaa\\rig_trajectories.json|50", "D:\\t2v8f\\scenes2\\sc000002-aaaa\\volume.nurec|9999"]
    open(f"{out}/sizes.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    res = MK.resume(f"{out}/sizes.txt", f"{out}/src.json", f"{out}/b.sftp")
    txt = open(f"{out}/b.sftp", encoding="utf-8").read()
    assert (res["skipped_complete"], res["reput"], res["put"]) == (1, 1, 2), res
    assert "reput F:/src/sc1/volume.nurec /D:/t2v8f/scenes2/sc000001-aaaa/volume.nurec" in txt and "put F:/src/sc2/volume.nurec" in txt and "sc3" not in txt and "sc000003" not in txt
    return res


# ============================================================================================================================ the stub rehearsal
def copy_small(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def fixture():
    """-> dict(pkg, kit, hf, scenes2_src, ids, plan). Built once per session: the fixture sources (16-byte volume placeholders, REAL rig and tracks), the overlay built by
    `make_stage2.build(..., fixture_tag='_fx')`, the REAL stage-1 tar extracted + the fixture overlay over it, a shadow kit with 7c72937c (real rig and tracks)."""
    if "fx" in _C:
        return _C["fx"]
    plan = json.load(open(PLAN, encoding="utf-8"))
    by = {s["scene"]: s for s in plan["scenes"]}
    kit_id = next(s for s in by if s.startswith("7c72937c"))
    one = next(s["scene"] for s in plan["scenes"] if len(s["starts"]) == 1)
    two = next(s["scene"] for s in plan["scenes"] if len(s["starts"]) == 2 and not s["scene"].startswith("7c72937c"))
    ids = [kit_id, two, one]
    tag = time.strftime("%Y%m%d_%H%M%S")
    root = f"{SCRATCH}/fx_{tag}"
    src, fx_scenes = f"{root}/src", []
    for sid in ids:
        s = by[sid]
        copy_small(next(f["location"] for f in s["files"] if f["member"] == "rig_trajectories.json"), f"{src}/{sid}/rig_trajectories.json")
        copy_small(next(f["location"] for f in s["files"] if f["member"] == "sequence_tracks.json"), f"{src}/{sid}/extracted/sequence_tracks.json")
        open(f"{src}/{sid}/volume.nurec", "wb").write(b"PLACEHOLDER-0000\n"[:16])
        rng = np.random.default_rng(len(fx_scenes))                                                  # the scene's reference frames (tiny, distinct per scene), as `extract_ref_frames.py` banks them
        np.savez_compressed(f"{src}/{sid}/ref_frames.npz", **{f"f{i}": (rng.random((36, 64, 3)) * 255).astype(np.uint8) for i in D.REF_IDX})
        files = []
        for mem, rel in MK.REL.items():
            p = f"{src}/{sid}/{rel}"
            files.append({"member": mem, "bytes": os.path.getsize(p), "sha256": MK.sha256_file(p), "location": p})
        fx_scenes.append(dict(s, files=files))
    fx_plan = f"{root}/fx_plan.json"
    json.dump({"scenes": fx_scenes}, open(fx_plan, "w"), indent=1)
    out = f"{root}/overlay_out"
    res = MK.build(fx_plan, out, f"{root}/stage", verify_src=True, plan_root=src, fixture_tag="_fx")
    json.dump({sid: {"path": f"{src}/{sid}/ref_frames.npz"} for sid in ids}, open(f"{root}/refs_fx.json", "w"), indent=1)
    res2 = MO.build(res["tar"], f"{root}/refs_fx.json", f"{root}/overlay2_out", tag="_fx2")                # overlay 2, derived from the fixture's overlay 1 by the REAL builder
    # the real stage-1 tar, extracted + checked like stage 1's rehearsal, then the fixture overlay over it
    want = open(T1.TAR + ".sha256").read().split()[0]
    assert T1.sha256_file(T1.TAR) == want, "the stage-1 tar does not match its .sha256"
    x = f"{root}/x"
    with tarfile.open(T1.TAR) as tf:
        tf.extractall(x, filter="data")
    with tarfile.open(res["tar"]) as tf:
        tf.extractall(x, filter="data")
    with tarfile.open(res2["tar"]) as tf:                                                            # then overlay 2 over it: MANIFEST2.json + the driver are replaced, scenes2/<id>/ref_frames.npz added
        tf.extractall(x, filter="data")
    pkg = f"{x}/pkg"
    assert T1.tree_ok(pkg), "the stage-1 files were changed by the overlay"
    m = json.load(open(f"{pkg}/MANIFEST.json", encoding="utf-8"))
    for e in m["files"]:
        if e["path"].startswith("code/"):
            assert T1.sha256_file(f"{CODE}/{e['path'][5:]}") == e["sha256"], f"STALE stage-1 TAR: {e['path']}"
    # shadow kit: the kit's code, 7c72937c with the REAL rig + tracks (+ placeholder volume), 00040136 as stage 1's placeholders; its own KIT_MANIFEST.json
    kit = f"{root}/shadow_kit"
    for sub in ("t2a", "tree/stack/experiments/alpasim-gsplat"):
        os.makedirs(f"{kit}/{sub}", exist_ok=True)
        for f in sorted(glob.glob(f"{T1.REAL_KIT}/{sub}/*.py")):
            shutil.copyfile(f, f"{kit}/{sub}/{os.path.basename(f)}")
    dst = "tree/stack/experiments/alpasim-gsplat/closedloop_drive.py"
    items = [{"kind": "overlay", "dst": dst, "size": os.path.getsize(f"{kit}/{dst}"), "md5": hashlib.md5(open(f"{kit}/{dst}", "rb").read()).hexdigest()}]
    os.makedirs(f"{kit}/scenes/00040136-e651-4abd-991d-0655ccda9430", exist_ok=True)
    for fn in ("volume.nurec", "rig_trajectories.json"):
        open(f"{kit}/scenes/00040136-e651-4abd-991d-0655ccda9430/{fn}", "wb").write(b"PLACEHOLDER-0000\n"[:16])
    shutil.copytree(f"{src}/{kit_id}", f"{kit}/scenes/{kit_id}")
    for sid in ("00040136-e651-4abd-991d-0655ccda9430", kit_id):
        for fn in ("volume.nurec", "rig_trajectories.json"):
            p = f"{kit}/scenes/{sid}/{fn}"
            items.append({"kind": "scene", "dst": f"scenes/{sid}/{fn}", "size": os.path.getsize(p), "md5": hashlib.md5(open(p, "rb").read()).hexdigest()})
    json.dump({"note": "SHADOW kit for the stage-2 rehearsal; the real kit is never written", "items": items}, open(f"{kit}/KIT_MANIFEST.json", "w"), indent=1)
    os.makedirs(f"{root}/hf", exist_ok=True)
    _C["fx"] = {"root": root, "pkg": pkg, "kit": kit, "hf": f"{root}/hf", "src": src, "ids": ids, "kit_id": kit_id, "two": two, "one": one, "build": res, "build2": res2, "x": x}
    return _C["fx"]


def new_scenes2(fx, tag):
    """The 2 copied scenes, as the sftp batch would leave them on box 2."""
    d = f"{fx['root']}/scenes2_{tag}_{time.strftime('%H%M%S')}"
    for sid in fx["ids"]:
        if sid != fx["kit_id"]:
            shutil.copytree(f"{fx['src']}/{sid}", f"{d}/{sid}", ignore=shutil.ignore_patterns("ref_frames.npz"))
            shutil.copyfile(f"{fx['x']}/scenes2/{sid}/ref_frames.npz", f"{d}/{sid}/ref_frames.npz")          # from the EXTRACTED overlay-2 tar, as `tar -xf` leaves it on box 2
    return d


def new_pull(fx, tag):
    p = f"{fx['root']}/pull_{tag}_{time.strftime('%H%M%S')}"
    assert not os.path.exists(p)
    return p


def run_s2(fx, pull, scenes2, extra=(), timeout=1500):
    t0 = time.time()
    cmd = [PY, "-I", "-u", f"{fx['pkg']}/code/t2v8f_stage2.py", "all", "--pkg", fx["pkg"], "--kit", fx["kit"], "--hf", fx["hf"], "--pull", pull, "--scenes2", scenes2, "--stub", "--inproc", *extra]
    p = subprocess.run(cmd, env=T1.clean_env(), capture_output=True, text=True, timeout=timeout)
    os.makedirs(os.path.dirname(pull), exist_ok=True)
    with open(f"{pull}.stdout.txt", "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n---stderr---\n" + p.stderr)
    return p.returncode, round(time.time() - t0, 1)


def res2(pull):
    return json.load(open(f"{pull}/STAGE2_RESULT.json", encoding="utf-8"))


def cells_of(pull):
    return sorted(glob.glob(f"{pull}/rollouts/*/objects/*/*/DONE"))


def positive():
    if "pos" not in _C:
        fx = fixture()
        scenes2, pull = new_scenes2(fx, "pos"), new_pull(fx, "pos")
        rc, secs = run_s2(fx, pull, scenes2)
        _C["pos"] = (pull, scenes2, rc, secs)
    return _C["pos"]


def check_positive(mut=None):
    """All three jobs PASS on the stub: STAGE2.done, 42 cells DONE (30 deciding then 12 beside, `executor=base` in the DONE text), 70 records stamped `deciding candidate`,
    GATE_SCENES.json finite and <= 1e-6, the cell takes CELL_MIB, the DONE mtimes keep the deciding-before-beside order."""
    fx = fixture()
    pull, _, rc, secs = positive()
    r = res2(pull)
    assert rc == 0, f"driver rc {rc}: {open(pull + '.stdout.txt', encoding='utf-8').read()[-1500:]}"
    assert os.path.exists(f"{pull}/STAGE2.done") and not os.path.exists(f"{pull}/STAGE2.fail") and not os.path.exists(f"{pull}/STAGE2.running")
    assert r["gates"] == {"preflight": "PASS", "gate": "PASS", "rollouts": "PASS"} and r["overall"] == "COMPLETE", r["gates"]
    assert r["executor"] == "base" and r["steps"]["preflight"]["failures"] == [] and r["steps"]["preflight"]["info"]["overlay_files_ok"] == r["steps"]["preflight"]["info"]["overlay_files"]
    assert r["steps"]["preflight"]["info"]["scene_files_ok"] == r["steps"]["preflight"]["info"]["scene_files"] == 9
    done = cells_of(pull)
    assert len(done) == 42 == r["steps"]["rollouts"]["n_cells_planned"], len(done)
    for f in done:
        t = open(f, encoding="utf-8").read().strip()
        assert t.startswith("rc=0 records=") and t.endswith("executor=base"), t
    recs = [json.loads(x) for f in glob.glob(f"{pull}/rollouts/**/records.jsonl", recursive=True) for x in open(f, encoding="utf-8")]
    assert len(recs) == 70 and {x["purpose"] for x in recs} == {"deciding candidate"} and all(len(x["plans"]) == 3 for x in recs)
    g = json.load(open(f"{pull}/GATE_SCENES.json", encoding="utf-8"))
    assert len(g["rows"]) == 5 and all(x["finite"] and x["max_abs_diff_m"] <= 1e-6 for x in g["rows"]) and not g["failures"]
    mt = {p: [os.path.getmtime(f) for f in done if f"/{p}/" not in "" and os.path.basename(os.path.dirname(f)).startswith("DAGOACC") == (p == "beside")] for p in ("deciding", "beside")}
    assert len(mt["deciding"]) == 30 and len(mt["beside"]) == 12 and max(mt["deciding"]) < min(mt["beside"]), "a beside cell finished before a deciding cell"
    assert T1.allocated_mib(os.path.dirname(done[0])) == MK.CELL_MIB - (1 if mut == "footprint" else 0), "a stub cell does not cost CELL_MIB"
    return {"seconds": secs, "cells": len(done), "records": len(recs)}


def cell_env(fx):
    env = T1.clean_env()
    env.update(T2A_STUB="1", T2A_KIT=fx["kit"], T2V8F_TREE=f"{fx['pkg']}/tree", T2V8F_V9_DIR=f"{fx['pkg']}/v9", T2V8F_TOOL=f"{fx['pkg']}/refs/tool/v8f_t1_eval_ri_lf6.py",
               T2V8F_SCREEN_TABLE=f"{fx['pkg']}/refs/SCREEN_TABLE_2.json", HF_HOME=fx["hf"], HF_HUB_OFFLINE="1")
    return env


def kit_cell(fx, scenes2):
    """(ctx-like, the manifest scene) of the kit scene 7c72937c for `cell_argv`."""
    f = types.SimpleNamespace(pkg=fx["pkg"], kit=fx["kit"], hf=fx["hf"], encoder="stub", device="cpu", scenes2=scenes2)
    f.scene_dir2 = lambda sc: D.Ctx2.scene_dir2(f, sc)
    return f, next(s for s in json.load(open(f"{fx['pkg']}/MANIFEST2.json", encoding="utf-8"))["scenes"] if s["scene"] == fx["kit_id"])


def run_cell_proc(fx, scenes2, tag, script, flags, mut):
    """ONE fresh process (the shipping path: the real driver runs a child per cell) for the kit scene, T2, C0_s0. -> its records."""
    f, sc = kit_cell(fx, scenes2)
    out = f"{fx['root']}/{tag}_{time.strftime('%H%M%S')}_{mut}"
    os.makedirs(out)
    p = subprocess.run([PY, "-I", "-u", f"{fx['pkg']}/code/{script}"] + flags + D.cell_argv(f, sc, "T2", "C0_s0", out), env=cell_env(fx), capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-1200:]
    return [json.loads(x) for x in open(f"{out}/records.jsonl", encoding="utf-8")]


def check_base_equals_direct(mut=None):
    """The stage-2 cell under `--executor base` is exactly `t2v8f_run.py`: scene 7c72937c (kit), T2, C0_s0, each in a fresh process; the records are equal bit for bit.
    The cell the DRIVER ran in-process (rehearsal only) agrees to < 0.05 m (in-process float drift of the stub policy: MEASURED 2e-3 m; the real driver uses one process per cell).
    RED: the stage-2 cell is run with `--executor hold --hold-eps 1e9` (heading never updated: must not equal base)."""
    fx = fixture()
    pull, scenes2, rc, _ = positive()
    assert rc == 0
    a = run_cell_proc(fx, scenes2, "direct", "t2v8f_run.py", [], mut)
    b = run_cell_proc(fx, scenes2, "cellbase", "t2v8f_stage2_cell.py", ["--executor", "hold", "--hold-eps", "1e9"] if mut == "hold_instead" else [], mut)
    assert len(a) == len(b) == 2
    for x, y in zip(a, b):
        for k in ("plans", "track", "cl", "g", "frame_md5", "v0", "purpose", "f0", "calls"):
            assert x[k] == y[k], f"record field {k} differs between the direct run and the stage-2 cell"
    c = [json.loads(x) for x in open(f"{pull}/rollouts/{fx['kit_id'][:8]}/objects/T2/C0_s0/records.jsonl", encoding="utf-8")]
    assert len(c) == 2
    for x, y in zip(a, c):
        assert x["g"] == y["g"] and x["v0"] == y["v0"] and x["f0"] == y["f0"] and len(x["frame_md5"]) == len(y["frame_md5"])
        assert float(np.abs(np.asarray(x["plans"]) - np.asarray(y["plans"])).max()) < 0.05, "the driver's in-process cell is not the same rollout"
    return len(a)


def check_hold_reachable(mut=None):
    """`--executor hold` reaches the executor through the real cell entry (caller reachability), each run a fresh process: hold(1e-3) equals the base cell bit for bit (G-ID through
    the caller) and hold(1e9) never updates the heading on the new rows (yaw constant). RED: the executor flag is dropped before the cell (base runs) and the yaw is NOT constant."""
    fx = fixture()
    pull, scenes2, rc, _ = positive()
    base = run_cell_proc(fx, scenes2, "hbase", "t2v8f_stage2_cell.py", [], mut)
    ident = run_cell_proc(fx, scenes2, "hid", "t2v8f_stage2_cell.py", [] if mut == "flag_dropped" else ["--executor", "hold", "--hold-eps", "0.001"], mut)
    frozen = run_cell_proc(fx, scenes2, "hfrozen", "t2v8f_stage2_cell.py", [] if mut == "flag_dropped" else ["--executor", "hold", "--hold-eps", "1e9"], mut)
    for x, y in zip(ident, base):
        assert x["track"] == y["track"] and x["cl"] == y["cl"], "hold(1e-3) differs from base through the cell entry"
    for r in frozen:
        trk = np.asarray(r["track"])
        n_hist = 2 * r["t0_step"] + 1
        assert np.all(trk[n_hist:, 2] == trk[n_hist - 1, 2]), "hold(1e9) changed the heading: the flag did not reach the executor"
    return len(frozen)


def refused(fx, tag, scenes2, extra=(), mutate=None):
    pull = new_pull(fx, tag)
    rc, secs = run_s2(fx, pull, scenes2, extra)
    return pull, rc, res2(pull)


def check_planted_missing(mut=None):
    """A scene file of a COPIED scene is missing (refuse): STAGE2.fail names `MISSING scenes2/<id>/volume.nurec`, the steps are [preflight] only, no cell ran.
    skip policy: the scene is named in the preflight info, the other scenes run (INCOMPLETE, rc 0, no cell of the missing scene)."""
    fx = fixture()
    miss = fx["two"]
    s2 = new_scenes2(fx, "miss")
    if mut != "not_applied":
        os.replace(f"{s2}/{miss}/volume.nurec", f"{s2}/{miss}/volume.nurec.aside_by_test")
    pull, rc, r = refused(fx, "miss", s2)
    tag = f"MISSING scenes2/{miss}/volume.nurec"
    assert rc == 1 and os.path.exists(f"{pull}/STAGE2.fail") and not os.path.exists(f"{pull}/STAGE2.done"), (rc, os.listdir(pull))
    assert list(r["steps"]) == ["preflight"] and r["overall"] == "INFRASTRUCTURE_FAILURE" and r["steps"]["preflight"]["failures"] == [tag], r["steps"]["preflight"]["failures"]
    assert tag in open(f"{pull}/STAGE2.fail", encoding="utf-8").read() and not os.path.exists(f"{pull}/rollouts") and not os.path.exists(f"{pull}/GATE_SCENES.json")
    return {"rc": rc}


def check_skip_policy(mut=None):
    fx = fixture()
    miss = fx["two"]
    s2 = new_scenes2(fx, "skip")
    if mut != "not_applied":
        os.replace(f"{s2}/{miss}/rig_trajectories.json", f"{s2}/{miss}/rig_trajectories.json.aside_by_test")
    pull, rc, r = refused(fx, "skip", s2, ["--scene-policy", "skip", "--passes", "beside"])
    assert rc == 0 and r["overall"] == "INCOMPLETE" and r["gates"] == {"preflight": "PASS", "gate": "PASS", "rollouts": "INCOMPLETE"}, (rc, r["gates"], r["overall"])
    assert list(r["steps"]["preflight"]["info"]["scenes_skipped"]) == [miss] and r["steps"]["preflight"]["info"]["scenes_skipped"][miss] == [f"MISSING scenes2/{miss}/rig_trajectories.json"]
    got = {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(f))))) for f in cells_of(pull)}
    assert got == {fx["kit_id"][:8], fx["one"][:8]} and len(cells_of(pull)) == 8, got
    return {"cells": len(cells_of(pull))}


def check_changed_byte(mut=None):
    """One byte of a copied scene's `rig_trajectories.json` changed (same size): `HASH_MISMATCH scenes2/<id>/rig_trajectories.json` is the ONLY failure, steps [preflight]."""
    fx = fixture()
    s2 = new_scenes2(fx, "flip")
    p = f"{s2}/{fx['one']}/rig_trajectories.json"
    if mut != "not_applied":
        b = bytearray(open(p, "rb").read())
        b[len(b) // 2] ^= 0x01
        open(p, "wb").write(bytes(b))
    pull, rc, r = refused(fx, "flip", s2)
    assert rc == 1 and list(r["steps"]) == ["preflight"] and r["steps"]["preflight"]["failures"] == [f"HASH_MISMATCH scenes2/{fx['one']}/rig_trajectories.json"], r["steps"]["preflight"]["failures"]
    return {"rc": rc}


def check_done_skipped(mut=None):
    """Relaunch on a copy of the finished pull: all 42 cells SKIPPED_DONE (none re-run), every records.jsonl unchanged byte for byte. RED: one DONE removed (that cell re-runs)."""
    fx = fixture()
    pull0, scenes2, rc0, _ = positive()
    assert rc0 == 0
    pull = new_pull(fx, "resume")
    shutil.copytree(pull0, pull)
    if mut == "one_not_done":
        os.remove(cells_of(pull)[0])
    before = {f: MK.sha256_file(f) for f in glob.glob(f"{pull}/rollouts/**/records.jsonl", recursive=True)}
    rc, _ = run_s2(fx, pull, scenes2)
    r = res2(pull)
    st = r["steps"]["rollouts"]["state"]
    assert rc == 0 and r["overall"] == "COMPLETE" and st["deciding"]["done"] == 0 and st["beside"]["done"] == 0 and st["deciding"]["skipped_done"] == 30 and st["beside"]["skipped_done"] == 12, st
    assert before == {f: MK.sha256_file(f) for f in glob.glob(f"{pull}/rollouts/**/records.jsonl", recursive=True)}, "a skipped cell was rewritten"
    return st


def check_executor_mismatch(mut=None):
    """A pull finished under `base` relaunched with `--executor hold` refuses (EXECUTOR_MISMATCH in STAGE2.fail), rewrites no record. RED: relaunched with base (no refusal)."""
    fx = fixture()
    pull0, scenes2, rc0, _ = positive()
    pull = new_pull(fx, "mismatch")
    shutil.copytree(pull0, pull)
    rc, _ = run_s2(fx, pull, scenes2, ["--executor", "base" if mut == "not_applied" else "hold", "--passes", "deciding"])
    assert rc == 1 and os.path.exists(f"{pull}/STAGE2.fail") and "EXECUTOR_MISMATCH" in open(f"{pull}/STAGE2.fail", encoding="utf-8").read(), rc
    return {"rc": rc}


def check_gate_red(mut=None):
    """A plan whose n_ticks does not match the rig (MANIFEST2 edited, the rest untouched): the preflight passes, the CPU gate FAILS before any render
    (overall GATE_FAILED, steps [preflight, gate], no cell). RED: the manifest is left as built."""
    fx = fixture()
    mp = f"{fx['pkg']}/MANIFEST2.json"
    orig = open(mp, "rb").read()
    try:
        if mut != "not_applied":
            m = json.loads(orig)
            next(s for s in m["scenes"] if s["scene"] == fx["one"])["n_ticks"] += 1
            open(mp, "w", encoding="utf-8", newline="\n").write(json.dumps(m, indent=1))
        pull, rc, r = refused(fx, "gate", new_scenes2(fx, "gate"))
    finally:
        open(mp, "wb").write(orig)
    assert r["overall"] == "GATE_FAILED" and list(r["steps"]) == ["preflight", "gate"] and r["gates"]["gate"] == "FAIL", r["gates"]
    assert any("ticks" in x for x in r["steps"]["gate"]["failures"]) and not os.path.exists(f"{pull}/rollouts"), r["steps"]["gate"]["failures"]
    return {"failures": r["steps"]["gate"]["failures"][:1]}


def check_builder_refuses_bad_plan(mut=None):
    """`make_stage2.build` refuses a plan whose n_ticks differs from the rig (before any tar). RED: the plan is left as it is."""
    fx = fixture()
    p = json.load(open(f"{fx['root']}/fx_plan.json", encoding="utf-8"))
    if mut != "not_applied":
        p["scenes"][1]["n_ticks"] += 1
    bad = f"{fx['root']}/bad_plan_{time.strftime('%H%M%S')}_{mut}.json"
    json.dump(p, open(bad, "w"))
    try:
        MK.build(bad, f"{fx['root']}/bad_out_{time.strftime('%H%M%S')}_{mut}", f"{fx['root']}/bad_stage_{time.strftime('%H%M%S')}_{mut}", verify_src=False, plan_root=fx["src"], fixture_tag="_fx")
    except AssertionError as e:
        assert "ticks" in str(e), str(e)
        return str(e)[:80]
    raise AssertionError("the builder accepted a plan whose n_ticks differs from the rig")



# ============================================================================================================================ the scorer: rows per start class
def check_scorer_classes(mut=None):
    """`t2v8f_stage2_score.py` on the MEASURED stage-1 pull (box 2): 00040136 f0 60 (v0 17.8 m/s) is MOVING, 7c72937c f0 22 (v0 0.0008 m/s) is a LAUNCH; the classes are never pooled:
    P10(T2) heading MAE reads 10.25 deg (moving) and 61.43 deg (launch) with yaw rate 5.06 and 283.03 deg/s (AUDIT_NUREC_SCORE section 1), while the pooled mean is 35.84.
    The class speed equals every record's own `v0` exactly. RED: reading the pooled file as if it were a class."""
    out = f"{SCRATCH}/score_cls_{time.strftime('%H%M%S')}_{mut}"
    os.makedirs(out, exist_ok=True)
    p = subprocess.run([PY, "-I", f"{CODE}/t2v8f_stage2_score.py", "--root", f"{T2}/stage1/pull_box2/pull/plumbing", "--out", f"{out}/SCORE2.json", "--md", f"{out}/SCORE2.md"],
                       env=T1.clean_env(), capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-1200:]
    r = json.load(open(f"{out}/SCORE2.json", encoding="utf-8"))
    assert {k: v["class"] for k, v in r["starts"].items()} == {"00040136|f0=60": "moving", "7c72937c|f0=22": "launch"}, r["starts"]
    assert r["checks"]["ok"] and r["checks"]["max_abs_v0_logged_minus_record_mps"] < 1e-9
    if mut == "pooled":
        pooled = json.load(open(f"{out}/SCORE2_POOLED.json", encoding="utf-8"))["groups"]["P10|T2|objects"]
        cl = {"moving": {"mean": pooled["mean"], "n_starts": pooled["n_starts"]}, "launch": {"mean": pooled["mean"], "n_starts": pooled["n_starts"]}}
    else:
        cl = {c: r["classes"][c]["P10|T2|objects"] for c in ("moving", "launch")}
    for c, h, y in (("moving", 10.25, 5.06), ("launch", 61.43, 283.03)):
        assert cl[c]["n_starts"] == 1, f"{c}: {cl[c]['n_starts']} starts in a group (the classes must not be pooled)"
        assert abs(cl[c]["mean"]["heading MAE"] - h) < 0.01 and abs(cl[c]["mean"]["yaw-rate MAE"] - y) < 0.01, (c, cl[c]["mean"]["heading MAE"], cl[c]["mean"]["yaw-rate MAE"])
    assert set(r["classes"]) == {"launch", "moving"} and "contrasts" in r and set(r["contrasts"]["launch"]["seed0"]) >= {"heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE"}
    return {k: v["v0_mps"] for k, v in r["starts"].items()}


def check_start_class_rule(mut=None):
    """The rule is `v0 < 0.5 m/s` and nothing else, in both the builder (logged rig, backward difference at f0) and the scorer. RED: a 0.1 m/s threshold (the audit's standstill rule)."""
    import t2v8f_stage2_score as SC
    thr = 0.1 if mut == "audit_rule" else 0.5
    assert SC.LAUNCH_V0_MPS == MK.LAUNCH_V0 == thr, (SC.LAUNCH_V0_MPS, MK.LAUNCH_V0)
    assert SC.start_class(0.4999) == "launch" and SC.start_class(0.5) == "moving" and SC.start_class(17.8) == "moving" and SC.start_class(0.0008) == "launch"
    gt = []
    for i in range(120):
        T = np.eye(4)
        T[0, 3] = 0.03 * i if i <= 30 else 0.9 + 1.5 * (i - 30)
        gt.append(T)
    assert MK.start_class(gt, 20)[0] == "launch" and MK.start_class(gt, 40)[0] == "moving"
    assert abs(SC.logged_v0(gt, 20) - 0.3) < 1e-9 and abs(MK.start_class(gt, 20)[1] - 0.3) < 1e-9
    return True


# ============================================================================================================================ pytest
def test_g_id():
    check_g_id()


def test_g_mov():
    check_g_mov()


def test_g_phys():
    check_g_phys()


def test_tie_documented():
    check_tie_documented()


def test_cell_switch():
    check_cell_switch()


def test_order():
    check_order()


def test_cell_argv():
    check_cell_argv()


def test_cell_footprint():
    check_cell_footprint()


def test_resume():
    check_resume()


def test_positive():
    out = check_positive()
    os.makedirs(f"{T2}/logs", exist_ok=True)
    json.dump(out, open(f"{T2}/logs/rehearsal_stage2_positive.json", "w"), indent=1)


def test_base_equals_direct():
    check_base_equals_direct()


def test_hold_reachable():
    check_hold_reachable()


def test_planted_missing():
    check_planted_missing()


def test_skip_policy():
    check_skip_policy()


def test_changed_byte():
    check_changed_byte()


def test_done_skipped():
    check_done_skipped()


def test_executor_mismatch():
    check_executor_mismatch()


def test_gate_red():
    check_gate_red()


def test_builder_refuses_bad_plan():
    check_builder_refuses_bad_plan()


def test_scorer_classes():
    check_scorer_classes()


def test_start_class_rule():
    check_start_class_rule()
