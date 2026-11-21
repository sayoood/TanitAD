"""Stage 2 of the v8F-mini T2 re-perception harness on box 2 (AGENT G): the DECIDING rollouts of PREREG_T2_CRAWL sections 3 and 5, as an OVERLAY on the stage-1 package.

    python -I t2v8f_stage2.py all  --pkg PKG --kit KIT --hf HF --pull PULL --scenes2 D:/t2v8f/scenes2 [--executor base|hold] [--scene-policy refuse|skip] [--stub] [--inproc]
    python -I t2v8f_stage2.py plan --manifest2 MANIFEST2.json              (no box needed: the cell order, counts and the disk estimate)

Jobs, strictly in this order, one child process at a time (safe mode):
  1 preflight  stage 1's preflight (471 shipped files, CUDA, toolchain, DINOv3, kit) + the overlay (sha256 of every overlay file, bound to stage 1's MANIFEST.json) + every scene file of
               MANIFEST2 (41 copied scenes under --scenes2, plus 7c72937c in the kit) by size and sha256, with a named reason per failure + free disk against the plan's estimate
  2 gate       CPU, no render: for every scene and start the LOGGED plan is played back by the T2 executor and must reproduce the logged future to 1e-6 m (card gate T2g on each rig
               trajectory); the rig is 10 Hz, n_ticks equals the plan, f0 >= 18 and f0 + 60 < n_ticks. A failed gate stops the run before any render (card section 4 / outcome 5)
  3 rollouts   pass `deciding` (C0_s0 C0_s1 P10 ZERO BLIND x {T2, RR}, scene-major, in the contrast-priority order of CELL_PRIORITY) over all scenes FIRST, then pass `beside`
               (DAGOACC_s0, DAGOACC_s1): a crash leaves the deciding cells. A cell is its own process; DONE cells are skipped on relaunch; a FAILED cell never stops the others.
Executor: `--executor base` (default; the frozen card) or `hold` (t2v8f_exec_hold.py; only under a later dated amendment). The DONE text carries it; a relaunch under the other one refuses.
Markers in --pull: STAGE2.running (heartbeat), <job>.done|fail, STAGE2.done|fail, STAGE2_RESULT.json after every job. Cells: rollouts/<scene8>/objects/<T2|RR>/<arm>/{DONE|FAIL,records.jsonl,...}.
Records carry purpose `deciding candidate` (no --t2-plumbing). Stage-1 rollouts are never pooled with these.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import t2v8f_stage1 as S1  # noqa: E402  (the stage-1 driver: preflight, markers, Ctx, RENDER_FLAGS)

TIER = S1.TIER
DECIDING = ("C0_s0", "C0_s1", "P10", "ZERO", "BLIND")
BESIDE = ("DAGOACC_s0", "DAGOACC_s1")
# Within one scene the deciding cells run in the order the card's contrasts need them: X1 = C0(T2) - P10(T2), X2 = C0(T2) - C0(RR), both seeds, then BLIND(T2), then the controls.
CELL_PRIORITY = (("T2", "C0_s0"), ("T2", "P10"), ("RR", "C0_s0"), ("T2", "C0_s1"), ("RR", "C0_s1"), ("T2", "BLIND"), ("T2", "ZERO"), ("RR", "P10"), ("RR", "ZERO"), ("RR", "BLIND"))
BESIDE_ORDER = (("T2", "DAGOACC_s0"), ("T2", "DAGOACC_s1"), ("RR", "DAGOACC_s0"), ("RR", "DAGOACC_s1"))
PASSES = {"deciding": CELL_PRIORITY, "beside": BESIDE_ORDER}
GATE_TOL_M = 1e-6
CELL = f"{HERE}/t2v8f_stage2_cell.py"


def cell_order(scenes, passes=("deciding", "beside")):
    """The run order: pass by pass, scene-major inside a pass. -> [{pass, idx, scene, tc, arm}]."""
    out = []
    for p in passes:
        for sc in scenes:
            for tc, arm in PASSES[p]:
                out.append({"pass": p, "idx": sc["idx"], "scene": sc["scene"], "tc": tc, "arm": arm})
    return out


def cell_dir(pull, sid, tc, arm):
    return f"{pull}/rollouts/{sid[:8]}/objects/{tc}/{arm}"


class Ctx2(S1.Ctx):
    def __init__(self, a):
        super().__init__(a)
        self.scenes2 = a.scenes2.replace("\\", "/")
        self.man2_path = a.manifest2.replace("\\", "/") if a.manifest2 else f"{self.pkg}/MANIFEST2.json"
        self.man2 = json.load(open(self.man2_path, encoding="utf-8"))
        self.res.update(stage="stage2", purpose="STAGE 2: deciding rollouts (PREREG_T2_CRAWL sections 3 and 5)", executor=a.executor, hold_eps=a.hold_eps if a.executor == "hold" else None,
                        manifest2_sha256=S1.sha256_file(self.man2_path), scene_policy=a.scene_policy, n_scenes_manifest=len(self.man2["scenes"]))
        self.skipped = {}

    def save(self):
        S1.jdump(self.res, f"{self.pull}/STAGE2_RESULT.json")

    def scene_dir2(self, sc):
        return f"{self.kit}/scenes/{sc['scene']}" if sc["where"] == "kit" else f"{self.scenes2}/{sc['scene']}"

    def live_scenes(self):
        return [sc for sc in self.man2["scenes"] if sc["scene"] not in self.skipped]

    def heartbeat(self, text):
        with open(f"{self.pull}/STAGE2.running", "w", newline="\n") as f:
            f.write(f"pid={os.getpid()} start={self.res['started']} {text}\n")


# ============================================================================================================================ 1 preflight
def preflight2(ctx):
    base = S1.preflight(ctx)
    bad, info, m2 = list(base["failures"]), dict(base["info"]), ctx.man2
    info["manifest2"] = ctx.man2_path
    if m2.get("stage1_manifest_sha256") != ctx.res["manifest_sha256"]:
        bad.append(f"HASH_MISMATCH MANIFEST.json (the overlay was built against stage-1 manifest {str(m2.get('stage1_manifest_sha256'))[:12]}, this package is {ctx.res['manifest_sha256'][:12]})")
    n_ok = 0
    for e in m2["files"]:                                                                       # the overlay's own files
        p = f"{ctx.pkg}/{e['path']}"
        if not os.path.isfile(p):
            bad.append(f"MISSING {e['path']}")
        elif os.path.getsize(p) != e["size"]:
            bad.append(f"SIZE_MISMATCH {e['path']}: {os.path.getsize(p)} != {e['size']}")
        elif S1.sha256_file(p) != e["sha256"]:
            bad.append(f"HASH_MISMATCH {e['path']}")
        else:
            n_ok += 1
    info["overlay_files_ok"], info["overlay_files"] = n_ok, len(m2["files"])
    per_scene, n_f, t0 = {}, 0, time.time()
    for k, sc in enumerate(m2["scenes"]):
        why = []
        for f in sc["files"]:
            p = f"{ctx.scene_dir2(sc)}/{f['rel']}"
            tag = f"{'kit:scenes' if sc['where'] == 'kit' else 'scenes2'}/{sc['scene']}/{f['rel']}"
            if not os.path.isfile(p):
                why.append(f"MISSING {tag}")
            elif os.path.getsize(p) != f["bytes"]:
                why.append(f"SIZE_MISMATCH {tag}: {os.path.getsize(p)} != {f['bytes']}")
            elif S1.sha256_file(p) != f["sha256"]:
                why.append(f"HASH_MISMATCH {tag}")
            else:
                n_f += 1
        if why:
            per_scene[sc["scene"]] = why
        ctx.heartbeat(f"phase=preflight scene={k + 1}/{len(m2['scenes'])}")
    info["scene_files_ok"], info["scene_files"], info["scene_hash_seconds"] = n_f, sum(len(sc["files"]) for sc in m2["scenes"]), round(time.time() - t0, 1)
    if ctx.a.scene_policy == "refuse":
        for sid, why in per_scene.items():
            bad.extend(why)
    else:                                                                                       # skip: the scene is named, never silently dropped
        ctx.skipped = {sid: why for sid, why in per_scene.items()}
        info["scenes_skipped"] = ctx.skipped
        if len(ctx.skipped) == len(m2["scenes"]):
            bad.append("NO_SCENE_LEFT every scene failed its check")
    pull_free = shutil.disk_usage(ctx.pull).free / 2**30
    need = m2["footprint"]["pass_mib"]["deciding"] / 1024
    info["free_gib_pull_after_scenes"] = round(pull_free, 2)
    if pull_free - need < ctx.a.min_free_gb:
        bad.append(f"DISK_WILL_RUN_LOW pass deciding needs {need:.2f} GiB, free {pull_free:.2f} GiB, floor {ctx.a.min_free_gb}")
    return {"verdict": "REFUSED" if bad else "PASS", "failures": bad, "info": info}


# ============================================================================================================================ 2 gate (CPU, no render)
class ReplayPolicy:
    """The T2g control: returns the LOGGED plan at NOW, so the car follows the human (AGENT F's audit, restated)."""
    window = 17

    def __init__(self, gt_T, X):
        self.X, self.rows = X, X.logged_rows(gt_T)

    def set_t0(self, gt_T, f0, **k):
        r = self.rows[:f0 + 1]
        self.raw_off = (len(r) - 1) % 2
        self.track = r[self.raw_off:].copy()
        self.log = self.rows[self.raw_off:]

    def set_nav(self, **k):
        pass

    def set_history(self, h, n_new=None):
        import numpy as np
        self.track = np.vstack([self.track, np.asarray(h)[len(h) - n_new:]])

    def plan(self, frames, intr, v0, nav_cmd):
        return self.X.logged_plan(self.log, len(self.track) - 1, pose=self.track[-1]), {}


def load_rig(path, cam="camera_front_wide_120fov"):
    """`rig_trajectories.json` -> (gt_T [n,4,4], gt_ts_us [n]) at 10 Hz (t2v8f_lead.load_rig, restated: the lead module is not on box 2)."""
    import numpy as np
    with open(path, "r", encoding="utf-8") as f:
        e = json.load(f)["rig_trajectories"][0]
    key = f"{cam}@{e['sequence_id']}"
    Ts, ts = e["cameras_frame_T_rig_worlds"][key], e["cameras_frame_timestamps_us"][key]
    stride = int(round(1e5 / (ts[1][1] - ts[0][1])))
    idx = range(0, len(Ts), stride)
    return np.stack([np.array(Ts[i][1], np.float64) for i in idx]), np.array([ts[i][1] for i in idx], np.int64)


def selfgate_main(argv):
    """Child process of job 2. Writes GATE_SCENES.json (per scene and start: max |executed - logged| over the 30 knots, finite, start class) and prints the verdict."""
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest2", required=True)
    ap.add_argument("--scenes2", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="")
    a = ap.parse_args(argv)
    import t2v8f_exec as X
    if os.environ.get("T2V8F_TOOL"):
        X.TOOL = os.environ["T2V8F_TOOL"]
    tool = X.load_tool()
    m2 = json.load(open(a.manifest2, encoding="utf-8"))
    skip = {s for s in a.skip.split(",") if s}
    rows, bad = [], []
    for sc in m2["scenes"]:
        if sc["scene"] in skip:
            continue
        sd = f"{a.kit}/scenes/{sc['scene']}" if sc["where"] == "kit" else f"{a.scenes2}/{sc['scene']}"
        try:
            gt, ts = load_rig(f"{sd}/rig_trajectories.json")
            n = len(ts)
            if n != sc["n_ticks"]:
                bad.append(f"{sc['scene'][:8]}: rig has {n} ticks, the plan {sc['n_ticks']}")
            if abs(float(np.median(np.diff(ts))) - 1e5) > 50:
                bad.append(f"{sc['scene'][:8]}: the rig is not 10 Hz")
            for f0 in sc["starts"]:
                if f0 < 18 or f0 + 60 >= n:
                    bad.append(f"{sc['scene'][:8]} f0 {f0}: needs f0 >= 18 and f0 + 60 < {n}")
                    continue
                rec = X.run_t2_rollout(ReplayPolicy(gt, X), lambda T, t: None, gt, ts, f0, None, nav_cmd=0, condition="T2", tool=tool)
                d = float(np.abs(np.asarray(rec["cl"]) - np.asarray(rec["g"])).max())
                fin = bool(np.isfinite(np.asarray(rec["cl"])).all() and np.isfinite(np.asarray(rec["g"])).all())
                rows.append({"scene": sc["scene"], "f0": f0, "max_abs_diff_m": d, "finite": fin, "v0_mps": float(rec["v0"])})
                if not (fin and d <= GATE_TOL_M):
                    bad.append(f"{sc['scene'][:8]} f0 {f0}: logged-plan self-score {d:.3g} m (> {GATE_TOL_M})")
        except Exception as e:                                                                  # noqa: BLE001
            bad.append(f"{sc['scene'][:8]}: {e!r}"[:200])
    S1.jdump({"tol_m": GATE_TOL_M, "rows": rows, "failures": bad}, a.out)
    print(json.dumps({"n_starts": len(rows), "failures": len(bad)}))
    return 1 if bad else 0


def gate(ctx):
    out = f"{ctx.pull}/GATE_SCENES.json"
    rc = ctx.run_child([f"{HERE}/t2v8f_stage2.py", "selfgate", "--manifest2", ctx.man2_path, "--scenes2", ctx.scenes2, "--kit", ctx.kit, "--out", out, "--skip", ",".join(sorted(ctx.skipped))],
                       f"{ctx.pull}/gate.log")
    if not os.path.exists(out):
        return {"verdict": "CRASH", "error": f"the gate child wrote no result (rc {rc})"}
    d = json.load(open(out, encoding="utf-8"))
    return {"verdict": "PASS" if rc == 0 and not d["failures"] else "FAIL", "n_starts": len(d["rows"]), "failures": d["failures"][:20], "max_abs_diff_m": max([r["max_abs_diff_m"] for r in d["rows"]] or [0.0]),
            "rule": f"logged plan played back by the T2 executor reproduces the logged future to {GATE_TOL_M} m on every scene and start"}


# ============================================================================================================================ 3 rollouts
def cell_argv(ctx, sc, tc, arm, d):
    """The stage-1 `entry_args` for the same cell, without `--t2-plumbing` (these are the deciding rollouts)."""
    return ["--scene-dir", ctx.scene_dir2(sc), "--arm", arm, "--condition", "objects", "--starts", ",".join(str(f) for f in sc["starts"]), "--steps", "60", "--out", d, "--t2-cond", tc,
            "--t2-scene-idx", str(sc["idx"]), "--t2-scene-id", sc["scene"], "--t2-encoder", ctx.encoder, "--t2-device", ctx.device, "--t2-hub", f"{ctx.hf}/hub", "--t2-ckpt-root", f"{ctx.pkg}/ckpt",
            "--t2-blind-mean", f"{ctx.pkg}/refs/blind_mean_eval141.pt", *S1.RENDER_FLAGS]


def run_one_cell(ctx, sc, tc, arm, log):
    d = cell_dir(ctx.pull, sc["scene"], tc, arm)
    os.makedirs(d, exist_ok=True)
    ex = ctx.a.executor
    want = len(sc["starts"])
    if os.path.exists(f"{d}/DONE"):
        txt = open(f"{d}/DONE", encoding="utf-8").read()
        if f"executor={ex}" not in txt:
            raise RuntimeError(f"EXECUTOR_MISMATCH {d}/DONE says {txt.strip()!r}, this run is --executor {ex}")
        return "SKIPPED_DONE"
    if os.path.exists(f"{d}/FAIL"):
        os.replace(f"{d}/FAIL", f"{d}/FAIL.prev")
    for stale in ("records.jsonl",):                                                              # a half-written earlier attempt must not be appended to
        if os.path.exists(f"{d}/{stale}"):
            os.replace(f"{d}/{stale}", f"{d}/{stale}.prev")
    av = ["--executor", ex, "--hold-eps", str(ctx.a.hold_eps)] + cell_argv(ctx, sc, tc, arm, d)
    try:
        if ctx.a.inproc:
            import t2v8f_run as R
            import t2v8f_stage2_cell as C
            R.STATE["nav"] = {}
            try:
                C.run_cell(av)
                rc = 0
            except SystemExit as e:
                rc = 1 if e.code else 0
        else:
            rc = ctx.run_child([CELL] + av, log, timeout=ctx.a.cell_timeout)
    except subprocess.TimeoutExpired:
        rc = -999
    except Exception as e:                                                                        # noqa: BLE001
        rc = 1
        open(f"{d}/error.txt", "w", encoding="utf-8").write(repr(e))
    n = sum(1 for _ in open(f"{d}/records.jsonl", encoding="utf-8")) if os.path.exists(f"{d}/records.jsonl") else 0
    ok = rc == 0 and n == want
    with open(f"{d}/{'DONE' if ok else 'FAIL'}", "w", newline="\n") as f:
        f.write(f"rc={rc} records={n}/{want} executor={ex}\n")
    return "DONE" if ok else "FAIL"


def rollouts(ctx):
    m2, cells = ctx.man2, {}
    order = cell_order(ctx.live_scenes(), tuple(p for p in ctx.a.passes.split(",") if p))
    state = {"deciding": {"done": 0, "skipped_done": 0, "failed": []}, "beside": {"done": 0, "skipped_done": 0, "failed": []}}
    passes_skipped, started, halted = [], set(), set()
    for k, c in enumerate(order):
        p = c["pass"]
        sc = next(s for s in ctx.man2["scenes"] if s["scene"] == c["scene"])
        if p not in started:
            started.add(p)
            free = shutil.disk_usage(ctx.pull).free / 2**30
            need = m2["footprint"]["pass_mib"][p] / 1024
            if free - need < ctx.a.min_free_gb:                                                  # refuse to START a pass that would end below the floor; never die mid-pass
                halted.add(p)
                passes_skipped.append({"pass": p, "free_gib": round(free, 2), "needs_gib": round(need, 2), "floor_gib": ctx.a.min_free_gb,
                                       "shortfall_gib": round(ctx.a.min_free_gb + need - free, 2), "reason": "DISK_LOW"})
        if p in halted:
            continue
        if shutil.disk_usage(ctx.pull).free / 2**30 < ctx.a.min_free_gb:
            halted.add(p)
            passes_skipped.append({"pass": p, "reason": "DISK_FLOOR_REACHED_MID_PASS", "at_cell": k})
            continue
        log = f"{ctx.pull}/rollouts_{p}.log"
        res = run_one_cell(ctx, sc, c["tc"], c["arm"], log)
        key = f"{sc['scene'][:8]}|{c['tc']}|{c['arm']}"
        cells[key] = res
        if res == "DONE":
            state[p]["done"] += 1
        elif res == "SKIPPED_DONE":
            state[p]["skipped_done"] += 1
        else:
            state[p]["failed"].append(key)
        ctx.heartbeat(f"phase=rollouts cell={k + 1}/{len(order)} pass={p} last={key}:{res}")
    n_fail = sum(len(v["failed"]) for v in state.values())
    n_cells = len(order)
    complete = n_fail == 0 and not passes_skipped and not ctx.skipped and all(v == "DONE" or v == "SKIPPED_DONE" for v in cells.values()) and len(cells) == n_cells
    return {"verdict": "PASS" if complete else "INCOMPLETE", "n_cells_planned": n_cells, "state": state, "passes_skipped_disk": passes_skipped, "scenes_skipped": ctx.skipped,
            "first_cells": [f"{c['scene'][:8]}|{c['tc']}|{c['arm']}" for c in order[:6]], "executor": ctx.a.executor}


# ============================================================================================================================ driver
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "selfgate":
        return selfgate_main(argv[1:])
    if argv and argv[0] == "plan":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--manifest2", required=True)
        a = ap.parse_args(argv)
        m2 = json.load(open(a.manifest2, encoding="utf-8"))
        order = cell_order(m2["scenes"])
        print(json.dumps({"n_scenes": len(m2["scenes"]), "n_starts": sum(len(s["starts"]) for s in m2["scenes"]), "n_cells": len(order),
                          "per_pass": {p: sum(1 for c in order if c["pass"] == p) for p in PASSES}, "first": [f"{c['scene'][:8]}|{c['tc']}|{c['arm']}" for c in order[:12]],
                          "pass_switch_at": next(i for i, c in enumerate(order) if c["pass"] == "beside"), "footprint": m2["footprint"]}, indent=1))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all",))
    ap.add_argument("--pkg", default=os.path.dirname(HERE))
    ap.add_argument("--kit", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--scenes2", required=True)
    ap.add_argument("--manifest2", default=None)
    ap.add_argument("--executor", choices=("base", "hold"), default="base")
    ap.add_argument("--hold-eps", type=float, default=0.1)
    ap.add_argument("--passes", default="deciding,beside")
    ap.add_argument("--scene-policy", choices=("refuse", "skip"), default="refuse")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--inproc", action="store_true", help="rehearsal only: run the cells in this process")
    ap.add_argument("--matrix-inproc", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--cell-timeout", type=float, default=3600.0)
    a = ap.parse_args(argv)
    ctx = Ctx2(a)
    ctx.heartbeat("phase=start")
    for stale in ("STAGE2.done", "STAGE2.fail"):
        if os.path.exists(f"{ctx.pull}/{stale}"):
            os.replace(f"{ctx.pull}/{stale}", f"{ctx.pull}/{stale}.prev")
    crashed = None
    for name, fn in (("preflight", preflight2), ("gate", gate), ("rollouts", rollouts)):
        t0 = time.time()
        ctx.heartbeat(f"phase={name}")
        try:
            out = fn(ctx)
        except Exception as e:                                                                    # noqa: BLE001
            out = {"verdict": "CRASH", "error": repr(e)[:400], "trace": traceback.format_exc()[-1500:]}
        out["seconds"] = round(time.time() - t0, 1)
        ctx.res["steps"][name] = out
        ctx.res["gates"][name] = out["verdict"]
        ctx.save()
        ok = out["verdict"] not in ("REFUSED", "CRASH")
        S1.marker(ctx, name, ok, f"{out['verdict']} {json.dumps(out.get('failures') or out.get('error') or '')[:600]}")
        if out["verdict"] in ("REFUSED", "CRASH"):
            crashed = f"{name} {out['verdict']}: {json.dumps(out.get('failures') or out.get('error'))[:800]}"
            break
        if name == "gate" and out["verdict"] == "FAIL":
            break
    g = ctx.res["gates"]
    ctx.res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ctx.res["overall"] = ("INFRASTRUCTURE_FAILURE" if crashed else "GATE_FAILED" if g.get("gate") == "FAIL" else "COMPLETE" if g.get("rollouts") == "PASS" else "INCOMPLETE")
    ctx.save()
    os.remove(f"{ctx.pull}/STAGE2.running")
    S1.marker(ctx, "STAGE2", crashed is None, crashed or f"{ctx.res['overall']} " + json.dumps(ctx.res["gates"]))
    print(json.dumps({"overall": ctx.res["overall"], "gates": ctx.res["gates"]}))
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
