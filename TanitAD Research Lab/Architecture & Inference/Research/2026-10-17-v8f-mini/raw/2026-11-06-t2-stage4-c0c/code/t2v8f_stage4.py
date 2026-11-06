"""Stage 4 of the v8F-mini T2 re-perception harness on box 2 (AGENT N): the composite arm C0C_s0 / C0C_s1 over the 62 scenes and 121 starts of stages 2 and 3, as a NEW driver on top of the
stage-2 driver (which it imports and never edits).

    python -I t2v8f_stage4.py all  --pkg PKG --kit KIT --hf HF --pull PULL4 --scenes2 D:/t2v8f/scenes2 --scenes3 C:/t2v8f/scenes3 --manifest4 PKG/MANIFEST4.json [--stub] [--inproc]
    python -I t2v8f_stage4.py plan --manifest4 MANIFEST4.json              (no box needed: the cell order, counts and the disk estimate)

Reused byte for byte from `t2v8f_stage2.py` (sha256 33b65b19..., the file stage 2 ran on box 2): the preflight (stage 1's, the overlay's files, `check_ref_frames`, the disk floor, the scene sha256
pass over ALL 62 scenes), the cell process and its DONE / FAIL markers, the resume rule, the CPU gate (run once per scene root). What is new here:
  * the arm: cells run `t2v8f_stage4_cell.py`, which registers C0C_s0 / C0C_s1 (`t2v8f_c0c.install`) and then runs the stage-2 cell. Passes: `deciding` = T2 x {C0C_s0, C0C_s1} over every scene, then
    `beside` = RR x {C0C_s0, C0C_s1} (124 + 124 cells). `D.PASSES` / `D.CELL` are rebound ONLY inside `bindings()` (restored on exit), so importing this module changes nothing else;
  * three scene roots: the kit (7c72937c), `--scenes2` (41 scenes) and `--scenes3` (20 guard scenes), by the manifest's `where`. NO scene is copied: the preflight sha256-checks them against the files
    recorded in MANIFEST2 / MANIFEST3 (carried verbatim in MANIFEST4) and REFUSES, named, if `--scenes3` is absent;
  * the RI tool (76,704 B, sha256 0417c838...) ships as a byte copy at `pkg/refs/tool/v8f_t1_eval_ri.py`; the preflight pins it and the driver points `T2V8F_RI_TOOL` at it (children inherit it);
  * SAFE MODE across stages: the stage-2 and stage-3 heartbeats must be absent (box 2 runs ONE job at a time);
  * markers and result are STAGE4.* / STAGE4_RESULT.json in `--pull` (`manifest4_sha256` = the sha256 of the MANIFEST4.json it ran).
Records carry purpose `deciding candidate`, tier T2 re-perception (RR beside). P10(T2) is REUSED from stages 2 and 3: the (scene idx, f0) keys pair by construction and the reader re-checks them.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import t2v8f_c0c as C0C  # noqa: E402
import t2v8f_stage1 as S1  # noqa: E402
import t2v8f_stage2 as D  # noqa: E402

STAGE = "STAGE4"
DECIDING4 = (("T2", "C0C_s0"), ("T2", "C0C_s1"))
BESIDE4 = (("RR", "C0C_s0"), ("RR", "C0C_s1"))
PASSES4 = {"deciding": DECIDING4, "beside": BESIDE4}
CELL4 = f"{HERE}/t2v8f_stage4_cell.py"
IDX3_0 = 42
N_LAUNCH, N_GUARD = 42, 20
RI_REL, RI_BYTES = "refs/tool/v8f_t1_eval_ri.py", 76704
BUSY2, BUSY3 = "D:/t2v8f/pull2/STAGE2.running", "D:/t2v8f/pull3/STAGE3.running"
F_MIN, TAIL = 18, 61


def cell_order4(scenes, passes=("deciding", "beside")):
    """pass by pass, scene-major inside a pass -> [{pass, idx, scene, tc, arm}] (the same shape as `t2v8f_stage2.cell_order`)."""
    return [{"pass": p, "idx": sc["idx"], "scene": sc["scene"], "tc": tc, "arm": arm} for p in passes for sc in scenes for tc, arm in PASSES4[p]]


@contextlib.contextmanager
def bindings():
    """Point the stage-2 machinery at the stage-4 passes and cell entry, and put it back afterwards (the stage-2 module is shared by every test in one process)."""
    saved = (D.PASSES, D.CELL)
    D.PASSES, D.CELL = PASSES4, CELL4
    try:
        yield
    finally:
        D.PASSES, D.CELL = saved


class Ctx4(D.Ctx2):
    """The stage-2 context with stage-4 names and THREE scene roots (kit / scenes2 / scenes3, by the manifest's `where`)."""

    def __init__(self, a):
        a.manifest2 = a.manifest4                                                  # the stage-2 context's own name for this path
        super().__init__(a)
        self.scenes3 = a.scenes3.replace("\\", "/")
        m4_sha = self.res.pop("manifest2_sha256", None)
        self.ri_tool = f"{self.pkg}/{RI_REL}"
        self.env["T2V8F_RI_TOOL"] = self.ri_tool
        os.environ["T2V8F_RI_TOOL"] = self.ri_tool
        self.res.update(stage="stage4", purpose="STAGE 4: the composite arm C0C (C0 + sg72 + kbox), T2 deciding then RR beside, 62 scenes (PREREG_T2_CRAWL section 6, outcome 1)",
                        manifest4_sha256=m4_sha, scenes_root_kit=f"{self.kit}/scenes", scenes_root_2=self.scenes2, scenes_root_3=self.scenes3, pull_root=self.pull, ri_tool=self.ri_tool)

    def save(self):
        S1.jdump(self.res, f"{self.pull}/STAGE4_RESULT.json")

    def scene_dir2(self, sc):
        w = sc["where"]
        return f"{self.kit}/scenes/{sc['scene']}" if w == "kit" else f"{self.scenes3 if w == 'scenes3' else self.scenes2}/{sc['scene']}"

    def heartbeat(self, text):
        with open(f"{self.pull}/STAGE4.running", "w", newline="\n") as f:
            f.write(f"pid={os.getpid()} start={self.res['started']} {text}\n")


def check_manifest4(m, n_launch=N_LAUNCH, n_guard=N_GUARD):
    """The cheap, named refusals of a stage-4 manifest. -> [reason]."""
    bad = []
    if m.get("stage") != 4:
        return [f"NOT_A_STAGE4_MANIFEST stage={m.get('stage')!r} package={m.get('package')!r} (this driver reads a MANIFEST4.json built by make_stage4.py)"]
    if not m.get("requires"):
        bad.append("NO_REQUIRES this MANIFEST4 does not pin the stage-2 files it depends on")
    ri = m.get("ri_tool") or {}
    if ri.get("sha256") != C0C.RI_TOOL_SHA or ri.get("path") != RI_REL or ri.get("bytes") != RI_BYTES:
        bad.append(f"BAD_RI_TOOL_PIN {ri!r}: need {RI_REL} {RI_BYTES} B sha256 {C0C.RI_TOOL_SHA}")
    if not any(e.get("path") == RI_REL and e.get("sha256") == C0C.RI_TOOL_SHA for e in m.get("files", [])):
        bad.append(f"RI_TOOL_NOT_IN_OVERLAY {RI_REL} is not an overlay file with the pinned sha256 (the preflight could not verify it)")
    if (m.get("arms") or {}).get("deciding") != list(C0C.ARMS_C0C) or [list(x) for x in m.get("cell_priority", [])] != [list(x) for x in DECIDING4] or [list(x) for x in m.get("beside_order", [])] != [list(x) for x in BESIDE4]:
        bad.append("BAD_ARMS the manifest's arms / cell order are not C0C_s0, C0C_s1 under T2 (deciding) then RR (beside)")
    sc = m.get("scenes", [])
    if len(sc) != n_launch + n_guard:
        bad.append(f"BAD_SCENE_COUNT {len(sc)} != {n_launch + n_guard}")
    ids = [s.get("scene") for s in sc]
    if len(set(ids)) != len(ids) or len({i[:8] for i in ids if i}) != len(ids):
        bad.append("DUPLICATE_SCENE a scene id (or its 8-character prefix, the cell directory name) occurs twice")
    for k, s in enumerate(sc):
        tag, g = f"{str(s.get('scene', '?'))[:8]}", k >= n_launch
        want_idx = IDX3_0 + (k - n_launch) if g else k
        if s.get("idx") != want_idx:
            bad.append(f"BAD_IDX {tag} idx={s.get('idx')!r}, expected {want_idx} (stage-2 scenes 0..{n_launch - 1}, stage-3 scenes {IDX3_0}.., so P10's keys pair)")
        if (s.get("where") == "scenes3") != g or s.get("where") not in ("kit", "scenes2", "scenes3"):
            bad.append(f"BAD_WHERE {tag} where={s.get('where')!r} for position {k}")
        n = int(s.get("n_ticks", 0))
        st = s.get("starts") or []
        if not st or any((f < F_MIN or f + 60 >= n) for f in st) or len(set(st)) != len(st):
            bad.append(f"BAD_STARTS {tag} starts={st} n_ticks={n}: need distinct f0 in [{F_MIN}, {n - TAIL}]")
        if not (s.get("ref_frames") or {}).get("sha256"):
            bad.append(f"NO_REF_FRAMES_IN_MANIFEST4 {tag}")
    return bad


def preflight4(ctx):
    """Cheap stage-4 refusals first (manifest identity, the stage-2 files, the other stages' heartbeats, scenes3, the RI tool, the checkpoints), then the stage-2 preflight on this context."""
    with bindings():
        bad = check_manifest4(ctx.man2, ctx.a.expect_launch, ctx.a.expect_guard)
        for p, e in (ctx.man2.get("requires") or {}).items():
            f = f"{ctx.pkg}/{p}"
            if not os.path.isfile(f):
                bad.append(f"MISSING_STAGE2_FILE {p}")
            elif S1.sha256_file(f) != e["sha256"]:
                bad.append(f"HASH_MISMATCH_STAGE2_FILE {p}: box 2 holds another version than the one MANIFEST4 was built against")
        for label, bf in (("STAGE2", ctx.a.busy_file2), ("STAGE3", ctx.a.busy_file3)):
            if os.path.exists(bf):
                bad.append(f"{label}_STILL_RUNNING {bf} exists: box 2 runs ONE job at a time (a stale file after a crash: read it, then move it aside)")
        if not os.path.isdir(ctx.scenes3):
            bad.append(f"SCENES3_ABSENT {ctx.scenes3} does not exist: stage 3 has not run (or its scenes were removed); stage 4 ships no scene")
        if not os.path.isdir(ctx.scenes2):
            bad.append(f"SCENES2_ABSENT {ctx.scenes2} does not exist")
        if not os.path.isfile(ctx.ri_tool):
            bad.append(f"MISSING_RI_TOOL {RI_REL}")
        elif os.path.getsize(ctx.ri_tool) != RI_BYTES or S1.sha256_file(ctx.ri_tool) != C0C.RI_TOOL_SHA:
            bad.append(f"HASH_MISMATCH_RI_TOOL {RI_REL}: not the pinned byte copy (sha256 {C0C.RI_TOOL_SHA[:12]}..., {RI_BYTES} B)")
        for arm, fn in C0C.CKPT_FILE_C0C.items():
            if not os.path.isfile(f"{ctx.pkg}/ckpt/{fn}"):
                bad.append(f"MISSING_CKPT ckpt/{fn} (arm {arm} runs C0's checkpoint)")
        if bad:
            return {"verdict": "REFUSED", "failures": bad, "info": {"scenes_root_2": ctx.scenes2, "scenes_root_3": ctx.scenes3, "pull_root": ctx.pull, "refused_before": "the stage-2 preflight (nothing was hashed)"}}
        out = D.preflight2(ctx)
        sizes = {r: round(shutil.disk_usage(p).free / 2**30, 2) for r, p in (("pull", ctx.pull),) if os.path.isdir(p)}
        out["info"].update(scenes_root_2=ctx.scenes2, scenes_root_3=ctx.scenes3, pull_root=ctx.pull, floor_checked_on=ctx.pull, free_gib=sizes, ri_tool_sha256=C0C.RI_TOOL_SHA)
        return out


def gate4(ctx):
    """The stage-2 CPU gate (logged plan played back must reproduce the logged future to 1e-6 m), run once per scene root with the stage-2 child's own `--skip`, then merged into GATE_SCENES.json."""
    sc = ctx.man2["scenes"]
    groups = {"a": ([s["scene"] for s in sc if s["where"] == "scenes3"], ctx.scenes2), "b": ([s["scene"] for s in sc if s["where"] != "scenes3"], ctx.scenes3)}
    rows, fails, rc_all = [], [], 0
    for g, (skip, root) in groups.items():
        out = f"{ctx.pull}/GATE_SCENES_{g}.json"
        rc = ctx.run_child([f"{D.HERE}/t2v8f_stage2.py", "selfgate", "--manifest2", ctx.man2_path, "--scenes2", root, "--kit", ctx.kit, "--out", out, "--skip", ",".join(sorted(skip))], f"{ctx.pull}/gate.log")
        if not os.path.exists(out):
            return {"verdict": "CRASH", "error": f"the gate child {g} wrote no result (rc {rc})"}
        d = json.load(open(out, encoding="utf-8"))
        rows += d["rows"]
        fails += d["failures"]
        rc_all = rc_all or rc
    S1.jdump({"tol_m": D.GATE_TOL_M, "rows": rows, "failures": fails}, f"{ctx.pull}/GATE_SCENES.json")
    return {"verdict": "PASS" if rc_all == 0 and not fails else "FAIL", "n_starts": len(rows), "failures": fails[:20], "max_abs_diff_m": max([r["max_abs_diff_m"] for r in rows] or [0.0]),
            "rule": f"logged plan played back by the T2 executor reproduces the logged future to {D.GATE_TOL_M} m on every scene and start (two scene roots)"}


def rollouts4(ctx):
    if ctx.a.inproc:                                                     # rehearsal only: the cells run in this process, so the composite arms are registered here (a child does it in the cell entry)
        import t2v8f_run as R
        C0C.install(R)
    return D.rollouts(ctx)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all",))
    ap.add_argument("--pkg", default=os.path.dirname(HERE))
    ap.add_argument("--kit", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--scenes2", required=True)
    ap.add_argument("--scenes3", required=True)
    ap.add_argument("--manifest4", required=True)
    ap.add_argument("--executor", choices=("base", "hold"), default="base")
    ap.add_argument("--hold-eps", type=float, default=0.1)
    ap.add_argument("--passes", default="deciding,beside")
    ap.add_argument("--scene-policy", choices=("refuse", "skip"), default="refuse")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--inproc", action="store_true", help="rehearsal only: run the cells in this process")
    ap.add_argument("--matrix-inproc", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--cell-timeout", type=float, default=3600.0)
    ap.add_argument("--busy-file2", default=BUSY2)
    ap.add_argument("--busy-file3", default=BUSY3)
    ap.add_argument("--expect-launch", type=int, default=N_LAUNCH, help="the card has 42 launch scenes; only a rehearsal on a fixture passes another number")
    ap.add_argument("--expect-guard", type=int, default=N_GUARD, help="the card has 20 guard scenes; only a rehearsal on a fixture passes another number")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plan":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--manifest4", required=True)
        a = ap.parse_args(argv)
        m = json.load(open(a.manifest4, encoding="utf-8"))
        order = cell_order4(m["scenes"])
        print(json.dumps({"n_scenes": len(m["scenes"]), "n_starts": sum(len(s["starts"]) for s in m["scenes"]), "n_cells": len(order),
                          "per_pass": {p: sum(1 for c in order if c["pass"] == p) for p in PASSES4}, "first": [f"{c['scene'][:8]}|{c['tc']}|{c['arm']}" for c in order[:8]],
                          "pass_switch_at": next(i for i, c in enumerate(order) if c["pass"] == "beside"), "footprint": m["footprint"], "refusals": check_manifest4(m)}, indent=1))
        return 0
    a = build_parser().parse_args(argv)
    ctx = Ctx4(a)
    ctx.heartbeat("phase=start")
    for stale in (f"{STAGE}.done", f"{STAGE}.fail"):
        if os.path.exists(f"{ctx.pull}/{stale}"):
            os.replace(f"{ctx.pull}/{stale}", f"{ctx.pull}/{stale}.prev")
    crashed = None
    with bindings():
        for name, fn in (("preflight", preflight4), ("gate", gate4), ("rollouts", rollouts4)):
            t0 = time.time()
            ctx.heartbeat(f"phase={name}")
            try:
                out = fn(ctx)
            except Exception as e:                                                                # noqa: BLE001
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
    os.remove(f"{ctx.pull}/{STAGE}.running")
    S1.marker(ctx, STAGE, crashed is None, crashed or f"{ctx.res['overall']} " + json.dumps(ctx.res["gates"]))
    print(json.dumps({"overall": ctx.res["overall"], "gates": ctx.res["gates"]}))
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
