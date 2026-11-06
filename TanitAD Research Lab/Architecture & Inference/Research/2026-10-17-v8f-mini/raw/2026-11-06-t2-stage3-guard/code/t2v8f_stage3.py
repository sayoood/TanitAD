"""Stage 3 of the v8F-mini T2 re-perception harness on box 2 (AGENT K): the 20 GUARD scenes of PREREG_T2_CRAWL section 3, as a NEW driver on top of the stage-2 driver (which it imports and never edits).

    python -I t2v8f_stage3.py all  --pkg PKG --kit KIT --hf HF --pull PULL3 --scenes3 C:/t2v8f/scenes3 --manifest3 PKG/MANIFEST3.json [--executor base] [--stub] [--inproc]
    python -I t2v8f_stage3.py plan --manifest3 MANIFEST3.json              (no box needed: the cell order, counts and the disk estimate)

What is reused, byte for byte, from `t2v8f_stage2.py` (sha256 33b65b19..., the file stage 2 is running on box 2): the preflight (stage 1's, the overlay's files, `check_ref_frames`, the disk
floor, the scene sha256 pass), the CPU gate, the cell order (`CELL_PRIORITY`, then `BESIDE_ORDER`), the cell argv, the cell process and its DONE / FAIL markers, the resume rule. What is new here:
  * the scenes live in their OWN root (`--scenes3`, box 2 `C:\\t2v8f\\scenes3`), on another drive than the pull (`--pull`, box 2 `D:`): scene files are read under `--scenes3`, and the free-space floor
    is checked on the PULL drive (`shutil.disk_usage(ctx.pull)`, as stage 2 does), never on the scenes drive;
  * the manifest must be a STAGE-3 manifest: `stage == 3`, `scene_root == "scenes3"`, 20 guard scenes (`slot` guard_*), no kit scene, `idx` 42..61, every start inside [18, n_ticks - 61]; a stage-2
    manifest is refused by name;
  * SAFE MODE across stages: the stage-2 heartbeat (`--busy-file`, default D:/t2v8f/pull2/STAGE2.running) must be absent, or box 2 would run two jobs;
  * markers and result are STAGE3.* / STAGE3_RESULT.json in `--pull`, so stage 3 never touches stage 2's pull.
Records carry purpose `deciding candidate` (no --t2-plumbing), tier T2 re-perception. The card's M1 contrast reads the LAUNCH scenes only; the guard scenes enter the four families (section 3).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import t2v8f_stage1 as S1  # noqa: E402
import t2v8f_stage2 as D  # noqa: E402

STAGE = "STAGE3"
IDX0, N_GUARD = 42, 20
BUSY_FILE = "D:/t2v8f/pull2/STAGE2.running"
F_MIN, TAIL = 18, 61


class Ctx3(D.Ctx2):
    """The stage-2 context with stage-3 names; every scene is under `--scenes3` (a stage-3 manifest has no kit scene)."""

    def __init__(self, a):
        a.scenes2, a.manifest2 = a.scenes3, a.manifest3                       # the stage-2 context's own names for the same two paths
        super().__init__(a)
        self.res.update(stage="stage3", purpose="STAGE 3: the 20 guard scenes, deciding rollouts (PREREG_T2_CRAWL section 3)", scenes_root=self.scenes2, pull_root=self.pull,
                        manifest3_sha256=self.res.get("manifest2_sha256"))

    def save(self):
        S1.jdump(self.res, f"{self.pull}/STAGE3_RESULT.json")

    def scene_dir2(self, sc):
        return f"{self.scenes2}/{sc['scene']}"

    def heartbeat(self, text):
        with open(f"{self.pull}/STAGE3.running", "w", newline="\n") as f:
            f.write(f"pid={os.getpid()} start={self.res['started']} {text}\n")


def check_manifest3(m, expect=N_GUARD):
    """The cheap, named refusals of a stage-3 manifest (`expect` = the scene count; 20, the card's, unless a rehearsal says otherwise). -> [reason]."""
    bad = []
    if m.get("stage") != 3:
        bad.append(f"NOT_A_STAGE3_MANIFEST stage={m.get('stage')!r} package={m.get('package')!r} (this driver reads a MANIFEST3.json built by make_stage3.py)")
        return bad
    if not m.get("requires"):
        bad.append("NO_REQUIRES this MANIFEST3 does not pin the stage-2 files it depends on")
    if m.get("scene_root") != "scenes3":
        bad.append(f"BAD_SCENE_ROOT {m.get('scene_root')!r} != 'scenes3'")
    sc = m.get("scenes", [])
    if len(sc) != expect:
        bad.append(f"BAD_SCENE_COUNT {len(sc)} != {expect}")
    seen = set()
    for k, s in enumerate(sc):
        tag = f"{s.get('scene', '?')[:8]}"
        if not str(s.get("slot", "")).startswith("guard"):
            bad.append(f"NOT_A_GUARD_SCENE {tag} slot={s.get('slot')!r}")
        if s.get("where") != "scenes3":
            bad.append(f"BAD_WHERE {tag} where={s.get('where')!r} (a stage-3 scene is never in the kit)")
        if s.get("idx") != IDX0 + k or s.get("idx") in seen:
            bad.append(f"BAD_IDX {tag} idx={s.get('idx')!r}, expected {IDX0 + k} (stage-2 scenes are 0..41, so records pool without a clash)")
        seen.add(s.get("idx"))
        n = int(s.get("n_ticks", 0))
        st = s.get("starts") or []
        if not st or any((f < F_MIN or f + 60 >= n) for f in st) or len(set(st)) != len(st):
            bad.append(f"BAD_STARTS {tag} starts={st} n_ticks={n}: need distinct f0 in [{F_MIN}, {n - TAIL}]")
    return bad


def preflight3(ctx):
    """Cheap stage-3 refusals first (manifest identity, the stage-2 heartbeat), then the stage-2 preflight on this context. -> the same dict, with the two roots and both drives' free space in `info`."""
    bad = check_manifest3(ctx.man2, ctx.a.expect_scenes)
    for p, e in (ctx.man2.get("requires") or {}).items():                    # the stage-2 files this driver imports and spawns: the driver stage 2 itself runs (sha256 33b65b19...) and its cell files
        f = f"{ctx.pkg}/{p}"
        if not os.path.isfile(f):
            bad.append(f"MISSING_STAGE2_FILE {p}")
        elif S1.sha256_file(f) != e["sha256"]:
            bad.append(f"HASH_MISMATCH_STAGE2_FILE {p}: box 2 holds another version than the one MANIFEST3 was built against")
    if os.path.exists(ctx.a.busy_file):
        bad.append(f"STAGE2_STILL_RUNNING {ctx.a.busy_file} exists: box 2 runs ONE job at a time (a stale file after a crash: read it, then move it aside)")
    if bad:
        return {"verdict": "REFUSED", "failures": bad, "info": {"scenes_root": ctx.scenes2, "pull_root": ctx.pull, "refused_before": "the stage-2 preflight (nothing was hashed)"}}
    out = D.preflight2(ctx)
    out["info"].update(scenes_root=ctx.scenes2, pull_root=ctx.pull, floor_checked_on=ctx.pull, free_gib_scenes_drive=round(shutil.disk_usage(ctx.scenes2).free / 2**30, 2) if os.path.isdir(ctx.scenes2) else None)
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all",))
    ap.add_argument("--pkg", default=os.path.dirname(HERE))
    ap.add_argument("--kit", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--scenes3", required=True)
    ap.add_argument("--manifest3", required=True)
    ap.add_argument("--executor", choices=("base", "hold"), default="base")
    ap.add_argument("--hold-eps", type=float, default=0.1)
    ap.add_argument("--passes", default="deciding,beside")
    ap.add_argument("--scene-policy", choices=("refuse", "skip"), default="refuse")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--inproc", action="store_true", help="rehearsal only: run the cells in this process")
    ap.add_argument("--matrix-inproc", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--cell-timeout", type=float, default=3600.0)
    ap.add_argument("--busy-file", default=BUSY_FILE)
    ap.add_argument("--expect-scenes", type=int, default=N_GUARD, help="the card has 20 guard scenes; only a rehearsal on a fixture passes another number")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plan":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--manifest3", required=True)
        a = ap.parse_args(argv)
        m = json.load(open(a.manifest3, encoding="utf-8"))
        order = D.cell_order(m["scenes"])
        print(json.dumps({"n_scenes": len(m["scenes"]), "n_starts": sum(len(s["starts"]) for s in m["scenes"]), "n_cells": len(order),
                          "per_pass": {p: sum(1 for c in order if c["pass"] == p) for p in D.PASSES}, "first": [f"{c['scene'][:8]}|{c['tc']}|{c['arm']}" for c in order[:12]],
                          "pass_switch_at": next(i for i, c in enumerate(order) if c["pass"] == "beside"), "footprint": m["footprint"], "refusals": check_manifest3(m)}, indent=1))
        return 0
    a = build_parser().parse_args(argv)
    ctx = Ctx3(a)
    ctx.heartbeat("phase=start")
    for stale in (f"{STAGE}.done", f"{STAGE}.fail"):
        if os.path.exists(f"{ctx.pull}/{stale}"):
            os.replace(f"{ctx.pull}/{stale}", f"{ctx.pull}/{stale}.prev")
    crashed = None
    for name, fn in (("preflight", preflight3), ("gate", D.gate), ("rollouts", D.rollouts)):
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
    os.remove(f"{ctx.pull}/{STAGE}.running")
    S1.marker(ctx, STAGE, crashed is None, crashed or f"{ctx.res['overall']} " + json.dumps(ctx.res["gates"]))
    print(json.dumps({"overall": ctx.res["overall"], "gates": ctx.res["gates"]}))
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
