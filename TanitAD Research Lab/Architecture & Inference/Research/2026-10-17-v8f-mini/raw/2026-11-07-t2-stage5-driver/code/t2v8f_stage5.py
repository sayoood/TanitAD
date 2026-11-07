"""Stage 5 of the v8F-mini T2 re-perception harness on box 2 (AGENT R): ANY composite arm NAME_s0 / NAME_s1 over the 62 scenes and 121 starts of stages 2 to 4, as a NEW driver on top of stage 4's and stage 2's
(which it imports and never edits).

    python -I t2v8f_stage5.py all  --pkg PKG --kit KIT --hf HF --pull PULL5 --scenes2 D:/t2v8f/scenes2 --scenes3 C:/t2v8f/scenes3 --manifest5 PKG/MANIFEST5.json [--stub] [--inproc]
    python -I t2v8f_stage5.py plan --manifest5 MANIFEST5.json              (no box needed: the cell order, counts and the disk estimate)

The recipe is stage 4's, line for line: the cell order (62 scenes, scene-major, T2 x {NAME_s0, NAME_s1} deciding then RR x the same beside), the cell argv (`t2v8f_stage2.cell_argv`, unchanged), the
CPU gate (`t2v8f_stage4.gate4`, unchanged), the preflight's scene sha256 pass, the resume rule, the markers. What this driver adds:
  * the arm and its weights come from MANIFEST5.json (`arm5`: name, tag, ckpt_file {NAME_sK: file in pkg/ckpt}, ckpt_sha256). The two checkpoints ship in the overlay (or, for a rehearsal of the identity,
    point at files already in pkg/ckpt). The preflight sha256-checks them and REFUSES, named, on a missing or changed byte (MISSING_CKPT, HASH_MISMATCH_CKPT);
  * the cell entry is `t2v8f_stage5_cell.py`; the driver exports `T2V8F_S5_MANIFEST` so every child reads the same arm table;
  * what must already be on box 2 (`requires`, sha256-pinned, checked): stage 2's three files, stage 4's `t2v8f_c0c.py` and `t2v8f_stage4.py`, and the RI tool copy (stage 4's overlay);
  * the other stages' heartbeats must be absent (stage 2, 3 AND 4): box 2 runs ONE job at a time;
  * markers and result are STAGE5.* / STAGE5_RESULT.json in `--pull`; the result carries `stage: "stage5"`, `manifest_sha256` (the stage-1 package) and `manifest5_sha256` (this overlay's manifest),
    the keys the pinned read (`read_t2_dominance.py --arm NAME --arm-stage 5`) reads.
Records carry purpose `deciding candidate`, tier T2 re-perception (RR beside). P10(T2) is REUSED from stages 2 and 3 (the (scene idx, f0) keys pair by construction; the reader re-checks them).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
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
import t2v8f_stage4 as D4  # noqa: E402

STAGE = "STAGE5"
SEEDS = (0, 1)
CELL5 = f"{HERE}/t2v8f_stage5_cell.py"
ENV_MANIFEST = "T2V8F_S5_MANIFEST"
BUSY4 = "D:/t2v8f/pull4/STAGE4.running"
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")                        # no underscore: the read appends `_s0` / `_s1` itself, and `|` / `/` would break the cell paths
NAME_RESERVED = ("P10", "ZERO", "BLIND", "C0", "DAGOACC", "C0C")     # the floor, the stage-2/3 arms, and stage 4's own arm
ENCODER_FROZEN = "frozen_dinov3"
SHA_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_STAGE2 = ("code/t2v8f_stage2.py", "code/t2v8f_stage2_cell.py", "code/t2v8f_exec_hold.py")
REQUIRED_STAGE4 = ("code/t2v8f_c0c.py", "code/t2v8f_stage4.py", D4.RI_REL)


def arm_names(name):
    return tuple(f"{name}_s{s}" for s in SEEDS)


def passes5(name):
    a = arm_names(name)
    return {"deciding": tuple(("T2", x) for x in a), "beside": tuple(("RR", x) for x in a)}


def cell_order5(scenes, name, passes=("deciding", "beside")):
    """pass by pass, scene-major inside a pass -> [{pass, idx, scene, tc, arm}] (the shape of `cell_order4`; `cell_order5(s, 'C0C')` IS `cell_order4(s)`)."""
    P = passes5(name)
    return [{"pass": p, "idx": sc["idx"], "scene": sc["scene"], "tc": tc, "arm": arm} for p in passes for sc in scenes for tc, arm in P[p]]


@contextlib.contextmanager
def bindings5(name):
    """Point the stage-2 machinery at this arm's passes and the stage-5 cell entry, and put it back afterwards (the stage-2 module is shared by every test in one process)."""
    saved = (D.PASSES, D.CELL)
    D.PASSES, D.CELL = passes5(name), CELL5
    try:
        yield
    finally:
        D.PASSES, D.CELL = saved


def arm_of(m):
    return m.get("arm5") or {}


def check_arm5(m):
    """The arm table of a MANIFEST5. -> [reason]."""
    a, bad = arm_of(m), []
    name = a.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        return [f"BAD_ARM_NAME {name!r}: letters, digits and '-' only, starting with a letter (the read appends _s0 / _s1)"]
    if name.upper() in NAME_RESERVED:
        bad.append(f"BAD_ARM_NAME {name!r} is reserved {NAME_RESERVED} (the floor, the stage-2/3 arms, stage 4's C0C)")
    want = list(arm_names(name))
    if a.get("arms") != want:
        bad.append(f"BAD_ARM_ARMS {a.get('arms')!r}, expected {want}")
    kind = (a.get("encoder") or {}).get("kind")
    if kind != ENCODER_FROZEN:                                           # a LoRA arm's head weights read the MERGED encoder's tokens: run on the frozen encoder they are the wrong arm, silently
        bad.append(f"ENCODER_UNSUPPORTED {kind!r}: this driver builds the frozen DINOv3-L encoder only (arm5.encoder.kind == {ENCODER_FROZEN!r}); the merged-LoRA encoder of the LC arm is not built (HANDOFF_S5.md, section LC)")
    ck, sha = a.get("ckpt_file") or {}, a.get("ckpt_sha256") or {}
    if sorted(ck) != sorted(want) or sorted(sha) != sorted(want):
        bad.append(f"BAD_CKPT_MAP ckpt_file keys {sorted(ck)} / ckpt_sha256 keys {sorted(sha)}, expected {want}")
    else:
        for k in want:
            if not str(ck[k]).endswith(".pt") or "/" in str(ck[k]) or "\\" in str(ck[k]):
                bad.append(f"BAD_CKPT_FILE {k}: {ck[k]!r} must be a bare file name inside pkg/ckpt")
            if not SHA_RE.fullmatch(str(sha[k])):
                bad.append(f"BAD_CKPT_SHA {k}: {sha[k]!r}")
    if not bad:
        P = passes5(name)
        if (m.get("arms") or {}).get("deciding") != want or [list(x) for x in m.get("cell_priority", [])] != [list(x) for x in P["deciding"]] or [list(x) for x in m.get("beside_order", [])] != [list(x) for x in P["beside"]]:
            bad.append("BAD_ARMS the manifest's arms / cell order are not NAME_s0, NAME_s1 under T2 (deciding) then RR (beside)")
    return bad


def check_manifest5(m, n_launch=D4.N_LAUNCH, n_guard=D4.N_GUARD):
    """The cheap, named refusals of a stage-5 manifest. The scene rules are STAGE 4's (`check_manifest4` on a copy whose stage and arm table read as stage 4's C0C): one definition of a valid scene list."""
    if m.get("stage") != 5:
        return [f"NOT_A_STAGE5_MANIFEST stage={m.get('stage')!r} package={m.get('package')!r} (this driver reads a MANIFEST5.json built by make_stage5.py)"]
    bad = check_arm5(m)
    proj = dict(m, stage=4, arms={"deciding": list(C0C.ARMS_C0C)}, cell_priority=[list(x) for x in D4.DECIDING4], beside_order=[list(x) for x in D4.BESIDE4])
    bad += [x for x in D4.check_manifest4(proj, n_launch, n_guard) if not x.startswith("RI_TOOL_NOT_IN_OVERLAY")]       # the RI tool is stage 4's file: it is in `requires`, checked below
    req = m.get("requires") or {}
    for p in REQUIRED_STAGE2:
        if p not in req:
            bad.append(f"REQUIRES_INCOMPLETE {p} is not pinned in `requires`")
    for p in REQUIRED_STAGE4:
        if p not in req:
            bad.append(f"REQUIRES_INCOMPLETE {p} (stage 4's overlay) is not pinned in `requires`")
    if (req.get(D4.RI_REL) or {}).get("sha256") != C0C.RI_TOOL_SHA:
        bad.append(f"BAD_RI_TOOL_REQUIRES {D4.RI_REL} must be pinned at sha256 {C0C.RI_TOOL_SHA}")
    return bad


class Ctx5(D4.Ctx4):
    """The stage-4 context (three scene roots, the RI tool env) with stage-5 names. `self.arm` is the manifest's arm name (a placeholder when the manifest is not a MANIFEST5: the preflight then REFUSES it)."""

    def __init__(self, a):
        a.manifest4 = a.manifest5                                                  # the stage-4 context's own name for this path
        super().__init__(a)
        self.arm = arm_of(self.man2).get("name") or "NOARM"
        m5_sha = self.res.pop("manifest4_sha256", None)
        self.env[ENV_MANIFEST] = self.man2_path
        os.environ[ENV_MANIFEST] = self.man2_path
        ar = arm_of(self.man2)
        self.res.update(stage="stage5", purpose=f"STAGE 5: the composite arm {self.arm} (tag {ar.get('tag')!r}; checkpoint sha256 {ar.get('ckpt_sha256')}), T2 deciding then RR beside, 62 scenes",
                        manifest5_sha256=m5_sha, arm=self.arm, arm_tag=ar.get("tag"), arm_ckpt_sha256=ar.get("ckpt_sha256"), ckpt_root=f"{self.pkg}/ckpt")

    def save(self):
        S1.jdump(self.res, f"{self.pull}/STAGE5_RESULT.json")

    def heartbeat(self, text):
        with open(f"{self.pull}/STAGE5.running", "w", newline="\n") as f:
            f.write(f"pid={os.getpid()} start={self.res['started']} {text}\n")


def preflight5(ctx):
    """Cheap stage-5 refusals first (manifest identity, the files that must already be on box 2, the other stages' heartbeats, the scene roots, the RI tool, the two checkpoints), then stage 2's preflight."""
    with bindings5(ctx.arm):
        bad = check_manifest5(ctx.man2, ctx.a.expect_launch, ctx.a.expect_guard)
        for p, e in (ctx.man2.get("requires") or {}).items():
            f = f"{ctx.pkg}/{p}"
            if not os.path.isfile(f):
                bad.append(f"MISSING_REQUIRED_FILE {p}")
            elif S1.sha256_file(f) != e["sha256"]:
                bad.append(f"HASH_MISMATCH_REQUIRED_FILE {p}: box 2 holds another version than the one MANIFEST5 was built against")
        for label, bf in (("STAGE2", ctx.a.busy_file2), ("STAGE3", ctx.a.busy_file3), ("STAGE4", ctx.a.busy_file4)):
            if os.path.exists(bf):
                bad.append(f"{label}_STILL_RUNNING {bf} exists: box 2 runs ONE job at a time (a stale file after a crash: read it, then move it aside)")
        if not os.path.isdir(ctx.scenes3):
            bad.append(f"SCENES3_ABSENT {ctx.scenes3} does not exist: stage 3 has not run (or its scenes were removed); stage 5 ships no scene")
        if not os.path.isdir(ctx.scenes2):
            bad.append(f"SCENES2_ABSENT {ctx.scenes2} does not exist")
        if not os.path.isfile(ctx.ri_tool):
            bad.append(f"MISSING_RI_TOOL {D4.RI_REL}")
        elif os.path.getsize(ctx.ri_tool) != D4.RI_BYTES or S1.sha256_file(ctx.ri_tool) != C0C.RI_TOOL_SHA:
            bad.append(f"HASH_MISMATCH_RI_TOOL {D4.RI_REL}: not the pinned byte copy (sha256 {C0C.RI_TOOL_SHA[:12]}..., {D4.RI_BYTES} B)")
        ar = arm_of(ctx.man2)
        for arm, fn in sorted((ar.get("ckpt_file") or {}).items()):
            f = f"{ctx.pkg}/ckpt/{fn}"
            if not os.path.isfile(f):
                bad.append(f"MISSING_CKPT ckpt/{fn} (arm {arm})")
            elif S1.sha256_file(f) != (ar.get("ckpt_sha256") or {}).get(arm):
                bad.append(f"HASH_MISMATCH_CKPT ckpt/{fn} (arm {arm}): not the checkpoint MANIFEST5 was built with")
        if bad:
            return {"verdict": "REFUSED", "failures": bad, "info": {"scenes_root_2": ctx.scenes2, "scenes_root_3": ctx.scenes3, "pull_root": ctx.pull, "refused_before": "the stage-2 preflight (nothing was hashed)"}}
        out = D.preflight2(ctx)
        sizes = {r: round(shutil.disk_usage(p).free / 2**30, 2) for r, p in (("pull", ctx.pull),) if os.path.isdir(p)}
        out["info"].update(scenes_root_2=ctx.scenes2, scenes_root_3=ctx.scenes3, pull_root=ctx.pull, floor_checked_on=ctx.pull, free_gib=sizes, ri_tool_sha256=C0C.RI_TOOL_SHA,
                           arm=ctx.arm, arm_ckpt_sha256=ar.get("ckpt_sha256"))
        return out


def rollouts5(ctx):
    if ctx.a.inproc:                                                     # rehearsal only: the cells run in this process, so the arm is registered here (a child does it in the cell entry)
        import t2v8f_run as R
        import t2v8f_stage5_cell as C5
        C5.install5(R, *C5.arm5_from_manifest(ctx.man2_path))
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
    ap.add_argument("--manifest5", required=True)
    ap.add_argument("--executor", choices=("base", "hold"), default="base")
    ap.add_argument("--hold-eps", type=float, default=0.1)
    ap.add_argument("--passes", default="deciding,beside")
    ap.add_argument("--scene-policy", choices=("refuse", "skip"), default="refuse")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--inproc", action="store_true", help="rehearsal only: run the cells in this process")
    ap.add_argument("--matrix-inproc", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--cell-timeout", type=float, default=3600.0)
    ap.add_argument("--busy-file2", default=D4.BUSY2)
    ap.add_argument("--busy-file3", default=D4.BUSY3)
    ap.add_argument("--busy-file4", default=BUSY4)
    ap.add_argument("--expect-launch", type=int, default=D4.N_LAUNCH, help="the card has 42 launch scenes; only a rehearsal on a fixture passes another number")
    ap.add_argument("--expect-guard", type=int, default=D4.N_GUARD, help="the card has 20 guard scenes; only a rehearsal on a fixture passes another number")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plan":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--manifest5", required=True)
        a = ap.parse_args(argv)
        m = json.load(open(a.manifest5, encoding="utf-8"))
        name = arm_of(m).get("name") or "NOARM"
        order = cell_order5(m["scenes"], name)
        print(json.dumps({"arm": name, "n_scenes": len(m["scenes"]), "n_starts": sum(len(s["starts"]) for s in m["scenes"]), "n_cells": len(order),
                          "per_pass": {p: sum(1 for c in order if c["pass"] == p) for p in ("deciding", "beside")}, "first": [f"{c['scene'][:8]}|{c['tc']}|{c['arm']}" for c in order[:8]],
                          "pass_switch_at": next(i for i, c in enumerate(order) if c["pass"] == "beside"), "footprint": m["footprint"], "refusals": check_manifest5(m)}, indent=1))
        return 0
    a = build_parser().parse_args(argv)
    ctx = Ctx5(a)
    ctx.heartbeat("phase=start")
    for stale in (f"{STAGE}.done", f"{STAGE}.fail"):
        if os.path.exists(f"{ctx.pull}/{stale}"):
            os.replace(f"{ctx.pull}/{stale}", f"{ctx.pull}/{stale}.prev")
    crashed = None
    with bindings5(ctx.arm):
        for name, fn in (("preflight", preflight5), ("gate", D4.gate4), ("rollouts", rollouts5)):
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
