"""Run the whole T2-A grid on box 2:  2 scenes x { refcv7-50,400 (x3 variants), the P10 floor (held + live), the ha0_ext echo
(held + live), the hold-speed zero control }  -> per-arm JSON stamped T2, paired files, one summary table.

    python run_on_box2.py --kit <kit dir> --out <results dir> [--scenes 00040136,7c72937c] [--arms tag,tag,...]
                          [--with-control] [--steps 50] [--starts 0,17,34,51,68,85,102,119,136] [--dry] [--no-resume]

It is RESUMABLE (an arm whose rollouts JSON exists is skipped) and ORDERED BY VALUE: the cheap floors first -- they validate
the render path end to end -- then the model arms. Every sub-run is `t2a_run.py` -> `closedloop_drive.main()`, i.e. the SAME code
that produced the banked Thor numbers, with the SAME render flags (`--cull-scale-quantile 0.95 --sky-gain 0.3`, rolling shutter
off, `empty` condition, 9 starts x 50 ticks).

Run `preflight_box2.py` first. Nothing here trains, downloads or writes outside --out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENES = {"00040136": "00040136-e651-4abd-991d-0655ccda9430", "7c72937c": "7c72937c-c620-4776-9555-d57222c0081f"}
STARTS = "0,17,34,51,68,85,102,119,136"            # run_panel_hq.sh's starts (9 clusters)
RENDER = ["--cull-scale-quantile", "0.95", "--sky-gain", "0.3"]


def arm_table(kit: Path, with_control: bool):
    """-> [(tag, --arm, extra argv[, env override])]"""
    ck, cfg, p10 = str(kit / "ckpt/ckpt_50400.pt"), str(kit / "ckpt/config.json"), str(kit / "p10/p10_S2.pt")
    A = [
        # (tag, --arm, extra argv)                                   what it is
        ("cl_ha0", "cl_ha0", []),                                    # zero control: steer & accel must read EXACTLY 0.0
        ("cl_ha0_ext", "cl_ha0_ext", []),                            # the ha0_ext ECHO, state held from t0 (harness arm)
        ("cl_p10_t0", "cl_p10_t0", ["--t2a-p10", p10]),              # P10 held from t0 (like-for-like with the line above)
        ("cl_p10", "cl_p10", ["--t2a-p10", p10]),                    # P10 as a per-tick driver
        ("cl_ha0_ext_live", "cl_ha0_ext_live", []),                  # the echo as a per-tick driver
        ("refcv7_vmax_off_s0", "refcv7", ["--ckpt", ck, "--t2a-config", cfg, "--t2a-vmax-mode", "withheld", "--t2a-infer-seed", "0"]),
        ("refcv7_vmax_off_s1", "refcv7", ["--ckpt", ck, "--t2a-config", cfg, "--t2a-vmax-mode", "withheld", "--t2a-infer-seed", "1"]),
        # The ceiling arm on the LAUNCH tree is a CONSISTENCY CONTROL, not a lever: SPEC_REFCV7 sec. 26.1 (INHERITED) says the ceiling never
        # reaches the emitted plan on fec3a0dc, and a 5 m/s ceiling left the plan bit-identical in the T2 path (MEASURED,
        # raw/refcv7_argv_reach_fixed5.json). Expected: plans identical to off_s0, pair exactly 0 (or moved only by the decoder-condition path).
        ("refcv7_vmax_log_s0", "refcv7", ["--ckpt", ck, "--t2a-config", cfg, "--t2a-vmax-mode", "logged_2_6s", "--t2a-infer-seed", "0"]),
    ]
    tree_r4 = kit / "tree_r4"
    if tree_r4.is_dir():       # the R4-restart inference code (0fcb1c1c): same weights, the ceiling reaches the EMITTED plan (SPEC 26.1 fix)
        A.append(("refcv7_r4code_vmax_log_s0", "refcv7",
                  ["--ckpt", ck, "--t2a-config", cfg, "--t2a-vmax-mode", "logged_2_6s", "--t2a-infer-seed", "0"], {"T2A_TREE": str(tree_r4)}))
    if with_control:   # cross-host repro control: refc-base through THIS box's render vs the banked Thor panel
        A.append(("refc-base", "refc-base", ["--ckpt", str(kit / "models/refc-base/ckpt.pt")]))
    return A


PAIRS = [   # (A, B): paired A - B on identical windows; negative error delta = A better
    ("refcv7_vmax_off_s0", "cl_ha0"), ("refcv7_vmax_off_s0", "cl_ha0_ext"), ("refcv7_vmax_off_s0", "cl_p10_t0"),
    ("refcv7_vmax_off_s0", "cl_p10"), ("refcv7_vmax_off_s0", "cl_ha0_ext_live"),
    ("refcv7_vmax_log_s0", "refcv7_vmax_off_s0"),        # launch tree: expected exactly 0 (SPEC 26.1) -- a plumbing control
    ("refcv7_r4code_vmax_log_s0", "refcv7_vmax_off_s0"),  # R4 inference code: what the oracle ceiling buys once it reaches the plan
    ("refcv7_vmax_off_s1", "refcv7_vmax_off_s0"),        # INFERENCE-seed replicate: the floor an effect must clear
    ("cl_p10_t0", "cl_ha0_ext"), ("cl_p10", "cl_ha0_ext_live"),
]


def zero_control(rollouts_json: Path) -> dict:
    """cl_ha0 plans a constant-velocity straight line; pure pursuit on a waypoint straight ahead gives steer == 0.0 and the
    speed target equals the speed, so accel == 0.0 -- bit-exactly, on any scene, because that arm never reads the image."""
    d = json.loads(Path(rollouts_json).read_text())
    st = [s for r in d["rollouts"] for s in r["steps"]]
    ms, ma = max(abs(s["steer"]) for s in st), max(abs(s["accel"]) for s in st)
    return {"n_steps": len(st), "max_abs_steer": ms, "max_abs_accel": ma, "exact_zero": bool(ms == 0.0 and ma == 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="00040136,7c72937c")
    ap.add_argument("--arms", default=None, help="comma list of tags (default: all, in value order)")
    ap.add_argument("--with-control", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--starts", default=STARTS)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    kit, out = Path(a.kit).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, T2A_TREE=str(kit / "tree"), REFCV6_KIT=str(kit / "refcv6_eval_kit"), PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="6")
    if os.name != "nt":                                   # expandable_segments is unsupported (a warning) on Windows
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.pop("CUDA_VISIBLE_DEVICES", None)               # box 2's GPU is THIS job's; nothing hides it here
    arms = arm_table(kit, a.with_control)
    if a.arms:
        want = a.arms.split(",")
        arms = [x for x in arms if x[0] in want]
    prog_path = out / "PROGRESS.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {}
    for s8 in a.scenes.split(","):
        sid = SCENES[s8]
        scene_dir = kit / "scenes" / sid
        for tag, arm, extra, *ov in arms:
            d = out / s8 / tag
            res = d / f"rollouts_{arm}_empty.json"
            key = f"{s8}/{tag}"
            if res.exists() and not a.no_resume:
                print(f"[skip] {key} (exists)")
                continue
            d.mkdir(parents=True, exist_ok=True)
            cmd = [a.python, str(kit / "t2a" / "t2a_run.py"), "--scene-dir", str(scene_dir), "--arm", arm, "--condition", "empty",
                   "--starts", a.starts, "--steps", str(a.steps), "--out", str(d), *RENDER, *extra]
            if a.dry:
                print((f"[T2A_TREE={ov[0]['T2A_TREE']}] " if ov else "") + " ".join(cmd))
                continue
            t = time.time()
            with open(d / "run.log", "w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, env=dict(env, **(ov[0] if ov else {})), stdout=lf, stderr=subprocess.STDOUT).returncode
            prog[key] = {"rc": rc, "wall_s": round(time.time() - t, 1), "ok": bool(rc == 0 and res.exists())}
            if tag == "cl_ha0" and prog[key]["ok"]:           # the zero control: a constant-velocity straight plan must command EXACTLY 0
                z = zero_control(res)
                prog[key]["zero_control"] = z
                if not z["exact_zero"]:
                    prog[key]["ok"] = False
                    print(f"[FAIL] {key}: zero control commanded steer/accel != 0.0 {z} -- the loop or the harness is broken; "
                          "do NOT trust any other arm from this run", flush=True)
            prog_path.write_text(json.dumps(prog, indent=1))
            print(f"[{'ok' if prog[key]['ok'] else 'FAIL'}] {key} rc={rc} {prog[key]['wall_s']} s  (log: {d / 'run.log'})", flush=True)
    if a.dry:
        return 0
    # ---- scoring: per-arm (tier-stamped) + paired ------------------------------------------------------------------- #
    score = [a.python, str(kit / "t2a" / "t2a_score.py")]
    env2 = dict(env, CUDA_VISIBLE_DEVICES="-1")        # scoring is CPU
    banked = kit / "banked" / "HQ_refc-base_empty.json"
    for s8 in a.scenes.split(","):
        if a.with_control and s8 == "00040136" and banked.exists():
            rc_ = out / s8 / "refc-base" / "rollouts_refc-base_empty.json"
            if rc_.exists():
                subprocess.run([*score, "pair", str(rc_), str(banked), "--out",
                                str(out / s8 / "T2A_PAIR_00040136_refc-base_THISBOX_minus_BANKED_THOR.json")], env=env2, check=False)
        for tag, arm, *_ in arms:
            r = out / s8 / tag / f"rollouts_{arm}_empty.json"
            if r.exists():
                subprocess.run([*score, "arm", str(r), "--out", str(out / s8 / f"T2A_{s8}_{tag}.json"), "--scene-label", SCENES[s8]],
                               env=env2, check=False)
        for A, B in PAIRS:
            ra = next((out / s8 / A / f"rollouts_{x[1]}_empty.json" for x in arms if x[0] == A), None)
            rb = next((out / s8 / B / f"rollouts_{x[1]}_empty.json" for x in arms if x[0] == B), None)
            if ra and rb and ra.exists() and rb.exists():
                subprocess.run([*score, "pair", str(ra), str(rb), "--out", str(out / s8 / f"T2A_PAIR_{s8}_{A}_minus_{B}.json")],
                               env=env2, check=False)
    subprocess.run([*score, "table", str(out)], env=env2, check=False)
    print(f"\nDONE. Read {out / 'T2A_SUMMARY.md'}; per-arm JSONs {out}/<scene>/T2A_*.json; progress {prog_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
