"""Per-arm driver of the ladder T0 read (resumable: an arm with results/arm_<name>.json is skipped unless --force). Tier T0.
Usage (cwd = this directory; env_ladder_read.cmd):  python -u run_ladder_read.py C0_s0 C0_s1 ... [--force] [--smoke]
The arm list is an argument: R35b_s0 is added later by passing it (run dir <PULLS>/ladder_R35b_s0). Memory guard: 6 GB free physical memory before each arm."""
import json, os, sys, time, traceback, hashlib
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import numpy as np
import torch
import lr_lib as L

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
CACHE = f"{L.SCR}/cache"


def seed_of(name):
    return int(name.rsplit("_s", 1)[1])


def frames_for(model, seed):
    p = f"{CACHE}/frames_s{seed}.npz"
    if not os.path.exists(p):
        os.makedirs(CACHE, exist_ok=True)
        L.build_eval_frames(model, p)
    fr = L.load_eval_frames(p)
    got = L.adapter_sha(model)
    if got != fr["adapter_sha"]:
        raise L.LadderMismatch(f"frozen adapter of this run differs from the cached frames' adapter for seed {seed}: {got[:12]} vs {fr['adapter_sha'][:12]}")
    return fr


def read_arm(name, smoke=False):
    t0 = time.time()
    free = L.memory_guard()
    run_dir = f"{L.PULLS}/ladder_{name}"
    model, cfg, argv, ck = L.load_arm(run_dir)
    built = L.built_values(model)
    checked = L.check_built_vs_argv(built, argv)                       # raises on any mismatch: the arm is not read
    fr = frames_for(model, seed_of(name))
    if int(argv["seed"][0]) != seed_of(name):
        raise L.LadderMismatch(f"argv --seed {argv['seed']} vs the run name {name}")
    Y = L.build_label_frames(fr["clips"], fr["T"])
    cells, states, cum, plan, built_ctx = L.gather(model, fr)
    W = L.IL.boot_weights(np.arange(len(fr["clips"])))
    worst = L.beside_copy_selfcheck(cells, W)
    if worst != 0.0:
        raise RuntimeError(f"Copy-Last against itself is not exactly 0.0 (max |value| {worst})")
    n_cells_checked = L.check_cells_against_states(cells, states, cum)
    ia0 = L.ia0_arm(cells, plan, cum, W, "built", smoke)
    ia0_ctx1 = L.ia0_arm(cells, plan, cum, W, "ctx1", smoke)
    beside = L.beside_arm(model, fr, Y, states, cells, cum, plan, W, "built", smoke)
    res = {"tier": "T0", "arm": name.rsplit("_s", 1)[0], "seed": seed_of(name), "name": name, "run_dir": run_dir, "smoke": bool(smoke),
           "argv_sha256": hashlib.sha256(open(f"{run_dir}/argv.txt", "rb").read()).hexdigest(), "ckpt_step": int(ck.get("step", -1)),
           "built": built, "argv_checked": checked, "adapter_sha256": L.adapter_sha(model), "copy_vs_itself_max_abs": worst, "cells_checked_against_states": n_cells_checked,
           "n_clips": len(fr["clips"]), "n_frames": int(sum(fr["T"])), "n_not_ok_frames": int(sum((~o).sum() for o in fr["ok"])),
           "cum_stride": cum, "ctx_built": built_ctx, "horizon_plan": {str(k): v for k, v in plan.items()},
           "ia0": ia0, "ia0_planning_context1": ia0_ctx1, "passes": L.level_passes(ia0), "beside": beside,
           "bootstrap": {"estimator": "episode-cluster bootstrap over the 141 eval clips", "B": L.B_BOOT, "seed": 0}, "reference_draws": 19 if smoke else L.R_REF,
           "free_gb_at_start": round(free, 2), "wall_s": round(time.time() - t0, 1)}
    return res


def main(argv):
    smoke, force = "--smoke" in argv, "--force" in argv
    names = [a for a in argv if not a.startswith("--")]
    rc = 0
    for n in names:
        out = os.path.join(HERE, "results", ("smoke_" if smoke else "") + f"arm_{n}.json")
        if os.path.exists(out) and not force:
            print(f"[{n}] done already, skipped", flush=True)
            continue
        try:
            r = read_arm(n, smoke)
            json.dump(r, open(out + ".tmp", "w", encoding="utf-8"))
            os.replace(out + ".tmp", out)
            print(f"[{n}] read in {r['wall_s']} s -> {out}", flush=True)
        except BaseException as e:
            traceback.print_exc()
            with open(os.path.join(HERE, "results", f"FAILED_{n}.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"{type(e).__name__}: {str(e)[:2000]}\n")
            print(f"[{n}] FAILED {type(e).__name__}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
