"""ACTSENS read, IA-3 / ladder read of ONE banked arm (card 4.1). Tier T0 (world-model diagnostic). Evidence MEASURED when run.
Code of record: ladder_read/run_ladder_read.py + lr_lib.py, imported UNCHANGED from the dagland folder; this wrapper only (a) takes the run dir as an argument (arms are banked as actsens_<tag>, the original
hard-codes ladder_<name>) and (b) can zero the level-1 action embedding (control C8, --zero-act). The body of read_arm below is a line-for-line copy with run_dir / zero_act added.
usage: python ia_read.py --name IDM_s0 --run-dir C:/Users/Admin/t2v8f_pulls/actsens_IDM_s0 --out F:/.../out/IDM_s0/ia_read.json [--zero-act] [--smoke]
Env: LADDER_READ_CODE_ROOT must be the ACTSENS tree (asserted by lr_lib). The seed (chassis) is the --name suffix _s<k>, except --chassis-seed overrides it (NULL_s2 -> 0, NULL_s3 -> 1)."""
import argparse, hashlib, json, os, sys, time
sys.dont_write_bytecode = True
LR = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/ladder_read"
sys.path.insert(0, LR)
import numpy as np
import torch
import run_ladder_read as RR   # chdirs to LR; imports lr_lib, which asserts tanitad comes from LADDER_READ_CODE_ROOT
import lr_lib as L

p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--run-dir", required=True); p.add_argument("--out", required=True)
p.add_argument("--chassis-seed", type=int, default=None); p.add_argument("--zero-act", action="store_true"); p.add_argument("--smoke", action="store_true")
a = p.parse_args()
out_path = os.path.abspath(a.out) if os.path.isabs(a.out) else a.out
if os.path.exists(out_path):
    sys.exit(f"REFUSED: {out_path} exists (an earlier output is never overwritten)")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
seed = a.chassis_seed if a.chassis_seed is not None else RR.seed_of(a.name)

t0 = time.time()
free = L.memory_guard()
model, cfg, argv, ck = L.load_arm(a.run_dir)
n_zeroed = 0
if a.zero_act:
    with torch.no_grad():
        for q in model.wm.stack.act_embed.parameters():
            q.zero_(); n_zeroed += q.numel()
built = L.built_values(model)
checked = L.check_built_vs_argv(built, argv)
fr = RR.frames_for(model, seed)
if int(argv["seed"][0]) != RR.seed_of(a.name):
    raise L.LadderMismatch(f"argv --seed {argv['seed']} vs the run name {a.name}")
Y = L.build_label_frames(fr["clips"], fr["T"])
cells, states, cum, plan, built_ctx = L.gather(model, fr)
W = L.IL.boot_weights(np.arange(len(fr["clips"])))
worst = L.beside_copy_selfcheck(cells, W)
if worst != 0.0:
    raise RuntimeError(f"Copy-Last against itself is not exactly 0.0 ({worst})")
n_cells = L.check_cells_against_states(cells, states, cum)
ia0 = L.ia0_arm(cells, plan, cum, W, "built", a.smoke)
ia0_ctx1 = L.ia0_arm(cells, plan, cum, W, "ctx1", a.smoke)
beside = L.beside_arm(model, fr, Y, states, cells, cum, plan, W, "built", a.smoke)
res = {"tier": "T0", "evidence": "MEASURED", "name": a.name, "chassis_seed": seed, "run_dir": a.run_dir, "zero_act": bool(a.zero_act), "n_zeroed_params": n_zeroed, "smoke": bool(a.smoke),
       "argv_sha256": hashlib.sha256(open(f"{a.run_dir}/argv.txt", "rb").read()).hexdigest(), "ckpt_step": int(ck.get("step", -1)),
       "wm_gamma_levels": getattr(cfg, "wm_gamma_levels", None), "tanitad_file": L.tanitad.__file__,
       "built": built, "argv_checked": checked, "adapter_sha256": L.adapter_sha(model), "copy_vs_itself_max_abs": worst, "cells_checked_against_states": n_cells,
       "cum_stride": cum, "ctx_built": built_ctx, "ia0": ia0, "ia0_planning_context1": ia0_ctx1, "passes": L.level_passes(ia0), "beside": beside,
       "bootstrap": {"estimator": "episode-cluster bootstrap over the 141 eval clips", "B": L.B_BOOT, "seed": 0}, "reference_draws": 19 if a.smoke else L.R_REF,
       "free_gb_at_start": round(free, 2), "wall_s": round(time.time() - t0, 1)}
json.dump(res, open(out_path + ".tmp", "w", encoding="utf-8"))
os.replace(out_path + ".tmp", out_path)
print(f"[{a.name}{' zero-act' if a.zero_act else ''}] IA read in {res['wall_s']} s -> {out_path}", flush=True)
