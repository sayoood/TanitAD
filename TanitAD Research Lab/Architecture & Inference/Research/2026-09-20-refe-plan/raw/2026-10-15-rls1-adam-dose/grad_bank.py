#!/usr/bin/env python3
"""H-ADAM section 3.1-3.2 input: the micro-batch gradient bank at theta_0 (R0's init = the base scorer).
For stream s (seed 20261015+s) the 302-set pool is cut into 37 random micro-batches of 8; for each, the trainer's own train_losses gives
  gB = d(BCE + lane BCE)/dtheta / accum   and   gE = d(UNIT EPO: -J + beta*KL, lam_pg=1)/dtheta / accum.
An update's gradient = mean over 8 micro-batches (the trainer sums loss/accum over accum micro-batches).   python grad_bank.py --streams 0 1 2 --out-dir <scratch>"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import adam_common as AC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=0)
    q = ap.parse_args()
    if q.threads: torch.set_num_threads(q.threads)
    os.makedirs(q.out_dir, exist_ok=True)
    SF = AC.load_trainer(); a, _ = AC.r0_args(SF); holder, cfg = AC.build_scorer(SF, a)
    pool = AC.Pool(AC.prepare_pool(SF, a, cfg, "train")).load_visual_all(); tr = AC.Trainer(SF, a, holder, pool)
    N = len(pool.keys)
    for s in q.streams:
        t0 = time.time(); perm = np.random.default_rng(20261015 + s).permutation(N)
        groups = perm[: (N // 8) * 8].reshape(-1, 8)
        gBs, gEs, infos = [], [], []
        for j, ids in enumerate(groups):
            ids = [int(i) for i in ids]; cxy, ctg, sctx, _ = pool.micro(ids)
            gB, gE, info = tr.micro_grads(ids, cxy, ctg, sctx)
            gBs.append(AC.flat(gB)); gEs.append(AC.flat(gE)); infos.append(info)
            if j % 6 == 0: print(f"stream {s} micro {j}/{len(groups)} {time.time()-t0:.0f}s", flush=True)
        torch.save({"gB": torch.stack(gBs), "gE": torch.stack(gEs), "groups": groups, "infos": infos, "names": tr.names,
                    "shapes": [tuple(p.shape) for p in tr.params], "seed": 20261015 + s}, os.path.join(q.out_dir, f"bank_s{s}.pt"))
        print(f"ZZBANK_DONE stream {s} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
