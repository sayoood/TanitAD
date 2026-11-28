#!/usr/bin/env python3
"""POST-HOC extension of H-ADAM (labelled as such in REPORT.md; not in PREREG_H_ADAM.md): the gradient-scale ratio r and the objective values J / BCE at the REAL late-run weights.

For each banked scorer state (R0-B / R0b-B / BCE-only arm A at u250, u500, u1000) plus the base: on 4 batches of 8 micro-batches (stream-0 partition, 256 sets) compute the trainer's
gB (BCE + lane BCE) and unit gE, report r_total = rms|gE| / rms|gB| (per module group too), mean J (expected reward), mean BCE, and -- as a check of the substituted sctx at the late state --
the same within-scene V against the banked RL-val logits of that state (scores_update{u}_{A|B}.npz) on the 62 cached RL-val frames.

    python late_probe.py --bank-dir <scratch/bank> --out late_probe.json [--threads 6]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adam_common as AC  # noqa: E402
from disp_real import group_of  # noqa: E402

STATES = [("base", None, None, None), ("R0_B_u250", "r0", 250, "B"), ("R0_B_u500", "r0", 500, "B"), ("R0_B_u1000", "r0", 1000, "B"), ("R0b_B_u250", "r0b", 250, "B"),
          ("R0b_B_u500", "r0b", 500, "B"), ("R0_A_u500", "r0", 500, "A"), ("C0_A_u500", "c0", 500, "A")]


def load_into(arm, run, u, which, SF):
    ss = torch.load(os.path.join(AC.DL, f"rls1-{run}", "scorer_states", f"scorer_states_u{u}.pt"), map_location="cpu", weights_only=False)
    for m in SF.SCORER:
        getattr(arm, m).load_state_dict(ss[which][m], strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--n-batches", type=int, default=4)
    q = ap.parse_args()
    if q.threads:
        torch.set_num_threads(q.threads)
    SF = AC.load_trainer()
    a, _ = AC.r0_args(SF)
    holder, cfg = AC.build_scorer(SF, a)
    pool = AC.Pool(AC.prepare_pool(SF, a, cfg, "train")).load_visual_all()
    tr = AC.Trainer(SF, a, holder, pool)
    pv = AC.Pool(AC.prepare_pool(SF, a, cfg, "val")).load_visual_all()
    groups = torch.load(os.path.join(q.bank_dir, "bank_s0.pt"), weights_only=False)["groups"]
    z = {}
    res = {"tool": "late_probe/1", "class": "ESTIMATED (real late-run weights; substituted sctx)", "states": {}}
    base_sd = {m: {k: v.clone() for k, v in getattr(tr.armB, m).state_dict().items()} for m in SF.SCORER}
    names = tr.names
    for tag, run, u, which in STATES:
        t0 = time.time()
        for m in SF.SCORER:
            getattr(tr.armB, m).load_state_dict(base_sd[m])
        if run:
            load_into(tr.armB, run, u, which, SF)
        GB, GE, Js, Bs, KLs = [], [], [], [], []
        for b in range(q.n_batches):
            gb, ge = None, None
            for j in range(8):
                ids = [int(i) for i in groups[b * 8 + j]]
                cxy, ctg, sctx, _ = pool.micro(ids)
                gB, gE, info = tr.micro_grads(ids, cxy, ctg, sctx)
                gb = AC.flat(gB) / 8 * 8 if gb is None else gb + AC.flat(gB)           # micro_grads already divides by accum (8); a batch gradient = SUM over its 8 micro-batches
                ge = AC.flat(gE) if ge is None else ge + AC.flat(gE)
                Js.append(info["J"]); Bs.append(info["bce"]); KLs.append(info["kl"])
            GB.append(gb); GE.append(ge)
        GB, GE = torch.stack(GB), torch.stack(GE)
        out = {"r_total_lam1": float(GE.pow(2).sum(1).mean().sqrt() / GB.pow(2).sum(1).mean().sqrt()), "rms_norm_gB": float(GB.pow(2).sum(1).mean().sqrt()),
               "rms_norm_gE": float(GE.pow(2).sum(1).mean().sqrt()), "mean_J": float(np.mean(Js)), "mean_BCE": float(np.mean(Bs)), "mean_KL_to_base": float(np.mean(KLs)), "groups": {}}
        o = 0
        gg = {}
        for n, p in zip(names, tr.params):
            m = p.numel()
            g = group_of(n)
            gg.setdefault(g, [0.0, 0.0, 0.0, 0.0, 0])
            gg[g][0] += float(GB[:, o:o + m].pow(2).sum(1).mean())
            gg[g][1] += float(GE[:, o:o + m].pow(2).sum(1).mean())
            rb = GB[:, o:o + m].pow(2).mean(0).sqrt()
            re = GE[:, o:o + m].pow(2).mean(0).sqrt()
            gg[g][2] += float((re > rb).float().sum())
            gg[g][3] += float((0.25 * re > rb).float().sum())
            gg[g][4] += m
            o += m
        for g, v in gg.items():
            out["groups"][g] = {"r_lam1": float(np.sqrt(v[1] / v[0])), "r_lam025": 0.25 * float(np.sqrt(v[1] / v[0])), "coord_share_EPO_dominates_lam1": v[2] / v[4], "coord_share_EPO_dominates_lam025": v[3] / v[4]}
        # sctx check at this state on the cached RL-val frames
        if run:
            zf = np.load(f"C:/Users/Admin/rls1b_dl/{run}/final/scores_update{u}_{which}.npz", allow_pickle=False) if os.path.exists(f"C:/Users/Admin/rls1b_dl/{run}/final/scores_update{u}_{which}.npz") else None
        else:
            zf = np.load("C:/Users/Admin/rls1b_dl/r0/final/scores_step0_base.npz", allow_pickle=False)
        if zf is not None:
            idx = {(str(zf["log"][i]), str(zf["token"][i]), int(zf["step"][i]), int(zf["rank"][i])): i for i in range(len(zf["token"]))}
            banked = np.stack([zf["scores"][idx[k]] for k in pv.keys]).astype(np.float64)
            mine = []
            for i in range(len(pv.keys)):
                with torch.no_grad():
                    mine.append(tr.armB(torch.from_numpy(pv.cxy[[i]]), AC.bf16bits_to_f32(pv.vis[i])[None]).float().numpy()[0])
            mine = np.stack(mine).astype(np.float64)
            rs = [np.mean([np.corrcoef(mine[k, :, j], banked[k, :, j])[0, 1] for k in range(mine.shape[0]) if mine[k, :, j].std() > 0 and banked[k, :, j].std() > 0]) for j in range(7)]
            out["V_vs_banked_RLval"] = float(np.median(rs))
            out["mean_abs_logit_diff_vs_banked"] = float(np.abs(mine - banked).mean())
        res["states"][tag] = out
        print(f"{tag}: r_total {out['r_total_lam1']:.3f} |gB| {out['rms_norm_gB']:.4f} |gE| {out['rms_norm_gE']:.4f} J {out['mean_J']:.4f} BCE {out['mean_BCE']:.4f} KL {out['mean_KL_to_base']:.4f} "
              f"V {out.get('V_vs_banked_RLval', float('nan')):.3f} {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(q.out, "w"), indent=1)
    print("ZZLATE_OK")


if __name__ == "__main__":
    main()
