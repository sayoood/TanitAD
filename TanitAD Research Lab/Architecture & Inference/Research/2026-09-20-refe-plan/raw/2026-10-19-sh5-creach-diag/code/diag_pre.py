"""SH5 C-REACH DIAGNOSIS driver (Research Lab, 2026-10-19).  NOT a registered stage: it trains ONLY the planted control arm (`plant`) on fold 0 (= `fit --stage PRE --folds 0`) under
DIAGNOSTIC recipe variants, to find why the branch cannot read the planted channel.  It imports the SEALED modules read-only from `par18/sh5/code` and changes nothing there; every variant is
applied in THIS process only (per-learner lr, query Fourier octaves, plant-code octaves, epochs).  No real arm (S, Snoprox, Sshuf) is built, no `analyze` is run.

The GPU is taken through the sealed `sh5_gpulock.run_locked_inprocess` (RAM gate, compute check, never touches a foreign lock), exactly as `fit` does.

usage: PY diag_pre.py --grid grid.json --out OUTDIR [--max-wait S]
grid.json = {"runs": [{"name": ..., "seed": 0, "cycles": 3, "inner": 4, "learners": [{"tag": ..., "lr": 3e-4, "fourier_k": 6, "plant_oct": 8, "warmup": 50}, ...]}, ...]}
All learners of one run train in LOCKSTEP on the same shards (one ctx read per shard); a run's learners share its epochs and seed.  After every cycle each learner is evaluated on fold 0
(held out): `auc_all_strata` (= the ZZPREFLIGHT number, `_fl_pairs`), F-AUC (Q4), `r2_reg`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

SEALED = "C:/Users/Admin/par18/sh5/code"
sys.path.insert(0, SEALED)
import sh5_lib as L                    # noqa: E402
import sh5_branch as B                 # noqa: E402
import sh5_train_eval as TE            # noqa: E402

DD = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19"


def plant_block_oct(pts, signed_clear, d_ctx, n_slots, n_oct):
    """sh5_branch.plant_block with `n_oct` octaves of the (x, y) code (sealed: 8 = wavelengths 128 m .. 1 m).  Channels: [0, 4*n_oct) code, 4*n_oct = clearance / 10, 4*n_oct + 1 = flag."""
    S, Pp, n, _ = pts.shape
    xy = pts[..., :2].reshape(-1, 2).astype(np.float64)
    sc = signed_clear.reshape(-1).astype(np.float64)
    if len(xy) > n_slots:
        raise ValueError("too many plant entries")
    code = []
    for j in range(n_oct):
        w = (2.0 ** j) * math.pi / 64.0
        code += [np.sin(w * xy[:, 0]), np.cos(w * xy[:, 0]), np.sin(w * xy[:, 1]), np.cos(w * xy[:, 1])]
    blk = np.zeros((len(xy), d_ctx), np.float64)
    blk[:, :4 * n_oct] = np.stack(code, axis=1)
    blk[:, 4 * n_oct] = np.clip(sc, -L.CLEAR_CAP, L.CLEAR_CAP) / 10.0
    blk[:, 4 * n_oct + 1] = 1.0
    return blk.astype(np.float32)


class DiagTrainer(TE.Trainer):
    """Per-learner lr / warm-up / plant octaves.  Everything else is the sealed Trainer."""

    def _batch(self, ln, toks, ctx_pos, shard_ctx, train):
        import torch
        abl = getattr(ln, "ablate", None)
        if getattr(ln, "plant_oct", 8) == 8 and abl is None:
            return super()._batch(ln, toks, ctx_pos, shard_ctx, train)        # the sealed path, byte for byte
        D = self.D
        sets = np.concatenate([D["tok_sets"][t] for t in toks])
        s2t = np.concatenate([np.full(len(D["tok_sets"][t]), i, np.int64) for i, t in enumerate(toks)])
        ctx = shard_ctx[torch.as_tensor([ctx_pos[t] for t in toks], device=shard_ctx.device)].clone()
        for i, t in enumerate(toks):
            s = D["tok_sets"][t]
            blk = plant_block_oct(D["pts"][s], D["sc_key"][s].astype(np.float64), ctx.shape[-1], ctx.shape[1], ln.plant_oct)
            if abl == "zero_clear":                        # ABLATION (eval only): the planted clearance channel is set to 0; code and flag stay
                blk[:, 4 * ln.plant_oct] = 0.0
            elif abl == "shuffle_clear":                   # ABLATION (eval only): the planted clearances are permuted across the token's entries (same marginal, wrong positions)
                blk[:, 4 * ln.plant_oct] = np.random.default_rng(int(t) + 17).permutation(blk[:, 4 * ln.plant_oct])
            if abl == "no_plant":
                continue
            ctx[i, ctx.shape[1] - len(blk):] = torch.as_tensor(blk, dtype=ctx.dtype, device=ctx.device)
        pts = D["pts"][sets]
        if train:
            sel = np.argsort(ln.rng.random((len(sets), 64)), axis=1)[:, :self.rc.p_sub]
            r = np.arange(len(sets))[:, None]
            pts = pts[r, sel]
            y, tgt = ln.y[sets][r, sel], ln.tgt[sets][r, sel]
        else:
            y, tgt = None, None
        f = lambda a: torch.as_tensor(a, device=self.rc.device)            # noqa: E731
        return f(pts), f(s2t), ctx, (None if y is None else f(y)), (None if tgt is None else f(tgt)), sets

    def _train_step(self, ln, toks, ctx_pos, shard_ctx):
        lr0, wu = L.LR, L.WARMUP
        L.LR, L.WARMUP = ln.diag_lr, ln.diag_warmup                       # the sealed step reads L.LR / L.WARMUP at call time
        try:
            return super()._train_step(ln, toks, ctx_pos, shard_ctx)
        finally:
            L.LR, L.WARMUP = lr0, wu


def evaluate(tr, D, f0):
    out = {}
    groups = tr.predict()                       # one group per learner: give each learner a unique group via its tag (below)
    for g_, v in groups.items():
        cov = v["n_cover"] == 1
        lab = D["lab"][..., L.DAC_COL][cov]
        FL, _ = L.refs()
        st = FL.pair_stats(v["z"][cov], lab == 0.0, lab == 1.0, D["code"][cov], names=("ALLQ", "Q4", "Q123"))
        a = {k: float(n.sum() / d.sum()) if d.sum() > 0 else None for k, (n, d) in st.items()}
        out[g_] = {"n_sets": int(cov.sum()), "auc_all_strata": a["ALLQ"], "auc_F": a["Q4"], "auc_Q123": a["Q123"], "r2_reg": TE.r2_reg(v["reg"][cov], D["tgt"][cov]),
                   "z_std": float(np.nanstd(v["z"][cov]))}
    return out, groups


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-wait", type=float, default=21600.0)
    ap.add_argument("--min-avail-gb", type=float, default=L.MIN_AVAIL_GB_FIT)
    x = ap.parse_args(argv)
    import sh5_gpulock as GL
    grid = json.load(open(x.grid, encoding="utf-8"))
    os.makedirs(x.out, exist_ok=True)
    allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
    keep, row_of = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
    D = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", keep)
    ctx_rows = [row_of[str(t)] for t in D["token"]]
    f0 = D["fold"] == 0
    print(f"ZZDIAG {len(grid['runs'])} runs; N {D['N']} T {D['T']} fold-0 sets {int(f0.sum())}", flush=True)

    def body():
        import torch
        GL.require_locked("cuda")
        res = {}
        for run in grid["runs"]:
            t0 = time.time()
            rc = TE.RunCfg(seed=run["seed"], cycles=run["cycles"], inner=run["inner"], epochs_total=run["cycles"] * run["inner"], device="cuda")
            provider = TE.NpyCtx(f"{DD}/ctx_local/ctx.npy", ctx_rows, "cuda")
            specs = [TE.Spec("plant", 0, run["seed"]) for _ in run["learners"]]
            tr = DiagTrainer(D, specs, rc, provider, log=lambda m: print(m, flush=True))
            if tr.n_shards == 1:
                raise RuntimeError("expected a sharded ctx (cycles x inner); refuse rather than silently change the schedule")
            lns = []
            for i, (sp, lc) in enumerate(zip(specs, run["learners"])):
                cfg = B.SH5Cfg(d_ctx=provider.Dc, fourier_k=lc.get("fourier_k", L.FOURIER_K))
                ln = TE.Learner(sp, cfg, D, "cuda")
                if lc.get("init_seed") is not None:
                    torch.manual_seed(lc["init_seed"]); ln.model = B.SH5Branch(cfg).to("cuda")
                    ln.opt = torch.optim.Adam(ln.model.parameters(), lr=lc["lr"], weight_decay=L.WD, betas=L.BETAS, eps=L.EPS)
                ln.diag_lr, ln.diag_warmup, ln.plant_oct, ln.tag = lc["lr"], lc.get("warmup", L.WARMUP), lc.get("plant_oct", 8), lc["tag"]
                lns.append(ln)
            tr.learners = lns
            # unique group names per learner: Spec.group ignores rep for 'plant', so patch through a tiny subclass
            for ln in lns:
                ln.spec = _TagSpec(ln.spec.arm, ln.spec.fold, ln.spec.seed, ln.spec.rep, ln.tag)
            curve = {ln.tag: [] for ln in lns}

            def ck(c):
                ev, _ = evaluate(tr, D, f0)
                for ln in lns:
                    e = ev[ln.spec.group]
                    e.update({"cycle": c + 1, "epochs": (c + 1) * run["inner"], "steps": ln.step})
                    curve[ln.tag].append(e)
                    print(f"ZZDIAGEVAL {run['name']} {ln.tag} cycle {c + 1}: auc_all {e['auc_all_strata']:.4f} F {e['auc_F']:.4f} r2 {e['r2_reg']:.4f} steps {ln.step}", flush=True)
                json.dump({"run": run, "curve": curve, "elapsed_s": time.time() - t0}, open(f"{x.out}/diag_{run['name']}.json", "w"), indent=1)
            tr.train(ckpt_fn=ck)
            res[run["name"]] = {ln.tag: curve[ln.tag][-1] for ln in lns}
            for abl in run.get("ablate", []):                # eval-only ablations of the TRAINED learners: does the read come from the planted clearance?
                for ln in lns:
                    ln.ablate = abl
                ev, _ = evaluate(tr, D, f0)
                for ln in lns:
                    ln.ablate = None
                    res[run["name"]][ln.tag][f"ablate_{abl}"] = ev[ln.spec.group]
                    print(f"ZZDIAGABL {run['name']} {ln.tag} {abl}: auc_all {ev[ln.spec.group]['auc_all_strata']:.4f} F {ev[ln.spec.group]['auc_F']:.4f} r2 {ev[ln.spec.group]['r2_reg']:.4f}", flush=True)
                json.dump({"run": run, "curve": curve, "final": res[run["name"]]}, open(f"{x.out}/diag_{run['name']}.json", "w"), indent=1)
            print(f"ZZDIAGRUN_DONE {run['name']} {time.time() - t0:.0f} s", flush=True)
            del tr, provider
            torch.cuda.empty_cache()
        json.dump(res, open(f"{x.out}/diag_summary.json", "w"), indent=1)
        print("ZZDIAG_OK " + json.dumps(res), flush=True)
        return 0
    rc = GL.run_locked_inprocess(body, max_wait=x.max_wait, min_avail_gb=x.min_avail_gb, owner="sh5_diag", job="sh5-creach-diag")
    print(f"ZZDIAG_EXIT {rc}", flush=True)
    return rc


import dataclasses  # noqa: E402


@dataclasses.dataclass
class _TagSpec(TE.Spec):
    tag: str = ""

    @property
    def name(self) -> str:
        return f"plant[{self.tag}]_f{self.fold}_s{self.seed}"

    @property
    def group(self) -> str:
        return f"plant[{self.tag}]_s{self.seed}"


if __name__ == "__main__":
    sys.exit(main())
