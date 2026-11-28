"""SH5 C-REACH DIAGNOSIS driver, grid 3+ (Research Lab, 2026-10-19).  Extends `diag_pre.py` (UNCHANGED: it produced diag1/diag2) with
  * per-learner ATTENTION levers (`diag_attn.apply_lever`: learnable temperature / cosine attention with learnable temperature / q-k init scale),
  * per-learner `p_sub`, eval only at `eval_cycles`, a checkpoint (state_dict) of every learner at the end of its run,
  * the attention-logit diagnostic (`diag_attn.attn_token_stats`) on the first `n_attn` fold-0 tokens of every evaluation (not under ablations),
  * `--skip-if-pass SUMMARY` (exit 0 without taking the lock when a learner of SUMMARY already PASSES, rule below) and `--best-from SUMMARY` (resolve learner entries with
    "lever": "BEST" to the pre-registered BEST lever of SUMMARY, rule in `best_lever`).
It trains ONLY the planted control arm (`plant`) on fold 0 (= `fit --stage PRE --folds 0`) and imports the sealed modules read-only.  The GPU is taken through the sealed
`sh5_gpulock.run_locked_inprocess`, owner `sh5_diag`, exactly as diag1/diag2.

PASS (PREREG_GRID3.md, written before any grid-3 number): on BOTH seeds, final cycle, fold 0: auc_F >= 0.97 AND auc_all_strata >= 0.95 AND r2_reg >= 0.50, AND the read collapses
when the planted clearance is removed: ablate_zero_clear AND ablate_shuffle_clear each read auc_all_strata <= 0.80 AND auc_F <= 0.80.

usage: PY diag_pre3.py --grid grid3.json --out OUTDIR [--max-wait S] [--skip-if-pass SUMMARY] [--best-from SUMMARY]"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "C:/Users/Admin/par18/sh5diag/code")
import diag_pre as DP                 # noqa: E402
import diag_attn as DA                # noqa: E402
L, B, TE = DP.L, DP.B, DP.TE
DD = DP.DD

PASS_F, PASS_ALL, PASS_R2, COLLAPSE_MAX = 0.97, 0.95, 0.50, 0.80


def learner_pass(rec: dict) -> bool:
    try:
        ok = rec["auc_F"] >= PASS_F and rec["auc_all_strata"] >= PASS_ALL and rec["r2_reg"] >= PASS_R2
        for a in ("ablate_zero_clear", "ablate_shuffle_clear"):
            ok = ok and rec[a]["auc_all_strata"] <= COLLAPSE_MAX and rec[a]["auc_F"] <= COLLAPSE_MAX
        return bool(ok)
    except (KeyError, TypeError):
        return False


def by_cell(summary: dict) -> dict:
    """{cell: {seed: final record}}; cell = the learner tag, seed = the run's seed (run names end in _s<seed>)."""
    out = {}
    for run, lrn in summary.items():
        seed = int(run.rsplit("_s", 1)[1])
        for tag, rec in lrn.items():
            out.setdefault(tag, {})[seed] = rec
    return out


def passing_cells(summary: dict) -> list:
    return sorted(c for c, s in by_cell(summary).items() if len(s) >= 2 and all(learner_pass(r) for r in s.values()))


def best_lever(summary: dict, grid: dict) -> dict:
    """PRE-REGISTERED (PREREG_GRID3.md): BEST = the LEVER cell (lever not None) of SUMMARY's grid with the highest MIN over its seeds of the final auc_F; ties -> higher min auc_all_strata.
    Returns its learner config (lever, fourier_k, plant_oct; lr / p_sub / epochs are then set by the new cell)."""
    cfg = {lc["tag"]: lc for run in grid["runs"] for lc in run["learners"] if lc.get("lever")}
    cells = by_cell(summary)
    cand = [(min(r["auc_F"] for r in cells[t].values()), min(r["auc_all_strata"] for r in cells[t].values()), t) for t in cfg if t in cells]
    if not cand:
        raise RuntimeError("no lever cell in the summary")
    cand.sort(reverse=True)
    t = cand[0][2]
    return {"from_tag": t, "min_F": cand[0][0], "min_all": cand[0][1], "lever": cfg[t]["lever"], "fourier_k": cfg[t]["fourier_k"], "plant_oct": cfg[t]["plant_oct"]}


class DiagTrainer3(DP.DiagTrainer):
    def _batch(self, ln, toks, ctx_pos, shard_ctx, train):
        if getattr(ln, "p_sub", self.rc.p_sub) != self.rc.p_sub:
            p0 = self.rc.p_sub
            self.rc.p_sub = ln.p_sub
            try:
                return DP.DiagTrainer._batch(self, _ForceDiag(ln), toks, ctx_pos, shard_ctx, train)
            finally:
                self.rc.p_sub = p0
        return super()._batch(ln, toks, ctx_pos, shard_ctx, train)

    def predict(self) -> dict:
        """the sealed Trainer.predict (copied; identical z / reg) + the attention diagnostic on the first `n_attn` fold-0 tokens of each learner (skipped under an ablation)."""
        import torch
        D, rc = self.D, self.rc
        out = {}
        for ln in self.learners:
            g = out.setdefault(ln.spec.group, {"z": np.full((D["N"], 64), np.nan, np.float32), "reg": np.zeros((D["N"], 64, L.N_REG), np.float16), "n_cover": np.zeros(D["N"], np.int64)})
            ln.model.eval()
            ln.attn_acc, ln.attn_n = None, 0
        order = np.random.default_rng(rc.seed + 424242).permutation(D["T"])
        for sh in range(self.n_shards):
            toks_sh = order[sh * self.shard_tokens:(sh + 1) * self.shard_tokens]
            self.ram_guard()
            shard_ctx = self.provider.load(toks_sh)
            ctx_pos = {int(t): i for i, t in enumerate(toks_sh)}
            for t in toks_sh:
                t = int(t)
                for ln in self.learners:
                    if D["tok_fold"][t] != ln.spec.fold:
                        continue
                    pts, s2t, ctx, _, _, sets = self._batch(ln, [t], ctx_pos, shard_ctx, False)
                    with torch.no_grad(), self._ac():
                        kv = ln.model.kv(ctx)
                        logit, reg = ln.model(pts, kv, s2t)
                    g = out[ln.spec.group]
                    g["z"][sets] = logit.float().cpu().numpy()
                    g["reg"][sets] = reg.float().cpu().numpy().astype(np.float16)
                    g["n_cover"][sets] += 1
                    if getattr(ln, "ablate", None) is None and ln.attn_n < getattr(self, "n_attn", 0):
                        st = DA.attn_token_stats(ln.model, pts, ctx, min(ln.model.c.fourier_k, ln.plant_oct), self._ac)
                        ln.attn_acc, ln.attn_n = DA.acc_add(ln.attn_acc, st), ln.attn_n + 1
            del shard_ctx
        return out


class _ForceDiag:
    """a proxy that sends a plant_oct == 8, un-ablated learner down diag_pre's own (byte-identical, smoke-tested) plant path, so a per-learner p_sub can be applied."""
    def __init__(self, ln):
        self.__dict__["_ln"] = ln

    def __getattr__(self, k):
        if k == "ablate":
            a = getattr(self._ln, "ablate", None)
            return "inert" if a is None else a
        return getattr(self._ln, k)


def make_learners(D, run, provider, device):
    import torch
    specs = [TE.Spec("plant", 0, run["seed"]) for _ in run["learners"]]
    lns = []
    for sp, lc in zip(specs, run["learners"]):
        cfg = B.SH5Cfg(d_ctx=provider.Dc, fourier_k=lc.get("fourier_k", L.FOURIER_K))
        ln = TE.Learner(sp, cfg, D, device)
        DA.apply_lever(ln.model, lc.get("lever"))
        ln.model.to(device)
        ln.opt = torch.optim.Adam(ln.model.parameters(), lr=lc["lr"], weight_decay=L.WD, betas=L.BETAS, eps=L.EPS)
        ln.diag_lr, ln.diag_warmup, ln.plant_oct, ln.tag = lc["lr"], lc.get("warmup", L.WARMUP), lc.get("plant_oct", 8), lc["tag"]
        ln.p_sub = lc.get("p_sub", L.P_SUB)
        ln.spec = DP._TagSpec(ln.spec.arm, ln.spec.fold, ln.spec.seed, ln.spec.rep, ln.tag)
        lns.append(ln)
    return specs, lns


def resolve_best(grid: dict, best: dict) -> dict:
    g = copy.deepcopy(grid)
    for run in g["runs"]:
        for lc in run["learners"]:
            if lc.get("lever") == "BEST":
                lc["lever"], lc["fourier_k"], lc["plant_oct"] = best["lever"], best["fourier_k"], best["plant_oct"]
                lc["best_from"] = best["from_tag"]
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-wait", type=float, default=21600.0)
    ap.add_argument("--min-avail-gb", type=float, default=L.MIN_AVAIL_GB_FIT)
    ap.add_argument("--skip-if-pass", default=None)
    ap.add_argument("--best-from", default=None, help="SUMMARY,GRID of the grid whose BEST lever resolves 'BEST' entries")
    ap.add_argument("--n-attn", type=int, default=24)
    x = ap.parse_args(argv)
    import sh5_gpulock as GL
    grid = json.load(open(x.grid, encoding="utf-8"))
    os.makedirs(x.out, exist_ok=True)
    if x.skip_if_pass:
        if not os.path.exists(x.skip_if_pass):
            print(f"ZZDIAG3_SKIP_REFUSED summary missing: {x.skip_if_pass}", flush=True)
            print("ZZDIAG_EXIT 3", flush=True)
            return 3
        pc = passing_cells(json.load(open(x.skip_if_pass, encoding="utf-8")))
        if pc:
            print(f"ZZDIAG3_SKIPPED a cell already passes: {pc}", flush=True)
            print("ZZDIAG_EXIT 0", flush=True)
            return 0
    if x.best_from:
        sp, gp = x.best_from.split(",")
        best = best_lever(json.load(open(sp, encoding="utf-8")), json.load(open(gp, encoding="utf-8")))
        grid = resolve_best(grid, best)
        print("ZZDIAG3_BEST " + json.dumps(best), flush=True)
        json.dump(grid, open(f"{x.out}/grid_resolved.json", "w"), indent=1)
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
            specs, lns = make_learners(D, run, provider, "cuda")
            tr = DiagTrainer3(D, specs, rc, provider, log=lambda m: print(m, flush=True))
            tr.n_attn = x.n_attn
            if tr.n_shards == 1:
                raise RuntimeError("expected a sharded ctx (cycles x inner); refuse rather than silently change the schedule")
            tr.learners = lns
            curve = {ln.tag: [] for ln in lns}
            ev_cycles = set(run.get("eval_cycles", range(1, run["cycles"] + 1)))

            def ck(c):
                if (c + 1) not in ev_cycles:
                    return
                ev, _ = DP.evaluate(tr, D, f0)
                for ln in lns:
                    e = ev[ln.spec.group]
                    e.update({"cycle": c + 1, "epochs": (c + 1) * run["inner"], "steps": ln.step, "tau": DA.taus(ln.model)})
                    e["attn"] = DA.acc_report(ln.attn_acc, ln.attn_n) if ln.attn_acc is not None else None
                    curve[ln.tag].append(e)
                    a = e["attn"] or [{}]
                    print(f"ZZDIAGEVAL {run['name']} {ln.tag} cycle {c + 1}: auc_all {e['auc_all_strata']:.4f} F {e['auc_F']:.4f} r2 {e['r2_reg']:.4f} steps {ln.step}", flush=True)
                    for A in a:
                        if A:
                            print(f"ZZATTN {run['name']} {ln.tag} cycle {c + 1} L{A['layer']}: beta_eff_max {A['beta_eff_max']} beta {A['beta_eff']} range_plant {A['range_plant_mean']} "
                                  f"plant_mass {A['plant_mass']} self_mass {A['self_mass']} top1 {A['top1_within_0p5m']} tau {e['tau'][A['layer']]}", flush=True)
                json.dump({"run": run, "curve": curve, "elapsed_s": time.time() - t0}, open(f"{x.out}/diag_{run['name']}.json", "w"), indent=1)
            tr.train(ckpt_fn=ck)
            res[run["name"]] = {ln.tag: curve[ln.tag][-1] for ln in lns}
            os.makedirs(f"{x.out}/ckpt", exist_ok=True)
            for ln in lns:
                torch.save({k: v.detach().cpu() for k, v in ln.model.state_dict().items()}, f"{x.out}/ckpt/{run['name']}__{ln.tag}.pt")
            for abl in run.get("ablate", []):
                for ln in lns:
                    ln.ablate = abl
                ev, _ = DP.evaluate(tr, D, f0)
                for ln in lns:
                    ln.ablate = None
                    res[run["name"]][ln.tag][f"ablate_{abl}"] = ev[ln.spec.group]
                    print(f"ZZDIAGABL {run['name']} {ln.tag} {abl}: auc_all {ev[ln.spec.group]['auc_all_strata']:.4f} F {ev[ln.spec.group]['auc_F']:.4f} r2 {ev[ln.spec.group]['r2_reg']:.4f}", flush=True)
                json.dump({"run": run, "curve": curve, "final": res[run["name"]]}, open(f"{x.out}/diag_{run['name']}.json", "w"), indent=1)
            print(f"ZZDIAGRUN_DONE {run['name']} {time.time() - t0:.0f} s", flush=True)
            json.dump(res, open(f"{x.out}/diag_summary.json", "w"), indent=1)
            del tr, provider
            torch.cuda.empty_cache()
        json.dump(res, open(f"{x.out}/diag_summary.json", "w"), indent=1)
        pc = passing_cells(res)
        print("ZZDIAG3_PASSING " + json.dumps(pc), flush=True)
        print("ZZDIAG_OK " + json.dumps({r: {t: {k: v for k, v in rec.items() if k not in ("attn",)} for t, rec in d.items()} for r, d in res.items()}), flush=True)
        return 0
    rc = GL.run_locked_inprocess(body, max_wait=x.max_wait, min_avail_gb=x.min_avail_gb, owner="sh5_diag", job="sh5-creach-diag3")
    print(f"ZZDIAG_EXIT {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
