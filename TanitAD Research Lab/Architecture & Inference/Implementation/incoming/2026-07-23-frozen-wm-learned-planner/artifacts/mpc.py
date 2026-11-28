"""Amortised-MPC prototype on the FROZEN v1 WM (TD-MPC2 shape).

(1) CEM/MPC search over the frozen WM to find, per window, the action sequence
    the WM maps to the lowest ADE-to-expert (teacher; uses GT as the imitation
    cost, like any imitation target).  Reported cold-init (search from scratch)
    AND warm-init (refine the GT actions -> strongest targets).
(2) Distil the warm search's found actions into a fast feed-forward PRIOR
    (same 3.77M planner as arm W), which at test runs feed-forward, no GT, no
    search -> the DEPLOYABLE amortised-MPC planner.
(3) Eval on the SAME 12-ep val, open-loop ADE@2s, PAIRED episode-cluster
    bootstrap vs arm W (0.599) and the oracle-action ceiling (0.405), read from
    the saved perwin.pt so windows are identically ordered.

Frozen encoder + predictor + step-readout (WM never updated). Cache path only
(no tanitad.lake).
"""
import sys, json, time, argparse
from pathlib import Path
import torch
sys.path.insert(0, "/root/frozenwm")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from run import (build_split, Planner, roll, ade2s, fde2s, append_v0,
                 ep_bootstrap, paired_boot, WP_IDX, K, DEV)   # reuse arm-W code
from taniteval.loaders import load


@torch.no_grad()
def cem_plan(predictor, step_readout, ST, AW, GTWP, PL, P, I, act_std,
             elite=0.1, sig_frac=0.3, init_mu=None, decay=0.8, chunk=4096):
    """CEM/MPPI over future actions [b,K,2] minimising 4-waypoint ADE-to-GT
    through the frozen WM. Elitism (incumbent kept every iter) + best-candidate
    tracking guarantee the returned plan is never worse than the init. sigma is
    action-scaled and annealed. Returns (best_plan [b,K,2], best_ade [b])."""
    b = ST.shape[0]
    warm = init_mu is not None
    mu = init_mu.clone() if warm else torch.zeros(b, K, 2, device=DEV)
    sig = ((sig_frac if warm else 1.0) * act_std.view(1, 1, 2)).expand(b, K, 2).clone()
    gt4 = GTWP.index_select(1, WP_IDX)                       # [b,4,2]
    best = mu.clone(); best_ade = torch.full((b,), float("inf"), device=DEV)
    ke = max(1, int(P * elite))
    for it in range(I):
        cand = mu[None] + sig[None] * torch.randn(P, b, K, 2, device=DEV)
        cand[0] = mu                                         # elitism: keep incumbent
        candf = cand.reshape(P * b, K, 2)
        ade = torch.empty(P * b, device=DEV)
        for s in range(0, P * b, chunk):
            e = min(P * b, s + chunk)
            bi = torch.arange(s, e, device=DEV) % b          # flat idx = p*b + bi
            wp = roll(predictor, step_readout, ST[bi], AW[bi], candf[s:e], PL[bi])
            ade[s:e] = (wp.index_select(1, WP_IDX) - gt4[bi]).norm(dim=-1).mean(1)
        ade = ade.reshape(P, b)
        v, i0 = ade.min(0)                                   # best-this-iter per window
        bc = torch.gather(cand, 0, i0.view(1, b, 1, 1).expand(1, b, K, 2))[0]
        imp = v < best_ade
        best = torch.where(imp[:, None, None], bc, best); best_ade = torch.minimum(best_ade, v)
        idx = ade.topk(ke, dim=0, largest=False).indices     # [ke,b]
        ec = torch.gather(cand, 0, idx[..., None, None].expand(-1, -1, K, 2))
        mu = ec.mean(0); sig = (ec.std(0).clamp_min(1e-4)) * decay
    return best, best_ade


@torch.no_grad()
def eval_actions(predictor, step_readout, sp, actions):
    """Roll a fixed action plan [N,K,2] through the frozen WM on split sp."""
    wp = roll(predictor, step_readout, sp["ST"], sp["AW"], actions, sp["PL"])
    return ade2s(wp, sp["GTWP"]).cpu(), fde2s(wp, sp["GTWP"]).cpu()


def distil_prior(sp_tr, targets, sp_va, steps, bs, lr, predictor, step_readout, log):
    """BC a feed-forward planner to predict the search-found actions; eval on val
    by rolling its feed-forward actions through the frozen WM."""
    torch.manual_seed(7)
    pl = Planner(out_dim=2).to(DEV)
    opt = torch.optim.AdamW(pl.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    N = sp_tr["ST"].shape[0]; g = torch.Generator(device=DEV).manual_seed(0)
    t0 = time.time()
    for it in range(steps):
        idx = torch.randint(0, N, (bs,), generator=g, device=DEV)
        pred = pl(sp_tr["ST"][idx])
        loss = (pred - targets[idx]).pow(2).mean()          # BC on search actions
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pl.parameters(), 1.0); opt.step(); sch.step()
        if (it+1) % log == 0 or it == 0:
            print(f"  [distil] {it+1}/{steps} loss={loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    pl.eval()
    with torch.no_grad():
        wp = roll(predictor, step_readout, sp_va["ST"], sp_va["AW"], pl(sp_va["ST"]), sp_va["PL"])
    return ade2s(wp, sp_va["GTWP"]).cpu(), fde2s(wp, sp_va["GTWP"]).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=256)
    ap.add_argument("--I", type=int, default=5)
    ap.add_argument("--n-train-win", type=int, default=2500)
    ap.add_argument("--distil-steps", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default="/root/frozenwm/mpc_results.json")
    args = ap.parse_args()

    entry = dict(key="flagship-30k", arch="flagship-worldmodel",
                 ckpt="/root/models/flagship-30k/ckpt.pt", speed_input=True)
    h = load(entry, device=DEV); model = h["model"]; sr = h["step_readout"]
    predictor = model.predictor
    for p in model.parameters(): p.requires_grad_(False)
    for p in h["grounding"].parameters(): p.requires_grad_(False)
    model.eval()
    va = build_split("val"); tr = build_split("train")
    print(f"val {va['n']}w/{va['neps']}e  train {tr['n']}w/{tr['neps']}e", flush=True)

    pw = torch.load("/root/frozenwm/perwin.pt", weights_only=False)  # W, oracle aligned
    W_pw, OR_pw, eid = pw["W"], pw["oracle"], pw["eid"]
    assert W_pw.shape[0] == va["n"], f"align mismatch {W_pw.shape} vs {va['n']}"

    R = {"config": vars(args), "refs": {}, "search": {}, "amortised": {}, "paired": {}}
    R["refs"] = {"W_ade2s": round(W_pw.mean().item(), 4),
                 "oracle_ade2s": round(OR_pw.mean().item(), 4)}
    ne = va["neps"]
    act_std = tr["FA"].reshape(-1, 2).std(0).clamp_min(1e-3)      # per-dim action scale
    print("act_std", [round(x, 4) for x in act_std.tolist()], flush=True)

    # ---- (1) search teacher on VAL: cold + warm ----
    for tag, init in (("cold", None), ("warm", va["FA"])):
        t0 = time.time()
        mu, _ = cem_plan(predictor, sr, va["ST"], va["AW"], va["GTWP"], va["PL"],
                         args.P, args.I, act_std, init_mu=init, chunk=args.chunk)
        a, f = eval_actions(predictor, sr, va, mu)
        lo, hi = ep_bootstrap(a, eid, ne)
        R["search"][tag] = dict(ade2s=round(a.mean().item(), 4), ci=[round(lo,4), round(hi,4)],
                                fde2s=round(f.mean().item(), 4),
                                miss2m=round((f>2.0).float().mean().item(),4))
        pw[f"search_{tag}"] = a
        print(f"[search-{tag}] val ADE2s={a.mean().item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        Path(args.out).write_text(json.dumps(R, indent=2))

    # ---- (2) search teacher on TRAIN subset (warm) -> distil targets ----
    n = min(args.n_train_win, tr["n"])
    g = torch.Generator(device=DEV).manual_seed(0)
    sel = torch.randperm(tr["n"], generator=g, device=DEV)[:n]
    tr_s = {k: (v[sel] if torch.is_tensor(v) and v.shape[:1] == (tr["n"],) else v)
            for k, v in tr.items()}
    t0 = time.time()
    targets, tgt_ade = cem_plan(predictor, sr, tr_s["ST"], tr_s["AW"], tr_s["GTWP"],
                                tr_s["PL"], args.P, args.I, act_std,
                                init_mu=tr_s["FA"], chunk=args.chunk)
    print(f"[search-train] {n} windows, teacher ADE={tgt_ade.mean().item():.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- (3) distil prior, eval on val ----
    a, f = distil_prior(tr_s, targets, va, args.distil_steps, 48, 3e-4,
                        predictor, sr, 1000)
    lo, hi = ep_bootstrap(a, eid, ne)
    R["amortised"]["prior"] = dict(ade2s=round(a.mean().item(), 4), ci=[round(lo,4), round(hi,4)],
                                   fde2s=round(f.mean().item(), 4),
                                   miss2m=round((f>2.0).float().mean().item(),4),
                                   n_distil=int(n))
    pw["amortised_prior"] = a
    print(f"[amortised-prior] val ADE2s={a.mean().item():.4f}", flush=True)

    # ---- paired episode-cluster bootstraps vs W and oracle ----
    def pb(x, y):
        d, lo, hi, pos = paired_boot(pw[x], pw[y], eid.cpu(), ne)
        return dict(delta=d, ci=[lo, hi], frac_gt0=pos, separated=(lo>0 or hi<0))
    pw["W"] = W_pw; pw["oracle"] = OR_pw
    for x, y in [("search_warm","W"), ("search_warm","oracle"), ("search_cold","W"),
                 ("amortised_prior","W"), ("amortised_prior","oracle"),
                 ("amortised_prior","search_warm")]:
        R["paired"][f"{x}_minus_{y}"] = pb(x, y)
    torch.save(pw, "/root/frozenwm/perwin_mpc.pt")
    Path(args.out).write_text(json.dumps(R, indent=2))
    print("MPC_DONE", flush=True); print(json.dumps(R, indent=2), flush=True)


if __name__ == "__main__":
    main()
