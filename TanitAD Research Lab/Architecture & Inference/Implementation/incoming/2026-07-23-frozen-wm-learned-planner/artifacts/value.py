"""Learned value/cost model + CEM search over the frozen v1 WM (the crux test).

CEM search with the GT future as its cost hits 0.132. A DEPLOYABLE contender needs
that search at TEST time WITHOUT the GT future. So: train a value model
V(state, rolled-trajectory) -> predicted rollout-cost (supervised by the ACTUAL
cost on train), then run CEM ranking candidates by the LEARNED V (no GT). Measure
the selected plan's TRUE open-loop ADE on the 12-ep val.

Pre-registered: V-search <= 0.45 -> frozen-WM+search-via-learned-value is a viable
flagship CONTENDER · ~0.599 -> V's error re-introduces the aleatoric gap, frozen-WM
stays the fallback · in-between -> report the fraction closed.

Diagnostic (explains the result): within-window rank-correlation between V's
predicted cost and the true cost across candidates — a value model can only learn
E[cost|state] (mean future), so if the corr is low, V cannot do the per-window
future-matching that makes GT-search reach 0.132.  Frozen WM, cache path only.
"""
import sys, json, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, "/root/frozenwm")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from run import (build_split, roll, ade2s, fde2s, ep_bootstrap, paired_boot,
                 WP_IDX, K, DEV)
from taniteval.loaders import load


class ValueModel(nn.Module):
    """V(z_last, rolled-waypoints) -> predicted rollout cost (ADE-to-future)."""
    def __init__(self, state_dim=2048, d=256):
        super().__init__()
        self.z = nn.Linear(state_dim, d)
        self.w = nn.Linear(K * 2, d)
        self.net = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, z_last, wp):
        h = torch.cat([self.z(z_last), self.w(wp.reshape(wp.shape[0], -1))], -1)
        return self.net(h).squeeze(-1)


@torch.no_grad()
def cem_run(predictor, sr, sp, P, I, act_std, mode, V=None, chunk=8192, seed=0,
            collect=False):
    """Cold-init CEM. mode='gt' ranks by TRUE ADE (needs GT); mode='v' ranks by the
    learned V (deployable). Returns (best_plan, true_ade_of_best) or, if collect,
    on-distribution (win_idx, rolled_wp, true_ade) tuples for training V."""
    b = sp["ST"].shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    mu = torch.zeros(b, K, 2, device=DEV)
    sig = act_std.view(1, 1, 2).expand(b, K, 2).clone()
    gt4 = sp["GTWP"].index_select(1, WP_IDX)                     # [b,4,2]
    z_last = sp["ST"][:, -1]                                     # [b,2048]
    best = mu.clone(); best_score = torch.full((b,), 1e9, device=DEV)
    ke = max(1, int(P * 0.1))
    WI, WP_, AD = [], [], []
    for it in range(I):
        cand = mu[None] + sig[None] * torch.randn(P, b, K, 2, generator=g, device=DEV)
        cand[0] = mu
        candf = cand.reshape(P * b, K, 2)
        biall = torch.arange(P * b, device=DEV) % b
        wp = torch.empty(P * b, K, 2, device=DEV)
        for s in range(0, P * b, chunk):
            e = min(P * b, s + chunk); bi = biall[s:e]
            wp[s:e] = roll(predictor, sr, sp["ST"][bi], sp["AW"][bi], candf[s:e], sp["PL"][bi])
        true_ade = (wp.index_select(1, WP_IDX) - gt4[biall]).norm(dim=-1).mean(1)   # [P*b]
        score = true_ade if mode == "gt" else V(z_last[biall], wp)
        if collect:
            WI.append(biall.cpu()); WP_.append(wp.half().cpu()); AD.append(true_ade.cpu())
        sc = score.reshape(P, b); v, i0 = sc.min(0)
        cb = torch.gather(cand, 0, i0.view(1, b, 1, 1).expand(1, b, K, 2))[0]
        imp = v < best_score
        best = torch.where(imp[:, None, None], cb, best); best_score = torch.minimum(best_score, v)
        idx = sc.topk(ke, dim=0, largest=False).indices
        ec = torch.gather(cand, 0, idx[..., None, None].expand(-1, -1, K, 2))
        mu = ec.mean(0); sig = (ec.std(0).clamp_min(1e-4)) * 0.8
    if collect:
        return torch.cat(WI), torch.cat(WP_), torch.cat(AD)
    return best, ade2s(roll(predictor, sr, sp["ST"], sp["AW"], best, sp["PL"]), sp["GTWP"])


@torch.no_grad()
def rank_corr(V, predictor, sr, sp, P, act_std, chunk=8192, seed=1):
    """Within-window Pearson corr between V's predicted cost and the TRUE cost across
    P candidates — the key diagnostic. High -> V ranks by true cost (can beat mean);
    low -> V only knows E[cost] (mean future) -> bounded by the aleatoric wall."""
    b = sp["ST"].shape[0]; g = torch.Generator(device=DEV).manual_seed(seed)
    z_last = sp["ST"][:, -1]; gt4 = sp["GTWP"].index_select(1, WP_IDX)
    cand = (act_std.view(1, 1, 2) * torch.randn(P, b, K, 2, generator=g, device=DEV))
    candf = cand.reshape(P * b, K, 2); biall = torch.arange(P * b, device=DEV) % b
    wp = torch.empty(P * b, K, 2, device=DEV)
    for s in range(0, P * b, chunk):
        e = min(P * b, s + chunk); bi = biall[s:e]
        wp[s:e] = roll(predictor, sr, sp["ST"][bi], sp["AW"][bi], candf[s:e], sp["PL"][bi])
    true = (wp.index_select(1, WP_IDX) - gt4[biall]).norm(dim=-1).mean(1).reshape(P, b)
    pred = V(z_last[biall], wp).reshape(P, b)
    tc = true - true.mean(0); pc = pred - pred.mean(0)
    corr = (tc * pc).sum(0) / (tc.norm(dim=0) * pc.norm(dim=0)).clamp_min(1e-6)   # [b]
    return corr.mean().item(), corr.std().item()


def train_V(Vv, z_all, WI, WP_, AD, steps, bs, lr, log):
    opt = torch.optim.AdamW(Vv.parameters(), lr=lr, weight_decay=1e-4)
    N = AD.shape[0]; g = torch.Generator().manual_seed(0)
    WPf = WP_.float().to(DEV); ADf = AD.to(DEV); WIf = WI.to(DEV); zt = z_all.float().to(DEV)
    ntr = int(0.9 * N); perm = torch.randperm(N, generator=g)
    tr, va = perm[:ntr], perm[ntr:]
    for it in range(steps):
        b = tr[torch.randint(0, ntr, (bs,), generator=g)].to(DEV)
        pred = Vv(zt[WIf[b]], WPf[b]); loss = (pred - ADf[b]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (it + 1) % log == 0:
            with torch.no_grad():
                vb = va.to(DEV); vl = (Vv(zt[WIf[vb]], WPf[vb]) - ADf[vb]).abs().mean().item()
            print(f"    V {it+1}/{steps} train_L1={loss.item():.4f} val_L1={vl:.4f}", flush=True)
    with torch.no_grad():
        vb = va.to(DEV); vl = (Vv(zt[WIf[vb]], WPf[vb]) - ADf[vb]).abs().mean().item()
    return round(vl, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-collect", type=int, default=350)
    ap.add_argument("--P", type=int, default=128)
    ap.add_argument("--I", type=int, default=4)
    ap.add_argument("--v-steps", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--out", default="/root/frozenwm/valuemodel_results.json")
    args = ap.parse_args()

    entry = dict(key="flagship-30k", arch="flagship-worldmodel",
                 ckpt="/root/models/flagship-30k/ckpt.pt", speed_input=True)
    h = load(entry, device=DEV); model = h["model"]; sr = h["step_readout"]; predictor = model.predictor
    for p in model.parameters(): p.requires_grad_(False)
    for p in h["grounding"].parameters(): p.requires_grad_(False)
    model.eval()
    tr = build_split("train"); va = build_split("val")
    act_std = tr["FA"].reshape(-1, 2).std(0).clamp_min(1e-3)
    pw = torch.load("/root/frozenwm/perwin.pt", weights_only=False)
    refs = {"W": round(pw["W"].mean().item(), 4), "oracle": round(pw["oracle"].mean().item(), 4),
            "cv": round(pw["cv"].mean().item(), 4)}
    try:
        pm = torch.load("/root/frozenwm/perwin_mpc.pt", weights_only=False)
        refs["gt_search_warm"] = round(pm["search_warm"].mean().item(), 4)
        refs["gt_search_cold"] = round(pm["search_cold"].mean().item(), 4)
    except Exception as e:
        print("no mpc perwin", e, flush=True)
    eid = va["EID"].cpu(); ne = va["neps"]
    print("REFS:", json.dumps(refs), flush=True)

    # ---- collect on-distribution (cold GT-CEM) training data on train windows ----
    sel = torch.randperm(tr["n"], generator=torch.Generator().manual_seed(0))[:args.n_collect].to(DEV)
    tr_s = {k: (v[sel] if torch.is_tensor(v) and v.shape[:1] == (tr["n"],) else v) for k, v in tr.items()}
    t0 = time.time()
    WI, WP_, AD = cem_run(predictor, sr, tr_s, args.P, args.I, act_std, "gt",
                          chunk=args.chunk, collect=True)
    z_all = tr_s["ST"][:, -1].half().cpu()
    print(f"collected {AD.shape[0]} candidates from {args.n_collect} win, "
          f"true-ADE range [{AD.min():.3f},{AD.max():.3f}] mean {AD.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- train value model ----
    Vv = ValueModel().to(DEV)
    vL1 = train_V(Vv, z_all, WI, WP_, AD, args.v_steps, 512, 1e-3, args.v_steps // 4)
    Vv.eval()
    print(f"V trained, held-out cost-pred L1={vL1}", flush=True)

    # ---- diagnostic: within-window rank corr on VAL ----
    rc_m, rc_s = rank_corr(Vv, predictor, sr, va, args.P, act_std, chunk=args.chunk)
    print(f"within-window rank-corr(V_pred, true_cost) on val: {rc_m:.3f} +/- {rc_s:.3f}", flush=True)

    # ---- test: cold CEM ranked by LEARNED V (no GT) on val ----
    _, a_v = cem_run(predictor, sr, va, args.P, args.I, act_std, "v", V=Vv, chunk=args.chunk)
    a_v = a_v.cpu()
    lo, hi = ep_bootstrap(a_v, eid, ne)
    # matched control: cold CEM ranked by GT on the SAME val (upper bound of this search)
    _, a_gt = cem_run(predictor, sr, va, args.P, args.I, act_std, "gt", chunk=args.chunk)
    a_gt = a_gt.cpu()

    Vr = dict(ade2s=round(a_v.mean().item(), 4), ci=[lo, hi], fde2s=None,
              cold_gt_search=round(a_gt.mean().item(), 4), v_costpred_L1=vL1,
              within_win_rank_corr=round(rc_m, 3))
    paired = {"Vsearch_minus_W": paired_boot(a_v, pw["W"], eid, ne),
              "Vsearch_minus_oracle": paired_boot(a_v, pw["oracle"], eid, ne),
              "Vsearch_minus_coldGT": paired_boot(a_v, a_gt, eid, ne),
              "Vsearch_minus_cv": paired_boot(a_v, pw["cv"], eid, ne)}
    R = {"config": vars(args), "refs": refs, "V_search": Vr, "paired": paired}
    Path(args.out).write_text(json.dumps(R, indent=2))
    torch.save({"Vsearch": a_v, "coldGT": a_gt, "W": pw["W"], "oracle": pw["oracle"], "eid": eid},
               "/root/frozenwm/perwin_value.pt")
    print("V_SEARCH:", json.dumps(Vr), flush=True)
    print("PAIRED:", json.dumps(paired), flush=True)
    print("VALUE_DONE", flush=True); print(json.dumps(R, indent=2), flush=True)


if __name__ == "__main__":
    main()
