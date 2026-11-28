"""40-ep hardening of the frozen-WM feed-forward planner (arm W) on the eval pod.

The eval pod has ONLY the clean 40-ep physicalai val (no train corpus), so W is
trained by EPISODE-DISJOINT k-fold CV within the 40 val episodes (train on k-1
folds, predict the held-out fold, pool all held-out predictions -> a decision-
grade held-out ADE for every one of the 40 episodes / 881 windows). Controls
(oracle-action ceiling, CV, hold-v0) are deterministic -> computed on all 40.
Frozen v1 WM (encoder+predictor+step-readout), read-only. Data kept in memory.
"""
import sys, json, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from taniteval.loaders import load
from tanitad.data.mixing import load_episode
from tanitad.models.metric_dynamics import rollout_decode, gt_ego_waypoints
from driving_diagnostic import WP_STEPS, baseline_waypoints

DEV = "cuda"; SPEED_SCALE = 10.0; WINDOW = 8; K = 20; STRIDE = 8
WP_IDX = torch.tensor([k - 1 for k in WP_STEPS], device=DEV)
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"


class Planner(nn.Module):
    def __init__(self, state_dim=2048, d=384, out_steps=K, out_dim=2, layers=2):
        super().__init__()
        self.proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, 6, 1024, activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Linear(512, out_steps * out_dim))
        self.out_steps, self.out_dim = out_steps, out_dim

    def forward(self, states):
        x = self.proj(states) + self.pos
        return self.head(self.tr(x)[:, -1]).view(-1, self.out_steps, self.out_dim)


def append_v0(a2, pl):
    v0 = (pl[:, 3:4] / SPEED_SCALE)[:, None].expand(-1, a2.shape[1], -1)
    return torch.cat([a2, v0], -1)


def roll(predictor, sr, ST, AW, fut, PL):
    wp, _ = rollout_decode(predictor, ST, append_v0(AW, PL), append_v0(fut, PL), sr, K)
    return wp


def ade2s(wp, gt):
    return (wp.index_select(1, WP_IDX) - gt.index_select(1, WP_IDX)).norm(dim=-1).mean(1)


def fde2s(wp, gt):
    return (wp[:, K - 1] - gt[:, K - 1]).norm(dim=-1)


def ep_bootstrap(pw, eid, neps, B=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    groups = [pw[eid == e] for e in range(neps)]; groups = [x for x in groups if x.numel() > 0]
    m = []
    for _ in range(B):
        idx = torch.randint(0, len(groups), (len(groups),), generator=g)
        m.append(torch.cat([groups[i] for i in idx]).mean().item())
    m.sort(); return round(m[int(.025 * B)], 4), round(m[int(.975 * B)], 4)


def paired_boot(a, b, eid, neps, B=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    ga = [a[eid == e] for e in range(neps)]; gb = [b[eid == e] for e in range(neps)]
    keep = [i for i in range(neps) if ga[i].numel() > 0]
    ga = [ga[i] for i in keep]; gb = [gb[i] for i in keep]; d = []
    for _ in range(B):
        idx = torch.randint(0, len(ga), (len(ga),), generator=g)
        d.append(torch.cat([ga[i] for i in idx]).mean().item() - torch.cat([gb[i] for i in idx]).mean().item())
    d.sort(); lo, hi = d[int(.025 * B)], d[int(.975 * B)]
    return dict(delta=round(a.mean().item() - b.mean().item(), 4), ci=[round(lo, 4), round(hi, 4)],
                frac_gt0=round(sum(x > 0 for x in d) / B, 3), separated=(lo > 0 or hi < 0))


def encode_all(model):
    files = sorted(Path(VAL).glob("ep_*.pt"))
    ST, AW, FA, PL, FP, EID = [], [], [], [], [], []
    CV, HV = [], []
    for ei, f in enumerate(files):
        ep = load_episode(str(f), mmap=True)
        T = min(ep.frames.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        st = []
        for i in range(0, T, 64):
            fr = ep.frames[i:i + 64].to(DEV).float().div_(255.0)
            with torch.no_grad():
                st.append(model.encode(fr).half().cpu())
        s = torch.cat(st)[:T]; a = ep.actions[:T].float(); p = ep.poses[:T].float()
        starts = list(range(0, T - WINDOW - K, STRIDE))
        last = torch.tensor([t + WINDOW - 1 for t in starts])
        bw = baseline_waypoints(p, last)
        gtp = torch.stack([p[t + WINDOW:t + WINDOW + K] for t in starts])
        gt4 = gt_ego_waypoints(p[last], gtp, WP_STEPS)
        CV.append((bw["constant_velocity"] - gt4).norm(dim=-1).mean(1))
        HV.append((bw["go_straight"] - gt4).norm(dim=-1).mean(1))
        for t in starts:
            ST.append(s[t:t + WINDOW]); AW.append(a[t:t + WINDOW]); FA.append(a[t + WINDOW:t + WINDOW + K])
            PL.append(p[t + WINDOW - 1]); FP.append(p[t + WINDOW:t + WINDOW + K]); EID.append(ei)
    ST = torch.stack(ST).float().to(DEV); AW = torch.stack(AW).to(DEV); FA = torch.stack(FA).to(DEV)
    PL = torch.stack(PL).to(DEV); FP = torch.stack(FP).to(DEV)
    GTWP = gt_ego_waypoints(PL, FP, range(1, K + 1))
    EID = torch.tensor(EID)
    return dict(ST=ST, AW=AW, FA=FA, PL=PL, GTWP=GTWP, EID=EID, n=ST.shape[0],
                neps=len(files), CV=torch.cat(CV), HV=torch.cat(HV))


def train_W(d, tr_mask, te_mask, predictor, sr, steps, bs, lr, seed, log):
    torch.manual_seed(seed)
    pl = Planner().to(DEV); opt = torch.optim.AdamW(pl.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    idx_tr = tr_mask.nonzero(as_tuple=True)[0].to(DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for it in range(steps):
        b = idx_tr[torch.randint(0, idx_tr.numel(), (bs,), generator=g, device=DEV)]
        wp = roll(predictor, sr, d["ST"][b], d["AW"][b], pl(d["ST"][b]), d["PL"][b])
        loss = (wp - d["GTWP"][b]).norm(dim=-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pl.parameters(), 1.0); opt.step(); sch.step()
        if log and (it + 1) % log == 0:
            print(f"    step {it+1}/{steps} loss={loss.item():.4f}", flush=True)
    pl.eval()
    te = te_mask.nonzero(as_tuple=True)[0].to(DEV)
    with torch.no_grad():
        wp = roll(predictor, sr, d["ST"][te], d["AW"][te], pl(d["ST"][te]), d["PL"][te])
        a = ade2s(wp, d["GTWP"][te]); f = fde2s(wp, d["GTWP"][te])
    return te.cpu(), a.cpu(), f.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-eps", type=int, default=0)   # 0=all; smoke uses small
    ap.add_argument("--log", type=int, default=0)
    ap.add_argument("--out", default="/root/frozenwm40/results_40ep.json")
    args = ap.parse_args()

    entry = dict(key="flagship-30k", arch="flagship-worldmodel",
                 ckpt="/root/models/flagship-30k/ckpt.pt", speed_input=True)
    h = load(entry, device=DEV); model = h["model"]; sr = h["step_readout"]; predictor = model.predictor
    for p in model.parameters(): p.requires_grad_(False)
    for p in h["grounding"].parameters(): p.requires_grad_(False)
    model.eval()
    print("frozen WM loaded, step", h["step"], flush=True)

    t0 = time.time(); d = encode_all(model)
    print(f"encoded {d['neps']} eps -> {d['n']} windows ({time.time()-t0:.0f}s)", flush=True)
    eid = d["EID"]; ne = d["neps"]

    # ---- deterministic controls + oracle-action ceiling on ALL 40 eps ----
    with torch.no_grad():
        wp_or = roll(predictor, sr, d["ST"], d["AW"], d["FA"], d["PL"])
        or_a = ade2s(wp_or, d["GTWP"]).cpu(); or_f = fde2s(wp_or, d["GTWP"]).cpu()
    refs = {}
    for name, arr in (("oracle", or_a), ("cv", d["CV"].cpu()), ("holdv0", d["HV"].cpu())):
        lo, hi = ep_bootstrap(arr, eid, ne)
        refs[name] = dict(ade2s=round(arr.mean().item(), 4), ci=[lo, hi])
    refs["oracle"]["fde2s"] = round(or_f.mean().item(), 4)
    refs["oracle"]["miss2m"] = round((or_f > 2.0).float().mean().item(), 4)
    print("REFS (40ep):", json.dumps(refs), flush=True)

    # ---- W via episode-disjoint k-fold CV ----
    W_a = torch.empty(d["n"]); W_f = torch.empty(d["n"])
    for fold in range(args.folds):
        te_mask = (eid % args.folds == fold); tr_mask = ~te_mask
        t1 = time.time()
        te, a, f = train_W(d, tr_mask, te_mask, predictor, sr, args.steps, args.bs,
                           args.lr, 100 + fold, args.log)
        W_a[te] = a; W_f[te] = f
        print(f"  [fold {fold}] test_eps={int((eid%args.folds==fold).sum()//1)} "
              f"n_te={te.numel()} ADE={a.mean().item():.4f} ({time.time()-t1:.0f}s)", flush=True)
    lo, hi = ep_bootstrap(W_a, eid, ne)
    Wr = dict(ade2s=round(W_a.mean().item(), 4), ci=[lo, hi],
              fde2s=round(W_f.mean().item(), 4), miss2m=round((W_f > 2.0).float().mean().item(), 4),
              n=d["n"], neps=ne, folds=args.folds, protocol="episode-disjoint k-fold CV on 40-ep val")

    # ---- paired bootstraps (same windows) ----
    paired = {}
    for y, arr in (("oracle", or_a), ("cv", d["CV"].cpu()), ("holdv0", d["HV"].cpu())):
        paired[f"W_minus_{y}"] = paired_boot(W_a, arr, eid, ne)

    R = {"config": vars(args), "refs": refs, "W": Wr, "paired": paired, "neps": ne}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=2))
    torch.save({"W_a": W_a, "W_f": W_f, "oracle": or_a, "cv": d["CV"].cpu(),
                "holdv0": d["HV"].cpu(), "eid": eid},
               str(Path(args.out).parent / "perwin_40ep.pt"))
    print("W_40EP:", json.dumps(Wr), flush=True)
    print("PAIRED:", json.dumps(paired), flush=True)
    print("DONE_40EP", flush=True); print(json.dumps(R, indent=2), flush=True)


if __name__ == "__main__":
    main()
