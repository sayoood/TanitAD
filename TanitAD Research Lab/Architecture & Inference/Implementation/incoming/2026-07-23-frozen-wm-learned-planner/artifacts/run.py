"""Frozen-WM planner experiment — STAGE 2: train planners on the FROZEN WM.

Three arms share one frozen v1 world model (encoder+readout+predictor+step-
readout all requires_grad=False) and the SAME cached per-frame states:

  F  frozen-encoder DIRECT trajectory decode   planner(state)->20 waypoints, ADE
     -> the frozen-ENCODER ceiling on v1's grounded state (REF-A regime probe)
  W  analytic-gradient through the frozen WM    planner(state)->20 actions ->
     rollout_decode(frozen predictor+readout) -> 20 waypoints, ADE.
     Gradient of ADE backprops THROUGH the frozen dynamics into the planner.
     -> the mission's core mechanism (Dreamer/SHAC analytic-gradient family)
  B  action behaviour-cloning                   planner(state)->20 actions, MSE
     to GT actions (NO WM in the loss); eval by rolling through the frozen WM.
     -> ablation isolating the analytic-gradient mechanism vs plain action BC

Eval (open-loop ADE@2s = mean L2 over waypoints [5,10,15,20], driving.py def):
  - each arm on the held-out val episodes
  - ORACLE-ACTION ceiling: roll GT future actions through the frozen WM (this is
    v1's own operative number, matched on THESE windows) = the frozen WM's
    action->trajectory fidelity bound
  - CV floor + hold-v0 (driving_diagnostic.baseline_waypoints)
Stats: episode-cluster bootstrap over the val episodes (program estimator).
"""
import sys, json, time, argparse, math
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from taniteval.loaders import load
from tanitad.models.metric_dynamics import rollout_decode, gt_ego_waypoints
from driving_diagnostic import WP_STEPS, baseline_waypoints

DEV = "cuda"; SPEED_SCALE = 10.0; WINDOW = 8; K = 20; STRIDE = 8
WP_IDX = torch.tensor([k - 1 for k in WP_STEPS], device=DEV)      # [4,9,14,19]
CACHE = Path("/root/frozenwm/cache")


# --------------------------------------------------------------------------- #
def build_split(split):
    """Load cached episodes -> big window tensors on GPU.
    Returns dict of ST[N,8,2048] AW[N,8,2] FA[N,20,2] PL[N,4] FP[N,20,4]
    GTWP[N,20,2] EID[N] (episode index)."""
    files = sorted((CACHE / split).glob("ep_*.pt"))
    ST, AW, FA, PL, FP, EID = [], [], [], [], [], []
    for ei, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        s, a, p = d["states"].float(), d["actions"].float(), d["poses"].float()
        T = min(s.shape[0], a.shape[0], p.shape[0])
        for t in range(0, T - WINDOW - K, STRIDE):
            ST.append(s[t:t+WINDOW]); AW.append(a[t:t+WINDOW])
            FA.append(a[t+WINDOW:t+WINDOW+K]); PL.append(p[t+WINDOW-1])
            FP.append(p[t+WINDOW:t+WINDOW+K]); EID.append(ei)
    ST = torch.stack(ST).to(DEV); AW = torch.stack(AW).to(DEV)
    FA = torch.stack(FA).to(DEV); PL = torch.stack(PL).to(DEV)
    FP = torch.stack(FP).to(DEV)
    GTWP = gt_ego_waypoints(PL, FP, range(1, K + 1))                 # [N,20,2]
    EID = torch.tensor(EID, device=DEV)
    return dict(ST=ST, AW=AW, FA=FA, PL=PL, FP=FP, GTWP=GTWP, EID=EID,
                n=ST.shape[0], neps=len(files))


class Planner(nn.Module):
    """Small learned planner: 8 frozen window states -> out_steps*out_dim.
    2-layer transformer over the window tokens, last-token readout, MLP head."""
    def __init__(self, state_dim=2048, d=384, out_steps=K, out_dim=2, layers=2):
        super().__init__()
        self.proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, 6, 1024, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(),
                                  nn.Linear(512, out_steps * out_dim))
        self.out_steps, self.out_dim = out_steps, out_dim

    def forward(self, states):                                       # [b,8,2048]
        x = self.proj(states) + self.pos
        h = self.tr(x)[:, -1]                                        # last token
        return self.head(h).view(-1, self.out_steps, self.out_dim)


def append_v0(a2, pose_last):
    """[b,H,2] steer/accel + constant v0 channel -> [b,H,3] (matches append_ego)."""
    v0 = (pose_last[:, 3:4] / SPEED_SCALE)[:, None].expand(-1, a2.shape[1], -1)
    return torch.cat([a2, v0], -1)


def roll(predictor, step_readout, ST, AW, fut_a2, PL):
    """Roll frozen predictor under [steer,accel]+v0 future actions -> [b,20,2]."""
    aw3 = append_v0(AW, PL); fa3 = append_v0(fut_a2, PL)
    wp, _ = rollout_decode(predictor, ST, aw3, fa3, step_readout, K)
    return wp


def ade2s(wp, gt):        # mean L2 over the 4 waypoints, per window [b]
    return (wp.index_select(1, WP_IDX) - gt.index_select(1, WP_IDX)).norm(dim=-1).mean(1)


def fde2s(wp, gt):
    return (wp[:, K-1] - gt[:, K-1]).norm(dim=-1)


def ep_bootstrap(perwin, eid, neps, B=2000, seed=0):
    """Episode-cluster bootstrap: resample episodes, mean of per-window values."""
    g = torch.Generator().manual_seed(seed)
    groups = [perwin[eid == e] for e in range(neps)]
    groups = [x for x in groups if x.numel() > 0]
    means = []
    for _ in range(B):
        idx = torch.randint(0, len(groups), (len(groups),), generator=g)
        means.append(torch.cat([groups[i] for i in idx]).mean().item())
    means.sort()
    return means[int(0.025*B)], means[int(0.975*B)]


def paired_boot(a, b, eid, neps, B=2000, seed=0):
    """Paired episode-cluster bootstrap of (mean a - mean b) on the SAME windows.
    Resample episodes; within each resample take BOTH arms' windows for those
    episodes. Returns (point_delta, ci_lo, ci_hi, frac_delta>0)."""
    g = torch.Generator().manual_seed(seed)
    ga = [a[eid == e] for e in range(neps)]; gb = [b[eid == e] for e in range(neps)]
    keep = [i for i in range(neps) if ga[i].numel() > 0]
    ga = [ga[i] for i in keep]; gb = [gb[i] for i in keep]
    d = []
    for _ in range(B):
        idx = torch.randint(0, len(ga), (len(ga),), generator=g)
        A = torch.cat([ga[i] for i in idx]).mean().item()
        Bm = torch.cat([gb[i] for i in idx]).mean().item()
        d.append(A - Bm)
    d.sort()
    pos = sum(x > 0 for x in d) / B
    return (round(a.mean().item()-b.mean().item(), 4),
            round(d[int(0.025*B)], 4), round(d[int(0.975*B)], 4), round(pos, 3))


def train_arm(arm, tr, va, predictor, step_readout, steps, bs, lr, log):
    out_dim = 2
    torch.manual_seed({"F": 1, "W": 2, "B": 3}.get(arm, 0))   # reproducible init
    pl = Planner(out_dim=out_dim).to(DEV)
    opt = torch.optim.AdamW(pl.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    N = tr["n"]; g = torch.Generator(device=DEV).manual_seed(0)
    t0 = time.time()
    for it in range(steps):
        idx = torch.randint(0, N, (bs,), generator=g, device=DEV)
        ST = tr["ST"][idx]; AW = tr["AW"][idx]; PL = tr["PL"][idx]
        GTWP = tr["GTWP"][idx]; FA = tr["FA"][idx]
        pred = pl(ST)
        if arm == "F":                       # direct trajectory decode
            loss = (pred - GTWP).norm(dim=-1).mean()
        elif arm == "B":                     # action BC (no WM)
            loss = (pred - FA).pow(2).mean()
        elif arm == "W":                     # analytic gradient through frozen WM
            wp = roll(predictor, step_readout, ST, AW, pred, PL)
            loss = (wp - GTWP).norm(dim=-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pl.parameters(), 1.0)
        opt.step(); sched.step()
        if (it+1) % log == 0 or it == 0:
            print(f"  [{arm}] {it+1}/{steps} loss={loss.item():.4f} "
                  f"{ (time.time()-t0):.0f}s", flush=True)
    # ---- eval on val ----
    pl.eval()
    with torch.no_grad():
        pred = pl(va["ST"])
        if arm == "F":
            wp = pred
        else:
            wp = roll(predictor, step_readout, va["ST"], va["AW"], pred, va["PL"])
        a = ade2s(wp, va["GTWP"]); f = fde2s(wp, va["GTWP"])
    lo, hi = ep_bootstrap(a, va["EID"], va["neps"])
    r = dict(arm=arm, val_ade2s=round(a.mean().item(), 4),
             val_ade2s_ci=[round(lo, 4), round(hi, 4)],
             val_fde2s=round(f.mean().item(), 4),
             miss2m=round((f > 2.0).float().mean().item(), 4),
             n_val=int(va["n"]), n_train=int(tr["n"]),
             params=sum(p.numel() for p in pl.parameters()))
    return r, a.detach().cpu(), f.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="F,W,B")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log", type=int, default=500)
    ap.add_argument("--out", default="/root/frozenwm/results.json")
    args = ap.parse_args()

    entry = dict(key="flagship-30k", arch="flagship-worldmodel",
                 ckpt="/root/models/flagship-30k/ckpt.pt", speed_input=True)
    h = load(entry, device=DEV)
    model, step_readout = h["model"], h["step_readout"]
    predictor = model.predictor
    for p in model.parameters(): p.requires_grad_(False)
    for p in h["grounding"].parameters(): p.requires_grad_(False)
    model.eval()
    print("frozen WM loaded (step", h["step"], ")", flush=True)

    tr = build_split("train"); va = build_split("val")
    print(f"train windows {tr['n']} ({tr['neps']} eps) | "
          f"val windows {va['n']} ({va['neps']} eps)", flush=True)

    # ---- reference numbers on the SAME val windows ----
    with torch.no_grad():
        wp_or = roll(predictor, step_readout, va["ST"], va["AW"], va["FA"], va["PL"])
        or_a = ade2s(wp_or, va["GTWP"]); or_f = fde2s(wp_or, va["GTWP"])
    or_lo, or_hi = ep_bootstrap(or_a, va["EID"], va["neps"])
    # CV + hold-v0 baselines (per episode, matched windows)
    cv_list, hv_list, e_list = [], [], []
    for f in sorted((CACHE / "val").glob("ep_*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        p = d["poses"].float(); T = p.shape[0]
        starts = list(range(0, T - WINDOW - K, STRIDE))
        last = torch.tensor([t + WINDOW - 1 for t in starts])
        bw = baseline_waypoints(p, last)
        gtp = torch.stack([p[s+WINDOW:s+WINDOW+K] for s in starts])
        gt4 = gt_ego_waypoints(p[last], gtp, WP_STEPS)               # [b,4,2]
        cv_list.append((bw["constant_velocity"] - gt4).norm(dim=-1).mean(1))
        hv_list.append((bw["go_straight"] - gt4).norm(dim=-1).mean(1))   # hold-v0 proxy
    cv = torch.cat(cv_list)
    refs = dict(oracle_action_ade2s=round(or_a.mean().item(), 4),
                oracle_action_ade2s_ci=[round(or_lo, 4), round(or_hi, 4)],
                oracle_action_fde2s=round(or_f.mean().item(), 4),
                oracle_action_miss2m=round((or_f > 2.0).float().mean().item(), 4),
                cv_ade2s=round(cv.mean().item(), 4),
                holdv0_ade2s=(round(torch.cat(hv_list).mean().item(), 4)
                              if hv_list else None))
    print("REFS:", json.dumps(refs), flush=True)

    results = {"refs": refs, "arms": {}, "config": vars(args),
               "val_eps": va["neps"], "train_eps": tr["neps"]}
    pw = {"oracle": or_a.cpu(), "cv": cv.cpu(),
          "holdv0": torch.cat(hv_list).cpu(), "eid": va["EID"].cpu()}
    for arm in args.arms.split(","):
        bs = 24 if arm == "W" else args.bs        # W: deeper graph, smaller batch
        print(f"=== ARM {arm} (bs={bs}) ===", flush=True)
        r, a_pw, f_pw = train_arm(arm, tr, va, predictor, step_readout,
                                  args.steps, bs, args.lr, args.log)
        results["arms"][arm] = r; pw[arm] = a_pw
        print("RESULT:", json.dumps(r), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    # ---- paired episode-cluster bootstraps (same windows) ----
    torch.save(pw, "/root/frozenwm/perwin.pt")            # persist BEFORE stats
    eid = va["EID"].cpu(); ne = va["neps"]
    def pb(x, y):
        return paired_boot(pw[x], pw[y], eid, ne)
    pairs = {}
    A = set(pw.keys())
    for x, y in [("W", "cv"), ("W", "B"), ("W", "oracle"), ("W", "F"),
                 ("W", "holdv0"), ("B", "cv"), ("F", "cv")]:
        if x in A and y in A:
            d, lo, hi, pos = pb(x, y)
            pairs[f"{x}_minus_{y}"] = dict(delta=d, ci=[lo, hi], frac_gt0=pos,
                                           separated=(lo > 0 or hi < 0))
    results["paired"] = pairs
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("PAIRED:", json.dumps(pairs), flush=True)
    print("ALL_ARMS_DONE", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
