"""Bigger feed-forward planner on the frozen v1 WM — capacity sweep.

Question (Sayed's contender call): is the W(0.599)->search(0.132) gap a
FEED-FORWARD CAPACITY limit (a bigger planner closes it) or does it fundamentally
need test-time search? Same frozen v1 WM, same 12-ep val, analytic-gradient
training, better recipe (warmup+cosine, longer). Paired vs W / search / oracle.

Pre-registered: a bigger planner <= 0.45 -> viable DEPLOYABLE feed-forward
flagship contender · stays ~0.599 -> architecturally limited, needs test-time
search (value-model path) / frozen-WM stays the cheap fallback.
"""
import sys, json, time, argparse, math
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

WINDOW = 8


class BigPlanner(nn.Module):
    """Configurable planner: window-state encoder + (query-decoder | MLP) head."""
    def __init__(self, state_dim=2048, d=768, enc=5, dec=3, heads=8, ff=2048,
                 head="query", out_steps=K, out_dim=2):
        super().__init__()
        self.proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        el = nn.TransformerEncoderLayer(d, heads, ff, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc)
        self.head_type = head
        if head == "query":
            self.q = nn.Parameter(torch.zeros(1, out_steps, d)); nn.init.trunc_normal_(self.q, std=0.02)
            dl = nn.TransformerDecoderLayer(d, heads, ff, activation="gelu", batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(dl, dec)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))
        else:
            self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.GELU(),
                                     nn.Linear(ff, out_steps * out_dim))
        self.out_steps, self.out_dim = out_steps, out_dim

    def forward(self, states):
        x = self.enc(self.proj(states) + self.pos)               # [b,8,d]
        if self.head_type == "query":
            q = self.q.expand(x.shape[0], -1, -1)
            return self.out(self.dec(q, x))                       # [b,K,2]
        return self.mlp(x[:, -1]).view(-1, self.out_steps, self.out_dim)


CONFIGS = {
    "wplus": dict(d=384, enc=2, dec=0, heads=6, ff=1024, head="mlp"),      # ~W arch, better recipe
    "med":   dict(d=512, enc=4, dec=2, heads=8, ff=1536, head="query"),
    "large": dict(d=768, enc=5, dec=3, heads=8, ff=2048, head="query"),
    "xl":    dict(d=1024, enc=6, dec=3, heads=8, ff=2048, head="query"),
    # clean capacity test — W's OWN head family (encoder+pool+MLP), just scaled up,
    # so size varies without the query-decoder head confound:
    "mlpbig": dict(d=768, enc=5, dec=0, heads=8, ff=2048, head="mlp"),
    "mlpwide": dict(d=1024, enc=4, dec=0, heads=8, ff=2560, head="mlp"),
}


def train_bp(cfg, tr, va, predictor, sr, steps, bs, lr, warmup, seed, log):
    torch.manual_seed(seed)
    pl = BigPlanner(**cfg).to(DEV)
    np_ = sum(p.numel() for p in pl.parameters())
    opt = torch.optim.AdamW(pl.parameters(), lr=lr, weight_decay=1e-4)
    def lam(s):
        if s < warmup: return (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup); return 0.5 * (1 + math.cos(math.pi * p))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    N = tr["n"]; g = torch.Generator(device=DEV).manual_seed(seed); t0 = time.time()
    tr_loss = 0.0
    for it in range(steps):
        idx = torch.randint(0, N, (bs,), generator=g, device=DEV)
        wp = roll(predictor, sr, tr["ST"][idx], tr["AW"][idx], pl(tr["ST"][idx]), tr["PL"][idx])
        loss = (wp - tr["GTWP"][idx]).norm(dim=-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pl.parameters(), 1.0); opt.step(); sch.step()
        tr_loss = loss.item()
        if log and (it + 1) % log == 0:
            print(f"    {it+1}/{steps} loss={tr_loss:.4f} {time.time()-t0:.0f}s", flush=True)
    pl.eval()
    with torch.no_grad():
        wp = roll(predictor, sr, va["ST"], va["AW"], pl(va["ST"]), va["PL"])
        a = ade2s(wp, va["GTWP"]).cpu(); f = fde2s(wp, va["GTWP"]).cpu()
    lo, hi = ep_bootstrap(a, va["EID"].cpu(), va["neps"])
    return a, f, dict(ade2s=round(a.mean().item(), 4), ci=[lo, hi],
                      fde2s=round(f.mean().item(), 4), miss2m=round((f > 2.0).float().mean().item(), 4),
                      params=int(np_), train_loss=round(tr_loss, 4), cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="large,med,xl,wplus")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--log", type=int, default=1000)
    ap.add_argument("--out", default="/root/frozenwm/bigplanner_results.json")
    args = ap.parse_args()

    entry = dict(key="flagship-30k", arch="flagship-worldmodel",
                 ckpt="/root/models/flagship-30k/ckpt.pt", speed_input=True)
    h = load(entry, device=DEV); model = h["model"]; sr = h["step_readout"]; predictor = model.predictor
    for p in model.parameters(): p.requires_grad_(False)
    for p in h["grounding"].parameters(): p.requires_grad_(False)
    model.eval()
    tr = build_split("train"); va = build_split("val")
    print(f"train {tr['n']}w val {va['n']}w", flush=True)

    pw = torch.load("/root/frozenwm/perwin.pt", weights_only=False)         # W, oracle, cv, holdv0
    refs = {"W": round(pw["W"].mean().item(), 4), "oracle": round(pw["oracle"].mean().item(), 4),
            "cv": round(pw["cv"].mean().item(), 4), "holdv0": round(pw["holdv0"].mean().item(), 4)}
    try:
        pm = torch.load("/root/frozenwm/perwin_mpc.pt", weights_only=False)
        pw["search_warm"] = pm["search_warm"]; refs["search_warm"] = round(pm["search_warm"].mean().item(), 4)
    except Exception as e:
        print("no mpc perwin:", e, flush=True)
    eid = va["EID"]; ne = va["neps"]
    print("REFS:", json.dumps(refs), flush=True)

    R = {"config": vars(args), "refs": refs, "arms": {}, "paired": {}}
    for arm in args.arms.split(","):
        cfg = CONFIGS[arm]
        bs = args.bs if arm != "xl" else min(args.bs, 16)
        print(f"=== {arm} {cfg} bs={bs} ===", flush=True)
        a, f, r = train_bp(cfg, tr, va, predictor, sr, args.steps, bs, args.lr,
                           args.warmup, 42, args.log)
        R["arms"][arm] = r; pw[arm] = a
        for y in ("W", "oracle", "cv", "search_warm"):
            if y in pw:
                R["paired"].setdefault(arm, {})[f"minus_{y}"] = paired_boot(a, pw[y], eid.cpu(), ne)
        print(f"RESULT {arm}: ADE={r['ade2s']} ci={r['ci']} params={r['params']/1e6:.1f}M "
              f"train_loss={r['train_loss']}", flush=True)
        print(f"  vs W: {json.dumps(R['paired'][arm].get('minus_W'))}", flush=True)
        Path(args.out).write_text(json.dumps(R, indent=2))
        torch.save({k: pw[k] for k in list(CONFIGS) + ["W", "oracle", "cv", "holdv0", "search_warm", "eid"]
                    if k in pw} | {"eid": eid.cpu()}, "/root/frozenwm/perwin_big.pt")
    print("BIG_DONE", flush=True); print(json.dumps(R, indent=2), flush=True)


if __name__ == "__main__":
    main()
