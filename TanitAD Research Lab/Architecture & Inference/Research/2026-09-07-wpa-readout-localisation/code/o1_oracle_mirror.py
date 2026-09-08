# -*- coding: utf-8 -*-
"""WP-A ORACLE RECHECK -- re-read the READOUT-CEILING ladder under BOTH azimuth senses.

⛔ WHY THIS EXISTS.  `R-2026-09-08-wpa-mirror` retracted WP-A's real-trunk column:
`s5_indexed.py` addresses azimuth with

    u = W/2 + f_ref * az                                   (s5_indexed.py:78)

while `bev_aux.azimuth_column` (`stack/tanitad/data/bev_aux.py:196`) and
`psg_targets.azimuth_column` (`stack/tanitad/data/psg_targets.py:72`) -- two
independently authored implementations that AGREE -- both use

    col = (hfov/2 - az) / daz        ⇒       u = W/2 - f_ref * az

⭐ `s6_oracle.py:81-82` uses the SAME formula as s5, i.e. the ORACLE LADDER CARRIES
THE MIRRORED ADDRESS TOO.  The retraction left the ladder open with the note that a
target-derived feature map *may* be self-consistent under a mirror -- **but that is an
argument, and this row is what happens when an argument is banked as a result.**
This script measures it.

⭐⭐ THE FALSIFIABLE PREDICTION, STATED BEFORE ANY NUMBER WAS SEEN (2026-09-08 19:47Z)
-----------------------------------------------------------------------------------
The two addresses are an EXACT mirror of each other about the column axis:

    colf_prog + colf_wpa = (W/2 - f*az)/P + (W/2 + f*az)/P - 1 = W/P - 1 = TW - 1

and every column-axis operation in the oracle path is equivariant under that mirror:
  * the bilinear column lookup S is exactly mirror-symmetric (floor/ceil weights swap);
  * every width pooling kernel used (1,2,5,10,20,40) DIVIDES TW=40, so the pooling
    partition maps onto itself under c -> TW-1-c, and so does the nearest-upsample;
  * the oracle map is BUILT with the same address the head READS with, so a sign flip
    permutes the intermediate map and un-permutes it at the lookup.
The only asymmetry is the learned 3x3 `mix` conv (a mirrored input is not the mirror of
a conv output unless the kernel is symmetric) -- which is a RELABELLING the head can
learn equally well either way.

⇒ **PREDICTION: the ORACLE ladder is address-sense INVARIANT.**  Both senses must give
   the same ladder to within the run-to-run replicate floor (banked: orc_16x40 spans
   0.4651-0.4773 = 0.0122 AP over train seeds 0/1/2), and the ladder must stay MONOTONE
   DECREASING as the pool coarsens under BOTH senses.
⇒ This is the OPPOSITE prediction from the real-trunk arms, where the mirror cost
   1.57-2.10x and turned the ladder MONOTONE UP.  That is what makes it discriminating:
   if the oracle behaved like the real trunk, the prediction is falsified.

⭐ THE MUTATION CONTROL THAT MAKES THE INVARIANCE MEAN SOMETHING (`xwire`).
An invariance result is worthless if the head simply ignores azimuth.  So the panel
also runs CROSS-WIRED arms: the map BUILT under one sense and READ under the other --
i.e. the real historical defect, deliberately reintroduced.  That MUST collapse toward
`pos_only`.  If it does not, the invariance claim proves nothing and must be discarded.

⛔ AP IS TIE-GROUP SUMMED (`R-2026-09-07-ap-ties`).  WP-A's `s6_oracle.py:182-188`
uses the naive per-sample form, which breaks ties by ARRAY ORDER: its own all-zero
control read 0.016340211 against a test base rate of 0.016197555 -- **+0.881 %, i.e.
the no-information floor read ABOVE the value that defines it.**  Both forms are run
here on the same control so the bias is quantified rather than asserted.

Estimator: paired EPISODE-CLUSTER bootstrap over the 30 test clips
(`taniteval/ci.py` machinery: `episode_index` + `_draws`), AP recomputed inside every
draw.  ⛔ `overlapping_holdout_se` is not used anywhere.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot")

import numpy as np
import torch
from torch import nn

from tanitad.data.bev_raster import BEVGrid, cell_centers_xy, fov_mask

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default=r"C:\Users\Admin\wpa-readout\bank\k8r1")
ap.add_argument("--out", default=r"C:\Users\Admin\wpa-readout\oracle_recheck\raw\oracle_mirror.json")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--cells", type=int, default=2048)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0, help="SPLIT seed -- keep fixed")
ap.add_argument("--train-seeds", default="0,1,2")
ap.add_argument("--h-cam", type=float, default=1.5)
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[env] torch {torch.__version__} dev {dev} "
      f"{torch.cuda.get_device_name(0) if dev=='cuda' else ''}", flush=True)

idx = json.load(open(os.path.join(a.bank, "idx.json"), encoding="utf-8"))
clips = idx["clips"]; plan = np.asarray(idx["plan"], dtype=np.int64)
N = int(idx["N"]); TH, TW = idx["token_grid"]
GH, GW = idx["grid"]; CELL = float(idx["cell_m"])
PATCH, W_PX, H_PX, F_REF = 16, 640, 256, 305.5774907364391
grid = BEVGrid(x_fwd_m=float(idx["x_fwd_m"]), y_half_m=float(idx["y_half_m"]),
               cell_m=CELL)
Yall = np.asarray(np.load(os.path.join(a.bank, "y.npy"), mmap_mode="r"))

# --------------------------------------------------------------------------- #
# AP -- TIE-GROUP SUMMED.  Verbatim from stack/tanitad/refs/refc_bev_aux.py:211 #
# (copied rather than imported: the repo lives on the G: mount, whose editable  #
#  install cannot be imported from this box -- CLAUDE.md's PYTHONPATH trap).    #
# --------------------------------------------------------------------------- #
def AP_tiegroup(score, target) -> float:
    s = np.asarray(score, dtype=np.float64).ravel()
    y = (np.asarray(target).ravel() > 0.5).astype(np.float64)
    n_pos = float(y.sum())
    if n_pos == 0.0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    ends = np.flatnonzero(np.r_[np.diff(s) != 0.0, True])
    prec = tp[ends] / (ends + 1).astype(np.float64)
    d_rec = np.diff(np.r_[0.0, tp[ends]]) / n_pos
    return float((prec * d_rec).sum())


def AP_tiegroup_t(s, y) -> float:
    """The SAME tie-group AP on the GPU -- used only inside the bootstrap, where
    the CPU form costs ~2 s per draw.  ⭐ Cross-checked against `AP_tiegroup`
    on the full test set before any bootstrap number is produced; the tie-run
    boundaries and the precision denominator are computed in float64 because
    `ends+1` reaches 2.1e7, beyond float32's exact-integer range."""
    n_pos = float(y.sum())
    if n_pos == 0.0:
        return float("nan")
    order = torch.argsort(s, descending=True, stable=True)
    ss, yy = s[order], y[order]
    tp = torch.cumsum(yy, 0)                       # <= n_pos, exact in float32
    brk = torch.cat([ss[1:] != ss[:-1],
                     torch.ones(1, dtype=torch.bool, device=s.device)])
    ends = torch.nonzero(brk, as_tuple=False).squeeze(1)
    tpe = tp[ends].to(torch.float64)
    prec = tpe / (ends + 1).to(torch.float64)
    d_rec = torch.cat([tpe[:1], tpe[1:] - tpe[:-1]]) / n_pos
    return float((prec * d_rec).sum())


def AP_naive(score, target) -> float:
    """WP-A's s6_oracle.py:182-188 form, kept ONLY to quantify the tie bias."""
    sc = torch.as_tensor(np.asarray(score, dtype=np.float64).ravel())
    la = torch.as_tensor((np.asarray(target).ravel() > 0.5).astype(np.float64))
    o = torch.argsort(sc, descending=True)
    l = la[o]
    tp = torch.cumsum(l, 0); fp = torch.cumsum(1 - l, 0)
    rec = tp / l.sum()
    dr = torch.diff(rec, prepend=torch.zeros(1))
    return float(((tp / (tp + fp)) * dr).sum())


# --------------------------------------------------------------------------- #
# address: BEV cell -> (token row, token col), under BOTH senses               #
# --------------------------------------------------------------------------- #
X_m, Y_m = cell_centers_xy(grid)
az = np.arctan2(Y_m, X_m)                                # +y LEFT -> +az LEFT
rng_m = np.sqrt(X_m ** 2 + Y_m ** 2)
vpx = H_PX / 2.0 + F_REF * a.h_cam / np.maximum(rng_m, 1e-6)
rowf = vpx / PATCH - 0.5

SIGN = {"prog": -1.0, "wpa": +1.0}   # prog == bev_aux/psg_targets; wpa == s5/s6


def colf_of(sign):
    return (W_PX / 2.0 + sign * F_REF * az) / PATCH - 0.5


in_fov = fov_mask(grid, math.radians(60.0))
valid = in_fov & (rowf >= 0) & (rowf <= TH - 1)
for sg in SIGN.values():                     # ⭐ same cells under BOTH senses
    cf = colf_of(sg)
    valid = valid & (cf >= 0) & (cf <= TW - 1)
vidx = np.flatnonzero(valid.reshape(-1))
NC = len(vidx)
ri = np.clip(np.rint(rowf.reshape(-1)[vidx]), 0, TH - 1).astype(np.int64)

TOKFLAT, S_BY = {}, {}
for nm, sg in SIGN.items():
    cf = colf_of(sg).reshape(-1)[vidx]
    ci = np.clip(np.rint(cf), 0, TW - 1).astype(np.int64)
    TOKFLAT[nm] = torch.from_numpy(ri * TW + ci).to(dev)
    c = np.clip(cf, 0, TW - 1 - 1e-6)
    c0 = np.floor(c).astype(np.int64); w1 = c - c0
    c1 = np.minimum(c0 + 1, TW - 1)
    S = np.zeros((NC, TW), dtype=np.float32)
    S[np.arange(NC), c0] += 1.0 - w1
    S[np.arange(NC), c1] += w1
    S_BY[nm] = torch.from_numpy(S).to(dev)
    u, cnt = np.unique(ri * TW + ci, return_counts=True)
    print(f"[address/{nm:4s}] sign {sg:+.0f}  addressable {NC}/{valid.size} "
          f"({100*valid.mean():.2f} %)  token cells used {len(u)}  "
          f"BEV cells/token cell median {np.median(cnt):.1f} max {cnt.max()}",
          flush=True)

# ⭐ ASSERT the two addresses are the exact mirror the prediction assumes.
mirror_err = float(np.abs(colf_of(-1.0) + colf_of(+1.0) - (TW - 1)).max())
print(f"[mirror] max |colf_prog + colf_wpa - (TW-1)| = {mirror_err:.3e} "
      f"(0 ⇒ exact mirror about the column axis)", flush=True)
assert mirror_err < 1e-9, "the two addresses are NOT an exact mirror"

xs = X_m.reshape(-1)[vidx]; ys = Y_m.reshape(-1)[vidx]; azv = az.reshape(-1)[vidx]
pf = [xs / 60.0, ys / 16.0, np.log1p(xs) / 4.2, azv / 1.05,
      rng_m.reshape(-1)[vidx] / 60.0]
for k in (1, 2, 4, 8):
    pf += [np.sin(k * azv), np.cos(k * azv), np.sin(k * xs / 20.0), np.cos(k * xs / 20.0)]
POS = torch.from_numpy(np.stack(pf, 1).astype(np.float32)).to(dev)
PD = POS.shape[1]

Yv = torch.from_numpy(Yall[:, vidx].copy())
base = float(Yv.float().mean())

# --------------------------------------------------------------------------- #
# the ORACLE maps: GT occupancy scattered into the token grid, one per sense    #
# --------------------------------------------------------------------------- #
ORACLE = {}
_y = Yv.float()
for nm in SIGN:
    O = torch.zeros(N, 2, TH * TW, dtype=torch.float16)
    for s in range(0, N, 512):
        blk = _y[s:s + 512].to(dev)
        m = torch.zeros(blk.shape[0], TH * TW, device=dev)
        m.index_add_(1, TOKFLAT[nm], blk)
        O[s:s + blk.shape[0], 0] = (m > 0).to(torch.float16).cpu()
        O[s:s + blk.shape[0], 1] = torch.log1p(m).to(torch.float16).cpu()
    ORACLE[nm] = O.reshape(N, 2, TH, TW)
    mp = float(ORACLE[nm][:, 0].float().mean())
    print(f"[oracle/{nm:4s}] mean presence {mp:.5f}", flush=True)
    assert mp > 1e-4, "ORACLE MAP IS DEGENERATE"
del _y

# ⭐ the maps must be exact column mirrors of each other (up to rint .5 ties)
_d = (ORACLE["prog"][:, 0].float() - torch.flip(ORACLE["wpa"][:, 0].float(), dims=[2]))
print(f"[mirror] map presence mismatch after column flip: "
      f"{float(_d.abs().mean()):.3e} mean, {int((_d.abs() > 0).sum())} cells "
      f"of {_d.numel()}", flush=True)
del _d

# --------------------------------------------------------------------------- #
# split (identical to s6_oracle.py)                                            #
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(clips))
n_te, n_va = 30, 25
te_c, va_c = set(order[:n_te].tolist()), set(order[n_te:n_te + n_va].tolist())
cl = plan[:, 0]
fi = np.where(~np.isin(cl, list(te_c | va_c)))[0]
vi = np.where(np.isin(cl, list(va_c)))[0]
ti = np.where(np.isin(cl, list(te_c)))[0]
print(f"[split] rows fit/val/test {len(fi)}/{len(vi)}/{len(ti)}  cells d={NC}", flush=True)

Ttest = Yv[ti].float()
_yt = (Ttest.numpy() > 0.5)
N_POS = int(np.count_nonzero(_yt)); N_SCORED = int(_yt.size)
BASE_TEST = N_POS / N_SCORED
del _yt
print(f"[base] test base rate {BASE_TEST:.12f} = {N_POS}/{N_SCORED}", flush=True)

# ⛔⛔ THE CONTROL THAT MUST READ A LITERAL INTEGER RATIO -- run BEFORE anything else.
_tflat = Ttest.reshape(-1).numpy()
ap_const_tie = AP_tiegroup(np.full(N_SCORED, 0.37), _tflat)
ap_zero_tie = AP_tiegroup(np.zeros(N_SCORED), _tflat)
ap_zero_naive = AP_naive(np.zeros(N_SCORED), _tflat)
print(f"[CONTROL] tie-group constant AP {ap_const_tie:.12f}  "
      f"base {BASE_TEST:.12f}  exact={ap_const_tie == BASE_TEST}", flush=True)
print(f"[CONTROL] tie-group all-zero AP {ap_zero_tie:.12f}  "
      f"exact={ap_zero_tie == BASE_TEST}", flush=True)
print(f"[CONTROL] NAIVE     all-zero AP {ap_zero_naive:.12f}  "
      f"bias {100*(ap_zero_naive/BASE_TEST - 1):+.3f} %   "
      f"(this is s6_oracle.py's own AP; WP-A banked 0.016340211)", flush=True)
if ap_const_tie != BASE_TEST or ap_zero_tie != BASE_TEST:
    raise SystemExit("REFUSING: the constant/all-zero control does not read the "
                     "base rate EXACTLY as the literal integer ratio")

# --------------------------------------------------------------------------- #
# head + ladder (byte-identical to s6_oracle.py)                               #
# --------------------------------------------------------------------------- #
D_RED = 16
S_t = S_BY["prog"]


class Head(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.red = nn.Conv2d(cin, D_RED, 1)
        self.mix = nn.Sequential(nn.Conv2d(D_RED, D_RED, 3, padding=1), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(TH * D_RED + PD, 192), nn.GELU(),
                                 nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, fmap, cellsel, zero_feat=False):
        b = fmap.shape[0]
        x = self.red(fmap)
        x = x + self.mix(x)
        x = x.reshape(b, TH * D_RED, TW)
        g = torch.einsum("bdw,kw->bkd", x, S_t[cellsel])
        if zero_feat:
            g = torch.zeros_like(g)
        p = POS[cellsel].unsqueeze(0).expand(b, -1, -1)
        return self.mlp(torch.cat([g, p], -1)).squeeze(-1)


def feat(build, rows, ker):
    x = ORACLE[build][rows].to(torch.float32)
    if ker != (1, 1):
        x = torch.nn.functional.avg_pool2d(x, ker)
        x = torch.nn.functional.interpolate(x, size=(TH, TW), mode="nearest")
    return x


LADDER = [("16x40", (1, 1)), ("8x20", (2, 2)), ("16x4", (1, 10)),
          ("4x40", (4, 1)), ("4x8", (4, 5)), ("4x4", (4, 10)),
          ("4x2", (4, 20)), ("1x1", (16, 40))]


@torch.no_grad()
def predict(head, build, rows, ker, zf):
    head.eval(); P = []
    allc = torch.arange(NC, device=dev)
    for s in range(0, len(rows), 48):
        r = rows[s:s + 48]
        P.append(torch.sigmoid(head(feat(build, r, ker).to(dev), allc, zf)).cpu())
    head.train()
    return torch.cat(P)


def run(name, build, read, ker, ts, zf=False):
    """build = which sense the MAP is scattered with; read = which sense the head
    LOOKS UP with.  build != read is the deliberate mutation control."""
    global S_t
    S_t = S_BY[read]
    torch.manual_seed(ts); np.random.seed(ts)
    head = Head(2).to(dev)
    npar = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - base) / base], device=dev))
    g = np.random.default_rng(ts)
    t0 = time.time()
    for step in range(a.steps):
        rows = np.sort(g.choice(fi, a.batch, replace=False))
        cs = torch.from_numpy(g.choice(NC, a.cells, replace=False)).to(dev)
        y = Yv[rows][:, cs.cpu()].float().to(dev)
        loss = lossf(head(feat(build, rows, ker).to(dev), cs, zf), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    Pv = predict(head, build, vi, ker, zf)
    Tv = Yv[vi].float()
    ap_val = AP_tiegroup(Pv.numpy(), Tv.numpy())
    qs = torch.quantile(Pv.reshape(-1)[::7].float(),
                        torch.linspace(0.90, 0.9999, 120))
    bf, bt = 0.0, 0.5
    for t in qs:
        pr = (Pv >= t).float()
        f1 = float(2 * (pr * Tv).sum() / (pr.sum() + Tv.sum() + 1e-9))
        if f1 > bf:
            bf, bt = f1, float(t)
    Pt = predict(head, build, ti, ker, zf)
    ap_t = AP_tiegroup(Pt.numpy(), Ttest.numpy())
    ap_t_naive = AP_naive(Pt.numpy(), Ttest.numpy())
    pr = (Pt >= bt).float(); inter = float((pr * Ttest).sum())
    iou = inter / float((pr + Ttest - pr * Ttest).sum() + 1e-9)
    f1 = float(2 * inter / (pr.sum() + Ttest.sum() + 1e-9))
    del head, opt
    torch.cuda.empty_cache()
    print(f"[{name:22s}] AP {ap_t:.4f} (naive {ap_t_naive:.4f})  val {ap_val:.4f}  "
          f"F1 {f1:.4f}  IoU {iou:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return ({"ap_test": ap_t, "ap_test_naive_tiebroken": ap_t_naive,
             "ap_val": ap_val, "f1_test": f1, "iou_test": iou,
             "thr_from_val": bt, "f1_val": bf, "params": int(npar),
             "pool": list(ker), "az_cols": TW // ker[1], "rows": TH // ker[0],
             "deg_per_col": 120.0 / (TW // ker[1]), "build": build, "read": read,
             "train_seed": ts, "wall_s": round(time.time() - t0, 1)},
            Pt.to(torch.float16).numpy())

# --------------------------------------------------------------------------- #
# the panel                                                                    #
# --------------------------------------------------------------------------- #
ARMS = []
for sg in ("prog", "wpa"):
    for nm, ker in LADDER:
        ARMS.append((f"orc_{nm}@{sg}", sg, sg, ker, False))
    ARMS.append((f"pos_only@{sg}", sg, sg, (1, 1), True))
# ⭐ MUTATION CONTROL: the real historical defect, deliberately reintroduced.
XWIRE = [("xwire_16x40", "prog", "wpa", (1, 1), False),
         ("xwire_8x20", "prog", "wpa", (2, 2), False),
         ("xwire_16x40_r", "wpa", "prog", (1, 1), False)]

TSEEDS = [int(x) for x in a.train_seeds.split(",") if x != ""]
res, PRED = {}, {}
t_all = time.time()
for ts in TSEEDS:
    print(f"\n===== TRAIN SEED {ts} (split seed {a.seed} fixed) =====", flush=True)
    # the xwire mutation control runs on the FIRST seed only (deadline budget)
    if ts == TSEEDS[0]:
        todo = ARMS + XWIRE
    else:
        # ⭐ the REPLICATE seed runs a 4-rung subset of the ladder under BOTH
        # senses -- enough to read the rig's own run-to-run floor at the top,
        # the middle and the bottom of the ladder, within the deadline.
        keep = {"16x40", "8x20", "4x4", "1x1"}
        todo = [x for x in ARMS
                if x[0].startswith("pos_only") or x[0].split("@")[0][4:] in keep]
    for name, build, read, ker, zf in todo:
        key = f"{name}|s{ts}"
        r, p = run(key, build, read, ker, ts, zf)
        res[key] = r
        PRED[key] = p
        # ⭐ bank incrementally -- a deadline abort still yields every finished arm
        json.dump({"_partial": True, "arms": res},
                  open(a.out + ".partial", "w", encoding="utf-8"), indent=1)

np.savez_compressed(os.path.join(os.path.dirname(a.out), "test_scores.npz"), **PRED)

# --------------------------------------------------------------------------- #
# paired EPISODE-CLUSTER bootstrap over the 30 TEST CLIPS, AP recomputed per draw #
# --------------------------------------------------------------------------- #
eid = np.asarray([clips[c] for c in cl[ti]])
uniq = np.unique(eid)
idx_by_ep = {u: np.flatnonzero(eid == u) for u in uniq}
Tnp = Ttest.numpy()
Tgpu = Ttest.to(dev)
print(f"\n[bootstrap] paired episode-cluster, {len(uniq)} test clips, "
      f"{a.n_boot} draws, AP recomputed inside every draw", flush=True)

# --------------------------------------------------------------------------- #
# ⭐ EXACT tie-group AP from PER-CLIP HISTOGRAMS.                               #
# The scores are float16, so there are at most 2**15 distinct non-negative      #
# values and each one is EXACTLY one tie group -- no sort is needed, and a      #
# histogram is ADDITIVE over clips.  So each bootstrap draw's AP is computed    #
# by summing 30 precomputed [32768] histograms instead of sorting 2.1e7 scores. #
# ⛔ This is an OPTIMISATION, not an approximation: it is cross-checked below    #
# against the sort-based CPU AP (the verbatim repo function) on the full test   #
# set, and it must agree to < 1e-9 or the run refuses.                          #
# --------------------------------------------------------------------------- #
NBIN = 1 << 15
CLIP_ROWS = [torch.from_numpy(idx_by_ep[u]).to(dev) for u in uniq]


def clip_hists(P):
    """[n_clips, NBIN] totals and positive-counts, one row per TEST clip."""
    Pg = torch.from_numpy(P).to(dev)                       # float16 [rows, NC]
    tot = torch.zeros(len(uniq), NBIN, dtype=torch.float64, device=dev)
    pos = torch.zeros(len(uniq), NBIN, dtype=torch.float64, device=dev)
    for j, rows in enumerate(CLIP_ROWS):
        b = Pg[rows].reshape(-1).view(torch.int16).to(torch.int64)
        y = Tgpu[rows].reshape(-1).to(torch.float64)
        tot[j] = torch.bincount(b, minlength=NBIN).to(torch.float64)
        pos[j] = torch.bincount(b, weights=y, minlength=NBIN)
    del Pg
    return tot, pos


def ap_from_hist(tot, pos):
    """Tie-group AP from a pooled histogram.  Buckets are fp16 BIT PATTERNS,
    which are monotone in value for non-negative floats, so descending score
    order is descending bucket id."""
    t = torch.flip(tot, [0]); p = torch.flip(pos, [0])
    k = t > 0
    t, p = t[k], p[k]
    n_pos = p.sum()
    if float(n_pos) == 0.0:
        return float("nan")
    prec = torch.cumsum(p, 0) / torch.cumsum(t, 0)
    return float((prec * (p / n_pos)).sum())

# ⭐ INDEPENDENT CROSS-CHECK before any bootstrap number: the GPU AP used inside
# the draws must reproduce the CPU AP (the verbatim repo function) on the full
# test set, AND the GPU AP's constant control must still read the literal ratio.
_k0 = f"orc_16x40@prog|s{TSEEDS[0]}"
_t0, _p0 = clip_hists(PRED[_k0])
_cpu = AP_tiegroup(PRED[_k0], Tnp)
_hist = ap_from_hist(_t0.sum(0), _p0.sum(0))
_hc = torch.zeros(NBIN, dtype=torch.float64, device=dev)
_hcp = torch.zeros(NBIN, dtype=torch.float64, device=dev)
_bconst = int(torch.tensor([0.37], dtype=torch.float16).view(torch.int16)[0])
_hc[_bconst] = float(N_SCORED); _hcp[_bconst] = float(N_POS)
_hist_const = ap_from_hist(_hc, _hcp)
print(f"[xcheck] AP  sort-based(cpu) {_cpu:.12f}   histogram {_hist:.12f}   "
      f"|d| {abs(_cpu-_hist):.3e}", flush=True)
print(f"[xcheck] histogram constant control {_hist_const:.12f}  "
      f"base {BASE_TEST:.12f}  exact={_hist_const == BASE_TEST}", flush=True)
if abs(_cpu - _hist) > 1e-9 or _hist_const != BASE_TEST:
    raise SystemExit("REFUSING: the histogram AP disagrees with the sort-based "
                     "AP, or its constant control does not read the base rate")
del _t0, _p0, _hc, _hcp

PICK = torch.stack([torch.from_numpy(
    np.random.default_rng(1000 + i).choice(len(uniq), size=len(uniq), replace=True)
).to(dev) for i in range(a.n_boot)])       # [n_boot, n_clips] clip picks


def paired(k_a, k_b):
    ta, pa = clip_hists(PRED[k_a])
    tb, pb = clip_hists(PRED[k_b])
    point = (AP_tiegroup(PRED[k_a], Tnp) - AP_tiegroup(PRED[k_b], Tnp))
    d = np.empty(a.n_boot)
    for i in range(a.n_boot):
        s = PICK[i]
        d[i] = (ap_from_hist(ta[s].sum(0), pa[s].sum(0))
                - ap_from_hist(tb[s].sum(0), pb[s].sum(0)))
    del ta, pa, tb, pb
    torch.cuda.empty_cache()
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return {"delta": point, "lo": lo, "hi": hi, "ci95": (hi - lo) / 2.0,
            "separated": bool(lo > 0 or hi < 0),
            "p_delta_gt0": float((d > 0).mean()),
            "n_episodes": int(len(uniq)), "n_boot": int(a.n_boot),
            "n_windows": int(Tnp.shape[0]), "n_cells": int(NC),
            "estimator": "paired_episode_cluster_bootstrap (AP recomputed per draw)"}


PAIRS = {}
ts = TSEEDS[0]
for nm, _ in LADDER:                           # THE question: does the sense matter?
    PAIRS[f"prog_minus_wpa@{nm}|s{ts}"] = (f"orc_{nm}@prog|s{ts}",
                                           f"orc_{nm}@wpa|s{ts}")
for sg in ("prog", "wpa"):                     # does the ladder actually descend?
    PAIRS[f"{sg}_16x40_minus_1x1|s{ts}"] = (f"orc_16x40@{sg}|s{ts}",
                                            f"orc_1x1@{sg}|s{ts}")
    PAIRS[f"{sg}_1x1_minus_pos_only|s{ts}"] = (f"orc_1x1@{sg}|s{ts}",
                                               f"pos_only@{sg}|s{ts}")
PAIRS[f"xwire_minus_prog@16x40|s{ts}"] = (f"xwire_16x40|s{ts}",
                                          f"orc_16x40@prog|s{ts}")
PAIRS[f"xwire_minus_pos_only@16x40|s{ts}"] = (f"xwire_16x40|s{ts}",
                                              f"pos_only@prog|s{ts}")
PAIRS[f"xwire_r_minus_wpa@16x40|s{ts}"] = (f"xwire_16x40_r|s{ts}",
                                           f"orc_16x40@wpa|s{ts}")
# ⭐ the REPLICATE contrast: same flags, same sense, different TRAINING seed --
# the rig's own run-to-run floor (H-ESTIM-SEED-1), against which any sense
# effect must be read.
if len(TSEEDS) > 1:
    for nm, _ in LADDER:
        PAIRS[f"REPLICATE_s{TSEEDS[0]}_minus_s{TSEEDS[1]}@{nm}_prog"] = (
            f"orc_{nm}@prog|s{TSEEDS[0]}", f"orc_{nm}@prog|s{TSEEDS[1]}")
boot = {}
for nm, (ka, kb) in PAIRS.items():
    if ka in PRED and kb in PRED:
        boot[nm] = paired(ka, kb)
        b = boot[nm]
        print(f"[boot] {nm:38s} d {b['delta']:+.4f} "
              f"[{b['lo']:+.4f},{b['hi']:+.4f}] sep={b['separated']}", flush=True)

out = {
    "_evidence_class": "MEASURED (ours; artifact = this file + WP-A's own bank "
                       "C:/Users/Admin/wpa-readout/bank/k8r1)",
    "tier": "NOT APPLICABLE - representation probe, no trajectory produced",
    "ORACLE": "⛔ the feature map is BUILT FROM THE TARGET. READOUT-CEILING "
              "measurement, NOT a perception result.",
    "question": "does WP-A's ORACLE ladder change under the CORRECTED azimuth "
                "address? Run on WP-A's own banked bytes, both senses, one "
                "variable moved, plus a cross-wired mutation control.",
    "address_senses": {"prog": "u = W/2 - f_ref*az  (bev_aux.azimuth_column:196, "
                               "psg_targets.azimuth_column:72 -- col 0 = +60 LEFT)",
                       "wpa": "u = W/2 + f_ref*az  (s5_indexed.py:78, "
                              "s6_oracle.py:82 -- the MIRRORED sense)"},
    "prediction_stated_before_measuring": (
        "INVARIANT. The two addresses are an exact mirror (colf_prog + colf_wpa "
        "= TW-1, asserted at runtime); the oracle map is BUILT with the same "
        "address the head READS with; every width pool kernel divides TW=40 so "
        "the pooling partition is mirror-symmetric; the bilinear lookup is "
        "mirror-symmetric. ⇒ both senses must match within the replicate floor "
        "(banked 0.0122 AP on orc_16x40 over train seeds 0/1/2) and the ladder "
        "must stay MONOTONE DOWN under BOTH. This is the OPPOSITE of the "
        "real-trunk arms (1.57-2.10x, monotone UP under mirror), which is what "
        "makes it falsifiable. The mutation control (map built one sense, read "
        "the other) MUST collapse toward pos_only, or the invariance means "
        "only that the head ignores azimuth."),
    "bank": a.bank, "ckpt": idx["ckpt"], "step": idx["step"],
    "token_grid": [TH, TW], "patch_px": PATCH, "f_ref": F_REF,
    "projection_scope": "column LINEAR in azimuth: OUR 256x640 cylindrical "
                        "corpus, f_ref 305.577, FOV 120 deg (pinhole gives 92.6 "
                        "and does not travel)",
    "row_model": {"v0_px": H_PX / 2, "h_cam_m": a.h_cam,
                  "formula": "v = v0 + f_ref*h/range"},
    "mirror_exactness_max_abs_err": mirror_err,
    "n_addressable_cells_both_senses": int(NC),
    "n_fit_rows": int(len(fi)), "n_val_rows": int(len(vi)),
    "n_test_rows": int(len(ti)), "n_scored_cells": N_SCORED, "n_pos_test": N_POS,
    "base_rate_all": base, "base_rate_test": BASE_TEST,
    "controls": {
        "constant_ap_tiegroup": ap_const_tie,
        "allzero_ap_tiegroup": ap_zero_tie,
        "equals_base_rate_exactly": bool(ap_const_tie == BASE_TEST
                                         and ap_zero_tie == BASE_TEST),
        "base_rate_literal": f"{N_POS}/{N_SCORED}",
        "allzero_ap_NAIVE_s6_form": ap_zero_naive,
        "naive_tie_bias_pct": 100 * (ap_zero_naive / BASE_TEST - 1.0),
        "note": "⛔ no headline number here is an accuracy: an all-zero "
                f"predictor scores {100*(1-BASE_TEST):.4f} % accuracy."},
    "split_seed": a.seed, "train_seeds": TSEEDS, "steps": a.steps,
    "arms": res, "paired_bootstrap": boot,
    "estimator_note": "paired episode-cluster bootstrap over the 30 TEST CLIPS "
                      "(taniteval/ci.py machinery), AP recomputed inside every "
                      "draw. overlapping_holdout_se is NOT used.",
}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print("\nWROTE", a.out, f"{time.time()-t_all:.0f}s total", flush=True)
