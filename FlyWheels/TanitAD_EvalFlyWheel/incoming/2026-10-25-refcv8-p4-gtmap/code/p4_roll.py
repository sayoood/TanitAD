"""p4_roll.py -- P4 GT-MAP ORACLE (PREREG_P4.md): ONE inference-seed roll of refcv8 EMA-48,852 (per-frame nav feed, os-only), with
the planner's BEV re-formed from features that DECODE TO A GIVEN MAP through the model's OWN frozen 10 cm map decoder.

WHY THIS DESIGN (from source, tree fb47a583763f, sha256-verified against the launch tree):
  * refc_v3.py:2189-2192  hi = mh(fmap_s8, grid, valid); pout = br(fmap_s16, bev_hires=hi["map_hires_bev"]); pout["map_hires_logits"]=...
  * map_head_hires.py:849-873  lift -> encoder = `map_hires_bev` [B,64,400,240] @0.25 m; logits = refine(feats [+ near lift]) (a LEAF)
  * refcv6_perception_branch.py:638-640 / 342-346  bev_feats = proj(avg_pool2d(crop(map_hires_bev))) ; bev_tokens = adaptive_avg_pool(bev_feats)
  * refc.py:4798 / 4826  the planner reads perception_out["bev_tokens"] (scene hook) and ["bev_feats"] (decoder BEV coupling)
  * refc_v3.py:2620-2630  the ONLY map-logit consumer is the refcv8 drivable critic `r8_drv`, which is NOT BUILT in this run
    (perc2plan census r8_drv_is_none = true).  => NO path lets a map raster enter the tokens; zero-training substitution is impossible.
  => the oracle injects the map AS FEATURES: per window, delta (64 x 240 x 128 @0.25 m, the planner's crop only) is optimised so that the
     model's own frozen decoder (refine + near path) reads the TARGET map on the crop's seen 10 cm cells; the planner then receives
     bev_feats/bev_tokens recomputed from map_hires_bev + delta through the run's own pool (crop -> 2x2 mean -> proj -> GN).
     Zero new parameters, zero training.  Arms (identical procedure, different TARGET):
       TRUE   = the SAM3 GT 10 cm codes of THIS window (the trainer's own target path: _map_fine_item(ep, t+w-1))
       OWNMAP = the model's OWN decoded map (fp32 argmax of refine(map_hires_bev)), on the SAME seen-cell mask as TRUE
       SHUF   = the GT of ANOTHER episode's window (cyclic next clip in the windows file; position-matched) -- must HURT (liveness)
     The oracle effect is TRUE - OWNMAP: the procedure's own artefact cancels.  BASE = the battery's own record (the unhooked forward).
  * mechanics: the r8oracle pattern -- the battery's proxy __call__ runs first (= BASE, its dump), then every arm re-runs the model on the
    SAME inputs from the SAME pre-call RNG state inside torch.random.fork_rng (common random numbers).  The 10 cm branch output is
    memoised per window (exact torch.equal guard), so an arm forward recomputes only what reads the BEV.
  * delta-p (2x2 mean of delta = exactly what the planner's pool sees) is cached fp16 per window (--delta-mode compute, seed 1) and READ
    by the other seed (--delta-mode read): both seeds apply the IDENTICAL injected tensor.

    R8_LAUNCH_TREE=<tree_fb47a58> PYTHONPATH='<tree>/stack;<tree>/taniteval' R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=4 \
      python -X utf8 p4_roll.py --p4-out <dir>/p4_s1.npz --delta-mode compute --delta-cache <F:/...> --map-gt-root <sam3_gt_v3_eval> \
          --remap-json <refcv8_eval_remap_devbox_p4.json> -- <battery child argv incl. --os-only>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREREG, SEAL = ROOT / "PREREG_P4.md", ROOT / "PREREG_P4.sha256"
PF_CODE = Path(os.environ.get("P4_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-perframe-final/code"))
ARMS_ALL = ("TRUE", "OWNMAP", "SHUF")
N_SLOT = 8
NOT_SEEN = 255
BANDS_FINE = ((0, 200), (200, 400), (400, 600))          # 0-20 / 20-40 / 40-60 m on the planner crop (10 cm rows)
PAIRS = ("own0_vs_gt", "TRUE_vs_gt", "OWNMAP_vs_gt", "SHUF_vs_gt", "SHUF_vs_donor", "OWNMAP_vs_own0",
         "TRUEpool_vs_gt", "SHUFpool_vs_donor", "OWNMAPpool_vs_own0")
ZERO_DELTA = os.environ.get("P4_ZERO_DELTA", "") == "1"   # identity control: every arm's delta-p forced to exact 0 (smoke only)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse():
    if "--" not in sys.argv:
        raise SystemExit("[p4] usage: p4_roll.py [p4 opts] -- <battery child argv>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4-out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS_ALL))
    ap.add_argument("--delta-mode", choices=("compute", "read"), required=True)
    ap.add_argument("--delta-cache", required=True, help="dir of per-episode delta-p npz (written in compute mode, read in read mode)")
    ap.add_argument("--map-gt-root", required=True)
    ap.add_argument("--remap-json", default=None, help="replaces r8_repoint.REMAP_OVERRIDES (a PATH; recorded)")
    ap.add_argument("--inv-steps", type=int, default=80)
    ap.add_argument("--inv-lr-rel", type=float, default=0.05, help="Adam lr = this x rms(map_hires_bev on the crop) of the window")
    ap.add_argument("--inv-lambda", type=float, default=0.01, help="weight of mean(delta^2)/mean(f0_crop^2)")
    ap.add_argument("--inv-amp", choices=("off", "bf16"), default="bf16", help="autocast for the optimisation steps (CUDA only)")
    ap.add_argument("--inv-chunk", type=int, default=3, help="arms optimised per batch (memory only; the arms are independent)")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--require-seal", action="store_true")
    ap.add_argument("--budget-probe-windows", type=int, default=0, help="after N windows project the total wall (marker)")
    ap.add_argument("--budget-max-s", type=float, default=0.0, help="compute mode: stop (exit 7) if projected s1 + s0 wall exceeds this")
    a = ap.parse_args(sys.argv[1:i])
    child = sys.argv[i + 1:]
    arms = tuple(x for x in a.arms.split(",") if x)
    bad = [x for x in arms if x not in ARMS_ALL]
    if bad:
        raise SystemExit(f"[p4] unknown arms {bad}")
    if "--os-only" not in child:
        raise SystemExit("[p4] the oracle rolls the battery's --os-only child; --os-only missing")
    return a, arms, child


A, ARMS, CHILD = parse()
SEAL_SHA = None
if SEAL.exists() and PREREG.exists():
    want = SEAL.read_text(encoding="utf-8").split()[0].strip()
    got = sha256_file(PREREG)
    SEAL_SHA = got if (len(want) == 64 and got == want) else None
if A.require_seal and SEAL_SHA is None:
    raise SystemExit("[p4] PREREG_P4.md does not match PREREG_P4.sha256: refusing a scored roll")

# ---- re-point the dev-box remap (the launch tree moved; one path in the devbox remap named it) -- BEFORE the child module imports it
_r8_code = Path(os.environ.get("PF_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                             "2026-10-15-refcv8-milestone-battery/code"))
sys.path.insert(0, str(_r8_code))
import r8_repoint  # noqa: E402
REMAP_REC = {"devbox_file_replaced": str(r8_repoint.REMAP_OVERRIDES), "launch_tree": str(r8_repoint.LAUNCH_TREE)}
if A.remap_json:
    REMAP_REC.update(file=A.remap_json, sha256=sha256_file(A.remap_json), map=json.load(open(A.remap_json, encoding="utf-8")))
    r8_repoint.REMAP_OVERRIDES = Path(A.remap_json)

sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py")] + CHILD
sys.path.insert(0, str(PF_CODE))
import roll_seed_r8_pf as PF  # noqa: E402  (overlay check, apply_child, nav-feed overlay; READ-ONLY, UNCHANGED)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RR = PF.RR
_Base = RR.Refcv6Proxy
DEV_IS_CUDA = "--device" in CHILD and CHILD[CHILD.index("--device") + 1].startswith("cuda")
ST = {"mode": "own", "apply": None, "cap": None, "cap_calls": 0, "applied": 0, "installed": False, "model": None, "mh": None, "br": None,
      "records": [], "n_windows": 0, "t0": time.time(), "cache_ep": None, "cache_rows": {}, "cache_loaded": {}, "donor": None,
      "c": {"own_calls": 0, "navnull_calls": 0, "recomp_zero_equal": 0, "recomp_zero_checked": 0, "recomp_zero_maxdiff": 0.0,
            "mh_memo_hit": 0, "mh_memo_miss": 0, "arm_forwards": 0, "arm_as_base_nogt": 0, "gt_store_parity_checked": 0,
            "gt_store_parity_equal": 0, "refine_repro_maxdiff": 0.0, "refine_repro_argmax_agree_min": 1.0,
            "applied_not_once": 0, "cap_not_once": 0}}
ARGS: dict = {}
CROP = {}


# ============================================================================================ dataset: attach the 10 cm GT (trainer path)
def _wrap_build_eval_dataset(L, map_root):
    """The l7cap pattern (fan_roll_r8_l7cap.py:_wrap_build_eval_dataset): the roll's OWN build (perception targets OFF), then ONLY the
    10 cm GT through the trainer's `enable_map_hires` -> every item carries `map_fine` (= _map_fine_item(ep, t+w-1)).  Index unchanged."""
    prev = L.build_eval_dataset

    def bed(model, cfg, args, config, *, with_perception_targets=True, dataset_cls=None):
        e_ds, e_eps, rec = prev(model, cfg, args, config, with_perception_targets=with_perception_targets, dataset_cls=dataset_cls)
        n0 = len(e_ds.index)
        tr = L.trainer()
        _clip, _ns = tr._clip_table_for_caches([args.eval_cache])
        ext = tr._mhr.declared_extent(args)
        store = tr._sem_fine.FineMapGTStore(Path(map_root), max_open=4, extent=ext)
        rec["p4_map_fine"] = e_ds.enable_map_hires(store, _clip, _ns, min_coverage=None)
        if len(e_ds.index) != n0:
            raise SystemExit("[p4] attaching the map GT changed the window index (%d -> %d): refusing" % (n0, len(e_ds.index)))
        ARGS.update(extent=ext, clip_of_ep=dict(_clip), n_stack=int(_ns), episodes=[int(e.episode_id) for e in e_ds.episodes],
                    donor_store=tr._sem_fine.FineMapGTStore(Path(map_root), max_open=2, extent=ext), sem=tr._sem_fine,
                    window=int(getattr(e_ds, "window", 8)))
        rec["p4_label_side"] = {"map_gt_root": str(map_root), "index_unchanged": True, "n_stack": int(_ns)}
        return e_ds, e_eps, rec

    L.build_eval_dataset = bed


# ============================================================================================ hooks
def _recompute(br, f0, dp):
    """bev_feats / bev_tokens from map_hires_bev f0 [n,64,X4,Y4] + delta-p [64,120,64] through the run's OWN pool modules.
    PlannerBEVPool.forward (refcv6_perception_branch.py:338-351) = proj(avg_pool2d(crop(x), k)); crop+mean are linear, so
    pool(f0 + upsample(dp)) == proj(avg_pool2d(crop(f0)) + dp) -- pinned by test_p4_pool_shortcut.py."""
    bp = br.bev_pool
    w = f0[:, :, :bp.rows, bp.col0:bp.col0 + bp.cols]
    p = F.avg_pool2d(w, kernel_size=bp.k, stride=bp.k)
    if dp is not None:
        p = p + dp.to(device=p.device, dtype=p.dtype)[None]
    feats = bp.proj(p)
    return feats, br.bev_tokens(feats)


def _install(model):
    mh, br = getattr(model, "_map_hires", None), getattr(model, "_perception", None)
    if mh is None or br is None or getattr(br, "bev_pool", None) is None:
        raise SystemExit("[p4] the model has no 10 cm branch / map_hires_pool perception branch: not refcv8 option (c)")
    if getattr(model, "r8_drv", None) is not None:
        raise SystemExit("[p4] the drivable critic is built: a map-logit -> planner path the prereg did not register")
    bp = br.bev_pool
    ext = ARGS["extent"]
    sem = ARGS["sem"]
    planner_ext = sem.MapExtent(float(bp.rows) * 0.25, float(bp.cols) * 0.25 / 2.0)
    r0, r1, c0, c1 = planner_ext.window_in(ext)
    if (r1 - r0) != bp.rows * 25 // 10 or (c1 - c0) != bp.cols * 25 // 10 or c0 != bp.col0 * 25 // 10 or bp.k != 2:
        raise SystemExit(f"[p4] crop geometry disagrees: fine {(r0, r1, c0, c1)} vs pool rows {bp.rows} cols {bp.cols} col0 {bp.col0} k {bp.k}")
    CROP.update(r0=r0, r1=r1, c0=c0, c1=c1, rows4=bp.rows, col04=bp.col0, cols4=bp.cols)
    cw = getattr(model, "_map_hires_class_weight", None)
    ST.update(model=model, mh=mh, br=br, class_weight=(cw.detach().float().clone() if torch.is_tensor(cw) else None))
    orig_mh_fwd = mh.forward

    def mh_forward(fmap_s8, grid, valid, **kw):
        cap = ST["cap"]
        if ST["mode"] != "own" and cap is not None and not kw and fmap_s8 is not None \
                and tuple(fmap_s8.shape) == tuple(cap["mh_in"][0].shape) and torch.equal(fmap_s8, cap["mh_in"][0]) \
                and torch.equal(grid, cap["mh_in"][1]) and torch.equal(valid, cap["mh_in"][2]):
            ST["c"]["mh_memo_hit"] += 1
            ST["last_mh"] = (cap["mh_in"], cap["mh_out"])
            return cap["mh_out"]
        r = orig_mh_fwd(fmap_s8, grid, valid, **kw)
        if ST["mode"] != "own":
            ST["c"]["mh_memo_miss"] += 1
        ST["last_mh"] = ((fmap_s8, grid, valid), r)
        return r
    mh.forward = mh_forward                                     # instance attribute; Module.__call__ resolves self.forward

    orig_factory = model._bev_hook                              # bound method

    def factory(*fa, **fk):
        h = orig_factory(*fa, **fk)
        if h is None:
            return None

        def hooked(*ha, **hk):
            pout = h(*ha, **hk)
            mh_in, mh_out = ST["last_mh"]
            f0 = mh_out["map_hires_bev"]
            if ST["mode"] == "own":
                if ST.get("capture_armed"):
                    ST["capture_armed"] = False
                    ST["cap_calls"] += 1
                    ST["cap"] = {"mh_in": mh_in, "mh_out": mh_out, "logits": mh_out["map_hires_logits"]}
                    with torch.no_grad():                        # identity of the recompute path: delta-p = 0 must be bit-exact
                        f2, t2 = _recompute(br, f0, None)
                    eq = torch.equal(f2, pout["bev_feats"]) and ("bev_tokens" not in pout or torch.equal(t2, pout["bev_tokens"]))
                    ST["c"]["recomp_zero_checked"] += 1
                    ST["c"]["recomp_zero_equal"] += int(eq)
                    ST["c"]["recomp_zero_maxdiff"] = max(ST["c"]["recomp_zero_maxdiff"],
                                                         float((f2.float() - pout["bev_feats"].float()).abs().max()))
                return pout
            dp = ST["apply"]
            if dp is None:
                raise RuntimeError("[p4] arm forward with no delta-p")
            new = dict(pout)
            nf, nt = _recompute(br, f0, dp)
            new["bev_feats"] = nf
            if "bev_tokens" in pout:
                new["bev_tokens"] = nt
            ST["applied"] += 1
            return new
        return hooked

    model._bev_hook = factory                                   # instance attribute shadows the bound method (refc_v3.py:2502)
    ST["installed"] = True
    print(f"[p4] hooks installed: crop fine {CROP} class_weight={'run' if ST['class_weight'] is not None else 'NONE'}", flush=True)


# ============================================================================================ targets + inversion
def _donor_table(windows_json):
    """clip -> (donor clip, ws list).  Cyclic NEXT clip in sorted episode-index order that has a GT file; position-matched window."""
    W = json.load(open(windows_json, encoding="utf-8"))
    eps = ARGS["episodes"]
    clip_of_ep = ARGS["clip_of_ep"]
    ci_of_clip = {clip_of_ep[eid]: i for i, eid in enumerate(eps) if eid in clip_of_ep}
    order = sorted((ci_of_clip[c], c) for c in W)
    store = ARGS["donor_store"]
    has = {}
    for _, c in order:
        has[c] = store.resolve(c) is not None
    tab = {}
    for k, (_, c) in enumerate(order):
        d = None
        for s in range(1, len(order) + 1):
            cand = order[(k + s) % len(order)][1]
            if cand != c and has[cand]:
                d = cand
                break
        tab[c] = {"donor": d, "ws": [int(x) for x in W[c]], "donor_ws": ([int(x) for x in W[d]] if d else [])}
    ST["donor"] = tab
    return {"n_clips": len(order), "n_with_gt": int(sum(has.values())), "no_gt_clips": sorted(c for c, v in has.items() if not v)}


def _donor_codes(clip, ws):
    t = ST["donor"].get(clip)
    if t is None or t["donor"] is None:
        return None, None
    j = t["ws"].index(int(ws))
    dws = t["donor_ws"][min(j, len(t["donor_ws"]) - 1)]
    fm = ARGS["donor_store"].frames_for_windows(t["donor"], [dws], n_stack=ARGS["n_stack"])
    return fm.codes[0], (t["donor"], int(dws))


def _band_stats(pred, tgt):
    """[3 bands, 8 classes, 2 (inter, union)] over cells where tgt != 255 (pred, tgt: [600,320] int tensors on the crop).
    Confusion by bincount (rows = target, cols = pred): inter = diag, union = rowsum + colsum - diag."""
    out = np.zeros((len(BANDS_FINE), 8, 2), np.int64)
    for b, (r0, r1) in enumerate(BANDS_FINE):
        p, t = pred[r0:r1].reshape(-1), tgt[r0:r1].reshape(-1)
        m = t != NOT_SEEN
        cm = torch.bincount(t[m] * 8 + p[m].clamp(0, 7), minlength=64)[:64].reshape(8, 8).cpu().numpy().astype(np.int64)
        dg = np.diag(cm)
        out[b, :, 0] = dg
        out[b, :, 1] = cm.sum(1) + cm.sum(0) - dg
    return out


def _decode(mh, f0_full, delta_pad, near):
    x = f0_full if delta_pad is None else f0_full + delta_pad
    if near is None:
        return mh.refine(x)
    return mh.refine(x, near=near, near_refine=mh.near_refine)


MARGIN4 = 6                                                  # coarse (0.25 m) cells of context around the crop = 15 fine cells


def _gn_capture(mods):
    """forward PRE-hooks recording each GroupNorm's per-(sample, group) input mean / biased var (the exact statistics GN uses)."""
    st, hs = {}, []
    for m in mods:
        def pre(mod, inp, _st=st):
            x = inp[0].float()
            g = mod.num_groups
            xg = x.reshape(x.shape[0], g, -1)
            _st[id(mod)] = (xg.mean(-1), xg.var(-1, unbiased=False))
        hs.append(m.register_forward_pre_hook(pre))
    return st, hs


def _gn_frozen(x, gn, st):
    m, v = st
    B, C, H, W = x.shape
    G = gn.num_groups
    xg = x.reshape(B, G, C // G, H, W)
    xg = (xg - m[:1].reshape(1, G, 1, 1, 1)) * torch.rsqrt(v[:1].reshape(1, G, 1, 1, 1) + gn.eps)
    return xg.reshape(B, C, H, W) * gn.weight.reshape(1, C, 1, 1) + gn.bias.reshape(1, C, 1, 1)


def _gn_mods(mh):
    R = mh.refine
    mods = [R.inp[1], R.norm1, R.norm2]
    for blk in (mh.near_refine or ()):
        mods.append(blk.n1)
    return mods


def _surrogate(mh, xc, near_c, gs, fine_hw):
    """HiresRefine.forward (map_head_hires.py) on a CROP of the 0.25 m map with every GroupNorm's statistics FROZEN at the exact
    full-extent values of delta = 0: local, so it can run on the planner crop + margin (~2.8x fewer 10 cm cells). Used ONLY for the
    optimisation steps; targets and reach are always the EXACT full head. Exactness at delta = 0 is measured per window."""
    R = mh.refine
    x = _gn_frozen(R.inp[0](xc), R.inp[1], gs[id(R.inp[1])])
    x = F.interpolate(x, size=fine_hw, mode="bilinear", align_corners=False)
    if near_c is not None:
        n = int(near_c.shape[2])
        top = x[:, :, :n] + near_c
        for blk in (mh.near_refine or ()):
            top = top + blk.c2(F.gelu(_gn_frozen(blk.c1(top), blk.n1, gs[id(blk.n1)])))
        x = torch.cat([top, x[:, :, n:]], dim=2)
    x = R.act(_gn_frozen(R.conv1(x), R.norm1, gs[id(R.norm1)]))
    x = R.act(_gn_frozen(R.conv2(x), R.norm2, gs[id(R.norm2)]))
    return R.cls(x)


def invert(cap, gt_codes, gt_label, donor_codes):
    """-> (dp dict arm -> [64,120,64] fp16-rounded tensor or None, stats dict).  fp32 decode for targets + reach; the optimisation
    steps under --inv-amp autocast on CUDA.  Only delta is optimised (torch.autograd.grad wrt delta: no parameter grads)."""
    mh = ST["mh"]
    fmap_s8, grid, valid = (t[:1] for t in cap["mh_in"])
    f0 = cap["mh_out"]["map_hires_bev"][:1].detach().float()
    r0, r1, c0, c1 = CROP["r0"], CROP["r1"], CROP["c0"], CROP["c1"]
    R4, C04, NC4 = CROP["rows4"], CROP["col04"], CROP["cols4"]
    X4, Y4 = int(f0.shape[2]), int(f0.shape[3])
    dev = f0.device
    stats = {"pairs": np.zeros((len(PAIRS), len(BANDS_FINE), 8, 2), np.int64), "loss0": {}, "loss1": {}, "rel_norm": {},
             "has_target": {}, "refine_repro_maxdiff": None, "wall": 0.0}
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        near = None if mh.near is None else mh.near(fmap_s8.float(), grid, valid).float()
        gs, hs = _gn_capture(_gn_mods(mh))
        try:
            lg0 = _decode(mh, f0, None, near)[:, :, r0:r1, c0:c1]
        finally:
            for h_ in hs:
                h_.remove()
        own0 = lg0.argmax(1)[0]
        ml = cap["logits"][:1, :, r0:r1, c0:c1].float()
        stats["refine_repro_maxdiff"] = float((ml - lg0).abs().max())
        stats["refine_repro_argmax_agree"] = float((ml.argmax(1)[0] == own0).float().mean())
    seen_gt = None
    gtc = None
    if gt_label and gt_codes is not None:
        gtc = torch.as_tensor(np.asarray(gt_codes)[r0:r1, c0:c1].astype(np.int64), device=dev)
        seen_gt = gtc != NOT_SEEN
        if not bool(seen_gt.any()):
            gtc = seen_gt = None
    targets = {}
    if gtc is not None:
        targets["TRUE"] = gtc
        targets["OWNMAP"] = torch.where(seen_gt, own0, torch.full_like(own0, NOT_SEEN))
    if donor_codes is not None:
        dc = torch.as_tensor(np.asarray(donor_codes)[r0:r1, c0:c1].astype(np.int64), device=dev)
        if bool((dc != NOT_SEEN).any()):
            targets["SHUF"] = dc
    arms = [a for a in ARMS if a in targets]
    for a in ARMS:
        stats["has_target"][a] = a in targets
    dp_out = {a: None for a in ARMS}
    if gtc is not None:
        stats["pairs"][PAIRS.index("own0_vs_gt")] = _band_stats(own0, gtc)
    if not arms:
        stats["wall"] = time.time() - t0
        return dp_out, stats
    f0c = f0[:, :, :R4, C04:C04 + NC4]
    ms = float(f0c.pow(2).mean())
    lr = float(A.inv_lr_rel) * math.sqrt(max(ms, 1e-12))
    w = ST["class_weight"].to(dev) if ST["class_weight"] is not None else None
    pad = (C04, Y4 - C04 - NC4, 0, X4 - R4)
    use_amp = (A.inv_amp == "bf16") and dev.type == "cuda"
    m4 = MARGIN4
    cr0, cr1, cc0, cc1 = 0, R4 + m4, C04 - m4, C04 + NC4 + m4          # coarse crop with context
    if cc0 < 0 or cc1 > Y4 or cr1 > X4:
        raise SystemExit("[p4] surrogate crop exceeds the map extent")
    fine_hw = (int(round((cr1 - cr0) * 2.5)), int(round((cc1 - cc0) * 2.5)))
    fo_c = int(round(cc0 * 2.5))                                          # fine column origin of the surrogate output
    lr0_, lr1_, lc0_, lc1_ = r0, r1, c0 - fo_c, c1 - fo_c                 # the loss region inside the surrogate output
    xc0 = f0[:, :, cr0:cr1, cc0:cc1]
    nc0 = None if near is None else near[:, :, :, fo_c:fo_c + fine_hw[1]]
    spad = (C04 - cc0, cc1 - C04 - NC4, 0, cr1 - R4)
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        sg0 = _surrogate(mh, xc0, nc0, gs, fine_hw)[:, :, lr0_:lr1_, lc0_:lc1_]
        stats["surrogate_maxdiff0"] = float((sg0 - lg0).abs().max())
        stats["surrogate_argmax_agree0"] = float((sg0.argmax(1)[0] == own0).float().mean())
    ch = max(1, int(A.inv_chunk))
    for j0 in range(0, len(arms), ch):                          # arms are independent problems: chunking changes memory, not math
        carms = arms[j0:j0 + ch]
        na = len(carms)
        tg = torch.stack([targets[a] for a in carms])           # [na, 600, 320]
        delta = torch.zeros((na,) + tuple(f0c.shape[1:]), device=dev, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        f0r = f0.expand(na, -1, -1, -1)
        nr = None if near is None else near.expand(na, -1, -1, -1)

        xcr = xc0.expand(na, -1, -1, -1)
        ncr = None if nc0 is None else nc0.expand(na, -1, -1, -1)

        def loss_of(dl, amp, exact=False):
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                if exact:
                    lg = _decode(mh, f0r, F.pad(dl, pad), nr)[:, :, r0:r1, c0:c1]
                else:
                    lg = _surrogate(mh, xcr + F.pad(dl, spad), ncr, gs, fine_hw)[:, :, lr0_:lr1_, lc0_:lc1_]
            lg = lg.float()
            ce = torch.stack([F.cross_entropy(lg[i:i + 1], tg[i:i + 1], weight=w, ignore_index=NOT_SEEN) for i in range(na)])
            reg = dl.pow(2).flatten(1).mean(1) / max(ms, 1e-12)
            return ce, reg

        with torch.enable_grad():
            for k in range(int(A.inv_steps)):
                ce, reg = loss_of(delta, use_amp)
                if k == 0:
                    for i, a in enumerate(carms):
                        stats["loss0"][a] = float(ce[i].detach())
                tot = (ce + float(A.inv_lambda) * reg).sum()
                g, = torch.autograd.grad(tot, [delta])
                delta.grad = g
                opt.step()
                opt.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
            d = delta.detach()
            ce, _ = loss_of(d, False, exact=True)
            dpool = F.avg_pool2d(d, kernel_size=2, stride=2).half().float()      # what the planner sees, fp16-rounded (= the cache)
            lg_full = _decode(mh, f0r, F.pad(d, pad), nr)[:, :, r0:r1, c0:c1].argmax(1)
            lg_pool = _decode(mh, f0r, F.pad(F.interpolate(dpool, scale_factor=2, mode="nearest"), pad), nr)[:, :, r0:r1, c0:c1].argmax(1)
            for i, a in enumerate(carms):
                stats["loss1"][a] = float(ce[i])
                stats["rel_norm"][a] = float(dpool[i].pow(2).mean().sqrt() / max(math.sqrt(ms), 1e-12))
                dp_out[a] = torch.zeros_like(dpool[i]) if ZERO_DELTA else dpool[i].clone()
                if a == "TRUE":
                    stats["pairs"][PAIRS.index("TRUE_vs_gt")] = _band_stats(lg_full[i], gtc)
                    stats["pairs"][PAIRS.index("TRUEpool_vs_gt")] = _band_stats(lg_pool[i], gtc)
                elif a == "OWNMAP":
                    stats["pairs"][PAIRS.index("OWNMAP_vs_gt")] = _band_stats(lg_full[i], gtc)
                    stats["pairs"][PAIRS.index("OWNMAP_vs_own0")] = _band_stats(lg_full[i], targets["OWNMAP"])
                    stats["pairs"][PAIRS.index("OWNMAPpool_vs_own0")] = _band_stats(lg_pool[i], targets["OWNMAP"])
                elif a == "SHUF":
                    stats["pairs"][PAIRS.index("SHUF_vs_donor")] = _band_stats(lg_full[i], targets["SHUF"])
                    stats["pairs"][PAIRS.index("SHUFpool_vs_donor")] = _band_stats(lg_pool[i], targets["SHUF"])
                    if gtc is not None:
                        stats["pairs"][PAIRS.index("SHUF_vs_gt")] = _band_stats(lg_full[i], gtc)
        del delta, opt, d, lg_full, lg_pool
    stats["lr"] = lr
    stats["wall"] = time.time() - t0
    return dp_out, stats


# ============================================================================================ delta-p cache (one npz per episode)
def _cache_path(ci):
    return Path(A.delta_cache) / f"ep{int(ci):04d}.npz"


def _cache_flush():
    ci = ST["cache_ep"]
    rows = ST["cache_rows"]
    if ci is None or not rows or A.delta_mode != "compute":
        return
    ws = sorted(rows)
    arr = np.zeros((len(ws), len(ARMS_ALL), 64, CROP["rows4"] // 2, CROP["cols4"] // 2), np.float16)
    has = np.zeros((len(ws), len(ARMS_ALL)), bool)
    for j, w_ in enumerate(ws):
        for k, a in enumerate(ARMS_ALL):
            v = rows[w_].get(a)
            if v is not None:
                arr[j, k] = v.detach().cpu().numpy().astype(np.float16)
                has[j, k] = True
    p = _cache_path(ci)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, ws=np.array(ws, np.int64), arms=np.array(ARMS_ALL), has=has, dp=arr)
    os.replace(tmp, p)


def _cache_get(ci, ws, dev):
    if ST["cache_loaded"].get("ci") != ci:
        p = _cache_path(ci)
        if not p.exists():
            raise SystemExit(f"[p4] read mode: no delta-p cache for episode {ci} at {p}")
        z = np.load(p, allow_pickle=False)
        ST["cache_loaded"] = {"ci": ci, "ws": {int(w): j for j, w in enumerate(z["ws"])}, "has": z["has"], "dp": z["dp"],
                              "arms": [str(x) for x in z["arms"]]}
    c = ST["cache_loaded"]
    j = c["ws"].get(int(ws))
    if j is None:
        raise SystemExit(f"[p4] read mode: window ({ci},{ws}) missing from the cache")
    out = {}
    for a in ARMS:
        k = c["arms"].index(a)
        out[a] = torch.as_tensor(c["dp"][j, k].astype(np.float32), device=dev) if c["has"][j, k] else None
    return out


# ============================================================================================ the proxy
def _np(x):
    return x.detach().float().cpu().numpy()


class P4Proxy(_Base):
    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        d = self.__dict__
        model = d["model"]
        if not ST["installed"]:
            ST["t_first"] = time.time()
            _install(model)
            ST["donor_census"] = _donor_table(CHILD[CHILD.index("--windows-json") + 1])
        if nav_cmd is None:                                      # not expected under --os-only; passed through, unhooked
            ST["c"]["navnull_calls"] += 1
            return _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        n, dev = int(frames.shape[0]), frames.device
        devs = [dev] if dev.type == "cuda" else []
        st_cpu = torch.get_rng_state()
        st_cuda = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
        ST["mode"], ST["cap"], ST["capture_armed"] = "own", None, True
        k_cap = ST["cap_calls"]
        t_w = time.time()
        out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        w_own = time.time() - t_w
        ST["c"]["own_calls"] += 1
        if ST["cap_calls"] != k_cap + 1 or ST["cap"] is None:
            ST["c"]["cap_not_once"] += 1
            raise RuntimeError(f"[p4] the OWN call captured the BEV {ST['cap_calls'] - k_cap} times (expected 1)")
        key = tuple(int(x) for x in self.ctx.index)
        it = self.ctx.item
        W = int(ARGS.get("window", 8))
        ci, ws = key[0], key[1] + W - 1
        clip = ARGS["clip_of_ep"][ARGS["episodes"][ci]]
        if clip not in ST["donor"] or ws not in ST["donor"][clip]["ws"]:
            raise SystemExit(f"[p4] window ({ci},{ws}) -> clip {clip} is not in the windows file: the e_i -> clip map drifted")
        gt_label = bool(it.get("map_fine_label", torch.tensor(False))) if "map_fine" in it else False
        gt_codes = it["map_fine"].numpy() if gt_label else None
        if gt_label and ST["c"]["gt_store_parity_checked"] < 50:  # the dataset's item == a direct store read of (clip, ws)
            fm = ARGS["donor_store"].frames_for_windows(clip, [ws], n_stack=ARGS["n_stack"])
            ST["c"]["gt_store_parity_checked"] += 1
            ST["c"]["gt_store_parity_equal"] += int(np.array_equal(fm.codes[0], gt_codes))
        rec = {"ci": ci, "ws": ws, "clip": clip, "gt_label": gt_label,
               "arms": {"OWN": {"traj": _np(out["traj"][0]), "sel": int(out["sel_idx"][0]), "forced": False, "wall": round(w_own, 3)}}}
        if A.delta_mode == "compute":
            dcodes, dkey = _donor_codes(clip, ws)
            dps, stt = invert(ST["cap"], gt_codes, gt_label, dcodes)
            rec["donor"] = dkey
            rec["inv"] = stt
            ST["c"]["refine_repro_maxdiff"] = max(ST["c"]["refine_repro_maxdiff"], stt["refine_repro_maxdiff"])
            ST["c"]["refine_repro_argmax_agree_min"] = min(ST["c"]["refine_repro_argmax_agree_min"], stt["refine_repro_argmax_agree"])
            if ST["cache_ep"] != ci:
                _cache_flush()
                ST["cache_ep"], ST["cache_rows"] = ci, {}
            ST["cache_rows"][ws] = dps
        else:
            dps = _cache_get(ci, ws, dev)
            if ZERO_DELTA:
                dps = {a: (None if v is None else torch.zeros_like(v)) for a, v in dps.items()}
        rec["dp_sum"] = {a: (None if v is None else float(v.double().sum())) for a, v in dps.items()}
        for arm in ARMS:
            dp = dps.get(arm)
            if dp is None:                                       # no target for this window: the arm IS BASE by construction (counted)
                rec["arms"][arm] = dict(rec["arms"]["OWN"], forced=False, wall=0.0)
                ST["c"]["arm_as_base_nogt"] += 1
                continue
            t_a = time.time()
            k_app = ST["applied"]
            with torch.random.fork_rng(devices=devs):
                torch.set_rng_state(st_cpu)
                if st_cuda is not None:
                    torch.cuda.set_rng_state(st_cuda, dev)
                kw2 = dict(kw)
                kw2.update(self._inputs(n, dev))                # ego_poses is one-shot: re-supplied for every call
                ST["mode"], ST["apply"] = arm, dp.to(dev)
                try:
                    with torch.no_grad():
                        o = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw2)
                finally:
                    ST["mode"], ST["apply"] = "own", None
            if ST["applied"] != k_app + 1:
                ST["c"]["applied_not_once"] += 1
                raise RuntimeError(f"[p4] arm {arm}: delta-p applied {ST['applied'] - k_app} times in one forward (window {key})")
            ST["c"]["arm_forwards"] += 1
            rec["arms"][arm] = {"traj": _np(o["traj"][0]), "sel": int(o["sel_idx"][0]), "forced": True, "wall": round(time.time() - t_a, 3)}
            del o
        ST["cap"] = None
        ST["records"].append(rec)
        ST["n_windows"] += 1
        nw = ST["n_windows"]
        if nw % int(A.save_every) == 0:
            save(False)
        if nw % 50 == 0 or nw <= 3:
            el = time.time() - ST["t0"]
            iw = rec.get("inv", {}).get("wall", 0.0)
            print(f"ZZP4PROGZZ {nw} el={el:.0f}s own={w_own:.2f}s inv={iw:.2f}s arms="
                  f"{sum(v['wall'] for k, v in rec['arms'].items() if k != 'OWN'):.2f}s", flush=True)
        if A.budget_probe_windows and nw == int(A.budget_probe_windows):
            el = time.time() - ST["t_first"]
            inv = sum(r.get("inv", {}).get("wall", 0.0) for r in ST["records"])
            proj_this = el / nw * 4754
            proj_other = (el - inv) / nw * 4754 if A.delta_mode == "compute" else proj_this
            proj = proj_this + proj_other
            print(f"ZZP4BUDGETZZ windows={nw} elapsed_s={el:.0f} inv_s={inv:.0f} projected_this_s={proj_this:.0f} "
                  f"projected_both_seeds_s={proj:.0f} budget_s={A.budget_max_s:.0f}", flush=True)
            if A.budget_max_s and proj > float(A.budget_max_s):
                save(False, {"over_budget": {"projected_both_seeds_s": proj, "budget_s": A.budget_max_s}})
                print("ZZP4OVERBUDGETZZ stopping before the budget is spent", flush=True)
                raise SystemExit(7)
        return out


# ============================================================================================ save
def save(final: bool, extra: dict | None = None):
    recs = ST["records"]
    if not recs:
        return
    out = {"clip_index": np.array([r["ci"] for r in recs], np.int64), "ws": np.array([r["ws"] for r in recs], np.int64),
           "gt_label": np.array([bool(r["gt_label"]) for r in recs])}
    for arm in ("OWN",) + ARMS:
        nan8 = np.full((N_SLOT, 2), np.nan, np.float32)
        out[f"has_{arm}"] = np.array([arm in r["arms"] for r in recs])
        out[f"traj_{arm}"] = np.stack([r["arms"][arm]["traj"] if arm in r["arms"] else nan8 for r in recs]).astype(np.float32)
        out[f"sel_{arm}"] = np.array([r["arms"][arm]["sel"] if arm in r["arms"] else -1 for r in recs], np.int64)
        out[f"forced_{arm}"] = np.array([bool(r["arms"][arm]["forced"]) if arm in r["arms"] else False for r in recs])
        out[f"wall_{arm}"] = np.array([r["arms"][arm]["wall"] if arm in r["arms"] else np.nan for r in recs], np.float64)
        if arm != "OWN":
            out[f"dpsum_{arm}"] = np.array([np.nan if r["dp_sum"].get(arm) is None else r["dp_sum"][arm] for r in recs], np.float64)
    if A.delta_mode == "compute":
        out["reach_pairs"] = np.stack([r["inv"]["pairs"] for r in recs]).astype(np.int64)
        for arm in ARMS:
            out[f"loss0_{arm}"] = np.array([r["inv"]["loss0"].get(arm, np.nan) for r in recs], np.float64)
            out[f"loss1_{arm}"] = np.array([r["inv"]["loss1"].get(arm, np.nan) for r in recs], np.float64)
            out[f"relnorm_{arm}"] = np.array([r["inv"]["rel_norm"].get(arm, np.nan) for r in recs], np.float64)
        out["inv_wall"] = np.array([r["inv"]["wall"] for r in recs], np.float64)
        out["refine_repro_maxdiff"] = np.array([r["inv"]["refine_repro_maxdiff"] for r in recs], np.float64)
        out["surrogate_maxdiff0"] = np.array([r["inv"].get("surrogate_maxdiff0", np.nan) for r in recs], np.float64)
        out["surrogate_argmax_agree0"] = np.array([r["inv"].get("surrogate_argmax_agree0", np.nan) for r in recs], np.float64)
        out["donor_ws"] = np.array([(-1 if r.get("donor") is None else r["donor"][1]) for r in recs], np.int64)
        out["donor_clip"] = np.array([("" if r.get("donor") is None else r["donor"][0]) for r in recs])
    meta = {"tool": "p4_roll.py", "prereg_sha256": SEAL_SHA, "script_sha256": sha256_file(__file__), "arms": list(ARMS),
            "pairs": list(PAIRS), "bands_fine_rows": [list(b) for b in BANDS_FINE], "crop": CROP, "child_argv": CHILD,
            "delta_mode": A.delta_mode, "delta_cache": A.delta_cache, "map_gt_root": A.map_gt_root, "remap": REMAP_REC,
            "inv": {"steps": A.inv_steps, "lr_rel": A.inv_lr_rel, "lambda": A.inv_lambda, "amp": A.inv_amp, "chunk": A.inv_chunk},
            "zero_delta_identity_mode": ZERO_DELTA, "final": bool(final), "n_windows": len(recs), "counters": ST["c"],
            "donor_census": ST.get("donor_census"), "wall_s": round(time.time() - ST["t0"], 1), "saved": time.strftime("%FT%T%z"),
            **(extra or {})}
    out["meta_json"] = np.array(json.dumps(meta, default=str))
    p = Path(A.p4_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)


def main() -> int:
    _wrap_build_eval_dataset(RR.L, A.map_gt_root)
    RR.Refcv6Proxy = P4Proxy                                     # install_patches' load_model resolves the name at call time
    print(f"[p4] prereg {'sealed ' + SEAL_SHA[:16] if SEAL_SHA else 'UNSEALED'} arms={ARMS} mode={A.delta_mode} "
          f"zero_delta={ZERO_DELTA} child={' '.join(CHILD)} ZZP4ROLLSTARTZZ", flush=True)
    t0 = time.time()
    if DEV_IS_CUDA:
        torch.cuda.reset_peak_memory_stats()
    rc = PF.main()
    _cache_flush()
    if ST["n_windows"] == 0:
        print("ZZP4NOWINDOWSZZ the P4Proxy never ran", flush=True)
        return 4
    walls = {arm: float(np.nansum([r["arms"][arm]["wall"] for r in ST["records"] if arm in r["arms"]])) for arm in ("OWN",) + ARMS}
    extra = {"pf_main_rc": rc, "wall_by_arm_s": walls, "roll_wall_s": round(time.time() - t0, 1),
             "inv_wall_s": float(sum(r.get("inv", {}).get("wall", 0.0) for r in ST["records"])),
             "cuda_max_mem_gib": (torch.cuda.max_memory_allocated() / 2 ** 30 if DEV_IS_CUDA else None)}
    save(True, extra)
    print(f"[p4] windows={ST['n_windows']} rc={rc} walls={json.dumps({k: round(v, 1) for k, v in walls.items()})} "
          f"inv={extra['inv_wall_s']:.0f}s counters={json.dumps(ST['c'])} {'ZZP4ROLLOKZZ' if rc == 0 else 'ZZP4ROLLBADZZ'}", flush=True)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
