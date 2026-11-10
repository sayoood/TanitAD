#!/usr/bin/env python3
"""S0 of PREREG_VDAC_P6.md (FROZEN): the true-raster ceiling of a drivable-area check on P6's selection. CPU only, navtrain only.

Population : F:/tanitad_scratch/p6fit/fit_pack8.npz (8,489 navtrain FIT sets), restricted to tokens present in the TRUE raster store
             (label_store, sha256 4a17db67...). A set's log comes from the PACK (`log`); the store shard of that log must hold the token.
Field      : the TRUE raster (int8 SDF in the store) -> MASK (sdf > 0) -> Euclidean distance transform on the raster's own 0.5 m grid
             (DESIGN 5.1 step 3, exactly as AUXVETO read every raster, GT included: `auxveto_read.sdf_from_mask`). Stored-SDF = sensitivity only.
d_k        : DESIGN 5.1: t0 pose + 8 poses, yaw unwrapped, linear to 0.1 s (41 poses), AUXVETO footprint (half-length 2.588, half-width 1.1485,
             centre 1.461 m ahead of the rear axle), 4 corners = 164 points, bilinear field samples, min over points inside the extent;
             no point inside -> q_k = 1.  POSE-ONLY (the tracked footprint is not admitted).
Arms       : O-DAC (drop NAVSIM DAC = 0), V0 / V25 (drop d < 0 / d < -0.25; all dropped -> P6's pick), Phi-GT sigma 0.5/1.0/2.0 (p_DAC x Phi(d/sigma)).
Pick       : argmax NC x DAC' x (5 EP + 5 TTC + 2 C) / 12 on P6's logits, torch float32, first index on ties (sel_common.agg_torch recipe).
Quantity   : R = gain(rule) / gain(O-DAC), gain = mean PDMS x100 of the pick minus that of P6's pick; paired log-cluster bootstrap,
             10,000 draws, seed 20260927 (sel_common.log_boot_idx draws).  R_max = largest point R over V0 and the three Phi-GT rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import zlib

import numpy as np

CARD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/vdac/PREREG_VDAC_P6.md"
CARD_SHA = "5594c2ae777a53ed4a8a873ca374c12943eadc74385cb07671c2ac90a3292ae4"
PACK = "F:/tanitad_scratch/p6fit/fit_pack8.npz"
STORE = "E:/Projects/TanitAD-artifacts/aux_drivable/label_store"
STORE_SHA_PREFIX = "4a17db67"
SELP = "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107"
B_DRAWS, SEED = 10000, 20260927
GO_R, NEAR_R = 0.33, 0.28
SIGMAS = (0.5, 1.0, 2.0)
VETO_MARGINS = {"V0": 0.0, "V25": 0.25}
BIG_M = 1.0e3                                    # all-drivable / no-background fields
CAVEAT = ("IN-SAMPLE CAVEAT (card S0): the FIT sets are in-sample for P6, so P6 fails on them less often than on navtest (PDMS-0 picks "
          "1.01 % in-sample vs 3.06 % on E1); R is a ratio of gains and is read as a CEILING ESTIMATE, not a navtest value.")

# ---- footprint: nuPlan Pacifica == geodac.py / navsim_dac.py:35-37 ----
HALF_LENGTH = (4.049 + 1.127) / 2.0
HALF_WIDTH = 2.297 / 2.0
REAR_AXLE_TO_CENTER = (4.049 - 1.127) / 2.0
SRC_DT, SRC_N, DST_DT, DST_N = 0.5, 8, 0.1, 41
# ---- the true raster's grid (aux_label_store.py): x in [-8, 72), y in [-32, 32), 0.5 m, [160, 128], index [ix, iy]
X0, Y0, RES, NX, NY = -8.0, -32.0, 0.5, 160, 128


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================ geometry (verbatim DESIGN 5.1 / geodac.py)
def densify(poses8):
    """[..., 8, 3] poses at 0.5..4.0 s -> [..., 41, 3] at 0.0..4.0 s; linear in x, y and the UNWRAPPED heading, t0 pose = origin."""
    p = np.asarray(poses8, np.float64)
    lead = p.shape[:-2]
    full = np.concatenate([np.zeros(lead + (1, 3)), p], axis=-2)
    hd = np.unwrap(full[..., 2], axis=-1)
    t_q = np.arange(DST_N) * DST_DT
    j = np.minimum((t_q / SRC_DT + 1e-9).astype(int), SRC_N - 1)
    w = (t_q - j * SRC_DT) / SRC_DT
    out = np.empty(lead + (DST_N, 3))
    for c in (0, 1):
        out[..., c] = full[..., j, c] * (1 - w) + full[..., j + 1, c] * w
    out[..., 2] = hd[..., j] * (1 - w) + hd[..., j + 1] * w
    return out


def corners(poses):
    """[..., T, 3] rear-axle poses -> [..., T, 4, 2] footprint corners."""
    p = np.asarray(poses, np.float64)
    ch, sh = np.cos(p[..., 2]), np.sin(p[..., 2])
    cx, cy = p[..., 0] + REAR_AXLE_TO_CENTER * ch, p[..., 1] + REAR_AXLE_TO_CENTER * sh
    out = np.empty(p.shape[:-1] + (4, 2))
    for i, (lo, la) in enumerate(((HALF_LENGTH, HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, -HALF_WIDTH))):
        out[..., i, 0] = cx + lo * ch - la * sh
        out[..., i, 1] = cy + lo * sh + la * ch
    return out


def sample_bilinear(field, pts, x0=X0, y0=Y0, res=RES):
    """Signed field at ego-frame points, bilinear between cell centres (edge replicate in the half-cell border); NaN outside the extent."""
    nx, ny = field.shape
    fx, fy = (pts[..., 0] - x0) / res - 0.5, (pts[..., 1] - y0) / res - 0.5
    ok = (fx >= -0.5) & (fx < nx - 0.5) & (fy >= -0.5) & (fy < ny - 0.5)
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    a0, a1 = np.clip(ix, 0, nx - 1), np.clip(ix + 1, 0, nx - 1)
    b0, b1 = np.clip(iy, 0, ny - 1), np.clip(iy + 1, 0, ny - 1)
    v = field[a0, b0] * (1 - tx) * (1 - ty) + field[a1, b0] * tx * (1 - ty) + field[a0, b1] * (1 - tx) * ty + field[a1, b1] * tx * ty
    return np.where(ok, v, np.nan), ok


def dk_proposals(field, poses8, x0=X0, y0=Y0):
    """poses8 [M, 8, 3] -> (d_k [M], n_inside [M], n_points): min bilinear field value over the points inside the extent; +inf if none inside."""
    pts = corners(densify(poses8)).reshape(poses8.shape[0], -1, 2)              # [M, 164, 2]
    v, ok = sample_bilinear(field, pts, x0, y0)
    d = np.where(ok, v, np.inf).min(axis=1)
    return d, ok.sum(axis=1), pts.shape[1]


def sdf_from_mask(mask, cell_m=RES):
    """The store's own signed_distance on the raster's OWN grid: metres, + inside, half-cell corrected, no pad, no clip; degenerate masks read +-BIG_M."""
    from scipy.ndimage import distance_transform_edt as edt
    m = np.asarray(mask, bool)
    if not m.any():
        return np.full(m.shape, -BIG_M)
    if m.all():
        return np.full(m.shape, BIG_M)
    return np.where(m, edt(m) - 0.5, -(edt(~m) - 0.5)) * cell_m


# ================================================================ selection
def p6_probs(logits):
    import torch
    return torch.as_tensor(np.asarray(logits, np.float32)).sigmoid()


def pick_from_probs(p, dac_mult=None):
    """navsim_v1 rule on probabilities p [N, M, 6]; optionally DAC' = p_DAC * dac_mult [N, M] (float32). torch float32, first index on ties."""
    import torch
    w1 = torch.tensor([5.0, 5.0, 2.0], dtype=p.dtype)
    dac = p[..., 1] if dac_mult is None else p[..., 1] * torch.as_tensor(np.asarray(dac_mult, np.float32))
    a = p[..., 0] * dac * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()
    return torch.argmax(a, dim=-1).numpy().astype(np.int64), a.numpy()


def masked_pick(p, veto, base_pick):
    """AUXVETO rule: highest P6 score among non-vetoed proposals (first index on ties); all vetoed -> P6's pick."""
    _, a = pick_from_probs(p)
    out = np.empty(len(base_pick), np.int64)
    for i in range(len(base_pick)):
        out[i] = base_pick[i] if veto[i].all() else int(np.argmax(np.where(veto[i], -np.inf, a[i])))
    return out


def phi(x):
    from scipy.special import ndtr
    return ndtr(x)


def arm_picks(p, base_pick, dk, dac_nav, sigmas=SIGMAS):
    """All S0 arms for one field. dk [N, M] (+inf = no point inside), dac_nav [N, M] NAVSIM DAC sub-score. -> {arm: pick [N]}."""
    out = {"O-DAC": masked_pick(p, dac_nav == 0, base_pick)}
    for nm, m in VETO_MARGINS.items():
        out[nm] = masked_pick(p, dk < -m, base_pick)
    for s in sigmas:
        out[f"Phi-GT{s}"] = pick_from_probs(p, phi(dk / s))[0]
    return out


def gains(pdms100, picks, base_pick):
    ar = np.arange(len(base_pick))
    return pdms100[ar, picks] - pdms100[ar, base_pick]


# ================================================================ estimator
class Boot:
    def __init__(self, log, B=B_DRAWS, seed=SEED):
        if SELP not in sys.path:
            sys.path.insert(0, SELP)
        import sel_common as sc
        self.ul, self.li, self.draws = sc.log_boot_idx(np.asarray(log), B, seed)
        self.nl = len(self.ul)

    def per_draw(self, vals):
        v = np.asarray(vals, np.float64)
        s = np.bincount(self.li, weights=v, minlength=self.nl)
        c = np.bincount(self.li, minlength=self.nl).astype(np.float64)
        return float(s.sum() / c.sum()), s[self.draws].sum(1) / c[self.draws].sum(1)


def ci(r):
    r = np.asarray(r)
    r = r[np.isfinite(r)]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def ratio_block(boot, g_rule, g_odac):
    pt_r, d_r = boot.per_draw(g_rule)
    pt_o, d_o = boot.per_draw(g_odac)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = d_r / d_o
    R = pt_r / pt_o if pt_o != 0 else float("nan")
    return {"gain": pt_r, "gain_ci": ci(d_r), "R": R, "R_ci": ci(rd), "n_changed": None}, rd


def outcome(r_max):
    """Card table: GO R_max >= 0.33; NEAR 0.28 <= R_max < 0.33; STOP < 0.28."""
    if not math.isfinite(r_max):
        return "INVALID"
    return "S0-GO" if r_max >= GO_R else ("S0-NEAR" if r_max >= NEAR_R else "S0-STOP")


# ================================================================ controls
def zero_control(p, base_pick, dk_free, dac_nav, pdms100):
    """An all-drivable field gives Delta = 0.0 exactly and 0 proposals changed, for every field-driven arm (O-DAC is not field-driven)."""
    picks = {k: v for k, v in arm_picks(p, base_pick, dk_free, dac_nav).items() if k != "O-DAC"}
    ch = {k: int((v != base_pick).sum()) for k, v in picks.items()}
    dl = {k: float(gains(pdms100, v, base_pick).sum()) for k, v in picks.items()}
    return {"ok": all(x == 0 for x in ch.values()) and all(x == 0.0 for x in dl.values()), "n_changed": ch, "delta_sum": dl}


def pick_reproduction(base_pick, logits):
    """The pack records NO pick: compare the reader's pick with sel_common.pick_from_logits recomputed from the pack's logits."""
    if SELP not in sys.path:
        sys.path.insert(0, SELP)
    import sel_common as sc
    ref = sc.pick_from_logits(np.asarray(logits, np.float32), "torch", lane=False).astype(np.int64)
    return {"ok": bool((ref == base_pick).all()), "n_sets": int(len(ref)), "n_equal": int((ref == base_pick).sum()),
            "note": "the pack records no pick; recomputed pick from the pack's logits via sel_common.pick_from_logits(torch, lane=False)"}


def odac_dominates(gain_true):
    """O-DAC's gain >= every rule's gain (points)."""
    od = gain_true["O-DAC"]
    rules = {k: v for k, v in gain_true.items() if k != "O-DAC"}
    return {"ok": all(od >= v for v in rules.values()), "O-DAC": od, "rules": rules}


def yflip_below(r_true, r_flip):
    """Decision quantity: the y-flipped field's R_max must lie below the true field's R_max (per-rule flags reported)."""
    rm_t, rm_f = max(r_true.values()), max(r_flip.values())
    return {"ok": bool(rm_f < rm_t), "R_max_true": rm_t, "R_max_flip": rm_f,
            "per_rule_flip_below_true": {k: bool(r_flip[k] < r_true[k]) for k in r_true}}


# ================================================================ data
def read_shard(path):
    with open(path, "rb") as f:
        if f.read(8) != b"AUXSDF1\n":
            raise ValueError("bad magic")
        (hl,) = struct.unpack("<I", f.read(4))
        head = json.loads(f.read(hl))
        base = f.tell()
        lab = {}
        for e in head["entries"]:
            f.seek(base + e["o"])
            lab[e["t"]] = np.frombuffer(zlib.decompress(f.read(e["l"])), np.int8).reshape(NX, NY)
    return head, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--limit-sets", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    got = sha256_file(CARD)
    if got != CARD_SHA:
        raise SystemExit(f"REFUSED: card sha256 {got} != frozen {CARD_SHA}")
    print("card sha256 OK", got)
    Z = np.load(PACK, allow_pickle=False)
    man = json.load(open(f"{STORE}/manifest.json"))
    if not str(man["store_sha256"]).startswith(STORE_SHA_PREFIX):
        raise SystemExit(f"label_store sha256 {man['store_sha256']} does not start with {STORE_SHA_PREFIX}")
    log_all, tok_all = Z["log"], Z["token"]
    keep = np.zeros(len(tok_all), bool)
    labels = {}
    for lg in sorted(set(log_all.tolist())):
        path = f"{STORE}/shards/{lg}.aux1"
        if not os.path.exists(path):
            continue
        _, lab = read_shard(path)
        for i in np.where(log_all == lg)[0]:
            t = str(tok_all[i])
            if t in lab:
                keep[i] = True
                labels[(lg, t)] = lab[t]
    sel = np.where(keep)[0]
    if a.limit_sets:
        sel = sel[: a.limit_sets]
    counts = {"pack_sets": int(len(tok_all)), "pack_logs": int(len(set(log_all.tolist()))), "sets_in_label_store": int(len(sel)),
              "logs_in_label_store": int(len(set(log_all[sel].tolist()))), "pack_has_pdm_all": bool(Z["has_pdm"].all()),
              "heldout_flagged_sets_in_population": int(Z["is_heldout"][sel].sum())}
    print("COUNTS BEFORE THE READ", json.dumps(counts), flush=True)
    log = log_all[sel]
    logits = Z["logits"][sel].astype(np.float32)
    pdm = Z["pdm"][sel].astype(np.float32)
    if str(Z["pdm_names"][6]) != "pdms" or str(Z["pdm_names"][1]) != "dac":
        raise SystemExit("pdm head order is not NC, DAC, EP, TTC, C, DDC, PDMS")
    pdms100 = 100.0 * pdm[..., 6].astype(np.float64)
    dac_nav = pdm[..., 1]
    poses = np.concatenate([Z["traj_xy"][sel].astype(np.float64), Z["yaw"][sel].astype(np.float64)[..., None]], axis=-1)   # [N, 64, 8, 3]
    N = len(sel)
    p = p6_probs(logits)
    base_pick, _ = pick_from_probs(p)

    dk = {k: np.empty((N, 64)) for k in ("true", "flip", "free", "stored")}
    n_in = np.empty((N, 64))
    npts = 164
    for n, i in enumerate(sel):
        lab = labels[(str(log_all[i]), str(tok_all[i]))]
        mask = lab > 0
        dk["true"][n], n_in[n], npts = dk_proposals(sdf_from_mask(mask), poses[n])
        dk["flip"][n] = dk_proposals(sdf_from_mask(mask[:, ::-1]), poses[n])[0]
        dk["free"][n] = dk_proposals(sdf_from_mask(np.ones((NX, NY), bool)), poses[n])[0]
        dk["stored"][n] = dk_proposals(lab.astype(np.float64) * 0.1, poses[n])[0]
    out_share = 1 - n_in.sum() / (n_in.size * npts)
    print(f"d_k done {time.time() - t0:.1f}s; footprint points outside the raster extent: {out_share:.5f}; proposals with no point inside: {int((n_in == 0).sum())}", flush=True)

    boot = Boot(log)
    picks_true = arm_picks(p, base_pick, dk["true"], dac_nav)
    picks_flip = arm_picks(p, base_pick, dk["flip"], dac_nav)
    picks_stored = arm_picks(p, base_pick, dk["stored"], dac_nav)
    g = {k: gains(pdms100, v, base_pick) for k, v in picks_true.items()}
    gf = {k: gains(pdms100, v, base_pick) for k, v in picks_flip.items()}
    gs = {k: gains(pdms100, v, base_pick) for k, v in picks_stored.items()}
    rule_arms = ["V0", "V25"] + [f"Phi-GT{s}" for s in SIGMAS]
    main_arms = ["V0"] + [f"Phi-GT{s}" for s in SIGMAS]
    arms = {"O-DAC": {"gain": boot.per_draw(g["O-DAC"])[0], "gain_ci": ci(boot.per_draw(g["O-DAC"])[1]), "n_changed": int((picks_true["O-DAC"] != base_pick).sum())}}
    rd = {}
    for k in rule_arms:
        arms[k], rd[k] = ratio_block(boot, g[k], g["O-DAC"])
        arms[k]["n_changed"] = int((picks_true[k] != base_pick).sum())
    r_true = {k: arms[k]["R"] for k in main_arms}
    r_flip = {k: float(gf[k].mean() / gf["O-DAC"].mean()) for k in main_arms}
    r_stored = {k: float(gs[k].mean() / gs["O-DAC"].mean()) for k in rule_arms}
    R_max = max(r_true.values())
    arg = max(main_arms, key=lambda k: r_true[k])
    rmax_draws = np.max(np.stack([rd[k] for k in main_arms]), axis=0)
    ctrl = {"zero_all_drivable": zero_control(p, base_pick, dk["free"], dac_nav, pdms100),
            "pick_reproduces": pick_reproduction(base_pick, logits),
            "odac_ge_all": odac_dominates({k: float(v.mean()) for k, v in g.items()}),
            "yflip_below_true": yflip_below(r_true, r_flip)}
    valid = all(v["ok"] for v in ctrl.values()) and arms["O-DAC"]["gain"] > 0
    oc = outcome(R_max) if valid else "INVALID"
    dz = dac_nav == 0
    align = {"proposals_dac0": int(dz.sum()), "share_dac0_with_d_lt0": float((dk["true"][dz] < 0).mean()),
             "share_dac1_with_d_lt0": float((dk["true"][~dz] < 0).mean()),
             "p6_pick_dac0_sets": int((dac_nav[np.arange(N), base_pick] == 0).sum())}
    res = {"stage": "S0", "card": CARD, "card_sha256": CARD_SHA, "caveat": CAVEAT, "counts": counts, "n_boot_logs": int(boot.nl),
           "bootstrap": {"draws": B_DRAWS, "seed": SEED, "estimator": "paired log-cluster bootstrap over the pack's logs (sel_common.log_boot_idx draws); "
                         "it answers 'another draw of logs' only; R CI = percentile of per-draw gain_rule / gain_O-DAC"},
           "field": "true raster mask (sdf>0) -> EDT on the 0.5 m grid, pose-only footprint, bilinear; the log of a set comes from the PACK",
           "footprint_points_outside_extent_share": float(out_share), "proposals_with_no_point_inside": int((n_in == 0).sum()),
           "p6_pick_pdms100": float(pdms100[np.arange(N), base_pick].mean()), "best_of_64_pdms100": float(pdms100.max(1).mean()),
           "arms": arms, "R_max": R_max, "R_max_arm": arg, "R_max_ci_of_max_over_arms": ci(rmax_draws),
           "outcome": oc, "valid": bool(valid), "controls": ctrl,
           "reported_not_deciding": {"R_yflipped_field": r_flip, "R_stored_sdf_field": r_stored, "alignment": align},
           "inputs": {"pack": PACK, "pack_sha256": sha256_file(PACK), "label_store_sha256": man["store_sha256"], "card_sha256": got,
                      "sel_common": SELP + "/sel_common.py", "sel_common_sha256": sha256_file(SELP + "/sel_common.py")},
           "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/s0_read.json", "w", newline="\n"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("counts", "arms", "R_max", "R_max_arm", "outcome", "valid")}, indent=1, default=float))
    print(json.dumps(ctrl, indent=1, default=float))
    print(CAVEAT)
    write_md(res, f"{a.out}/s0_read.md")
    print(f"ZZS0_DONE outcome={oc} R_max={R_max:.4f} valid={valid}")


def write_md(r, path):
    L = ["# S0 read: true-raster ceiling of a drivable-area check on P6 (navtrain FIT sets, zero GPU)", "",
         f"**Outcome: {r['outcome']}** (controls valid: {r['valid']}). R_max = **{r['R_max']:.4f}** ({r['R_max_arm']}). Card bars: GO >= 0.33, NEAR [0.28, 0.33), STOP < 0.28.", "",
         f"> {r['caveat']}", "",
         "Evidence class: MEASURED (ours; artifacts below), open-loop selection read on navtrain, no navtest token read.", "",
         "## Counts (before the read)", "", "```", json.dumps(r["counts"], indent=1), "```", "",
         f"Bootstrap: paired log-cluster, {r['bootstrap']['draws']} draws, seed {r['bootstrap']['seed']}, {r['n_boot_logs']} logs. {r['bootstrap']['estimator']}.", "",
         f"P6's pick PDMS x100 on these sets: {r['p6_pick_pdms100']:.3f}; best-of-64: {r['best_of_64_pdms100']:.3f}. "
         f"Footprint points outside the raster extent: {r['footprint_points_outside_extent_share']:.5f}; proposals with no point inside: {r['proposals_with_no_point_inside']}.", "",
         "## Arms (gain = PDMS x100 over the sets minus P6's pick; R = gain / gain(O-DAC))", "",
         "| arm | gain [95 % CI] | R [95 % CI] | picks changed |", "|---|---|---|---|"]
    for k, v in r["arms"].items():
        gc = f"{v['gain']:+.4f} [{v['gain_ci'][0]:+.4f}, {v['gain_ci'][1]:+.4f}]"
        rc = "1 (reference)" if k == "O-DAC" else f"{v['R']:.4f} [{v['R_ci'][0]:.4f}, {v['R_ci'][1]:.4f}]"
        L.append(f"| {k} | {gc} | {rc} | {v['n_changed']} |")
    lo, hi = r["R_max_ci_of_max_over_arms"]
    L += ["", f"R_max (V0 and the three Phi-GT rules; V25 is reported, not in the max) = {r['R_max']:.4f}, arm {r['R_max_arm']}; "
          f"CI of the max over those four arms [{lo:.4f}, {hi:.4f}] (the point decides, per the card).", "",
          "## Controls", "", "```", json.dumps(r["controls"], indent=1), "```", "",
          "## Reported, not deciding", "", "```", json.dumps(r["reported_not_deciding"], indent=1), "```", "",
          "## Inputs", "", "```", json.dumps(r["inputs"], indent=1), "```", "",
          f"Elapsed {r['elapsed_s']} s. Method: `s0_read.py` (this folder), DESIGN_VDAC_P6.md 5.1 verbatim, pose-only.", ""]
    open(path, "w", newline="\n").write("\n".join(L))


if __name__ == "__main__":
    main()
