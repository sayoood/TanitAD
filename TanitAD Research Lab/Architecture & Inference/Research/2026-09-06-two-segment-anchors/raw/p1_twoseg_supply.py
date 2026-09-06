"""P1 - THE TWO-SEGMENT SUPPLY MEASUREMENT, against the PREREG bars.

⛔ Model-free and DETERMINISTIC. The fan is reconstructed from the checkpoint's
own `anchor_controls` through the programme's own integrator, and the target is
the RECORDED 6 s ego path. No training draw, no inference sampling, no episode
resampling ⇒ the only variance the addendum had was its 5,000-window subsample,
and this scores EVERY scoreable window instead of removing it with a CI.

⛔ THIS IS A SUPPLY CEILING. It says what the best possible selector could reach
with the extended fan. It says nothing about what refcv4b's ranker would pick and
may never be quoted beside an achievement
(`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`).

⛔ NOT `a_star` as computed in `taniteval/tools/refcv3_arm.py` -- that binds
against `decoder.anchors`, the bank rolled once at ref_speed_ms, and scores WORSE
than the arm it bounds. Everything here binds against the PER-WINDOW
v0-conditioned roll, which is the binding the TRAINER uses.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

import _env  # noqa: F401
from tanitad.refs import anchor_twoseg as ts    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPS = r"C:\Users\Admin\refav1_eval_full\eps"
CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"
HZ = 10.0
T_S = 6.0
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOTS = [h - 1 for h in HORIZONS]
SLOT_T = np.array([h / HZ for h in HORIZONS])              # 0.5 ... 6.0 s
#: ⛔ per-slot dt: 0.5 s for the first four slots, 1.0 s after. A single dt here
#: would misread SPEED by 2x on the tail -- the units family in a grid costume.
SLOT_DT = np.diff(np.concatenate([[0.0], SLOT_T]))
MIN_DS = 0.25                                              # selstab's mask
LANE_LO, LANE_HI = 2.5, 5.0
RET_NET_DEG, RET_PEAK_DEG, TURN_DEG = 10.0, 3.0, 30.0
KAPPA_CAP, ALAT_V_FLOOR, DT_TICK = 0.12, 4.0, 0.1
CHUNK = 800


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def sha(t):
    return hashlib.sha256(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


# ------------------------------------------------------------------ geometry
def path_geom(p, dt):
    """p [..., K, 2] ego-frame -> arc lengths, heading, curvature, speed."""
    z = np.zeros(p.shape[:-2] + (1, 2))
    q = np.concatenate([z, p], axis=-2)
    d = np.diff(q, axis=-2)
    ds = np.linalg.norm(d, axis=-1)
    th = np.arctan2(d[..., 1], d[..., 0])
    dth = wrap(np.diff(th, axis=-1))
    seg = 0.5 * (ds[..., :-1] + ds[..., 1:])
    ok = seg >= MIN_DS
    kap = np.where(ok, dth / np.maximum(seg, 1e-6), 0.0)
    return {"ds": ds, "heading": th, "kappa": kap, "kappa_ok": ok,
            "speed": ds / dt}


def families(p, g, dt):
    """The FOUR FAMILIES for a picked path p against GT g. Never pooled."""
    P, G = path_geom(p, dt), path_geom(g, dt)
    dv = p - g
    ade = np.linalg.norm(dv, axis=-1).mean(-1)
    gh = np.arctan2(g[..., 1], g[..., 0])
    u = np.stack([np.cos(gh), np.sin(gh)], -1)
    nvec = np.stack([-np.sin(gh), np.cos(gh)], -1)
    both = P["kappa_ok"] & G["kappa_ok"]
    nk = both.sum(-1)
    kmae = np.where(nk > 0,
                    (np.abs(P["kappa"] - G["kappa"]) * both).sum(-1)
                    / np.maximum(nk, 1), np.nan)
    hd = np.abs(wrap(P["heading"] - G["heading"]))
    return {"ade_m": ade,
            "along_mae_m": np.abs((dv * u).sum(-1)).mean(-1),
            "cross_mae_m": np.abs((dv * nvec).sum(-1)).mean(-1),
            "speed_mae_mps": np.abs(P["speed"] - G["speed"]).mean(-1),
            "curv_mae_1pm": kmae,
            "heading_mae_deg": np.degrees(hd.mean(-1)),
            "n_kappa": nk}


def roll(ctrl, v0, slots):
    """[n, N, len(slots), 2] emitted fan, chunked over windows."""
    out = np.empty((len(v0), ctrl.shape[0], len(slots), 2), dtype=np.float64)
    for s in range(0, len(v0), CHUNK):
        vv = torch.as_tensor(v0[s:s + CHUNK], dtype=torch.float32)
        out[s:s + CHUNK] = ts.roll_bank(
            ctrl, vv, control_units="alat", steps=max(HORIZONS), slots=slots,
            dt=DT_TICK, alat_v_floor=ALAT_V_FLOOR,
            kappa_cap=KAPPA_CAP).numpy().astype(np.float64)
    return out


def best(fan, GT):
    """(ade [n], idx [n]) -- oracle-in-vocabulary over the given slots."""
    a = np.linalg.norm(fan - GT[:, None], axis=-1).mean(-1)     # [n, N]
    i = a.argmin(1)
    return a[np.arange(len(i)), i], i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = ALL")
    a = ap.parse_args()

    # ---- the vocabulary, from the checkpoint itself ------------------------
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    d = sd.get("model", sd)
    C0 = d["core.decoder.anchor_controls"].float()
    print(f"[bank] source controls {tuple(C0.shape)} sha256 {sha(C0)}")
    n_base = C0.shape[0]
    C0_3 = ts.as_three_column(C0, T_S)
    EXT = ts.extend_controls(C0, T_S)                       # 123 x 3
    # deliberate-regression arms
    R1 = ts.extend_controls(C0, T_S, t_split_s=(6.0, 6.0, 6.0))   # never flips
    R2 = ts.extend_controls(C0, T_S, t_split_s=(0.0, 0.0, 0.0))   # flips at t0
    R3 = ts.extend_controls(C0, T_S, t_split_s=(3.0,))            # 1 split, 2
    # controls at KNOWN values
    STRAIGHT_FAM = C0[C0[:, 1] == 0.0]                  # the a_lat = 0 sub-fan
    CV = torch.tensor([[0.0, 0.0]])                     # one constant-v line
    print(f"[bank] ext {tuple(EXT.shape)} R1 {tuple(R1.shape)} "
          f"R2 {tuple(R2.shape)} R3 {tuple(R3.shape)} "
          f"straight-family {tuple(STRAIGHT_FAM.shape)}")

    # ---- windows + recorded 6 s GT ----------------------------------------
    with open(a.arm_json, encoding="utf-8") as fh:
        man = json.load(fh)["refcv3"]["manifest"]
    clip_of = {int(e["file_index"]): e["clip_id"] for e in man["episodes"]}
    off = int(man["corpus"]["frames"]["provider_to_raw_frame_offset"])
    n_fut = int(round(T_S * HZ))
    v0s, gts, tag, n_dump = [], [], [], 0
    for fi in sorted(clip_of):
        z = np.load(os.path.join(a.dump, f"ep{fi:03d}.npz"), allow_pickle=True)
        ws, v0 = z["ws"].astype(int), z["v0"].astype(np.float64)
        n_dump += len(ws)
        p = torch.load(os.path.join(EPS, clip_of[fi] + ".v2ep.pt"),
                       map_location="cpu",
                       weights_only=False)["poses"].numpy().astype(np.float64)
        for i, w in enumerate(ws):
            j0 = w + off
            if j0 + n_fut >= len(p):
                continue
            seg = p[j0:j0 + n_fut + 1]
            c, s = np.cos(-seg[0, 2]), np.sin(-seg[0, 2])
            dx, dy = seg[:, 0] - seg[0, 0], seg[:, 1] - seg[0, 1]
            lon = c * dx - s * dy
            lat = s * dx + c * dy
            yaw = wrap(seg[:, 2] - seg[0, 2])
            gts.append(np.stack([lon[list(HORIZONS)], lat[list(HORIZONS)]], -1))
            v0s.append(v0[i])
            dyt, dyp = np.degrees(yaw[-1]), np.degrees(np.abs(yaw).max())
            lc = (LANE_LO <= abs(lat[-1]) < LANE_HI) and \
                 abs(dyt) <= RET_NET_DEG and dyp >= RET_PEAK_DEG
            tag.append((1 if lc else 0, 1 if abs(dyt) >= TURN_DEG else 0))
    GT = np.asarray(gts)
    V0 = np.asarray(v0s)
    TAG = np.asarray(tag)
    if a.max_windows and len(V0) > a.max_windows:
        keep = np.sort(np.concatenate([
            np.flatnonzero(TAG[:, 0] == 1),
            np.random.default_rng(0).choice(
                np.flatnonzero(TAG[:, 0] == 0),
                a.max_windows - int(TAG[:, 0].sum()), replace=False)]))
        GT, V0, TAG = GT[keep], V0[keep], TAG[keep]
    n = len(V0)
    LC, TN = TAG[:, 0] == 1, TAG[:, 1] == 1
    print(f"[grid] corpus = the 141 B1-v7.2 EVAL clips, refcv4b stride-1 dump: "
          f"{n_dump} dumped windows -> {n} with a full {T_S} s recorded future")
    print(f"[grid] n_lane_change = {int(LC.sum())}  n_turn_control = "
          f"{int(TN.sum())}  n_other = {int((~LC & ~TN).sum())}")
    print(f"[grid] v0 range {V0.min():.2f} - {V0.max():.2f} m/s")

    res = {"corpus": "141 B1-v7.2 EVAL clips (refcv4b T1 stride-1 dump)",
           "n_dumped_windows": int(n_dump), "n_scoreable_windows": int(n),
           "n_lane_change": int(LC.sum()), "n_turn": int(TN.sum()),
           "n_other": int((~LC & ~TN).sum()),
           "source_controls_sha256": sha(C0), "arms": {}}

    banks = {"base117": C0, "base117_3col": C0_3, "ext123": EXT,
             "R1_split_at_horizon": R1, "R2_split_at_zero": R2,
             "R3_single_split_3s": R3,
             "C_straight_family": STRAIGHT_FAM, "C_const_velocity": CV}
    fans, ades, idxs = {}, {}, {}
    for k, ctrl in banks.items():
        fans[k] = roll(ctrl, V0, SLOTS)
        ades[k], idxs[k] = best(fans[k], GT)
        print(f"[roll] {k:22s} N={ctrl.shape[0]:>3d}  "
              f"ALL {ades[k].mean():.4f}  LC {ades[k][LC].mean():.4f}  "
              f"TURN {ades[k][TN].mean():.4f}")
    # the ZERO path: no information at all
    z_ade = np.linalg.norm(GT, axis=-1).mean(-1)
    print(f"[roll] {'C_zero_path':22s} N=  0  ALL {z_ade.mean():.4f}  "
          f"LC {z_ade[LC].mean():.4f}  TURN {z_ade[TN].mean():.4f}")

    base = ades["base117"]
    for k in banks:
        res["arms"][k] = {
            "n_candidates": int(banks[k].shape[0]),
            "ade_all": float(ades[k].mean()),
            "ade_lane_change": float(ades[k][LC].mean()),
            "ade_turn": float(ades[k][TN].mean()),
            "gain_lane_change_vs_base": float(base[LC].mean()
                                              - ades[k][LC].mean()),
            "gain_all_vs_base": float(base.mean() - ades[k].mean()),
            "gain_turn_vs_base": float(base[TN].mean() - ades[k][TN].mean()),
            "n_windows_improved": int((ades[k] < base - 1e-12).sum()),
            "n_lc_windows_improved": int((ades[k][LC] < base[LC] - 1e-12).sum()),
        }
    res["arms"]["C_zero_path"] = {
        "n_candidates": 0, "ade_all": float(z_ade.mean()),
        "ade_lane_change": float(z_ade[LC].mean()),
        "ade_turn": float(z_ade[TN].mean())}

    # ---- B1' : the 3-column base MUST be bit-identical to the 2-column base
    same = np.array_equal(fans["base117"], fans["base117_3col"])
    print(f"\n[B2] base117 vs base117_3col over {fans['base117'].size:,} "
          f"floats: {'BIT-IDENTICAL' if same else 'DIFFERS'}")
    res["B2_single_segment_limit_bit_identical"] = bool(same)

    # ---- B5 : the 2 s grid is a STRUCTURAL ZERO ---------------------------
    s2 = list(range(4))
    f2b = roll(C0, V0, [SLOTS[i] for i in s2])
    f2e = roll(EXT, V0, [SLOTS[i] for i in s2])
    a2b, i2b = best(f2b, GT[:, s2])
    a2e, i2e = best(f2e, GT[:, s2])
    d2 = float(np.abs(a2b - a2e).max())
    picks_new_2s = int((i2e >= n_base).sum())
    print(f"[B5] 2 s grid (slots 0-3): max |ADE change| = {d2:.10e} m; "
          f"windows whose 2 s argmin is a NEW candidate = {picks_new_2s}")
    res["B5_2s_max_ade_change"] = d2
    res["B5_2s_windows_picking_new"] = picks_new_2s

    # ---- B7 : SUPERVISION rate -- where the geometric a_star lands ---------
    ie = idxs["ext123"]
    new = ie >= n_base
    print(f"\n[B7] geometric a_star on the EXTENDED bank lands on a "
          f"two-segment candidate:")
    print(f"     lane-change windows  {int(new[LC].sum()):>6d} / "
          f"{int(LC.sum()):>6d}  = {100.0 * new[LC].mean():6.2f} %")
    print(f"     turn windows         {int(new[TN].sum()):>6d} / "
          f"{int(TN.sum()):>6d}  = {100.0 * new[TN].mean():6.2f} %")
    print(f"     ALL windows          {int(new.sum()):>6d} / {n:>6d}  "
          f"= {100.0 * new.mean():6.2f} %")
    res["B7_supervision_rate"] = {
        "lane_change": {"n": int(LC.sum()), "k": int(new[LC].sum()),
                        "rate": float(new[LC].mean())},
        "turn": {"n": int(TN.sum()), "k": int(new[TN].sum()),
                 "rate": float(new[TN].mean())},
        "all": {"n": int(n), "k": int(new.sum()), "rate": float(new.mean())}}
    # which of the six, and at which split
    per = {}
    for j in range(n_base, EXT.shape[0]):
        m = ie == j
        per[f"a_lat={float(EXT[j,1]):+.2f}_tsplit={float(EXT[j,2]):.1f}s"] = {
            "n_all": int(m.sum()), "n_lane_change": int((m & LC).sum())}
    res["B7_per_candidate"] = per
    print("     per candidate (all / lane-change):")
    for k2, v2 in per.items():
        print(f"       {k2:32s} {v2['n_all']:>6d} / {v2['n_lane_change']:>5d}")

    # ---- FOUR FAMILIES on the picked path, never pooled --------------------
    print(f"\n[families] the picked (oracle-in-vocabulary) path vs GT, 6 s grid")
    hdr = (f"{'arm':<22}{'n':>7}{'ADE':>9}{'along':>9}{'cross':>9}"
           f"{'speed':>9}{'curv':>10}{'head':>8}")
    fam = {}
    for stratum, mask in (("lane_change", LC), ("turn", TN), ("all",
                          np.ones(n, bool))):
        print(f"  -- {stratum} (n = {int(mask.sum())}) --")
        print("  " + hdr)
        fam[stratum] = {}
        for k in ("base117", "ext123", "R1_split_at_horizon",
                  "C_straight_family", "C_const_velocity"):
            pick = fans[k][np.arange(n), idxs[k]]
            f = families(pick[mask], GT[mask], SLOT_DT)
            row = {kk: float(np.nanmean(vv)) for kk, vv in f.items()
                   if kk != "n_kappa"}
            row["n"] = int(mask.sum())
            row["n_kappa_valid"] = int((f["n_kappa"] > 0).sum())
            fam[stratum][k] = row
            print(f"  {k:<22}{row['n']:>7d}{row['ade_m']:>9.4f}"
                  f"{row['along_mae_m']:>9.4f}{row['cross_mae_m']:>9.4f}"
                  f"{row['speed_mae_mps']:>9.4f}{row['curv_mae_1pm']:>10.6f}"
                  f"{row['heading_mae_deg']:>8.3f}")
    res["four_families_6s"] = fam

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
