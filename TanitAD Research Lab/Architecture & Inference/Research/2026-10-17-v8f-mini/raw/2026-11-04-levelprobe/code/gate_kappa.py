#!/usr/bin/env python3
"""gate_kappa.py -- G-KAPPA of PREREG_FROZEN.md (H-KAPPA-T1-1).  Offline, no rollout, numpy only.

Exact additive partition of the curvature-MAE excess E = mean_w (row_X - row_P10) of the matched-thread S runs (out_m16) into DIP / HIGHK / REST,
the low-pass fraction f_LF, and the committed lever decision (a) / (b) / (c) / (d).  Refuses to run on a card that does not hash to FROZEN_SHA.txt.
Tier T1-kin; evidence class MEASURED (ours).  No interval is computed here: the gate is a point decomposition (no estimator; it decides a lever, not a claim).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
OUT_M16 = "F:/TanitAD-scratch/smooth_t1/out_m16"
MANIFEST = "F:/TanitAD-scratch/soup_t1_out/runs_soup.json"
LANDED_Y = f"{ART}/dagland/gapdecomp/raw/arrays/CTRL_s0.npz"
RUNS = {"CTRL_s0": ("SMOOTH_DAGOACC_s0", "CTRL", 0), "CTRL_s1": ("SMOOTH_DAGOACC_s1", "CTRL", 1),
        "SOUP_s0": ("SMOOTH_SOUP_s0", "SOUP", 0), "SOUP_s1": ("SMOOTH_SOUP_s1", "SOUP", 1)}
DT, MIN_DS_MPS, EPS = 0.2, 0.5, 1e-8          # taniteval.four_families: DT of the cache grid, MIN_DS_MPS, _EPS
KAPPA_MAX = 0.2                               # feasible_decode.KAPPA_MAX_1PM
V_DIP = 0.5                                   # m/s, the card's dip speed
V_SLOW_HIGHK = 2.0                            # m/s, beside only
BAND = (3.0, 8.0)
SHARE_BAR = 0.5


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_dump(d):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


# ------------------------------------------------------------------------------------------------ pure layer
def geom(wp):
    """numpy float64 transcription of taniteval.four_families._seq_geometry(wp, dt=0.2) (KG1 pins it against the landed arrays)."""
    wp = np.asarray(wp, np.float64)
    p = np.concatenate([np.zeros_like(wp[:, :1]), wp], 1)
    d = p[:, 1:] - p[:, :-1]
    ds = np.linalg.norm(d, axis=-1)
    valid = ds > MIN_DS_MPS * DT
    hd = np.arctan2(d[..., 1], d[..., 0])
    dh = hd[:, 1:] - hd[:, :-1]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    ds_mid = 0.5 * (ds[:, 1:] + ds[:, :-1])
    return {"speed": ds / DT, "valid": valid, "pair_valid": valid[:, 1:] & valid[:, :-1], "curvature": dh / (ds_mid + EPS),
            "yaw_rate_deg": np.rad2deg(dh / DT), "v_mid": ds_mid / DT}


def terms(GX, GG, key):
    """-> (t [N, K] per-step terms whose row-sum is the window row, row [N] with NaN where the mask is empty)."""
    m = GX["pair_valid"] & GG["pair_valid"]
    e = np.abs(np.where(m, GX[key] - GG[key], 0.0))
    n = m.sum(1)
    t = np.where(n[:, None] > 0, e / np.maximum(n, 1)[:, None], 0.0)
    row = np.where(n > 0, e.sum(1) / np.maximum(n, 1), np.nan)
    return t, row


def partition(X, P, G, key="curvature"):
    """Exact partition of E (X - P10 on the row `key`) into DIP / HIGHK / REST by X's own path (card, Part 1)."""
    GX, GP, GG = geom(X), geom(P), geom(G)
    tX, rX = terms(GX, GG, key)
    tP, rP = terms(GP, GG, key)
    ok = np.isfinite(rX) & np.isfinite(rP)
    N = int(ok.sum())
    c = (tX - tP)[ok]
    E = float(np.mean(rX[ok] - rP[ok]))
    dipwin = ((GX["speed"].min(1) < V_DIP) & (GG["speed"].min(1) >= V_DIP))[ok]
    highk = (~dipwin[:, None]) & GX["pair_valid"][ok] & (np.abs(GX["curvature"][ok]) > KAPPA_MAX)
    dip = np.broadcast_to(dipwin[:, None], c.shape)
    rest = ~(dip | highk)
    EC = {"DIP": float(c[dip].sum() / N), "HIGHK": float(c[highk].sum() / N), "REST": float(c[rest].sum() / N)}
    slow = highk & (GX["v_mid"][ok] < V_SLOW_HIGHK)
    return {"N": N, "E": E, "E_parts": EC, "partition_sum_minus_E": float(sum(EC.values()) - E), "shares": {k: (v / E if E != 0 else float("nan")) for k, v in EC.items()},
            "n_dip_windows": int(dipwin.sum()), "n_highk_steps": int(highk.sum()), "n_highk_steps_masked_in_row": int((highk & (GX["pair_valid"] & GG["pair_valid"])[ok]).sum()),
            "highk_share_from_steps_under_2mps": float(c[slow].sum() / N / EC["HIGHK"]) if EC["HIGHK"] != 0 else float("nan"),
            "_ok": ok, "_rX": rX, "_rP": rP}


def low_pass_fraction(X, P, G, E):
    from scipy.signal import savgol_filter
    Xlo = savgol_filter(np.asarray(X, np.float64), 7, 2, axis=1, mode="interp")
    _, rXl = terms(geom(Xlo), geom(G), "curvature")
    _, rP = terms(geom(P), geom(G), "curvature")
    ok = np.isfinite(rXl) & np.isfinite(rP)
    E_lo = float(np.mean(rXl[ok] - rP[ok]))
    return {"E_lo": E_lo, "f_LF": E_lo / E if E != 0 else float("nan"), "N_lo": int(ok.sum())}


def rho_clip(X, P, G, E):
    GX, GG, GP = geom(X), geom(G), geom(P)
    GX = dict(GX, curvature=np.clip(GX["curvature"], -KAPPA_MAX, KAPPA_MAX))
    _, rX = terms(GX, GG, "curvature")
    _, rP = terms(GP, GG, "curvature")
    ok = np.isfinite(rX) & np.isfinite(rP)
    Ec = float(np.mean(rX[ok] - rP[ok]))
    return {"E_clipped": Ec, "rho_a": (E - Ec) / E if E != 0 else float("nan")}


def decide(s0, s1):
    """The card's rule on DAG-OACC; s* = {"E", "shares", "f_LF"} per seed.  First match wins."""
    if s0["E"] <= 0 or s1["E"] <= 0:
        return {"lever": None, "rule": "REFUSED: E <= 0 at a DAG-OACC seed"}
    sh = lambda c: (s0["shares"][c], s1["shares"][c])  # noqa: E731
    if all(v >= SHARE_BAR for v in sh("HIGHK")):
        return {"lever": "a", "rule": "(a) s_HIGHK >= 0.5 at both seeds"}
    if all(v >= SHARE_BAR for v in sh("DIP")):
        return {"lever": "b", "rule": "(b) s_DIP >= 0.5 at both seeds"}
    if all(v >= SHARE_BAR for v in sh("REST")) and s0["f_LF"] >= SHARE_BAR and s1["f_LF"] >= SHARE_BAR:
        return {"lever": "c", "rule": "(c) s_REST >= 0.5 and f_LF >= 0.5 at both seeds"}
    mh, md = float(np.mean(sh("HIGHK"))), float(np.mean(sh("DIP")))
    if mh <= 0 and md <= 0:
        return {"lever": "c", "rule": "(d) both mean shares <= 0 -> (c)", "mean_s_HIGHK": mh, "mean_s_DIP": md}
    return {"lever": "a" if mh >= md else "b", "rule": "(d) larger of mean s_HIGHK and mean s_DIP ((a) on a tie)", "mean_s_HIGHK": mh, "mean_s_DIP": md}


def public(p):
    return {k: v for k, v in p.items() if not k.startswith("_")}


# ------------------------------------------------------------------------------------------------ heavy layer
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{PK}/raw/GATE_KAPPA.json")
    ap.add_argument("--out-m16", default=OUT_M16)
    a = ap.parse_args(argv)
    want = [ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_FROZEN.md sha256")][0]
    got = sha256_file(f"{PK}/PREREG_FROZEN.md")
    if want != got:
        print(f"REFUSED: PREREG_FROZEN.md {got} != frozen {want}", file=sys.stderr)
        return 2
    man = json.load(open(MANIFEST, encoding="utf-8"))
    res = {"gate": "G-KAPPA", "hypothesis": "H-KAPPA-T1-1", "tier": "T1-kin", "evidence_class": "MEASURED (ours)", "prereg_sha256": got, "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "estimator": "none: an exact point decomposition of the window-mean excess (decides a lever, not a claim)", "inputs": {}, "controls": {}, "runs": {}, "beside": {}}
    # KG1: the row definition against the landed per-window arrays of the BANKED Y (DAG-OACC s0, no S) and P10
    yb = load_dump(man["arms"]["CTRL"]["0"]["t1"]["dump_dir"])
    L = np.load(LANDED_Y, allow_pickle=True)
    labs = [str(x) for x in L["labels"]]
    i_c, i_y = labs.index("curvature MAE"), labs.index("yaw-rate MAE")
    same_w = bool(np.array_equal(L["eid"], yb["eid"]) and np.array_equal(L["g"], yb["g"]))
    _, ry = terms(geom(yb["cl"]), geom(yb["g"]), "curvature")
    _, rp = terms(geom(yb["fl"]), geom(yb["g"]), "curvature")
    _, ryy = terms(geom(yb["cl"]), geom(yb["g"]), "yaw_rate_deg")
    dev = lambda u, v: float(np.nanmax(np.abs(u - v))) if np.array_equal(np.isfinite(u), np.isfinite(v)) else float("inf")  # noqa: E731
    kg1 = {"windows_equal": same_w, "curv_Y_max_abs_dev": dev(ry, L[f"a_{i_c}"]), "curv_P10_max_abs_dev": dev(rp, L[f"b_{i_c}"]), "yaw_Y_max_abs_dev": dev(ryy, L[f"a_{i_y}"])}
    kg1["PASS"] = bool(same_w and kg1["curv_Y_max_abs_dev"] <= 1e-9 and kg1["curv_P10_max_abs_dev"] <= 1e-9)
    res["controls"]["KG1_row_definition_vs_landed_arrays"] = kg1
    res["inputs"]["landed_arrays_sha256"] = sha256_file(LANDED_Y)
    res["inputs"]["manifest_sha256"] = sha256_file(MANIFEST)
    kg2, kg3 = {}, {}
    for k, (tag, arm, seed) in RUNS.items():
        d = load_dump(f"{a.out_m16}/{tag}/dump_{tag}")
        res["inputs"][tag] = json.load(open(f"{a.out_m16}/done_{tag}.txt", encoding="utf-8"))
        p = partition(d["cl"], d["fl"], d["g"])
        lp = low_pass_fraction(d["cl"], d["fl"], d["g"], p["E"])
        rc = rho_clip(d["cl"], d["fl"], d["g"], p["E"])
        py = partition(d["cl"], d["fl"], d["g"], key="yaw_rate_deg")
        band = (d["v0"] >= BAND[0]) & (d["v0"] < BAND[1])
        ok = p["_ok"]
        dl = (p["_rX"] - p["_rP"])[ok]
        res["runs"][k] = {"tag": tag, "arm": arm, "seed": seed, **public(p), **lp, **rc, "band_3_8_part_of_E": float(dl[band[ok]].sum() / p["N"]),
                          "n_band_windows": int(band[ok].sum()), "yaw_rate_partition": public(py)}
        kg2[k] = abs(p["partition_sum_minus_E"]) <= 1e-12
        z = partition(d["cl"], d["cl"], d["g"])
        kg3[k] = bool(z["E"] == 0.0 and all(v == 0.0 for v in z["E_parts"].values()))
        print(f"{k}: E={p['E']:+.5f} shares DIP {p['shares']['DIP']:+.3f} HIGHK {p['shares']['HIGHK']:+.3f} REST {p['shares']['REST']:+.3f} f_LF {lp['f_LF']:+.3f} rho_a {rc['rho_a']:+.3f} "
              f"dipwin {p['n_dip_windows']} highk {p['n_highk_steps']}", flush=True)
    res["controls"]["KG2_partition_sums_to_E"] = {"runs": kg2, "PASS": all(kg2.values())}
    res["controls"]["KG3_self_is_zero"] = {"runs": kg3, "PASS": all(kg3.values())}
    # beside: the banked no-S Y of DAG-OACC (how S changed the composition)
    for s in (0, 1):
        d = load_dump(man["arms"]["CTRL"][str(s)]["t1"]["dump_dir"])
        p = partition(d["cl"], d["fl"], d["g"])
        res["beside"][f"banked_Y_noS_CTRL_s{s}"] = {**public(p), **low_pass_fraction(d["cl"], d["fl"], d["g"], p["E"])}
    ctl_ok = all(v["PASS"] for v in res["controls"].values())
    r0, r1 = res["runs"]["CTRL_s0"], res["runs"]["CTRL_s1"]
    res["decision"] = decide(r0, r1) if ctl_ok else {"lever": None, "rule": "REFUSED: a gate known answer failed"}
    res["decision"]["controls_pass"] = ctl_ok
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("controls", {k: v["PASS"] for k, v in res["controls"].items()}, "KG1", kg1)
    print("DECISION", res["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
