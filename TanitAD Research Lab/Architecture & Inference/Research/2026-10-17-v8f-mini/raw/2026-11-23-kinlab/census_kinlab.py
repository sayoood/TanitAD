"""KINLAB census (CPU, one python process): the own-versus-human kinematic gap at NOW on one shard of the RH bank s0.

Reads shard members ci, r, reason, tick, j, kin3 and the LAST row of p10f (in row chunks: low RAM), the human's native 10 Hz pose tracks, and computes with the trainer's own spelling
(``v8f_mini_data.yaw_kin_from_track``, tree at df250a00) the human kin3 = (v, a, kappa) and the measured yaw rate omega at every cache step of every bank clip.
  own   : kin3 of the record (the own T1-kin state at clock row j)           [v m/s, a m/s^2, kappa 1/m]; own yaw rate = kappa * max(v, 2) (exact unless |kappa| is at the 0.3 clamp)
  human : the same quantities of the LOGGED track at r' + k, k = 0..10 (0 .. 2 s, 0.2 s steps); r' = the matched row (ok records only)
  yaw   : primary = native-track one-frame yaw rate (the SAME definition as the own); secondary = the 0.2 s cache-grid polyline's heading change (wrap(dyaw)/0.2, backward)
  control: |v_own - u_h[1]|, u_h[1] = chord speed of the first 0.2 s step of the human polyline from r' (v_own = p10f[:, -1, 0]); must reproduce REAL_SHARD_CENSUS_2.json (p50 0.7725, p95 6.2210, n 372,791).
  scale : human vs human, x[r' + d] against x[r'] (d = 1 step = the next logged window; and pooled over d <= 5 / <= 10).
usage: python census_kinlab.py <shard.npz> <out.json> <tracks.pt> [<tracks2.pt> ...]
"""
import hashlib
import json
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, "F:/TanitAD-scratch/kinlab/tree/stack")
sys.path.insert(0, "F:/TanitAD-scratch/kinlab/tree/taniteval")
import torch  # noqa: E402
import tanitad  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402

assert D.__file__.replace("\\", "/").startswith("F:/TanitAD-scratch/kinlab/tree/"), D.__file__
RAD2DEG = 180.0 / np.pi
DT = 0.2
KMAX = 10
YAW_T = (1.0, 2.0, 5.0)       # deg/s
ACC_T = (0.5, 1.0, 2.0)       # m/s^2
PCT = (50, 90, 95, 99)


def read_member(zf, name, last_row_only=False, chunk=65536):
    with zf.open(name + ".npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(fh)
        assert not fortran
        n = shape[0]
        rest = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if not last_row_only:
            return np.frombuffer(fh.read(n * rest * dtype.itemsize), dtype).reshape(shape).copy()
        out = np.empty((n,) + tuple(shape[2:]), dtype)
        per = rest * dtype.itemsize
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            blk = np.frombuffer(fh.read((b - a) * per), dtype).reshape((b - a,) + tuple(shape[1:]))
            out[a:b] = blk[:, -1]
        return out


def stats(x, thr=None):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    out = {"n": int(len(x))}
    if len(x):
        for p in PCT:
            out[f"p{p}"] = float(np.percentile(x, p))
        out["mean"] = float(x.mean())
        if thr:
            out["share_gt"] = {str(t): float((x > t).mean()) for t in thr}
    return out


def main():
    shard, outp, tpaths = sys.argv[1], sys.argv[2], sys.argv[3:]
    t0 = time.time()
    h = hashlib.sha256()
    with open(shard, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    zf = zipfile.ZipFile(shard)
    hdr = json.loads(bytes(read_member(zf, "header_json")).decode("utf-8"))
    ci, rr, reason, tick = (read_member(zf, k) for k in ("ci", "r", "reason", "tick"))
    kin_own = read_member(zf, "kin3").astype(np.float64)
    p10_last = read_member(zf, "p10f", last_row_only=True)          # [N, 8] float32
    print("loaded", len(ci), "records", round(time.time() - t0, 1), "s", flush=True)
    names = {int(k): v for k, v in hdr["clips"].items()}
    tracks = {}
    for p in tpaths:
        tracks.update({c: t.double().numpy() for c, t in torch.load(p, map_location="cpu", weights_only=True)["tracks"].items()})
    ok = reason == 0
    n_ok = int(ok.sum())
    need = sorted({names[int(c)] for c in np.unique(ci[ok])})
    missing = [c for c in need if c not in tracks]
    have = np.array([names[int(c)] in tracks for c in ci], bool)
    sel = ok & have
    print("ok", n_ok, "with track", int(sel.sum()), "clips needed", len(need), "missing", len(missing), flush=True)
    # per-clip human series, concatenated
    off, T_of, cl_idx = {}, {}, {}
    K, OM, GY, GXY = [], [], [], []
    o = 0
    for c in need:
        if c in missing:
            continue
        tr = tracks[c]
        kin, omega = D.yaw_kin_from_track(tr)                          # [T, 3] (v, a, kappa), omega [T] rad/s
        g = tr[0::2][:, :3]
        T = len(kin)
        gy = np.full(T, np.nan)
        dy = g[1:, 2] - g[:-1, 2]
        gy[1:] = ((dy + np.pi) % (2 * np.pi) - np.pi) / DT              # cache-grid backward heading change / 0.2 s
        K.append(kin); OM.append(np.where(np.isfinite(kin[:, 0]), omega, np.nan)); GY.append(gy); GXY.append(g[:, :2])
        off[c], T_of[c] = o, T
        o += T
    K, OM, GY, GXY = np.concatenate(K), np.concatenate(OM), np.concatenate(GY), np.concatenate(GXY)
    # records
    I = np.flatnonzero(sel)
    cn = [names[int(c)] for c in ci[I]]
    base = np.array([off[c] for c in cn], np.int64)
    Tn = np.array([T_of[c] for c in cn], np.int64)
    r = rr[I].astype(np.int64)
    v_own, a_own, kap_own = kin_own[I, 0], kin_own[I, 1], kin_own[I, 2]
    om_own = kap_own * np.maximum(v_own, 2.0) * RAD2DEG
    clamped = np.abs(np.abs(kap_own) - 0.3) < 1e-9
    tk = tick[I]
    res = {"inputs": {"shard_sha256": h.hexdigest(), "tracks": tpaths, "n_records_shard": int(len(ci)), "n_ok": n_ok, "n_ok_with_track": int(len(I)),
                      "clips_needed": len(need), "clips_missing_track": len(missing), "ticks_ok": {str(t): int((tk == t).sum()) for t in (1, 2)},
                      "header_keys": {k: hdr[k] for k in hdr if k in ("readout", "acc_gain", "path_exec", "kappa_max", "generator", "init_ckpt_sha256", "schema", "split")}},
           "units": {"yaw": "deg/s", "acc": "m/s^2", "speed": "m/s", "kappa": "1/m"},
           "own_kappa_at_clamp_share": float(clamped.mean())}
    # control: |v_own - u_h[1]|, v_own = p10f[:, -1, 0]
    vp = p10_last[I, 0].astype(np.float64)
    g0 = GXY[base + r]
    g1 = GXY[base + r + 1]
    uh1 = np.hypot(*(g1 - g0).T) / DT
    ctrl = np.abs(vp - uh1)
    res["control_speed"] = {"abs_v_own_minus_u_h1": stats(ctrl), "ref_REAL_SHARD_CENSUS_2": {"p50": 0.7725374488841155, "p95": 6.22098895312836, "n": 372791},
                            "max_abs_p10f_v_minus_kin3_v": float(np.nanmax(np.abs(vp - v_own)))}
    # gaps own(NOW) vs human(r'+k)
    quant = {
        "yaw_rate_native": (om_own, lambda ix: OM[ix] * RAD2DEG, YAW_T),
        "yaw_rate_gridpoly": (om_own, lambda ix: GY[ix] * RAD2DEG, YAW_T),
        "kappa_1pm": (kap_own, lambda ix: K[ix, 2], (0.005, 0.01, 0.02)),
        "accel": (a_own, lambda ix: K[ix, 1], ACC_T),
        "speed": (v_own, lambda ix: K[ix, 0], (0.5, 1.0, 2.0)),
    }
    own_vs_h, hh = {}, {}
    for qn, (own, fh_, thr) in quant.items():
        G = np.full((len(I), KMAX + 1), np.nan)
        for k in range(KMAX + 1):
            G[:, k] = np.abs(own - fh_(base + r + k))
        H = np.full((len(I), KMAX + 1), np.nan)
        x0 = fh_(base + r)
        for k in range(1, KMAX + 1):
            H[:, k] = np.abs(fh_(base + r + k) - x0)
        pools = {"NOW_k0": (0, 0), "first_1s_k0to5": (0, 5), "first_2s_k0to10": (0, 10)}
        blk = {"by_k": {str(k): stats(G[:, k], thr) for k in range(KMAX + 1)},
               "by_bank_tick_at_NOW": {str(t): stats(G[tk == t, 0], thr) for t in (1, 2)}}
        for pn, (a, b) in pools.items():
            sub = G[:, a:b + 1]
            blk[pn] = stats(sub.ravel(), thr)
            fin = np.isfinite(sub).all(1)
            blk[pn]["share_records_max_over_window_gt"] = {str(t): float((np.nanmax(sub[fin], 1) > t).mean()) for t in thr}
        own_vs_h[qn] = blk
        hb = {"consecutive_window_d1": stats(H[:, 1], thr)}
        for pn, (a, b) in {"human_vs_human_d1to5": (1, 5), "human_vs_human_d1to10": (1, 10)}.items():
            hb[pn] = stats(H[:, a:b + 1].ravel(), thr)
        hh[qn] = hb
    res["own_vs_human"], res["human_vs_human_scale"] = own_vs_h, hh
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps({q: {"NOW": own_vs_h[q]["NOW_k0"], "scale_d1": hh[q]["consecutive_window_d1"]} for q in own_vs_h}, indent=1)[:3500])
    print(json.dumps(res["control_speed"], indent=1))


if __name__ == "__main__":
    main()
