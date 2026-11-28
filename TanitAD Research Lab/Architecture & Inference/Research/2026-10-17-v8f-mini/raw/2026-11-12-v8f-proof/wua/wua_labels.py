"""wua_labels.py -- card ADD155 section 1 targets on eval-141, per (clip, cached frame g). Labels only: no model is loaded.

  python -I wua_labels.py --out C:/Users/Admin/wua_work/labels/WUA_LABELS.npz
Frame convention: cached frame g = 0.2 s steps; join row = off[clip] + g; agent join and native tracks are raw 10 Hz frames, raw frame = 2 g (as labels_latsem2.py).
"""
import argparse, gzip, hashlib, json, lzma, os, sys
sys.dont_write_bytecode = True
import numpy as np
import torch

CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache"
JOIN = "F:/TanitAD-scratch/v8proof/rhacc10k/pins/join_eval141.npz"
TRACKS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
AGENTS = "E:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
TL = "C:/Users/Admin/wua_work/labels/s2_labels_v7.2_eval.jsonl.gz"
FOV_HALF = np.deg2rad(60.0)
R_MAX = 50.0
CLOSE_CAP = 20.0


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    man = [json.loads(l) for l in open(f"{CACHE}/eval/manifest.jsonl")]
    clips = sorted(r["clip"] for r in man)
    nfr = {r["clip"]: int(r["n_frames"]) for r in man}
    J = np.load(JOIN)
    jc = {str(c): i for i, c in enumerate(J["clip"])}
    TR = {k: np.asarray(v, np.float64) for k, v in torch.load(TRACKS, weights_only=False)["tracks"].items()}
    AG = {}
    with lzma.open(AGENTS, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            AG[(r["clip_id"], int(r["frame"]))] = r["agents"]
    tl = {}
    for line in gzip.open(TL, "rt"):
        r = json.loads(line)
        g = ((r.get("g_tac") or {}).get("goals") or {})
        if "TRAFFIC_LIGHT_REACT_RED" in g:
            tl[r["clip_id"]] = (1.0, float(r.get("t0_s", np.nan)))
        elif "TRAFFIC_LIGHT_REACT_GREEN" in g:
            tl[r["clip_id"]] = (0.0, float(r.get("t0_s", np.nan)))
    names = ["lead_present", "lead_gap", "lead_closing", "lead_rel_acc", "gap_change_2s", "curv_ahead", "lon7", "lat_cls", "tl_red_vs_green",
             "ag1_x", "ag1_y", "ag2_x", "ag2_y", "ag3_x", "ag3_y", "ag1_vx", "ag1_vy", "ag1_inview_n"]
    rows = []
    audit = {"clips": len(clips), "no_join": 0, "agent_line_missing": 0, "tl_clips": 0, "track_short": 0}
    CI, GG = [], []
    for j, c in enumerate(clips):
        T = nfr[c]
        Y = {n: np.full(T, np.nan) for n in names}
        if c not in jc:
            audit["no_join"] += 1
        else:
            ci = jc[c]; off = int(J["off"][ci]); n = int(J["n"][ci])
            for g in range(min(T, n)):
                row = off + g
                if J["lead_valid"][row] == 1 and np.isfinite(J["lead_gap"][row]):
                    Y["lead_gap"][g] = float(J["lead_gap"][row])
                    Y["lead_present"][g] = 1.0
                else:
                    Y["lead_present"][g] = 0.0
                if J["has"][row] and int(J["lon_cls"][row]) >= 0:
                    Y["lon7"][g] = float(J["lon_cls"][row])
                if J["has"][row] and int(J["lat_cls"][row]) >= 0:
                    Y["lat_cls"][g] = float(J["lat_cls"][row])
        gap = Y["lead_gap"]
        for g in range(T):
            if g >= 2 and np.isfinite(gap[g]) and np.isfinite(gap[g - 2]):
                cl = (gap[g - 2] - gap[g]) / 0.4
                Y["lead_closing"][g] = cl if abs(cl) <= CLOSE_CAP else np.nan
            if g + 10 < T and np.isfinite(gap[g]) and np.isfinite(gap[g + 10]):
                Y["gap_change_2s"][g] = gap[g + 10] - gap[g]
        clo = Y["lead_closing"]
        for g in range(5, T):
            if np.isfinite(clo[g]) and np.isfinite(clo[g - 5]):
                Y["lead_rel_acc"][g] = (clo[g] - clo[g - 5]) / 1.0
        tr = TR.get(c)
        if tr is not None:
            for g in range(T):
                r0, r1 = 2 * g, 2 * g + 40
                if r1 < len(tr) and np.all(np.isfinite(tr[r0:r1 + 1, :3])):
                    arc = float(np.hypot(np.diff(tr[r0:r1 + 1, 0]), np.diff(tr[r0:r1 + 1, 1])).sum())
                    if arc >= 5.0:
                        Y["curv_ahead"][g] = wrap(tr[r1, 2] - tr[r0, 2]) / arc
        else:
            audit["track_short"] += 1
        for g in range(T):
            ln = AG.get((c, 2 * g))
            if ln is None:
                audit["agent_line_missing"] += 1
                continue
            P = [(x["cx"], x["cy"], x["track_id"]) for x in ln]
            P = [p for p in P if p[0] > 0.5 and abs(np.arctan2(p[1], p[0])) <= FOV_HALF and np.hypot(p[0], p[1]) <= R_MAX]
            P.sort(key=lambda p: np.hypot(p[0], p[1]))
            Y["ag1_inview_n"][g] = float(len(P))
            for k in range(min(3, len(P))):
                Y[f"ag{k + 1}_x"][g], Y[f"ag{k + 1}_y"][g] = P[k][0], P[k][1]
            if P and g >= 2:
                prev = AG.get((c, 2 * g - 4))
                if prev is not None:
                    pm = {x["track_id"]: (x["cx"], x["cy"]) for x in prev}
                    if P[0][2] in pm:
                        Y["ag1_vx"][g] = (P[0][0] - pm[P[0][2]][0]) / 0.4
                        Y["ag1_vy"][g] = (P[0][1] - pm[P[0][2]][1]) / 0.4
        if c in tl:
            audit["tl_clips"] += 1
            lab, t0 = tl[c]
            for g in range(T):
                if np.isfinite(t0) and abs(0.2 * g - t0) <= 1.0:
                    Y["tl_red_vs_green"][g] = lab
        rows.append(np.stack([Y[n] for n in names], 1))
        CI.append(np.full(T, j, np.int32)); GG.append(np.arange(T, dtype=np.int16))
    L = np.concatenate(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, Y=L, names=np.array(names), clip_idx=np.concatenate(CI), t=np.concatenate(GG), clips=np.array(clips))
    audit["n_per_target"] = {n: int(np.isfinite(L[:, i]).sum()) for i, n in enumerate(names)}
    audit["clips_with_lead"] = int(len(set(np.concatenate(CI)[np.isfinite(L[:, 1])].tolist())))
    audit["sha256"] = sha(a.out)
    json.dump(audit, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
