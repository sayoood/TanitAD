"""PREREG_V8F_IMAG1 labels (labels only; no latent statistic is computed here). Tier T0.
Per dataset (eval141 = WMACT eval windows, train = WMACT TRAIN windows) the frame bank is LATSEM2's rule with EVERY clip kept (no frame is read: 141 episodes on both),
and per bank frame (episode e, cache step g; raw frame 2g) the ten targets of the card:
  speed [m/s]          native_tracks.pt column 3 at raw row 2g (the loader's v0 = poses[2t, 3])
  accel [m/s^2]        logged action channel 0 of the v8F loader (A_raw[..., 0], the WM's longitudinal action)
  yaw_rate [rad/s]     logged action channel 1 of the loader in kin_mode yaw (measured, one-frame backward wrapped difference)
  curvature [1/m]      the loader's kappa0 = clip(yaw_rate / max(speed, 2), -0.3, 0.3)
  lead_present, lead_gap, lead_closing, agents30, lon7   LATSEM2 / LATSEM2B arrays, unchanged (looked up by (episode, g))
  ttc [s]              lead frames with a finite lead_closing: 10 if closing <= 0.1 m/s else min(10, lead_gap / lead_closing)
Run from theory/imag1 with WMACT_CODE_ROOT set: python labels_imag1.py"""
import hashlib, json, os, sys
sys.dont_write_bytecode = True
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wmact")); sys.path.insert(0, os.path.join(HERE, "..", "latsem2b")); sys.path.insert(0, os.path.join(HERE, "..", "latsem2"))
import wmact_lib as W  # noqa: E402
import labels_latsem2b as LBb  # noqa: E402

TRACKS = LBb.TRACKS
TARGETS = ("speed", "accel", "lead_gap", "lead_closing", "ttc", "yaw_rate", "curvature", "lead_present", "lon7", "agents30")
FAMILY = {"speed": "LON", "accel": "LON", "lead_gap": "LON", "lead_closing": "LON", "ttc": "LON", "yaw_rate": "LAT", "curvature": "LAT",
          "lead_present": "TAC", "lon7": "TAC", "agents30": "SCENE"}
KIND = {t: "cont" for t in TARGETS}; KIND.update({"lead_present": "bin", "lon7": "cls"})
LATENTS = {"eval141": os.path.join(HERE, "..", "wmact", "latents.pt"), "train": os.path.join(HERE, "..", "wmact", "latents_train.pt")}
OLD = {"eval141": os.path.join(HERE, "..", "latsem2", "labels_latsem2.npz"), "train": os.path.join(HERE, "..", "latsem2b", "labels_latsem2b_train.npz")}
TTC_CAP, TTC_CLOSE_MIN = 10.0, 0.1


def build(ds):
    B = LBb.build_bank_train(LATENTS[ds])
    beid, bg = B["beid"], B["bg"]
    old = np.load(OLD[ds])
    key = {(str(e), int(g)): i for i, (e, g) in enumerate(zip(old["beid"], old["bg"]))}
    idx = np.array([key.get((str(e), int(g)), -1) for e, g in zip(beid, bg)])
    Y = {}
    for q in ("lead_gap", "lead_closing", "lead_present", "lon7", "agents30"):
        Y[q] = np.where(idx >= 0, old[q][np.maximum(idx, 0)], np.nan)
    TR = {k: np.asarray(v, np.float64) for k, v in torch.load(TRACKS, weights_only=False)["tracks"].items()}
    sp = np.full(len(beid), np.nan)
    for b, (e, g) in enumerate(zip(beid, bg)):
        tr = TR[str(e)]
        if 2 * int(g) < len(tr):
            sp[b] = tr[2 * int(g), 3]
    A = B["A_raw"].numpy().astype(np.float64)
    acc, yaw = A[B["bw"], B["bf"], 0], A[B["bw"], B["bf"], 1]
    copy_diff = float(np.abs(A[B["occ_w"], B["occ_f"]] - A[B["bw"][B["occ_b"]], B["bf"][B["occ_b"]]]).max())
    Y["speed"], Y["accel"], Y["yaw_rate"] = sp, acc, yaw
    Y["curvature"] = np.where(np.isfinite(sp), np.clip(yaw / np.maximum(sp, 2.0), -0.3, 0.3), np.nan)
    gap, cl = Y["lead_gap"], Y["lead_closing"]
    ttc = np.where(np.isfinite(gap) & np.isfinite(cl), np.where(cl <= TTC_CLOSE_MIN, TTC_CAP, np.minimum(TTC_CAP, gap / np.maximum(cl, 1e-9))), np.nan)
    Y["ttc"] = ttc
    return B, Y, dict(copy_maxdiff_actions=copy_diff, n_in_old_array=int((idx >= 0).sum()), old_sha256=hashlib.sha256(open(OLD[ds], "rb").read()).hexdigest(),
                      speed_equal_old=(float(np.nanmax(np.abs(sp - old["speed"]))) if ds == "train" and "speed" in old.files else None))


if __name__ == "__main__":
    for ds in ("eval141", "train"):
        B, Y, audit = build(ds)
        fold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in B["beid"]])
        sel = B["wkeep"][B["ok"][B["wkeep"]].all(1)]
        rep = {"dataset": ds, "n_bank_frames": int(len(B["beid"])), "n_episodes": int(len(set(B["beid"]))), "n_windows_total": int(len(B["eid"])), "n_windows_all16_ok": int(len(sel)),
               "n_episodes_windows": int(len(set(B["eid"][sel]))), "audit": audit, "n": {}, "n_by_fold": {}, "n_episodes_by_target": {}, "summary": {}}
        for q in TARGETS:
            f = np.isfinite(Y[q]); rep["n"][q] = int(f.sum()); rep["n_by_fold"][q] = [int((f & (fold == k)).sum()) for k in (0, 1)]
            rep["n_episodes_by_target"][q] = int(len(set(B["beid"][f])))
            rep["summary"][q] = ({str(int(c)): int((Y[q][f] == c).sum()) for c in np.unique(Y[q][f])} if KIND[q] != "cont" else
                                 {k: float(v) for k, v in zip(("min", "p5", "median", "p95", "max", "mean", "sd"), (Y[q][f].min(), *np.percentile(Y[q][f], [5, 50, 95]), Y[q][f].max(), Y[q][f].mean(), Y[q][f].std()))})
        np.savez(os.path.join(HERE, f"labels_imag1_{ds}.npz"), beid=B["beid"].astype("U36"), bg=B["bg"], **{q: Y[q] for q in TARGETS})
        json.dump(rep, open(os.path.join(HERE, f"labels_imag1_{ds}.report.json"), "w", encoding="utf-8"), indent=1)
        print(ds, json.dumps({k: rep[k] for k in ("n_bank_frames", "n_episodes", "n_windows_all16_ok", "n_episodes_windows", "audit", "n", "n_episodes_by_target")}), flush=True)
