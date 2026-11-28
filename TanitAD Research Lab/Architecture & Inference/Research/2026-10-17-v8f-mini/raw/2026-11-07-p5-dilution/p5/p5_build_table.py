"""p5_build_table: the in-band (v0 in [3, 8) m/s) TRAIN windows of the base's clips with the model's a0, the kappa0 it reads, the independent a0, and the label-side accelerations.
The canonical train window list (306,566 windows, 4,512 clips, --exclude 540:600, window_start 3) is rebuilt as wsp_edit/build_train_ids.py does and REFUSED unless its sha256 is the banked e144adcd....
Reads: the manifest, native_tracks.pt (no cache clip, no eval window, no checkpoint). CPU only.   Output: raw/window_table_band.npz + raw/WINDOW_TABLE.json
"""
import os
import sys
import json
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import p5_lib as L


def main():
    t0 = time.time()
    L.assert_card_frozen()
    D = L.setup_tree()
    ents = {}
    for ln in open(L.TRAIN_MANIFEST, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            ents[r["clip"]] = int(r["n_frames"])
    clips_all = list(ents)
    ex = set(sorted(clips_all, key=D.sha12)[540:600])
    clips = D.select_train_clips([c for c in clips_all if c not in ex], 1.0)
    first = D.effective_window_start(4, 3)
    h = hashlib.sha256()
    n_total = 0
    for c in clips:
        for t in range(first, ents[c] - D.HORIZON):
            h.update((("\n" if n_total else "") + f"{c}:{t}").encode("utf-8"))
            n_total += 1
    digest = h.hexdigest()
    if digest != L.WINDOW_LIST_SHA:
        raise SystemExit(f"REFUSED: the rebuilt train window list is not the banked one ({digest})")
    print("window list ok", n_total, "clips", len(clips), "first", first, f"{time.time()-t0:.0f}s", flush=True)
    tracks = D.load_native_tracks(L.TRACKS)
    cols = {k: [] for k in ("clip_idx", "t", "v0", "a0_model", "kappa0", "a0_indep", "a0_lag2", "a0_fwd", "a_pm0.3s", "a_pm0.5s", "a_pm0.8s")}
    n_all = 0
    for ci, c in enumerate(clips):
        tr = tracks[c]
        assert len(tr[0::2]) == ents[c], (c, len(tr), ents[c])
        kin3, _ = D.yaw_kin_from_track(tr)                 # THE MODEL'S (v0, a0, kappa0): ClipStore.get(kin_mode='yaw') calls exactly this
        ts = np.arange(first, ents[c] - D.HORIZON)
        n_all += len(ts)
        v0 = kin3[ts, 0]
        m = (v0 >= L.BAND[0]) & (v0 < L.BAND[1])
        ts = ts[m]
        if not len(ts):
            continue
        k0 = 2 * ts
        v = tr[:, 3]
        cols["clip_idx"].append(np.full(len(ts), ci, np.int32))
        cols["t"].append(ts.astype(np.int32))
        cols["v0"].append(kin3[ts, 0])
        cols["a0_model"].append(kin3[ts, 1])
        cols["kappa0"].append(kin3[ts, 2])
        cols["a0_indep"].append(L.a0_backward(v, k0, 1))
        cols["a0_lag2"].append(L.a0_backward(v, k0, 2))
        cols["a0_fwd"].append(L.a0_forward(v, k0))
        for nm, mm in L.SG_M.items():
            cols["a_" + nm].append(L.sg_accel(v, k0, mm))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(L.HERE + "/raw/window_table_band.npz", clips=np.array(clips), **out)
    n = len(out["t"])
    fin = np.isfinite(out["a_pm0.3s"]) & np.isfinite(out["a_pm0.5s"]) & np.isfinite(out["a_pm0.8s"]) & np.isfinite(out["a0_model"])
    rep = {"n_windows_total": n_all, "window_list_sha256": digest, "n_clips": len(clips), "n_excluded_clips": len(ex), "n_band_3_8": int(n), "n_band_clips": int(len(np.unique(out["clip_idx"]))),
           "n_finite_all_three_sg_widths": int(fin.sum()), "n_dropped_for_sg_support": int(n - fin.sum()),
           "dropped_by_t": {str(int(t)): int(((~fin) & (out["t"] == t)).sum()) for t in np.unique(out["t"][~fin])},
           "manifest": L.TRAIN_MANIFEST, "manifest_sha256": L.sha256_file(L.TRAIN_MANIFEST), "tracks": L.TRACKS, "tracks_sha256": L.sha256_file(L.TRACKS),
           "tip": L.TIP, "wall_s": time.time() - t0}
    json.dump(rep, open(L.HERE + "/raw/WINDOW_TABLE.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
