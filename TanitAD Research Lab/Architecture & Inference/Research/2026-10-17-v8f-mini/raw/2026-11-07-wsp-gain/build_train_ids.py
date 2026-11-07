"""build_train_ids: the E-GAIN fit set (PREREG_WSP_GAIN.md section 2): a fixed random subset of 20,000 TRAIN windows with v0 in [3, 8) m/s from the 4,512 trained clips, seed 0.
The canonical train window list is rebuilt exactly as dagland/crawl_diag/crawl_diag.py::train_inputs does and REFUSED unless its sha256 is the banked e144adcd... (306,566 windows).
No eval window is read. CPU only.  Output: raw/train_window_ids.txt (LF, one 'clip:t' per line, flat order) + raw/TRAIN_IDS.json.
"""
import os
import sys
import json
import hashlib
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import wsp_common as W

N_SUB = 20000
SEED = 0


def main():
    t0 = time.time()
    T = W.load_tool()
    D = T.D
    ents = {}
    for ln in open(W.TRAIN_MANIFEST, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            ents[r["clip"]] = int(r["n_frames"])
    clips_all = list(ents)
    ex = set(sorted(clips_all, key=D.sha12)[540:600])                    # --exclude 540:600 (argv of the C0 trainer)
    clips = D.select_train_clips([c for c in clips_all if c not in ex], 1.0)   # summary.json: data_frac 1.0
    first = D.effective_window_start(4, 3)                               # summary.json levers: hist 4, window_start 3
    wins = [(c, t) for c in clips for t in range(first, ents[c] - D.HORIZON)]
    digest = hashlib.sha256("\n".join(f"{c}:{t}" for c, t in wins).encode("utf-8")).hexdigest()
    if digest != W.WINDOW_LIST_SHA:
        raise SystemExit(f"REFUSED: the rebuilt train window list is not the banked one ({digest})")
    print("window list ok", len(wins), "clips", len(clips), "first", first, f"{time.time()-t0:.0f}s", flush=True)
    tracks = D.load_native_tracks(W.TRACKS)
    V0 = []
    for c in clips:
        tr = tracks[c]
        pg = tr[0::2]
        assert len(pg) == ents[c], (c, len(pg), ents[c])
        kin3, _ = D.yaw_kin_from_track(tr)
        gt, gok = D.gt_future(pg)
        ts = np.arange(first, len(pg) - D.HORIZON)
        assert gok[ts].all()
        V0.append(kin3[ts, 0])
    v0 = np.concatenate(V0)
    assert len(v0) == len(wins)
    pool = np.flatnonzero((v0 >= W.BAND[0]) & (v0 < W.BAND[1]))
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(pool, size=N_SUB, replace=False))
    lines = [f"{wins[i][0]}:{wins[i][1]}" for i in pick]
    txt = ("\n".join(lines) + "\n").encode("utf-8")
    os.makedirs(W.HERE + "/raw", exist_ok=True)
    open(W.HERE + "/raw/train_window_ids.txt", "wb").write(txt)
    rep = {"n_windows_total": len(wins), "window_list_sha256": digest, "n_clips": len(clips), "n_excluded": len(ex), "n_pool_3_8": int(len(pool)), "n_subset": int(len(pick)),
           "n_subset_clips": int(len({wins[i][0] for i in pick})), "subset_v0_mean": float(v0[pick].mean()), "subset_v0_min": float(v0[pick].min()), "subset_v0_max": float(v0[pick].max()),
           "seed": SEED, "rule": "pool = flat indices (window-list order) with 3.0 <= v0 < 8.0; pick = np.sort(np.random.default_rng(0).choice(pool, 20000, replace=False))",
           "train_window_ids_txt_sha256": hashlib.sha256(txt).hexdigest(), "manifest": W.TRAIN_MANIFEST, "manifest_sha256": W.sha256_file(W.TRAIN_MANIFEST), "wall_s": time.time() - t0}
    json.dump(rep, open(W.HERE + "/raw/TRAIN_IDS.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
