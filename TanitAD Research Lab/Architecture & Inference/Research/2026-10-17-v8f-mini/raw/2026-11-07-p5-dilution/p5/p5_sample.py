"""p5_sample: P5's window set (THEORY_CRAWL_AMEND_3.md section 3): TRAIN windows of the base's clips (the banked 306,566-window list, sha e144adcd...), v0 in [3, 8) m/s,
a seeded sample of 20,000 from the windows whose +-0.8 s Savitzky-Golay support is inside the clip (so the three SG widths run on IDENTICAL windows).
rule: pool = table rows with a finite a at +-0.3/0.5/0.8 s;  pick = np.sort(np.random.default_rng(0).choice(pool, 20000, replace=False)).  No eval window is read.
Output: raw/p5_window_ids.txt (LF, 'clip:t' per line, table order), raw/p5_sample_rows.npy, raw/P5_WINDOWS.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import p5_lib as L

N_SUB, SEED = 20000, 0


def main():
    L.assert_card_frozen()
    tab = np.load(L.HERE + "/raw/window_table_band.npz")
    clips = tab["clips"]
    ok = np.isfinite(tab["a_pm0.3s"]) & np.isfinite(tab["a_pm0.5s"]) & np.isfinite(tab["a_pm0.8s"]) & np.isfinite(tab["a0_model"]) & np.isfinite(tab["kappa0"])
    pool = np.flatnonzero(ok)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(pool, size=N_SUB, replace=False))
    lines = [f"{clips[tab['clip_idx'][i]]}:{int(tab['t'][i])}" for i in pick]
    txt = ("\n".join(lines) + "\n").encode("utf-8")
    open(L.HERE + "/raw/p5_window_ids.txt", "wb").write(txt)
    np.save(L.HERE + "/raw/p5_sample_rows.npy", pick)
    ep = np.unique(tab["clip_idx"][pick])
    rep = {"seed": SEED, "rule": "pool = in-band (3 <= v0 < 8) rows of window_table_band.npz with a finite SG a at +-0.3/0.5/0.8 s; pick = np.sort(np.random.default_rng(0).choice(pool, 20000, replace=False))",
           "n_pool": int(len(pool)), "n_subset": int(len(pick)), "n_subset_episodes": int(len(ep)), "min_episodes_required": 500,
           "list_sha256": L.sha256_bytes(txt), "list_file": "raw/p5_window_ids.txt", "window_list_sha256": L.WINDOW_LIST_SHA,
           "subset_v0_mean": float(tab["v0"][pick].mean()), "subset_v0_min": float(tab["v0"][pick].min()), "subset_v0_max": float(tab["v0"][pick].max()),
           "windows_per_episode_median": float(np.median(np.bincount(tab["clip_idx"][pick])[ep])), "eval141_read": False}
    json.dump(rep, open(L.HERE + "/raw/P5_WINDOWS.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
