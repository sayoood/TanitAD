"""How many TRAINING WINDOWS does the 100 % clip coverage actually supervise?

The report quotes "n = 2,400 instead of n = 190". That is a CLIP count.
CE on tac_lat/tac_lon is computed PER WINDOW, and
v7_labels.window_in_band admits a window only when
    |t_now - t0_s| <= (hi - lo)/2      (bands.tactical_s = [2,6] -> +-2.0 s)
so with t0_s = 8.0 the admitted NOW is [6.0, 10.0] s.
"""
import glob, json, os, sys
from pathlib import Path
import gzip
from tanitad.data.v2_dataset import build_v2_providers, stable_episode_id
from tanitad.data import v7_labels as v7l

CACHE = r"C:/Users/Admin/.cache/huggingface/hub/datasets--Sayood--tanitad-physicalai-w120-256x640cyl/snapshots/328b0e50b90603ddd763bab94dec056e244b764a/physicalai-train-e438721ae894-w120-256x640cyl"
n_files = len(glob.glob(os.path.join(CACHE, "*.v2ep.pt")))
print("local parity v2 files:", n_files, " (CONTROL: dir entries =",
      len(os.listdir(CACHE)), ")")
eps = build_v2_providers(CACHE, lru_size=4, verbose=False)
print("providers:", len(eps))

A = Path(r"C:/Users/Admin/tanitad-review-20260906/art")
labels, _ = v7l.load_v7_labels(str(A/"s2_labels_parity-v7geom-1_train.jsonl.gz"),
                               allow_oracle_nav=True)
by_sid = {stable_episode_id(l.clip_id): l for l in labels}

WINDOW, MAXH, DT = 4, 20, 0.1     # refc_v3_train: kw window=cfg.core.window, max_horizon=20
for WINDOW in (4,):
    tot = inb = 0; per = []
    T_list = []
    for e in eps:
        T = int(e.frames.shape[0]); T_list.append(T)
        lab = by_sid.get(int(e.episode_id))
        n_w = max(T - WINDOW - MAXH + 1, 0)
        k = 0
        for t in range(n_w):
            t_now = (t + WINDOW - 1) * DT
            if lab is not None and v7l.window_in_band(lab, t_now):
                k += 1
        tot += n_w; inb += k; per.append((n_w, k))
    print()
    print("window=%d max_horizon=%d dt=%.1f" % (WINDOW, MAXH, DT))
    print("  episode frame counts: min %d max %d" % (min(T_list), max(T_list)))
    print("  windows per episode: %.1f   in-band per episode: %.1f"
          % (tot/len(eps), inb/len(eps)))
    print("  TOTAL windows %d   SUPERVISED (in band) %d   = %.2f %%"
          % (tot, inb, 100*inb/max(tot,1)))
    print("  => %.2f %% of every window carries IGNORE_ID (-100) on tac_lat/tac_lon"
          % (100*(tot-inb)/max(tot,1)))
    # CONTROL: an out-of-band time must be refused, an in-band one admitted
    l0 = labels[0]
    print("  CONTROL window_in_band(t0=8.0) =", v7l.window_in_band(l0, 8.0),
          " (must be True);  at t=0.0 =", v7l.window_in_band(l0, 0.0),
          " (must be False);  bands:", l0.bands["tactical_s"], " t0_s:", l0.t0_s)
