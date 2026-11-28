"""PREREG_V8F_LAUNCH_INTENT dose rule: measure f, the fraction of TRAIN windows (the DAG-OACC recipe's window list) that are GT-launch, among windows with a valid along-track label.

CPU only, no GPU, no model, no feature cache: the launch label needs only the logged pose track (v0 = kin3[t, 0], the GT chord speed over the last 0.2 s of the 6 s horizon).
  * the window list is REBUILT from the train manifest with the trainer's own functions and CHECKED against the banked run: its sha256 must equal `train_window_list_sha256` of
    `v8f-dagoacc/M2b_s0_dagoacc/summary.json` (and s1), else the script exits 2 and writes no f;
  * the mask is computed twice, by `v8f_mini.launch_mask` (the trainer's function, torch) and by an independent numpy spelling written here from the definition; they must agree on every window;
  * the dose is  1 + W = (1 - f) / (2 f)  (arm A: launch samples carry one third of the along-track weight)  and  1 + W = (1 - f) / (5 f)  (arm B: one sixth), rounded to 6 significant digits
    like OACC's w_acc;
  * the realised launch weight share of each arm's seed is SIMULATED with the trainer's own ChunkedSampler (3,000 steps, batch 64) so the K-SHARE band (1/3 +- 0.02) is known to be reachable.

Usage:  python measure_f.py --stack <tree>/stack --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ap = argparse.ArgumentParser()
ap.add_argument("--stack", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--manifest", default="C:/Users/Admin/box2_ship/logs/v8f_dscale/manifest_train.jsonl")
ap.add_argument("--tracks", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt")
ap.add_argument("--banked", nargs=2, default=["E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/summary.json", "E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s1_dagoacc/summary.json"])
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
sys.path[:0] = [a.stack, str(Path(a.stack) / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(a.threads)
from tanitad.data import v8f_dagger as DG  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

t0 = time.time()
DAGOACC_ARGV = ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--out", "x", "--kin-mode", "yaw", "--exclude", "540:600", "--batch", "64", "--steps-mult", "4", "--steps", "3000",
                "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0", "--dagger-p", "0.1", "--w-acc", "49.6969"]
pa = TR.build_parser().parse_args(DAGOACC_ARGV)
cfg = TR.make_config(pa)
first = D.effective_window_start(cfg.hist, cfg.window_start)

ents = {}
for ln in open(a.manifest, encoding="utf-8"):
    if ln.strip():
        d = json.loads(ln)
        ents[d["clip"]] = int(d["n_frames"])
clips_all = list(ents)
excluded = sorted(clips_all, key=D.sha12)[540:600]
ex = set(excluded)
clips_all = [c for c in clips_all if c not in ex]
clips = D.select_train_clips(clips_all, pa.data_frac)
names = list(ents)                                                                      # the join's row order is irrelevant to the digest (clip-name based)
wins = [(c, t) for c in clips for t in range(first, ents[c] - D.HORIZON)]
digest = hashlib.sha256("\n".join(f"{c}:{t}" for c, t in wins).encode("utf-8")).hexdigest()
banked = [json.load(open(p, encoding="utf-8")) for p in a.banked]
want = {b["train_window_list_sha256"] for b in banked}
rec = {"hist": cfg.hist, "window_start_effective": first, "n_clips": len(clips), "n_windows": len(wins), "window_list_sha256": digest,
       "banked_window_list_sha256": sorted(want), "banked_n_windows": [b["n_windows"] for b in banked], "banked_n_clips": [b["n_clips"] for b in banked]}
if want != {digest} or any(b["n_windows"] != len(wins) for b in banked):
    rec["error"] = "the rebuilt window list is NOT the banked run's: f would be a claim about another list"
    json.dump(rec, open(os.path.join(a.out, "f_measure.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))
    sys.exit(2)

tracks = D.load_native_tracks(a.tracks)
n_tot = n_launch = n_valid = n_nan = n_disagree = 0
per_clip = []
flags = {}                                                                              # clip -> bool [n_windows_of_clip] in window order (for the sampler simulation)
v0_launch, v6_launch = [], []
for ci, c in enumerate(clips):
    tr = tracks[c]
    pg = tr[0::2]
    T = len(pg)
    assert T == ents[c], (c, T, ents[c])
    kin3, _ = D.yaw_kin_from_track(tr)
    gt, gok = D.gt_future(pg)
    ts = np.arange(first, T - D.HORIZON)
    v0 = kin3[ts, 0]
    g = gt[ts]
    ok = gok[ts]
    fin = np.isfinite(v0) & np.isfinite(g).all((1, 2))
    n_nan += int((~fin).sum())
    m_t = M.launch_mask(torch.from_numpy(v0.astype(np.float64)), torch.from_numpy(g), torch.from_numpy(ok)).numpy()          # the trainer's function (float32 gt, as the collated batch holds it)
    v6 = np.linalg.norm((g[:, 29].astype(np.float64) - g[:, 28].astype(np.float64)), axis=-1) / 0.2                                # the definition, written again with numpy
    m_n = ok & np.isfinite(v0) & (v0 < 6.0) & (v6 >= v0 + 1.0)
    n_disagree += int((m_t != m_n).sum())
    n_tot += len(ts)
    n_valid += int(ok.sum())
    n_launch += int(m_t.sum())
    flags[c] = m_t
    if m_t.any():
        v0_launch.append(v0[m_t])
        v6_launch.append(v6[m_t])
    per_clip.append(int(m_t.sum()))
f = n_launch / n_valid
pc = np.array(per_clip)
v0l, v6l = np.concatenate(v0_launch), np.concatenate(v6_launch)
rec.update({"n_windows_all": n_tot, "n_gt_valid": n_valid, "n_nonfinite_v0_or_gt": n_nan, "n_launch": n_launch, "f": f, "mask_disagreements_torch_vs_numpy": n_disagree,
            "clips_with_a_launch_window": int((pc > 0).sum()), "launch_per_clip_max": int(pc.max()), "launch_v0_median": float(np.median(v0l)), "launch_v6_median": float(np.median(v6l)),
            "seconds_so_far": time.time() - t0})
# the dose rule, fixed before any run
rs = lambda x: float(f"{x:.6g}")   # noqa: E731  (OACC's round_sig, 6 significant digits)
w_for = lambda share: rs((1.0 - f) / (f * (1.0 / share - 1.0)) - 1.0)   # noqa: E731  share s: f w / (f w + 1 - f) = s  =>  w = (1-f) s / (f (1-s));  here w = 1 + W
wA = rs((1.0 - f) / (2.0 * f) - 1.0)
wB = rs((1.0 - f) / (5.0 * f) - 1.0)
rec["dose"] = {"rule_A": "1 + W = (1 - f) / (2 f)   (launch samples hold 1/3 of the along-track weight)", "W_A": wA, "rule_B": "1 + W = (1 - f) / (5 f)   (1/6)", "W_B": wB,
               "share_A_at_f_exact": (1 + wA) * f / ((1 + wA) * f + 1 - f), "share_B_at_f_exact": (1 + wB) * f / ((1 + wB) * f + 1 - f)}

# the realised share by the trainer's own sampler, per arm seed: ESTIMATED (a simulation of the sampler's index stream with the measured flags; the run record is the measurement)
ci_of = {c: i for i, c in enumerate(clips)}
flat = {}
wl = []
for c in clips:
    for k, t in enumerate(range(first, ents[c] - D.HORIZON)):
        wl.append((ci_of[c], t, 0, 0))
        flat[(ci_of[c], t)] = bool(flags[c][k])
lru = max(8, int(1.5e9 // (max(ents.values()) * cfg.n_tok * 1024 * 2)))
chunk = max(4, min(48, lru - 2))
sim = {}
for seed in (0, 1, 10, 11):
    smp = D.ChunkedSampler(wl, 64, seed=seed, chunk=chunk, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    rows = nl = 0
    for step, b in enumerate(smp.epoch(0)):
        if step >= 3000:
            break
        rows += len(b)
        nl += sum(flat[(w[0], w[1])] for w in b)
    sim[str(seed)] = {"rows": rows, "launch": nl, "f_hat": nl / rows,
                      "share_A": (1 + wA) * nl / ((1 + wA) * nl + rows - nl), "share_B": (1 + wB) * nl / ((1 + wB) * nl + rows - nl)}
rec["sampler_simulation_3000_steps_batch64"] = {"lru": lru, "chunk": chunk, "seeds": sim,
                                                "K_SHARE_A_band": [1 / 3 - 0.02, 1 / 3 + 0.02], "K_SHARE_B_band": [1 / 6 - 0.02, 1 / 6 + 0.02],
                                                "A_in_band": {s: (1 / 3 - 0.02 <= v["share_A"] <= 1 / 3 + 0.02) for s, v in sim.items()},
                                                "B_in_band": {s: (1 / 6 - 0.02 <= v["share_B"] <= 1 / 6 + 0.02) for s, v in sim.items()}}
rec["seconds"] = time.time() - t0
rec["evidence_class"] = "MEASURED (f, the window list, the dose); ESTIMATED (the simulated shares: the sampler stream, not a training run)"
json.dump(rec, open(os.path.join(a.out, "f_measure.json"), "w"), indent=1)
print(json.dumps(rec, indent=1))
sys.exit(0 if n_disagree == 0 else 3)
