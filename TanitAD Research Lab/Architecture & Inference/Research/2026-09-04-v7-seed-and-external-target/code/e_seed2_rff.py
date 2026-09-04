"""E-SEED-2c — the SAME arms through the REGISTER'S OWN environment instrument.

⛔ WHY. E-SEED-2's environment column came back VOID: every arm, INCLUDING frozen
DINOv3, read below a constant predictor on `n_agents`, and the instrument-validity
contrast `dino_hf - scratch` was not separated. Two candidate causes, with
opposite consequences:

  (I) THE INSTRUMENT. The register's environment panels do NOT use a linear ridge
      scored by pooled R2. They use `rangeprobe_rff.rff_fold` -- PCA(96) ->
      standardise -> random Fourier features (D=1024, RBF, median-heuristic
      bandwidth) -> ridge -- scored by `within_clip_r`, a WITHIN-CLIP Pearson r
      whose constant control reads EXACTLY 0.0. That is a DIFFERENT ESTIMAND:
      pooled R2 asks "does it beat the scored-set mean", within-clip r asks "does
      it track the target inside each clip". On a target whose variance is mostly
      BETWEEN clips the two can disagree completely.
      ⇒ if this is the cause, the register's +0.2754 and my -0.0769 are not
      comparable numbers and neither is wrong.

  (P) POWER. 24 clips is too few for a clip-level count target however it is
      scored. ⇒ then the whole question needs the 130-clip bank.

ONE VARIABLE: the probe (linear ridge / pooled R2  ->  RFF / within-clip r).
The FEATURES are the byte-identical cached arrays from `e_seed2_panel.py`.

COMMITTED IN ADVANCE (written before this was run):
  * if `dino_hf - scratch` becomes SEPARATED under the RFF/within-clip read, the
    VOID was (I) -- an instrument mismatch -- and the seed question IS answerable
    on these 24 clips, by this instrument;
  * if it stays not separated, it is (P), the panel stays VOID, and the answer
    needs the 130-clip bank;
  * either way the constant control must read EXACTLY 0.0 and the time-shuffled
    control must sit at it, or the panel is VOID for a third reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE / "eseed2"
ASSETS = Path(r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901")
sys.path.insert(0, str(ASSETS))
sys.path.insert(0, str(HERE))

from panel_kfold import kfold_clip_scores                        # noqa: E402
from rangeprobe_rff import rff_fold, within_clip_r               # noqa: E402

ARMS = ["dino_hf", "seed_asis", "seed_imnet", "seed_imnet_pos0",
        "scratch", "pixel"]
N_BOOT = 2000


def boot(vals, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    d = np.array([np.nanmean(rng.choice(vals, len(vals), True))
                  for _ in range(n_boot)])
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def paired(a, b, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    d = np.array([np.nanmean(rng.choice(diff, len(diff), True))
                  for _ in range(n_boot)])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"delta": float(np.nanmean(diff)), "ci95": [lo, hi],
            "separated": bool(lo > 0 or hi < 0)}


def per_clip(F, y, clip):
    cl = np.unique(clip)
    Xc = [F[clip == c] for c in cl]
    Yc = [y[clip == c].reshape(-1, 1) for c in cl]
    return kfold_clip_scores(Xc, Yc, rff_fold, within_clip_r,
                             k_folds=6, seed=0)


def main() -> int:
    z = np.load(BANK / "eseed2b_preds.npz")
    clip = z["clip"]
    y_env = z["n_agents_psg"].astype(np.float64)
    y_spd = z["speed"].astype(np.float64)
    rng = np.random.default_rng(11)
    y_shuf = y_env[rng.permutation(len(y_env))]      # time/row-shuffled control

    out = {"meta": {
        "instrument": ("rangeprobe_rff.rff_fold (PCA 96 -> RFF D=1024, RBF, "
                       "median-heuristic bandwidth -> ridge; lambda on a "
                       "CLIP-DISJOINT inner split) scored by "
                       "rangeprobe_rff.within_clip_r; 6-fold clip-disjoint "
                       "out-of-fold via panel_kfold.kfold_clip_scores -- ALL "
                       "IMPORTED from the register's own modules, not "
                       "re-implemented"),
        "estimator": "clip bootstrap over the 24 per-clip scores, 2000 draws; "
                     "paired for contrasts",
        "targets": ["n_agents_psg", "speed", "n_agents_psg_row_shuffled"],
        "n_rows": int(len(y_env)), "n_clips": int(len(np.unique(clip))),
        "evidence_class": "MEASURED (ours; dev-box, CPU)"},
        "arms": {}, "scores": {}}

    for arm in ARMS:
        F = np.load(BANK / f"featsp_{arm}.npy")
        row = {}
        for tname, y in (("n_agents_psg", y_env), ("speed", y_spd),
                         ("n_agents_psg_row_shuffled", y_shuf)):
            s = per_clip(F, y, clip)
            out["scores"][f"{arm}::{tname}"] = [float(v) for v in s]
            row[tname] = {"mean_within_clip_r": float(np.nanmean(s)),
                          "ci95": boot(s), "n_clips": int(len(s)),
                          "d_ambient": int(F.shape[1])}
        out["arms"][arm] = row
        print(f"[{arm:16s}] " + " | ".join(
            f"{t} {row[t]['mean_within_clip_r']:+.4f} "
            f"[{row[t]['ci95'][0]:+.3f},{row[t]['ci95'][1]:+.3f}]"
            for t in row), flush=True)
        (BANK / "eseed2c_rff.json").write_text(json.dumps(out, indent=2),
                                               encoding="utf-8")

    pairs = [("dino_hf", "scratch"), ("dino_hf", "pixel"),
             ("seed_asis", "scratch"), ("seed_imnet_pos0", "scratch"),
             ("dino_hf", "seed_asis"), ("dino_hf", "seed_imnet_pos0"),
             ("seed_imnet_pos0", "seed_asis"), ("seed_imnet_pos0", "pixel")]
    con = {}
    for tname in ("n_agents_psg", "speed"):
        for a, b in pairs:
            con[f"{tname}::{a}-{b}"] = paired(
                out["scores"][f"{a}::{tname}"], out["scores"][f"{b}::{tname}"])
            r = con[f"{tname}::{a}-{b}"]
            print(f"  {tname:14s} {a}-{b:18s} d={r['delta']:+.4f} "
                  f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
                  f"{'SEPARATED' if r['separated'] else '.'}")
    out["contrasts"] = con
    out["controls"] = {
        "constant_only": ("within_clip_r returns EXACTLY 0.0 for a zero-variance "
                          "prediction -- the no-information value is a known "
                          "value, not an estimate"),
        "row_shuffled": "n_agents_psg_row_shuffled; must sit at the constant value",
        "raw_input_floor": "pixel",
        "deliberate_regression": "scratch"}
    (BANK / "eseed2c_rff.json").write_text(json.dumps(out, indent=2),
                                           encoding="utf-8")
    print("\nwrote", BANK / "eseed2c_rff.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
