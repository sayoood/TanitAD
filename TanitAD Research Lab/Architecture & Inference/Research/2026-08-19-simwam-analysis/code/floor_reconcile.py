"""H-RANK-16 / H-RANK-21 — settle the floor contradiction on REAL DINOv3.

    `O6_PARTICIPATION_FLOOR = 8.56`   sourced "frozen DINOv3, n=1440"
    `E_TRUNK_3_LADDER.md`      40.77  frozen DINOv3, 5,617 frames / 130 episodes

Same encoder, same programme, 4.76x apart -- and EVERY collapse verdict we have
issued (champ30k's FAIL at 6.499 among them) is measured against the smaller one.

⭐ WHY THIS RUN CAN SETTLE IT WITH NO GPU.  `dinov3_fields/` holds the banked
per-clip patch tokens [F, 640, 1024] fp16 for 130 clips ~= 5,617 frames -- i.e.
THE VERY SAMPLE the 40.77 was computed on.  Mean-pooling those tokens and then
SUBSAMPLING the frame axis varies n while holding the encoder, the corpus, the
preprocessing and the statistic FIXED.  Any movement is attributable to n alone.

⛔ MEMORY.  130 x [43, 640, 1024] fp16 is ~7.3 GB dense, ~29 GB as float64 --
the dense-tensor trap that thrashed a box for 2.5 h.  Tokens are mean-pooled per
FILE on load, so the bank is 5,617 x 1024 float64 = 46 MB.

Reported per n: mean +- sd over repeated draws, never a single draw and never a
maximum (C133).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
FIELDS = SP / "dinov3_fields"
OUT = SP / "h_rank16_floor_reconcile.json"


def main() -> int:
    from tanitad.models.v6 import spectrum_report, O6_PARTICIPATION_FLOOR

    files = sorted(FIELDS.glob("*.npy"))
    assert files, f"no banked DINOv3 fields under {FIELDS}"

    # ---- mean-pool per file (OUR instrument's treatment) ---------------------
    rows, per_clip = [], []
    for f in files:
        a = np.load(f, mmap_mode="r")             # [F, n_tok, 1024] fp16
        pooled = np.asarray(a, dtype=np.float32).mean(axis=1)   # [F, 1024]
        rows.append(pooled.astype(np.float64))
        per_clip.append(pooled.shape[0])
    Z = np.concatenate(rows)
    n_tok = int(np.load(files[0], mmap_mode="r").shape[1])
    print(f"\n  banked frozen DINOv3: {len(files)} clips, {Z.shape[0]} frames, "
          f"d={Z.shape[1]}, {n_tok} patch tokens mean-pooled per frame")

    # ⛔ content assertion: an all-zero / non-finite bank scores like anything
    assert np.isfinite(Z).all(), "non-finite values in the banked fields"
    assert float(np.abs(Z).mean()) > 0, "the banked fields are all zero"
    print(f"  content check: |mean| = {float(np.abs(Z).mean()):.4f}  (non-zero, finite)\n")

    def partic(X: np.ndarray) -> float:
        return float(spectrum_report(torch.from_numpy(X).float())["participation_ratio"])

    full = partic(Z)
    rep = {"_evidence_class": "MEASURED (ours; banked frozen DINOv3 fields, dev-box CPU)",
           "hypothesis": "H-RANK-16 / H-RANK-21",
           "encoder": "facebook/dinov3-vitl16-pretrain-lvd1689m (frozen), patch tokens mean-pooled",
           "instrument": "tanitad.models.v6.spectrum_report -> participation_ratio (p prop sigma^2)",
           "n_clips": len(files), "n_frames_total": int(Z.shape[0]), "d": int(Z.shape[1]),
           "participation_full_sample": round(full, 3),
           "reference_E_TRUNK_3": 40.77,
           "code_floor_O6_PARTICIPATION_FLOOR": float(O6_PARTICIPATION_FLOOR),
           "by_n": {}}
    print(f"  FULL SAMPLE  n={Z.shape[0]:>5}  participation = {full:8.3f}   "
          f"(E-TRUNK-3 reports 40.77)\n")

    print(f"     {'n':>7}{'draws':>7}{'participation':>16}{'sd':>8}   vs the two numbers")
    rng = np.random.default_rng(0)
    for n in (360, 720, 1440, 2160, 2880, 4320, min(5617, Z.shape[0])):
        if n > Z.shape[0]:
            continue
        draws = 12 if n < Z.shape[0] else 1
        vals = [partic(Z[rng.choice(Z.shape[0], n, replace=False)]) for _ in range(draws)]
        m, s = float(np.mean(vals)), float(np.std(vals))
        rep["by_n"][str(n)] = {"participation_mean": round(m, 3), "sd": round(s, 3),
                               "draws": draws}
        note = ""
        if n == 1440:
            note = f"   <- the floor's n; code says {O6_PARTICIPATION_FLOOR}"
        print(f"     {n:>7}{draws:>7}{m:>16.3f}{s:>8.3f}{note}")

    # ---- the verdict --------------------------------------------------------
    p1440 = rep["by_n"].get("1440", {}).get("participation_mean")
    ratio = (full / p1440) if p1440 else float("nan")
    rep["ratio_full_over_1440"] = None if not np.isfinite(ratio) else round(ratio, 3)
    rep["programme_ratio_40p77_over_8p56"] = 4.763

    near_floor = p1440 is not None and abs(p1440 - O6_PARTICIPATION_FLOOR) <= 0.35 * O6_PARTICIPATION_FLOOR
    near_ref = abs(full - 40.77) <= 0.35 * 40.77
    if near_floor and near_ref:
        rep["verdict"] = (
            f"SETTLED — the SAME banked DINOv3 reads {p1440:.2f} at n=1440 and {full:.2f} at "
            f"n={Z.shape[0]}. Both published numbers are reproduced from ONE representation by "
            f"changing n ALONE ⇒ 8.56 and 40.77 were never in conflict; participation is "
            f"N-DEPENDENT and neither number is 'the' DINOv3 reference. ⛔ CONSEQUENCE: a floor "
            f"is only meaningful at MATCHED n, and every arm compared to 8.56 must itself be "
            f"measured at n=1440 (champ30k's 6.499 IS at n=1440, so that comparison stands).")
    elif p1440 is not None and not near_floor:
        rep["verdict"] = (
            f"NOT explained by n alone — at n=1440 this bank reads {p1440:.2f}, not ~8.56. "
            f"The floor's 8.56 came from a different corpus or a different pooling; it must be "
            f"re-derived on the 12 val clips before any arm is judged against it.")
    else:
        rep["verdict"] = f"partial: full={full:.2f} vs 40.77, n=1440 -> {p1440}"
    print(f"\n  ratio full/1440 = {ratio:.3f}   (the programme's 40.77/8.56 = 4.763)")
    print(f"\n  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
