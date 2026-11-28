"""LL2 — ⭐ THE POSITIVE CONTROL FOR THE SANITY ANCHOR.

⛔ WHY THIS EXISTS. The ladder's anchor row is NEGATIVE: a ridge on the v6
`cells` cannot recover the ego's own speed (it loses to a constant). The D1
withdrawal is the programme's standing lesson that **a negative from an
instrument with no positive control is not a fact about the model** — five
negative controls proved the slot probe was not cheating and none proved it
could measure. So before "the latent does not encode ego speed" may be said,
this run establishes that THE READOUT CAN MEASURE EGO SPEED AT THIS n/p when
the information is linearly present.

⚠️ AND IT IS DELIBERATELY THE *DISTRIBUTED* CONTROL, NOT THE EASY ONE. The
banked `ORC-DIRECT` control puts its answer at a FIXED ADDRESS (cell 0), which
is the easiest possible linear problem and does not test whether a ridge with
2 049 features and ~2 200 training windows can find a signal SMEARED ACROSS ALL
DIMENSIONS — which is how a real learned latent would carry it. Here `v0` is
pushed through a fixed random projection into every one of the 16x128 cells, so
no single feature is a copy of the label.

CONSTRUCTION — `pc1_oracle_cache.py`'s recipe, with the ego instead of the boxes
  * take the REAL @11250 cache and replace ONLY `cells`, keeping every window,
    target, class, rate, clip_id, episode_uid and the declared split, so the
    paired bootstrap's clusters are literally the same objects;
  * cell k = P @ [v0 / 10, 1.0] with P a fixed [128, 2] N(0,1) matrix
    (proj seed 20260817, the same as pc1) tiled over all 16 cells with an
    independent projection per cell;
  * rescale to the REAL cells' global std, then add Gaussian noise at
    `--noise-rel` x that std. `--noise-rel 0.10` is pc1's level.
  * ⭐ A NOISE SWEEP IS RUN, not a single point: 0.10 / 1.0 / 3.0 / 10.0. A
    control that only ever passes at a noiseless operating point says nothing
    about the SNR at which this readout stops working, and the SNR is exactly
    the quantity in dispute for a real latent.

⛔ SYNTHETIC POSITIVE CONTROL. An upper bound on the apparatus. NEVER a model
result and never comparable to a v6 arm's number.
"""
from __future__ import annotations

import pyarrow  # noqa: F401  # isort: skip  (pyarrow BEFORE torch on this box)

import argparse
import json
from pathlib import Path

import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--noise-rel", type=float, nargs="+",
                    default=[0.10, 1.0, 3.0, 10.0])
    ap.add_argument("--proj-seed", type=int, default=20260817)
    ap.add_argument("--noise-seed", type=int, default=17)
    a = ap.parse_args(argv)

    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    n_c, d_r = rows[0]["cells"].shape
    real = torch.stack([r["cells"].float() for r in rows])
    real_std = float(real.std())

    gp = torch.Generator().manual_seed(int(a.proj_seed))
    P = torch.randn(n_c, d_r, 2, generator=gp)              # per-cell [128, 2]
    v0 = torch.tensor([float(r["v0"]) for r in rows]).unsqueeze(1)   # [N, 1]
    f = torch.cat([v0 / 10.0, torch.ones_like(v0)], 1)               # [N, 2]
    sig = torch.einsum("cdk,nk->ncd", P, f)                          # [N,16,128]
    sig = sig * (real_std / float(sig.std()))

    for nr in a.noise_rel:
        gn = torch.Generator().manual_seed(int(a.noise_seed))
        cells = (sig + torch.randn(sig.shape, generator=gn)
                 * (nr * real_std)).to(torch.float16)
        out_rows = []
        for i, r in enumerate(rows):
            q = dict(r)
            q["cells"] = cells[i].clone()
            q["tokens"] = None
            out_rows.append(q)
        m = dict(meta)
        m["run_stamp"] = f"EGO-ORACLE-n{nr:g}@{meta.get('step')}"
        m["_evidence_class"] = (
            "SYNTHETIC POSITIVE CONTROL (the REAL cache with `cells` replaced "
            "by a fixed per-cell random projection of the window's own banked "
            "v0, rescaled to the real cells' std and noised). An UPPER BOUND "
            "ON THE READOUT, never a model result.")
        m["ego_oracle_cells"] = {
            "features": ["v0/10", "1.0"], "encoding": "DISTRIBUTED — an "
            "independent random [128,2] projection per cell; no feature is a "
            "bare copy of the label",
            "proj_seed": int(a.proj_seed), "noise_seed": int(a.noise_seed),
            "noise_rel": float(nr),
            "real_cells_std": round(real_std, 6),
            "signal_rescale_factor": round(real_std / float(sig.std()) if
                                           float(sig.std()) else 1.0, 6)}
        d = Path(a.out_dir) / f"cache_egoorc_n{nr:g}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"rows": out_rows, "meta": m}, d / "latents.pt")
        (d / "ll2_meta.json").write_text(
            json.dumps({k: v for k, v in m.items()
                        if k in ("run_stamp", "_evidence_class",
                                 "ego_oracle_cells", "step", "n_frames")},
                       indent=1), "utf-8")
        print(f"  EGO-ORACLE noise_rel={nr:<5g} -> {d}  "
              f"(real_cells_std {real_std:.6f}, n {len(out_rows)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
