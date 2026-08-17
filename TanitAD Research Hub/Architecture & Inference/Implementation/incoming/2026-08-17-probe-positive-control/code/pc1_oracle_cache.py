"""PC1 — the TRIVIALLY-SUFFICIENT UPPER BOUND cache: `cells` REPLACED BY AN
ENCODING OF THE GT BOXES THEMSELVES.

⛔ WHY THIS EXISTS. Every control in `…/2026-08-17-slot-probe-parity/` is
NEGATIVE (C-CONST, C-SHUF, C-EPMEAN, C-SHUF-XEP, random-latent-matched). They
prove the probe is not cheating. NOT ONE proves the probe can SUCCEED AT ALL.
So "the v6 latent does not carry agents" is confounded with "this
probe/label/window/fit pipeline cannot read agents from anything".

This is the cheapest possible apparatus check: hand the probe a memory tensor
that CONTAINS THE ANSWER and see whether it recovers it. A probe that cannot
read the lead gap out of an explicit encoding of the GT boxes is BROKEN, and
then D1 is a statement about the instrument, not about the world model.

⭐ IT IS THE SAME MOVE AS `pA_null_matched.py`, WITH SIGNAL INSTEAD OF NOISE:
take the REAL cache and replace ONLY the `cells` tensor, keeping every window,
every target, every clip_id, the split, and therefore the estimator's clusters.
Everything downstream — sp2_probe.py, byte-identical — is unchanged.

═══════════════════════════════════════════════════════════════════════════════
THE ENCODING — DECLARED HERE, BEFORE ANY FIT
═══════════════════════════════════════════════════════════════════════════════
`agents` is [A, 6] = (cx, cy, yaw, l, w, occ) in the EGO frame, already
restricted by sp1 to the in-grid box (0 < cx <= 60 m, |cy| <= 16 m).

  1. sort the frame's agents by `cx` ASCENDING (nearest first);
  2. take the first `n_cells` (= 16, the real cache's own memory length — the
     shape must match or the head geometry changes and the comparison dies);
  3. cell k for k < min(A, 16):   x_k = P @ f_k
     with f_k = [cx/60, cy/16, sin(yaw), cos(yaw), l/10, w/5, 1.0]  (7 dims,
     the last being an explicit PRESENCE flag) and P a FIXED [128, 7] N(0,1)
     matrix drawn from seed 20260817 — a random projection, so no dimension of
     the memory is a bare copy of a label field;
  4. cells for k >= A: the ZERO vector (an explicitly empty slot);
  5. the whole tensor is rescaled so its global std equals the REAL cache's
     global `cells` std, so the head's `mem_proj`/cross-attention sees the same
     input scale it saw on the real arms (memory is NOT layer-normed inside
     `nn.TransformerDecoderLayer`, so scale is not free);
  6. Gaussian noise at `--noise-rel` x that same std is added.

⚠️ WHAT THIS IS AND IS NOT. This is an UPPER BOUND on the apparatus, not a
model. It says what the probe does when the representation is essentially
perfect. It does NOT establish that the probe would succeed on a realistic
learned representation — that is what the `--noise-rel` ladder is for, and what
a real image-encoder positive control (PC2) is for.

⚠️ It is deliberately NOT a copy of the answer: the metric is the nearest
IN-CORRIDOR agent (|cy| <= 1.75 m) within 30 m, while the encoding is sorted by
`cx` over the whole 32 m-wide grid, is randomly projected, is noised, and is
truncated at 16. The head must still learn the corridor rule, the projection,
and the set structure. `--report-coverage` measures how often the GT lead even
survives the 16-cell truncation, and that number is published.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a REAL sp1 cache latents.pt")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--noise-rel", type=float, default=0.10,
                    help="noise std as a fraction of the signal std")
    ap.add_argument("--lead-cell", action="store_true",
                    help="⭐ THE TRIVIALLY-SUFFICIENT VARIANT (ORC-DIRECT). Put "
                         "the GT LEAD — the very agent the metric asks for, "
                         "selected by `gt_lead_gap`'s own rule — in CELL 0, and "
                         "the remaining agents (cx-sorted) in cells 1..n-1. The "
                         "answer is then at a FIXED, KNOWN address, so a linear "
                         "map can read it and a slot head has only to bind one "
                         "query to one cell. ⚠️ This deliberately leaks the "
                         "corridor rule into the FEATURE — that is the point of "
                         "an upper bound, and it is stated, not hidden. A probe "
                         "that fails HERE is broken, full stop.")
    ap.add_argument("--proj-seed", type=int, default=20260817)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--meta-out", default=None)
    a = ap.parse_args(argv)

    blob = torch.load(a.src, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], dict(blob["meta"])
    n_c, d_r = rows[0]["cells"].shape

    real = torch.stack([r["cells"] for r in rows]).float()
    real_std = float(real.std())
    real_mean = float(real.mean())
    del real

    gp = torch.Generator().manual_seed(int(a.proj_seed))
    P = torch.randn(d_r, 7, generator=gp)                    # [128, 7]
    gn = torch.Generator().manual_seed(int(a.noise_seed))

    # --- pass 1: build the raw (unscaled, unnoised) signal -------------------
    sig = torch.zeros(len(rows), n_c, d_r)
    n_trunc = 0
    lead_in_cells = 0
    lead_windows = 0
    for i, r in enumerate(rows):
        ag = r["agents"].float()                             # [A, 6]
        A = int(ag.shape[0])
        if A == 0:
            continue
        order = torch.argsort(ag[:, 0])                      # cx ascending
        if a.lead_cell:
            cx0, cy0 = ag[:, 0], ag[:, 1]
            mm = (cx0 > 0) & (cy0.abs() <= 1.75) & (cx0 <= 30.0)
            if bool(mm.any()):
                nz = torch.nonzero(mm).flatten()
                jlead = int(nz[cx0[nz].argmin()])
                rest = [int(t) for t in order.tolist() if int(t) != jlead]
                order = torch.tensor([jlead] + rest, dtype=torch.long)
        keep = order[:n_c]
        if A > n_c:
            n_trunc += 1
        f = torch.stack([
            ag[keep, 0] / 60.0,
            ag[keep, 1] / 16.0,
            torch.sin(ag[keep, 2]),
            torch.cos(ag[keep, 2]),
            ag[keep, 3] / 10.0,
            ag[keep, 4] / 5.0,
            torch.ones(keep.numel()),
        ], dim=1)                                            # [k, 7]
        sig[i, :keep.numel()] = f @ P.T

        # coverage diagnostic: is the GT LEAD inside the kept 16?
        cx, cy = ag[:, 0], ag[:, 1]
        m = (cx > 0) & (cy.abs() <= 1.75) & (cx <= 30.0)
        if bool(m.any()):
            lead_windows += 1
            j = int(torch.nonzero(m).flatten()[cx[torch.nonzero(m).flatten()]
                                               .argmin()])
            if j in set(keep.tolist()):
                lead_in_cells += 1

    s = float(sig.std())
    scale = (real_std / s) if s > 0 else 1.0
    sig = sig * scale
    noise = torch.randn(sig.shape, generator=gn) * (a.noise_rel * real_std)
    out = (sig + noise).to(torch.float16)

    for i, r in enumerate(rows):
        r["cells"] = out[i].clone()
        r["tokens"] = None

    cov = lead_in_cells / max(lead_windows, 1)
    meta["run_stamp"] = (("GT-ORACLE-DIRECT" if a.lead_cell
                          else "GT-ORACLE-CELLS")
                         + f"-n{a.noise_rel:g}@{meta.get('step')}")
    meta["_evidence_class"] = (
        "SYNTHETIC POSITIVE CONTROL (the REAL cache with `cells` replaced by a "
        "fixed random projection of the frame's own GT boxes, rescaled to the "
        "real cells' std and noised). An UPPER BOUND ON THE APPARATUS, never a "
        "model result.")
    meta["tokens_banked"] = False
    meta["oracle_cells"] = {
        "n_cells": n_c, "d_readout": d_r,
        "features": ["cx/60", "cy/16", "sin(yaw)", "cos(yaw)", "l/10", "w/5",
                     "presence=1"],
        "sort": ("GT LEAD in cell 0, then cx ascending" if a.lead_cell
                 else "cx ascending, first n_cells kept"),
        "lead_cell_variant": bool(a.lead_cell),
        "proj_seed": int(a.proj_seed), "noise_seed": int(a.noise_seed),
        "noise_rel": float(a.noise_rel),
        "real_cells_std": round(real_std, 6),
        "real_cells_mean": round(real_mean, 6),
        "signal_rescale_factor": round(scale, 6),
        "windows_truncated_at_16": n_trunc,
        "n_rows": len(rows),
        "gt_lead_windows": lead_windows,
        "gt_lead_survives_truncation": lead_in_cells,
        "gt_lead_coverage": round(cov, 6),
    }
    torch.save({"rows": rows, "meta": meta}, a.dst)
    js = json.dumps(meta["oracle_cells"], indent=1)
    print(f"[pc1] wrote {a.dst}\n{js}", flush=True)
    if a.meta_out:
        with open(a.meta_out, "w", encoding="utf-8") as fh:
            json.dump({"src": a.src, "dst": a.dst,
                       "run_stamp": meta["run_stamp"],
                       "oracle_cells": meta["oracle_cells"]}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
