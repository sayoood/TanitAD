"""The apples-to-apples floor: frozen DINOv3 through OUR instrument on the SAME
12 val clips the 8.56 is sourced to, at n=1440.

This is the one measurement that can settle H-RANK-16.  Established so far:

  * on the 130-clip lead corpus (5,617 frames), `spectrum_report` over the banked
    DINOv3 fields reads **20.52**, and it is FLAT in n (20.23 at n=1440) --
    so neither the code's 8.56 nor E-TRUNK-3's 40.77 is reproduced there, and
    sample size is NOT the explanation (H-RANK-21 REFUTED).
  * but that corpus is not the corpus 8.56 came from. Corpus was never held fixed.

Here the corpus IS held fixed: the same 12 physicalai-val clips, 120 frames each,
n = 1440, frozen DINOv3 ViT-L/16, patch tokens mean-pooled per frame -- i.e. the
identical treatment `spectrum_report` gives z_op.

⛔ bf16 NEVER fp16 (a DINOv3 fp16 forward returns ALL-NaN silently and a full
extraction prints DONE on garbage), and the features are content-asserted before
use, not merely checked for existence.

Runs on the dev-box 4060 alongside two live training arms, so batch is small and
peak device memory is reported via torch.cuda.max_memory_allocated().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")

VAL = SP / "sp2/cache/physicalai-val-w120-256x640cyl"
OUT = SP / "h_rank16_floor_valclips.json"


def main() -> int:
    import v7tiny_probe as P
    from tanitad.models.v6 import spectrum_report, O6_PARTICIPATION_FLOOR

    dev = torch.device("cuda")
    clips = sorted(VAL.glob("*.v2ep.pt"))[:12]
    assert clips, f"no val clips under {VAL}"
    print(f"\n  frozen DINOv3 on the SAME 12 val clips, 120 frames each (target n=1440)\n",
          flush=True)

    feats = P.dinov3_encode(clips, 120, dev)
    Z = np.concatenate(feats).astype(np.float64)
    assert np.isfinite(Z).all(), "non-finite DINOv3 features (the fp16 NaN mode)"
    assert float(np.abs(Z).mean()) > 0, "all-zero DINOv3 features"

    rep_full = spectrum_report(torch.from_numpy(Z).float())
    partic = float(rep_full["participation_ratio"])
    peak = float(torch.cuda.max_memory_allocated()) / 1e9

    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "hypothesis": "H-RANK-16 — the apples-to-apples floor",
           "encoder": "facebook/dinov3-vitl16-pretrain-lvd1689m (frozen), patch tokens "
                      "mean-pooled per frame",
           "instrument": "tanitad.models.v6.spectrum_report -> participation_ratio",
           "corpus": "physicalai-val, the SAME 12 clips as the 8.56 provenance",
           "n": int(Z.shape[0]), "d": int(Z.shape[1]),
           "participation": round(partic, 3),
           "top8_share": round(float(rep_full["top_k_share"]), 4),
           "feature_abs_mean": round(float(np.abs(Z).mean()), 4),
           "code_floor": float(O6_PARTICIPATION_FLOOR),
           "e_trunk_3_reference": 40.77,
           "lead_corpus_same_instrument": 20.52,
           "cuda_max_mem_gb": round(peak, 2)}

    d856 = abs(partic - O6_PARTICIPATION_FLOOR) / O6_PARTICIPATION_FLOOR
    d4077 = abs(partic - 40.77) / 40.77
    d2052 = abs(partic - 20.52) / 20.52
    if d856 <= 0.15:
        out["verdict"] = (f"the code floor is REPRODUCED on its own corpus ({partic:.2f} vs "
                          f"8.56). ⇒ 8.56 is corpus-specific and CORRECT for val-clip "
                          f"comparisons; E-TRUNK-3's 40.77 and the lead-corpus 20.52 are "
                          f"different corpora and must never be quoted as the val floor.")
    elif d4077 <= 0.15:
        out["verdict"] = (f"the val corpus reproduces E-TRUNK-3's 40.77 ({partic:.2f}), NOT "
                          f"the code's 8.56 ⇒ O6_PARTICIPATION_FLOOR is WRONG and every arm "
                          f"failed against it was failed against a number ~4.8x too small.")
    elif d2052 <= 0.15:
        out["verdict"] = (f"the val corpus agrees with the lead corpus ({partic:.2f} vs 20.52) "
                          f"⇒ our instrument is corpus-stable and BOTH published numbers "
                          f"(8.56 and 40.77) are unreproducible; the floor must be reset to "
                          f"the measured value with a pinning test.")
    else:
        out["verdict"] = (f"a FOURTH value ({partic:.2f}) — distinct from 8.56, 40.77 and the "
                          f"lead-corpus 20.52. The floor cannot be set until the discrepancy "
                          f"is explained; do not fail any arm on it.")
    print(f"\n  participation = {partic:.3f}   (n={Z.shape[0]}, d={Z.shape[1]}, "
          f"top8 {out['top8_share']:.4f}, peak {peak:.2f} GB)")
    print(f"  code floor 8.56 · E-TRUNK-3 40.77 · lead corpus (same instrument) 20.52")
    print(f"\n  VERDICT: {out['verdict']}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
