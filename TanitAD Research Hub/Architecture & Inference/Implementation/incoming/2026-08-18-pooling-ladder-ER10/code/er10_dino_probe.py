"""JOB 2 PRE-FLIGHT — is a FROZEN EXTERNAL HIGH-DIVERSITY ENCODER obtainable on
this dev box? (POOLING_BOTTLENECK_R1R2.md §8.1, open item §12.1.5.)

⚠️ THE SPEC LEAVES THIS OPEN AND THE BRIEF SAYS SO: "DINOv3 weight availability
is an open question — flagged in the spec, not resolved." This resolves it by
MEASUREMENT, and it probes MORE THAN ONE LOCATION AND MORE THAN ONE NAME,
because absence found at one location is not absence.

⭐ WHY GRID PARITY DECIDES THE SUBSTITUTION, not popularity:
our encoder tiles 256x640 px at patch 16 into EXACTLY 16x40 = 640 tokens. Only a
/16 backbone reproduces that grid, so the 40:1 / 10:1 / 4:1 / 1:1 ladder is the
IDENTICAL operator on the IDENTICAL grid. A /14 backbone (DINOv2) does not tile
256x640 and would force either a resize or a different grid — i.e. it would
change TWO things at once, which is the --v2 conflation the spec refuses.

⛔ THE TOKEN IS READ IN PLACE AND NEVER PRINTED, COPIED OR PASSED AS AN ARG.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()      # certifi fails behind this box's TLS proxy

CANDIDATES = [
    # (repo_id, patch, d_model, why)
    ("facebook/dinov3-vitb16-pretrain-lvd1689m", 16, 768,
     "⭐ PREFERRED — /16 reproduces our 16x40 grid EXACTLY; LVD-1689M is the "
     "high-diversity corpus the discriminator needs"),
    ("facebook/dinov3-vits16-pretrain-lvd1689m", 16, 384,
     "smaller /16 fallback, same corpus and same grid"),
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", 16, 1024,
     "larger /16, same corpus and same grid"),
    ("facebook/dinov3-convnext-base-pretrain-lvd1689m", None, None,
     "convnext variant — NOT token-grid comparable"),
    ("facebook/dinov2-base", 14, 768,
     "⚠️ /14 — does NOT tile 256x640; a substitution costs grid parity"),
    ("facebook/dinov2-with-registers-base", 14, 768,
     "⚠️ /14, registers variant"),
    ("timm/vit_base_patch16_224.dino", 16, 768,
     "⚠️ ORIGINAL DINO (v1), /16 — ImageNet-1k only, i.e. NOT high-diversity, "
     "so it does not serve the corpus-narrowness argument"),
]


def main() -> int:
    keys = Path(sys.argv[1] if len(sys.argv) > 1
                else r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt")
    tok = None
    if keys.exists():
        m = re.findall(r"hf_[A-Za-z0-9]+", keys.read_text("utf-8", "ignore"))
        tok = m[0] if m else None
    out = {"_evidence_class": "MEASURED (ours; live HF probe from the dev box)",
           "token_found_in_keys_txt": bool(tok),
           "token_value_ever_printed": False,
           "truststore_injected": True,
           "probes": {}}

    from huggingface_hub import HfApi
    from huggingface_hub.utils import (EntryNotFoundError,  # noqa: F401
                                       GatedRepoError, RepositoryNotFoundError)
    api = HfApi(token=tok)

    for repo, patch, d, why in CANDIDATES:
        rec = {"patch": patch, "d_model": d, "why": why}
        # --- probe 1: the model-info API -----------------------------------
        try:
            info = api.model_info(repo, files_metadata=False)
            rec["probe1_model_info"] = "OK"
            rec["gated"] = getattr(info, "gated", None)
            rec["files"] = sorted(s.rfilename for s in (info.siblings or []))[:40]
        except Exception as e:                                # noqa: BLE001
            rec["probe1_model_info"] = f"{type(e).__name__}: {str(e)[:200]}"
        # --- probe 2: a DIFFERENT path — the file listing endpoint ---------
        # (the absence rule: a second path and the tool that owns the fact)
        try:
            fl = api.list_repo_files(repo)
            rec["probe2_list_repo_files"] = "OK"
            rec["n_files"] = len(fl)
            rec["has_safetensors"] = any(f.endswith(".safetensors") for f in fl)
            rec["has_config"] = any(f.endswith("config.json") for f in fl)
        except Exception as e:                                # noqa: BLE001
            rec["probe2_list_repo_files"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["REACHABLE"] = (rec.get("probe1_model_info") == "OK"
                            or rec.get("probe2_list_repo_files") == "OK")
        out["probes"][repo] = rec
        print(f"  {repo:52s} p1={rec.get('probe1_model_info','')[:38]:38s} "
              f"p2={rec.get('probe2_list_repo_files','')[:28]:28s} "
              f"REACHABLE={rec['REACHABLE']}", flush=True)

    # --- what is installed locally, because weights are not the only gate ---
    loc = {}
    for mod in ("transformers", "timm", "safetensors", "torch"):
        try:
            m = __import__(mod)
            loc[mod] = getattr(m, "__version__", "?")
        except Exception as e:                                # noqa: BLE001
            loc[mod] = f"ABSENT ({type(e).__name__})"
    out["local_packages"] = loc
    out["grid_parity_note"] = (
        "our encoder: 256x640 px, patch 16 -> 16x40 = 640 tokens "
        "(sp1 meta token_grid). Only a /16 backbone reproduces it.")
    print(json.dumps(loc), flush=True)
    dst = Path(__file__).resolve().parents[1] / "raw" / "dino_availability.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), "utf-8")
    print(f"[probe] wrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
