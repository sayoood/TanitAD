"""Second channel: download the pushed augmentation components back from HF into a fresh folder and verify them with the loader the
dataset ships (tools/augment_loader.py, as downloaded): sha256 of every listed file, the datacard edit additive against the card that
was on the Hub before, and a load of each component for one clip."""
import json, re, sys
from pathlib import Path
import truststore; truststore.inject_into_ssl()
from huggingface_hub import snapshot_download

REPO = "Sayood/tanitad-v7-training-corpus"; BASE = Path(r"D:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915")
ROOT = BASE / "verify_download"
tok = re.search(r"hf_[A-Za-z0-9]+", open(r"D:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
snapshot_download(REPO, repo_type="dataset", token=tok, local_dir=str(ROOT),
                  allow_patterns=["calibration/*", "lidar_bev_gt/*", "agents/*", "alpamayo/vqa_bank_500.json", "tools/augment_loader.py",
                                  "AUG_2026_09_MANIFEST.json", "DATACARD.md", "semantic_maps/SEMANTIC_MAPS_MANIFEST.json"])
sys.path.insert(0, str(ROOT / "tools")); import augment_loader as L
res = {c: L.verify(ROOT, c) for c in ("calibration", "agents", "lidar_bev_gt", "alpamayo")}
NL = "\n"; anchor = "Ground-truth register row: `D-LABEL-GT`." + NL
old = (BASE / "hub_copies" / "DATACARD.md").read_text(encoding="utf-8"); new = (ROOT / "DATACARD.md").read_text(encoding="utf-8")
head, tail = old.split(anchor)
additive = new.startswith(head + anchor) and tail.rstrip(NL) in new and "## Augmentations added 2026-09-15" in new
aug = json.loads((ROOT / "AUG_2026_09_MANIFEST.json").read_text(encoding="utf-8"))
lid = json.loads((ROOT / "lidar_bev_gt" / "LIDAR_BEV_GT_MANIFEST.json").read_text(encoding="utf-8")); clip = next(iter(lid["files"]))
b = L.load_lidar_bev(ROOT, clip); a = L.load_agents(ROOT, clip); i, e = L.load_calibration(ROOT, clip)
print(f"clip {clip[:8]}: lidar {b['cart_occ'].shape}, agents {len(a)} rows, calibration fw_poly_1 {float(i['fw_poly_1']):.3f}")
bad = sum(v[1] for v in res.values()); missing = sum(v[2] for v in res.values())
print("verify:", res, "| card additive:", additive, "| AUG components:", len(aug["components"]))
print(f"ZZHFVERIFY-{int(bad == 0 and missing == 0 and additive)}-{bad}-{missing}ZZ")
