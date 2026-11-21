"""Set up S0 (PREREG_V7_SEED_POS §6): E-SEED-2 re-run at the v7F LAUNCH geometry, ViT-B/16.

Copies the E-SEED-2 code verbatim from tip b3f7ea6f, then applies ONLY these patches:
  panel : DINO_SNAP + MODEL_ID  vitl16 -> vitb16 (local HF cache snapshot, no download)
          EncoderConfig          1024x24x16 -> 768x12x12 (B/16)
          feature buffers        1024-wide  -> D = cfg.d_model
          torch.manual_seed(0)   before every model build (the L/16 run left the random-init
                                 arms -- scratch, and the random `pos` of seed_asis/seed_imnet --
                                 unseeded, so they were not reproducible run to run)
  rff   : ASSETS -> HERE (the tip copies of panel_kfold / rangeprobe_rff are code-identical to
          the rescued ones E-SEED-2c imported; only a 13-line rescue header differs)
Every patch asserts its anchor exists exactly once; the patched file must still parse.
"""
from __future__ import annotations

import ast
import hashlib
import shutil
import sys
from pathlib import Path

SNAP = Path("C:/Users/Admin/tipsnap/b3f7ea6f")
PK = SNAP / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-v7-seed-and-external-target/code"
TOOLS = SNAP / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-instrument-repair/tools"
W = Path("C:/Users/Admin/s0_b16/code")
B16_SNAP = ("C:\\Users\\Admin\\.cache\\huggingface\\hub"
            "\\models--facebook--dinov3-vitb16-pretrain-lvd1689m\\snapshots"
            "\\5931719e67bbdb9737e363e781fb0c67687896bc")

(W / "clone_scripts").mkdir(parents=True, exist_ok=True)
(W / "eseed2").mkdir(exist_ok=True)
for f in ("e_seed2_bank.py", "e_seed2_panel.py", "e_seed2_relabel.py", "e_seed2_rff.py",
          "e_seed2_table.py"):
    shutil.copy2(PK / f, W / f)
for f in ("panel_kfold.py", "rangeprobe_rff.py"):
    shutil.copy2(TOOLS / f, W / f)
shutil.copy2(SNAP / "stack/scripts/dinov3_seed_checkpoint.py", W / "clone_scripts/dinov3_seed_checkpoint.py")


def patch(path: Path, pairs):
    s = path.read_text(encoding="utf-8")
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit("anchor count %d (want 1) in %s: %r" % (n, path.name, old[:80]))
        s = s.replace(old, new, 1)
    ast.parse(s)
    path.write_text(s, encoding="utf-8")


patch(W / "e_seed2_panel.py", [
    ('r"\\models--facebook--dinov3-vitl16-pretrain-lvd1689m\\snapshots"\n'
     '    r"\\ea8dc2863c51be0a264bab82070e3e8836b02d51")',
     'r"\\models--facebook--dinov3-vitb16-pretrain-lvd1689m\\snapshots"\n'
     '    r"\\5931719e67bbdb9737e363e781fb0c67687896bc")  # S0: ViT-B/16'),
    ('MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"',
     'MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # S0: ViT-B/16'),
    ("patch_size=16, d_model=1024, depth=24, n_heads=16)",
     "patch_size=16, d_model=768, depth=12, n_heads=12)  # S0: ViT-B/16"),
    ("    cfg = make_enc_cfg()\n    state, hfcfg, _prov = S.load_source(DINO_SNAP)",
     "    cfg = make_enc_cfg()\n    torch.manual_seed(0)  # S0: deterministic `pos` init\n"
     "    state, hfcfg, _prov = S.load_source(DINO_SNAP)"),
    ("    return ViTEncoder(make_enc_cfg()).eval()",
     "    torch.manual_seed(0)  # S0: deterministic scratch init\n"
     "    return ViTEncoder(make_enc_cfg()).eval()"),
    ("    G = np.empty((n, 1024), dtype=np.float32)\n    S_ = np.empty((n, 16 * 1024), dtype=np.float32)",
     "    D = make_enc_cfg().d_model  # S0: 768 at B/16\n"
     "    G = np.empty((n, D), dtype=np.float32)\n    S_ = np.empty((n, 16 * D), dtype=np.float32)"),
])
patch(W / "e_seed2_rff.py", [
    ('ASSETS = Path(r"C:\\Users\\Admin\\tanitad-caches\\mm-e19-assets-20260901")',
     'ASSETS = Path(__file__).resolve().parent  # S0: tip copies, code-identical to the rescued ones'),
])

for f in sorted(W.rglob("*.py")):
    print("%-44s md5 %s" % (f.relative_to(W), hashlib.md5(f.read_bytes()).hexdigest()[:12]))
print("S0 SETUP OK")
