"""Record the DEPLOYED-VAL40 overlap in the manifest, before any build is armed.

MEASURED 2026-08-30: 6 of the B1 corpus's 4,719 clips are in the deployed val40
(`stack/tanitad/data/deployed_val40_clip_digests.json`, matched by
sha256(clip_id)). Training any of the four consumers on them would contaminate
THE published open-loop statistic — the val40 number would be partly measured on
its own training data. This is the C6/leak family, and it is precisely what
`v2_compressed.py --corpus-role train` refuses to let happen.

⚠️ It is ALSO a known, already-sanctioned case: the builder's own `--help` names
"the Alpamayo 4,472 build (6 deployed-val episodes)" — the same six. The
sanctioned repair is `--exclude-parity-overlap`, which drops them and records the
exclusion in `_geometry.json` instead of refusing.

⚠️ The validation gate's "exclusions: 0" was correct and INCOMPLETE: it is a
DATA-QUALITY verdict (no corrupt clip), not a parity verdict. Both belong in the
ledger, labelled, or a reader concludes the corpus needs no exclusions at all.

🔒 Ids are recorded as sha256 DIGESTS, matching the confidentiality convention of
the source file ("per-clip sha256 ONLY — the ids are gated-confidential").
"""
import hashlib
import json
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
BUNDLE = REL / "tanitad-v7-training-corpus"
VAL = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/tanitad/data"
           "/deployed_val40_clip_digests.json")

val = json.loads(VAL.read_text(encoding="utf-8"))
need = {json.loads(x)["clip_id"] for x in open(
    "C:/Users/Admin/tanitad-wt/_s2build/v7_sup/s2_labels_v7.jsonl",
    encoding="utf-8") if x.strip()}
dig = {hashlib.sha256(c.encode("utf-8")).hexdigest(): c for c in need}
overlap = sorted(set(val["clip_id_digests"]) & set(dig))
assert len(overlap) == 6, f"expected the known 6, found {len(overlap)}"

man = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
man["exclusions"] = {
    "policy": "append-with-reason; never a silent skip",
    "data_quality": {
        "count": 0,
        "basis": "validate_release.py — 4,719/4,719 decode-probed, 0 failures",
    },
    "parity_deployed_val40": {
        "count": len(overlap),
        "clip_id_sha256": overlap,
        "confidentiality": "🔒 digests only — the val40 ids are gated-confidential",
        "source": "stack/tanitad/data/deployed_val40_clip_digests.json "
                  f"(corpus_key {val['corpus_key']}, n_clips {val['n_clips']})",
        "why": ("These clips are in THE deployed val40 — the split that produces "
                "the programme's published open-loop statistic. Training any "
                "consumer on them measures the model partly on its own training "
                "data (C6/leak family)."),
        "required_build_flag": "--corpus-role train --exclude-parity-overlap",
        "effect": "4,719 - 6 = 4,713 clips enter the epcache; the exclusion is "
                  "recorded in the build's _geometry.json.",
        "precedent": "the builder's own --help names the Alpamayo 4,472 build's "
                     "6 deployed-val episodes — the same sanctioned repair.",
    },
}
man["camera"]["note_parity"] = (
    "The camera bank ships all 4,719 clips; the 6 val40-overlap clips are "
    "excluded AT BUILD TIME, not from the dataset — so an eval consumer can "
    "still read them legitimately.")

out = BUNDLE / "MANIFEST.json"
out.write_text(json.dumps(man, indent=1), encoding="utf-8")
print(f"MANIFEST updated — parity exclusions: {len(overlap)} (digests recorded)")
print(f"  build must carry: --corpus-role train --exclude-parity-overlap")
print(f"  MANIFEST sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
