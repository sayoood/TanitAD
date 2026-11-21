"""MM binding item 6 (2026-10-04): "apply the 693-frame ego-box mask". The trainer applies it through WP-C's
``--join-defect-masks`` list (keyed by the JOIN's frame_idx); WP-A's corpus manifest lists the same boxes keyed by the
RAW frame k. Two independently mined lists -- this checks they name the SAME frames, and at which offset.

Writes ``raw/mask_crosscheck.json`` (sha12 / frame counts only). CPU, < 1 s.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WPC = Path("C:/Users/Admin/r8_wpb_49a/stack/tanitad/configs/refcv8_join_label_defects.json")      # tip 49a0655
WPA = Path("D:/Projects/TanitAD/FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/raw/"
           "refcv8_corpus_manifest.json")


def main() -> None:
    import hashlib
    c = json.loads(WPC.read_text(encoding="utf-8"))
    m = json.loads(WPA.read_text(encoding="utf-8"))
    ego = c["ego_footprint"]["frames"]
    a = [b for b in m["mask_boxes"] if b["split"] == "train"]
    aset = {(b["sha12"], int(b["k"])) for b in a}
    by_off = {}
    for off in range(-1, 5):
        cset = {(s, int(f) + off) for s, fr in ego.items() for f in fr}
        by_off[off] = {"wpc_frames": len(cset), "both": len(cset & aset), "wpa_only": len(aset - cset),
                       "wpc_only": len(cset - aset)}
    out = {"what": "WP-C ego-footprint list vs WP-A corpus-manifest mask_boxes (train), MEASURED",
           "wpc_file_md5": hashlib.md5(WPC.read_bytes()).hexdigest(),
           "wpa_file_md5": hashlib.md5(WPA.read_bytes()).hexdigest(),
           "wpa_train_boxes": len(a), "wpa_train_frames": len(aset), "wpa_clips": len({b["sha12"] for b in a}),
           "wpc_clips": len(ego), "clip_sets_equal": set(ego) == {b["sha12"] for b in a},
           "by_offset_k_minus_frame_idx": by_off,
           "wpa_other_split_boxes_not_in_any_corpus": sum(1 for b in m["mask_boxes"] if b["split"] != "train")}
    (HERE.parent / "raw" / "mask_crosscheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
