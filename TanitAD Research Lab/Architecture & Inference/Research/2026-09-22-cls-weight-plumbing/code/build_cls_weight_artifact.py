"""Build a class-weight artifact FROM a census JSON. The producer the first one never had.

⛔ WHY THIS FILE EXISTS. `agent_cls_weights_train2400.json` was written by hand in a heredoc, and
its `_self_digest_sha256_of_weights` was a string I typed — no code could reproduce it
(`RETR-2026-09-22-SELF-ATTESTING-DIGEST`). An artifact with no producer cannot be rebuilt,
audited, or diffed against its source; this one can.

⛔ AND IT DECLARES ITS CORPUS LINE, because the second defect was worse than the first: the vector
was counted on the PARITY line and reported as refcv6 readiness, while the parity join overlaps
refcv6's corpus by **4.09 %** (`RETR-2026-09-22-WRONG-CORPUS-FOR-REFCV6`). The line is written
into the field `load_cls_class_weight` REFUSES on.

⭐ The digest is computed by `agent_slots.cls_weight_digest` -- the SAME function the loader
verifies with, so producer and verifier cannot drift. That is deliberate and is NOT the
"a check that shares the defect it checks for" failure: the shared thing here is a pure,
independently-tested FUNCTION, not a hand-typed value. `cls_weight_digest` has its own mutation
arms (S2: ignore class names; S3: skip verification), so the function itself is pinned.

⛔ CPU only.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/Projects/TanitAD/stack")

import torch                                                       # noqa: E402
from tanitad.models import agent_slots as A                        # noqa: E402

OUT_DIR = pathlib.Path("D:/Projects/TanitAD/stack/tanitad/data")


def build(census_path: str, out_name: str, corpus_line: str, source: dict,
          ruling: str) -> dict:
    cen = json.loads(pathlib.Path(census_path).read_text(encoding="utf-8"))
    counts = {c: int(cen["counts"][c]) for c in A.AGENT_CLASSES}
    weights = {c: float(v) for c, v in cen["WEIGHT_VECTOR_inv_freq_mean1"].items()}
    missing = [c for c in A.AGENT_CLASSES if c not in weights]
    if missing:
        raise SystemExit(f"census does not weight {missing}")

    # ⛔ ROUND ONCE, HERE, and digest the ROUNDED values -- because the rounded values are what
    # the file carries and what the loader will rebuild. Digesting the unrounded ones would make
    # the stated digest unreproducible from the artifact, which is the original defect.
    w6 = {c: round(weights[c], 6) for c in A.AGENT_CLASSES}
    vec = torch.tensor([w6[c] for c in A.AGENT_CLASSES], dtype=torch.float32)
    digest = A.cls_weight_digest(vec)

    tot = sum(counts.values())
    art = {
        "_what": ("inverse-frequency class weights for the agent/box slot head cls term "
                  "(H-BOXCLS-1)"),
        "_evidence_class": "MEASURED (ours), CPU, read-only over the named agent join",
        "_ruling": ruling,
        "corpus_line": corpus_line,
        "_corpus_line_note": (
            "\u26d4 LOAD-BEARING. `load_cls_class_weight` REFUSES a vector whose declared line "
            "does not match the arm's, and refuses an artifact that declares none. MEASURED "
            "2026-09-22: the parity and v7-B1 lines share only 4.09 % of their clips, so their "
            "frequency vectors are NOT interchangeable (max weight ratio 1.857x on `rider`)."),
        "_source": source,
        "_producer": ("qland/work/pbox/build_cls_weight_artifact.py -- this file HAS a producer, "
                      "unlike the first one; see RETR-2026-09-22-SELF-ATTESTING-DIGEST"),
        "_normalisation": ("inverse frequency, normalised to MEAN 1 -- scale is absorbed by "
                           "slot_set_loss's weight-following denominator, so only the RELATIVE "
                           "emphasis is expressed"),
        "_out_of_vocabulary": dict(
            cen.get("unknown_classes") or {},
            _handling=("targets_from_join maps these to -1 and slot_set_loss masks ok = ct >= 0, "
                       "so they are EXCLUDED from the cls term, never relabelled")),
        "counts": counts,
        "share": {c: round(counts[c] / tot, 6) for c in A.AGENT_CLASSES},
        "imbalance_majority_to_rarest": cen["imbalance_majority_to_rarest"],
        "weights_inv_freq_mean1": w6,
        "_digest_recipe": ('sha256("|".join(f"{class}={weight:.6f}" for class in AGENT_CLASSES '
                           'order))[:16] -- tanitad.models.agent_slots.cls_weight_digest; '
                           "VERIFIED on every load"),
        "_self_digest_sha256_of_weights": digest,
    }
    p = OUT_DIR / out_name
    p.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {p}  ({p.stat().st_size:,} B)  digest {digest}  imbalance "
          f"{art['imbalance_majority_to_rarest']}:1")
    return art


def main() -> int:
    art = build(
        census_path="C:/Users/Admin/qland/work/pbox/b1_cls_census.json",
        out_name=A.CLS_WEIGHTS_B1,
        corpus_line=A.CORPUS_LINE_B1,
        source={"join": "a40-rescue/b1_train_plus_eval_agents.jsonl.xz",
                "n_clips": 4566, "n_frames": 875657, "n_boxes_in_vocab": 28929493,
                "v7_corpus_coverage": "4,566 / 4,719 = 96.76 %, 0 clips outside",
                "_identity": ("these three numbers reproduce the coverage pass exactly; the "
                              "census ABORTS if they do not")},
        ruling=("H-BOXCLS-1 for the v7/B1 line refcv6 trains on. Supersedes the train2400 "
                "vector for refcv6; that vector remains correct for a PARITY-line arm."))
    # \u2b50 the round trip through the real loader, immediately: an artifact that cannot be
    # loaded by the code that will load it is not an artifact.
    vec, stamp = A.load_cls_class_weight(A.CLS_WEIGHTS_B1,
                                         expect_corpus_line=A.CORPUS_LINE_B1)
    print(f"round trip OK: shape {tuple(vec.shape)} mean {float(vec.mean()):.6f} "
          f"digest {stamp['digest']} line {stamp['corpus_line']}")
    assert stamp["digest"] == art["_self_digest_sha256_of_weights"]
    # \u26d4 and the REFUSAL must fire on the wrong line -- a guard never shown to fire is not one.
    try:
        A.load_cls_class_weight(A.CLS_WEIGHTS_B1,
                                expect_corpus_line=A.CORPUS_LINE_PARITY)
        print("ZZABORT the line check did NOT refuse a mismatched line")
        return 4
    except SystemExit as e:
        print(f"refusal control fired: {str(e)[:110]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
