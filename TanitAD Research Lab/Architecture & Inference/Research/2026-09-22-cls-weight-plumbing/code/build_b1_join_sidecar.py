"""Give the B1 agent join the sidecar it never had — built with `join_meta`, not by hand.

⛔ WHY. `refcv5_preflight.py` run against the real data reports:
    [INCONCLUSIVE] join digest declares its scope
      no sidecar at b1_train_plus_eval_agents.jsonl.xz.meta.json -- the digest scope is
      UNDECLARED, and a checker inherited from the other join REFUSES a good file
The parity join has a full sidecar; the join refcv6 actually trains against has none. So the
artifact the programme depends on declares nothing about itself — the same defect shape as the
class-weight vector that declared no corpus line, and the same fix: **declare it, and let the
consumer refuse when it is absent.**

⭐ BUILT WITH THE MODULE, NOT BY HAND. `RETR-2026-09-22-SELF-ATTESTING-DIGEST` is what a
hand-typed attestation costs: `join_meta.file_digest` computes the hash and
`join_meta.declare` writes the block, so producer and consumer share one spelling and the
scope cannot be inferred wrongly.

⛔ WHAT THIS DOES **NOT** DO, deliberately: it does not copy the parity sidecar's `conventions`
block. I have not verified the B1 join's frame convention, its time base, or its occlusion-flag
semantics against its builder. Copying them because the other sidecar has them is exactly how an
INHERITED fact becomes a MEASURED-looking one. The fields below are either MEASURED here or
explicitly marked NOT ESTABLISHED.

⛔ CPU only. Writes ONE new file beside the join; touches the join itself not at all.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/Projects/TanitAD/stack")

from tanitad.data import join_meta as jm                       # noqa: E402
from tanitad.models import agent_slots as A                    # noqa: E402

JOIN = pathlib.Path("C:/Users/Admin/a40-rescue/b1_train_plus_eval_agents.jsonl.xz")
SIDE = pathlib.Path(str(JOIN) + ".meta.json")

# ⛔ LITERALS, each MEASURED with a control earlier today and re-asserted by the census's
# four-way abort. They are written here so the sidecar states facts, not guesses.
N_CLIPS, N_FRAMES, N_BOXES = 4_566, 875_657, 28_958_699
N_BOXES_IN_VOCAB = 28_929_493
OOV_TRAIN_OR_TRAM = 29_206


def main() -> int:
    if not JOIN.exists():
        raise SystemExit(f"join not found: {JOIN}")
    if SIDE.exists():
        print(f"sidecar already exists at {SIDE} — refusing to overwrite")
        return 3

    print(f"hashing {JOIN.name} ({JOIN.stat().st_size:,} B) ...", flush=True)
    dig = jm.file_digest(JOIN, algo="md5")
    print(f"  md5(compressed) = {dig}")

    meta = {
        "_what": "sidecar for the B1 train+eval agent join (the join refcv6 trains against)",
        "_evidence_class": "MEASURED (ours), CPU, read-only over the join itself",
        "_written": "2026-09-22, by qland/work/pbox/build_b1_join_sidecar.py",
        "_why_it_exists": (
            "refcv5_preflight reported the digest scope UNDECLARED and noted that 'a checker "
            "inherited from the other join REFUSES a good file'. The join the programme actually "
            "trains on had no sidecar at all."),
        "corpus_line": A.CORPUS_LINE_B1,
        "_corpus_line_note": (
            "\u26d4 LOAD-BEARING. This join serves the v7/B1 line, NOT the parity line. MEASURED "
            "2026-09-22: it covers 4,566 / 4,719 = 96.76 % of the v7 corpus with 0 clips outside, "
            "while the PARITY join (train2400_agents.jsonl.xz) covers 193 / 4,719 = 4.09 %. The "
            "two are not interchangeable -- see RETR-2026-09-22-WRONG-CORPUS-FOR-REFCV6."),
        "summary": {
            "n_clips": N_CLIPS, "n_frames": N_FRAMES, "n_agent_boxes": N_BOXES,
            "n_agent_boxes_in_AGENT_CLASSES": N_BOXES_IN_VOCAB,
            "out_of_vocabulary": {"train_or_tram_car": OOV_TRAIN_OR_TRAM,
                                  "_handling": ("targets_from_join maps it to -1 and slot_set_loss "
                                                "masks ok = ct >= 0, so it is EXCLUDED from the "
                                                "cls term, never relabelled")},
            "md5": dig,
        },
        "reader_verify": {
            "n_records": N_FRAMES, "n_clips": N_CLIPS,
            "_control": ("these two reproduce the independent coverage pass; the class census "
                         "ABORTS if a read does not reproduce them, because a partial read yields "
                         "a plausible vector from a partial corpus"),
        },
        "joins_to": {
            "cache": "v2ep-eval139-416x1024cyl",
            "episodes_joined": "139/139 on the STABLE id (1.0000)",
            "on_colliding_legacy_key_only": 0,
            "_measured_by": "stack/scripts/refcv5_preflight.py --agent-join --v2-cache, 2026-09-22",
        },
        "_NOT_ESTABLISHED": {
            "conventions": (
                "\u26d4 The parity join's sidecar declares a frame convention, a time base and "
                "occlusion-flag semantics. NONE of those have been verified for THIS join against "
                "its builder, so none are stated here. Copying them because the sibling sidecar "
                "has them would turn an INHERITED fact into a MEASURED-looking one -- which is "
                "the exact error class this file exists to close. Verify against the builder "
                "before quoting any of them."),
            "builder": "not identified in this pass; the file arrived via a40-rescue",
        },
    }
    meta = jm.attach(
        meta, dig, scope="compressed", filename=JOIN.name, algo="md5",
        declared_by="builder-backfill-MEASURED",
        note=("scope is `compressed`: the md5 above is of the .xz itself. Only the .xz exists "
              "for this join -- there is no decompressed sibling on this box -- so the other "
              "candidate scope is not merely unlikely, it is unavailable."))

    SIDE.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {SIDE} ({SIDE.stat().st_size:,} B)")

    # \u2b50 ROUND TRIP THROUGH THE CONSUMER, immediately -- a sidecar the checker cannot read
    # is not a sidecar. This is the same discipline as the weight artifact's load-back.
    back = json.loads(SIDE.read_text(encoding="utf-8"))
    sc = jm.read_digest_scope(back, where=str(SIDE))
    print(f"read_digest_scope OK: scope={sc.scope} algo={sc.algo} digest={sc.digest}")
    assert sc.digest == dig and sc.scope == "compressed"
    ok = jm.verify(JOIN, back)
    print(f"join_meta.verify -> {ok}")

    # \u26d4 and the REFUSAL must fire on a sidecar with the block removed -- a guard never shown
    # to fire is not a guard.
    stripped = {k: v for k, v in back.items() if k != jm.BLOCK}
    try:
        jm.read_digest_scope(stripped, where="<stripped>")
        print("ZZABORT read_digest_scope accepted a sidecar with NO declaration")
        return 4
    except Exception as e:
        print(f"refusal control fired: {type(e).__name__}: {str(e)[:90]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
