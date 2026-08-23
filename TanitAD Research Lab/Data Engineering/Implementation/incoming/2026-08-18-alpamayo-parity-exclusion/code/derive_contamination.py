#!/usr/bin/env python3
"""Re-derive EVERY number in ALPAMAYO_PARITY_EXCLUSION.md from primary artifacts.

⛔ Nothing here reads a summary, a report or a prose count. Each figure comes from
an id list, a clip index or the committed manifest, and the script writes what it
computed so the doc can never drift from it (C81: where a fact is written twice,
the copies must be checkable against each other).

Run from the repo root with the stack on the path:

    PYTHONUTF8=1 PYTHONPATH=stack python "TanitAD Research Hub/Data Engineering/\
Implementation/incoming/2026-08-18-alpamayo-parity-exclusion/code/\
derive_contamination.py"
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]      # code/ <pkg>/ incoming/ Implementation/ <hub area>/ hub/ REPO
sys.path.insert(0, str(REPO / "stack"))

from tanitad.data import parity                                  # noqa: E402

HUB = REPO / "TanitAD Research Hub"
PILOT = (HUB / "Architecture & Inference" / "Implementation" / "incoming"
         / "2026-08-17-thor-concurrency-pilot")
S2 = (HUB / "Data Engineering" / "Implementation" / "incoming"
      / "2026-08-16-s2-v1-labels")
VAL40 = (HUB / "Architecture & Inference" / "Implementation" / "incoming"
         / "2026-08-18-thor-stranded-rescue" / "rescued_beyond_a11" / "leadwork"
         / "val40_lead_index.json")
VAL40_ANON = (HUB / "Data Engineering" / "Implementation" / "incoming"
              / "2026-08-04-instrument-durability" / "raw"
              / "val40_lead_index_ANON.json")


def lines(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def main() -> int:
    out: dict = {"_evidence_class": "MEASURED (ours; this script)",
                 "_repo": str(REPO)}

    # ---- 1. the parity train clip set, proven against the committed manifest --
    ls = lines(PILOT / "parity_ls.txt")
    ptrain = sorted({n.split()[-1][: -len(parity.V2_SUFFIX)] for n in ls
                     if n.split()[-1].endswith(parity.V2_SUFFIX)})
    cm = parity.clip_membership_of(parity.PARITY_TRAIN_KEY)
    out["parity_train"] = {
        "source": "…/2026-08-17-thor-concurrency-pilot/parity_ls.txt",
        "n_clips": len(ptrain),
        "sha256_sorted": parity.uid_digest(ptrain),
        "manifest_sha256_sorted": cm["clip_id_sha256_sorted"],
        "REPRODUCES_COMMITTED_MANIFEST":
            parity.uid_digest(ptrain) == cm["clip_id_sha256_sorted"],
    }

    # ---- 2. the Alpamayo corpus and the overlap -----------------------------
    alpa = lines(PILOT / "alpamayo_clip_ids.txt")
    banked = lines(PILOT / "alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt")
    derived = parity.clips_in_parity_train(alpa)
    out["alpamayo_x_parity_train"] = {
        "n_alpamayo": len(alpa),
        "n_in_parity_train": len(derived),
        "frac": round(len(derived) / len(alpa), 6),
        "derived_equals_banked_list": derived == sorted(banked),
        "note": "DERIVED from the committed per-clip digest oracle; the banked "
                "list is used only to confirm the derivation, never as input",
    }

    # ---- 3. the S2 label legs — the one place a split is declared today ------
    for leg in ("labels", "review/labels_v2"):
        idx = json.loads((S2 / leg / "clip_index.json").read_text(
            encoding="utf-8"))
        by: dict[str, list[str]] = {}
        for cid, ent in idx["clips"].items():
            by.setdefault(ent.get("label_split", "?"), []).append(cid)
        out.setdefault("s2_label_splits", {})[leg] = {
            k: {"n_clips": len(v),
                "n_in_parity_train": len(parity.clips_in_parity_train(v)),
                "n_in_alpamayo_records": len(set(v) & set(alpa)),
                "n_excluded_in_index": sum(
                    1 for c in v if idx["clips"][c].get("excluded"))}
            for k, v in sorted(by.items())}

    # ---- 4. what a split BUILDABLE TODAY would look like ---------------------
    # Only clips that HAVE w120 video can be built into an eval set at all. The
    # S2 clip index enumerates exactly the built w120 set (201 train-side +
    # 600 val-side = 801), so `alpamayo ∩ index` reproduces aug120_pipeline.py's
    # own "257 have w120 video" from ids rather than from its prose.
    idx = json.loads((S2 / "review/labels_v2" / "clip_index.json").read_text(
        encoding="utf-8"))["clips"]
    have_video = set(idx)
    buildable = sorted(set(alpa) & have_video)
    bad = parity.clips_in_parity_train(buildable)
    out["buildable_eval_split_today"] = {
        "_what": "an Alpamayo eval split can only contain clips that HAVE w120 "
                 "video built; 4472 of the 4729 do not. This is the rate that "
                 "matters for a split someone could build TODAY.",
        "n_alpamayo_with_w120_video": len(buildable),
        "n_in_parity_train": len(bad),
        "frac_contaminated": round(len(bad) / len(buildable), 6),
        "vs_corpus_wide_rate": round(len(parity.clips_in_parity_train(alpa))
                                     / len(alpa), 6),
    }

    # ---- 5. THE OTHER DIRECTION — deployed val inside the Alpamayo corpus ----
    full = json.loads(VAL40.read_text(encoding="utf-8"))
    anon = json.loads(VAL40_ANON.read_text(encoding="utf-8"))
    xcheck = sum(1 for ep, e in full.items()
                 if "clip_" + hashlib.sha256(
                     e["clip_id"].encode()).hexdigest()[:8]
                 == anon[ep]["clip_sha8"])
    v40 = {e["clip_id"] for e in full.values()}
    out["deployed_val_x_alpamayo"] = {
        "n_val_episodes": len(v40),
        "second_source_sha8_agreements": f"{xcheck}/{len(full)}",
        "n_in_alpamayo_records": len(v40 & set(alpa)),
        "frac_of_val_split": round(len(v40 & set(alpa)) / len(v40), 6),
        "n_in_parity_train": len(parity.clips_in_parity_train(v40)),
        "note": "blast radius TODAY is ZERO — no trainer consumes the Alpamayo "
                "labels. The trigger is the 4472-clip build becoming "
                "supervision.",
    }

    print(json.dumps(out, indent=1, ensure_ascii=False))
    (HERE.parent / "raw" / "contamination.json").parent.mkdir(
        parents=True, exist_ok=True)
    (HERE.parent / "raw" / "contamination.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
