"""Q5 — the 22-token vocabulary against the ACTUAL label pack, by LAYER.

SPEC §4: *"Losses: **v8.1 GT labels**"*, and *"17 of 22 tokens are trainable
(5 masked for lack of negatives); 10 sit under the n = 200 floor."*

⛔ THE PROGRAMME DISTINGUISHES THREE LAYERS and conflating them is a registered
error class (`CLAUDE.md`, the feature-count table). For LABELS the three are:

  L1  PUBLISHED CORPUS  — PhysicalAI-AV's own features. Carries NO tactical
                          behaviour vocabulary at all.
  L2  EPISODE BUILD     — what `physicalai.py` writes per episode.
  L3  AUGMENTED LABEL RELEASE — `s2_labels_v*.jsonl.gz`, one record per CLIP.
                          EVERY count below is L3 and says which FILE.

⛔ AND WITHIN L3 THE **VERSION** IS ITS OWN SCOPE. A census computed on v8.0
is not a census of v8.1, however close they are, and the spec cites v8.1.

⭐ The census is run through `v7_labels.goal_supervision_census` with
`negatives="measured"` — the SAME negative policy the trainer uses — because a
naive presence/absence count gives a different (and wrong) trainable set.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
while ROOT.name != "TanitAD":
    ROOT = ROOT.parent
OUT = HERE.parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)

from tanitad.data import v7_labels as v7l              # noqa: E402
from tanitad.models import vocab_v7 as V               # noqa: E402

DATA = ROOT / "TanitAD Research Lab" / "Data Engineering" / "Implementation" / "incoming"
PACKS = {
    "v7.2_train": DATA / "2026-09-04-v72-label-release" / "raw" / "s2_labels_v7.2_train.jsonl.gz",
    "v7.2_eval": DATA / "2026-09-04-v72-label-release" / "raw" / "s2_labels_v7.2_eval.jsonl.gz",
    "v8.0_train": DATA / "2026-09-10-v8-speed-max-label-release" / "raw" / "s2_labels_v8.0_train.jsonl.gz",
    "v8.0_eval": DATA / "2026-09-10-v8-speed-max-label-release" / "raw" / "s2_labels_v8.0_eval.jsonl.gz",
    # ⛔ the version the SPEC cites. Listed so its ABSENCE is an explicit row
    # rather than a gap between rows.
    "v8.1_train_AS_SPEC_CITES": DATA / "2026-09-01-v8-tacsit-release" / "raw" / "s2_labels_v8.1_train.jsonl.gz",
}
FLOOR = V.GOAL_MIN_N_FOR_METRIC


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def census(path: Path) -> dict:
    if not path.exists():
        # ⛔ absence is reported WITH the directory listing that proves the
        # search reached the right place — "not found" from a wrong path is
        # indistinguishable from "does not exist".
        d = path.parent
        return {"exists": False, "path": str(path),
                "parent_exists": d.exists(),
                "siblings": (sorted(x.name for x in d.iterdir())[:20]
                             if d.exists() else None)}
    labels, manifest = v7l.load_v7_labels(str(path))
    cen = v7l.goal_supervision_census(labels)       # negatives="measured"
    rows = {}
    for tok in V.TACTICAL_GOAL_TOKENS_V7:
        c = cen.get(tok) or {}
        pos, neg = int(c.get("pos") or 0), int(c.get("neg") or 0)
        rows[tok] = {"pos": pos, "neg": neg,
                     "ignored": int(c.get("ignored") or 0),
                     "supervised_negative": bool(c.get("supervised_negative")),
                     "trainable": bool(pos > 0 and neg > 0),
                     "under_floor": bool(pos < FLOOR)}
    trainable = [t for t, r in rows.items() if r["trainable"]]
    masked = [t for t, r in rows.items() if not r["trainable"]]
    under = [t for t, r in rows.items() if r["under_floor"]]
    both = [t for t in trainable if t not in under]
    return {
        "exists": True, "path": str(path), "md5": _md5(path),
        "manifest": {k: getattr(manifest, k, None) for k in
                     ("schema_version", "vocab_version", "md5", "n_records",
                      "split", "source")},
        "n_records": len(labels),
        "n_tokens_in_vocab": len(V.TACTICAL_GOAL_TOKENS_V7),
        "n_trainable": len(trainable), "trainable": sorted(trainable),
        "n_masked": len(masked), "masked": sorted(masked),
        "floor": FLOOR, "n_under_floor": len(under),
        "under_floor": sorted(under),
        "n_trainable_AND_scoreable": len(both),
        "trainable_AND_scoreable": sorted(both),
        "per_token": rows,
        "UNDERPOWERED_constant_matches_census": (
            sorted(under) == sorted(V.TACTICAL_GOAL_UNDERPOWERED)),
        "TACTICAL_GOAL_UNDERPOWERED_constant": sorted(
            V.TACTICAL_GOAL_UNDERPOWERED),
    }


def main():
    res = {
        "LAYER": "L3 augmented label release (one record per CLIP)",
        "vocab_module": V.__file__,
        "vocab_tokens": list(V.TACTICAL_GOAL_TOKENS_V7),
        "n_vocab_tokens": len(V.TACTICAL_GOAL_TOKENS_V7),
        "n_lat_actions": len(V.TACTICAL_LAT_ACTIONS_V7),
        "n_lon_actions": len(V.TACTICAL_LON_ACTIONS_V7),
        "GOAL_MIN_N_FOR_METRIC": FLOOR,
        "packs": {},
    }
    for name, p in PACKS.items():
        try:
            res["packs"][name] = census(p)
        except Exception as e:
            res["packs"][name] = {"ERROR": f"{type(e).__name__}: {e}"}
    p = OUT / "q5_vocab_census.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(f"vocab: {res['n_vocab_tokens']} goal / {res['n_lat_actions']} lat /"
          f" {res['n_lon_actions']} lon | floor n={FLOOR}")
    for name, c in res["packs"].items():
        if not c.get("exists"):
            print(f"\n== {name}: ABSENT  ({c.get('path')})")
            print(f"   parent_exists={c.get('parent_exists')} "
                  f"siblings={c.get('siblings')}")
            continue
        print(f"\n== {name}  n={c['n_records']}  md5={c['md5'][:12]}")
        print(f"   trainable {c['n_trainable']}/22   masked {c['n_masked']}"
              f"   under floor {c['n_under_floor']}"
              f"   trainable AND scoreable {c['n_trainable_AND_scoreable']}")
        print(f"   masked: {c['masked']}")
        print(f"   UNDERPOWERED constant reproduces census: "
              f"{c['UNDERPOWERED_constant_matches_census']}")
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
