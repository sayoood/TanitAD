"""One-command, guarded HF push for BOTH TanitDataSet tiers (C and R).

Blocked 2026-07-25: the publishing step was DENIED by the Claude Code permission
classifier, so this script was written but NOT executed. Everything upstream of
the network calls (build integrity, license guard, payload provenance audit) is
MEASURED and PASSED — see NOTE.md and safety_check_C.json.

Run only with Sayed's explicit go:

    python push_tanitdataset.py --verify-only      # re-runs the safety check, no network writes
    python push_tanitdataset.py --tier C
    python push_tanitdataset.py --tier R
    python push_tanitdataset.py --tier both

INVARIANTS THIS SCRIPT ENFORCES (do not remove):
  1. The safety check re-runs and must PASS before ANY network write.
  2. `gated="manual"` is set BEFORE the repo is ever public — there is never a
     window where content is public and ungated.
       - C already exists PRIVATE with content -> gate first, then flip public.
       - R does not exist -> create EMPTY + public, gate the empty repo, THEN upload.
  3. The token is read from Keys.txt in place; never printed, never in argv.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
STACK = REPO / "stack"
KEYS = REPO / "Keys.txt"
ROOT = Path(r"C:/Users/Admin/tanitad-data/tanitdataset")
STAGES = {"C": ROOT / "hf_stage_C", "R": ROOT / "hf_stage_R"}
RIDS = {"C": "Sayood/TanitDataSet-C", "R": "Sayood/TanitDataSet-R"}

sys.path.insert(0, str(STACK))


# ------------------------------------------------------------------ safety -- #
def safety_check(stage: Path) -> dict:
    """Two independent legs. Raises on ANY failure. Returns the report."""
    import numpy as np
    from tanitad.lake.schema import SOURCE_REGISTRY
    from tanitad.lake.view import owned_safe_commercial_view
    from tanitad.lake.license_guard import verify_license_scope

    fails: list[str] = []

    # LEG A — the repo's own export guard over the view that produced the set
    members = owned_safe_commercial_view(ROOT / "lake").resolve()
    n = verify_license_scope(members, allowed_classes={"owned-safe"},
                             require_commercial_ok=True, forbid_share_alike=True,
                             context="push:safety-check")

    # LEG B — payload only: trust nothing but the bytes inside the staged tars
    srcs, classes, names, splits = Counter(), Counter(), Counter(), Counter()
    seen: dict[int, str] = {}
    sha_ok = sha_bad = 0
    split_of: dict[int, str] = {}
    shards = sorted(stage.glob("shards/**/*.tar"))
    for sp in shards:
        path_split = sp.parent.name
        with tarfile.open(sp, "r") as tf:
            blobs, metas = {}, {}
            for ti in tf:
                if not ti.isfile():
                    continue
                key, _, ext = ti.name.partition(".")
                data = tf.extractfile(ti).read()
                if ext == "frames.npy":
                    blobs[key] = data
                elif ext == "meta.json":
                    metas[key] = json.loads(data.decode("utf-8"))
            for key, meta in metas.items():
                eid = int(meta.get("episode_id", key))
                if eid in seen:
                    fails.append(f"duplicate episode id {eid}")
                seen[eid] = sp.name
                srcs[meta.get("source")] += 1
                classes[meta.get("license_class")] += 1
                names[meta.get("license_name")] += 1
                splits[meta.get("split")] += 1
                split_of[eid] = meta.get("split")
                if meta.get("split") != path_split:
                    fails.append(f"ep {eid}: split/path mismatch")
                arr = np.load(io.BytesIO(blobs[key]), allow_pickle=False)
                if hashlib.sha256(arr.tobytes()).hexdigest() == meta.get("sha256"):
                    sha_ok += 1
                else:
                    sha_bad += 1
                    fails.append(f"ep {eid}: sha256 MISMATCH")

    GATED = {s for s, l in SOURCE_REGISTRY.items() if l.license_class == "gated-confidential"}
    REFUSE = {s for s, l in SOURCE_REGISTRY.items() if l.license_class == "refuse"}
    NC = {s for s, l in SOURCE_REGISTRY.items() if l.license_class == "nc-research"}
    SA = {s for s, l in SOURCE_REGISTRY.items() if l.share_alike}
    present = set(srcs)
    for label, bad in (("GATED", GATED), ("REFUSE", REFUSE), ("NC", NC), ("SHARE-ALIKE", SA)):
        if present & bad:
            fails.append(f"{label} source in payload: {sorted(present & bad)}")
    if present != {"comma2k19"}:
        fails.append(f"payload sources not 100% comma2k19: {sorted(present)}")
    if set(classes) != {"owned-safe"}:
        fails.append(f"license classes not all owned-safe: {dict(classes)}")
    if len(shards) != 14:
        fails.append(f"expected 14 shards, found {len(shards)}")
    if len(seen) != 90:
        fails.append(f"expected 90 episodes, found {len(seen)}")
    tr = {e for e, s in split_of.items() if s == "train"}
    va = {e for e, s in split_of.items() if s == "val"}
    if tr & va:
        fails.append(f"train/val id overlap: {sorted(tr & va)}")
    if {int(m['episode_id']) for m in members} != set(seen):
        fails.append("catalog/payload episode-id mismatch")

    rep = {"guard_rows": n, "shards": len(shards), "episodes": len(seen),
           "sources": dict(srcs), "classes": dict(classes), "licenses": dict(names),
           "splits": dict(splits), "sha256_ok": sha_ok, "sha256_bad": sha_bad,
           "failures": fails, "verdict": "ABORT" if fails else "SAFE-TO-PUSH"}
    print(json.dumps(rep, indent=2))
    if fails:
        raise SystemExit("SAFETY CHECK FAILED — refusing to push:\n  " + "\n  ".join(fails))
    return rep


# -------------------------------------------------------------------- push -- #
def _api():
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    tok = re.search(r"hf_[A-Za-z0-9]+", KEYS.read_text(errors="ignore")).group(0)
    return HfApi(token=tok), tok


def push(tier: str) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api, tok = _api()
    rid, stage = RIDS[tier], STAGES[tier]
    safety_check(stage)

    api.create_repo(rid, repo_type="dataset", private=(tier == "C"), exist_ok=True)

    # (2) GATE BEFORE PUBLIC — always, in both orders of existence.
    api.update_repo_settings(rid, repo_type="dataset", gated="manual", token=tok)
    info = api.repo_info(rid, repo_type="dataset", token=tok)
    assert getattr(info, "gated", None) == "manual", "gate not set — ABORT"
    print(f"[{tier}] gated=manual confirmed (private={info.private})")

    # (2b) REVEAL NOW — *after* the gate is provably up, *before* any commit.
    # Why the reorder (MEASURED 2026-07-25): committing while still PRIVATE returns
    #   403 "Private repository storage limit reached"
    # because the account's private bucket is at cap (75.97 GB). Flipping public first
    # both unblocks the commit AND frees ~15.9 GB of private quota. The safety invariant
    # is UNCHANGED — gated="manual" is already asserted above, so the content is never
    # publicly fetchable; only the (already-gated) shards are briefly visible without the
    # card, which the very next commit supplies.
    if info.private:
        api.update_repo_settings(rid, repo_type="dataset", private=False, token=tok)
        chk = api.repo_info(rid, repo_type="dataset", token=tok)
        assert getattr(chk, "gated", None) == "manual", "gate lost on reveal — ABORT"
        assert not chk.private, "reveal did not take — ABORT"
        print(f"[{tier}] revealed public with gate intact (gated={chk.gated})")

    # metadata + card first, so the gate prompt and card are correct on reveal
    small = [p for p in sorted(stage.rglob("*"))
             if p.is_file() and p.suffix != ".tar" and ".cache" not in p.parts]
    ops = [CommitOperationAdd(p.relative_to(stage).as_posix(), str(p)) for p in small]
    # Retire the exporter's auto-generated DATA_CARD.md: HF renders README.md only,
    # so DATA_CARD.md is an invisible second card WITHOUT the honest-limits section.
    # (Preserved in repo:TanitAD Research Hub/Data Engineering/tanitdataset-build-2026-07-22/.)
    existing = {s.rfilename for s in api.repo_info(rid, repo_type="dataset", token=tok).siblings}
    if "DATA_CARD.md" in existing:
        ops.append(CommitOperationDelete("DATA_CARD.md"))
    api.create_commit(rid, repo_type="dataset", operations=ops,
                      commit_message="Dataset card (README.md) + Parquet catalog + provenance/verification")
    print(f"[{tier}] metadata committed: {[o.path_in_repo for o in ops]}")

    # shards — only if the remote does not already hold them byte-identically
    remote = {s.rfilename: (s.lfs or {}).get("sha256") if isinstance(s.lfs, dict)
              else getattr(s.lfs, "sha256", None)
              for s in api.repo_info(rid, repo_type="dataset",
                                     files_metadata=True, token=tok).siblings}
    need = []
    for p in sorted(stage.glob("shards/**/*.tar")):
        rel = p.relative_to(stage).as_posix()
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(8 << 20), b""):
                h.update(c)
        if remote.get(rel) != h.hexdigest():
            need.append(p)
    if need:
        print(f"[{tier}] uploading {len(need)} shard(s)…")
        api.upload_large_folder(repo_id=rid, repo_type="dataset", folder_path=str(stage))
    else:
        print(f"[{tier}] all 14 shards already present remotely, byte-identical — skipped")

    # (3) reveal LAST — the gate is already up
    api.update_repo_settings(rid, repo_type="dataset", private=False, token=tok)
    final = api.repo_info(rid, repo_type="dataset", token=tok)
    print(f"[{tier}] DONE https://huggingface.co/datasets/{rid} "
          f"private={final.private} gated={getattr(final,'gated',None)} "
          f"files={len(final.siblings)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["C", "R", "both"])
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if a.verify_only or not a.tier:
        for t, s in STAGES.items():
            if s.exists():
                print(f"=== {t} ===")
                safety_check(s)
        raise SystemExit(0)
    for t in (["C", "R"] if a.tier == "both" else [a.tier]):
        push(t)
