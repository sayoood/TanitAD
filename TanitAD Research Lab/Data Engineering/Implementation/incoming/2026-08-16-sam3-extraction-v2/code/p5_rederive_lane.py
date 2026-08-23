"""STEP 5 — re-derive the ego-lane block from banked primitives. ZERO GPU, dev box.

⭐ WHY THIS IS POSSIBLE AT ALL, AND WHY IT IS THE POINT OF BANKING PRIMITIVES.
`derive_ego_lane` is a PURE FUNCTION of the scene detections already in each
record. When its definition changed mid-run — `lane_idx_est` was being counted
from the LEFT, and `ego_lane_idx`'s own spec says *"0-based lane index from the
right"* — the corpus did not need re-detecting. It needed re-deriving, from the
masks it already carries, on a laptop, in the time an HTTP round-trip takes.

⛔ CONTRAST WITH `confidence_threshold`, WHICH IS THE OPPOSITE CASE AND IS WHY
THE 115 HAD TO BE RE-RUN AT ALL: that filter runs INSIDE the vendor forward
pass, so what it removed is not in the record and no amount of re-analysis
brings it back. ⇒ **Bank the primitive, derive the rest at read time** — the
same rule that deleted the stored `live` boolean (a cache of a rule that
changed) and the same rule that put the RLE mask next to the lossy contour.

⚠️ It rewrites `ego_lane` and NOTHING else, and it refuses any record whose
schema or detection floor is not the one it was told to expect — a re-derivation
that silently touched a v1 record would mix the corpus it exists to keep clean.

⛔ ONE COMMIT FOR THE WHOLE CORPUS, NOT ONE PER RECORD. MEASURED 2026-08-16:
the first version pushed per file and died at
`429 … you have exceeded the rate limit for repository commits (128 per hour)`
— the 115-clip run had already spent 116 of them. A per-file loop is also what
leaves a corpus HALF-CONVERTED when it fails, which is the exact state this
script exists to eliminate. `create_commit` with N `CommitOperationAdd`s is one
commit, atomic, and unaffected by the limit.

⚠️ **Idempotent by construction**: the target state is a pure function of each
record's own banked scene detections, so a record already in the target state
compares equal and is skipped. Re-running after ANY partial failure converges.

usage:  python p5_rederive_lane.py [--dry-run] [--prefix sam3_backfill_v2/]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

REPO_ROOT = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
DS = "Sayood/tanitad-ph0-aug120"


def token() -> str:
    with open(os.path.join(REPO_ROOT, "Keys.txt"), encoding="utf-8",
              errors="replace") as fh:
        m = re.findall(r"hf_[A-Za-z0-9]+", fh.read())
    if not m:
        raise SystemExit("no HF token in Keys.txt")
    return max(m, key=len)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("p5_rederive_lane")
    ap.add_argument("--prefix", default="sam3_backfill_v2/")
    ap.add_argument("--schema", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "raw", "p5_rederive_lane.json"))
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(REPO_ROOT, "stack", "scripts"))
    import ph0_sam3
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    api = HfApi(token=tok)
    far = [f.rfilename for f in api.dataset_info(DS, files_metadata=True
                                                 ).siblings
           if f.rfilename.startswith(a.prefix)
           and f.rfilename.endswith(".json") and "/_runs/" not in f.rfilename]
    print(f"[p5] {len(far)} records under {a.prefix}")

    stat = {"n": len(far), "rewritten": 0, "unchanged": 0, "skipped": 0,
            "bounded_before": 0, "bounded_after": 0, "frames": 0,
            "skipped_clips": []}
    pending: list[tuple[str, bytes]] = []
    for i, rf in enumerate(sorted(far)):
        rec = json.load(open(hf_hub_download(DS, rf, repo_type="dataset",
                                             token=tok, force_download=True),
                             encoding="utf-8"))
        eng = rec.get("engine") or {}
        if (int(rec.get("schema_version") or 0) < a.schema
                or eng.get("confidence_threshold") != a.conf):
            stat["skipped"] += 1
            stat["skipped_clips"].append(rec.get("clip_id"))
            continue
        W, H = rec["frame_wh"]
        old = (rec.get("ego_lane") or {}).get("frames") or {}
        new = {}
        for fk, fr in (rec.get("frames") or {}).items():
            new[fk] = ph0_sam3.derive_ego_lane(fr.get("scene") or [], (W, H))
        stat["frames"] += len(new)
        stat["bounded_before"] += sum(
            1 for v in old.values() if v.get("lane_idx_est") is not None)
        stat["bounded_after"] += sum(
            1 for v in new.values() if v.get("lane_idx_est") is not None)
        if new == old:
            stat["unchanged"] += 1
            continue
        rec["ego_lane"] = {"frames": new,
                           "note": "DERIVED, never prompted — see "
                                   "ph0_sam3.derive_ego_lane"}
        pending.append((rf, json.dumps(rec, separators=(",", ":"))
                        .encode("utf-8")))
        stat["rewritten"] += 1
        if (i + 1) % 25 == 0:
            print(f"[p5] {i+1}/{len(far)} read · pending "
                  f"{len(pending)}", flush=True)

    if pending and not a.dry_run:
        from huggingface_hub import CommitOperationAdd
        api.create_commit(
            repo_id=DS, repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=rf,
                                           path_or_fileobj=io.BytesIO(p))
                        for rf, p in pending],
            commit_message=f"p5: re-derive ego_lane on {len(pending)} v2 "
                           "records (lane_idx_est 0-based FROM THE RIGHT, "
                           "matching s2_derive.LANE_CONTEXT_INPUTS)")
        # ⛔ FAR-SIDE VERIFY BY BYTES — never the push log. A sample, because a
        # single commit either landed whole or not at all.
        import random
        for rf, payload in random.Random(0).sample(pending,
                                                   min(5, len(pending))):
            back = open(hf_hub_download(DS, rf, repo_type="dataset",
                                        token=tok, force_download=True),
                        "rb").read()
            if back != payload:
                raise SystemExit(f"FARSIDE VERIFY FAILED for {rf}")
            stat["verified"] = stat.get("verified", 0) + 1

    stat["class"] = "MEASURED"
    stat["dry_run"] = a.dry_run
    stat["bounded_frac_after"] = (round(stat["bounded_after"]
                                        / stat["frames"], 4)
                                  if stat["frames"] else None)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        json.dump(stat, fh, indent=1)
    print(json.dumps(stat, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
