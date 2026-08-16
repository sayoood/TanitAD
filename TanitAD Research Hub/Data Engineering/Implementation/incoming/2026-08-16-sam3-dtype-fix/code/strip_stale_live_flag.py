"""Remove the derived `liveness.live` / `liveness.all_fired` booleans from the
banked SAM3 records — the field was deleted from the schema on 2026-08-16.

⛔ WHY. `live` was a CACHE of a rule that changed mid-corpus (`all(...)` ->
`any(...)`, after clip `24b6948f` returned `road 2 · sky 0` under an underpass
and was flagged dead while carrying 22 real detections). The far-side census
then found that record on disk with **`live: False` contradicting its own
`n_det`** — so any consumer reading the flag (the `aug120_pipeline` batch gate,
the overlay's liveness row, a future re-fuse, a human six months out) would
score a healthy clip as the one dead-engine failure that blocks a PASS.

The counts are the primitive. `ph0_sam3.is_live()` is the only derivation, and
it runs at read time. **A field that cannot be stale beats a field that must be
kept in sync**, so this strips the field rather than correcting it — correcting
it would leave the same trap armed for the next rule change.

No GPU: this rewrites JSON in place, per record, far-side verified by byte
round-trip. Reports the exact set it changed and, separately, the set whose
stored flag DISAGREED with its own counts (the ones that were actively wrong).

    python strip_stale_live_flag.py [--dry-run] [--out report.json]
"""
import argparse
import json
import io
import os
import re
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
KEYS = os.path.join(REPO, "Keys.txt")
DS = "Sayood/tanitad-ph0-aug120"
PREFIX = "sam3_backfill/"
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
from ph0_sam3 import is_live                                     # noqa: E402

DERIVED = ("live", "all_fired")


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("strip_stale_live_flag")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    api = HfApi(token=tok)
    rfs = sorted(f.rfilename for f in api.dataset_info(DS).siblings
                 if f.rfilename.startswith(PREFIX)
                 and f.rfilename.endswith(".json")
                 and "/_runs/" not in f.rfilename)
    print(f"[scan] {len(rfs)} records under {PREFIX}")

    carried, disagreed, stripped, no_control = [], [], [], []
    for i, rf in enumerate(rfs):
        cid = rf[len(PREFIX):-len(".json")]
        rec = json.load(open(hf_hub_download(DS, rf, repo_type="dataset",
                                             token=tok, force_download=True),
                             encoding="utf-8"))
        lv = rec.get("liveness")
        if lv is None:
            no_control.append(cid)
            continue
        present = [k for k in DERIVED if k in lv]
        if not present:
            continue
        carried.append(cid)
        # was the stored flag actually WRONG, not merely redundant?
        if "live" in lv and bool(lv["live"]) != is_live(lv):
            disagreed.append({"clip_id": cid, "stored_live": bool(lv["live"]),
                              "recomputed": is_live(lv),
                              "n_det": lv.get("n_det")})
        if a.dry_run:
            continue
        for k in present:
            lv.pop(k, None)
        payload = json.dumps(rec, indent=1).encode("utf-8")
        api.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=rf,
                        repo_id=DS, repo_type="dataset")
        back = open(hf_hub_download(DS, rf, repo_type="dataset", token=tok,
                                    force_download=True), "rb").read()
        if back != payload:
            raise RuntimeError(f"FARSIDE VERIFY FAILED for {rf}")
        stripped.append(cid)
        if len(stripped) % 20 == 0:
            print(f"[strip] {len(stripped)} rewritten", flush=True)

    out = {"n_records": len(rfs),
           "carried_derived_field": len(carried),
           "stored_flag_disagreed_with_counts": disagreed,
           "n_disagreed": len(disagreed),
           "stripped": len(stripped),
           "records_without_control": len(no_control),
           "dry_run": a.dry_run,
           "evidence_class": "MEASURED"}
    print(json.dumps({k: v for k, v in out.items()
                      if k != "stored_flag_disagreed_with_counts"}, indent=1))
    for d in disagreed:
        print("  DISAGREED", d["clip_id"], "stored", d["stored_live"],
              "recomputed", d["recomputed"], d["n_det"])
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print("wrote", a.out)
    print("STRIP_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
