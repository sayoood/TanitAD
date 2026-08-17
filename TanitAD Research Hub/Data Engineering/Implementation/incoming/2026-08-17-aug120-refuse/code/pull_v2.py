"""Pull the v2 SAM3 corpus (sam3_backfill_v2/) to a local dir, then census the
clip coverage against the aug120 todo=201 population.

Token read IN PLACE from Keys.txt — never printed, never argv.
"""
from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sam3_backfill_v2/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args(argv)

    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    api = HfApi(token=tok)
    info = api.dataset_info(DS, files_metadata=True)
    allf = {f.rfilename: f.size for f in info.siblings}
    far = {k: v for k, v in allf.items() if k.startswith(a.prefix)}
    recs = sorted(rf for rf in far
                  if rf.endswith(".json") and "/_runs/" not in rf)
    print(f"[far] prefix={a.prefix}: {len(recs)} records, "
          f"{len(far)-len(recs)} other")
    # what other prefixes exist?
    tops = {}
    for k in allf:
        tops[k.split("/")[0]] = tops.get(k.split("/")[0], 0) + 1
    print("[far] top-level prefixes: " + json.dumps(tops, indent=1))
    if a.list_only:
        json.dump({"prefix": a.prefix, "records": recs,
                   "top_level": tops,
                   "zero_byte": [r for r in recs if far.get(r) == 0],
                   "bytes_total": sum(far.get(r) or 0 for r in recs)},
                  open(os.path.join(os.path.dirname(a.out) or ".",
                                    "v2_listing.json"), "w"), indent=1)
        return 0

    os.makedirs(a.out, exist_ok=True)
    got = 0
    for i, rf in enumerate(recs):
        cid = rf[len(a.prefix):-len(".json")]
        dst = os.path.join(a.out, f"{cid}.json")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            got += 1
            continue
        p = hf_hub_download(DS, rf, repo_type="dataset", token=tok,
                            force_download=True)
        with open(p, "rb") as fh:
            raw = fh.read()
        rec = json.loads(raw.decode("utf-8"))
        if rec.get("clip_id") != cid:
            raise SystemExit(f"{rf} carries clip_id={rec.get('clip_id')!r}")
        with open(dst, "wb") as fh:
            fh.write(raw)
        got += 1
        if (i + 1) % 25 == 0:
            print(f"[far] {i+1}/{len(recs)}", flush=True)
    print(f"PULL_DONE n={got} out={a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
