"""Anonymous access probes for the 2026-07-26 corpus sweep — >=2 paths per claim.

RULE BEING OBEYED: "absence found at ONE location is not absence." The Lyft L5
probe once returned `NoSuchBucket` on the first try, which was the WRONG answer —
the second probe found the bucket exists and 403s. So every candidate below is
probed on at least two independent paths/names before anything is written down.

DEV-BOX TRAP, MEASURED: bare curl here fails with CRYPT_E_NO_REVOCATION_CHECK and
reports HTTP=000 — indistinguishable from an outage. `--ssl-no-revoke` is
mandatory; without it EVERY row below reads as a false negative.

⛔ Read-only. Nothing here downloads a corpus, creates an account, submits a form,
accepts terms, or touches a torrent.

Usage: python access_probes_2026-07-26.py [out.json]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time

CURL = ["curl", "-sS", "--ssl-no-revoke", "-m", "45", "-o", "-",
        "-w", "\n<<<HTTP=%{http_code} BYTES=%{size_download}>>>"]

PROBES: list[tuple[str, str, str, str]] = [
    # (corpus, probe_id, url, what this probe decides)
    ("argoverse2", "av2-list-sensor-train",
     "https://s3.amazonaws.com/argoverse?list-type=2&prefix=datasets%2Fav2%2Fsensor%2Ftrain%2F&delimiter=%2F&max-keys=2",
     "anonymous ListBucket on the split we pulled"),
    ("argoverse2", "av2-object-get-map",
     "https://s3.amazonaws.com/argoverse/datasets/av2/sensor/val/02678d04-cc9f-3148-9f95-1ba66347dff9/map/log_map_archive_02678d04-cc9f-3148-9f95-1ba66347dff9____PIT_city_71109.json",
     "anonymous object GET of a real lane graph"),

    ("overture", "overture-list-global-endpoint",
     "https://overturemaps-us-west-2.s3.amazonaws.com?list-type=2&prefix=release%2F&delimiter=%2F&max-keys=10",
     "anonymous ListBucket, global S3 endpoint"),
    ("overture", "overture-list-regional-endpoint",
     "https://s3.us-west-2.amazonaws.com/overturemaps-us-west-2?list-type=2&prefix=release%2F&delimiter=%2F&max-keys=10",
     "anonymous ListBucket, regional endpoint (independent path)"),

    ("zod", "zod-download-page",
     "https://zod.zenseact.com/download/",
     "licensor's own statement of the access procedure"),
    ("zod", "zod-readme-raw",
     "https://raw.githubusercontent.com/zenseact/zod/main/README.md",
     "licensor's own README — independent path to the same claim"),
    ("zod", "zod-license-page",
     "https://zod.zenseact.com/license/",
     "licence AS A DOCUMENT (not a short name)"),

    ("metadrive", "metadrive-license",
     "https://raw.githubusercontent.com/metadriverse/metadrive/main/LICENSE.txt",
     "licence as a document"),
    ("metadrive", "metadrive-pypi",
     "https://pypi.org/pypi/metadrive-simulator/json",
     "distribution channel — pip, no gate"),

    ("interaction", "interaction-site",
     "https://interaction-dataset.com/",
     "access mechanism"),
    ("interaction", "interaction-download",
     "https://interaction-dataset.com/details-and-format",
     "second path to the access mechanism"),

    ("levelxdata", "levelx-site",
     "https://levelxdata.com/highd-dataset/",
     "access mechanism (highD/inD/rounD/exiD family, lanelet2 maps)"),
    ("levelxdata", "levelx-root",
     "https://levelxdata.com/",
     "second path"),

    ("pandaset", "pandaset-hf",
     "https://huggingface.co/api/datasets/georghess/pandaset",
     "is there an ungated host for the CC-BY-4.0 PandaSet"),
    ("pandaset", "pandaset-scale",
     "https://scale.com/open-av-dataset/pandaset",
     "original publisher channel"),

    ("openstreetmap", "osm-copyright",
     "https://www.openstreetmap.org/copyright",
     "licence as a document (the source Overture transportation inherits)"),
    ("openstreetmap", "geofabrik-index",
     "https://download.geofabrik.de/",
     "ungated bulk channel"),
]


def run(url: str) -> dict:
    t0 = time.time()
    r = subprocess.run(CURL + [url], capture_output=True, text=True,
                       errors="replace")
    body = r.stdout or ""
    m = re.search(r"<<<HTTP=(\d+) BYTES=(\d+)>>>", body)
    http = int(m.group(1)) if m else None
    nbytes = int(m.group(2)) if m else None
    body = body[: m.start()] if m else body
    snippet = re.sub(r"\s+", " ", body)[:400]
    return {"http": http, "bytes": nbytes, "elapsed_s": round(time.time() - t0, 2),
            "stderr": (r.stderr or "")[:300], "body_head": snippet}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "access_probes_2026-07-26.json"
    rows = []
    for corpus, pid, url, decides in PROBES:
        res = run(url)
        rows.append({"corpus": corpus, "probe_id": pid, "url": url,
                     "decides": decides, **res})
        print(f"{corpus:14s} {pid:32s} HTTP={res['http']} "
              f"bytes={res['bytes']}", file=sys.stderr)
    doc = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "box": "dev box (Windows), curl --ssl-no-revoke, NO credential of any kind",
        "note": ("HTTP=000 on this box means the TLS-revocation trap, NOT an "
                 "outage. Every probe here uses --ssl-no-revoke."),
        "hard_constraint": ("no third-party mirrors of gated corpora, no scraped "
                            "copies, no torrents, no account creation, no form "
                            "submission, no Terms acceptance"),
        "n_probes": len(rows),
        "per_corpus_probe_counts": {c: sum(1 for r in rows if r["corpus"] == c)
                                    for c in {r["corpus"] for r in rows}},
        "probes": rows,
    }
    json.dump(doc, open(out, "w"), indent=2)
    print(json.dumps({"n_probes": len(rows),
                      "codes": {r["probe_id"]: r["http"] for r in rows}}, indent=2))


if __name__ == "__main__":
    main()
