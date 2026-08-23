"""Overture lane-level probe — INDEPENDENT path from the 2026-07-26 av2-zod-ingest agent.

That agent read ONE segment part and reported a 21-column schema with no `lanes`.
This probe re-derives the answer from a DIFFERENT release/part and dumps the FULL
arrow schema verbatim, plus a row sample, so the "no lane-level detail" claim rests
on two independent reads rather than one.

Anonymous: unsigned HTTP range GETs against the foundation's own bucket.
DEV-BOX TRAP: stdlib TLS fails behind this box's intercepting proxy ->
truststore.inject_into_ssl() (the Python sibling of curl --ssl-no-revoke).
"""
from __future__ import annotations

import json
import sys
import urllib.request

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

import pyarrow.parquet as pq

BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"


class HttpRangeFile:
    def __init__(self, url: str):
        self.url = url
        self._pos = 0
        self.bytes_fetched = 0
        self.n_requests = 0
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=180
        ) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def closed_check(self):
        return False

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        if whence == 0:
            self._pos = off
        elif whence == 1:
            self._pos += off
        else:
            self._pos = self.size + off
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        if n == 0:
            return b""
        end = min(self._pos + n, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            buf = r.read()
        self.n_requests += 1
        self.bytes_fetched += len(buf)
        self._pos += len(buf)
        return buf

    def close(self):
        pass

    @property
    def closed(self):
        return False


def main(key: str, out: str, nrows: int = 3000):
    url = f"{BUCKET}/{key}"
    f = HttpRangeFile(url)
    pf = pq.ParquetFile(f)
    schema = pf.schema_arrow
    fields = [{"name": fl.name, "type": str(fl.type)} for fl in schema]

    lane_hits = [
        fl["name"] for fl in fields if "lane" in fl["name"].lower()
    ]
    # deep scan: any nested field name anywhere in the schema mentioning "lane"
    deep = []

    def walk(t, path):
        s = str(t)
        if "lane" in s.lower():
            deep.append(path + " :: " + s[:400])

    for fl in schema:
        walk(fl.type, fl.name)

    # read a small sample of the road-relevant columns
    want = [
        c
        for c in ["id", "subtype", "class", "connectors", "road_flags",
                  "access_restrictions", "destinations", "prohibited_transitions",
                  "speed_limits", "width_rules", "routes", "subclass",
                  "subclass_rules", "level_rules"]
        if c in schema.names
    ]
    tbl = pf.read_row_group(0, columns=want) if pf.num_row_groups else None
    sample = []
    if tbl is not None:
        d = tbl.slice(0, min(nrows, tbl.num_rows)).to_pylist()
        sample = d[:5]

    res = {
        "probe": "overture-lane-level-independent",
        "date": "2026-07-26",
        "url": url,
        "file_bytes": f.size,
        "bytes_fetched": f.bytes_fetched,
        "n_http_requests": f.n_requests,
        "num_rows_in_file": pf.metadata.num_rows,
        "num_row_groups": pf.num_row_groups,
        "n_columns": len(fields),
        "columns": fields,
        "columns_whose_NAME_contains_lane": lane_hits,
        "nested_types_anywhere_containing_lane": deep,
        "parquet_created_by": pf.metadata.created_by,
        "sample_rows_first5": sample,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"columns={len(fields)}  lane_name_hits={lane_hits}  deep_lane_hits={len(deep)}")
    print("names:", ", ".join(schema.names))
    print(f"fetched {f.bytes_fetched/1e6:.2f} MB in {f.n_requests} requests "
          f"(file is {f.size/1e9:.3f} GB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
