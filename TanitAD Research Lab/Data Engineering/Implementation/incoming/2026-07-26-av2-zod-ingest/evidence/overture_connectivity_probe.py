"""Byte-verify Overture Maps' transportation LANE/ROAD CONNECTIVITY — anonymously.

P2 in the standing pre-registration demands the EDGE RELATION byte-verified, not
"the docs say it is routable". This probe reads the parquet FOOTER over HTTP range
requests and then a single column-projected slice, so it proves connectivity while
moving a few MB instead of the ~14 GB theme.

Access is unauthenticated: `s3://overturemaps-us-west-2` answers unsigned
ListObjectsV2 and ranged GETs. No account, no token, no Terms click.

DEV-BOX TRAP: stdlib/certifi TLS fails behind this box's intercepting proxy;
`truststore.inject_into_ssl()` is required (the Python sibling of curl's
`--ssl-no-revoke`). Without it every probe here reads as a network outage.

Usage:  python overture_connectivity_probe.py [release] [out.json]
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request

try:                                            # dev-box TLS proxy
    import truststore
    truststore.inject_into_ssl()
except Exception:                               # noqa: BLE001  pragma: no cover
    pass

BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"
DEFAULT_RELEASE = "2026-07-22.0"


class HttpRangeFile:
    """Minimal random-access file over HTTP Range — enough for pyarrow.parquet."""

    def __init__(self, url: str):
        self.url = url
        self._pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])
        self.bytes_fetched = 0
        self.n_requests = 0

    # -- file protocol ---------------------------------------------------- #
    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        self._pos = (off if whence == 0 else
                     self._pos + off if whence == 1 else self.size + off)
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        if n == 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + n, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        self.n_requests += 1
        self.bytes_fetched += len(data)
        self._pos += len(data)
        return data

    def close(self):
        pass

    @property
    def closed(self):
        return False


def s3_list(prefix: str, max_keys: int = 5):
    url = (f"{BUCKET}?list-type=2&prefix={prefix.replace('/', '%2F').replace('=', '%3D')}"
           f"&max-keys={max_keys}")
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read().decode()
    out = []
    for blk in re.findall(r"<Contents>(.*?)</Contents>", body, re.S):
        k = re.search(r"<Key>([^<]+)</Key>", blk)
        s = re.search(r"<Size>(\d+)</Size>", blk)
        if k:
            out.append((k.group(1), int(s.group(1)) if s else None))
    return out


def probe(release: str = DEFAULT_RELEASE) -> dict:
    import pyarrow.parquet as pq

    res: dict = {"release": release, "bucket": "s3://overturemaps-us-west-2",
                 "access": "anonymous unsigned HTTPS (no account/token/Terms click)",
                 "types": {}}
    for typ in ("segment", "connector"):
        pre = f"release/{release}/theme=transportation/type={typ}/"
        files = s3_list(pre, max_keys=5)
        if not files:
            res["types"][typ] = {"error": "no objects listed", "prefix": pre}
            continue
        key, size = files[0]
        f = HttpRangeFile(f"{BUCKET}/{key}")
        pf = pq.ParquetFile(f)
        sch = pf.schema_arrow
        entry = {
            "probe_file": key,
            "probe_file_bytes": size,
            "n_files_listed_sample": len(files),
            "n_rows_in_file": pf.metadata.num_rows,
            "n_row_groups": pf.metadata.num_row_groups,
            "columns": [n for n in sch.names],
            "schema": {n: str(sch.field(n).type)[:400] for n in sch.names},
        }
        # --- the edge relation, read for real -------------------------------
        if typ == "segment" and "connectors" in sch.names:
            cols = [c for c in ("id", "connectors", "class", "subtype") if c in sch.names]
            tbl = pf.read_row_group(0, columns=cols)
            n = min(20000, tbl.num_rows)
            conn = tbl.column("connectors").to_pylist()[:n]
            deg = [len(c or []) for c in conn]
            entry["edge_relation"] = {
                "field": "connectors",
                "sampled_rows": n,
                "rows_with_at_least_one_connector": sum(1 for d in deg if d >= 1),
                "rows_with_at_least_two_connectors": sum(1 for d in deg if d >= 2),
                "total_connector_refs": sum(deg),
                "max_connectors_on_one_segment": max(deg) if deg else 0,
                "example_connector_struct": (conn[0][0] if conn and conn[0] else None),
            }
            if "subtype" in cols:
                st: dict = {}
                for v in tbl.column("subtype").to_pylist()[:n]:
                    st[v] = st.get(v, 0) + 1
                entry["subtype_counts_sampled"] = st
            if "class" in cols:
                cl: dict = {}
                for v in tbl.column("class").to_pylist()[:n]:
                    cl[v] = cl.get(v, 0) + 1
                entry["class_counts_sampled"] = dict(
                    sorted(cl.items(), key=lambda kv: -kv[1])[:15])
        entry["bytes_fetched_for_this_probe"] = f.bytes_fetched
        entry["http_range_requests"] = f.n_requests
        res["types"][typ] = entry
    return res


if __name__ == "__main__":
    rel = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RELEASE
    out = sys.argv[2] if len(sys.argv) > 2 else "overture_connectivity_probe.json"
    r = probe(rel)
    json.dump(r, open(out, "w"), indent=2, default=str)
    print(json.dumps(r, indent=2, default=str)[:4000])
