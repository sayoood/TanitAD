"""Strategic level supply (v7/v8.1 label release, `labels/s2_labels_v8_{train,eval}.jsonl.gz`): one record per clip, anchor t0_s = 8.0 s,
strategic band [8, 30] s. The v9 release has NO strategic labels, so Level 1 is trained on rows within +-`window_s` of the anchor and is
IGNORE elsewhere (effectively an [8,12] s decision on 20 s clips; every strategic number is reported with its n)."""
from __future__ import annotations
import gzip, hashlib, json
import numpy as np

ANCHOR_S = 8.0


def load_strategic(jsonl_gz: str) -> dict:
    """sha12(clip_id) -> dict(token, g_token, t0_s). Provenance of each label is stored in the record (geometry vs teacher text)."""
    out = {}
    for line in gzip.open(jsonl_gz, "rt"):
        r = json.loads(line)
        out[hashlib.sha256(r["clip_id"].encode()).hexdigest()[:12]] = dict(token=r["a_str"]["token"], g=r["g_str"]["token"], t0_s=float(r["t0_s"]),
                                                                           prov=r["a_str"].get("provenance"))
    return out


def token_table(records: dict, extra: tuple = ()) -> list[str]:
    """Stable id order: sorted observed tokens (5 usable in the v8.1 corpus) + declared-but-empty placeholders appended."""
    return sorted({v["token"] for v in records.values()}) + list(extra)


def strategic_rows(v9, records: dict, tokens: list[str], window_s: float = 2.0) -> np.ndarray:
    """strat_y per v9 row (-100 = IGNORE). Rows with |now_s - (grid_start_s + t0)| <= window_s receive the clip's anchor label."""
    y = np.full(len(v9["lat_cls_a"]), -100, np.int64); tid = {t: i for i, t in enumerate(tokens)}
    for sha, rec in records.items():
        if sha not in v9.index: continue
        m = v9.clip(sha); r = m["rows"]
        now = v9["now_s"][r]
        sel = np.abs(now - rec["t0_s"]) <= window_s
        y[r.start:r.stop][sel] = tid.get(rec["token"], -100)
    return y
