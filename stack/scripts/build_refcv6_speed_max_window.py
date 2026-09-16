#!/usr/bin/env python
"""REGENERATE the max-speed channel under the PI's containing-window rule (refcv6).

⭐ THE PI, 2026-09-16: *"I think we need to revise and generate the max speed
values in the data set according to my new logic and its ok to derive this from
future ego data but put in the right quantization window."*

THE RULE, restated so a reader needs nothing else:

    (0, 30]  km/h -> 30       (100, 120] km/h -> 120
    (30, 50] km/h -> 50       > 120      km/h -> 120, CLAMPED and COUNTED
    (50, 100] km/h -> 100

It models a max **SET** speed — a limiter a driver sets — so the realised speed
must sit INSIDE the window and never above it. ⛔ That is why it is a
CONTAINING-WINDOW rule and not a nearest-value rule: 96 km/h goes to 100, not to
the nearer-looking 100-vs-120 comparison, and 31 km/h goes to 50, not to 30.

⛔⛔ IT DOES NOT OVERWRITE THE LABEL BLOB. The v8.1 release is a hashed artifact
that other arms resume against; rewriting it in place would silently redefine
every banked ``speed_max_input`` and is the class of change the V8 manifest's
``hash_authority`` block exists to forbid. This script writes a **SIDECAR**:
one JSON-lines record per clip, ``{clip_id, v_hi_ms, bin, limit_kmh, one_hot,
over_ceiling}``, plus a ``*.meta.json`` carrying the source path, the source
md5, the rule, the census and this module's stamp.

USAGE
    python -m scripts.build_refcv6_speed_max_window \\
        --labels  <s2_labels_v8*.jsonl.gz> \\
        --out     <sidecar.jsonl> \\
        [--split train]

⚠️ THE SOURCE BLOB IS NAMED, HASHED AND PRINTED. The refcv6 spec's census was
taken on the v8.1 train blob (md5 ``b45377a1…``), which is **not present on the
dev box** — only the 2026-09-10 v8.0 release is. This script therefore refuses
to guess: it stamps the md5 of whatever it actually read, and the RESULT names
it. Re-run it against the v8.1 blob when that blob is on the box; nothing in the
code changes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tanitad.refs.refcv6_max_speed import (        # noqa: E402
    SPEED_MAX_DERIVATION_V6, SPEED_MAX_STEPS_KMH_V6, N_SPEED_MAX_BINS_V6,
    ladder_census, speed_max_bin)


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open(path: str):
    return (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz")
            else open(path, "rt", encoding="utf-8"))


def build(labels_path: str, out_path: str, split: str = "train") -> dict:
    src_md5 = _md5(labels_path)
    rows: list[dict] = []
    v_hi: list[float] = []
    n = 0
    n_no_band = 0
    no_band_clips: list[str] = []
    for line in _open(labels_path):
        rec = json.loads(line)
        n += 1
        clip = rec.get("clip_id")
        sb = ((rec.get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND")
        v = (sb or {}).get("v_hi_ms")
        if v is None:
            # ⛔ A CLIP WITH NO SPEED BAND GETS NO CEILING AND SAYS SO. It is
            # emitted with `bin = null` and an ALL-ZERO one-hot, which is the
            # "not known" state `speed_max_onehot` defines — never bin 0, which
            # would assert a 30 km/h limiter nobody measured.
            n_no_band += 1
            if len(no_band_clips) < 20:
                no_band_clips.append(clip)
            rows.append({"clip_id": clip, "v_hi_ms": None, "bin": None,
                         "limit_kmh": None,
                         "one_hot": [0.0] * N_SPEED_MAX_BINS_V6,
                         "over_ceiling": None, "valid": 0})
            continue
        v = float(v)
        i, over = speed_max_bin(v)
        oh = [0.0] * N_SPEED_MAX_BINS_V6
        oh[i] = 1.0
        rows.append({"clip_id": clip, "v_hi_ms": round(v, 6), "bin": i,
                     "limit_kmh": int(SPEED_MAX_STEPS_KMH_V6[i]),
                     "one_hot": oh, "over_ceiling": bool(over), "valid": 1})
        v_hi.append(v)

    cen = ladder_census(v_hi) if v_hi else {"n": 0}
    meta = {
        "produced_by": "stack/scripts/build_refcv6_speed_max_window.py",
        "rule": "CONTAINING WINDOW over {30, 50, 100, 120} km/h; > 120 clamps "
                "to 120 and is counted",
        "speed_max_derivation_v6": SPEED_MAX_DERIVATION_V6,
        "source_labels": os.path.basename(labels_path),
        "source_md5": src_md5,
        "split": split,
        "n_clips": n,
        "n_with_speed_band": len(v_hi),
        "n_without_speed_band": n_no_band,
        "clips_without_speed_band_sample": no_band_clips,
        "census": cen,
        "sidecar": os.path.basename(out_path),
        "⛔": ("this is a SIDECAR; the label blob is NOT modified. Join on "
               "clip_id. A clip with no SPEED_BAND carries bin=null and an "
               "all-zero one-hot (the 'not known' state), never bin 0."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    meta["sidecar_md5"] = _md5(out_path)
    return meta


def _print_census(meta: dict) -> None:
    c = meta["census"]
    n = meta["n_clips"]
    print(f"source      : {meta['source_labels']}  md5 {meta['source_md5']}")
    print(f"split       : {meta['split']}")
    print(f"clips       : {n}")
    print(f"with band   : {meta['n_with_speed_band']}")
    print(f"NO band     : {meta['n_without_speed_band']}"
          + (f"   e.g. {meta['clips_without_speed_band_sample'][:3]}"
             if meta["n_without_speed_band"] else ""))
    print()
    print("CONTAINING-WINDOW census over {30, 50, 100, 120} km/h")
    tot = c.get("n", 0)
    for kmh, cnt_, sh in zip(c["steps_kmh"], c["counts"], c["shares"]):
        print(f"  ({'0' if kmh == 30 else c['steps_kmh'][c['steps_kmh'].index(kmh)-1]}"
              f", {kmh}] km/h -> {kmh:>4} : {cnt_:>6}  ({100.0*sh:5.2f} %)")
    # ⚠️ THE CLAMPED CLIPS LIVE IN THE 120 BUCKET. Printing the two counts
    # without saying so invites the reading that they are a fifth group.
    print(f"  ...of which CLAMPED (v_hi > 120 km/h) : {c['over_ceiling']:>6}  "
          f"({100.0*c['over_ceiling_share']:5.2f} % of {tot}) — they are "
          f"INSIDE the 120 bucket above, not a fifth group")
    print(f"  entropy  {c['entropy_bits']} bits   "
          f"slack p50 {c['slack_kmh_p50']} km/h  p95 {c['slack_kmh_p95']} km/h")
    print()
    print(f"sidecar     : {meta['sidecar']}  md5 {meta.get('sidecar_md5')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True,
                    help="s2_labels_v8*.jsonl(.gz) — the SOURCE, never written")
    ap.add_argument("--out", required=True, help="sidecar .jsonl to write")
    ap.add_argument("--split", default="train")
    a = ap.parse_args(argv)
    if os.path.abspath(a.labels) == os.path.abspath(a.out):
        raise SystemExit(
            "[refcv6-vmax] ⛔ --out is the label blob itself. This script "
            "writes a SIDECAR; overwriting a hashed label release in place "
            "silently redefines every arm that resumes against it.")
    meta = build(a.labels, a.out, a.split)
    _print_census(meta)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
