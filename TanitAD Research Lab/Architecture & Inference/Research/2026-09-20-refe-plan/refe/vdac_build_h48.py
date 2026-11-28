#!/usr/bin/env python3
"""H48 build (DESIGN_VDAC_P6 3.2, PREREG_VDAC_P6 section 1): the held-out log set of the VDAC drivable heads.
H48 = the 24 logs of PK/splits/sft_heldout_logs.txt  +  for each of SG, BOS, PIT the 8 navtrain logs of that city with the LOWEST sha256(log name utf-8) hex digest
among the label store's logs outside the 24.  City per log = the label store manifest's per-shard `map` field (manifest.json files[shards/<log>.aux1].map, written by the
store build from nuPlan's map names): us-ma-boston = BOS, us-pa-pittsburgh-hazelwood = PIT, sg-one-north = SG, us-nv-las-vegas-strip = LV.  Frames per log = the shard's `n`.
CPU only; reads the store's manifest.json (the pinned store sha256 4a17db67... is asserted) and the 24-log file; writes h48_logs.txt (one log per line, sorted, LF) and its sha256.
Usage: build_h48.py --store <label_store dir> --sft24 <sft_heldout_logs.txt> --out <dir>
"""
import argparse
import hashlib
import json
import os
import sys

PINNED_STORE_SHA256 = "4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a"
CITY_OF_MAP = {"us-nv-las-vegas-strip": "LV", "us-ma-boston": "BOS", "us-pa-pittsburgh-hazelwood": "PIT", "sg-one-north": "SG"}
PER_CITY = 8
EXTRA_CITIES = ("SG", "BOS", "PIT")


def log_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def read_sft24(path: str) -> list:
    raw = open(path, "rb").read()
    names = [x.strip() for x in raw.decode("utf-8").splitlines() if x.strip()]
    if len(names) != 24 or len(set(names)) != 24:
        raise SystemExit(f"REFUSED: {path} holds {len(names)} names ({len(set(names))} distinct), the card says 24")
    return names


def store_logs(store: str, pinned: str = PINNED_STORE_SHA256) -> dict:
    man = json.load(open(os.path.join(store, "manifest.json"), encoding="utf-8"))
    if man.get("store_sha256") != pinned:
        raise SystemExit(f"REFUSED: store_sha256 {man.get('store_sha256')} is not the pinned {pinned}")
    out = {}
    for rel, ent in man["files"].items():
        if not (rel.startswith("shards/") and rel.endswith(".aux1")):
            raise SystemExit(f"REFUSED: unexpected manifest entry {rel}")
        log = rel[len("shards/"):-len(".aux1")]
        if ent["map"] not in CITY_OF_MAP:
            raise SystemExit(f"REFUSED: log {log} has an unknown map {ent['map']!r}")
        out[log] = {"city": CITY_OF_MAP[ent["map"]], "frames": int(ent["n"])}
    if len(out) != man["counts"]["logs"] or sum(v["frames"] for v in out.values()) != man["counts"]["stored_keys"]:
        raise SystemExit("REFUSED: the manifest's per-log entries do not add up to its own counts")
    return out


def build(store: str, sft24_path: str, pinned: str = PINNED_STORE_SHA256) -> dict:
    logs = store_logs(store, pinned)
    sft = read_sft24(sft24_path)
    missing = [x for x in sft if x not in logs]
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} of the 24 SFT held-out logs are not in the label store: {missing[:3]}")
    extra = {}
    for city in EXTRA_CITIES:
        pool = sorted((k for k, v in logs.items() if v["city"] == city and k not in set(sft)), key=lambda k: (log_key(k), k))
        extra[city] = pool[:PER_CITY]
        if len(extra[city]) != PER_CITY:
            raise SystemExit(f"REFUSED: city {city} has only {len(pool)} candidate logs")
    h48 = sorted(set(sft) | {x for v in extra.values() for x in v})
    if len(h48) != 48:
        raise SystemExit(f"REFUSED: H48 holds {len(h48)} logs, not 48")
    per_city = {}
    for k in h48:
        c = per_city.setdefault(logs[k]["city"], {"logs": 0, "frames": 0})
        c["logs"] += 1
        c["frames"] += logs[k]["frames"]
    return {"h48": h48, "sft24": sorted(sft), "extra": extra, "per_city": per_city, "frames": sum(v["frames"] for v in per_city.values()),
            "city_of": {k: logs[k]["city"] for k in h48}, "store_logs": len(logs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--sft24", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = build(a.store, a.sft24)
    os.makedirs(a.out, exist_ok=True)
    body = ("\n".join(r["h48"]) + "\n").encode("utf-8")
    with open(os.path.join(a.out, "h48_logs.txt"), "wb") as f:
        f.write(body)
    sha = hashlib.sha256(body).hexdigest()
    with open(os.path.join(a.out, "h48_logs.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{sha} *h48_logs.txt\n")
    rep = {"h48_sha256": sha, "n_logs": len(r["h48"]), "frames": r["frames"], "per_city": r["per_city"], "sft24_sha256": hashlib.sha256(open(a.sft24, "rb").read()).hexdigest(),
           "extra_logs": r["extra"], "city_source": "label_store/manifest.json files[shards/<log>.aux1].map", "store_sha256": PINNED_STORE_SHA256}
    with open(os.path.join(a.out, "h48_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(json.dumps({k: rep[k] for k in ("h48_sha256", "n_logs", "frames", "per_city")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
