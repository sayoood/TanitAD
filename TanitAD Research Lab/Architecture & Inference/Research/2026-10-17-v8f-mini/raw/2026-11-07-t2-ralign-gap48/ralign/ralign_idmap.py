"""RALIGN gate G-IDMAP on pulled box-2 cells (PREREG_T2_RALIGN.md section 4): with a := 1 and b := 0 the wrapped composite reproduces C0C's stage-4 cells bit for bit on 2 scenes. CPU only.

    python -I ralign_idmap.py compare --ra-root <pull5_C0C-RA_ID>/rollouts --stage4-root <stage-4 pull>/rollouts --manifest <the ID package's MANIFEST5.json> --out G_IDMAP.json [--min-scenes 2]

Compares, for every scene both roots hold and each seed, the cells `<scene8>/objects/<T2|RR>/C0C-RA_s<k>` (identity-map run) and `.../C0C_s<k>` (stage 4) record by record, keyed by (scene, f0):
every key of the record is compared EXACTLY (plans, tracks, calls, frame_md5, cl, g, v0, `extra` per tick) except `arm` and `render_ms` (the arm name and a timing), and inside `extra` the keys `arm`
and `ralign` (the wrapper's own stamp). Equal means json-equal on the pulled files: the plans are float64 values printed by Python's repr, which round-trips exactly.
Verdict G-IDMAP-PASS needs: at least --min-scenes scenes compared, every compared cell present on both sides with the same number of records, every record equal, the ID package's manifest has
status IDMAP and identity maps. The verdict file also records the sha256 of the shipped `ralign_policy.py` / `ralign_lib.py` (from the manifest), which `make_ralign_stage5.py build` (status
GATED) requires to equal the current files: edit the policy and the gate must be run again. A perturbed-identity package must FAIL this comparison (tests do that).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ralign_lib as L  # noqa: E402

SKIP_TOP = ("arm", "render_ms")
SKIP_EXTRA = ("arm", "ralign")


def read_records(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def norm(r):
    o = {k: v for k, v in r.items() if k not in SKIP_TOP}
    o["extra"] = [{k: v for k, v in e.items() if k not in SKIP_EXTRA} for e in r["extra"]]
    return o


def compare_cell(ra_file, s4_file):
    a, b = read_records(ra_file), read_records(s4_file)
    key = lambda r: (int(r["scene"]), int(r["f0"]))      # noqa: E731
    a, b = sorted(a, key=key), sorted(b, key=key)
    out = {"n_ra": len(a), "n_stage4": len(b), "keys_equal": [key(x) for x in a] == [key(x) for x in b], "records_equal": 0, "records_differ": [], "tokens_mapped_max": 0}
    if not out["keys_equal"]:
        return out
    for x, y in zip(a, b):
        if json.dumps(norm(x), sort_keys=True) == json.dumps(norm(y), sort_keys=True):
            out["records_equal"] += 1
        else:
            out["records_differ"].append({"scene": x["scene"], "f0": x["f0"], "keys": sorted(k for k in set(norm(x)) | set(norm(y)) if json.dumps(norm(x).get(k)) != json.dumps(norm(y).get(k)))})
        out["tokens_mapped_max"] = max([out["tokens_mapped_max"]] + [int((e.get("ralign") or {}).get("n_token_sets_mapped", 0)) for e in x["extra"]])
    return out


def compare(ra_root, s4_root, manifest, arm="C0C-RA", min_scenes=2):
    man = json.load(open(manifest, encoding="utf-8"))
    blk = (man.get("arm5") or {}).get("ralign") or {}
    cells, scenes, problems = {}, set(), []
    if blk.get("status") != "IDMAP":
        problems.append(f"MANIFEST_NOT_IDMAP status {blk.get('status')!r}: the evidence must come from the identity package")
    for d in sorted(glob.glob(f"{ra_root}/*/objects/*/{arm}_s*/records.jsonl")):
        d = d.replace("\\", "/")
        scene8, tc, a = d.split("/")[-5], d.split("/")[-3], d.split("/")[-2]
        s4 = f"{s4_root}/{scene8}/objects/{tc}/C0C_s{a[-1]}/records.jsonl"
        if not os.path.isfile(s4):
            problems.append(f"NO_STAGE4_CELL {scene8}/{tc}/C0C_s{a[-1]}")
            continue
        r = compare_cell(d, s4)
        cells[f"{scene8}|{tc}|{a}"] = r
        scenes.add(scene8)
        if not r["keys_equal"] or r["n_ra"] != r["n_stage4"] or r["n_ra"] == 0:
            problems.append(f"RECORD_SET_DIFFERS {scene8}|{tc}|{a}: {r['n_ra']} vs {r['n_stage4']} records")
        elif r["records_differ"]:
            problems.append(f"RECORDS_DIFFER {scene8}|{tc}|{a}: {len(r['records_differ'])} of {r['n_ra']} (keys {r['records_differ'][0]['keys']})")
        elif r["tokens_mapped_max"] < 1:
            problems.append(f"MAP_NEVER_RAN {scene8}|{tc}|{a}: no `ralign` stamp with mapped token sets (an identity that proves nothing)")
    if len(scenes) < min_scenes:
        problems.append(f"TOO_FEW_SCENES {len(scenes)} compared, need {min_scenes}")
    verdict = "G-IDMAP-PASS" if not problems else "G-IDMAP-FAIL"
    return {"card": "PREREG_T2_RALIGN.md", "card_sha256": L.CARD_SHA256, "gate": "G-IDMAP", "verdict": verdict, "problems": problems, "n_scenes": len(scenes), "scenes": sorted(scenes), "n_cells": len(cells),
            "n_records_equal": sum(c["records_equal"] for c in cells.values()), "cells": cells, "manifest5_sha256": L.sha256_file(manifest), "maps_status": blk.get("status"),
            "code_sha256": {fn: blk.get("code_sha256", {}).get(fn) for fn in ("ralign_policy.py", "ralign_lib.py")}, "ra_root": ra_root, "stage4_root": s4_root}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ra-root", required=True)
    c.add_argument("--stage4-root", required=True)
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--arm", default="C0C-RA")
    c.add_argument("--min-scenes", type=int, default=2)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"ZZ G-IDMAP-REFUSED output exists {a.out} (move it aside) ZZ")
        return 4
    res = compare(a.ra_root.replace("\\", "/"), a.stage4_root.replace("\\", "/"), a.manifest, a.arm, a.min_scenes)
    L.jdump(res, a.out)
    print(f"ZZ {res['verdict']} scenes {res['n_scenes']} cells {res['n_cells']} equal records {res['n_records_equal']} problems {len(res['problems'])} ZZ")
    return 0 if res["verdict"] == "G-IDMAP-PASS" else 5


if __name__ == "__main__":
    sys.exit(main())
