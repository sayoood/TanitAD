"""PROOF that strata.json does not depend on any label: copy every bank unit and every per-log store with ALL label members REMOVED from the zip, run the strata script on the copies
(`--no-write`, a label path that does not exist) and require the SAME content digest as the one recorded in strata.json.  Writes strata_label_free_proof.json; deletes the copies.
    python strata_label_free_proof.py --strata-dir RUN/strata --scratch RUN/strata_proof
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastneg_lib as L                                                               # noqa: E402
import fastneg_strata as ST                                                           # noqa: E402


def strip(src: str, dst: str) -> list:
    removed = []
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zo:
        for n in zi.namelist():
            if "label" in n.lower():
                removed.append(n)
                continue
            zo.writestr(n, zi.read(n))
    return removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata-dir", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--bank-dir", default=ST.DEFAULT_BANK)
    ap.add_argument("--stores-dir", default=ST.DEFAULT_STORES)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    rec = json.load(open(os.path.join(a.strata_dir, "strata.json"), encoding="utf-8"))
    sb, ss = os.path.join(a.scratch, "bank"), os.path.join(a.scratch, "sets")
    os.makedirs(sb, exist_ok=True)
    os.makedirs(ss, exist_ok=True)
    removed_bank, removed_store = {}, {}
    for n in sorted(os.listdir(a.bank_dir)):
        if n.endswith(".sets.npz"):
            removed_bank[n] = strip(os.path.join(a.bank_dir, n), os.path.join(sb, n))
    shutil.copy(ST.DEFAULT_MANIFEST, os.path.join(sb, "MANIFEST.sha256.json"))
    for n in sorted(os.listdir(a.stores_dir)):
        if n.endswith(".npz"):
            removed_store[n] = strip(os.path.join(a.stores_dir, n), os.path.join(ss, n))
    left = []
    for d in (sb, ss):
        for n in os.listdir(d):
            if n.endswith(".npz"):
                with zipfile.ZipFile(os.path.join(d, n)) as z:
                    left += [(n, m) for m in z.namelist() if "label" in m.lower()]
    if left:
        raise SystemExit(f"label members remain in the copies: {left[:3]}")
    res = ST.main(["--bank-dir", sb, "--stores-dir", ss, "--manifest", os.path.join(sb, "MANIFEST.sha256.json"), "--out-dir", os.path.join(a.scratch, "unused"),
                   "--labels-path", "C:/NO_SUCH_DIR/labels6_MUST_NOT_BE_OPENED.npz", "--no-write"])
    ok = res["content_digest"] == rec["speeds_npz"]["content_digest"]
    out = {"strata_json_content_digest": rec["speeds_npz"]["content_digest"], "stripped_run_content_digest": res["content_digest"], "identical": bool(ok),
           "n_bank_units_stripped": len(removed_bank), "n_stores_stripped": len(removed_store),
           "members_removed_bank": sorted({m for v in removed_bank.values() for m in v}), "members_removed_store": sorted({m for v in removed_store.values() for m in v}),
           "label_members_remaining_in_copies": 0, "members_read_in_stripped_run": res["members_read"],
           "cut_points_equal": res["payload"]["cut_points"] == rec["cut_points"], "rows_per_stratum_equal": res["payload"]["rows_per_stratum"] == rec["rows_per_stratum"]}
    with open(os.path.join(a.strata_dir, "strata_label_free_proof.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    if not a.keep:
        shutil.rmtree(a.scratch, ignore_errors=True)
    print("LABEL_FREE_PROOF identical =", ok)
    print(json.dumps(out, indent=1, sort_keys=True)[:1500])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
