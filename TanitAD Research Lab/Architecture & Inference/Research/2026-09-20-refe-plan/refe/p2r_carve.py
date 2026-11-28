"""Carve P2's TRAINING on-policy sets out of the restored bank: every row of the 24 SFT held-out logs goes OUT.

Why (MEASURED 2026-10-11 on the P2 pod). PREREG_P2 / PREREG_P2_AMENDMENT_1: P2 trains on the **174,912** served training
sets. The restored bank (`onpolicy_sets_logs_full`, md5 4fc97993...) serves **177,836** = 174,912 + **2,924** in the 24
held-out logs of `splits/sft_heldout_logs.txt` (the list inside HF `refe_heldout_v5.tar.zst`, md5 1f43409b...). It is the
same split SFT-3 logged (`raw/2026-10-04-sft3/sft.log`: train_sets 174,912 of train_sets_in_bank 177,836,
train_rows_in_heldout_logs 0). u400 trained with those logs OUT (scorer_finetune --heldout-logs); train.py has no such
flag, so P2 would have trained on them and both the u400 comparison and the held-out read would have moved.

IN PLACE, so the pre-registered training argv stays byte-identical (`--onpolicy-targets .../onpolicy/sets`): each original
file moves to --stash (a rename, no copy) and `sets/` keeps only the training rows. Re-runnable after any interruption: the
input of every file is its stash copy when one exists, so a half-finished carve is rebuilt, never carved twice.

    python p2r_carve.py --sets <onpolicy/sets> --stash <onpolicy/sets_full> --heldout-logs <txt> --expect 174912 --json <out.json>

Prints `ZZCARVE_OK <served> <dropped rows>`; anything else ends `ZZCARVE_FAIL <reason>` with rc 1.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_bank_subset as SB  # noqa: E402
from onpolicy_relabel_pdm import served_keys  # noqa: E402  -- the relabeller's own definition, so the count cannot drift

PAT = "onpolicy_*.jsonl"


def log_name_of(line):
    """The row's log_name: the relabeller's fast head parse, and the full JSON when that misses (a row the parse cannot
    place must not slip through as 'not held out')."""
    f = SB.head_fields(line[:2000] + line[-400:])
    if "log_name" in f:
        return f["log_name"]
    try:
        return json.loads(line).get("log_name")
    except json.JSONDecodeError:
        return None


def carve(sets, stash, heldout, expect=None):
    """Returns (report dict, None) on success or (report dict, reason) on refusal."""
    ho = set(open(heldout, encoding="utf-8").read().split())
    rep = {"sets": sets, "stash": stash, "heldout_logs": len(ho),
           "heldout_sha256": hashlib.sha256(open(heldout, "rb").read()).hexdigest(), "expect": expect, "files": {}}
    if not ho:
        return rep, "empty_heldout_list"
    os.makedirs(stash, exist_ok=True)
    names = sorted({os.path.basename(p) for p in glob.glob(os.path.join(sets, PAT)) + glob.glob(os.path.join(stash, PAT))})
    if not names:
        return rep, "no_set_files"
    for nm in names:
        src_stash, dst = os.path.join(stash, nm), os.path.join(sets, nm)
        if not os.path.exists(src_stash):
            os.replace(dst, src_stash)                       # the original is kept, whole, beside the carved copy
        part = os.path.join(sets, ".carve_" + nm + ".part")   # a leading dot: no consumer's onpolicy_*.jsonl glob sees it
        kept = dropped = 0
        with open(src_stash, encoding="utf-8") as fi, open(part, "w", encoding="utf-8", newline="") as fo:
            for line in fi:
                if log_name_of(line) in ho:
                    dropped += 1
                    continue
                fo.write(line)
                kept += 1
        os.replace(part, dst)
        rep["files"][nm] = {"kept_rows": kept, "dropped_rows": dropped}
    full = served_keys([os.path.join(stash, n) for n in names])
    train = served_keys([os.path.join(sets, n) for n in names])
    want = {k: v for k, v in full.items() if k[0] not in ho}
    rep.update(served_full=len(full), served_in_heldout_logs=len(full) - len(want), served_train=len(train),
               dropped_rows=sum(f["dropped_rows"] for f in rep["files"].values()))
    if any(k[0] in ho for k in train):
        return rep, "heldout_rows_survived"
    if train != want:
        return rep, f"train_keys_differ_from_full_minus_heldout:{len(train)}!={len(want)}"
    if expect is not None and len(train) != expect:
        return rep, f"served_train_{len(train)}_expected_{expect}"
    return rep, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--stash", required=True)
    ap.add_argument("--heldout-logs", required=True)
    ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rep, why = carve(a.sets, a.stash, a.heldout_logs, a.expect)
    rep["verdict"] = "OK" if why is None else "FAIL"
    rep["reason"] = why
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    if why is not None:
        print(f"ZZCARVE_FAIL {why}", flush=True)
        return 1
    print(f"ZZCARVE_OK {rep['served_train']} {rep['dropped_rows']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
