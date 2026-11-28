# assemble_run_root.py -- lead's glue between the two kits of card PREREG_V8F_PROOF_ADD99 (CROPREAD). The data kit wrote its outputs under enc_out/ and named the pass
# flag of p1 `pass_all`; the fit kit reads <root>/ctl/*.json with a top-level `pass` and looks for the sha256 of the train list in <root>/lists/LISTS_RECORD.json.
# This script builds a new run root that points at the data kit's outputs and carries the two records in the form the fit kit reads. It computes no value of its own:
# the pass of p1 is the AND of the three clause flags the data kit wrote (and has to equal its `pass_all`); the list record is the two records of the data kit put
# together, with the sha256 of the list files as they are on disk. It never overwrites and never deletes; it refuses when the run root exists.
import hashlib
import json
import os
import shutil
import sys
import time

SRC = "/workspace/v8proof/cropread/enc_out"
RUN = sys.argv[1] if len(sys.argv) > 1 else "/workspace/v8proof/cropread/run1"
KIT = "/workspace/v8proof/cropread/kit"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(RUN):
        print("REFUSED: %s exists" % RUN)
        return 2
    for p in ("ctl/C1.json", "ctl/C7.json", "ctl/P1.json", "lists/eval141.txt", "lists/train_final.txt", "lists/LISTS_RECORD.json", "lists/TRAIN_FINAL_RECORD.json", "tok/D/MANIFEST.json", "tok/B/MANIFEST.json", "DONE.txt"):
        if not os.path.isfile(os.path.join(SRC, p)):
            print("REFUSED: missing %s" % p)
            return 2
    os.makedirs(os.path.join(RUN, "ctl"))
    os.makedirs(os.path.join(RUN, "lists"))
    os.symlink(os.path.join(SRC, "tok"), os.path.join(RUN, "tok"))
    os.symlink(KIT, os.path.join(RUN, "kit"))
    for n in ("eval141.txt", "train_final.txt", "train_final_ids.txt", "train_order.txt", "TRAIN_FINAL_RECORD.json"):
        s = os.path.join(SRC, "lists", n)
        if os.path.isfile(s):
            os.symlink(s, os.path.join(RUN, "lists", n))
    for n in ("C1.json", "C7.json"):
        shutil.copyfile(os.path.join(SRC, "ctl", n), os.path.join(RUN, "ctl", n))
        assert sha(os.path.join(SRC, "ctl", n)) == sha(os.path.join(RUN, "ctl", n))
    p1 = json.load(open(os.path.join(SRC, "ctl", "P1.json"), encoding="utf-8"))
    flags = [p1["clause_i"]["pass"], p1["clause_ii"]["pass"], p1["clause_iii"]["pass"]]
    assert all(isinstance(f, bool) for f in flags), flags
    both = bool(all(flags))
    assert both == p1["pass_all"], (flags, p1["pass_all"])
    assert "pass" not in p1
    out = dict(p1)
    out["pass"] = both
    out["adapter"] = {"tool": "assemble_run_root.py", "source": os.path.join(SRC, "ctl", "P1.json"), "source_sha256": sha(os.path.join(SRC, "ctl", "P1.json")),
                      "what": "the key `pass` is added: the AND of the three clause flags of the source, equal to its `pass_all`; nothing else is changed"}
    with open(os.path.join(RUN, "ctl", "P1.json"), "x", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    lr = json.load(open(os.path.join(SRC, "lists", "LISTS_RECORD.json"), encoding="utf-8"))
    tf = json.load(open(os.path.join(SRC, "lists", "TRAIN_FINAL_RECORD.json"), encoding="utf-8"))
    rec = {"card": "PREREG_V8F_PROOF_ADD99", "what": "the two list records of the data kit put together for the fit kit, with the sha256 of the list files as on disk; written before the first fit",
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "train_final_sha256": sha(os.path.join(SRC, "lists", "train_final.txt")), "eval141_sha256": sha(os.path.join(SRC, "lists", "eval141.txt")),
           "sources": {"LISTS_RECORD.json": sha(os.path.join(SRC, "lists", "LISTS_RECORD.json")), "TRAIN_FINAL_RECORD.json": sha(os.path.join(SRC, "lists", "TRAIN_FINAL_RECORD.json"))},
           "lists_record": lr, "train_final_record": tf}
    with open(os.path.join(RUN, "lists", "LISTS_RECORD.json"), "x", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    rows = [ln.split() for ln in open(os.path.join(RUN, "lists", "train_final.txt"), encoding="utf-8") if ln.strip()]
    ev = [ln.strip() for ln in open(os.path.join(RUN, "lists", "eval141.txt"), encoding="utf-8") if ln.strip()]
    roles = [r[1] for r in rows]
    print("ZZ RUN-ROOT %s: train %d (fit %d, sel %d), eval %d, overlap %d, p1 pass %s, c1 pass %s, c7 pass %s, no fits dir yet %s" % (
        RUN, len(rows), roles.count("fit"), roles.count("sel"), len(ev), len(set(r[0] for r in rows) & set(ev)), both,
        json.load(open(os.path.join(RUN, "ctl", "C1.json")))["pass"], json.load(open(os.path.join(RUN, "ctl", "C7.json")))["pass"], not os.path.exists(os.path.join(RUN, "fits"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
