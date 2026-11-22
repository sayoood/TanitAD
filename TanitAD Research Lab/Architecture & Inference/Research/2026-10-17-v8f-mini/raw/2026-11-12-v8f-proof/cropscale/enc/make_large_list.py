#!/usr/bin/env python3
"""Lists of card ADD106 (CROPSCALE10K), section 2. Pure Python, no GPU, no network. Three modes:

  new-order  eligible minus the pilot's 500 clips, in the order of train_eligible.txt  -> <out>/train_new_order.txt
  staged     the clips of train_new_order.txt that the stage ledger marks ok           -> <out>/train_new_staged.txt
  final      the pilot's 500 lines unchanged, then every new clip that has a token file (role fit)
             -> <out>/train_final.txt and <out>/LISTS_LARGE_RECORD.json (counts, hashes, clips left out with a reason)

Nothing is deleted or moved; every output is written whole under a .part name and renamed once (never over an existing file with other bytes:
a second run that would change a finished file stops).
"""
import argparse
import glob
import hashlib
import json
import os
import sys

POD_ROOT = "/workspace/v8proof/cropscale"
ELIGIBLE = "/workspace/v8proof/cropread/enc_out/lists/train_eligible.txt"
PILOT_FINAL = "/workspace/v8proof/cropread/run1/lists/train_final.txt"
EVAL141 = "/workspace/v8proof/cropread/run1/lists/eval141.txt"
EXPECT_ELIGIBLE, EXPECT_PILOT, EXPECT_EVAL = 1679, 500, 141


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_ids(path):
    """first column of every non-empty line (a role column or a position column is not an id: positions are numeric)."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            t = ln.split()
            if not t:
                continue
            out.append(t[-1] if (len(t) == 2 and t[0].isdigit()) else t[0])
    return out


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            t = ln.split()
            if not t:
                continue
            if len(t) != 2 or t[1] not in ("fit", "sel"):
                raise SystemExit("REFUSED: %s has a line that is not '<clip id> fit|sel': %r" % (path, ln[:80]))
            rows.append((t[0], t[1]))
    return rows


def write_text(path, text):
    """whole file under .part then one rename; an existing file with other bytes is refused."""
    data = text.encode("utf-8")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            if fh.read() == data:
                return False
        raise SystemExit("REFUSED: %s exists with other content (this tool never overwrites a finished list)" % path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return True


def new_order(eligible, pilot_ids):
    ps = set(pilot_ids)
    return [c for c in eligible if c not in ps]


def ledger_last(path):
    d = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    try:
                        r = json.loads(ln)
                        d[r["clip"]] = r
                    except Exception:                                     # noqa: BLE001
                        pass
    return d


def staged_list(order, ledger):
    return [c for c in order if (ledger.get(c) or {}).get("ok")]


def read_encode_fails(tok_root):
    """{clip: message} from every ENCODE_FAILS*.json next to tok/D (the encode tool rewrites one file per call; the queue keeps a copy per call)."""
    d = {}
    for p in sorted(glob.glob(os.path.join(tok_root, "ENCODE_FAILS*.json"))):
        try:
            for c, msg in json.load(open(p, encoding="utf-8")):
                d[c] = msg
        except Exception:                                                 # noqa: BLE001
            pass
    return d


def tokens_present(clips, tok_d):
    """clips with a token file that the manifest names with the file's byte size."""
    mp = os.path.join(tok_d, "MANIFEST.json")
    man = json.load(open(mp, encoding="utf-8"))["clips"] if os.path.exists(mp) else {}
    have = set()
    for c in clips:
        p = os.path.join(tok_d, c + ".npy")
        m = man.get(c)
        if m and os.path.exists(p) and os.path.getsize(p) == m["bytes"]:
            have.add(c)
    return have


def build_final(eligible, eval_ids, pilot_rows, new_ord, ledger, have_tok, enc_fails):
    """-> (rows, left_out). rows = pilot rows unchanged + every new clip with a token file (role fit), in eligible order."""
    rows = list(pilot_rows) + [(c, "fit") for c in new_ord if c in have_tok]
    left = []
    for c in new_ord:
        if c in have_tok:
            continue
        r = ledger.get(c)
        if not (r and r.get("ok")):
            left.append({"clip": c, "stage": "did not stage", "reason": "did not stage: " + str((r or {}).get("reason", "no ledger line (stage did not reach it)"))})
        else:
            left.append({"clip": c, "stage": "staged", "reason": "did not encode: " + str(enc_fails.get(c, "no token file in the manifest"))})
    return rows, left


def cmd_new_order(a):
    elig, ev, prow = read_ids(a.eligible), read_ids(a.eval), read_rows(a.pilot_final)
    if a.strict:
        if len(elig) != EXPECT_ELIGIBLE or len(set(elig)) != len(elig):
            raise SystemExit("REFUSED: eligible list has %d lines, %d distinct (expected %d)" % (len(elig), len(set(elig)), EXPECT_ELIGIBLE))
        if len(prow) != EXPECT_PILOT:
            raise SystemExit("REFUSED: the pilot's train_final has %d lines (expected %d)" % (len(prow), EXPECT_PILOT))
        if len(ev) != EXPECT_EVAL:
            raise SystemExit("REFUSED: eval list has %d lines (expected %d)" % (len(ev), EXPECT_EVAL))
    if set(elig) & set(ev):
        raise SystemExit("REFUSED: %d eligible clips are evaluation clips" % len(set(elig) & set(ev)))
    pids = [c for c, _ in prow]
    if not set(pids) <= set(elig):
        raise SystemExit("REFUSED: %d pilot clips are not in the eligible list" % len(set(pids) - set(elig)))
    new = new_order(elig, pids)
    if a.strict and len(new) != EXPECT_ELIGIBLE - EXPECT_PILOT:
        raise SystemExit("REFUSED: %d new clips (expected %d)" % (len(new), EXPECT_ELIGIBLE - EXPECT_PILOT))
    ch = write_text(os.path.join(a.out, "train_new_order.txt"), "".join(c + "\n" for c in new))
    print("new-order: %d eligible, %d pilot, %d new%s" % (len(elig), len(pids), len(new), "" if ch else " (already written, same bytes)"))
    return 0


def cmd_staged(a):
    new = read_ids(os.path.join(a.out, "train_new_order.txt"))
    st = staged_list(new, ledger_last(a.stage_ledger))
    body = "".join(c + "\n" for c in st)
    path = os.path.join(a.out, "train_new_staged_%s.txt" % sha_bytes(body.encode("utf-8"))[:8])     # content-addressed: a longer list is a new file
    if a.write:
        write_text(path, body)
    print("staged: %d of %d new clips are ok in the stage ledger" % (len(st), len(new)))
    print("LIST=%s" % path)
    return 0


def cmd_final(a):
    elig, ev, prow = read_ids(a.eligible), read_ids(a.eval), read_rows(a.pilot_final)
    new = read_ids(os.path.join(a.out, "train_new_order.txt"))
    led = ledger_last(a.stage_ledger)
    have = tokens_present(new, a.tok_dir)
    rows, left = build_final(elig, ev, prow, new, led, have, read_encode_fails(os.path.dirname(a.tok_dir.rstrip("/"))))
    body = "".join("%s %s\n" % (c, r) for c, r in rows)
    changed = write_text(os.path.join(a.out, "train_final.txt"), body)
    rec = {"card": "ADD106", "n_eligible": len(elig), "n_eval": len(ev), "n_pilot_lines": len(prow), "n_new_order": len(new),
           "n_new_staged": len(staged_list(new, led)), "n_new_with_token_file": len(have), "n_left_out": len(left),
           "n_large_list": len(rows), "n_fit": sum(1 for _, r in rows if r == "fit"), "n_sel": sum(1 for _, r in rows if r == "sel"),
           "sha256": {"train_eligible": sha_file(a.eligible), "eval141": sha_file(a.eval), "pilot_train_final": sha_file(a.pilot_final),
                      "train_new_order": sha_file(os.path.join(a.out, "train_new_order.txt")),
                      "train_final": sha_bytes(body.encode("utf-8")),
                      "tok_manifest": sha_file(os.path.join(a.tok_dir, "MANIFEST.json")) if os.path.exists(os.path.join(a.tok_dir, "MANIFEST.json")) else None},
           "left_out": left,
           "left_out_counts": {"did not stage": sum(1 for x in left if x["stage"] == "did not stage"),
                               "did not encode": sum(1 for x in left if x["stage"] == "staged")}}
    write_text_rec = json.dumps(rec, indent=1) + "\n"
    rp = os.path.join(a.out, "LISTS_LARGE_RECORD.json")
    if os.path.exists(rp):
        # a record of a finished list is never overwritten with other numbers
        old = json.load(open(rp, encoding="utf-8"))
        if old.get("sha256", {}).get("train_final") != rec["sha256"]["train_final"]:
            raise SystemExit("REFUSED: LISTS_LARGE_RECORD.json exists for another list")
    else:
        write_text(rp, write_text_rec)
    print("final: %d lines (%d fit, %d sel), %d new clips with tokens, %d left out%s" % (
        len(rows), rec["n_fit"], rec["n_sel"], len(have), len(left), "" if changed else " (list already written, same bytes)"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["new-order", "staged", "final"])
    ap.add_argument("--eligible", default=ELIGIBLE)
    ap.add_argument("--eval", default=EVAL141)
    ap.add_argument("--pilot-final", default=PILOT_FINAL)
    ap.add_argument("--out", default=POD_ROOT + "/lists")
    ap.add_argument("--stage-ledger", default=POD_ROOT + "/stage/ledger.jsonl")
    ap.add_argument("--tok-dir", default=POD_ROOT + "/enc_out/tok/D")
    ap.add_argument("--no-strict", dest="strict", action="store_false", help="synthetic data: skip the 1,679 / 500 / 141 count checks")
    ap.add_argument("--write", action="store_true", help="staged: write train_new_staged.txt")
    a = ap.parse_args(argv)
    return {"new-order": cmd_new_order, "staged": cmd_staged, "final": cmd_final}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
