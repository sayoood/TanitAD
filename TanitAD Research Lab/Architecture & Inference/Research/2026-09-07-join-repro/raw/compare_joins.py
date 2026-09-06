"""Compare two b1 agent-join .xz artifacts by CONTAINER and by CONTENT.

Headline claim = the DECOMPRESSED stream. The .xz digest is reported beside it,
never instead of it: xz is not deterministic across preset/threads, so a
container-only mismatch says nothing about the data.

Emits, per file:
  xz_md5                container digest (the digest the receipts quote)
  xz_bytes
  raw_sha256            sha256 of the DECOMPRESSED byte stream, ORDER-SENSITIVE
  raw_bytes             decompressed size
  n_lines               rows
  n_agent_boxes         sum(len(rec["agents"]))
  n_clips               |{clip_id}|
  clipset_sha256        sha256 over the SORTED clip ids -> clip SET identity
  key_seq_sha256        sha256 over (clip_id, frame) in FILE ORDER -> ordering
  multiset_sha256       sha256 over the SORTED per-line digests -> content
                        identity INDEPENDENT of row order
"""
import hashlib
import json
import lzma
import sys


def scan(path):
    xz = hashlib.md5()
    xz_bytes = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                break
            xz.update(b)
            xz_bytes += len(b)

    raw = hashlib.sha256()
    keyseq = hashlib.sha256()
    raw_bytes = 0
    n_lines = 0
    n_boxes = 0
    clips = set()
    digests = []

    with lzma.open(path, "rb") as fh:
        for line in fh:
            raw.update(line)
            raw_bytes += len(line)
            n_lines += 1
            digests.append(hashlib.blake2b(line, digest_size=16).digest())
            rec = json.loads(line)
            clips.add(rec["clip_id"])
            n_boxes += len(rec["agents"])
            keyseq.update(b"%s|%d\n" % (rec["clip_id"].encode(), rec["frame"]))

    digests.sort()
    ms = hashlib.sha256()
    for d in digests:
        ms.update(d)

    cs = hashlib.sha256()
    for c in sorted(clips):
        cs.update(c.encode() + b"\n")

    return {
        "path": path,
        "xz_md5": xz.hexdigest(),
        "xz_bytes": xz_bytes,
        "raw_sha256": raw.hexdigest(),
        "raw_bytes": raw_bytes,
        "n_lines": n_lines,
        "n_agent_boxes": n_boxes,
        "n_clips": len(clips),
        "clipset_sha256": cs.hexdigest(),
        "key_seq_sha256": keyseq.hexdigest(),
        "multiset_sha256": ms.hexdigest(),
        "_clips": clips,
    }


def main():
    a_path, b_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    A = scan(a_path)
    print("scanned A", flush=True)
    B = scan(b_path)
    print("scanned B", flush=True)

    a_clips, b_clips = A.pop("_clips"), B.pop("_clips")

    # ---- digest sanity: a short or empty digest is INCONCLUSIVE, never a match
    bad = []
    for side, D in (("A", A), ("B", B)):
        for k, n in (("xz_md5", 32), ("raw_sha256", 64), ("multiset_sha256", 64),
                     ("clipset_sha256", 64), ("key_seq_sha256", 64)):
            if len(D[k]) != n:
                bad.append("%s.%s len=%d expected %d" % (side, k, len(D[k]), n))
    EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e"
    EMPTY_SHA = hashlib.sha256(b"").hexdigest()
    for side, D in (("A", A), ("B", B)):
        if D["xz_md5"] == EMPTY_MD5:
            bad.append("%s.xz_md5 is the EMPTY digest" % side)
        if D["raw_sha256"] == EMPTY_SHA:
            bad.append("%s.raw_sha256 is the EMPTY digest" % side)

    # ---- same-breath non-zero controls
    controls = {
        "A_n_lines_nonzero": A["n_lines"] > 0,
        "B_n_lines_nonzero": B["n_lines"] > 0,
        "A_boxes_nonzero": A["n_agent_boxes"] > 0,
        "B_boxes_nonzero": B["n_agent_boxes"] > 0,
        "A_clips_nonzero": A["n_clips"] > 0,
        "B_clips_nonzero": B["n_clips"] > 0,
    }
    # ---- comparator control: two DIFFERENT strings must compare unequal
    controls["comparator_works"] = (
        hashlib.sha256(b"x").hexdigest() != hashlib.sha256(b"y").hexdigest())

    res = {
        "A": A, "B": B,
        "digest_length_problems": bad,
        "nonzero_controls": controls,
        "equal": {
            "xz_md5": A["xz_md5"] == B["xz_md5"],
            "xz_bytes": A["xz_bytes"] == B["xz_bytes"],
            "raw_sha256_ORDER_SENSITIVE": A["raw_sha256"] == B["raw_sha256"],
            "raw_bytes": A["raw_bytes"] == B["raw_bytes"],
            "n_lines": A["n_lines"] == B["n_lines"],
            "n_agent_boxes": A["n_agent_boxes"] == B["n_agent_boxes"],
            "n_clips": A["n_clips"] == B["n_clips"],
            "clipset_sha256": A["clipset_sha256"] == B["clipset_sha256"],
            "key_seq_sha256": A["key_seq_sha256"] == B["key_seq_sha256"],
            "multiset_sha256_ORDER_FREE": A["multiset_sha256"] == B["multiset_sha256"],
        },
        "clips_only_in_A": sorted(a_clips - b_clips)[:20],
        "clips_only_in_B": sorted(b_clips - a_clips)[:20],
        "n_clips_only_in_A": len(a_clips - b_clips),
        "n_clips_only_in_B": len(b_clips - a_clips),
    }

    if bad:
        res["VERDICT"] = "INCONCLUSIVE -- a digest was short or empty"
    elif not all(controls.values()):
        res["VERDICT"] = "INCONCLUSIVE -- a non-zero control read zero"
    elif res["equal"]["xz_md5"] and res["equal"]["raw_sha256_ORDER_SENSITIVE"]:
        res["VERDICT"] = "1 BYTE-IDENTICAL"
    elif res["equal"]["raw_sha256_ORDER_SENSITIVE"]:
        res["VERDICT"] = ("1b DECOMPRESSED-IDENTICAL, container digest differs "
                          "(xz encoder settings only)")
    elif res["equal"]["multiset_sha256_ORDER_FREE"]:
        res["VERDICT"] = "2 SEMANTICALLY IDENTICAL, ROW ORDER DIFFERS"
    elif (res["equal"]["n_lines"] and res["equal"]["n_agent_boxes"]
          and res["equal"]["clipset_sha256"]):
        res["VERDICT"] = ("2 SAME COUNTS+CLIPS, RECORD CONTENT DIFFERS "
                          "(inspect field-level)")
    else:
        res["VERDICT"] = "3 DIFFERENT"

    txt = json.dumps(res, indent=1)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
