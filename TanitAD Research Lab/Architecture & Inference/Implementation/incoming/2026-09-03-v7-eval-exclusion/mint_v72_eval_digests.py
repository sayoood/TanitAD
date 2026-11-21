"""MINT `stack/tanitad/data/v72_eval_clip_digests.json` (D-V7-EVAL-EXCLUSION).

The provenance of a COMMITTED data file, banked beside it rather than left in a
scratch dir — `load_clip_digests` can re-verify the file's self-consistency but
not where it came from, and "the agent ran a script once" is not provenance.

It REFUSES to write unless THREE independent artifacts of the v7.2 release agree
on the same 147 clip ids: the label blob (md5 aa12c948f062181c3297265b51526ec5 =
`intrain_eval.V72['eval']`), the clip index beside it (md5
2f68790b0a6fc9d7041331a190b245d4 = `V72_INDEX['eval_index']`) and the release's
own `eval_v72_clip_digests.txt`.

🔒 The output is per-clip sha256 ONLY — no clip id ever leaves the release copy.

    python mint_v72_eval_digests.py <repo>/stack/tanitad/data/v72_eval_clip_digests.json

MEASURED 2026-09-03 on the dev box against C:/Users/Admin/tanitad-wt/_s2build/
release/v72 (`REL` below is that path; point it at the release root on any other
host). Reproduces clip_id_sha256_sorted 20ec68bd7b021b22… and
digest_of_digests 931af1c1f23be760…
"""
import gzip, hashlib, json, sys
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release/v72")
OUT = Path(sys.argv[1])
BLOB = REL / "s2_labels_v7.2_eval.jsonl.gz"
IDX = REL / "clip_index_eval.json"
TXT = REL / "eval_v72_clip_digests.txt"

def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()

def cd(c):
    return hashlib.sha256(str(c).encode("utf-8")).hexdigest()

def uid_digest(us):
    us = sorted(str(u) for u in us)
    assert len(set(us)) == len(us)
    return hashlib.sha256("\n".join(us).encode("utf-8")).hexdigest()

ids_blob = []
with gzip.open(BLOB, "rt", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            ids_blob.append(json.loads(line)["clip_id"])
idx = json.loads(IDX.read_text(encoding="utf-8"))
ids_idx = sorted(idx["clips"])
digs_txt = sorted(l.strip() for l in TXT.read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.startswith("#"))

assert md5(BLOB) == "aa12c948f062181c3297265b51526ec5", md5(BLOB)
assert md5(IDX) == "2f68790b0a6fc9d7041331a190b245d4", md5(IDX)
assert len(ids_blob) == 147 and len(set(ids_blob)) == 147, len(ids_blob)
assert sorted(set(ids_blob)) == ids_idx, "blob and clip index disagree"
digs = sorted(cd(c) for c in ids_blob)
assert digs == digs_txt, "release digest file disagrees"

doc = {
    "schema": "tanitad.parity_clip_digests/1",
    "corpus_key": "physicalai-b1-w120-256x640cyl",
    "is_full_corpus": False,
    "deployment": "v7.2-eval",
    "role": "the v7.2 EVAL split — the 147 label records every v7 T1/val number "
            "is computed over. 141 of them HAVE THEIR PIXELS INSIDE the 4,713-clip "
            "B1 train cache, so a v7 trainer pointed at that cache without an "
            "exclusion trains its world-model objectives on the evaluation split "
            "(BACKLOG R8 / D-V7-EVAL-EXCLUSION).",
    "split": "eval",
    "n_clips": 147,
    "digest_algorithm": "sha256(clip_id.encode('utf-8')).hexdigest()",
    "clip_id_sha256_sorted": hashlib.sha256(
        "\n".join(sorted(set(ids_blob))).encode("utf-8")).hexdigest(),
    "verified_against": (
        "THREE independent artifacts of the v7.2 release agree on the same 147 "
        "ids: the label blob (md5 aa12c948f062181c3297265b51526ec5, the "
        "intrain_eval.V72['eval'] pin), the clip index beside it (md5 "
        "2f68790b0a6fc9d7041331a190b245d4, the V72_INDEX['eval_index'] pin) and "
        "the release's own eval_v72_clip_digests.txt. The whole-corpus manifest "
        "digest is NOT applicable to a 147-of-4,719 subset and is deliberately "
        "not claimed."),
    "digest_of_digests": uid_digest(digs),
    "source": "release/v72/s2_labels_v7.2_eval.jsonl.gz "
              "(md5 aa12c948f062181c3297265b51526ec5, 147 records)",
    "cross_check_source": "release/v72/clip_index_eval.json "
                          "(md5 2f68790b0a6fc9d7041331a190b245d4, 147 clips) "
                          "+ release/v72/eval_v72_clip_digests.txt",
    "cross_check_episodes": 147,
    "minted": "2026-09-03 on the dev box from the local release copy "
              "C:/Users/Admin/tanitad-wt/_s2build/release/v72 "
              "(ArchInf FlyWheel, D-V7-EVAL-EXCLUSION)",
    "confidentiality": "\U0001f512 per-clip sha256 ONLY — the ids are gated-confidential.",
    "_read": "membership oracle for `train_v6_staged.py --exclude-eval-clips`. "
             "The v7.2 EVAL LABEL BLOB is the PRIMARY source and this file is "
             "the fallback for a host that does not hold it; when both are "
             "present the trainer cross-checks them and refuses on a mismatch.",
    "clip_id_digests": digs,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_bytes((json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
                .replace("\n", "\r\n").encode("utf-8"))
print("wrote", OUT, "n=", len(digs))
print("clip_id_sha256_sorted", doc["clip_id_sha256_sorted"])
print("digest_of_digests", doc["digest_of_digests"])
