"""Resolve the 18 `untracked_differ` files from the 2026-09-19 D: sync — mechanically.

For each path, compare D:'s pre-sync copy (frozen in the LOCAL ref
`refs/backup/d-pre-sync-20260919`) against HEAD, and CLASSIFY the difference by an
exact test, never by eye:

  BLOB-IDENTICAL   the two git blobs are the same object. The sync flagged these
                   only because it compared on-disk bytes (CRLF worktree vs LF blob).
  HEADER+BACKUP    HEAD == an N-line header prepended to the backup, exactly.
  REDACTION-ONLY   HEAD == redact(backup) exactly, where redact() maps every raw clip
                   UUID -> sha12, every 8-hex clip-id PREFIX -> the sha12 of its
                   full id, and the session scratchpad path -> `<scratchpad>`.
                   (sample_plan.json: compared as JSON, `clip` -> `clip_sha12`.)
  STUB-FILLED      backup == HEAD's first N lines + a `SEE_MANIFEST` placeholder
                   that HEAD replaced with the real manifest.
  UNEXPLAINED      none of the above — would need a human merge.

⛔ A REDACTION-ONLY or STUB-FILLED backup carries NOTHING HEAD lacks except raw clip
ids, which the programme keeps out of the repo (ids are gated-confidential; sha12
only). Merging such a backup back would RE-LEAK the ids. It is never merged.

🔒 Output carries no raw clip id. Usage:
    python resolve_divergence.py <d_sync_stage2.json> <out.json>
"""
import difflib
import hashlib
import json
import re
import subprocess
import sys

R = "D:/Projects/TanitAD"
B = "refs/backup/d-pre-sync-20260919"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SEP = r"[/\\]"
SCRATCH = re.compile(r"C:" + SEP + r"Users" + SEP + r"Admin" + SEP + r"AppData" + SEP
                     + r"Local" + SEP + r"Temp" + SEP + r"claude" + SEP
                     + r"G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD" + SEP
                     + r"[0-9a-f-]{36}" + SEP + r"scratchpad")


def git(*a):
    return subprocess.run(["git", "-C", R, *a], capture_output=True).stdout


def text(ref, p):
    return git("show", "%s:%s" % (ref, p)).replace(b"\r\n", b"\n").decode("utf-8")


def s12(u):
    return hashlib.sha256(u.encode("utf-8")).hexdigest()[:12]


def build_pref(files):
    """prefix -> sha12, from the FULL clip ids present in the backup copies
    (session ids inside scratchpad paths are stripped first: they are not clip ids)."""
    full = set()
    for p in files:
        full |= set(UUID.findall(SCRATCH.sub("", text(B, p))))
    pref = {u[:8]: s12(u) for u in full}
    assert len(pref) == len(full), "8-char prefix collision"
    return pref


pref = {}


def redact(s):
    s = SCRATCH.sub("<scratchpad>", s)
    s = UUID.sub(lambda m: s12(m.group(0)), s)
    # hex-boundary lookarounds, NOT \b: in `seq_<8-hex prefix>` the underscore is a word
    # character, so \b never fires and a prefix would survive un-redacted.
    return re.sub(r"(?<![0-9a-f])([0-9a-f]{8})(?![0-9a-f])",
                  lambda m: pref.get(m.group(1), m.group(1)), s)


def norm_json(o):
    if isinstance(o, dict):
        o = {k: norm_json(v) for k, v in o.items()}
        if "clip" in o:
            o["clip_sha12"] = s12(o.pop("clip"))
        return o
    if isinstance(o, list):
        return [norm_json(x) for x in o]
    return o


def classify(p, b, h, bb, hb):
    """The exact tests, as a function, so a mutation check can call the REAL classifier."""
    if bb == hb:
        v = "BLOB-IDENTICAL"
    elif p.endswith(".json") and b.lstrip().startswith(("{", "[")) and "clip" in b \
            and norm_json(json.loads(b)) == json.loads(h):
        v = "REDACTION-ONLY (semantic JSON: clip -> clip_sha12)"
    elif redact(b) == h:
        v = "REDACTION-ONLY (exact)"
    else:
        rh, rb = h.splitlines(), redact(b).splitlines()
        if len(rh) > len(rb) and rh[len(rh) - len(rb):] == rb:
            v = "HEADER+BACKUP (HEAD prepends %d lines)" % (len(rh) - len(rb))
        elif rb and rb[-1].strip() == "SEE_MANIFEST" and rh[:len(rb) - 1] == rb[:-1]:
            v = "STUB-FILLED (HEAD replaces the SEE_MANIFEST stub, +%d lines)" % (len(rh) - len(rb) + 1)
        else:
            v = "UNEXPLAINED"
    return v


def main():
    global pref
    files = json.load(open(sys.argv[1], encoding="utf-8"))["untracked_differ"]
    pref.update(build_pref(files))
    out = []
    for p in files:
        bb = git("rev-parse", "%s:%s" % (B, p)).strip().decode()
        hb = git("rev-parse", "HEAD:%s" % p).strip().decode()
        assert len(bb) == 40 and len(hb) == 40, "INCONCLUSIVE: a blob id did not resolve"
        b, h = text(B, p), text("HEAD", p)
        hl, bl = h.splitlines(), b.splitlines()
        sm = difflib.SequenceMatcher(None, hl, bl, autojunk=False)
        only_b = sum(j2 - j1 for t, i1, i2, j1, j2 in sm.get_opcodes() if t in ("replace", "insert"))
        only_h = sum(i2 - i1 for t, i1, i2, j1, j2 in sm.get_opcodes() if t in ("replace", "delete"))
        v = classify(p, b, h, bb, hb)
        hlog = git("log", "-1", "--format=%h %ad %s", "--date=short", "HEAD", "--", p).decode(
            "utf-8", "replace").strip()
        out.append({"path": p, "verdict": v,
                    "authoritative": "HEAD" if v != "UNEXPLAINED" else "?",
                    "backup_blob": bb, "head_blob": hb,
                    "lines_backup": len(bl), "lines_head": len(hl),
                    "lines_only_in_backup": only_b, "lines_only_in_head": only_h,
                    "raw_clip_ids_in_backup": len(UUID.findall(SCRATCH.sub("", b))),
                    "raw_clip_ids_in_head": len(UUID.findall(h)),
                    "head_landed_by": UUID.sub("<id>", hlog)[:160]})

    summary = {}
    for x in out:
        k = x["verdict"].split(" ")[0]
        summary[k] = summary.get(k, 0) + 1
    res = {"backup_ref": B, "backup_commit": git("rev-parse", "--short", B).decode().strip(),
           "head_commit": git("rev-parse", "--short", "HEAD").decode().strip(),
           "n_files": len(out), "summary": summary,
           "n_unexplained": sum(1 for x in out if x["verdict"] == "UNEXPLAINED"),
           "n_to_merge": 0 if all(x["verdict"] != "UNEXPLAINED" for x in out) else None,
           "files": out}
    json.dump(res, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=1))
    for x in out:
        print("  %-8s %-60s %s" % (x["authoritative"], x["path"].split("Research/")[1][:60],
                                   x["verdict"]))


if __name__ == "__main__":
    main()
