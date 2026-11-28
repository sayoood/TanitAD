#!/usr/bin/env python3
"""restage_v2 helpers.  stdlib only (python >= 3.8).  The token is read INSIDE this process from a file (first `hf_...` match) and is never printed,
never an argument, never in a URL.  Every verdict is one line starting `ZZREV_` / `ZZGUARD_`.

  place   --repo R --rev REV --run RUNDIR --expect-step N --report FILE [--tag NAME] [--dry-run] [--token-file F]
          put run/{ckpt_last.pt, config.json, metrics.jsonl, snap_epoch001..006.pt} of REV into RUNDIR BEFORE the chain starts.
          Verified, in this order: every file exists at REV; sizes; sha256 of the .pt files against HF's LFS oid; the LAST `"event": "ckpt"` line
          of the downloaded metrics.jsonl has step == N (read from the file, not trusted from the revision); config.json byte-equal to main's.
          Nothing is placed unless all pass.  An existing different ckpt_last.pt is REFUSED, never overwritten.  Nothing in the HF repo is
          modified (optional --tag adds a tag on main's head, warn-only).  --dry-run: resolves REV, lists, downloads ONLY the small json/jsonl, no .pt.
  verify  --run RUNDIR --expect-step N [--ckpt-sha256 HEX]       re-read the resume point from DISK (used after the chain's `recover` stage)
  guard   --metrics F --offset BYTES --wait-s S [--lines 5] --alert FILE
          after the trainer starts: the first labeller events after OFFSET must all have no_parts == 0; else write the ALERT file (never kills anything).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co"
SNAPS = ["snap_epoch%03d.pt" % i for i in range(1, 7)]
NEED = ["ckpt_last.pt", "config.json", "metrics.jsonl"] + SNAPS


def clean(s):
    return re.sub(r"hf_[A-Za-z0-9]+", "hf_***", str(s))


def die(msg, code=2):
    print("ZZREV_FAIL " + clean(msg), flush=True)
    sys.exit(code)


def token(path):
    try:
        m = re.search(r"hf_[A-Za-z0-9]+", open(path, encoding="utf-8", errors="ignore").read())
    except OSError as e:
        die("cannot read token file: %s" % type(e).__name__)
    if not m:
        die("no token in the token file")
    return m.group(0)


class _DropAuthCrossHost(urllib.request.HTTPRedirectHandler):
    """HF redirects LFS downloads to a signed CDN URL; an Authorization header must not follow it to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        n = super().redirect_request(req, fp, code, msg, headers, newurl)
        if n is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            n.headers.pop("Authorization", None)
            n.unredirected_hdrs.pop("Authorization", None)
        return n


OPENER = urllib.request.build_opener(_DropAuthCrossHost)


def call(url, tok, method="GET", body=None):
    h = {"Authorization": "Bearer " + tok}
    if body is not None:
        h["Content-Type"] = "application/json"
    try:
        return OPENER.open(urllib.request.Request(url, data=body, headers=h, method=method), timeout=300)
    except urllib.error.HTTPError as e:
        die("HTTP %s on %s" % (e.code, urllib.parse.urlsplit(url).path))
    except Exception as e:
        die("%s on %s" % (type(e).__name__, urllib.parse.urlsplit(url).path))


def jget(url, tok):
    return json.load(call(url, tok))


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)


def download(url, dest, tok):
    r = call(url, tok)
    h = hashlib.sha256()
    part = dest + ".part"
    with open(part, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            h.update(b)
            f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, dest)
    return h.hexdigest()


def resolve_rev(repo, rev, tok):
    if re.fullmatch(r"[0-9a-f]{40}", rev):
        return rev
    if not re.fullmatch(r"[0-9a-f]{7,39}", rev):
        die("revision is neither a full sha nor a hex prefix")
    hits = []
    for p in range(0, 80):
        page = jget("%s/api/models/%s/commits/main?p=%d" % (API, repo, p), tok)
        if not page:
            break
        hits += [c["id"] for c in page if c["id"].startswith(rev)]
        if hits:
            break
    if len(hits) != 1:
        die("revision prefix %s matched %d commits on main" % (rev[:12], len(hits)))
    return hits[0]


def last_ckpt(metrics_path):
    """(step of the last `"event": "ckpt"` line, (event, step) of the very last line)."""
    last, tail = None, None
    with open(metrics_path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.strip():
                continue
            tail = ln
            if '"event": "ckpt"' in ln:
                try:
                    last = json.loads(ln)
                except ValueError:
                    pass
    try:
        t = json.loads(tail) if tail else {}
    except ValueError:
        t = {}
    return (last.get("step") if last else None), (t.get("event"), t.get("step"))


def cmd_place(a):
    tok = token(a.token_file)
    rev = resolve_rev(a.repo, a.rev, tok)
    tree = {os.path.basename(x["path"]): x for x in jget("%s/api/models/%s/tree/%s/run?expand=1" % (API, a.repo, rev), tok) if x.get("type") == "file"}
    miss = [n for n in NEED if n not in tree]
    if miss:
        die("revision %s lacks run/%s" % (rev[:12], ",".join(miss)))
    main_sha = jget("%s/api/models/%s/revision/main" % (API, a.repo), tok)["sha"]
    tmp = a.run.rstrip("/\\") + ".rollback_tmp"
    os.makedirs(tmp, exist_ok=True)
    sha, sizes = {}, {}
    for n in NEED:
        if a.dry_run and n.endswith(".pt"):
            continue
        x = tree[n]
        oid = (x.get("lfs") or {}).get("oid")
        d = os.path.join(tmp, n)
        h = download("%s/%s/resolve/%s/run/%s" % (API, a.repo, rev, n), d, tok)
        if os.path.getsize(d) != x["size"]:
            die("size mismatch on run/%s: %d != %d" % (n, os.path.getsize(d), x["size"]))
        if oid and h != oid:
            die("sha256 mismatch on run/%s (HF lfs oid)" % n)
        sha[n], sizes[n] = h, x["size"]
    step, tail = last_ckpt(os.path.join(tmp, "metrics.jsonl"))
    if step != a.expect_step:
        die("the restored metrics.jsonl's last ckpt event is step %s, expected %d" % (step, a.expect_step))
    main_cfg = call("%s/%s/resolve/main/run/config.json" % (API, a.repo), tok).read()
    if main_cfg != open(os.path.join(tmp, "config.json"), "rb").read():
        die("config.json at the revision differs from main's (the run identity would change)")
    rep = {"repo": a.repo, "rev": rev, "main_head_before": main_sha, "ckpt_step": step, "metrics_last_line": list(tail),
           "ckpt_sha256": sha.get("ckpt_last.pt"), "files": sizes, "dry_run": bool(a.dry_run)}
    if a.dry_run:
        for n in os.listdir(tmp):
            os.remove(os.path.join(tmp, n))
        os.rmdir(tmp)
        print("ZZREV_DRY_OK rev=%s main_head=%s ckpt_step=%s metrics_last=%s/%s config=equal_to_main pt_files_skipped=%d" %
              (rev[:12], main_sha[:12], step, tail[0], tail[1], len([n for n in NEED if n.endswith(".pt")])), flush=True)
        return 0
    os.makedirs(a.run, exist_ok=True)
    ck = os.path.join(a.run, "ckpt_last.pt")
    if os.path.exists(ck):
        if sha256_file(ck) != sha["ckpt_last.pt"]:
            die("run dir already holds a DIFFERENT ckpt_last.pt: refusing to overwrite it")
    else:
        for n in ("config.json", "metrics.jsonl"):
            p = os.path.join(a.run, n)
            if os.path.exists(p) and open(p, "rb").read() != open(os.path.join(tmp, n), "rb").read():
                die("run dir already holds a different run/%s without a checkpoint: refusing" % n)
    for n in NEED:
        os.replace(os.path.join(tmp, n), os.path.join(a.run, n))
    os.rmdir(tmp)
    if a.tag:
        try:
            body = json.dumps({"tag": a.tag, "message": "main head before P6 rollback to %s" % rev[:12]}).encode()
            call("%s/api/models/%s/tag/%s" % (API, a.repo, main_sha), tok, "POST", body).read()
            print("ZZREV_TAG_OK %s -> %s" % (a.tag, main_sha[:12]), flush=True)
        except SystemExit:
            print("ZZREV_TAG_WARN could not create tag %s (history on HF is kept regardless)" % a.tag, flush=True)
    tmpr = a.report + ".tmp"
    open(tmpr, "w").write(json.dumps(rep, indent=1))
    os.replace(tmpr, a.report)
    print("ZZREV_PLACED rev=%s main_head_before=%s ckpt_step=%s ckpt_sha=%s files=%d" % (rev[:12], main_sha[:12], step, rep["ckpt_sha256"][:16], len(NEED)), flush=True)
    return 0


def cmd_verify(a):
    ck = os.path.join(a.run, "ckpt_last.pt")
    if not os.path.isfile(ck) or os.path.getsize(ck) == 0:
        die("no ckpt_last.pt in " + a.run)
    h = sha256_file(ck)
    if a.ckpt_sha256 and h != a.ckpt_sha256:
        die("ckpt_last.pt sha256 %s != the placed %s (something overwrote it)" % (h[:16], a.ckpt_sha256[:16]))
    step, tail = last_ckpt(os.path.join(a.run, "metrics.jsonl"))
    if step != a.expect_step:
        die("metrics.jsonl on disk: last ckpt event step %s, expected %d" % (step, a.expect_step))
    print("ZZREV_VERIFY_OK ckpt_step=%s ckpt_sha=%s metrics_last=%s/%s" % (step, h[:16], tail[0], tail[1]), flush=True)
    return 0


def cmd_guard(a):
    t0, first_at, seen, buf = time.time(), None, [], b""
    pos = a.offset
    while True:
        try:
            with open(a.metrics, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk:
            buf += chunk
            pos += len(chunk)
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if b'"event": "labeller"' in ln:
                    try:
                        seen.append(json.loads(ln))
                    except ValueError:
                        continue
                    first_at = first_at or time.time()
        if len(seen) >= a.lines:
            break
        if first_at is None and time.time() - t0 > a.wait_s:
            print("ZZGUARD_TIMEOUT no labeller event within %ds (nothing judged)" % a.wait_s, flush=True)
            return 0
        if first_at is not None and time.time() - first_at > 900:
            break
        time.sleep(20)
    vals = [e.get("no_parts") for e in seen]
    if all(v == 0 for v in vals):
        print("ZZGUARD_OK n=%d no_parts=%s steps=%s" % (len(vals), vals, [e.get("step") for e in seen]), flush=True)
        return 0
    msg = "COVERAGE ALERT: labeller no_parts != 0 on the first %d labeller events after the launch: %s (steps %s). Samples train WITHOUT scorer targets." % (
        len(vals), vals, [e.get("step") for e in seen])
    open(a.alert, "w").write(msg + "\nTraining was NOT stopped. Check the twin parts link (restage_v2 step twin_link) and p6 lanes_distinct_*.out.\n")
    print("ZZGUARD_ALERT " + msg, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("place")
    p.add_argument("--repo", required=True); p.add_argument("--rev", required=True); p.add_argument("--run", required=True)
    p.add_argument("--expect-step", type=int, required=True); p.add_argument("--report", required=True)
    p.add_argument("--tag", default=None); p.add_argument("--dry-run", action="store_true"); p.add_argument("--token-file", default="<TOKEN_FILE>")
    p = sp.add_parser("verify")
    p.add_argument("--run", required=True); p.add_argument("--expect-step", type=int, required=True); p.add_argument("--ckpt-sha256", default=None)
    p = sp.add_parser("guard")
    p.add_argument("--metrics", required=True); p.add_argument("--offset", type=int, required=True); p.add_argument("--wait-s", type=int, default=14400)
    p.add_argument("--lines", type=int, default=5); p.add_argument("--alert", required=True)
    a = ap.parse_args()
    return {"place": cmd_place, "verify": cmd_verify, "guard": cmd_guard}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
