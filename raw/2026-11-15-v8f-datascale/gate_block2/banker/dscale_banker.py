#!/usr/bin/env python3
"""dscale_banker.py -- the DSCALE10K banker (pod, CPU + network only). It is the ONLY writer of /root/dscale10k/bank_ok/<arm>.<stage>.sha256.

For every bank_ready/<arm>.<stage>.txt (written by block1.sh / block2.sh: "<ckpt path> <sha256> <utc>") that has no matching bank_ok marker:
  1. REFUSE unless the record is well formed, the path is exactly <root>/runs/<arm>/<stage>/ckpt_last.pt (a regular file, not a link), the file's sha256 equals the record's, and, when the
     chain's ARM_<stage>_DONE marker exists, equals its sha256 too.
  2. Upload the checkpoint and its provenance files in ONE commit to a PRIVATE dataset repo (<account of /root/.hf_token>/tanitad-v8f-dscale10k-runs), under <arm>/<stage>/.
  3. Read every uploaded file back FROM THAT COMMIT and compare size and sha256; for the checkpoint the hub's LFS object id must be present and equal the sha256 as well.
  4. Only then write bank_ok/<arm>.<stage>.sha256 ("<sha256> <repo> <revision> <remote path> <size> <utc>", written atomically). chainlib.py `banked` reads the first token.
Any mismatch, any missing LFS id, any error: no marker, rc 1, the reason on the output. Idempotent (a banked tag is skipped; a marker whose sha differs from the file is reported as a CONFLICT and
never overwritten). It never deletes or moves anything (the chain's sweep deletes, and only against bank_ok). One instance at a time (flock). The token is read in place, never printed.
    python dscale_banker.py [--root /root/dscale10k] [--only TAG ...] [--repo-name N] [--poll-once]       # real hub
    python dscale_banker.py --local-hub DIR ...                                                       # TEST ONLY: a directory stands in for the hub
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
REPO_NAME = "tanitad-v8f-dscale10k-runs"
TAG_RE = re.compile(r"^([A-Za-z0-9]+_[A-Za-z0-9]+)\.([AB])$")
STAGE_FILES = ("argv.json", "selfcheck.json", "summary.json", "summary_check.json", "post.json", "train_log.jsonl")


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class HubError(Exception):
    pass


class LocalHub:
    """TEST ONLY: a directory plays the hub. corrupt_next: the next upload lands with one flipped byte; lfs_mode 'good' | 'none' | 'wrong'; fail_next: injected upload failures."""

    def __init__(self, root, private=True):
        self.root, self.private, self.repo_id = Path(root), private, "testuser/" + REPO_NAME
        self.corrupt_next = self.fail_next = 0
        self.lfs_mode = "good"
        self.n_commits = 0

    def ensure_repo(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.private:
            raise HubError(f"{self.repo_id} is NOT private; refusing to upload")

    def head(self):
        return f"rev{self.n_commits:04d}"

    def upload_many(self, pairs, message=""):
        if self.fail_next > 0:
            self.fail_next -= 1
            raise HubError("injected upload failure")
        self.n_commits += 1
        for local, remote in pairs:
            dst = self.root / remote
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dst)
            if self.corrupt_next > 0:
                self.corrupt_next -= 1
                b = bytearray(dst.read_bytes())
                b[len(b) // 2] ^= 0xFF
                dst.write_bytes(bytes(b))
        return self.head()

    def remote_size_sha(self, remote, revision):
        p = self.root / remote
        if not p.is_file():
            raise HubError("remote file missing: " + remote)
        s = sha256_file(p)
        lfs = {"good": s, "none": None, "wrong": "0" * 64}[self.lfs_mode]
        return p.stat().st_size, s, lfs


class Hub:
    """The real hub: huggingface_hub, classic LFS upload (Xet off), a PRIVATE dataset repo on the account of the token file."""

    def __init__(self, token_file="/root/.hf_token", repo_name=REPO_NAME):
        from huggingface_hub import HfApi, HfFileSystem
        self.tok = Path(token_file).read_text().strip()
        self.api = HfApi(token=self.tok)
        self.repo_id = f"{self.api.whoami()['name']}/{repo_name}"
        self.fs = HfFileSystem(token=self.tok)

    def ensure_repo(self):
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        if getattr(self.api.repo_info(self.repo_id, repo_type="dataset"), "private", None) is not True:
            raise HubError(f"{self.repo_id} is NOT private; refusing to upload")

    def head(self):
        return self.api.repo_info(self.repo_id, repo_type="dataset").sha

    def upload_many(self, pairs, message=""):
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(l)) for l, r in pairs]
        info = self.api.create_commit(self.repo_id, ops, repo_type="dataset", commit_message=message)
        return getattr(info, "oid", None) or self.head()

    def remote_size_sha(self, remote, revision):
        h, n = hashlib.sha256(), 0
        self.fs.invalidate_cache()
        with self.fs.open(f"datasets/{self.repo_id}@{revision}/{remote}", "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
                n += len(b)
        lfs = None
        try:
            i = self.api.get_paths_info(self.repo_id, [remote], repo_type="dataset", revision=revision)[0]
            lfs = getattr(getattr(i, "lfs", None), "sha256", None)
        except Exception:   # noqa: BLE001
            lfs = None
        return n, h.hexdigest(), lfs


class Banker:
    def __init__(self, root, hub, log=print, max_attempts=3, sleep=time.sleep):
        self.D, self.hub, self.log, self.max_attempts, self.sleep = Path(root), hub, log, max_attempts, sleep
        self.ready, self.ok = self.D / "bank_ready", self.D / "bank_ok"
        self.ledger = self.D / "banker_ledger.jsonl"

    def _led(self, **rec):
        rec["t"] = utc()
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def pending(self):
        return sorted(p.stem for p in self.ready.glob("*.txt")) if self.ready.is_dir() else []

    def files_for(self, arm, stage, ckpt):
        r = self.D / "runs" / arm / stage
        pairs = [(ckpt, f"{arm}/{stage}/ckpt_last.pt")]
        for n in STAGE_FILES:
            if (r / n).is_file():
                pairs.append((r / n, f"{arm}/{stage}/{n}"))
        tk = self.D / "gate" / "tokens" / f"{arm}.{stage}"
        for n in ("gate_token.json", "checks.json"):
            if (tk / n).is_file():
                pairs.append((tk / n, f"{arm}/{stage}/gate/{n}"))
        if stage == "A":
            for p, n in ((self.D / "dry" / f"{arm}.A.dry.json", "dry.json"), (self.D / "gate" / f"evidence_{arm}.json", "gate/evidence.json")):
                if p.is_file():
                    pairs.append((p, f"{arm}/A/{n}"))
        else:
            for p, n in ((self.D / "runs" / arm / "argv_B.json", "argv_B.json"), (self.D / "runs" / arm / "bank" / "bank.json", "bank/bank.json"), (self.D / "runs" / arm / "bank" / "PROVENANCE.json", "bank/PROVENANCE.json")):
                if p.is_file():
                    pairs.append((p, f"{arm}/B/{n}"))
        return pairs

    def check_record(self, tag):
        """-> (arm, stage, ckpt Path, sha256) or raises ValueError(reason)."""
        m = TAG_RE.match(tag)
        if not m:
            raise ValueError(f"tag {tag!r} is not <arm>.<A|B>")
        arm, stage = m.groups()
        f = (self.ready / f"{tag}.txt").read_text().split()
        if len(f) < 2 or not re.fullmatch(r"[0-9a-f]{64}", f[1]):
            raise ValueError(f"bank_ready record malformed: {f}")
        want = self.D / "runs" / arm / stage / "ckpt_last.pt"
        if Path(os.path.abspath(f[0])) != want:
            raise ValueError(f"record path {f[0]} is not {want}")
        if want.is_symlink() or not want.is_file():
            raise ValueError(f"{want} is not a regular file")
        s = sha256_file(want)
        if s != f[1]:
            raise ValueError(f"sha256 of {want} is {s[:12]}, the bank_ready record says {f[1][:12]} (the file changed after it was announced)")
        done = self.D / "runs" / arm / stage / f"ARM_{stage}_DONE"
        if done.is_file():
            d = done.read_text().split()
            if d and d[0] != s:
                raise ValueError(f"{done.name} sha {d[0][:12]} != file sha {s[:12]}")
        return arm, stage, want, s

    def bank_one(self, tag):
        """'BANKED' | 'ALREADY' | raises ValueError (refused: nothing written)."""
        arm, stage, ckpt, sha = self.check_record(tag)
        mk = self.ok / f"{tag}.sha256"
        if mk.is_file():
            if mk.read_text().split()[0] == sha:
                return "ALREADY"
            raise ValueError(f"CONFLICT: {mk} holds {mk.read_text().split()[0][:12]} but the checkpoint is {sha[:12]}; not overwritten")
        pairs = self.files_for(arm, stage, ckpt)
        sums = {r: (Path(l).stat().st_size, sha256_file(l)) for l, r in pairs}
        err, rev = None, None
        for k in range(1, self.max_attempts + 1):
            try:
                rev = self.hub.upload_many(pairs, message=f"bank {tag} (sha256 {sha[:12]})")
                bad = []
                for l, r in pairs:
                    n, s, lfs = self.hub.remote_size_sha(r, rev)
                    if (n, s) != sums[r]:
                        bad.append(f"{r}: remote size/sha {n}/{s[:12]} != local {sums[r][0]}/{sums[r][1][:12]}")
                    if r.endswith("ckpt_last.pt") and lfs != sha:
                        bad.append(f"{r}: hub LFS id {None if lfs is None else lfs[:12]} != sha256 {sha[:12]}")
                if not bad:
                    break
                err = "; ".join(bad)
            except Exception as e:   # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:300]}"
            self.log(f"[banker] {tag} attempt {k}/{self.max_attempts}: {err}")
            self._led(tag=tag, status="ATTEMPT-FAIL", attempt=k, error=err)
            rev = None
            if k < self.max_attempts:
                self.sleep(5 * k)
        if rev is None:
            raise ValueError(f"upload not verified after {self.max_attempts} attempts: {err}")
        self.ok.mkdir(parents=True, exist_ok=True)
        body = f"{sha} {self.hub.repo_id} {rev} {arm}/{stage}/ckpt_last.pt {sums[arm + '/' + stage + '/ckpt_last.pt'][0]} {utc()}\n"
        tmp = self.ok / f".{tag}.sha256.part"
        tmp.write_text(body)
        os.replace(tmp, mk)
        self._led(tag=tag, status="BANKED", repo=self.hub.repo_id, revision=rev, sha256=sha, files={r: list(v) for r, v in sums.items()})
        return "BANKED"

    def run(self, only=None):
        self.hub.ensure_repo()
        res = {}
        for tag in only or self.pending():
            try:
                res[tag] = self.bank_one(tag)
            except Exception as e:   # noqa: BLE001
                res[tag] = "REFUSED: " + str(e)
                self._led(tag=tag, status="REFUSED", error=str(e)[:400])
            self.log(f"[banker] {tag}: {res[tag]}")
        return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/root/dscale10k")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repo-name", default=REPO_NAME)
    ap.add_argument("--local-hub", help="TEST ONLY")
    a = ap.parse_args(argv)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    lk = open(root / "banker.lock", "w")
    try:
        import fcntl                                   # POSIX only (the pod); the tests call Banker directly
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[banker] another banker holds banker.lock; exiting 0 (it banks the same set)")
        return 0
    hub = LocalHub(a.local_hub) if a.local_hub else Hub(repo_name=a.repo_name)
    res = Banker(root, hub).run(a.only)
    print("BANKER-SUMMARY " + json.dumps(res, sort_keys=True))
    return 0 if all(v in ("BANKED", "ALREADY") for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
