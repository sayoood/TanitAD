#!/usr/bin/env python3
"""The REFe launch-gate TOKEN: mint, verify, and the primitives they stand on.  Route and rationale: eval/REFE_LAUNCH_GATE.md.

SAME SCHEMA AND KEY AS THE PROGRAMME'S GENERIC GATE (`tanitad.launch_gate/1`, banked at products/P4-training-pipelines/
2026-09-27-v7f-refav1-state/v7f_ckpt/code/fix/stack/scripts/launch_gate.py): `canonical_json`, `argv_sha256`, `sign`, `signature_ok`,
`fingerprint`, `fp_key` are byte-identical re-implementations, and test_launch_gate.py loads the banked module (when present) and
asserts equal outputs, so a token minted here verifies there once that gate knows the refe profiles.  Never hand-written: the
HMAC refuses an edited token.

WHY A SEPARATE MODULE (not a copy of the 7,806-line generic gate): the pod ships only `code/ refe/ splits/` (no git, no `stack/`),
and REFe is not a `tanitad.train` trainer, so everything the generic gate wraps around the token (trainer loader, declared_vs_built,
smoke runners) does not apply.  What REFe needs from it is exactly this: a token bound to (profile, commit, tree, argv, data).

THE COMMIT WITHOUT GIT ON THE POD: `ship-manifest` (dev box) writes SHIP_MANIFEST.json = {commit, files: {path: sha256}} after proving
every shipped file equals its blob at that commit; the pod's `mint` proves the pod's tree equals the manifest.  So the commit in the
token is a measured fact about the bytes, not a label.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

GATE_SCHEMA = "tanitad.launch_gate/1"
KEY_ENV = "TANITAD_GATE_KEY_FILE"
SHIP_MANIFEST = "SHIP_MANIFEST.json"
TREE_ROOTS = ("refe", "code", "splits")
_TREE_SUFFIX = (".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".patch")
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
_DIR_SKIP_NAMES = frozenset({"_v2manifest.pt", "_v2manifest.pt.tmp", "__pycache__"})


class GateError(RuntimeError):
    """A condition under which the gate cannot certify anything."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Key-sorted, compact, UTF-8: the ONE byte form every hash and HMAC here is taken over (== the generic gate's)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def argv_sha256(argv: list) -> str:
    """sha256 of the argv as an ORDERED JSON array (order is meaning; never sorted) (== the generic gate's)."""
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise GateError("argv must be a list of strings")
    return sha256_bytes(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def argv_from_string(s: str) -> list:
    """The shell word-splitting of an unquoted `$ARGS` (default IFS) -- what `train.py $ARGS` passes.  No path in ARGS holds a blank
    or a glob character; test_launch_gate pins that on the chain's real ARGS."""
    return s.split()


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ key and signature
def default_key_file() -> Path:
    env = os.environ.get(KEY_ENV)
    return Path(env) if env else Path.home() / ".config" / "tanitad" / "launch_gate.key"


def load_key(path=None, *, create: bool = True, forbid_under: Iterable = ()) -> bytes:
    """32 random bytes, hex on disk, created on first use with mode 0600.  Refused inside a code tree: a key that can be shipped is
    not a key.  On the pod the key is created by the first gate run on THAT pod; a token minted elsewhere does not verify there."""
    p = Path(path) if path else default_key_file()
    rp = os.path.normcase(str(p.resolve()))
    for t in forbid_under:
        rt = os.path.normcase(str(Path(t).resolve()))
        if rp == rt or rp.startswith(rt + os.sep):
            raise GateError(f"refusing a key file inside the code tree {t}: {p}")
    if not p.exists():
        if not create:
            raise GateError(f"no gate key at {p} (run the gate on this host first)")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(secrets.token_hex(32) + "\n")
    txt = p.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", txt):
        raise GateError(f"gate key {p} is not 64 hex chars")
    return bytes.fromhex(txt)


def key_id(key: bytes) -> str:
    return sha256_bytes(b"tanitad-launch-gate-key-id:" + key)[:16]


def sign(token_wo_sig: dict, key: bytes) -> str:
    return hmac.new(key, canonical_json(token_wo_sig), hashlib.sha256).hexdigest()


def signature_ok(token: dict, key: bytes) -> bool:
    body = {k: v for k, v in token.items() if k != "hmac_sha256"}
    got = str(token.get("hmac_sha256", ""))
    return len(got) == 64 and hmac.compare_digest(sign(body, key), got)


# ------------------------------------------------------------------------------------------ fingerprints
def fingerprint(path) -> dict:
    """A file: its sha256.  A directory: its LISTING (relpath + size of every file; contents NOT hashed -- a 449 GB bank is not re-read
    per launch; stated in `method`).  Missing: missing.  (== the generic gate's)"""
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "bytes": p.stat().st_size, "sha256": sha256_file(p), "method": "sha256-v1"}
    if p.is_dir():
        lines, n, total = [], 0, 0
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(d for d in dirnames if d not in _DIR_SKIP_NAMES)
            for fn in filenames:
                if fn in _DIR_SKIP_NAMES:
                    continue
                f = Path(dirpath) / fn
                st = f.stat()
                lines.append(f"{f.relative_to(p).as_posix()}\t{st.st_size}\n")
                n += 1
                total += st.st_size
        lines.sort()
        return {"kind": "dir", "n_files": n, "bytes": total, "listing_sha256": sha256_bytes("".join(lines).encode("utf-8")),
                "method": "dir-listing-v1 (relpath + size of every file; contents NOT hashed; _v2manifest.pt excluded -- a cache "
                          "rewrites it on first use)"}
    return {"kind": "missing"}


def fp_key(fp: dict) -> str:
    if fp.get("kind") == "file":
        return f"file:{fp.get('sha256')}"
    if fp.get("kind") == "dir":
        return f"dir:{fp.get('n_files')}:{fp.get('bytes')}:{fp.get('listing_sha256')}"
    return f"{fp.get('kind')}:{fp.get('sha256', fp.get('why', ''))}"


def flag_pairs(argv: list) -> list:
    """[(flag, [values...])] -- a token starting with `--` opens a flag, the tokens after it are its values."""
    out, cur = [], None
    for t in argv:
        if t.startswith("--"):
            cur = (t, [])
            out.append(cur)
        elif cur is not None:
            cur[1].append(t)
    return out


def flag_values(argv: list, flag: str):
    for f, v in flag_pairs(argv):
        if f == flag:
            return v
    return None


def data_manifest(argv: list, data_flags: Iterable[str], gate_inputs: dict | None = None, path_map=None) -> dict:
    """Every input the launch READS, fingerprinted on THIS host: the value of each data flag in the argv, plus the gate's own inputs
    (`gate:*`).  A data flag that is passed and missing on the host is a finding (`data_problems`)."""
    man: dict = {}
    want = set(data_flags)
    for flag, vals in flag_pairs(argv):
        if flag in want and vals:
            local = vals[0]
            for a, b in (path_map or []):
                if local.startswith(a):
                    local = b + local[len(a):]
            man[flag] = {"path": vals[0], **({"local": local} if local != vals[0] else {}), **fingerprint(local)}
    for key, path in sorted((gate_inputs or {}).items()):
        man[key] = {"path": str(path), **fingerprint(path)}
    return man


def data_problems(data: dict) -> list:
    return [f"data input {k} ({v.get('path', '')}) is {v.get('kind')} on this host" for k, v in sorted(data.items())
            if v.get("kind") in ("missing", "unresolved")]


# ------------------------------------------------------------------------------------------ the tree
def tree_manifest(pkg) -> dict:
    """{relpath: sha256} of the shipped tree: refe/ code/ splits/ under `pkg`, source-like files only, never __pycache__."""
    pkg = Path(pkg)
    out: dict = {}
    for root in TREE_ROOTS:
        base = pkg / root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for fn in sorted(filenames):
                if fn.endswith(_TREE_SUFFIX):
                    f = Path(dirpath) / fn
                    out[f.relative_to(pkg).as_posix()] = sha256_file(f)
    return out


def tree_digest(man: dict) -> str:
    return sha256_bytes(canonical_json(man))


def tree_diff(want: dict, got: dict, limit: int = 12) -> dict:
    return {"missing": sorted(set(want) - set(got))[:limit], "extra": sorted(set(got) - set(want))[:limit],
            "changed": sorted(k for k in set(want) & set(got) if want[k] != got[k])[:limit]}


def ship_manifest(pkg, *, commit: str | None = None, git_dir: str | None = None, repo_prefix: str = "") -> dict:
    """DEV BOX: write the manifest the pod's `mint` will be held to.  With `commit`, every shipped file must equal its blob at that
    commit (`git show <commit>:<repo_prefix><path>`): a file that is not landed, or edited since, REFUSES -- so the commit in the token
    names bytes that exist in the repository."""
    man = tree_manifest(pkg)
    if commit:
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise GateError("commit must be a full 40-hex sha")
        bad = []
        for rel, sha in sorted(man.items()):
            cmd = ["git"] + (["--git-dir", git_dir] if git_dir else []) + ["show", f"{commit}:{repo_prefix}{rel}"]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                bad.append(f"{rel}: not in {commit[:12]}")
            elif sha256_bytes(r.stdout) != sha:
                bad.append(f"{rel}: differs from its blob at {commit[:12]}")
        if bad:
            raise GateError(f"{len(bad)} shipped file(s) are not the committed bytes: {bad[:6]}")
    return {"schema": "refe.ship_manifest/1", "commit": commit, "created_utc": utc_now(), "files": man,
            "tree_sha256": tree_digest(man)}


# ------------------------------------------------------------------------------------------ mint and verify
def binding_of(profile: str, commit: str | None, tree_sha: str, argv: list) -> dict:
    return {"profile": profile, "commit": commit, "tree_sha256": tree_sha, "argv_sha256": argv_sha256(argv)}


def mint(*, profile: str, commit: str | None, tree_man: dict, argv: list, trainer: str, data: dict, checks: dict, reasons: list,
         key: bytes, out_dir, gate_inputs: dict | None = None, gate_self_sha: str = "", verdict_override: str | None = None):
    """Assemble, sign and write the token.  PASS only when every entry of `checks` is PASS and there is no reason; the file is
    `<verdict>_<commit12 or tree12>.json`, exactly ONE current token per binding (a stale PASS is DEMOTED to superseded/)."""
    tsha = tree_digest(tree_man)
    binding = {**binding_of(profile, commit, tsha, argv), "argv": list(argv), "trainer": trainer, "gate_inputs": dict(gate_inputs or {}),
               "data": data, "data_sha256": sha256_bytes(canonical_json(data))}
    reasons = list(reasons)
    bad = [c for c, v in checks.items() if v.get("status") != "PASS"]
    if bad:
        reasons.append(f"checks not PASS: {bad}")
    verdict = verdict_override or ("PASS" if not reasons else "FAIL")
    token = {"schema": GATE_SCHEMA, "kind": "launch-gate-token", "verdict": verdict, "binding": binding, "checks": checks,
             "reasons": reasons, "gate_sha256": gate_self_sha, "created_utc": utc_now(), "key_id": key_id(key)}
    token["hmac_sha256"] = sign(token, key)
    out = Path(out_dir)
    tag = (commit or "nocommit")[:12] if commit else tsha[:12]
    stamp = utc_now().replace(":", "")
    for other in ("PASS", "FAIL", "INCOMPLETE"):
        op = out / f"{other}_{tag}.json"
        if op.exists():
            (out / "superseded").mkdir(parents=True, exist_ok=True)
            os.replace(op, out / "superseded" / f"{other}_{tag}.{stamp}.json")
    path = write_json(out / f"{verdict}_{tag}.json", token)
    return verdict, path, token


def verify_token(token_path, argv: list, pkg, *, key: bytes, profile: str | None = None, data_flags: Iterable[str] = (),
                 commit: str | None = None, path_map=None):
    """THE SUPERVISOR'S SIDE.  Every field the token binds is RE-MEASURED here; any difference refuses.  Returns (ok, reasons, info)."""
    reasons: list = []
    info: dict = {"token": str(token_path)}
    try:
        tok = read_json(token_path)
    except Exception as e:                                                  # noqa: BLE001 -- a refusal, never a pass
        return False, [f"token unreadable: {type(e).__name__}: {e}"], info
    if tok.get("kind") != "launch-gate-token" or tok.get("schema") != GATE_SCHEMA:
        reasons.append("not a launch-gate token")
    if not signature_ok(tok, key):
        reasons.append(f"HMAC does not verify -- edited, or signed with another key (token key_id {tok.get('key_id')}, this host's "
                       f"{key_id(key)})")
    if tok.get("verdict") != "PASS":
        reasons.append(f"token verdict is {tok.get('verdict')!r}, not PASS")
    b = tok.get("binding") or {}
    info["binding"] = {k: b.get(k) for k in ("profile", "commit", "tree_sha256", "argv_sha256")}
    if profile and b.get("profile") != profile:
        reasons.append(f"token profile {b.get('profile')!r} != launch profile {profile!r}")
    if commit and b.get("commit") != commit:
        reasons.append(f"token commit {str(b.get('commit'))[:12]} != launch commit {commit[:12]}")
    a_sha = argv_sha256(argv)
    info["argv_sha256"] = a_sha
    if b.get("argv_sha256") != a_sha or b.get("argv") != argv:
        changed = [f"{x} -> {y}" for x, y in zip(b.get("argv") or [], argv) if x != y][:5]
        reasons.append(f"argv differs from the gated argv (sha {a_sha[:12]} vs {str(b.get('argv_sha256'))[:12]}; "
                       f"{changed or 'length differs'})")
    try:
        tsha = tree_digest(tree_manifest(pkg))
        info["tree_sha256"] = tsha
        if tsha != b.get("tree_sha256"):
            reasons.append(f"code tree differs from the gated tree ({tsha[:16]} vs {str(b.get('tree_sha256'))[:16]})")
    except Exception as e:                                                  # noqa: BLE001
        reasons.append(f"cannot measure the code tree: {e}")
    flags = list(data_flags) or sorted({k for k in (b.get("data") or {}) if k.startswith("--")})
    data = data_manifest(argv, flags, b.get("gate_inputs") or {}, path_map)
    want = b.get("data") or {}
    d_bad = [k for k in sorted(set(want) | set(data)) if k not in want or k not in data or fp_key(want[k]) != fp_key(data[k])]
    info["data_differs"] = d_bad
    if d_bad:
        reasons.append(f"data differs from the gated data on {d_bad[:8]}")
    return not reasons, reasons, info
