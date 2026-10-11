#!/usr/bin/env python
"""HF egress for pod chains (eval/PREREG_SFT4R.md, Changed 3). Push, then prove the push by the REMOTE listing.

Why this exists: STOP_CONSEQUENCES_2026-10-11.md. SFT-4's checkpoints and logs lived only on the pod, the pod stopped,
and nothing beyond update 400 survived. A chain's output exists once the remote listing shows it with the local size,
and not before.

Subcommands
  push    push files to a PRIVATE HF repo; writes `ZZEGRESS_OK <tag> ...` / `ZZEGRESS_FAIL <tag> <reason>` to --chain-log
  gate    G-EGR: push the gate's own JSON (must be OK) and a deliberate regression to a repo the token cannot write
          (must be REFUSED BY THE SERVER). Prints ZZEGR_PASS / ZZEGR_FAIL.
  verify  read-only: the listing check against a known upload (model_final.pt), plus a size+1 control that must read
          as a mismatch. Records the account's private storage total.
  watch   beside a training run: push every new scorer_states checkpoint plus the logs; halt on repeated failure.

Auth: the host's own HF login (HF_TOKEN or the cached token) or --token-file, read in place (first hf_ match). The
token is never printed, never written and never in argv; exception text is scrubbed of anything token-shaped.
Quota (binding, PI): the account's private total (API listing, usedStorage) plus this push must stay inside
--allowance-bytes (default 1e12 = the Pro allowance). An unknown total FAILS CLOSED.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")
ALLOWANCE = 1e12


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize(s) -> str:
    return TOKEN_RE.sub("hf_***", str(s))


def _append(path, text: str) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except Exception:
        pass


def _read_token(path):
    if not path:
        return None
    m = re.search(r"hf_[A-Za-z0-9]+", Path(path).read_text(encoding="utf-8", errors="ignore"))
    if not m:
        raise SystemExit("no hf_ token found in --token-file")      # never echo the file
    return m.group(0)


def make_api(token_file=None):
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    from huggingface_hub import HfApi
    return HfApi(token=_read_token(token_file))


def _add_op(path_in_repo, local):
    from huggingface_hub import CommitOperationAdd
    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local))


def _lfs_field(it, field):
    lfs = getattr(it, "lfs", None)
    if lfs is None:
        return None
    return lfs.get(field) if isinstance(lfs, dict) else getattr(lfs, field, None)


def _item_size(it):
    """The file's real size: the LFS size when the file is in LFS (the bare `size` may be the pointer's)."""
    size = getattr(it, "size", None)
    if size is None:                                                  # a folder
        return None
    ls = _lfs_field(it, "size")
    return int(ls if ls is not None else size)


def _remote_sizes(api, repo, repo_type):
    out = {}
    for it in api.list_repo_tree(repo, recursive=True, repo_type=repo_type):
        s = _item_size(it)
        if s is not None:
            out[it.path] = s
    return out


def private_used_bytes(api, author):
    """Sum of usedStorage over the author's PRIVATE models, datasets and spaces. Raises if any value is unknown.

    MEASURED 2026-10-11 against the live API: the LIST endpoints reject expand=["usedStorage"] (HTTP 400, "Invalid
    option"); only the per-repo info endpoint serves it. So: list with expand=["private"], then one repo_info call per
    private repo.
    """
    total = 0
    for repo_type, lister in (("model", api.list_models), ("dataset", api.list_datasets), ("space", api.list_spaces)):
        for r in lister(author=author, expand=["private"]):
            rid = getattr(r, "id", "?")
            priv = getattr(r, "private", None)
            if priv is None:
                raise RuntimeError(f"private flag unknown for {rid}")
            if not priv:
                continue
            u = getattr(api.repo_info(rid, repo_type=repo_type, expand=["usedStorage"]), "used_storage", None)
            if u is None:
                raise RuntimeError(f"used_storage unknown for {rid}")
            total += int(u)
    return total


def _finish(ok, tag, reason, rec, chain_log, egress_log):
    reason = _sanitize(reason)
    line = f"{_ts()} {'ZZEGRESS_OK' if ok else 'ZZEGRESS_FAIL'} {tag} {reason}"
    _append(chain_log, line)
    _append(egress_log, json.dumps({"ts": _ts(), "tag": tag, "ok": ok, "reason": reason, **rec}))
    print(line, flush=True)
    return ok, reason


def push(api, repo, repo_type, files, tag, chain_log=None, egress_log=None, names=None, prefix="",
         max_push_bytes=None, allowance_bytes=ALLOWANCE, author=None, message=None):
    """Push `files` to a private repo and verify them by the remote listing. Never raises; returns (ok, reason)."""
    files = [Path(f) for f in files]
    names = list(names) if names else [f.name for f in files]
    paths = [f"{prefix.strip('/')}/{n}" if prefix else n for n in names]
    rec = {"repo": repo, "repo_type": repo_type, "paths": paths}
    try:
        if len(names) != len(files):
            return _finish(False, tag, f"bad_names:{len(names)}!={len(files)}", rec, chain_log, egress_log)
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            return _finish(False, tag, "missing_local:" + ",".join(missing), rec, chain_log, egress_log)
        sizes = {p: f.stat().st_size for p, f in zip(paths, files)}
        n = sum(sizes.values())
        rec["sizes"] = sizes
        if max_push_bytes is not None and n > max_push_bytes:
            return _finish(False, tag, f"over_budget:{n}>{int(max_push_bytes)}", rec, chain_log, egress_log)
        try:
            author = author or api.whoami()["name"]
            used = private_used_bytes(api, author)
        except Exception as e:
            return _finish(False, tag, f"usage_unknown:{type(e).__name__}:{_sanitize(e)[:200]}", rec, chain_log,
                           egress_log)
        rec["used"] = used
        if used + n > allowance_bytes:
            return _finish(False, tag, f"over_allowance:{used}+{n}>{int(allowance_bytes)}", rec, chain_log,
                           egress_log)
        api.create_repo(repo, private=True, repo_type=repo_type, exist_ok=True)
        if not api.repo_info(repo, repo_type=repo_type).private:
            return _finish(False, tag, "not_private", rec, chain_log, egress_log)
        ops = [_add_op(p, f) for p, f in zip(paths, files)]
        api.create_commit(repo, operations=ops, commit_message=message or f"egress {tag}", repo_type=repo_type)
        remote = _remote_sizes(api, repo, repo_type)
        miss = [p for p in paths if p not in remote]
        if miss:
            return _finish(False, tag, "missing_remote:" + ",".join(miss), rec, chain_log, egress_log)
        bad = [p for p in paths if remote[p] != sizes[p]]
        if bad:
            return _finish(False, tag, "size_mismatch:" + ",".join(f"{p}={remote[p]}!={sizes[p]}" for p in bad),
                           rec, chain_log, egress_log)
        return _finish(True, tag, f"{len(paths)} files {n} B", rec, chain_log, egress_log)
    except Exception as e:
        return _finish(False, tag, f"exception:{type(e).__name__}:{_sanitize(e)[:200]}", rec, chain_log, egress_log)


def gate(api, repo, repo_type, out_dir, chain_log=None, egress_log=None,
         regression_repo="google/tanitad-egress-regression-must-fail", allowance_bytes=ALLOWANCE):
    """G-EGR. PASS iff the gate JSON reaches `repo` (by the listing) AND a push to `regression_repo` is refused BY THE
    SERVER (reason `exception:...`). A refusal by our own pre-checks would prove nothing about the token."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = out / "g_egr.json"
    g.write_text(json.dumps({"check": "G-EGR", "ts": _ts(), "repo": repo, "repo_type": repo_type}, indent=1),
                 encoding="utf-8")
    ok_main, r_main = push(api, repo, repo_type, [g], "G-EGR", chain_log, egress_log, names=["g_egr.json"],
                           prefix="gate", allowance_bytes=allowance_bytes)
    ok_reg, r_reg = push(api, regression_repo, "model", [g], "G-EGR-REGRESSION", chain_log, egress_log,
                         names=["g_egr.json"], prefix="gate", allowance_bytes=allowance_bytes)
    refused = (not ok_reg) and r_reg.startswith("exception:")
    passed = ok_main and refused
    res = {"check": "G-EGR", "ts": _ts(), "pass": passed, "main": {"repo": repo, "ok": ok_main, "reason": r_main},
           "regression": {"repo": regression_repo, "ok": ok_reg, "reason": r_reg, "refused_by_server": refused}}
    (out / "g_egr_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    line = f"{_ts()} {'ZZEGR_PASS' if passed else 'ZZEGR_FAIL'} main_ok={ok_main} regression_refused={refused}"
    _append(chain_log, line)
    print(line, flush=True)
    return 0 if passed else 1


def verify(api, repo, local, out_json, author=None):
    """Read-only check of the listing logic against a file whose upload is already known to be good."""
    local = Path(local)
    lsize = local.stat().st_size
    rows = []
    for it in api.list_repo_tree(repo, recursive=True, repo_type="model"):
        if _item_size(it) is None or it.path.rsplit("/", 1)[-1] != local.name:
            continue
        s = _item_size(it)
        rows.append({"path": it.path, "size_attr": getattr(it, "size", None), "lfs_size": _lfs_field(it, "size"),
                     "lfs_pointer_size": _lfs_field(it, "pointer_size"), "listing_size": s,
                     "match_local": s == lsize, "control_plus1_reads_mismatch": s != lsize + 1})
    author = author or api.whoami()["name"]
    try:
        used, used_err = private_used_bytes(api, author), None
    except Exception as e:
        used, used_err = None, f"{type(e).__name__}:{_sanitize(e)[:200]}"
    passed = bool(rows) and all(r["match_local"] and r["control_plus1_reads_mismatch"] for r in rows)
    res = {"check": "egress-verify", "ts": _ts(), "repo": repo, "local": str(local), "local_size": lsize,
           "rows": rows, "private_used_bytes": used, "private_used_error": used_err,
           "allowance_bytes": ALLOWANCE, "pass": passed}
    Path(out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{'ZZVERIFY_PASS' if passed else 'ZZVERIFY_FAIL'} rows={len(rows)} private_used={used}", flush=True)
    return 0 if passed else 1


def _read_update_torch(p):
    import torch
    return int(torch.load(str(p), map_location="cpu", weights_only=True)["update"])


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _kill(pid):
    try:
        os.killpg(pid, signal.SIGTERM)                               # the chain starts the run under setsid
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


class Watcher:
    """Push every new `scorer_states_last.pt` the trainer writes (refe/scorer_finetune.py saves it, NON-atomically,
    at every held-out read and at the final update) plus snapshots of the logs.

    * A checkpoint is acted on only when its (size, mtime_ns) is unchanged across two polls, then COPIED, and the copy
      is kept only if the source did not change during it. The update number is read from the copy.
    * One push per checkpoint. A failed push leaves its states pending, and they ride with the next push (the retry).
    * HALT, the literal PREREG_SFT4R reading: "two consecutive failures stop the chain AFTER the next checkpoint is
      written". After the second consecutive failure the halt is ARMED; at the next checkpoint the watcher pushes once
      more and then kills the run, whether or not that last push succeeded.
    * The run is finished when `ZZSFT_DONE` is in the log or its pid is gone. The final push (tag `exit`) carries every
      unpushed state and all logs; its result is the watcher's exit code.
    """

    def __init__(self, out_dir, stage_dir, logs, push_fn, halt_pid=None, poll=60, sft_log=None, chain_log=None,
                 read_update=_read_update_torch, pid_alive=_pid_alive, kill=_kill, sleep=time.sleep):
        self.out = Path(out_dir)
        self.stage = Path(stage_dir)
        self.state = self.out / "scorer_states_last.pt"
        self.logs = [Path(p) for p in logs]
        self.sft_log = Path(sft_log) if sft_log else self.out / "sft.log"
        self.chain_log = chain_log
        self.push_fn, self.halt_pid, self.poll = push_fn, halt_pid, poll
        self.read_update, self.pid_alive, self.kill, self.sleep = read_update, pid_alive, kill, sleep
        self.prev_stat = None        # the stat seen at the previous poll
        self.done_stat = None        # the stat last acted on: never re-read
        self.pushed: set[int] = set()
        self.pending: list[tuple[int, Path]] = []
        self.fails = 0
        self.halt_armed = False
        self.killed = False

    def _say(self, text):
        line = f"{_ts()} {text}"
        _append(self.chain_log, line)
        print(line, flush=True)

    def _stat(self):
        try:
            st = self.state.stat()
            return (st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            return None

    def _stage_state(self, st):
        """Copy the settled state; return its update number, or None if it moved or cannot be read yet."""
        self.stage.mkdir(parents=True, exist_ok=True)
        tmp = self.stage / "scorer_states_incoming.pt"
        shutil.copy2(self.state, tmp)
        if self._stat() != st:                                         # rewritten during the copy
            tmp.unlink(missing_ok=True)
            return None
        try:
            upd = int(self.read_update(tmp))
        except Exception:
            tmp.unlink(missing_ok=True)
            return None
        self.done_stat = st
        if upd in self.pushed or any(u == upd for u, _ in self.pending):
            tmp.unlink(missing_ok=True)
            return None
        dst = self.stage / f"scorer_states_u{upd}.pt"
        os.replace(tmp, dst)
        self.pending.append((upd, dst))
        return upd

    def _bundle(self, tag):
        files = [p for _, p in self.pending]
        names = [f"scorer_states/{p.name}" for _, p in self.pending]
        d = self.stage / "logs" / tag
        d.mkdir(parents=True, exist_ok=True)
        for lp in self.logs:
            if lp.is_file():
                dst = d / lp.name
                shutil.copy2(lp, dst)
                files.append(dst)
                names.append(f"logs/{tag}/{lp.name}")
        return files, names

    def _push(self, tag):
        files, names = self._bundle(tag)
        ok, reason = self.push_fn(files, names, tag)
        if ok:
            self.pushed.update(u for u, _ in self.pending)
            self.pending.clear()
            self.fails = 0
        else:
            self.fails += 1
        return ok, reason

    def poll_once(self):
        st = self._stat()
        settled = st is not None and st == self.prev_stat and st != self.done_stat
        self.prev_stat = st
        if not settled:
            return None
        upd = self._stage_state(st)
        if upd is None:
            return None
        halt_now = self.halt_armed
        self._push(f"u{upd}")
        if halt_now and not self.killed:
            self.killed = True
            self._say(f"ZZEGRESS_HALT u{upd}: two consecutive failed pushes, then this checkpoint; killing the run")
            if self.halt_pid:
                self.kill(self.halt_pid)
        elif self.fails >= 2 and not self.halt_armed:
            self.halt_armed = True
            self._say(f"ZZEGRESS_HALT_ARMED u{upd}: two consecutive failed pushes; the run stops after the next checkpoint")
        return upd

    def _finished(self):
        try:
            with open(self.sft_log, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 65536))
                if b"ZZSFT_DONE" in fh.read():
                    return True
        except FileNotFoundError:
            pass
        return self.halt_pid is not None and not self.pid_alive(self.halt_pid)

    def run(self):
        while not self._finished():
            self.poll_once()
            self.sleep(self.poll)
        st = self._stat()                                              # the state written at the final update
        if st is not None and st != self.done_stat:
            self._stage_state(st)
        ok, reason = self._push("exit")
        self._say(f"ZZEGRESS_WATCH_END ok={ok} pushed={sorted(self.pushed)} pending={[u for u, _ in self.pending]}")
        return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--token-file", default=None, help="read in place; the first hf_ match is used")
        p.add_argument("--allowance-bytes", type=float, default=ALLOWANCE)
        p.add_argument("--chain-log", default=None)
        p.add_argument("--egress-log", default=None)

    p = sub.add_parser("push")
    common(p)
    p.add_argument("--repo", required=True)
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--prefix", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--max-push-bytes", type=float, default=None)
    p.add_argument("--author", default=None)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--names", nargs="+", default=None)

    p = sub.add_parser("gate")
    common(p)
    p.add_argument("--repo", required=True)
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--out", required=True)
    p.add_argument("--regression-repo", default="google/tanitad-egress-regression-must-fail")

    p = sub.add_parser("verify")
    p.add_argument("--token-file", default=None)
    p.add_argument("--repo", default="Sayood/tanitad-refe-vitl16-navtrain10-grow-final")
    p.add_argument("--local", default="E:/Projects/TanitAD/data/refe_runs_eval/model_final.pt")
    p.add_argument("--author", default=None)
    p.add_argument("--out-json", required=True)

    p = sub.add_parser("watch")
    common(p)
    p.add_argument("--repo", required=True)
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--out", required=True, help="the trainer's --out (holds scorer_states_last.pt and sft.log)")
    p.add_argument("--stage", required=True)
    p.add_argument("--log", nargs="*", default=[])
    p.add_argument("--halt-pid", type=int, required=True)
    p.add_argument("--poll", type=float, default=60)
    p.add_argument("--max-push-bytes", type=float, default=None)

    a = ap.parse_args(argv)
    api = make_api(a.token_file)
    if a.cmd == "push":
        ok, _ = push(api, a.repo, a.repo_type, a.files, a.tag, a.chain_log, a.egress_log, names=a.names,
                     prefix=a.prefix, max_push_bytes=a.max_push_bytes, allowance_bytes=a.allowance_bytes,
                     author=a.author)
        return 0 if ok else 1
    if a.cmd == "gate":
        return gate(api, a.repo, a.repo_type, a.out, a.chain_log, a.egress_log, a.regression_repo, a.allowance_bytes)
    if a.cmd == "verify":
        return verify(api, a.repo, a.local, a.out_json, a.author)

    def push_fn(files, names, tag):
        return push(api, a.repo, a.repo_type, files, tag, a.chain_log, a.egress_log, names=names,
                    max_push_bytes=a.max_push_bytes, allowance_bytes=a.allowance_bytes)

    return Watcher(a.out, a.stage, a.log, push_fn, halt_pid=a.halt_pid, poll=a.poll, chain_log=a.chain_log).run()


if __name__ == "__main__":
    sys.exit(main())
