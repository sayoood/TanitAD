#!/usr/bin/env python3
"""Pod <-> HF traffic for the P2 re-run chain (refe/p2r_pod.sh, eval/PREREG_P2_AMENDMENT_1.md). Built on hf_egress.py.

Why it exists (STOP_CONSEQUENCES_2026-10-11.md): the previous pod stopped and took every output that lived only on it.
A P2 output exists once the REMOTE listing shows it with the local size, and not before.

  restore   pull model_final + the two training banks from the private HF repos, verify each against a PINNED md5,
            extract with member validation. Prints ZZRESTORE_OK / ZZRESTORE_FAIL <reason>.
  recover   on a NEW pod: pull back what a lost pod pushed (PDM labels + their log; ckpt_last.pt and the run files), each
            checked against its sha256 / size. Prints ZZRECOVER_OK labels=yes|no ckpt=yes|no.
  watch     beside the chain: every --interval-min push the small files (logs, metrics, summary); every --heavy-min
            also the checkpoints (a pod loss then costs the 20-min checkpoint interval + this cadence). Failures are logged (ZZSYNC_DEGRADED after 5 in a row) and never stop the run.
  tick      one non-forced pass of `watch`, checkpoints included: the chain's exit trap uses it so a REFUSED line lands.
  final     the hard gate: push EVERYTHING, require model_final.pt's remote sha256 == summary.json's. Prints
            ZZP2R_FINAL_OK / ZZP2R_FINAL_FAIL <reason>, exit code 0 / 1.
  pdm-push  tar the PDM labels (cost ~11 h of CPU) + a sha256 MANIFEST and push them BEFORE training starts.

Auth: HF_TOKEN from the environment (hf_egress.make_api). The token is never an argument, never printed, never written.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_egress as E  # noqa: E402

MODEL_REPO = "Sayood/tanitad-refe-vitl16-navtrain10-grow-final"       # private; STOP_CONSEQUENCES_2026-10-11.md
DATA_REPO = "Sayood/tanitad-refe-navtrain-banks"                      # private dataset; same source
OUT_REPO = "Sayood/tanitad-refe-p2-joint-pdm"                         # private; created by the first push

# stem -> (md5, bytes, [extensions the file may carry]). md5s are the pod-computed MANIFEST values re-verified on the dev
# box against the banked copies in D:/Projects/TanitAD-artifacts (2026-10-11); a download that disagrees is REFUSED.
PINS = {
    "model_final": ("b54773f8c302b2ec8ff4c5971d1e1370", 1288484600, [".pt"]),
    "refe_train_grow": ("e07c7d6a5c0156f31a5de3af0843c325", 1043549966, [".tar.zst", ".tar"]),
    "onpolicy_sets_logs_full": ("4fc979939c5f9b5140ba59a952da77fb", 2070541308, [".tar.zst", ".tar"]),
}
# tar root -> where it is extracted (relative to --work). MEASURED roots: train_grow/...; and for the on-policy bank
# (MEASURED 2026-10-11 on the P2 pod, `tar -tv` of the pinned 2,070,541,308 B archive, 31 members) sets/ 9, logs/ 11,
# pids/ 6, smoke_archive/ 5. The first pin listed only sets/ and the restore REFUSED "unexpected_root:logs" on the pod.
# Only sets/ is read downstream; the other three are the bank's own provenance and are kept with it.
EXTRACT = {
    "refe_train_grow": ("refe_navtrain", {"train_grow"}),
    "onpolicy_sets_logs_full": ("refe_navtrain/onpolicy", {"sets", "logs", "pids", "smoke_archive"}),
}
INIT_REL = "refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt"      # the P2 --init-from path, byte for byte


def _hash(path, algo, chunk=1 << 22):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _say(msg):
    print(E._sanitize(msg), flush=True)


# ---------------------------------------------------------------- restore
def zstd_stream(path):
    """(readable stream, closer). Tries python 3.14 `compression.zstd`, the zstandard module, then the zstd binary."""
    try:
        from compression import zstd
        return zstd.open(path, "rb"), None
    except Exception:
        pass
    try:
        import zstandard
        fh = open(path, "rb")
        return zstandard.ZstdDecompressor().stream_reader(fh), fh
    except Exception:
        pass
    exe = shutil.which("zstd")
    if exe:
        p = subprocess.Popen([exe, "-dc", str(path)], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        return p.stdout, p
    raise RuntimeError("no zstd decoder: need the zstd binary, the zstandard module, or python 3.14")


def member_verdict(m, roots):
    """(ok, reason). Regular files and directories only; relative paths; first component in `roots`."""
    n = m.name.replace("\\", "/")
    while n.startswith("./"):
        n = n[2:]
    if n in ("", "."):
        return True, ""
    parts = n.split("/")
    if n.startswith("/") or ".." in parts or ":" in parts[0]:
        return False, "unsafe_path:" + n
    if roots and parts[0] not in roots:
        return False, "unexpected_root:" + parts[0]
    if not (m.isfile() or m.isdir()):
        return False, "unsupported_member_type:" + n
    return True, ""


def safe_extract(archive, dest, roots):
    """Extract a .tar / .tar.zst into dest, refusing any unsafe member BEFORE writing it. Returns (files, bytes)."""
    archive = str(archive)
    os.makedirs(dest, exist_ok=True)
    closer = None
    if archive.endswith(".zst"):
        stream, closer = zstd_stream(archive)
        tar = tarfile.open(fileobj=stream, mode="r|")
    else:
        tar = tarfile.open(archive, mode="r|*")                     # plain .tar or .tar.gz
    n_files = n_bytes = 0
    try:
        for m in tar:
            ok, why = member_verdict(m, roots)
            if not ok:
                raise ValueError(why)
            try:
                tar.extract(m, dest, filter="data")
            except TypeError:                                  # python < 3.12 has no filter= (member already validated)
                tar.extract(m, dest)
            if m.isfile():
                n_files += 1
                n_bytes += m.size
    finally:
        tar.close()
        if closer is not None:
            closer.terminate() if hasattr(closer, "terminate") else closer.close()
    return n_files, n_bytes


def default_fetch(repo, repo_type, path, local_dir):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=path, repo_type=repo_type, local_dir=local_dir))


def pick_remote(sizes, stem, want_bytes, exts):
    """The remote path for `stem`: name match AND exact pinned size, else exact pinned size alone. [] -> not found."""
    names = {stem + e for e in exts}
    by_name = [p for p, s in sizes.items() if os.path.basename(p) in names and s == want_bytes]
    if by_name:
        return by_name
    return [p for p, s in sizes.items() if s == want_bytes]


def restore(api, work, fetch=default_fetch, pins=None, model_repo=MODEL_REPO, data_repo=DATA_REPO):
    pins = pins or PINS
    work = Path(work)
    dl = work / "hf_dl"
    marks = work / "hf_dl" / "extracted"
    marks.mkdir(parents=True, exist_ok=True)
    listing = {}
    for repo, rtype in ((model_repo, "model"), (data_repo, "dataset")):
        try:
            listing[repo] = E._remote_sizes(api, repo, rtype)
        except Exception as e:
            return False, f"list_failed:{repo}:{type(e).__name__}:{E._sanitize(e)[:160]}"
    plan = [("model_final", model_repo, "model"), ("refe_train_grow", data_repo, "dataset"),
            ("onpolicy_sets_logs_full", data_repo, "dataset")]
    for stem, repo, rtype in plan:
        md5, nbytes, exts = pins[stem]
        found = pick_remote(listing[repo], stem, nbytes, exts)
        if len(found) != 1:
            near = sorted(p for p in listing[repo] if stem in p)[:6]
            return False, f"remote_not_unique:{stem}:found={len(found)}:near={near}"
        local = dl / repo.replace("/", "__") / found[0]
        if not (local.is_file() and local.stat().st_size == nbytes and _hash(local, "md5") == md5):
            _say(f"{stem}: downloading {found[0]} ({nbytes} B)")
            try:
                got = Path(fetch(repo, rtype, found[0], str(dl / repo.replace("/", "__"))))
            except Exception as e:
                return False, f"download_failed:{stem}:{type(e).__name__}:{E._sanitize(e)[:160]}"
            local = got
        have = _hash(local, "md5")
        if have != md5 or local.stat().st_size != nbytes:
            return False, f"md5_mismatch:{stem}:{have}!={md5}"
        _say(f"{stem}: md5 {have} == pin")
        if stem == "model_final":
            dst = work / INIT_REL
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not (dst.is_file() and dst.stat().st_size == nbytes and _hash(dst, "md5") == md5):
                shutil.copyfile(local, dst)
            if _hash(dst, "md5") != md5:
                return False, "init_copy_mismatch"
            continue
        sub, roots = EXTRACT[stem]
        mk = marks / (stem + ".md5")
        if mk.is_file() and mk.read_text().strip() == md5:
            _say(f"{stem}: already extracted")
            continue
        try:
            nf, nb = safe_extract(local, work / sub, roots)
        except Exception as e:
            return False, f"extract_failed:{stem}:{type(e).__name__}:{E._sanitize(e)[:200]}"
        mk.write_text(md5)
        _say(f"{stem}: extracted {nf} files {nb} B into {sub}")
    tg = work / "refe_navtrain" / "train_grow"
    need = [(tg / "calib_table.json").is_file(), bool(glob.glob(str(tg / "scorer_targets*.jsonl"))),
            bool(glob.glob(str(tg / "targets_rank*.jsonl"))),
            bool(glob.glob(str(work / "refe_navtrain" / "onpolicy" / "sets" / "onpolicy_*.jsonl")))]
    if not all(need):
        return False, f"layout_incomplete:calib,scorer_targets,targets_rank,sets={need}"
    return True, "ok"


# ---------------------------------------------------------------- outbox (watch / final)
SMALL_RUN = ["summary.json", "metrics.jsonl", "config.json"]
HEAVY_RUN = ["ckpt_last.pt", "snap_epoch*.pt", "model_final.pt"]
SMALL_LOGS = ["*.log", "*.json", "gate/*.json", "gate/*.log"]
SMALL_CAP = 200 << 20


def collect(run, logs, heavy, tails=()):
    """[(source Path, repo name, is_heavy)] -- what exists now. Names: run/<f>, logs/<f>, logs/<tail name>."""
    out = []
    for pat in SMALL_RUN:
        out += [(Path(p), "run/" + os.path.basename(p), False) for p in glob.glob(os.path.join(run, pat))]
    for pat in SMALL_LOGS:
        out += [(Path(p), "logs/" + os.path.relpath(p, logs).replace("\\", "/"), False)
                for p in glob.glob(os.path.join(logs, pat))]
    if heavy:
        for pat in HEAVY_RUN:
            out += [(Path(p), "run/" + os.path.basename(p), True) for p in glob.glob(os.path.join(run, pat))]
    return [t for t in out if t[0].is_file() and (t[2] or t[0].stat().st_size <= SMALL_CAP)]


def _tail_to(src, dst, n):
    with open(src, "rb") as fh:
        lines = fh.readlines()[-n:]
    with open(dst, "wb") as fh:
        fh.writelines(lines)


def stage(items, outbox, state, force, tails=()):
    """Copy each changed file into the outbox (a stable snapshot: the size we record is the size we upload). Returns
    [(staged Path, name, (size, mtime_ns))]."""
    sd = Path(outbox) / "stage"
    if sd.exists():
        shutil.rmtree(sd, ignore_errors=True)
    sd.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (src, name, _heavy) in enumerate(items):
        st = src.stat()
        sig = [st.st_size, st.st_mtime_ns]
        if not force and state.get(name) == sig:
            continue
        dst = sd / f"{i:04d}__{os.path.basename(name)}"
        shutil.copyfile(src, dst)
        st2 = src.stat()
        if (st2.st_size, st2.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
            dst.unlink()                                     # the file moved under us: next tick
            continue
        out.append((dst, name, sig))
    for spec in tails:                                       # "path:name:nlines"
        p, nm, n = spec.rsplit(":", 2)
        if os.path.isfile(p):
            dst = sd / f"tail__{nm}"
            _tail_to(p, dst, int(n))
            out.append((dst, "logs/" + nm, None))
    return out


def tick(api, a, n, heavy, force=False, state_path=None):
    state_path = Path(state_path or Path(a.outbox) / "state.json")
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        state = {}
    items = collect(a.run, a.logs, heavy)
    staged = stage(items, a.outbox, state, force, tails=getattr(a, "tail", []) or [])
    if not staged:
        return True, "nothing_changed"
    ok, why = E.push(api, a.repo, "model", [s[0] for s in staged], f"p2r_{'final' if force else 'tick'}_{n}",
                     chain_log=a.chain_log, egress_log=os.path.join(a.outbox, "egress.jsonl"),
                     names=[s[1] for s in staged], max_push_bytes=a.max_push_bytes)
    if ok:
        for _d, name, sig in staged:
            if sig is not None:
                state[name] = sig
        state_path.write_text(json.dumps(state))
    shutil.rmtree(Path(a.outbox) / "stage", ignore_errors=True)
    return ok, why


def watch(api, a):
    os.makedirs(a.outbox, exist_ok=True)
    n, bad, last_heavy = 0, 0, 0.0
    while True:
        n += 1
        heavy = (time.time() - last_heavy) >= a.heavy_min * 60
        ok, why = tick(api, a, n, heavy)
        if ok and heavy:
            last_heavy = time.time()
        bad = 0 if ok else bad + 1
        if bad == 5:
            E._append(a.chain_log, f"{E._ts()} ZZSYNC_DEGRADED 5 consecutive failed pushes; last: {E._sanitize(why)[:160]}")
        waited = 0
        while waited < a.interval_min * 60:
            if os.path.exists(a.stop_file):
                return 0
            time.sleep(5)
            waited += 5
        if os.path.exists(a.stop_file):
            return 0


def remote_sha256(api, repo, path):
    for it in api.list_repo_tree(repo, recursive=True, repo_type="model"):
        if it.path == path:
            return E._lfs_field(it, "sha256")
    return None


def final(api, a):
    """Push everything (forced), then prove model_final.pt by sha256 against summary.json. Returns (ok, reason)."""
    run = Path(a.run)
    try:
        summ = json.loads((run / "summary.json").read_text())
    except Exception as e:
        return False, f"summary_unreadable:{type(e).__name__}"
    if summ.get("done") is not True:
        return False, "summary_not_done"
    fin = run / "model_final.pt"
    if not fin.is_file():
        return False, "model_final_missing"
    local_sha = _hash(fin, "sha256")
    if summ.get("model_final_sha256") and summ["model_final_sha256"] != local_sha:
        return False, "local_sha_disagrees_with_summary"
    os.makedirs(a.outbox, exist_ok=True)
    ok, why = tick(api, a, 0, True, force=True)
    if not ok:
        return False, "push_failed:" + why
    rs = remote_sha256(api, a.repo, "run/model_final.pt")
    if rs is None:
        E._append(a.chain_log, f"{E._ts()} ZZP2R_NOTE remote sha256 unavailable; size match only")
    elif rs != local_sha:
        return False, f"remote_sha_mismatch:{rs}!={local_sha}"
    return True, f"model_final sha256 {local_sha} remote={'sha256-verified' if rs else 'size-only'}"


def pdm_push(api, a):
    lab = Path(a.labels)
    files = sorted(glob.glob(str(lab / "pdm_*.jsonl")))
    if not files:
        return False, "no_pdm_files"
    os.makedirs(a.outbox, exist_ok=True)
    tgz = Path(a.outbox) / "pdm_train_labels.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        for f in files:
            tf.add(f, arcname="train_pdm/" + os.path.basename(f))
    man = {"files": {os.path.basename(f): os.path.getsize(f) for f in files}, "n_files": len(files),
           "tar_sha256": _hash(tgz, "sha256"), "tar_bytes": tgz.stat().st_size, "created": E._ts()}
    mp = Path(a.outbox) / "pdm_train_labels.MANIFEST.json"
    mp.write_text(json.dumps(man, indent=1))
    push_files, names = [tgz, mp], [tgz.name, mp.name]
    if getattr(a, "log", None) and os.path.isfile(a.log):
        # the relabeller's own log (its "served sets: N" and ZZRELABEL_PDM_DONE lines): a new pod re-runs the completeness
        # audit on the RESTORED files instead of trusting that they were complete
        lg = Path(a.outbox) / "pdm_train.log"
        shutil.copyfile(a.log, lg)
        push_files.append(lg)
        names.append(lg.name)
    return E.push(api, a.repo, "model", push_files, "p2r_pdm_labels", chain_log=a.chain_log,
                  egress_log=os.path.join(a.outbox, "egress.jsonl"), names=names, prefix="pdm_labels",
                  max_push_bytes=a.max_push_bytes)


# ---------------------------------------------------------------- recover (a NEW pod picks up where a lost one stopped)
def remote_meta(api, repo):
    """{path: (size, lfs sha256 or None)}. A repo that does not exist yet is {} (a first run); any other error raises."""
    try:
        return {it.path: (E._item_size(it), E._lfs_field(it, "sha256"))
                for it in api.list_repo_tree(repo, recursive=True, repo_type="model") if E._item_size(it) is not None}
    except Exception as e:
        if "NotFound" in type(e).__name__:
            return {}
        raise


def recover(api, work, run, repo=OUT_REPO, fetch=default_fetch):
    """Restore what a previous pod pushed to the OUTPUT repo: the PDM labels (with the relabeller's log, so the audit can be
    re-run) and the run directory (ckpt_last.pt, config, metrics, snapshots). Returns (ok, 'labels=yes|no ckpt=yes|no').
    Anything partial is 'no' (recompute / retrain); anything that fails its checksum is a FAILURE, never a silent 'no'."""
    work, run = Path(work), Path(run)
    meta = remote_meta(api, repo)
    dl = work / "hf_dl" / repo.replace("/", "__")
    got = {"labels": "no", "ckpt": "no"}
    tdir = work / "refe_sft2" / "train_pdm"
    need = ["pdm_labels/pdm_train_labels.tar.gz", "pdm_labels/pdm_train_labels.MANIFEST.json", "pdm_labels/pdm_train.log"]
    if not glob.glob(str(tdir / "pdm_*.jsonl")) and all(p in meta for p in need):
        paths = {p: Path(fetch(repo, "model", p, str(dl))) for p in need}
        man = json.loads(paths[need[1]].read_text())
        tgz = paths[need[0]]
        if tgz.stat().st_size != man["tar_bytes"] or _hash(tgz, "sha256") != man["tar_sha256"]:
            return False, "labels_tar_mismatch"
        safe_extract(tgz, work / "refe_sft2", {"train_pdm"})
        if not all((tdir / f).is_file() and (tdir / f).stat().st_size == s for f, s in man["files"].items()):
            return False, "labels_extract_mismatch"
        shutil.copyfile(paths[need[2]], work / "refe_sft2" / "pdm_train.log")
        got["labels"] = "yes"
    if not (run / "ckpt_last.pt").is_file() and "run/ckpt_last.pt" in meta:
        run.mkdir(parents=True, exist_ok=True)
        keep = [p for p in sorted(meta) if p.startswith("run/") and p.count("/") == 1
                and (p.endswith(".pt") or p in ("run/config.json", "run/metrics.jsonl", "run/summary.json"))]
        for p in keep:
            f = Path(fetch(repo, "model", p, str(dl)))
            size, sha = meta[p]
            if f.stat().st_size != size or (sha and p.endswith(".pt") and _hash(f, "sha256") != sha):
                return False, "run_file_mismatch:" + p
            shutil.copyfile(f, run / Path(p).name)
        got["ckpt"] = "yes"
    return True, f"labels={got['labels']} ckpt={got['ckpt']}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("restore")
    r.add_argument("--work", default="/workspace/data")
    rc = sub.add_parser("recover")
    rc.add_argument("--work", default="/workspace/data")
    rc.add_argument("--run", required=True)
    rc.add_argument("--repo", default=OUT_REPO)
    for nm in ("watch", "final", "tick", "pdm-push"):
        p = sub.add_parser(nm)
        p.add_argument("--repo", default=OUT_REPO)
        p.add_argument("--chain-log", required=True)
        p.add_argument("--outbox", required=True)
        p.add_argument("--max-push-bytes", type=float, default=12e9)
        if nm in ("watch", "final", "tick"):
            p.add_argument("--run", required=True)
            p.add_argument("--logs", required=True)
            p.add_argument("--tail", action="append", default=[])
        if nm == "watch":
            p.add_argument("--stop-file", required=True)
            p.add_argument("--interval-min", type=float, default=30)
            p.add_argument("--heavy-min", type=float, default=30)
        if nm == "pdm-push":
            p.add_argument("--labels", required=True)
            p.add_argument("--log", default=None)
    a = ap.parse_args(argv)
    if not os.environ.get("HF_TOKEN"):
        print("ZZSYNC_FAIL HF_TOKEN not set in the environment", flush=True)
        return 2
    api = E.make_api()
    if a.cmd == "restore":
        ok, why = restore(api, a.work)
        print(("ZZRESTORE_OK " if ok else "ZZRESTORE_FAIL ") + E._sanitize(why), flush=True)
        return 0 if ok else 1
    if a.cmd == "recover":
        try:
            ok, why = recover(api, a.work, a.run, repo=a.repo)
        except Exception as e:
            ok, why = False, f"exception:{type(e).__name__}:{E._sanitize(e)[:200]}"
        print(("ZZRECOVER_OK " if ok else "ZZRECOVER_FAIL ") + E._sanitize(why), flush=True)
        return 0 if ok else 1
    if a.cmd == "watch":
        return watch(api, a)
    if a.cmd == "tick":
        os.makedirs(a.outbox, exist_ok=True)
        ok, why = tick(api, a, 0, True)
        print(("ZZSYNC_TICK_OK " if ok else "ZZSYNC_TICK_FAIL ") + E._sanitize(why), flush=True)
        return 0 if ok else 1
    if a.cmd == "final":
        ok, why = final(api, a)
        line = ("ZZP2R_FINAL_OK " if ok else "ZZP2R_FINAL_FAIL ") + E._sanitize(why)
        E._append(a.chain_log, f"{E._ts()} {line}")
        print(line, flush=True)
        return 0 if ok else 1
    ok, why = pdm_push(api, a)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
