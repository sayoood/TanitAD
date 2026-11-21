#!/usr/bin/env python3
"""nt_e1_run.py -- H-NT-DECIDE-1 E1 production pipeline: per shard, stream -> extract L0/F0/R0 t0 jpgs -> stitch -> PNG ->
token-episode ``*.v2ep.pt`` (INTERFACE.md v1) -> verify -> manifest -> delete the shard.  Download-bound, resumable, detached.

Shard sources, in order: shard 1 (``/home/nvidia/r8ns/navtrain_dl/navtrain_current_1.tgz``, the r8ns agent's download, READ
ONLY, never deleted), then ``/home/nvidia/r8ns/nt/dl/navtrain_current_{2..32}.tgz`` as ``nt_download.sh`` verifies them
(``<name>.ok``).  A consumed shard of ours is deleted with its ``.ok`` and named in ``dl/CONSUMED.txt``.

State (all under /home/nvidia/r8ns/nt): ``E1_STATE.json`` (shards done), ``MANIFEST_A1.jsonl`` (written + verified tokens),
``pending/`` (tokens whose LAW frame -- the next token's t0 stitch -- has not been seen yet), ``carry/`` (jpgs of a frame split
across shards), ``RIGS_A1.json`` + ``rigs/``, ``PROGRESS.json`` (heartbeat), at the end ``MISSING_A1.json``,
``CLIP_CLOCK_A1.jsonl``, ``v2ep_a1/_v2manifest.pt`` and ``DONE_A1.json`` (last).  Rerunning resumes.
Markers: ``ZZE1SHARD <name> ...`` per shard, ``ZZE1DONE`` / ``ZZE1FAIL_<why>``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import gzip
import json
import os
import re
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nt_e1 as E  # noqa: E402

NT = "/home/nvidia/r8ns/nt"
OUT = f"{NT}/v2ep_a1"
DL = f"{NT}/dl"
SHARD1 = "/home/nvidia/r8ns/navtrain_dl/navtrain_current_1.tgz"
SHARD_SIZES = {}
CAMS_L = ("cam_l0", "cam_f0", "cam_r0")
CAMS_U = ("CAM_L0", "CAM_F0", "CAM_R0")
FLOOR_B = 10_000_000_000          # keep >= 10 GB free beyond a shard's expected output
EST_TOKEN_B = 1_250_000           # MEASURED smoke mean 1,145,479 B/file (2 PNGs), +10 % headroom


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def free_b() -> int:
    st = os.statvfs("/home/nvidia")
    return st.f_bavail * st.f_frsize


def dir_bytes(d: str) -> int:
    tot = 0
    for dp, _dn, fns in os.walk(d):
        for fn in fns:
            try:
                tot += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return tot


def dl_stats() -> dict:
    """Parse nt_download.sh's ``ZZDLFILE OK`` lines (opaque marker) for the transfer rate and an ETA."""
    ok, b, s, last = 0, 0, 0, None
    try:
        for ln in open(f"{NT}/logs/dl.log", encoding="utf-8", errors="replace"):
            if ln.startswith("ZZDLFILE OK "):
                m = re.search(r"bytes=(\d+)/\d+ secs=(\d+) MBps=([\d.]+)", ln)
                if m:
                    ok += 1
                    b += int(m.group(1))
                    s += int(m.group(2))
                    last = float(m.group(3))
    except OSError:
        pass
    total = sum(SHARD_SIZES.values())
    mean = (b / s / 1e6) if s else None
    rem = max(total - b, 0)
    return {"shards_downloaded": ok, "bytes_downloaded": b, "bytes_total_31": total, "mean_MBps": round(mean, 2) if mean else None,
            "last_MBps": last, "eta_h": round(rem / (mean * 1e6) / 3600, 2) if mean else None}


class Run:
    def __init__(self, a):
        self.a = a
        os.makedirs(OUT, exist_ok=True)
        for d in ("pending", "carry", "rigs", "scratch", "logs"):
            os.makedirs(f"{NT}/{d}", exist_ok=True)
        tl = json.load(open(f"{NT}/TOKENS_A1.json"))
        self.tokens = list(tl["tokens"])
        self.pool = set(self.tokens)
        if E.N.sha256_tokens(self.tokens) != tl["sha256_sorted_tokens"]:
            raise SystemExit("ZZE1FAIL_token_list_sha")
        self.index = json.load(gzip.open(f"{NT}/TOKEN_INDEX_A1.json.gz", "rt"))["index"]
        self.recs = json.load(gzip.open(f"{NT}/inputs/a1_inputs.json.gz", "rt"))["tokens"]
        if set(self.recs) != self.pool:
            raise SystemExit("ZZE1FAIL_inputs_do_not_cover_the_pool")
        self.state = json.load(open(f"{NT}/E1_STATE.json")) if os.path.exists(f"{NT}/E1_STATE.json") else \
            {"shards_done": [], "started": now(), "shard_reports": {}}
        self.done = {}
        if os.path.exists(f"{NT}/MANIFEST_A1.jsonl"):
            for ln in open(f"{NT}/MANIFEST_A1.jsonl", encoding="utf-8"):
                if ln.strip():
                    r = json.loads(ln)
                    self.done[r["token"]] = r
        self.pending = {}
        for p in glob.glob(f"{NT}/pending/*.json"):
            m = json.load(open(p))
            if m["token"] in self.done:
                self._drop_pending(m["token"])
            else:
                self.pending[m["token"]] = m
        self.rigs = json.load(open(f"{NT}/RIGS_A1.json")) if os.path.exists(f"{NT}/RIGS_A1.json") else {}
        self.rig_cache = {}
        self.missing = (json.load(open(f"{NT}/MISSING_A1.partial.json"))["missing"]
                        if os.path.exists(f"{NT}/MISSING_A1.partial.json") else {})
        self.peak = self.state.get("peak", {"nt_bytes": 0, "min_free_b": None})
        self.log_cache = (None, None)
        self.ex = cf.ThreadPoolExecutor(max_workers=1)
        self.cur = None

    # ---------------------------------------------------------------------------------------------------------------
    def frames_of(self, log):
        if self.log_cache[0] != log:
            fr, _ = E.N.load_log_pickle(f"/home/nvidia/r8ns/navtrain/navsim_logs/trainval/{log}.pkl")
            self.log_cache = (log, fr)
        return self.log_cache[1]

    def heartbeat(self, status: str, extra: dict | None = None):
        bank_b = sum(int(r["bytes"]) for r in self.done.values())
        nt_b = bank_b + dir_bytes(DL) + dir_bytes(f"{NT}/scratch") + dir_bytes(f"{NT}/pending") + dir_bytes(f"{NT}/carry")
        fb = free_b()
        self.peak["nt_bytes"] = max(self.peak.get("nt_bytes", 0), nt_b)
        self.peak["min_free_b"] = fb if self.peak.get("min_free_b") is None else min(self.peak["min_free_b"], fb)
        self.state["peak"] = self.peak
        prog = {"status": status, "updated": now(), "shards_done": self.state["shards_done"],
                "n_shards_done": len(self.state["shards_done"]), "n_shards_total": 32, "shard_in_progress": self.cur,
                "tokens_written": len(self.done), "tokens_total": len(self.pool), "tokens_pending_law": len(self.pending),
                "missing": len(self.missing), "law_next_token": sum(1 for r in self.done.values() if r["law_src"] == "next_token"),
                "bank_bytes": bank_b, "bank_bytes_per_token": round(bank_b / max(len(self.done), 1)),
                "nt_bytes_now": nt_b, "disk_free_bytes": fb, "disk_peak_nt_bytes": self.peak["nt_bytes"],
                "disk_min_free_bytes": self.peak["min_free_b"], "download": dl_stats(), "pid": os.getpid()}
        if extra:
            prog.update(extra)
        prog["missing_reasons"] = {}
        for why in self.missing.values():
            k = re.sub(r"[0-9]+", "N", why)[:70]
            prog["missing_reasons"][k] = prog["missing_reasons"].get(k, 0) + 1
        jdump({"n": len(self.missing), "missing": self.missing}, f"{NT}/MISSING_A1.partial.json")
        jdump(prog, f"{NT}/PROGRESS.json")
        jdump(self.state, f"{NT}/E1_STATE.json")

    # ---------------------------------------------------------------------------------------------------------------
    def shard_queue(self) -> list:
        q = [("navtrain_current_1.tgz", SHARD1, False)]
        for ln in open(f"{NT}/code/shards_2_32.txt"):
            if ln.strip():
                p, sz, _sha = ln.split()
                name = os.path.basename(p)
                SHARD_SIZES[name] = int(sz)
                q.append((name, f"{DL}/{name}", True))
        return q

    def wait_for(self, name: str, path: str, ours: bool):
        while True:
            if not ours and os.path.exists(path):
                return
            if ours and os.path.exists(path + ".ok") and os.path.exists(path):
                return
            self.heartbeat(f"WAITING_FOR_{name}")
            time.sleep(30)

    def wait_disk(self, need_b: int):
        while free_b() < need_b + FLOOR_B:
            self.heartbeat("PAUSED_DISK", {"disk_need_bytes": need_b + FLOOR_B})
            time.sleep(60)

    # ---------------------------------------------------------------------------------------------------------------
    def process_shard(self, name: str, path: str):
        t0 = time.time()
        todo = [t for t in self.tokens if t not in self.done and t not in self.pending and t not in self.missing]
        need = {}
        for t in todo:
            for c in CAMS_L:
                need[str(self.recs[t]["cams"][c][-1]).replace("\\", "/")] = t
        scratch = f"{NT}/scratch/{name}"
        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch)
        n_seen = n_got = n_dup = 0
        got_tok = {}
        for rel, read in E.BNF._tar_readers(path):
            n_seen += 1
            t = need.get(rel)
            if t is None:
                continue
            dst = os.path.join(scratch, rel)
            if os.path.exists(dst):
                n_dup += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(read())
            n_got += 1
            got_tok[t] = got_tok.get(t, 0) + 1
        t_ext = time.time() - t0
        # complete in this shard, or completed with jpgs carried from an earlier shard
        complete, partial = [], []
        for t, k in got_tok.items():
            rels = [str(self.recs[t]["cams"][c][-1]).replace("\\", "/") for c in CAMS_L]
            have = [os.path.exists(os.path.join(scratch, r)) or os.path.exists(os.path.join(f"{NT}/carry", r)) for r in rels]
            if all(have):
                for r in rels:
                    cp = os.path.join(f"{NT}/carry", r)
                    if not os.path.exists(os.path.join(scratch, r)) and os.path.exists(cp):
                        os.makedirs(os.path.dirname(os.path.join(scratch, r)), exist_ok=True)
                        shutil.move(cp, os.path.join(scratch, r))
                complete.append(t)
            else:
                partial.append(t)
                for r in rels:
                    sp = os.path.join(scratch, r)
                    if os.path.exists(sp):
                        os.makedirs(os.path.dirname(os.path.join(f"{NT}/carry", r)), exist_ok=True)
                        shutil.move(sp, os.path.join(f"{NT}/carry", r))
        self.wait_disk(len(complete) * EST_TOKEN_B)
        # stitch (main thread, torch threads) + PNG encode (1 worker thread)
        t1 = time.time()
        fm = {}
        complete.sort(key=lambda t: (self.index[t]["log"], self.index[t]["pos"]))
        n_fail = 0
        for i, t in enumerate(complete):
            cams = self.frames_of(self.index[t]["log"])[self.index[t]["pos"]]["cams"]
            try:
                hwc, src, rk, obs, mpx = E.stitch_frame(cams, scratch, self.rig_cache)
            except Exception as e:                                   # noqa: BLE001
                self.missing[t] = f"stitch: {type(e).__name__}: {str(e)[:160]}"
                n_fail += 1
                continue
            if rk not in self.rigs:
                np.save(f"{NT}/rigs/{rk}.src.npy", src)
                self.rigs[rk] = {**sys.modules["frames416"].rig_record(cams), "observed_frac": round(float((src >= 0).mean()), 6),
                                 "first_token": t}
                jdump(self.rigs, f"{NT}/RIGS_A1.json")
            fm[t] = {"sha16": E.sha16(hwc), "rig_key": rk, "observed_frac": round(obs, 6), "mean_px": round(mpx, 2),
                     "png": self.ex.submit(E.encode_png, hwc)}
            if (i + 1) % 500 == 0:
                self.heartbeat(f"STITCHING_{name}", {"shard_stitched": i + 1, "shard_to_stitch": len(complete)})
        for t in fm:
            fm[t]["png"] = fm[t]["png"].result()
        t_st = time.time() - t1
        shutil.rmtree(scratch, ignore_errors=True)
        # write every token whose LAW requirement is now satisfiable
        t2 = time.time()
        n_w = n_p = 0
        cands = sorted(set(fm) | set(self.pending), key=lambda t: (self.index[t]["log"], self.index[t]["pos"]))
        for t in cands:
            if t in self.done:
                continue
            r = self.write_token(t, name, fm, final=False)
            if r == "written":
                n_w += 1
            elif r == "pending":
                n_p += 1
            if (n_w + 1) % 500 == 0:
                self.heartbeat(f"WRITING_{name}", {"shard_written": n_w})
        t_wr = time.time() - t2
        rep = {"name": name, "cam_jpgs_seen": n_seen, "needed_extracted": n_got, "dup_members": n_dup,
               "tokens_complete": len(complete), "tokens_partial_carried": len(partial), "stitch_failures": n_fail,
               "written": n_w, "pending_after": len(self.pending), "secs_extract": round(t_ext, 1), "secs_stitch_encode": round(t_st, 1),
               "secs_write_verify": round(t_wr, 1), "secs_total": round(time.time() - t0, 1),
               "s_per_frame": round((t_st / max(len(fm), 1)), 4), "finished": now()}
        return rep

    def png_of(self, t: str, fm: dict):
        if t in fm:
            return fm[t]["png"], fm[t]["sha16"]
        if t in self.pending:
            return torch.from_numpy(np.fromfile(f"{NT}/pending/{t}.png", dtype=np.uint8)), self.pending[t]["fm"]["sha16"]
        if t in self.done:
            d = torch.load(f"{OUT}/{self.done[t]['file']}", map_location="cpu", weights_only=False)
            return d["jpeg_buf"][:int(d["jpeg_len"][0])].clone(), self.done[t]["t0_frame_sha16"]
        return None, None

    def write_token(self, t: str, shard: str, fm: dict, final: bool) -> str:
        ix = self.index[t]
        nxt = ix["next_token"] if ix["next_is_navtrain_token"] and ix["next_token"] in self.pool else None
        t0_png, t0_sha = self.png_of(t, fm)
        fmeta = fm[t] if t in fm else self.pending[t]["fm"]
        if nxt is not None:
            law_png, law_sha = self.png_of(nxt, fm)
            if law_png is None and not final:
                if t not in self.pending:                      # park it until the next token's frame is stitched
                    np.asarray(t0_png.numpy(), dtype=np.uint8).tofile(f"{NT}/pending/{t}.png")
                    m = {"token": t, "shard": shard, "fm": {k: v for k, v in fmeta.items() if k != "png"}, "parked": now()}
                    jdump(m, f"{NT}/pending/{t}.json")
                    self.pending[t] = m
                return "pending"
            law = ({"src": "next_token", "token": nxt, "sha16": law_sha, "png": law_png} if law_png is not None else
                   {"src": "fallback_t0_missing", "token": nxt, "sha16": None, "png": None})
        else:
            law = {"src": "fallback_t0_not_in_navtrain_current", "token": None, "sha16": None, "png": None}
        rec = self.recs[t]
        try:
            tr = E.token_poses(rec, self.frames_of(ix["log"]), ix["pos"])
        except Exception as e:                                      # noqa: BLE001
            self.missing[t] = f"poses: {type(e).__name__}: {str(e)[:160]}"
            self._drop_pending(t)
            return "missing"
        src_shard = self.pending[t]["shard"] if t in self.pending else shard
        meta = E.token_meta(t, rec, ix["pos"], src_shard, fmeta, law, tr)
        path = f"{OUT}/{E.clip_id_of(t)}.v2ep.pt"
        nb = E.save_atomic(E.build_payload(t, t0_png, law["png"], tr, meta), path)
        ver = E.verify_file(path, t0_sha, law["sha16"], tr)
        if ver["problems"]:
            os.remove(path)
            raise SystemExit(f"ZZE1FAIL_verify {t}: {ver['problems']}")
        row = E.manifest_row(t, path, src_shard, meta, tr, nb)
        with open(f"{NT}/MANIFEST_A1.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.done[t] = row
        self._drop_pending(t)
        return "written"

    def _drop_pending(self, t):
        self.pending.pop(t, None)
        for ext in ("png", "json"):
            try:
                os.remove(f"{NT}/pending/{t}.{ext}")
            except OSError:
                pass

    # ---------------------------------------------------------------------------------------------------------------
    def finalize(self):
        self.heartbeat("FINALIZING")
        for t in sorted(self.pending, key=lambda t: (self.index[t]["log"], self.index[t]["pos"])):
            self.write_token(t, self.pending[t]["shard"], {}, final=True)
        never = sorted(t for t in self.pool if t not in self.done and t not in self.missing)
        for t in never:
            self.missing[t] = "t0 jpgs (L0/F0/R0) not found in any navtrain_current shard"
        if self.missing:
            jdump({"n": len(self.missing), "missing": self.missing}, f"{NT}/MISSING_A1.json")
        sids = [int(r["sid"]) for r in self.done.values()]
        if len(set(sids)) != len(sids):
            raise SystemExit("ZZE1FAIL_sid_collision")
        if not set(self.done) <= self.pool:
            raise SystemExit("ZZE1FAIL_written_token_outside_the_pool")
        with open(f"{NT}/CLIP_CLOCK_A1.jsonl.tmp", "w", encoding="utf-8") as fh:
            for t in sorted(self.done):
                r = self.done[t]
                fh.write(json.dumps({"sid": int(r["sid"]), "clip_id": r["clip_id"], "grid_start_s": 0.0, "dt_s": 0.1}) + "\n")
        os.replace(f"{NT}/CLIP_CLOCK_A1.jsonl.tmp", f"{NT}/CLIP_CLOCK_A1.jsonl")
        jdump({"schema": "tanitad.clip_clock/1", "what": "NT-DECIDE E1 token-episodes: raw row r is at r * 0.1 s; t0 = raw row 9 "
               "(0.9 s). Exact by construction (DECLARED, INTERFACE.md section 3).", "n_rows": len(self.done)},
              f"{NT}/CLIP_CLOCK_A1.jsonl.meta.json")
        files = sorted(f for f in os.listdir(OUT) if f.endswith(".v2ep.pt"))
        if len(files) != len(self.done) or set(files) != {r["file"] for r in self.done.values()}:
            raise SystemExit(f"ZZE1FAIL_dir_vs_manifest files={len(files)} manifest={len(self.done)}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("v2_dataset_patched", f"{HERE}/loader_patch/v2_dataset_patched.py")
        P = importlib.util.module_from_spec(spec)
        sys.modules["v2_dataset_patched"] = P
        spec.loader.exec_module(P)
        t0 = time.time()
        man = P.load_or_build_manifest(OUT, rebuild=True, verbose=True)
        law = sum(1 for r in self.done.values() if r["law_src"] == "next_token")
        done = {"done": True, "finished": now(), "n_written": len(self.done), "n_pool": len(self.pool), "n_missing": len(self.missing),
                "n_law_next_token": law, "law_fallback": {k: sum(1 for r in self.done.values() if r["law_src"] == k)
                                                          for k in ("fallback_t0_not_in_navtrain_current", "fallback_t0_missing")},
                "bank_bytes": sum(int(r["bytes"]) for r in self.done.values()),
                "manifest_sha256": E.file_sha256(f"{NT}/MANIFEST_A1.jsonl"),
                "v2manifest": {"files": len(man["files"]), "secs": round(time.time() - t0, 1)},
                "peak": self.peak, "shard_reports": self.state.get("shard_reports")}
        jdump(done, f"{NT}/DONE_A1.json")
        self.heartbeat("DONE")
        print(f"ZZE1DONE written={len(self.done)} missing={len(self.missing)} law={law}", flush=True)

    def run(self):
        for name, path, ours in self.shard_queue():
            if name in self.state["shards_done"]:
                continue
            self.cur = name
            self.wait_for(name, path, ours)
            self.heartbeat(f"PROCESSING_{name}")
            rep = self.process_shard(name, path)
            self.state["shards_done"].append(name)
            self.state.setdefault("shard_reports", {})[name] = rep
            if ours:
                for p in (path, path + ".ok"):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                with open(f"{DL}/CONSUMED.txt", "a") as fh:
                    fh.write(name + "\n")
            self.cur = None
            self.heartbeat(f"SHARD_DONE_{name}")
            print("ZZE1SHARD " + json.dumps(rep), flush=True)
        self.finalize()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3, help="torch threads for the stitch (+1 PNG encode thread = 4)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    if os.path.exists(f"{NT}/DONE_A1.json"):
        print("ZZE1DONE already", flush=True)
        return 0
    r = Run(a)
    try:
        r.run()
    except SystemExit:
        raise
    except Exception:                                                # noqa: BLE001
        traceback.print_exc()
        r.heartbeat("CRASHED", {"traceback": traceback.format_exc()[-2000:]})
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
