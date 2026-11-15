#!/usr/bin/env python3
"""q5_frames_pod.py -- AMENDMENT 7 stage Q5-0 pod driver: the 416x1024 frame build + the NOW-frame encodes at 416 and 256p.  Runs ON THE LORA POD (env_lora).
No address, port or pod id lives in this file; the HF token is read in place from /root/.hf_token and never printed.

Per clip (one pass, resumable): stage the mp4 from HF at the PINNED revision (sha256 vs that revision's own camera_sha256.json) -> `build_v2ep_wide.build_one` (UNEDITED, 416x1024) -> verify
-> delete the mp4 -> main process: the sibling encoder on the NOW frames (raw frame 2t) at 416 and, from the pod's native 256 v2ep, at 256p -> P-A pools + fp8 -> rows/<clip>.npz -> ledger line
(C6b MAE of resize416_to_256 against the native 256 frames, C6c bitwise n_frames / poses / actions against the 256 v2ep) -> delete the 416 v2ep for TRAIN clips (kept for eval: C6a).
Final marker in the log: 'ZZ Q5BUILD DONE ZZ' or 'ZZ Q5BUILD FAIL ZZ <why>'.

  python q5_frames_pod.py run    [--root R] [--samples S1,S2] [--eval-only-n N]   detached: nohup setsid ... < /dev/null
  python q5_frames_pod.py c8     [--root R] --clips a,b,c                         C8 on those eval clips (256, full every-2nd-frame order, batches of 8)
  python q5_frames_pod.py hashes [--root R] --out file.json                        per-frame sha256 of the decoded uint8 frames of every 416 v2ep under R/v2ep416/eval (C6a)
  python q5_frames_pod.py gather [--root R]                                        rows/ -> S0-P-format vision dirs under R/out/
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "V2_TORCH_THREADS", "PAI_DECODE_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_HUB_OFFLINE", None)          # env_lora sets both (huggingface_hub also honours TRANSFORMERS_OFFLINE): the mp4 download needs the hub; the model load uses local_files_only
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.setdefault("TANITAD_STACK", "/root/tad/stack")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ["TANITAD_STACK"])
sys.path.insert(0, os.path.join(os.environ["TANITAD_STACK"], "scripts"))

HF = "Sayood/tanitad-v7-training-corpus"
REV = "a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0"            # pinned (SCOPE section 2); an unpinned revision is refused
S_SHA = {"S1": "843e4017de3c40ae91618ac878cc2f361c798a62633eb9fa6805b457b4c1609e", "S2": "32ad96be507893b91c673c87e3088a3d294ff0e047b61e37ff8b39951740136a"}
S_SEED = {"S1": 20261111, "S2": 20261115}
STAGE = "/root/data/stage"
CAM = f"{STAGE}/r0/camera_front_wide"
V256 = "/root/data/v2ep"
LEDGER256 = "/root/data/v2ep_ledger.jsonl"
PEP = os.environ.get("V8F_LORA_BUILDER", "/root/identity_pod/code/pull_encode_pooled.py")
SHIP = os.environ.get("V8F_LORA_SHIP", "/root/tad/stack/scripts/dinov3_fp8_encode_ship.py")
BASE = "/root/q5_416"


def log(root, m):
    line = f"[q5 {time.strftime('%F %T')}] {m}"
    print(line, flush=True)
    with open(f"{root}/q5build.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def free_gb(p="/root"):
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 1e9


def tok():
    return open("/root/.hf_token").read().strip()


def jsonl_last(path, key="clip"):
    out = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    d = json.loads(ln)
                    out[d[key]] = d
                except Exception:
                    pass                                     # a torn trailing line is not an entry yet
    return out


# ------------------------------------------------------------------------------------------------------------ worker: mp4 -> 416 v2ep
def pinned_sha_table(root):
    """camera_sha256.json AT THE PINNED REVISION (the pod's own copy is the unpinned HEAD's)."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF, "camera/camera_sha256.json", repo_type="dataset", revision=REV, token=tok(), local_dir=f"{root}/hf_pin")
    j = json.load(open(p, encoding="utf-8"))
    return j["files"] if isinstance(j, dict) and isinstance(j.get("files"), dict) else j


def _sha_of(tab, cid):
    for k in (cid, f"{cid}.mp4", f"camera/{cid}.mp4"):
        if k in tab:
            v = tab[k]
            return v if isinstance(v, str) else v.get("sha256")
    return None


def build_worker(item):
    """(cid, kind, root) -> dict.  Never raises.  The mp4 is deleted only after the 416 v2ep is built AND content-verified."""
    cid, kind, root = item
    t0 = time.time()
    res = {"clip": cid, "kind": kind}
    try:
        sys.path.insert(0, f"{root}/code") if f"{root}/code" not in sys.path else None
        sys.path.insert(0, f"{BASE}/code") if f"{BASE}/code" not in sys.path else None
        from huggingface_hub import hf_hub_download
        import build_v2ep_wide as W
        from pod_pull_b1_epcache import verify
        out_dir = f"{root}/v2ep416/{kind}"
        out = f"{out_dir}/{cid}.v2ep.pt"
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out):
            ok, _ = verify(out)
            if ok:
                res.update(status="OK", bytes=os.path.getsize(out), secs=0.0, reused=True)
                return res
            os.unlink(out)
        tab = json.load(open(f"{root}/hf_pin/camera/camera_sha256.json", encoding="utf-8"))
        tab = tab["files"] if isinstance(tab, dict) and isinstance(tab.get("files"), dict) else tab
        want = _sha_of(tab, cid)
        assert want, "no sha256 at the pinned revision"
        mp4 = f"{CAM}/{cid}.mp4"
        got = sha256_file(mp4) if os.path.exists(mp4) else None
        tries = 0
        while got != want:
            tries += 1
            if tries > 3:
                res.update(status="FAIL", err=f"mp4 sha mismatch after 3 downloads ({got} != {want})")
                return res
            if os.path.exists(mp4):
                os.unlink(mp4)
            p = hf_hub_download(HF, f"camera/{cid}.mp4", repo_type="dataset", revision=REV, token=tok(), local_dir=f"{root}/hf_dl")
            os.replace(p, mp4)
            got = sha256_file(mp4)
        _, nb, secs, n, geo, err = W.build_one((cid, STAGE, out_dir, 416, 1024))
        if err:
            res.update(status="FAIL", err=err, secs=secs)
            return res
        res.update(status="OK", bytes=nb, secs=secs, mp4_sha256=got, geo=geo, n_frames_built=n)
        os.unlink(mp4)
        return res
    except Exception as e:                                           # noqa: BLE001
        res.update(status="FAIL", err=f"{type(e).__name__}: {str(e)[:200]}", tb=traceback.format_exc()[-300:])
        return res


# ------------------------------------------------------------------------------------------------------------ main-process encode of one clip
class Enc:
    def __init__(self):
        import importlib.util
        import torch
        import q5_now_encode as Q
        spec = importlib.util.spec_from_file_location("pep_q5", PEP)
        self.pep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pep)
        self.ship = self.pep.load_encoder_module(SHIP)
        self.model, self.mean, self.std, self.n_special = self.pep.build_model("cuda", "bf16")
        self.Q, self.torch = Q, torch
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.led256 = jsonl_last(LEDGER256)

    def load256(self, cid, kind):
        p = f"{V256}/{kind}/{cid}.v2ep.pt"
        led = self.led256[cid]
        assert os.path.getsize(p) == led["bytes"] and sha256_file(p) == led["sha256"], f"{cid}: native 256 v2ep differs from the pod ledger"
        return self.torch.load(p, map_location="cpu", weights_only=False), led["sha256"]

    def clip(self, cid, kind, ts, root):
        """-> (rows dict for npz, ledger fields)."""
        torch, Q = self.torch, self.Q
        o4 = torch.load(f"{root}/v2ep416/{kind}/{cid}.v2ep.pt", map_location="cpu", weights_only=False)
        o2, sha2 = self.load256(cid, kind)
        raw = [2 * int(t) for t in ts]
        assert max(raw) < len(o4["jpeg_len"]), f"{cid}: window start beyond the clip"
        t0 = time.time()
        fr4 = Q.decode_frames(o4, raw, self.pool)
        fr2 = Q.decode_frames(o2, raw, self.pool)
        r4 = Q.rows_from_field(Q.encode_frames(fr4, self.model, self.mean, self.std, self.n_special, "cuda", 1664), "416")
        r2 = Q.rows_from_field(Q.encode_frames(fr2, self.model, self.mean, self.std, self.n_special, "cuda", 640), "256")
        enc_s = time.time() - t0
        # C6b: the 416 NOW frames resampled as resize416_to_256 does (x0.625 to 260x640, bilinear, no antialias, rows 2:258) against the native 256 frames, in 8-bit counts
        import torch.nn.functional as F
        rs = F.interpolate(fr4.float(), size=(260, 640), mode="bilinear", align_corners=False, antialias=False)[:, :, 2:258, :].round().clamp(0, 255)
        d = (rs - fr2.float()).abs()
        c6c = {"n_frames": len(o4["jpeg_len"]) == len(o2["jpeg_len"]), "poses": bool(torch.equal(o4["poses"], o2["poses"])), "actions": bool(torch.equal(o4["actions"], o2["actions"]))}
        rows = {"t": np_i64(ts)}
        for tag, r in (("416", r4), ("256p", r2)):
            for k, v in r.items():
                rows[f"{k}_{tag}"] = v
        return rows, {"native256_sha256": sha2, "enc_s": round(enc_s, 2), "c6b_abs_sum": float(d.sum()), "c6b_n": int(d.numel()), "c6c": c6c, "n_frames": len(o4["jpeg_len"])}


def np_i64(a):
    import numpy as np
    return np.asarray(a, np.int64)


# ------------------------------------------------------------------------------------------------------------ the need map
def load_need(root, samples, eval_only_n):
    """-> need {clip: (kind, sorted unique t list)}, targets {key: dict(clip, ws, y, eid)}"""
    import numpy as np
    need, targets = {}, {}
    if eval_only_n == 0:
        for s in samples:
            sj = json.load(open(f"{BASE}/dry_{S_SEED[s]}/sample.json", encoding="utf-8"))
            assert sj["sha256"] == S_SHA[s], f"{s}: sample sha256 {sj['sha256']} != frozen {S_SHA[s]}"
            targets[s] = sj
            for c, w in zip(sj["clip"], sj["ws"]):
                need.setdefault(c, ("train", set()))[1].add(int(w))
    ew = json.load(open(f"{BASE}/eval_windows.json", encoding="utf-8"))
    ev_clips = [l.strip() for l in open("/root/ops/clips141.txt", encoding="utf-8") if l.strip()]
    first = sorted(set(ew["clip"]))[:eval_only_n] if eval_only_n else None
    if eval_only_n == 0:
        for c in ev_clips:
            need.setdefault(c, ("eval", set()))
    for c, w in zip(ew["clip"], ew["ws"]):
        if first is not None and c not in first:
            continue
        need.setdefault(c, ("eval", set()))[1].add(int(w))
    targets["eval"] = ew
    return {c: (k, sorted(t)) for c, (k, t) in need.items()}, targets


# ------------------------------------------------------------------------------------------------------------ run
def clip_done(led, c, root, need):
    """A clip is done iff its last ledger line is OK and, when it has windows, its rows file exists.
    MM 2026-11-15: 2 of the 141 eval clips have no eval window, so they write no rows file; the old predicate (OK AND rows file) never held for them
    and the run loop rebuilt both 80 times (pod ledger 4,757 lines over 4,599 clips). A zero-window clip is done on its OK ledger line alone."""
    if led.get(c, {}).get("status") != "OK":
        return False
    return len(need[c][1]) == 0 or os.path.exists(f"{root}/rows/{c}.npz")


def cmd_run(a):
    root = a.root
    for d in ("code", "rows", "v2ep416", "hf_pin", "hf_dl"):
        os.makedirs(f"{root}/{d}", exist_ok=True)
    if a.revision != REV:
        print("REFUSED: unpinned or different revision")
        return 2
    if root != BASE and not os.path.exists(f"{root}/code/build_v2ep_wide.py"):
        shutil.copyfile(f"{BASE}/code/build_v2ep_wide.py", f"{root}/code/build_v2ep_wide.py")
    led_p, mark = f"{root}/ledger.jsonl", f"{root}/q5build.marker"
    try:
        samples = [s for s in a.samples.split(",") if s]
        need, targets = load_need(root, samples, a.eval_only_n)
        # the parity ingest gate, run here as v2ep_pod.stage does (build_one is called directly)
        from tanitad.data import parity
        parity.require_ingest_gate("pod_build_b1_epcache")
        tr = [c for c, (k, _) in need.items() if k == "train"]
        ev = [c for c, (k, _) in need.items() if k == "eval"]
        for ids, role in ((tr, "train"), (ev, "eval")):
            if ids:
                kept, _ = parity.guard_corpus_build(ids, label="q5_frames_pod", role=role, mode="exclude")
                if role == "eval":
                    # MM 2026-11-15: informational only on eval, as the cache builder does (pull_encode_pooled.py:196-208, "the 141 is fixed by the
                    # prereg"). MEASURED: 11 of the 141 are inside the REF-C parity train corpus and 0 inside v8f-mini's 4,572-clip train cache.
                    with open(f"{root}/parity_eval_record.json", "w", encoding="utf-8") as fh:
                        json.dump({"n": len(ids), "n_kept": len(kept), "inside_parity_train": sorted(set(ids) - set(kept)), "informational_only": True}, fh, indent=1)
                else:
                    assert len(kept) == len(ids), f"the parity gate would DROP {len(ids) - len(kept)} {role} clips"
        pinned_sha_table(root)
        order = sorted(ev) + sorted(tr, key=lambda c: hashlib.sha256(c.encode()).hexdigest())
        if a.limit:
            order = order[:a.limit]
        enc = Enc()
        att = {}
        t_all = time.time()
        log(root, f"START clips={len(order)} (eval {len(ev)}, train {len(tr)}) samples={samples} free={free_gb():.0f}GB")
        n_done = 0
        while True:
            led = jsonl_last(led_p)
            todo = [c for c in order if not clip_done(led, c, root, need) and att.get(c, 0) < 2]
            if not todo:
                break
            part = todo[:a.chunk]
            if free_gb() < a.min_free_gb:
                raise RuntimeError(f"free disk {free_gb():.1f} GB < {a.min_free_gb} GB")
            try:
                with ProcessPoolExecutor(max_workers=a.workers) as ex:
                    futs = {ex.submit(build_worker, (c, need[c][0], root)): c for c in part}
                    for fu in as_completed(futs):
                        c = futs[fu]
                        r = fu.result()
                        kind, ts = need[c]
                        if r["status"] != "OK":
                            att[c] = att.get(c, 0) + 1
                            log(root, f"FAIL {c[:8]} {r.get('err')}")
                            continue
                        try:
                            if ts:
                                rows, fields = enc.clip(c, kind, ts, root)
                                tmp = f"{root}/rows/{c}.npz.tmp.npz"
                                import numpy as np
                                np.savez(tmp, **rows)
                                os.replace(tmp, f"{root}/rows/{c}.npz")
                            else:
                                fields = {}
                            p4 = f"{root}/v2ep416/{kind}/{c}.v2ep.pt"
                            line = {"clip": c, "kind": kind, "status": "OK", "v2ep416_sha256": sha256_file(p4), "v2ep416_bytes": os.path.getsize(p4), "mp4_sha256": r.get("mp4_sha256"),
                                    "build_s": r.get("secs"), "n_t": len(ts), "t": time.strftime("%F %T"), **fields}
                            with open(led_p, "a", encoding="utf-8") as f:
                                f.write(json.dumps(line) + "\n")
                            if kind == "train":
                                os.unlink(p4)                       # the 416 v2ep of a train clip is not kept (380 GB); its sha256 is in the ledger
                            n_done += 1
                            if n_done % 25 == 0:
                                w = time.time() - t_all
                                log(root, f"PROGRESS {n_done} this run, {len(order) - len(todo) + n_done} of {len(order)} total, {n_done / w * 3600:.0f} clips/h, free {free_gb():.0f}GB")
                        except Exception as e:                       # noqa: BLE001
                            att[c] = att.get(c, 0) + 1
                            log(root, f"FAIL-ENC {c[:8]} {type(e).__name__}: {str(e)[:200]}")
            except BrokenProcessPool as e:
                for c in part:
                    att[c] = att.get(c, 0) + 1
                log(root, f"BrokenProcessPool: {e}")
        led = jsonl_last(led_p)
        bad = [c for c in order if not clip_done(led, c, root, need)]
        if bad:
            raise RuntimeError(f"{len(bad)} clips not done: {[c[:8] for c in bad[:10]]}")
        agg = aggregate(root)
        json.dump(agg, open(f"{root}/controls_c6.json", "w", encoding="utf-8"), indent=1)
        log(root, f"ZZ Q5BUILD DONE ZZ clips={len(order)} controls={json.dumps(agg)[:300]}")
        open(mark, "w").write("DONE\n")
        return 0
    except Exception as e:                                           # noqa: BLE001
        log(root, f"ZZ Q5BUILD FAIL ZZ {type(e).__name__}: {str(e)[:300]}")
        log(root, "TB " + " | ".join(traceback.format_exc()[-600:].splitlines()))
        open(mark, "w").write(f"FAIL {type(e).__name__}\n")
        return 1


def aggregate(root):
    led = jsonl_last(f"{root}/ledger.jsonl")
    s, n = 0.0, 0
    c6c = True
    for r in led.values():
        if "c6b_abs_sum" in r:
            s += r["c6b_abs_sum"]
            n += r["c6b_n"]
            c6c = c6c and all(r["c6c"].values())
    return {"c6b_mae_255": (s / n) if n else None, "c6b_n_values": n, "c6c_bitwise": bool(c6c), "n_clips": len(led)}


# ------------------------------------------------------------------------------------------------------------ C8, C6a hashes, gather
def cmd_c8(a):
    import torch
    e = Enc()
    out = {}
    for cid in a.clips.split(","):
        o, _ = e.load256(cid, "eval")
        ref = e.ship.encode_episode(o, e.model, e.mean, e.std, e.n_special, "cuda")
        sib = e.Q.encode_episode_sib(o, e.model, e.mean, e.std, e.n_special, "cuda", 640, e.pool)
        fl = e.Q.decode_frames(o, list(range(0, len(o["jpeg_len"]), 2)), e.pool)
        ts = list(range(0, ref.shape[0], 3))[:7]                        # a sparse subset: batch composition differs
        sp = e.Q.encode_frames(fl[ts], e.model, e.mean, e.std, e.n_special, "cuda", 640)
        rel = float((sp - ref[ts]).norm() / ref[ts].norm())
        out[cid] = {"bitwise_equal": bool(torch.equal(ref, sib)), "shape": list(sib.shape), "sparse_vs_full_rel_l2": rel}
    out["all_bitwise"] = all(v["bitwise_equal"] for v in out.values())
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{a.root}/c8.json", "w", encoding="utf-8"), indent=1)
    return 0 if out["all_bitwise"] else 1


def frame_hashes(path):
    """-> {n_frames, poses_sha, actions_sha, frames: [sha256 of each decoded uint8 frame]}  (device-independent; C6a compares these with E:'s copy)"""
    import torch
    import torchvision
    o = torch.load(path, map_location="cpu", weights_only=False)
    lens, buf = o["jpeg_len"], o["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    hs = []
    for j in range(len(lens)):
        if int(lens[j]) == 0:
            hs.append(None)
            continue
        fr = torchvision.io.decode_png(buf[offs[j]:offs[j + 1]].clone())
        hs.append(hashlib.sha256(fr.numpy().tobytes()).hexdigest())
    return {"n_frames": len(lens), "shape": [int(o["image_h"]), int(o["image_w"])], "frames": hs, "file_sha256": sha256_file(path)}


def cmd_hashes(a):
    d = f"{a.root}/v2ep416/eval"
    out = {f[:-len(".v2ep.pt")]: frame_hashes(f"{d}/{f}") for f in sorted(os.listdir(d)) if f.endswith(".v2ep.pt")}
    json.dump(out, open(a.out, "w"))
    print(len(out), "clips hashed ->", a.out)
    return 0


def cmd_gather(a):
    import numpy as np
    root = a.root
    need, targets = load_need(root, [s for s in a.samples.split(",") if s], a.eval_only_n)
    from tanitad.data import v8f_mini_data as D
    for key, tg in targets.items():
        clips = tg["clip"]
        ws = np.asarray(tg["ws"], np.int64)
        y = np.asarray(tg["y"], np.uint8)
        eid = np.asarray(tg["eid"], np.int64) if key == "eval" else np.asarray([D.stable_episode_id(str(c)) for c in clips], np.int64)
        for tag in ("416", "256p"):
            od = f"{root}/out/{key}_{tag}"
            os.makedirs(od, exist_ok=True)
            n = len(ws)
            mm = {"v40": np.lib.format.open_memmap(f"{od}/v40.npy", "w+", np.uint8, (n, 40, 1024)), "v160": np.lib.format.open_memmap(f"{od}/v160.npy", "w+", np.uint8, (n, 160, 1024)),
                  "mean": np.lib.format.open_memmap(f"{od}/mean.npy", "w+", np.float16, (n, 1024))}
            by = {}
            for i, c in enumerate(clips):
                by.setdefault(c, []).append(i)
            for c, idx in by.items():
                z = np.load(f"{root}/rows/{c}.npz")
                tpos = {int(t): j for j, t in enumerate(z["t"])}
                for i in idx:
                    j = tpos[int(ws[i])]
                    for k in ("v40", "v160", "mean"):
                        mm[k][i] = z[f"{k}_{tag}"][j]
            for m in mm.values():
                m.flush()
            np.save(f"{od}/y.npy", y)
            np.save(f"{od}/clip.npy", np.asarray(clips).astype("U40"))
            np.save(f"{od}/ws.npy", ws)
            np.save(f"{od}/eid.npy", eid)
            json.dump({"key": key, "tag": tag, "n": int(n), "sample_sha256": tg.get("sha256"), "pipeline": "q5-sibling-P-A-v1"}, open(f"{od}/meta.json", "w"))
            print("gathered", od)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "c8", "hashes", "gather"))
    ap.add_argument("--root", default=BASE)
    ap.add_argument("--samples", default="S1,S2")
    ap.add_argument("--eval-only-n", type=int, default=0, help="smoke: only the first N eval clips (with windows), no train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--chunk", type=int, default=65)
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    ap.add_argument("--revision", default=REV)
    ap.add_argument("--clips", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    os.makedirs(a.root, exist_ok=True)
    return {"run": cmd_run, "c8": cmd_c8, "hashes": cmd_hashes, "gather": cmd_gather}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
