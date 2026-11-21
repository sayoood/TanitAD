#!/usr/bin/env python3
"""Pod-side DETACHED, RESUMABLE cache re-encode that trails the frame build (PREREG_V8F_LORA, B' base).

Input : every clip of /root/data/v2ep_ledger.jsonl (last line per clip), once its line exists; the v2ep file is read once, its size and sha256 are
        checked against the ledger line, and that exact byte string is what is encoded.
Encode: the cache BUILDER's own path, imported unedited: pull_encode_pooled.encode_clip -> dinov3_fp8_encode_ship.encode_episode (bf16 weights, batches
        of 8, no autocast), pool_field 4x10 and 8x20 in fp32 on the CPU, then fp8 e4m3fn; kinematics, mean, p10 as the builder writes them.
Output: <out>/{train,eval}/clips/<clip>.pt + manifest.jsonl + meta.json + runs.jsonl, the SAME layout, keys and manifest-row schema as box 2's cache_full
        (the row is built by the same lines as pull_encode_pooled.main). Extra files, never read by the trainer: pod_provenance.jsonl (v2ep sha256 per clip)
        and compare_box2.jsonl (rel-L2 / bitwise of the pod tokens against box 2's stored tokens, for the clips box 2's cache was pulled for).
Resume: a clip is done iff its manifest row, its file (size) and its provenance line (same v2ep sha256) exist. A partial file never carries a final name.
Finish: 'ZZ CACHEPOD DONE ZZ' after the build's 'ZZ V2EP DONE ZZ' is seen AND every ledger clip (4,713) is encoded; 'ZZ CACHEPOD FAIL ZZ <why>' otherwise.
No address, port or pod id lives in this file.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch

PEP = os.environ.get("V8F_LORA_BUILDER", "/root/identity_pod/code/pull_encode_pooled.py")
SHIP = os.environ.get("V8F_LORA_SHIP", "/root/tad/stack/scripts/dinov3_fp8_encode_ship.py")
LEDGER = Path("/root/data/v2ep_ledger.jsonl")
BUILD_LOG = Path("/root/logs/v2ep_run.log")
V2EP = Path("/root/data/v2ep")
BOX2_PULLED = Path("/root/identity_pod/cache_c")          # box 2 cache files pulled by sftp get (a few clips)
N_EXPECT = 4713
GRIDS = ["4x10", "8x20"]
KIN_KEYS = ("mean", "v", "a", "steer", "yaw_rate", "pose_grid", "kin3", "p10")


def log(m):
    print(f"[cache_pod {time.strftime('%F %T')}] {m}", flush=True)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("pull_encode_pooled", PEP)
    pep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pep)
    ship = pep.load_encoder_module(SHIP)
    return pep, ship


def ledger_last():
    last = {}
    if LEDGER.exists():
        for ln in open(LEDGER, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    d = json.loads(ln)
                    last[d["clip"]] = d
                except Exception:
                    pass                                   # a torn trailing line is not an entry yet
    return last


def build_marker():
    """-> 'DONE' | 'FAIL' | None: the LAST ZZ V2EP marker in the build log."""
    if not BUILD_LOG.exists():
        return None
    mk = None
    with open(BUILD_LOG, "rb") as fh:
        for ln in fh:
            if b"ZZ V2EP DONE ZZ" in ln:
                mk = "DONE"
            elif b"ZZ V2EP FAIL ZZ" in ln:
                mk = "FAIL"
    return mk


def read_jsonl_last(path, key="clip"):
    out = {}
    if Path(path).exists():
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    d = json.loads(ln)
                    out[d[key]] = d
                except Exception:
                    pass
    return out


class Store:
    def __init__(self, out: Path, split: str):
        self.root = out / split
        (self.root / "clips").mkdir(parents=True, exist_ok=True)
        (self.root / ".tmp").mkdir(exist_ok=True)
        self.mpath, self.ppath, self.cpath = self.root / "manifest.jsonl", self.root / "pod_provenance.jsonl", self.root / "compare_box2.jsonl"
        self.ents = read_jsonl_last(self.mpath)
        self.prov = read_jsonl_last(self.ppath)

    def done(self, clip, v2ep_sha):
        e, p = self.ents.get(clip), self.prov.get(clip)
        if not e or not p or p.get("v2ep_sha256") != v2ep_sha:
            return False
        f = self.root / "clips" / f"{clip}.pt"
        return f.exists() and f.stat().st_size == e["bytes"]


def prepare(item):
    """Worker thread: read the v2ep once, check size + sha256 against the ledger line, load. Raises on any mismatch."""
    clip, kind, led = item
    p = V2EP / kind / f"{clip}.v2ep.pt"
    if p.stat().st_size != led["bytes"]:
        raise RuntimeError(f"size {p.stat().st_size} != ledger {led['bytes']}")
    data = p.read_bytes()
    sh = sha256_bytes(data)
    if sh != led["sha256"]:
        raise RuntimeError(f"sha256 {sh[:12]} != ledger {led['sha256'][:12]}")
    o = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    assert o.get("clip_id") == clip and str(o["codec"]) == "png"
    assert (int(o["image_h"]), int(o["image_w"])) == (256, 640)
    return o, sh


def compare_box2(blob, clip, split):
    p = BOX2_PULLED / split / f"{clip}.pt"
    if not p.exists():
        return None
    ref = torch.load(p, map_location="cpu", weights_only=True)
    out = {"clip": clip}
    for g in ("pool4x10", "pool8x20"):
        x, y = blob[g], ref[g]
        a, b = x.double(), y.double()
        d = a - b
        out[g] = {"rel_l2": float(d.norm() / b.norm()), "max_abs": float(d.abs().max()),
                  "frac_bitwise_equal_values": float((x.view(torch.uint8) == y.view(torch.uint8)).double().mean()),
                  "frac_bitwise_equal_tokens": float((x.view(torch.uint8) == y.view(torch.uint8)).all(dim=-1).double().mean())}
    out["kinematics_equal"] = {k: bool(torch.allclose(blob[k].double(), ref[k].double(), rtol=0, atol=0, equal_nan=True)) for k in KIN_KEYS}
    out["keys_equal"] = sorted(blob) == sorted(ref)
    out["dtypes_equal"] = {k: str(blob[k].dtype) == str(ref[k].dtype) for k in ref if k != "meta" and k in blob}
    out["shapes_equal"] = {k: list(blob[k].shape) == list(ref[k].shape) for k in ref if k != "meta" and k in blob}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/root/data/cache_pod")
    ap.add_argument("--only", nargs="*", default=None, help="smoke: encode just these clip ids (they must have a ledger line), then stop")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    ap.add_argument("--max-consecutive-fail", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".tmp").mkdir(exist_ok=True)
    stores = {s: Store(out, s) for s in ("train", "eval")}
    pep, ship = load_builder()
    assert torch.cuda.is_available(), "no CUDA: refusing a silent CPU fallback"
    qdtype = torch.float8_e4m3fn if pep.fp8_ok(out / ".tmp") else torch.float16
    assert qdtype is torch.float8_e4m3fn, "fp8 preflight failed: the cache must be fp8 e4m3fn"
    qname = "float8_e4m3fn"
    model, mean, std, n_special = pep.build_model("cuda", "bf16")
    wsha = hashlib.sha256()
    with open("/root/models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors", "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            wsha.update(b)
    import transformers
    prov = {"pipeline": pep.PIPELINE, "script_sha256": pep.file_hash(PEP, "sha256"), "encoder_module": SHIP, "encoder_module_sha256": pep.file_hash(SHIP, "sha256"),
            "model": pep.MID, "model_safetensors_sha256": wsha.hexdigest(), "torch": torch.__version__, "transformers": transformers.__version__, "device": "cuda",
            "cuda_device": torch.cuda.get_device_name(0), "enc_dtype": "bf16", "argv": sys.argv[1:], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "grids": GRIDS, "stored_dtype": qname, "source": {"kind": "local256", "dir": str(V2EP), "remote": False, "resize": False},
            "frames": "pod-built v2ep (Linux CPU cylindrical rectify), ledger-checked by size and sha256", "pod_stage": "PREREG_V8F_LORA B' base re-encode"}
    for s, st in stores.items():
        (st.root / "meta.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
        with open(st.root / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"started": prov["started"], "argv": prov["argv"], "pid": os.getpid()}) + "\n")
    log(f"ready: torch {torch.__version__} {prov['cuda_device']} weights {prov['model_safetensors_sha256'][:12]} out={out}")

    pool = ThreadPoolExecutor(max_workers=1)
    n_done = n_fail = consec = 0
    t_start = time.time()
    first_idle = True
    while True:
        led = ledger_last()
        bm = build_marker()
        todo = []
        for clip, d in led.items():
            if a.only is not None and clip not in a.only:
                continue
            st = stores[d["kind"]]
            if not st.done(clip, d["sha256"]):
                todo.append((clip, d["kind"], d))
        if a.only is not None:
            missing = [c for c in a.only if c not in led]
            if missing:
                log(f"FAIL: --only clips without a ledger line: {missing}")
                return 2
        if not todo:
            if a.only is not None:
                log(f"smoke done: {n_done} encoded")
                return 0
            if bm == "DONE":
                if len(led) == N_EXPECT:
                    n_tr = len(stores["train"].ents)
                    n_ev = len(stores["eval"].ents)
                    log(f"ZZ CACHEPOD DONE ZZ train {n_tr} eval {n_ev} encoded {n_done} this launch, {time.time() - t_start:.0f}s")
                    return 0
                log(f"ZZ CACHEPOD FAIL ZZ build says DONE but the ledger has {len(led)} clips, expected {N_EXPECT}")
                return 3
            if bm == "FAIL":
                log("ZZ CACHEPOD FAIL ZZ the build reported ZZ V2EP FAIL ZZ; every OK clip is encoded")
                return 4
            if first_idle:
                log(f"caught up with the ledger ({len(led)} clips); polling every {a.poll:.0f}s")
                first_idle = False
            time.sleep(a.poll)
            continue
        first_idle = True
        fut = pool.submit(prepare, todo[0])
        for i, item in enumerate(todo):
            clip, kind, d = item
            nxt = pool.submit(prepare, todo[i + 1]) if i + 1 < len(todo) else None
            st = stores[kind]
            te = time.time()
            try:
                free = shutil.disk_usage(out).free / 1e9
                if free < a.min_free_gb:
                    log(f"ZZ CACHEPOD FAIL ZZ disk free {free:.1f} GB < {a.min_free_gb}")
                    return 7
                o, v2sha = fut.result()
                payload = pep.encode_clip(o, ship, model, mean, std, n_special, "cuda", GRIDS, qdtype)
                meta = pep.make_meta(clip, o, payload, GRIDS, qdtype)
                meta["source"] = "local256"
                blob = dict(payload, meta=meta)
                tmp = st.root / ".tmp" / f"{clip}.pt"
                torch.save(blob, tmp)
                final = st.root / "clips" / f"{clip}.pt"
                os.replace(tmp, final)
                ent = {"clip": clip, "sha12": pep.sha12(clip), "file": f"clips/{clip}.pt", "bytes": final.stat().st_size, "md5": pep.file_hash(final),
                       "content_md5": pep.content_md5(payload), "n_frames": meta["n_frames"], "t_ep": meta["t_ep"], "dtype": qname, "enc_dtype": "bf16",
                       "device": "cuda", "source": "local256", "mean_dtype": "float16", "grids": GRIDS,
                       "shapes": {k: list(v.shape) for k, v in payload.items()}, "enc_s": round(time.time() - te, 2), "pipeline": pep.PIPELINE}
                cmp_row = compare_box2(blob, clip, kind)
                with open(st.ppath, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"clip": clip, "v2ep_sha256": v2sha, "v2ep_bytes": d["bytes"], "encoded": time.strftime("%F %T")}) + "\n")
                if cmp_row is not None:
                    with open(st.cpath, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(cmp_row) + "\n")
                with open(st.mpath, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ent) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                st.ents[clip] = ent
                st.prov[clip] = {"clip": clip, "v2ep_sha256": v2sha}
                n_done += 1
                consec = 0
                del o, payload, blob
            except Exception as e:                       # one clip: loud, counted, never half-written
                n_fail += 1
                consec += 1
                (st.root / ".tmp" / f"{clip}.pt").unlink(missing_ok=True)
                log(f"SKIP {clip[:8]} {type(e).__name__}: {str(e)[:200]}")
                if consec >= a.max_consecutive_fail:
                    log(f"ZZ CACHEPOD FAIL ZZ {consec} consecutive clip failures")
                    return 5
            fut = nxt
            if n_done and (n_done <= 3 or n_done % 25 == 0):
                el = time.time() - t_start
                total_done = len(stores["train"].ents) + len(stores["eval"].ents)
                rate = n_done / el
                log(f"{total_done}/{N_EXPECT} encoded ({n_done} this launch, {el / n_done:.2f} s/clip, last {time.time() - te:.1f}s, ledger {len(led)}, free {free:.0f} GB, "
                    f"eta_encode_only {(N_EXPECT - total_done) / rate / 60:.0f} min)")
            if a.only is None and i + 1 >= 200:
                break                                     # re-read the ledger and the build marker regularly
        if a.only is not None:
            log(f"smoke done: {n_done} encoded, {n_fail} failed")
            return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
