#!/usr/bin/env python3
"""DATASCALE pod-side v2ep BUILD for a staged clip list: the production builder pod_build_b1_epcache.build_one, UNEDITED, on /root/datascale/stage.
Resumable (an existing, content-verified v2ep is a SKIP). Writes a ledger in the v2ep_ledger.jsonl schema and ends the build log with 'ZZ V2EP DONE ZZ'
(the marker cache_pod waits for). CPU only. No address, port or pod id in this file."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "V2_TORCH_THREADS", "PAI_DECODE_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TANITAD_STACK", "/root/tad/stack")
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

STACK = os.environ["TANITAD_STACK"]
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))
ROOT = "/root/datascale/stage"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def stage_row(cid):
    for ln in open("/root/datascale/ledger_stage.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        if r["clip"] == cid and r.get("status") == "OK":
            last = r
    return last


def process(item):
    cid, out_dir = item
    t0 = time.time()
    res = {"clip": cid, "kind": "train"}
    out = f"{out_dir}/{cid}.v2ep.pt"
    try:
        from pod_pull_b1_epcache import verify
        if os.path.exists(out):
            ok, why = verify(out)
            if ok:
                res.update(status="SKIP", bytes=os.path.getsize(out), sha256=sha256(out), secs=0.0)
                return res
            os.unlink(out)
        import pod_build_b1_epcache as B
        cid2, nb, secs, err = B.build_one((cid, ROOT, out_dir))
        if err:
            res.update(status="FAIL", err=err, secs=secs)
            return res
        res.update(status="OK", bytes=nb, secs=secs, sha256=sha256(out))
        return res
    except Exception as e:                                          # noqa: BLE001
        res.update(status="FAIL", err=f"{type(e).__name__}: {str(e)[:200]}", secs=time.time() - t0)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--v2ep-root", required=True, help="dir that will hold train/<clip>.v2ep.pt")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--log", required=True, help="build log that receives the ZZ V2EP marker")
    ap.add_argument("--workers", type=int, default=13)
    a = ap.parse_args()
    clips = [ln.strip() for ln in open(a.clips, encoding="utf-8") if ln.strip()]
    out_dir = f"{a.v2ep_root}/train"
    os.makedirs(out_dir, exist_ok=True)
    from tanitad.data import parity
    parity.require_ingest_gate("ds_build")
    done = set()
    if os.path.exists(a.ledger):
        done = {json.loads(ln)["clip"] for ln in open(a.ledger, encoding="utf-8") if ln.strip()}
    todo = [c for c in clips if c not in done]
    lf = open(a.log, "a", encoding="utf-8", buffering=1)
    lf.write(f"[build {time.strftime('%F %T')}] {len(clips)} clips, {len(todo)} to build, workers {a.workers}\n")
    led = open(a.ledger, "a", encoding="utf-8", buffering=1)
    t0 = time.time()
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, (c, out_dir)) for c in todo]
        for fu in as_completed(futs):
            r = fu.result()
            if r["status"] in ("OK", "SKIP"):
                n_ok += 1
                led.write(json.dumps({k: r.get(k) for k in ("clip", "kind", "bytes", "sha256", "secs")} | {"t": time.strftime("%F %T")}) + "\n")
            else:
                n_fail += 1
                lf.write(f"[build] FAIL {r['clip']} {r.get('err')}\n")
                with open(a.ledger + ".fail", "a", encoding="utf-8") as ff:
                    ff.write(json.dumps(r) + "\n")
            if (n_ok + n_fail) % 20 == 0:
                lf.write(f"[build {time.strftime('%T')}] {n_ok + n_fail}/{len(todo)} ok={n_ok} fail={n_fail} {(n_ok + n_fail) / (time.time() - t0) * 3600:.0f} clips/h\n")
    wall = time.time() - t0
    lf.write(f"[build] wall {wall:.0f}s ok={n_ok} fail={n_fail} -> {len(todo) / max(wall, 1e-9) * 3600:.0f} clips/h\n")
    # DONE even with failures: cache_pod encodes the OK ones; failures stay in <ledger>.fail with their reasons
    lf.write("ZZ V2EP DONE ZZ\n")
    lf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
