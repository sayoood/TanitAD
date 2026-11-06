#!/usr/bin/env python3
"""Pod-side driver: rebuild the native 256x640 cylinder v2ep frames by the PRODUCTION route.

Route (ROUTE.md rows 0-5): HF private B1 bank Sayood/tanitad-v7-training-corpus camera/<clip>.mp4
  -> sha256 vs camera/camera_sha256.json -> pod_build_b1_epcache.build_one (UNEDITED, imported)
  -> <clip>.v2ep.pt (PNG, n_stack 3, 256x640 cylindrical, hfov 120) -> build_one's own content verify.

Subcommands:  stage | gate | run     (all idempotent; `run` resumes by content)
No address, port or pod id lives in this file.  HF token is read in place from /root/.hf_token, never printed.
"""
import os
# pin threads BEFORE torch/av import (pod_build_b1_epcache does the same; TRAPS: unpinned pools make no progress)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "V2_TORCH_THREADS", "PAI_DECODE_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU build, as on the dev box and box 2; no GPU touched
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("TANITAD_STACK", "/root/tad/stack")

import argparse, hashlib, json, shutil, sys, tarfile, time, traceback, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

STACK = os.environ["TANITAD_STACK"]
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))

HF = "Sayood/tanitad-v7-training-corpus"
OPS = "/root/ops"
DATA = "/root/data"
STAGE = f"{DATA}/stage"
CAM = f"{STAGE}/r0/camera_front_wide"
SCRATCH = f"{DATA}/_hf"
V2EP = f"{DATA}/v2ep"                      # V2EP/train, V2EP/eval
LEDGER = f"{DATA}/v2ep_ledger.jsonl"
BANKED = f"{OPS}/banked_hashes.json"
WORKERS_DEFAULT = 13                       # cgroup 15.3 CPUs minus 2; never nproc

_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        _TOK = open("/root/.hf_token").read().strip()
    return _TOK


def log(msg):
    print(f"[v2ep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_ids(name):
    return [l.strip() for l in open(f"{OPS}/{name}", encoding="utf-8") if l.strip()]


def all_clips():
    """[(clip, kind)] in build order: the six AR-2 clips first, then eval-141, then train in manifest order."""
    train, ev = read_ids("train4572.txt"), read_ids("clips141.txt")
    assert len(train) == 4572 and len(set(train)) == 4572, len(train)
    assert len(ev) == 141 and len(set(ev)) == 141, len(ev)
    assert not (set(train) & set(ev)), "train/eval overlap"
    b = json.load(open(BANKED))
    first = [tuple(k.split("/")[::-1]) for k in list(b["gate"]) + list(b["extra"])]      # (clip, kind)
    seen = set(c for c, _ in first)
    order = list(first)
    order += [(c, "eval") for c in ev if c not in seen]
    order += [(c, "train") for c in train if c not in seen]
    assert len(order) == 4713 and len({c for c, _ in order}) == 4713
    return order


def hf_dl(path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF, path, repo_type="dataset", token=tok(), local_dir=SCRATCH)


def ledger_lookup():
    j = json.load(open(f"{SCRATCH}/camera/camera_sha256.json", encoding="utf-8"))
    if isinstance(j, dict) and "files" in j and isinstance(j["files"], dict):
        j = j["files"]

    def hf_sha(c):
        for k in (c, f"{c}.mp4", f"camera/{c}.mp4"):
            if k in j:
                v = j[k]
                return v if isinstance(v, str) else v.get("sha256")
        return None
    return hf_sha


# ----------------------------------------------------------------------------------------------------------- stage
def stage(a):
    import pandas as pd
    import numpy as np
    order = all_clips()
    ids = [c for c, _ in order]
    want = set(ids)
    for d in (CAM, f"{STAGE}/labels/egomotion", f"{STAGE}/calibration", SCRATCH, f"{V2EP}/train", f"{V2EP}/eval"):
        os.makedirs(d, exist_ok=True)
    rep = {"started": time.strftime("%F %T"), "n_clips": len(ids)}

    # the production builder re-asserts the parity ingest gate on its clip list; we call build_one directly, so re-assert it here
    from tanitad.data import parity
    parity.require_ingest_gate("pod_build_b1_epcache")
    kept, gate = parity.guard_corpus_build(ids, label="v2ep_pod stage -> " + V2EP, role="train", mode="exclude")
    rep["parity_gate_kept"] = len(kept)
    rep["parity_gate_dropped"] = sorted(want - set(kept))
    log(f"parity gate: kept {len(kept)} dropped {len(rep['parity_gate_dropped'])}")
    assert len(kept) == len(ids), f"the builder's gate would DROP {rep['parity_gate_dropped']}"

    hf_dl("camera/camera_sha256.json")
    hf_sha = ledger_lookup()
    miss = [c for c in ids if hf_sha(c) is None]
    rep["ledger_missing"] = miss
    assert not miss, f"no HF sha256 for {len(miss)} clips"
    log("mp4 ledger ok for all clips")

    # timestamps
    n_ts = sum(os.path.exists(f"{CAM}/{c}.timestamps.parquet") for c in ids)
    if n_ts < len(ids):
        tp = hf_dl("timestamps/timestamps.tar")
        n_ts = 0
        with tarfile.open(tp) as t:
            for m in t:
                b = os.path.basename(m.name)
                if b.endswith(".timestamps.parquet") and b[:-len(".timestamps.parquet")] in want:
                    with open(f"{CAM}/{b}", "wb") as fh:
                        fh.write(t.extractfile(m).read())
                    n_ts += 1
    rep["timestamps"] = n_ts
    assert n_ts == len(ids), f"timestamps {n_ts} != {len(ids)}"
    log(f"timestamps {n_ts}")

    # egomotion -> one STORED zip with members <clip>.egomotion.parquet (the loader's expected names)
    zpath = f"{STAGE}/labels/egomotion/egomotion_all.zip"
    have = set()
    if os.path.exists(zpath):
        try:
            with zipfile.ZipFile(zpath) as z:
                have = {n[:-len(".egomotion.parquet")] for n in z.namelist()}
        except Exception:                                           # noqa: BLE001
            have = set()
    if not want <= have:
        ep = hf_dl("egomotion/egomotion_alpamayo.tar")
        seen = {}
        with tarfile.open(ep) as t, zipfile.ZipFile(zpath + ".tmp", "w", zipfile.ZIP_STORED) as z:
            for m in t:
                if not m.isfile():
                    continue
                b = os.path.basename(m.name)
                cid = b.split(".")[0]
                if cid in want and cid not in seen:
                    data = t.extractfile(m).read()
                    seen[cid] = hashlib.sha256(data).hexdigest()
                    z.writestr(f"{cid}.egomotion.parquet", data)
        os.replace(zpath + ".tmp", zpath)
        have = set(seen)
    rep["egomotion"] = len(want & have)
    assert want <= have, f"egomotion missing {len(want - have)}"
    log(f"egomotion {len(want & have)} in {zpath}")

    # intrinsics: the dev box's per-clip table (what the production stagers copy), cross-checked to HF's parquet
    csvp = f"{STAGE}/calibration/physicalai_front_wide_intrinsics.csv"
    shutil.copy2(f"{OPS}/intrinsics_dev.csv", csvp)
    rep["intrinsics_sha256"] = sha256(csvp)
    ci = pd.read_parquet(hf_dl("calibration/camera_intrinsics.parquet"))
    camcol = next((c for c in ci.columns if c in ("camera_name", "sensor_name", "camera", "sensor", "feature")), None)
    sub = ci[ci[camcol].astype(str).str.contains("front_wide")] if camcol else ci
    sub = sub.drop_duplicates("clip_id").set_index("clip_id")
    csv = pd.read_csv(csvp, dtype={"clip_id": str}).set_index("clip_id")
    cols = ["cx", "cy", "width", "height"] + [f"fw_poly_{k}" for k in range(5)]
    mism, maxabs, nomatch = 0, 0.0, 0
    for c in ids:
        if c not in csv.index or c not in sub.index:
            nomatch += 1
            continue
        d = float(np.max(np.abs(csv.loc[c, cols].astype(float).to_numpy() - sub.loc[c, cols].astype(float).to_numpy())))
        maxabs = max(maxabs, d)
        mism += d > 1e-9
    rep["intrinsics_vs_hf"] = {"camera_column": camcol, "n_mismatch": int(mism), "max_abs_diff": maxabs, "n_unmatched": nomatch}
    log(f"intrinsics vs HF: mismatch={mism} unmatched={nomatch} max|diff|={maxabs:.3g}")
    assert nomatch == 0 and mism == 0, "intrinsics table does not cover/agree for every clip"
    rep["finished"] = time.strftime("%F %T")
    json.dump(rep, open(f"{DATA}/stage_report.json", "w"), indent=1)
    log("ZZ STAGE DONE ZZ")
    return 0


# ------------------------------------------------------------------------------------------------------ per clip
def process(item):
    """Worker: ensure <clip>.v2ep.pt exists and is content-verified; delete the mp4 only after that. Never raises."""
    cid, kind = item
    t0 = time.time()
    out_dir = f"{V2EP}/{kind}"
    out = f"{out_dir}/{cid}.v2ep.pt"
    res = {"clip": cid, "kind": kind}
    try:
        from pod_pull_b1_epcache import verify
        mp4 = f"{CAM}/{cid}.mp4"
        if os.path.exists(out):
            ok, why = verify(out)
            if ok:
                res.update(status="SKIP", bytes=os.path.getsize(out), sha256=sha256(out), secs=0.0, dl=0.0)
                return res
            os.unlink(out)
        hf_sha = ledger_lookup()
        want = hf_sha(cid)
        td = time.time()
        got = sha256(mp4) if os.path.exists(mp4) else None
        tries = 0
        while got != want:
            tries += 1
            if tries > 3:
                res.update(status="FAIL", err=f"mp4 sha mismatch after 3 downloads ({got} != {want})")
                return res
            if os.path.exists(mp4):
                os.unlink(mp4)
            p = hf_dl(f"camera/{cid}.mp4")
            os.replace(p, mp4)
            got = sha256(mp4)
        res["dl"] = time.time() - td
        import pod_build_b1_epcache as B
        cid2, nb, secs, err = B.build_one((cid, STAGE, out_dir))
        if err:
            res.update(status="FAIL", err=err, secs=secs)
            return res
        res.update(status="OK", bytes=nb, secs=secs, sha256=sha256(out), mp4_sha256=got)
        os.unlink(mp4)                       # only after the v2ep is built AND content-verified (build_one's verify)
        return res
    except Exception as e:                                          # noqa: BLE001
        res.update(status="FAIL", err=f"{type(e).__name__}: {str(e)[:200]}", tb=traceback.format_exc()[-400:])
        return res


def banked_check(res, banked):
    """Compare a finished clip to the banked production hash if it has one. Returns None | True | False."""
    k = f"{res['kind']}/{res['clip']}"
    want = {**banked["gate"], **banked["extra"]}.get(k)
    if want is None or res.get("status") not in ("OK", "SKIP"):
        return None
    return res["sha256"] == want


def run_items(items, workers, chunk, banked, stop_on_banked_mismatch=True):
    ledger = open(LEDGER, "a", buffering=1)
    n_total, n_ok, n_skip, fails, t0 = len(items), 0, 0, [], time.time()
    mism = []
    done = 0
    sum_secs = 0.0
    for c0 in range(0, len(items), chunk):
        part = items[c0:c0 + chunk]
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:       # fresh executor per chunk, no max_tasks_per_child
                futs = [ex.submit(process, it) for it in part]
                for fu in as_completed(futs):
                    r = fu.result()
                    done += 1
                    st = r.get("status")
                    bm = banked_check(r, banked)
                    if st == "OK":
                        n_ok += 1
                        sum_secs += r.get("secs", 0.0)
                    elif st == "SKIP":
                        n_skip += 1
                    else:
                        fails.append(r)
                    if st in ("OK", "SKIP"):
                        ledger.write(json.dumps({k: r.get(k) for k in ("clip", "kind", "bytes", "sha256", "mp4_sha256", "secs", "dl")} | {"t": time.strftime("%F %T")}) + "\n")
                    tag = "" if bm is None else (" BANKED=MATCH" if bm else " BANKED=MISMATCH")
                    log(f"{st:4s} {r['clip'][:8]} {r['kind']:5s} {r.get('bytes', 0) / 1e6:6.1f}MB dl={r.get('dl', 0):4.1f}s "
                        f"build={r.get('secs', 0):5.1f}s{tag}" + (f" ERR {r.get('err')}" if st == "FAIL" else ""))
                    if bm is False:
                        mism.append(r)
                    if done % 50 == 0 or done == n_total:
                        w = time.time() - t0
                        eph = (n_ok + n_skip) / max(w, 1e-9) * 3600
                        left = (n_total - done) / max(eph / 3600, 1e-9)
                        log(f"PROGRESS {done}/{n_total} ok={n_ok} skip={n_skip} fail={len(fails)} {eph:.0f}clips/h "
                            f"elapsed={w / 60:.1f}min eta={left / 60:.1f}min mean_build_s={sum_secs / max(n_ok, 1):.2f}")
        except BrokenProcessPool as e:
            log(f"BrokenProcessPool in chunk starting {c0}: {e}")
            fails.append({"clip": f"chunk@{c0}", "status": "FAIL", "err": "BrokenProcessPool"})
        if mism and stop_on_banked_mismatch:
            log(f"STOP: banked-hash mismatch on {[m['clip'][:8] for m in mism]}")
            break
    ledger.close()
    return dict(n_total=n_total, ok=n_ok, skip=n_skip, fails=fails, mismatch=mism, wall_s=time.time() - t0, sum_build_s=sum_secs)


# --------------------------------------------------------------------------------------------------------- gate/run
def gate(a):
    banked = json.load(open(BANKED))
    items = [tuple(k.split("/")[::-1]) for k in banked["gate"]]      # (clip, kind)
    log(f"GATE: building {[(c[:8], k) for c, k in items]}")
    summ = run_items(items, workers=len(items), chunk=len(items), banked=banked, stop_on_banked_mismatch=False)
    rows = {}
    for line in open(LEDGER):
        r = json.loads(line)
        rows[f"{r['kind']}/{r['clip']}"] = r
    out = {"expected": banked["gate"], "got": {}, "verdict": {}}
    for k, want in banked["gate"].items():
        kind, c = k.split("/")
        p = f"{V2EP}/{kind}/{c}.v2ep.pt"
        got = sha256(p) if os.path.exists(p) else None
        out["got"][k] = got
        out["verdict"][k] = "MATCH" if got == want else "DIFF"
    out["all_match"] = all(v == "MATCH" for v in out["verdict"].values())
    out["run"] = {k: v for k, v in summ.items() if k != "mismatch"}
    json.dump(out, open(f"{DATA}/gate_result.json", "w"), indent=1)
    for k, v in out["verdict"].items():
        log(f"GATE {k[:20]} {v}  got={str(out['got'][k])[:16]} want={banked['gate'][k][:16]}")
    log("ZZ GATE PASS ZZ" if out["all_match"] else "ZZ GATE FAIL ZZ")
    return 0 if out["all_match"] else 3


def run(a):
    banked = json.load(open(BANKED))
    gr = json.load(open(f"{DATA}/gate_result.json")) if os.path.exists(f"{DATA}/gate_result.json") else None
    if not (gr and gr.get("all_match")):
        if not a.waive_gate:
            log("REFUSED: no passing gate_result.json (byte gate must pass first). "
                "A PI/Master-Mind waiver is recorded with --waive-gate '<reason>'.")
            return 4
        log(f"GATE WAIVED by explicit flag: {a.waive_gate}")
        json.dump({"waived": a.waive_gate, "gate_result": gr, "at": time.strftime("%F %T")}, open(f"{DATA}/gate_waiver.json", "w"), indent=1)
    order = all_clips()
    log(f"RUN: {len(order)} clips, workers={a.workers}, chunk={a.chunk}")
    summ = run_items(order, a.workers, a.chunk, banked, stop_on_banked_mismatch=not a.waive_gate)   # under a waiver mismatches are recorded, not stopped on
    # one retry pass for failures
    if summ["fails"] and not summ["mismatch"]:
        retry = [(f["clip"], f["kind"]) for f in summ["fails"] if f.get("kind")]
        if retry:
            log(f"RETRY {len(retry)} failed clips once")
            s2 = run_items(retry, min(a.workers, len(retry)), a.chunk, banked, stop_on_banked_mismatch=not a.waive_gate)
            summ["fails"] = s2["fails"]
            summ["ok"] += s2["ok"]
    # final accounting by content: every clip present, in the ledger, banked extras matched
    led = {}
    for line in open(LEDGER):
        r = json.loads(line)
        led[(r["kind"], r["clip"])] = r
    missing = [(k, c) for c, k in order if (k, c) not in led or not os.path.exists(f"{V2EP}/{k}/{c}.v2ep.pt")]
    bank_bad = []
    for k, want in {**banked["gate"], **banked["extra"]}.items():
        kind, c = k.split("/")
        if led.get((kind, c), {}).get("sha256") != want:
            bank_bad.append(k)
    total_b = sum(os.path.getsize(f"{V2EP}/{k}/{c}.v2ep.pt") for c, k in order if os.path.exists(f"{V2EP}/{k}/{c}.v2ep.pt"))
    final = {"n_expected": len(order), "n_present_in_ledger": len(order) - len(missing), "missing": [c for _, c in missing][:50],
             "n_missing": len(missing), "banked_bad": bank_bad, "n_fail_final": len(summ["fails"]),
             "bytes_total": total_b, "wall_s": summ["wall_s"], "mean_build_s": summ["sum_build_s"] / max(summ["ok"], 1),
             "workers": a.workers, "finished": time.strftime("%F %T")}
    json.dump(final, open(f"{DATA}/v2ep_final.json", "w"), indent=1)
    log("FINAL " + json.dumps(final))
    ok = not missing and not summ["fails"] and (not bank_bad or bool(a.waive_gate))   # banked hashes cannot match under a waiver
    return 0 if ok else 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "gate", "run"])
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--waive-gate", default="", help="reason string; lets `run` proceed after a failed byte gate (needs an authorised decision)")
    a = ap.parse_args()
    return {"stage": stage, "gate": gate, "run": run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
