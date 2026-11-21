"""Dev-box side of NAVSIM-on-Colab P0: assemble the PRIVATE sub200 inputs for the relay, and the PUBLIC DB fetch list.

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe colab/navtest/build_sub200_bundle.py --out <scratch dir>
        [--evidence colab/raw/<date>-navtest-p0]

Writes, under --out (the scratchpad, never the repo -- ~224 MB of derived private data):
  p0_harness_sub200.tar  the harness inputs: the dev box's 200 metric-cache pickles (WindowsPath, loaded on Linux
                         with the shim), the export subset, the token list, and the ep016 REFERENCES (seam + report,
                         E-6 proposal dump, E-6 table, per-token CSV + counts) that G-H / G-E6 / G-S / G-P read
  p0_frames_sub200.tar   one <log>_<CAM>.zip per (log, camera) holding EXACTLY the JPEG members the planner's own
                         FrameResolver resolves for each sub200 token (found by running the resolver, not assumed)
and, under --evidence (small, no private bytes):
  db_manifest_sub200.json  the 93 test-DB members of the PUBLIC nuplan-v1.1_test.zip: local-header offset, sizes, CRC,
                           read from the dev box's copy of the same zip (the plan MEASURED 136/136 navtest members
                           identical in csz/usz/CRC to the remote central directory) + the dev box's own DB CRC32, so a
                           VM-side CRC match proves the VM's DB bytes equal the dev box's
  frames_sub200_manifest.json  per token, per camera: the resolved member, its sha256, and whether it equals W3's
                               tokens_frames.json pick
  bundle_manifest.json   every tar member with bytes + sha256

The frame resolution runs in the driverl-eval venv with eval_checkpoint.env_driverl() (REFE_SIM_HZ=10 and the rest)
and CUDA hidden -- the seam's scenario build + FrameResolver.resolve at ego.time_point.time_us, no model.
CPU: one process, OMP_NUM_THREADS=1 (the dev box's M6 chain owns the rest).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile
import zlib

PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TOKENS = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
          "A1_sub200_tokens.json")
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
DATA = "D:/Projects/TanitAD/data/refe_navtest"
FRAMES = f"{DATA}/frames"
DB_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
TEST_ZIP = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1_test.zip"
TEST_ZIP_URL = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip"
MC = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
NAME = "sub200_ep016"
REFS = {
    "ref/refe_sub200_ep016.npz": f"{DATA}/seams/refe_{NAME}.npz",
    "ref/refe_sub200_ep016.report.json": f"{DATA}/seams/refe_{NAME}.report.json",
    "ref/proposals.npz": f"{DATA}/proptable/{NAME}/proposals.npz",
    "ref/table.npz": f"{DATA}/proptable/{NAME}/table.npz",
    "ref/gates.json": f"{DATA}/proptable/{NAME}/gates.json",
    "ref/refe_sub200_ep016.csv": f"{DATA}/score/refe_{NAME}/refe_{NAME}.csv",
    "ref/refe_sub200_ep016.counts.json": f"{DATA}/score/refe_{NAME}/refe_{NAME}.counts.json",
}
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def crc32_file(p: str) -> int:
    c = 0
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


# ---- runs INSIDE the driverl-eval venv (spawned by main) -------------------------------------------------------
def resolve_frames(out_json: str) -> int:
    sys.path.insert(0, f"{PKG}/refe")
    import navtrain_scenarios as NS                       # noqa: E402  (REFE_SIM_HZ read at import)
    from planner import FrameResolver                     # noqa: E402
    assert NS.SIM_HZ == 10, f"REFE_SIM_HZ must be 10 (the eval's), got {NS.SIM_HZ}"
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    want = set(json.load(open(TOKENS, encoding="utf-8"))["tokens"])
    toks = [t for t in exp if t in want]                  # the seam's own order
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    fr = FrameResolver(DB_DIR, FRAMES)
    out, order, t0 = {}, [], time.time()
    for li, (log, lt) in enumerate(sorted(by_log.items())):
        for sc in NS.build_scenarios_for_log(os.path.join(DB_DIR, f"{log}.db"), lt, history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            ego = sc.get_ego_state_at_iteration(0)
            paths = fr.resolve(sc.log_name, int(ego.time_point.time_us))
            out[tok] = {"log": sc.log_name, "time_us": int(ego.time_point.time_us),
                        "paths": paths, "miss": None if paths else fr.miss_reason}
            order.append(tok)
        if (li + 1) % 10 == 0:
            print(f"  resolved {li + 1}/{len(by_log)} logs, {len(out)} tokens, {time.time() - t0:.0f} s", flush=True)
    json.dump({"order": order, "tokens": out, "n_asked": len(toks), "seconds": round(time.time() - t0, 1)},
              open(out_json, "w", encoding="utf-8"), indent=1)
    print(f"ZZRESOLVED {len(out)}/{len(toks)} in {time.time() - t0:.0f} s")
    return 0 if len(out) == len(toks) and all(v["paths"] for v in out.values()) else 1


# ---- dev box orchestration ------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=False)
    ap.add_argument("--evidence", default="D:/Projects/TanitAD/colab/raw/2026-09-27-navtest-p0")
    ap.add_argument("--resolve-only", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.resolve_only:
        return resolve_frames(a.resolve_only)
    t0 = time.time()
    out = os.path.abspath(a.out)
    if "D:/Projects/TanitAD" in out.replace("\\", "/"):
        sys.exit("write the private bundle to the scratchpad, never the repo")
    os.makedirs(out, exist_ok=True)
    os.makedirs(a.evidence, exist_ok=True)
    doc = json.load(open(TOKENS, encoding="utf-8"))
    toks = list(doc["tokens"])
    logs = sorted({doc["token_log"][t] for t in toks})
    print(f"sub200: {len(toks)} tokens / {len(logs)} logs")

    # 1. frames: run the planner's own resolver in the driverl venv
    sys.path.insert(0, f"{PKG}/eval")
    import eval_checkpoint as EC                          # noqa: E402  (constants + env_driverl only)
    res_json = os.path.join(out, "frames_resolved.json")
    if not os.path.exists(res_json):
        env = EC.env_driverl()
        env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        rc = subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), "--resolve-only", res_json], env=env)
        if rc != 0:
            sys.exit(f"frame resolution failed rc={rc}")
    R = json.load(open(res_json, encoding="utf-8"))
    assert set(R["tokens"]) == set(toks), "the resolver did not return every sub200 token"
    tf = {r["token"]: r for r in json.load(open(f"{FRAMES}/tokens_frames.json", encoding="utf-8"))}
    members: dict = {}                                    # (log, cam) -> {member: bytes}
    fman = {}
    for tok in R["order"]:
        rec = R["tokens"][tok]
        row = {"log": rec["log"], "time_us": rec["time_us"], "cams": {}}
        w3 = {os.path.basename(c.replace("\\", "/")) for c in tf.get(tok, {}).get("cams", [])}
        for cam, p in zip(CAMERAS, rec["paths"]):
            zp, mem = p.split("::", 1)
            assert os.path.basename(zp) == f"{rec['log']}_{cam}.zip", (zp, cam)
            b = zipfile.ZipFile(zp).read(mem)
            members.setdefault((rec["log"], cam), {})[mem] = b
            row["cams"][cam] = {"member": mem, "bytes": len(b), "sha256": sha256_bytes(b), "in_w3_tokens_frames": mem in w3}
        fman[tok] = row
    # 2. the harness tar
    exp_all = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))
    sub_exp = {k: v for k, v in exp_all.items() if k != "tokens"}
    sub_exp["tokens"] = {t: v for t, v in exp_all["tokens"].items() if t in set(toks)}
    sub_exp["subset_note"] = f"sub200 subset of {EXPORT} (key order preserved) for NAVSIM-on-Colab P0"
    assert len(sub_exp["tokens"]) == len(toks)
    bman: dict = {}

    def add(tar: tarfile.TarFile, arc: str, data: bytes):
        ti = tarfile.TarInfo(arc)
        ti.size = len(data)
        ti.mtime = 0
        tar.addfile(ti, io.BytesIO(data))
        bman.setdefault(os.path.basename(tar.name), {})[arc] = {"bytes": len(data), "sha256": sha256_bytes(data)}

    ht = os.path.join(out, "p0_harness_sub200.tar")
    with tarfile.open(ht, "w") as tar:
        add(tar, "tokens/A1_sub200_tokens.json", open(TOKENS, "rb").read())
        add(tar, "inputs/navtest_inputs_sub200.json.gz",
            gzip.compress(json.dumps(sub_exp).encode("utf-8"), mtime=0))
        for arc, src in REFS.items():
            add(tar, arc, open(src, "rb").read())
        n_mc = 0
        for t in toks:
            p = f"{MC}/{doc['token_log'][t]}/unknown/{t}/metric_cache.pkl"
            add(tar, f"metric_cache_navtest/{doc['token_log'][t]}/unknown/{t}/metric_cache.pkl", open(p, "rb").read())
            n_mc += 1
        assert n_mc == 200
    ft = os.path.join(out, "p0_frames_sub200.tar")
    with tarfile.open(ft, "w") as tar:
        for (log, cam), mems in sorted(members.items()):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                for mem, b in sorted(mems.items()):
                    zi = zipfile.ZipInfo(mem, date_time=(2026, 9, 27, 0, 0, 0))
                    z.writestr(zi, b)
            add(tar, f"frames/{log}_{cam}.zip", buf.getvalue())
    # 3. the public DB fetch list, from the dev box's copy of the same public zip, + the dev box's own DB CRC
    z = zipfile.ZipFile(TEST_ZIP)
    info = {os.path.basename(i.filename)[:-3]: i for i in z.infolist() if i.filename.endswith(".db")}
    dbm = {"url": TEST_ZIP_URL, "zip_bytes": os.path.getsize(TEST_ZIP), "members": {}}
    for log in logs:
        i = info[log]
        local = os.path.join(DB_DIR, f"{log}.db")
        dbm["members"][log] = {"name": i.filename, "lho": i.header_offset, "csz": i.compress_size,
                               "usz": i.file_size, "crc": i.CRC, "method": i.compress_type,
                               "devbox_db_bytes": os.path.getsize(local), "devbox_db_crc": crc32_file(local)}
    dbm["n"] = len(dbm["members"])
    dbm["csz_total"] = sum(m["csz"] for m in dbm["members"].values())
    dbm["usz_total"] = sum(m["usz"] for m in dbm["members"].values())
    dbm["devbox_equals_zip"] = all(m["devbox_db_crc"] == m["crc"] and m["devbox_db_bytes"] == m["usz"]
                                   for m in dbm["members"].values())
    json.dump(dbm, open(os.path.join(a.evidence, "db_manifest_sub200.json"), "w", encoding="utf-8"), indent=1)
    json.dump({"what": "per sub200 token (seam order), per camera: the member planner.FrameResolver resolves at "
                       "ego.time_point.time_us on the dev box, its bytes and sha256",
               "n_tokens": len(fman), "n_jpeg": sum(len(r["cams"]) for r in fman.values()),
               "n_zip": len(members),
               "n_not_in_w3_tokens_frames": sum(1 for r in fman.values() for c in r["cams"].values()
                                                if not c["in_w3_tokens_frames"]),
               "tokens": fman}, open(os.path.join(a.evidence, "frames_sub200_manifest.json"), "w", encoding="utf-8"),
              indent=1)
    for p in (ht, ft):
        bman[os.path.basename(p) + "::tar"] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    json.dump(bman, open(os.path.join(a.evidence, "bundle_manifest.json"), "w", encoding="utf-8"), indent=1)
    print(f"ZZBUNDLE_OK harness {os.path.getsize(ht):,} B, frames {os.path.getsize(ft):,} B "
          f"({len(members)} zips, {sum(len(v) for v in members.values())} JPEGs); DB list {dbm['n']} members "
          f"{dbm['csz_total']:,} B compressed, devbox==zip {dbm['devbox_equals_zip']}; {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
