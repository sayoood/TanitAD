"""fresh2_build_inputs.py -- COPY of refcv9_l1c/code/l1c_build_inputs.py (L1C package) with ONE added set `fresh2` and the v2 sanction test
extended to kit_fresh2 (FRESH2 clips are outside the corpus, exactly like the first pull). Every builder call is unchanged.
Original docstring:
l1c_build_inputs.py -- build, for a list of clips, EVERY input the eval139 roll reads, with the launch tree's own builders (AMENDMENT 1 A1.3).

  prep      mini egomotion zips + per-clip intrinsics csv + sel parquet                      (fetched root layout the v2 builder already supports)
  v2        `v2_compressed.py build` CLI UNCHANGED, K shards                                 -> <kit>/data/refcv6-b1-416x1024-eval139/
  labels    s2_geom_emit_v7.emit_one (Alpamayo absent, stamped) -> build_v8_speed_max_labels.build_split -> <kit>/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz
  speed     build_refcv6_speed_max_window.build  (v_lim sidecar)                              -> <kit>/data/refcv6_speed_max_v8_eval.jsonl
  clock     build_clip_clock_sidecar.py (via l1c_run.py)                                      -> <kit>/data/clock_sidecar.jsonl
  v9        build_v9_labels.py build (no lead / lc pass)                                      -> <kit>/data/v9/v9_labels_eval.npz
  extr      shipped 4,711-clip extrinsics table + these clips' sensor_extrinsics rows         -> <kit>/data/extrinsics_union.json
  kit       anchors + class-weights copies                                                    -> <kit>/data/...
  windows   t % 5 == 0 over the dataset index                                                 -> <kit>/windows_s0.json (raw ids; F: only)
    python l1c_build_inputs.py --set platform --steps prep,v2,labels,speed,clock,v9,extr,kit,windows
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TREE = "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SRC = "C:/Users/Admin/tanitad-data/physicalai"
EKIT6, EKIT8 = "E:/refcv6_eval_kit", "E:/refcv8_eval_kit"
SETS = {
    "fresh": {"root": "F:/l1c/pai_root", "kit": "F:/l1c/kit_fresh", "ids": "F:/l1c/select/final_ids_fresh.json"},
    "platform": {"root": "F:/l1c/pai_root_plat", "kit": "F:/l1c/kit_plat", "ids": "F:/l1c/select/platform_ids.json"},
    "repro": {"root": "F:/l1c/pai_root_repro", "kit": "F:/l1c/kit_repro", "ids": "F:/l1c/select/repro_ids.json"},
    # FRESH2 (PREREG_FRESH2 s1): its OWN root view + kit; the first pull's root and kit_fresh* are never written
    "fresh2": {"root": "F:/l1c/pai_root_fresh2", "kit": "F:/l1c/kit_fresh2", "ids": "F:/l1c/fresh2/fresh2_ids.json"},
}
CAMN = "camera_front_wide_120fov"


def env_for(s) -> dict:
    e = dict(os.environ)
    e.update(TANITAD_STACK=TREE + "/stack", PYTHONPATH=f"{TREE}/stack;{TREE}/stack/scripts;{TREE}/taniteval", TMP="F:/l1c/tmp", TEMP="F:/l1c/tmp",
             OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
             V2_TORCH_THREADS="3", PAI_DECODE_THREADS="3", L1C_EGO_FLAT=s["root"] + "/labels/egomotion_flat",
             TANITAD_ALPAMAYO_RECORDS="F:/l1c/work/__no_alpamayo_records_for_fresh_clips__.parquet", TANITAD_ALPAMAYO_RECORDS_OPTIONAL="1")
    for k in ("REFCV6_KIT", "REFCV7_KIT", "R8_EVAL_KIT"):
        e.pop(k, None)
    return e


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def load_ids(s):
    return json.load(open(s["ids"], encoding="utf-8"))          # [{clip_id, chunk}, ...] in processing order


def step_prep(s):
    root = Path(s["root"])
    ids = load_ids(s)
    (root / "labels" / "egomotion").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "egomotion_flat").mkdir(parents=True, exist_ok=True)
    (root / "r0" / "camera_front_wide").mkdir(parents=True, exist_ok=True)
    (root / "calibration").mkdir(parents=True, exist_ok=True)
    by_chunk = {}
    for r in ids:
        by_chunk.setdefault(int(r["chunk"]), []).append(r["clip_id"])
        c = r["clip_id"]
        fl = root / "labels" / "egomotion_flat" / f"{c}.parquet"
        if not fl.exists():
            src = f"{SRC}/labels/egomotion_alpamayo/{c}.parquet"
            if not os.path.exists(src):
                raise SystemExit(f"[prep] no egomotion for {sha12(c)}")
            fl.write_bytes(open(src, "rb").read())
        for ext in ("mp4", "timestamps.parquet"):
            dst = root / "r0" / "camera_front_wide" / f"{c}.{CAMN}.{ext}"
            if not dst.exists():
                src = f"{SRC}/camera/{CAMN}/{c}.{ext}"
                if os.path.exists(src):
                    dst.write_bytes(open(src, "rb").read())
    for ch, cl in by_chunk.items():
        zp = root / "labels" / "egomotion" / f"egomotion.chunk_{ch:04d}.zip"
        if zp.exists():
            zp.unlink()
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for c in cl:
                z.write(root / "labels" / "egomotion_flat" / f"{c}.parquet", f"{c}.egomotion.parquet")
    # per-clip intrinsics csv (the corpus csv's schema), from the dataset's OWN camera_intrinsics parquet when present, else the local corpus csv
    rows = []
    local = pd.read_csv(f"{SRC}/calibration/physicalai_front_wide_intrinsics.csv").set_index("clip_id")
    for r in ids:
        c, ch = r["clip_id"], int(r["chunk"])
        p = root / "calibration" / "camera_intrinsics" / f"camera_intrinsics.chunk_{ch:04d}.parquet"
        if p.exists():
            d = pd.read_parquet(p).loc[(c, CAMN)]
            rows.append({"clip_id": c, "cx": float(d["cx"]), "cy": float(d["cy"]), "width": int(d["width"]), "height": int(d["height"]),
                         **{f"fw_poly_{i}": float(d[f"fw_poly_{i}"]) for i in range(5)}, "_src": "camera_intrinsics parquet"})
        else:
            d = local.loc[c]
            rows.append({"clip_id": c, **{k: float(d[k]) for k in ("cx", "cy")}, "width": int(d["width"]), "height": int(d["height"]),
                         **{f"fw_poly_{i}": float(d[f"fw_poly_{i}"]) for i in range(5)}, "_src": "local corpus csv"})
    df = pd.DataFrame(rows)
    df.drop(columns=["_src"]).to_csv(root / "calibration" / "physicalai_front_wide_intrinsics.csv", index=False)
    pd.DataFrame([{"clip_id": r["clip_id"], "chunk": int(r["chunk"])} for r in ids]).to_parquet(root / "sel.parquet")
    print(f"[prep] {len(ids)} clips, {len(by_chunk)} chunks; intrinsics sources {df['_src'].value_counts().to_dict()}", flush=True)


def step_v2(s, shards):
    root, kit = Path(s["root"]), Path(s["kit"])
    out = kit / "data" / "refcv6-b1-416x1024-eval139"
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(shards):
        cmd = [PY, "-X", "utf8", TREE + "/stack/scripts/v2_compressed.py", "build", "--sel", str(root / "sel.parquet"), "--root", str(root), "--out", str(out),
               "--hfov", "120", "--height", "416", "--width", "1024", "--projection-mode", "cylindrical", "--codec", "png", "--corpus-role", "eval",
               "--shard", f"{i}/{shards}"]
        if os.path.basename(s["kit"]) not in ("kit_fresh", "kit_fresh2"):      # FRESH2 clips are outside the corpus like the first pull's. The platform control rebuilds EVAL139 clips (some are parity-train members): sanctioned, stamped non-decision-grade
            cmd += ["--sanctioned-audit", "PREREG_L1C A1.5 platform control: eval139 clips rebuilt on the dev box to measure the build-platform effect; not a training corpus"]
        lg = open(f"F:/l1c/logs/v2_{os.path.basename(s['kit'])}_{i}.log", "w")
        procs.append((subprocess.Popen(cmd, env=env_for(s), stdout=lg, stderr=subprocess.STDOUT), lg))
    for p, lg in procs:
        p.wait()
        lg.close()
    n = len(list(out.glob("*.v2ep.pt")))
    print(f"[v2] built files: {n} of {len(load_ids(s))}  rcs={[p.returncode for p, _ in procs]}", flush=True)


def step_labels(s):
    kit = Path(s["kit"])
    sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts"]
    os.environ.update(TANITAD_ALPAMAYO_RECORDS="F:/l1c/work/__no_alpamayo_records_for_fresh_clips__.parquet", TANITAD_ALPAMAYO_RECORDS_OPTIONAL="1")
    import warnings
    warnings.filterwarnings("ignore")
    from tanitad.data import egomotion_source as ES
    ES.EGO_FLAT = s["root"] + "/labels/egomotion_flat/*.parquet"
    ES.EGO_GLOB = s["root"] + "/labels/egomotion_flat/__none__/*.zip"
    ES._index.cache_clear()
    import s2_geom_emit_v7 as E
    import build_v8_speed_max_labels as V8
    built = {p.name[:-8] for p in (kit / "data" / "refcv6-b1-416x1024-eval139").glob("*.v2ep.pt")}
    ids = [r["clip_id"] for r in load_ids(s) if r["clip_id"] in built]
    recs, fail = [], []
    for c in ids:
        try:
            recs.append(E.emit_one(c))
        except Exception as e:      # noqa: BLE001
            fail.append({"sha12": sha12(c), "why": f"{type(e).__name__}: {str(e)[:160]}"})
    ldir = kit / "data" / "v8labels" / "labels"
    ldir.mkdir(parents=True, exist_ok=True)
    v7 = ldir / "_s2_labels_v7_emit.jsonl.gz"
    with gzip.open(v7, "wt", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    cen = V8.build_split(v7, ldir / "s2_labels_v8_eval.jsonl.gz")
    json.dump({"n_emitted": len(recs), "failed": fail, "census": {k: v for k, v in cen.items() if k not in ("src", "dst")},
               "alpamayo": "ABSENT (stamped opt-out)", "emit_one_sha256": hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest()},
              open(kit / "labels_report.json", "w"), indent=1, default=str)
    print(f"[labels] emitted {len(recs)} failed {len(fail)}: {[f['why'] for f in fail][:3]}", flush=True)


def step_speed(s):
    kit = Path(s["kit"])
    sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts"]
    import build_refcv6_speed_max_window as W
    out = kit / "data" / "refcv6_speed_max_v8_eval.jsonl"
    meta = W.build(str(kit / "data" / "v8labels" / "labels" / "s2_labels_v8_eval.jsonl.gz"), str(out), split="eval")
    print(f"[speed] {out} -> {type(meta).__name__}", flush=True)


def run_launch(s, script, args, log):
    cmd = [PY, "-X", "utf8", str(HERE / "l1c_run.py"), script] + args
    with open(log, "w") as lg:
        p = subprocess.run(cmd, env=env_for(s), stdout=lg, stderr=subprocess.STDOUT)
    return p.returncode


def step_clock(s):
    kit = Path(s["kit"])
    rc = run_launch(s, TREE + "/stack/scripts/build_clip_clock_sidecar.py",
                    ["--cache", str(kit / "data" / "refcv6-b1-416x1024-eval139"), "--out", str(kit / "data" / "clock_sidecar.jsonl")],
                    f"F:/l1c/logs/clock_{os.path.basename(s['kit'])}.log")
    p = kit / "data" / "clock_sidecar.jsonl"
    n = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else -1
    print(f"[clock] rc={rc} rows={n}", flush=True)


def step_v9(s):
    kit = Path(s["kit"])
    rc = run_launch(s, TREE + "/stack/scripts/build_v9_labels.py",
                    ["build", "--split", "eval", "--manifest", str(kit / "data" / "refcv6-b1-416x1024-eval139" / "_v2manifest.pt"),
                     "--sidecar", str(kit / "data" / "clock_sidecar.jsonl"), "--labels", str(kit / "data" / "v8labels" / "labels" / "s2_labels_v8_eval.jsonl.gz"),
                     "--ego", s["root"] + "/labels/egomotion_flat/{}.parquet", "--out-dir", str(kit / "data" / "v9")],
                    f"F:/l1c/logs/v9_{os.path.basename(s['kit'])}.log")
    p = kit / "data" / "v9" / "v9_labels_eval.npz"
    md5 = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    if md5:
        (kit / "data" / "v9" / "v9_labels_eval.md5").write_text(str(md5))
    print(f"[v9] rc={rc} npz={p.exists()} md5={md5}", flush=True)


def step_extr(s):
    kit, root = Path(s["kit"]), Path(s["root"])
    base = json.load(open(f"{EKIT8}/data/refcv8_train4572_eval139_extrinsics.json", encoding="utf-8"))
    out, added = dict(base), 0
    for r in load_ids(s):
        c, ch = r["clip_id"], int(r["chunk"])
        if c in base:
            continue
        p = root / "calibration" / "sensor_extrinsics" / f"sensor_extrinsics.chunk_{ch:04d}.parquet"
        df = pd.read_parquet(p)
        if "clip_id" not in df.columns:
            df = df.reset_index()
        nm = "sensor_name" if "sensor_name" in df.columns else "name"
        d = df[(df["clip_id"].astype(str) == c) & (df[nm].astype(str) == CAMN)]
        if len(d) != 1:
            raise SystemExit(f"[extr] {sha12(c)}: {len(d)} front-wide rows")
        d = d.iloc[0]
        out[c] = {"qx": float(d["qx"]), "qy": float(d["qy"]), "qz": float(d["qz"]), "qw": float(d["qw"]), "x": float(d["x"]), "y": float(d["y"]),
                  "z": float(d["z"]), "chunk": ch, "sensor": CAMN, "source": f"sensor_extrinsics.chunk_{ch:04d}.parquet (PhysicalAI-AV sensor_extrinsics)",
                  "_read": next(iter(base.values())).get("_read")}
        added += 1
    json.dump(out, open(kit / "data" / "extrinsics_union.json", "w", encoding="utf-8"))
    print(f"[extr] union {len(out)} (shipped {len(base)}, added {added})", flush=True)


def step_kit(s):
    import shutil
    kit = Path(s["kit"])
    for rel, src in (("data/anchors/refc_anchors_6s_v0cond_alat_117.pt", f"{EKIT6}/data/anchors/refc_anchors_6s_v0cond_alat_117.pt"),
                     ("data/refcv7/map_hires_class_weights_train_100x30.json", f"{EKIT6}/data/refcv7/map_hires_class_weights_train_100x30.json")):
        dst = kit / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
    print("[kit] anchors + class weights in place", flush=True)


def step_windows(s):
    kit = Path(s["kit"])
    sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts"]
    from tanitad.data.v2_dataset import build_v2_providers, load_or_build_manifest
    cache = str(kit / "data" / "refcv6-b1-416x1024-eval139")
    man = load_or_build_manifest(cache, verbose=False)
    clip_ids = [str(c) for c in man["clip_id"]]
    import importlib.util
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_arm", TREE + "/stack/scripts/refc_v3_train.py")
    tr = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_arm"] = tr
    spec.loader.exec_module(tr)
    eps = build_v2_providers([cache], lru_size=2)
    W = 8
    ds = tr.V3Dataset(eps, window=W, max_horizon=20, channels=9)
    win = {}
    for e_i, t in ds.index:
        if int(t) % 5 == 0:
            win.setdefault(clip_ids[e_i], []).append(int(t) + W - 1)
    win = {c: sorted(v) for c, v in win.items()}
    json.dump(win, open(kit / "windows_s0.json", "w", encoding="utf-8"))
    n = sum(len(v) for v in win.values())
    print(f"[windows] {len(win)} clips / {n} windows (index {len(ds.index)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--steps", default="prep,v2,labels,speed,clock,v9,extr,kit,windows")
    ap.add_argument("--shards", type=int, default=3)
    a = ap.parse_args()
    s = SETS[a.set]
    fns = {"prep": step_prep, "v2": lambda s_: step_v2(s_, a.shards), "labels": step_labels, "speed": step_speed, "clock": step_clock, "v9": step_v9,
           "extr": step_extr, "kit": step_kit, "windows": step_windows}
    for st in a.steps.split(","):
        t0 = time.time()
        fns[st](s)
        print(f"[step {st}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
