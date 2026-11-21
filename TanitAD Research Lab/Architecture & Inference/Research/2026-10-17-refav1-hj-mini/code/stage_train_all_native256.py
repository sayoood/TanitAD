"""RA-1 FULL CORPUS (PREREG_RA1_FULL.md): stage the INPUTS for the native 256x640 cylindrical v2ep of ALL 4,572 TRAIN clips.
Derived from stage_train600_native256.py with three changes only: --n defaults to 0 (= all), the stack import is repo-relative
(the E: working tree lags the tip), and clips_rest.txt lists the clips NOT in the 600 already built (the ones box 2 builds).

(original header follows)
S4b: stage the INPUTS for rebuilding the native 256x640 cylindrical v2ep of the 600 stage-A TRAIN clips.

The builder is the banked one (`stack/scripts/pod_build_b1_epcache.py` -> `v2_compressed.build_compressed`), driven
UNCHANGED. This script only materialises the layout it expects and PROVES each input is the HF corpus's bytes:

    <stage>/r0/camera_front_wide/<clip>.mp4                hardlink of the local copy of HF camera/<clip>.mp4,
                                                           sha256-checked against HF camera/camera_sha256.json
    <stage>/r0/camera_front_wide/<clip>.timestamps.parquet extracted from HF timestamps/timestamps.tar
    <stage>/labels/egomotion/egomotion_all.zip             STORED zip of <clip>.egomotion.parquet members taken from
                                                           HF egomotion/egomotion_alpamayo.tar (the pod build's route)
    <stage>/calibration/physicalai_front_wide_intrinsics.csv   local per-clip table, cross-checked to HF
                                                           calibration/camera_intrinsics.parquet
    <stage>/clips600.txt                                   the picker's 600 (pull_encode_pooled.pick_clips, train, n=600)

CPU only, no Thor. HF token is read in place and never printed.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import argparse, hashlib, json, re, shutil, sys, tarfile, time, zipfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[4] / "stack"))
import numpy as np, pandas as pd, torch
assert not torch.cuda.is_available(), "CPU only: CUDA must be hidden"

import pull_encode_pooled as P

HF = "Sayood/tanitad-v7-training-corpus"
LOCAL_MP4 = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
LOCAL_INTR = Path("C:/Users/Admin/tanitad-data/physicalai/calibration/physicalai_front_wide_intrinsics.csv")
LOC_EGO = Path("C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo")


def tok():
    return re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def picks(n=600):
    a = SimpleNamespace(eval_eps=P.D_EVAL_EPS, eval_cache=P.D_EVAL_CACHE, eval_labels=P.D_EVAL_LABELS,
                        train_labels=P.D_TRAIN_LABELS, listing=P.D_INDEX, thor=P.THOR)
    sets = P.clip_sets(a)
    pk = P.pick_clips(sets, "train", n)
    assert len(set(pk)) == len(pk) and (n == 0 or len(pk) == n), (len(pk), n)   # n = 0 means ALL (pick_clips)
    assert not (set(pk) & set(sets["eval"])), "pick overlaps eval-141"
    return pk, sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="C:/Users/Admin/tanitad-data/refav1-train4572-256_stage")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--scratch", default="C:/Users/Admin/tanitad-data/refav1-train4572-256_stage/_hf")
    ap.add_argument("--built", default="C:/Users/Admin/tanitad-data/refav1-train600-256_stage/clips600.txt",
                    help="the clips already built (excluded from clips_rest.txt)")
    a = ap.parse_args()
    stage, scratch = Path(a.stage), Path(a.scratch)
    cam = stage / "r0" / "camera_front_wide"
    for d in (cam, stage / "labels" / "egomotion", stage / "calibration", scratch):
        d.mkdir(parents=True, exist_ok=True)
    rep = {"started": time.strftime("%F %T")}
    pk, sets = picks(a.n)
    (stage / "clips_all.txt").write_text("\n".join(pk) + "\n")
    built = [l.strip() for l in open(a.built, encoding="utf-8") if l.strip()]
    assert set(built) <= set(pk), "a built clip is not in the full pick"
    rest = [c for c in pk if c not in set(built)]
    (stage / "clips_rest.txt").write_text("\n".join(rest) + "\n")
    rep["n_built_before"], rep["n_rest"] = len(built), len(rest)
    print(f"[stage] all {len(pk)} = built {len(built)} + rest {len(rest)}", flush=True)
    rep["n_pick"] = len(pk)
    rep["pick_eval_overlap"] = len(set(pk) & set(sets["eval"]))
    rep["split_checks"] = sets["checks"]
    print(f"[stage] picked {len(pk)}; overlap with eval-141 = {rep['pick_eval_overlap']}", flush=True)

    from huggingface_hub import hf_hub_download
    T = tok()

    def dl(f):
        return hf_hub_download(HF, f, repo_type="dataset", token=T, local_dir=str(scratch))

    # 1) mp4: local copy of the HF file, sha256 against HF's own ledger
    shaj = json.load(open(dl("camera/camera_sha256.json"), encoding="utf-8"))
    if isinstance(shaj, dict) and "files" in shaj and isinstance(shaj["files"], dict):
        shaj = shaj["files"]

    def hf_sha(c):
        for k in (c, f"{c}.mp4", f"camera/{c}.mp4"):
            if k in shaj:
                v = shaj[k]
                return v if isinstance(v, str) else v.get("sha256")
        return None

    t0 = time.time(); bad = []; missing = []
    for i, c in enumerate(pk):
        src = LOCAL_MP4 / f"{c}.mp4"
        dst = cam / f"{c}.mp4"
        if not src.exists():
            missing.append(c); continue
        want = hf_sha(c)
        if want is None:
            raise SystemExit(f"no HF sha256 for {c}")
        if sha256(src) != want:
            bad.append(c); continue
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        if (i + 1) % 100 == 0:
            print(f"[stage] mp4 sha ok {i + 1}/{len(pk)} {time.time() - t0:.0f}s", flush=True)
    rep["mp4_missing_local"] = missing
    rep["mp4_sha_mismatch"] = bad
    print(f"[stage] mp4: missing={len(missing)} sha_mismatch={len(bad)} ({time.time() - t0:.0f}s)", flush=True)
    for c in missing + bad:           # fall back to the HF copy for any clip the local copy cannot vouch for
        p = hf_hub_download(HF, f"camera/{c}.mp4", repo_type="dataset", token=T, local_dir=str(scratch))
        assert sha256(p) == hf_sha(c), f"HF download sha mismatch {c}"
        shutil.copy2(p, cam / f"{c}.mp4")
        os.unlink(p)
    rep["mp4_fetched_from_hf"] = len(missing) + len(bad)

    # 2) timestamps + egomotion from the HF tars
    want = set(pk)
    ts_tar = dl("timestamps/timestamps.tar")
    n_ts = 0
    with tarfile.open(ts_tar) as t:
        for m in t:
            b = os.path.basename(m.name)
            if b.endswith(".timestamps.parquet") and b[:-len(".timestamps.parquet")] in want:
                (cam / b).write_bytes(t.extractfile(m).read())
                n_ts += 1
    rep["timestamps_from_hf_tar"] = n_ts
    assert n_ts == len(pk), f"timestamps {n_ts} != {len(pk)}"
    ts_diff = [c for c in pk if (LOCAL_MP4 / f"{c}.timestamps.parquet").exists()
               and sha256(LOCAL_MP4 / f"{c}.timestamps.parquet") != sha256(cam / f"{c}.timestamps.parquet")]
    rep["timestamps_local_differs_from_hf"] = ts_diff
    print(f"[stage] timestamps {n_ts}; local!=hf: {len(ts_diff)}", flush=True)

    eg_tar = dl("egomotion/egomotion_alpamayo.tar")
    out_zip = stage / "labels" / "egomotion" / "egomotion_all.zip"
    seen = {}
    with tarfile.open(eg_tar) as t, zipfile.ZipFile(str(out_zip) + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for m in t:
            if not m.isfile():
                continue
            b = os.path.basename(m.name)
            cid = b.split(".")[0]
            if cid in want and cid not in seen:
                data = t.extractfile(m).read()
                seen[cid] = hashlib.sha256(data).hexdigest()
                z.writestr(f"{cid}.egomotion.parquet", data)
    os.replace(str(out_zip) + ".tmp", out_zip)
    rep["egomotion_from_hf_tar"] = len(seen)
    assert len(seen) == len(pk), f"egomotion {len(seen)} != {len(pk)}"
    ego_diff = [c for c in pk if (LOC_EGO / f"{c}.parquet").exists() and sha256(LOC_EGO / f"{c}.parquet") != seen[c]]
    rep["egomotion_local_differs_from_hf"] = ego_diff
    rep["egomotion_local_present"] = sum((LOC_EGO / f"{c}.parquet").exists() for c in pk)
    print(f"[stage] egomotion {len(seen)}; local present {rep['egomotion_local_present']}, local!=hf {len(ego_diff)}",
          flush=True)

    # 3) intrinsics: local per-clip CSV vs HF camera_intrinsics.parquet (front-wide rows)
    shutil.copy2(LOCAL_INTR, stage / "calibration" / "physicalai_front_wide_intrinsics.csv")
    ci = pd.read_parquet(dl("calibration/camera_intrinsics.parquet"))
    rep["hf_intrinsics_columns"] = list(ci.columns)
    print("[stage] HF intrinsics columns:", list(ci.columns), flush=True)
    csv = pd.read_csv(LOCAL_INTR, dtype={"clip_id": str}).set_index("clip_id")
    camcol = next((c for c in ci.columns if c in ("camera_name", "sensor_name", "camera", "sensor", "feature")), None)
    sub = ci[ci[camcol].astype(str).str.contains("front_wide")] if camcol else ci
    sub = sub.drop_duplicates("clip_id").set_index("clip_id")
    cols = ["cx", "cy", "width", "height"] + [f"fw_poly_{k}" for k in range(5)]
    mism = 0; maxabs = 0.0
    for c in pk:
        a_, b_ = csv.loc[c, cols].astype(float).to_numpy(), sub.loc[c, cols].astype(float).to_numpy()
        d = float(np.max(np.abs(a_ - b_)))
        maxabs = max(maxabs, d)
        mism += d > 1e-9
    rep["intrinsics_vs_hf"] = {"camera_column": camcol, "n_front_wide_rows": int(len(sub)), "n_mismatch": int(mism),
                               "max_abs_diff": maxabs}
    print(f"[stage] intrinsics vs HF: mismatch={mism} max|diff|={maxabs:.3g}", flush=True)
    rep["finished"] = time.strftime("%F %T")
    (stage / "stage_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("ZZSTAGE-DONEZZ", flush=True)


if __name__ == "__main__":
    main()
