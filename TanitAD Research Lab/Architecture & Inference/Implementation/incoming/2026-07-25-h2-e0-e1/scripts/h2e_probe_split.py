"""H2 E1/E0 — establish the HELD-OUT split. Counts only; no labels computed here.

Purpose: prove that the episodes we will test on were NOT used in the threshold sweep
that produced the 2.22x @ 3.0 m headline (H2_SUBSTRATE_AND_LABELING.md Sec 6.4).
"""
import glob, io, os, zipfile
import pandas as pd

DR = r"C:\Users\Admin\tanitad-data\physicalai"
S = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad"


def chunks_of(pat):
    return {os.path.basename(p).split("_")[-1].split(".")[0] for p in glob.glob(pat)}


ob = chunks_of(DR + r"\labels\obstacle.offline\*.zip")
ci = chunks_of(DR + r"\calibration\camera_intrinsics\*.parquet")
se = chunks_of(DR + r"\calibration\sensor_extrinsics\*.parquet")
eg = chunks_of(DR + r"\labels\egomotion\*.zip")

usable = sorted(ob & ci & se & eg)
print("obstacle.offline chunks :", len(ob))
print("camera_intrinsics chunks:", len(ci))
print("sensor_extrinsics chunks:", len(se))
print("egomotion chunks        :", len(eg))
print("USABLE (all four)       :", usable)

# --- what the SWEEP actually used ---
sweep = pd.read_parquet(S + r"\crux3.parquet", columns=["clip_id", "chunk"])
sweep_clips = set(sweep.clip_id.unique())
print("\nSWEEP (crux3.parquet): chunks", sorted(sweep.chunk.unique()),
      "clips", len(sweep_clips))

held = [c for c in usable if c not in set(sweep.chunk.unique())]
print("HELD-OUT chunk candidates:", held)

# --- per-chunk clip availability for the held-out chunks ---
CI = pd.concat([pd.read_parquet(f) for f in
                glob.glob(DR + r"\calibration\camera_intrinsics\*.parquet")]).reset_index()
calib_clips = set(CI.clip_id.astype(str))
sel = pd.read_parquet(DR + r"\r0\phase0_selection.parquet")
corpus = set(sel.clip_id.astype(str))

rows = []
for ch in usable:
    z = zipfile.ZipFile(DR + rf"\labels\obstacle.offline\obstacle.offline.chunk_{ch}.zip")
    clips = [n.split("/")[-1].split(".")[0] for n in z.namelist() if n.endswith(".parquet")]
    ze = zipfile.ZipFile(DR + rf"\labels\egomotion\egomotion.chunk_{ch}.zip")
    egoclips = {n.split("/")[-1].split(".")[0] for n in ze.namelist() if n.endswith(".parquet")}
    ok = [c for c in clips if c in calib_clips and c in egoclips]
    rows.append(dict(chunk=ch, n_obstacle=len(clips), n_with_calib_and_ego=len(ok),
                     n_in_phase0=sum(c in corpus for c in ok),
                     n_in_sweep=sum(c in sweep_clips for c in ok)))
print("\n", pd.DataFrame(rows).to_string(index=False))

# country mix
m = sel[sel.clip_id.astype(str).isin(calib_clips)]
print("\ncountry mix of clips with local calibration (top 12):")
print(m.country.value_counts().head(12).to_string())
