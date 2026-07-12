"""PhysicalAI-AV Phase-0 builder — SCENARIO-STRATIFIED (DATA_STRATEGY.md §2,
D-012; usage tagged `data:physicalai`).

Strategy: egomotion labels are ~40 MB/chunk while camera chunks are ~2 GB, so
selection runs on egomotion FIRST, then only selected clips' camera chunks are
fetched. Scenario has no explicit tag in the dataset; we derive it from an
egomotion MOTION PROXY per clip:

    highway   mean_speed >= 15 m/s (~54 km/h), stop_frac < 0.05, low yaw
    urban     low speed (< 8 m/s) WITH interaction (stops and/or turns)
    suburban  everything else (arterial / mixed) — the residual bucket

The Stage-R0 predecessor scored an URBAN-ONLY score with a hard gate
`2 <= mean_v <= 14`, which scored every >=15 m/s clip 0 and yielded a corpus
that was 0 % highway (all <= 41 km/h). This builder scores clips across MANY
chunks (stratified over countries) and stratified-samples the selection over

    scenario  x  country  x  time-of-day  x  platform (hyperion_8 / 8.1)

so the Phase-0 corpus includes highway (target >= 25 %, roughly thirds).

Stages:
  select        score every cached egomotion chunk (optionally download more
                first, stratified by country), classify scenario, stratified-
                sample >= `--target` clips, write phase0_selection.parquet +
                PHASE0_REPORT.{md,json} (achieved scenario/country/tod/platform
                distribution, speed histogram, total hours).
  fetch-camera  download camera chunks + per-clip calibration chunks for the
                selected clips, extract only selected members, delete the zips.

Usage:
  python scripts/physicalai_r0.py select --download 167 --target 2800
  python scripts/physicalai_r0.py select --target 2800          # cached only
  python scripts/physicalai_r0.py fetch-camera --max-chunks 999
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TANITAD_PHYSICALAI_ROOT",
                           r"C:\Users\Admin\tanitad-data\physicalai"))
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
EGO_TMPL = "labels/egomotion/egomotion.chunk_{chunk_id:04d}.zip"
CAM_TMPL = ("camera/camera_front_wide_120fov/"
            "camera_front_wide_120fov.chunk_{chunk_id:04d}.zip")
CALIB_TMPL = "calibration/{kind}/{kind}.chunk_{chunk_id:04d}.parquet"
CALIB_KINDS = ("camera_intrinsics", "sensor_extrinsics")

# --- scenario thresholds (egomotion motion proxies; calibrated on 30 chunks) --
HW_MIN_SPEED = 15.0     # m/s (~54 km/h) sustained mean -> highway floor
HW_MAX_STOP = 0.05      # highway does not stop
HW_MAX_YAW = 0.03       # rad/s mean |kappa|*v — highway is near-straight
URBAN_MAX_SPEED = 8.0   # m/s — urban is low-speed
URBAN_MIN_STOP = 0.05   # ... with stopping ...
URBAN_MIN_YAW = 0.04    # ... and/or turning interaction
PARKED_SPEED = 1.0      # mean_v below this = parked/invalid, dropped


def _hf():
    try:
        # dev machine: proxy TLS + Keys.txt. On pods, export HF_TOKEN instead.
        from tanitad.keys import enable_tls, load_keys
        enable_tls()
        load_keys()
    except Exception as e:  # noqa: BLE001
        print(f"[p0] keys helper unavailable ({e}) — relying on env HF_TOKEN")
    from huggingface_hub import hf_hub_download
    return hf_hub_download


def _load_catalog() -> pd.DataFrame:
    idx = pd.read_parquet(ROOT / "clip_index.parquet")
    dc = pd.read_parquet(ROOT / "metadata" / "data_collection.parquet")
    cat = idx.join(dc)
    cat = cat[cat["clip_is_valid"] & (cat["split"].astype(str) == "train")]
    return cat


def tod_bucket(hour: int) -> str:
    """Coarse time-of-day from hour_of_day (local): the two commute shoulders
    are the dawn/dusk bucket, mid-day is `day`, the rest is `night`."""
    h = int(hour)
    if 6 <= h < 9 or 17 <= h < 20:
        return "dawn_dusk"
    if 9 <= h < 17:
        return "day"
    return "night"


# --------------------------------------------------------------------------- #
# Scenario scoring from egomotion.                                            #
# --------------------------------------------------------------------------- #
def classify_scenario(mean_v: float, stop_frac: float, yaw_act: float) -> str:
    if mean_v >= HW_MIN_SPEED and stop_frac < HW_MAX_STOP and yaw_act < HW_MAX_YAW:
        return "highway"
    if mean_v < URBAN_MAX_SPEED and (stop_frac >= URBAN_MIN_STOP
                                     or yaw_act >= URBAN_MIN_YAW):
        return "urban"
    return "suburban"


def scenario_stats_from_egomotion(df: pd.DataFrame) -> dict:
    """Motion statistics + scenario label against the ACTUAL egomotion schema:
    timestamp, q{x,y,z,w}, x/y/z [m], v{x,y,z} [m/s], a{x,y,z}, curvature."""
    need = {"vx", "vy", "curvature"}
    if not need.issubset(df.columns):
        raise ValueError(f"unexpected egomotion schema: {list(df.columns)}")
    v = np.hypot(df["vx"].to_numpy(np.float64), df["vy"].to_numpy(np.float64))
    curv = df["curvature"].to_numpy(np.float64)
    yaw_rate = np.abs(curv) * v                      # |kappa| * v = yaw rate
    mean_v = float(np.nanmean(v))
    stop_frac = float(np.mean(v < 0.5))
    yaw_act = float(np.nanmean(yaw_rate))
    return {
        "mean_speed": mean_v,
        "p85_speed": float(np.nanpercentile(v, 85)),
        "max_speed": float(np.nanmax(v)),
        "stop_frac": stop_frac,
        "yaw_activity": yaw_act,
        "distance": mean_v * 20.0,                   # clips are 20 s
        "scenario": classify_scenario(mean_v, stop_frac, yaw_act),
    }


def _score_zip(zp: Path) -> list[dict]:
    ch = int(zp.stem.split("_")[-1])
    rows = []
    try:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.endswith(".egomotion.parquet"):
                    continue
                clip_id = name.split("/")[-1].split(".")[0]
                try:
                    df = pd.read_parquet(io.BytesIO(z.read(name)))
                    s = scenario_stats_from_egomotion(df)
                except Exception:
                    continue
                rows.append({"clip_id": clip_id, "chunk": ch, **s})
    except zipfile.BadZipFile:
        print(f"[p0] bad zip skipped: {zp.name}", flush=True)
    return rows


def _score_all_cached(workers: int = 8) -> pd.DataFrame:
    zips = sorted((ROOT / "labels" / "egomotion").glob("egomotion.chunk_*.zip"))
    print(f"[p0] scoring {len(zips)} cached egomotion chunks "
          f"({workers} workers)...", flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(_score_zip, zips), 1):
            rows.extend(part)
            if i % 20 == 0:
                print(f"[p0]  scored {i}/{len(zips)} chunks, "
                      f"{len(rows)} clips", flush=True)
    scored = pd.DataFrame(rows)
    if scored.empty:
        raise SystemExit("[p0] nothing scored — inspect egomotion cache/schema")
    return scored


def _attach_catalog(scored: pd.DataFrame) -> pd.DataFrame:
    meta = _load_catalog().reset_index().rename(columns={"index": "clip_id"})
    meta["clip_id"] = meta["clip_id"].astype(str)
    scored["clip_id"] = scored["clip_id"].astype(str)
    scored = scored.merge(
        meta[["clip_id", "country", "hour_of_day", "platform_class"]],
        on="clip_id", how="left")
    scored = scored.dropna(subset=["country", "platform_class"])
    scored["tod"] = scored["hour_of_day"].apply(tod_bucket)
    return scored


# --------------------------------------------------------------------------- #
# Balanced stratified sampler.                                                #
# --------------------------------------------------------------------------- #
def _waterfill(avail: dict, total: int) -> dict:
    """Allocate `total` across keys AS EQUALLY AS POSSIBLE, bounded by each
    key's availability (classic water-filling). Sum(alloc) == min(total,
    sum avail)."""
    alloc = {k: 0 for k in avail}
    remaining = min(total, sum(avail.values()))
    while remaining > 0:
        active = [k for k in avail if alloc[k] < avail[k]]
        if not active:
            break
        share = max(1, remaining // len(active))
        for k in active:
            add = min(share, avail[k] - alloc[k], remaining)
            alloc[k] += add
            remaining -= add
            if remaining <= 0:
                break
    return alloc


def stratified_sample(scored: pd.DataFrame, target: int,
                      scenario_mix: dict, seed: int = 0) -> pd.DataFrame:
    """Balance over scenario x country x (tod x platform).

    Per scenario we take a fair share of `target`; within a scenario we
    water-fill across countries (equal where availability allows), and within a
    country we water-fill across the tod x platform cells before random
    sampling. A final top-up draws uniformly from a scenario's leftovers to hit
    the scenario target when fine cells run short.
    """
    rng = np.random.RandomState(seed)
    scen_target = {s: int(round(target * f)) for s, f in scenario_mix.items()}
    picks = []
    for scen, tgt in scen_target.items():
        sub = scored[scored["scenario"] == scen]
        if sub.empty or tgt <= 0:
            continue
        c_alloc = _waterfill(sub.groupby("country").size().to_dict(), tgt)
        chosen_ids: set[str] = set()
        for country, cn in c_alloc.items():
            if cn <= 0:
                continue
            csub = sub[sub["country"] == country]
            cell_avail = csub.groupby(["tod", "platform_class"]).size().to_dict()
            for (t, p), pn in _waterfill(cell_avail, cn).items():
                if pn <= 0:
                    continue
                cell = csub[(csub["tod"] == t) & (csub["platform_class"] == p)]
                take = cell.sample(n=min(pn, len(cell)), random_state=rng)
                chosen_ids.update(take["clip_id"])
        # top-up to the scenario target from remaining clips of this scenario
        short = tgt - len(chosen_ids)
        if short > 0:
            pool = sub[~sub["clip_id"].isin(chosen_ids)]
            if len(pool):
                extra = pool.sample(n=min(short, len(pool)), random_state=rng)
                chosen_ids.update(extra["clip_id"])
        picks.append(sub[sub["clip_id"].isin(chosen_ids)])
    out = pd.concat(picks).drop_duplicates("clip_id").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Download helpers (egomotion pre-fetch, stratified across countries).        #
# --------------------------------------------------------------------------- #
def _pick_download_chunks(n_chunks: int) -> list[int]:
    cat = _load_catalog().reset_index()
    ch_country = cat.groupby("chunk")["country"].agg(lambda s: s.value_counts().index[0])
    cached = {int(p.stem.split("_")[-1])
              for p in (ROOT / "labels" / "egomotion").glob("*.zip")}
    rng = np.random.RandomState(7)
    per_country = max(2, n_chunks // max(1, ch_country.nunique()))
    picked: list[int] = []
    for _country, grp in ch_country.reset_index().groupby("country"):
        avail = sorted(c for c in grp["chunk"].tolist() if c not in cached)
        if not avail:
            continue
        k = min(per_country, len(avail))
        picked.extend(int(c) for c in rng.choice(avail, size=k, replace=False))
    rng.shuffle(picked)
    return picked[:n_chunks]


def _download_more_egomotion(n_chunks: int) -> None:
    dl = _hf()
    for ch in _pick_download_chunks(n_chunks):
        try:
            _download_file(EGO_TMPL.format(chunk_id=ch), dl)
            print(f"[p0] egomotion chunk {ch} cached", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[p0] egomotion chunk {ch} failed: {e}", flush=True)


def _download_file(rel_path: str, dl) -> Path:
    """Windows: curl.exe (resumable, --ssl-no-revoke for the MITM proxy —
    hf_hub_download zombies on network drops here). Linux/pod: hf_hub_download."""
    import platform
    import subprocess
    if platform.system() != "Windows":
        return Path(dl(REPO, rel_path, repo_type="dataset", local_dir=str(ROOT)))
    zp = ROOT / rel_path
    zp.parent.mkdir(parents=True, exist_ok=True)
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel_path}"
    cmd = ["curl.exe", "-L", "-C", "-", "--ssl-no-revoke", "--retry", "10",
           "--retry-delay", "5", "-o", str(zp), url]
    if token:      # gated dataset; curl drops the header on the CDN redirect
        cmd += ["-H", f"Authorization: Bearer {token}"]
    subprocess.run(cmd, check=True)
    return zp


# --------------------------------------------------------------------------- #
# Reporting.                                                                  #
# --------------------------------------------------------------------------- #
def _speed_hist(sel: pd.DataFrame) -> dict:
    bins = [0, 2, 5, 8, 11, 14, 17, 20, 25, 30, 100]
    cut = pd.cut(sel["mean_speed"], bins)
    return {str(k): int(v) for k, v in cut.value_counts().sort_index().items()}


def _write_selection_and_report(sel: pd.DataFrame, scored: pd.DataFrame,
                                target: int) -> dict:
    out_dir = ROOT / "r0"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(out_dir / "phase0_selection.parquet")

    n = len(sel)
    hours = n * 20.0 / 3600.0
    scen_counts = sel["scenario"].value_counts().to_dict()
    scen_pct = {k: round(100 * v / n, 1) for k, v in scen_counts.items()}
    report = {
        "target": target,
        "selected": n,
        "total_hours": round(hours, 2),
        "highway_pct": scen_pct.get("highway", 0.0),
        "meets_highway_25pct": scen_pct.get("highway", 0.0) >= 25.0,
        "meets_15h": hours >= 15.0,
        "scenario_counts": scen_counts,
        "scenario_pct": scen_pct,
        "by_country": sel["country"].value_counts().to_dict(),
        "by_tod": sel["tod"].value_counts().to_dict(),
        "by_platform": sel["platform_class"].value_counts().to_dict(),
        "speed_hist_mean_mps": _speed_hist(sel),
        "mean_speed_selected": round(float(sel["mean_speed"].mean()), 2),
        "countries_covered": int(sel["country"].nunique()),
        "chunks_scored": int(scored["chunk"].nunique()),
        "clips_scored": int(len(scored)),
        "camera_chunks_needed": sorted(sel["chunk"].unique().tolist()),
    }
    (out_dir / "PHASE0_REPORT.json").write_text(json.dumps(report, indent=2))

    def _tbl(d):
        return "\n".join(f"| {k} | {v} |" for k, v in
                         sorted(d.items(), key=lambda kv: -kv[1]))

    md = f"""# PhysicalAI-AV Phase-0 selection

**{n} clips = {hours:.2f} h** (20 s each) | target {target}
scored {report['clips_scored']} clips over {report['chunks_scored']} chunks.

## Scenario (target: highway >= 25 %, roughly thirds)
| scenario | clips | % |
|---|---|---|
""" + "\n".join(
        f"| {s} | {scen_counts.get(s, 0)} | {scen_pct.get(s, 0)} |"
        for s in ("highway", "suburban", "urban")) + f"""

Highway >= 25 %: **{report['meets_highway_25pct']}** ({report['highway_pct']} %) |
Total >= 15 h: **{report['meets_15h']}** ({hours:.2f} h)

## Platform (rig)
| platform | clips |
|---|---|
{_tbl(report['by_platform'])}

## Time of day
| tod | clips |
|---|---|
{_tbl(report['by_tod'])}

## Country (top strata; {report['countries_covered']} covered)
| country | clips |
|---|---|
{_tbl(report['by_country'])}

## Mean-speed histogram (m/s)
| band | clips |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in
                report['speed_hist_mean_mps'].items()) + "\n"
    (out_dir / "PHASE0_REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[p0] selection -> {out_dir/'phase0_selection.parquet'}")
    print(f"[p0] report     -> {out_dir/'PHASE0_REPORT.md'}")
    return report


def stage_select(target: int, highway_frac: float, urban_frac: float,
                 download: int, seed: int) -> None:
    if download:
        _download_more_egomotion(download)
    scored = _attach_catalog(_score_all_cached())
    scored = scored[scored["mean_speed"] >= PARKED_SPEED]         # drop parked
    print(f"[p0] {len(scored)} driving clips scored; scenario mix in corpus: "
          f"{scored['scenario'].value_counts(normalize=True).round(3).to_dict()}",
          flush=True)
    mix = {"highway": highway_frac, "urban": urban_frac,
           "suburban": round(1.0 - highway_frac - urban_frac, 4)}
    sel = stratified_sample(scored, target, mix, seed=seed)
    _write_selection_and_report(sel, scored, target)


# --------------------------------------------------------------------------- #
# Camera + calibration fetch for the selected clips.                          #
# --------------------------------------------------------------------------- #
def stage_fetch_camera(max_chunks: int, selection: str) -> None:
    dl = _hf()
    sel = pd.read_parquet(ROOT / "r0" / selection)
    out = ROOT / "r0" / "camera_front_wide"
    out.mkdir(parents=True, exist_ok=True)
    by_chunk: dict[int, set[str]] = {}
    for _, r in sel.iterrows():
        by_chunk.setdefault(int(r["chunk"]), set()).add(str(r["clip_id"]))
    have = {p.name.split(".")[0] for p in out.rglob("*.mp4")}
    for ch in sorted(by_chunk)[:max_chunks]:
        # per-clip calibration chunks (intrinsics + extrinsics) drive the
        # per-clip cy crop; small parquet, keep it.
        for kind in CALIB_KINDS:
            rel = CALIB_TMPL.format(kind=kind, chunk_id=ch)
            if not (ROOT / rel).exists():
                try:
                    _download_file(rel, dl)
                except Exception as e:  # noqa: BLE001
                    print(f"[p0] calib {kind} chunk {ch} failed: {e}", flush=True)
        missing = by_chunk[ch] - have
        if not missing:
            print(f"[p0] chunk {ch}: camera complete, skipping", flush=True)
            continue
        zp = _download_file(CAM_TMPL.format(chunk_id=ch), dl)
        n = 0
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                clip_id = name.split("/")[-1].split(".")[0]
                if clip_id in by_chunk[ch]:
                    z.extract(name, out)
                    n += 1
        zp.unlink()                       # 2 GB each — never keep the zips
        print(f"[p0] chunk {ch}: extracted {n} files, zip deleted", flush=True)
    print("[p0] camera + calibration fetch done ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "fetch-camera"])
    ap.add_argument("--target", type=int, default=2800,
                    help="clips to select (>=2700 for >=15 h)")
    ap.add_argument("--highway-frac", type=float, default=0.33)
    ap.add_argument("--urban-frac", type=float, default=0.33)
    ap.add_argument("--download", type=int, default=0,
                    help="egomotion chunks to fetch (country-stratified) "
                         "before scoring; 0 = score cached only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chunks", type=int, default=999)
    ap.add_argument("--selection", default="phase0_selection.parquet")
    a = ap.parse_args()
    if a.stage == "select":
        stage_select(a.target, a.highway_frac, a.urban_frac, a.download, a.seed)
    else:
        stage_fetch_camera(a.max_chunks, a.selection)
