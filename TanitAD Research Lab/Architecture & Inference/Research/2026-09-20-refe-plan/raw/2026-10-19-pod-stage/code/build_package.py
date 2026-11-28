"""Stage the 2026-10-19 pod-stage package: MANIFEST.md (counts; the sha256 lists live on D:), out/<repo path>/..., out/paths.json ([repo path, abs source, "raw"]).  Re-runnable, tolerant of a partial state.  No git.

    python build_package.py
"""
import glob
import json
import os
import shutil
import time

ROOT = "C:/Users/Admin/par18/podstage"
BASE = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-pod-stage"
OUT = f"{ROOT}/out/{BASE}"
WORK = f"{ROOT}/work"
SHA_DIR = "D:/Projects/TanitAD-artifacts/pod_stage_2026-10-19"
FRAMES = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/frames_full"
PLAN_FULL = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/plan_full"
PLAN_PART = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/plan_partial_5776"
DB_DEST = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"


def jl(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    fs, vf = jl(f"{WORK}/frames_stat.json"), jl(f"{WORK}/verify_frames.json")
    ds, vd = jl(f"{WORK}/db_stat.json"), jl(f"{WORK}/verify_dbs.json")
    cov = jl(f"{PLAN_FULL}/coverage.json")
    log = open(f"{ROOT}/logs/podstage.log", encoding="utf-8", errors="replace").read() if os.path.exists(f"{ROOT}/logs/podstage.log") else ""
    plog = [ln for ln in log.splitlines() if "ZZ" in ln]
    n_wait = sum(1 for ln in log.splitlines() if "P6_GUARD" in ln and "waiting" in ln)
    batches_f = sum(1 for ln in log.splitlines() if " batch " in ln and "navtrain_current" in ln)
    batches_d = sum(1 for ln in log.splitlines() if " batch " in ln and "navtrain_dbs" in ln)
    free = shutil.disk_usage("D:/").free / 2 ** 30
    L = ["# MANIFEST: pod reads under P6's overlap rule (2026-10-19)", "",
         "Two pod reads, PI-approved 2026-10-19 under the P6 OVERLAP RULE: every pod-side command at `nice -n 19 ionice -c3`, ONE stream, batches of <= 120 MB, a `P6_GUARD` line in P6's chain log at the start of each pod command pauses (it is checked in the SAME remote shell as the read), no GPU process, no pod-side python, no `git` on the pod, never `pkill -f`/`pgrep -f`, P6's processes and files untouched. Pod-side work per batch: `tail` (guard), `tar -c`, `sha256sum` (the PI's approval for this task). The pod's address, port, id and token files appear in NO staged file (the ssh alias lives in a local file outside the repo; every logged string is redacted).", ""]
    L += ["## READ 1: SH5 FULL frames (priority 1)", ""]
    if fs and vf:
        tot = vf.get("n_listed_total", vf["n_listed"])
        part = vf.get("partial_disk_guard", False)
        full_ok = vf["n_failed"] == 0 and vf["n_verified"] == tot
        L += [f"- **Frames staged: {vf['n_verified']}/{tot} jpg, {vf['bytes_verified'] / 1e9:.3f} GB ({vf['bytes_verified'] / 2**30:.3f} GiB); every staged file verified: {'YES' if vf['n_failed'] == 0 else 'NO, ' + str(vf['n_failed']) + ' failed'}; COMPLETE: {'YES' if full_ok else 'NO'}** (MEASURED, `raw/verify_frames.json`).",
              (f"- ⛔ **STOPPED BY THE DISK GUARD, NOT BY AN ERROR.** D: (exFAT, 1 MiB clusters: MEASURED 200 files of 200 KiB = 200 MiB) fell from 35.3 GiB free (15:34) to 20.2 GiB while this copy added only ~{vf['n_verified'] / 1024:.1f} GiB allocated: other writers on D: took ~9 GiB meanwhile. The copier refuses a batch that would leave < 20 GiB free (the S0 guard, kept). The remaining {tot - vf['n_verified']} files are {(tot - vf['n_verified']) * 0.2 / 1000:.2f} GB of data but ~{(tot - vf['n_verified']) / 1024:.1f} GiB ALLOCATED, which would leave ~{20.2 - (tot - vf['n_verified']) / 1024:.1f} GiB free. Resume = one command after space is freed (or the guard is lowered by the Master Mind): `PODSTAGE_MIN_FREE_GB=<n> python code/podstage.py frames`." if part else "- Copy complete."),
              f"- Source list: `needed_frames.json` of the SH5 plan = 1,966 tokens x 4 cameras; {fs['n_listed']} distinct names; on the pod {fs['n_found_on_pod']} found, {fs['n_missing_on_pod']} missing; {fs['n_shards']} pod shards; pod-side `stat` total {fs['bytes_on_pod']} B = {fs['bytes_on_pod'] / 1e9:.3f} GB (READY.md estimated ~1.5 GB).",
              f"- Checks on EVERY staged file: exists; size = pod `stat` = manifest; non-zero; JPEG magic `FF D8 FF`; sha256 of the bytes on disk = sha256 of the received stream = pod-side `sha256sum` (computed at nice/ionice). {vf.get('jpeg_ends_with_FFD9')} of {vf['n_verified']} end with `FF D9`. The checker was mutation-tested in scratch (flipped byte, file truncated by 1 byte, zero-size file, broken magic: all 4 caught; clean control 6/6, 0 failed).",
              f"- Where: `{FRAMES}` (layout `<log>/<CAM>/<hash>.jpg`, exactly the names the DBs give). sha256 list (lives on D:, not in the repo): `{vf['sha_list']}`."]
    else:
        L += ["- NOT FINISHED at the time this file was built (see the markers below)."]
    L += ["", "### SH5 plan re-run for FULL (CPU, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`; READY.md Q1 verbatim)", ""]
    covp = jl(f"{PLAN_PART}/coverage.json")
    if cov:
        L += [f"- Command: `sh5_ctx_build.py plan --inputs DD/data/sh5_inputs.npz --out-dir DD/plan_full --images-root DD/frames_full`  (DD = `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19`).",
              f"- **FULL coverage: tokens {cov['n_tokens_with_frames']}/{cov['n_tokens']}, sets {cov['n_sets_kept']}/{cov['n_sets']}, logs {cov['n_logs_with_any']}** (before: 816/2,782 tokens, 1,444/4,941 sets, 38/41 logs). Sets kept by fold {cov['sets_kept_by_fold']} of {cov['sets_by_fold']}.",
              f"- Output: `{PLAN_FULL}`; `coverage.json` also staged as `raw/plan_full_coverage.json`. FULL needs its own out-dirs (`ctx_full`, `fits_full`): the ctx rows follow the plan's token order. **No GPU job was launched** (Master Mind's)."]
    elif covp:
        L += [f"- **`plan_full` NOT produced: the frames are incomplete (disk guard).** The same command was re-run on the {vf['n_verified'] if vf else '?'} staged frames into `{PLAN_PART}` (CPU, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`, ~24 s), so it is a PARTIAL plan, not the FULL variant.",
              f"- **PARTIAL coverage: tokens {covp['n_tokens_with_frames']}/{covp['n_tokens']}, sets {covp['n_sets_kept']}/{covp['n_sets']}, logs {covp['n_logs_with_any']}** (before any staging: 816/2,782 tokens, 1,444/4,941 sets, 38/41 logs). Sets kept by fold {covp['sets_kept_by_fold']} of {covp['sets_by_fold']}. The staged 5,776 frames are exactly 1,444 complete tokens (4 cameras each); the other 522 tokens need the 2,088 frames that did not fit.",
              f"- Output: `{PLAN_PART}` (`plan.json`, `coverage.json`, ...); `coverage.json` also staged as `raw/plan_partial_coverage.json`. Do NOT use the partial plan as the FULL variant (the prereg's FULL is all 2,782 tokens)."]
    else:
        L += ["- NOT RUN yet."]
    L += ["", "## READ 2: nuPlan DBs of the 76 f10 logs PRIVGEOM-SAFE could not read (priority 2)", ""]
    if ds:
        L += [f"- The 117 f10 logs = the `*.npz` of `C:/Users/Admin/srttc_s0/work/sets` (PRIVGEOM's `SETS_DIR`); 41 have a DB in the local folder `{DB_DEST}` (PRIVGEOM's `NS.index_dbs()` = `navtrain_scenarios.DEFAULT_DB_ROOT`, on D:); the other **{ds['n_wanted']}** are needed. All {ds['n_found']} exist on the pod (`navtrain_dbs`), {len(ds['missing'])} missing.",
              f"- **MEASURED TOTAL SIZE (pod-side `stat`, nice/ionice): {ds['total_bytes']} B = {ds['total_GB']:.3f} GB = {ds['total_GiB']:.3f} GiB** (min {ds['min_MiB']:.1f} MiB, max {ds['max_MiB']:.0f} MiB). <= 20 GB, so the transfer was allowed."]
        if vd:
            L += [f"- **DBs transferred: {vd['n_verified']}/{vd['n_wanted']} verified, {vd['bytes_verified'] / 1e9:.3f} GB.** Checks: as for frames (magic = `SQLite format 3`) + a read-only sqlite open with `lidar_pc`/`image`/`camera` present. {vd['n_failed']} failed. sha256 list (on D:): `{vd['sha_list']}`.",
                  f"- Where: `{DB_DEST}` (the flat folder PRIVGEOM read; on D:, so no D: fallback was needed). Copy order = ascending size."]
        else:
            L += [f"- ⛔ **DBs transferred: 0/{ds['n_wanted']} (NOT transferred, deliberately).** The size passes the 20 GB rule ({ds['total_GB']:.2f} GB), but the staging drive does not have the room: D: has {free:.1f} GiB free now, 1 MiB clusters make the {ds['total_GiB']:.2f} GiB occupy ~{ds['total_GiB'] + 0.05:.2f} GiB, so the transfer would leave ~{free - ds['total_GiB'] - 0.05:.1f} GiB, under the 20 GiB floor the S0 stager keeps (and other jobs are writing to D: right now: ~9 GiB vanished in ten minutes during the frames copy). Priority 1 (frames) is also still incomplete for the same reason, so nothing was started for priority 2.",
                  f"- What would unblock it: free ~{max(0.0, 20 + ds['total_GiB'] + 0.05 - free):.0f} GiB on D: (DBs only) or the Master Mind lowers the guard (`PODSTAGE_MIN_FREE_GB`) or names another drive; then `python code/podstage.py dbs` (resumable, ascending size, verifies by sha256 pod = stream = disk and by a read-only sqlite open). The pod-side list is already made (`work/copy_list_dbs.tsv`)."]
    else:
        L += ["- NOT MEASURED yet."]
    L += ["", f"## Run facts", "", f"- Batches: {batches_f} frame batches + {batches_d} DB batches logged; P6_GUARD waits: {n_wait}. D: free after the run: {free:.1f} GiB (the copier refuses a batch that would leave < 20 GiB).", "- Markers in the log (`logs/podstage.log`):", ""]
    L += ["```"] + plog[-25:] + ["```", "", "## Deliverable manifest", "", "| artifact | where |", "|---|---|"]
    L += [f"| staged frames ({vf['n_verified'] if vf else '?'} jpg) | `{FRAMES}` (D: ONLY: re-fetchable from the pod, which is training P6 and is not storage) |",
          f"| sha256 lists (frames / DBs) | `{SHA_DIR}` (D: only; counts above) |", f"| SH5 FULL plan | `{PLAN_FULL}` (D: only; coverage.json also in the repo) |",
          f"| 76 nuPlan DBs | `{DB_DEST}` (D: only; re-fetchable from the pod) |",
          f"| scripts: `podcopy2.py`, `podstage.py`, `build_package.py`, `launch.ps1` | `repo:{BASE}/code/` (staged by the lander from `out/paths.json`); also `{ROOT}/code/` |",
          f"| `MANIFEST.md`, `HANDOFF.md`, `raw/*.json`, `raw/podstage.log` | `repo:{BASE}/`; also `{ROOT}/` |", "", "Never staged for the repo: the frames, the DBs, the sha lists, anything with the pod's address.", ""]
    os.makedirs(OUT, exist_ok=True)
    txt = "\n".join(L)
    open(f"{ROOT}/MANIFEST.md", "w", encoding="utf-8", newline="\n").write(txt)
    srcs = [(f"{ROOT}/MANIFEST.md", "MANIFEST.md"), (f"{ROOT}/HANDOFF.md", "HANDOFF.md"), (f"{ROOT}/logs/podstage.log", "raw/podstage.log")]
    for f in ("podcopy2.py", "podstage.py", "build_package.py", "launch.ps1"):
        srcs.append((f"{ROOT}/code/{f}", f"code/{f}"))
    for f, d in (("frames_stat.json", "raw"), ("verify_frames.json", "raw"), ("db_stat.json", "raw"), ("verify_dbs.json", "raw"), ("db_logs.json", "raw")):
        srcs.append((f"{WORK}/{f}", f"{d}/{f}"))
    srcs.append((f"{PLAN_FULL}/coverage.json", "raw/plan_full_coverage.json"))
    srcs.append((f"{PLAN_PART}/coverage.json", "raw/plan_partial_coverage.json"))
    srcs.append((f"{ROOT}/logs/plan_partial.log", "raw/plan_partial.log"))
    paths = []
    for s, rel in srcs:
        if not os.path.exists(s):
            continue
        d = f"{OUT}/{rel}"
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        paths.append([f"{BASE}/{rel}", d, "raw"])
    json.dump(paths, open(f"{ROOT}/out/paths.json", "w", encoding="utf-8"), indent=1)
    print(f"staged {len(paths)} files; paths.json written; built {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
