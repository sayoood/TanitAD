#!/usr/bin/env python3
"""build_bundle.py -- stage the SECOND TanitScenarioFlyWheel scenario bundle (same layout as tsf_nuplan_cruise_bundle/).

    python build_bundle.py prepare   # candidates -> wanted image members (reads the OpenScene meta pkls)
    python build_bundle.py extract   # stream the 32 sensor tgz archives (resumable; early stop when everything is found)
    python build_bundle.py sky       # CPU sky pre-screen of the top third of CAM_F0, first 6 s (13 key frames at 2 Hz)
    python build_bundle.py assemble  # copy the per-log pkls, calib rows, write MANIFEST.json (sha256 of every file)

The bundle goes to F: (BUNDLE), small code/results to E: (ART).  nuPlan/OpenScene data is non-commercial research data: it never goes into the repo tree.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107"
RUN = ART + "/run2"
BUNDLE = "F:/TanitAD_tsf_bundle2_20261107/bundle"
WORK = "F:/TanitAD_tsf_bundle2_20261107/work"
META_DIR = "F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
BLOB_DIR = "F:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera"
MEMBER_PREFIX = "openscene-v1.1/sensor_blobs/test/"
BANK_DIR = "E:/Projects/TanitAD/data/refe_navtest/frames"
DB_DIR = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
PK = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
CAMS8 = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]   # the cruise bundle's order
CALIB_CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]
N_SKY = 13           # key frames at 2 Hz over the first 6 s, t0 included
SLOTS = ["cruise", "steady_lead", "lead_braking", "stop_go", "merge_cutin"]


def sha256_file(p, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def candidates():
    """token -> {slots: [(reading, slot, rank)], log}"""
    out = {}
    for reading, fn in (("literal", "SUITE_CANDIDATES.json"), ("alt", "SUITE_CANDIDATES_ALT.json")):
        d = json.load(open(f"{RUN}/{fn}", encoding="utf-8"))
        for slot in SLOTS:
            for e in d["slots"][slot]["ordered_one_per_log"][:5]:
                t = out.setdefault(e["token"], dict(log=e["log"], roles=[]))
                t["roles"].append(dict(reading=reading, slot=slot, rank=e["rank"]))
    return out


def prepare():
    os.makedirs(WORK, exist_ok=True)
    cand = candidates()
    by_log = {}
    for t, v in cand.items():
        by_log.setdefault(v["log"], []).append(t)
    wanted, tokinfo, missing_in_meta = {}, {}, []
    for log, toks in sorted(by_log.items()):
        with open(f"{META_DIR}/{log}.pkl", "rb") as fh:
            frames = pickle.loads(fh.read())          # ONE big read: pickle.load() issues many small reads, which crawl when the page cache is under pressure
        pos = {f["token"]: j for j, f in enumerate(frames)}
        for t in sorted(toks):
            j = pos.get(t)
            if j is None or j + N_SKY - 1 >= len(frames):
                missing_in_meta.append(t)
                continue
            tokinfo[t] = dict(log=log, list_idx=j, t0_timestamp_us=int(frames[j]["timestamp"]), roles=cand[t]["roles"], cams={}, sky_members=[])
            for cam in CAMS8:
                m = MEMBER_PREFIX + frames[j]["cams"][cam]["data_path"]
                tokinfo[t]["cams"][cam] = frames[j]["cams"][cam]["data_path"]
                wanted.setdefault(m, []).append(dict(dest=f"frames/{t}/{cam}.jpg", token=t, cam=cam, kind="t0"))
            # coordinator addition (sim session): the rollout window starts at t0 - 2 s, so chunk 0 needs the 8 views of the key frame 4 samples (2 Hz) before the token
            tinfo = tokinfo[t]
            tinfo["cams_tm2s"], tinfo["tm2s_list_idx"] = {}, j - 4
            if j - 4 < 0:
                missing_in_meta.append(t + "@tm2s")
            else:
                tinfo["tm2s_timestamp_us"] = int(frames[j - 4]["timestamp"])
                tinfo["tm2s_dt_s"] = round((int(frames[j]["timestamp"]) - int(frames[j - 4]["timestamp"])) / 1e6, 4)
                for cam in CAMS8:
                    m = MEMBER_PREFIX + frames[j - 4]["cams"][cam]["data_path"]
                    tinfo["cams_tm2s"][cam] = frames[j - 4]["cams"][cam]["data_path"]
                    wanted.setdefault(m, []).append(dict(dest=f"frames/{t}/tm2s/{cam}.jpg", token=t, cam=cam, kind="tm2s"))
            for k in range(1, N_SKY):
                m = MEMBER_PREFIX + frames[j + k]["cams"]["CAM_F0"]["data_path"]
                tokinfo[t]["sky_members"].append(frames[j + k]["cams"]["CAM_F0"]["data_path"])
                wanted.setdefault(m, []).append(dict(dest=f"sky_prescreen_frames/{t}/CAM_F0_m{k:02d}.jpg", token=t, cam="CAM_F0", kind="sky", m=k))
    json.dump(dict(wanted=wanted, tokinfo=tokinfo, missing_in_meta=missing_in_meta), open(f"{WORK}/wanted.json", "w"), indent=0)
    print("tokens", len(tokinfo), "logs", len(by_log), "wanted members", len(wanted), "missing_in_meta", missing_in_meta, "tm2s dt range", (min(v["tm2s_dt_s"] for v in tokinfo.values() if "tm2s_dt_s" in v), max(v["tm2s_dt_s"] for v in tokinfo.values() if "tm2s_dt_s" in v)), flush=True)


# ------------------------------------------------------------------------------------------------------------------------------------------
def scan_archive(args):
    path, wanted = args
    name = os.path.basename(path)
    st_path = f"{WORK}/status_{name}.json"
    rem = {m for m, ds in wanted.items() if not all(os.path.exists(f"{BUNDLE}/{d['dest']}") for d in ds)}
    rep = dict(archive=name, found={}, scanned_mb=0.0, members_seen=0, stopped="", secs=0.0)
    if not rem:
        rep["stopped"] = "nothing_remaining_at_start"
        return rep
    t0 = time.time()
    size = os.path.getsize(path)
    with open(path, "rb", buffering=1 << 25) as raw:     # 32 MB read buffer: tarfile reads 10 KB at a time, which crawls (~2 MB/s) on this drive without it
        tf = tarfile.open(fileobj=raw, mode="r|gz")
        n = 0
        for ti in tf:
            n += 1
            if ti.name in rem:
                data = tf.extractfile(ti).read()
                sha = hashlib.sha256(data).hexdigest()
                for d in wanted[ti.name]:
                    dst = f"{BUNDLE}/{d['dest']}"
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst + ".tmp", "wb") as f:
                        f.write(data)
                    os.replace(dst + ".tmp", dst)
                rep["found"][ti.name] = dict(sha256=sha, bytes=len(data), tgz=name)
                rem.discard(ti.name)
            if n % 2000 == 0:
                rem = {m for m in rem if not all(os.path.exists(f"{BUNDLE}/{d['dest']}") for d in wanted[m])}
                json.dump(dict(archive=name, members_seen=n, pos_gb=round(raw.tell() / 1e9, 2), size_gb=round(size / 1e9, 2), found=len(rep["found"]), remaining_here=len(rem),
                               secs=round(time.time() - t0)), open(st_path, "w"))
                if not rem:
                    rep["stopped"] = "all_wanted_found_globally"
                    break
        else:
            rep["stopped"] = "end_of_archive"
        rep["scanned_mb"] = round(raw.tell() / 1e6, 1)
        rep["members_seen"] = n
    rep["secs"] = round(time.time() - t0, 1)
    json.dump(dict(rep, done=True, found=len(rep["found"])), open(st_path, "w"))
    return rep


def extract(workers=6):
    d = json.load(open(f"{WORK}/wanted.json"))
    wanted = d["wanted"]
    archives = sorted(glob.glob(BLOB_DIR + "/*.tgz"), key=lambda p: int(re.search(r"_(\d+)\.tgz", p).group(1)))
    print(len(archives), "archives;", len(wanted), "wanted members", flush=True)
    t0 = time.time()
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rep in ex.map(scan_archive, [(p, wanted) for p in archives]):
            reports.append(rep)
            print(f"[{time.strftime('%H:%M:%S')}] {rep['archive']}: found {len(rep['found'])} scanned {rep['scanned_mb']} MB in {rep['secs']} s  stopped: {rep['stopped']}", flush=True)
    found = {}
    for r in reports:
        found.update(r["found"])
    missing = sorted(set(wanted) - set(found))
    # members that were already on disk from an earlier (resumed) run have no record in this run: hash them from disk
    for m in missing:
        ds = wanted[m]
        if all(os.path.exists(f"{BUNDLE}/{x['dest']}") for x in ds):
            p = f"{BUNDLE}/{ds[0]['dest']}"
            found[m] = dict(sha256=sha256_file(p), bytes=os.path.getsize(p), tgz="(earlier run)")
    missing = sorted(set(wanted) - set(found))
    old = json.load(open(f"{WORK}/extract_report.json")) if os.path.exists(f"{WORK}/extract_report.json") else {}
    old_found = old.get("found", {})
    for m, v in old_found.items():
        found.setdefault(m, v)
    missing = sorted(set(wanted) - set(found))
    json.dump(dict(found=found, missing=missing, secs=round(time.time() - t0, 1), archives=[(r["archive"], r["scanned_mb"], r["stopped"], r["secs"]) for r in reports]),
              open(f"{WORK}/extract_report.json", "w"), indent=0)
    print("extract done: found", len(found), "missing", len(missing), "secs", round(time.time() - t0), flush=True)


# ------------------------------------------------------------------------------------------------------------------------------------------
def sky():
    sys.path.insert(0, HERE)
    import sky_prescreen as sp
    d = json.load(open(f"{WORK}/wanted.json"))
    tokinfo = d["tokinfo"]
    out = dict(
        method=sp.__doc__.strip().split("\n\n")[0] if sp.__doc__ else "",
        method_detail=dict(resize=list(sp.SMALL), top_rows_of_270=sp.TOP_ROWS, smooth_std_max=sp.STD_MAX, blue_hue_cv2=list(sp.BLUE_H), blue_s_min=sp.BLUE_S_MIN, blue_v_min=sp.BLUE_V_MIN,
                           grey_s_max=sp.GREY_S_MAX, grey_v_min=sp.GREY_V_MIN, frac_min=sp.FRAC_MIN, share_min=sp.SHARE_MIN),
        evidence_class="ESTIMATED proxy (colour + texture, no learned model); UNVERIFIED that it predicts the lifter's failure; the sim's QC render decides. No candidate is skipped here.",
        frames="13 key frames of CAM_F0 at 2 Hz: t0 and t0+0.5 .. t0+6.0 s (the OpenScene key frames; the 10 Hz sweeps are not in the sensor blobs we hold)",
        rule="scorecard 3.1 QC bullet 1: sky fraction of the top third of CAM_F0 >= 0.5 in >= 90 % of the first 6 s of frames", tokens={})
    for t, ti in sorted(tokinfo.items()):
        paths = [f"{BUNDLE}/frames/{t}/CAM_F0.jpg"] + [f"{BUNDLE}/sky_prescreen_frames/{t}/CAM_F0_m{k:02d}.jpg" for k in range(1, N_SKY)]
        fr = [sp.sky_fraction(p) if os.path.exists(p) else None for p in paths]
        v = sp.verdict(fr)
        out["tokens"][t] = dict(log=ti["log"], roles=ti["roles"], fractions=[None if f is None else round(f, 3) for f in fr], **v)
    json.dump(out, open(f"{ART}/SKY_PRESCREEN.json", "w"), indent=1)
    n_pass = sum(1 for v in out["tokens"].values() if v["pass_proxy"])
    print("sky: tokens", len(out["tokens"]), "pass_proxy", n_pass)


# ------------------------------------------------------------------------------------------------------------------------------------------
def assemble():
    d = json.load(open(f"{WORK}/wanted.json"))
    tokinfo = d["tokinfo"]
    rep = json.load(open(f"{WORK}/extract_report.json"))
    found, missing_members = rep["found"], set(rep["missing"])
    os.makedirs(f"{BUNDLE}/openscene_meta_test", exist_ok=True)
    logs = sorted({ti["log"] for ti in tokinfo.values()})
    for lg in logs:
        dst = f"{BUNDLE}/openscene_meta_test/{lg}.pkl"
        if not os.path.exists(dst):
            shutil.copyfile(f"{META_DIR}/{lg}.pkl", dst)
    # calib rows, exactly the cruise bundle's source (refe/calib_table.py read_db on the DB camera table)
    sys.path.insert(0, PK + "/refe")
    import calib_table
    ref = json.load(open("E:/Projects/TanitAD-artifacts/tsf_nuplan_cruise_bundle/calib_rows_51logs.json"))
    calib = dict(meta=dict(ref["meta"]), logs={})
    calib["meta"]["source"] = ref["meta"]["source"]
    for lg in logs:
        calib["logs"][lg] = {k: [float(x) for x in v] for k, v in calib_table.read_db(f"{DB_DIR}/{lg}.db", CALIB_CAMS).items()}
    # self-check of the calib reader against the cruise bundle's file for the logs they share
    shared = [lg for lg in logs if lg in ref["logs"]]
    calib_selfcheck = dict(n_shared_logs=len(shared), all_equal=bool(all(calib["logs"][lg] == ref["logs"][lg] for lg in shared)))
    cpath = f"{BUNDLE}/calib_rows_{len(logs)}logs.json"
    json.dump(calib, open(cpath, "w"), indent=0)
    # token -> cam -> member (cruise bundle's `token_cams`), image table
    token_cams, images, wanted_n, found_n, missing_images = {}, {}, 0, 0, []
    for t, ti in sorted(tokinfo.items()):
        token_cams[t] = ti["cams"]
        for cam in CAMS8:
            wanted_n += 1
            m = MEMBER_PREFIX + ti["cams"][cam]
            p = f"{BUNDLE}/frames/{t}/{cam}.jpg"
            if m in found and os.path.exists(p):
                found_n += 1
                images[ti["cams"][cam]] = dict(token=t, cam=cam, member=m, tgz=found[m]["tgz"], sha256=found[m]["sha256"], bytes=found[m]["bytes"])
            else:
                missing_images.append(ti["cams"][cam])
    tm2s_images, tm2s_wanted, tm2s_found, tm2s_missing, token_cams_tm2s = {}, 0, 0, [], {}
    for t, ti in sorted(tokinfo.items()):
        token_cams_tm2s[t] = ti.get("cams_tm2s", {})
        for cam in CAMS8:
            tm2s_wanted += 1
            dp = ti.get("cams_tm2s", {}).get(cam)
            p = f"{BUNDLE}/frames/{t}/tm2s/{cam}.jpg"
            m = MEMBER_PREFIX + dp if dp else None
            if dp and m in found and os.path.exists(p):
                tm2s_found += 1
                tm2s_images[dp] = dict(token=t, cam=cam, member=m, tgz=found[m]["tgz"], sha256=found[m]["sha256"], bytes=found[m]["bytes"])
            else:
                tm2s_missing.append(dp or f"{t}/tm2s/{cam}: no frame j-4 in the meta pkl")
    sky_missing = []
    for t, ti in tokinfo.items():
        for k, dp in enumerate(ti["sky_members"], 1):
            if not os.path.exists(f"{BUNDLE}/sky_prescreen_frames/{t}/CAM_F0_m{k:02d}.jpg"):
                sky_missing.append(dp)
    # bank cross-check: the REFe frame bank's four cameras must be byte-identical to the tgz copies
    bank = dict(checked=0, equal=0, not_in_bank=0, differ=[])
    zc = {}
    for t, ti in sorted(tokinfo.items()):
        for cam in CALIB_CAMS:
            zp = f"{BANK_DIR}/{ti['log']}_{cam}.zip"
            p = f"{BUNDLE}/frames/{t}/{cam}.jpg"
            if not os.path.exists(zp) or not os.path.exists(p):
                bank["not_in_bank"] += 1
                continue
            z = zc.setdefault(zp, zipfile.ZipFile(zp))
            nm = os.path.basename(ti["cams"][cam])
            if nm not in z.namelist():
                bank["not_in_bank"] += 1
                continue
            bank["checked"] += 1
            if hashlib.sha256(z.read(nm)).hexdigest() == sha256_file(p):
                bank["equal"] += 1
            else:
                bank["differ"].append([t, cam])
    # A0 model inputs: the forward-hook export (REFe u400, CPU) if it is complete, else the capture-only export (pick null)
    a0_info = dict(file=None, status="MISSING")
    for mode, src in (("forward-cpu", "F:/TanitAD_tsf_bundle2_20261107/a0_forward/a0_model_inputs.jsonl"), ("capture-only", "F:/TanitAD_tsf_bundle2_20261107/a0_capture/a0_model_inputs.jsonl")):
        if os.path.exists(src):
            lines = [l for l in open(src, encoding="utf-8") if l.strip()]
            toks = {json.loads(l)["token"] for l in lines}
            if toks == set(tokinfo):
                shutil.copyfile(src, f"{BUNDLE}/a0_model_inputs.jsonl")
                picks = [json.loads(l)["pick"] for l in lines]
                a0_info = dict(file="a0_model_inputs.jsonl", mode=mode, n_lines=len(lines), n_pick_null=sum(p is None for p in picks), keys=["token", "log", "ego7_as_fed", "goal_as_fed", "pick", "device", "forward"],
                               producer="code/refe_a0_inputs_b2.py (lean copy of refe_ego_probe_20261106/refe_a0_inputs.py hook; reproduces the cruise bundle's ego7/goal bit-exactly on 5/5 tokens)",
                               note="pick from a CPU fp32 forward; the cruise bundle's picks came from the GPU run, so a pick can differ in rare near-tie cases (see A0_CPU_VALIDATION.json)" if mode == "forward-cpu" else "pick is null: run the GPU command in HANDOFF.md")
                break
    # files table (sha256 of every file in the bundle)
    files = {}
    for root, _, fs in os.walk(BUNDLE):
        for f in fs:
            if f.endswith(".tmp") or f == "MANIFEST.json":
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, BUNDLE).replace("\\", "/")
            files[rel] = dict(sha256=sha256_file(p), bytes=os.path.getsize(p))
    roles = {t: ti["roles"] for t, ti in tokinfo.items()}
    primary = {s: [t for t, ti in tokinfo.items() if any(r["reading"] == "literal" and r["slot"] == s and r["rank"] <= 5 for r in ti["roles"])] for s in SLOTS}
    man = dict(
        purpose="TanitScenarioFlyWheel nuPlan-anchored closed loop, bundle 2: the closed-loop BEHAVIOUR-SUITE candidate tokens selected by the frozen rule "
                "(BEHAVIOUR_SCORECARD.md section 3.1, select_suite_tokens.py)",
        licence="nuPlan/OpenScene non-commercial research data: keep private (never a public repo)",
        scorecard_sha256="fac6c7f150eb026de0708c6f6557df52481216f76b35f3c8c8b8045bcb8ec0c5",
        n_tokens=len(tokinfo), n_logs=len(logs), n_images_wanted=wanted_n, n_images_found=found_n, missing_images=missing_images,
        missing_tokens_in_meta=d["missing_in_meta"],
        tm2s_frames=dict(layout="frames/<token>/tm2s/<CAM>.jpg = the 8 views of the key frame 4 samples (2 Hz) before the token, t0 - 2.0 s (coordinator request for the sim's chunk 0)",
                         dt_to_t0_s={t: ti.get("tm2s_dt_s") for t, ti in sorted(tokinfo.items())}, n_images_wanted=tm2s_wanted, n_images_found=tm2s_found, missing_images=tm2s_missing,
                         token_cams=token_cams_tm2s, images=tm2s_images),
        sky_prescreen_frames=dict(layout="sky_prescreen_frames/<token>/CAM_F0_m<NN>.jpg, NN=01..12 = t0+0.5*NN s (m00 is frames/<token>/CAM_F0.jpg)", n_missing=len(sky_missing),
                                  missing=sky_missing, note="audit extra, not part of the cruise bundle's layout"),
        frames_layout="frames/<token>/<CAM>.jpg (1920x1080 JPEG as in OpenScene); 8 cameras at t0 (same as the cruise bundle: one key frame per token)",
        candidate_roles=roles, primary_literal_candidates_per_slot=primary,
        reading_note="roles[].reading = 'literal' (the frozen text as written, the primary list) or 'alt' (secondary, NOT the frozen rule: event inside [t0, t0+15 s]; merge: same direction). See AMBIGUITIES.md.",
        token_cams=token_cams, images=images,
        calib_rows=dict(file=os.path.basename(cpath), order_16=calib["meta"]["order_16"], camera_order=calib["meta"]["camera_order"], selfcheck_vs_cruise_bundle=calib_selfcheck),
        a0_model_inputs=a0_info,
        bank_crosscheck=bank,
        sources=dict(blobs=BLOB_DIR, meta=META_DIR, db=DB_DIR, candidates=RUN),
        files=files, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(man, open(f"{BUNDLE}/MANIFEST.json", "w"), indent=1)
    print("assemble: tokens", len(tokinfo), "images", found_n, "/", wanted_n, "missing", len(missing_images), "sky missing", len(sky_missing), "bank", {k: (v if k != 'differ' else len(v)) for k, v in bank.items()},
          "calib selfcheck", calib_selfcheck, "files", len(files))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare()
    elif cmd == "extract":
        extract(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif cmd == "sky":
        sky()
    elif cmd == "assemble":
        assemble()
    elif cmd == "all":
        prepare(); extract(int(sys.argv[2]) if len(sys.argv) > 2 else 6); sky(); assemble()
    else:
        print(__doc__)
