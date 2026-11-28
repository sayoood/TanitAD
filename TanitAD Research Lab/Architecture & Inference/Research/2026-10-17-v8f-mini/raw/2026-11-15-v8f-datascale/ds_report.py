#!/usr/bin/env python3
"""DATASCALE pilot report (pod-side): reads the ledgers/manifests/phase timings and writes /root/datascale/PILOT_REPORT.json (+ prints it).
Every number here is MEASURED from the pilot's own files. No address, port or pod id."""
import json
import os
import statistics as st

B = "/root/datascale"


def jl(p):
    out = []
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:                                      # noqa: BLE001
                    pass
    return out


def du(p):
    n = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            n += os.path.getsize(os.path.join(r, f))
    return n


def main():
    rep = {}
    ph = {}
    if os.path.exists(f"{B}/phases.tsv"):
        for ln in open(f"{B}/phases.tsv", encoding="utf-8"):
            a = ln.split()
            if len(a) == 3:
                ph[a[0]] = round(float(a[2]) - float(a[1]), 1)
    rep["phase_wall_s"] = ph
    stg = {}
    for r in jl(f"{B}/ledger_stage.jsonl"):
        stg[r["clip"]] = r
    pil = [ln.strip() for ln in open(f"{B}/selection/pilot_clips.txt", encoding="utf-8") if ln.strip()]
    ctl = [ln.strip() for ln in open(f"{B}/selection/control_clips.txt", encoding="utf-8") if ln.strip()]
    ps = [stg[c] for c in pil if c in stg]
    rep["stage"] = {"n": len(pil), "ok": sum(r.get("status") == "OK" for r in ps), "fail": [{"clip": r["clip"][:8], "err": r.get("err")} for r in ps if r.get("status") != "OK"],
                    "mp4_MB_mean": round(st.mean(r["mp4_bytes"] for r in ps if r.get("mp4_bytes")) / 1e6, 2) if any(r.get("mp4_bytes") for r in ps) else None,
                    "ego_MB_mean": round(st.mean(r["ego_bytes"] for r in ps if r.get("ego_bytes")) / 1e6, 3) if any(r.get("ego_bytes") for r in ps) else None,
                    "obs_MB_mean": round(st.mean(r["obs_bytes"] for r in ps if r.get("obs_bytes")) / 1e6, 3) if any(r.get("obs_bytes") for r in ps) else None,
                    "n_with_obstacle": sum(1 for r in ps if r.get("obs_bytes")),
                    "n_distinct_chunks": len({r["chunk"] for r in ps})}
    if ph.get("stage_pilot"):
        rep["stage"]["clips_per_h"] = round(len(pil) / ph["stage_pilot"] * 3600, 0)
    bl = jl(f"{B}/ledger_pilot.jsonl")
    bf = jl(f"{B}/ledger_pilot.jsonl.fail")
    rep["build"] = {"ok": len(bl), "fail": [{"clip": r["clip"][:8], "err": r.get("err")} for r in bf],
                    "mean_build_s_per_clip_per_worker": round(st.mean(r["secs"] for r in bl if r.get("secs")), 2) if bl else None,
                    "v2ep_MB_mean": round(st.mean(r["bytes"] for r in bl) / 1e6, 2) if bl else None}
    if ph.get("build_pilot"):
        rep["build"]["clips_per_h_wall_13_workers"] = round(len(pil) / ph["build_pilot"] * 3600, 0)
    mp = jl(f"{B}/cache_pilot/train/manifest.jsonl")
    rep["encode"] = {"cached": len(mp), "cache_MB_mean": round(st.mean(r["bytes"] for r in mp) / 1e6, 2) if mp else None,
                     "enc_s_mean": round(st.mean(r["enc_s"] for r in mp), 2) if mp else None, "n_frames_mean": round(st.mean(r["n_frames"] for r in mp), 1) if mp else None}
    if ph.get("encode_pilot"):
        rep["encode"]["clips_per_h_wall"] = round(len(mp) / ph["encode_pilot"] * 3600, 0)
    # controls
    led = {}
    for r in jl("/root/data/v2ep_ledger.jsonl"):
        led[r["clip"]] = r
    cl = {r["clip"]: r for r in jl(f"{B}/ledger_ctl.jsonl")}
    cmp_ = {r["clip"]: r for r in jl(f"{B}/cache_ctl/train/compare_box2.jsonl")}
    ctrl = []
    for c in ctl:
        row = {"clip": c[:8], "mp4_sha_equal_corpus": stg.get(c, {}).get("mp4_sha256") == led[c]["mp4_sha256"],
               "v2ep_sha_equal_corpus": cl.get(c, {}).get("sha256") == led[c]["sha256"]}
        k = cmp_.get(c)
        if k:
            row["pool4x10"] = k["pool4x10"]
            row["pool8x20"] = k["pool8x20"]
            row["kinematics_equal"] = k["kinematics_equal"]
            row["keys_equal"] = k["keys_equal"]
        ctrl.append(row)
    rep["controls"] = ctrl
    rep["disk_GB"] = {"stage_total": round(du(f"{B}/stage") / 1e9, 2), "v2ep_pilot": round(du(f"{B}/v2ep_pilot") / 1e9, 2), "cache_pilot": round(du(f"{B}/cache_pilot") / 1e9, 2)}
    jn = f"{B}/join_pilot.jsonl"
    if os.path.exists(jn):
        rows = jl(jn)
        rep["obstacle_join"] = {"lines": len(rows), "clips": len({r["clip_id"] for r in rows}), "MB": round(os.path.getsize(jn) / 1e6, 2)}
    if os.path.exists(f"{B}/labelprobe.json"):
        rep["label_probe"] = json.load(open(f"{B}/labelprobe.json", encoding="utf-8"))
    json.dump(rep, open(f"{B}/PILOT_REPORT.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
