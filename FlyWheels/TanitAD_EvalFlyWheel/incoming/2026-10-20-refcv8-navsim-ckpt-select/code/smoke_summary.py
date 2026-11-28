"""Summarise the Thor smoke from its ARTIFACTS (manifests, rows, records). VALIDATION ONLY; no score is read or produced."""
import glob, json, os, statistics, sys
O = sys.argv[1]; out = {"label": "VALIDATION ONLY (Thor CUDA navtest smoke5; not refcv8's NavSim score)", "steps": {}}
for d in sorted(glob.glob(os.path.join(O, "smoke_ema*"))):
    step = d.rsplit("ema", 1)[1]; b = os.path.join(d, "bridge_navtest"); s = {}
    cc = json.load(open(os.path.join(d, "cuda_controls.json")))
    s["cuda_controls"] = {k: cc.get(k) for k in ("K0_pass", "KD_pass", "premises_pass", "precision", "cuda_max_memory_allocated_gb", "world_s", "torch")}
    s["cuda_controls"]["failed_seam_checks"] = [k for k, v in cc["seam_checks"].items() if not v]
    w = json.load(open(os.path.join(b, "bridge_wrap_record.json")))
    s["bridge_wrap"] = {k: w[k] for k in ("rc", "wall_s", "import_s", "cuda_max_memory_allocated_gb", "torch")}
    s["arms"] = {}
    for mp in sorted(glob.glob(os.path.join(b, "seam_*.manifest.json"))):
        m = json.load(open(mp)); arm = m["arm"]
        e = {k: m.get(k) for k in ("n_expected", "n_rows", "n_missing", "partial", "n_model_rows", "n_cv_standin_rows", "device", "precision", "s_per_scene_forward_mean", "seconds_this_launch")}
        e["KPR"] = m.get("KPR"); e["label"] = m.get("label")
        if m.get("model"):
            e["ckpt_md5"] = m["model"].get("ckpt_md5"); e["step"] = m["model"].get("step"); e["exact_dedup"] = m["model"].get("exact_dedup"); e["load_s"] = m["model"].get("load_s")
        rp = os.path.join(b, f"rows_{arm}.jsonl")
        if os.path.exists(rp) and arm in ("R8_LEGAL", "R8_LEGAL_s1"):
            rows = [json.loads(l) for l in open(rp)]
            e["row_s"] = [r.get("s") for r in rows]
            e["row_nav"] = [r["nav"]["name"] for r in rows]
            e["row_r8_nav_allzero"] = all(all(x == 0.0 for x in r["r8_nav"]) for r in rows)
            e["row_vmax_valid"] = sorted({r["vmax"]["v_max_valid"] for r in rows})
            e["row_seed"] = [r["seed"] for r in rows]
        s["arms"][arm] = e
    leg = s["arms"].get("R8_LEGAL", {}).get("row_s", []); l1 = s["arms"].get("R8_LEGAL_s1", {}).get("row_s", [])
    s["timing"] = {"first_forward_s": leg[0] if leg else None, "warm_forwards_s": sorted(leg[1:] + l1),
                   "warm_median_s": statistics.median(leg[1:] + l1) if (leg[1:] + l1) else None, "n_warm": len(leg[1:] + l1)}
    out["steps"][step] = s
rt = os.path.join(O, "RT_ASSERT.json")
out["rt_assert"] = json.load(open(rt)) if os.path.exists(rt) else None
print(json.dumps(out, indent=1, default=str))
