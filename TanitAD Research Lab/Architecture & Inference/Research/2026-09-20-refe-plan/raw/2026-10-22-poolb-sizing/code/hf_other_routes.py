#!/usr/bin/env python3
"""Top-level (non-recursive) HF tree listings of candidate third-party repos (metadata only). Writes raw/hf_other_routes.json {repo: [(type,size,path)]}"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlog, urllib.error
repos = ["introvoyz041/navsim", "HQH11111/navsim-data", "ZijianZhang1121/NAVSIM-part2", "Shengzhen/navsim_training_datac_ache", "WaltYoung/nuplan_110w",
         "casko97/nuplanCarBostonAll", "casko97/nuplanTruckBostonAll", "lzrbit/avcl_nuplan", "GALLERVICH/mini_test_nuplan", "Arlolo0/Nuplan-Occupancy",
         "chendylan/navsim_occ", "gray311/NuPlan-examples", "rohit2324/navtrain_tar_files_onkar", "ChiChi2000/Nuplan-QA", "c-chua/navsim-hj-field",
         "rudals/resmap-navsim-teacher-kd", "MrShallow/navsim-generator", "richardyann/navsim-select", "tabguigui/ADJEPA-NAVSIM"]
out = {}
for r in repos:
    url = "https://huggingface.co/api/datasets/%s/tree/main?limit=200" % r
    try:
        body, _ = netlog.get_listing(url, "tree-listing")
        out[r] = [[i["type"], i["size"], i["path"]] for i in json.loads(body)]
    except urllib.error.HTTPError as e:
        out[r] = ["HTTP %s" % e.code]
url = "https://huggingface.co/api/datasets/zhough1201/Nuplan-Dataset/tree/main/v1.1?limit=200"
try:
    body, _ = netlog.get_listing(url, "tree-listing"); out["zhough1201/Nuplan-Dataset/v1.1"] = [[i["type"], i["size"], i["path"]] for i in json.loads(body)]
except urllib.error.HTTPError as e:
    out["zhough1201/Nuplan-Dataset/v1.1"] = ["HTTP %s" % e.code]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "hf_other_routes.json"), "w"), indent=0)
for k, v in out.items():
    print(k, len(v), v[:6])
