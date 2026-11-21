#!/usr/bin/env python3
"""c2prep: turn the fixture chain's LAST run into raw/fixture_result_c2prep.json (CPU, read-only on the work dir). Mirrors finish_navpipe.fixture_result with c2prep's paths."""
import json, os, re, shutil, sys, hashlib

C2 = "C:/Users/Admin/par18/c2prep"
RUN = "C:/Users/Admin/c2prep_run"
BANKED_RESULT = "C:/Users/Admin/par18/l2stepb/run/RESULT_STEPB.json"
log = open(f"{C2}/logs/fixture_chain.log", encoding="utf-8", errors="replace").read()
log = log[max(log.rfind("ZZNAVPIPE_START fixture"), 0):]
res = {"verdict": "NOT_FINISHED", "tip": "0d5cef5d4c24a786d6414b110e29ea2da66d585c"}
m = re.search(r"^ZZNAVPIPE_(FIXTURE_IDENTICAL|FIXTURE_DIFFERS) (.*)$", log, re.M)
if m:
    res["verdict"] = m.group(1)
    res["payload"] = json.loads(m.group(2))
s = re.search(r"^ZZNAVPIPE_STOP (.*)$", log, re.M)
if s:
    res["verdict"] = "STOPPED"
    res["stop"] = s.group(1)
res["chain_log_tail"] = log[-1500:]
fp = f"{RUN}/fixture/final/RESULT_NAVPIPE.json"
if os.path.isfile(fp):
    ours = json.load(open(fp))
    theirs = json.load(open(BANKED_RESULT))["arms"]["P2HEAD"]
    res["ours"] = {k: ours.get(k) for k in ("outcome", "endpoint_dPDMS_points", "per_seed_points", "n_tokens", "n_logs")}
    res["stepB_banked"] = {k: theirs.get(k) for k in ("endpoint_dPDMS_points", "per_seed_points")}
    for k, n in (("endpoint_dPDMS_points", "endpoint_json_equal"), ("per_seed_points", "per_seed_json_equal"), ("families", "families_json_equal")):
        res[n] = json.dumps(ours[k], sort_keys=True) == json.dumps(theirs[k], sort_keys=True)
st = f"{RUN}/fixture/navpipe_state.json"
if os.path.isfile(st):
    s_ = json.load(open(st))
    res["chunk_status"] = {k: v["status"] for k, v in s_["chunks"].items()}
    res["chunk_seconds"] = {k: v["seconds"] for k, v in s_["chunks"].items()}
rec = f"{RUN}/fixture/FIXTURE_RECORD.json"
if os.path.isfile(rec):
    shutil.copyfile(rec, f"{C2}/raw/FIXTURE_RECORD_c2prep.json")
    res["fixture_record_sha256"] = hashlib.sha256(open(rec, "rb").read()).hexdigest()
    res["fixture_record_verdict"] = json.load(open(rec)).get("verdict")
os.makedirs(f"{C2}/raw", exist_ok=True)
json.dump(res, open(f"{C2}/raw/fixture_result_c2prep.json", "w"), indent=1)
print("VERDICT", res["verdict"], res.get("fixture_record_verdict"), res.get("stop", ""))
