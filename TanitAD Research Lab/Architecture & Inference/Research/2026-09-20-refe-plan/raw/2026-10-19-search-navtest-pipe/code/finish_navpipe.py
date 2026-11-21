#!/usr/bin/env python3
"""FINISH of the SEARCH-NAVTEST-PIPE stream: idempotent, CPU only, safe to re-run.  Launched by `watch_finish.ps1` once the detached fixture chain and the detached mutation check have both ended (no model turn polls).

  1. reads the fixture chain's verdict (ZZNAVPIPE_FIXTURE_IDENTICAL | FIXTURE_DIFFERS | STOP) and `final/RESULT_NAVPIPE.json`, side by side with Step B's banked P2HEAD read  -> raw/fixture_result.json
  2. re-runs the fixture's chunk 0 with the FINAL code (the chain that produced the verdict was loaded before the S0/table additions)            -> raw/fixture_final_code_chunk0.json
  3. runs the full suite once more                                                                                                               -> logs/pytest_final.log
  4. generates COST.md from the measured fixture rates (cost_navpipe.py)
  5. checks that the fixture's merged outputs are accepted by the EP-ADMISSION bank adapter (`epadm_searchbank.py build`, carve-B data: not navtest, not S0) -> raw/adapter_compat.json
  6. stages ../out/<repo path>/..., out/paths.json and out/RESULT.md; appends the verdict to HANDOFF.md
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-search-navtest-pipe"
OUT = os.path.join(PKG, "out")
STAGE = os.path.join(OUT, *REPO.split("/"))
LOG = os.path.join(PKG, "logs")
BANKED_RESULT = "C:/Users/Admin/par18/l2stepb/run/RESULT_STEPB.json"
ADAPTER = "C:/Users/Admin/par18/epadm_search/out/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-epadm-searchbank/code/epadm_searchbank.py"
GATE_DIR = "C:/Users/Admin/par18/epadm_search/tipgate"
MAX_COPY = 6 * 2 ** 20


def sh(cmd, log, cwd=HERE, env_extra=None, timeout=3600):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", **(env_extra or {}))
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return r.returncode


def tail(path, n=3000):
    return open(path, encoding="utf-8", errors="replace").read()[-n:] if os.path.isfile(path) else ""


def fixture_result():
    log = open(os.path.join(LOG, "fixture_chain.log"), encoding="utf-8", errors="replace").read()
    log = log[max(log.rfind("ZZNAVPIPE_START fixture"), 0):]                    # the log is appended across launches: only the LAST run counts
    res = {"verdict": "NOT_FINISHED"}
    m = re.search(r"^ZZNAVPIPE_(FIXTURE_IDENTICAL|FIXTURE_DIFFERS) (.*)$", log, re.M)
    if m:
        res["verdict"] = m.group(1)
        res["payload"] = json.loads(m.group(2))
    s = re.search(r"^ZZNAVPIPE_STOP (.*)$", log, re.M)
    if s:
        res["verdict"] = "STOPPED"
        res["stop"] = s.group(1)
    res["chain_log_tail"] = log[-1200:]
    fp = os.path.join(PKG, "run", "fixture", "final", "RESULT_NAVPIPE.json")
    if os.path.isfile(fp):
        ours = json.load(open(fp))
        theirs = json.load(open(BANKED_RESULT))["arms"]["P2HEAD"]
        res["ours"] = {k: ours.get(k) for k in ("outcome", "endpoint_dPDMS_points", "per_seed_points", "n_tokens", "n_logs")}
        res["stepB_banked"] = {k: theirs.get(k) for k in ("endpoint_dPDMS_points", "per_seed_points")}
        res["endpoint_json_equal"] = json.dumps(ours["endpoint_dPDMS_points"], sort_keys=True) == json.dumps(theirs["endpoint_dPDMS_points"], sort_keys=True)
        res["per_seed_json_equal"] = json.dumps(ours["per_seed_points"], sort_keys=True) == json.dumps(theirs["per_seed_points"], sort_keys=True)
        res["families_json_equal"] = json.dumps(ours["families"], sort_keys=True) == json.dumps(theirs["families"], sort_keys=True)
    st = os.path.join(PKG, "run", "fixture", "navpipe_state.json")
    if os.path.isfile(st):
        s_ = json.load(open(st))
        res["chunk_status"] = {k: v["status"] for k, v in s_["chunks"].items()}
        res["chunk_seconds"] = {k: v["seconds"] for k, v in s_["chunks"].items()}
    return res


def adapter_compat():
    fin = os.path.join(PKG, "run", "fixture", "final")
    out = {"status": "NOT_RUN"}
    if not (os.path.isfile(ADAPTER) and os.path.isfile(os.path.join(fin, "scores_merged.npz"))):
        return {"status": "NOT_RUN", "why": "adapter or fixture outputs missing"}
    d = os.path.join(PKG, "run", "adapter_compat")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    # the fixture's bundle has no traj_raw in the merged copy; the adapter reads logits / sub / pick (checked from its source), so the merged triple is what it is given
    rc = sh([PY, ADAPTER, "build", "--bundle", os.path.join(fin, "bundle_merged.npz"), "--search", os.path.join(fin, "search_merged.npz"), "--labels", os.path.join(fin, "scores_merged.npz"), "--out-dir", os.path.join(d, "bank")],
            os.path.join(LOG, "adapter_compat.log"), env_extra={"EP_ADMISSION_DIR": GATE_DIR}, timeout=1800)
    out = {"status": "BUILD_OK" if rc == 0 else "BUILD_FAILED", "rc": rc, "log_tail": tail(os.path.join(LOG, "adapter_compat.log"), 1500),
           "note": "carve-B fixture data (token-fresh held-out, NOT navtest, NOT S0); shows the merged npz triple is the layout the adapter reads; no admission decision is computed"}
    return out


def final_code_chunk0():
    w = os.path.join(PKG, "run", "fixture_final1")
    shutil.rmtree(w, ignore_errors=True)
    log = os.path.join(LOG, "fixture_final_code_chunk0.log")
    rc = sh([PY, "-u", os.path.join(HERE, "navpipe_chain.py"), "--mode", "fixture", "--work", w, "--procs", "1", "--min-free-gib", "2.0", "--max-chunks", "1"], log, timeout=7200)
    t = tail(log, 100000)
    m = re.search(r"^ZZNAVPIPE_(FIXTURE_ROWS_IDENTICAL|FIXTURE_ROWS_DIFFER) (.*)$", t, re.M)
    return {"rc": rc, "verdict": m.group(1) if m else "NO_VERDICT", "payload_head": (m.group(2)[:400] if m else t[-600:])}


def stage(extra_files):
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(STAGE)
    items = []

    def put(src, rel):
        try:
            if not os.path.isfile(src) or os.path.getsize(src) > MAX_COPY:
                return
            dst = os.path.join(STAGE, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            items.append([f"{REPO}/{rel}", dst.replace("\\", "/"), "raw"])
        except OSError as e:
            print("ZZFINISH_SKIP", rel, type(e).__name__, flush=True)
    for f in ("DESIGN.md", "COST.md", "HANDOFF.md", "BRIEF.md"):
        put(os.path.join(PKG, f), f)
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        put(f, "code/" + os.path.basename(f))
    for f in sorted(glob.glob(os.path.join(PKG, "ref", "*"))):
        put(f, "ref/" + os.path.basename(f))
    for f in sorted(glob.glob(os.path.join(PKG, "raw", "*.json"))):
        put(f, "raw/" + os.path.basename(f))
    for f in ("fixture_chain.log", "mutation_check.log", "pytest_final.log", "adapter_compat.log", "fixture_final_code_chunk0.log", "finish.log"):
        put(os.path.join(LOG, f), "logs/" + f)
    fx = os.path.join(PKG, "run", "fixture")
    for f in ("chunk_plan.json", "navpipe_state.json"):
        put(os.path.join(fx, f), "fixture/" + f)
    put(os.path.join(fx, "final", "RESULT_NAVPIPE.json"), "fixture/final/RESULT_NAVPIPE.json")
    for f in glob.glob(os.path.join(fx, "chunks", "c*", "bank_manifest.json")) + glob.glob(os.path.join(fx, "chunks", "c*", "identity.json")):
        put(f, "fixture/chunks/" + os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f))
    for f in glob.glob(os.path.join(fx, "logs", "*.log")):
        put(f, "fixture/logs/" + os.path.basename(f))
    for f in extra_files:
        put(f, "raw/" + os.path.basename(f))
    return items


def main():
    t0 = time.time()
    os.makedirs(os.path.join(PKG, "raw"), exist_ok=True)
    fr = fixture_result()
    json.dump(fr, open(os.path.join(PKG, "raw", "fixture_result.json"), "w"), indent=1)
    print("ZZFINISH_FIXTURE", fr["verdict"], flush=True)
    f0 = final_code_chunk0() if fr["verdict"] == "FIXTURE_IDENTICAL" else {"verdict": "SKIPPED (the main fixture did not end IDENTICAL)"}
    json.dump(f0, open(os.path.join(PKG, "raw", "fixture_final_code_chunk0.json"), "w"), indent=1)
    print("ZZFINISH_FINAL_CODE_CHUNK0", f0["verdict"], flush=True)
    rc_t = sh([PY, "-m", "pytest", "test_navpipe.py", "test_navpipe_s0.py", "test_navpipe_guards.py", "-q", "-p", "no:cacheprovider", f"--basetemp={os.path.join(PKG, 'pt_final')}"], os.path.join(LOG, "pytest_final.log"))
    pt = tail(os.path.join(LOG, "pytest_final.log"), 400)
    print("ZZFINISH_PYTEST", rc_t, pt.strip().splitlines()[-1] if pt.strip() else "", flush=True)
    sh([PY, os.path.join(HERE, "cost_navpipe.py")], os.path.join(LOG, "cost.log"))
    ac = adapter_compat()
    json.dump(ac, open(os.path.join(PKG, "raw", "adapter_compat.json"), "w"), indent=1)
    print("ZZFINISH_ADAPTER", ac["status"], flush=True)
    mut = {}
    mp = os.path.join(PKG, "raw", "mutations.json")
    if os.path.isfile(mp):
        j = json.load(open(mp))
        mut = {"caught": j.get("caught"), "judged": j.get("judged"), "survived": [x["guard"] for x in j.get("survived_or_inapplicable", [])], "untestable": [x["guard"] for x in j.get("untestable_without_gpu", [])]}
    status = {"fixture": fr["verdict"], "fixture_final_code_chunk0": f0["verdict"], "pytest_rc": rc_t, "pytest_tail": pt.strip().splitlines()[-1] if pt.strip() else "", "adapter_compat": ac["status"], "mutations": mut,
              "seconds": round(time.time() - t0)}
    json.dump(status, open(os.path.join(PKG, "raw", "finish_status.json"), "w"), indent=1)
    with open(os.path.join(PKG, "HANDOFF.md"), "a", encoding="utf-8") as f:
        f.write("\n\n## FINISHED " + time.strftime("%Y-%m-%d %H:%M") + "\n```json\n" + json.dumps(status, indent=1) + "\n```\n")
    write_result(fr, f0, status, ac)
    items = stage([os.path.join(PKG, "raw", "finish_status.json")])
    put_result = os.path.join(OUT, "RESULT.md")
    shutil.copyfile(os.path.join(PKG, "RESULT.md"), put_result)
    items.append([f"{REPO}/RESULT.md", os.path.join(STAGE, "RESULT.md").replace("\\", "/"), "raw"])
    shutil.copyfile(os.path.join(PKG, "RESULT.md"), os.path.join(STAGE, "RESULT.md"))
    json.dump(items, open(os.path.join(OUT, "paths.json"), "w"), indent=1)
    print("ZZFINISH_DONE", len(items), "files staged", flush=True)
    return 0


def write_result(fr, f0, status, ac):
    o, b = fr.get("ours", {}), fr.get("stepB_banked", {})
    e, be = (o.get("endpoint_dPDMS_points") or {}), (b.get("endpoint_dPDMS_points") or {})
    m = status["mutations"]
    txt = f"""# RESULT, SEARCH-NAVTEST-PIPE (generated by `code/finish_navpipe.py`; the design is `DESIGN.md`, the cost table `COST.md`)

Stamps: package build, CPU only. The fixture is carve B (token-fresh, held-out; **not navtest, not S0**); label-only route; tier T0/selection as in Step B. No navtest or S0 label, score or search output was computed.

## Verdicts
| deliverable | result |
|---|---|
| 1. DESIGN | `DESIGN.md`: navtest + token-list-generic `s0bank` (section 9) |
| 2. gates (a) seal, (b) disk, (c) per-chunk identity, (d) bank-then-delete, (e) RED arms | all wired in `code/`; suite: **{status['pytest_tail']}** (rc {status['pytest_rc']}); mutation check: **{m.get('caught')} of {m.get('judged')} guard lines caught**, survived/inapplicable {m.get('survived')}, untestable without the GPU {m.get('untestable')} |
| 3. fixture on carve B reproduces Step B's P2HEAD read | **{fr['verdict']}**: chain endpoint {json.dumps(e)} vs Step B {json.dumps(be)}; per-seed equal {fr.get('per_seed_json_equal')}, families equal {fr.get('families_json_equal')}, endpoint equal {fr.get('endpoint_json_equal')}; chunk statuses {fr.get('chunk_status')}. Re-run of chunk 0 with the FINAL code: **{f0['verdict']}** |
| 4. measured cost table | `COST.md` (navtest and S0 bank) |
| 5. staging | `out/paths.json`, this file, `HANDOFF.md` |
| adapter layout | `epadm_searchbank.py build` on the fixture's merged triple: **{ac['status']}** (`raw/adapter_compat.json`) |

## Proposed register rows (for the Master Mind; nothing was edited)
* BACKLOG `SEARCH-NAVTEST-PIPE`: built and tested; fixture {fr['verdict']}; the GPU stage (`make_navtest_bundle.py`, `navpipe_bridge.py`, `navpipe_gpu_stage.py`) has NEVER run (first run = chunk 0's bridge); the T1 route was never run on a navtest token.
* BACKLOG `SEARCH-S0-BANK` (new consumer, before navtest): `navpipe_chain.py --mode s0bank` ready on planted data and the fixture's stage functions; needs the S0 amendment's pins, a plan json for 117 logs (dev box has DB + frames for 22/117), and P2's labelled FAN on the 8,922 S0 tokens (not found).
* RETRACTION_LOG (class: *absence found at ONE location is not absence*): `spc/raw/cost.json` `d_navtest` and the brief's "no banked P2 navtest fan exists" are wrong. `D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz` holds proposals [12146,64,8,3], logits [12146,64,6] and pick for all 12,146 navtest tokens. The pipeline uses token/proposals/logits/pick only (`--table-ref`).
* OBSERVATION for the C2 amendment: that table also carries `pdms`/`sub`/`valid` for all 64 proposals (navtest labels). If the in-fan decomposition were ever wanted at T1 it could be read from there; that is a navtest-label read and belongs under the sealed C2 amendment, not here.
* TRAP candidate: the D: copy of `score_navtest_refe.py` (`91ec3e65`) is not the tip's NT-FIX1 version (`94f7b024`); the pipeline vendors and pins the tip's and guards with the T1 planted identity.
* BLOCKER (disk): D: free fell from 24 GB to ~0 during this build; C2 needs >= 9.7 GiB free on its work volume (13.4 with prefetch); the S0 bank the same (largest chunk 991 tokens).
"""
    open(os.path.join(PKG, "RESULT.md"), "w", encoding="utf-8", newline="\n").write(txt)


if __name__ == "__main__":
    sys.exit(main())
