#!/usr/bin/env python3
"""finalize.py -- run AFTER the detached build chain printed CHAIN_DONE.  Verifies the bundle BY CONTENT, writes FINAL_STATUS.json + MANIFEST.md, validates the CPU A0 run
against the cruise bundle, then deletes this package's own temporary data (literal absolute paths, sizes logged to DELETIONS.json).  It deletes nothing if verification fails.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I code/finalize.py
"""
import hashlib, json, os, shutil, sys, time

ART = "E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107"
F = "F:/TanitAD_tsf_bundle2_20261107"
BUNDLE = F + "/bundle"
CHAIN_LOG = F + "/work_build_chain.log"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def dsize(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def main():
    if not os.path.exists(CHAIN_LOG) or "CHAIN_DONE" not in open(CHAIN_LOG, encoding="utf-8", errors="replace").read():
        print("CHAIN NOT DONE (no CHAIN_DONE in", CHAIN_LOG, ") - nothing done")
        return 2
    import build_bundle as B
    st = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checks={})
    man = json.load(open(BUNDLE + "/MANIFEST.json", encoding="utf-8"))
    # a complete CPU forward may have finished after `assemble` ran: re-assemble once so the bundle carries the picks
    a0 = man.get("a0_model_inputs", {})
    fwd = F + "/a0_forward/a0_model_inputs.jsonl"
    if a0.get("mode") != "forward-cpu" and os.path.exists(fwd) and os.path.exists(F + "/a0_forward/DONE"):
        B.assemble()
        man = json.load(open(BUNDLE + "/MANIFEST.json", encoding="utf-8"))
        a0 = man["a0_model_inputs"]
    # ---- verify by CONTENT -------------------------------------------------------------------------------------------------------------
    bad, missing = [], []
    for rel, v in man["files"].items():
        p = f"{BUNDLE}/{rel}"
        if not os.path.exists(p):
            missing.append(rel)
        elif os.path.getsize(p) != v["bytes"] or sha(p) != v["sha256"]:
            bad.append(rel)
    on_disk = {os.path.relpath(os.path.join(r, f), BUNDLE).replace("\\", "/") for r, _, fs in os.walk(BUNDLE) for f in fs if f != "MANIFEST.json"}
    extra = sorted(on_disk - set(man["files"]))
    tm = man["tm2s_frames"]
    st["checks"].update(
        files_in_manifest=len(man["files"]), files_missing=missing, files_hash_mismatch=bad, files_on_disk_not_in_manifest=extra,
        t0_images=dict(wanted=man["n_images_wanted"], found=man["n_images_found"], missing=man["missing_images"]),
        tm2s_images=dict(wanted=tm["n_images_wanted"], found=tm["n_images_found"], missing=tm["missing_images"]),
        sky_frames_missing=man["sky_prescreen_frames"]["n_missing"], missing_tokens_in_meta=man["missing_tokens_in_meta"],
        calib_selfcheck=man["calib_rows"]["selfcheck_vs_cruise_bundle"], bank_crosscheck=dict(checked=man["bank_crosscheck"]["checked"], equal=man["bank_crosscheck"]["equal"],
                                                                                          differ=man["bank_crosscheck"]["differ"], not_in_bank=man["bank_crosscheck"]["not_in_bank"]),
        a0=a0, tm2s_dt_range_s=[min(tm["dt_to_t0_s"].values()), max(tm["dt_to_t0_s"].values())], n_tokens=man["n_tokens"], n_logs=man["n_logs"])
    ok = not (missing or bad or man["missing_images"] or tm["missing_images"] or man["bank_crosscheck"]["differ"])
    st["bundle_verified_ok"] = bool(ok)
    # ---- sky summary -------------------------------------------------------------------------------------------------------------------
    sky = json.load(open(ART + "/SKY_PRESCREEN.json", encoding="utf-8"))
    rows = []
    for t, v in sorted(sky["tokens"].items(), key=lambda kv: (kv[1]["roles"][0]["slot"], kv[1]["roles"][0]["rank"])):
        rows.append((t, ",".join(f"{r['reading'][0]}:{r['slot']}#{r['rank']}" for r in v["roles"]), v["n_frames"], v["share"], v["pass_proxy"], v.get("frac_median")))
    st["sky"] = dict(n_tokens=len(rows), n_pass_proxy=sum(1 for r in rows if r[4]), rows=rows)
    # ---- CPU A0 vs the cruise bundle (picks) -------------------------------------------------------------------------------------------
    try:
        ref = json.load(open(F + "/a0_test/cruise5_ref.json"))
        new = {json.loads(l)["token"]: json.loads(l) for l in open(F + "/a0_fwd_cruise5/a0_model_inputs.jsonl") if l.strip()}
        cap = {json.loads(l)["token"]: json.loads(l) for l in open(F + "/a0_test/cruise5/a0_model_inputs.jsonl") if l.strip()}
        val = dict(n=len(ref), ego7_goal_equal_capture_only=sum(cap[t]["ego7_as_fed"] == r["ego7_as_fed"] and cap[t]["goal_as_fed"] == r["goal_as_fed"] for t, r in ref.items() if t in cap),
                   ego7_goal_equal_forward=sum(new[t]["ego7_as_fed"] == r["ego7_as_fed"] and new[t]["goal_as_fed"] == r["goal_as_fed"] for t, r in ref.items() if t in new),
                   pick_equal_cpu_vs_gpu=sum(new[t]["pick"] == r["pick"] for t, r in ref.items() if t in new), n_forward_done=len(new),
                   picks=[[t, r["pick"], new.get(t, {}).get("pick")] for t, r in ref.items()], reference="tsf_nuplan_cruise_bundle/a0_model_inputs.jsonl (GPU run)")
    except Exception as e:
        val = dict(error=repr(e))
    json.dump(val, open(ART + "/A0_CPU_VALIDATION.json", "w"), indent=1)
    st["a0_cpu_validation"] = val
    # ---- copies of small evidence into ART ------------------------------------------------------------------------------------------------
    shutil.copyfile(BUNDLE + "/MANIFEST.json", ART + "/BUNDLE_MANIFEST.json")
    for src, dst in ((F + "/work/extract_report.json", "extract_report.json"), (F + "/work/wanted.json", "wanted.json"), (CHAIN_LOG, "build_chain.log")):
        if os.path.exists(src):
            shutil.copyfile(src, f"{ART}/{dst}")
    # ---- deletions (own temp only; re-creatable or banked) ---------------------------------------------------------------------------------
    dels = []
    if ok:
        cand = [F + "/a0_test", F + "/a0_fwd_test", F + "/a0_fwd_cruise5", F + "/a0_fwd_cruise5.out", F + "/work", F + "/work_build_all.log", F + "/work_build_chain.log",
                ART + "/run1", ART + "/code/__pycache__",
                "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/c216c849-5fe7-4d43-a3be-32429241f811/scratchpad/b2"]
        if a0.get("mode") == "forward-cpu" and sha(fwd) == sha(BUNDLE + "/a0_model_inputs.jsonl"):
            cand += [F + "/a0_forward", F + "/a0_forward.out", F + "/a0_capture"]
        for p in cand:
            if os.path.exists(p):
                n = dsize(p)
                (shutil.rmtree if os.path.isdir(p) else os.remove)(p)
                dels.append(dict(path=p, bytes=n))
    json.dump(dels, open(ART + "/DELETIONS.json", "w"), indent=1)
    st["deletions"] = dels
    # ---- MANIFEST.md -----------------------------------------------------------------------------------------------------------------------
    pr = man["primary_literal_candidates_per_slot"]
    L = ["# MANIFEST -- stream BUNDLE2 deliverables (2026-11-07)", "",
         f"Bundle verified by sha256 over {len(man['files'])} files: **{'OK' if ok else 'NOT OK - see FINAL_STATUS.json'}**.  Nothing is staged or committed (brief).", "",
         "| artifact | where | notes |", "|---|---|---|",
         "| `select_suite_tokens.py` | `E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/code/` | frozen rule 3.1; sha256 in SUITE_PIN.json; **E: only** |",
         "| `SUITE_CANDIDATES.json` (PRIMARY, literal) | `tsf_bundle2_20261107/` (copy of `run2/`) | **the list for the sim**; E: only |",
         "| `SUITE_CANDIDATES_ALT.json` (secondary, NOT the frozen rule) | `tsf_bundle2_20261107/` | see AMBIGUITIES.md; E: only |",
         "| `SUITE_PIN.json`, `run_report.json`, `suite_features.csv.gz`, `per_token_results.pkl` | `tsf_bundle2_20261107/` and `run2/` | E: only |",
         "| `AMBIGUITIES.md`, `HAND_VERIFY.txt`, `HANDOFF.md`, `MANIFEST.md` | `tsf_bundle2_20261107/` | E: only |",
         "| `SKY_PRESCREEN.json`, `SKY_METHOD_CHECK.png`, `A0_CPU_VALIDATION.json`, `BUNDLE_MANIFEST.json`, `extract_report.json`, `wanted.json`, `build_chain.log`, `DELETIONS.json`, `FINAL_STATUS.json` | `tsf_bundle2_20261107/` | E: only |",
         "| code: `hand_verify.py`, `sky_prescreen.py`, `build_bundle.py`, `refe_a0_inputs_b2.py`, `finalize.py` | `tsf_bundle2_20261107/code/` | E: only |",
         f"| **the bundle** ({man['n_tokens']} tokens, {man['n_logs']} logs; t0 + tm2s 8 views each; per-log pkls; calib rows; a0_model_inputs.jsonl; MANIFEST.json) | `F:/TanitAD_tsf_bundle2_20261107/bundle/` | **F: only** (nuPlan data, private); re-creatable from the sensor tgz in about 30 min with `build_bundle.py` |",
         "", "## PRIMARY (literal) candidates per slot", ""]
    for s, ts in pr.items():
        L.append(f"* {s}: " + ", ".join(f"`{t}`" for t in ts))
    L += ["", "## Sky pre-screen (ESTIMATED proxy; the sim's QC render decides)", "", "| token | roles (l=literal, a=alt) | frames | share >= 0.5 | pass proxy | median fraction |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    L += ["", f"Single copies: the bundle (F:) and every E: artifact above. Deleted own temp (see DELETIONS.json): {len(dels)} paths, {sum(d['bytes'] for d in dels) / 1e6:.1f} MB."]
    open(ART + "/MANIFEST.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(st, open(ART + "/FINAL_STATUS.json", "w"), indent=1)
    print(json.dumps({k: st[k] for k in ("bundle_verified_ok",)}), "sky pass", st["sky"]["n_pass_proxy"], "/", st["sky"]["n_tokens"], "deleted", len(dels))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
