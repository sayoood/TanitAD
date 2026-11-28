#!/usr/bin/env python3
"""Tests of the launch-gate profile `refe_sft4r` (eval/PREREG_SFT4R.md Amendment 3; chain refe/sft4r_chain.sh).  Every positive assertion has its negative twin.

What the profile must make impossible (CLAUDE.md BINDING, PI 2026-09-26: no training launch without a launch-gate PASS token):
  * a run whose argv is not THE argv (the whole argv is pinned, token for token), or whose EFFECTIVE recipe moved under it (a parser default),
  * a run on INCOMPLETE labels -- SFT-4 launched on 146,185 of 174,912 -- asserted from the trainer's OWN `data` event, not from a log line,
  * a run whose egress was never proven (G-EGR is a required check bound into the token),
  * a token that does not verify for the exact (commit, tree, argv, data) being launched.
The `data` event and the preflight row below are the pod's own, REAL, copied from the S1 chain's N-s0 arm (MEASURED 2026-10-14, /workspace/data/s1aux/runs/N-s0/preflight.jsonl; the same trainer
file, the same inputs: train_pdm 174,912, held-out 3,137, train_teacher_lane 2,500).  Copies live in raw/2026-10-16-sft4r-readiness/evidence/.

Run: python refe/test_sft4r_gate.py   -> ZZTEST_PASS <n>      (CPU only; needs torch to import scorer_finetune.py and bash on PATH).  A skip prints ZZTEST_SKIP and is NOT counted.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_checks as C  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

N = 0
PROF = L.PROFILES["refe_sft4r"]
DATA_EVENT = json.loads('{"event": "data", "train_sets": 174912, "train_sets_in_bank": 174912, "train_label_versions": {"3": 174912}, "navsim_dac_sets": 174912, "heldout_sets": 3137, "heldout_in_bank": 3137, "heldout_logs": 24, "train_rows_in_heldout_logs": 0, "ckpt_format": "refe-full-v1", "per_sample_calib": true, "lane_labels": {"train_sets": 0, "train_stale": 0, "train_superseded": 0, "heldout_sets": 3137, "pdm_train": 174912, "pdm_train_stale": 0, "pdm_train_superseded": 0, "pdm_heldout": 3137, "teacher_lane_train": 2500, "teacher_lane_heldout": 3137, "heldout_stale": 0}, "b_mode": "expreward", "b_compw": "1,1,1,1,1,1", "at": "2026-10-14T18:34:36"}')
PREFLIGHT_EVENT = json.loads('{"event": "preflight", "verdict": "PASS", "gates": {"G1_trainable_is_exactly_the_scorer": {"ok": true, "n_trainable": 78, "n_scorer": 78, "extra": [], "missing": []}, "G7_heldout_disjoint": {"ok": true, "heldout_sets": 3137, "train_rows_in_heldout_logs": 0}, "G3_copies_equal_model_at_step0": {"max_base_vs_A": 0.0, "max_B_vs_A": 0.0, "n_out": 7, "ok": true}, "G4_listnet_identity_and_live": {"ok": true, "bceA": 0.1696706861257553, "lossB_lam0": 0.1696706861257553, "lossB": -0.7154045104980469, "listnet": -0.8850752115249634}, "G8_lane_labels_in_bank": {"train_lane_sets": 0, "train_stale": 0, "train_superseded": 0, "heldout_lane_sets": 3137, "heldout_stale": 0, "train_sets_with_ddc_0.5": 31973, "heldout_sets_with_ddc_0.5": 503, "ok": true}, "G9_pdm_targets_in_bank": {"train_pdm": 174912, "train_stale": 0, "heldout_pdm": 3137, "heldout_stale": 0, "train_sets_served": 174912, "heldout_sets_served": 3137, "train_with_soft_ep": 153426, "heldout_with_soft_ep": 2708, "require_pdm": true, "ok": true}, "G10_teacher_lane_in_bank": {"train_lane_sets": 2500, "heldout_lane_sets": 3137, "train_sets_with_soft_target": 2292, "batch_lane_labelled": 0, "ok": true}, "G2_live": {"ok": true, "per_module": {"score_q_mlp": {"A_nonzero": true, "B_nonzero": true}, "score_dec": {"A_nonzero": true, "B_nonzero": true}, "score_head": {"A_nonzero": true, "B_nonzero": true}}, "frozen_params_with_grad": 0, "base_params_with_grad": 0}, "G5_ckpt_roundtrip_planner_loader": {"ok": true, "max_abs_reload_vs_saved": 0.0, "max_abs_saved_vs_base": 2.46875}}, "s_per_microbatch": 2.2, "est_hours_per_epoch": 13.37, "gpu_mem_gb": 5.0, "at": "2026-10-14T18:35:03"}')
# hf_egress.py gate's g_egr_result.json, in its real shape (hf_egress.gate): the main push verified by the listing, the regression push refused BY THE SERVER
EGR_OK = {"check": "G-EGR", "ts": "2026-10-16T20:00:00Z", "pass": True, "main": {"repo": "Sayood/tanitad-refe-sft4r", "ok": True, "reason": "1 files 100 B"},
          "regression": {"repo": "google/tanitad-egress-regression-must-fail", "ok": False, "reason": "exception:PermissionError:403 Forbidden", "refused_by_server": True}}


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def argv_of(s=None):
    return L.parse_args_string(s if s is not None else " ".join(PROF["pinned_argv"]))


def refs(argv, prof=None):
    return L.profile_refusals(prof or PROF, argv)


def red(argv, frag, prof=None):
    rs = refs(argv, prof)
    ok(any(frag in r for r in rs), (frag, rs))


def pf(data=None, pre=None, tmp=None):
    d = copy.deepcopy(DATA_EVENT)
    d.update(data or {})
    p = copy.deepcopy(PREFLIGHT_EVENT)
    p.update(pre or {})
    f = Path(tmp or tempfile.mkdtemp()) / f"pf_{N}.jsonl"
    f.write_text(json.dumps(d) + "\n" + json.dumps(p) + "\n", encoding="utf-8")
    return C.load_scorer_preflight(f), f


def main() -> int:
    import scorer_finetune as S
    base = argv_of()
    # ======================================================================== 1. the registry: three copies of ONE argv
    ok(sorted(L.PROFILES)[-1] == "refe_sft4r" or "refe_sft4r" in L.PROFILES)
    ok(PROF["runner"] == "scorer" and PROF["parser_module"] == "scorer_finetune" and PROF["trainer_src"] == "scorer_finetune.py" and PROF["trainer"] == "refe/scorer_finetune.py")
    ok(PROF["mintable"] and PROF["require_commit"] and not PROF["rehearsal"] and not PROF.get("arm_rules"), "SFT-4R is one arm: no arm table")
    ok(PROF["checks"] == ("G-HYG", "G-DVB", "G-PREFLIGHT", "G-SUITE", "G-EGR") and "G-CLOCK" in PROF["na_ok"], PROF["checks"])
    ok(all(c in L.CHECKS for c in tuple(PROF["checks"]) + tuple(PROF["na_ok"])), "the profile names a check the gate does not know")
    for e in PROF["suite"]:                                                      # a renamed PASS token would make G-SUITE fail on the pod, not here
        ok((HERE / e["script"]).is_file() and e["token"] in (HERE / e["script"]).read_text(encoding="utf-8"), e)
    ok({e["script"] for e in PROF["suite"]} >= {"test_sft4r_chain.py", "test_sft4r_gate.py", "test_launch_gate.py"})
    r = subprocess.run(["bash", str(HERE / "sft4r_pod.sh")], capture_output=True, text=True, env=dict(os.environ, MODE="argv", SFT4R_WS="/workspace"))
    pod_argv = r.stdout.strip()
    ok(r.returncode == 0 and pod_argv == " ".join(PROF["pinned_argv"]) == L.SFT4R_ARGV, (r.returncode, pod_argv[:200], r.stderr[-200:]))   # pod script == launch_gate constant == pinned argv
    ok(" ".join(argv_of(pod_argv)) == " ".join(base))
    chain = (HERE / "sft4r_chain.sh").read_text(encoding="utf-8")
    ok('ARGS="$(MODE=argv SFT4R_WS=$WS bash $RUN/sft4r_pod.sh)"' in chain and "launch_gate.py run --profile refe_sft4r" in chain and "launch_gate.py\" verify --profile refe_sft4r" in chain,
       "the chain must take its argv from the pod script and name the profile")
    # the argv carries every input the token fingerprints, and the unpinned recipe values are NOT in it (they are the parser's, asserted by sft4r_refusals)
    for f in ("--ckpt", "--targets", "--images", "--calib", "--onpolicy", "--heldout", "--heldout-logs", "--teacher-lane-train", "--teacher-lane-heldout", "--lane-labels-heldout",
              "--pdm-labels-train", "--pdm-labels-heldout"):
        ok(f in PROF["data_flags"] and f in base, f)
    for f in ("--epochs", "--warmup", "--batch", "--accum", "--eval-every", "--seed", "--max-updates"):
        ok(f not in base, f)
    # ======================================================================== 2. the profile is CLEAN on its argv and REFUSES every lever that moves
    ok(refs(base) == [], refs(base))
    ap = S.build_parser()
    dests = {x.dest for x in ap._actions if x.dest != "help"}
    ok(set(C.DVB_KINDS_SCORER) == dests, (sorted(dests - set(C.DVB_KINDS_SCORER)), sorted(set(C.DVB_KINDS_SCORER) - dests)))     # house rule: no trainer flag without a G-DVB entry (this change adds none)
    for flag, nv in (("--require-pdm", 0), ("--lane-head", 0), ("--pdm-labels-train", 1), ("--pdm-labels-heldout", 1), ("--teacher-lane-train", 1), ("--teacher-lane-heldout", 1),
                     ("--lane-labels-heldout", 1), ("--b-mode", 1), ("--lam", 1), ("--tau-s", 1), ("--lr", 1), ("--workers", 1), ("--ckpt", 1), ("--lane-weight", 1)):
        rs = refs(C._drop(base, flag, nv))
        ok(rs and any(flag in r for r in rs), (flag, rs))                        # a flag dropped is REFUSED, by name (required, or pinned, or both)
    for flag, bad in (("--lr", "2e-4"), ("--lam", "0.5"), ("--tau-s", "0.2"), ("--b-mode", "listnet"), ("--workers", "5"), ("--lane-weight", "0.5"),
                      ("--ckpt", "/w/other.pt"), ("--pdm-labels-train", "/w/other_pdm"), ("--out", "/workspace/data/refe_sft4r_other"), ("--images", "/w/pix")):
        red(C._set(base, flag, bad), "PINS " + flag)
    for extra, frag in ((["--max-updates", "400"], "FORBIDS"), (["--epochs", "2"], "requires --epochs 1.0"), (["--seed", "1"], "requires --seed 0"), (["--warmup", "10"], "requires --warmup 50"),
                        (["--batch", "4"], "requires --batch 8"), (["--accum", "4"], "requires --accum 8"), (["--eval-every", "200"], "requires --eval-every 400"),
                        (["--epo"], "FORBIDS"), (["--aux-drv"], "FORBIDS"), (["--preflight"], "FORBIDS"), (["--lane-labels-train", "/w/x"], "FORBIDS"),
                        (["--b-compw", "1,1,1,1,1,1"], "FORBIDS"), (["--dump-scores", "/w/d"], "does not pin"), (["--require-pdm"], "carried 2 times")):
        red(base + extra, frag)
    red(["--require-p" if t == "--require-pdm" else t for t in base], "not an exact option")      # argparse accepts the prefix; the gate's token tests do not
    red(["x"] + base, "belongs to no flag")
    # a parser DEFAULT that moves under a pinned argv (the argv does not name --epochs / --warmup / --batch / --accum / --eval-every / --seed): the recipe rule catches it
    orig = S.build_parser
    for dest, bad, frag in (("epochs", 2.0, "--epochs 1.0"), ("warmup", 20, "--warmup 50"), ("batch", 4, "--batch 8"), ("accum", 4, "--accum 8"), ("eval_every", 100, "--eval-every 400"),
                            ("seed", 3, "--seed 0"), ("max_updates", 400, "--max-updates 0")):
        def patched(dest=dest, bad=bad):
            p_ = orig()
            p_.set_defaults(**{dest: bad})
            return p_
        S.build_parser = patched
        try:
            rs = refs(base)
        finally:
            S.build_parser = orig
        ok(any(frag in r and "parser DEFAULT" in r for r in rs), (dest, rs))
    ok(refs(base) == [])                                                          # restored
    # other profiles do not see the new rules, and the new profile does not see theirs
    ok(all(not p.get("recipe_rules") for n, p in L.PROFILES.items() if n != "refe_sft4r"))
    ok(L.profile_refusals(L.PROFILES["refe_aux_s1"], argv_of(" ".join(PROF["pinned_argv"]))) != [])      # SFT-4R's argv is not an S1 argv (and says so)
    # ======================================================================== 3. G-DVB: the labels are COMPLETE, from the trainer's own data event
    pfl, _ = pf()
    ev = C.check_dvb_scorer(base, pfl, PROF)
    ok(ev["status"] == "PASS", ev["reasons"])
    rows = {r["flag"]: r for r in ev["details"]["rows"] if r["dest"] == "data_expect"}
    ok(len(rows) == len(L.SFT4R_DATA_EXPECT) + len(L.SFT4R_DATA_MIN) + 1 and rows["schedule"]["built"] == 2733 == L.SFT4R_UPDATES, sorted(rows))
    ok(L.SFT4R_UPDATES == -(-174912 // 8 // 8) == 2733)                          # ceil(174,912 / 8 / 8): PREREG_SFT4R's "2,335" is the lost run's 146,185-label epoch
    ok(C.check_dvb_scorer(base, pfl, PROF, arm="data_incomplete")["status"] == "FAIL")                # the deliberate-regression arm: 83.6 % of the sets (SFT-4's 146,185 of 174,912)
    evi = C.check_dvb_scorer(base, pfl, PROF, arm="data_incomplete")
    ok(evi["arm"] == "data_incomplete" and any("data.train_sets" in r for r in evi["reasons"]) and any(r.startswith("schedule") for r in evi["reasons"]), evi["reasons"])
    for tag, mut, frag in (("146,185 sets", {"train_sets": 146185}, "data.train_sets"), ("bank short", {"train_sets_in_bank": 174911}, "data.train_sets_in_bank"),
                           ("a set too many", {"train_sets": 174913}, "schedule"),                                    # ceil(174913/64) = 2734: the schedule is re-derived, not trusted
                           ("held-out leak", {"train_rows_in_heldout_logs": 3}, "data.train_rows_in_heldout_logs"), ("23 held-out logs", {"heldout_logs": 23}, "data.heldout_logs"),
                           ("held-out short", {"heldout_sets": 3136}, "data.heldout_sets"), ("held-out not in bank", {"heldout_in_bank": 3000}, "data.heldout_in_bank")):
        p_, _ = pf(data=mut)
        e_ = C.check_dvb_scorer(base, p_, PROF)
        ok(e_["status"] == "FAIL" and any(frag in r for r in e_["reasons"]), (tag, e_["reasons"]))
    for tag, path, val, frag in (("PDM labels 146,185", "pdm_train", 146185, "lane_labels.pdm_train"), ("stale PDM labels", "pdm_train_stale", 5, "pdm_train_stale"),
                                 ("superseded PDM labels", "pdm_train_superseded", 1, "pdm_train_superseded"), ("held-out PDM short", "pdm_heldout", 3000, "lane_labels.pdm_heldout"),
                                 ("held-out stale", "heldout_stale", 1, "heldout_stale"), ("held-out teacher lane short", "teacher_lane_heldout", 3000, "teacher_lane_heldout"),
                                 ("lane under SFT-4's 2,109", "teacher_lane_train", 2108, "teacher_lane_train"), ("held-out lane short", "heldout_sets", 3000, "lane_labels.heldout_sets")):
        ll = dict(DATA_EVENT["lane_labels"], **{path: val})
        p_, _ = pf(data={"lane_labels": ll})
        e_ = C.check_dvb_scorer(base, p_, PROF)
        ok(e_["status"] == "FAIL" and any(frag in r for r in e_["reasons"]), (tag, e_["reasons"]))
    p_, _ = pf(data={"lane_labels": dict(DATA_EVENT["lane_labels"], teacher_lane_train=2109)})
    ok(C.check_dvb_scorer(base, p_, PROF)["status"] == "PASS")                   # the floor is a floor: exactly SFT-4's count is admitted
    p_, _ = pf(data={"lane_labels": dict(DATA_EVENT["lane_labels"], teacher_lane_train=15000)})
    ok(C.check_dvb_scorer(base, p_, PROF)["status"] == "PASS")                   # ... and so is the full 15,000-set subset
    d_ = {k: v for k, v in DATA_EVENT.items() if k != "train_sets"}
    f_ = Path(tempfile.mkdtemp()) / "nodata.jsonl"
    f_.write_text(json.dumps(d_) + "\n" + json.dumps(PREFLIGHT_EVENT) + "\n", encoding="utf-8")
    e_ = C.check_dvb_scorer(base, C.load_scorer_preflight(f_), PROF)
    ok(e_["status"] == "FAIL" and any("data.train_sets: declared 174912 != built None" in r for r in e_["reasons"]), e_["reasons"])         # a missing count is a FAIL, never a pass
    ev_no = C.check_dvb_scorer(base, None, PROF)
    ok(ev_no["status"] == "FAIL" and any("no preflight `data` event" in r for r in ev_no["reasons"]), ev_no["reasons"])
    # the other scorer profiles carry no data_expect: their G-DVB is unchanged row for row
    s1 = L.PROFILES["refe_aux_s1"]
    ok(not s1.get("data_expect") and not L.PROFILES["refe_rls1"].get("data_expect"))
    # ======================================================================== 4. G-PREFLIGHT: the trainer's own gates, consumed (never grepped)
    pfl, f_ok = pf()
    ev = L.consume_scorer_preflight(f_ok, base, PROF)
    ok(ev["status"] == "PASS", ev["reasons"])
    ok(sorted(ev["details"]["required"]) == ["G1", "G10", "G2", "G3", "G4", "G5", "G7", "G8", "G9"], ev["details"]["required"])        # G11 (EPO) and G12 (aux head) are NOT obliged
    for tag, mut, frag in (("verdict FAIL", {"verdict": "FAIL"}, "verdict"),):
        _, f_ = pf(pre=mut)
        e_ = L.consume_scorer_preflight(f_, base, PROF)
        ok(e_["status"] == "FAIL" and any(frag in r for r in e_["reasons"]), (tag, e_["reasons"]))
    for g in ("G1_trainable_is_exactly_the_scorer", "G3_copies_equal_model_at_step0", "G9_pdm_targets_in_bank", "G10_teacher_lane_in_bank", "G7_heldout_disjoint"):
        gates = copy.deepcopy(PREFLIGHT_EVENT["gates"])
        gates[g]["ok"] = False
        _, f_ = pf(pre={"gates": gates})
        e_ = L.consume_scorer_preflight(f_, base, PROF)
        ok(e_["status"] == "FAIL" and any(g in r for r in e_["reasons"]), (g, e_["reasons"]))
        gates = {k: v for k, v in PREFLIGHT_EVENT["gates"].items() if k != g}                         # a gate ABSENT is a FAIL too (the preflight ran another configuration)
        _, f_ = pf(pre={"gates": gates})
        e_ = L.consume_scorer_preflight(f_, base, PROF)
        ok(e_["status"] == "FAIL" and any("ABSENT" in r for r in e_["reasons"]), (g, e_["reasons"]))
    gates = dict(PREFLIGHT_EVENT["gates"], G11_epo_live_and_identity={"ok": True})                    # an EPO gate on a plain scorer fine-tune: ANOTHER configuration ran
    _, f_ = pf(pre={"gates": gates})
    e_ = L.consume_scorer_preflight(f_, base, PROF)
    ok(e_["status"] == "FAIL" and any("does not oblige" in r for r in e_["reasons"]), e_["reasons"])
    # ======================================================================== 5. G-EGR: hf_egress.py gate's own result, required, bound into the token
    tmp = Path(tempfile.mkdtemp())
    eg = tmp / "g_egr_result.json"
    eg.write_text(json.dumps(EGR_OK), encoding="utf-8")
    ev = C.check_egr(eg, require_ckpt=False)
    ok(ev["status"] == "PASS", ev["reasons"])
    ok(C.check_egr(eg, require_ckpt=True)["status"] == "FAIL")                                        # P6's rule (a first-checkpoint size) is NOT this arm's: the flag matters
    ok(PROF["egr_require_ckpt"] is False and L.PROFILES["refe_p6"].get("egr_require_ckpt", True) is True)
    for tag, mut in (("pass false", {"pass": False}), ("main push failed", {"main": {"ok": False, "reason": "size_mismatch"}}),
                     ("regression not refused", {"regression": {"ok": True, "reason": "1 files", "refused_by_server": False}}),
                     ("refused by our own pre-check", {"regression": {"ok": False, "reason": "over_allowance", "refused_by_server": False}})):
        eg.write_text(json.dumps({**EGR_OK, **mut}), encoding="utf-8")
        e_ = C.check_egr(eg, require_ckpt=False)
        ok(e_["status"] == "FAIL", (tag, e_["reasons"]))
    eg.write_text(json.dumps(EGR_OK), encoding="utf-8")
    ok(C.check_egr(tmp / "absent.json", require_ckpt=False)["status"] == "FAIL")
    # ======================================================================== 6. end to end: run -> evidence -> token -> verify, on a miniature of the pod (the real checks, a stub G-SUITE)
    work = Path(tempfile.mkdtemp())
    pkg = work / "pkg"
    for rel, txt in (("refe/a.py", "print(1)\n"), ("code/c.sh", "echo 3\n"), ("splits/d.yaml", "x: 1\n")):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(txt, encoding="utf-8", newline="\n")
    dd = work / "data"
    for rel, txt in (("model_final.pt", "init"), ("calib.json", "{}"), ("heldout_logs.txt", "L1\n")):
        (dd / rel).parent.mkdir(parents=True, exist_ok=True)
        (dd / rel).write_text(txt, encoding="utf-8")
    for rel in ("train_grow", "pix", "onp", "ho", "tl_train", "tl_held", "lane_held", "pdm_train", "pdm_held"):
        (dd / rel).mkdir(parents=True, exist_ok=True)
        (dd / rel / "x.jsonl").write_text('{"k": 1}\n', encoding="utf-8")
    remap = {"/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt": dd / "model_final.pt", "/workspace/data/refe_navtrain/train_grow": dd / "train_grow",
             "/workspace/data/navtrain_pixels": dd / "pix", "/workspace/data/refe_navtrain/train_grow/calib_table.json": dd / "calib.json",
             "/workspace/data/refe_navtrain/onpolicy/sets": dd / "onp", "/workspace/data/refe_heldout/sets": dd / "ho", "/workspace/data/refe_heldout/heldout_logs.txt": dd / "heldout_logs.txt",
             "/workspace/data/refe_sft2/train_teacher_lane": dd / "tl_train", "/workspace/data/refe_sft2/heldout_teacher_lane": dd / "tl_held", "/workspace/data/refe_sft2/heldout_lane": dd / "lane_held",
             "/workspace/data/refe_sft2/heldout_pdm": dd / "pdm_held", "/workspace/data/refe_sft2/train_pdm": dd / "pdm_train", L.SFT4R_RUN_DIR: work / "run"}
    margv = [remap.get(t, t) for t in base]
    margv = [str(t).replace("\\", "/") if isinstance(t, Path) else t for t in margv]
    stub_suite = [{"raw": [sys.executable, "-c", "print('ZZTEST_PASS')"], "token": "ZZTEST_PASS"}]
    mprof = {**PROF, "suite": stub_suite, "pinned_argv": tuple(margv)}
    ok(L.profile_refusals(mprof, margv) == [], L.profile_refusals(mprof, margv))
    key = GT.load_key(work / "k" / "gate.key", forbid_under=[pkg])
    pfl, f_pre = pf(tmp=work)
    out = work / "lg"
    st = L.run_checks(mprof, margv, pkg, out, preflight_json=f_pre, egr_json=eg, commit="a" * 40, key=key)
    ok(st == {"G-HYG": "PASS", "G-DVB": "PASS", "G-PREFLIGHT": "PASS", "G-SUITE": "PASS", "G-EGR": "PASS", "G-CLOCK": "NA"}, st)
    verdict, path, tok = L.finalize(mprof, margv, pkg, out, key, commit="a" * 40, preflight_json=f_pre, egr_json=eg)
    ok(verdict == "PASS", tok["reasons"])
    ok(set(tok["checks"]) == {"G-HYG", "G-DVB", "G-PREFLIGHT", "G-SUITE", "G-EGR", "G-CLOCK"} and tok["checks"]["G-EGR"]["status"] == "PASS", tok["checks"])
    bd = tok["binding"]
    ok("gate:egr_json" in bd["data"] and "gate:preflight_json" in bd["data"] and set(bd["gate_inputs"]) == {"gate:egr_json", "gate:preflight_json"}, (sorted(bd["data"]), bd["gate_inputs"]))   # the token binds the egress result and the preflight file it was judged on
    good, why, _info = GT.verify_token(path, margv, pkg, key=key, profile="refe_sft4r", data_flags=PROF["data_flags"], commit="a" * 40)
    ok(good is True, why)
    bad, why, _ = GT.verify_token(path, C._set(margv, "--lr", "2e-4"), pkg, key=key, profile="refe_sft4r", data_flags=PROF["data_flags"])
    ok(bad is False and why, why)                                                                      # another argv: no launch
    (dd / "pdm_train" / "x.jsonl").write_text('{"k": 1}\n{"k": 2}\n', encoding="utf-8")               # a label file changed SIZE after the gate: the dir fingerprint no longer matches
    bad, why, _ = GT.verify_token(path, margv, pkg, key=key, profile="refe_sft4r", data_flags=PROF["data_flags"])
    ok(bad is False and why, why)
    # KNOWN LIMIT, pinned here so it stays visible: a DIRECTORY is fingerprinted by listing (relpath + size), contents are NOT hashed (gate_token.fingerprint), so a SAME-SIZE rewrite of a
    # label is invisible to the token.  The chain's PDM egress pins the CONTENT instead (sha256 of the two tars, pushed to HF before training).
    (dd / "pdm_train" / "x.jsonl").write_text('{"k": 2}\n', encoding="utf-8")
    same_size, why, _ = GT.verify_token(path, margv, pkg, key=key, profile="refe_sft4r", data_flags=PROF["data_flags"])
    ok(same_size is True, "the token's directory fingerprint is listing-only; if it ever hashes contents, drop this pin and the READINESS caveat")
    (dd / "pdm_train" / "x.jsonl").write_text('{"k": 1}\n', encoding="utf-8")
    (pkg / "refe" / "a.py").write_text("print(99)\n", encoding="utf-8", newline="\n")                   # the shipped code changed after the gate
    bad, why, _ = GT.verify_token(path, margv, pkg, key=key, profile="refe_sft4r", data_flags=PROF["data_flags"])
    ok(bad is False and why, why)
    (pkg / "refe" / "a.py").write_text("print(1)\n", encoding="utf-8", newline="\n")
    # no egress result: no token (G-EGR is REQUIRED); incomplete labels (the regression arm): no token; a regression arm can never mint
    out2 = work / "lg2"
    L.run_checks(mprof, margv, pkg, out2, preflight_json=f_pre, egr_json=None, commit="a" * 40, key=key)
    v2, _, t2 = L.finalize(mprof, margv, pkg, out2, key, commit="a" * 40, preflight_json=f_pre, egr_json=None)
    ok(v2 != "PASS" and any("G-EGR" in r for r in t2["reasons"]), t2["reasons"])
    out3 = work / "lg3"
    L.run_checks(mprof, margv, pkg, out3, preflight_json=f_pre, egr_json=eg, commit="a" * 40, key=key, arm="data_incomplete")
    v3, _, t3 = L.finalize(mprof, margv, pkg, out3, key, commit="a" * 40, preflight_json=f_pre, egr_json=eg)
    ok(v3 != "PASS" and any("G-DVB" in r for r in t3["reasons"]), t3["reasons"])
    pf146, f146 = pf(data={"train_sets": 146185, "lane_labels": dict(DATA_EVENT["lane_labels"], pdm_train=146185)}, tmp=work)               # SFT-4's actual coverage, with no regression arm at all
    out4 = work / "lg4"
    L.run_checks(mprof, margv, pkg, out4, preflight_json=f146, egr_json=eg, commit="a" * 40, key=key)
    v4, _, t4 = L.finalize(mprof, margv, pkg, out4, key, commit="a" * 40, preflight_json=f146, egr_json=eg)
    ok(v4 != "PASS" and any("G-DVB" in r for r in t4["reasons"]), t4["reasons"])
    out5 = work / "lg5"                                                                                  # the PINNED argv is the real one: its /workspace paths are absent here, so the REAL profile cannot mint on the dev box
    L.run_checks(PROF, base, pkg, out5, preflight_json=f_pre, egr_json=eg, commit="a" * 40, key=key, only={"G-DVB"})
    v5, _, t5 = L.finalize(PROF, base, pkg, out5, key, commit="a" * 40, preflight_json=f_pre, egr_json=eg)
    ok(v5 != "PASS", t5["reasons"][:2])
    # ======================================================================== 7. the CLI the chain calls
    def cli(*args):
        return subprocess.run([sys.executable, str(HERE / "launch_gate.py"), *args], capture_output=True, text=True, encoding="utf-8")
    rc = cli("refusals", "--profile", "refe_sft4r", "--args", " ".join(base))
    ok(rc.returncode == 0 and "ZZGATE-CLEANZZ" in rc.stdout, rc.stdout[-300:] + rc.stderr[-300:])
    rc = cli("refusals", "--profile", "refe_sft4r", "--args", " ".join(base) + " --max-updates 400")
    ok(rc.returncode == 5 and "ZZGATE-REFUSEDZZ" in rc.stdout and "FORBIDS" in rc.stdout, rc.stdout[-300:])
    rc = cli("refusals", "--profile", "refe_sft4r", "--arm-name", "N-s0", "--args", " ".join(base))
    ok(rc.returncode != 0 and "only meaningful for a profile with arm rules" in (rc.stdout + rc.stderr), rc.stderr[-300:])      # one arm: naming one is an error, not a silent no-op
    rc = cli("run", "--profile", "refe_sft4r", "--args", " ".join(base), "--pkg", str(work / "nopkg"), "--out-dir", str(work / "o"))
    ok(rc.returncode != 0 and "mints for ONE named arm" not in (rc.stdout + rc.stderr), (rc.stdout + rc.stderr)[-300:])             # minting needs no --arm-name (it fails later, on the missing package/preflight)
    print(f"ZZTEST_PASS {N}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"ZZTEST_FAIL after {N} checks: {e}")
        raise
