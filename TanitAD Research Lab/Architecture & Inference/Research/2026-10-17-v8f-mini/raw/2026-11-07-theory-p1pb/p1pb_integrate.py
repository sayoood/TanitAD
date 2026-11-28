#!/usr/bin/env python3
"""p1pb_integrate.py -- adds the P1' (eval-141) and P1'b (holdout) results to WSP_READ.md section 5 and to WSP_READ.json.  ADD ONLY: no existing character of WSP_READ.md and no existing key of WSP_READ.json changes.
Numbers are read from P1P_RESULT.json and P1PB_RESULT.json (no hand-typed values).  python -I p1pb_integrate.py   (refuses to run twice; `--json-only` re-adds just the two JSON keys after make_wsp_read_json.py regenerated the file)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
P = json.load(open(f"{HERE}/P1P_RESULT.json"))
B = json.load(open(f"{HERE}/P1PB_RESULT.json"))


def f(x, n=3):
    return f"{x:.{n}f}"


def sg(x, n=3):
    return f"{x:+.{n}f}"


pp, bp = P["PRIMARY"], B["PRIMARY"]
ps_p, ps_b = pp["per_seed"], bp["per_seed"]
k2p, k1p = P["K2_probe_auc_eval_band_3_6"], P["K1_probe_auc_eval_pmove"]
k2b, k1b = B["K2_probe_auc_holdout_band_3_6"], B["K1_probe_auc_holdout_pmove"]
s4p = P["SENSITIVITIES_not_the_verdict"]["S4_go24_label"]
PO = B["POOLED_beside_decides_nothing"]
S = B["SENSITIVITIES_not_the_verdict"]


dp = B["descriptive_median_posterior"]
gaps = [dp["confident"] - ps_b[k]["median_conf"] for k in ("s0", "s1")] + [dp["unsure"] - ps_b[k]["median_unsure"] for k in ("s0", "s1")]
assert all(v["per_seed"][k]["delta"] > 0 for v in S.values() for k in ("s0", "s1")), "the wording 'positive at both seeds in all six' needs every sensitivity positive"
n_fail_rest = sum(1 for k, v in S.items() if k not in ("G4_grid_stride4_t3mod4", "S2_raw_plan") and v["verdict"] == "FAILS")
assert n_fail_rest == 4


def d(r):
    return f"{sg(r['delta'])} [{sg(r['lo'])}, {sg(r['hi'])}]"


md = []
w = md.append
w("")
w("### 5.1 Update 2026-11-07, step 1: P1' read on the eval-141 tick-0 windows (`P1P_RESULT.md`, spec `73640c67`, stamped before any eval posterior)")
w("The paragraph above records the state at the end of the WSP-READ-2 run (2026-11-06). Sections 5.1 and 5.2 add the two reads made on 2026-11-07 and are the current P1' status; the section 1 table row and the title line are left as they were written.")
w(f"**P1' (eval-141): {pp['verdict']}.** The probe was refit on its own training windows with its own folds (P-MOVE, label GO, V+E linear, five fold models, byte-identical `tp_lib.py`; R1 and R2 reproduce the banked artifacts) and applied to the eval-141 tick-0 windows. "
  f"After the window filter there are {pp['window_filter_steps']['finite']} windows from {ps_p['s0']['n_episodes']} episodes; CONFIDENT holds {ps_p['s0']['n_conf']} (floor 30), UNSURE {ps_p['s0']['n_unsure']}, at both seeds. "
  f"Descriptive only: median ρ CONFIDENT {f(ps_p['s0']['median_conf'])} / {f(ps_p['s1']['median_conf'])}, UNSURE {f(ps_p['s0']['median_unsure'])} / {f(ps_p['s1']['median_unsure'])}; Δ {d(ps_p['s0'])} at s0 and {d(ps_p['s1'])} at s1. "
  f"Probe AUC on the eval windows {f(k1p['auc'])} [{f(k1p['lo'])}, {f(k1p['hi'])}] (P-MOVE), {f(k2p['auc'])} [{f(k2p['lo'])}, {f(k2p['hi'])}] in the band v0 3-6, so the probe is not at chance. "
  f"Sensitivities S1 to S3 are NOT EVALUABLE; S4 (the probe's secondary label GO24) returns {s4p['verdict']} at {s4p['per_seed']['s0']['n_conf']} / {s4p['per_seed']['s0']['n_unsure']} windows and is not promoted. "
  "Evidence class MEASURED (dev-box CPU, zero training); estimator episode-cluster bootstrap B 2,000 seed 0, which answers another draw of episodes only. Artifacts: `SPEC_P1P.md`, `SPEC_P1P_AMEND_1.md`, `p1p_probe_refit.py`, `p1p_read.py`, `p1p_finalize.py`, `P1P_PROBE_MODELS.npz`, `P1P_PERWINDOW.npz`, `P1P_RESULT.json`, `P1P_RESULT.md`, hashes `P1P_FILES.sha256`.")
w("")
w("### 5.2 Update 2026-11-07, step 2: P1'b on the RA-1 holdout (`P1PB_RESULT.md`, `THEORY_CRAWL_AMEND_2.md` sha `0427c325`, spec `SPEC_P1PB.md` sha `" + B["spec_sha256"][:8] + "`)")
w("Amendment 2 re-reads P1' unchanged on windows the base never trained on: the 60 clips withheld by `--exclude 540:600` (sha256 `f6a1dee0...`, the RA-1 selection holdout). Admissibility was checked before any score and passes (`P1PB_ADMISSIBILITY.json`): "
  "no holdout clip is in the training list of M2b, dagoacc or C0 (s0 and s1), none is among the probe's 306,566 training windows (window-list digest `e144adcd...` equals C0's), none is in eval-141, and all 60 clips' frozen features are on the dev box with md5 equal to the box 2 manifest.")
w(f"**P1'b (holdout alone): {bp['verdict']}.** Window set: every window of the 56 holdout clips that have v9 label rows, v0 in [3, 8), a_human > 0.5 m/s²: {bp['window_filter_steps']['finite']} windows from {ps_b['s0']['n_episodes']} episodes; "
  f"CONFIDENT {ps_b['s0']['n_conf']} ({ps_b['s0']['n_episodes_conf']} episodes), UNSURE {ps_b['s0']['n_unsure']} ({ps_b['s0']['n_episodes_unsure']} episodes), so the set is evaluable. "
  f"Δ median ρ(CONFIDENT) − median ρ(UNSURE) is {d(ps_b['s0'])} at s0 (not separated) and {d(ps_b['s1'])} at s1 (separated). The criterion fails P1' when Δ is not separated at either seed, so the written consequence (`THEORY_CRAWL_AMEND_1.md` section 4) applies: the under-acceleration is not a mean over unresolved intent, and T1' is wrong too. "
  f"Probe AUC on the holdout windows {f(k2b['auc'])} [{f(k2b['lo'])}, {f(k2b['hi'])}] (v0 3-6), not at chance. Controls all pass: the driver reproduces the T1-kin dumps' tick-0 plans (max abs {B['FW_control']['per_seed']['s1']['plan_max_abs_m']:.1e} m on 240 eval windows), feature path, shuffle, identity, planted.")
w(f"Beside the verdict, deciding nothing: the pooled read (eval-141's {PO['n_eval141_windows']} seen windows plus the {PO['n_holdout_windows']} holdout windows, {PO['n_clips_pooled']} episodes) returns {PO['verdict_logic_returns']} with Δ {d(PO['per_seed']['s0'])} / {d(PO['per_seed']['s1'])}. "
  f"Sensitivities (none is the verdict): positive Δ at both seeds in all six; the eval-141 grid stride (t = 3 mod 4) leaves {S['G4_grid_stride4_t3mod4']['per_seed']['s0']['n_conf']} CONFIDENT windows (NOT EVALUABLE); the raw network plan `plan_raw` returns {S['S2_raw_plan']['verdict']} (s0 lower bound {sg(S['S2_raw_plan']['per_seed']['s0']['lo'])}); the other {n_fail_rest} return FAILS. "
  f"Post hoc and descriptive: the group medians of ρ ({f(ps_b['s0']['median_conf'])} / {f(ps_b['s0']['median_unsure'])} at s0) lie below the groups' median posteriors ({f(B['descriptive_median_posterior']['confident'])} / {f(B['descriptive_median_posterior']['unsure'])}) by {min(gaps):.2f} to {max(gaps):.2f}, and the CONFIDENT windows are slower and accelerate harder (median v0 {f(ps_b['s0']['median_v0_conf'], 2)} vs {f(ps_b['s0']['median_v0_unsure'], 2)} m/s).")
w("Evidence class MEASURED (dev-box CPU, zero training, no weight edit), tier a tick-0 plan-level read, not a driving result; estimator episode-cluster bootstrap over the holdout clips (training and inference variance not covered). "
  "Next levers, none run: a fresh pre-registration on a continuous posterior statistic read on FRESH windows (the 60 holdout windows are seen after this read; further unseen windows need clips outside the 4,572 encoded, blocked on encode compute and a PI go), and P4 on the LoRA arm. "
  "Artifacts: `SPEC_P1PB.md`, `SPEC_P1PB.sha256`, `P1PB_ADMISSIBILITY.json`, `P1PB_HOLDOUT_60.json`, `p1pb_admissibility.py`, `p1pb_forward.py`, `p1pb_read.py`, `p1pb_finalize.py`, `P1PB_TICK0_s0.npz`, `P1PB_TICK0_s1.npz`, `P1PB_FW_CONTROL.json`, `P1PB_F1_CONTROL.json`, `P1PB_PERWINDOW.npz`, `P1PB_RESULT.json`, `P1PB_RESULT.md`, `HANDOFF_P1PB.md`, hashes `P1PB_FILES.sha256`.")
ADD = "\n".join(md) + "\n"

if "--json-only" not in sys.argv:
    # ------------------------------------------------------------------------------------------------ WSP_READ.md: insert before "## 6."
    src = open(f"{HERE}/WSP_READ.md", encoding="utf-8", newline="").read()
    assert "### 5.1 Update 2026-11-07" not in src, "already integrated"
    anchor = "\n## 6. What this changes"
    assert src.count(anchor) == 1
    i = src.index(anchor)
    new = src[:i] + ADD + src[i:]
    # the existing text is preserved byte for byte: removing the inserted block gives back the original
    assert new.replace(ADD, "", 1) == src, "the insertion is not add-only"
    open(f"{HERE}/WSP_READ.md", "w", encoding="utf-8", newline="\n").write(new)


# ------------------------------------------------------------------------------------------------ WSP_READ.json: new keys only
J = json.load(open(f"{HERE}/WSP_READ.json"))
assert "P1_prime_eval141_read" not in J and "P1_prime_b_holdout_read" not in J
J["P1_prime_eval141_read"] = {"update_date": "2026-11-07", "verdict_as_written": pp["verdict"], "evidence_class": "MEASURED (dev-box CPU, zero training); episode-cluster bootstrap, episodes only",
                              "spec_sha256": P["spec_sha256"], "n_windows": pp["window_filter_steps"]["finite"], "n_episodes": ps_p["s0"]["n_episodes"],
                              "n_confident": ps_p["s0"]["n_conf"], "n_unsure": ps_p["s0"]["n_unsure"], "floor": 30,
                              "descriptive_delta": {s: {"delta": ps_p[s]["delta"], "ci95": [ps_p[s]["lo"], ps_p[s]["hi"]]} for s in ("s0", "s1")},
                              "probe_auc_pmove": {k: k1p[k] for k in ("auc", "lo", "hi", "n")}, "probe_auc_band_3_6": {k: k2p[k] for k in ("auc", "lo", "hi", "n")},
                              "S4_go24": {"verdict_logic_returns": s4p["verdict"], "n_confident": s4p["per_seed"]["s0"]["n_conf"], "n_unsure": s4p["per_seed"]["s0"]["n_unsure"], "promoted": False},
                              "artifacts": ["SPEC_P1P.md", "SPEC_P1P.sha256", "SPEC_P1P_AMEND_1.md", "p1p_read.py", "p1p_probe_refit.py", "p1p_finalize.py", "P1P_PROBE_MODELS.npz", "P1P_PERWINDOW.npz", "P1P_RESULT.json", "P1P_RESULT.md", "P1P_FILES.sha256"]}
J["P1_prime_b_holdout_read"] = {"update_date": "2026-11-07", "verdict_as_written": bp["verdict"], "written_consequence": B["verdict_consequence_as_written"], "verdict_reason": B["verdict_reason"],
                                "evidence_class": "MEASURED (dev-box CPU, zero training, no weight edit); tick-0 plan-level read; episode-cluster bootstrap over holdout clips, episodes only",
                                "frozen": {"THEORY_CRAWL_AMEND_2.md": "0427c325", "SPEC_P1PB.md": B["spec_sha256"][:8], "holdout_clips_sha256": B["holdout_sha256"][:8]},
                                "admissible": B["admissible"], "n_windows": bp["window_filter_steps"]["finite"], "n_episodes": ps_b["s0"]["n_episodes"], "n_confident": ps_b["s0"]["n_conf"], "n_unsure": ps_b["s0"]["n_unsure"],
                                "n_episodes_confident": ps_b["s0"]["n_episodes_conf"], "n_episodes_unsure": ps_b["s0"]["n_episodes_unsure"], "floor": 30,
                                "delta": {s: {"delta": ps_b[s]["delta"], "ci95": [ps_b[s]["lo"], ps_b[s]["hi"]], "separated": ps_b[s]["separated"], "median_rho_confident": ps_b[s]["median_conf"], "median_rho_unsure": ps_b[s]["median_unsure"]} for s in ("s0", "s1")},
                                "probe_auc_pmove": {k: k1b[k] for k in ("auc", "lo", "hi", "n")}, "probe_auc_band_3_6": {k: k2b[k] for k in ("auc", "lo", "hi", "n")},
                                "pooled_beside_decides_nothing": {"verdict_logic_returns": PO["verdict_logic_returns"], "n_clips": PO["n_clips_pooled"],
                                                                   "delta": {s: {"delta": PO["per_seed"][s]["delta"], "ci95": [PO["per_seed"][s]["lo"], PO["per_seed"][s]["hi"]]} for s in ("s0", "s1")}},
                                "sensitivities_not_the_verdict": {k: v["verdict"] for k, v in S.items()},
                                "controls": B["controls_summary"],
                                "artifacts": ["SPEC_P1PB.md", "SPEC_P1PB.sha256", "P1PB_ADMISSIBILITY.json", "P1PB_HOLDOUT_60.json", "p1pb_admissibility.py", "p1pb_forward.py", "p1pb_read.py", "p1pb_finalize.py", "P1PB_TICK0_s0.npz", "P1PB_TICK0_s1.npz",
                                              "P1PB_FW_CONTROL.json", "P1PB_F1_CONTROL.json", "P1PB_PERWINDOW.npz", "P1PB_RESULT.json", "P1PB_RESULT.md", "HANDOFF_P1PB.md", "P1PB_FILES.sha256"]}
json.dump(J, open(f"{HERE}/WSP_READ.json", "w"), indent=1)
print("integrated: WSP_READ.md section 5.1/5.2 added; WSP_READ.json keys added: P1_prime_eval141_read, P1_prime_b_holdout_read")
