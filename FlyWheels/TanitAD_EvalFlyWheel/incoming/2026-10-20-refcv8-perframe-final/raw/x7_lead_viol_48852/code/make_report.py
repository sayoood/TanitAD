"""Derive the SUMMARY block of X7_48852.json and write X7_48852.md. Reads ONLY X7_48852.json (+ file hashes); no new measurement.

usage: python make_report.py  (cwd-independent; paths fixed)
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path("C:/Users/Admin/refcv8_pf/x7_48852")
J = OUT / "X7_48852.json"
d = json.load(open(J, encoding="utf-8"))
d.pop("summary", None)
d.pop("provenance", None)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def ci(lo, hi):
    return f"[{lo:+.4f}, {hi:+.4f}]"


def yn(b):
    return "yes" if b else "no"


ARMS = [("refcv8_48852.os", "refcv8-48,852 (EMA, per-frame feed)"), ("refcv8_40k.os", "refcv8-40k (EMA, per-frame feed)"),
        ("refcv7_50400.os", "refcv7-50,400 (banked)")]
CTRL = [("ha0_ext", "echo (ha0_ext)"), ("ha0", "hold-v0 (ha0)"), ("stop", "STOP")]
PAIRS = [("refcv8_48852_minus_refcv7_50400", "refcv8-48,852 - refcv7-50,400"),
         ("refcv8_40k_minus_refcv7_50400", "refcv8-40k - refcv7-50,400"),
         ("refcv8_48852_minus_refcv8_40k", "refcv8-48,852 - refcv8-40k"),
         ("refcv8_48852_minus_echo_ha0_ext", "refcv8-48,852 - echo (ha0_ext)"),
         ("refcv8_48852_minus_hold_v0_ha0", "refcv8-48,852 - hold-v0 (ha0)")]

# ---------------- summary block (machine-readable, derived) ----------------
summary = {"_what": "per surface, per inference seed: counts, share_of_at_risk, paired deltas [95 % CI], separated, controls. Derived from this file's own blocks.",
           "tier": d["meta"]["tier"], "estimator": d["meta"]["estimator"], "training_seeds": d["meta"]["training_seeds"]}
for sf, SS in d["surfaces"].items():
    srow = {}
    for s, S in SS["seeds"].items():
        r = {"n_windows": S["join"]["n_windows"], "n_lead_windows": S["join"]["n_lead_windows"], "n_at_risk": S["join"]["n_at_risk_recount"],
             "arms": {}, "paired_share_of_at_risk": {}, "controls_all_pass": S["controls_internal"]["all_known_value_controls_pass"],
             "controls": {k: S["controls_internal"][k] for k in ("human_as_plan", "half_speed_plan", "crawl_in_half_overshoot_in_half", "stop_plan")}}
        for nm, _ in ARMS + CTRL:
            a = S["arms"][nm]
            r["arms"][nm] = {"n_viol": a["n_viol"], "n_lead_windows": a["n_lead_windows"], "n_at_risk": a["n_at_risk"],
                             "share_of_at_risk": a["share_of_at_risk"], "ci": [a["share_of_at_risk_ci"]["lo"], a["share_of_at_risk_ci"]["hi"]],
                             "lead_progress_ratio_clipped": a["progress"]["lead_progress_ratio_clipped"],
                             "verdict": a["verdict"]["verdict"]}
        for pn, _ in PAIRS:
            p = S["paired"][pn]
            ar = p["at_risk_denominator"]
            r["paired_share_of_at_risk"][pn] = {"delta": ar["delta"], "lo": ar["lo"], "hi": ar["hi"], "separated": ar["separated"],
                                                "x7_direction_met_at_risk": p["x7_direction_met_at_risk"],
                                                "progress_delta": p["progress_paired"]["delta"],
                                                "progress_delta_ci": [p["progress_paired"]["lo"], p["progress_paired"]["hi"]]}
        srow[s] = r
    both = lambda pn: all(srow[s]["paired_share_of_at_risk"][pn]["x7_direction_met_at_risk"] for s in ("0", "1"))
    summary[sf] = {"seeds": srow,
                   "x7_met_at_both_inference_seeds": {pn: bool(both(pn)) for pn, _ in PAIRS[:3]},
                   "x7_met_per_seed": {pn: {s: bool(srow[s]["paired_share_of_at_risk"][pn]["x7_direction_met_at_risk"]) for s in ("0", "1")} for pn, _ in PAIRS[:3]}}
d["summary"] = summary
inputs = {}
for t, root in d["meta"]["trees"].items():
    for s in (0, 1):
        for pat in ("s6_s%d", "panel_s%d"):
            p = Path(root) / (pat % s) / "manifest.json"
            inputs[f"{t}/{pat % s}/manifest.json"] = sha256(p)
for s in (0, 1):
    for pat in ("s6_s%d", "panel_s%d"):
        p = Path(d["meta"]["refcv7_kit"]) / (pat % s) / "manifest.json"
        inputs[f"refcv7_kit/{pat % s}/manifest.json"] = sha256(p)
d["provenance"] = {
    "repo_ref": "refs/remotes/origin/agent/arch-inf-20260803 @ 75ce238d657b270566e3367d9b8bcec08462de81 (read via git --git-dir=C:/Users/Admin/tanitad-push/.git show)",
    "extracted_reference_files": "ref/ (r7base_stageB.py, r7base_stageD_leadprog.py, lead_metrics.py, ci.py, refcv3_arm_repo.py, refav1_arm_repo.py, banked r7base JSONs); code is imported from the C:/lgs/Lfb47a58376 tree whose lead_metrics.py / ci.py blobs equal the repo ref's",
    "code_sha256": {"x7_r8.py": sha256(OUT / "code" / "x7_r8.py"), "make_report.py": sha256(OUT / "code" / "make_report.py")},
    "input_manifest_sha256": inputs,
    "env": "C:/Users/Admin/venvs/tanitad, CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4, PYTHONPATH=C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval, PYTHONDONTWRITEBYTECODE=1",
    "clip_ids": "none written: episodes are file indices; the panel identity is a sha12 of the joined clip list",
}
json.dump(d, open(J, "w", encoding="utf-8"), indent=1, default=str)

# ---------------- markdown ----------------
L = []
w = L.append
S6, S2 = d["surfaces"]["S6"], d["surfaces"]["S2"]
w("# X7 LEAD-VIOL + distance keeping on refcv8 (48,852 EMA, per-frame feed) - EvalFlyWheel, 2026-10-20")
w("")
w("* **Tier: T1 self-action OPEN LOOP on logged eval139 frames - NOT driving performance.** A one-shot plan from t0 scored against the logged lead track.")
w("* **Estimator.** `taniteval.ci` episode-cluster bootstrap (B 2000, seed 0); deltas are the PAIRED version. It answers *'would another draw of EPISODES say this?'* and nothing else.")
w("  It is blind to *another TRAINING run* (H-ESTIM-SEED-1): **refcv8 and refcv7 are each ONE training seed**, so a separated interval here is necessary, not sufficient, for a lever effect.")
w("  It is also not an *inference* replicate: both models sample (DDIM); seeds 0 and 1 are two inference draws of the same checkpoint, reported side by side, never pooled in the bar.")
w("* **Evidence class: MEASURED** (ours; `X7_48852.json` in this folder). The refcv7 row is re-derived here from banked per-window plans and checked against the banked r7base numbers (below).")
w("* **Instrument:** `taniteval/lead_metrics.py` (`lead_viol`, `lead_viol_verdict`, `paired_lead_viol`, `paired_distance_keeping`, `path_arc_length`) as at repo ref `75ce238d` (blob `bd89520d`, identical in the `C:/lgs/Lfb47a58376` tree that was imported). Join: `refcv3_arm.lead_block_common_grid` + `refav1_arm.join_lead_block` (function sources hash-identical to the repo ref), 6 s lead block `b1_eval_lead_block_k12_dt0p5.npz`, md5 `c346dff3b3d356f55ef2786f327c6c91` (checked), identity frame map, raw offset +2. Adapted call for call from `r7base_stageB.py` / `r7base_stageD_leadprog.py` (no new instrument was written).")
w("")
w("## 1. Verdict on the registered bar (S6, `share_of_at_risk`, human keeps >= 1.0 s; admissible only if `lead_progress_ratio_clipped` >= 0.90)")
w("")
w("The bar text, as quoted in the r7base package (INHERITED, SPEC itself not re-read): *refcv8 - refcv7 < 0, separated at both seeds*. Instrument key: `x7_direction_met_at_risk`.")
w("")
w("| paired delta (negative = refcv8 violates less) | seed 0 | seed 1 | bar met at BOTH seeds |")
w("|---|---|---|---|")
for pn, lab in PAIRS[:2]:
    cells = []
    for s in ("0", "1"):
        p = S6["seeds"][s]["paired"][pn]["at_risk_denominator"]
        cells.append(f"{p['delta']:+.4f} {ci(p['lo'], p['hi'])} separated **{yn(p['separated'])}**")
    met = d["summary"]["S6"]["x7_met_at_both_inference_seeds"][pn]
    w(f"| {lab} | {cells[0]} | {cells[1]} | **{'MET' if met else 'NOT MET'}** |")
w("")
w("* **refcv8-48,852: the bar is NOT MET as registered.** The point delta is negative at both draws (-0.0145 / -0.0205 = 12 and 17 fewer violating windows out of 830 at-risk), and seed 1 separates, but **seed 0's interval reaches +0.0024** - one draw out of two fails the 'both seeds' text. Reported as written; the criterion is not moved.")
w("* **refcv8-40k: MET at both seeds** (-0.0169 [-0.0348, -0.0024] and -0.0169 [-0.0334, -0.0044]).")
w("* **48,852 vs 40k are not distinguishable on this metric** (paired 48,852 - 40k: +0.0024 [-0.0072, +0.0108] and -0.0036 [-0.0103, +0.0022]). The ADE got worse after 40k; the distance-keeping read did not - it is the SAME ~4 % violation share, a different question.")
w("* **Progress control clears for every refcv8 arm**: `lead_progress_ratio_clipped` 0.943-0.946 (>= 0.90), and the paired progress delta vs refcv7 is not separated at either seed (48,852: -0.0136 [-0.0658, +0.0480] / -0.0001 [-0.0462, +0.0634]). The lower violation share is not bought by braking (STOP reads exactly 0 with ratio 0 and is INADMISSIBLE-CRAWL).")
w("* **Inference-draw floor (would another INFERENCE run say this?)**: the 48,852 delta moved **0.0060** between the two draws (-0.0145 -> -0.0205), larger than the whole 48,852-vs-40k gap, and the seed-0 shortfall (CI hi +0.0024) is smaller than that floor. 12 of 1,151 lead windows flip their violation flag between the two draws of the same 48,852 checkpoint (9 for 40k, 15 for refcv7).")
ex = S6["exploratory_post_hoc_two_draw_mean"]["rows"]
w(f"* **EXPLORATORY, POST-HOC, NOT THE BAR:** averaging the two inference draws per window (removes sampler noise only) gives 48,852 - refcv7 = {ex['refcv8_48852_minus_refcv7_50400']['delta']:+.4f} {ci(ex['refcv8_48852_minus_refcv7_50400']['lo'], ex['refcv8_48852_minus_refcv7_50400']['hi'])}, separated; 40k - refcv7 = {ex['refcv8_40k_minus_refcv7_50400']['delta']:+.4f} {ci(ex['refcv8_40k_minus_refcv7_50400']['lo'], ex['refcv8_40k_minus_refcv7_50400']['hi'])}. Decided after seeing the per-draw result; it does not replace the registered reading and does not address training-seed variance.")
w("* **S2 (2 s) is near-blind, as the X7 package said**: refcv7 has 5 / 8 violating windows out of 927 at-risk; no refcv8 - refcv7 delta separates at either seed on either checkpoint (48,852: +0.0011 [0.0000, 0.0036] / -0.0032 [-0.0071, 0.0000]).")
w("")
for sf, SS, title in (("S6", S6, "## 2. S6 (6 s, instants 1..6 s, dt 1 s) - the bar surface"), ("S2", S2, "## 3. S2 (2 s, instants 0.5..2.0 s, dt 0.5 s) - beside S6")):
    w(title)
    w("")
    j = SS["seeds"]["0"]["join"]
    w(f"Windows {j['n_windows']}, episodes {SS['seeds']['0']['panels']['refcv8_48852']['n_episodes']}; join states {j['state_counts']}; **lead windows {j['n_lead_windows']}** ({j['n_episodes_with_lead_window']} episodes), **at-risk windows {j['n_at_risk_recount']}** ({j['n_episodes_with_at_risk_window']} episodes); label-free speed proof max |dv0| = {j['speed_check_max_mps']} m/s over {j['n_episodes_ok']}/{j['n_episodes']} episodes; identical in all three panels and both seeds (asserted).")
    w("")
    w("| seed | arm | n viol / n lead | LEAD-VIOL (all lead) | **share_of_at_risk** [95 % CI] | n at-risk | plan lost lead | share incl. lost | progress ratio (clipped) | verdict |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in ("0", "1"):
        for nm, lab in ARMS + CTRL:
            a = SS["seeds"][s]["arms"][nm]
            w(f"| {s} | {lab} | {a['n_viol']}/{a['n_lead_windows']} | {f4(a['share'])} | **{f4(a['share_of_at_risk'])}** [{f4(a['share_of_at_risk_ci']['lo'])}, {f4(a['share_of_at_risk_ci']['hi'])}] | {a['n_at_risk']} | {a['n_plan_lost_lead']} | {f4(a['share_incl_lost_lead'])} | {f4(a['progress']['lead_progress_ratio_clipped'])} | {a['verdict']['verdict']} |")
    w("")
    w("**Paired, `share_of_at_risk` denominator (the bar key), episode-cluster paired bootstrap, B 2000:**")
    w("")
    w("| seed | pair (A - B) | n viol A vs B | delta [95 % CI] | separated | bar met (`x7_direction_met_at_risk`) | progress ratio A vs B | paired progress delta [95 % CI] |")
    w("|---|---|---|---|---|---|---|---|")
    for s in ("0", "1"):
        for pn, lab in PAIRS:
            p = SS["seeds"][s]["paired"][pn]
            ar, pg = p["at_risk_denominator"], p["progress_paired"]
            w(f"| {s} | {lab} | {ar['n_viol_a']} vs {ar['n_viol_b']} | {ar['delta']:+.4f} {ci(ar['lo'], ar['hi'])} | {yn(ar['separated'])} | {p['x7_direction_met_at_risk']} | {pg['lead_progress_ratio_a']:.3f} vs {pg['lead_progress_ratio_b']:.3f} | {pg['delta']:+.4f} {ci(pg['lo'], pg['hi'])} |")
    w("")
    w("**Distance keeping (headway / time-gap / TTC), per arm over the lead windows, mean [95 % CI]** (min over the surface's instants; time gap undefined where ego v0 ~ 0; TTC pinned at 30 s where the plan never closes - `n_closing` is the denominator that matters):")
    w("")
    w("| seed | arm | headway_min [m] | time_gap_min [s] | min_TTC [s] | n closing |")
    w("|---|---|---|---|---|---|")
    h = SS["seeds"]["0"]["join"]["human_distance_keeping_level"]
    w(f"| - | human path (reference) | {h['mean_headway_min_m']['mean']:.2f} [{h['mean_headway_min_m']['lo']:.2f}, {h['mean_headway_min_m']['hi']:.2f}] | {h['mean_time_gap_min_s']['mean']:.3f} [{h['mean_time_gap_min_s']['lo']:.3f}, {h['mean_time_gap_min_s']['hi']:.3f}] | {h['mean_min_ttc_s']['mean']:.2f} [{h['mean_min_ttc_s']['lo']:.2f}, {h['mean_min_ttc_s']['hi']:.2f}] | {h['n_closing']} |")
    for s in ("0", "1"):
        for nm, lab in ARMS:
            L_ = SS["seeds"][s]["arms"][nm]["distance_keeping_level_all_lead_windows"]
            hw, tg, tt = L_["mean_headway_min_m"], L_["mean_time_gap_min_s"], L_["mean_min_ttc_s"]
            w(f"| {s} | {lab} | {hw['mean']:.2f} [{hw['lo']:.2f}, {hw['hi']:.2f}] | {tg['mean']:.3f} [{tg['lo']:.3f}, {tg['hi']:.3f}] | {tt['mean']:.2f} [{tt['lo']:.2f}, {tt['hi']:.2f}] | {L_['n_closing']} |")
    w("")
    w("**Paired distance-keeping deltas** (jointly-finite windows only; `n_used` shown; positive headway/gap/TTC = more room):")
    w("")
    w("| seed | pair (A - B) | headway_min [m] | time_gap_min [s] | min_TTC [s] | n_used (hw / tg / ttc) |")
    w("|---|---|---|---|---|---|")
    for s in ("0", "1"):
        for pn, lab in PAIRS[:3]:
            p = SS["seeds"][s]["paired_distance_keeping"][pn]
            cells = []
            for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
                b = p[k]
                cells.append(f"{b['delta']:+.3f} {ci(b['lo'], b['hi'])} sep **{yn(b['separated'])}**")
            w(f"| {s} | {lab} | {cells[0]} | {cells[1]} | {cells[2]} | {p['headway_min_m']['n_used']} / {p['time_gap_min_s']['n_used']} / {p['min_ttc_s']['n_used']} |")
    w("")
    fl = SS["inference_seed_floor"]["rows"]
    w("**Inference-draw replicate (seed 0 vs seed 1, same windows, same checkpoint):** " + "; ".join(
        f"{k.replace('.os', '')}: {v['s0']:.4f} vs {v['s1']:.4f} (|diff| {v['abs_diff']:.4f}, {v['n_lead_windows_whose_violation_flag_differs_between_the_two_draws']} flag flips)"
        for k, v in fl.items() if k.endswith(".os")) + ".")
    w("")
w("## 4. Controls that must read known values (MEASURED, every surface x seed)")
w("")
w("| surface | seed | human-as-plan: ratio / n_viol (expect 1.0 / 0) | half-speed plan: ratio / verdict (expect 0.5 / CRAWL) | crawl-in-half, overshoot-in-half: unclipped / clipped / masks (expect ~1 / ~0.5 / True) | STOP: n_viol / ratio / verdict (expect 0 / 0 / CRAWL) | all pass |")
w("|---|---|---|---|---|---|---|")
for sf, SS in (("S6", S6), ("S2", S2)):
    for s in ("0", "1"):
        c = SS["seeds"][s]["controls_internal"]
        h, hf, cr, st = c["human_as_plan"], c["half_speed_plan"], c["crawl_in_half_overshoot_in_half"], c["stop_plan"]
        w(f"| {sf} | {s} | {h['lead_progress_ratio']} / {h['n_viol']} | {hf['lead_progress_ratio']} / {hf['verdict']} | {cr['lead_progress_ratio']:.3f} / {cr['lead_progress_ratio_clipped']:.3f} / {cr['overshoot_masks_crawl']} | {st['n_viol']} / {st['lead_progress_ratio']} / {st['verdict']} | **{'PASS' if c['all_known_value_controls_pass'] else 'FAIL'}** |")
w("")
w("The controls run on the shared human path `g` (identical in all three panels), so their values are properties of the instrument and the join, not of the model; they are reproduced here on the refcv8 panels and equal the banked refcv7 values (S6: human 1.0 / half 0.5, `controls_equal_banked_r7base_controls` = True at both seeds). Content asserts (STOP all-zero, every `os` finite and non-degenerate, refcv8 plans differ from refcv7's and from 40k's) hold - a control that cannot fail proves nothing, so these are the guard against a no-op run.")
w("")
w("## 5. Join / pairing checks (all asserted, none skipped)")
w("")
for sf, SS in (("S6", S6), ("S2", S2)):
    for s in ("0", "1"):
        S = SS["seeds"][s]
        rec = S["checks"]["recompute_vs_banked_r7base"]
        w(f"* **{sf} seed {s}:** window grid identical across refcv8-48,852 / refcv8-40k / refcv7 panels (ws, v0, GT path, episode order, model-free arms: equal). Lead windows {S['join']['n_lead_windows']}, at-risk {S['join']['n_at_risk_recount']} - **identical to the refcv7 baseline's counts** ({'1,151 / 830' if sf == 'S6' else '1,188 / 927'}); the grid matches, so no re-definition was needed. The seed-{s} refcv7 slot `b_refcv7_50400_s{s}` carried in the refcv8 panels equals refcv7's own banked `os` (seed {s}) bit-for-bit, and the OTHER seed's slot does not (so a seed mix-up would have been caught). Re-scoring refcv7 and the model-free arms with the current instrument reproduces the banked r7base numbers: {sum(1 for v in rec.values() if v['equal'])}/{len(rec)} comparisons equal (n_lead, n_viol, n_at_risk, share_of_at_risk{', clipped progress ratio' if sf == 'S6' else ''}).")
w("")
w("## 6. Caveats - read before quoting")
w("")
w("1. **One training seed per model.** The bar compares ONE refcv8 training run to ONE refcv7 training run. A separated interval is necessary, not sufficient (H-ESTIM-SEED-1); a replicate (same flags, other training seed) is what would make 'refcv8 lowers LEAD-VIOL' a lever claim.")
w("2. **Small event counts.** 33-37 violating windows (refcv8) vs 49-50 (refcv7) among 830 at-risk, from 63 episodes. The signal is a difference of ~12-17 windows.")
w("3. **Feed asymmetry in the baseline.** refcv8 is scored on its per-frame nav feed (manifest `nav_feed.mode = perframe_full`); the refcv7 panel's manifest carries no `nav_feed` block - refcv7 is the banked battery roll (`D:/refcv7_eval_kit/battery/step50400_a6text`), not re-rolled on the per-frame feed. This is the same pairing the programme's `PAIRED_READ_step48852ema_pf` uses; it was not changed here. Both navs are ego-future-derived oracle tokens (optimistic by construction), so neither row is a deployable-nav number.")
w("4. **Lost-lead windows are not violations under the registered text** (a plan that leaves the lead's corridor has an undefined time gap). They are counted separately: refcv8-48,852 loses the lead on 27 / 23 windows (refcv7: 28) and `share_incl_lost_lead` ranks the same way (0.0539 / 0.0460 vs 0.0652 / 0.0660). Sensitivity, not the bar.")
w("5. **Lead windows are the B1 block's `LEAD` state, not the v9 `lead_valid`** (same definition as the refcv7 baseline; both sides must use it). S-ROUTE LEAD-VIOL (SPEC names S2 and S-ROUTE) is NOT computed - blocker below.")
w("6. **`separated` for S2 `+0.0011 [0.0000, 0.0036]` is a display rounding of a lower bound a hair above/below zero**; the instrument decides on unrounded bounds and says not separated.")
w("")
w("## 7. Next lever (Rule Zero) - and what blocks each")
w("")
w("The bar misses by one draw on 48,852 (seed 0 CI upper +0.0024) while 40k passes at both and the two checkpoints are statistically the same here. The largest-effect, cheapest items, in order:")
w("")
w("1. **DONE on CPU in this run:** two-draw mean (exploratory, labelled) - separated. It is evidence the seed-0 miss is inference-sampler noise at the edge, not that the bar is met.")
w("2. **More inference draws of the 48,852 EMA on the S6 panel (seeds 2, 3, ...).** Settles whether the registered per-draw reading is stable. Blocked on: a GPU re-roll of the checkpoint (this brief was CPU-only; needs a dev-box/Thor GPU slot and the battery roll). Unblock: Master Mind schedules the roll.")
w("3. **A refcv8 training-seed replicate** - the only thing that turns a one-seed paired difference into a lever effect. Blocked on: compute + a PI decision.")
w("4. **S-ROUTE LEAD-VIOL**: blocked on the A6 window list + a plan capture on it (not in any tree here). Unblock: capture the 48,852 plans on the A6 windows.")
w("")
w("## 8. Deliverable manifest (nothing written outside this folder; nothing staged, committed or pushed)")
w("")
w("Folder `C:/Users/Admin/refcv8_pf/x7_48852/`:")
w("")
w("* `X7_48852.json` - all numbers above (+ `summary` and `provenance` blocks).")
w("* `X7_48852.md` - this file.")
w("* `code/x7_r8.py` - the measurement (imports `lead_metrics` / `ci` from `C:/lgs/Lfb47a58376`, joins via `refcv3_arm`/`refav1_arm`); `code/make_report.py` - derives `summary` + this md from the JSON.")
w("* `ref/` - read-only extracts from repo ref `75ce238d` used as the template and as banked comparison (r7base stage B/D scripts, `lead_metrics.py`, `ci.py`, `refcv3_arm_repo.py`, `refav1_arm_repo.py`, banked `leadprog_s6_s*.json`, `lead_viol_s*.json`, the two RESULT.md texts, `real_block_controls.json`, `test_lead_viol.py`, `r7_panel.py`, `r7_roll.py`, `run_battery_r8_pf.py`, `four_families.py`).")
w("* `work/` - run logs (`run3.log`) only; the temporary `shim_*` join files were removed.")
w("")
w("Inputs read (read-only): `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/{s6_s0,s6_s1,panel_s0,panel_s1}`, `C:/Users/Admin/refcv8_pf/battery40k_pf/step40000ema_pf/{...}`, `D:/refcv7_eval_kit/battery/step50400_a6text/{...}`, `C:/Users/Admin/qland/landed_hold_x7/artifacts/b1_eval_lead_block_k12_dt0p5.npz`. Manifest hashes are in `provenance.input_manifest_sha256`.")
open(OUT / "X7_48852.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", OUT / "X7_48852.md", "and summary/provenance into", J)
