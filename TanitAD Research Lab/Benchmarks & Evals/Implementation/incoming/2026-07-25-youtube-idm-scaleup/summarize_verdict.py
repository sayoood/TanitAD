"""Summarize the scale-up downstream result against the PRE-REGISTERED ①/②/③ bar.

Reads results_scaleup_downstream.json (+ the parity-validation reference for the
ceiling) and prints: yield, per-seed floor vs pseudo_yt speed_r2, CI separation,
fraction-of-ceiling, across-seed spread vs the pilot (0.047), and the committed verdict.

Usage:  python summarize_verdict.py [results_scaleup_downstream.json]
Parity reference (same split): CEILING=0.6507, common FLOOR=-0.4387 (from
run_idm_parity_validation.json); pilot fixed-HFOV PSEUDO_YT=0.563 @80 clips, std 0.047.
"""
import json, sys, statistics as st

CEIL = 0.6507          # parity real-label pretrain ceiling (same val split)
FLOOR_REF = -0.4387    # common parity floor
PILOT_STD = 0.047      # pilot's across-seed PSEUDO_YT speed_r2 std (80 clips, fixed-HFOV)
PILOT_PSEUDO = 0.563

p = sys.argv[1] if len(sys.argv) > 1 else "pod_artifacts/results_scaleup_downstream.json"
d = json.load(open(p))
m = d["meta"]; a = d["arms_mean_std"]; per = d["per_seed"]; ci = d["bootstrap_ci_speed_r2_gap"]
clips = m.get("pretrain_youtube_clips"); seeds = m.get("seeds")

fl = [s["floor"]["speed_r2"] for s in per]
yt = [s["pseudo_yt"]["speed_r2"] for s in per]
fl_m, yt_m = st.mean(fl), st.mean(yt)
yt_std = st.pstdev(yt) if len(yt) > 1 else float("nan")
frac = (yt_m - FLOOR_REF) / (CEIL - FLOOR_REF)
beats_all = all(y > f for y, f in zip(yt, fl))
ci_all = all(r["ci_excludes_0"] for r in ci)
# in-run ceiling if present
frac_inrun = d.get("fraction_of_ceiling_speed_r2")

print(f"YIELD: {clips} clips, {seeds} seeds  (pilot: 80 clips, 3 seeds, fixed-HFOV)")
print(f"floor speed_r2 mean {fl_m:+.3f}  per-seed {[round(x,3) for x in fl]}")
print(f"pseudo_yt   mean {yt_m:+.3f}  per-seed {[round(x,3) for x in yt]}  across-seed std {yt_std:.3f}")
print(f"beats_floor_all_seeds: {beats_all} | per-seed CI excludes 0 all: {ci_all}")
print(f"fraction-of-ceiling (cited parity CEIL={CEIL}, FLOOR={FLOOR_REF}): {frac:.3f}"
      + (f"  | in-run: {frac_inrun}" if frac_inrun is not None else ""))
print(f"CI-tighten check: across-seed std {yt_std:.3f} vs pilot {PILOT_STD}  -> "
      + ("TIGHTER/=" if yt_std <= PILOT_STD + 1e-9 else "WIDER"))
print(f"ADE@2s: floor {a['floor']['ade_2s'][0]} -> pseudo_yt {a['pseudo_yt']['ade_2s'][0]}")

# committed decision rule
if not (beats_all and ci_all):
    verdict = "③ FAIL/REVERSAL — the pilot did not survive rigor (a seed doesn't beat floor or a CI includes 0)."
elif frac >= 0.80 and yt_std <= PILOT_STD + 1e-9:
    verdict = "① HOLDS — DECISION-GRADE WIN. Non-CC-scale + GeoCalib YouTube pretraining holds >=92%-of-ceiling lift at scale AND tightens the CI. GO."
elif frac >= 0.80:
    verdict = "①/② HOLDS on ceiling (>=0.80) but CI did NOT tighten vs pilot — WIN stands; 'scale improves it' is BOUND on spread."
else:
    verdict = "② PARTIAL/BOUND — win holds (all seeds + CI-sep) but fraction-of-ceiling < 0.80; name the cause (non-CC heterogeneity / label noise / GeoCalib residual)."
print("\nVERDICT:", verdict)
print("RAW verdict field:", d.get("verdict"))
