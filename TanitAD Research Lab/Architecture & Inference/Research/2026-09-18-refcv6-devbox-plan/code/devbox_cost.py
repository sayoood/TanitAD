"""Dev-box cost arithmetic for PREREG_REFCV6_DEVBOX_PREPARATION.

EVERY input is labelled with its evidence class. Nothing is typed twice:
the markdown quotes this file's output, so a number cannot drift between them.
No GPU, no model, no network -- arithmetic over banked measurements.
"""
import json

# ---------------------------------------------------------------- MEASURED --
# 2026-09-18-occupancy-floor/raw/maphead_1k.json: resnet34, 416x1024, batch 2,
# every head live, 1,000 steps in 29,196.7 s on the dev-box RTX 4060 (8 GiB).
S_PER_STEP_MEASURED = 29196.7 / 1000.0
# the 30-step probe's prediction for the same configuration, kept as the
# optimistic end of the band.
S_PER_STEP_PROBE = 28.30

# 2026-09-17-refcv6-pipeline-validation: cover_evalA 3,221.9 s for 500 windows,
# cover_evalB 2,534.3 s for 400 windows (each pass also ran 1 train step, so
# these are UPPER bounds on the per-window eval cost).
EVAL_S_PER_WINDOW = (3221.9 / 500.0 + 2534.3 / 400.0) / 2.0

# refc_v3_train BASE argv (PREREG_REFCV6 §4): --batch 20 on the A40.
POD_BATCH = 20
DEVBOX_BATCH = 2
FULL_STEPS = 40284      # refcv5-v2's own budget
CUT_STEPS = 12000       # the WP-D precedent 'cut'

# pipeline-validation map gate: 23,772 windows over 139 clips.
WINDOWS_139 = 23772
CLIPS_139 = 139

# clean_split.json (this package): 139 cache clips, 135 SAM3 maps, 11 inside
# the parity TRAIN corpus, 4 without a map, and the two losses are DISJOINT.
CLIPS_CLEAN_WITH_MAP = 124

# cost_model.json (2026-09-17-refcv6-e2e-1024): timm MAC counts per window
# position at 256x1024.
MACS_R101 = 128685441024
MACS_R34 = 57818480640

# PI ruling: the dev box is the only compute until preparation is proven.
SEVEN_DAYS_S = 7 * 24 * 3600.0

DAY = 86400.0
HOUR = 3600.0


def band(steps):
    """(fast, slow) wall-clock seconds for `steps` at the measured band."""
    return steps * S_PER_STEP_PROBE, steps * S_PER_STEP_MEASURED


def row(steps):
    f, s = band(steps)
    return {"steps": steps,
            "hours_fast": round(f / HOUR, 2), "hours_slow": round(s / HOUR, 2),
            "days_fast": round(f / DAY, 3), "days_slow": round(s / DAY, 3)}


windows_per_clip = WINDOWS_139 / CLIPS_139
windows_clean = CLIPS_CLEAN_WITH_MAP * windows_per_clip
windows_half = windows_clean / 2.0
steps_one_epoch_half = windows_half / DEVBOX_BATCH

out = {
    "evidence": {
        "s_per_step": {
            "value_measured": round(S_PER_STEP_MEASURED, 4),
            "value_probe": S_PER_STEP_PROBE,
            "class": "MEASURED (ours)",
            "config": "resnet34.a1_in1k, 416x1024, batch 2, every head live, RTX 4060 8 GiB",
            "artifact": "TanitAD Research Lab/Architecture & Inference/Research/"
                        "2026-09-18-occupancy-floor/raw/maphead_1k.json",
        },
        "eval_s_per_window": {
            "value": round(EVAL_S_PER_WINDOW, 3),
            "class": "ESTIMATED (basis MEASURED)",
            "basis": "cover_evalA 3221.9 s / 500 windows; cover_evalB 2534.3 s / 400 windows",
            "caveat": "each pass also ran 1 train step -> this is an UPPER bound",
            "artifact": "TanitAD Research Lab/Architecture & Inference/Research/"
                        "2026-09-17-refcv6-pipeline-validation/RESULT.md",
        },
    },
    "devbox_budget": {
        "seven_days_s": SEVEN_DAYS_S,
        "steps_in_seven_days_fast": int(SEVEN_DAYS_S / S_PER_STEP_PROBE),
        "steps_in_seven_days_slow": int(SEVEN_DAYS_S / S_PER_STEP_MEASURED),
        "note": "SEQUENTIAL -- this box runs ONE GPU job at a time.",
    },
    "unit_costs": {str(n): row(n) for n in
                   (40, 100, 200, 500, 1000, 2000, 5000, CUT_STEPS, FULL_STEPS)},
    "corpus": {
        "windows_per_clip_measured_mean": round(windows_per_clip, 2),
        "clips_usable_clean_and_mapped": CLIPS_CLEAN_WITH_MAP,
        "windows_clean_estimated": int(round(windows_clean)),
        "windows_per_disjoint_half_estimated": int(round(windows_half)),
        "steps_for_ONE_EPOCH_on_a_half_batch2": int(round(steps_one_epoch_half)),
        "one_epoch_hours_slow": round(steps_one_epoch_half * S_PER_STEP_MEASURED / HOUR, 1),
        "class": "MEASURED clip count (clean_split.json) x MEASURED window mean -> ESTIMATED windows",
    },
    "full_arm": {
        "step_matched": row(FULL_STEPS),
        "step_matched_note": "same STEP count as refcv5-v2 but at batch 2, i.e. ONE TENTH the samples",
        "sample_matched_steps": FULL_STEPS * POD_BATCH // DEVBOX_BATCH,
        "sample_matched": row(FULL_STEPS * POD_BATCH // DEVBOX_BATCH),
        "cut_budget": row(CUT_STEPS),
        "arms_in_prereg_refcv6": 10,
        "ten_arms_step_matched_days_slow":
            round(10 * FULL_STEPS * S_PER_STEP_MEASURED / DAY, 1),
        "ten_arms_sample_matched_days_slow":
            round(10 * FULL_STEPS * POD_BATCH / DEVBOX_BATCH * S_PER_STEP_MEASURED / DAY, 1),
        "ten_arms_CUT_days_slow": round(10 * CUT_STEPS * S_PER_STEP_MEASURED / DAY, 1),
    },
    "eval_costs": {
        "windows_1000_hours": round(1000 * EVAL_S_PER_WINDOW / HOUR, 2),
        "full_held_out_half_hours": round(windows_half * EVAL_S_PER_WINDOW / HOUR, 1),
    },
    "resnet101": {
        "mac_ratio_vs_resnet34": round(MACS_R101 / MACS_R34, 4),
        "cpu_s_per_step_measured": round(382.9 / 5.0, 2),
        "cpu_artifact": "GOALS_AND_CLAIMS.md D-REFCV6-R101: {'done': true, 'step': 5, "
                        "'wallclock_s': 382.9} at 416x1024 batch 1, every head live",
        "hypothetical_gpu_s_per_step_if_it_fitted": round(
            MACS_R101 / MACS_R34 * S_PER_STEP_MEASURED, 1),
        "hypothetical_class": "ESTIMATED -- trunk-MAC scaling only; the heads do not scale "
                              "with the trunk, so treat as an ORDER, not a rate",
        "measured_status": "OOM at 256x1024 batch 2 AND at 416x1024 batch 1 on the 8 GiB card",
    },
    "tactical_floor": {
        "n_floor": 200,
        "windows_per_half": int(round(windows_half)),
        "min_prevalence_to_clear_floor_on_one_half":
            round(200.0 / windows_half, 5),
        "note": "SPEC §4: 10 of 22 tokens already sit under the n=200 floor on the FULL "
                "corpus, so they are structurally unreachable on a 62-clip half.",
    },
}

print(json.dumps(out, indent=1))
with open("devbox_cost_table.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
