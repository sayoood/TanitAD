"""Derive DriveZero's training scale from the paper's own stated settings.

Every INPUT here is quoted from the paper (Table A1, A12, section 2.2/3.1/3.2); every OUTPUT is
arithmetic on those inputs and is labelled DERIVED, not MEASURED-by-the-authors. The point is that
the paper never states "driving hours" anywhere -- so if we are going to quote one, it has to be
reconstructed in the open.
"""

# ---- inputs, all quoted from the paper -------------------------------------
WORLDS_PER_RANK = 2_048          # Table A1
RANKS = 96                       # "12 nodes / 96 GPUs"
GLOBAL_WORLDS = 196_608          # Table A1 "Global rollout batch"
ROLLOUT_STEPS = 110              # Table A1 "110 steps at 5 Hz"
HZ = 5
UPDATES = 2_400                  # Table A1 "Training updates"
WALLCLOCK_H = 21                 # "approximately 21 h"
TRANS_PER_RANK = 225_280         # Table A1 "Nominal transitions / rank"

VFM_S1_ITERS, VFM_S2_ITERS, VFM_BATCH = 600_000, 200_000, 2_048  # section 2.2

STUDENT_SCENES = 100_000 + 237_000   # navtrain + SimScale
STUDENT_GPUS, STUDENT_H = 16, 38     # Table A12: 16 x H20, 38 h
STUDENT_FULL_M, STUDENT_TRAIN_M = 338.46, 18.58
TEACHER_M = 5.7                      # section 2.1.1

# ---- consistency checks: the paper must agree with itself ------------------
assert WORLDS_PER_RANK * RANKS == GLOBAL_WORLDS, "worlds/rank x ranks != global batch"
assert WORLDS_PER_RANK * ROLLOUT_STEPS == TRANS_PER_RANK, "worlds x steps != transitions/rank"
print("[check] 2,048 x 96 = %d == stated global batch %d  OK" % (WORLDS_PER_RANK * RANKS, GLOBAL_WORLDS))
print("[check] 2,048 x 110 = %d == stated transitions/rank %d  OK" % (WORLDS_PER_RANK * ROLLOUT_STEPS, TRANS_PER_RANK))

# ---- DERIVED: DriveRL teacher ----------------------------------------------
trans_per_update = GLOBAL_WORLDS * ROLLOUT_STEPS
total_trans = trans_per_update * UPDATES
sec_per_trans = 1.0 / HZ
sim_hours_per_update = trans_per_update * sec_per_trans / 3600
sim_hours_total = total_trans * sec_per_trans / 3600
gpu_hours = WALLCLOCK_H * RANKS

print()
print("DriveRL teacher (DERIVED from Table A1)")
print("  transitions / update      %15s" % f"{trans_per_update:,}")
print("  transitions total         %15s  (%.1f billion)" % (f"{total_trans:,}", total_trans / 1e9))
print("  SIM DRIVING HOURS / update%15s" % f"{sim_hours_per_update:,.0f}")
print("  SIM DRIVING HOURS total   %15s  (%.2f million)" % (f"{sim_hours_total:,.0f}", sim_hours_total / 1e6))
print("  sim YEARS of driving      %15.1f" % (sim_hours_total / 8766))
print("  wall clock                %15s h" % WALLCLOCK_H)
print("  GPU-hours                 %15s" % f"{gpu_hours:,}")
print("  sim-hours per GPU-hour    %15s" % f"{sim_hours_total/gpu_hours:,.0f}")
print("  rollout wall-time each    %15.1f s of sim time per world" % (ROLLOUT_STEPS * sec_per_trans))
print("  teacher params            %15s M" % TEACHER_M)

# ---- DERIVED: DriveVFM ------------------------------------------------------
vfm_samples = (VFM_S1_ITERS + VFM_S2_ITERS) * VFM_BATCH
print()
print("DriveVFM backbone (DERIVED from section 2.2)")
print("  stage 1 (256x256)         %15s samples" % f"{VFM_S1_ITERS*VFM_BATCH:,}")
print("  stage 2 (512x512)         %15s samples" % f"{VFM_S2_ITERS*VFM_BATCH:,}")
print("  TOTAL image samples       %15s  (%.2f billion)" % (f"{vfm_samples:,}", vfm_samples / 1e9))
print("  GPU count / hours         %15s" % "NOT STATED in the paper -- named gap")

# ---- DERIVED: student -------------------------------------------------------
print()
print("DriveZero student (Table A12)")
print("  scenes                    %15s" % f"{STUDENT_SCENES:,}")
print("  GPU-hours                 %15s  (%d x H20 x %d h)" % (f"{STUDENT_GPUS*STUDENT_H:,}", STUDENT_GPUS, STUDENT_H))
print("  params full / trainable   %15s" % ("%.2f M / %.2f M" % (STUDENT_FULL_M, STUDENT_TRAIN_M)))
print("  trainable fraction        %15.2f %%" % (100 * STUDENT_TRAIN_M / STUDENT_FULL_M))

# ---- the ratio that matters -------------------------------------------------
NUPLAN_SCENES = 922_703
print()
print("THE RATIO THAT MATTERS")
print("  real nuPlan scenes seeding the worlds   %15s" % f"{NUPLAN_SCENES:,}")
print("  simulated driving hours generated       %15s" % f"{sim_hours_total:,.0f}")
print("  => the teacher's EXPERIENCE is simulated; the real logs only seed scenes and goals.")

# ---- ablation deltas we will quote -----------------------------------------
print()
print("ABLATION DELTAS (Table 7, ViT-S, navtrain only, navtest PDMS) -- the buried headline")
sup = {"human trajectories": 93.92, "DriveRL trajectories": 93.61, "DriveRL + goal aug": 94.41}
for k, v in sup.items():
    print("  %-24s %6.2f   (vs human %+.2f)" % (k, v, v - sup["human trajectories"]))
print("  => RL-teacher supervision ALONE is %.2f BELOW human imitation." % (sup["human trajectories"] - sup["DriveRL trajectories"]))
print("  => the whole gain, %+.2f, comes from GOAL AUGMENTATION." % (sup["DriveRL + goal aug"] - sup["DriveRL trajectories"]))

print()
print("TEST-TIME SEARCH (Table 3) -- I-1's quality half, answered externally")
tts = {"N=8": 93.12, "N=16": 93.13, "N=32": 93.40, "N=64": 93.57}
base = 93.01
for k, v in tts.items():
    print("  %-6s mean %6.2f   delta %+.2f" % (k, v, v - base))
print("  => 8x more candidates buys %+.2f CLS points (%.2f%% relative); below N=32 it is noise."
      % (tts["N=64"] - base, 100 * (tts["N=64"] - base) / base))
