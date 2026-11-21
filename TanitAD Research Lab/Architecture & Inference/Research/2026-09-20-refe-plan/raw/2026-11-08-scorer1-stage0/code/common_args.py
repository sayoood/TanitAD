"""Stage 0 argv: COMMON of refe/scorer1_pod.sh (ARM=C0 -> no --pair-*), with the pod paths mapped to the F: layout and two data differences (flagged in HANDOFF)."""
S = "F:/tanitad_scratch/stage0_20261108"
D = S + "/data"
PK = S + "/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
CKPT = "E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt"
SRC = PK + "/raw/2026-10-22-p7-session-reserve/raw/src"
OUT = S + "/run"
DUMP = S + "/dump"
def argv(preflight=False):
    a = ["--ckpt", CKPT, "--targets", D + "/refe_navtrain/train_grow_s0", "--images", D + "/navtrain_pixels",
         "--calib", D + "/refe_navtrain/train_grow_s0/calib_table.json", "--onpolicy", D + "/train_bank/sets",
         "--heldout", D + "/ho/refe_heldout/sets", "--heldout-logs", D + "/ho/refe_heldout/heldout_logs.txt", "--workers", "3",
         "--lr", "0" if not preflight else "1e-4", "--b-mode", "expreward", "--lam", "1.0", "--tau-s", "0.1", "--lane-head", "--lane-weight", "1.0",
         "--teacher-lane-train", D + "/train_bank/train_teacher_lane", "--teacher-lane-heldout", D + "/s1aux/heldout_teacher_lane/heldout_teacher_lane",
         "--lane-labels-heldout", D + "/s1aux/heldout_lane/heldout_lane", "--pdm-labels-heldout", D + "/s1aux/heldout_pdm/heldout_pdm",
         "--pdm-labels-train", D + "/train_bank/train_pdm", "--require-pdm", "--navtest-logs", SRC + "/navtest_logs.json", "--navtest-tokens", SRC + "/navtest_tokens.txt.gz",
         "--max-updates", "1", "--eval-every", "200", "--seed", "0", "--out", OUT]
    if preflight: a.append("--preflight")
    else: a += ["--dump-scores", DUMP]
    return a
