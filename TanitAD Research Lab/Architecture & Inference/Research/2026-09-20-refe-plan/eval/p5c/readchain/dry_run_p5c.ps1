# DRY RUN of the P5-C read chain (serial, detached): 1) seal REGRESSION arm (one sha file tampered: the chain MUST refuse, exit 4),
# 2) identity run: P7's model_final.pt stands in for BOTH arms, P6's banked seam is the base, 64-token navtest subset, read in --rehearsal mode.
# NEVER a P5-C result. Driver: p5c_dry_driver.sh. Progress: <root>\driver.log, <root>\dryrun_*\CHAIN.log.
$hold = 'E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c_readchain'
$d = 'E:/Projects/TanitAD/data/refe_navtest'
$env:P5C_DRY = '1'
$env:P5C_DRY_N = '64'
$env:P5C_TIP = 'F:/tanitad_scratch/readchain/tip'
$env:P5C_DRYWORK_ROOT = 'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain'
$env:P5C_WORK = "$($env:P5C_DRYWORK_ROOT)/dryrun_driver"
$env:P5C_SCRIPT = "$($hold -replace '\\','/')/p5c_dry_driver.sh"
$env:P5C_COMMIT_WAIT_MIN = '720'
foreach ($a in 'C1','C0') {
  Set-Item -Path "Env:P5C_DRY_SEAM_$a" -Value "$d/seams/stage2_p7_full.npz"
  Set-Item -Path "Env:P5C_DRY_DUMP_$a" -Value "$d/seams/stage2_p7_full.proposals.npz"
  Set-Item -Path "Env:P5C_DRY_CSV_$a"  -Value "$d/score/refe_stage2_p7_full/refe_stage2_p7_full.csv"
}
$env:P5C_DRY_CKPT_SRC = 'F:/TanitAD_p7read/run/model_final.pt'
$env:P5C_BASE_SEAM = "$d/seams/refe_p6final.npz"
$env:P5C_BASE_DUMP = 'NONE'
$env:P5C_BASE_CSV  = "$d/score/refe_p6final/refe_p6final.csv"
& "$hold\launch_p5c_read.ps1"
