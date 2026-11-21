# Anchored action-divergence (Delta-JEPA Fig. 6) on BOTH banked refav1 step-1,000
# checkpoints. CPU only — the RTX 4060 is left free. Launched DETACHED.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/colab;C:/Users/Admin/tanitad-wt/taniteval"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "6"
$py  = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$wt  = "C:\Users\Admin\tanitad-wt"
$out = "C:\Users\Admin\tanitad-wt\TanitAD Research Lab\Architecture & Inference\Research\2026-09-03-anchored-actdiv-refav1\raw\actdiv_anchored_refav1_step1000.json"
$log = "C:\Users\Admin\refav1_eval_slice\actdiv_anchored.log"
Set-Location $wt
"START $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
& $py "$wt\taniteval\tools\actdiv_anchored.py" `
    --family refav1 `
    --arm "incumbent_fp32=C:\Users\Admin\refav1_eval_slice\ckpt\ckpt.pt" `
    --arm "clean_epoch_ema_bf16=C:\Users\Admin\refav1_eval_slice\ckpt_ep2\ckpt.pt" `
    --episodes-n 20 --window-stride 10 --n-perm 200 --seed 0 `
    --device cpu --batch 4 `
    --out "$out" *>> $log
"EXIT $LASTEXITCODE $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
"ZZDONE-$LASTEXITCODE-ZZ" | Out-File -FilePath $log -Append -Encoding utf8
