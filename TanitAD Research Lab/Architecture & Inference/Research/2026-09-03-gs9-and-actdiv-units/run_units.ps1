# D-ACTDIV-UNITS-INVARIANCE: actdiv_anchored d(a) on BOTH banked refav1 step-1,000
# checkpoints under BOTH channel-1 conventions. CPU only; the RTX 4060 is left free.
# ONE VARIABLE between the two passes: the number fed on channel 1.
param([string]$Units)
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/colab;C:/Users/Admin/tanitad-wt/taniteval"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "6"
$py  = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$wt  = "C:\Users\Admin\tanitad-wt"
$scr = "C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad\out"
$out = "$scr\actdiv_units_$Units.json"
$log = "$scr\actdiv_units_$Units.log"
Set-Location $wt
"START $Units $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
& $py "$wt\taniteval\tools\actdiv_anchored.py" `
    --family refav1 `
    --arm "incumbent_fp32=C:\Users\Admin\refav1_eval_slice\ckpt\ckpt.pt" `
    --arm "clean_epoch_ema_bf16=C:\Users\Admin\refav1_eval_slice\ckpt_ep2\ckpt.pt" `
    --episodes-n 20 --window-stride 10 --n-perm 200 --seed 0 `
    --device cpu --batch 4 `
    --action-units $Units `
    --out "$out" *>> $log
$rc = $LASTEXITCODE
"EXIT $rc $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
"ZZ${Units}-${rc}-ZZ" | Out-File -FilePath "$scr\status_$Units.txt" -Encoding utf8
