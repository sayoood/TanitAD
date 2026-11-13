# one-off after setup: unpack the code overlay, give the PK tree a clean path, stage the bank + smoke lists (all inside D:\tanitad\rj5)
$R='D:\tanitad\rj5'; $T='D:\tanitad\tmp'
tar -xf "$T\rj5_w2.tar" -C $R
$src = Get-ChildItem "$R\w" -Directory | Select -First 1
$pk = Join-Path $src.FullName 'Architecture & Inference\Research\2026-09-20-refe-plan'
if (-not (Test-Path "$R\pk")) { Move-Item -LiteralPath $pk -Destination "$R\pk" }
New-Item -ItemType Directory -Force "$R\bank","$R\db","$R\out","$R\cdcache" | Out-Null
Move-Item -LiteralPath "$T\rj5_bank_p7_rank0_slim.jsonl.gz" -Destination "$R\bank\bank_p7_rank0_slim.jsonl.gz" -Force
Move-Item -LiteralPath "$T\rj5_smoke_logs.txt" -Destination "$R\bank\smoke_logs.txt" -Force
Move-Item -LiteralPath "$T\rj5_smoke_tokens.txt" -Destination "$R\bank\smoke_tokens.txt" -Force
Remove-Item -LiteralPath "$T\rj5_w2.tar"
Get-ChildItem $R -Name
