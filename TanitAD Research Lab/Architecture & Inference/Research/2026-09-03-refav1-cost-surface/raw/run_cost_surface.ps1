# R10 — the refav1 planner COST SURFACE on BOTH banked step-1,000 checkpoints,
# the same 140 windows / 20 episodes / stride 10 as the two banked T1 reads.
# Detached (Start-Process) because the pair is > 55 min. Yields the GPU if another
# python appears on it.
$ErrorActionPreference = "Continue"
$slice = "C:\Users\Admin\refav1_eval_slice"
$outdir = "$slice\costsurface"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$log = "$outdir\run.log"
"[$(Get-Date -Format u)] cost-surface run up" | Out-File -FilePath $log -Encoding utf8

Set-Location "C:\Users\Admin\tanitad-wt"
$env:PYTHONPATH = "C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/colab;C:/Users/Admin/tanitad-wt/taniteval"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "6"
$py = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"

$runs = @(
  @{ name = "incumbent"; ckpt = "$slice\ckpt\ckpt.pt";     cfg = $null;                    dump = "$slice\t1_dump" },
  @{ name = "ep2";       ckpt = "$slice\ckpt_ep2\ckpt.pt"; cfg = "$slice\ckpt_ep2\config.json"; dump = "$slice\t1_dump_ep2" }
)

foreach ($r in $runs) {
  "[$(Get-Date -Format u)] === $($r.name) ===" | Add-Content $log
  $a = @(
    "taniteval\tools\cost_surface_probe.py",
    "--ckpt", $r.ckpt,
    "--cache", "$slice\fp8",
    "--episodes", "$slice\eps",
    "--labels", "C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_eval.jsonl.gz",
    "--nav", "C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_eval.jsonl.gz",
    "--banked-dump", $r.dump,
    "--device", "cuda", "--window-stride", "10", "--episodes-n", "20",
    "--gate-n", "5", "--zero-model-n", "140",
    "--wheelbase", "2.9", "--wheelbase-sweep", "--wheelbase-sweep-n", "40",
    "--save-surfaces-n", "8", "--n-boot", "10000",
    "--out", "$outdir\cost_surface_$($r.name).json"
  )
  if ($r.cfg) { $a += @("--config", $r.cfg) }
  & $py @a *>> $log
  "[$(Get-Date -Format u)] $($r.name) EXIT $LASTEXITCODE" | Add-Content $log
}
"[$(Get-Date -Format u)] ALL DONE" | Add-Content $log
