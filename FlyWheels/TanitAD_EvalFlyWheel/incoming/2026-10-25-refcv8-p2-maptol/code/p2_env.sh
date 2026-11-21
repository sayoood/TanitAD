# sourced by every P2 launcher. Evidence: CLAUDE.md traps (PYTHONPATH required, OMP_NUM_THREADS, HF offline).
P2=/c/Users/Admin/refcv8_pf/perc8tol
P2W=C:/Users/Admin/refcv8_pf/perc8tol
export R8_LAUNCH_TREE=$P2W/tree R8_REMAP_FILE=refcv8_eval_remap_p2.json R8_NO_POSE_SYNC=1
export PYTHONPATH="$P2W/tree/stack;$P2W/tree/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WIN=E:/refcv8_eval_kit/windows/s2_windows.json
CK8=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt;  CF8=E:/refcv8_eval_kit/ckpt/config.json
CK7=E:/refcv7_eval_kit/ckpt/ckpt_50400.pt;      CF7=E:/refcv7_eval_kit/ckpt/config.json
free_virt_gb() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)" | tr -d '\r' | tr ',' '.'; }
