#!/bin/bash
echo "ZZ-NVSMI"; nvidia-smi 2>&1 | head -30
echo "ZZ-TEGRA"; timeout 3 tegrastats --interval 1000 2>&1 | head -2
echo "ZZ-PY"; ps -eo pid,etime,pcpu,rss,args | awk '$5 ~ /python|torch|sup_|supervise|bash.*\.sh/' | grep -v awk | cut -c1-220
echo "ZZ-DISK"; df -h /home/nvidia 2>&1 | tail -1
echo "ZZ-MEM"; grep -E "MemTotal|MemAvailable" /proc/meminfo
echo "ZZ-DATA"; ls /home/nvidia/data 2>&1 | head -80
echo "ZZ-FIND"; timeout 60 find /home/nvidia -maxdepth 4 \( -iname "*navsim*" -o -iname "*openscene*" -o -iname "*navtrain*" -o -iname "*navtest*" -o -iname "*nuplan*" \) 2>/dev/null | head -40
echo "ZZ-VENVS"; ls ~/venvs; ~/venvs/tanitad-edge/bin/python -c "import torch,sys;print(sys.version.split()[0],torch.__version__,torch.cuda.is_available())" 2>&1 | tail -1; ~/venvs/tanitad-train/bin/python -c "import torch,sys;print(sys.version.split()[0],torch.__version__,torch.cuda.is_available())" 2>&1 | tail -1
echo "ZZ-END"
