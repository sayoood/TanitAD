---
name: fleet-status
description: One-shot live status of the whole TanitAD compute fleet (pod1 training, pod2 arms, local, workflows)
---

Produce a compact live fleet status. Run these checks (parallel where possible) and present ONE table:

1. **pod1 (training):** `ssh tanitad-pod 'grep -o "\"step\": [0-9]*" /workspace/experiments/p0-sB01-realmix.log | tail -1; pgrep -fc "train_worldmode[l]"; echo WD=$(pgrep -fc "stall_watchdo[g]"); tail -1 /workspace/experiments/watchdog.log 2>/dev/null'` — report step/30000, trainer count (must be 1), watchdog alive, last watchdog action. Compute pace only if asked (takes 3+ min).
2. **pod2 (arms):** `ssh tanitad-pod2 'grep -o "\"step\": [0-9]*" /workspace/experiments/arm_base.log | tail -1; grep -o "\"step\": [0-9]*" /workspace/experiments/arm_kstep.log | tail -1; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'` — both arm steps + GPU.
3. **Local:** CARLA process state (`Get-Process CarlaUE4* -ErrorAction SilentlyContinue`), 4060 free.
4. **Background:** TaskList for running workflows/monitors; CronList for the drumbeat + report crons (flag if missing — they expire after 7 days).

Close with: anything anomalous (duplicate trainers, stalled steps vs last known, dead watchdog, missing crons) flagged in bold, and the single next scheduled milestone.
