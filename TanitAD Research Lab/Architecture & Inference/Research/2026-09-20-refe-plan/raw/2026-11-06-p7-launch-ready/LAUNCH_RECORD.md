# P7 chain started in wait-for-P6 mode (2026-11-06 20:20:38Z, Master Mind; PI go in chat: "start P7 in wait mode")

- **Package:** `/workspace/refe-plan-p7`, exported raw-blob from commit `58255f30cf15` (remote tip `939176e7` touches no package path: `git diff --stat 58255f30 939176e7` over `refe/ code/ splits/ eval/` is empty).
  - Contents: `code/ refe/ splits/`, plus the 8 listed files outside them.
  - `SHIP_MANIFEST.json` made by `launch_gate.py ship-manifest --commit 58255f30... --git-dir <mirror> --repo-prefix "<PK>/"` gave `ZZSHIP_OK 285 files tree e38704487de0`. The prefix needs its trailing `/`: without it every file reads "not in 58255f30".
- **Ship-list check:** `p7_ship_list.py check` gave `ZZP7SHIP_OK files=292 scripts=103 closure=fccec3323a4b` on the dev box AND on the pod. `p7_pod.sh` md5 `ecbd4bfd73ee` on both.
- **Aux label store:** fetched from HF `Sayood/tanitad-refe-navtrain-banks` into `/workspace/data/refe_navtrain/aux_label_store_p7`.
  - 353,802,240 B, `store_sha256` 4a17db67, 103,139 keys, 1,196 files.
- **Start:** `P7_LABEL_WORKERS=13 P7_WAIT_P6=1 nohup setsid bash /workspace/refe-plan-p7/refe/p7_pod.sh`, with HF_TOKEN in the env only.
  - `chain.log`: `chain start (p7_pod.sh md5 ecbd4bfd73ee)`, then `ZZP7_WAITING_FOR_P6 ... (waited 0s)`.
  - The chain proceeds only if P6's lock is released AND P6's chain.log has `ZZP6_DONE`.
- **Note:** tar extraction on the pod exits 2 on "Cannot change ownership" for a Windows-made tar; the files are complete. Use `--no-same-owner`, and do not chain `&&` after it.
