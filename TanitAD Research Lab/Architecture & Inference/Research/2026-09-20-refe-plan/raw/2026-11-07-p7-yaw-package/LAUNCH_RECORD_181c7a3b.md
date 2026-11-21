# P7 (with the yaw-rate input) restarted in wait-for-P6 mode (2026-11-07 03:30:06Z, Master Mind; PI go in chat 2026-11-06: "do 1 and 2")

- **Package:** `/workspace/refe-plan-p7`, a raw-blob export (`git -c core.autocrlf=false archive`) of commit `181c7a3b86d4`: `code/ refe/ splits/` plus the 9 ship-list files outside them (`eval/` x6, `raw/` x3).
  - `launch_gate.py ship-manifest --commit 181c7a3b... --repo-prefix "<PK>/"` gave `ZZSHIP_OK 288 files tree b1ebeb46ee0df753`. That is the same tree digest as the YP7 FINAL pre-run, so the shipped bytes are exactly the tested bytes.
  - Tar sha256 `55a11805...` equal on the dev box and the pod. `p7_pod.sh` md5 `a3003f28762d` on both; `bash -n` OK; 0 CR bytes.
- **Dev-box suite-prerun on the exported package:** `suite_prerun_LANDED.out`, PASS, 9/9 scripts, 0 skips, 1,437 s (profile refe_p7, tree b1ebeb46, commit 181c7a3b). This is a sanity check; the pod's own launch gate (stage 11) runs G-SUITE on the real argv and data.
  - The earlier FINAL pre-run FAILED only on `test_p6_levers.py`; reproduced as `OSError: [Errno 28] No space left on device` on the dev box's C: (0.65 GB free), not a code defect.
- **Ship-list check on the pod:** `ZZP7SHIP_OK files=296 scripts=106 closure=adf9d8f49e4a`, the same as on the dev box.
- **Replaced chain:** the previous waiting chain (`p7_pod.sh` md5 `ecbd4bfd73ee`, package from `58255f30`, started 2026-11-06 20:20:38Z) was stopped by explicit PID. It had written no stage marker. The lock read free before the restart. Its package and logs are kept on the pod as `*.old-ecbd4bfd-20261107`.
- **Start:** `P7_LABEL_WORKERS=13 P7_WAIT_P6=1 nohup setsid bash /workspace/refe-plan-p7/refe/p7_pod.sh`, with HF_TOKEN in the environment only. `chain.log`: `chain start (p7_pod.sh md5 a3003f28762d)`, then `ZZP7_WAITING_FOR_P6 ... (waited 0s)`.
- **After P6 releases its lock (in order):** stage 5b builds the navtrain yaw table from the pod's 1,192 navtrain DBs (refuses on any missing token), `check-bank` must read 100 % coverage, and the table sha256 enters the launch token's data manifest. Then comes the launch gate (stage 11), which includes G-LIVE's non-zero gradient on `ego_enc.0.weight[:, 4]`.
