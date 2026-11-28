# ⛔ `mm_e19_read_step30000.json` POINTS AT LOGS THAT NO LONGER HOLD THIS ARM

**Written 2026-09-01.** Read this before quoting anything from this directory.

`mm_e19_read_step30000.json` (arm `k60clip05p30k`, ckpt md5 `7e3c776d19a6…`, step 30000)
records each probe stage as `{cmd, rc, seconds, log}` — **it stores no numbers**. The numbers
lived only in the files its `log` fields point to:

```
C:\Users\Admin\tanitad-wt\mm-e19-probes\mm_e19_out\raw\latentmotion_*.log
C:\Users\Admin\tanitad-wt\mm-e19-probes\mm_e19_out\raw\actdiv_local.log
```

⛔ **Those paths were OVERWRITTEN on 2026-09-01 at 22:09–22:42 UTC+2** by the MM-E19 **k=8**
control read, which reuses the same fixed output directory. They now contain **k=8 data under a
`k60clip05p30k` label** (the runner's `--arm-name` defaulted to this arm's name — see
`…/2026-09-01-mm-e19-k8-attribution/RESULT.md` §6). Following those pointers today yields the
wrong arm with no warning.

## What is still true here

* the arm identity and **ckpt md5 `7e3c776d19a6…`** in the JSON are correct;
* the checkpoint itself survives on Thor at `/home/nvidia/v7tiny/k60clip05p30k/ckpt.pt`
  and locally at `C:\Users\Admin\mm-e19-pull\ckpt.pt` — **verify by md5 before use**;
* `RUNBOOK.md` in the parent directory is unaffected.

## The general lesson

**Banking a JSON that references external logs by path is not banking the raw data.** A record
whose evidence lives outside the repo is one overwrite away from being uncitable — and it fails
silently, because the JSON still parses and still looks complete. Bank the bytes, not a path to
them.

Both defects that produced this are fixed in `mm-e19-probes/mm_e19_read.py`: `--arm-name` is
now required, and installing a checkpoint over a **different** one under the same arm name is
refused unless `--replace-arm` is passed.
