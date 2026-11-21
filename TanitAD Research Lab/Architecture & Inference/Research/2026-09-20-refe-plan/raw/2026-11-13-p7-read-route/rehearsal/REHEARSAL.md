# P7 read route: families-stage rehearsal on real data (P6 vs u400). PASS end to end in 35 min

Master Mind, 2026-11-13. This closes risk 1 of `../P7_READ_ROUTE.md`: until now, the families stage had run only against a fake runner.

## What ran
- **Route:** `eval/p7_read_chain.py`, from a tip export of `7f7ef7c7`:
  - `PK/eval`, `refe`, `raw/2026-10-05-sft-stage2`, `raw/p2_r2_instrument/baseline`, `raw/p2_readout/p2_full`, `raw/2026-11-12-instruments`, `stack/tanitad`;
  - `taniteval` without `results/`.
- **Arguments:**
  - `--ckpt` = P6 `model_final.pt` (sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`, MEASURED today);
  - `--name p6rh --base u400 --also p2 --only families`.
  - The arm's seam is a copy of the banked `raw/p2_r2_instrument/baseline/p6_navtest_full_seam.npz`, saved as `seams/stage2_p6rh.npz` in a scratch data dir with `navtest_full_tokens.json` (md5 `3a8b3653`).
- **Launch:** `launch_p6rh.cmd`, via WMI, CPU only.
- **Preflight:** the strict real preflight passed all 7 checks (ckpt sha, yaw table, tokens, 21 harness pins, `baseline_u400`, `baseline_p2`, navsim root).
- **Steps** (`CHAIN.log`):
  - `plans8`: 12,146 tokens each;
  - `transp_score`: 2 shards over both arms in one pass, harness MetricCache route, rc 0, 18:01 to 18:33;
  - `families_run --per-window` for each arm: rc 0;
  - `fam_ci` paired: rc 0.
  - `ZZP6RH_EXIT 0` at 18:35 local.
- **Wall time: 35 min on 2 processes.** `P7_READ_ROUTE.md` ESTIMATED 5.7 h from `INSTRUMENTS.md`. The difference is the `--metric-cache-root` route.

## Checks (MEASURED)
- **Tracked PDMS from the families rows equals the pinned stage-3 harness means to 4 decimals.**
  - P6: **91.1965**. Pin `BASELINES["p6"]` 91.1965; `level_ci.json` 91.19648.
  - u400: **88.2469**. Pin 88.2469.
  - Δ = +2.9496, against stage 3's +2.9495 [+2.1872, +3.7669].
  - Transparent frame: P6 91.3877, u400 88.4745.
- **`fam_ci` output** (`fam_ci_U400_vs_P6RH.json`):
  - estimator: paired log-cluster bootstrap over 136 logs, 12,146 tokens;
  - it answers whether another draw of logs would give the same result;
  - 61 metrics per frame.
  - The `reference` frame equals `executed_transparent` on every metric shown. This is expected by construction: the transparent frame executes the plan exactly.

## A descriptive note (not a claim; both arms were read before)
On the executed-tracked frame, P6 − u400 is separated on 49 of 61 metrics.

| family | metric | P6 − u400 |
|---|---|---|
| LONGITUDINAL | speed bias | **+0.485 m/s** [+0.451, +0.522]; P6 is +0.291, u400 −0.194 |
| LONGITUDINAL | speed MAE | +0.113 [+0.075, +0.151] |
| LONGITUDINAL | headway min | −0.29 m [−0.43, −0.13] |
| LONGITUDINAL | min TTC | −0.48 s [−0.66, −0.28] |
| LATERAL | heading MAE | +0.21 deg [+0.11, +0.31] |

So P6's +2.95 PDMS over u400 comes with faster-than-human driving, closer headway and lower TTC. **The P7 read should report this longitudinal family beside PDMS, never PDMS alone.**

## For the P7 read
- The families stage is now verified on real data.
- What remains first-use for P7 is unchanged: the GPU seam (stage-2 attempt 1, then 2), stage 3 against `--base p6`, and adherence.
- `families` against `--base p6` uses the same code path exercised here, with `p6` as the base.

## Not banked
- The per-token rows (`rows_s*.jsonl`, 57 MB), `fam_*.json` (100 MB) and `plans8` npz.
- All of these can be re-created in 35 min from this commit's code plus the banked seams.
