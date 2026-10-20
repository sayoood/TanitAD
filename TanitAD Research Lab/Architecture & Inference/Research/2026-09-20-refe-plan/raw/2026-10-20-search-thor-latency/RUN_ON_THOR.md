# RUN_ON_THOR: SEARCH-THOR-LATENCY (the Master Mind runs this; the agent that wrote it has not touched Thor)

Written 2026-10-20 (~04:40 Berlin) against tip `6a0cfae` (`MM_RULING_C2AMEND.md` DQ3 / DQ4, `AMENDMENT_C2_DRAFT.md` section 7). Instrument: `thor_search_latency.py` (sha256 `cf5cdfd1686adfd4f591ba851d2eeb7b99d48dba146aa7c30f0a8aa796f5729d`).
Everything below is **UNRUN on Thor**. The only run so far is 5 tokens on the dev-box CPU (`raw/devbox5/`), which proves the code path on the real checkpoint, ctx and fan; it is **not a Thor number**.

## 0. What it measures, in four lines
1. Per token, batch 1, one search seed (20260927), P2-s0 scorer modules only, banked post-trunk `visual_ctx` of carve-B tokens: **(a)** the scorer-only pick (1 set-scoring + aggregate + argmax), **(b)** Step B's CEM search with C8 (14 scorer calls, 896 rows), **(c)** the same with O4 (38 calls, 3,392 rows).
2. Output: p50 / p95 / p99 ms per arm, the ratio to (a), the added ms over (a), the scorer-call and row counts, a scorer-time vs host-numpy split. **No pass / fail** (DQ4): until an absolute budget exists there is no deployability claim.
3. **(a) is narrow on purpose**: the trunk (frames to ctx) and the proposal generator are not in the banked ctx, so they are not timed. Quote `added_ms_vs_a` first; `ratio_to_a` is "search over the scorer-only pick", which is LARGER than the ratio over a full forward.
4. The labels (`sub`, `pdms`, `valid` in the bundle) are never read; the code raises `LabelAccess` if asked.

## 1. What to copy to Thor (flat layout `/home/nvidia/thorlat/`), about 2.87 GB
| item | dev-box source | bytes | sha256 (the script refuses a different one) | Thor destination |
|---|---|---|---|---|
| checkpoint P2-s0 | `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt` | 1,288,484,600 | `ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51` | `ckpt/model_final.pt` |
| carve-B ctx | `D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle/ctx.npy` | 1,572,864,128 | `c8c287cf6bdb6b1b91785d6251b1027897cc16d11d13857b524f1731a23bd03b` | `data/ctx.npy` |
| carve-B bundle (fan, pick, banked logits; also holds labels, never read) | same dir, `bundle.npz` | 8,309,154 | `580b6cc395aa9074c75eb7a62fe99e8b493abd9ea5671be1a5765d8ae4fd673f` | `data/bundle.npz` |
| `thor_search_latency.py` (this package) | `raw/2026-10-20-search-thor-latency/` | 52 KB | see `MANIFEST.sha256` (and the header line above) | `code/` |
| sealed Step B: `stepB.py`, `l2_scorer.py` | `raw/2026-10-19-l2-stepB/code/` (sha256 `74e60d3d...`, `8584bb1e...`) | 24 KB, 8 KB | pinned in the script | `code/` |
| L1: `stepA.py`, `tts_cem.py`, `real_scorer.py`, `label_perturbations.py` | `Research/2026-10-18-ep-rl-lit/l1_stepA/` (`05d182c7...`, `9dea9603...`, `05c4b4fb...`, `9e2caca1...`) | 30, 6, 5, 6 KB | pinned in Step B | `code/` |
| `refe/model.py` | `Research/2026-09-20-refe-plan/refe/model.py` | 55,372 | `e6011cd83c846dfbd9756301ae731f4b84541b7b8e07a139a2c2c2b663a8a46d` | `refe/model.py` |
| configs C8 / O4 | `raw/2026-10-19-l2-stepB/raw/stepB_config.json`, `stepB_config_O4.json` | 276, 278 | `299c6e63...c441`, `1ebd7196...9013` | `cfg/` |

All code files in ONE directory (`code/`): the script finds Step B and L1 next to itself (override: `STEPB_CODE`, `L1_CODE`). Code and config shas are verified at start (`verify_stepb_files`, `stepB.verify_l1_code`, `validate_config`); the checkpoint, ctx and bundle are hashed in full (about 2.9 GB read once; not timed on Thor).
Copy (dev box, Git Bash; `scp` reads `D:` as a host name, so `cd` first):
```
ssh -n tanitad-thor-wifi 'mkdir -p /home/nvidia/thorlat/{code,ckpt,data,refe,cfg,env,out}'
cd /d/Projects/TanitAD/data/refe_runs_eval/p2 && scp model_final.pt tanitad-thor-wifi:/home/nvidia/thorlat/ckpt/
cd /d/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle && scp ctx.npy bundle.npz tanitad-thor-wifi:/home/nvidia/thorlat/data/
# then the code files, refe/model.py and the two configs from a checkout of the tip into the matching dirs
```
Then ON THOR: `cd /home/nvidia/thorlat && sha256sum ckpt/model_final.pt data/* cfg/*` and compare with the table (CRLF warning: Step B hashes code with CRLF folded to LF; copy from a Linux checkout or `git show`, and the script will tell you if it differs).
Needs about 4 GB free in `/home/nvidia` (Thor free disk is UNVERIFIED: the 2026-10-04 cleanup inventory says it was tight). Python: `/home/nvidia/venvs/tanitad-train/bin/python` (path INHERITED from `thor_launch_arm.sh`, 2026-09; UNVERIFIED now).
Import smoke (UNVERIFIED that `refe/model.py` and this torch build import on aarch64): `python -c "import torch,numpy;print(torch.__version__,torch.cuda.is_available())"`.

## 2. The busy-test: it refuses to start if Thor is in use (exit 3)
Run it alone first (it changes nothing; `preflight` tests the GPU flock and releases it at once, `run` holds it):
```
cd /home/nvidia/thorlat/code && hostname        # expected thor6 (INHERITED from cleanup_thor_A.sh, 2026-10-04); pass what it really prints
/home/nvidia/venvs/tanitad-train/bin/python thor_search_latency.py preflight --target thor --expect-hostname thor6
```
It reads FOUR things and all must say idle; an unreadable instrument is a failed check, never a pass:
1. **the GPU flock** `/home/nvidia/refcv7_post/thor_gpu.lock` (held means a job owns the GPU; this is the lock `cleanup_thor_A.sh` obeys). A free lock is HELD by the run until it ends; an absent file is recorded, not read as idle.
2. **the process table** (`ps -eo pid,ppid,user,etimes,args`, parsed in-process, never `pgrep -f`): any command line matching `tools/fleet_probe.py`'s `JOB_RE` (`train*.py`, `scripts/train`, `vlm_*.py`, `taniteval.runner`, `refa/refb/refc_train`) or a supervisor (`supervise_run`, `sup_*.sh`, `torchrun`), plus any other `python` of the invoking user (the `cleanup_thor_A.sh` rule). Itself and its ancestors are excluded; root's system python daemons are not counted.
3. **GPU utilisation** (`nvidia-smi --query-gpu=utilization.gpu`, 5 samples, 1 s apart): any sample >= 5 % refuses (the amendment's "< 5 % before start"); `[N/A]` or no output refuses.
4. **`nvidia-smi --query-compute-apps`**: any listed app refuses (unavailable is recorded only).
**Not used, on purpose: `free`, `tegrastats` memory, `torch.cuda.mem_get_info`, VmRSS.** All three lie on Thor in both directions (`TRAPS.md`, "On the Jetson Thor, THREE standard memory probes all lie").
A `ZZTHORLAT_STOP THOR_BUSY_OR_UNVERIFIED` names the reasons. **There is no override flag and none is wanted:** if a trainer is live, wait for a gap; this script never pauses or kills anything. At the end of the run the process table is read again: a job that appeared DURING the run makes `no_foreign_job_at_end` RED.

## 3. Record the power mode and clocks (read-only), then run
`nvpmodel -q` works as the user; `jetson_clocks --show` normally needs root. The script tries live first, and falls back to captures made within the last 30 minutes in `--env-dir`:
```
sudo nvpmodel -q          > /home/nvidia/thorlat/env/nvpmodel_q.txt
sudo jetson_clocks --show > /home/nvidia/thorlat/env/jetson_clocks_show.txt     # --show ONLY
```
**Never run bare `jetson_clocks`: it SETS the clocks.** The script's subprocess allowlist refuses it too (and `nvpmodel -m`, `free`, `tegrastats`). Do NOT change the power mode or the clocks for this measurement: the number is "Thor as configured" and the JSON names the configuration (E-THOR-CLK precedent: mode 1 "120W", GPU 1386 MHz, EMC 3200 or 4266 MHz). Missing `nvpmodel_q`, `jetson_clocks_show`, `devfreq` or `thermal` makes the run REFUSE (exit 4, `REFUSED_DEVICE_FIELDS.json`) before any timing.

Smoke (3 tokens, about 1 min, same code path), then the two real runs. **Detach the real runs** (the budget rule: no agent waits on a long job); read the log file, not the terminal (`ssh -n`, TRAPS "polling monitor"):
```
cd /home/nvidia/thorlat/code && export PY=/home/nvidia/venvs/tanitad-train/bin/python
COMMON="--target thor --expect-hostname thor6 --env-dir /home/nvidia/thorlat/env --bundle ../data/bundle.npz --ctx ../data/ctx.npy --ckpt ../ckpt/model_final.pt --refe-dir ../refe --c8 ../cfg/stepB_config.json --o4 ../cfg/stepB_config_O4.json"
$PY thor_search_latency.py run $COMMON --autocast bf16 --n-tokens 3 --warmup 1 --repeats 1 --out-dir ../out/smoke
setsid nohup sh -c "$PY -u thor_search_latency.py run $COMMON --autocast bf16 --out-dir ../out/bf16 > ../out/run_bf16.log 2>&1; $PY -u thor_search_latency.py run $COMMON --autocast none --out-dir ../out/fp32 > ../out/run_fp32.log 2>&1" < /dev/null > /dev/null 2>&1 &
# done when:  grep -E '^ZZTHORLAT_(CONTROLS|STOP)' /home/nvidia/thorlat/out/run_*.log
```
`--autocast bf16` is the trainer's numerics (the ctx is banked as bf16); `--autocast none` is fp32. Run both: the amendment does not say which one the vehicle runs, and they differ. Defaults: 200 tokens, 30 warm-up, 3 repeats, `--threads 4`.
Exit codes: 0 all controls green; 2 usage; **3 Thor busy**; **4 device fields missing**; 5 a pin or the carve refused; **6 a control is RED (the JSON is still written; the numbers are not quotable)**.

## 4. Expected runtime (ESTIMATED from the dev-box CPU, NOT a Thor number)
MEASURED on the dev-box CPU, 5 tokens, fp32, 4 threads, 1 repeat, 1 warm-up token (`raw/devbox5/`): (a) 61.1 ms, (b) 870.3 ms, (c) 2,703.3 ms per token at p50, i.e. 3.63 s for the three arms; of that, the host numpy part is 0.1 / 55.5 / 214.5 ms. (A first run on the previous code revision read 58.5 / 797.5 / 2,402.9 ms: about 10 % run-to-run on this box at n = 5.)
The protocol runs each token 1,030 times over all arms (2 x 200 reference runs, 30 warm-up, 3 x 200 timed) plus a small fixed tail (no-op floor, scaling, n=0 control):
- **if Thor's GPU were no faster than this CPU: about 62 min per mode** (1,030 x 3.63 s). That is the ceiling I can state; the Thor GPU is expected to be faster on the scorer (UNVERIFIED), so the scorer share should shrink;
- **the part that cannot shrink with the GPU is the host numpy**: 1,030 x 0.27 s = about 4.6 min on this x86 CPU, and the ARM host may be slower (factor UNMEASURED, ASSUMED 1 to 3, i.e. 5 to 14 min).
- Budget one slot of **60 to 65 min per mode, about 2 h for both**, plus about 15 min of copying and hashing. At CPU speed one mode already fills the amendment's "about 1 Thor-hour"; both modes fit in about 1 h only if the GPU is roughly twice as fast as this CPU (UNMEASURED).

## 5. How to read the result (`out/<mode>/THOR_SEARCH_LATENCY_thor_<mode>_<utc>.json`)
- First `controls_ok` and `is_thor_number`. RED controls: do not quote. `identity_*` = the timed path returned exactly the untimed reference output on every sample; `determinism_*` = the device repeats itself (if this is red the device is nondeterministic and identity is unattributable); `call_counts_*` = 1 / 14 / 38 calls and 64 / 896 / 3,392 rows; `timer_floor`, `sees_work` (p50 a < b < c), `n0_pick`.
- Then `arms.{a,b,c}.ms` (mean, p50, p95, p99, min, max over repeats x tokens; `per_repeat_ms` shows stability), `arms.{b,c}.added_ms_vs_a`, `ratio_to_a`, and `scorer_calls_ms` vs `host_rest_ms`.
- Informative only: `informative.noop_scorer_search_ms` (the search around a zero scorer = the host cost), `per_call_ms_by_population` and `row_scaling_visible_256_vs_64` (a launch-bound GPU is flat in M: then O4's cost is its call count, 38 vs 14), `scorer_parity_vs_banked_logits` (the order of magnitude of the device difference; no tolerance is gated), `pick_agreement_with_banked_pick`.
- Wording (DQ3 / DQ4): "MEASURED (ours, Thor, nvpmodel mode X, autocast Y, one search seed, 200 carve-B tokens, post-trunk ctx): the search adds Z ms per token at p50 (P ms p99) over the scorer-only pick"; **no deployability claim until a budget exists.** A dev-box record says `devbox-cpu` and `is_thor_number false`.

## 6. Where this differs from `AMENDMENT_C2_DRAFT.md` section 7 (stated, not hidden)
1. **(a) is the scorer-only pick, not the full-forward pick with frames.** The banked ctx has no frames and the 322 M trunk is not loaded; a full-forward baseline needs the frames of the 200 tokens (not banked with carve B) and is a separate measurement.
2. **Both C8 and O4 are timed** (the amendment's protocol named C8; the MM ruling's ADDITION makes the C2 config a mechanical choice of C8 or O4).
3. **Per-stage split** is wall time per scorer call (device-synchronised, includes the set H2D and logits D2H, because the closure ends in `.cpu()`), versus the rest of the token (host numpy); no CUDA events, which would need the closure opened.
4. **"Doubling M must raise the per-scoring time"** is reported (`row_scaling_visible_256_vs_64`), not gated: on a launch-bound GPU it may legitimately be flat. The gated control is that the timer sees the call-count work (a < b < c).
5. **The parity check** uses the first 3 carve-B tokens' banked logits, not the 3 bridge tokens; informative only (cost-only; Thor results are not used for any endpoint).
6. Warm-up is the first 30 of the 200 timed tokens (no extra ctx to copy); the untimed reference phase has already run every token twice per arm before the warm-up.

## 7. UNVERIFIED (named)
Thor free disk; the venv path; that `refe/model.py` and this torch build import on aarch64 sm_110; that `nvidia-smi --query-gpu=utilization.gpu` prints a number (the 2026-09-03 note says it does: `97 %`; if it prints `[N/A]` the preflight refuses and nothing here is wrong); that `sudo` is non-interactive; the format of `jetson_clocks --show` (the `gpu_at_max` parse is best-effort; the raw text is the record); the hostname `thor6`; the ARM host-numpy factor.
