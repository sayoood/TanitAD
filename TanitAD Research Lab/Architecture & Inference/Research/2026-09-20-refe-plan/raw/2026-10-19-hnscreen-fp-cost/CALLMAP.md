# CALLMAP -- every caller that re-verifies the ctx cache inside one HN screen chain

Base: mirror tip `5ad79d09eec9dc806bace47642f559f83ce5f656` (`agent/arch-inf-20260803`, pinned once; `TIP_SHA.txt`). Line numbers are of that tip, in
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/` (REFE below). Evidence class: **MEASURED by reading source** (grep + read of every hop);
the per-screen read COUNT is separately MEASURED by instrumenting the reader (section 4, `logs/reads_per_screen.json`).

A "full read" = `gate_token._sha256_and_size` on every file of the cache (`content_check` -> `_hash_files`, gate_token.py:328-345). It is the only primitive that reads cache bytes in a gate check.

## 1. The hops (`file:line`), in chain order

| # | stage / caller | hop (each line is a call) | what reads the cache | full reads |
|---|---|---|---|---|
| 0 | **S2 build** (`commands.s2_cache`, run-box command) | `ctx_cache.finalize` ctx_cache.py:241 -> `seal(out_dir)` :254 -> `seal` :258 -> `GT.seal_content_manifest` gate_token.py:404 -> `_hash_files` :328 | the SEAL: sha256 of every file, once | **1** |
| 0' | **the FILL** (outside the chain, before the pin is committed) | `hn_fingerprints.fingerprints_expected` hn_fingerprints.py:64 (the CLI `main` :69 calls `_keyed` :78) -> `_keyed` :45 -> `describe` :40 -> `GT.content_fingerprint` gate_token.py:379 -> `content_check` :345 | every file re-hashed against the seal | **1** |
| 1 | **T<s> `launch_gate.py run`**, part 1 `run_checks` | `Chain.stage_T` hnscreen_devbox.py:607 -> `self.exec(cmd, "T<s>_gate_run")` :627 -> `launch_gate.cmd_run` launch_gate.py:1278 -> `run_checks` :1294 -> (a) `GT.data_manifest(..., content_flags=...)` :959 -> `_timed_fingerprint` gate_token.py:454 -> `content_fingerprint` :379 | the token's data fingerprint of `--ctx-cache` | **1** |
| 2 | (same call) `run_checks` -> `profile_refusals` :964 -> `hnscreen_refusals` :474 -> `GT.content_fingerprint(vals[0][0])` :750 | the registered-pin compare | **1** |
| 3 | **T<s> `launch_gate.py run`**, part 2 `finalize` (the MINT; `cmd_run` calls it right after `run_checks`) | `cmd_run` :1298 -> `finalize` :1191 -> `GT.data_manifest` :1199 -> `content_fingerprint` | the token's data fingerprint, again | **1** |
| 4 | (same call) `finalize` -> `profile_refusals` :1202 -> `hnscreen_refusals` -> :750 | the pin compare, again | **1** |
| 5 | **T<s> mint verify** | `Chain.stage_T` :631 -> `verify_pass_token(seed, where="mint")` :549 -> `launch_gate.py verify` -> `cmd_verify` launch_gate.py:1336 -> `GT.verify_token` :1340 -> `data_manifest` gate_token.py:630 -> `content_fingerprint` | the supervisor re-measures the token's data | **1** |
| 6 | **P<s> start verify** (immediately before the trainer starts) | `Chain.stage_P` :634 -> `verify_pass_token(seed, where="start")` :642 -> same chain of hops as #5 | the supervisor re-measures again | **1** |

**Per seed: 6 full reads** (#1-#6). Three seeds: 18. Plus the seal (#0) and the fill (#0'): **20 per 3-seed screen**.

### The number in the ruling is 17; the call map says 20. Why they differ (SETTLED by the measurement in section 4: it is 20)
hnfp2's COST.md counts "mint = run 2 + finalize 2, then verify 1 = 5 per seed -> 15 + seal + fill = 17". That table lists `launch_gate.py run` as **2** reads (it is `run_checks` only) and
`verify` as **1** per seed. The CLI `run` is `run_checks` **and** `finalize` (launch_gate.py:1294 + :1298 = 4 reads), and the chain verifies the token **twice** per seed (at the mint, stage_T:631, and at the start of the pass,
stage_P:642 = 2 reads). So the structural count is 6 per seed, 20 per screen, not 17. The (b') arithmetic (4 reads per screen) is unaffected; the saving is 20 -> 4, not 17 -> 4.

## 2. Probed and NOT a re-verifier (absence checked at two paths each: grep over `refe/ eval/ code/`, and a read of the owning function)

| candidate | finding |
|---|---|
| the trainer (`scorer_finetune.py:1318` `CC.CtxCache(a.ctx_cache, expect_trunk_fingerprint=...)`) | `verify=False` (default, ctx_cache.py:266); reads `manifest.json` + the small per-unit sidecars; `CtxCache.verify()` (:280, a full unit re-hash) is called only by `eval/test_hn_screen_read.py:563` (a test), never by the chain |
| G15 (`scorer_finetune.g15_ctx_cache_identity`, :1075) | a bounded sample of scenes (G15's `min_scenes`), not a full pass |
| `gate_checks` G-DVB (`gate_checks.py:1174-1185`) | `_file_sha256(<cache>/manifest.json)` only (one small file), never the units |
| `Marks.is_done` :271 / `Marks.write` :280 for S2 (`hnscreen_devbox.py` `class Marks` :254; `stage_S2` :590 returns `[manifest.json, I6, G15]`) | re-hashes `manifest.json` only: "the cache is bound by its own manifest" |
| `hn_screen_read.py`, `hn_screen_families.py`, `hn_screen_offsets_score.py` (stages S4) | never open the cache |
| `ctx_cache.py check` (:332) | a manual CLI, not called by the chain |
| `launch_gate.py refusals` (`cmd_refusals` :1359) | 1 read, but it is a human/test entry point, not a chain stage |

## 3. Where (b') cuts (the edit surface, i.e. the smallest change)

| hop | change |
|---|---|
| #0, #0' | unchanged (the seal and the fill keep their full reads) |
| NEW: **START anchor** (after S2, before T<first>) | one `launch_gate.py anchor`: 1 full read + the stat manifest; the chain's content anchor |
| #1-#6 | `--ctx-mode anchor-bound --anchor <file> --chain-id <id> --chain-start-ns <ns>`: stat compare of every file `(relpath, size, mtime_ns)`, **0 bytes read** |
| NEW: **END verify** (after the last P<s>, before S4) | one `launch_gate.py end-verify`: 1 full read vs the anchor + a final stat compare; writes the signed `END_VERIFY_OK` / `VOID` record |
| NEW: reader gate | `hn_screen_read.py` refuses unless every pass token is content-bound, or anchor-bound with an `END_VERIFY_OK` bound to the same anchor |

Total: seal 1 + fill 1 + start anchor 1 + end verify 1 = **4**.

## 4. MEASURED read counts (not read off the source: counted by instrumenting `gate_token._sha256_and_size`)

Instrument: `scripts/measure_reads.py` (raw `logs/reads_per_screen.json`, log `logs/measure_reads_run1.log`). A real sealed cache built by `ctx_cache.finalize` (**10 files**, so one full pass = 10 reads), the REAL
`launch_gate.py` (anchor / run / verify / end-verify) run as a subprocess per call, the seal (`ctx_cache.seal`) and the fill (`hn_fingerprints.fingerprints_expected`) counted in-process. Stub trainer / stage tools; the cache-touching lines of
`run_checks` kept. **Estimator question this answers: a deterministic COUNT of file reads on one synthetic cache - not a timing, not a statistic; there is no sampling variance to interval.**

| screen | today (content mode, the control) | (b') anchor-bound | saving |
|---|---|---|---|
| 1 seed | seal 10 + fill 10 + run 40 + verify 10 + verify 10 = 80 reads = **8 passes** | anchor 10 + end 10 + seal 10 + fill 10 = 40 = **4 passes** | 2.0x |
| 2 seeds | 140 = **14 passes** | 40 = **4 passes** | 3.5x |
| **3 seeds (the real screen)** | **200 = 20 passes** (3 x run 40 + 6 x verify 10 + seal 10 + fill 10) | **40 = 4 passes** (anchor 10 + end 10 + seal 10 + fill 10; all 11 intermediate gate calls read **0**) | **5.0x** |

Per-call reads, anchored 3-seed: `anchor 10, run 0, verify 0, run 0, verify 0, run 0, verify 0, verify 0, verify 0, verify 0, end-verify 10`. The (b') cost is **constant in the number of seeds**; today's grows 6 passes per seed.
**The ruling's and hnfp2's 17 is an undercount (INHERITED, not re-verified by them): measured 20.** The saving is 20 -> 4 (5x), not 17 -> 4.

## 5. The edit surface as landed (mine; every other hop above is untouched)

| hop | file | edit |
|---|---|---|
| anchor, end verify, stat compare, reader gate | `refe/gate_token.py` | new block after `seal_content_manifest`: `mint_anchor`, `load_anchor`, `anchor_fingerprint`, `end_verify`, `ctx_chain_refusals`; `fp_key` knows `dir-anchor`; `data_manifest` / `verify_token` take `ctx_anchor=` (None = today) |
| profile refusal + run/finalize + CLI | `refe/launch_gate.py` | `hnscreen_refusals` uses the anchor when the profile carries one; new subcommands `anchor`, `end-verify`; `--ctx-mode` (default content) / `--anchor` / `--chain-id` / `--chain-start-ns` on `run`, `verify`, `refusals` |
| the chain | `refe/hnscreen_devbox.py` | stages ANCHOR (after S2) and ENDV (after the last P<s>); `T`/`verify` get `ctx_args()`; the reader command gets `--chain-root` + the key file; config `gate.ctx_mode` (default `anchor-bound`) |
| the read | `eval/hn_screen_read.py` | `--chain-root` (required) + `--key-file`; the gate runs before any score file is opened (rc 6, `ZZHNSCREEN_REFUSED`) |
