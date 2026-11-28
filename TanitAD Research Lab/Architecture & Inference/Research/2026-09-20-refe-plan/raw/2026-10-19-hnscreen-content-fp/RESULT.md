# HNSCREEN-CACHE-CONTENT-FP -- the P6-HN screen's `--ctx-cache` is now bound by CONTENT (a per-file sha256 manifest), not by a size listing

Architecture FlyWheel agent (Sonnet), 2026-10-19, 0 GPU, CPU only, nothing written to D:, nothing staged/committed/pushed. Built on tip `d2f4a062783f3fd9757ef10cc3266a13b42d8815` (`agent/arch-inf-20260803`, `git fetch` + `git cat-file --batch`
export; `raw/.../TIP_SHA.txt`). Evidence class of every number: **MEASURED**, artifact under `raw/2026-10-19-hnscreen-content-fp/`.

**INTEGRATION NEEDED (headline).** (1) Land the 6 files in the manifest (4 edited: `refe/gate_token.py`, `refe/launch_gate.py`, `refe/ctx_cache.py`, `refe/hn_fingerprints.py`; `refe/test_launch_gate.py` edited; `refe/test_hn_content_fp.py` NEW).
`refe/launch_gate.py` is a hot file (other agents add profiles to it): my hunks are `HNSCREEN_CONTENT_FLAGS`, `content_flags_of`, three `content_flags=` keyword additions, the `hnscreen_refusals` loop and one `HNSCREEN_SUITE` entry; none touches `PROFILES` other than adding the suite entry through the constant.
(2) **The cache must be SEALED.** `ctx_cache.finalize` seals it; the real S2 cache driver is the run-box command `commands.s2_cache` (not in the repo): it must end in `ctx_cache.finalize(...)`, or the cache is sealed afterwards with `python refe/ctx_cache.py seal --dir <cache>`.
An unsealed cache is REFUSED by the gate (that is the fail-closed behaviour, not a bug). (3) **Cost:** every gate command re-reads the whole cache (about 70 GiB); CPU is never the limit (1.3 GB/s/thread), the DISK is: 17 full reads for the screen; 6-8 min on a fast local disk,
about 8 h at the 44 MB/s measured on D: today. See `COST.md`. Decide where the cache lives before queuing the screen. (4) The profile's `fingerprints_expected` is still `{}` (refuses by design until filled); the fill is now `hn_fingerprints.py` on the SEALED cache and prints a `dirc:` key.

## 1. The defect, reproduced through the gate's own entry point (before / after)

`raw/.../scripts/repro_same_size_flip.py` builds a real cache (real `build_units` + `finalize`), fills the profile with each tree's OWN fill procedure, flips ONE bit of one payload file (same size), and runs `launch_gate.py refusals`:

| tree | pristine cache | same-size 1-bit flip |
|---|---|---|
| base (tip) | rc 0 `ZZGATE-CLEAN` | **rc 0 `ZZGATE-CLEAN` -- the gate PASSES the flip** (`logs/repro_same_size_flip_base.json`) |
| fixed | rc 0 `ZZGATE-CLEAN` | **rc 5 `ZZGATE-REFUSED`**: "the content fingerprint of --ctx-cache cannot be taken: 1 problem(s): BYTES CHANGED at the same size (576): units/log0.ctx.u16 (sha256 manifest 04f3ac74a3ca, now 2182ce7c0d2e)" (`logs/repro_same_size_flip_mine.json`) |

## 2. Everywhere the fingerprint is computed or checked (file:line)

| where | base line | what it did / does now |
|---|---|---|
| `refe/gate_token.py` `fingerprint` / `fp_key` | 203 / 226 | UNCHANGED (file = sha256, dir = LISTING). Every other flag of every other profile keeps exactly this. |
| `refe/gate_token.py` `data_manifest`, `_timed_fingerprint` | 259 / 253 | new keyword `content_flags` (default `()` = identical behaviour); a flag in it is fingerprinted by `content_fingerprint` |
| `refe/gate_token.py` `verify_token` (the SUPERVISOR) | 386, data compare 429 | new keyword `content_flags`; the compare is unchanged; the reasons now name the file |
| `refe/launch_gate.py` `hnscreen_refusals`, the pin compare | 737 | `--ctx-cache` is fingerprinted by CONTENT (new lines 745-757); a `dir:` pin is refused as stale |
| `refe/launch_gate.py` `run_checks` / `finalize` (the MINT) | 942 / 1182 | pass `content_flags=content_flags_of(prof)` (new 959 / 1199) |
| `refe/launch_gate.py` `cmd_verify` (what `hnscreen_devbox.verify_pass_token` runs, line 549) | 1323 | passes the content flags (new 1341) |
| `refe/hn_fingerprints.py` (the FILL) | 36-37, 44, 68 | computes the cache's content key once and refuses an unsealed / tampered / empty cache (exit 2) |
| `refe/ctx_cache.py` `finalize` (the BUILD) | 238-250 | now ends with `seal(out_dir)` (new 254): sha256 of every file, ONCE |
| `refe/gate_checks.py` G-DVB `ctx_cache.manifest_sha256` | 1179 | UNCHANGED (binds `manifest.json`, which is now also a member of the content manifest) |
| `refe/hnscreen_devbox.py` `verify_pass_token` / `stage_S2` | 549 / 605 | UNCHANGED (it calls `launch_gate.py verify`; its DONE marker lists `manifest.json`, not the seal -- see section 7) |

## 3. The design (gate_token.py lines 250-440 in the fixed tree)

* **Build (once):** `seal_content_manifest(dir)` writes `content_manifest.json` = canonical key-sorted JSON (LF, indent 1), `{tool: content_manifest/1, algorithm: sha256, n_files, bytes, files: {relpath: {bytes, sha256}}}`, relative POSIX paths sorted. The manifest itself is not a member; re-sealing rewrites it.
* **Fingerprint:** `dirc:<n_files>:<bytes>:<sha256 of the manifest's canonical bytes>`, **recomputed from the LIVE files** (so a hand-edited manifest cannot vouch for itself); for a valid sealed cache it equals `sha256sum content_manifest.json` (asserted against an independent hashlib oracle).
* **Check (every gate command):** structural problems end the check BEFORE any byte is read (missing / unreadable / malformed / non-canonical manifest; a missing, extra, renamed or resized file; a symlinked directory; an unlistable directory; an empty cache). Otherwise every file is re-hashed (4 threads, `TANITAD_GATE_HASH_THREADS`) and a changed byte at the same size is a problem.
  Any problem makes the key `dir-content-INVALID:...`, which equals no registered key. No cache, no memo, no switch that skips it.
* **Two independent mechanisms refuse a tampered cache:** (i) the stored manifest vs the live files, (ii) the live digest vs the registered pin. The tests isolate each (section 4) so that neither hides the other.
* **Scope:** `content_flags_of(prof)` = `("--ctx-cache",)` iff the profile has `hn_rules`, else `()`; derived from `hn_rules`, not from a profile key that could be absent (a default that disables the check). Asserted: every other profile returns `()`, and `data_manifest` without `content_flags` is byte-identical to before.

## 4. RED arms -- every one runs a REAL entry point (`raw/.../logs/test_hn_content_fp_FINAL.log`: 62 arms green on the real tree, 0 red)

| variant of the pristine sealed cache (a real cache from `build_units` + `finalize`) | `launch_gate.py refusals`, registered pin | same, pin REFILLED from the tampered bytes (only the manifest check can refuse) | `launch_gate.py verify` on a token minted over the pristine cache, in process AND as the subprocess the chain runs |
|---|---|---|---|
| **same-size 1-bit flip** | REFUSED (BYTES CHANGED at the same size) | REFUSED | REFUSED (names `units/log0.ctx.u16`) |
| renamed file | REFUSED | REFUSED | REFUSED |
| added file | REFUSED | REFUSED | REFUSED |
| deleted file | REFUSED | REFUSED | REFUSED |
| **missing manifest** (registered bytes, unsealed) | REFUSED | REFUSED (nothing else could) | REFUSED |
| **stale manifest** (a unit grown after the seal) | REFUSED | REFUSED | REFUSED |
| non-canonical manifest (one extra space) | REFUSED | REFUSED | REFUSED |
| malformed manifest (`{not json`) | REFUSED | REFUSED | REFUSED |
| foreign manifest (another cache's, valid) | REFUSED | REFUSED | REFUSED |
| payload flipped AND manifest regenerated (self-consistent) | REFUSED (not the registered one) | n/a | n/a |

Also REFUSED: a stale `dir:` (size-listing) pin; a wrong `dirc:` pin on the sealed cache; the shipped (unfilled) profile. A symlinked directory and an empty directory are INVALID (unit arms).

**Clean controls (all PASS):** the sealed cache through `refusals` (rc 0 `ZZGATE-CLEAN`); `launch_gate.finalize` mints a PASS token whose `--ctx-cache` data is `kind: dir-content` and whose key equals the independent oracle; `launch_gate.py verify` MATCHES (in process and as a subprocess);
verify MATCHES again once the tampered cache is restored byte for byte. `run_checks` on the flipped cache fails every check and records the `dir-content` fingerprint with its problem in the evidence (`inputs_read`); `finalize` on it is a FAIL token naming the file.

## 5. Mutation matrix (19 source mutants, each a fresh copy of the tree with ONE edit, each must turn the named checks RED; all 19 did)

| source edit | RED checks (count) |
|---|---|
| `content_check`: **the per-file byte comparison deleted** (the task's mutation) | 4: incl. `G_refused_flip_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse` -- the gate PASSES that same-size flip (rc 0) |
| `hnscreen_refusals`: the comparison with the registered pin deleted | 2: the resealed flip and a wrong `dirc:` pin PASS |
| `hnscreen_refusals`: **both** content comparisons deleted | 5: the same-size flip, the add, the delete and the resealed flip all PASS |
| `verify_token`: the data fingerprint comparison deleted (supervisor trusts the token) | 4: the flip, add, delete, missing-manifest all MATCH |
| `cmd_verify`: content flags not passed | 1: the CLEAN control breaks (fail CLOSED: a content-minted token no longer matches a listing; the flip is never silently passed) |
| `finalize` AND `cmd_verify` both drop the content flags (the wiring class "built, tested, unreachable from its caller") | 2: the supervisor PASSES the same-size flip |
| `finalize` drops the content flags / `run_checks` drops them | 2 / 1 |
| `content_flags_of` returns `()` | 2 |
| `hnscreen_refusals` fingerprints `--ctx-cache` by listing again (the defect) | 1 |
| `content_check`: missing manifest silent / extra file / missing file / resized file / canonical-form check / symlink check each dropped | 3 / 2 / 2 / 2 / 2 / 1 |
| `fp_key`: INVALID keyed like a valid one | 1 |
| `ctx_cache.finalize` does not seal (the builder) | 1 |
| `hn_fingerprints` describes the cache by listing | 1 |

**Honest reading of the first rows.** Deleting ONE of the two comparisons leaves the flip refused by the other (defence in depth), so on its own the pinned-flip arm stays red-for-the-right-reason only through its message; the matrix therefore PAIRS each mechanism with an arm that isolates it
(the "pin refilled from the tampered bytes" arm for the manifest comparison, the "resealed" arm for the pin comparison). Deleting the manifest comparison alone makes the isolating arm see a PASS; deleting both makes every flip arm see a PASS.

## 6. Tests, base vs mine (all `CUDA_VISIBLE_DEVICES=-1`, CPU; the base tree is the tip's bytes)

| file | base | mine |
|---|---|---|
| `test_launch_gate.py hnscreen_static()` (non-aborting harness) | 278 assertions, 0 failing | **298**, 0 failing (+20: HNCFP-0..8 arms, same-size flip / rename / add / delete / no-manifest / stale / foreign / resealed / stale pin / CLI rc 5) |
| `test_launch_gate.py --static --no-banked` (the G-SUITE form) | `ZZLG_STATIC_PASS 1147` | `ZZLG_STATIC_PASS 1167` |
| `test_ctx_cache.py` | `ZZTEST_PASS 15` | `ZZTEST_PASS 15` |
| `test_hn_screen_wiring.py` | `ZZTEST_PASS 120` | `ZZTEST_PASS 120` |
| `eval/test_hn_screen_read.py` (needs `taniteval/` + `raw/2026-10-17-zero-tail-mechanism`, exported into BOTH trees) | `ZZTEST_PASS 31` | `ZZTEST_PASS 31` |
| `test_dump_scores.py` / `test_sft4r_gate.py` / `test_carve_feed_gate.py` / `test_gate_gpu_probes.py` | 16 / 153 / 10 / 14 | 16 / 153 / 10 / 14 |
| `test_suite_nesting.py` | 110 | 111 (one more suite entry is enumerated) |
| `test_rl_s1b.py` / `test_rl_r0c.py` / `test_rl_pickcon.py` | 18 / 26 / 20 | 18 / 26 / 20 |
| `test_aux_g12_bf16.py` | rc 0, **1 SKIP** (no CUDA: the bf16 G12 probe did NOT run; not a pass of that probe) | identical |
| `test_gate_gpu_probes.py` | prints `SKIP_CUDA: the GPU check did NOT run` | identical |
| **`test_hn_content_fp.py` (NEW, also added to `HNSCREEN_SUITE` so the launch gate runs it)** | -- | `ZZTEST_PASS 81` = 62 arms + 19 RED mutations, 120 s |

**Regressions so far: 0.** The rest of the sibling list (every `test_*.py` that names an edited file, 28 files, plus `test_launch_gate.py` in its full form) is the table below, generated by `scripts/finish_pkg.py` from `logs/suite_{mine,base}.json`.

<!--SIBLINGS-->
| file | base rc | mine rc | base count | mine count | skips b/m | verdict |
|---|---|---|---|---|---|---|
| refe/test_launch_gate.py:--static,--no-banked | 0 | 0 | 1147 | 1167 | 0/0 | MORE (new checks) |
| refe/test_ctx_cache.py | 0 | 0 | 15 | 15 | 0/0 | SAME |
| refe/test_hn_screen_wiring.py | 0 | 0 | 120 | 120 | 0/0 | SAME |
| eval/test_hn_screen_read.py | 0 | 0 | 31 | 31 | 0/0 | SAME |
| refe/test_dump_scores.py | 0 | 0 | 16 | 16 | 0/0 | SAME |
| refe/test_hn_content_fp.py | - | 0 | - | 81 | -/0 | NEW FILE (mine only) |
| refe/test_sft4r_gate.py | 0 | 0 | 153 | 153 | 0/0 | SAME |
| refe/test_suite_nesting.py | 0 | 0 | 110 | 111 | 0/0 | MORE (new checks) |
| refe/test_carve_feed_gate.py | 0 | 0 | 10 | 10 | 0/0 | SAME |
| refe/test_gate_gpu_probes.py | 0 | 0 | 14 | 14 | 0/0 | SAME |
| refe/test_aux_g12_bf16.py | 0 | 0 | - | - | 1/1 | SAME |
| refe/test_p2r_chain.py | 0 | 0 | 139 | 139 | 0/0 | SAME |
| refe/test_rl_s1b.py | 0 | 0 | 18 | 18 | 0/0 | SAME |
| refe/test_rl_r0c.py | 0 | 0 | 26 | 26 | 0/0 | SAME |
| refe/test_rl_pickcon.py | 0 | 0 | 20 | 20 | 0/0 | SAME |
| refe/test_rls1_chain.py | 0 | 0 | 522 | 522 | 0/0 | SAME |
| refe/test_s1aux_chain.py | 0 | 0 | 256 | 256 | 1/1 | SAME |
| refe/test_sft4r_chain.py | 0 | 0 | 31 | 31 | 0/0 | SAME |
| refe/test_rls1e_chain.py | 0 | 0 | 936 | 936 | 0/0 | SAME |
| refe/test_hnscreen_chain.py | 0 | 0 | 16 | 16 | 0/0 | SAME |
| refe/test_launch_gate.py | 0 | 0 | 1192 | 1212 | 0/0 | MORE (new checks) |
| refe/test_s1c.py | 0 | 0 | 220 | 220 | 2/2 | SAME |
| refe/test_rl_safeobj.py | 0 | 0 | 61 | 61 | 0/0 | SAME |
| refe/test_rls1_queue.py | 1 | 0 | - | 254 | 0/0 | base FAILS, mine passes |
| refe/test_p2rep_chain.py | 0 | 0 | 353 | 353 | 0/0 | SAME |
| refe/test_p6_chain.py | 0 | 0 | 351 | 351 | 0/0 | SAME |
| refe/test_p6_levers.py | 0 | 0 | - | - | 0/0 | SAME |
| refe/test_aux_s1_wiring.py | 0 | 0 | 20 | 20 | 0/0 | SAME |
| eval/test_s1_read.py | 0 | 0 | 100 | 100 | 0/0 | SAME |
| refe/test_gate_suite_prerun.py | 0 | 0 | 144 | 144 | 0/0 | SAME |

NEW FAILURES: none; LOWER COUNTS: none; PENDING: none
<!--/SIBLINGS-->

Environment notes for the sibling runs (identical for both trees, so they cannot explain a difference): my first launch used a WMI bare `PATH` and the bash-driven chain tests failed on BOTH trees at once; the runner was restarted with Git's `usr/bin` on `PATH` and those entries re-run.
Layout files the tests read from `raw/` and `taniteval/` were exported into BOTH trees (`raw/rl_pool`, `raw/rl_fan`, `raw/2026-10-13-aux-s1-data-audit`, `raw/2026-10-13-aux-head`, `raw/2026-10-04-sft4-smoke`, `raw/2026-10-17-zero-tail-mechanism`, `taniteval/adapters`, `taniteval/taniteval`).
`test_s1aux_chain.py` prints one `ZZTEST_SKIP` (sandbox_runs: the navtest list under `FlyWheels/` is absent in this layout), on base and mine alike.

## 7. Things the Master Mind must know

1. **Where the cache is built decides what the gate costs** (`COST.md`): 17 full reads; disk-bound; 44 MB/s on D: today vs minutes on a fast local disk (C: NVMe cold rate UNMEASURED). Two cheap levers exist and were deliberately not pulled (a per-process memo: 5 reads per seed -> 3).
2. **Fill procedure (updated):** build bank + cache (`finalize` seals) -> `python refe/hn_fingerprints.py --hn-bank <bank> --ctx-cache <cache>` (one more full read; refuses an unsealed or tampered cache, exit 2) -> paste the JSON into `PROFILES["refe_hnscreen"]["fingerprints_expected"]` (the `--ctx-cache` value is `dirc:...`) -> commit -> gate.
   Until then the profile REFUSES (unchanged). A pre-existing listing fill (`dir:...`) is now refused as stale.
3. **`hnscreen_devbox.py` is unchanged on purpose** (it only calls `launch_gate.py verify`; the chain's own test stubs the gate). Consequences: the T<s> mint REFUSES an unsealed cache with "no content_manifest.json" (correct); the chain's S2 DONE marker still lists `manifest.json`, not `content_manifest.json`
   (optional follow-up: list it too, which needs the chain test's stub driver to write it).
4. **`build_units` does not delete a stale seal** when it rebuilds a unit in a finalised cache. The gate refuses such a cache anyway (BYTES CHANGED / SIZE changed, which is a correct but not self-explanatory message); an extra `os.remove` before a rebuild would make the message "unsealed". Not done (scope).
5. **Every other profile and every other directory flag is untouched** (asserted: `content_flags_of` is `()` for them; `data_manifest` without `content_flags` equals the old listing; `test_sft4r_gate.py`'s "listing-only" pin still holds).
6. The G-SUITE of `refe_hnscreen` now includes `test_hn_content_fp.py` (+120 s).
7. **UNVERIFIED:** the fixed gate on the REAL 70 GiB cache (it does not exist yet); C: NVMe cold throughput; the hash-thread default (4) on the cache's real drive (a USB SSD may prefer 1-2 threads).

## 8. Proposed register text (NOT applied -- register files untouched)

* **BACKLOG** `HNSCREEN-CACHE-CONTENT-FP` -> **DONE (pending landing)**: "`--ctx-cache` is fingerprinted by a per-file sha256 content manifest sealed at build; the gate re-hashes every file; same-size flip / rename / add / delete / missing / stale manifest REFUSED through `launch_gate.py refusals` and `verify`; 62 arms + 19 RED mutations (`raw/2026-10-19-hnscreen-content-fp/`). Remaining: `HNSCREEN-FP-FILL` (needs the built bank + sealed cache) and the cache-drive decision (COST.md)."
* **GAC** note for the P6-HN screen row: the earlier caveat "the `--ctx-cache` fingerprint is a size listing" is CLOSED in code.
* Retraction-log class: *a documented limit that stayed documented* -- the listing's blindness to a same-size flip was written into a docstring and a test pin (`test_sft4r_gate.py:278`) weeks ago as "by design" and was only measured against the gate's entry point on 2026-10-19.
  Trap-log class: *two sufficient mechanisms hide each other's mutant* -- with both a manifest check and a pin compare, deleting either line leaves the flip refused; isolate each with an arm that disables the other (pin refilled from the tampered bytes; manifest regenerated).

## Manifest (also `paths.json`; everything is in ONE place: `C:/Users/Admin/par18/hnfp2/`; nothing on a pod, nothing in git)

Code (real repo paths under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`), staged copies under `out/refe/`:
see `raw/.../code_vs_tip.diff` for the exact diff vs the tip. sha256 (LF) of the six files is printed by `make_pkg.py` and recorded in `HANDOFF.md`.
Evidence under `.../raw/2026-10-19-hnscreen-content-fp/`: `RESULT.md`, `HANDOFF.md`, `COST.md`, `TIP_SHA.txt`, `code_vs_tip.diff`, `scripts/`, `logs/`.
