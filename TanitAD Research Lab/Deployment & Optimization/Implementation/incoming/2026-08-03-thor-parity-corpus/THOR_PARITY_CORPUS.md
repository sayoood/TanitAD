# Can Thor train on the PARITY corpus? — capacity, transfer, and a guard-passing REF-C step

**Date** 2026-08-03 · **Device** `tanitad-thor` (thor6, aarch64, L4T R38 / JetPack 7, CUDA 13.0)
**Arm** REF-C-base (`refc-diffusion-base`, **104,191,577** params) · **Trainer** `stack/scripts/refc_train.py`
**Venv** `~/venvs/tanitad-train` (two-venv rule honoured; `tanitad-edge` never invoked)
**Follows** `…/incoming/2026-08-03-thor-training-benchmark/` — which proved Thor is fast enough and
left open whether it can hold the corpus at all.

---

## 0. The answer, first

**YES — Thor can hold the full parity training corpus, with 2.8× headroom, and the corpus exists
in a verifiable form. The blocker was never Thor's disk; it was that nobody knew where the raw
corpus still lived.**

| question | answer | class |
|---|---|---|
| How big is the parity train corpus REF-C actually reads? | **278,782,982,625 B = 278.78 GB** (2,376 `ep_*.pt` + 24 `skip_*` + `DONE`) | **MEASURED** (§1) |
| Can Thor hold it? | **Yes.** 845.4 GB free; **192.8 GB written with `O_DIRECT`** in one test and 605 GB still free | **MEASURED** (§2) |
| Does any reachable host still have it? | ⛔ **No.** pod1 / pod3 / eval are **`Connection refused`**; pod4 and `tanitad-new` hold **no raw epcache** | **MEASURED** (§3) |
| So where is it? | **HuggingFace** — `Sayood/tanitad-physicalai-w120-256x640cyl` → `epcache-256px-phase0/physicalai-train-e438721ae894` | **MEASURED** (§3) |
| Is the HF copy the CANONICAL corpus? | **Yes — strict parity PASSES.** uid `sha256 9877bef6…7386` **matches the committed manifest**, 2376/2376, all 24 skip indices match | **MEASURED** (§4) |
| Did parity survive the move? | **Yes so far.** **208/208** transferred episodes match the source by **size AND `sha256`**, `torch.load` OK, guard PASSES | **MEASURED** (§5) |
| Is the full corpus on Thor yet? | ⛔ **NOT YET.** **35.2 of 278.78 GB (300/2376 episodes)** at 2026-08-03T20:06Z; the pull is **still running** at ~23 MB/s → **ETA ≈ 2.9 h** | **MEASURED** (§5) |
| Does REF-C really train on it, guard PASSING? | **Yes** — 104,191,577 params, 25,966 windows, real optimizer steps, on genuine PhysicalAI parity episodes | **MEASURED** (§6) |
| Was any guard weakened to get that? | ⛔ **No.** The same command in **strict** mode **REFUSES** with `PARITY VIOLATION … TRUNCATED by 2224`, exit 1 | **MEASURED** (§6.2) |

### What I did NOT do — stated plainly

1. ⛔ **I did not complete the transfer.** 35.2 GB of 278.78 GB (300/2376 episodes) had landed at
   2026-08-03T20:06Z; the background pull continues. **There is therefore NO strict-parity training
   step on Thor yet** — §6 is a **subset-mode** run over the first 152/208 episodes and is *not*
   cross-arm comparable.
2. ⛔ **I did not verify the other 254 GB.** The verifier is written, shipped and proven on 208
   episodes; §5.3 is the one command that finishes the job.
3. ⛔ **I did not touch either live pod** beyond read-only `ls` / `nvidia-smi` / `find`. Both were at
   GPU 100 %.
4. ⛔ **I did not record the val uid digest** into `parity_manifest.json`, though I now have it —
   see §7.2 for why that is an escalation and not an edit.
5. ⛔ **I did not train anything decision-grade.** Every loss number here exists to prove the
   optimizer ran. The BINDING four-metric-family rule governs **evals**; nothing here is an eval.

---

## 1. WHICH cache, and how big — read from source, not guessed

`refc_train.py:696` calls `refb_train.load_cached_episodes(args.data_root, "*train*", args.episodes)`,
which resolves the newest `*train*` dir and reads **raw epcache `ep_*.pt`** (`refb_train.py:233-250`).
⚠️ **REF-C has no v2 path at all** — that matters in §3.

Size, derived two independent ways that agree to **0.006 %**:

| route | value |
|---|---|
| **A. from the committed manifest.** `sum_T_out = 472,627` frames (`parity_manifest.json` → `cross_checks`), × `9·256·256` uint8 = 589,824 B/frame | **278,766,747,648 B** |
| **B. from the actual file listing** (HF tree API, §4) | **278,782,982,592 B** (episodes) |
| difference | **+16,234,944 B** = **6,832.9 B/episode** of poses/actions/maneuvers/zip overhead |

*(The per-frame constant is itself MEASURED, not assumed: Thor's `ep_00000.pt` is 117,383,256 B and
`torch.load` reports `frames_u8 [199, 9, 256, 256]` → 199 × 589,824 = 117,374,976, leaving 8,280 B.)*

---

## 2. Thor's disk — a real `dd` test, never `df` alone

⛔ `df` is not evidence here (the MooseFS lesson), and **Thor's memory probes lie in both
directions**, so the same scepticism was applied to disk.

```
avail_before (df --output=avail -B1)     845,406,420,992 B   (= 787.4 GiB; `df -h` says 788G)
dd bs=1M count=70000 oflag=direct  #1     73,400,320,000 B    24.20 s   3.0 GB/s
dd bs=1M count=70000 oflag=direct  #2     73,400,320,000 B   290.54 s   253 MB/s
dd #3 (killed by explicit PID)            45,464,158,208 B
--------------------------------------------------------------------------
simultaneously present on disk           192,264,798,208 B  = 192.3 GB
alongside the in-flight corpus pull         8,500,000,000 B  (approx)
df at that moment                          284G used / 605G avail
```

**Verdict: MEASURED capacity is sufficient.** 278.78 GB against 845.4 GB free leaves ~567 GB.
⚠️ **I stopped the test at 192.3 GB rather than the full 280 GB** — it was competing with the corpus
transfer for the same NVMe, and the transfer is the better proof anyway (real bytes, verified).
So the capacity claim is *192.3 GB written and verified present*, plus *567 GB of `df` headroom*,
not *280 GB written*. Stated that way on purpose.

⚠️ **Second-order finding:** sustained write drops **3.0 GB/s → 253 MB/s** after ~73 GB. That is the
SLC-cache cliff and it means **a 278 GB write is a ~20-minute disk operation at best**, not a
90-second one. Any future "rebuild the cache on Thor" plan must budget for the sustained rate.

---

## 3. Where the corpus actually lives — absence at one location is not absence

| host | probe | result |
|---|---|---|
| `tanitad-pod` (pod1) | ssh | ⛔ `Connection refused` |
| `tanitad-pod3` | ssh | ⛔ `Connection refused` |
| `tanitad-eval` | ssh | ⛔ `Connection refused` |
| `tanitad-pod4` | `ls /workspace/data/*/_epcache/*/` | **empty** (GPU 100 %, A40, training) |
| `tanitad-new` (v5f) | `ls`, **plus** `find /workspace -maxdepth 5 -type d -name 'physicalai-train-e438721ae894*'`, **plus** `find -name ep_00000.pt` | **no raw epcache.** Only the **v2 wide** siblings: `physicalai-train-…-w120-256x640cyl` 80 GB / 2400 `.v2ep.pt`, and the val sibling 20 GB / 600 |
| **HuggingFace** | tree API | ⭐ **`Sayood/tanitad-physicalai-w120-256x640cyl` carries the RAW epcache** under `epcache-256px-phase0/` |

⚠️ **Three probes before the absence claim** (path, name, and the tool that owns the fact), per the
standing rule. The first `ls` alone would have concluded "the corpus is gone".

⚠️ **The v2 cache is NOT a substitute.** `refc_train` reads `ep_*.pt`; a v2 cache is a flat set of
`<clip_id>.v2ep.pt` in a different uid space (`parity.py` §9). Pointing REF-C at one is a
`no ep_*.pt files in this directory` refusal, not a training run.

**HF repo inventory** (`raw/hf_census_*.json`):

| path in repo | files | bytes | note |
|---|---|---|---|
| `epcache-256px-phase0/physicalai-train-e438721ae894` | 2401 | **278.78 GB** | ⭐ the raw parity TRAIN corpus |
| `epcache-256px-phase0/physicalai-val-0c5f7dac3b11` | 601 | **70.39 GB** | the full 600-episode clean val build |
| `physicalai-train-e438721ae894-w120-256x640cyl` | 2403 | 85.00 GB | v2 wide sibling |
| `physicalai-val-0c5f7dac3b11-w120-256x640cyl` | 603 | 21.21 GB | v2 wide val sibling |

---

## 4. ⭐ Parity verified BEFORE moving a byte

`parity.check_uids` operates on the **set of filenames**, which the HF tree API returns for free.
So the parity verdict costs ~2 s and the 3-hour download only starts if it PASSES —
`code/hf_parity_census.py`, output `raw/hf_census_train.json`:

```
corpus_key                physicalai-train-e438721ae894
mode                      strict
episodes_loaded           2376  /  2376 expected
episode_uid_sha256        9877bef64da35f384b380b23ab0e760f3ef5396c6f3e849d5de81c7243ac7386
expected                  9877bef64da35f384b380b23ab0e760f3ef5396c6f3e849d5de81c7243ac7386   ← MATCH
content_check             sha256(sorted uids) MATCHES the committed manifest
skip_indices              24 observed == 24 expected (1798 … 1941)   MATCH
skip-hash                 f09e44db
byte cross-check          within 0.006 % of 472,627 frames × 589,824 B
```

⚠️ **What this does not prove**, said out loud because the digest invites over-reading: `parity.py`
states the digest "does NOT hash episode CONTENT (tensor bytes)". A file of the right name and right
size can still be corrupt — Thor had a live example (§7.1). Byte integrity is a **destination-side**
check, §5.

---

## 5. The move — and proving parity survived it

### 5.1 The bus and the real rate

HF, as instructed. ⚠️ **MEASURED 23 MB/s down to Thor**, over a 60-second uncontended window
(`delta 1,407,398,684 B / 60 s`) with `snapshot_download(max_workers=8)`.

⚠️ **This is 4× slower than the 93 MB/s the brief quotes.** That figure is INHERITED and was
measured elsewhere; on this link, into this box, the honest number is 23 MB/s ⇒
**278.78 GB ≈ 3.4 h**. Anyone planning around 93 MB/s will be wrong by 2.5 hours.

### 5.2 What has actually landed, and how it was checked

`code/verify_epcache_bytes.py` checks **four** things and is explicit that no one of them suffices:
the **name set** (parity), the **size** (catches truncation), the **sha256** (catches corruption a
size cannot see), and a real **`torch.load`** (the only check that runs the code path training will).
Expected sizes and digests come from the **source's own LFS metadata**
(`code/mint_hf_expected.py` → `raw/hf_expected_train.json`), so this compares the destination
against the source's record of itself, not against a number someone typed.

```
cache               /home/nvidia/epcache_prefix/physicalai-train-e438721ae894
parity_verdict      PASS  (subset: 208 of 2376, shortfall printed LOUD)
n_episodes_on_disk  208          bytes_on_disk  24,396,457,072
size_checked        208          size_mismatches      []
sha256_checked      208          sha256_mismatches    []      (22.0 s → 1.11 GB/s)
load_failures       []           VERDICT  PASS
```

⛔ **Exit codes are not evidence** — every line above is a counted, printed fact. Silent truncation
with exit 0 has bitten this programme three times in one day.

### 5.3 The ONE command that finishes the job

When `n_ep == 2376` under `~/epcache/epcache-256px-phase0/physicalai-train-e438721ae894`:

```bash
ssh -n tanitad-thor 'cd ~/parity_verify && PYTHONPATH=$HOME/TanitAD/stack \
  ~/venvs/tanitad-train/bin/python verify_epcache_bytes.py \
    --cache ~/epcache/epcache-256px-phase0/physicalai-train-e438721ae894 \
    --expected hf_expected_train.json --mode strict --sha256 all --load 8 \
    --out verify_train_full.json'
```
At the MEASURED 1.11 GB/s hash rate the full 278.78 GB sweep takes **≈ 251 s**. `--mode strict` is
the point: it demands 2376/2376 and the exact manifest digest.

---

## 6. A real REF-C training step on parity data, with the guard PASSING

### 6.1 Preconditions — the stack was synced and PROVEN, not assumed

⚠️ Thor's `~/TanitAD` sat at `4954544` while the dev box was at `7e18d68` — **a launch from it would
have run stale code.** `git log` is not proof, so the check was an import:

```
refc_train        /home/nvidia/TanitAD/stack/scripts/refc_train.py
sha256            6636d95e56108749cc05424a7cc7b81d5640049d817dca45437daca1f8c953cc
   dev box        6636d95e56108749cc05424a7cc7b81d5640049d817dca45437daca1f8c953cc   ← IDENTICAL
manifest digest   9877bef64da35f384b380b23ab0e760f3ef5396c6f3e849d5de81c7243ac7386
torch 2.13.0+cu130   cuda True
```

### 6.2 ⭐ RED FIRST — the guard really refuses

Same script, same data, **`--episodes 0` (strict)**, `EXIT=1`:

```
PARITY VIOLATION [*train* cache] — corpus physicalai-train-e438721ae894
  cache      : /home/nvidia/epcache_prefix/physicalai-train-e438721ae894
  episodes   : 152 loaded, 2376 expected   <-- TRUNCATED by 2224
  missing    : ep_00152.pt, ep_00153.pt, … (+2218 more)
```

⛔ **Nothing was weakened, bypassed or `--force`d.** `parity.py` deliberately offers no environment
variable that disables the check, and none was invented. **The refusal is a result, and it is the
thing that makes the PASS below mean something.**

### 6.3 GREEN — the subset run

`--episodes 152` puts the loader in **subset mode**, the only non-full episode set the guard admits:
a sorted **PREFIX** of the manifest list. It is a self-labelling truncation of the canonical corpus,
never a re-selection — one foreign, renumbered or out-of-order episode and it refuses.

```
[parity] ⚠ *train* cache: physicalai-train-e438721ae894 SUBSET — 152 of 2376 episodes
         (2224 absent). … NOT strict parity and must not be cross-compared with full-corpus arms.
[refc] train: 152 episodes / 25966 windows  (mode=diffusion)
{"step": 0, "loss": 13.66377, "traj": 3.79683, "cls": 8.28656, "law": 2.6919, …}
{"step":19, "loss": 13.63405, "traj": 3.35399, …}
n_params_trainable 104,191,577      ← byte-identical to the A40 run's recorded count
```

⛔ **Those losses are NOT results.** 20 steps at lr ≈ 1e-6 during warmup; they prove the optimizer
ran end-to-end on real PhysicalAI parity frames, and nothing else.

### 6.4 Throughput and the ONLY admissible memory number

⛔ On Thor's unified memory, `mem_get_info` / `free` / `tegrastats` / `VmRSS` all lie. Only in-process
`torch.cuda.max_memory_allocated()` is quotable, which is why `run_refc_on_parity.py` calls
`refc_train.main()` **in-process** instead of shelling out.

| batch | p50 step_s (n=20, steps 4-23) | min/max | windows/s | **max_memory_allocated** | max_memory_reserved |
|---|---|---|---|---|---|
| 8 | **0.65 s** | 0.6 / 0.7 | 12.31 | **7.38 GB** | 10.65 GB |
| 20 | **1.50 s** | 1.5 / 1.5 | **13.33** | **16.47 GB** | 18.16 GB |

This **confirms** the prior benchmark independently: its "saturates at ~13 windows/s" and its
1.500 s at batch 20 both reproduce **exactly**, on different data, a different day, a re-synced stack.

⚠️ **A single-sample step time is not admissible on this box.** My *first* batch-8 run reported a
steady **1.5 s/step** across steps 1-8; the p50-over-20 measurement 20 minutes later gave
**0.65 s**. A **2.3× discrepancy I did not explain** (cold page cache and the concurrent download are
the candidates; neither was isolated). The windowed p50 is the quotable number; the single read was
not, and I nearly quoted it. Same family as the `step_s`-is-accumulated false alarm.

---

## 7. Two defects repaired, one escalated

### 7.1 The truncated val episode — REPAIRED

`~/valdata/physicalai-val-0c5f7dac3b11/ep_00028.pt` was **92,299,264 B** against a true
**117,383,256 B** — 21.4 % of the episode missing, in a directory every listing called healthy.

Repaired from the HF source and verified three ways (**not** by exit code):
`size 117,383,256` ✓ · `sha256 e9a5e8ed…525d` ✓ (the source's own LFS digest) ·
`torch.load → ['frames_u8','actions','poses','episode_id','maneuvers']` ✓ · atomic `os.replace`.

**Then the whole deployment was swept**, because one bad file means the set was never checked:

```
cache /home/nvidia/valdata/physicalai-val-0c5f7dac3b11   parity PASS (subset 40/600)
40 episodes / 4,697,689,792 B · size_mismatches [] · sha256_checked 40, mismatches [] · VERDICT PASS
```

⭐ **Thor's val-40 is now byte-verified against source for the first time.** Every Thor eval before
this ran against a 21 %-truncated episode 28 and nothing could have told you.

### 7.2 ⬆️ ESCALATION — the val uid digest is recordable and I did not record it

`parity_manifest.json` carries **no `episode_uid_sha256` for `physicalai-val-0c5f7dac3b11`**
(`uid_source: count-only-unrecorded`), so **every val check in the fleet is COUNT-ONLY** — a
substituted val episode at the right count passes today.

I now have the digest over the full 600-episode HF build: **`75a4d429be8cef8ea47a319e2033d792ee9eecbff033fad27dbb624b5634df20`**
(`raw/hf_census_val.json`; the 40-episode Thor deployment digests to `c36c2168…828a`).

⛔ **I did not write it into the manifest**, on purpose. The manifest's own TODO requires a cache
that `compute_skipset.py` has verified; my source is an HF mirror I did not build, and for the train
split the mirror could be checked against a committed digest whereas for val — by definition — it
cannot. Recording an unverifiable digest is how a wrong invariant becomes load-bearing.
**Decision needed:** either (a) run `make_parity_manifest.py --record --split val` on a host holding
a `compute_skipset`-verified 600-episode val cache, or (b) accept the HF mirror as the source of
record *because its train sibling reproduced the committed digest exactly* and record it with that
provenance stated. **(b) is cheap and defensible; it is a PI/registry call, not mine.**

### 7.3 `train_flagship_v4.py` — a preflight that named a remedy that did not exist

`preflight_asserts` refused a `--v2-train-cache` run without `--require-parity` and told the operator
to *"record why this arm is deliberately non-parity"* — **through a flag the parser did not have**.
Following the instruction produced `error: unrecognized arguments`. A legitimately non-parity v2 arm
(toy episodes, the 9,000-clip corpus `4b7eeeac222d`, an OOD probe) was **unrunnable through this
trainer**.

⚠️ Same class as `PREFLIGHT: OK` covering an input it never looked at: the guard's **output** and the
guard's **behaviour** disagree, and the message is the thing that misleads.

**Implemented** (`stack/scripts/train_flagship_v4.py`), mirroring `--heldout-off-reason` exactly:

- `--parity-off-reason <why>` — declared in `NOT_A_PATH`, **required instead of** `--require-parity`.
  ⛔ Deliberately **not** a bare `--force` boolean: a boolean records that someone wanted past the
  guard, a reason records **why**, and only the second survives into `config.json` as evidence.
- The two flags are **mutually exclusive** — a command asserting both "enforce parity" and "this arm
  is deliberately non-parity" would write a false provenance record.

**Three latent defects in the precedent itself surfaced while mirroring it, and are fixed for BOTH flags:**

| defect | evidence | fix |
|---|---|---|
| whitespace unlocks the guard | `--heldout-off-reason '   '` passed a bare truthiness test — a boolean in a string's clothes | `_off_reason()` strips |
| the reason did **not** survive `_staged_command` | absent since the day it was added; the copied command trips its own preflight and the run never starts | both flags emitted, `shlex.quote`d |
| the reason was **never echoed**, despite the help text saying "printed at launch" | `heldout_off_reason` reached exactly 3 code sites: parser, `NOT_A_PATH`, the preflight requiring it — its only surface was the `args` blob in `config.json` | `_off_reason_banner()`, printed on `--print-launch` **and** on the real run |

**Tests:** `stack/tests/test_v4_off_reasons.py` — **17 tests, all passing**, RED-first, incl. a
round-trip (reconstruct → re-parse → re-check) so a dropped argument is impossible rather than
unlikely. ⭐ **The whitespace hole was found BY the test, not by inspection** — the test was written
to assert a property I assumed held, and it did not.

**Full suite:** `cd stack && pytest -q` → **1932 passed, 12 skipped, 2 xfailed** in 389 s
(was 1900/12/2; +17 mine, the rest from sibling streams already in the tree — the repo advanced
from `59d2097` to `7e18d68` during this session).

---

## 8. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `THOR_PARITY_CORPUS.md` (this file) | `repo:TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-08-03-thor-parity-corpus/` | no |
| `code/hf_parity_census.py` | same, `code/` | no |
| `code/mint_hf_expected.py` | same, `code/` | no |
| `code/verify_epcache_bytes.py` | repo + `thor:~/parity_verify/` | no |
| `code/make_parity_prefix.py` | repo + `thor:~/parity_verify/` | no |
| `code/run_refc_on_parity.py` | repo + `thor:~/parity_verify/` | no |
| `code/watch_and_verify.sh` — **RUNNING on Thor** (PID 28167): waits for 2376 episodes, then runs the STRICT verify unattended | repo + `thor:~/parity_verify/`, log `thor:/tmp/watch_verify.log` | no |
| `raw/hf_census_train.json`, `hf_census_val.json` | repo, `raw/` | no |
| `raw/hf_expected_train.json` (2401 size+sha256), `hf_expected_val.json` (601) | repo `raw/` + `thor:~/parity_verify/` | no |
| `raw/verify_val40.json`, `verify_train_prefix.json`, `refc_*.log`, `run_record.json` | repo `raw/` + `thor:/tmp/`, `thor:~/parity_verify/` | no |
| `--parity-off-reason` + banner + staged-command fix | `repo:stack/scripts/train_flagship_v4.py` | no |
| 17 regression tests | `repo:stack/tests/test_v4_off_reasons.py` | no |
| REPAIRED `ep_00028.pt` | `thor:~/valdata/physicalai-val-0c5f7dac3b11/` | ⚠️ **yes — it is a data file; the authoritative copy is the HF source it was restored from** |
| the transferred epcache (in flight) | `thor:~/epcache/epcache-256px-phase0/physicalai-train-e438721ae894/` | ⚠️ **yes on Thor — but the HF source is authoritative and verified** |
| synced stack | `thor:~/TanitAD/stack` | no (= dev-box `7e18d68`, proven by sha256) |

**Staged, never pushed. Nothing committed.**

## 9. Escalations (these need a decision, not a doc)

1. ⬆️ **The 243 GB still in flight — but it verifies ITSELF.** The pull is `nohup`'d on Thor
   (PID 22414, `/tmp/pull_parity.log`) and `code/watch_and_verify.sh` is running beside it
   (PID 28167, `/tmp/watch_verify.log`): it polls for 2376 episodes and then runs the **STRICT**
   verification unattended, writing `~/parity_verify/verify_train_full.json`.
   ⇒ **Nobody has to remember to check.** Read `/tmp/watch_verify.log` for the verdict.
   **If Thor reboots, re-launch BOTH** — `snapshot_download` resumes from what is on disk.
2. ⬆️ **The val uid digest** — §7.2, (a) or (b).
3. ⬆️ **pod1 / pod3 / eval are all `Connection refused`.** The precedent is that a RunPod volume
   resize stops the pod and reassigns its SSH port. **If those pods are gone, HF is now the ONLY
   copy of the raw parity corpus** — a single point of failure for the programme's most sacred asset.
4. ⬆️ **The 93 MB/s HF figure in circulation is 4× the rate this link delivers** (§5.1). Any plan
   sized on it is wrong by hours.
