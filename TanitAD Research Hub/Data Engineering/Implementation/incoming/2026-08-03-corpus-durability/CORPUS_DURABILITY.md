# Corpus durability census — 2026-08-03

**Author:** Data/ops stream · **Branch:** `agent/arch-inf-20260803`
**Machine-readable:** `CORPUS_DURABILITY_CENSUS.json` (this dir)
**Instrument:** `tools/corpus_census.py` + `tools/tests/test_corpus_census.py` (34 tests)
**Registry sweep:** `tools/registry_paths.py` + `tools/tests/test_registry_paths.py` (33 tests)

---

## 0. The premise I was briefed with is RETRACTED

> *"The raw parity training corpus … was probed for and **not found on any live machine**."*

**FALSE. MEASURED 2026-08-03 21:5x local.** `tanitad-thor` was holding it the whole time:

| path | held |
|---|---|
| `thor:/home/nvidia/epcache/epcache-256px-phase0/physicalai-train-e438721ae894` | **438** `ep_*.pt` at first probe (51.40 GB) |
| `thor:/home/nvidia/epcache_prefix/physicalai-train-e438721ae894` | **208** `ep_*.pt` (24.40 GB) |
| `thor:/home/nvidia/valdata/physicalai-val-0c5f7dac3b11` | **40/40** val episodes (4.70 GB) |

**Root-cause class: absence found at ONE location reported as absence** — the same class as
the "our pods cannot render" claim that stood for 12 days. The earlier probe swept the ssh
config's pod aliases; Thor is in that config but its corpus lives under `~/epcache` and
`~/valdata`, not under any `/workspace/...` path a pod-shaped probe looks at.

**Second-order finding, and the reason this matters:** the retraction does *not* clear the risk.
Thor's train holding is **PARTIAL**, and a partial corpus is not a copy of the corpus. The
headline risk survives the retraction — but it had to be re-derived, not inherited.

⚠️ **A third fact I nearly mis-stated.** ICMP fails for **every** host including `pod4`, which
answers on TCP in the same second. **ICMP is blocked fleet-wide and is not evidence of anything.**

---

## 1. LEAD — what currently has exactly ONE copy

`copies` counts **distinct machines** (two paths on one host die together).
`durable` excludes **rented RunPod pods** — pod2 was terminated today, and pod1/pod3/eval went
to `Connection refused` this week. A pod is not storage.

| # | artifact | size | copies | durable | where | status |
|---|---|---|---|---|---|---|
| **1** | **raw parity TRAIN epcache 256 px** (2376 eps) | **278.78 GB** | **1** | **1** | HF only | 🟠 mitigation **in flight** |
| **2** | **raw parity VAL epcache 256 px** (600 eps) | **70.39 GB** | **1** | **1** | HF only | 🟠 mitigation **armed by me** |
| **3** | **9 checkpoints on `pod4`** (rescued from terminated pod2) | **~28 GB** | **1** | **0** | rented pod ONLY | 🔴 **needs PI decision** |

And two that pass the copy-count bar but should not be read as safe:

| artifact | size | copies | durable | why it is still exposed |
|---|---|---|---|---|
| `w120-train-cyl` (2400 files) | 85.00 GB | 2 | **1** | HF + **pod5 (rented)** — one termination from single-copy |
| `w120-val-cyl` (600 files) | 21.21 GB | 2 | **1** | HF + **pod5 (rented)** — same |

### 1.1 The nine pod-only checkpoints

`pod4:/workspace/rescue/experiments/` — rescued off pod2 before its termination, and pod4 is
itself a rented A40. **No HF repo under `Sayood/` holds a checkpoint of the matching size.**

| arm | `ckpt.pt` bytes |
|---|---|
| `flagship4b-v3enc-30k` | 3 415 808 330 |
| `flagship4b-v3enc-expA-nodrop-2k` | 3 415 808 330 |
| `flagship-v2corpus-30k` | 3 415 808 330 |
| `refb-speed-30k` | 3 153 889 214 |
| `refb-refbpatch-30k` | 3 157 099 838 |
| `refb-phase0-30k` | 3 150 737 694 |
| `finetune_traj` | 3 187 302 926 |
| `ft_trial` | 3 187 302 926 |
| `axis6-relaxed` | 787 360 702 |

⚠️ **Evidence class: MEASURED (size comparison against the full `Sayood/` HF file tree), NOT
sha256.** Size is a strong negative signal — a mirrored checkpoint would have exactly that size —
but it is **not proof**, and I did not upgrade it, because hashing 28 GB on `pod4` would add
sustained disk I/O to a **live training run** (`v1arch-v2bal`). Flagged, not assumed.

⚠️ Two pod4 arms **do** size-match HF and are therefore probably fine:
`refb-refbpatch-v2-30k` → `Sayood/tanitad-refb-speed/ckpt.pt`;
`p0-sB01-realmix` → `Sayood/tanitad-internal/ckpt_step8500.pt`.

---

## 2. What I made durable this turn

### 2.1 The REF-C val raster is now a VERIFIED two-copy artifact

The brief's second wall — *"one reachable copy, and one clip already produced a transient
unreadable load"* — is **closed for the 40 episodes that decide every published REF-C number**.

**MEASURED**, `raw/thor_val40_verify.json`: all **40/40** episodes on
`thor:/home/nvidia/valdata/physicalai-val-0c5f7dac3b11`

* match the **HF LFS sha256 bit-for-bit**, and
* `torch.load` cleanly with the episode contract intact,
* totalling **4 697 689 792 B**, which equals the HF total exactly.

That includes `ep_00028`, the previously 21 %-truncated episode — it is repaired.
**Size alone was not accepted as evidence, and neither was the exit code.**

### 2.2 The val-600 second copy is armed and sequenced

A **concurrent stream is already pulling the 278 GB train epcache to Thor**
(`pull_parity.py`, PID **22414**, 24.6 MB/s MEASURED — consistent with the retracted-to-23 MB/s
HF download figure). **I did not duplicate it.**

I armed the val-600 pull to run **after** it, not alongside it: HF download is bandwidth-bound,
so concurrency would not finish either sooner — it would just halve the more critical pull's rate.

* supervisor `/tmp/pull_val600.sh` — PID **34909**, polls `/proc/22414`, then launches
* `/tmp/pull_val600.py` — `snapshot_download` → `~/epcache/epcache-256px-phase0/physicalai-val-0c5f7dac3b11`,
  then **verifies all 600 against the HF sha256 manifest** and `torch.load`s a 1-in-25 sample.
* Digest manifests staged on Thor: `/tmp/hf_val600_sha.txt`, `/tmp/hf_train2376_sha.txt`.
* Disk checked with a **real `dd` write** (1.2 GB/s), not `df` alone: 670 GB free, need ~348 GB.

⚠️ The wait target was confirmed by reading `/proc/22414/cmdline` before arming — **not** by a
`pgrep -f` pattern, which self-matches the probing ssh command.

### 2.3 The anchors are confirmed genuinely mirrored

`flagship_v4_anchors_dense.pt` differs in **size** between repo (42 983 B) and the live v5f run's
copy on pod5 (42 550 B), which looks like corruption. It is not: the tensors are
**numerically identical** (`anchors` 256×20×2, `maxabsdiff = 0.0`, every scalar field equal).
The 433-byte delta is the `source` metadata string. The repo copy is a valid backup.

Also checked and **clear**: `core.autocrlf=true` with **no `.gitattributes`** in this repo — but
all four sampled tracked `.pt` blobs round-trip worktree↔blob with identical sha256. Git is
detecting them as binary.

### 2.4 Four artifacts were wrongly on my own single-copy list

My first census counted only the working tree and reported `anchors-dev256`,
`anchors-refc-small64`, `anchors-flagship-v4-dense` and the `windows_*/fan_*` eval dumps as
**SINGLE_COPY**. They are all **pushed to GitHub** on `origin/agent/arch-inf-20260803` and two
other remote branches. I committed the module's own error class inside the module, caught it, and
the census now models `github` as a location — with the honest caveat that `origin/*` refs prove
the blob was pushed *as of the last fetch*.

---

## 3. Needs the PI's authorisation — I stopped rather than acted

**Staged, NOT run:** `code/mirror_pod4_rescue_to_hf.py`

Mirroring the nine pod-only checkpoints into the **already-existing** archive repo
`Sayood/tanitad-archive-pod2-2026-08` is arguably inside the established migration precedent —
that repo exists and already holds exactly this class of artifact (pod2 rescue checkpoints).
I did **not** run it, for two independent reasons, either of which alone is sufficient:

1. ⛔ **`pod4` is training.** `flagship-v1arch-v2bal-30k` is live. An HF push reads, hashes and
   uploads ~28 GB from that pod — sustained disk and network load on a training run. The standing
   rule is absolute, and no durability argument outranks it.
2. **The authorisation is genuinely ambiguous.** These are nine *new* checkpoints, not a re-upload
   of already-published content. "Already holds that class" is an argument, not a decision.

**The PI's call, in one question:** *may the nine pod4 rescue checkpoints be pushed to
`Sayood/tanitad-archive-pod2-2026-08`, and if so, do we wait for `v1arch-v2bal` to finish first?*

My recommendation: **yes, and wait.** The arm is at step ~18k of 30k; mirroring after it lands
costs nothing and removes the programme's only zero-durable-copy exposure.

**Second, smaller decision:** the w120 caches (106 GB total) are HF + pod5 only. A Thor copy would
make them 2-durable. Thor has the disk for it after the two pulls land (~285 GB free). This is a
pure download, inside my authorisation — but it is *third* in priority behind the raw corpus, so I
have not started it rather than contend for the same bandwidth.

---

## 4. The census

```
artifact                           kind      copies dur  verdict      where
--------------------------------------------------------------------------------------------------
!!raw-train-epcache-256px          corpus         1   1  SINGLE_COPY  hf:…w120-256x640cyl
!!raw-val-epcache-256px            raster         1   1  SINGLE_COPY  hf:…w120-256x640cyl
  anchors-dev256                   anchors        2   2  OK           github,repo
  anchors-refc-small64             anchors        2   2  OK           github,repo
  raw-val-epcache-256px-eval40     raster         2   2  OK           hf:…w120-256x640cyl,thor
~~w120-train-cyl                   corpus         2   1  OK           hf:…,pod5  [VOLATILE]
~~w120-val-cyl                     raster         2   1  OK           hf:…,pod5  [VOLATILE]
  anchors-flagship-v4-dense        anchors        3   2  OK           github,pod5,repo
  evaldumps-windows-fan            evaldump       3   2  OK           github,pod5,repo
```

**Parity re-verified against the committed manifest, on HF, MEASURED:**
`epcache-256px-phase0/physicalai-train-e438721ae894` = **2376** `ep_*.pt`
(`ep_00000`…`ep_02399`) + **24 `skip_*` markers** + `DONE` = 2401 files, **278.78 GB**.
The 24 skip indices are present and nothing has re-selected episodes.
Val = **600** `ep_*.pt` + `DONE`, **70.39 GB**.

### 4.1 Fleet, as probed (not as configured)

⚠️ **An ssh config is a cache of what someone wrote down, never the fleet inventory.** I scraped
every pod address out of the repo and probed the ones the config does **not** contain.

| endpoint | in ssh config | TCP |
|---|---|---|
| `69.30.85.48:22192` (pod4) | yes | ✅ **UP** — training `v1arch-v2bal` |
| `69.30.85.106:22039` (pod5 / `tanitad-new`) | yes | ✅ **UP** — training `v5f` |
| `192.168.178.194` (thor) | yes | ✅ **UP** |
| `38.147.83.15:39198` (pod1) · `69.30.85.16:22079` (pod3) · `69.30.85.106:22073` (eval) | yes | ❌ refused |
| **`38.147.83.18:34126`** · **`69.30.85.75:22022`** · `38.147.83.15:30107` · `69.30.85.123:22091` | **NO — scraped from repo docs** | ❌ refused |

The two bolded addresses appear only in `FLEET_INVENTORY_2026-07-28-migration.md` and
`RESOURCE_LEDGER.md`. Both are dead — but they were probed, so "three live machines" is a
**measurement**, not a reading of the config.

---

## 5. Registry path reconciliation (P4)

`tools/registry_paths.py` sweeps every backtick-quoted artifact path in
`Project Steering/MODEL_REGISTRY.md`. **252 citations**: 115 EXISTS · **4 MISSING** ·
22 NOT_CHECKED (pod paths) · **17 NOT_A_PATH** · rest are bare filenames.

**It does not repair a path by pointing it at a plausible file.** Re-pointing a citation at a
lookalike is how a wrong number becomes a sourced number.

### 5.1 MISSING — malformed citations, not missing artifacts (4)

| citation | reality |
|---|---|
| `eval_flagship_v15/v16.py` | prose shorthand; **both** `stack/scripts/eval_flagship_v15.py` and `…v16.py` exist |
| `refb-speed-30k/ckpt.pt` | a *fragment* of a pod path — the file exists at `pod4:/workspace/rescue/experiments/refb-speed-30k/ckpt.pt` (MEASURED) |
| `refb-speed-30k/ckpt_prepatch_step8500.pt` | same — exists on pod4 |
| `resolve/ckpt.pt` | prose, not a path |

**Fix required:** the two `refb-speed-30k/…` citations should carry their host
(`tanitad-pod4:/workspace/rescue/experiments/…`), since a bare relative fragment reads as a repo
path and resolves to nothing.

### 5.2 NOT_A_PATH — brace expansions and templates used as citations (17)

The `{base,xl}` defect the brief already found is **not isolated**. Sixteen more citations name no
file: `ckpt_step{5,10,15,20}000.pt`, `taniteval/results/{windows,fan}_refc-{base,xl}-30k.pt`,
`gate_step{1k,5k,10k}.json`, `results/<key>.json`, `eff_*.json`, `eval_*.py`, `ep_*.pt`,
`…/{DESIGN,LAUNCH_PLAN,PRE_REGISTRATION}.md`, and others. Individually harmless as prose;
collectively they mean **a sweep cannot verify them**, so they are unsourced by construction.

### 5.3 Stranded pod paths — cited, with NO repo counterpart (5)

| citation | |
|---|---|
| `/workspace/experiments/refc_anchors_full.pt` | anchors — same class as the ones we do bank |
| `/workspace/experiments/flagship-v16-ab-ft/eval_v16.json` | an eval result behind a registry row |
| `/workspace/ops/heartbeats/flagship-v2corpus-30k.json` | |
| `/root/refb_orig_backup.py`, `/root/refb_train_orig_backup.py` | |

The pod they refer to (pod2) is **terminated**. These are either already lost or need locating on
pod4's rescue dump. **`refc_anchors_full.pt` is the one worth chasing** — it is an anchor set, and
anchors define what a planner is evaluated against.

### 5.4 Two false-defect classes I had to fix in my own sweeper

Both would have manufactured registry defects that do not exist:

1. **Bare filenames counted as dead paths.** First run: 85 MISSING, ~70 of them tokens like
   `ckpt.pt` / `ci.py` that the registry uses as *names*. Real defects were buried in noise.
2. **Ellipsis matching that required whole path components.** `…/own-dynamics-encoder/RESULTS_camcond.md`
   was reported MISSING; the file exists under `2026-07-22-own-dynamics-encoder/`. The ellipsis
   elides a *prefix of a component*, not only whole components.

Also: `.claude/worktrees/*` are transient **copies** of this repo and were turning every genuine
hit into an 8-way AMBIGUOUS. They are excluded from the index.

---

## 6. The standing check (P3)

The failure being prevented is exactly the one that created this task: **the count silently went
to 1 and nobody noticed until two unrelated streams tripped over it weeks later.**

`python tools/corpus_census.py --json <out>`

* exit **0** every artifact ≥ 2 machines · **1** something at exactly one copy · **2** zero copies
  **or the census could not be completed**.
* **A failed probe returns `UNKNOWN`, never `0`.** A truncated remote payload downgrades every
  `MISS` in that stream to `UNKNOWN`, because a network hiccup read as absence is how this risk was
  manufactured in the first place. Guarded by a dedicated test.
* **`PARTIAL` does not count as a copy.** Thor's 438-of-2376 train pull is not a backup.
* Host discovery is delegated to `tools/fleet_probe.load_fleet()` so the census never becomes a
  *second* hardcoded cache of the ssh config.

⚠️ **The instrument caught one of its own bugs on the first live run** and it is worth recording:
the payload was piped to `ssh -n … sh`, and `-n` *is* "redirect stdin from `/dev/null`", so the
script never arrived. Every host reported "incomplete payload" simultaneously — indistinguishable
from a fleet-wide outage. Because the module refuses to read that as absence, it reported
`UNRESOLVED` instead of inventing nine zero-copy artifacts. Fixed (payload travels in argv, stdin
explicitly `DEVNULL`) and regression-tested.

**Wiring it in is an integration request, not something I silently did:** it should join the
nightly `pod_git_drift.py` slot. See §8.

---

## 7. Test status

| suite | result |
|---|---|
| `tools/tests/` (incl. 34 census + 33 registry-path tests) | **222 passed** |
| `stack/` full suite | see §9 — reported as measured, not as a baseline claim |

⚠️ The stack baseline is **moving** (1900 → 1913 → 1932 within one session), so a count mismatch
is not by itself a regression.

---

## 8. ESCALATION — three things that need a decision, not a doc

1. 🔴 **The nine pod4-only checkpoints** (§1.1, §3). Zero durable copies. Needs the PI's
   authorisation to mirror, and should wait for `v1arch-v2bal` to finish. **This is the one item
   that can still lose program artifacts.**
2. 🟠 **Wire `tools/corpus_census.py` into the nightly job.** An instrument that is not scheduled
   is a document. Requested here rather than in a README — a README request sat unread for 10 days
   once.
3. 🟡 **`MODEL_REGISTRY.md` citation hygiene** (§5.1, §5.2): 2 host-less pod paths and 17 brace/template
   citations. I have **not** edited the registry — a corpus-durability agent silently rewriting the
   single quotable source is exactly the wrong move.

---

## 9. Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| `CORPUS_DURABILITY.md` (this file) | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-03-corpus-durability/` | no — staged |
| `CORPUS_DURABILITY_CENSUS.json` | same dir | no — staged |
| `raw/thor_val40_verify.json` | same dir · also `thor:/tmp/verify_val40.json` | no |
| `raw/registry_path_sweep.json` | same dir | no — staged |
| `tools/corpus_census.py` | `repo:tools/` | no — staged |
| `tools/tests/test_corpus_census.py` | `repo:tools/tests/` | no — staged |
| `tools/registry_paths.py` | `repo:tools/` | no — staged |
| `tools/tests/test_registry_paths.py` | `repo:tools/tests/` | no — staged |
| `code/pull_val600.py`, `code/pull_val600.sh` | `repo:…/code/` · **running on** `thor:/tmp/` | no |
| `code/verify_val40.py` | `repo:…/code/` · also `thor:/tmp/` | no |
| `code/mirror_pod4_rescue_to_hf.py` | `repo:…/code/` | no — **STAGED, NOT RUN** (§3) |
| `code/hf_enum.py` | `repo:…/code/` | no |
| in-flight: val-600 pull | `thor` supervisor PID 34909 → `~/epcache/…/physicalai-val-0c5f7dac3b11` | n/a |

**Nothing produced this turn lives only on a pod or only on Thor.**
