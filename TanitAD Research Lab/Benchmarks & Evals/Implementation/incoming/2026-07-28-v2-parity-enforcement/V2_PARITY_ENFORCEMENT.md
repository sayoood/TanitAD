# v2 parity enforcement — closing the hole in the guarantee this program calls sacred

**Date:** 2026-07-27 (dev box, Europe/Berlin) · **Stream:** Benchmarks & Eval
**Status:** staged, never committed, never pushed. **No pod was touched.** pod1 is training and pod2 is
building the 120° cache; both were left alone, and the fixture for every test below is synthetic.

**Predecessor:** `Architecture & Inference/.../2026-07-28-wide-fov-build/WIDE_FOV_BUILD.md` §6 —
the escalation. It found the hole, supplied `code/verify_v2_parity.py` as a stop-gap, and
**deliberately did not edit `parity.py`** because that is another stream's file. This document is
that edit, reviewed rather than smuggled in behind a build.

---

## 0. Headline

| # | finding | class |
|---|---|---|
| **1** | ⛔ **The hole reproduces at HEAD and I measured it RED.** With the new guard removed, `train_flagship4b --v2-cache … --require-parity` reaches `build_v2_providers` on a cache referencing **no registered corpus at all**. Restored: refused before the loader, `reached['loader'] is False` asserted. | **MEASURED** (`raw/wiring_redgreen_2026-07-27.json`) |
| **2** | ⭐ **The v2 path now proves MEMBERSHIP, not a count** — and the difference is demonstrated, not claimed. **4 of 9 defect classes are invisible to a count check and are refused by this one**, including a swapped clip and a wholly re-selected split *at identical cardinality*. | **MEASURED** (`raw/redgreen_2026-07-27.json`) |
| **3** | ⭐ **A new independent cross-check nobody had run: the 2 376 episode positions and the 24 skip positions tile `0…2399` exactly** — no gap, no overlap. That simultaneously confirms the split is **2 400 clips** and that `skip_%05d` indexes the *same ordered list* as `ep_%05d.pt`, which is the premise the identity-of-the-24 check rests on. | **MEASURED** (`raw/membership_facts_2026-07-27.json`) |
| **4** | ⭐ **The proof is strictly stronger than the stop-gap it adopts**, in three named ways — chiefly that a shortfall of the **right size made of the wrong clips** is now refused (`verify_v2_parity.py` accepted any `len(missing) == 24`). | **MEASURED** |
| **5** | 🔴 **`train_flagship_v4` gets NO v2 support, deliberately — and v5's trainer is UNDECIDED, with two live documents disagreeing.** `flagship-v5-retrain.PREP.md` presumes `train_flagship_v4`; `WIDE_FOV_BUILD.md` §7 gives a `train_flagship4b --v2-cache` command. v4 cannot read v2 and the wide corpus can only be v2. **PI call, before the run.** | **MEASURED** (both docs read directly) |
| **6** | ⚠️ **And the trainer question is bigger than parity: `train_flagship4b` HAS NO VAL LOOP AT ALL.** `ds_val` is assigned on both branches and **never read** — the whole file mentions it twice, both times on the left of `=`. "No held-out early-stop signal" is **cause #1 of the previous v5 failure (~29.5 GPU-h)**. ⚠️ This also **corrects** `WIDE_FOV_BUILD.md` §7, which frames it as a `--v2-cache` branch property. | **MEASURED** (`grep -n ds_val scripts/train_flagship4b.py` → lines 421, 429, both assignments) |
| **7** | ⛔ **What it still cannot prove, stated rather than buried** — §5. The sharpest: **the v2 manifest entry has no repo-side preimage.** The raw entry enumerates all 2 376 uids in the repo; clip ids are gated-confidential, so the v2 entry carries only a digest — and a refusal off-pod can therefore name *counts*, never *which clips*. | **MEASURED** |
| **8** | ✅ **No default moved.** `--require-parity` is opt-in and off; the unregistered-cache path still warns-and-proceeds without it, which is what keeps the deliberately non-parity `physicalai-v2bal` arm (training right now) running. Asserted in a test, not promised. | **MEASURED** |
| **9** | ✅ **The published runbook was EXECUTED, not written** — real subprocess runs of `register_v2_sibling.py` with the exact key §7 publishes, `ALL_PASS: true`. ⚠️ And a defect in that probe was **caught in my own instrument** before it shipped (§7.1). | **MEASURED** (`raw/runbook_smoke_2026-07-27.json`) |

**Evidence-class note.** Everything above is a **code-behaviour** fact: deterministic, `n` = 1 run, no
estimator and no interval — a path either executes or it does not. Tier language (DIRECTIONAL /
CONFIRMED) does not apply and is not borrowed. Where a corpus number appears it is read from the
**artifact that owns it**, never from prose — the C21 discipline. Storage figures (697 GB / ~95 GB)
are **INHERITED** from `WIDE_FOV_BUILD.md` §3 and were **not** re-measured here; they are cited as
motivation, and no decision in this document turns on them.

---

## 1. What the RAW path proves

`train_flagship_v4._assert_parity` → `parity.assert_parity_corpus(require=True)` →
`parity.check_uids`. On a directory whose path contains a registered corpus key:

| # | check | where |
|---|---|---|
| 1 | the directory is **not** a known-leaky split (`physicalai-val-f1b378f295ae`) — never downgradable | `assert_parity_corpus` |
| 2 | `len(ep_*.pt) == manifest.episode_count` (**2 376**) | `check_uids` count branch |
| 3 | `sha256(sorted ep_*.pt basenames) == manifest.episode_uid_sha256` — so a **substituted or re-selected** set of the right size is refused too | `check_uids` content branch |
| 4 | on failure it names **which** uids are missing and which are extra | `_diff_lines` |
| 5 | it runs **before any GPU allocation**, and `ParityViolation(SystemExit)` gives the pod supervisor a non-zero exit | module contract |

⚠️ **What the raw path does NOT do, so the comparison below is fair:** the *trainer* guard reports
`skip_markers_present` but does **not** assert it. Only `register_geometry_sibling` asserts the skip
indices, and only at registration. Neither hashes episode **content** bytes (`notes.limitations`).

---

## 2. What the v2 path could NOT prove — the hole

`train_flagship4b`'s `--v2-cache` branch had **no parity check of any kind**. The guard lives in
`_cache_split`, which only `--cache-dirs` calls. The branch said so in as many words: *"This is a
SEPARATE corpus and does NOT touch the raw parity path."*

That reasoning held exactly as long as the only v2 corpus was the deliberately non-parity 9 000-clip
`physicalai-v2bal`. **It stopped holding the moment v5's corpus became a v2 re-cache of the sacred
split**, and nothing in the code noticed the premise had changed.

⚠️ **`register_geometry_sibling()` could not have closed it, and the reason is structural, not an
oversight.** Two uid spaces:

| | raw epcache | v2 compressed |
|---|---|---|
| file | `ep_%05d.pt` | `<clip_id>.v2ep.pt` |
| identity | **position** in the ordered clip list | **the clip id** |
| build failure | leaves `skip_%05d` **at that index** | leaves **nothing** — the file is simply absent |
| shape | dense, self-describing | a flat **set** |

`register_geometry_sibling` compares `sha256(sorted ep_*.pt)`. A v2 cache has no `ep_*.pt`, so it
refuses — **correctly**: it cannot tell "different format" from "different episodes".

**And the v2 cache is not optional.** The raw epcache at 120° / 256×640 is 293.4 MB/episode ⇒ **697 GB
for the train split alone** and fits on no host in the fleet, against **~95 GB** for the v2 PNG build.
*(INHERITED — `WIDE_FOV_BUILD.md` §3, not re-measured here.)*

⇒ **The v5 wide cache was trainable today with zero parity enforcement.** Reproduced RED at HEAD:
`raw/wiring_redgreen_2026-07-27.json`.

---

## 3. What it NOW proves

### 3.1 The manifest learns the clip-id uid space

`stack/tanitad/data/parity_manifest.json` gains a **`clip_membership`** block on both PhysicalAI
entries. Purely additive (`git diff --stat` → **39 insertions, 1 deletion**, the deletion being a
trailing comma in `notes`).

```
physicalai-train-e438721ae894.clip_membership
  n_clips                2400          <-- CLIPS
  decode_failures          24
  clip_id_sha256_sorted  e61a04553df5b9d52a0810be32cf31927bd92644d9d12ada563910b8a0ada4de
  ordered_equals_sorted  true
(and episode_count stays 2376           <-- EPISODES.  2400 - 24 = 2376)
```

**Provenance:** `…/2026-07-28-wide-fov-build/raw/parity_split_meta_2026-07-27.json`, written on pod1 by
`parity_split_export.py`, which **refuses to write anything** unless that host reproduces *both*
canonical corpus keys first (`keys_match_parity: true`). 🔒 The clip ids stay on the pods; the repo
carries only these digests.

⚠️ **`clip_membership` exists as a separate block, not as a re-use of `episode_count`, precisely
because the two are different facts.** A check written against 2 400-as-episodes or
2 376-as-clips is wrong in opposite directions — this corrected the brief that commissioned the work.
`tests/test_v2_parity.py::test_clips_are_not_episodes` pins `n_clips − decode_failures ==
episode_count` so the two can never drift apart silently.

⭐ **And it is verified, not asserted** (`raw/membership_facts_2026-07-27.json`, `ALL_PASS: true`):

```
union(episode_uids' indices, skip_indices) == exactly {0 … 2399}     ← no gap, no overlap
N = 2400 == export.train_clips                                      ← the split IS 2400 clips
ordered digest == sorted digest                                     ← discovered order IS clip-id order
manifest.clip_id_sha256_sorted == export.train_ids_sha256_sorted    ← the manifest matches its source
```

The first line is load-bearing and had not been checked anywhere: it is what makes
`expect_clips[i] for i in skip_indices` a valid way to name the 24 corrupt clips. The third means a
**set** proof — all the v2 format can offer — is here **exactly as strong** as an ordered one.
⚠️ That is a property of *this corpus*, not a general guarantee, and the checker records it per run.

### 3.2 `parity.py` §9 — three functions

| function | role |
|---|---|
| `verify_v2_membership(cache_dirs, expect_clips=…)` | **the proof.** Set-diff of the built clip ids against the exported parity split. Refuses on any extra; accepts a shortfall only if it is *exactly the recorded decode failures*. |
| `register_v2_geometry_sibling(cache_dirs, new_key=…, geometry=…)` | **the registration IS the proof** — mints a manifest entry only if the above passes. Keeps `register_geometry_sibling`'s contract; adds `uid_kind: "v2ep_clipid"`. |
| `assert_v2_parity_cache(cache_dirs, label=…, require=…)` | **the trainer guard.** Mirrors `assert_parity_corpus` decision-for-decision. |

**Adopted, not reinvented** — from the sibling stream's `code/verify_v2_parity.py`: its pass criteria
(extras fatal; shortfall must equal the recorded 24), its order-independent sorted digest, its
counts-and-digests-only output discipline. **Three deliberate strengthenings:**

1. ⭐ **The 24 are checked by IDENTITY, not by count.** The original accepted any `len(missing) == 24`.
   Because the skip indices index the ordered clip list (§3.1), the **same 24 clips must fail again** —
   which `WIDE_FOV_BUILD.md` §8 itself calls *"a strong independent check"* while testing only
   cardinality. Measured: a shortfall of the right size made of two *other* clips now yields
   `of those, 0 are the RECORDED failures and 2 are NOT`.
2. ⭐ **The expectation is bound to the committed manifest.** The original compared the built digest
   against a digest carried in the *same sidecar* as the clip list, so a **self-consistent wrong pair
   would have verified**. The supplied list must now first reproduce
   `clip_membership.clip_id_sha256_sorted`.
3. ⭐ **It refuses instead of returning a verdict string.** The original wrote `VERDICT: NOT VERIFIED`
   into JSON and **exited 0** unless there were extra clips — a truncated build did not fail the
   command.

### 3.3 One behaviour change outside §9, and its proof of harmlessness

`corpus_key_of` now breaks ties **longest key first** (then lexicographically). A sibling cache
legitimately lives under its parent's path, and the old rule resolved it to the **parent**.

⛔ **This may not move any pre-existing answer, and that is proven, not hoped:** the three keys that
existed before siblings are pairwise non-overlapping, so on every path where several match, the two
orderings agree — enumerated exhaustively in
`test_longest_match_cannot_change_legacy_resolution`.

---

## 4. ⛔ What it STILL cannot prove

**This section is the point of the document.** The two paths do **not** enforce the same thing, and
saying they do would be the more comfortable lie.

| # | the raw path | the v2 path |
|---|---|---|
| 1 | manifest **enumerates all 2 376 uids in the repo** — the digest has a repo-side preimage, so any host can recompute it and name *which* episodes are missing | 🔒 clip ids are gated-confidential ⇒ the entry carries **only a digest**. **Off-pod a refusal can name counts and digests, never which clips.** `_diff_lines` is deliberately NOT reachable from the v2 refusals. |
| 2 | a failed build leaves `skip_%05d`, so *"absent because uncacheable"* is **on disk** | a failed build leaves **nothing**. Absence is indistinguishable from deletion **without `--expect-clips`** |
| 3 | `mode="subset"` supports deliberate `[:n]` episode subsets | ⛔ **no subset mode.** A v2 sibling is all-or-nothing; `--episodes N` is not honoured on this branch anyway |
| 4 | — | ⚠️ **digest-only mode refuses a CORRECT incomplete build.** With no clip list it cannot tell a legitimate decode failure from a lost clip, so it refuses both. Measured; the refusal says so verbatim. **`--expect-clips` is mandatory on the pod.** |

**Shared blind spots (neither path closes these):**

- ⛔ **Nothing hashes pixel or tensor CONTENT.** Membership proves *which clips*, never *which pixels*.
  A cache built at the wrong FOV with the right clips **passes**. The geometry is *recorded*
  (`_geometry.json`, `provenance.geometry`) and asserted **pre-decode** by the builder
  (`_assert_geometry_deliverable`) — hours earlier, by a different guard, in a different stream. If
  that assert is wrong, nothing here catches it. Said plainly in the code and repeated here.
- ⚠️ **A registered key is only as good as its registration.** `register_v2_geometry_sibling` refuses an
  unproven cache, and `register_v2_sibling.py --write-manifest` refuses to overwrite without `--force`
  — but a hand-edited `parity_manifest.json` defeats both. The guard refuses a v2 entry carrying no
  digest for that reason; it cannot refuse a *plausible* forged one.
- 🔴 **A manifest entry that is not COMMITTED is worse than none.** The registration runs on a pod; if
  the diff is not staged, the cache reads NON-PARITY on every other host and `--require-parity`
  **refuses to start**. The script prints this at write time. It is step 3 of the runbook for a reason.
- ⚠️ **`--v2-cache` runs no val loop — and neither does `train_flagship4b` at all** (§0 finding 6).
  Parity of the *training* corpus is now enforced; **held-out selection is a different hole and is
  still open.**

---

## 5. Both directions, with the failure REASON asserted

`raw/redgreen_2026-07-27.json` — nine cases through `assert_v2_parity_cache`, each reporting what a
**count-only** check would have concluded beside what the guard concludes.

| case | clips | a COUNT check | the GUARD | reason it gave |
|---|---:|---|---|---|
| **CORRECT** — the registered cache | 12/12 | pass | ✅ **PASS** | `sha256(sorted clip ids) MATCHES the committed manifest entry` |
| DROPPED one clip | 11/12 | refuse | ⛔ REFUSE | `11 present, 12 registered <-- TRUNCATED by 1` |
| ⭐ **SWAPPED one clip** | **12/12** | **pass** ← | ⛔ **REFUSE** | `<-- count OK — MEMBERSHIP DIFFERS AT THE SAME COUNT` |
| ⭐ **RE-SELECTED split** | **12/12** | **pass** ← | ⛔ **REFUSE** | `<-- count OK — MEMBERSHIP DIFFERS AT THE SAME COUNT` |
| EXTRA foreign clip | 13/12 | refuse | ⛔ REFUSE | `13 present, 12 registered <-- EXTRA 1` |
| ⭐ **UNREGISTERED cache** *(the v5 state as built today)* | **12/12** | **pass** ← | ⛔ **REFUSE** | `--require-parity was passed and none of these dirs references a registered corpus key` |
| **same, WITHOUT `--require-parity`** | 12/12 | pass | ✅ **PASS-THROUGH, warned** | `NON-PARITY v2 corpus` — ⚠️ **the no-default-moved case** |
| ⭐ **v2 cache wearing a RAW epcache key** | **12/12** | **pass** ← | ⛔ **REFUSE** | `uid kind : this directory holds *.v2ep.pt (uid_kind 'v2ep_clipid')` |
| same clip in two `--v2-cache` dirs | 14/12 | refuse | ⛔ REFUSE | `2 clip(s) appear in more than one --v2-cache dir … would contribute its windows TWICE` |

**The C13 self-check.** *A guard whose refusals are a subset of what a count check already refuses is a
count check wearing a membership check's name.* **Four rows** — swapped, re-selected, unregistered,
uid-kind — are refused where a count sees nothing wrong. That column is the evidence, and it is in the
raw JSON under `c13_self_check.count_only_would_have_missed`.

**The membership proof's own three cases:**

| case | verdict | reason |
|---|---|---|
| shortfall **is** the recorded decode failures | ✅ PASS | `identity_checked: true`, `missing: 2` |
| ⭐ shortfall of the **right size, wrong clips** | ⛔ REFUSE | `of those, 0 are the RECORDED failures and 2 are NOT` |
| digest-only on a **correct incomplete** build | ⛔ REFUSE | `DIGEST-ONLY MODE cannot say WHICH clips differ` — ⚠️ §4 #4, a deliberate false-refusal |

**And the guard was proven to fail on the real defect, not a synthetic one** —
`raw/wiring_redgreen_2026-07-27.json`. Remove the one line that wires it in and
`tests/test_v2_parity.py` goes **1 failed / 33 passed** with
`AssertionError: build_v2_providers was REACHED despite an unregistered cache under --require-parity`,
raised at `scripts/train_flagship4b.py:407` — the real call site. Restore it: **34 passed**.

🔒 One test exists only to prove confidentiality: `test_every_v2_refusal_withholds_clip_ids` asserts
no synthetic clip id appears anywhere in a refusal message.

---

## 6. 🔴 The `train_flagship_v4` decision

**`train_flagship_v4` gets NO `--v2-cache` support. Decided, not deferred.** Three reasons:

1. ⛔ **v5's trainer is UNDECIDED and two live documents disagree.** `Project Steering/Gates/
   flagship-v5-retrain.PREP.md` §3 presumes v4 — *"`train_flagship_v4` (`require=True`) REFUSES TO
   TRAIN ON IT. v5 stalls unless a new key is registered."* `WIDE_FOV_BUILD.md` §7 gives a
   `train_flagship4b --v2-cache` launch command. **v4 cannot read v2, and the wide corpus can only be
   v2.** Bolting a branch onto v4 would silently pick the answer.
2. ⛔ **It is not a flag, it is a stream.** v4 does not merely load episodes: `FlagshipV4Dataset`,
   curriculum phases, the canary controller and a held-out gate all sit on the raw path. None is
   exercised on lazy providers. Untested code on the critical path of the most expensive run in the
   program is the wrong trade.
3. ⛔ **It may be the wrong fix entirely** if v5 runs on `4b`.

**What was done instead — the absence made EXPLICIT rather than latent.** Pointing v4 at a v2 cache
used to produce *"does not reference the canonical corpus"*, which sends the reader off to rename a
directory. It now appends (`parity._V2_HINT`, fired on both raw refusal paths):

```
⚠️ THIS IS A V2 COMPRESSED CACHE (*.v2ep.pt), not a raw epcache.
Raw-epcache trainers (train_flagship_v4 --train-cache, train_flagship4b --cache-dirs)
CANNOT read it: episode identity here is a CLIP ID, not an ep_%05d.pt position.
Use:  train_flagship4b.py --v2-cache <dir> --require-parity
and register the corpus first: scripts/register_v2_sibling.py (parity.py §9).
```

Pinned by `test_raw_guard_on_a_v2_dir_names_the_format_and_the_right_trainer`.

🔴 **ESCALATION — needs the PI, before the run, not after.** Choosing `train_flagship4b` for v5 to get
the wide corpus **re-opens cause #1 of the previous v5 failure**: `train_flagship4b` has **no val loop
at all** (finding 6), and *"no held-out early-stop signal"* cost **~29.5 GPU-h — half the run —
training past the best checkpoint**. Parity is now enforced on that path; **held-out selection is
not, and this document does not fix it.** The three ways out are a v4 v2-loader, a val loop in `4b`,
or an out-of-band mid-run gate — and they are not equal in cost. **That is the decision, and it is
larger than the one I was sent to close.**

---

## 7. ⭐ The exact command a v5 launch must run before training

The runbook is **rebuild → register → commit manifest → train**. Steps 2–4, exactly:

```bash
# ---- 2. REGISTER, on pod2, when the 8-shard build finishes -----------------
# 2a. prove membership; writes nothing, changes nothing. Safe on a busy host.
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack \
python3 scripts/register_v2_sibling.py --verify-only \
  --cache        /workspace/data/pai_wide120_v2png_train \
  --expect-clips /workspace/wfov/paritysplit/parity_train_clips.txt \
  --out          /workspace/wfov/v2_parity_verify.json
# PASS CRITERIA, fixed here BEFORE the number exists (GATE_PROTOCOL §0.3):
#   extra_count == 0                     — ANY extra clip is a re-selection
#   AND ( membership_identical           — all 2400 built
#         OR missing_count == 24 AND shortfall_matches_recorded_skips == true )
#                                        — and they must be THE recorded 24
#   Anything else: DO NOT REGISTER, DO NOT TRAIN, report it.

# 2b. the cache dir MUST carry the key — corpus_key_of resolves by path substring
mv /workspace/data/pai_wide120_v2png_train \
   /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl

# 2c. mint the manifest entry (refuses if 2a would not have passed)
PYTHONPATH=/workspace/TanitAD/stack python3 scripts/register_v2_sibling.py \
  --cache        /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --new-key      physicalai-train-e438721ae894-w120-256x640cyl \
  --expect-clips /workspace/wfov/paritysplit/parity_train_clips.txt \
  --out          /workspace/wfov/v2_sibling_entry.json \
  --write-manifest

# ---- 3. COMMIT THE MANIFEST ------------------------------------------------
# 🔴 NOT OPTIONAL. An entry that lives only on pod2 makes the cache read
#    NON-PARITY everywhere else, and --require-parity then REFUSES TO START.
#    Copy stack/tanitad/data/parity_manifest.json back to the repo, then:
#      git add stack/tanitad/data/parity_manifest.json
#    (check `git status --short` for foreign staged entries FIRST — CLAUDE.md)

# ---- 4. TRAIN --------------------------------------------------------------
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack \
python3 -u scripts/train_flagship4b.py \
  --v2-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --config flagship4b --v2  ...            # remaining args unchanged
```

⛔ **`--require-parity` is the whole point of step 4.** Without it an unregistered or mismatched cache
prints one `NON-PARITY` line and **trains anyway** — that is the pre-existing default and it was left
untouched on purpose (§0 finding 8). **A v5-class run that omits the flag is unenforced.**

⭐ **The proposed key deliberately CONTAINS its parent's key.** It reads as *"e438721ae894, re-cached at
120° / 256×640 cylindrical"*, and §3.3's longest-match tie-break is what makes it resolve to the
sibling rather than the parent. Before registration the guard refuses it with the **uid-kind** message,
which is the informative pre-registration state, not a silent pass.

⚠️ **Before 2a the geometry must be re-read from `_geometry.json`**, because membership proves clips
and never pixels (§4). The build recorded `achieved_hfov_deg: 120.0`, `f_eff: 305.5775`.

### 7.1 The runbook was RUN, not written

⚠️ *A runbook that has never been executed is a hypothesis.* `code/runbook_smoke.py` drives the real
`register_v2_sibling.py` as a subprocess through steps 2a → 2c → key resolution → the step-4 guard →
a re-registration attempt, on a synthetic corpus **with the exact key published above**.
`raw/runbook_smoke_2026-07-27.json`, **`ALL_PASS: true`**:

| step | result |
|---|---|
| 2a `--verify-only` | exit 0; both published pass criteria true; **manifest untouched** |
| 2c `--write-manifest` | exit 0; `uid_kind: v2ep_clipid`, `derived_from` set, geometry recorded, **and it printed the "NOW STAGE IT" reminder** |
| key resolution | resolved **to the sibling**; the old lexicographic rule resolves to `physicalai-train-e438721ae894` — `old_rule_was_wrong: true` |
| 4 guard under `--require-parity` | `parity: true`, `sha256(sorted clip ids) MATCHES` |
| re-register without `--force` | **exit 1, refused** — a truncated cache cannot re-record itself into a passing manifest |

⚠️ **A defect in this probe, caught and fixed rather than shipped.** Its first draft simulated the
"old rule" over the *manifest's* keys only — where the parent key is not even a candidate — so it
reported the sibling for **both** rules and printed a confident "reading" about a comparison it had
not made. Corrected to use the same candidate set `corpus_key_of` uses; **the identical defect was
present in the matching pytest case and is fixed there too**, with an assertion that the old rule
*must* resolve to the parent or the test is not exercising the tie-break at all.

---

## 8. Test counts — before and after

| suite | before | after | delta |
|---|---|---|---|
| `stack/` | **1264 passed, 12 skipped** | ✅ **1298 passed, 12 skipped** | **+34**, all in `tests/test_v2_parity.py` |
| `taniteval/` | **559 passed** | ✅ **559 passed** | 0 — nothing in this change reaches it |

*(dev box, `C:\Users\Admin\venvs\tanitad`; the system `python` 3.14 has no pytest.)*
⚠️ **Zero tests are skipped by this change.** The two wiring tests deliberately stub
`tanitad.data.v2_dataset` into `sys.modules` instead of `importorskip("torchvision")` — the dev box
has no torchvision, so an importorskip would make the guard tests **skip exactly where they are most
likely to be run**, which is a guard that cannot fail.

---

## 9. Deliverable manifest

**Everything `git add`ed into the working tree. Nothing committed. Nothing pushed. No pod touched.**

| artifact | where it lives | only one copy? |
|---|---|---|
| `V2_PARITY_ENFORCEMENT.md` (this file) | `repo:…/incoming/2026-07-28-v2-parity-enforcement/` | no |
| ⭐ `stack/tanitad/data/parity.py` — §9 (+`_V2_HINT`, `corpus_key_of` tie-break) | `repo:` **staged** | no |
| ⭐ `stack/tanitad/data/parity_manifest.json` — `clip_membership` on both splits | `repo:` **staged** | no |
| ⭐ `stack/scripts/train_flagship4b.py` — the guard + `--require-parity` | `repo:` **staged** | no |
| ⭐ `stack/scripts/register_v2_sibling.py` — **new**, verify/register/write | `repo:` **staged** | no |
| ⭐ `stack/tests/test_v2_parity.py` — **new**, 34 tests | `repo:` **staged** | no |
| `code/redgreen_probe.py` + `raw/redgreen_2026-07-27.json` | `repo:` | no (regenerable) |
| `code/verify_clip_membership_facts.py` + `raw/membership_facts_2026-07-27.json` | `repo:` | no (regenerable) |
| `code/runbook_smoke.py` + `raw/runbook_smoke_2026-07-27.json` — §7 executed | `repo:` | no (regenerable) |
| `raw/wiring_redgreen_2026-07-27.json` — the RED/GREEN record | `repo:` | **yes** (the RED state is a manual edit; the procedure to reproduce it is in the file) |

⚠️ **Nothing here exists on a pod, and the registration deliberately has not been run** — the build
was still going (638/2 400 at last probe) and the brief was explicit that it must not be disturbed.
**Step 2 of §7 is owed to whoever finishes the build.**

⚠️ `git status --short` will show **other streams' staged work** (the wide-FOV `v2_compressed.py`
fix, the HF push bundle). Per `CLAUDE.md`: read it first, and prefer a pathspec-free
`git commit -F <msgfile>` after confirming every index entry is intended program work — `git commit
-- <pathspec>` segfaults on this repo.

### What this unblocks

- 🔴 **`Project Steering/Gates/flagship-v5-retrain.PREP.md` §3 item 7** — the wide-geometry v5 run. Its
  blocking sub-item was *"`parity.corpus_key_of()` matches on the DIRECTORY NAME, so a re-cropped cache
  reads NON-PARITY and `train_flagship_v4` REFUSES TO TRAIN ON IT. v5 stalls unless a new key is
  registered."* **The registration path now exists for the format v5's corpus is actually in.**
- 🔴 **`WIDE_FOV_BUILD.md` §12 escalation 1** — *"the parity guarantee is currently NOT enforced for the
  artifact v5 would train on"*. **Closed for the training corpus**, by the second of the three routes
  it named (a guard on `--v2-cache`) plus the first (a `uid_kind` registration path). The third
  (`--v2-cache` in `train_flagship_v4`) is **refused, with reasons** — §6.
- The v5 runbook's **rebuild → register → commit manifest → train** is now executable end to end; §7
  is the command list.

---

## 10. Escalations

0. 🔴 **v5's TRAINER IS NOT DECIDED, and the choice re-opens the previous run's #1 failure.** v4 cannot
   read v2; `4b` can but has **no val loop at all**. Parity is now enforced on `4b`; **held-out
   selection is not.** PI call, before the run. (§6)
1. 🔴 **Someone must run §7 step 2 when pod2's build finishes**, and **step 3 is the part that gets
   forgotten** — an unstaged manifest turns `--require-parity` into a refusal-to-start on every other
   host.
2. ⚠️ **`train_flagship4b` builds a val dataset on the raw path and never reads it** (`ds_val`, lines
   421/429). On `--cache-dirs` that means every val episode is loaded and discarded each run. Dead
   code with a real cost, and it makes the "no val loop" gap easy to misread as v2-specific — which
   `WIDE_FOV_BUILD.md` §7 does. Flagged for a separate fix; **not** touched here.
3. ⚠️ **`GATE_PROTOCOL` / `MODEL_REGISTRY` consequence:** a wide-geometry arm is a **different corpus
   key**. The registry already needs a GEOMETRY column (PREP §3 item 7); it now also needs the
   **corpus key** per row, because two rows can share `e438721ae894`'s *episodes* and differ in pixels.
   `provenance.derived_from` makes the lineage machine-readable — nothing reads it yet.
4. ⚠️ **`parity_manifest.json`'s val entry is still `episode_uid_sha256: null`** (count-only, an
   inherited TODO). Its **clip** membership is now recorded, so a v2 val sibling is registrable —
   but only from a COMPLETE 600-clip build, since no skip indices are committed for the val split.
5. ⚠️ **Non-ASCII in refusal messages.** `parity.py` refusals contain `…` and `⚠️` (pre-existing —
   `_refuse` already used `…`). On a cp1252 console the traceback print raises `UnicodeEncodeError`
   and **masks the parity message**. Harmless on pods (UTF-8); it bit this session's dev-box probe run
   and the workaround is `PYTHONIOENCODING=utf-8`. Not changed, because changing it now would touch
   the raw path's messages too.
