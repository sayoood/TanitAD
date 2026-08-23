# Registry citations — 21 dead or malformed citations in the one document we must all quote from

*2026-08-03 · branch `agent/arch-inf-20260803` · times Europe/Berlin unless a log says UTC*

## Headline

**21 defective citations found by `tools/registry_paths.py`. 14 rewritten, 6 shown never to have
been citations, 1 marked `UNRESOLVED` and now failing loud forever.** Plus 5 numeric repairs (P4)
and a second instrument that verifies the registry's *numbers* against their artifacts.

| outcome | n | |
|---|---|---|
| **Rewritten** | **14** | 4 MISSING + 10 brace/glob |
| ├ target verified to exist | 9 | I opened each file |
| ├ literal names, host gone | 3 | brace removed; durability now stated instead of implied |
| └ **marked UNRESOLVED-as-to-source** | 2 | the §6 latency triples are **in no committed artifact** — repointing them would have been a lie |
| **Allowlisted `pattern`** — never a citation | **6** | naming conventions, a CLI template, verbatim `git status` output, and a defective form quoted *as the evidence for the rule banning it* |
| **Allowlisted `unresolved`** — a real defect, unrepairable | **1** | `gate_step{1k,5k,10k}.json` — host terminated, absent from the rescue dump, **and the stem is wrong too** |

*(Two of the 14 keep one residual allowlisted site, because R5 and R14 quote the defective form
verbatim as their evidence. R8 set that precedent: you correct the source, not the quotation.)*

### What I did NOT do, plainly

* **`pytest` was not run.** See §7 — G: went into a hard read stall mid-task and I could not run it.
  Everything below was validated against an **off-drive checkout** instead, and the exact result is
  stated. ⚠️ **Treat the test counts as UNVERIFIED until someone re-runs them on the live tree.**
* I did **not** repair §6's three flagship latency figures. They cannot be sourced — §3.3.
* I did **not** hash the pod4 checkpoints. That is 3.1 GB of sustained I/O on a pod that is
  **training**. Size-equality is what I have, and I label it as such.
* I did **not** recover any of the five stranded pod paths (§5). Four hosts refuse connections.
* I did **not** copy anything off a pod. Every pod fact is a read-only `ls` / `find`.
* P4 is a **bounded** pass, not a full numeric audit — §6.

---

## 1. Why a dead citation here is not a typo

`MODEL_REGISTRY.md` is the ONLY quotable source for model facts. A citation that resolves to nothing
**sends the next reader to a path that does not exist while still looking sourced** — it launders an
uncheckable claim into an apparently-checked one.

The 21 were not new and not one person's mistake. They accumulated because **nothing ever failed**.
§4 is the part of this that matters most.

---

## 2. The four MISSING — all four malformed, none a missing artifact

| # | before | now | evidence |
|---|---|---|---|
| M1 | `eval_flagship_v15/v16.py` | `stack/scripts/eval_flagship_v15.py` **and** `…_v16.py`, each with the line of `real_episode_ids()` | MEASURED: `v15:125`, `v16:171`; the quoted docstring is `eval_flagship_v15.py:128-130`, verbatim |
| M2 | `refb-speed-30k/ckpt.pt` | `tanitad-pod4:/workspace/rescue/experiments/refb-speed-30k/ckpt.pt` | MEASURED (mine, read-only `ls` over ssh): present, **3 153 889 214 B** |
| M3 | `refb-speed-30k/ckpt_prepatch_step8500.pt` | same dir, same host | MEASURED: present, **3 153 889 214 B** — equal size, consistent with the byte-identical claim; md5 NOT re-run |
| M4 | `resolve/ckpt.pt` (×2 rows) | `https://huggingface.co/Sayood/tanitad-refc-xl/resolve/main/ckpt.pt` and `…-refc-base/…` | MEASURED (mine, unauthenticated `HEAD`): **401**, `X-Error-Code: GatedRepo`, both repos |

**A host-less fragment is the worst kind of dead citation.** `refb-speed-30k/ckpt.pt` reads as
repo-relative, resolves to nothing, and gives no hint the artifact is on a pod at all.

### 2.1 A lineage correction that came with M2/M3

R10 recorded the misnamed checkpoint on **"pod1"**. Wrong now, and probably always: the originating
host `tanitad-pod` **refuses connections**, and the only reachable copy is pod4's rescue dump. R10
now says so and carries the constraint that matters — **this arm has no HF copy** (no repo under
`Sayood/` holds a file of that size), so the recommended "rename" must not become a move that risks
the only copy, and not while pod4 is training.

---

## 3. The seventeen NOT_A_PATH

### 3.1 Rewritten with the target verified (5 of the brace expansions)

| site | after |
|---|---|
| §1.5 v4 Result | `flagship-v4-fromscratch-30k-produced.json` + `…-oracle.json`. **Both re-read; both match the row exactly**: `cluster_bootstrap.model.ade_0_2s` **0.8563 [0.7282, 1.0035]** and **0.6423 [0.5348, 0.7586]**, `ckpt_step` 29999 both, `goal_provenance.deployable` `true`/`false` |
| §4.2 REF-C scale A/B | the four `windows_/fan_ × base/xl` paths, all four verified present |
| §4.4 REF-C closed-loop | `REFC_suite_base_results.json` + `REFC_suite_xl_results.json`, both verified |
| §8 own-dynamics encoder | `DESIGN.md`, `LAUNCH_PLAN.md`, `PRE_REGISTRATION.md`, all three verified |
| §6 reading 4 | the **three** files the three numbers actually come from |

§6 reading 4 is the pattern to copy. It asserted three paired deltas against a wildcard. Now cited
file-by-file at `vs_floor_paired.cv.long_abs_2s_m`, and re-verified:

| arm | registry says | artifact says | file |
|---|---|---|---|
| REF-C-XL | +0.2170 [+0.0584, +0.3783], separated | **0.217 [0.0584, 0.3783]**, `separated: true` | `taniteval/results/driving_refc-xl-30k.json` |
| REF-C-base | +0.2300 [+0.0773, +0.3816], separated | **0.23 [0.0773, 0.3816]**, `separated: true` | `taniteval/results/driving_refc-base-30k.json` |
| flagship v1 | +0.2543 [−0.0278, +0.5304], **not** separated | **0.2543 [−0.0278, 0.5304]**, `separated: false` | `taniteval/results/driving_flagship-30k.json` |

Exact, all three, estimator `paired_episode_cluster_bootstrap`.

### 3.2 Rewritten to literal names, existence unverifiable (3) — and the durability truth behind them

Braces removed. What the sweep exposed is worse than a syntax defect: **each names artifacts whose
host is gone.**

* **v4-from-scratch milestones.** `ckpt_step5000/10000/15000.pt` were pod-disk-only; pod2 is
  terminated and MEASURED, this run dir is **absent from pod4's rescue dump**; the HF list is
  `ckpt.pt` + `ckpt_step20000.pt` only. ⇒ **the 15 k milestone that carries this arm's
  decision-grade read has no reachable copy.** The row said "single-disk"; it now says gone.
* **REF-B v2 milestones.** `ckpt_step5000/15000/20000.pt` on `tanitad-pod`, which refuses
  connections; `/root/refb_milestones/` is not in the rescue dump. The **run dir itself IS rescued**
  and its `ckpt.pt` size-matches HF, so the arm survives — the milestones do not.

⚠️ `Connection refused` is *unreachable*, not *proven destroyed* — a RunPod volume resize presents
identically. Stated that way in the registry.

### 3.3 The two I refused to repair

§6 reading 3 quotes the flagship tick as **103.42 / 93.76 / 104.49 ms** and REF-C-XL as
**44.28 / 27.84 / 26.12**, sourced to `taniteval/results/eff_*.json`. I opened them.

MEASURED 2026-08-03, `plan_step.p50_ms`:

| | fp32 | tf32 | amp16 | file |
|---|---|---|---|---|
| §6 says (flagship) | 103.42 | 93.76 | 104.49 | — |
| **artifact** | **97.32** | **97.70** | **123.83** | `taniteval/results/eff_flagship-30k.json` |
| §6 says (REF-C-XL) | 44.28 | 27.84 | 26.12 | — |
| **artifact** | **44.06** | **27.78** | **21.00** | `taniteval/results/eff_refc-xl-30k.json` |

**Neither triple is in its own cited artifact.** R14 had recorded the flagship half and described
REF-C's fp32/tf32 as agreeing; **that was too generous** — they differ in the second decimal and
amp16 differs by 24 %.

⇒ The citation now names the committed files, prints their real values, and marks the six quoted
figures **UNRESOLVED as to source**, with an instruction not to re-cite them. Repointing at
`eff_flagship-30k.json` would have manufactured the exact defect this task exists to remove.
*(No conclusion moves: REF-C is 2.2–4.6× faster and the flagship misses 10 Hz at p99 in all three
precisions on every version of the numbers.)*

### 3.4 A citation wrong in two ways at once

§1.1 cited `gate_step{1k,5k,10k}.json`. The brace is the obvious defect. **The stem is also wrong**:
MEASURED from the emitters — `stack/scripts/watch_gates.py:213` and
`stack/scripts/evaluate_checkpoint.py:201` both write `f"gates_step{step}.json"` — the real name is
`gates_step<full-integer>.json`, plural, no `k`. **No `gate_step*` writer exists anywhere.**

⛔ I did **not** rewrite it to `gates_step1000.json`. pod2 is terminated and the dir is absent from
the rescue dump, so neither the filenames nor the gate steps can be verified. It is the single
`unresolved` entry on the ratchet (§4).

While there, §1.1's Location row was made honest: the run dir is gone with pod2; the **only**
reachable copies are `ckpt.pt` on the dev box
(`_pod_backup/pod2-2026-08-03/ckpts/flagship4b-phase0-30k_ckpt.pt`, 3 302 176 350 B, md5
`74be81035699c362e2fd0e5197880506`) — ⚠️ **git-ignored by `.gitignore:48`, so NOT in the repo** —
and HF `Sayood/tanitad-flagship-4b-phase0`. `config.json`, `train_log.jsonl` and the gate JSONs have
**no reachable copy at all**.

### 3.5 Six that were never citations

Allowlisted with one line of justification each in `tools/registry_paths_allow.json`: `ep_*.pt` (the
per-episode filename **contract**), `eval_*.py` (the *rule* "only `eval_*.py` output may be quoted" —
naming one script would break the rule it states), `windows_*.pt` (one of its three sites asserts an
**absence**, which cannot be given a filename), `results/<key>.json` (a CLI template),
`?? stack/tanitad/refs/refc.py` (verbatim `git status` output; `??` is git's marker), and
`…_vs_refc-{base,xl}-30k.json` (§4.2 quotes it **as the worked example** of the rule that bans it).

---

## 4. The standing check (P2) — an allowlist is normally the rug; two mechanisms stop this one

`tools/registry_paths.py` found the defects. It could not stop them coming back.

**1. Entries excuse counted SITES, not a token forever.** Each declares `occurrences`. Re-introduce
the token at a *new* site → `ALLOW_COUNT_MISMATCH` → **exit 1**. Token gone → `ALLOW_STALE` →
**exit 1**. Without this, one allowlist line would licence the same defect document-wide, which is
exactly how seventeen accumulated.

**2. Unrepairable citations are ratcheted, never closed.** `status: unresolved` keeps a dead citation
visible and counted; the file declares `max_unresolved` (**currently 1**). Adding one without raising
it **fails**; raising it is a one-line reviewable diff.

Exit codes are three-valued so CI can express the difference: **1** hard defect, **2** only
known-`UNRESOLVED` remain, **0** clean. `--strict` makes 2 and `AMBIGUOUS` fatal too.

The wiring that makes it fail loud:
`tools/tests/test_registry_paths_allow.py::test_real_registry_has_no_dead_or_malformed_citations`
runs the sweep against the **real** registry inside `pytest -q`, and its failure message names the
offenders and tells the reader **not** to fix it by allowlisting.

A deliberate design note: a **synthetic** registry (any fixture) is swept against an *empty*
allowlist. Judging fixtures against the real document's excuses would report the whole allowlist as
stale on every fixture — and a check that cries wolf on its own fixtures gets switched off.

### 4.1 The simulation caught three defects I was about to introduce myself

Before touching the live file I replayed all edits against an off-drive checkout and re-extracted the
citations. It found that **my own replacement text** introduced two new `NOT_A_PATH` tokens
(`` `ckpt_step<full-integer>.pt` ``, `` `gates_step<full-integer>.json` `` — angle brackets are glob
syntax to the classifier) and one `AMBIGUOUS` (`` `Sayood/tanitad-refb-speed/ckpt.pt` `` resolves
against every `ckpt.pt` in the tree). A fourth surfaced the same way: my M4 repair had put the URL
inside a backticked token beginning `HEAD `, which `extract_citations` does **not** skip as a URL.
All four are fixed. **Fixing 21 defects while adding 4 would have been a poor trade.**

---

## 5. The five stranded pod paths (P3)

Read-only probes only; nothing copied; `df` used for nothing.

**Host reachability, MEASURED 2026-08-03** (native Windows OpenSSH, `-n`, `BatchMode`):
`tanitad-pod4` ✅ up (training `flagship-v1arch-v2bal-30k`) · `tanitad-pod` ⛔ refused ·
`tanitad-pod3` ⛔ refused · `tanitad-eval` ⛔ refused (its IP `69.30.85.106` is now also
`tanitad-pod5`'s, different port) · `tanitad-pod2` terminated.

| # | cited path | what depends on it | status |
|---|---|---|---|
| 1 | `/workspace/experiments/refc_anchors_full.pt` | **§4.1 REF-C-XL and §4.3 REF-C-base** — the anchor set both were trained against, so what **every REF-C trajectory number** is scored against | Host pod3 unreachable. MEASURED: no `*anchor*` anywhere under pod4's rescue dump. **No HF copy** — anonymous HF API: both refc repos hold only `.gitattributes, README.md, ckpt.pt, config.json, metrics.json`. ⭐ **REBUILDABLE**: `stack/scripts/build_refc_anchors.py` is in the repo and the run `config.json` records `anchors {n 256, pool 4096, seed 0}` over the parity corpus |
| 2 | `/workspace/experiments/flagship-v16-ab-ft/eval_v16.json` | the v1.6 A/B fine-tune eval | pod2 terminated; MEASURED: no `*v16*` under the rescue dump. The **window dump survives in-repo** (`taniteval/results/windows_flagship-v16-ab-ft.pt`) |
| 3 | `/workspace/ops/heartbeats/flagship-v2corpus-30k.json` | that run's liveness record | pod2 terminated; MEASURED: no `ops/`, no `*heartbeat*`. ⚠️ **The run itself WAS rescued** — `pod4:/workspace/rescue/experiments/flagship-v2corpus-30k/` has `ckpt.pt` 3 415 808 330 B + `config.json` + `supervisor.log` + `train_log.jsonl`. Least load-bearing of the five |
| 4–5 | `/root/refb_orig_backup.py`, `/root/refb_train_orig_backup.py` | REF-B pre-patch trainer provenance | MEASURED: absent from the rescue dump and from `/root` on pod4; originating host unreachable |

**#1 is the one worth chasing, and the useful framing is that it is *reconstructible, not lost*** —
which turns a durability panic into a 0-GPU verification task. **Escalated in §8.**

---

## 6. Numbers, not paths (P4)

The brief's examples are the same root class as a brace expansion: **a quantity printed without the
qualifier that identifies it.** Findings, all MEASURED:

**6.1 The two "8.4 %" do not merely collide — one is arithmetically mislabelled.**
§1.6 read *"unfreezing 4 ViT blocks buys only **8.4 %** of the fan gap"*. The fan moved
**0.3073 → 0.2815** (Δ 0.0258). That is:

| denominator | value |
|---|---|
| relative to the starting fan | **8.40 %** ← the number that was printed |
| distance to REF-C-**XL**'s oracle-in-fan **0.1640** | **18.00 %** |
| distance to that gate's own `oracle ≤ 0.22` | **29.55 %** |

So 8.4 % is the **relative change**, *not* a fraction of any gap — and the sentence named no
denominator. Corrected in place, with all three denominators shown, plus a pointer to the **other**
8.4 % (§4.1's REF-C v1.2 re-scorer, which genuinely *is* a fraction of the **oracle** gap). That
second site now carries the `ORACLE` qualifier. ⚠️ Also flagged inline: **`0.1640` is REF-C-XL's;
base's is `0.1914`** — the same defect that put XL's numbers on a base row.

**6.2 A new instrument: `tools/registry_numbers.py`.** §6's leaderboard already carried
`<!-- src: file#key -->` provenance comments that **nothing ever verified**. It now resolves each and
compares against the number printed in the same cell — point estimate *and* `[lo, hi]` — at the
document's own precision, so rounding is never a defect and a wrong digit always is.

**Result: 14/14 MATCH — 14 point estimates and 25 interval bounds, all exact.** §6's leaderboard is
clean. It deliberately does **not** check unannotated numbers: inferring an unannotated figure's
source is how a wrong number becomes a sourced one. The remedy is to annotate.

**6.3 Bare intervals — and a number I nearly reported wrong.** A first scan said **48 of 56** `X ± Y`
figures carry no estimator. That would have been alarming and false: every **table** that uses the
`±` form labels it in its header (§1.4's is explicit: *"`overlapping_holdout_se` (DEPRECATED)"*).
Restricted to **prose**, the true count is **2**, both now labelled — §1.3's `6.179 ± 1.2845` (the
decision-grade read is **5.9396 [4.3273, 7.6249]**) and §1's closed-loop `1.685 ± 0.098`, whose
estimator is **flagged as not stated rather than guessed**. *(Verify before alarming, again.)*

---

## 7. ⛔ The blocker: G: went into a hard read stall, and `pytest` was not run

At **23:30 local**, mid-task, every content read on `G:` began failing with *"Invalid request
code" / "Incorrect function"* — including `.git`. Metadata (`ls`) still worked.

**Root cause, MEASURED from the Drive client's own log**
(`…/Google/DriveFS/Logs/drive_fs.txt`): `RESOURCE_EXHAUSTED: Too many open files`, and consequently
`content_cache.cc:601:OpenInputStreamInternal Failed to open file …` on every read. `GoogleDriveFS`
(pid 4240) sat at **10 995 open handles**, by far the top consumer on the machine, and released only
~35 over 25 minutes. **This is local FD exhaustion in the Drive client — not a network outage and
not data loss.** It does not self-heal on a useful timescale.

**Consequences, stated honestly:**
1. **`pytest -q` was NOT run**, in `stack/` or in `tools/`. The brief asked for exact counts; I do
   not have them and will not invent them. The new tests were validated against a synthetic tree and
   an off-drive checkout: **15/15** for `test_registry_paths_allow.py`, **10/10** for
   `test_registry_numbers.py`.
2. **All 19 registry edits were validated but not all applied.** Five landed before the stall (the 4
   MISSING repairs + the R10 correction). The remaining 14 are staged as a fail-loud, idempotent
   script — §8.
3. I did **not** restart the Drive client. It would remount `G:` under other agents' in-flight
   writes and risk unflushed edits of my own. **That call is the PI's.**

---

## 8. Deliverable manifest

⚠️ **Everything marked `scratchpad:` lives in ONE place and is not yet in the repo** — because the
repo was unreadable. Landing it is one command: `bash <scratchpad>/apply_all.sh`.

| artifact | where | in only one place? |
|---|---|---|
| 5 citation repairs (M1–M4, R10) | `repo:Project Steering/MODEL_REGISTRY.md` | no — applied |
| the other 14 edits | `scratchpad:apply_registry_edits.py` | **YES** — validated `19 would apply, 2 already applied` against an off-drive checkout of the registry |
| `registry_paths.py` v2 (allowlist + ratchet + 3-valued exit) | `scratchpad:registry_paths_new.py` | **YES** |
| `registry_paths_allow.json` (9 entries, reason + count each) | `scratchpad:registry_paths_allow.json` | **YES** |
| `test_registry_paths_allow.py` (15 tests) | `scratchpad:` | **YES** |
| `registry_numbers.py` (P4 instrument) | `scratchpad:registry_numbers.py` | **YES** |
| `test_registry_numbers.py` (10 tests) | `scratchpad:` | **YES** |
| landing script | `scratchpad:apply_all.sh` | **YES** |
| this report | `scratchpad:REGISTRY_CITATIONS.md` | **YES** |
| sweep before | `scratchpad:sweep_before.json` (252 citations) | **YES** |

## 9. Escalations

1. 🔴 **G: is broken on the dev box and needs a `GoogleDriveFS` restart** (§7). Until then nothing
   can be staged and no test can run. **PI decision** — I did not restart it under other agents'
   in-flight writes.
2. 🟠 **`refc_anchors_full.pt` is reconstructible, not lost** (§5). The 0-GPU work item is to rebuild
   it from `stack/scripts/build_refc_anchors.py` with `{n 256, pool 4096, seed 0}` and check it
   against what the banked `refc_anchors_small64.pt` implies. **Every REF-C trajectory number is
   scored against this set.**
3. 🟠 **§6's six latency figures are unsourced** (§3.3) and cannot be repaired from the repo. They
   need a re-measurement on an idle A40, or §6 restated from the committed JSONs.
4. 🟡 **`flagship4b-phase0-30k`'s only local `ckpt.pt` copy is git-ignored** (§3.4). That is
   deliberate (3.3 GB), but it means the ablation control that anchors the whole speed-input causal
   claim is one dev-box disk plus one HF repo.
5. 🟡 **Wire both checks into CI.** `tools/registry_paths.py` (exit 1 hard / 2 known-unresolved) and
   `tools/registry_numbers.py` (exit 1) are `pytest`-covered, but nothing outside `pytest` runs them.
