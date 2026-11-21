# pod3 and tanitad-eval are v5-capable — and the guard's first field test found four defects

*2026-07-27. Closes `STALE_IMPORT_GUARD.md` §8 escalation 2 (the fleet-capacity constraint) and
`FLEET_REFILL.md` §2.5 item 2 (the seed-0-only checkpoint).*

⛔ **pod2 WAS NOT TOUCHED** — not probed, not signalled, not logged into. The small validation's
chain (3695397), trainer (3695401) and armed eval chain (3696611) are untouched by me.
⛔ **pod1 WAS NEVER CONTACTED.**
Work happened on **`tanitad-pod3`** and **`tanitad-eval`** only, both of which were idle (GPU 0 %,
0 MiB) at start.

All host times are **UTC**; the dev box reads Europe/Berlin (UTC+2).

---

## 0. HEADLINE

⭐⭐ **1. THE CONSTRAINT IS GONE. Both pod3 and tanitad-eval now pass
`python3 -m taniteval.stack_check --require v5` with exit 0.** pod2 is no longer the only
v5-capable eval host. §2 has the full output for each host.

⭐⭐ **2. THE FIELD TEST WAS WORTH MORE THAN THE SYNC — four defects, all MEASURED, all
reproducible.** The brief predicted this and it was right:

| # | defect | class |
|---|---|---|
| **F1** ⛔ | **a GREEN does not say what it probed.** `--require v5` (5 capabilities checked) and a bare `stack_check` (**zero** checked) print **byte-identical** summaries: `"ok": true, "problems": []`. A dropped or misspelled flag reads as a pass. | *the same shape the guard exists to close* — **FIXED here + test** |
| **F2** ⛔ | **`--require v5` is a CAPABILITY probe, not an IDENTITY probe.** `tanitad-eval:/root/vtband/stack` passes with **exit 0** while its `heldout_gate.py` is md5 `8c146f82`, **not** HEAD's `c2882830`. | escalated (§4.2) |
| **F3** ⛔ | **the harness-shadowing class is real and it bit on the first try.** `idm2_lib.py:19` and `idm3_arms.py` each run an unconditional `sys.path.insert(0, "/root/taniteval")` at import, which **silently defeated my explicit estimator pin**. `stack_check` cannot see it — it pins `tanitad`, not `taniteval`. | escalated (§4.3) |
| **F4** ⚠️ | **a third exit code, 1, that the contract does not document.** When the resolved `taniteval` has no `stack_check`, the probe **never runs** and exits **1** — after printing the reassuring `[taniteval] tanitad OVERRIDE -> …` banner, which the **pre-guard** harness prints too. | escalated (§4.4) |

⭐ **3. THE SECOND-TREE CLASS IS FLEET-WIDE, AND IT IS THE ESTIMATOR.** I checked what the brief
asked and found more: across the two hosts there are **12 `taniteval/ci.py` copies and 19 `tanitad`
trees. Exactly 2 of each — the two I synced — are at HEAD.** The other **10 `ci.py` copies are all
the old `ef925f06` estimator**. pod3's `/root/taniteval` is a **symlink created 2026-07-27 05:37**
pointing at a stale harness; tanitad-eval's `/root/taniteval` is a real **224 MB** stale tree and is
the **deployed** harness. §3.

⭐⭐ **4. THE 3-SEED ENSEMBLE IS BUILT, MEASURED AND STAGED — and it reproduces from the file.**
See §6–§10.

⛔ **5. NOTHING WAS PUSHED TO HF.** The ensemble checkpoint is staged in the repo. §10.6.

---

## 1. WHAT WAS SYNCED, AND WHY NOT IN PLACE

⚠️ **THE REPO ADVANCED WHILE THIS RAN.** The brief named HEAD `37ccfea`; the orchestrator committed
**`8ab5327`** ("comma yaw_rate re-issued…") mid-session. **Both hosts were re-synced to `8ab5327`
and re-probed** — every transcript in §2 is at `8ab5327`, not at the briefed SHA.
⭐ **That is a bonus, not a complication:** `8ab5327` swept in my F1 fix, so the field test in §2 now
runs *against the fixed guard on a real host* and demonstrates the fix rather than only asserting it.
**MEASURED:** `git diff 37ccfea 8ab5327` touches **none** of the five capability modules, so nothing
in §4's findings is invalidated by the move.

`origin/main` is `2d903ba`, **26 commits behind** the branch. Both hosts fetched
`+refs/heads/*:refs/remotes/origin/*` (their `remote.origin.fetch` was pinned to `main` only, so a
plain `git fetch` silently brought nothing — **MEASURED**, it returned rc 0 and
`origin/agent/benchmarks-eval-20260721` was still an unknown revision). Then each checked out the
branch tip **by SHA**.

| host | tree now at HEAD | how | the pre-existing tree, and why it was LEFT |
|---|---|---|---|
| **tanitad-pod3** | **`/workspace/TanitAD-main`** @ **`8ab5327`**, `git status` **clean** | existing clean worktree moved `2d903ba` → `37ccfea` → `8ab5327` | `/workspace/TanitAD` stays at `0f93b98` with **185 changed entries under `stack/` + `taniteval/` alone (69 modified tracked + 116 untracked)**. `FLEET_REFILL.md` §2.1 measured 33 files / 3,680 insertions that survive `--ignore-all-space`, i.e. real local edits. Resetting it would destroy them. |
| **tanitad-eval** | **`/workspace/TanitAD-head`** @ **`8ab5327`**, `git status` **clean**, 3,301 files, 1.2 GB | **new worktree** of `/root/TanitAD` | `/root/TanitAD` stays at `0f93b98` on branch `main` with 61 dirty entries. ⚠️ **It is also a truncated checkout — `git ls-files` = 357, against 3,301 at HEAD.** It is the deployed tree and the brief forbids breaking the deployed eval path. |

⚠️ **Consequence, stated plainly rather than papered over:** on both hosts the *canonical* path
(`/workspace/TanitAD/stack`, `/root/TanitAD/stack`) is **still pre-v5**. Anyone pasting the published
v5 command against those paths now gets **exit 2 with the five missing symbols named** — which is the
correct outcome and a loud one, not a wrong number. The v5-capable paths are the two in the table.
**Whether the canonical paths should be force-synced is a decision for whoever owns those local
edits** (§11, escalation 1).

⚠️ **Disk was judged with a real `dd`, never `df`:** pod3 `/workspace` 500 MiB @ **508 MB/s**;
tanitad-eval `/workspace` 500 MiB @ **449 MB/s**, `/root` 200 MiB @ **4.5 GB/s**. No quota pressure.

---

## 2. ⭐ THE CAPABILITY PROBE, PER HOST — pasted, not paraphrased

Full transcripts: `raw/guard_field_test_pod3.txt`, `raw/guard_field_test_eval.txt`.
Banked reports: `raw/stack_guard_v5_pod3_GREEN.json`, `raw/stack_guard_v5_eval_GREEN.json`
(and the two RED twins).

### 2.1 `tanitad-pod3` — **EXIT 0**

```
$ cd /workspace/TanitAD-main/stack && \
  TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD-main/stack \
  PYTHONPATH=/workspace/TanitAD-main/stack:/workspace/TanitAD-main/stack/scripts:/workspace/TanitAD-main/taniteval \
  python3 -m taniteval.stack_check --require v5

[taniteval] tanitad OVERRIDE -> /workspace/TanitAD-main/stack (/workspace/TanitAD-main/stack/tanitad/__init__.py)
{
  "pinned_root": "/workspace/TanitAD-main/stack",
  "provenance": "explicit-argument",
  "mode": "error",
  "tanitad_file": "/workspace/TanitAD-main/stack/tanitad/__init__.py",
  "required": [
    "tanitad.train.heldout_gate:PRIMARY_NAME",
    "tanitad.train.heldout_goal:make_goal_kwargs_fn",
    "tanitad.data.parity:register_v2_geometry_sibling",
    "tanitad.geometry:frame_from_args",
    "train_flagship_v4:resolve_v2_frames"
  ],
  "ok": true,
  "problems": []
}
EXIT=0
```

⭐ The `required` block is **the F1 fix, running on a real host** — at the briefed `37ccfea` this
same GREEN printed no `required` at all and was byte-identical to a probe that checked nothing (§4.1).
The banked JSON additionally records `sentinel_installed: true`, `legacy_present: false`.

### 2.2 `tanitad-eval` — **EXIT 0**

```
$ cd /workspace/TanitAD-head/stack && \
  TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD-head/stack \
  PYTHONPATH=/workspace/TanitAD-head/stack:/workspace/TanitAD-head/stack/scripts:/workspace/TanitAD-head/taniteval \
  python3 -m taniteval.stack_check --require v5

[taniteval] tanitad OVERRIDE -> /workspace/TanitAD-head/stack (/workspace/TanitAD-head/stack/tanitad/__init__.py)
{
  "pinned_root": "/workspace/TanitAD-head/stack",
  "provenance": "explicit-argument",
  "mode": "error",
  "tanitad_file": "/workspace/TanitAD-head/stack/tanitad/__init__.py",
  "required": [
    "tanitad.train.heldout_gate:PRIMARY_NAME",
    "tanitad.train.heldout_goal:make_goal_kwargs_fn",
    "tanitad.data.parity:register_v2_geometry_sibling",
    "tanitad.geometry:frame_from_args",
    "train_flagship_v4:resolve_v2_frames"
  ],
  "ok": true,
  "problems": []
}
EXIT=0
```

### 2.3 ⭐ The RED twins — the probe is not a rubber stamp on either host

⚠️ **A GREEN alone is not evidence the probe works**; both hosts were also run against their own
pre-v5 tree in the same session.

| host, tree | verdict |
|---|---|
| pod3 `/workspace/TanitAD/stack` | **EXIT 2**, all five named, incl. the subtle one: `train_flagship_v4:resolve_v2_frames -> module resolved from …/scripts/train_flagship_v4.py has NO attribute 'resolve_v2_frames' (a PRE-v5 tree looks exactly like this)` |
| tanitad-eval `/root/TanitAD/stack` (**the deployed tree**) | **EXIT 2**, all five named |
| pod3 `/workspace/rigfix/stack_head` | **EXIT 2** — carries `heldout_gate` but not `heldout_goal`; a *partial* v5 tree is caught |

**⇒ neither host's v5 pass is an artefact of a permissive probe.** MEASURED, both directions,
same session, same command shape.

---

## 3. 🔴 THE SECOND-TREE QUESTION — asked, and the answer is worse than pod2's

The brief asked whether pod3 / eval carry an equivalent of pod2's stale `/root/taniteval`.
**Both do**, and a full sweep (`raw/fleet_tree_inventory.txt`, read-only) says the class is fleet-wide.

| host | `/root/taniteval` | what it is |
|---|---|---|
| **tanitad-pod3** | ✅ exists | a **symlink → `/workspace/idmretrain/taniteval`**, created **2026-07-27 05:37** by the IDM retrain setup. Stale `ci.py` (`ef925f06`). |
| **tanitad-eval** | ✅ exists | a **real 224 MB directory**, no `.git`, 79 `.py` at depth ≤ 3, containing a `taniteval/` package. Stale `ci.py`. ⛔ **This is the DEPLOYED harness on the host that produced every published closed-loop number.** |

### 3.1 The sweep — MEASURED, `raw/fleet_tree_inventory.txt`

| | tanitad-eval | tanitad-pod3 | **total** |
|---|---:|---:|---:|
| `taniteval/ci.py` copies | 8 | 4 | **12** |
| …at HEAD (`c92618a0`) | **1** | **1** | **2** |
| …stale (`ef925f06`) | 7 | 3 | **10** |
| `tanitad` trees | 14 | 5 | **19** |
| …at HEAD `heldout_gate.py` (`c2882830`) | **1** | **1** | **2** |
| …carrying a **different** `heldout_gate.py` | 2 (`8c146f82`, `64f40c37`) | 1 (`e545e2f6`) | **3** |
| …pre-v5 | 11 | 3 | 14 |

⭐ **`ci.py` is not an ordinary file — it is the program's `CI-separated` predicate.** The two
versions differ: `episode_cluster_bootstrap` is **byte-identical**, and
`paired_episode_cluster_bootstrap` differs **only in display rendering** (the 2026-07-27 adaptive-
precision fix for `{"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": true}`). **The statistics are
unchanged** — verified by reading the diff, not assumed. So this is not a silent-wrong-number today;
it is the *mechanism* for one, sitting under the decision-grade estimator on 10 of 12 copies.

---

## 4. ⭐⭐ THE GUARD'S FIRST FIELD TEST — four defects

### 4.1 F1 ⛔ **A GREEN DOES NOT SAY WHAT IT PROBED** — FIXED HERE

**MEASURED on pod3**, same host, same tree, same second, at the briefed `37ccfea`:

| command | capabilities actually probed | printed summary **at `37ccfea`** | **at `8ab5327` (fixed)** |
|---|---:|---|---|
| `… stack_check --require v5` | **5** | `"ok": true, "problems": []` | `"required": [5 entries], "ok": true` |
| `… stack_check` | **0** | `"ok": true, "problems": []` — ⛔ **byte-identical** | `"required": [], "ok": true` ✅ distinguishable |

⭐ **Both halves are banked as captured host output, not reconstructed:**
`raw/guard_field_test_pod3_PREFIX_37ccfea.txt` is the RED (its §1 and §2 are byte-identical), and
`raw/guard_field_test_pod3.txt` is the re-run at `8ab5327` where §1 carries the five `required`
entries and §2 carries `"required": []`. **The fix is demonstrated on a real host, not only in a
unit test.**

The banked JSON always carried `required`; **only the human surface lied** — and the human surface is
the whole product here, because the guard's value proposition is *"paste this one line in front of
your eval and read the answer"*.

⚠️ **This is the guard's own failure shape, one level up:** a GREEN that looks the same whether or
not the check happened. `STALE_IMPORT_GUARD.md` §2.4 says *"a probe list that does not describe the
real tree is a rubber stamp"*; this is the adjacent case — **a probe invocation that checked nothing,
indistinguishable from one that checked five.**

**FIX (minimal, display-only, no exit code and no statistic changed):** `stack_guard.main()` now
prints `required`, on **both** the pass path and the refusal path.
**Pinned by two new tests** (`taniteval/tests/test_stack_guard.py`):
`test_a_GREEN_says_what_it_PROBED_and_a_no_require_GREEN_does_not` (asserts the two outputs are now
different, `len(required) == 5` vs `[]`) and
`test_a_capability_REFUSAL_also_prints_what_was_demanded` (the refusal path builds its report
separately and needed its own pin).

### 4.2 F2 ⛔ **CAPABILITY ≠ IDENTITY** — escalated, not silently redesigned

**MEASURED on tanitad-eval** (`raw/guard_field_test_eval.txt` §3):

```
$ TANITEVAL_STACK_OVERRIDE=/root/vtband/stack … python3 -m taniteval.stack_check --require v5
  "pinned_root": "/root/vtband/stack",
  "ok": true,
EXIT=0
```

while

```
/root/vtband/stack/tanitad/train/heldout_gate.py   md5 8c146f82ec08fae2856df04e379076c7
/workspace/TanitAD-head/…/heldout_gate.py (HEAD)   md5 c2882830e34a13ef75c78c83d861bac7
```

**Three trees on the fleet carry a `heldout_gate.py` at three different md5s and all would be called
"v5" by the symbol list.** `pinned_root` in the JSON *does* let a careful reader see which tree ran,
so this is not a wrong number today — but **`exit 0` alone does not mean "the pinned release"**, and
that is exactly what an operator will read it as.

⇒ **DECISION OWED** (§10): should `--require v5` also assert an *identity* (a commit SHA / a content
digest of the capability modules), or is capability-only the intended contract? Adding identity
would be a semantics change to a guard that is about to sit in front of every v5 eval, so I did
**not** make it unilaterally.

### 4.3 F3 ⛔ **THE HARNESS-SHADOW CLASS IS REAL — it silently defeated my own pin on the first try**

This is `STALE_IMPORT_GUARD.md` §3.4 residual 1 / §8 escalation 3, observed in the field within one
hour of the guard existing.

I set out to run the IDM job against the **pinned** estimator, and put
`/workspace/TanitAD-main/taniteval` at `sys.path[0]`. **MEASURED, it did not hold:**

```
taniteval.ci -> /root/taniteval/taniteval/ci.py      (md5 ef925f06 — the STALE tree)
```

**Cause, verified in code:** `idm2_lib.py:19` and `idm3_arms.py` each execute, at import time,

```python
sys.path.insert(0, "/root/taniteval")
```

unconditionally — jumping the stale tree back in **front** of the caller's pin. No error, no warning.

⚠️ **`python3 -m taniteval.stack_check --require v5` returns `ok: true` throughout this.** It pins
`tanitad`. It cannot see `taniteval`, and `taniteval.ci` is where every interval in the program
comes from.

**Cure applied in this deliverable** (`code/idm5_ensemble.py`) — the same one
`TANITEVAL_STACK_OVERRIDE` uses for `tanitad`: import the pinned package **first** so `sys.modules`
caches it, because the module cache beats every later `sys.path.insert`; then **assert the pin
survived**:

```python
assert L.tci.__file__ == _PINNED_CI, "ESTIMATOR PIN DEFEATED: …"
```

**MEASURED after the cure:** `taniteval.ci -> /workspace/TanitAD-main/taniteval/taniteval/ci.py`
(`c92618a0`), and `idm2_lib` uses the same file. Every interval in §8 comes from the pinned tree.

### 4.4 F4 ⚠️ **A THIRD EXIT CODE THE CONTRACT DOES NOT DOCUMENT**

**MEASURED on tanitad-eval** (`raw/guard_field_test_eval.txt` §4), against
`/workspace/_egoin/lib`, which carries **both** a `tanitad/` and a stale `taniteval/`:

```
[taniteval] tanitad OVERRIDE -> /workspace/_egoin/lib (/workspace/_egoin/lib/tanitad/__init__.py)
/usr/bin/python3: No module named taniteval.stack_check
EXIT=1
```

Two things are wrong here and both matter:

1. **The documented contract is "exit 0 / exit 2".** `exit 1` means **the probe never ran**. A
   wrapper written as `stack_check … ; [ $? -ne 2 ] && proceed` proceeds. Any operator habit built
   on "2 is the bad one" is unsafe.
2. ⭐ **The `[taniteval] tanitad OVERRIDE -> …` banner is printed by the PRE-guard harness too.**
   Seeing it is **not** evidence the guard is armed. Here it printed, and then nothing was checked.

---

## 5. SUITES — zero new skips

Run twice — once at the briefed `37ccfea`, and again after the repo advanced to `8ab5327`.

| suite | brief's baseline | **at `37ccfea` + my changes** | **at `8ab5327` (current tree)** | new skips |
|---|---|---|---|---|
| `stack/` | 1534 passed, 12 skipped | ✅ **1534 passed, 12 skipped** (159.7 s) | ✅ **1557 passed, 12 skipped** (109.1 s) | **0** |
| `taniteval/` | 661 passed | ✅ **663 passed** (101.4 s) | ✅ **663 passed** (84.9 s) | **0** |

`taniteval` **+2 = my two new guard tests** (§4.1). `stack` **+23 is not mine** — it is the sibling
`comma2k19` work that arrived in `8ab5327` and in the still-staged `stack/tanitad/data/comma2k19.py`.
**Skips are unchanged at 12 in every run.**

Run with `C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -q`.

---

# JOB 2 — the IDM `steer` head: the 3-seed ensemble is built, and the seed-0 file would have MISSED

## 6. ⛔ THE FINDING THAT JUSTIFIES THE WHOLE JOB

`FLEET_REFILL.md` §2.5 established rung-757 `steer` **R² +0.7993**, separated on both corpora. But
`idm4_steer.py:295` saves a checkpoint only `if … sd == a.seeds[0]`, so the staged
`idm_head_v4_steer.pt` is **seed 0 alone** while the headline is the **3-seed ensembled prediction**.

I trained all three seeds and scored **both** arms on the identical paired read. **MEASURED
(ours) · DECISION-GRADE**, `raw/idm5_ensemble.json`:

| arm | steer ΔMAE vs A0 — **PhysicalAI** (n=14 ep) | steer ΔMAE vs A0 — **comma2k19** (n=22 ep) |
|---|---|---|
| ⭐ **3-seed ensemble** | **−0.0042 [−0.0066, −0.0013]** ✅ **SEPARATED** | **−0.0030 [−0.0048, −0.0017]** ✅ **SEPARATED** |
| ⛔ **seed 0 only — the currently staged file** | **−0.0025 [−0.0056, +0.0007]** ⛔ **NOT SEPARATED** | −0.0026 [−0.0043, −0.0012] ✅ separated |

⭐⭐ **The staged seed-0 checkpoint DOES NOT MEET THE SHIP BAR ON PhysicalAI.** The claim
*"separated independently on both corpora"* is **true of the ensemble and false of the artifact that
was on disk**. Shipping that file under the ensemble's number would have been a fabricated result —
the brief's warning was not hypothetical, it is now measured.

Its R² is short too: `steer` **+0.7866** pooled / **+0.7737** pai / **+0.7663** cm, against the
ensemble's **+0.7993 / +0.7858 / +0.8071**.

## 7. 🔒 PROTOCOL — stated explicitly, because a `steer` number is only comparable within one

| | |
|---|---|
| **labels** | ⭐ **REPAIRED** — `idm3_labels.heading_repair`, **`v_min = 0.5 m/s`**, extended to the linked-in `cmx_` extras via `repair_labels_ext` (stock `repair_labels` tests `tag.startswith("cm_")`, which is False for the extras and would have left 79 comma episodes on the broken arctan2-at-standstill heading — a mixed protocol inside one train set). 50 val yaw labels rewritten. ⚠️ The repair rewrites **yaw/heading**, not `steer`; it is applied for protocol identity with v3 and with the 2026-07-27 retrain. |
| **recipe** | R0 — k=4 (9 frames), `d_model` 256, no winsorisation, no clip context, 50 epochs, AdamW 3e-4 / wd 0.01, cosine. |
| **rung** | **757 episodes** (cm 121 / pai 636), **141,628 train windows** |
| **val** | **36 episodes / 4,195 windows** — frozen, byte-identical to v3's and to A0's stored predictions (asserted at runtime: `a0["S"].shape[0] == n_val`) |
| **estimator** | ⭐ `taniteval.ci.paired_episode_cluster_bootstrap`, unit = **EPISODE**, **B = 2000**, **per corpus, never pooled alone**. `overlapping_holdout_se` **never called**. Source **PINNED and ASSERTED** to `/workspace/TanitAD-main/taniteval/taniteval/ci.py` (md5 `c92618a0`, = repo HEAD) — see §4.3 for why that assertion has to exist. |
| **`steer` is not one quantity** | PhysicalAI `atan(L·κ)`; comma2k19 `STEER_RATIO = 15.3`. Pooled numbers are reported only for continuity with the model card. |

## 8. ⭐ THE LEAK CHECK — by CONTENT, with the path and the count

**Path checked: `/root/idm2/lat`** (874 `.pt` files: the 104-episode v3 cache plus the 770 extras
linked in under `cmx_` / `paix_`). Method: **md5 of each episode's float32 `poses`** — the latents
store only `{z, poses, actions}`, with **no `episode_id` and no `src`**, so identity cannot be read
off metadata and a filename check would prove nothing.

**MEASURED**, `raw/idm5_ensemble.json → leak_check`:

| quantity | value |
|---|---:|
| candidate extra episodes fingerprinted | **770** |
| val episodes | **36** |
| ⭐ **val episodes found in the pool BY CONTENT and EXCLUDED** | ⭐ **4 (11.1 %)** |
| internal duplicates dropped | **77** |
| final training pool | **757** (cm 121 / pai 636) |
| ⭐ **residual content overlap between pool and val** | ⭐ **0** — asserted at runtime, the run aborts otherwise |

The four, by content (extra → the val episode it *is*):
`cmx_00008 → cm_00018` · `cmx_00020 → cm_00039` · `paix_00600 → pai_00000` · `paix_00612 → pai_00018`.

⚠️ **Without this check the run leaks 4 of its 36 val episodes** — the REF-A I-JEPA failure mode at
11 % instead of 80 %. The exclusion is enforced by two runtime assertions (tag overlap **and**
fingerprint overlap) and recorded in the output JSON.

⚠️ **Reconciliation with `FLEET_REFILL.md` §2.3's "62 internal duplicates", which counts the same
thing differently** and is *not* a discrepancy: 62 are duplicates **within** the 770-file pod3 cache;
my 77 = those 62 **plus 15** extras that duplicate one of the 68 v3 *train* episodes.
`770 − 4 − 77 + 68 = 757` exactly, and the pool composition (**cm 121 / pai 636**) and window count
(**141,628**) are identical to the 2026-07-27 run.

## 9. ⭐⭐ THE RESULT — the ensemble, and an artifact that reproduces it

### 9.1 Reproduction first — this is not a new measurement, it is the same one

**MEASURED**, `raw/idm5_ensemble.json → reproduction_of_2026_07_27`:

| | seed 0 | seed 1 | seed 2 | **3-seed ensemble** |
|---|---:|---:|---:|---:|
| 2026-07-27 retrain | 0.7866273580445887 | 0.7857364125973343 | 0.7891320475182658 | 0.7992634007131648 |
| **here** | 0.7866273580445887 | 0.7857364125973343 | 0.7891320475182658 | 0.7992634007131648 |
| delta | **0** | **0** | **0** | ⭐ **exactly 0.0** |

**Bit-exact on all three seeds and on the ensemble.** ⭐ And it was produced through the **pinned
HEAD `ci.py`** while the original ran on the stale `ef925f06` copy — so this doubles as a MEASURED
confirmation that the two estimator versions agree on this read (§3.1's "display-only" claim, tested
rather than asserted).

### 9.2 The channels — per corpus (**MEASURED (ours)** · DECISION-GRADE)

| channel | A0 (deployed `idm_head_v1`) | **3-seed ensemble** | pai | cm |
|---|---:|---:|---:|---:|
| ⭐ **steer** | +0.7419 (pai +0.7340 / cm +0.5648) | ⭐ **+0.7993** | **+0.7858** | **+0.8071** |
| ⭐ **yaw_rate** | +0.8108 (pai +0.9035 / cm +0.3308) | ⭐ **+0.9188** | +0.9624 | +0.6948 |
| **speed** | +0.8651 (pai +0.9070 / cm +0.7590) | **+0.8650** | +0.9312 | +0.7453 |
| ~~long_accel~~ | −0.2398 | −0.0591 | −0.0369 | −0.2258 |

### 9.3 Paired ΔMAE vs A0 — episode-cluster bootstrap, B = 2000, **per corpus**

Negative = the ensemble is better. "SEPARATED" = the 95 % interval excludes 0.

| channel | **PhysicalAI** (n = 14 ep, 1,203 win) | **comma2k19** (n = 22 ep, 2,992 win) |
|---|---|---|
| ⭐ **steer** | ⭐ **−0.0042 [−0.0066, −0.0013] SEPARATED** | ⭐ **−0.0030 [−0.0048, −0.0017] SEPARATED** |
| ⭐ **yaw_rate** | ⭐ **−0.0135 [−0.0183, −0.0091] SEPARATED** | ⭐ **−0.0089 [−0.0119, −0.0061] SEPARATED** |
| speed | −0.2625 [−0.9878, +0.3366] not sep. | +0.4262 [−0.3476, +1.2179] not sep. |
| long_accel | −0.0381 [−0.1205, +0.0474] not sep. | +0.0135 [−0.0534, +0.0819] not sep. |

⭐ **`steer` and `yaw_rate` both beat the deployed head with each interval separated on its own
corpus. `speed` is unmoved and unseparated on both** — the clean control that the effect is
rotation-specific and not "more data helps everything".

⚠️ **`long_accel` STAYS UNSHIPPED.** −0.0591 beats A0's −0.2398 and is still **negative**, and its
paired delta is unseparated on both corpora. The v3 contract that excludes it is unchanged; the
0.9999 discretisation-ceiling refutation is not revisited.

⚠️ **The PhysicalAI interval rests on n = 14 episodes and is wide** ([−0.0066, −0.0013]): the *sign*
of the PhysicalAI win is established, its *size* is not well pinned. The val set is fixed at 36 by
the pairing requirement against A0's stored predictions.

### 9.4 ⭐ THE ARTIFACT REPRODUCES ITS OWN HEADLINE — checked, not assumed

The checkpoint stores **three** state_dicts plus the ensembling rule. It was then **reloaded from
disk**, re-predicted, re-ensembled, and compared to the training-time prediction:

```
RELOAD CHECK: max|delta| = 0.000e+00 -> REPRODUCES        (3 state_dicts, 34,841,012 bytes)
```

⚠️ **Why this check exists:** the ensemble is the **mean of the three heads' scalar predictions**,
not a weight average — averaging the weights of independently-seeded transformers is a different
estimator and would not reproduce the headline. Recording the rule *and* proving it from the file is
what makes the number shippable.

⚠️ **Defect in the previous artifact, fixed here:** `idm_head_v4_steer.pt` stores
`config = {"head_kwargs": {}}` — `train_arm` never writes `head_kwargs` into `meta`, so **the old
file does not record how to rebuild its own head**. The new checkpoint carries explicit
`head_kwargs` (`state_dim 2048, window 9, d_model 256, use_ctx False, side_dim 0, acc_bins 0,
input_slice [4,13], class idm3_arms.IDMHeadV3`), plus `cfg`, `scalars`, `ensemble_rule`,
`label_protocol` and `seeds`.

## 10. ⚖️ RE-SHIP RECOMMENDATION — plainly

⭐ **YES — re-ship `steer` from the 3-seed ensemble, and mark the model card's prohibition
SUPERSEDED.**

1. ✅ **The bar is met by the ENSEMBLE and only by the ensemble.** *"Do not replace `steer` unless
   the retrain beats 0.742 on a paired, episode-disjoint read"* — **+0.7993 vs +0.7419**, paired on
   4,195 identical windows, episode-disjointness **measured by content fingerprint (4 leaks caught,
   0 residual)**, and CI-separated **on each corpus independently**.
2. ⛔ **Ship `idm_head_v4_steer_ens3.pt`, NOT `idm_head_v4_steer.pt`.** The seed-0 file is **not
   separated on PhysicalAI** (§6). If a single-seed head is wanted for cost reasons it must be
   re-quoted at **+0.7866 / pai −0.0025 [−0.0056, +0.0007] not separated** — which does not clear
   the bar on PhysicalAI.
3. ⭐ **`MODEL_CARD_IDM_V3.md` §"Known failures" item 2 is SUPERSEDED.** It reads *"`steer` is WORSE
   than the previous head (0.408 vs 0.742) … Do not use v3 for `steer` — this is a data-budget
   regression, not a recipe improvement."* The **diagnosis is confirmed** (the rung-68 control
   reproduces v3 at +0.4175 vs published 0.408 — INHERITED from `FLEET_REFILL.md` §2.5) and the
   **prohibition no longer applies at ≥ 400 episodes**. Suggested replacement: *"`steer` at the v3
   budget (68 clips) is worse than the previous head. This is a data-budget effect, confirmed by a
   4-rung ladder; at 757 episodes the same recipe reaches +0.7993 and beats the previous head
   CI-separated on both corpora. Use the 757-episode 3-seed ensemble for `steer`; do not use the
   68-clip v3 head."*
4. ⭐ **`yaw_rate` is a second, unclaimed re-ship**: **+0.9188** vs A0's +0.8108 **and** above IDM
   v3's shipped **+0.841**, separated on both corpora.
5. ⚠️ **`MODEL_REGISTRY.md` needs the new head and this ladder as its provenance.** I did not edit
   the registry — it is the program's source of truth, and a registry entry for a head the PI has
   not agreed to ship would invert that relationship.
6. ⛔ **NOTHING WAS PUSHED TO HF.** Publishing beyond what is already decided was not authorised.
   The checkpoint is staged in this directory.

## 11. 🔴 ESCALATIONS — decisions, not documentation

1. ⭐⭐ **The v5-capable path on pod3 and tanitad-eval is NOT the canonical path** (`…-main` /
   `…-head`, not `/workspace/TanitAD` / `/root/TanitAD`). Someone must decide whether the canonical
   trees get force-synced — that means adjudicating **185 changed entries on pod3** and 61 on
   tanitad-eval, some of which `FLEET_REFILL.md` §2.1 measured as real local edits. **Until then the
   published v5 commands must name the new paths**, which is why §2 spells them out in full.
2. ⭐ **F2 — should `--require v5` assert IDENTITY, not just capability?** Three trees on the fleet
   carry three different `heldout_gate.py`, and one of them (`tanitad-eval:/root/vtband/stack`)
   passes with exit 0. A commit-SHA or content-digest assertion would close it. **This is a
   semantics change to a guard about to sit in front of every v5 eval — a decision, and I did not
   take it unilaterally.**
3. ⭐ **F3/F4 — the harness (`taniteval`) has no guard at all, and the class is live.** 10 of 12
   `ci.py` copies on the fleet are stale, `idm2_lib.py:19` defeats any caller pin silently, and
   `stack_check` structurally cannot see it. The cure that works is the `sys.modules`-cache pin plus
   an assertion (`code/idm5_ensemble.py`); **it should become a helper in `taniteval`, not a pattern
   each agent re-derives.** Also: `stack_check` can exit **1** (the probe never ran) — the
   documented contract says 0 or 2, and the `[taniteval] tanitad OVERRIDE ->` banner prints then too.
4. ⚠️ **`tanitad-eval:/root/TanitAD` is a truncated checkout** — `git ls-files` = **357** against
   **3,301** at HEAD. It is the deployed eval tree. Not obviously broken today, but it is not the
   repo either.
5. ⚠️ **pod3 carries 4 stale `until [ … ]; do sleep …; done` watcher loops** (PIDs 30782, 30872,
   31164, 38171) spinning **6+ days** on `/root/vlmprod/phase1/*` files that never arrived. Same
   family as the `pgrep -f` self-match watcher `FLEET_REFILL.md` §1.2 reaped. **I did not kill them
   — they are not mine and they are cheap** — but they are almost certainly dead chains.

## 12. DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| `FLEET_SYNC_IDM_STEER.md` (this) | `repo:…/incoming/2026-07-27-fleet-sync-idm-steer/` **(staged)** | no |
| ⭐ **`idm_head_v4_steer_ens3.pt`** — the 3-seed ensemble head, 34.8 MB, md5 `ab8f0e49364435fafa927a7986ea948d` | `repo:` **(staged)** + `tanitad-pod3:/workspace/idmretrain/out/idm_head_v4_steer_ens3.pt` — **md5-verified identical on both** | no |
| ⭐ `code/idm5_ensemble.py` — ensemble build + leak check + reload proof + estimator pin | `repo:` **(staged)** + `tanitad-pod3:/workspace/idmretrain/idm5_ensemble.py` | no |
| ⭐ `raw/idm5_ensemble.json` — every number in §6–§9 | `repo:` **(staged)** | no |
| `raw/idm5.log` — the full run log | `repo:` **(staged)** | no |
| ⭐ `taniteval/taniteval/stack_guard.py` — F1 fix (`required` printed on both paths) | `repo:` **(staged)** | no |
| ⭐ `taniteval/tests/test_stack_guard.py` — +2 tests pinning F1 | `repo:` **(staged)** | no |
| `raw/guard_field_test_pod3.txt`, `raw/guard_field_test_eval.txt` — the field-test transcripts incl. every RED | `repo:` **(staged)** | ⚠️ **yes** (transcripts; reproducible in one command each) |
| `raw/fleet_tree_inventory.txt` — the 12-`ci.py` / 19-tree sweep | `repo:` **(staged)** | ⚠️ **yes** (same) |
| `raw/stack_guard_v5_{pod3,eval}_GREEN.json` + the two RED twins | `repo:` **(staged)** | no |
| `raw/guard_field_test_*_PREFIX_37ccfea.txt` — the PRE-fix RED transcripts (recovered from `8ab5327`, captured host output) | `repo:` **(staged)** | no |
| pod3 tree at HEAD | `tanitad-pod3:/workspace/TanitAD-main` @ **`8ab5327`**, clean | n/a — a checkout |
| eval tree at HEAD | `tanitad-eval:/workspace/TanitAD-head` @ **`8ab5327`**, clean | n/a — a checkout |

**Nothing is left running.** The IDM job (**PID 1312766**, `tanitad-pod3`) exited cleanly —
`IDM5_DONE 1871.2s` — and the PID is gone. Log: `tanitad-pod3:/workspace/idmretrain/out/idm5.log`
(pulled to `raw/idm5.log`). **pod3 and tanitad-eval are both idle again** (GPU 0 %, 0 MiB).

**I ran no `git commit`, no `git push`, and switched no branch.** Staging verified with
`git ls-files --cached`, not exit codes.

⚠️ **BUT — read this before committing anything.** I banked incrementally, and while I did so the
orchestrator ran a whole-index commit. **`8ab5327` therefore already contains part of my
deliverable** — `code/idm5_ensemble.py`, the four `stack_guard_v5_*.json` reports, the two pre-fix
field-test transcripts, `taniteval/taniteval/stack_guard.py` and `taniteval/tests/test_stack_guard.py`
— under a commit message about comma `yaw_rate`. That is CLAUDE.md's documented hazard (*"`git
commit` commits the ENTIRE INDEX"*) happening again, and it is **not a loss** — the work is in git —
but the lineage now reads wrong. ⇒ **whoever commits the remainder should say in the message that
`8ab5327` also carries the fleet-sync guard fix and its tests.**
**Still uncommitted and staged by me:** `FLEET_SYNC_IDM_STEER.md`,
`idm_head_v4_steer_ens3.pt`, `raw/idm5_ensemble.json`, `raw/idm5.log`,
`raw/fleet_tree_inventory.txt`, the two re-run field-test transcripts and the two `_PREFIX_` ones.
⚠️ **The index also holds a sibling's in-progress work** (`…/incoming/2026-07-27-heading-default/*`,
`stack/tanitad/data/comma2k19.py`, `stack/tanitad/train/train_worldmodel.py`,
`stack/tests/test_comma2k19.py`, `stack/tests/test_comma_heading_regime.py`) — **not mine, do not
attribute it to this deliverable.**

## 13. Provenance and evidence class

| claim | class · tier | source |
|---|---|---|
| pod3 + tanitad-eval pass `--require v5`, exit 0 | **MEASURED (ours)** · DECISION-GRADE | `raw/guard_field_test_*.txt`, `raw/stack_guard_v5_*_GREEN.json` |
| both hosts' pre-v5 trees are refused, exit 2, all five named | **MEASURED (ours)** | same files |
| F1 — `--require v5` and no-`--require` print byte-identical summaries | **MEASURED (ours)** · DECISION-GRADE | `raw/guard_field_test_pod3.txt` §1 vs §2 |
| F2 — `/root/vtband/stack` passes exit 0 at `heldout_gate` md5 `8c146f82` ≠ HEAD `c2882830` | **MEASURED (ours)** · DECISION-GRADE | `raw/guard_field_test_eval.txt` §3, `raw/fleet_tree_inventory.txt` |
| F3 — `idm2_lib.py:19` defeats a `sys.path` estimator pin | **MEASURED (ours)** · VERIFIED IN CODE | pod3 probe; `idm2_lib.py:19`, `idm3_arms.py` headers |
| F4 — exit 1 when the resolved `taniteval` lacks `stack_check` | **MEASURED (ours)** | `raw/guard_field_test_eval.txt` §4 |
| 12 `ci.py` copies / 19 `tanitad` trees, 2 of each at HEAD | **MEASURED (ours)** · read-only sweep | `raw/fleet_tree_inventory.txt` |
| `ci.py` HEAD-vs-stale difference is display-only | **MEASURED (ours)** | the diff, **plus** §9.1's identical intervals through both copies |
| ⭐ ensemble steer +0.7993; pai −0.0042 [−0.0066,−0.0013]; cm −0.0030 [−0.0048,−0.0017]; both separated | **MEASURED (ours)** · DECISION-GRADE | `raw/idm5_ensemble.json` |
| ⭐ seed-0 file is NOT separated on PhysicalAI (−0.0025 [−0.0056, +0.0007]) | **MEASURED (ours)** · DECISION-GRADE | same |
| leak: 4/36 val episodes excluded by content, 0 residual, path `/root/idm2/lat` | **MEASURED (ours)** · DECISION-GRADE | same, `leak_check` block |
| reload of the ensemble ckpt reproduces exactly (max abs delta = 0.0) | **MEASURED (ours)** | same, `reload_verification` |
| bit-exact reproduction of the 2026-07-27 per-seed and ensemble R² | **MEASURED (ours)** | same, `reproduction_of_2026_07_27` |
| suites 1534/12 and 663 | **MEASURED (ours)** | run here, §5 |
| v3 published steer 0.408 / A0 0.742; the model-card prohibition text | **PUBLISHED (cited)** | `MODEL_CARD_IDM_V3.md` §"Known failures" item 2; table L46 / L55 |
| the rung-68/200/400 ladder; the 33-file pod3 local-edit count | **INHERITED** (`FLEET_REFILL.md` §2.1 / §2.5) — **not re-run by me** | that doc |
| pod2's state, pod1's state | ⛔ **NOT PROBED** — neither host was contacted | — |

🔒 **Gated-confidential:** counts, paths, digests, tags and module names only. **No PhysicalAI-AV
clip UUID appears anywhere in this deliverable.** `Keys.txt` was never read, printed, or passed in
argv — no credential was needed for any step.
