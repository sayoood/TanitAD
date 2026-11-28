# D-PWVOCAB-1 — the per-window 7x6 vocabulary: the cache is GONE, the RECIPE is not

**2026-09-06 · Architecture & Inference · per-window wiring stream**
**Evidence class MEASURED unless stamped otherwise. No model was trained here —
this is a SUPERVISION-SOURCE and OWNERSHIP result, so it carries no T-tier, no
ADE and no metric-family table.** The four families enter at P2, which is
pre-registered in `PREREG_D-PWVOCAB-1.md` and **not run** — see §5.

---

## 0. The headline, in three lines

1. ⛔ **`labels_train_v4.pt` no longer exists anywhere.** Its only recorded home
   was `tanitad-pod2:/workspace/v15/`, and pod2 was **TERMINATED 2026-08-03**;
   the evacuation carried **7 checkpoints and no label cache**.
2. ⭐⭐ **It did not need to.** `lat_target` / `lon_target` are **pure functions
   of `poses [T,4]`** (`stack/scripts/v4_labels.py:92,101`), `v4_labels` already
   ships **`mint_window`** — documented as *"the on-the-fly path"* and
   *"bit-identical to the corresponding row of `mint_episode`"* — and
   `V3Dataset` **already holds the episode's own poses per item**. The lost
   `.pt` was a **cache of a pure function**, not a source of truth. **Nothing
   needs to be generated, rebuilt, or asked of the PI to obtain these labels.**
3. ⛔ **P1 is BLOCKED ON OWNERSHIP, not on data.** The 7x6 targets can only be
   consumed through a `tac_vocab_version` the head sizing reads from
   **`stack/tanitad/models/v6.py`** and **`stack/tanitad/refs/refc_v3.py`** —
   both explicitly outside this stream. §4 carries the exact diff.

---

## 1. P0 — where the artifact is, with the absence proved twice

⛔ **Every absence below carries a same-breath non-zero control**, per the
mount's history of correct-size all-NUL files and under-reporting search.

| probe | mechanism | result | same-breath control |
|---|---|---|---|
| repo working tree | `find . -name "labels_*.pt"` | **0** | 170 `*.md` at depth 2 |
| repo index | `git ls-files \| grep v4-labels` | 3 files, **all JSON/MD** | 11,295 tracked files |
| `tanitad-pod2` | ssh config | ⛔ **`Host tanitad-pod2-TERMINATED`** — *"TERMINATED 2026-08-03 by the PI"* | pod3/4/5 entries present |
| pod2 evacuation | `_pod_backup/pod2-2026-08-03/ckpts/BACKUP_LOG.txt` | **7 checkpoints, 0 label caches** | 8 `OK`/`SIZE MISMATCH` lines |
| `tanitad-pod3/4/5` | ssh | **Connection refused** (all three) | thor answered on the same key |
| `tanitad-thor` | `find / -xdev -name "labels_*v4*.pt" -o -name "labels_train.pt"` | **0** | **18,534** `*.pt` on the box |
| **live A40** `69.30.85.211:22001` | `ls /workspace/v15`, `find -maxdepth 3 -name "labels_*.pt"` | **no such dir, 0 hits** | 82 entries in `/workspace` |
| dev box | `find /c/Users/Admin -maxdepth 4` | **0** | **2,154** `*.pt` |
| **HuggingFace `Sayood`** | file listing of **all 45 repos** | **0** label caches (7 hits, all the v7 per-clip JSONL line) | **18,573** files enumerated |

⇒ **VERDICT: the `.pt` is DESTROYED, not misplaced.** Two independent
mechanisms (host filesystem enumeration; index/registry enumeration), each with
a non-zero control, on eight distinct locations. **No md5 can be given because
no copy exists to hash** — and an md5 is exactly what the brief asked for, so
this is stated as the reason it is absent rather than as an omission.

⚠️ **The evacuation record over-claims.** `~/.ssh/config` says pod2 was *"fully
evacuated and verified first"*; `BACKUP_LOG.txt` enumerates checkpoints only.
Both statements are true of the checkpoints and the first is false of the label
caches. **`MANIFEST.md` had already declared the risk** — *"the multi-GB label
tensors are the cache and stay pod-side"* — so this loss was documented as a
policy, then executed as one.

### 1.1 What DOES survive, and it is the part that matters

| survivor | path | what it gives |
|---|---|---|
| the minter | `stack/scripts/v4_labels.py` (md5 `5eac3b82f9b4ffea2223842730bb1c2d`) | `lat_target`, `lon_target`, `mint_window`, `mint_episode` |
| the vocabularies | `stack/scripts/refb_labels.py:1207,1210` (md5 `53787a08250bc824a9708cf7c99c060d`) | `LAT_KINEMATIC_TOKENS` (7), `LON_KINEMATIC_TOKENS` (6) |
| its tests | `stack/tests/test_v4_labels.py` | 14 tests incl. vocab widths + masking |
| the provenance | `TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-07-22-v4-labels/labels_train_v4_provenance.json` | the census figures, `parity_key`, `skip_hash` |

⚠️ **A NAME IS NOT PROVENANCE, and neither is a sidecar.** The brief required
the parity key be read *from inside the artifact*. **That is not possible and
will not become possible** — the artifact is gone. What is verified instead is
that the surviving **provenance JSON** carries `parity_key
physicalai-train-e438721ae894` and `skip_hash f09e44db (unchanged — labels
re-derived on the existing pose cache; no episode re-selection)`, and that
`stack/scripts/make_parity_manifest.py` **cross-checks that same JSON's
`n_episodes` against two independently-scanned artifacts** before it will write
a manifest. ⇒ the census figures are **INHERITED-but-cross-checked**, and every
number in §2 below is **MEASURED here, independently, and does not rest on
them.**

⚠️ **Path drift found in passing (not fixed — out of this stream's ownership):**
`make_parity_manifest.py:72` points at `Benchmarks & Eval/` (singular). The
directory is `Benchmarks & Evals/` (plural, per the 2026-08-27 rename). The
guard is written `if V4_TRAIN_PROV.exists()`, so **the cross-check silently
does not run** — an absence that reads as a pass. Reported, not touched.

---

## 2. The census, re-measured independently — and it holds

⛔ **Not a re-read of the provenance.** The minters were run over a **real
episode cache**, from `poses`, at the dataset's own window geometry
(`WINDOW 8`, `MAX_HORIZON 20`, `last = t + WINDOW - 1`).

**Corpus: `physicalai-train-14231cd29c74`, 400 episodes, 68,377 windows,
0 unreadable.** ⚠️ **This is NOT the parity corpus** (`…-e438721ae894`), which
exists on no reachable machine. ⇒ **counts are not parity-comparable; the
structural findings are corpus-independent** and that is the distinction to
carry.

| | coverage | K | IGNORE |
|---|---|---|---|
| `lat_target` | **1.0000** | 7 | **0** |
| `lon_target` | **1.0000** | 6 | **0** |

⭐ **Coverage 1.0000 reproduces on a DIFFERENT corpus** ⇒ it is a property of
**the minter** (every window gets a class; nothing is IGNORE), not a lucky
property of the lost cache. This is the single most load-bearing claim in the
brief and it now stands on our own measurement.

| `lat_target` | n | % | | `lon_target` | n | % |
|---|---:|---:|---|---|---:|---:|
| `lane_keep` | 56,959 | 83.301 | | `free_cruise` | 41,669 | 60.940 |
| **`lc_left`** | **3,282** | **4.800** | | `stop_at_point` | 5,239 | 7.662 |
| **`lc_right`** | **3,211** | **4.696** | | `hold_stop` | 6,777 | 9.911 |
| `nudge_left` | 2,318 | 3.390 | | `launch` | 1,308 | 1.913 |
| `nudge_right` | 2,603 | 3.807 | | `creep` | 3,463 | 5.065 |
| ⛔ `abort_lc` | **4** | **0.006** | | `coast` | 9,921 | 14.509 |
| ⛔ `pull_over` | **0** | **0.000** | | | | |

⭐ **The lane-change tokens are real and plentiful** — 6,493 windows (9.50 %)
carry `lc_left`/`lc_right`. Against the v7.2 per-clip line's **0 of 4,572**,
this line supplies lane changes at ~1 in 10 windows. The 2,201-window audit's
`lc_right` 95 / `lc_left` 73 = 4.3 % / 3.3 % is **consistent** with 4.80 % /
4.70 % here.

⛔⛔ **AND THE CLASS-IMBALANCE TRAP IS WORSE THAN THE BRIEF FEARED — `pull_over`
IS NOT RARE, IT IS ABSENT.** 0 of 68,377. `abort_lc` is 4. ⇒ **the effective
lateral vocabulary is FIVE tokens, not seven**, and a 7-wide head would carry
two units that can never receive a gradient. This is a design input, not a
footnote: §3 of the pre-registration sizes the head at **5** and states the
masking rule.

**Controls, each reading its known value:**
* constant-predictor(`lane_keep`): pooled **0.8330**, macro-recall **0.1667**
  = **exactly 1/6** over the 6 present classes. ⭐ This is the brief's warning
  reproduced on our own data: **a constant predictor reads 0.833 pooled** — it
  would look like a working lateral head — **against 0.167 macro.** Pooled
  accuracy is inadmissible for this head.
* constant-predictor(`free_cruise`): pooled **0.6094**, macro **0.1667** = 1/6.
* `n` = 68,377, `d` = `poses[T,4]` **only** — no cache file, no label artifact.

**Equivalence control (the claim that the on-the-fly path is the cached path):**
on episode 0, n = 171 windows, direct `lat_target`/`lon_target` calls are
**bitwise equal** to `mint_episode`'s rows for both axes. **Negative control:**
the same comparison with the anchor shifted by one window reads **False** — so
the equality is not vacuous.

**Cost of deriving it in-loader: 0.383 ms/window** (26.2 s for 68,377,
single process, no GPU). ⚠️ Call `lat_target`/`lon_target` directly, **not**
`mint_window`, in a hot loop: `mint_window` re-runs `savgol` + `vtarget_v2`
over the whole track per call for fields the tactical heads do not use.

---

## 3. ⭐⭐ THE RESULT THAT DECIDES WHETHER THIS IS WORTH DOING

The brief's claim to test is *"a finer vocabulary on the same windows helps"*.
Before any GPU, that claim has a **necessary precondition** which is cheap to
check and which nobody had checked: **does kin3 actually lose information?**

Cross-tab, **same windows, same poses**, the trainer's own call
(`tac.window_factored_labels(pose_last, fut_ext[:, :20])`) against v4:

### LATERAL — v4 (7) x kin3 (`lane_keep`, `turn_left`, `turn_right`)

| v4 token | -> `lane_keep` | `turn_left` | `turn_right` | row n |
|---|---:|---:|---:|---:|
| `lane_keep` | 34,009 | 12,569 | 10,381 | 56,959 |
| `lc_left` | **2,368** | 914 | 0 | 3,282 |
| `lc_right` | **2,358** | 0 | 853 | 3,211 |
| `nudge_left` | **2,239** | 79 | 0 | 2,318 |
| `nudge_right` | **2,565** | 0 | 38 | 2,603 |
| `abort_lc` | 4 | 0 | 0 | 4 |

⛔ **kin3's `lane_keep` is the sole destination of all five lane-change and
nudge tokens.** **11,418 / 68,377 = 16.70 %** of windows carry a v4 lateral
identity kin3 **cannot express at all**.

⚠️ **And the reverse error is larger than the forward one.** v4 `lane_keep`
lands in kin3 `lane_keep` at purity **0.597** — **22,950 windows of ordinary
lane-keeping are called `turn_left`/`turn_right` by kin3**, which is **92.4 %
of everything kin3 calls a turn** (22,950 of 24,834). ⇒ **kin3's two "turn"
classes are predominantly curve-following, not manoeuvres.** That is a
mislabelling of the live trainer's supervision, found here, and it is a
stronger argument for the finer vocabulary than the collision count is.

### LONGITUDINAL — v4 (6) x kin3 (`brake_stop`, `steady`, `accelerate`)

| v4 token | `brake_stop` | `steady` | `accelerate` | row n |
|---|---:|---:|---:|---:|
| `free_cruise` | 0 | 24,677 | 16,992 | 41,669 |
| `stop_at_point` | 4,343 | 777 | 119 | 5,239 |
| `hold_stop` | 0 | 6,219 | 558 | 6,777 |
| `launch` | 0 | 0 | 1,308 | 1,308 |
| `creep` | 108 | 2,066 | 1,289 | 3,463 |
| `coast` | 9,921 | 0 | 0 | 9,921 |

⛔ kin3 `steady` <- {`free_cruise`, `hold_stop`, `creep`}; kin3 `brake_stop` <-
{`stop_at_point`, `coast`}. **15,479 / 68,377 = 22.64 %** unrepresentable.

⭐ **`coast` vs `stop_at_point` is the one that should worry us**: kin3 calls
both `brake_stop`, i.e. **"lift off" and "there is a stop line ahead" are the
same word.** 88.7 % of the programme's oracle gap is longitudinal, and this is
a concrete mechanism by which the current supervision cannot see the
difference.

⇒ ⭐⭐ **MEASURED HEADROOM: 16.70 % (lateral) and 22.64 % (longitudinal) of ALL
windows carry a distinction the live trainer's labels destroy.** This is the
honest size of the prize, at zero GPU. **It is a precondition, not a result:**
it proves the information is thrown away, **not** that a model can recover it
or that recovering it improves driving. That is what P2 is for.

---

## 4. ⛔ P1 — the wiring, and the exact ownership blocker

**The seam.** `compute_losses_v3` already supports **two** tactical label
sources on the `z_tac` heads — kin3 (runtime, from poses) and v7.2 (per-clip,
via `batch["lat_v7"]`) — selecting on `use_v7 = "lat_v7" in batch`, with a
width refusal on each. **The 7x6 set is a natural third source through the same
seam.** No new mechanism is needed.

**Why it cannot land in this stream.** The head width is not chosen in the
trainer:

```
stack/tanitad/refs/refc_v3.py:909-919
    _vv = getattr(cfg, "tac_vocab_version", "kin3")
    if _vv == "kin3":
        _nlat, _nlon = tac.N_LAT, tac.N_LON
    else:
        from tanitad.models.v6 import (tactical_lat_actions,
                                       tactical_lon_actions_v)
        _nlat = len(tactical_lat_actions(_vv))
        _nlon = len(tactical_lon_actions_v(_vv))
    self.lat_head_tac = nn.Linear(cfg.d_tac, _nlat)
```

and `tactical_lat_actions` resolves against `TACTICAL_VOCAB_VERSIONS`, a dict
in **`stack/tanitad/models/v6.py`**, which raises `ValueError` on an unknown
version. ⇒ a new vocabulary version **must** be added to `v6.py`, and
`refc_v3.py` is what reads it. **Both are named in this stream's do-not-edit
list, and both have live siblings in them tonight.** No edit was made to
either. The exact diff is §4.2.

### 4.1 ⛔ A COUPLING THE DIFF MUST NOT HIDE

```
stack/tanitad/refs/refc_v3.py:1189
    man5 = (tac.derive_man5_logprobs(lat, lon)
            if self.tac_vocab_version == "kin3" else None)
```

⇒ **any** `tac_vocab_version` other than `"kin3"` sets `man5 = None`, and
`hook_out["maneuver_logits"]` (the H19 lateral prior) is then never populated.
The consumer is guarded (`if man5 is not None`), so **this does not crash — it
silently removes a capability**, which is precisely the "an operator typed a
weight a later layer zeroes" failure the effective-weight guard exists to stop.
⚠️ **The v7.2 arm already pays this cost today** and it is not stamped anywhere.
⇒ **the escalated change must stamp `man5_active` into `config.json`**, or it
creates a second silent path. That stamp is the reviewable part of the diff.

### 4.2 The exact diff, for the owners of `v6.py` and `refc_v3.py`

```diff
--- a/stack/tanitad/models/v6.py
+++ b/stack/tanitad/models/v6.py
@@ TACTICAL_VOCAB_VERSIONS
+    # ⭐ kin76 — the v4 per-window KINEMATIC vocabulary (v4_labels.LAT_TOKENS).
+    # ⛔ FIVE tokens, not seven: `abort_lc` (4/68,377) and `pull_over`
+    # (0/68,377) are MEASURED dead on real poses and are masked to
+    # IGNORE_INDEX by the emitter, never given a head unit.
+    "kin76": ("lane_keep", "lc_left", "lc_right", "nudge_left", "nudge_right"),

@@ TACTICAL_LON_ACTION_VOCAB_VERSIONS
+    "kin76": ("free_cruise", "stop_at_point", "hold_stop", "launch",
+              "creep", "coast"),
```

```diff
--- a/stack/tanitad/refs/refc_v3.py
+++ b/stack/tanitad/refs/refc_v3.py
@@ -1189,3 +1189,7 @@
-        man5 = (tac.derive_man5_logprobs(lat, lon)
-                if self.tac_vocab_version == "kin3" else None)
+        # ⛔ man5 is DEFINED on [B,3]x[B,3]; any other vocabulary drops the H19
+        # prior. Recorded, not assumed: `man5_active` is stamped into
+        # config.json by the trainer so the run says which arm paid this.
+        man5 = (tac.derive_man5_logprobs(lat, lon)
+                if self.tac_vocab_version == "kin3" else None)
+        self._man5_active = man5 is not None
```

**The trainer-side half — `refc_v3_train.py`, which IS this stream's — is
specified in `refc_v3_train.perwindow.patch.md`** and is deliberately **not
applied**, because a flag whose consumer cannot be built is the "module built,
wired to nothing" defect the refcv5 plan lists as **P6**. ⇒ **It lands in one
commit with the two files above, or it does not land.**

---

## 5. What was NOT done, and why — stated plainly

⛔ **P2 was not run. There is no PASS and no FAIL, because no arm was
trained.** The pre-registered bar exists (`PREREG_D-PWVOCAB-1.md`, both
outcomes committed, deliberate-regression arm, controls, four families,
variance named) and it is **executable the moment §4.2 lands**. Reporting a
verdict here would be reporting on an experiment that did not happen.

⛔ **No per-class recall on a trained head is reported**, for the same reason.
The per-class **label** distribution and the **majority-class control** are in
§2 and they are what §4.2's owners need to size the head correctly.

⚠️ **The A40 was not touched** beyond two read-only `ls`/`find` calls with
`ssh -n`; it is training `refcv5-ddim-b1-v72-40k`. Thor was probed read-only.
No GPU was used by this stream.

---

## 6. The one-line answer to the brief's closing question

⭐ **Yes — the model can be taught the 7x6 vocabulary, and it needs no PI
decision and no label generation to obtain it**, because the labels are a pure
function of poses the loader already holds and the lost `.pt` was only a cache
of that function. ⛔ **But it cannot be taught it in this stream**: the head
sizing lives in `v6.py` and `refc_v3.py`. **Whether it HELPS is unmeasured** —
what is measured is that the current labels destroy the distinction on
**16.70 %** (lateral) and **22.64 %** (longitudinal) of all windows, that the
effective lateral vocabulary is **five** tokens rather than seven, and that a
constant predictor scores **0.833** pooled on this head, so the validation must
read per-class recall or it will fool itself.

---

## 7. Deliverable manifest

| artifact | where |
|---|---|
| this report | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-perwindow-wiring/PERWINDOW_WIRING.md` |
| pre-registration | `…/2026-09-06-perwindow-wiring/PREREG_D-PWVOCAB-1.md` |
| trainer-side patch (**specified, NOT applied**) | `…/2026-09-06-perwindow-wiring/refc_v3_train.perwindow.patch.md` |
| census reproduction (script + JSON) | `…/2026-09-06-perwindow-wiring/raw/repro_perwindow.py`, `raw/repro_perwindow.json` |
| kin3 x v4 cross-tab (script + JSON) | `…/2026-09-06-perwindow-wiring/raw/crosstab_kin3_vs_v4.py`, `raw/crosstab_kin3_vs_v4.json` |
| register entry | appended to `Project Steering/GOALS_AND_CLAIMS.md` |

⛔ Nothing is stranded: both scripts run from a repo checkout against a local
episode cache, and both write their JSON beside them.
