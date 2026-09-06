# DATA — the RL-pilot agent join: 15 → 34 val episodes, and the 4 episodes that were the wrong clip

**Agent:** TanitAD data-engineering (TanitAD_TrainingFlyWheel) · **Date:** 2026-09-05
**Ask:** extend `pilot_val_agents.jsonl` (2 983 lines, **15** distinct `clip_id`) to as many of the
100 val epcache episodes as the local label data allows, identical line schema, verify by content.

> ## ⛔ HEADLINE — READ THIS BEFORE THE COVERAGE NUMBER
>
> **4 of the live file's 15 episodes carry agent boxes from the WRONG CLIP** (`ep_00013`,
> `ep_00065`, `ep_00076`, `ep_00087` — 26.7 %). Every one of them passed the builder's own
> ambiguity guard. The boxes are real NVIDIA cuboids at plausible ranges, from a real recording
> that is **not this episode**, so `collision`/`headway` fire on agents that were never in front of
> this ego and **there is no downstream symptom**. Any probe-panel result computed on those
> episodes is contaminated.
>
> Coverage did go up — **15 → 34 episodes (2.27×)**, 2 983 → 6 763 lines, 89 423 → 241 228 boxes —
> but the identity fix **removes 3 episodes** (their true clip has no local obstacle data at all;
> the live file joined a prefix-sibling instead) and **re-joins 1** with the correct clip.
>
> ⛔ **AND THE TRAINING HALF IS WORSE.** The same audit run on `pilot_train_agents.jsonl` (not in
> the brief; run because a failed arm must leave the next arm behind):
> **13 of its 54 episodes — 24.1 % — carry the wrong clip**, and coverage under verified identity
> is **137 of 400** against the 54 it has. Both halves are now rebuilt.
>
> ⛔ **ESCALATION:** the live `pilot_val_agents.jsonl` and `pilot_train_agents.jsonl` should be
> treated as **contaminated**, not merely low-power. Both are unchanged on disk (as instructed);
> `pilot_val_agents_ext.jsonl` / `pilot_train_agents_ext.jsonl` are the replacements. Whoever owns
> the probe panel and the pilot must decide whether banked results stand.

---

## 1. Diagnosis — why coverage was 15

**It was the `--chunks-dir` argument, and nothing else.** Not corpus order, not registration,
not a missing v2 val corpus.

The live file was **not** produced by `build_obstacle_join.py` at all — it was produced by
`stack/scripts/rl_pilot_join.py`, which is the epcache-native builder. Proven by exact
reproduction, not by inference:

| check | result | evidence class |
|---|---|---|
| `rl_pilot_join.py --chunks-dir C:/Users/Admin/tanitad-data/rl-pilot/obstacle.offline.chunks` | md5 **`85e1bb71b9a4847fb00ff0db184b2fce`** | MEASURED |
| live `C:/Users/Admin/tanitad-data/rl-pilot/pilot_val_agents.jsonl` | md5 **`85e1bb71b9a4847fb00ff0db184b2fce`** | MEASURED |

That directory holds **30** chunk zips / **2 872** clips. The 100 val episodes intersect it at
exactly **15**. Measured per label source (artifact: the sweep in §6):

| local obstacle source | files | clips | val episodes matched (4-char prefix) | of which unambiguous |
|---|---:|---:|---:|---:|
| `rl-pilot/obstacle.offline.chunks` | 30 zips | 2 872 | 15 | **15** ← the live file |
| `physicalai/labels/obstacle.offline` | 57 zips | 5 317 | 32 | 31 |
| `physicalai/labels/obstacle_offline_b1eval` | 145 parquet | 145 | 1 | 1 |
| **union** | 200 files | **7 267** | 40 | 38 |

⚠️ **`build_obstacle_join.py` was never a candidate route.** Its `corpus_first_clips()` requires
`*.v2ep.pt` and no v2 val corpus exists on this box (only the `_epcache` `ep_*.pt` form, keys
`frames_u8, actions, poses, episode_id, maneuvers`). More decisively, its **geometry and its
NO_LABEL contract differ from the live file's**, so a file built with it would disagree with the
live file on every box — the extension would not be an extension. The correct route was to extend
the builder that actually made the file.

---

## 2. ⛔ The defect the diagnosis surfaced — a 16-bit hash used as an identity

The epcache carries **no `clip_id`**. `physicalai.build_episode` stores
`episode_id = int.from_bytes(clip_id.encode()[:4].ljust(4, b"\0"), "big")`
(`stack/tanitad/data/physicalai.py:740`) — **the first four characters of the uuid**. `rl_pilot_join.py`
inverted that against the local clip universe and dropped any episode whose prefix matched more
than one clip.

**That guard asks the wrong question.** It asks *"is this prefix unique among the chunk zips that
happen to be cached?"*, not *"which clip is this episode?"*. Widening the universe makes it worse:
over the 7 267-clip union the prefix route calls **38** episodes unambiguous and **6 of those are
the wrong clip**.

### The correct identity, and its control

The episode's own poses ARE the answer. `physicalai.signals_at` builds
`poses = [col("x"), col("y"), yaw, v]` (`physicalai.py:641`) — i.e. **columns 0–1 are egomotion
`x`/`y` interpolated at the query times**, in the clip's own frame. So the true clip's egomotion
polyline passes *through* the episode's pose track and a sibling's does not.
`taniteval.lead_source.register_poses_to_time` fits exactly that and **refuses loudly** when it
fails.

| MEASURED over all 100 val episodes (`--identity registered`) | value |
|---|---|
| episodes resolved to **exactly one** registering candidate | **100 / 100** |
| prefix groups of size 1 / 2 / 3 among the 34 joined | 24 / 9 / 1 |
| median probe residual of the accepted clip (34 joined) | **0.00105 m** |
| worst probe residual (34 joined / all 100) | **0.00278 m / 0.00401 m** |
| a rejected sibling — `ep_00013`'s `276cb0b3…` | **4.403 m**, refused at the 0.25 m bar |

⭐ **This is the control the prefix guard never had: a candidate that is not the clip cannot be
made to register.** The separation is ~1 600× (0.0028 m vs 4.4 m), not a threshold judgement.

### The four contaminated episodes in the live file

| epcache stem | `real_clip_id` in the live file | the clip that registers | fate in the new file |
|---|---|---|---|
| `ep_00013` | `276cb0b3-577b-4f2c-a093-9357f308829f` | `276c0b12-01ca-44e1-b050-d7878a672d50` | **dropped** — true clip has no local obstacle data |
| `ep_00065` | `7e145c4e-a495-4036-b297-6fb30558c0d3` | `7e145616-a5a1-4c4c-be01-c987de11ce9e` | **dropped** — same |
| `ep_00087` | `dc878b88-7411-4b9d-a0b7-c2430437dec6` | `dc8732e8-0eb2-49a8-805a-e7155563182b` | **dropped** — same |
| `ep_00076` | `a7e26d9a-f958-47fb-8c9f-56c4d14a89dd` | `a7e2a63f-f5b7-4e3c-b219-ee5ace6039c4` | **re-joined with the correct clip** |

⇒ 15 − 3 dropped = 12 overlap; of those 11 keep their clip and 1 changes.

### ⭐ The proof that needs no registration at all

The guard is **directional**. It detects *one episode → many clips*; it cannot see *many episodes →
one clip*. In `pilot_train_agents.jsonl`, **`ep_00107` and `ep_00108` are both assigned
`4ab1f61b-8d6e-418f-a31d-782104ff7c39`** — 54 `clip_id`s carrying only 53 distinct
`real_clip_id`s. Two different episodes with byte-identical agent boxes is impossible for correct
identities, and it is visible **in the live file alone**, with no egomotion and no fitting. (Their
true clips share the prefix `4ab1`: `ep_00107` is `4ab1e3eb-…`, `ep_00108` is `4ab1f61b-…`, and
only the latter was cached — so the guard called both "unambiguous" and handed them the same clip.)

*Root-cause class: the `df` / Thor `free` / cgroup `usage_in_bytes` family — **a probe answering a
narrower question than the claim hung on it**. Here the probe answers "unique in the cache?" and
the claim is "this is the clip". Sibling of `H-ESTIM-SEED-1`.*

---

## 3. `clip_id` — what it looks like, and how the consumer matches it

⚠️ **The match is EXACT, not substring, in both pilot consumers.** Stated because the brief
assumed substring:

* `stack/scripts/rl_a0_coverage.py:131-132` — `clip_ids = {basename(p).split(".")[0] for p in eps}`
  then `if clips is not None and cid not in clips: continue`. **Set membership.**
* `stack/scripts/rl_pilot_refc21.py:69,92` — keys by `r["clip_id"]` and then reloads
  `torch.load(os.path.join(epdir, stem + ".pt"))`. **The `clip_id` IS the filename stem.**

| thing | real example |
|---|---|
| epcache file | `C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836/ep_00013.pt` |
| epcache **stem** the consumer computes | `ep_00013` |
| join's **`clip_id`** (must equal the stem verbatim) | `ep_00013` |
| join's **`real_clip_id`** (the recovered uuid, audit only) | `276c0b12-01ca-44e1-b050-d7878a672d50` |

A uuid written into `clip_id` would join **nothing** — `set` membership fails and
`torch.load(epdir/<uuid>.pt)` raises. The extension keeps `clip_id = stem`. Pinned by
`test_written_clip_id_is_the_epcache_stem_matched_exactly_by_consumers`.

---

## 4. Route taken — additive gates on the builder that made the file

`stack/scripts/rl_pilot_join.py` gained four levers. **Every default is the v1 behaviour**, and the
v1 path is proven byte-identical after every edit (§5, control row).

| flag | default (= v1) | what the non-default does |
|---|---|---|
| `--chunks-dir` | single dir | now `nargs="+"`; each dir probed for **both** layouts (chunk zips *and* loose `<uuid>.parquet`), **in the order given, first hit wins** |
| `--ego-dir` | `[]` | egomotion sources (zip and/or loose), required by the three modes below |
| `--identity` | `prefix` | `registered` — fit every prefix candidate to the episode's pose track; **never substitutes a sibling** |
| `--t0-mode` | `hypothesis` | `registered` — per-episode affine fit `t = a + b·i` instead of `0.2 + i/10` |
| `--geometry` | `rig-raw` | `ego-compensated` — rig@sample → world → ego@frame, **importing** `build_obstacle_join.rig_to_world` + `bev_raster.ego_frame_agents` |
| `--occ-mode` | `zero` | `fov` — the real front-camera azimuth flag via `build_obstacle_join.visibility_occ` |

`labels/obstacle_offline_b1eval` carries the **identical 16-column schema** as a chunk member
(MEASURED: column lists equal, `reference_frame == "rig"` in both), so mixing the layouts is safe.
It nonetheless contributes **0** of the 34 joined episodes — its single prefix match was a false one.

### ⛔ A bug the overlap check caught, that the run reported as success

The first revision adopted the registration's `(a, b)` **whenever `--identity registered` was set**,
fusing the identity lever and the time-base lever. Result: the `--t0-mode hypothesis` arm and the
`--t0-mode registered` arm came out **byte-identical (md5 `c57bf7fb3140` twice)**, and the
"identity-only" arm silently carried the time-base correction. Both runs printed success.

It was visible only as a *predicted-agreement failure*: LIVE vs EXT-A on the 11 same-clip episodes
read **20.0 % exactly equal** where it had to read 100 %. Fixed, and pinned by
`test_time_base_is_independent_of_identity_mode` — **which was mutation-tested**: the defect was
reintroduced, the test FAILED, the file was restored and byte-compared against a saved good copy.
*(A guard that has never seen the defect is not a guard — `guards-need-mutation-not-inspection`.)*

---

## 5. Verification by content

### 5.1 The arms (MEASURED; artifacts are the paths in §7)

| arm | levers vs v1 | lines | clips | boxes | frames w/ agents | agents/frame | ∩ 100 epcache eps | md5 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **CONTROL** v1 | none | 2 983 | 15 | 89 423 | 2 710 | 29.98 | 15 | `85e1bb71b9a4847fb00ff0db184b2fce` |
| **EXT-A** | +sources +identity | 6 763 | **34** | 241 773 | 6 439 | 35.75 | **34** | `a20f12382f021b746a8686d8fe2fa86e` |
| **EXT-B** | + time base | 6 763 | 34 | 241 228 | 6 424 | 35.67 | 34 | *(scratch, not shipped)* |
| **EXT-C — PRIMARY** | + ego-comp + fov occ | 6 763 | **34** | 241 228 | 6 424 | 35.67 | **34** | `2200e0ac4f71c28c7bc27fcc52c6bc52` |

**CONTROL is bit-identical to the live file** — the extension is additively gated, verified after
every edit including the final one.

### 5.2 Class histogram

| arm | automobile | person | heavy_truck | bus | rider | trailer | protruding_object | stroller | other_vehicle | train_or_tram_car | animal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LIVE | 70 958 | 14 214 | 492 | 1 431 | 1 215 | 584 | 295 | 89 | — | 77 | 68 |
| **EXT-C** | **199 279** | **29 924** | **3 277** | **2 412** | **2 385** | **1 940** | **1 419** | **255** | **183** | **80** | **74** |

`other_vehicle` appears only in the extended file — a class the 15-episode panel never saw.

### 5.3 ⭐ Overlap consistency — the interchangeability proof

**LIVE vs EXT-A, on the 11 episodes whose identity was already correct:
2 189 / 2 189 frames EXACTLY EQUAL — 100.0 %.**

Adding two label sources and switching to content-based identity **perturbed nothing** that was
already right. The complete set of differences is enumerable and each is explained:

| difference | count | why |
|---|---:|---|
| frames bit-identical on same-clip overlap | 2 189 / 2 189 | — |
| episodes dropped | 3 (`ep_00013`, `ep_00065`, `ep_00087`) | verified clip has no local obstacle data; the live file had joined a prefix-sibling |
| episodes with `real_clip_id` corrected | 1 (`ep_00076`) | prefix collision resolved by registration |
| episodes added | 22 | the two additional label sources |

⇒ `pilot_val_agents_ext_v1sem.jsonl` **is a drop-in higher-power replacement** for
`pilot_val_agents.jsonl` with no semantic change beyond the identity fix.

### 5.4 Per-lever attribution (paired, same frames)

| lever | frames identical | agent-count changes | per-box displacement, **decision band `0 < cx < 80 m`** | occ flips |
|---|---:|---:|---|---:|
| **time base** `0.2 + i/10` → `a + b·i` | 868 / 6 763 (12.8 %) | 2 370 frames | median **0.0084 m**, p95 **1.766 m**, max 10.01 m (n = 95 863) | 0 |
| **ego-compensation** rig-raw → composed | 378 / 6 763 (5.6 %) | **0** | median **0.1309 m**, p95 **0.658 m**, max 3.128 m (n = 100 325) | 144 431 |
| **both**, LIVE → EXT-C (11 same clips) | 221 / 2 189 (10.1 %) | 674 frames | median **0.3913 m**, p95 **2.352 m**, max 8.007 m (n = 26 610) | 45 770 |

* **Time base.** Recovered `b` ∈ [0.100496, 0.101010] s (median 0.100668) — the 0.1 s hypothesis
  drifts **0.098–0.200 s over a 199-frame episode**. Recovered `a` ∈ [0.0354, 0.2183] s (median
  0.1128), not 0.2. Combined per-episode longitudinal shift: **median 0.2369 m, max 0.764 m**.
  It also moves tracks in and out of the ±0.06 s match window, which is the 2 370 count changes.
* **Ego-compensation** changes geometry only — **0 count changes**, exactly as it must, which is
  itself the control that the two levers are separated.
* ⚠️ The 34.78 m worst case in the raw (unbanded) time-base column is **autolabel noise, not a join
  error**: track 169 of `ep_00035` moves −208 → −198 → −172 m in consecutive 0.1 s rows (≈350 m/s)
  at ~200 m *behind* the ego. Reported so the number is not read as a defect.

### 5.5 Three data-quality controls (each had to read a known value)

| control | MEASURED | verdict |
|---|---|---|
| distinct `source` values per clip | **1** (`scene:obstacles:autolabels:v2`) for 34/34; track_ids used by >1 source: **0 / 3 843** | the per-`track_id` nearest-sample rule is safe |
| duplicate `(track_id, timestamp_us)` rows | **0 / 72 989** over 10 clips | nearest-per-track is well defined |
| `occ` in v1-mode files | **100.0 % zero** (89 423 / 89 423 and 241 773 / 241 773) | v1 `occ` is a **constant, not a measurement** |

⛔ **The `occ` finding is load-bearing.** In EXT-C the real front-camera flag reads
**occ=0: 96 797 (40.1 %) · occ=1: 144 431 (59.9 %)** — i.e. **~60 % of the joined boxes are outside
the 120° field**, and the v1 files call every one of them *visible*. No RL-pilot consumer reads
`occ` today (MEASURED: the only `occ` readers under `stack/scripts` are `refc_v3_train.py`'s
unrelated raster tensor and `train_p8_occupancy.py`'s docstring), so nothing is broken **yet** —
but any future use of that field on a v1-mode file reads a constant.

### 5.6 ⛔ NO_LABEL is emitted as "road clear" — inherited, now visible

v1 writes one line per frame **always**, so a frame outside the clip's ~20 s obstacle span emits
`agents: []`, which every consumer reads as *labelled clear*
(`rl_pilot_refc21.py:113` → no lead → `FAR_LEAD_X`; `rl_a0_coverage.py:172` → no obstacles).

**MEASURED over the 34 joined episodes: 292 of 6 763 lines (4.3 %) are outside the label span,
against only 47 genuinely-clear frames inside it — so ~86 % of the "clear" frames are
manufactured.** This is the bias `build_obstacle_join`'s join-doc §4 forbids.

Dropping the line does **not** fix it (both consumers use `.get(t0, [])`, which yields `[]` for a
missing frame too) and adding a key would break the schema this file must keep. ⇒ the per-episode
`label_span_s` is now **banked in the meta sidecar** (`identity_log[].label_span_s`,
`n_frames_outside_label_span`) and the count is printed at build time, so a consumer can mask on
it. **This needs a consumer-side decision; it is not fixed by this work.**

### 5.7 Consumer-side load proof (the readers themselves, not a re-implementation)

Both real reader functions were executed against the shipped files — `rl_a0_coverage.read_agents`
and `rl_pilot_refc21.load_agents`, extracted from their own source, not re-written:

| file | `rl_a0_coverage.read_agents` | boxes | every `clip_id` resolves to an `ep_*.pt` |
|---|---:|---:|---|
| `pilot_val_agents.jsonl` | 15 episodes | 89 423 | ✔ |
| **`pilot_val_agents_ext.jsonl`** | **34 episodes** | **241 228** | ✔ |
| `pilot_val_agents_ext_v1sem.jsonl` | 34 episodes | 241 773 | ✔ |

`rl_pilot_refc21.load_agents` → **34 stems, 0 with no `epdir/<stem>.pt`**; `frame_idx` range for
`ep_00000` is 0..198 against episode `T = 199` — **in range**. The `clip_id`↔stem contract of §3
holds against the code that enforces it.

---

## 6. The honest ceiling — what the local label data can and cannot cover

**N = 34 of 100.** The binding constraint is the **obstacle** labels, not egomotion and not the
builder.

| | measured |
|---|---|
| val episodes with a **verified identity** | **100 / 100** (egomotion covers the whole set) |
| verified clip present in the local obstacle union (7 267 clips) | **34** |
| verified clip **absent** from all three local obstacle sources | **66** |
| joined-episode source split | `labels/obstacle.offline` **23** · `rl-pilot/…chunks` **11** · `b1eval` **0** |

The 34 stems: `ep_00000 ep_00004 ep_00006 ep_00008 ep_00010 ep_00015 ep_00018 ep_00020 ep_00021
ep_00022 ep_00023 ep_00026 ep_00027 ep_00030 ep_00033 ep_00034 ep_00035 ep_00048 ep_00049 ep_00051
ep_00052 ep_00054 ep_00056 ep_00059 ep_00063 ep_00064 ep_00074 ep_00076 ep_00078 ep_00086 ep_00089
ep_00091 ep_00092 ep_00095`.

⇒ **Going past 34 requires downloading more `obstacle.offline` chunks, which is an HF fetch and
therefore a PI decision** (HF quota is a hard programme ceiling; `--no-download` was mandatory
here and no network call was made). The exact ask is computable from the meta: the 66 unjoined
episodes' verified uuids are in `pilot_val_agents_ext.jsonl.meta.json → identity_log[]`, each with
`why: "verified clip has no local obstacle.offline"`. **This is a named blocker, not an idle stop.**

⚠️ **Power, stated honestly.** 34 episodes is the clustering unit for the episode-cluster
bootstrap, so it is a **2.27× increase in the binding power limit** — not a fix for it. And per
`H-ESTIM-SEED-1`, a separated CI over 34 episodes is still only the *episode-draw* question; a
lever claim on this panel still needs a replicate arm.

---

## 6b. The train half — audited and rebuilt in the same turn

Not in the brief. Run because *"a failed arm must leave behind the NEXT arm"* and the instrument
was already written; the audit is one command.

| | live `pilot_train_agents.jsonl` | corrected `pilot_train_agents_ext.jsonl` |
|---|---:|---:|
| lines | 10 744 | **27 253** |
| distinct `clip_id` | 54 | **137** (of 400 epcache episodes) |
| agent boxes | — | **1 026 834** |
| **episodes carrying the WRONG clip** | **13 / 54 = 24.1 %** | 0 (identity registered) |
| episodes sharing one clip with another episode | **1 pair** (`ep_00107`/`ep_00108`) | 0 |
| identity resolved by registration | — | **400 / 400** |
| frames outside the label span (NO_LABEL as clear) | — | 642 / 27 253 (2.4 %) |
| time-base displacement corrected | — | median **0.2519 m**, max **2.3703 m** per episode |
| `occ` | constant 0 | **61.3 % out of field** (629 359 / 1 026 834) |
| md5 | `44881b76fab05cd633e1e158fbc117ac` | `c01f4834b5849e98d84389610b9b4ed6` |

The 13 misidentified train episodes are `ep_00018 ep_00044 ep_00071 ep_00107 ep_00122 ep_00147
ep_00167 ep_00174 ep_00246 ep_00279 ep_00300 ep_00319` + 1 more; the full live-vs-registered pairs
are in the audit artifact and in the new file's `identity_log`.

### ⭐ Leak control — train/val clip disjointness

| set | value |
|---|---|
| verified val clips ∩ verified train clips (34 vs 137) | **0** |
| live val clips ∩ live train clips (15 vs 53) | **0** |

The extension does **not** introduce a train/val leak, and the live files did not have one either.
This control had to read exactly 0 and does.

---

## 6c. Test suite

| run | result |
|---|---|
| `pytest -q tests/test_rl_pilot_join.py` (real tree, G:) | **12 passed** |
| mutation control — defect reintroduced | the guard **FAILED** as required; file restored and byte-compared to a saved good copy |
| **`pytest -q` full suite, REAL TREE (G:)** — authoritative | **6 507 passed, 60 skipped, 2 xfailed, 52 failed** in 1:14:50 |
| `pytest -q` full suite, off-Drive copy (corroborating) | 6 456 passed, 118 skipped, 2 xfailed, 47 failed, 29 errors |
| the 12 new tests inside **both** full runs | **passed** — absent from every FAILED/ERROR line in either |

⚠️ **Two environment facts the raw numbers need, or they read as a regression.**

1. ⚠️ **A correction I owe this document.** I first reported the real-tree run as *"died with zero
   output"* — it had not. `pytest -q … | tail -30` buffers everything until the pipeline ends, so
   the output file sat at 0 bytes for **75 minutes** while the run was healthy, and a `ps` check
   that found no matching PID looked like confirmation. *Same family as the traps this repo already
   documents: an absence read as an answer, and a second probe through the same channel read as a
   second sample. **Never pipe a long suite through `tail`** — redirect to a log and poll the log.*
2. **The G: mount was separately in a genuine `Errno 22` flap.** Two *later* attempts on G: died
   for real — one with **61 collection errors**, all `OSError: [Errno 22] Invalid argument`,
   including `pytest` failing to read `stack/pyproject.toml`, a file this work never touched. That
   is the documented Drive hard-failure mode, not a code result. The off-Drive copy
   (`gdrive-mount-hard-failure`, `devbox-run-recipe-off-drive`) was built to get past it, with
   `tanitad.__file__` asserted to resolve into the copy — and it **corroborates** the real-tree
   result rather than replacing it.
3. **Attribution of the 52.** None of them is this work: the 13 failing files are
   `test_mktree_commit` (git), `test_o6_spectrum_power`, `test_refc_agents`,
   `test_rl_refc_adapter_robust`, `test_runbook_commands`, `test_text_encoding_is_explicit`,
   `test_trunk_anchor`, `test_v6_agent_slots`, `test_v6_anchor_loss`, `test_v6_chain`,
   `test_v6_domain_mix`, `test_v6_s2_loss`, `test_v6_seam_dump` — and **not one of them contains
   the string `rl_pilot_join`** (checked file by file), because the modified script is a standalone
   CLI that **nothing in `stack/`, `taniteval/` or `tools/` imports** except its own new test.
   ⚠️ Several of them (`test_refc_agents`, `test_rl_refc_adapter_robust`) look like the same
   in-flight `agent_gt` feature as the defect below — another agent's live work on this shared
   branch, not measured here and not claimed either way.

⛔ **One of them is a live defect at HEAD worth naming** (it fails in **both** runs, real tree and
copy) — `test_rl_refc_adapter_robust.py::test_forward_kwargs_plumbs_every_channel_the_forward_accepts`:

```
FORWARD_KEYS ['ego_state','lan','nav_cmd','nav_known','v0','withheld_speed']
   != forward's optional params [... + 'agent_gt']
```

Attribution is a positive assertion, not an inference: `stack/tanitad/rl/refc_adapter.py` and
`stack/tanitad/refs/refc.py` **both blob-match `HEAD`** (unmodified in the worktree), `HEAD`'s
`refc.py` already carries `agent_gt` (6 occurrences) and `HEAD`'s `FORWARD_KEYS` does not list it.
⇒ committed before this work, in two files it never touched. **Consequence:
`refc_adapter.py:126` does `kw = {k: batch.get(k) for k in FORWARD_KEYS}`, so `agent_gt` is
silently dropped and the oracle path never receives its agent GT through the RL adapter.**
Escalated as its own item.

---

## 7. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `stack/scripts/rl_pilot_join.py` | `repo:` — **modified** (4 additive gates, v1 byte-identical) | staged |
| `stack/tests/test_rl_pilot_join.py` | `repo:` — **new**, 12 tests | staged |
| this document | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-design-dialogue/DATA_AGENTJOIN_EXTENSION.md` | staged |
| **`pilot_val_agents_ext.jsonl`** — **PRIMARY (val)** | `devbox:C:/Users/Admin/tanitad-data/rl-pilot/pilot_val_agents_ext.jsonl` | ⚠️ **ONE PLACE ONLY** (28.7 MB, data-tree, not a repo path) |
| `pilot_val_agents_ext.jsonl.meta.json` | same dir | ⚠️ one place only |
| `pilot_val_agents_ext_v1sem.jsonl` (+meta) — drop-in, v1 semantics | same dir | ⚠️ one place only |
| **`pilot_train_agents_ext.jsonl`** (+meta) — **PRIMARY (train)** | same dir (122 MB) | ⚠️ **ONE PLACE ONLY** |
| `pilot_val_agents.jsonl` | same dir | **UNTOUCHED**, md5 `85e1bb71b9a4847fb00ff0db184b2fce` |
| `pilot_train_agents.jsonl` | same dir | **UNTOUCHED**, md5 `44881b76fab05cd633e1e158fbc117ac` |

### The two shipped files

| file | lines | distinct `clip_id` | boxes | md5 | use it when |
|---|---:|---:|---:|---|---|
| **`pilot_val_agents_ext.jsonl`** | **6 763** | **34** | **241 228** | `2200e0ac4f71c28c7bc27fcc52c6bc52` | **default** — correct identity, correct time base, ego-compensated, real `occ` |
| `pilot_val_agents_ext_v1sem.jsonl` | 6 763 | 34 | 241 773 | `a20f12382f021b746a8686d8fe2fa86e` | when a result must stay comparable to a banked v1 number (bit-identical to LIVE on the 11 clean overlap episodes) |

### Reproduce

```bash
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack"
D=C:/Users/Admin/tanitad-data
"$PY" stack/scripts/rl_pilot_join.py \
  --epdir      $D/physicalai/_epcache/physicalai-val-bb543bdf7836 \
  --chunks-dir $D/rl-pilot/obstacle.offline.chunks \
               $D/physicalai/labels/obstacle.offline \
               $D/physicalai/labels/obstacle_offline_b1eval \
  --ego-dir    $D/physicalai/labels/egomotion \
               $D/physicalai/labels/egomotion_alpamayo \
  --identity registered --t0-mode registered \
  --geometry ego-compensated --occ-mode fov \
  --out $D/rl-pilot/pilot_val_agents_ext.jsonl
# the train half: same command, --epdir .../physicalai-train-14231cd29c74
#                               --out   .../pilot_train_agents_ext.jsonl
# the v1-semantics drop-in:     drop --t0-mode/--geometry/--occ-mode, keep --identity registered
# the CONTROL (must md5 to 85e1bb71b9a4847fb00ff0db184b2fce):
#   --chunks-dir $D/rl-pilot/obstacle.offline.chunks   (and no other flag)
```

No HuggingFace call is made — `rl_pilot_join.py` has no download path at all, and
`--labels-root` / `--no-download` belong to `build_obstacle_join.py`, which was not used.

### ⛔ Escalation (do not leave in a doc)

1. **Both live pilot joins are contaminated** — val 4/15 (26.7 %), train 13/54 (24.1 %). Decide
   whether banked pilot/probe results stand. Re-run on `pilot_val_agents_ext.jsonl` and
   `pilot_train_agents_ext.jsonl`. **This is a claims-register item, not a data-engineering one.**
2. **The corrected joins live on ONE DISK** (dev box, `C:/Users/Admin/tanitad-data/rl-pilot/`,
   151 MB combined). They are data, not repo content — but nothing else holds them. If that box is
   lost they cost ~4 minutes to rebuild from the command in this manifest, which is why the command
   is here rather than a copy of the file.
3. **263 train + 66 val episodes are blocked on an HF `obstacle.offline` chunk fetch** = a PI
   decision (HF quota is a hard ceiling; nothing was downloaded). The exact uuids to fetch are in
   each file's `.meta.json → identity_log[]` under `why: "verified clip has no local
   obstacle.offline"`. Fetching them would take val to ~100 and train to ~400.
4. **NO_LABEL-as-road-clear needs a consumer-side fix** (§5.6); the span is now banked so it can be
   masked, but no consumer masks on it today.
5. **`--identity prefix` should be considered deprecated.** It is retained only to reproduce the
   historical files; every new join must pass `--identity registered`.

**Evidence class of every number above: MEASURED (ours).** Artifacts: the four jsonl files and
their `.meta.json` sidecars named in this manifest; the reproduction commands above.
