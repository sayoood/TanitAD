# The val40 obstacle join EXISTS — and distance-keeping is scored on ALL 27 banked dumps

- **Date:** 2026-08-18 · **Discipline:** Benchmarks & Eval · **Status:** PENDING orchestrator triage
- **Evidence class:** `MEASURED (ours, this run)` unless a row says otherwise. Artifacts in `raw/`,
  durable copies in `C:/Users/Admin/tanitad-caches/val40-obstacle-20260818/` (MANIFEST.md5).
- ⛔ **Tier stamp on every number here: T0 — teacher-forced WM diagnostic, NEVER "driving
  performance"** (EVAL_DOCTRINE.md §1.12). The dumps are open-loop, true-future-conditioned.
- **Estimator:** episode-cluster bootstrap / **paired** episode-cluster bootstrap
  (`taniteval/ci.py`), B = 2000, seed 0. ⛔ Never `overlapping_holdout_se`.
- **Cost:** dev-box CPU only, 0 GPU, no pod or Thor traffic, **no HF download, no credentials
  read** (`Keys.txt` untouched — every needed byte was already local and sha256-verifiable).

---

## Lead

**The gap DUMP_LEAD_WIRING.md named is closed.** The VAL-corpus agents JSONL join
(39 clips / 7,400 labelled frames / 195,805 boxes) now exists, built by
`stack/scripts/build_obstacle_join.py`'s own imported functions; `attach_lead` runs green on the
banked dumps (speed cross-check **bit-exact 0.0 on 40/40 episodes**, every dump); and the
**distance-keeping half of the LONGITUDINAL family is scored on all 27 banked arms** — 25 on the
canonical shared 881-window surface, 2 smoke dumps on their own 88-window surface.

| headline (T0) | result |
|---|---|
| Window states (label-side, all shared-surface dumps) | **LEAD 272 · NO_LEAD 548 · NO_LABEL 61** of 881 (LEAD coverage 30.9 %) |
| Cross-validation vs the independent 2026-08-04 pipeline | **874/881 windows same state (99.2 %)**; GT n **247 = exact**; every qualitative conclusion reproduced |
| Instrument control on this surface | GT − CV separated on all three metrics (headway **+0.480 [+0.098, +1.034]**, time-gap **+0.097 [+0.017, +0.201]**, min-TTC **+1.245 [+0.294, +2.623]**) |
| Arms vs the CV floor (paired) | **3 arms separated BETTER on all three** (refc-base-30k, refb-v2-30k, flagship-v4.2-step4000); **overfit/dynin arms separated WORSE than the floor**; flagship-30k better on tg+TTC, ns on headway |
| flagship-30k vs refc-base-30k (paired) | **indistinguishable** — Δhw +0.017 [−0.179, +0.252], Δtg +0.015 [−0.011, +0.049], Δttc +0.237 [−0.607, +1.084], n 220 / 20 episodes |
| ⛔ Admissibility gate | `_longitudinal_claim_admissible` is **False for 25/27 arms** (hold-v0 not separated) — these numbers are **fidelity diagnostics**, not longitudinal capability results. True only for `refc-v12-k16reg` and `flagship-v16-ab-ft`. |

---

## 1. The 40 clip UUIDs — recovered and pinned by FOUR mutually independent constraints

`code/recover_val40_uuids.py` → `raw/uuid_recovery_verify.json`: **16/16 checks pass** (exit
non-zero on any miss). The admissible standard was two agreeing sources; this has four:

| source | class | what it pins |
|---|---|---|
| S1 Thor leadwork index (`…/2026-08-18-thor-stranded-rescue/rescued_beyond_a11/leadwork/val40_lead_index.json`) | INHERITED (in-repo) | ep_00000–39 → full UUID + HF chunk + per-clip label sha256s; 40 distinct UUIDs, 40 distinct id4 prefixes |
| S2 `stack/tanitad/data/deployed_val40_clip_digests.json` | INHERITED (in-repo, itself cross-checked vs `val40_lead_index_ANON.json`) | sha256 of every recovered UUID matches the 40 committed digests **as a set** (order-independent), plus `digest_of_digests` and `clip_id_sha256_sorted` recomputed exactly |
| S3 poses cache `.pt` files (this run) | **MEASURED** | packed `episode_id` big-endian-decodes to UUID[:4] for 40/40 **in val-list order**; poses bytes re-hashed = `manifest_EVALPOD_val40.json` 40/40; T consistent 40/40 |
| S4 NVIDIA's own chunk zips (this run) | **MEASURED** | member `{uuid}.{kind}.parquet` exists in **exactly the chunk S1 names** — egomotion 40/40, obstacle 39/39 — and member **bytes sha256-match S1** (40/40 + 39/39); the single absent obstacle is `ep_00037` (clip `0e0fedfc…`, chunk 768), **confirmed absent in the chunk itself** |

Output: `raw/val40_clipmap.json` (`{eid: uuid}`, the exact `--clip-map` shape `dump_lead_join`
consumes) + `raw/val40_clips.json` (per-episode UUID/chunk/sha record).
🔒 The digests file flags clip ids gated-confidential; full UUIDs already live in-repo in the S1
index — this package adds no new exposure class, but the flag travels here too.

## 2. Label provenance — zero download

All 79 needed parquets (40 egomotion + 39 obstacle.offline) were already inside the local NVIDIA
chunk zips at `C:/Users/Admin/tanitad-data/physicalai/labels/` (the 2026-08-04 package verified
"37/37 needed chunks present"; S4 above re-verified per member, byte-for-byte, against the Thor
index). Durable extract: `C:/Users/Admin/tanitad-caches/val40-obstacle-20260818/{egomotion,
obstacle.offline}/` — **34.9 MB, 79 files, MANIFEST.md5** (plus join/, leadblocks/, clipmap in the
same manifest, 136 files total). The brief's HF-fetch step (and its `Keys.txt` gate) was not
needed; no credentials were read.

## 3. The join — built by the existing builder's own functions

`code/build_val40_join_local.py` is a **thin ingest driver**: poses from the sha-verified
poses-only view + identity from the verified clip map + parquets from the durable extract, then
**everything that computes is `build_obstacle_join.py` imported** (`EgoTrack`, `join_clip` —
registration, rig→world→ego composition, span rule, P4 occ flag — `open_out`, `write_records`,
`verify_with_reader`, `md5_of`, `assert_occ_matches_fov_mask`). No logic forked.

| join fact | value |
|---|---|
| episodes joined | **39/40** (ep_00037 skipped: no `obstacle.offline` — reported, stays NO_LABEL downstream) |
| labelled frames / agent boxes | **7,400 / 195,805** · visible_frac 0.3966 (hfov 120° sensor default; P4 stamp travels in the meta) |
| registration | 40/40 registered, 0 refusals; residual medians 0.0001–0.0098 m; **worst 0.00979 m — identical to the 2026-08-04 run's worst**; grid b ∈ [0.100496, 0.101006] s — **the exact range 2026-08-04 published** |
| reader-verify | `train_p8_occupancy.JoinFileReader`: 7,400 records / 39 clips / occ flags — **OK** |
| outputs | durable `join/val40_agents.jsonl` (md5 `6d3552ffcff5bef31d4ced5484f040f5`, 41 MB) + repo `raw/val40_agents.jsonl.xz` (2.2 MB, xz-roundtrip md5-verified) + `.meta.json` both places |

## 4. `attach_lead` coverage — and the honest 7-window delta vs the 2026-08-04 block

Every shared-surface dump attaches with **speed cross-check max 0.0 m/s over 40/40 episodes**
(the label-free alignment proof: eid→episode mapping, window grid and row order simultaneously).
Episode ep_00037 refuses as `NO_JOIN` with its reason — never counted as free flow.

**NO_LABEL 61 =** ep37 **22** (no obstacle.offline) + span exits ep16 **2**, ep20 **1**, ep28
**9**, ep34 **8**, ep38 **19** (label spans 12.2–19.3 s where the episode runs ~20 s) — the
binding per-family absence reporting, at the episode grain, in every `lead_*.pt.coverage.json`.

Vs the independently-built 2026-08-04 npz block (`…/2026-08-04-distance-keeping-arms/raw/
val40_lead_block.npz`, states 270/551/60): **7/881 windows differ** — 4 NO_LEAD→LEAD (gap0 15.1,
44.6, 59.6, 79.6 m), 2 LEAD→NO_LEAD (old gap0 43.7, 66.1 m), 1 NO_LEAD→NO_LABEL (ep20's span
edge). All far-lead corridor-margin or span-edge cases. **MEASURED mechanism probe:** rebuilding
with `QUERY_EPS_S = 0` lands at 274/546/61 and still differs from the old block on 7 windows — so
the two pipelines' *query time grids* differ at the ±1-frame level (independent implementations,
rounded row times), and the state assignment is boundary-sensitive there in both directions.
**The new block is the quotable surface**: its causal-sampling rule is pinned by a hand-computable
fixture in the committed test suite (13 + 78 neighbour tests green on this box, this run), and its
span rule is conservative by construction (labels-ended ≠ road-clear). The 2026-08-04 npz block
was val40 lead material that DUMP_LEAD_WIRING's four probes (scoped to agents-JSONL joins) did not
count; recorded here so the absence-claim history stays straight.

## 5. The scored panel — T0, all 27 banked dumps

`raw/val40_dk_panel.json` (+ per-arm full four-family blocks in `raw/families/`, each with
LONGITUDINAL incl. `by_speed` strata, LATERAL, and TACTICAL/STRATEGIC reporting their
UNAVAILABLE reasons + n on this surface — clause-5 compliant). dt for the closing rate =
**0.503332 s** (the measured grid; `four_families`' internal sparse inference uses 0.500 —
min-TTC differs by ≤0.02 s, recorded per arm as `_warn`).

**References on the shared surface** (denominator: 272 LEAD windows; an arm keeps a lead only
while its own predicted path stays behind it in-corridor — by design):

| reference | n | mean min-headway (m) | mean min-time-gap (s) | mean min-TTC (s) · n_closing |
|---|---|---|---|---|
| **GT (human)** | **247** | 29.638 | 3.205 (n 221) | 25.271 · 102 |
| **CV floor** (banked `constant_velocity`) | 235 | 28.109 | 3.276 (n 208) | 23.103 · 99 |

*(2026-08-04, old block, for the record: GT 247 / 28.886 / 3.161 / 25.023 — the +0.4–0.9 m
headway shift across ALL arms is the 2 extra far leads (44.6 + 59.6 m gap0) entering the mean.)*

**Headline arms** (paired episode-cluster bootstrap; SEP = CI excludes 0):

| contrast | Δ min-headway (m) | Δ min-time-gap (s) | Δ min-TTC (s) |
|---|---|---|---|
| GT − CV *(instrument control)* | **+0.480 [+0.098, +1.034]** ✅ | **+0.097 [+0.017, +0.201]** ✅ | **+1.245 [+0.294, +2.623]** ✅ |
| refc-base-30k − CV | **+0.322 [+0.011, +0.839]** ✅ | **+0.066 [+0.011, +0.149]** ✅ | **+1.248 [+0.330, +2.765]** ✅ |
| flagship-30k − CV | +0.463 [−0.004, +1.161] ✗ | **+0.090 [+0.013, +0.199]** ✅ | **+1.730 [+0.628, +3.324]** ✅ |
| GT − refc-base-30k | +0.142 [−0.113, +0.431] ✗ | +0.030 [−0.015, +0.087] ✗ | +0.123 [−0.753, +1.008] ✗ |
| GT − flagship-30k | +0.091 [−0.131, +0.264] ✗ | +0.012 [−0.019, +0.038] ✗ | +0.014 [−0.694, +0.809] ✗ |
| flagship-30k − refc-base-30k | +0.017 [−0.179, +0.252] ✗ | +0.015 [−0.011, +0.049] ✗ | +0.237 [−0.607, +1.084] ✗ |

Every 2026-08-04 qualitative conclusion reproduces through the new, durable pipeline: the
instrument separates human from floor; the two headline arms are indistinguishable from each
other **and** from the human (underpowered upper half — ~20 lead-bearing episode clusters);
flagship's lead-retention deficit reproduces (keeps 229/272 vs refc 244/272, GT 247/272; in the
paired cross contrast flagship-only losses 24 vs refc-only 9).

**Full sweep** (pred arm; n = windows where the arm keeps the lead, of 272 LEAD; vs-CV verdicts
paired, ✅⁺ = better separated, ✅⁻ = WORSE separated, ✗ = not separated):

| arm | n | headway | time-gap | min-TTC · n_cl | hw / tg / TTC vs CV | adm. |
|---|---|---|---|---|---|---|
| refa-dinov2 | 250 | 27.876 | 3.039 | 19.15 · 141 | ✗ / ✅⁻ / ✅⁻ | ✗ |
| refc-v12 | 246 | 29.602 | 3.191 | 25.06 · 95 | ✗ / ✗ / ✗ | ✗ |
| refc-v12-identity | 246 | 29.663 | 3.200 | 24.75 · 96 | ✗ / ✗ / ✗ | ✗ |
| refc-v12-k16reg | 246 | 29.616 | 3.193 | 25.00 · 97 | ✗ / ✗ / ✗ | **✅** |
| refc-xl-30k | 246 | 29.663 | 3.200 | 24.75 · 96 | ✗ / ✗ / ✗ | ✗ |
| refc-base-30k | 244 | 29.508 | 3.205 | 25.36 · 92 | ✅⁺/ ✅⁺/ ✅⁺ | ✗ |
| overfit_refa-dynin-20k | 242 | 27.567 | 2.718 | 17.45 · 139 | ✅⁻/ ✅⁻/ ✅⁻ | ✗ |
| refc-xl-live | 242 | 29.526 | 3.187 | 24.36 · 96 | ✗ / ✗ / ✗ | ✗ |
| flagship-nospeed | 241 | 28.748 | 2.986 | 21.35 · 116 | ✅⁻/ ✗ / ✗ | ✗ |
| flagship-speed | 237 | 30.162 | 3.152 | 24.91 · 112 | ✗ / ✗ / ✅⁺ | ✗ |
| flagship-v16-ab-ft | 236 | 30.594 | 3.120 | 24.75 · 95 | ✗ / ✅⁺/ ✅⁺ | **✅** |
| overfit_refa-dynin-30k | 233 | 27.429 | 2.700 | 19.27 · 134 | ✅⁻/ ✅⁻/ ✅⁻ | ✗ |
| refa-dynin-30k | 233 | 27.429 | 2.700 | 19.27 · 134 | ✅⁻/ ✅⁻/ ✅⁻ | ✗ |
| refb | 233 | 29.756 | 3.369 | 23.50 · 85 | ✗ / ✗ / ✗ | ✗ |
| refb-10k | 232 | 29.029 | 3.303 | 23.21 · 90 | ✗ / ✗ / ✗ | ✗ |
| refc-xl | 231 | 31.397 | 3.240 | 23.92 · 92 | ✗ / ✗ / ✗ | ✗ |
| refb-v2-30k | 230 | 29.858 | 3.167 | 24.83 · 94 | ✅⁺/ ✅⁺/ ✅⁺ | ✗ |
| flagship-30k (v1) | 229 | 31.362 | 3.164 | 25.26 · 112 | ✗ / ✅⁺/ ✅⁺ | ✗ |
| flagship-v3enc-10k | 226 | 27.984 | 3.293 | 23.29 · 88 | ✅⁺/ ✅⁺/ ✗ | ✗ |
| refb-v2-20k | 221 | 30.708 | 3.276 | 24.26 · 91 | ✗ / ✅⁺/ ✗ | ✗ |
| flagship-v4.2-step4000 | 218 | 30.855 | 3.334 | 24.50 · 69 | ✅⁺/ ✅⁺/ ✅⁺ | ✗ |
| flagship-v2-6k | 217 | 24.129 | 3.040 | 17.37 · 118 | ✅⁻/ ✅⁻/ ✅⁻ | ✗ |
| flagship-v4.1-10k | 210 | 31.481 | 3.137 | 24.18 · 86 | ✗ / ✗ / ✅⁺ | ✗ |
| overfit_refa-dynin-15k | 172 | 20.503 | 3.295 | 19.33 · 77 | ✗ / ✗ / ✗ | ✗ |
| overfit_refa-dynin-5k | 171 | 20.663 | 3.455 | 17.63 · 103 | ✗ / ✗ / ✅⁻ | ✗ |
| refc-v12-smoke-reg *(own 88-win surface)* | 18/20 | 50.564 | 6.569 | 28.01 · 7 | ✗ (n≤18, 4 eps — unpowered scale) | ✗ |
| refc-v12-smoke-t0 *(own surface)* | 17/20 | 50.885 | 6.727 | 29.03 · 7 | ✗ (unpowered scale) | ✗ |

Readings that survive the gates: **(1)** the dynin/overfit line and flagship-v2-6k are
**separated WORSE than the trivial CV floor** — a clean regression signal this instrument now
sees; **(2)** heavy lead LOSS (n ≪ 247) is itself the failure mode for the overfit arms (their
paths leave the corridor); **(3)** two exact-duplicate rows exist —
`refa-dynin-30k` ≡ `overfit_refa-dynin-30k` and `refc-v12-identity` ≡ `refc-xl-30k` (identical
per-window values; same underlying rollouts banked under two keys) — deduplicate before any
census over "arms".

⛔ **Framing that must travel with the table:** T0 throughout; and for 25/27 arms the anti-echo
hold-v0 control is NOT separated (`_longitudinal_claim_admissible: False`), so these are
world-model **fidelity diagnostics**. The speed-stratified read lives in each
`families_<arm>.json` (`by_speed`: OK strata 3–6, 6–10, 15+ m/s; 0–1, 1–3, 10–15 m/s UNPOWERED
with reasons, e.g. flagship-30k 15+ band: headway 44.62 [30.46, 63.00], time-gap 1.84
[1.39, 2.36] — the high-speed regime carries the *shortest* time gaps).

## 6. Relationship to "the program's first distance-keeping numbers"

The **first** distance-keeping numbers were 2026-08-04's (two arms, one-off npz path, dev-box).
Today's package is the first through the **durable program pipeline**
(`build_obstacle_join.py` jsonl → `dump_lead_join.attach_lead` → `four_families`), the first
covering **all 27 banked dumps**, and it doubles as an independent reproduction of the 2026-08-04
result (§4–§5). DUMP_LEAD_WIRING.md's step 1 ("build the VAL-corpus jsonl join … pod-side") is
discharged **without any pod**: the label parquets were local all along.

## 7. Reproduce

```bash
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PP="C:/Users/Admin/wt-tanitad-local/taniteval;C:/Users/Admin/wt-tanitad-local/stack;C:/Users/Admin/wt-tanitad-local/stack/scripts"
cd /c/Users/Admin/wt-tanitad-local
IN="TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-18-val40-lead-join"

# 1. UUID recovery + 16-check verification (writes clipmap + verify jsons)
PYTHONUTF8=1 $PY "$IN/code/recover_val40_uuids.py" --out-dir "$IN/raw"

# 2. the join (5.5 s)
PYTHONUTF8=1 PYTHONPATH=$PP $PY "$IN/code/build_val40_join_local.py" \
  --out C:/Users/Admin/tanitad-caches/val40-obstacle-20260818/join/val40_agents.jsonl \
  --xz-copy "$IN/raw/val40_agents.jsonl.xz"

# 3. attach one dump via the module CLI (what the wiring report specified)
PYTHONUTF8=1 PYTHONPATH=$PP $PY -m taniteval.dump_lead_join \
  --windows taniteval/results/windows_flagship-30k.pt \
  --agents C:/Users/Admin/tanitad-caches/val40-obstacle-20260818/join/val40_agents.jsonl \
  --epdir C:/Users/Admin/tanitad-caches/val40-poses-20260818/physicalai-val-0c5f7dac3b11 \
  --clip-map "$IN/raw/val40_clipmap.json" --out lead_flagship-30k.pt

# 4. the full 27-dump panel (48 s)
PYTHONUTF8=1 PYTHONPATH=$PP $PY "$IN/code/score_val40_dumps.py" --out-dir "$IN/raw"
```

## 8. Escalations

1. **Registry:** the LONGITUDINAL distance-keeping rows for the banked arms can now cite
   `raw/val40_dk_panel.json`. `MODEL_REGISTRY.md` is owned by a sibling stream right now —
   **not edited here**; orchestrator to route the pointer.
2. **Duplicate dumps** (`refa-dynin-30k`≡`overfit_refa-dynin-30k`,
   `refc-v12-identity`≡`refc-xl-30k`): flag to whoever owns `taniteval/results/` naming.
3. The 2026-08-04 npz block as an uncounted val40 lead surface (§4) is a small
   absence-claim correction to DUMP_LEAD_WIRING.md's coverage table — report-only here, since
   that package is another agent's deliverable.
4. `four_families._distance_keeping` consumes `path_steps`, `attach_lead` emits `wp_steps` —
   NOT a defect on today's sparse dumps (shapes already match; block `dt_s` carried the measured
   grid), but the first DENSE dump scored against this block will need `path_steps=[4,9,14,19]`
   set caller-side, or the shapes refuse. Noted for the wiring's owner; nothing edited.
