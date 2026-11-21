# COMMS — `2026-08-30-label-source-adjudication`

## Integration status: ⛔ ESCALATED — register rows NOT applied by me

`Project Steering/GOALS_AND_CLAIMS.md` moved under this run (modified 09:15:06, after my 08:53
read). With four agents live I did not bulk-edit it. Paste-ready deltas below.
**Owner to apply: TanitAD Master Mind.** ⚠️ **`D-LAT-AGREE` itself is NOT touched here** — the
PI assigned it to the DataFlyWheel and this package asserts no adjudication.

---

## Delta 1 — CORRECT `D-NUDGE-ABSORB`'s auditability caveat (it went stale the day it was written)

The row states: *"the deciding quantity is NOT PERSISTED. `ego_manoeuvre.py:318` decides NUDGE on
`abs(lat) >= NUDGE_LAT_M` and then **discards `lat`** — no numeric lateral field survives anywhere
in the 4,719-record schema."* **That is no longer true.** Proposed replacement:

> ⚠️ **AUDITABILITY GAP — FIXED 2026-08-30, verify the blob.** `lat_peak_m` is now a `Manoeuvre`
> dataclass field (`stack/tanitad/data/ego_manoeuvre.py:112`) and is emitted (`:379`,
> `lat_peak_m=round(_lat_peak, 3)`); the lateral track is computed **unconditionally**, per the
> `:316` comment — *"It used to live inside the `not is_turn` branch and be thrown away
> immediately after the comparison, which left every NUDGE label with its deciding evidence
> unrecoverable."* ⛔ **STILL OPEN: whether the 4,719-record RELEASE blob was RE-EMITTED with the
> field.** Persisting it in the extractor does not retroactively populate a blob built before the
> change — that check decides whether stratification by `|lat_peak_m|` is available now or needs
> a re-emit.

**Also correct the predicate quotation** (RESULT.md A5): the live condition is
`abs(lat[j]) >= NUDGE_LAT_M and abs(peak) >= 5.0` (`:335`) — **a 5° yaw gate is conjoined**. The
row's bare `abs(lat) >= NUDGE_LAT_M` is incomplete and would mis-predict any low-yaw lateral
translation. ✅ Re-verified unchanged: no `LANE_CHANGE` class exists in `lateral_class` (`:98`),
and `NUDGE_LAT_M = 1.0` (`:82`) has no upper bound.

## Delta 2 — ADD to `D-NUDGE-ABSORB`'s consequence: the instrument gap is one run from closed

> ⭐ **The banked `2026-08-29-lane-detector-deployment` package does NOT close this gap (zero
> detector runs, zero measured transfer — its own F4 requires transfer be MEASURED, precedent
> spanning mild degradation to "near-zero" F1) — but its GEOMETRY already covers this scope.** A
> lane change crosses the boundary of the current lane at **~1.6–1.85 m** (half of 3.2–3.7 m),
> which is **inside** the package's ≤3.1 %-deviation near-pinhole ego corridor (±25°) ⇒ **as-is
> CLRerNet DLA-34-EMA is geometrically sufficient for this adjudication; no cylindrical→pinhole
> rectify primitive is required.** ⚠️ The **39 `LANE_CHANGE` records are absent from that
> package's waiting-consumer list** (68 `turn_suppression` + 41 CoT `CORRIDOR_OFFSET`) and should
> be added as a third set — minutes at its own F5 sizing.

## Delta 3 — Background note for whoever owns `D-LAT-AGREE` (offered, not a ruling)

| point | evidence |
|---|---|
| ⭐⭐ **The two sources are probably NOT conditionally independent** — Alpamayo-R1's auto-labeler is given *"the ego vehicle's trajectory, dynamic states, and meta actions"* ⇒ the VLM saw the geometric source's own input. This **invalidates Dawid-Skene, data programming and spectral meta-learners by their own identifiability assumption**; [`2601.22336`] gives finite-K counterexamples where ignoring the dependence **reverses** the aggregate. ⚠️ **UNVERIFIED for our clips — a 0-GPU provenance check that should precede any method choice.** | PUBLISHED, banked `2511.00088` |
| **At n = 2 the disagreement rate is not evidence about either source** — one observed agreement rate, ≥2 unknown accuracies plus the prior: underdetermined. Breaking it needs a third independent source or a small gold set. | ESTIMATED (my derivation), structure of `1303.3257`/`1407.7644`; `2202.02016` formally |
| ⭐ **The register may already answer the LATERAL axis:** `D-DATA-ALPA-LAT` is MEASURED at chance (lateral 31.2 % vs 23.9 % shuffled, p = 0.335) — and the VLM literature **predicts that a priori** (`2507.20174` LRR-Bench: human-level *"only on the two simplest tasks"* of left/right; `2310.19785`: 56 % vs human 99 %). | MEASURED (ours) + PUBLISHED, banked |
| ⛔ **Do not run cleanlab or co-teaching here** — both assume class-conditional noise; a geometric rule's errors are a function of *x* (instance-dependent by construction), so both select the majority convention and report it clean. | PUBLISHED, banked `1911.00068`, `1804.06872`, `2110.12088` |
| **Admissible instead:** per-annotator heads on a shared trunk (`2110.05719`, preserves attribution), soft targets (`2511.14117`), ordinal adaptive boundary (`2509.02351`), noise-ignorant ERM on frozen features as the first baseline (`2411.00079`); `1805.08877` is the one aggregator needing no independence assumption. | PUBLISHED, banked |

## Decision asked

**Add the 39 `LANE_CHANGE` records to the lane-detector run** (Delta 2) — it converts
`D-NUDGE-ABSORB` from *awaiting an instrument* to *measured*, at minutes of marginal cost, and
the geometry is already established. Owner: DataFlyWheel.
