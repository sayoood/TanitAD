<title>The sign oracle is frozen — and its committed gate turns out to be one test, not two, with no recall clause at all</title>

# `E-DE-SIGN-4` (LR15-5): the 42-clip oracle is **frozen and pinned** (`sha256 f9931330…`) — and characterising the gate it will run found **three defects in the gate itself**: the Wilson clause is **vacuous**, a reader that exactly meets the 0.80 bar passes only **53 %** of the time, and **the one published reader for this task would either fail outright or score 100 %, depending on a convention the gate never states**

**2026-09-20 · Research Lab (LAB-RUN-017) · Data Engineering · consumes LR15-5 · serves the MM standing question (max-speed INPUT supplier, 2026-09-10)**
⛔ **0 GPU, no network for the freeze; arithmetic only.** The freeze is deterministic — two independent runs produced byte-identical output (`sha256 f9931330b2cdc1e1b27a3bfebae976f2e602f7987efeb7211621e37f849150d3`).

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ✅ **The oracle is frozen.** `raw/sign_value_oracle_v1.json` — **42 gated** Vienna-convention clips (km/h) plus **1 carried** US clip (35 mph, excluded from the gate), each with clip-id, value, unit, country, day/night and the flag state. Provenance pins the source artifact's own `sha256 99662d0b…`, so the freeze traces to a measurement, not to this script's reading of one. Day **25** / night **17**; **9 distinct values**; **14 countries** (France 7, Portugal 6, Austria 5, Denmark 4, …). | MEASURED |
| **F2** | ⛔⭐ **The gate is ONE test, not two. The Wilson clause is vacuous at n = 42.** LR15-5 committed *"agreement ≥ 0.80 **AND** Wilson 95 % LB ≥ 0.65"*. Every k clearing 0.80 on 42 also clears 0.65 on the lower bound (k = 34 ⇒ point 0.8095, LB **0.6670**), so the gate reduces exactly to **k ≥ 34 of 42**. The second clause would bind only for **n ∈ 5…36** — it was written for a smaller set than the one that exists. | MEASURED (exact, `raw/sign_value_oracle_v1.json` → `gate_characterisation.per_k_table`) |
| **F3** | ⛔⭐⭐ **A reader whose true accuracy is exactly the nominal bar passes the gate 53 % of the time.** Exact binomial: P(pass \| p = 0.80) = **0.5309**. To pass 4 times in 5 a reader needs true accuracy **0.8438**, not 0.80. And the gate leaks downward: P(pass \| p = 0.75) = **0.2429** — a reader 5 pp *below* the bar is admitted about **once in four**. ⇒ **"≥ 80 % accurate" is not what this gate tests**, and quoting it that way to the PI would misstate the instrument. | MEASURED (exact binomial, n = 42, k₀ = 34) |
| **F4** | ⛔⭐⭐⭐ **The gate has no recall clause, and the only published operating point for this exact task is recall-limited.** `2606.08860` (banked, **re-read locally from the PDF**, not via the summariser): on **39 posted signs** in 35 minutes of in-house data, a VLM speed-limit reader attains *"95.45 % precision and 53.85 % recall, with no incorrect speed classifications and a single false positive"*, the recall loss attributed to *"poor visibility conditions such as nighttime rain or heavy fog"*. ⇒ **Run that reader against our gate and the verdict flips on an unstated convention**: score a no-read as a disagreement and it gets ≈ 22.6 of 42 (**FAILS**, needs 34); exclude no-reads and it gets **100 % (PASSES perfectly)**. LR15-5 never says which. | MEASURED (ours) + PUBLISHED (`2606.08860`, local pypdf re-read) |
| **F5** | ⭐ **The gate is not trivially beatable, which is the one thing it does well.** A reader that answers the plurality value (30 km/h) on every clip scores **0.3095**; one sampling from the oracle's own value marginal scores **0.1803** in expectation. Both are far below k ≥ 34 — so a pass is not obtainable without reading. **This is why F2–F4 are repairs, not a rejection.** | MEASURED |
| **F6** | ⚠️ **The night subgroup cannot see the published failure mode.** 17 night clips vs 25 day; the smallest day–night accuracy gap distinguishable at 95 % is **24.7 pp**. `2606.08860` locates its recall loss *at night* — and our set, at **40.5 % night**, has the exposure but **not the power** to measure it. | MEASURED (normal approximation at p = 0.80 in both arms) |

---

## 1 · Controls

| control | bar | read | |
|---|---|---|---|
| source record count | 4,572 | **4,572** | ✅ |
| valued clips = landed E-DE-SIGN-3 | 43 | **43** | ✅ |
| system split | 42 Vienna / 1 US | **42 / 1** | ✅ |
| multi-value clips / illegal steps | 0 / 0 | **0 / 0** | ✅ |
| clip-ids unique | 43 distinct | **43** | ✅ |
| ⭐ **determinism** — a freeze that is not reproducible is not a freeze | two runs, same sha256 | **f9931330… twice** | ✅ |
| ⭐ **null floors must NOT pass** (a gate any guesser clears is not a gate) | majority & marginal both fail | **0.3095 / 0.1803, both fail** | ✅ |

⚠️ **What the assertions cannot check:** every count here is inherited from the E-DE-SIGN-3 parser. This package re-verifies the source's *internal* consistency and pins its hash; it does **not** re-parse the CoT. **Recall remains unbounded** — 43 is a lower bound, and a clip's absence from this file is not evidence that it has no sign.

## 2 · The gate, measured

Operating characteristic of **k ≥ 34 of 42**, exact binomial:

| true accuracy | 0.65 | 0.70 | 0.75 | **0.80** | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|
| **P(pass)** | 0.019 | 0.080 | 0.243 | **0.531** | 0.831 | 0.979 | 1.000 |

⭐ Read the row as a decision, not a table: the bar the programme would *say* is 0.80 behaves like a bar at **0.844** for a reader that wants to pass reliably, and like a bar at **≈ 0.75** for one that gets lucky.

## 3 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | The MM standing question is blocked on a **supplier**, and `E-DE-SIGN-1` (a sign reader) is the one in-corpus, ego-free candidate. This file is the validation set that request lacked; the gate is what will admit or refuse the supplier. |
| **CONSEQUENCE** | `E-DE-SIGN-1` can now go to the PI with a frozen, hashed validation set — **and must go with a corrected gate.** As written, the gate (a) is single-clause, (b) is a coin flip at its own nominal bar, and (c) is silent on abstention, which is the axis the only published reader actually fails on. ⛔ Fixing it is a **pre-registration change and therefore a Master Mind / PI motion**, not the Lab's: the Lab measured the instrument and states the repair, it does not adopt it. |
| **COMBINATION** | Sits directly on `FS19-2`'s result (2026-09-19): nuPlan **LV+PIT** is a 100 %-coverage, map-sourced, zero-ego posted-limit supplier. ⇒ The programme now has **two** candidate suppliers with opposite shapes — a *map* source with full coverage but a **non-parity corpus** (a PI parity decision), and a *vision* source on the parity corpus whose coverage is 43 of 4,572 and whose published analogue reads barely half the signs. F4 is what makes them comparable: **map coverage is 100 % by construction; reader coverage is a recall number nobody has measured on our corpus.** |
| **CHANCES / RISKS** | **Chance:** the gate's defects are all cheap to fix and were found *before* it refused anything. **Risks:** (a) the CoT is itself a VLM reading, so agreement here is reader-vs-VLM, never reader-vs-truth — a reader could score 42/42 and still be wrong about the road; (b) **eval is 0/147**, so a reader validated here has **no** validation reference where it would be used; (c) 14 countries × 9 values × 42 clips means most cells hold one clip — no per-country or per-value claim is available at any n; (d) F6: the published failure mode is exactly the one our set is underpowered for. |
| **EXPERIMENT** | ⭐ **`E-DE-SIGN-5` (proposed DE20-1, 0 GPU, arithmetic + a spec edit):** re-specify the gate before it runs, as **three** explicit clauses — **(i) value accuracy on READ clips ≥ 0.80**, **(ii) an explicit abstention convention** (proposed: a no-read is a FAIL for the gate and is *also* reported separately as recall), **(iii) a recall clause: reads ≥ 0.70 of the 42**, with the operating characteristic of each clause published alongside it. **Committed:** the re-specified gate must reject a reader matching `2606.08860`'s published profile (100 % value accuracy at 53.85 % recall) — if it admits that reader, the clause set is still wrong, because a supplier that misses **half** the signs cannot feed a max-speed input channel. ⛔ The gate change itself needs MM/PI sign-off; the Lab supplies the arithmetic. |

## 4 · Stopping condition (Rule Zero)

**(3a) for the freeze** — LR15-5's deliverable exists, is hashed and is reproducible. **(3c) for the gate** — characterising it turned up three defects, so the honest output is a **named repair (`E-DE-SIGN-5`) and an escalation**, not a quiet adoption of a bar the Lab now knows does not mean what it says.

`Deliverables: RESULT.md · code/freeze_and_characterise.py · raw/{sign_value_oracle_v1.json, sign_value_oracle_v1.sha256, freeze_and_characterise.log, search_log.md}`
