<title>SPEC E-DE-SIGN-3 — re-derive the VALUED speed-limit readings from the CoT text</title>

# `E-DE-SIGN-3` — pre-registration

`Research Lab · 2026-09-18 (LAB-RUN-015) · Data Engineering · serves LR14-10 and the MM standing research question (max-speed supplier)`
⛔ Pinned before the code was run (`raw/prereg_pin.json`).

## 1 · Why

`E-DE-SIGN-2` (09-17) stratified the **69** clips whose `cot_tokens.speed_limit` flag is set. But that
field is a **boolean**: 69 is a *mention* count, not a count of readings that carry a value. The
programme has been quoting **"31 of 4,572"** valued CoT readings (the MM's standing question). That
figure has no artifact path in the 09-17 package. `E-DE-SIGN-1`'s positive control needs **valued**
readings (a sign *number* to check a reader against), so the count must be re-derived from the text.

## 2 · Method (0 GPU, text only, no ego)

Source text: `semantics.cot` (full CoT string) of every record in the v7.2 release
(`…/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_{train,eval}.jsonl.gz`).
A **value** is a 1–3-digit integer bound to a speed-limit context by one of four literal patterns:
`N km/h|kph|kmh`, `N mph`, `N speed limit`, and `speed limit (sign)? (of|to|indicates|indicating|shows|showing|reads|reading|is|at)? N`.
Durations (`N seconds`, `N-M seconds`) and list ordinals (`N. type`) are excluded by construction.
⛔ Nothing is derived from ego speed. That is the refuted path.

**Legal-step check, independently authored from road law rather than from the data:** km/h values ∈
{5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130}; mph values ∈ multiples of 5 in [15, 85].

## 3 · Controls (each must read a known value)

| id | bar |
|---|---|
| record counts | train **4,572**, eval **147** — else VOID |
| flag count | train `speed_limit` flags **= 69** (09-17's number) — else the read is not the same corpus |
| parser literal tests | `"a 50 km/h sign is posted"`→50 · `"overhead speed limit sign indicates 100"`→100 · `"the 30 speed limit sign"`→30 · `"1. Type: speed limit sign"`→None · `"duration: 0-4 seconds"`→None. All five must pass before the corpus is read |
| mutation | a naive parser (*first integer in the sentence*) must FAIL at least one literal test. Otherwise the tests cannot discriminate |

## 4 · Committed outcomes

| branch | condition | consequence |
|---|---|---|
| **SPECIFIED** | ≥ **20** valued readings in train from **Vienna-convention** countries **and** ≥ 95 % of all valued readings on the legal-step set | `E-DE-SIGN-1`'s positive control is fully specified |
| **THIN** | < **10** valued Vienna readings | `E-DE-SIGN-1` carries an explicit *"no validation on eval, weak validation on train"* clause into its approval request |
| **MIDDLE** | 10–19 | usable as a smoke check only; state it |
| **UNLAWFUL** | < 95 % legal-step | the parser or the CoT is unreliable; the control is not usable as a value oracle |

Also reported, not barred: whether the quoted **"31"** reproduces; eval count (09-17 measured 0/147 flags).
