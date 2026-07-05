# License Review — NVIDIA PhysicalAI-Autonomous-Vehicles (D-002 blocker note)

**Agent:** Data Engineering. **Date:** 2026-07-07. **Purpose:** close the D-002 / PROJECT_STATE §4 open
item "PhysicalAI-AV license review before use in any public claim". **Access:** verified as HF user
`Sayood` (gated=auto, accepted). **Source:** the dataset card + `LICENSE.pdf` of
[`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
read 2026-07-07.

> ⚠️ **This is an engineer's read to route a decision, not legal advice.** Any *public* use of the real
> PhysicalAI-AV datasets needs Sayed / NVIDIA-legal sign-off. The safe default until then is in §5.

## 1. The AV family and its licenses (measured via `HfApi.dataset_info`)

| HF dataset | Gated | License | Role |
|---|---|---|---|
| `nvidia/PhysicalAI-Autonomous-Vehicles` (real) | auto | **other** = NVIDIA AV Dataset License | multi-view + CoC/VLA labels |
| `nvidia/PhysicalAI-Autonomous-Vehicles-NCore` (real) | auto | **other** = same | NCore variant |
| `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (real) | auto | **other** = same | NuRec neural-reconstruction |
| `nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams` (synthetic) | **False** | **CC-BY-4.0** | Cosmos-generated driving |

The three **real** datasets carry the custom **NVIDIA Autonomous Vehicle Dataset License Agreement**. The
**synthetic** Cosmos-Drive-Dreams is plain **CC-BY-4.0** and ungated.

## 2. What the NVIDIA AV Dataset License actually says (key clauses, verbatim-sourced)

- **§1 License Grant.** "non-exclusive, revocable, non-transferable, non-sublicensable … to download, use,
  modify, and reproduce the Dataset, in each case **solely for your internal development of autonomous
  vehicles and automated driving assisted systems using NVIDIA technology** ('Purpose')." NVIDIA may force
  you to update to a new version and delete prior ones on request.
- **§2 Authorized Users.** Only your (and Affiliates') employees/contractors, from your **secure network**,
  for the Purpose.
- **§3 Confidentiality.** The Dataset is treated as **NVIDIA Confidential Information** — **do not disclose
  to third parties** except Authorized Users under equivalent confidentiality terms.
- **§4 Limitations.** §4.1 no surveillance and **"will not … enable law enforcement or any public authority
  to enforce any rules or regulations including any road traffic laws"**; §4.2 keep all proprietary
  notices; §4.3 rights are **"for the Purpose only."** Elsewhere: no labelling of race/ethnicity/gender/
  age/health, **no biometric processing, no face/gaze analysis.**
- **§6 Ownership.** NVIDIA retains all IP; **no implied license.**
- **§7 Feedback.** Anything you send back → perpetual, irrevocable, sublicensable, royalty-free license to
  NVIDIA.
- **§8 Term.** **Expires 12 months** after download; auto-terminates on breach or if you litigate NVIDIA;
  on termination you **must destroy all copies**, and **the granted licenses do NOT survive** termination.

## 3. Consequence for TanitAD (the decision this note exists to make)

- **Public claims / benchmark numbers from the REAL PhysicalAI-AV datasets: NO (blocked).** The grant is
  **internal-development-only** and the data is **confidential**; there is no publication/attribution
  permission (contrast CC-BY). Publishing "TanitAD scores X on PhysicalAI-AV" or releasing derived
  weights/pseudo-labels as a public artifact is outside the Purpose and breaches §3/§4.3 absent explicit
  NVIDIA permission. **This directly qualifies the D-009 line** that named PhysicalAI-AV the "second
  source": fine for *internal* dev, **not** for any public number.
- **Internal development use: PLAUSIBLY yes, with one caveat.** The Purpose is AV/ADAS development **"using
  NVIDIA technology"** — we train on NVIDIA GPUs (RTX 4060 / A40 / Orin/Thor target), which plausibly
  satisfies it, but "using NVIDIA technology" is exactly the phrase a lawyer should confirm.
- **12-month expiry is a reproducibility poison pill.** Any pipeline or public result that *depends* on
  this data has a legal shelf-life and a "destroy all copies" obligation on termination — unsuitable for
  the reproducible public artifact P7 wants.
- **Data-ethics restrictions** (no biometric/face/gaze/demographic labelling) are compatible with our
  no-perception-label thesis, but constrain any future attribute-aware work.

## 4. G-D1 fields (for the recommended-dataset ledger)

| Field | PhysicalAI-AV (real) | Cosmos-Drive-Dreams (synthetic) |
|---|---|---|
| License | NVIDIA AV Dataset License (**internal-dev-only, confidential, 12-mo**) | **CC-BY-4.0** (attribution) |
| Gated | yes (auto-approve, token) | no |
| Size | ~70,775 files (real multi-view; tens of GB+) | ~99,352 files |
| Actions available | poses/ego-state + CoC/VLA labels (multi-view) | synthetic driving (Cosmos) |
| Public-claim safe? | **No** without NVIDIA sign-off | **Yes** (with attribution) |
| Cost to first batch | gated loader + secure-network handling; **defer** until internal-only need is real | standard loader |

## 5. Recommendation (actionable)

1. **Keep comma2k19 (MIT) as the public-claims corpus** — all D1–D3 public numbers ride on it (already the
   D-009 primary). PhysicalAI-AV real data does **not** back any public claim.
2. **If** we want PhysicalAI-AV's multi-view / CoC-VLA richness, use it **internal-only**, on a secure
   store, and **get Sayed + NVIDIA-legal written confirmation** that (a) "using NVIDIA technology" is
   satisfied and (b) publishing *aggregate* results is permitted — before any such number leaves the repo.
   Track it as a `Project Steering/Proposals/` item, not a silent adoption.
3. **For publicly-claimable synthetic AV data, prefer `Cosmos-Drive-Dreams` (CC-BY-4.0)** — the one AV
   asset in the family that is redistribution- and publication-safe with attribution. Candidate second
   corpus for public work; loader hardening (backlog #2) should target it first, not the gated real sets.
4. **Do not build the gated real loader on the dev box** under a 12-month clock without a concrete
   internal-only use; it adds legal surface for no Phase-0 public benefit.

## 6. Verdict

**D-002 resolved:** real PhysicalAI-AV = **internal-development-only, confidential, 12-month** NVIDIA
license → **excluded from all public claims**; comma2k19 (MIT) remains the public corpus; Cosmos-Drive-
Dreams (CC-BY-4.0) is the publicly-safe PhysicalAI-AV alternative. Escalate any internal-only adoption to a
proposal with legal sign-off.

Sources: [PhysicalAI-Autonomous-Vehicles card + LICENSE.pdf](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) ·
[Cosmos-Drive-Dreams (CC-BY-4.0)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) ·
license clauses §1–§10 read 2026-07-07 as HF user `Sayood`.
