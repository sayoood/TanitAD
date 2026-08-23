# ⚠️ PARTIAL / INTERRUPTED — Cosmos-Drive-Dreams + Cosmos-Reason survey (2026-07-21)

> **This is not a survey result. It is a salvage note.**
> The commissioned primary-source survey (Drive-Dreams pipeline / consumption / throughput /
> licence, plus the Reason-family confirmation) **was aborted before it returned any web content**.
> Written so the context is not lost a second time. **Pick up after Jul 26, 00:00 Berlin** (weekly
> API cap reset).

**Session outcome in one line: ZERO new externally-verified facts were established today.**
Both primary fetches (`arxiv.org/abs/2506.09042`, the GitHub raw README) failed with a
tool-classifier availability error and returned no content. Everything in §1 below is **prior repo
work that I collated and cross-read today** — it is *not* today's verification, and nobody should
cite it as such.

---

## 1. What is established (all of it pre-existing, none of it re-verified today)

### 1a. Dataset shape — PUBLISHED (cited), inherited citation, NOT re-verified 2026-07-21

Source of these numbers is the **2026-07-14 Data-Engineering note**
(`Research/2026-07-14-cosmos-drive-dreams-loader-and-landscape.md` §2), which cites the
[HF dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams),
[cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits) and
[arXiv 2506.09042](https://arxiv.org/abs/2506.09042).

| fact | value | status |
|---|---|---|
| repo ID | `nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams` | PUBLISHED (inherited) |
| real RDS-HQ clips | **5 843**, 10 s each | PUBLISHED (inherited) |
| synthetic videos | **81 802** | PUBLISHED (inherited) |
| weathers | **7** — Foggy / Golden_hour / Morning / Night / Rainy / Snowy / Sunny | PUBLISHED (inherited) |
| synthetic RGB | mp4 **@ 30 fps** ⚠️ see §3a | PUBLISHED (inherited) |
| pose | per-frame 4×4 `vehicle_pose` `.npy`, ego→world | PUBLISHED (inherited) |
| intrinsics | `pinhole_intrinsic` `[fx,fy,cx,cy,w,h]` | PUBLISHED (inherited) |
| also on disk | HD map + LiDAR (our loader ignores both) | PUBLISHED (inherited) |
| front camera | `front_wide_120fov` — same 120° HFOV as PhysicalAI-AV front-wide | PUBLISHED (inherited) |
| licence | **CC-BY-4.0**, "confirmed on the HF card" | PUBLISHED (inherited) ⚠️ see §3b |

**Useful corroboration:** the commissioning brief independently guessed "5 843 real / 81 802
synthetic" and the repo's 2026-07-14 note — written a week earlier — carries **exactly** those two
numbers. Two independent arrivals at the same figures. But both ultimately trace to the *same* HF
card, so this raises confidence, it does not constitute a second source.

### 1b. Licence / redistribution — MEASURED (ours), and this is the decision-relevant part

The upstream CC-BY-4.0 is **not** what governs us. `TANITDATASET_TIER_INTEGRATION_2026-07-21.md` §4
states the binding rule:

```
tier(derivative) = strictest( tier(source_record), tier(generator_model), tier(conditioning_labels) )
```

> **Synthetic augmentation does not launder a licence.**

Therefore **the Drive-Dreams renders we already hold of PhysicalAI clips
(`tanitad-eval:/root/vlm_pilot/frames/*_{Rainy,Foggy,Snowy,Golden_hour}`) remain `gated`** — not
shippable and not publishable, however synthetic they look. This is settled and needs no further
web research. It also means *any* future Drive-Dreams work on our parity corpus inherits `gated`,
which caps the strategic upside of the whole direction before a single GPU-hour is spent.

### 1c. Our own value evidence for re-rendering — MEASURED (ours), and it is weak

Photometric re-render (weather/night/fog) is logged as augmentation **A4** with the note:
⚠️ *"Our own OOD numbers are unimpressive"* — **cosmos 29.4 % win-rate vs 49.7 % in-distribution.*
Cost recorded only as the qualitative **"H100-class"**. No clips/GPU-hour figure exists anywhere in
the repo.

### 1d. Cosmos-Reason gating — MEASURED (ours), 2026-07-20 byte-pull with the Sayood token

- `nvidia/Cosmos-Reason1-7B` — **UNGATED**
- `nvidia/Cosmos-Reason2-32B` — **UNGATED**
- `nvidia/Cosmos-Reason2-2B`, `-8B` — **gated (auto-approve)**
- Cosmos3-Nano/Super — OpenMDW-1.1 omnimodel, commercial-OK, served via vllm-omni/sglang, **not**
  vanilla vllm.

Related prior artifact: `Data Engineering/Implementation/incoming/2026-07-20-vlm-reason1-vs-reason2/`.

---

## 2. What is ruled out (and why)

1. **Drive-Dreams as a licence-upgrade path for our parity corpus — RULED OUT.** Not by research
   but by the §1b derivative-tier rule. Re-rendering PhysicalAI clips in 7 weathers yields `gated`
   output. No survey finding can change this, so it should not be re-litigated on pickup.
2. **A4 photometric re-render as a headline robustness lever — RULED OUT as a priority** on our own
   measurement (29.4 % vs 49.7 %, §1c). Demoted below A1/A2 (cross-source label transfer, temporal
   restride), which are zero-GPU and zero-licence-risk.
3. **Fan-out subagent research on this topic — RULED OUT operationally.** Parallel fan-out is what
   exhausted the weekly budget. Pickup must be serial and fetch-budgeted.

Nothing in Parts A or B was ruled out *on evidence gathered today*, because none was gathered.

---

## 3. Open conflicts worth carrying forward

### 3a. ⚠️ The 121-frame cap does not reconcile with our recorded 30 fps — RESOLVE FIRST

The brief asks about a **121-frame generation cap**. Our repo records **10-second clips at 30 fps**.
These do not fit:

- 10 s × 30 fps = **300 frames** ≫ 121
- 121 frames @ 30 fps = **4.03 s** — far short of a 10 s clip
- 121 frames @ 12 fps = **10.08 s ≈ 10 s** ✅ arithmetically clean

**ESTIMATED (mine, unverified):** the likely reconciliation is that generation is ~121 frames at
~12 fps and the *container//released* mp4 is re-timed or interpolated to 30 fps — i.e. our "30 fps"
is the container rate, not the generation rate. **This is arithmetic, not evidence.** Do not build
on it.

Why it matters: it decides whether a **20 s clip is one generation, two chained generations, or
impossible**. Every throughput and cost estimate depends on the answer, so this is the single
highest-value item to settle on pickup — and it is cheap (one HTML fetch of the paper).

### 3b. The CC-BY-4.0 claim should be re-confirmed at the byte level

Recorded as "confirmed on the HF card" in July 2026 by an agent that did not re-check it later.
Given the tier machinery keys off `license_class` constants, an upstream card edit would be silent.
Low probability, but it is a one-fetch check.

---

## 4. NOT REACHED — the actual pickup list (after Jul 26)

Nothing below was attempted successfully. Ordered by value, cheapest-decisive first.

**Part A — Drive-Dreams**
1. **⭐ Throughput** — GPU-hours for the 81 802-video set, per-clip seconds, GPU class (A40 vs
   H100), VRAM. *Nothing whatsoever exists in the repo.* This was flagged CRITICAL in the brief and
   is the main gap.
2. **⭐ Consumes-vs-emits** — does the pipeline require HD-map + 3D-bbox renders + LiDAR *rendered to
   video* as conditioning? **Is the ego trajectory baked into the render** (change the path ⇒
   re-render map/boxes from a new pose) or is there a separate action/control input? Needs a
   verbatim paper quote. This is the finding that decides whether Drive-Dreams can serve
   augmentation **A5 (counterfactual rollout, same past / different ego action)** — the only
   augmentation that attacks the ~92 % aleatoric oracle bound. **A5 is the one worth reviving.**
3. Frame cap / multi-view — confirm 121, confirm whether 7-camera multi-view generation is
   supported and its cost multiple. See §3a.
4. Full pipeline stage list + **exact HF model IDs** (Cosmos-Drive-Dreams-7B?
   Cosmos-Transfer1-7B-Sample-AV? single-view→multi-view model? LLM prompt-rewriter?).
5. Downstream gains — 3D lane detection in rain, the "93× real data" 3D-object-detection
   equivalence — exact numbers **with the table they came from**.
6. Resolution, total size GB/TB, licence URL, commercial-use statement.

**Part B — Reason family** (beyond the §1d gating facts, none of this was reached)
7. Confirm from the HF cards that Reason-1 / Reason-2 are **VLM reasoning/understanding models, NOT
   video generators** — one quoted sentence per family.
8. Exact model IDs, release dates, params, input/output modalities, licence + URL.

**Suggested pickup budget:** ~6–8 fetches, serial, no subagents. Items 1–3 alone justify the run;
items 4–8 are follow-on.

---

## 5. Provenance of this note

- Written directly, no subagents (per coordinator instruction).
- Prior-work sources read today:
  `TanitAD Research Hub/Data Engineering/Research/2026-07-14-cosmos-drive-dreams-loader-and-landscape.md`,
  `TanitAD Research Hub/Data Engineering/TANITDATASET_TIER_INTEGRATION_2026-07-21.md`.
- §1d is from operator memory of a 2026-07-20 byte-pull, not re-checked today.
- Staged, not committed, not pushed. pod3 / pod1 untouched.
