# YouTube-dashcam → IDM: ToS / licensing tier verdict

**Purpose:** land a clear TanitDataSet-tier verdict for the three artifacts the IDM/YouTube line would
produce, so the ingest decision (S0 of `PIPELINE_DESIGN.md`) has a rule to follow, not a vibe.

> ⚠️ **This is engineering tier-mapping and risk-flagging, NOT legal advice.** I am not a lawyer. Every
> `refuse`/`nc` call below is the *conservative program rule* under the operating-standard principle
> **"unverified ≠ permissive"** (`TANITDATASET_TIER_INTEGRATION_2026-07-21.md` §2). Anything at
> `refuse` or "pending review" needs a human legal sign-off before it is relied on. Evidence classes are
> marked; the ToS/precedent facts are `PUBLISHED (cited, verified this task)`.

---

## 1. The machinery this plugs into (recap)

From `TANITDATASET_TIER_INTEGRATION_2026-07-21.md`:
- Tiers: **`ship`** (`commercial_ok`) · **`ship-sa`** (share-alike, segregated) · **`nc`** (research
  only, not shippable) · **🔴 `refuse`** (never ingested, on any tier — contamination survives training).
- **`license_class` is a `SOURCE_REGISTRY` CONSTANT, never inferred**; the firewall is code
  (`assemble_lake_record` raises).
- ⭐ **The augmentation rule (§4):** `tier(derivative) = strictest( tier(source), tier(generator),
  tier(conditioning_labels) )`. **Synthetic/derived processing does NOT launder a licence.**

The IDM pseudo-labeler is an *augmentation generator*; YouTube frames are a *source*. The rule decides
everything below.

---

## 2. The verdict — three artifacts

### (a) Raw YouTube dashcam frames  →  🔴 `refuse` to ship/re-host · `nc`-with-caveats internal-only
**Two independent restrictions stack, and they do not behave the same way:**

1. **YouTube ToS (contract, vs Google).** "You may not download any Content unless a 'download' button
   or similar link is displayed by YouTube for that Content"; scraping is barred unless you are a public
   search engine, use the official API, or have written permission [YouTube ToS, verified this task].
   yt-dlp ingest **breaches this contract** — independent of copyright. This is a restriction on the
   **act of obtaining** the bytes.
2. **Copyright (property, vs each uploader).** Each dashcam video is the uploader's copyrighted work,
   default all-rights-reserved. We hold **no licence** to reproduce or redistribute the frames.

**Tier call:** **`refuse` for any re-hosting or shipping of the frames** — this is the same call the
tier survey already made for **TLD**, flagged `refuse`-adjacent *specifically because its provenance is
"scraped YouTube + Honda LOKI/NC"* (§2 of the tier doc). We must not become a second TLD. Internal
research *use* of self-downloaded frames is the grey zone the H7 strategy already scoped as
"training-internal only pending legal review" — treat as **`nc`, provenance-logged, never in a shippable
shard**, and even that carries the unresolved ToS-breach question, so it is **`nc`-with-caveats**, not
clean `nc`.

### (b) IDM pseudo-labels (steer / yaw / accel / target-speed / ego-traj)  →  inherits the source: 🔴 `refuse` if bundled with frames; **`nc` at best as a standalone annotation layer**
By the augmentation rule, the labels' tier = strictest(source=YouTube-frame, generator=IDM-model-tier,
conditioning=none). The **source dominates** → the labels inherit YouTube's `refuse`/`nc`-caveats tier.
- **Bundled with the frames** (a re-hosted labeled clip set): 🔴 `refuse` — contamination survives, this
  is the TLD failure mode again.
- **Standalone annotation layer** keyed to public YouTube **URLs + timestamps** (no frames): the *numbers*
  are our measurements/facts, not the copyrighted expression — this is the only potentially-shippable
  form, and only as **`nc`** pending review (it still *points at* ToS-encumbered content and is only
  meaningful once a user re-downloads the frames themselves). **This is the OpenDV model** (§4).

> Subtlety worth flagging to legal: pure numeric telemetry (a trajectory) is closer to a *fact* than to
> a *derivative work* of the video. That is the argument for an annotations-only layer being lighter than
> the frames. It is an argument, not a settled answer — mark `HYPOTHESIS`, do not build a shipping shard
> on it without sign-off.

### (c) A world model PRETRAINED on YouTube-derived data  →  internal-research OK, provenance-stamped; **public/commercial release gated on legal review**
This is the artifact the `refuse` class was *invented* to reason about — but YouTube is **not** Waymo:
- **Waymo/Waymax are `refuse`** because "terms **follow the trained weights** into vehicle operation" —
  the licence reaches our end product (§2 tier doc).
- **YouTube's ToS does NOT claim rights over models** trained on downloaded content. The ToS breach is a
  **contract** matter (with Google, over the act of downloading) that does not, on its face, propagate a
  property interest into our weights. The residual risk in the weights is **copyright fair-use** (is
  training on copyrighted frames fair use?) — **unsettled and actively litigated**, with no controlling
  autonomous-driving precedent.

**Tier call:** the pretrained WM is **usable INTERNALLY for research**, with a **mandatory provenance
stamp** ("YouTube-pretrained prefix") and the **parity firewall** keeping it a *separate checkpoint* from
the parity fine-tune (so it is traceable and removable — `PIPELINE_DESIGN.md` §4). **Public release or
commercial deployment of a YouTube-pretrained WM is gated on legal review** — do not treat internal
research latitude as deployment clearance. Because the prefix is firewalled, a clean-provenance model
(parity-only, or `ship`-tier-only pretrain) remains reproducible if legal says no.

---

## 3. Why the two risk vectors must be tracked separately

| vector | who | triggered by | "spent" when | reaches the weights? |
|---|---|---|---|---|
| **ToS contract** | Google | the **act of downloading/scraping** | at download time (it's about access) | **No** (a contract breach, not a property right in the model) |
| **Copyright** | each uploader | **reproduction/redistribution**; training is the fair-use question | never fully — persists in any re-hosted frame | frames: yes; weights: **unsettled** |

The practical consequence: **the frames are the hot potato, not the model.** Minimise frame handling
(never re-host), and the heavier of the two vectors is contained.

---

## 4. The mitigation is a known, precedented pattern — adopt it verbatim

**OpenDV-YouTube** (the 1700 h corpus behind GenAD/Vista, our nearest published anchors) ships a **CSV
of YouTube video IDs + a download script + language annotations — NOT the video bytes**; it is
"designed for you to download the videos yourself from YouTube rather than providing pre-hosted versions,
respecting YouTube's licensing requirements" [OpenDV/GenAD, arXiv:2403.09630, verified this task].

**Adopt this exactly** (`PIPELINE_DESIGN.md` S0):
1. TanitDataSet ships **pointers** (video ID + time range + our IDM annotations), never frames.
2. Frames are downloaded **on the user's own machine** at use time; they live in the `nc`/gated,
   never-shipped zone, exactly like PhysicalAI renders (`gated`, §4 tier doc: "still `gated`, however
   synthetic they look").
3. The IDM annotation layer is the *only* candidate for a shippable artifact, at **`nc` pending review**.

This keeps us on all-fours with the field's accepted practice and out of the TLD/`refuse` hole.

---

## 5. `refuse`-adjacent flags (the "unverified ≠ permissive" checklist)

- 🔴 **Do not re-host YouTube frames or frame-bundled labels** under any tier. (= TLD's mistake.)
- 🔴 **Do not add a YouTube source to `SOURCE_REGISTRY` with an inferred permissive class** — it needs an
  explicit constant, and the honest constant is `refuse`-for-shipping / `nc`-caveats-internal.
- ⚠️ **Channel-level licences vary:** some dashcam channels post under Creative Commons (YouTube's CC-BY
  setting). **CC-BY frames are a genuinely lighter tier** (`ship-sa`-like, attribution) — but this must
  be **read per-video from the actual licence field, never assumed**, and re-hosting still meets the ToS
  download-mechanism restriction. A CC-licensed *subset* is the cleanest first slice if S0 is greenlit.
- ⚠️ **Speedometer OCR / on-screen text** (a candidate metric-scale cue, IDM design §4) may itself be a
  copyright/overlay concern per channel — tag, don't assume.

---

## 6. Recommendation

| question | verdict | class |
|---|---|---|
| Train **internally** on self-downloaded YouTube frames? | Yes, `nc`-caveats, provenance-logged, firewalled from parity | program-rule |
| Re-host the frames / labeled clips? | **No — `refuse`** | conservative rule |
| Ship an **annotations-only** layer (URLs + IDM labels)? | Maybe, **`nc` pending legal review**; use the OpenDV pointer model | `HYPOTHESIS` |
| Ship / deploy a **YouTube-pretrained WM**? | **Gated on legal review**; internal research OK with provenance stamp | conservative rule |
| Cheapest clean first slice, if greenlit? | **CC-BY-licensed dashcam channels only**, read per-video | recommendation |

**Bottom line for the IDM line:** the licensing does **not** block the *internal* research use that the
proof + pretraining need — so the IDM/YouTube thesis can be *investigated* without a legal gate. It
**does** block re-hosting frames and gates any public artifact. The firewall that makes this safe is the
same one already in the stack: **ship pointers, never bytes; keep the YouTube prefix a separate,
provenance-stamped checkpoint.** Route the three `HYPOTHESIS`/`gated` rows to Sayed + legal before any
public or commercial step.

**Sources (verified this task):** YouTube ToS download/scraping clause · OpenDV-YouTube / GenAD
(arXiv:2403.09630) distribution model · Vista (arXiv:2405.17398, in-domain anchor). Tier machinery:
`TANITDATASET_TIER_INTEGRATION_2026-07-21.md`.
