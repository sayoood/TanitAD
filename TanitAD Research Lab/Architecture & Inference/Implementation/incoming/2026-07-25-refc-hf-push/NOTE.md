# REF-C XL + base → HuggingFace — ✅ **COMPLETE.** Both 30 k FINAL checkpoints are HF-backed, public + gated-manual

**Date:** 2026-07-25 · **Agent:** HF-push subagent · **Pod used:** `tanitad-eval` (not training; upload is
network/CPU only — no GPU or RAM load added to any training pod)

---

## Headline

The two missing REF-C rungs are published and md5-verified. **The REF-C ladder is now fully HF-backed and
no longer single-disk.**

| model | repo | visibility | md5 vs registry | `ckpt.pt` |
|---|---|---|---|---|
| **REF-C-XL** (251,932,584 p) | `https://huggingface.co/Sayood/tanitad-refc-xl` | **public + gated `manual`** | ✅ `966d4eff…4e198` (§4.1) | ✅ **3,024,021,445 B** |
| **REF-C-base / medium** (104,191,577 p) | `https://huggingface.co/Sayood/tanitad-refc-base` | **public + gated `manual`** | ✅ `8f10d6f9…74939` (§4.3) | ✅ **1,250,838,325 B** |

Uploaded 3.02 GB in **12 s (246 MB/s)** and 1.25 GB in **6 s (215 MB/s)** — pushing from the pod, not the
~1 MB/s dev-box relay.

**Authorization:** Sayed chose **Option A** (public + `gated="manual"`) after the private-storage 403 was
reported; relayed by the coordinator. Visibility was **not** changed on any other repo.

---

## 1. Why this took two passes — the 403 (MEASURED, kept for the record)

First attempt created both repos **private** (the brief's tie-break: never more open than the siblings,
and the direct REF-C sibling `tanitad-refc-small-evalonly` is private). Cards, `config.json` and
`metrics.json` uploaded fine; **both `ckpt.pt` uploads then failed:**

```
HTTP 403  POST https://huggingface.co/api/models/Sayood/tanitad-refc-xl/commit/main
{"error":"Private repository storage limit reached, please upgrade your plan to
          increase your private storage limit"}
```

**Not auth, not a size limit.** Token (`Keys.txt`, displayName `TanitAD`) authenticates as **Sayood, role
`write`** — the same session created the repos and uploaded four small files each. *(This also clears the
earlier "invalid HF token on pod3" worry: the token in `Keys.txt` is good.)*
Account is free tier (`isPro: false`, `canPay: false`, `orgs: []`).

The freed space was **public**; **private** was the exhausted bucket. Public + gated-manual is the tier all
five other checkpoint-bearing `tanitad-*` repos already use.

⚠️ **The numeric private-storage limit remains UNVERIFIED.** HF exposes no quota endpoint I could find —
probed `…/users/Sayood/storage`, `/quota`, `/billing`, `/api/settings/storage` (all **404**) and
`…/overview?expand=storage`, `whoami-v2?expand=storage` (**200**, no storage fields). Only the server's
refusal is measured, never the threshold.

---

## 2. Execution order — the weights were never world-downloadable, not even briefly

Sayed's mandated sequence, enforced in `hf_gate_publish_push.py` with a hard abort between steps:

| step | XL | base |
|---|---|---|
| 0 · before | `private=True gated=false` | `private=True gated=false` |
| 1 · **set `gated="manual"` while still PRIVATE** | readback `private=True gated=manual` ✅ | readback `private=True gated=manual` ✅ |
| 2 · flip `private=False` | requested | requested |
| 3 · **hard gate — verify public AND gated=manual** | `public=True gated_manual=True` ✅ | `public=True gated_manual=True` ✅ |
| 4 · **re-verify md5 vs registry** | MATCH ✅ | MATCH ✅ |
| 5 · upload `ckpt.pt` | 12 s, 246 MB/s | 6 s, 215 MB/s |
| 6 · final API readback | `private=False gated=manual` | `private=False gated=manual` |

Gating stuck **while the repo was still private**, so at no point was there a public-ungated window. The
script aborts without uploading weights if step 3 or step 4 fails.

### The gate demonstrably bites (MEASURED — anonymous, no `Authorization` header)

```
GET https://huggingface.co/Sayood/tanitad-refc-xl/resolve/main/ckpt.pt    -> 401
    x-error-code: GatedRepo
    "Access to model Sayood/tanitad-refc-xl is restricted. You must have access to it
     and be authenticated to access it. Please log in."
GET https://huggingface.co/Sayood/tanitad-refc-base/resolve/main/ckpt.pt  -> 401  (same)
GET https://huggingface.co/api/models/<repo>                              -> 200  (card metadata public)
```

**Card and file list are world-readable; the PhysicalAI-derived weights are not fetchable without Sayed
approving each access request.** That is the property the gating requirement existed to guarantee, and it
is verified rather than assumed.

---

## 3. Integrity — md5 gate passed at every stage (MEASURED)

| file | md5 measured (×2: pre-flight and again pre-upload) | registry | bytes local | bytes on HF |
|---|---|---|---|---|
| `refc-xl-30k/ckpt.pt` | `966d4eff1ea5ddf86efba01b8344e198` | §4.1 identical ✅ | 3,024,021,445 | **3,024,021,445** ✅ |
| `refc-base-30k/ckpt.pt` | `8f10d6f934f4199e11ddc7352e074939` | §4.3 identical ✅ | 1,250,838,325 | **1,250,838,325** ✅ |

Contents confirmed by direct load (`map_location='cpu'`, mmap):
`{model: state_dict, opt: optimizer state, step: 29999}` — **673 tensors** (XL) / **487** (base).
`step == 29999` in both, matching "COMPLETE at step 29,999". **No drift at any point.**

Final file lists (independent API readback, separate process from the uploader):

```
Sayood/tanitad-refc-xl    public gated=manual  sha 6c6e92e787d21581719fe86ee4723faccf0e017f
    .gitattributes 1519 · README.md 12092 · ckpt.pt 3024021445 · config.json 2241 · metrics.json 920
Sayood/tanitad-refc-base  public gated=manual  sha 17f477d7a29ccec4ce988a37d75b103ce8c98b05
    .gitattributes 1519 · README.md 14490 · ckpt.pt 1250838325 · config.json 3174 · metrics.json 1015
```

---

## 4. `Sayood/` storage — before → after (MEASURED, summed over `list_repo_tree`)

| | before | after | Δ |
|---|---|---|---|
| **PRIVATE** | 75.974 GB | **75.974 GB** | 0.000 |
| **PUBLIC** | 144.230 GB | **148.505 GB** | **+4.275** |
| **GRAND TOTAL** | 220.204 GB | **224.479 GB** | +4.275 |

Δ is exactly 3.024 + 1.251 GB — the two checkpoints, nothing else. Private is untouched, so the quota
that blocked pass 1 is no more consumed than before.

**Private headroom remains at zero.** Largest private consumers, for whenever the next private push is
needed:

| repo | GB |
|---|---|
| `Sayood/final_technical_dataset` (dataset) | 18.233 ⬅ **byte-size-identical to the next row — probable duplicate pair, 36.5 GB** |
| `Sayood/final_user_friendly_dataset` (dataset) | 18.233 ⬅ same |
| `Sayood/final_technical_model_Qwe25_7B` | 16.584 |
| `Sayood/TanitDataSet-C` (dataset) | 15.926 |
| `Sayood/mixed_dataset_reduced_7k` (dataset) | 3.606 |
| `Sayood/tanitad-internal` | 3.134 |

---

## 5. ⚠️ Sibling-visibility inconsistency — **FLAGGED FOR SAYED, deliberately NOT acted on**

The visibility survey found a **pre-existing inconsistency**, and this change makes one part of it sharper.
Recording it because it is a real inconsistency, not because anything should be changed automatically.

| repo | private | gated | holds weights? |
|---|---|---|---|
| `Sayood/tanitad-refc-small-evalonly` | **True** | False | 219 MB `ckpt_evalonly.pt` (no card, no config) |
| `Sayood/tanitad-internal` | **True** | False | 3.13 GB `ckpt_step8500.pt` |
| `Sayood/tanitad-refc-base` | False | **manual** | ✅ new |
| `Sayood/tanitad-refc-xl` | False | **manual** | ✅ new |
| `Sayood/tanitad-flagship-4b-speedjerk` | False | **manual** | 3.31 GB |
| `Sayood/tanitad-flagship-4b-phase0` | False | **manual** | 3.30 GB |
| `Sayood/tanitad-refb-speed` | False | **manual** | 3.26 GB |
| `Sayood/tanitad-refa-dinov2-4b` | False | **manual** | 1.91 GB |
| `Sayood/tanitad-refa-dynin-4b` | False | **manual** | 1.91 GB |

**Two things worth Sayed's eye:**

1. **The REF-C ladder is now internally inconsistent** — `small` is **private/ungated**, while `base` and
   `xl` are **public/gated-manual**. The three rungs are quoted together constantly (§4.2/§4.3 scale
   study), so a reader with access to two of them cannot reproduce the third.
2. **`tanitad-refc-small-evalonly` and `tanitad-internal` are private but UNGATED.** Private makes that
   harmless today, but if either is ever flipped public it would become world-downloadable **immediately**
   — no gate to catch it. Setting `gated="manual"` on both *while they are still private* is a free,
   zero-visibility-change hardening step (it is exactly what step 1 above did, and it stuck cleanly).

**Neither was touched.** Changing the small rung's visibility is Sayed's call, not an agent's.

---

## 6. Model cards — what they carry

Registry-accurate. Every number cites `MODEL_REGISTRY.md` §4.1 / §4.3 or the raw eval JSON; every interval
names its estimator (**episode-cluster bootstrap, B=2000 over 40 val episodes / 881 windows**, paired form
for deltas); eval split named as the clean held-out **`physicalai-val-0c5f7dac3b11`**; corpus parity key
**`physicalai-train-e438721ae894`**, skip-hash **`f09e44db`**; exact training command, optimizer (**Adam**,
not AdamW), loss weights and measured per-module param breakdown.

Caveats carried openly rather than hidden: the deprecated `overlapping_holdout_se` `±`
(**1.28–2.06× too narrow**; 1.45× for XL), the flagship-v1 **tie** (Δ +0.0443 [−0.0544, +0.1465], not
separated), the **base≈XL tie** (Δ +0.0013 [−0.0281, +0.0316], per-window corr 0.789 so the test is not
weak), the **un-refined-anchor selection flaw**, the **~92 % irreducible** oracle gap (with the refuted
speed-cost and refined-confidence dead ends), the **v1-vs-v2.1 route-label confound** and that the clean
control run has not been done, the **reconstruction-OOD-confounded** AlpaSim closed-loop numbers, and the
declared **TanitEval-is-uncommitted** reproducibility gap.

---

## 7. Proposed MODEL_REGISTRY edits — **staged text, NOT applied by me** (coordinator will apply)

⚠️ Anchor on row text, not line numbers — the registry moved ~40 lines during this session.

### §4.1 REF-C-XL — **replace** the existing row

Current (last row of the §4.1 field table): `| **HF** | none |`

```
| **HF** | ✅ **`Sayood/tanitad-refc-xl`** — public + **gated `manual`** (access by owner approval), pushed 2026-07-25. Files: `ckpt.pt` **3,024,021,445 B** (md5 `966d4eff1ea5ddf86efba01b8344e198`, re-verified against this row immediately before upload), `config.json`, `metrics.json`, model card. Anonymous `resolve/ckpt.pt` returns **401 `GatedRepo`** — weights are not world-downloadable. Note: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-25-refc-hf-push/NOTE.md` |
```

### §4.3 REF-C-base — **ADD a row (it currently has NO `HF` row at all)**

Insert as the last row of the §4.3 field table, after the
`| **Note** | …2026-07-20-refc-medium-scaling.md… |` row:

```
| **HF** | ✅ **`Sayood/tanitad-refc-base`** — public + **gated `manual`**, pushed 2026-07-25. Files: `ckpt.pt` **1,250,838,325 B** (md5 `8f10d6f934f4199e11ddc7352e074939`, re-verified immediately before upload), `config.json`, `metrics.json`, model card. Anonymous `resolve/ckpt.pt` returns **401 `GatedRepo`**. Note: `…/incoming/2026-07-25-refc-hf-push/NOTE.md` |
```

### §4.2 REF-C-small — **ADD a row (also missing)**

```
| **HF** | ⚠️ **`Sayood/tanitad-refc-small-evalonly`** — **private, UNGATED**, holds `ckpt_evalonly.pt` (219,057,698 B) only; **no card, no `config.json`**. Inconsistent with §4.1/§4.3 (public + gated-manual, full ckpt + card + config). Tidy-up for Sayed — see §5 of `…/incoming/2026-07-25-refc-hf-push/NOTE.md`. |
```

### §11 risk register — **narrowed, not added**

The originally-proposed **R16** ("REF-C XL + base have no HF backup") is **resolved on arrival** — both are
now HF-backed and md5-verified, so no new risk row is needed. What remains worth a line is the *class*:

`R15` (`dynenc-branchB`, push blocked for lack of HF auth on pod3) is now the **only** remaining blocked
backup, and this session establishes the working recipe for it: **push from an HF-authenticated pod with
the `Keys.txt` token on stdin, to a public + `gated="manual"` repo** (private storage is full). Suggest
updating R15's "unblock" column to point at
`…/incoming/2026-07-25-refc-hf-push/hf_gate_publish_push.py` as the precedent.

---

## 8. Root-cause class — for `RETRACTION_LOG.md` discipline

No retraction (nothing false was published), but the near-miss has a nameable class worth recording:

**Class: "backup path assumed available, never probed."** The task was scoped as *"Sayed freed HF storage,
push the two models"* — the freed space was **public**, while the conservative visibility choice landed in
the one bucket that was full. A 3 GB transfer was started before anything confirmed the target could
accept it. **The 30-second check that catches this: write one small file to a repo at the intended final
visibility before starting the large transfer.** Same failure *shape* as R15 (`dynenc-branchB`): a backup
believed in progress that never produced a second copy.

Second, smaller class: **"sibling convention" is not always singular.** Two conflicting conventions existed
side by side (§5); a brief that says "match the siblings" is underdetermined whenever the siblings disagree,
and the tie-break has to be stated in advance — as this brief did, correctly.

---

## Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| This note | `repo: TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-25-refc-hf-push/NOTE.md` (**staged**) | no |
| REF-C-XL model card | `repo: …/README_xl.md` (**staged**) · `HF: Sayood/tanitad-refc-xl/README.md` · `tanitad-eval:/root/refc_hf/` | no |
| REF-C-base model card | `repo: …/README_base.md` (**staged**) · `HF: Sayood/tanitad-refc-base/README.md` · `tanitad-eval:/root/refc_hf/` | no |
| Gate→publish→push script (the reusable recipe) | `repo: …/hf_gate_publish_push.py` (**staged**) · `tanitad-eval:/root/refc_hf/` | no |
| Initial push script (md5-gated, idempotent) | `repo: …/hf_push_refc.py` (**staged**) · `tanitad-eval:/root/refc_hf/` | no |
| 403 diagnostic script | `repo: …/hf_diag_retry.py` (**staged**) · `tanitad-eval:/root/refc_hf/` | no |
| Storage-accounting script | `repo: …/hf_storage_total.py` (**staged**) · `tanitad-eval:/root/refc_hf/` | no |
| Anonymous gate-effect check | `repo: …/hf_gate_effect_check.py` (**staged**) · `tanitad-eval:/root/refc_hf/` | no |
| All raw logs (403 detail + gate/publish/push) | `repo: …/push_result.txt` (**staged**) · `tanitad-eval:/root/refc_hf/*.log` | no |
| **REF-C-XL `ckpt.pt`** | **`HF: Sayood/tanitad-refc-xl`** · `tanitad-eval:/root/models/refc-xl-30k/` · `tanitad-pod3:/workspace/experiments/refc-diffusion-xl-30k/` | ✅ **no — now 3 copies** |
| **REF-C-base `ckpt.pt`** | **`HF: Sayood/tanitad-refc-base`** · `tanitad-eval:/root/models/refc-base-30k/` · `tanitad-pod3:/workspace/experiments/refc-diffusion-base-v21-30k/` | ✅ **no — now 3 copies** |

No token was ever written to a file, an argv, or a log: read in place from `Keys.txt`, piped over the ssh
channel to the script's **stdin**. Verified no HF token cache was created on the pod
(`/root/.cache/huggingface/token` does not exist).

## Escalation (headline, per Operating Standard rule 3)

1. **Registry rows in §7 need applying** — §4.1 replace, §4.3 **add**, §4.2 **add**, R15 unblock-column
   update. The brief said both sections read "none"; in fact **only §4.1 had an `HF` row at all**.
2. **§5 sibling-visibility inconsistency is for Sayed** — the REF-C ladder is now split
   (small private/ungated vs base+xl public/gated), and two private repos carry **no gate** should they
   ever be published. Zero-cost hardening available; not done without his say-so.
