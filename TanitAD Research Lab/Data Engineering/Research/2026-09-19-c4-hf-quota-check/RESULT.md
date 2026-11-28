# E18 / C4 — HF storage against the corpus-rebuild push: **it fits today, but the ceiling will not stop us if it doesn't**

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind (task 3 of 3, PI-approved)
**Answers** `PREREG_REFCV6_DEVBOX_PREPARATION.md` §3.3 **C4** and checklist item **E18**.
⛔ **CHECK ONLY.** GET requests and metadata reads. Nothing was uploaded, created, deleted or
changed. Token read in place from the git-ignored `Keys.txt`, never printed or passed on argv.

## The answer

| | GB | class |
|---|---|---|
| PRIVATE storage used (20 repos, sum of the Hub's own per-repo `usedStorage`) | **176.554** | MEASURED · `raw/hf_quota_check.json` |
| PRO private storage included | **1,000** (1 TB, read as decimal — the conservative reading) | PUBLISHED · HF storage-limits doc |
| ⇒ **PRIVATE remaining** | **823.446** | derived |
| PUBLIC storage used (28 repos), against "up to 10 TB included" | 901.340 | MEASURED · PUBLISHED |

| push, PRIVATE (the rule for augmented data) | private after | headroom |
|---|---|---|
| the pre-registration's **+112.9** | 289.454 | +710.5 |
| ⭐ **the 408×1024 corpus cache, 386.5** | **563.054** | **+436.9** ✅ |
| the 256×1024 corpus cache, 273.6 | 450.154 | +549.8 ✅ |

⇒ **E18 passes as the account stands today: the rebuild push fits inside the included private
storage with 436.9 GB to spare.**

## ⛔ Three things the C4 row gets wrong or leaves out

### 1. +112.9 GB is not the size of any push

It is **386.5 − 273.6**, the marginal disk of choosing 408×1024 over 256×1024. Nothing of
273.6 GB exists on HF to be replaced — the 256×1024 corpus cache was projected, never built.
A push adds **all 386.5 GB**. Deleting files afterwards does not give it back: the Hub keeps
LFS history until a super-squash, which is irreversible and takes up to 36 h to show in the
quota (PUBLISHED). ⚠️ **386.5 is itself ESTIMATED and probably HIGH**: it is 4,713 × the
139-clip sample's 82.0 MB, and that sample's source clips run **1.121×** the corpus mean. It
was not rescaled. ⚠️ And `corpus_cost_projection_408.json` files it under the key
`cache_gb_256x1024` — a 408 figure under a 256 label, in a banked artifact that should not be
rewritten. Quote it as *408×1024, 386.5 GB*.

### 2. ⛔ The ceiling does not enforce itself

HF's published policy: above the included 1 TB, PRO private storage is **charged to the
payment method, pay-as-you-go, base $18/TB/month** — not refused. The account reads
**`canPay: True`, `billingMode: prepaid`** (MEASURED, `whoami-v2`). ⇒ **no 402 or 413 will
ever fire to stop an over-ceiling push.** The standing "abort on a quota signal" guard is
blind to this case. **Our own arithmetic before the push is the only thing that enforces the
PI's hard ceiling.** ⚠️ Whether prepaid mode would refuse at zero credit rather than bill is
**not established** here.

⛔ **No endpoint states the ceiling or the account total.** Four probes: `whoami-v2` 200 and
`users/{u}/overview` 200, both with no storage field; `settings/billing/usage` 404;
`settings/storage` 404. ⇒ "remaining" = PUBLISHED limit − MEASURED sum of per-repo
`usedStorage`. The sum is the proxy; the billed figure itself is not queryable.

### 3. ⛔ The biggest repo on the account is PUBLIC, and flipping it private changes the answer

`Sayood/tanitad-physicalai-w120-256x640cyl` — **455.377 GB, PUBLIC** — holds the **parity**
corpus caches (`epcache-256px-phase0` 349.5 GB, parity train 85.0 GB, parity val 21.2 GB;
current tree, `raw/public_256x640_breakdown.json`). **3,000 of its 6,061 file names carry a
raw clip id.** The programme treats clip ids as gated-confidential, and augmented or changed
datasets belong PRIVATE (CLAUDE.md, programme redesign 2026-08-22). That makes it a candidate
for going private — and then:

| if the 256×640 cache goes PRIVATE | private after | headroom |
|---|---|---|
| nothing else | 631.931 | +368.1 |
| ⛔ **+ the 408×1024 push** | **1,018.431** | **−18.4 → OVER, billed** |
| + the 256×1024 push instead | 905.531 | +94.5 |

⇒ **Two decisions that are each policy-consistent breach the PI's ceiling together.** Their
order matters, and both are the PI's: repo visibility is outward-facing, and a squash is
irreversible. Ways to stay under: push 256×1024 instead of 408×1024; retire the phase-0
`epcache` (349.5 GB, if nothing still reads it) via super-squash **before** the push; or keep
the rebuild off HF. **None was actioned here.**

## Adjacent facts, measured

* `periodEnd` is **2026-10-01** (MEASURED). The auto-memory's *"period ends 2026-12-31"*
  (2026-08-22) is stale.
* 48 repos; `usedStorage` returned for all 48.

## Sources

* HF, *Storage limits* — `https://huggingface.co/docs/hub/storage-limits`, fetched
  2026-09-19: PRO private 1 TB + pay-as-you-go at $18/TB/mo base; PRO public up to 10 TB
  included, responsible use asked; LFS pointer deletion frees nothing; squash reflects within
  36 h. ⚠️ **Not banked in the Library** — a policy page, and pricing pages change. Re-fetch
  before relying on the dollar figure.
* `code/hf_quota_check.py` → `raw/hf_quota_check.json`, `raw/public_256x640_breakdown.json`.
