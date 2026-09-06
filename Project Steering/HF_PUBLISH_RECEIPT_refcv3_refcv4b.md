# HF publish receipt — refcv3 / refcv4b

**Date:** 2026-09-06 (Europe/Berlin) · **Account:** `Sayood` (HF Pro, verified via `whoami`)
**Written to `C:\` because the G: mount was down at run start** (`git rev-parse` failed through
8 retries + a Drive-client restart in the preceding attempt; the mount also dropped mid-run here,
deleting the shell cwd). Copy into the repo when G: is stable.

---

## P0 — storage, measured before any upload

| reading | value |
|---|---|
| `GET /api/users/Sayood/storage` | **HTTP 404** — endpoint does not exist |
| `GET /api/users/Sayood/billing` | **HTTP 404** — endpoint does not exist |
| `whoami` | `isPro=True`, `periodEnd=1790812800`, **no storage/quota/limit key anywhere in the payload** |
| per-repo `?expand[]=usedStorage`, summed over all 46 owned repos | **1,064,523,423,631 B = 991.415 GiB** (43 repos report a value; 3 report 0) |

⛔ **No numeric ceiling is exposed by any endpoint.** 991.415 GiB is consumption, not headroom.
The account-wide `usedStorage: 0` seen on the `?author=` listing is a field HF does not populate
there — the per-repo endpoint does populate it, which is why the sweep is per-repo.
Largest single consumers: `tanitad-v6` 139.32 GB, `tanitad-physicalai-w120-256x640cyl` 455.38 GB,
`tanitad-comma2k19-episodes` 88.43 GB, `tanitad-v7-training-corpus` 63.67 GB.
Nothing observed suggests the account is near a limit — but that is an absence of a signal, not a
measured margin.

---

## What was already there (checked before uploading — no duplicates created)

`Sayood/tanitad-refc-v3` already existed, **created 2026-09-03, PUBLIC**, and already held all
three refcv3 checkpoints. The registry line was right. Nothing was re-uploaded for refcv3.

`Sayood/tanitad-refc-v4b` already existed, **created 2026-09-06 20:38 UTC, PRIVATE**, holding the
README / config.json / anchors / metrics / train.log banked by attempt 2 — but **no checkpoint**.

---

## Uploads performed by this run

Repo: **`Sayood/tanitad-refc-v4b`** — https://huggingface.co/Sayood/tanitad-refc-v4b —
**PRIVATE** (`private=True` verified immediately before and after every upload).

| file | bytes | md5 at source | sha256 at source | sha256 as HF reports it | action | verified |
|---|---|---|---|---|---|---|
| `ckpt_40284_FINAL.pt` | 1,285,301,425 | `99b573e8277d94a5e3bfbf630cb4d751` | `090ee8d2584acc81e3738ab28f8f5bf2ed6bfb0d0cc6fba887de2a0cbddcbaa4` | **identical** | uploaded (1255.9 s) | ✅ |
| `ckpt_40284.pt` | 428,616,885 | `b99c66a13ce42059d4201be14f965fab` | `6558731cfe8183ebed71de3a6ed3c0ea1567bedf51136c2553b545c74490a606` | **identical** | already present → **skipped** | ✅ |
| `eval/*.json` (8 files) | ~2.86 MB | — | — | — | already present → skipped | ✅ |

The source md5 for `ckpt_40284_FINAL.pt` matches the value the brief carried
(`99b573e8…4d751`) and the value in the card's own Files table. HF exposes sha256, not md5, so
identity against HF is asserted by sha256; both match byte-for-byte.

### ⚠️ A concurrent publisher was writing to this repo during the run

`list_repo_commits` shows a second process committing with the prefix
`publish refcv4b @ step 40284 (...)` — `ckpt_40284.pt` at **21:42:39 UTC** (one minute *after* my
pre-flight survey showed the repo checkpoint-less) and the eval JSONs plus its own
`ckpt_40284_FINAL.pt` at **22:02:11–22:02:17 UTC**. My own commit,
`Add refcv4b checkpoint @ step 40284 (ckpt_40284_FINAL.pt)`, landed at **22:04:46 UTC**.

Almost certainly attempt 2's uploader was still alive after its host process was declared dead.
**No damage:** the skip-if-sha256-matches guard did its job — `ckpt_40284.pt` was correctly
detected and *not* re-uploaded — and the two writers produced byte-identical content, confirmed by
sha256 after the fact. But it means the repo had two writers, and that is worth knowing.

### Not uploaded, because it was already there

**refcv3 needed no upload at all.** `Sayood/tanitad-refc-v3` already held:

| file | bytes | HF sha256 | check |
|---|---|---|---|
| `ckpt_30000.pt` | 428,519,790 | `ff3da459…06e9` | **byte-identical to the local file** (local sha256 recomputed and matched; local md5 `00da81c6efcd91e7b618a1fbddb3b78f`) |
| `ckpt_40284.pt` | 428,518,255 | `5a163eaf…4203` | published 2026-09-03/04 |
| `ckpt_40284_FINAL.pt` | 1,284,991,701 | `cb40fb76…fab9a` | the v4b card records this exact sha256 as pod-verified |

⇒ **refcv3@40284 was NOT pulled from `tanitad-refc-v3` pod in this run — it did not need to be.**
It was already published, and its sha256 on HF equals the pod-verified digest recorded in the
sibling card. The A40 was left alone.

---

## Verification

HF stores an LFS/Xet **sha256** per binary file, not an md5 — so identity against HF is asserted by
sha256, and md5 is recorded at source for the card and for cross-checking the pod.

### Content assertion (`torch.load`, not filename)

| file | `ckpt['step']` | tensor count | sampled non-zero norm |
|---|---|---|---|
| refcv4b `ckpt_40284_FINAL.pt` | **40284** | 1592 (551 model + optimizer) | **40/40** |
| refcv4b `ckpt_40284.pt` | **40284** | 551 | **40/40** |
| refcv3 `ckpt_30000.pt` | **30000** | 544 | **40/40** |

The step is embedded **inside** each checkpoint, so the step labels are not filename-trust.

### Identity assertion (from config, not name)

- **refcv4b** — `--anchors …/anchors.pt`, `--anchor-v0-conditioned`, `--anchor-control-units alat`;
  `anchors.shape = [117, 8, 2]` → **117 anchors**; `v0_conditioned: true`. 56-item `argv`.
  The `anchors.pt` already in the repo has `sha256 e86cf507…e8fb`, which equals the config's
  recorded `file_sha256` — the bank on HF is the bank the run used.
- **refcv3** — **no anchor flag and no `anchors` key at all**; 43-item `argv`.
### Strongest evidence — anchor counts read out of the WEIGHTS themselves

The published `ckpt_40284.pt` of each arm was opened and its anchor tensors measured. This does not
rely on the filename, the repo name, or even the config:

| tensor | refcv3 @ 40284 | refcv4b @ 40284 |
|---|---|---|
| `core.decoder.anchors` | **(128, 8, 2)** | **(117, 8, 2)** |
| `core.decoder.anchor_controls` | *absent* | (117, 2) |
| `core.decoder.lat_to_anchor.weight` | (128, 3) | (117, 3) |
| `core.decoder.lon_to_anchor.weight` | (128, 3) | (117, 3) |
| `ego_inj` / `ego_to_tac` / `ego_to_str` | *absent* | (32,5)+(32,) / (512,32)+(512,) / (256,32)+(256,) |

⇒ **refcv3 really is a 128-anchor arm (chance = 1/128 = 0.0078125)** and **refcv4b really is a
117-anchor arm (chance = 1/117 = 0.008547)**. The 7-tensor difference (551 − 544) is exactly
accounted for: `anchor_controls` + the three ego-injection weight/bias pairs.

Note a subtlety worth recording: refcv3 has **no `--anchors` CLI flag and no `anchors` config
key** — but it *does* carry a built-in 128-entry `decoder.anchors` buffer. "No anchors flag" is
not "no anchors"; the vocabulary is compiled in rather than loaded from a bank.

### Round-trip check on the pre-existing refcv3 artifact

`Sayood/tanitad-refc-v3/ckpt_40284.pt` was **downloaded back** and re-hashed:
sha256 `5a163eaf…4203` — equal to what HF reports. It loads to `ckpt['step'] = 40284`,
**544 tensors**, 107,082,365 elements, **40/40** sampled non-zero. So the published refcv3@40284 is
genuine, is step 40284, and is refcv3 architecture — not a mislabelled copy of v4b.

---

## Caveats the cards carry (audited, not assumed)

`tanitad-refc-v4b/README.md` (13,431 B) was audited term-by-term and already carries:

- **Corpus** `physicalai-b1-w120-256x640cyl` (**B1, NOT the parity corpus**) — §2 titled
  "⛔ Corpus — NOT the parity corpus". Inadmissible for any parity claim.
- **Tier** — §3 "⛔ Tier — T1 is self-action OPEN loop, not closed loop".
- **Landing comparison** −0.1444 m [−0.1647, −0.1227], 4,823 windows / 141 episodes, paired
  episode-cluster bootstrap — §1.1, whose own heading is
  "⛔ It beats the previous arm, and it TIES the do-nothing baselines"
  (`os − ha` −0.0021, `os − ha0_ext` +0.0101, neither separated).
- **`anchor_acc` / `oracle_sel` / `sel_agrees_oracle` CONTAMINATED** — §1.2; corrected full-grid
  `anchor_acc` **0.5275**, not 0.0993.
- **Chance is 1/117 = 0.008547**, and §1.3 explicitly flags the **stale `1/128` string** in the
  shipped JSON rather than repeating it as fact.
- **`os` ADE unsettled in the third decimal** (0.2975 vs 0.2965) — §1.5.
- **Anchor units are OPERATOR-ASSERTED** (`alat`; the bank declares none) — §5 sub-heading.
- Also §1.4: a separated CI from a **one-seed** arm is necessary, not sufficient.

`tanitad-refc-v3/README.md` (43,783 B, public, pre-existing) carries its own tier lock
("⛔ TIER — OPEN LOOP ONLY. NO CLOSED-LOOP CLAIM IS MADE."), the parity discussion, a
"Honest caveats" section, and a STRATEGIC family marked "⛔ UNAVAILABLE, n = 0". It predates the
v4b landing comparison, so the −0.1444 delta and the `anchor_acc` contamination correctly live on
the v4b card only.

---

## ⚠️ Flag for the PI — visibility

The brief required **private** repos. Everything **this run uploaded** went to
`tanitad-refc-v4b`, which is **private**.

But `Sayood/tanitad-refc-v3` **was already public** before this run (created 2026-09-03) and holds
the refcv3 weights. This run did **not** create it, did not upload to it, and did **not** change its
visibility — flipping a repo that has been public for days would break any link already shared and
is the PI's call, not mine. **Say the word and it flips to private in one call.**
