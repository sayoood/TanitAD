# 🔴 v1arch trained WITHOUT parity, and the canonical val is INSIDE its training pool

**MEASURED 2026-08-05 from pod4 and tanitad-new. Evidence class: MEASURED (manifests read directly).**
Found while preparing the 30k gate — i.e. **before** a number was published, not after.

## 1. Parity was never asserted — from the arm's OWN config

`/workspace/experiments/flagship-v1arch-v2bal-30k/config.json`:

```json
"v2_parity": {"parity": false, "corpus_key": null, "checked": false, "clips_present": 9000}
"require_parity": false
```

The arm trained on a **9000-clip** pool (`epcache-physicalai-v2bal-4b7eeeac222d`), not the canonical
**2376-episode** `physicalai-train-e438721ae894` with skip-hash `f09e44db`. `--require-parity` was
not passed, so the trainer's own guard never ran (`checked: false`).

## 2. The canonical val episodes are IN that training pool

The canonical val is `physicalai-val-0c5f7dac3b11`; its 40 scored episodes are identified by the
4-hex prefixes recovered from the packed-`eid` dumps (`0002`, `0084`, `00ba`, … `0e32`) — confirmed
correct here because those 40 prefixes select **exactly 40 of the 600** episodes in the val cache
on `tanitad-new` (`md5 2e2011d7…`).

Intersecting them with pod4's 9000-clip train manifest:

| | |
|---|---|
| train clips | **9000** |
| canonical-val prefixes present in train | **21 of 40** |
| expected by chance at 4 hex | **5.1** |
| **exact full 36-char UUID matches verified** | **≥16** |

The ≥16 were confirmed by cross-reading two independent clean listings (pod4's prefix hits and
tanitad-new's val manifest) — e.g. `0084596e-465e-42c3-a893-4e51714e3592`,
`026ef99a-2283-4c08-83e3-c0c3b4bae865`, `0b9b8bb3-9717-427b-b863-c2ef8ab08e07`,
`0e32d872-4ed1-4099-8d17-edc8eed71bb2`. A 36-character UUID matching exactly is not coincidence.

⚠️ **The 21 is a prefix count** (≈5 of which could be chance); **≥16 is the hard, exact-match
floor**. Either way ≥40 % of the canonical val was trained on.

## 3. What follows

⛔ **Do NOT run the 30k gate for `flagship-v1arch-v2bal-30k` against the canonical val.** It would
produce a strong, well-formed, invalid number — and `g_op_fwd_ade_m` **0.0289**, the best curve in
the programme, is exactly the kind of number nobody would question.

⚠️ **This is the REF-A I-JEPA failure again** (~80 % of val inside train), which `CLAUDE.md` already
carries as a root-cause class. The generalisation that applies: *for any arm, ask whether its
training pool intersects the set it will be scored on* — and check it from the MANIFESTS, not from
the run's prose.

⚠️ **Nothing above says the run is worthless.** It says the canonical val cannot score it. The
training curve is still a valid statement about optimisation, and the arm may be fine on a
**disjoint** val. What is not available is a comparison against the arms scored on
`physicalai-val-0c5f7dac3b11`.

## 4. Open questions for the PI (not decidable from here)

1. Was v2bal **intended** to be a different corpus with its own split? If so, where is its val
   manifest? Neither pod carries one (verified by filesystem-wide `find` on pod4).
2. Do the other v2bal-corpus arms (`flagship-v2corpus-30k`, on the same pod) share this defect?
   The same check applies and is cheap.
3. Does `MODEL_REGISTRY.md` quote any v1arch number against the canonical val? If so it needs the
   same treatment as the `overlapping_holdout_se` blast radius.

## 5. Method note

Both pods were **read-only** throughout: manifests and config JSON only. v1arch had already
finished (step 29999, clean supervisor exit 03:15:27Z) and its GPU is idle; v5f continues untouched.

---

## 6. UPDATE — the eval was attempted, and the HARNESS ITSELF refused the number

**PI ruling 2026-08-05:** the 9000-clip pool was **deliberate** — v1arch exists to test the effect of
*more and better-distributed data*. The corpus choice is therefore settled and not a defect. The
val-overlap below is a separate fact and survives that ruling.

MEASURED on pod4 (GPU idle, training finished): building a val split from the canonical episodes
that exist in v1arch's own geometry yields **23 of 40** (57.5 %) — and every one of them is in the
training pool, because *being in the pool is what makes them available*. The other **17 exist only
at 256×640 cylindrical**, which this 256×256 checkpoint cannot consume.

`eval_flagship_v4.py` ran to completion on those 23 (3937 windows, ckpt step 29999) and then
**refused to certify its own output**:

```
"mode": "MODE_A_canary_only_validation",
"canary_ade_2s_MEASURED": 0.6838,     vs v1's known FULL-SET 0.4271, tolerance 0.05
"HARNESS_VALIDATED": false,
"verdict": "HARNESS NOT VALIDATED — DO NOT proceed to score any v4 checkpoint
            with this harness until the discrepancy is found and fixed."
```

⭐ **That is the guard working, not a bug.** The canary reproduces v1's canonical-val number; fed a
non-canonical split it cannot, and it stops rather than emitting a plausible figure. ⇒ **No
certified number for v1arch is obtainable from any data now on either pod.**

## 7. What a clean eval corpus requires

The minimal correct set is the **17 canonical val episodes that are NOT in the 9000-clip pool**,
built as v2 caches at **256×256** (v1arch's geometry). That yields a genuinely held-out,
canonical-subset val — smaller n, but admissible and comparable episode-for-episode.

BLOCKED ON A CREDENTIAL, and it must not travel through chat:
* `Keys.txt` is git-ignored and lives on the dev box; it is **not** in the repo checkout and **not**
  on either pod (checked: `/root/.cache/huggingface/token`, `/root/.huggingface/token`,
  `/workspace/Keys.txt`, `/root/Keys.txt`, `/workspace/TanitAD/Keys.txt` — all absent;
  `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` unset on both).
* `huggingface_hub` is installed on `tanitad-new` (1.26.0) but **absent on pod4**.
* pod4 has internet (github 200, hf 200) and its GPU is now idle, so it is the right host.
