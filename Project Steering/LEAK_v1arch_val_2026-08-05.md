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
