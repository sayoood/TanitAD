
<!-- FULL-SUITE-HAS-BEEN-RED-AT-THE-TIP-2026-09-21 -->

### ⛔⛔ 2026-09-21 — the FULL suite has been RED at the tip: **5 failures, 8,606 passed**, none of them caused by this session — and the task harness reported the run as EXIT CODE 0

MEASURED by me, CPU only, full run, 1,069 s
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-21-full-suite-triage/`).

I found one red test by ACCIDENT while landing `6a472d1` and fixed it in `38a8ddd`. ⚠️ **That was
luck, not coverage.** Every suite run this session — and by the shape of this, for some time — was
a **FILTERED SLICE**, and a green slice is a claim about the slice. The full run says:

| | |
|---|---|
| failed | **5** |
| passed | 8,606 · skipped 55 · xfailed 2 |

⛔⛔ **AND THE HARNESS REPORTED THE RUN AS "EXIT CODE 0".** The command was
`pytest … > file 2>&1; echo …`, so the status that came back was the **`echo`'s**, not pytest's.
⇒ *never read a status through a chain or a pipe — assert on the **artifact***, which here is the
summary line in the output file. This is the `${PIPESTATUS[0]}` rule with a semicolon instead of a
pipe, and it would have let a 5-failure run be reported as clean.

**The five, each with its defect named. None is mine:**

| test | defect |
|---|---|
| `test_runbook_commands` | the **RUNBOOK tells an operator to run a v6F launch line the trainer's own preflight REFUSES** — `--nav-cond` missing (PI directive 2026-08-30, mandatory for every arm trained since) and `--horizons 1 2 4` declaring heads `[2, 4]` that **no loss consumes** ⚠️ and v6F is **RETIRED** by the PI, so the runbook carries a retired arm *and* a pre-directive config |
| `test_speed_max_derivation_stamp` | **the stamp guard runs AFTER `config.json` is written** (offset 425435 > 190431) — a guard that cannot prevent the write it exists to guard |
| `test_text_encoding_is_explicit` | `p_runner.py:260`, `:296` — text-mode subprocess without `encoding=` ⇒ **FIXED HERE** |
| `test_v6_effective_weights` | three refcv7 scorer sub-weights **ungated**: `r7_w_comf`, `r7_w_ep`, `r7_w_ttc` ⭐ **same family** as the `--w-r7-wta` provenance gap closed in `38a8ddd` — the refcv7 landing left several guards uncovered |
| `test_refcv3_arm` | the four-families guard |

**Fixed in this commit — `p_runner.py`, both sites.** ⭐ It is not cosmetic, and I have first-hand
evidence from today: `count_a7_procs` decodes **PowerShell** output, and its `except` branch returns
**1 = BUSY by design** (*"assume BUSY, never assume free"*). A cp1252 `UnicodeDecodeError` would
therefore reach that fail-safe **for a decoding reason** and hold A7 paused on a stray character.
The fail-safe is correct; arriving there by decode error is not. ⚠️ The identical defect bit my own
mutation prover twice today (`6a472d1`, `305debd`) — and the repo already had a test for it, red,
naming two other sites.

**Verified by exercising the patched paths, not only the linter:** `test_text_encoding_is_explicit`
**3 passed** (was red); `boxstat()` returns `1605 10`; `count_a7_procs()` returns **0** — readable,
and independently consistent with A7 being paused.

⇒ The remaining **four** are enumerated with their defects and left for their own landings rather
than folded in here. ⛔ **And the standing lesson: a filtered suite run is not a suite run.**
