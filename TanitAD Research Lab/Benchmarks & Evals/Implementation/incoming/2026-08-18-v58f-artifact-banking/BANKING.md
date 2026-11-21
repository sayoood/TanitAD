# v5.8f artifact banking — 2026-08-18

**Task:** close the single-remote risk on the ~200 KB of decision-carrying v5.8f eval JSONs that
existed ONLY on HF, per the registry-citation reconciliation's escalation §7.2
(`…/incoming/2026-08-18-registry-citation-reconciliation/REGISTRY_CITATIONS.md`). The pods that
produced them (pod4/pod5) are terminated; until this banking, one HF outage or repo deletion
would have severed eleven §1.14 registry citations from their raw artifacts.
**Author:** banking subagent, clone `C:/Users/Admin/wt-tanitad-local`, branch
`agent/arch-inf-20260803` (base `42852555`). Evidence class throughout: **MEASURED** this
session; machine-readable raw evidence in `VERIFICATION.json` beside this file.

## What landed (11 files, 222,729 B), from where

Source: HF `Sayood/tanitad-flagship-v5f-w120`, prefix `release/v58f/` — the v5.8f release bundle
built 2026-08-12T01:11:46Z **on the source pod** by `stack/scripts/release_v58f.py` (its
`MANIFEST.json`, uploaded last per the manifest-last protocol, records each file's md5 + bytes
**computed pod-side from the original run outputs before upload**). An md5 match against that
manifest therefore proves byte-identity with the **pod originals at release time**, not merely
with what HF serves today.

Layout mirrors the release (`gates/`, `gates/four_families/`). One deliberate rename:

⚠️ **`ff_comparison.json` → `ff_comparison.full.json`** — the repo already tracks a 3,916 B
**demo** of the same basename
(`…/incoming/2026-08-07-hierarchical-wm-redesign/ff_rescore_val40_demo/ff_comparison.json`,
under Architecture & Inference). The §1.14 artifact is this 55,705 B file; banking it under a
disambiguated name means a basename search can never land on the demo believing it found the
full comparison. Both registry citation and this note say so.

## Verification table

Checks per file: **(a)** bytes vs HF tree listing (`/api/models/...
/tree/main/release/v58f/gates?recursive=true`, unauthenticated), **(b)** bytes vs
`release/v58f/MANIFEST.json`, **(c)** md5 vs `MANIFEST.json`, **(d)** git-blob sha1 of the
downloaded bytes vs the HF tree `oid`, **(e)** JSON parses, **(f)** md5 of the bytes re-read
from the landed file on disk. All six checks passed for all 11 files (raw per-file flags +
blob sha1s: `VERIFICATION.json`).

| banked as (under this dir) | HF path under `release/v58f/` | bytes | md5 (= manifest = disk) |
|---|---|---|---|
| `gates/i4a_none.json` | `gates/i4a_none.json` | 11,824 | `f8a1084cc591a4602c03ce9458fb8d1d` |
| `gates/i4a_zero.json` | `gates/i4a_zero.json` | 11,818 | `251fec72d7b38f4adf738ad353e64175` |
| `gates/i4a_shuffle.json` | `gates/i4a_shuffle.json` | 11,831 | `bedc48e803a433af61b211d8406e983c` |
| `gates/w7_w4r_k32_gate.json` | `gates/w7_w4r_k32_gate.json` | 8,406 | `511b53eca006d68c2fe15eba66c2588b` |
| `gates/four_families/ff_stageA_cl.json` | `gates/four_families/ff_stageA_cl.json` | 20,423 | `9029350e51149c8f25cbf4272498e88c` |
| `gates/four_families/ff_stageA_ol.json` | `gates/four_families/ff_stageA_ol.json` | 20,781 | `7d2ec5c74320deadcdfde89757f0d895` |
| `gates/four_families/ff_stageA_ha.json` | `gates/four_families/ff_stageA_ha.json` | 20,390 | `8b04c48d094fdcb914617debc856fd73` |
| `gates/four_families/ff_v5f30k_cl.json` | `gates/four_families/ff_v5f30k_cl.json` | 20,396 | `2516a2954f156b3ff95bb4c97d71f16b` |
| `gates/four_families/ff_v5f30k_ol.json` | `gates/four_families/ff_v5f30k_ol.json` | 20,765 | `35e84ad350a8f2617f3977973db7d73e` |
| `gates/four_families/ff_v5f30k_ha.json` | `gates/four_families/ff_v5f30k_ha.json` | 20,390 | `b2517474a162b091af508e6a56a580b4` |
| `gates/four_families/ff_comparison.full.json` | `gates/four_families/ff_comparison.json` | 55,705 | `afd05c5de9fa94c9bcedea6901841ec3` |

Content spot-checks (structure, not just bytes): the i4a files carry `imagination_ablate` +
`c2_scorer`; the W7 gate carries `gate_W7_selgap_closed` + `calibration_p7` + `tier`; the six
ff files carry `four_families` + `intervals` + `_estimator`; the comparison carries `arms` +
`paired` + `_binding`.

⚠️ **CRLF guard:** this dir carries a `.gitattributes` with `* -text`. The dev box runs
`core.autocrlf=true`, which would rewrite these LF files to CRLF on the next checkout and
silently break every md5 above — the same artifact family as the "7 of 10 drift rows were
CRLF" retraction. With `-text`, checkout reproduces the exact banked bytes on any box, so the
table's md5s stay re-verifiable forever. Additionally verified: the **staged index blobs**
themselves md5-match the manifest (not just the worktree copies) — see the staging check in
the deliverable manifest.

## Provenance chain, per file family

- **i4a trio** — pod5 `/workspace/experiments/i4a/`, local stems
  `flagship-v5f-w120-30k-i4a-{none,zero,shuffle}` → release names per `release_v58f.py:40–45`.
  Serves registry §1.14 *I4a IMAGINATION ABLATION* (manifest `quoted_by`: "imagination
  intact/zeroed/shuffled").
- **`w7_w4r_k32_gate.json`** — pod5 `/workspace/experiments/w7-repaired-w4r-k32/w7_gate.json`
  → release name per `release_v58f.py:38`. Serves registry §1.14 *W4r + W7-w4r*.
- **four-families seven** — pod5 `/workspace/experiments/t1-v58f/four_families/`, same
  basenames, per `release_v58f.py:58–65`. Serve registry §1.14 (the six per-tier
  `ff_{stageA,v5f30k}_{cl,ol,ha}.json` + "the four-family rescore index").

## Registry edit + zero-changed-numbers proof

Three citation sites in `Project Steering/MODEL_REGISTRY.md` (§1.14: I4a block, W4r+W7-w4r
block, four-families instrument-change block) got the in-repo path **appended**; the pod-path
provenance and HF location at each site are untouched. **PATHS ONLY** — proved by the same
multiset method as the reconciliation's §3, over the whole file, old (`git show HEAD:`) vs new:

```
decimals (\d+\.\d+):              IDENTICAL multisets (3216 tokens; only-old {} only-new {})
comma-grouped (\d{1,3}(,\d{3})+): IDENTICAL multisets (345 tokens)
scientific:                       IDENTICAL multisets (82 tokens)
ALL digit runs (\d+):             only-old {} — nothing removed; only-new = 25 runs, every one
                                  inside the added citation lines: path components
                                  2026/08/18/58 ×3 (the banking dir), 2026/08/07/40 (the demo
                                  cross-ref), i4a/w7/w4r/k32 name digits, and "md5" ×3.
                                  No unit, no measurement, no table value.
```

(Checker: scratchpad `multiset_check.py`, line-level accounting included; it exits nonzero on
any unaccounted digit. Result: `MULTISET_OK`.)

**Registry lint after the edit:** `PYTHONUTF8=1 python -m pytest tools/tests/test_registry_lint.py
tools/tests/test_registry_paths_allow.py -q` → **40 passed in 0.44s**. `tools/registry_paths.py
--only-bad`: MISSING 0, NOT_A_PATH 0, EXISTS 188→191, unresolved ratchet 1/1 (unchanged).

## Access note (no secret anywhere in this package)

The repo's **tree API is public** but **blob downloads are gated (HTTP 401)**. G:'s `Keys.txt`
is unreachable (mount down); a local copy was found at `C:/Users/Admin/Desktop/Keys.txt` after
the brief's two named locations came up absent (absence-needs-two-probes). The token was read
in place — `grep -oE 'hf_[A-Za-z0-9]+'` piped directly into an env var consumed by the fetch
script — never printed, never written to a file or argument, and does not appear in
`VERIFICATION.json`, this file, or the git history.

## Deliberately NOT banked

- `release/v58f/ckpt/ckpt_stage_a.pt` (3,248,642,418 B) — stays HF-only **by design** (task
  scope; the registry §1.13c citation already records the HF location).
- The rest of the release (`stage_a_gate.json`, `w4r_gate.json`, `p8_gate_attempt2.json`,
  `w7_full_gate.json`, `t1_*.json`, `media/*`) — out of this task's scope; note the gate JSONs
  among them that the registry leans on are already banked in-repo per the reconciliation's
  verdict table (its §2, rows 3/5) or cited with their own HF pointers.
