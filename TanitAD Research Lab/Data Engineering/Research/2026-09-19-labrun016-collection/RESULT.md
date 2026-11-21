# LAB-RUN-016 collected: 79 files verified against the tip, one register redaction staged — and the clip-id rule is broken tip-wide

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind
**Tip** `362ce88` (read from the push mirror by plumbing; D:'s own branch ref is still `37645fc`).
**Content verified** from D:'s **index blobs**, the exact bytes a lander takes.

## 1. What the Lab run left in D: and not on the tip

Every one of these was already **staged** by the Lab. None was only in the worktree, and no
Lab-created file is hidden by `.gitignore` (checked: none modified today).

| group | paths | new | superset of tip |
|---|---|---|---|
| run report `2026-09-19-LAB-RUN-016.md` | 1 | 1 | — |
| package `Data Engineering/Research/2026-09-19-nuplan-posted-limit-coverage/` | 3 | 3 | — |
| its script + test `stack/scripts/nuplan_speed_limit_coverage.py`, `stack/tests/test_nuplan_speed_limit_coverage.py` | 2 | 2 | — |
| `Frontier Scan/Daily/2026-09-19/` (RESULT + search log) | 2 | 2 | — |
| Frontier Scan ledgers | 11 | 1 (`LEDGER_B8_tokenizers.md`) | 10 ✅ |
| `Frontier Scan/TRACKS.md` | 1 | — | ✅ |
| knowledge bases (A&I, Data Eng, Opponent) + `OPPONENT_CLAIMS_REGISTER.md` | 4 | — | 4 ✅ |
| banked papers `Library/papers/*.pdf` | 55 | 55 | — |
| **total** | **79** | **64** | **15 / 15** |

⭐ **The 55 papers are what makes the tip's Library whole.** The tip's `library.json` (landed in
`d36e2ab`) references **exactly 55 files absent from the tip tree — and they are exactly these
55.** Every one: **sha256 = the library entry, byte count = the entry, a complete PDF**
(`%PDF-` header, `%%EOF` in the last 2 KB). Landing the JSON without them would make a fresh
checkout's `--verify` report 55 MISSING.

The new test passes: **4 passed** (`raw/nuplan_test_pytest.txt`). The two stack files belong to
the Lab's FS19-2 posted-limit item, not to A7 or the S1 harness.

**Not collected, and why:** 13 untracked files under `FlyWheels/TanitAD_EvalFlyWheel/` (the Eval
FlyWheel's). D:'s index also holds three **stale** steering copies — `PI_DECISION_QUEUE.md`,
`PREREG_REFCV6_DEVBOX_PREPARATION.md`, `RETRACTION_LOG.md` — older than the tip. They are not
the Lab's, and ⛔ **committing any of them from D: would revert tip content.** D:'s branch ref
and index want re-syncing to the tip; that is the Master Mind's call.

## 2. Scan

Every UUID-shaped string is **classified against the full PhysicalAI clip index — 306,152 ids —
plus 2,587 parity / val40 / v7.2-eval digests**, not flagged by shape. Clip ids found in the 79
files' content: **0**. Credentials: **0**. One 8-hex clip-id **prefix** in
`Data Engineering/Research/KNOWLEDGE_BASE.md` — on a line **already on the tip** (line 110, a
deliberately truncated `xxxxxxxx-…` reference), not in the two lines the Lab added. It belongs
to §4.

## 3. The two UUID-shaped strings in `GOALS_AND_CLAIMS.md`

| sha12 | what it is | evidence | action |
|---|---|---|---|
| `feec332cd22a` | **a clip id** | in the 306,152-id index; introduced `e69039c` (2026-09-02) | ⭐ **redacted** — 2 occurrences → `sha12:feec332cd22a`, tip-based, staged |
| `7d4b0d340f98` | **not a clip id** — a claude.ai artifact id | in no id list or digest set; both occurrences are the published Training Watch page (`claude.ai/code/artifact/…`) | kept: redacting it only breaks the link |

The redaction changes **exactly 2 lines**; the artifact id is untouched. The write was
compare-and-swap against a copy verified identical to the tip.

⭐ **The register also carried 36 TRUNCATED clip ids — 8-hex prefixes used as short names in
prose** (*"clip ___ is a road bend"*, *"___'s real −78° corner"*), 32 distinct clips, each the
prefix of **exactly one** of the 306,152 ids. The lander treats a prefix as an id (it redacted
them in the 18-file set), so these are redacted by the same rule: `sha12:<digest>`, a
trailing `-…` absorbed. Guards: skip any token that resolves as a git object (a commit hash),
any token inside a file path, any prefix shared by two clips — **0 skipped**. Proof: inverting
the exact spans written reproduces the pass-1 file **byte for byte**, and that file is itself
the tip plus pass 1. `code/redact_prefixes.py`. **The register now carries 0 raw clip ids and
0 clip-id prefixes.**
⚠️ **My first proof failed and blocked the write — correctly, for the wrong reason.** It undid
substitutions by string replacement, which is ambiguous when a prefix maps to the same sha12
as a pass-1 redaction (1 such token). The positional proof has no such ambiguity.

## 4. ⛔ The clip-id rule is not satisfied anywhere near the register

The same clip id appears in **26 files** on the tip. Measured tip-wide:

| | |
|---|---|
| files carrying ≥ 1 raw clip id | **686** |
| **distinct clip ids exposed** | **16,868** |
| by type | json 520 · py 32 · **md 30** · log 28 · sh 27 · jsonl 23 · txt 11 · csv 7 |
| largest carriers | full clip lists and per-clip label files, **~4,700–4,950 ids each** |

MEASURED · `raw/clip_exposure_summary.json` (paths with full ids masked as `<id>` and
8-hex prefixes as `<id8>` — 20 tip file NAMES embed a clip prefix). Most carriers are
**banked raw artifacts and functional id lists** — redacting them rewrites the record of what
ran and breaks inputs, and the ids would remain in git **history** regardless. ⇒ **This is a
PI decision** (history rewrite or not; private-repo exposure acceptable or not; a sha12-keyed
convention for future id lists). Nothing beyond the register was touched. Related, already in
the queue as item 26: the public HF parity-cache repo exposes 3,000 ids in file names.

## Reproduce

`code/landscape.py` (D: vs tip) → `code/collect_verify.py` (supersets, PDFs, scan) →
`code/redact_goals.py` (the register). The 306k-id universe is built from
`C:/Users/Admin/tanitad-data/physicalai/clip_index.parquet` plus
`stack/tanitad/data/*_clip_digests.json`, and is **deliberately not banked** — it is itself a
list of raw ids.


---

## ⛔ MASTER MIND, AT LANDING (2026-09-19): the GOALS_AND_CLAIMS redaction above was NOT landed

It contradicts the PI's ruling on queue item 17 (2026-09-17, *"Its hygiene and reproducibility"* ⇒ option 2): **the banked record is left alone and a guard (`tools/clipid_scan.py`, counts only) stops the count growing** — git history keeps every identifier regardless, so redacting existing lines buys appearance only. The redaction was requested by the Master Mind's own brief, which did not check that ruling first: the error is the brief's, not this package's. The two redaction scripts are banked as record; the staged register rewrite stays unlanded. The systemic finding (686 tip files, 16,868 distinct ids) is therefore NOT a new PI decision — it is the legacy exposure that ruling already covers; the operative check is that the guard's count does not grow.
