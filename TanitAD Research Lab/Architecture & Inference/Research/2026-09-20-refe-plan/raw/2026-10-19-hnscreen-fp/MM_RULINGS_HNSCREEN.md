# MM rulings on the P6-HN screen's AMBIGUITIES 1-12 (Master Mind, 2026-10-19)

Source: `raw/2026-10-18-p6hn-prereg/impl3/RESULT.md` § AMBIGUITIES (landed `1e62e02`). The screen has NOT run and no screen number exists, so these rulings are pre-data.

| # | ruling |
|---|---|
| 1 | **ACCEPTED as an amendment (pre-data).** G14's identity is the loss-path two-call term against ONE call, `== 0` under strict determinism. The masked-call fan rows are informational. ⛔ If G14 is not exactly 0 on the GPU under bf16, the screen STOPS and reports; no tolerance is introduced after a reading. |
| 2 | **ACCEPTED; the prereg text is corrected.** PREREG 4.1's "strict deterministic algorithms are on (`scorer_finetune.py:1369`)" is wrong about the source: strict determinism is on only inside the G12 / G14 probes. Training steps are not bit-deterministic, as in every prior REFe arm. The seed replicate (`H-ESTIM-SEED-1`) covers that variance; determinism does not. |
| 3 | ACCEPTED: `G15_MIN_SCENES = 64` per side; a side with fewer compares all. |
| 4 | ACCEPTED: G15 holds at the run's own micro-batch, fixed in the pinned argv. A micro-batch change re-runs G15. |
| 5 | ACCEPTED: the `fan_fingerprint` construction (`keys_fingerprint` over the RL-train fan keys + `ckpt_step`) is the contract. |
| 6 | ⛔ **REJECTED as shipped: a LAUNCH BLOCKER.** The profile ships `"fingerprints_expected": {}`, and `refe/launch_gate.py:597` iterates `(prof.get("fingerprints_expected") or {})`. An empty dict therefore runs NO content check, and the gate would PASS a screen on any `--hn-bank` / `--ctx-cache` content. This is the "a default that disables the check" class (CLAUDE.md, *built, tested, and unreachable*, rule 4). **Fix (BACKLOG `HNSCREEN-FP-REQUIRED`):** under `hn_rules`, an empty or incomplete `fingerprints_expected` (missing `--hn-bank` or `--ctx-cache`), or an expected flag absent from the argv, is a gate FAIL. A deliberate-regression arm proves it. The values are filled at bank / cache build time from the builder's own fingerprint, never typed by hand. |
| 7 | ACCEPTED: `--b-mode listnet` with the hn flags is refused. |
| 8 | ACCEPTED: S3 / S4 stay non-mintable (`arms_need_pi`); the profile mints S0..S2 only. |
| 9 | ACCEPTED (wasted work, not a correctness issue); not pursued unless the screen's wall clock decides it. |
| 10 | ACCEPTED: the CPU stub rig's `ManualMHA` is test scaffolding; the module term is proven against the real `nn.MultiheadAttention` in `test_hn_bank.py`. |
| 11 | ACCEPTED: `refe_rls1` / aux profiles forbid the hn flags; their pinned argvs are unchanged. |
| 12 | ACCEPTED: `HN_LABEL_VERSION = 5` and `HN_PARTITION_ID` are trainer constants; a bank or cache minted under another value is refused. |

**Order before any screen launch:** `HNSCREEN-FP-REQUIRED` landed → bank + cache built → `fingerprints_expected` filled from the builder → launch gate PASS → screen (dev-box GPU, queued after L1 Step A).
