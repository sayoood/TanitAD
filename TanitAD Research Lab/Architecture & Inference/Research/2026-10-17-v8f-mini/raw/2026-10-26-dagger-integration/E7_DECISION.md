# E7 (Master Mind decision, 2026-10-26): `launch_gate.py` gets NO bank-manifest entry

**Question (integration NOTES, E7).** Should `stack/scripts/launch_gate.py` gain an explicit entry for the DAgger bank's sha256?

**Decision: no.** For `v8f_mini_train.py` the launch gate is the trainer's own in-process gate (`--selfcheck-only`; PREREG_V8F_ORECOVER
Amendment 2, reused by PREREG_V8F_DAGGER §6). `launch_gate.py` is the refcv / v8f-profile gate and does not run on this trainer. The bank's
bytes are already bound end to end, by three links, each enforced in code (file:line at tree `di/tree`, base tip 2a78f059):

1. **The token binds the argv.** The run's PASS token binds its argv sha256 (`v8f_mini_train.py` `argv_sha`, :1429; the ORC reader's GATE-TOKEN
   control checks the binding).
2. **The argv declares the bank.** At p > 0 `--dagger-bank-sha256` is REQUIRED, and it must equal the sha256 of the bank's `bank.json`
   (`bad_dagger`, `v8f_mini_train.py`:430-453). At p = 0 a declared hash is refused (a lever declared that does nothing).
3. **The manifest pins every shard.** `DaggerBank.load(..., verify=True, expect_sha256=...)` (`stack/tanitad/data/v8f_dagger.py`:337)
   refuses a manifest whose sha256 differs from the declared one, and re-hashes every shard's FILE sha256 and CONTENT sha256 against the
   manifest.

⇒ Changing one byte of any shard breaks link 3; swapping the manifest breaks link 2; editing the declared hash changes the argv sha256 and
breaks link 1. A separate `launch_gate.py` entry would be a fourth copy of the same fact with no new failure it could catch.

**What stays open (UNVERIFIED):** links 2-3 are proven by the unit and mutation suites on synthetic banks only. The first real bank's gate
run on box 2 is the first real-data test.
