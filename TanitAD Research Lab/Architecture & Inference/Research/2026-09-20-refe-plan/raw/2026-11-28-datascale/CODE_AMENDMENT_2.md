# DATA-SCALE CODE AMENDMENT 2: the test-side token mint binds the shipped commit (MM, 2026-11-28)

**What changed:** `refe/test_safehead_ds.py` only, at two sites (`mint_env` and `test_e2e`).
- Before, the test minted its synthetic PASS token with `commit="ab" * 20` in both `L.run_checks` and `L.finalize`.
- Now both resolve the commit through the gate's own `L._commit_of(HERE.parent, "ab" * 20)`. That returns the `SHIP_MANIFEST.json` commit in a shipped tree, else the same fake commit as before.
- No product code changed. `safehead_lc.py`, `safehead_train.py`, `safehead_gate.py` and `safehead_cvm.py` are byte-identical to CODE AMENDMENT 1 (`98e934e5`).

**Why (MEASURED, 2026-11-28):** the first real gate run on the amended build refused. That was the run of profile `refe_ds_ep_s0` for the M6 n = 35 shakedown, `datascale/run1/m6_35/gate/FAIL_98e934e5d33b.json`.
- G-SUITE ran `test_safehead_ds.py --fast`, and CODE AMENDMENT 1's new `test_learning_curve` failed in `mint_env` with *"binding differs"* on every check.
- The cause is that `launch_gate.finalize` binds the **SHIP_MANIFEST's commit** whenever the package carries one. The test's evidence had been bound to the fake commit.
- Every real gate run ships a manifest (`launch_gate.py ship-manifest`), so the test could never pass inside a gate. It passed for the build agent only because that agent's export had no manifest.
- **Reproduced directly, outside the gate,** in the same tip:
  - with `SHIP_MANIFEST.json`: `ZZSHDS_FAIL`, the same "binding differs";
  - with the manifest moved aside: `ZZSHDS_PASS 17`.
  So it is the tree, not an environment leak from the outer gate.
- `test_e2e` had the same latent defect. It is never seen in a gate because it runs only in full mode.

**Proof obligations:**
1. The unpatched file FAILS in a shipped tree. That is the measurement above, and it serves as the deliberate-regression arm of this change.
2. The patched file passes full mode unshipped (`ZZSHDS_PASS`, log `datascale/amend2/full_patched.log`).
3. The patched file passes inside the real gate's G-SUITE on the landed commit. That is the re-run of `datascale/run1/scripts/m6_35.sh`, which must mint a PASS token.

**Scope:** this is made before any real-data fit (no 185-log index exists, and the n = 35 shakedown has not trained), and the card (`PREREG_DATASCALE.md`, sha256 `4e52b00e...`) is unchanged.
