# MANIFEST: irep/ (the I-rep, PREREG_SAFEOBJ section 7 "inference" floor)

Staged, not committed, not pushed; no pod contact, no GPU. Written 2026-10-17 after the last edit; sha256 are of the staged bytes (CRLF/LF as they sit on disk).
Read-only inputs (NOT part of this package, hashes unchanged before/after, see logs/readonly_trees_*.sha256): `navtestE/fp/fp_forward.py` md5 `82bf21514f919ee4eafab72690004e34`; `navtestE/navtest_E.py` sha256 `a749661374ca6c3429b53b57a9affdfe3edde89ff21c679b9ffd9109af9837cc`; `outcome/safeobj_outcome.py` sha256 `e534e691d69f202b172a6057aa30a396e6b116d102b92a4502aa00f07e333faf`.
Re-verify: `python make_variant.py --check` (ZZVARIANT_CHECK_OK), then `python test_irep.py` (ZZTEST_PASS 11 expected), by hand `sha256sum` against the table.

| file | bytes | sha256 | what |
|---|---|---|---|
| `BUILD_IREP.md` | 16,606 | `30aeab04da76c53d1ba53aca3a217f071604915e4360614556f4747bcab96830` | what was built, the evidence, real vs synthetic, the VRAM finding, decisions, open points |
| `PLAN_IREP.md` | 20,975 | `648b18fc5ea8afbf94202239c77203a9b627777b44b350178c01bcefcf3a7d91` | the exact command sequence, VRAM table, cost, gates (incl. I-G5), failure handling |
| `fp_forward_irep.diff` | 7,064 | `8a769be7fa049ac22bec7fc1bebcbb80c60ef5e384a30aa465ba1ff423129a8f` | unified diff original -> variant (pure addition) |
| `fp_forward_irep.py` | 21,572 | `218b115a17ce5062e21e9b271a66fa3df71bc78ffd1525394312ebc537d1ce92` | THE VARIANT: the navtest forward + --eval-batch / --token-order-seed (82 lines added, 0 removed) |
| `irep_tool.py` | 22,790 | `e4253e57d823d0bf688946da37db424eb3ca97f751fbe2ea279f6146a066a05f` | drive / expected / seams / make: the glue, the guard, the consumer hand-off |
| `make_variant.py` | 10,698 | `95b719f8039f6aea646aa58c15fdbb7c60613c8fc30635e5d1edb9aae40a0223` | builds fp_forward_irep.py from the pinned original by 5 exact-string hunks (--check re-derives) |
| `test_irep.py` | 49,042 | `adcc30b7dabf6b7058d7e00afe28a26eefd3e71311e230515106c33058a8bd0f` | 29 checks + 10 RED mutations (full form: ZZTEST_PASS 11) |
| `vram_estimate.py` | 10,053 | `ebcfe15e3ade23de0f8f94596b1dda663d6cf25ad629da14553038fda01ce628` | VRAM probe: the real REFe forward on the meta device with a live-byte tracker (+ 2 analytic controls) |
| `logs/ZZTEST_irep.log` | 18,776 | `521d9f26b6bf093a8491755be5511af533b580cf179991c445044ee2918631f0` | FULL-form run on the final bytes: ZZTEST_PASS 11 (29 checks, 10 RED mutations), rc 0, 467 s (MEASURED) |
| `logs/determinism_flags_grep.log` | 505 | `5af187083275d6892bf13d42c07d037bcc05d9593d61ebdccb82ef7f5fabb360` | MEASURED: 0 determinism/precision flags in the 5 files the forward runs, with same-breath controls |
| `logs/dev_batch_order.log` | 733 | `a36df81f98997b568a32d1dabc3f546a8b8de9449fbc406502a38bba06e0a347` | AUDIT TRAIL: development run, batch_order group (earlier bytes) |
| `logs/dev_chain2.log` | 780 | `b1762f39542bb3189c65acbf937df4dad91876b926e3a180374219bb5b947d14` | AUDIT TRAIL: development run of the chain group after the expected/serve checks were added |
| `logs/dev_chain_guard.log` | 1,268 | `f52ae6abf1fd59b568f6b268b74dfda9edee21a2f4abfd68a9382845b2c4291d` | AUDIT TRAIL: development run, chain+guard+real groups (earlier bytes) |
| `logs/dev_full_before_seed_layers_fix.log` | 17,534 | `2148b8905af6f11a21d1b1b4234b1fcb24439888c4419aff49984ec5b4b1306a` | AUDIT TRAIL: first full run; identical_seeds_accepted (one layer) stayed GREEN -> mutation changed to remove both layers |
| `logs/dev_keep.log` | 1,106 | `a78438328dc7701bfd966b9e4a88991483de00b2fe38c834c19059469f49607d` | AUDIT TRAIL: development run with --keep, used to read the fixture's irep.json and sidecar |
| `logs/dev_variant_identity.log` | 916 | `63c52b60addc1a0183fee6d50ccd6d34f0063a6ead62e9d51d198efe1df4e4d7` | AUDIT TRAIL: development run, variant+identity groups (earlier bytes of the tool/test) |
| `logs/identity_proof.json` | 763 | `d5614e8c97faa65eb5f218a9b1c405bea9d7bcf45694014aef1e2f2ebdddc7d2` | MEASURED: original and variant npz at the defaults, sha256 equal (byte_identical true) |
| `logs/nvidia_smi_devbox.log` | 75 | `f96ed109b89d055db5c2e0adedc93b64a517215e247eb445c7ba8c9c3cc6d8db` | MEASURED dev-box GPU occupancy at 15:40 (shared card) |
| `logs/readonly_trees_after.sha256` | 3,609 | `5e8f9da546a13b0f7e19e2238edb59b9eb2bc727c5e78b307280eac038f116e2` | the same, after the final test run: identical (diff empty) |
| `logs/readonly_trees_before.sha256` | 3,609 | `5e8f9da546a13b0f7e19e2238edb59b9eb2bc727c5e78b307280eac038f116e2` | sha256 of the 36 files of navtestE/ + outcome/ before this work (taken at the start) |
| `logs/vram_estimate.log` | 2,550 | `23d84727819e59213103a5ef61157d1c929b152d56f6a6e8c46221b28207dec4` | MEASURED VRAM table (meta device), controls first, biggest live tensors at the peak |
