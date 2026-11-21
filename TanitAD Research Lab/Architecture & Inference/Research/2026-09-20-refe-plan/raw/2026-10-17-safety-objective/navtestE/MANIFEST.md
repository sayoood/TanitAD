# MANIFEST: navtestE/ (IMPL_PLAN step 9 of PREREG_SAFEOBJ)

Staged, not committed, not pushed; no pod contact. Written 2026-10-17 after the last edit; sha256 are of the staged bytes (CRLF files as they sit on disk).
`navtest_E.py` text digest as printed in the test-log header (CRLF normalised to LF by `read_text`): `b034c6e9df0ffe128862b9c741c2a9ddcf2daebc040927022bb18730844fe198`.
Reader the suite was run against (READ-ONLY, not part of this package): `outcome/safeobj_outcome.py` sha256 `e534e691d69f202b172a6057aa30a396e6b116d102b92a4502aa00f07e333faf`.
Re-verify: `sha256sum -c`-style by hand, or `python test_navtest_E.py --anchors` (every mutation anchor occurs exactly once) then the full form.

| file | bytes | sha256 | what |
|---|---|---|---|
| `BUILD_E.md` | 14,227 | `1b6561cf7cb447240901820b3e70bfc44dc91d33278c84f8009fab03ffca514a` | this pass: what it reads, the evidence, fixtures, changes, open points |
| `PLAN_E.md` | 15,548 | `0f0e97bcc60d49cbf08fc87c4da4a2139579a5303e1adf0e9ed41c25e813ad9d` | the exact command sequence, gates, cost, failure handling |
| `example/FIXTURE_P2REP_navtest_E.json` | 2,161 | `a9e56ea888f7625f20c625fcff73c2c2a3ed455c2e2b4a328299a932326aabf0` | FIXTURE: real pinned CLI build on the P2REP stand-in tables, NOT SAFEOBJ numbers; never hand to the reader for a verdict |
| `example/FIXTURE_P2REP_navtest_E.provenance.json` | 4,449 | `1c2b32cfb9e6bc47878c7ac202dde416b26cc8838147782d960b96c79fd13bbc` | FIXTURE: real pinned CLI build on the P2REP stand-in tables, NOT SAFEOBJ numbers; never hand to the reader for a verdict |
| `example/FIXTURE_P2REP_navtest_families.json` | 25,845 | `ad44087f0f0676ac98a2cf26bfe2f37fb6fe1e6fcf5705af2ed07b18be871c1f` | FIXTURE: real pinned CLI build on the P2REP stand-in tables, NOT SAFEOBJ numbers; never hand to the reader for a verdict |
| `fp/fp_chain.py` | 3,010 | `9cff33ff195ed8c5c6552efa83a7f85d3a171c016e4c1c3685a622313fc5e199` | byte-identical COPY of the free-point module (md5 in fp_COPIES.md5) |
| `fp/fp_checks.py` | 9,674 | `a9296aa375c460a1746e64a31b552662604ec31bdccd74cdec3063c912e44d42` | byte-identical COPY of the free-point module (md5 in fp_COPIES.md5) |
| `fp/fp_forward.py` | 15,894 | `0f65dcd911ff9869d0fe9d9825037f7a0fbd30f81bdb4521cd86c119462453f4` | byte-identical COPY of the free-point module (md5 in fp_COPIES.md5) |
| `fp/fp_picks.py` | 11,508 | `8d427f8f87ff0b9f608b13f0821b7c4c6ad62a92ccaf7829e6ab992ebece7022` | byte-identical COPY of the free-point module (md5 in fp_COPIES.md5) |
| `fp/fp_read.py` | 12,237 | `381ddece8bf791cca9414a72800ab682eda0fa3eba81683edb53923b4ec72330` | byte-identical COPY of the free-point module (md5 in fp_COPIES.md5) |
| `fp/fp_score.py` | 5,421 | `ff08d6e192dde04d9f8561eb0f9c556e6cbd96135a8dd43bedf3aa9ffafd01ea` | byte-identical COPY of the free-point module (md5 in fp_COPIES.md5) |
| `fp/fp_ship.py` | 9,926 | `f246b3581cf8e99cbe5aa18c6aa835b6dfd04a729c44c610671d06d26146daec` | byte-identical COPY of the free-point module (md5 in fp_COPIES.md5) |
| `fp_COPIES.md5` | 323 | `c200cedfac76dab161338bc558eb99d5e85cec57e0f66888b1fd1dc29d42cc35` | md5 of the byte-identical free-point COPIES in fp/ (pinned by a test) |
| `logs/ZZTEST_navtest_E.log` | 12,388 | `277af7023277dc90717aebd9ddf9cde2b88e07289e866a2849af3903e4f9fa0f` | FULL-form run, current contract: ZZTEST_PASS 49, 74 checks, 48 mutations, rc 0, 484 s (MEASURED) |
| `logs/ZZTEST_navtest_E.old_contract.log` | 5,758 | `1dc225659705329763dccf431fe1028c14fca4b6a0b5cc46b5b93247de5b1eba` | SUPERSEDED: old contract (68 checks, 42 mutations), audit trail |
| `logs/dev_atrisk4.log` | 1,686 | `10753bad7a2a98ac48f8b608693555d5267454d804d73b41ec3598490215c6a7` | the 3 at-risk mutations + the new plan mutation, one by one (MEASURED); CLI guard in its two-layer form (stays GREEN by design) |
| `logs/dev_atrisk_cli3.log` | 814 | `7494dda7a5dc42510fe8afecbad2bbdc0363d1abf8953093d7f28a4a4b1837d3` | CLI guard mutation with all three layers removed: RED at the stated check (MEASURED) |
| `logs/dev_mut.old_contract.log` | 10,601 | `ddbec8b24ac9e483700cff731b1bd9606a3e5c8b4d143bcf57b89af119f83ae3` | SUPERSEDED: old-contract mutation run with the 3 at-risk '!!!' lines |
| `logs/dev_unmutated.before_this_agent.log` | 4,399 | `b0be35ffb8d793e5787bf90c943d3be0f5c9222f669fc59b0fbeb9f3ca47d296` | previous agent's unmutated run (73 checks), audit trail |
| `logs/dev_unmutated.log` | 4,457 | `ece878452c1b86b6ba20a07acbbdfee8c2f54383fba70ef380204b64a5528a5f` | unmutated suite on the current bytes: 74/74 green (MEASURED) |
| `logs/navtest_E.py.before_this_agent` | 45,387 | `986d1e6c37019f97bbd3a78471aa14787a49397dd9ee69b5b4d97b8244726158` | tool as left by the previous agent (CRLF), before this pass |
| `logs/test_navtest_E.py.before_this_agent` | 83,270 | `4e94ed139a45a7ce34b587dedeef612e485046884008fdf2e389fe74fded1b54` | test as left by the previous agent (CRLF), before this pass |
| `logs/write_example.log` | 899 | `0c3dd1caa6b1942bf6dadea14091d671f5fd9aed6b4f585a7bb73a9b65adf1d5` | the --write-example run that produced example/ (MEASURED) |
| `navtest_E.py` | 45,567 | `a749661374ca6c3429b53b57a9affdfe3edde89ff21c679b9ffd9109af9837cc` | the tool (plan / expected-states / seams / families / build / irep / selftest) |
| `sobj_ship.py` | 1,853 | `9704753a538b2381605f7c12c640961a0a490bf61e87cf576d32138527d2b7f9` | free-point shipper wrapper for /workspace/ops/sobjE |
| `test_navtest_E.py` | 86,268 | `8a1eaa3149807c5704bde1a07f1ffb882b637d17d7a96ea60b481f1b36b51454` | its self-test: 74 checks, 48 RED mutations, `--write-example`, `--anchors`, `--unmutated` |
