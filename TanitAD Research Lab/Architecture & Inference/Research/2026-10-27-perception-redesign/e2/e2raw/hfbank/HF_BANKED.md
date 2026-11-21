# HF bank of the refcv9 E2 dev-box pulls (MM 4a3d05, 2026-11-20/21)

**Why.** F: on the dev box ran out of room for the launch gate's dev-box pinned run (it needs >= 20 GB free). Seven E2 stage B pulls on
`F:/refcv9_e3/` had **no other copy** (their Thor run dirs were gone). Each was banked to the PI's **private** HF dataset
`Sayood/tanitad-refcv9-e2-raw-runs` before any deletion (PI, 2026-11-20: *"Yes, delete them"*, *"Yes, include e2wise"*).

**How** (tools in this folder):

| tool | what it does |
|---|---|
| `hf_bank_dir.py <src> <name> <out>` | tar stream of `<src>` in 1,500 MB parts; sha256 per member (one read); each part uploaded (`HF_HUB_DISABLE_XET=1`) and checked against the **remote LFS sha256** before its local temp part is removed; manifest (members, parts, revisions) uploaded beside the parts. Refuses a non-private repo. Never touches `<src>`. |
| `hf_bank_restore.py <name> <dest_parent> [--check-only]` | streams the parts over HTTPS in order (**no temp files**), checks every part's sha256 and byte count and every member's sha256; writes only manifest members (relative, no `..`) under `<dest_parent>/<top>/`; **refuses an existing target** and a disk without room. `--check-only` hashes everything and writes nothing. Ends `RESTORE-PASS` / `RESTORE-FAIL`. |
| `hf_bank_restore_check.py <name> <dl_dir>` | the older check: downloads the parts to `<dl_dir>` (needs the full size on disk), then streams them. Superseded by `--check-only`. |

**MEASURED checks of `hf_bank_restore.py`** (2026-11-21, dev box): self-test bank `hfbank_selftest` (33 members, 3 parts) restored
**33/33 sha256-equal to its source tree**; a second restore into the same target **refused (rc 4)**; two mutations — one expected member
sha256 wrong, one expected part sha256 wrong — each gave **RESTORE-FAIL**; `--check-only` wrote **0 files**. On the real bank
`e2tail_stageB`: `--check-only` **RESTORE-PASS 14,426/14,426 members, 0 mismatches, parts 000-002 OK** (04:39:57Z-04:44:24Z).

**The banks** (manifest sha256 = sha256 of `<name>/<name>.manifest.json` on HF; rev = the manifest's commit):

| name | members | bytes | parts | banked (ZZBANK-VERIFIED) | manifest sha256 | rev | F: copy |
|---|---|---|---|---|---|---|---|
| e2raw_stageB | 14,427 | 3,508,942,224 | 3 | 2026-11-20T19:41:31Z | 66ca2ee997ff83e97544ca095c52365b1568096848f885e574ca67dcb2d106a7 | 0c1dba286a71 | deleted 2026-11-20T21:22:51Z |
| e2prst_stageB | 14,428 | 3,489,632,855 | 3 | 2026-11-20T20:49:15Z | a5ddd4fafe276aa844a36672cf4df0640f68442cc2be7edd20879091862b7876 | d241b6715d38 | deleted 2026-11-20T22:08:23Z |
| e2wboth_stageB | 14,429 | 3,468,668,820 | 3 | 2026-11-20T21:50:44Z | 8ffc3efe44e7d5e70005cf229678db4d85beb49e431c8601247ac736939be85b | 04f55a90f60e | deleted 2026-11-20T22:34:23Z |
| e2woff_stageB | 14,429 | 3,490,741,917 | 3 | 2026-11-20T22:55:06Z | 778bdb76b703d6925a877e3dd852e798d0f0de6a880f2d8c16c4273fa261855e | 69166202791f | deleted 2026-11-21T03:00:46Z |
| e2wise_enc_stageB | 14,426 | 3,466,461,365 | 3 | 2026-11-20T23:49:01Z | a54dee2219dcfe1baab1fc05a6e9098fde8f30dad63e63cb0b4fca8da361804b | f28689e515c1 | deleted 2026-11-21T04:15:16Z |
| e2wise_plan_stageB | 14,426 | 3,489,221,978 | 3 | 2026-11-21T00:43:48Z | 76932dfa00aeb669973fb905bfeb28e7fab09119209b37f84e8406920ca7f122 | a9ab51410fb5 | deletion running 11-21 |
| e2tail_stageB | 14,426 | 3,488,888,356 | 3 | 2026-11-21T04:28:58Z | 557e7b30e3d35f589d04cac578d81265fa6fdab4b7fcac7c3b0723b2619f0eec | 88f37e9f6659 | on F: (the K2DC02s1 tail kit's K0-slot source, `F:/refcv9_e3/e2tail_stageB/s6c`; restore before that read if deleted) |

Each deletion was done by a PI-run script that re-checked the bank (ZZBANK-VERIFIED, all parts verified, member count equal to the files on
F: at that moment) right before the delete; log `E:/Projects/TanitAD-artifacts/refcv9_e2/hfbank/DELETIONS_2026-11-20.log`, manifests
copied beside it.

**Restore:** `python -I hf_bank_restore.py <name> F:/refcv9_e3` recreates `F:/refcv9_e3/<name>/` byte for byte (the members are stored
relative to the source's parent folder).
