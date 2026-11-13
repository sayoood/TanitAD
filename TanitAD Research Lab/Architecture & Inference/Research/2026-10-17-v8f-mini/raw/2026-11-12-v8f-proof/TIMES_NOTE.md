# Clock note for the v8 proof package (written 2026-11-13 00:08:39)

The freeze times written inside the cards, the specs and the first state entries were estimates and run about one to three hours ahead of the dev-box clock. The file modification times below are the record. The order card-before-data and spec-before-proof holds in them. No card or spec text was changed after its sha256 was taken.

| file | modified (dev-box clock) | sha256 (first 12) |
|---|---|---|
| `fv/FV_SPEC_V1.md` | 2026-11-12 23:42:56 | 4e746f09ab36 |
| `fv/fv_check.py` | 2026-11-12 23:43:52 | 7eec446f53a1 |
| `fv/FV_RESULT_V1.json` | 2026-11-12 23:44:12 | cfbd5addf387 |
| `fv/FV_SPEC_V2.md` | 2026-11-12 23:50:04 | c4fd0f7bd5bf |
| `fv/fv_check_v2.py` | 2026-11-12 23:50:04 | 79f64e8103ad |
| `fv/FV_RESULT_V2.json` | 2026-11-12 23:50:13 | d0f4ecc76780 |
| `PREREG_V8F_PROOF.md` | 2026-11-12 23:48:24 | 38d33abdd51b |
| `PREREG_V8F_PROOF_ADD1.md` | 2026-11-12 23:54:44 | 898f9da0a37d |
| `wm/RESULT_WMCEIL.json` | 2026-11-12 23:55:01 | 8b57a3ccae84 |
| `PREREG_V8F_PROOF_ADD2.md` | 2026-11-12 23:57:16 | 4a290133a305 |
| `wm/RESULT_WMSTACK.json` | 2026-11-13 00:00:15 | 1482944263e0 |
| `PREREG_V8F_PROOF_ADD3.md` | 2026-11-13 00:02:49 | c92fa618f50b |
| `wsp/TASKVEC.json` | 2026-11-12 23:59:46 | e1bc273f59d0 |

The proof results were written on the pod and copied back, so their times are copy times (later than the run).

Note: `fv/FV_SPEC_V2.md` and `fv/fv_check_v2.py` were written by one command, the spec first; their times are equal to the second.
PREREG_V8F_PROOF_ADD5.md  frozen (file time) 2026-11-13 01:03:38
PREREG_V8F_PROOF_ADD6.md  frozen (file time) 2026-11-13 01:07:29
PREREG_V8F_PROOF_ADD6B.md  frozen (file time) 2026-11-13 01:26:47
PREREG_V8F_PROOF_ADD7.md  frozen (file time) 2026-11-13 01:38:31
PREREG_V8F_PROOF_ADD8.md  frozen (file time) 2026-11-13 01:38:44
PREREG_V8F_PROOF_ADD9.md  frozen 2026-11-13 02:20:51  (file time; no FULLRES encode or probe output existed: full/ was empty)
PREREG_V8F_PROOF_ADD10.md  frozen 2026-11-13 02:29:04  (file time; no LATVIS output existed)
* PREREG_V8F_PROOF_ADD11.md (HIERREP, replicate arms of the hierarchy / flat pair on the pod): file time 2026-11-13 07:03:09, sha256 94647519b965e5d0ac5b6decc0475719ed16a57757decd6a1994ebb25ee29d1c. No pod training output existed at that time (pod cache not yet built).
* PREREG_V8F_PROOF_ADD11_N1.md (note N1 to HIERREP: gate waiver replaced by control r6, eval cache of the arms, label files named only): file time 2026-11-13 07:25:29, sha256 7bdef4ef6e41d2461ce772c529b6f366f83e1be40db2cab56a61c3895cb47c9c. No token cache and no arm existed on the pod at that time.
* PREREG_V8F_PROOF_ADD12.md (SPATIAL): sha256 1e696630edd44306608feb4e5add8fb46d385c9397bd9be9ece0de82774b8ea6, file time 2026-11-13 12:55:18 local, frozen before any stage-0 or stage-1 data of the card was read.
* PREREG_V8F_PROOF_ADD13.md (DATAEFF): sha256 6207fa50b07d0db6cb06a4b3ad63bc64821e4e9f112f4abb66e3a7167b18c199, file time 2026-11-13 12:58:59 local, frozen before any arm of the card existed and before the label statistics of gate g1 were computed.
* PREREG_V8F_PROOF_ADD12_13_N1.md (reference note to ADD12 and ADD13): sha256 d5ddf0744a66d3504161012800057c67d88c8cf0702c6afebf801f12f5e20fa5, file time 2026-11-13 13:00:13 local, before any data of either card.
* fv/FV_SPEC_V5_POOL.md (group P, pooling): sha256 3f12d5b992bb8a9a2cbc8e38850d8ab7f27feb8c9ac3c7f98bcdab83d90ac632, file time 2026-11-13 13:03:07 local (its header says 13:15; the file time decides), written before fv/fv_check_v5.py (13:03:22) and before control s2 of SPATIAL stage 0 and control c3 of stage 1 were measured.
* PREREG_V8F_PROOF_ADD13_N2.md (note N2 to DATAEFF: chosen set passed as a restricted cache manifest with the full join; gate g1 PASS 1.4475; list facts): sha256 19dd882a473d3f1c153157983545e1404e174c77bfeac894cceaca0247b2c873, file time 2026-11-13 13:15:13 local, before any arm of the card.
* PREREG_V8F_PROOF_ADD12_N3.md (note N3 to SPATIAL stage 0: control s2 in two forms, rows, actions, r-c not run): sha256 f1cb2863c023fa608445b5cf9c0d4fe3daf607a19423d1843839cda978007026, file time 2026-11-13 13:19:51 local; spatial/s0/s0_read.py sha256 86d3159741c9787f and S0_DEFS.md sha256 f2a1769d0069281d at that time. The full stage-0 job was launched after this line was written; no RESULT_S0.json existed.
* PREREG_V8F_PROOF_ADD12_N4.md (note N4 to SPATIAL: first stage-0 read missed control s3 and is not quoted; stage 0b = r-b probe fitted on bank frames, second ego floor, control s5; written after all first-read numbers were seen): sha256 edc85a44d9953858582e5c123a855116158c4e181765654e4ede1badfb16aecc, file time 2026-11-13 13:28:48 local; s0_read.py sha256 74e6936212e374ba at launch of 0b (first read: 86d3159741c9787f). First-read outputs kept as RESULT_S0_first.json / VERDICT_S0_first.md / s0_job_first.log. The 0b job was launched after this line; no 0b number existed.
