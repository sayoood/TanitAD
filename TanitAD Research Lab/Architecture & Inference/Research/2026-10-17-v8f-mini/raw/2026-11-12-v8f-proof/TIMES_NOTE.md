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
