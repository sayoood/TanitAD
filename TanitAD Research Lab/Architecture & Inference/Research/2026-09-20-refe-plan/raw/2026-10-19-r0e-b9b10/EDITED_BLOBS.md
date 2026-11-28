# R0E-B9B10 -- edited blobs (git blob sha1, LF bytes; all LF, 0 CR)

BASE = the landed branch tip `1588fac2e654e8607fdcd0dc277c25db6effb4ff` (`agent/arch-inf-20260803`, R0E-B2B5 landed). `tip blob` is read from the mirror's object store (`git rev-parse 1588fac:<path>`) and must equal the sha1 of the exported base file (asserted).
Every edited file has base blob = tip blob (no 3-way merge needed: my file replaces the tip's wholesale); a new file has no base.

| file | base blob = tip blob (`1588fac`) | new (mine) blob | base == tip |
|---|---|---|---|
| `refe/rls1_checks.py` | 62ff89e0680a30a40b2414297a8a9c67f66c4389 | a82dd3a11a8a23f383fba0bacf39d90129a52dee | yes |
| `refe/rls1_pod.sh` | 95399d20c1bdae97ede1d0c525840a668c3cdf83 | 4cd6f3c15d1fd82987eebba1bc1f9077b419bc5d | yes |
| `refe/test_rl_safeobj.py` | 0eae9540bf0d6cb053d6391febe4f23899c4b46e | 75b9493ed31f21d1bdb1a2856cf63a410a80cbeb | yes |
| `refe/test_rls1e_chain.py` | 7831a3e6b0225e96aeb6a340d67c67a97d7fc32f | df3f78c39c146432e41a4af3c42264aada8032e9 | yes |
| `eval/r0e_id_read.py` (new) | -- (absent at the tip) | ecad59ee68bef6e548709e679699b59337a79b3f | n/a |
| `eval/r0d_id_reference.json` (new) | -- (absent at the tip) | bd07cabe6da298f56140f34f657d46c384cb2779 | n/a |
| `refe/test_r0e_b9b10.py` (new) | -- (absent at the tip) | cea5c28886d86fc6d6003abacfc9bee7a0bb831a | n/a |
| `raw/2026-10-19-r0e-b9b10/build/make_r0d_id_reference.py` (new) | -- (absent at the tip) | ed9b5bec26f7e947db9ac941da4c6c30c00aee1b | n/a |
| `raw/2026-10-19-r0e-b9b10/RESULT.md` (new) | -- (absent at the tip) | cb4a1e2134a54cd17b552beec47a93a86b52348e | n/a |
