# EOL note for the TF-N7 freeze land (MM, 2026-11-13)

The lander stores text files with LF line endings. Two files of this hold were CRLF when the run tree was built and pinned. Their git copies are therefore the LF form of the bytes the chain executes. The content is identical after stripping CR, checked byte for byte at land time.

| file | sha256 of the run-tree copy | sha256 of the git copy |
|---|---|---|
| `n7_nt_build.py` | `626526e93d1154d17f503472524e02c2cfd848bb0a29f4942119c92bf25e57b1` (CRLF, executed, in `N7_PINS.json`) | `0249828472a9684fe90a3542cf1f0b85b7d159ec995aa73fbf89c12796d85791` (LF, as stored in git) |
| `fixture_navtrain_tiers.json` | `4eb91e2b95ebe5cdaf32fc7302c508d670ec618068ba47080d3b5da0a9dc2e0c` (CRLF, executed, in `N7_PINS.json`) | `4d32a7e5a3ff106bd176ce9d35fac1430ab7d2e696f45f0f161391f6a6401b12` (LF, as stored in git) |

To check a pin against git, convert the git copy to CRLF, or compare the CR-stripped bytes. Every other pinned N7 file is byte-identical between the run tree and git.
