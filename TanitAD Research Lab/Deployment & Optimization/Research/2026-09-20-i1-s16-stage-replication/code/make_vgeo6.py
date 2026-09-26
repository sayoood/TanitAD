"""Derive vgeo6_s16.py from the LANDED vgeo5_screened.py by enumerated substitutions.

Every substitution is asserted to fire exactly once. The point of generating rather than
retyping is that the diff is provably the stage swap and nothing else.
"""
import hashlib, os, sys

SRC = ("D:/Projects/TanitAD/TanitAD Research Lab/Deployment & Optimization/Research/"
       "2026-09-18-i1-collinearity-screen/code/vgeo5_screened.py")
DST = ("D:/Projects/TanitAD/TanitAD Research Lab/Deployment & Optimization/Research/"
       "2026-09-20-i1-s16-stage-replication/code/vgeo6_s16.py")

s = open(SRC, encoding="utf-8").read()
print("src sha256", hashlib.sha256(s.encode()).hexdigest())

SUBS = [
    # 0 -- the docstring: say what this file is, and that it is generated
    ('"""E-DEP-VGEO-5 -- VGEO-4 + the PRE-REGISTERED collinearity screen (LR14-1 / DP17-1). See ../SPEC.md (pinned).',
     '"""E-DEP-VGEO-6 -- the VGEO-5 panel, UNCHANGED, on the s16 trunk stage (LR15-1 / DP18-1). See ../SPEC.md (pinned).\n\n'
     'GENERATED from vgeo5_screened.py by code/make_vgeo6.py through enumerated substitutions; the\n'
     'derivation is the audit trail. The ONLY change is the encoder STAGE the latent arms read:\n'
     's32 (704ch, 8x20) -> s16 (352ch, 16x40), pooled to the SAME 2x5 readout geometry. Split seed,\n'
     'screen, estimator, pixel control and P2 triplets are byte-identical, so pixel_1440 MUST\n'
     'reproduce the landed numbers exactly -- that is this run\'s harness control.'),
    # 1 -- the token file
    ('tokens_s32_fp16.npy', 'tokens_s16_fp16.npy'),
    # 2 -- feature allocation
    ('feats = {"latent_704": np.empty((len(rows), 704), np.float32),\n'
     '         "latent_7040": np.empty((len(rows), 7040), np.float32),',
     'feats = {"latent_352": np.empty((len(rows), 352), np.float32),\n'
     '         "latent_3520": np.empty((len(rows), 3520), np.float32),'),
    # 3 -- global pool
    ('    feats["latent_704"][i:i + n] = t.mean(axis=(2, 3))',
     '    feats["latent_352"][i:i + n] = t.mean(axis=(2, 3))'),
    # 4 -- 2x5 grid pool: 8x20 in blocks of 4x4  ->  16x40 in blocks of 8x8 (same 2x5 output)
    ('    feats["latent_7040"][i:i + n] = t.reshape(n, 704, 2, 4, 5, 4).mean(axis=(3, 5)).reshape(n, -1)',
     '    feats["latent_3520"][i:i + n] = t.reshape(n, 352, 2, 8, 5, 8).mean(axis=(3, 5)).reshape(n, -1)'),
    # 5 -- the control block keys on the gridded latent arm
    ('    if name == "latent_7040":', '    if name == "latent_3520":'),
    # 6/7 -- control arm names
    ('"C_presid|latent_7040_diag"', '"C_presid|latent_3520_diag"'),
    ('"C_mut|latent_7040_diag"', '"C_mut|latent_3520_diag"'),
    # 8 -- paired differences
    ('for lat in ("latent_704", "latent_7040"):', 'for lat in ("latent_352", "latent_3520"):'),
    # 9 -- the spec string written into the json
    ('"spec": "SPEC.md E-DEP-VGEO-5 (VGEO-4 + pre-registered |r_lp|>=0.80 collinearity screen)"',
     '"spec": "SPEC.md E-DEP-VGEO-6 (the VGEO-5 screened panel, unchanged, on the s16 stage)"'),
    # 10 -- default output name
    ('raw/vgeo5_step4.json', 'raw/vgeo6_step4.json'),
]
for i, (a, b) in enumerate(SUBS):
    assert s.count(a) == 1, (i, s.count(a), a[:60])
    s = s.replace(a, b)

# The harness control is a POST-HOC comparison against the landed json, not a code change:
# pixel_1440's arms must be bit-identical. Assert nothing here; the RESULT reports it.
os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, "w", encoding="utf-8", newline="\r\n").write(s)  # match the source's CRLF (repo convention)
print("dst sha256", hashlib.sha256(s.encode()).hexdigest())
print("wrote", DST, len(s), "bytes,", s.count("\n"), "lines")
for tok in ("s32", "704", "7040"):
    print("residual", tok, "=", s.count(tok))
