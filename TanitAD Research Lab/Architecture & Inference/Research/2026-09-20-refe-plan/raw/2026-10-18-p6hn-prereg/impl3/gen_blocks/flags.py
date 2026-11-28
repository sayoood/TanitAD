    # P6-HN-S (raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md): both default OFF = this trainer exactly.
    ap.add_argument("--hn-bank", default="", help="P6-HN-S: the offset hard-negative bank file (refe/hn_bank.py; its manifest sits beside it).  Arm B ONLY gains L_hn (hn_bank.hn_term); arm A stays the BCE-only control.  "
                    "Refused with --epo, --lam != 0, --b-mode != listnet, --aux-drv, or a bank of 0 hard-negative rows.  '' = no bank")
    ap.add_argument("--ctx-cache", default="", help="P6-HN-S: DIR of the frozen trunk's visual_ctx cache (refe/ctx_cache.py).  The scoring context is SERVED from it instead of recomputed by the trunk "
                    "(the cache's trunk fingerprint must equal the checkpoint's).  '' = the live trunk forward")
