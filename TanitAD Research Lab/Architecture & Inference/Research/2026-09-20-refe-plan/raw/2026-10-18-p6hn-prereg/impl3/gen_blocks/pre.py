        if hn_obj is not None:                               # G14 (P6-HN-S): the hard-negative term is live on arm B only, its controls read EXACTLY 0.0, the identities hold
            gates["G14_hn_live_and_identity"] = g14_hn_live_and_identity(hn_env)
        if cc_obj is not None:                               # G15 (P6-HN-S): the cache IS the live trunk forward, bit for bit, and a row-shuffled cache is not
            gates["G15_ctx_cache_identity"] = g15_ctx_cache_identity(hn_env)
