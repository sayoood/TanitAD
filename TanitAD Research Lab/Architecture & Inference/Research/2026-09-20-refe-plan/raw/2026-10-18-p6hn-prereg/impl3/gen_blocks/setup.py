    # ---------------- P6-HN-S (--hn-bank / --ctx-cache; default OFF = nothing below runs).  The bank and the cache are VERIFIED here, before the first step: a bank of another partition / label_version / fan
    # fingerprint / bytes, a bank with no hard negative, a cache of another trunk, or a scene the cache does not hold REFUSES (the join is strict; nothing is skipped or defaulted).
    hn_obj = cc_obj = trunk_fp = None
    hn_cur: dict = {"tail": (), "off": False, "acc": {"rows": 0, "expected": 0, "loss": 0.0, "n": 0, "nonempty": 0}}
    if a.hn_bank or a.ctx_cache:
        import ctx_cache as CC
        import hn_bank as HN
        hn_train_keys = [key(ds.rows[i]) for i in tr_idx]
        hn_held_keys = [key(ds.rows[i]) for i in ho_idx]
        hn_ckpt_of = {k_: int(op.by[k_][0]) for k_ in hn_train_keys}
        if a.ctx_cache:
            trunk_fp = CC.trunk_fingerprint(model.state_dict())
            try:
                cc_obj = CC.CtxCache(a.ctx_cache, expect_trunk_fingerprint=trunk_fp)
            except (CC.StaleCache, OSError, ValueError, KeyError) as e_:
                print(f"REFUSED: --ctx-cache {a.ctx_cache}: {e_}"); return 4
            cc_missing = [k_ for k_ in hn_train_keys + hn_held_keys if k_ not in cc_obj.index]
            if cc_missing:
                print(f"REFUSED: --ctx-cache holds no context for {len(cc_missing)} of the run's {len(hn_train_keys) + len(hn_held_keys)} scenes (first {cc_missing[0]}): the join is strict"); return 4
        if a.hn_bank:
            hn_lv = sorted({int(op.by[k_][3][1]) for k_ in hn_train_keys})
            if hn_lv != [HN_LABEL_VERSION]:
                print(f"REFUSED: the training sets carry label versions {hn_lv}; the bank is labelled at version {HN_LABEL_VERSION} (the fan labels the screen trains on)"); return 4
            hn_expect = {"partition_id": HN_PARTITION_ID, "label_version": hn_lv[0], "fan_fingerprint": HN.keys_fingerprint([k_ + (hn_ckpt_of[k_],) for k_ in hn_train_keys])}
            try:
                hn_obj = HN.load_bank(a.hn_bank, expect=hn_expect)                       # REFUSED on another partition / label_version / fan fingerprint, or bytes changed after minting
                hn_obj.batch(hn_train_keys, [hn_ckpt_of[k_] for k_ in hn_train_keys])    # the strict join: every training key, at the ckpt_step the run serves
            except (HN.BankMismatch, OSError, ValueError, KeyError) as e_:
                print(f"REFUSED: --hn-bank {a.hn_bank}: {e_}"); return 4
            if hn_obj.n_rows() == 0:
                print(f"REFUSED: --hn-bank {a.hn_bank} holds 0 hard-negative rows: a launch whose lever has nothing to act on is not an experiment"); return 4
            hn_mask_all = hn_obj.hn_mask()
        ds = ScreenItems(ds, cc_obj, hn_obj, hn_ckpt_of)                   # the TRAINING items: + the cached context, + the bank's rows
        ds_ho = ScreenItems(ds_ho, cc_obj, None, None)                      # the held-out items: the context only -- evaluation reads no bank (I4)
