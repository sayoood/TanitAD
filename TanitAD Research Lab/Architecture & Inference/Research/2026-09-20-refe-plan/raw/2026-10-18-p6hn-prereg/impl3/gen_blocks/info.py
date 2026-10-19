    if hn_obj is not None:                                   # P6-HN-S: BUILT facts of the bank the run holds, for G-DVB (absent without --hn-bank: every pre-HN data event is byte-identical)
        info["hn"] = hn_obj.hn_block(HN_ARM)
    if cc_obj is not None:                                   # ... and of the cache (the trunk fingerprint is the CHECKPOINT's: CtxCache refused the load unless it equals the manifest's)
        info["ctx_cache"] = {"manifest_sha256": CC.sha256_file(os.path.join(a.ctx_cache, "manifest.json")), "trunk_fingerprint": trunk_fp, "n_scenes": int(cc_obj.man["n_scenes"]), "dtype": cc_obj.man["dtype"]}
