import repro_lib as L, torch
print("r8 cfg from the trainer's pin:", L.r8_from_prov())
for kind in ("r7", "r8"):
    cfg, m = L.build(kind)
    bt = L.R.batch(cfg, b=3)
    o1 = L.digests(L.fwd_forked(cfg, m, bt)); o2 = L.digests(L.fwd_forked(cfg, m, bt))
    d = L.differ(o1, o2)
    print(f"{kind}: n_outputs {len(o1)} differ {len(d)}: {d}")
