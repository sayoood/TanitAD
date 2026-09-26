"""FIX-5 (sidecar meta) on refcv6_max_speed.py and the FIX-3 legacy as-trained rebuild on
taniteval/tools/refcv3_arm.py -- exact-match edits on the LF working copies."""
import sys

MS, ARM = sys.argv[1], sys.argv[2]


def patch(path, edits):
    s = open(path, encoding="utf-8", newline="").read()
    assert "\r" not in s, path
    for old, new, tag in edits:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"[patch] {path}: {tag}: anchor found {n} times")
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(s)
    print("patched", path.rsplit("/", 1)[-1], [t for _, _, t in edits])


patch(MS, [(
    '''    meta: dict = {}
    if _os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = _json.load(fh)
    if label_md5 and meta.get("source_md5") and \\
''', '''    meta: dict = {}
    if _os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = _json.load(fh)
    # ⛔⛔ FIX-5 (A16 audit Q6, G1 regression R3, 2026-09-26): the md5 cross-check below was
    # SKIPPED whenever the `.meta.json` was absent -- MEASURED: a sidecar built over a different
    # label blob read clean on the dev-box kit, which ships no meta. The builder
    # (`scripts/build_refcv6_speed_max_window.py`) ALWAYS writes one, so an absent meta is a copy
    # that lost its provenance. When the caller names the blob it loaded, the meta is REQUIRED.
    if label_md5 and not meta.get("source_md5"):
        raise SpeedMaxStampError(
            f"[refcv6-vmax] ⛔ {meta_path!r} is "
            f"{'ABSENT' if not _os.path.isfile(meta_path) else 'missing `source_md5`'}, so this "
            f"sidecar cannot be shown to be built over the label blob this run loaded "
            f"({label_md5!r}). The builder always writes it; copy it beside the sidecar (or "
            f"rebuild). Refusing rather than skipping the only check that catches a sidecar from "
            f"another release.")
    if label_md5 and meta.get("source_md5") and \\
''', "meta required with label_md5")])

patch(ARM, [(
    '''        cfg = tr._pin_trainer_cfg(base, args)
        if getattr(args, "graft_lan", False) or getattr(args, "goal_str", False):
            cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))
        src = ("config.json[argv] -> refc_v3_train.build_parser + "
               "_pin_trainer_cfg (the trainer's own build path)")
''', '''        cfg = tr._pin_trainer_cfg(base, args)
        if getattr(args, "graft_lan", False) or getattr(args, "goal_str", False):
            cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))
        src = ("config.json[argv] -> refc_v3_train.build_parser + "
               "_pin_trainer_cfg (the trainer's own build path)")
        # ⛔⛔ D-REFCV6-EQUALIZE-DROPPED (FIX-3, 2026-09-26): rebuild the trunk AS TRAINED. Before
        # the fix, `--equalize-bottom-rows N` with `--image-hw` never reached the trunk, so every
        # pre-fix checkpoint (refcv6-r101-s0 included: argv 43, trunk 0) was trained un-equalised.
        # The fixed pin now honours argv, which would evaluate a model the weights never were --
        # so a record that predates the stamp is rebuilt with the rows its trunk really zeroed.
        # Only the TRUNK moves; the lift received the argv value all along.
        _as_trained = getattr(tr, "trunk_equalize_rows_as_trained", None)
        if _as_trained is not None and hasattr(cfg.core.encoder, "trunk_equalize_bottom_rows"):
            _rows, _why = _as_trained(config)
            if int(cfg.core.encoder.trunk_equalize_bottom_rows) != int(_rows):
                cfg.core.encoder.trunk_equalize_bottom_rows = int(_rows)
                src += f"; TRUNK equalize_bottom_rows -> {int(_rows)} ({_why})"
''', "legacy as-trained trunk rows")])
