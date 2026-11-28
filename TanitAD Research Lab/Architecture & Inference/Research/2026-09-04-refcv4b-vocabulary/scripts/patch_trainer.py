"""refcv4-b trainer edits: the v0-conditioned vocabulary end-to-end."""
import io
import sys

P = sys.argv[1]
raw = io.open(P, "rb").read()
CRLF = raw.count(b"\r\n")
BARE = raw.count(b"\n") - CRLF
assert BARE == 0 or CRLF == 0, "mixed line endings (%d/%d)" % (CRLF, BARE)
NL = "\r\n" if CRLF else "\n"
src = raw.decode("utf-8").replace("\r\n", "\n")
orig = src
EDITS = []
print("line endings: %s" % ("CRLF" if CRLF else "LF"))


def sub(old, new, tag):
    global src
    n = src.count(old)
    assert n == 1, "edit %r matched %d times (expected 1)" % (tag, n)
    src = src.replace(old, new)
    EDITS.append(tag)


# --------------------------------------------------------------------------
# 1. the config lever — must be set BEFORE the model is built
# --------------------------------------------------------------------------
sub("""    if getattr(args, "sel_accel_max", None) is not None:
        cfg.core.sel_accel_max = float(args.sel_accel_max)
    return cfg""",
    """    if getattr(args, "sel_accel_max", None) is not None:
        cfg.core.sel_accel_max = float(args.sel_accel_max)
    # ⭐ refcv4-b — the v0-CONDITIONED vocabulary. Applied to BOTH arms, so
    # `config_delta` (hier vs flat) is unchanged and C122 still passes.
    if getattr(args, "anchor_v0_conditioned", False):
        cfg.core.anchors.v0_conditioned = True
        cfg.core.anchors.ref_speed_ms = float(
            getattr(args, "anchor_ref_speed", 10.0))
    return cfg""",
    "pin-cfg")

# --------------------------------------------------------------------------
# 2. CLI
# --------------------------------------------------------------------------
sub('''    ap.add_argument("--anchors", default=None,
                    help="6 s anchor vocabulary (build_refc_anchors.py over "
                         "V3_HORIZONS; the model's synthetic default otherwise)")''',
    '''    ap.add_argument("--anchors", default=None,
                    help="6 s anchor vocabulary (build_refc_anchors.py over "
                         "V3_HORIZONS; the model's synthetic default otherwise)")
    ap.add_argument("--anchor-v0-conditioned", action="store_true",
                    help="refcv4-b: the anchor file carries `controls` [N, 2] = "
                         "(accel, curvature) and the bank is ROLLED PER WINDOW "
                         "from that window's measured v0 instead of being a "
                         "fixed set of paths in absolute metres. MEASURED on the "
                         "4,823-window banked surface: the fixed-path set reads "
                         "0.3773 m oracle-in-vocabulary (+0.0777 [+0.0528, "
                         "+0.1044] SEPARATED WORSE than ha = 0.2996); the "
                         "v0-conditioned family at 117 candidates reads 0.2610 "
                         "(-0.0387 [-0.0652, -0.0104] BEATS ha). Requires a "
                         "`controls` entry in --anchors; refused without one.")
    ap.add_argument("--anchor-ref-speed", type=float, default=10.0,
                    help="m/s the bank is rolled at where the ego channel was "
                         "WITHHELD by --ego-dropout. Rolling a withheld row from "
                         "its true v0 would put the withheld channel into the "
                         "candidate GEOMETRY, which is a harder leak than the "
                         "ranking one S2 guards.")''',
    "cli")

# --------------------------------------------------------------------------
# 3. loading — the controls travel with the paths, and a mismatch is REFUSED
# --------------------------------------------------------------------------
sub("""    if args.anchors:
        anc = torch.load(args.anchors, map_location=device,
                         weights_only=True)
        anc = anc["anchors"] if isinstance(anc, dict) else anc
        model.core.decoder.load_anchors(anc.to(device))""",
    """    if args.anchors:
        anc = torch.load(args.anchors, map_location=device,
                         weights_only=True)
        _ctrl = anc.get("controls") if isinstance(anc, dict) else None
        anc = anc["anchors"] if isinstance(anc, dict) else anc
        # ⛔ BOTH DIRECTIONS ARE REFUSED, LOUDLY. A v0-conditioned build given a
        # controls-free file would silently fall back to fixed paths — the exact
        # vocabulary this arm exists to replace — and a fixed build given a
        # controls file would train against a bank rolled at one reference speed
        # while the file's author meant per-window. Neither failure is visible in
        # any artifact the run leaves behind, so neither is allowed to be silent.
        _want = bool(getattr(args, "anchor_v0_conditioned", False))
        if _want and _ctrl is None:
            raise SystemExit(
                "[v3] ⛔ --anchor-v0-conditioned given but "
                f"{args.anchors} carries no `controls` [N, 2]. The bank is "
                "rolled per window from those controls; there is nothing to "
                "roll.")
        if _ctrl is not None and not _want:
            raise SystemExit(
                f"[v3] ⛔ {args.anchors} carries `controls` [N, 2] (a "
                "v0-CONDITIONED vocabulary) but --anchor-v0-conditioned was NOT "
                "given. Its paths are the family rolled at the REFERENCE speed "
                "only, and training on them fixed would silently be a different "
                "experiment.")
        model.core.decoder.load_anchors(
            anc.to(device), None if _ctrl is None else _ctrl.to(device))""",
    "load")

sub('''        print(f"[v3] anchors: loaded {tuple(anc.shape)} from {args.anchors} "
              f"(sha256 {_sha})", flush=True)''',
    '''        print(f"[v3] anchors: loaded {tuple(anc.shape)} from {args.anchors} "
              f"(sha256 {_sha})", flush=True)
        if _ctrl is not None:
            _ok = bool(((_ctrl[:, 0] == 0) & (_ctrl[:, 1] == 0)).any())
            print(f"[v3] anchors: v0-CONDITIONED, controls {tuple(_ctrl.shape)} "
                  f"(accel, curvature), rolled per window; withheld rows at "
                  f"{getattr(args, 'anchor_ref_speed', 10.0)} m/s. "
                  f"straight-ahead control {{a=0, kappa=0}} present: {_ok}",
                  flush=True)
            if not _ok:
                raise SystemExit(
                    "[v3] ⛔ the control grid does not contain {a=0, kappa=0} "
                    "EXACTLY. An even-count linspace omits it and the set then "
                    "reads 1.2768 m oracle-in-vocabulary against 0.2610 — a "
                    "4.9x artifact that looks exactly like a resolution "
                    "finding. Rebuild with odd counts.")''',
    "load-print")

# --------------------------------------------------------------------------
# 4. THE ANCHOR TARGET — measured against the geometry that was DECODED
# --------------------------------------------------------------------------
sub("""    anchors = model.core.decoder.anchors.to(traj_tgt.dtype)  # [N, S, 2]
    dist = (((traj_tgt[:, None] - anchors[None]) ** 2).sum(-1)
            * sv[:, None]).sum(-1)                          # [B, N] valid-only""",
    """    # ⛔ THE TARGET MUST BE MEASURED AGAINST THE BANK THAT WAS ACTUALLY
    # DECODED. With a v0-conditioned vocabulary `decoder.anchors` is the family
    # rolled at the REFERENCE speed, not this window's fan, so scoring `a_star`
    # against it would supervise the anchor classifier on a geometry the model
    # never emitted — silently, and with `anchor_acc` still reading plausibly.
    # `out["anchor_bank"]` is [B, N, S, 2] and is EXACTLY `x0`; for a fixed
    # vocabulary it is `anchors[None].expand(...)`, so this line is unchanged
    # arithmetic there (verified bit-identical, 2026-09-04).
    anchors = out["anchor_bank"].to(traj_tgt.dtype)          # [B, N, S, 2]
    dist = (((traj_tgt[:, None] - anchors) ** 2).sum(-1)
            * sv[:, None]).sum(-1)                          # [B, N] valid-only""",
    "anchor-target")

# --------------------------------------------------------------------------
# 5. provenance — the controls are part of the vocabulary's identity
# --------------------------------------------------------------------------
sub('''    stamp = {"path": str(path) if path else None,
             "shape": list(a.shape),
             "sha256_installed": h,
             "source": "file" if path else "refc.default_anchors (SYNTHETIC)"}''',
    '''    stamp = {"path": str(path) if path else None,
             "shape": list(a.shape),
             "sha256_installed": h,
             "source": "file" if path else "refc.default_anchors (SYNTHETIC)"}
    if controls is not None:
        c = controls.detach().to("cpu", torch.float32).contiguous()
        stamp["controls_shape"] = list(c.shape)
        stamp["controls_sha256_installed"] = hashlib.sha256(
            c.numpy().tobytes()).hexdigest()
        stamp["v0_conditioned"] = True
        stamp["straight_ahead_control_present"] = bool(
            ((c[:, 0] == 0) & (c[:, 1] == 0)).any())
    else:
        stamp["v0_conditioned"] = False''',
    "stamp-body")

sub('''def _anchor_stamp(path, anchors) -> dict:''',
    '''def _anchor_stamp(path, anchors, controls=None) -> dict:''',
    "stamp-sig")

sub('''        "anchors": _anchor_stamp(getattr(args, "anchors", None),
                                 model.core.decoder.anchors),''',
    '''        "anchors": _anchor_stamp(
            getattr(args, "anchors", None), model.core.decoder.anchors,
            model.core.decoder.anchor_controls
            if getattr(model.core.decoder, "anchor_v0_cond", False) else None),''',
    "stamp-call")

io.open(P, "wb").write(src.replace("\n", NL).encode("utf-8"))
print("applied %d edits:" % len(EDITS))
for e in EDITS:
    print("  - " + e)
print("delta bytes: %+d" % (len(src) - len(orig)))
