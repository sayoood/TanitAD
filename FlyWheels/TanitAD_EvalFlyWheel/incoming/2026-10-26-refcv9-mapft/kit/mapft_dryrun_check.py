#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mapft_dryrun_check.py -- dev-box DRY CHECK of the MAPFT argv (no GPU, no Thor, no training).

  PYTHONPATH=<tree>/stack;<tree>/taniteval  python mapft_dryrun_check.py --tree F:/refcv9_mapft_scratch/tree \
        [--s-step 3.0 --smoke-wall-s 600] [--ema-file E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt] [--json OUT]

<tree> is a CLEAN EXPORT of the landed commit (git archive 70211d6 stack taniteval). The script ASSERTS that
`tanitad.__file__` lives under <tree> (CLAUDE.md: the venv's editable install points at the G: mount) and then checks,
on the SAMPLE budget it is given (the real N comes from the Thor smoke; this proves the derived argv is well-formed):

  P1  the 50-step SMOKE argv and the FULL argv PARSE with the landed trainer's own parser (refc_v3_train.build_parser)
  P2  map_only_ft.pin_args ACCEPTS both (the flag's argv refusals)
  P3  the gate's argv rules for profile refcv8-mapft report NO refusal on the FULL argv (init-from md5 bound to the
      shipped EMA model by CONTENT -- the real 403 MB file is hashed when --ema-file exists)
  R*  DELIBERATE-REGRESSION arms: each broken argv must be REFUSED, by trainer-side pin_args AND/OR the gate's rules:
        keep --grad-share-every | keep --b1-guard-ref-ap2m | keep --b1-guard-step | --conflict-detector auto |
        keep --conflict-every | drop --init-from | drop --r8-train-only-map | add --r8-critic-drivable |
        add --trunk-bn-recalib | --init-from a file with the wrong md5 | add --b1-guard-* | --w-map-hires 0
  The expectation for each arm is a LITERAL (which side must refuse); an arm that is NOT refused is a FAILURE.

Verdict is the printed table + the JSON; markers ZZMAPFT-DRYCHECK-{PASS,FAIL}ZZ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_mapft_argv as MK  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--s-step", default="3.0")
    ap.add_argument("--smoke-wall-s", default="600")
    ap.add_argument("--ema-file", default="E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt")
    ap.add_argument("--base", default=str(HERE / "refcv8_launch_argv_base.json"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tree = Path(a.tree).resolve()
    sys.path[:0] = [str(tree / "stack"), str(tree / "stack" / "scripts"), str(tree / "taniteval")]
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    import tanitad  # noqa: E402
    tf = Path(tanitad.__file__).resolve()
    assert str(tf).lower().startswith(str(tree).lower()), f"tanitad imported from {tf}, not from {tree}"
    print(f"IMPORT-OK tanitad -> {tf}")
    import refc_v3_train as T  # noqa: E402
    import launch_gate as LG  # noqa: E402
    from tanitad.train import map_only_ft as MOF  # noqa: E402
    print(f"IMPORT-OK trainer -> {Path(T.__file__).resolve()} | gate -> {Path(LG.__file__).resolve()}")

    base = MK.load_base(a.base)
    bud = MK.derive_budget(a.s_step, a.smoke_wall_s)
    assert not bud["stop"], bud
    full, _ = MK.build_full(base, bud)
    smoke, _ = MK.build_smoke(base)
    print(f"SAMPLE BUDGET s_step={a.s_step} smoke_wall_s={a.smoke_wall_s} -> N={bud['N']} eval_every={bud['eval_every']} "
          f"ema_decay={bud['ema_decay']} (SAMPLE only: the real N comes from the Thor smoke)")

    rows, ok = [], True

    def rec(name, expect, got, detail=""):
        nonlocal ok
        good = (expect == got)
        ok &= good
        rows.append({"check": name, "expect": expect, "got": got, "ok": good, "detail": detail[:300]})
        print(f"{'OK  ' if good else 'FAIL'} {name:46s} expect={expect:9s} got={got:9s} {detail[:140]}")

    def trainer_side(argv):
        """-> ('accepted'|'REFUSED', detail). parse + map_only_ft.pin_args (+ the trainer's other argv-only pins that
        the build_parser/parse path runs)."""
        try:
            args = T.build_parser().parse_args(list(argv))
        except SystemExit as e:                       # argparse error
            return "REFUSED", f"parse: {e}"
        try:
            MOF.pin_args(args)
        except SystemExit as e:
            return "REFUSED", str(e).replace("\n", " | ")[:300]
        return "accepted", f"steps={args.steps} warmup={args.warmup} ema={args.ema_decay} eval_every={args.eval_every}"

    # ---- the gate's rules need: the shipped EMA model by content, and the tau record bound to the argv's v2-cache ----
    prof = LG.PROFILES["refcv8-mapft"]
    tmp = Path(tempfile.mkdtemp(prefix="mapft_dry_", dir="F:/refcv9_mapft_scratch" if os.path.isdir("F:/refcv9_mapft_scratch") else None))
    pmap = []
    init_flag = MK.flag_val(full, "--init-from")[0]
    ema = Path(a.ema_file)
    if ema.is_file():
        pmap = [(init_flag.rsplit("/", 1)[0], str(ema.parent))]
        have_real = (hashlib.md5(ema.read_bytes()).hexdigest() == prof["map_only"]["init_from_md5"])
        print(f"EMA file {ema} md5 matches the profile pin: {have_real}")
    else:
        have_real = False
        print(f"NOTE: {ema} not found: the gate's init-md5 rule will REFUSE (that is the rule working, not a pass)")
    # the FIX-4 tau record: the REAL one cites the Thor-only labels file's sha256 ("cannot be checked on this host"), so
    # -- exactly as stack/tests/test_map_only_ft.py does -- a stand-in keyed to THIS argv is used. The tau check is not
    # what is under test here; every arm below additionally requires ITS OWN rule's signature in the refusal text, so a
    # refusal for this unrelated reason can never be credited to an arm (it would read REFUSED-OTHER).
    tau_rec = tmp / "nav_tau.json"
    tau_rec.write_text(json.dumps({
        "status": "OK", "tau": float(MK.flag_val(full, "--nav-compliance-tau-rad")[0]),
        "split_v2_cache": MK.flag_val(full, "--v2-cache")[0], "labels": MK.flag_val(full, "--v7-labels")[0]}),
        encoding="utf-8")
    opts = {"nav_tau_record": str(tau_rec)}

    def gate_side(argv, options=None):
        try:
            refusals, pending = LG.profile_argv_rules(prof, list(argv), options or opts, pmap)
        except SystemExit as e:
            return "REFUSED", str(e)[:300]
        except Exception as e:                       # noqa: BLE001 -- surfaced, never swallowed
            return "ERROR", f"{type(e).__name__}: {e}"
        return ("REFUSED", " | ".join(r[:140] for r in refusals[:4])) if (refusals or pending) else ("accepted", "")

    # ----------------------------------------------------------------------------------------- P1/P2/P3
    for nm, av in (("SMOKE", smoke), ("FULL", full)):
        try:
            args = T.build_parser().parse_args(list(av))
            rec(f"P1 {nm} argv parses with the landed parser", "parses", "parses", f"{len(av)} tokens")
        except SystemExit as e:
            rec(f"P1 {nm} argv parses with the landed parser", "parses", "REFUSED", str(e))
            continue
        try:
            MOF.pin_args(args)
            rec(f"P2 {nm} argv accepted by map_only_ft.pin_args", "accepted", "accepted")
        except SystemExit as e:
            rec(f"P2 {nm} argv accepted by map_only_ft.pin_args", "accepted", "REFUSED", str(e))
    g, why = gate_side(full)
    rec("P3 FULL argv: gate rules (profile refcv8-mapft)", "accepted", g, why)
    if not have_real:
        print("  (P3 is only fully meaningful with the real EMA file present)")

    # ----------------------------------------------------------------------------------------- RED arms
    def with_(av, remove=(), add=()):
        gs = MK.groups(list(av))
        gs = [(f, v) for f, v in gs if f not in set(remove)]
        for f, v in add:
            gs = [(ff, vv) for ff, vv in gs if ff != f] + [(f, [str(x) for x in v])]
        return MK.flatten(gs)

    base_args = {  # values from the refcv8 launch argv, re-added to show the flag is what is refused
        "gse": ("--grad-share-every", ["250"]), "b1r": ("--b1-guard-ref-ap2m", ["0.21211"]),
        "b1s": ("--b1-guard-step", ["1000"]), "ce": ("--conflict-every", ["50"]),
    }
    # (name, argv, who must refuse: 'trainer' | 'gate' | 'both', signature: a substring the refusal text MUST contain)
    arms = [
        ("R1 keep --grad-share-every 250", with_(full, add=[base_args["gse"]]), "both", "grad-share-every"),
        ("R2 keep --b1-guard-ref-ap2m", with_(full, add=[base_args["b1r"]]), "both", "b1-guard"),
        ("R3 keep --b1-guard-step", with_(full, add=[base_args["b1s"]]), "gate", "b1-guard-step"),
        ("R4 --conflict-detector auto", with_(full, add=[("--conflict-detector", ["auto"])]), "both", "conflict-detector"),
        ("R5 keep --conflict-every 50", with_(full, add=[base_args["ce"]]), "both", "conflict-every"),
        ("R6 drop --init-from", with_(full, remove=["--init-from"]), "both", "init-from"),
        ("R7 drop --r8-train-only-map (flag absent)", with_(full, remove=["--r8-train-only-map"]), "gate", "requires --r8-train-only-map"),
        ("R8 add --r8-critic-drivable", with_(full, add=[("--r8-critic-drivable", []), ("--w-r8-drivable", ["1.0"])]), "both", "critic-drivable"),
        ("R9 add --trunk-bn-recalib 1", with_(full, add=[("--trunk-bn-recalib", ["1"])]), "both", "trunk-bn-recalib"),
        ("R10 --w-map-hires 0", with_(full, add=[("--w-map-hires", ["0"])]), "trainer", "w-map-hires"),
        ("R11 add --b1-guard-ref-ap2m + --b1-guard-step", with_(full, add=[base_args["b1r"], base_args["b1s"]]), "both", "b1-guard"),
    ]
    def credited(res, why, sig):
        """REFUSED only when the refusal text carries THIS arm's signature; any other refusal is REFUSED-OTHER."""
        if res != "REFUSED":
            return res
        return "REFUSED" if sig.lower() in why.lower() else "REFUSED-OTH"

    for name, av, who, sig in arms:
        t_res, t_why = trainer_side(av)
        g_res, g_why = gate_side(av)
        if who in ("trainer", "both"):
            rec(f"{name} [trainer pin_args]", "REFUSED", credited(t_res, t_why, sig), t_why)
        if who in ("gate", "both"):
            rec(f"{name} [gate rules]", "REFUSED", credited(g_res, g_why, sig), g_why)
    # a wrong-content warm start: the gate must refuse by MD5 (a 1 KiB stand-in at the same path)
    bad_dir = tmp / "badinit"
    bad_dir.mkdir()
    (bad_dir / ema.name).write_bytes(b"not the shipped EMA model" * 40)
    g_res, g_why = ("REFUSED", "")
    try:
        r, _ = LG.profile_argv_rules(prof, list(full), opts, [(init_flag.rsplit("/", 1)[0], str(bad_dir))])
        g_res, g_why = ("REFUSED", " | ".join(x[:140] for x in r[:3])) if r else ("accepted", "")
    except Exception as e:                           # noqa: BLE001
        g_res, g_why = "ERROR", f"{type(e).__name__}: {e}"
    rec("R12 --init-from file with the WRONG md5 [gate rules]", "REFUSED", credited(g_res, g_why, "md5"), g_why)
    # control: the SAME refusal machinery must NOT fire on the unbroken argv (a gate that refuses everything proves nothing)
    t_res, t_why = trainer_side(full)
    rec("C1 control: unbroken FULL argv [trainer pin_args]", "accepted", t_res, t_why)

    out = {"tree": str(tree), "tanitad_file": str(tf), "sample_budget": bud, "full_argv_sha256": MK.sha_compact(full),
           "smoke_argv_sha256": MK.sha_compact(smoke), "real_ema_present_and_md5_ok": have_real, "rows": rows,
           "all_ok": bool(ok)}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"SUMMARY: {sum(r['ok'] for r in rows)}/{len(rows)} checks as expected")
    print("ZZMAPFT-DRYCHECK-PASSZZ" if ok else "ZZMAPFT-DRYCHECK-FAILZZ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
