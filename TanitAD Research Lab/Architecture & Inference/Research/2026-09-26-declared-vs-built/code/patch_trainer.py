"""Apply the declared-vs-built fixes to the LF working copy of stack/scripts/refc_v3_train.py.

Every edit is an EXACT-MATCH replacement that must hit exactly once; the script refuses
otherwise, so it cannot silently apply half a patch to a moved tip.
"""
import sys

P = sys.argv[1]
s = open(P, encoding="utf-8", newline="").read()
assert "\r" not in s, "work on the LF copy"
EDITS = []


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times (need 1)")
    s = s.replace(old, new)
    EDITS.append(tag)


# ---- E0: imports -------------------------------------------------------------------------- #
edit("from tanitad.train import grad_conflict as _gcf  # noqa: E402  (refcv6 §6)\n",
     "from tanitad.train import grad_conflict as _gcf  # noqa: E402  (refcv6 §6)\n"
     "# ⛔⛔ SPEC_REFCV7 §2 launch gate, 2026-09-26: G-HYG (config dataclasses refuse undeclared\n"
     "# attributes) and G-DVB (every argv lever equals the BUILT model's value).\n"
     "from tanitad.train import config_hygiene as _hyg  # noqa: E402\n"
     "from tanitad.train import declared_vs_built as _dvb  # noqa: E402\n",
     "E0 imports")

# ---- E1: FIX-3 -- the equalisation pin, and its dead-lever refusal -------------------------- #
edit("    # ⛔ C26: equalize the rig-correlated black strip (see `--equalize-bottom-rows`).\n"
     "    cfg.core.encoder.trunk_equalize_bottom_rows = int(\n"
     "        getattr(args, \"equalize_bottom_rows\", 0) or 0)\n",
     "    # ⛔ C26: equalize the rig-correlated black strip (see `--equalize-bottom-rows`).\n"
     "    # ⛔⛔ FIX-3 (D-REFCV6-EQUALIZE-DROPPED, 2026-09-26): this line set an UNDECLARED attribute\n"
     "    # and the `--image-hw` `dataclasses.replace` below dropped it, so refcv6-r101-s0 trained\n"
     "    # with `--equalize-bottom-rows 43` in argv and an un-equalised trunk. It is now a DECLARED\n"
     "    # field (`CNNEncoderConfig.trunk_equalize_bottom_rows`) that the rebuild carries, the\n"
     "    # config class REFUSES any undeclared attribute (G-HYG), and G-DVB reads the BUILT trunk.\n"
     "    _eq_rows = int(getattr(args, \"equalize_bottom_rows\", 0) or 0)\n"
     "    if _eq_rows < 0:\n"
     "        raise SystemExit(\"[v3] ⛔ --equalize-bottom-rows %d: a row COUNT cannot be negative.\"\n"
     "                         % _eq_rows)\n"
     "    if _eq_rows > 0 and _trunk != \"timm\":\n"
     "        raise SystemExit(\n"
     "            \"[v3] ⛔ --equalize-bottom-rows %d with --trunk %s: the in-repo `refc` trunk \"\n"
     "            \"implements no equalisation (only `TimmResNetTrunk.normalise` zeroes rows) and the \"\n"
     "            \"BEV lift that shares the mask needs --trunk timm, so the flag would be stamped and \"\n"
     "            \"reach nothing. Pass --trunk timm, or drop the flag.\" % (_eq_rows, _trunk))\n"
     "    cfg.core.encoder.trunk_equalize_bottom_rows = _eq_rows\n",
     "E1 FIX-3 pin")

# ---- E2: G-HYG at the end of the pin ------------------------------------------------------ #
edit("    _pin_refcv6_tactical(cfg, args)\n    _pin_refcv7(cfg, args)\n    return cfg\n",
     "    _pin_refcv6_tactical(cfg, args)\n    _pin_refcv7(cfg, args)\n"
     "    # ⛔⛔ G-HYG: the strict config classes refuse an undeclared attribute AT ASSIGNMENT; this\n"
     "    # walk catches the two routes that bypass `__setattr__` (a `__dict__` write, and a config\n"
     "    # class of another module that is not decorated) before any model is built from it.\n"
     "    _hyg.assert_config_hygiene(cfg, where=\"_pin_trainer_cfg\")\n"
     "    return cfg\n",
     "E2 G-HYG walk")

# ---- E3: FIX-4 -- the refcv6 selection levers reach the config ---------------------------- #
edit("                \"--tac-decoder-v6, or drop --graft-behaviour-sel.\")\n"
     "        cfg.core.graft_behaviour_sel = True\n",
     "                \"--tac-decoder-v6, or drop --graft-behaviour-sel.\")\n"
     "        cfg.core.graft_behaviour_sel = True\n"
     "    # ---- ⛔⛔ FIX-4 (D-REFCV6-CONFIG-BUILD, 2026-09-26): the OTHER three refcv6 selection\n"
     "    # levers. `RefCConfig` declared them and the decoder builds them (refc.py), but NO FLAG\n"
     "    # reached them, while `provenance_roles()` declared all of them whenever v6 was on. OFF by\n"
     "    # default -- a run that passes none is bit-identical -- and every dead combination REFUSES.\n"
     "    # ⚠️ Turning any of them ON changes the arm; that is a design decision, not wiring.\n"
     "    if bool(getattr(args, \"graft_tac8_prior\", False)):\n"
     "        if not _on6:\n"
     "            raise SystemExit(\n"
     "                \"[v3] ⛔ --graft-tac8-prior without --tac-decoder-v6: the 8x8 tactical \"\n"
     "                \"posterior is produced by the behaviour decoder's scene hook and by nothing \"\n"
     "                \"else, so the zero-init 8 -> n_anchors grafts would be built and never fed.\")\n"
     "        cfg.core.graft_tac8_prior = True\n"
     "    _navc = bool(getattr(args, \"graft_nav_compliance\", False))\n"
     "    _navc_tau = float(getattr(args, \"nav_compliance_tau_rad\", 0.0) or 0.0)\n"
     "    if _navc and not _navc_tau > 0.0:\n"
     "        raise SystemExit(\n"
     "            \"[v3] ⛔ --graft-nav-compliance needs --nav-compliance-tau-rad > 0. The tolerance \"\n"
     "            \"is DERIVED per corpus (`taniteval.nav_compliance.derive_tolerance(pos, neg)`, or \"\n"
     "            \"scripts/refcv7_derive_nav_tau.py on the TRAIN split) and the arm must state it; \"\n"
     "            \"an invented one decides what 'compliant' means with no evidence class.\")\n"
     "    if _navc_tau and not _navc:\n"
     "        raise SystemExit(\n"
     "            \"[v3] ⛔ --nav-compliance-tau-rad %.6g without --graft-nav-compliance: the \"\n"
     "            \"tolerance of a term that is not built -- stamped and read by nothing.\" % _navc_tau)\n"
     "    if _navc:\n"
     "        cfg.core.graft_nav_compliance = True\n"
     "        cfg.core.nav_compliance_tau_rad = _navc_tau\n"
     "    if bool(getattr(args, \"speed_ceiling_filter\", False)):\n"
     "        if not bool(getattr(args, \"max_speed_input_v6\", False)):\n"
     "            raise SystemExit(\n"
     "                \"[v3] ⛔ --speed-ceiling-filter without --max-speed-input-v6: the filter's \"\n"
     "                \"ceiling IS the fed 4-way set speed; with no channel every row's limit is \"\n"
     "                \"+inf and the argmax filter is built and INERT.\")\n"
     "        cfg.core.speed_ceiling_filter = True\n",
     "E3 FIX-4 pins")

# ---- E4: FIX-4 -- the parser ------------------------------------------------------------- #
edit("                          \"the flag would be silently inert while \"\n"
     "                          \"config.json stamps it on.\")\n"
     "    # ---- refcv7 (PI 2026-09-19): DrivoR's proven planning heads ---------- #\n",
     "                          \"the flag would be silently inert while \"\n"
     "                          \"config.json stamps it on.\")\n"
     "    # ⛔⛔ FIX-4 (D-REFCV6-CONFIG-BUILD, 2026-09-26): the three refcv6 selection mechanisms\n"
     "    # `provenance_roles()` declared with no flag to build them. OFF by default (bit-identical).\n"
     "    g6t.add_argument(\"--graft-tac8-prior\", action=\"store_true\",\n"
     "                     help=\"refcv6 §4: the 8x8 tactical posterior REPLACES the image-only \"\n"
     "                          \"lat3/lon3 anchor prior, through NEW zero-init 8 -> n_anchors \"\n"
     "                          \"grafts (1,872 params at 117 anchors). Needs --tac-decoder-v6. \"\n"
     "                          \"⚠️ CHANGES THE ARM.\")\n"
     "    g6t.add_argument(\"--graft-nav-compliance\", action=\"store_true\",\n"
     "                     help=\"refcv6 §5: the PARAMETER-FREE nav-compliance predicate reaches \"\n"
     "                          \"the RANKED score behind ONE zero-init gate. Needs \"\n"
     "                          \"--nav-compliance-tau-rad. ⚠️ CHANGES THE ARM.\")\n"
     "    g6t.add_argument(\"--nav-compliance-tau-rad\", type=float, default=0.0,\n"
     "                     help=\"the terminal-heading tolerance of --graft-nav-compliance, DERIVED \"\n"
     "                          \"on the TRAIN split (no default that means anything; 0 = unset).\")\n"
     "    g6t.add_argument(\"--speed-ceiling-filter\", action=\"store_true\",\n"
     "                     help=\"refcv6 §5: the fed 4-way set speed FILTERS THE ARGMAX (the S2 \"\n"
     "                          \"pattern; the score stays unmasked). Needs --max-speed-input-v6. \"\n"
     "                          \"⚠️ CHANGES THE ARM, and the channel is EGO-FUTURE derived.\")\n"
     "    # ---- refcv7 (PI 2026-09-19): DrivoR's proven planning heads ---------- #\n",
     "E4 FIX-4 parser")

# ---- E5: the seam stamp ------------------------------------------------------------------- #
edit("        \"trunk_compile\": bool(getattr(core.encoder, \"trunk_compile\", False)),\n",
     "        \"trunk_compile\": bool(getattr(core.encoder, \"trunk_compile\", False)),\n"
     "        # ⛔⛔ FIX-3: C26 in the RECORD. Absent from every config.json before 2026-09-26, which\n"
     "        # is also how `trunk_equalize_rows_as_trained` tells a pre-fix record from a new one.\n"
     "        \"trunk_equalize_bottom_rows\": int(core.encoder.trunk_equalize_bottom_rows),\n",
     "E5a stamp equalize")
edit("        \"w_u0\": float(getattr(args, \"w_u0\", U0_WEIGHT_DEFAULT)),\n",
     "        \"w_u0\": float(getattr(args, \"w_u0\", U0_WEIGHT_DEFAULT)),\n"
     "        # ⭐ `--ack-ddim-no-u0` promised this stamp; before G-HYG it lived only as an ad-hoc\n"
     "        # attribute nothing read (the refcv6-r101-s0 config.json carries no such key).\n"
     "        \"u0_absent_under_ddim\": getattr(cfg, \"u0_absent_under_ddim\", None),\n"
     "        # ⭐ FIX-4: the four refcv6 selection levers, as the CONFIG carries them (the built\n"
     "        # facts are `provenance_roles.selection_mechanisms_built`, checked by G-DVB).\n"
     "        \"graft_tac8_prior\": bool(getattr(core, \"graft_tac8_prior\", False)),\n"
     "        \"graft_behaviour_sel\": bool(getattr(core, \"graft_behaviour_sel\", False)),\n"
     "        \"graft_nav_compliance\": bool(getattr(core, \"graft_nav_compliance\", False)),\n"
     "        \"nav_compliance_tau_rad\": float(getattr(core, \"nav_compliance_tau_rad\", 0.0)),\n"
     "        \"speed_ceiling_filter\": bool(getattr(core, \"speed_ceiling_filter\", False)),\n",
     "E5b stamp u0 + selection")

# ---- E6: assert_seams_are_built -- the equalisation stamp vs the built trunk -------------- #
edit("    elif ms_built:\n"
     "        bad.append(\n"
     "            \"the seam stamp carries no `max_speed_input` block but the \"\n"
     "            \"conditioner WAS BUILT -- a live seam absent from the record\")\n"
     "\n"
     "    if bad:\n",
     "    elif ms_built:\n"
     "        bad.append(\n"
     "            \"the seam stamp carries no `max_speed_input` block but the \"\n"
     "            \"conditioner WAS BUILT -- a live seam absent from the record\")\n"
     "\n"
     "    # --- FIX-3: C26 -- the stamped equalisation vs the rows the BUILT trunk zeroes --------- #\n"
     "    if \"trunk_equalize_bottom_rows\" in stamp and _is_timm:\n"
     "        _eq_b = int(getattr(getattr(_enc, \"cfg\", None), \"equalize_bottom_rows\", -1))\n"
     "        if int(stamp[\"trunk_equalize_bottom_rows\"]) != _eq_b:\n"
     "            bad.append(\n"
     "                f\"stamp says trunk_equalize_bottom_rows=\"\n"
     "                f\"{stamp['trunk_equalize_bottom_rows']} but the BUILT trunk zeroes {_eq_b} \"\n"
     "                f\"rows -- D-REFCV6-EQUALIZE-DROPPED\")\n"
     "\n"
     "    if bad:\n",
     "E6 assert_seams equalize")

# ---- E7: G-DVB before config.json and before step 1 --------------------------------------- #
edit("    assert_seams_are_built(model, _seams)\n",
     "    assert_seams_are_built(model, _seams)\n"
     "    # ⛔⛔ G-DVB (SPEC_REFCV7 §2): ARGV against the BUILT model, skipping the config in between.\n"
     "    # `assert_seams_are_built` above compares the stamp (built FROM the config) with the model,\n"
     "    # so a lever lost between argv and the config is lost from both and they agree -- that is\n"
     "    # how refcv6-r101-s0 passed with `--equalize-bottom-rows 43` and an un-equalised trunk.\n"
     "    # Here, after the withheld bank, the optimizer and any resume, and before config.json and\n"
     "    # the first step, every registered lever is read off the modules. Any mismatch REFUSES.\n"
     "    _dvb.refuse_on_mismatch(model, args, build_parser(), where=\"train\")\n",
     "E7 G-DVB call")
edit("        \"trunk_bn_recalib\": None,\n",
     "        \"trunk_bn_recalib\": None,\n"
     "        # ⭐ G-DVB passed (it refuses otherwise): what it compared, for the record\n"
     "        \"declared_vs_built\": {\"registry_entries\": len(_dvb.REGISTRY),\n"
     "                              \"built_checks\": sum(1 for _l in _dvb.REGISTRY.values()\n"
     "                                                  if _l.kind in (\"built\", \"loss\")),\n"
     "                              \"mismatches\": 0,\n"
     "                              \"module\": \"tanitad/train/declared_vs_built.py\"},\n",
     "E7b G-DVB record")

# ---- E8: G3 -- the label-time guard ------------------------------------------------------- #
edit("    #: ⛔ THE DELIBERATE-REGRESSION SWITCH, for tests ONLY: True restores the historical\n"
     "    #: ``(t + w - 1) * v7_dt`` so a guard can prove it goes RED on the defect it exists for.\n"
     "    legacy_label_clock: bool = False\n",
     "    #: ⛔ THE DELIBERATE-REGRESSION SWITCH, for tests ONLY: True restores the historical\n"
     "    #: ``(t + w - 1) * v7_dt`` so a guard can prove it goes RED on the defect it exists for.\n"
     "    legacy_label_clock: bool = False\n"
     "\n"
     "    def assert_label_clock_true(self, sidecar_path: str | None, *, split: str,\n"
     "                                synthetic: bool = False,\n"
     "                                tol_s: float | None = None,\n"
     "                                max_unverified_frac: float | None = None) -> dict:\n"
     "        \"\"\"⛔⛔ G3 / G-CLOCK (SPEC_REFCV7 FIX-2): every label read lands within ``tol_s`` of the\n"
     "        clip's MEASURED clock, or the run REFUSES.\n"
     "\n"
     "        ``t_true`` is computed HERE from the sidecar table read afresh and the literal rule\n"
     "        *provider row r is raw row r + n_stack - 1, and raw row k sits at grid_start + k * dt*\n"
     "        -- never through :meth:`_now_s`, which is the code under test. Three rows per clip\n"
     "        (its first, middle and last window). The historical ``(t + w - 1) * 0.1`` read labels\n"
     "        0.369 s early at the median (A16, q4d) and is this guard's regression arm.\n"
     "\n"
     "        * a synthetic corpus has no true clock: skipped and STAMPED as skipped;\n"
     "        * a real corpus with no ``--clip-clock-sidecar``: REFUSED -- the fallback reads\n"
     "          ``grid_start_s = 0.0``, +0.113 s off at the corpus median, so nothing is verifiable;\n"
     "        * clips the sidecar does not cover are UNVERIFIED; more than ``max_unverified_frac`` of\n"
     "          the split REFUSES, fewer are counted and stamped by name-count.\n"
     "        \"\"\"\n"
     "        from tanitad.data import clip_clock as _cc\n"
     "        tol = float(LABEL_CLOCK_TOL_S if tol_s is None else tol_s)\n"
     "        cap = float(LABEL_CLOCK_MAX_UNVERIFIED_FRAC if max_unverified_frac is None\n"
     "                    else max_unverified_frac)\n"
     "        if synthetic:\n"
     "            return {\"g3\": \"SKIPPED: synthetic corpus (no true clock exists)\", \"split\": split}\n"
     "        if not sidecar_path:\n"
     "            raise SystemExit(\n"
     "                f\"[v3] ⛔ G3 ({split}): v7/v8 labels on a REAL corpus with no \"\n"
     "                f\"--clip-clock-sidecar. The pose-dt fallback clocks every clip from \"\n"
     "                f\"grid_start 0.0, which is +0.113 s off at the corpus median -- no label time \"\n"
     "                f\"can be shown within {tol} s of the truth. Build it with \"\n"
     "                f\"scripts/build_clip_clock_sidecar.py and pass it.\")\n"
     "        table = _cc.read_clip_clock_sidecar(str(sidecar_path)).table\n"
     "        by_ep: dict = {}\n"
     "        for e_i, t in self.index:\n"
     "            by_ep.setdefault(int(e_i), []).append(int(t))\n"
     "        w = int(self.window)\n"
     "        worst, bad, unverified, n_checked = 0.0, [], [], 0\n"
     "        for e_i, ts in by_ep.items():\n"
     "            ep = self.episodes[e_i]\n"
     "            sid = int(ep.episode_id)\n"
     "            if sid not in table:\n"
     "                unverified.append(sid)\n"
     "                continue\n"
     "            g0, dt = table[sid]\n"
     "            n_stack = int(ep.frames.shape[1]) // 3\n"
     "            ts = sorted(ts)\n"
     "            for t in {ts[0], ts[len(ts) // 2], ts[-1]}:\n"
     "                want = float(g0) + (t + w - 1 + (n_stack - 1)) * float(dt)\n"
     "                got = float(self._now_s(ep, t))\n"
     "                err = abs(got - want)\n"
     "                n_checked += 1\n"
     "                worst = max(worst, err)\n"
     "                if err > tol:\n"
     "                    bad.append((sid, t, round(got, 4), round(want, 4)))\n"
     "        if bad:\n"
     "            raise SystemExit(\n"
     "                f\"[v3] ⛔ G3 ({split}): {len(bad)} of {n_checked} label reads are more than \"\n"
     "                f\"{tol} s from the clip's MEASURED clock (worst {worst:.4f} s). The trainer \"\n"
     "                f\"would supervise the tactical heads at the wrong instant \"\n"
     "                f\"(D-REFCV6-LABEL-CLOCK). First (sid, t, t_trainer, t_true): {bad[:4]}\")\n"
     "        n_ep = len(by_ep)\n"
     "        frac = len(unverified) / max(n_ep, 1)\n"
     "        if frac > cap:\n"
     "            raise SystemExit(\n"
     "                f\"[v3] ⛔ G3 ({split}): {len(unverified)} of {n_ep} clips ({frac:.2%}) have no \"\n"
     "                f\"measured clock in {sidecar_path} -- above the {cap:.0%} the guard admits \"\n"
     "                f\"(LABEL_CLOCK_MAX_UNVERIFIED_FRAC). Rebuild the sidecar over this split.\")\n"
     "        return {\"g3\": \"PASS\", \"split\": split, \"tol_s\": tol,\n"
     "                \"n_clips\": n_ep, \"n_reads_checked\": n_checked,\n"
     "                \"worst_abs_err_s\": round(worst, 6),\n"
     "                \"n_unverified_clips\": len(unverified),\n"
     "                \"max_unverified_frac\": cap,\n"
     "                \"rule\": \"t_true = grid_start_s + (t + w - 1 + n_stack - 1) * dt_s (sidecar)\"}\n",
     "E8a G3 method")
edit("        clip_clock_stats = ds.enable_clip_clock(getattr(args, \"clip_clock_sidecar\", None))\n",
     "        clip_clock_stats = ds.enable_clip_clock(getattr(args, \"clip_clock_sidecar\", None))\n"
     "        # ⛔⛔ G3: the clock is VERIFIED against the measured one before any batch is read.\n"
     "        clip_clock_stats[\"g3\"] = ds.assert_label_clock_true(\n"
     "            getattr(args, \"clip_clock_sidecar\", None), split=\"train\",\n"
     "            synthetic=bool(getattr(args, \"synth_episodes\", 0)))\n",
     "E8b G3 train call")
edit("            eval_clip_clock_stats = e_ds.enable_clip_clock(\n"
     "                getattr(args, \"clip_clock_sidecar\", None))\n",
     "            eval_clip_clock_stats = e_ds.enable_clip_clock(\n"
     "                getattr(args, \"clip_clock_sidecar\", None))\n"
     "            eval_clip_clock_stats[\"g3\"] = e_ds.assert_label_clock_true(\n"
     "                getattr(args, \"clip_clock_sidecar\", None), split=\"eval\",\n"
     "                synthetic=bool(getattr(args, \"synth_episodes\", 0)))\n",
     "E8c G3 eval call")

# ---- E9: module-level constants + the legacy as-trained resolver -------------------------- #
edit("def _pin_trainer_cfg(cfg: v3.RefCV3Config, args) -> v3.RefCV3Config:\n",
     "#: ⛔ G3 (SPEC_REFCV7 FIX-2): the largest admissible |t_trainer - t_true| of a label read, s.\n"
     "LABEL_CLOCK_TOL_S: float = 0.05\n"
     "#: ⛔ G3: the largest fraction of a split's clips that may lack a MEASURED clock (they are\n"
     "#: counted and stamped). 0.01 admits refcv6's train split (22 of 4,369 = 0.50 % uncovered,\n"
     "#: MEASURED `config.json.label_clock`); set 0.0 to require full sidecar coverage.\n"
     "LABEL_CLOCK_MAX_UNVERIFIED_FRAC: float = 0.01\n"
     "\n"
     "\n"
     "def trunk_equalize_rows_as_trained(config: dict) -> tuple[int, str]:\n"
     "    \"\"\"``(rows, why)`` -- the C26 rows a RECORDED run's trunk ACTUALLY zeroed.\n"
     "\n"
     "    ⛔⛔ FOR EVERY REBUILD OF A PRE-FIX CHECKPOINT. Before 2026-09-26 (FIX-3) the value was an\n"
     "    undeclared attribute: with ``--image-hw`` the rebuild dropped it and the trunk trained with\n"
     "    0 rows (refcv6-r101-s0: argv 43, trunk 0); without ``--image-hw`` it survived. A tree with\n"
     "    the fix rebuilds EVERY such argv with the rows argv names -- a model the weights never\n"
     "    were. An eval loader rebuilding a recorded run must override the trunk (not the lift) to\n"
     "    this value when ``config.json`` predates the stamp.\n"
     "\n"
     "    * ``seams.trunk_equalize_bottom_rows`` present -> that (a post-fix record states it);\n"
     "    * else argv ``--equalize-bottom-rows N > 0`` with ``--image-hw`` -> ``0`` (DROPPED);\n"
     "    * else argv ``N`` (no rebuild ran, the attribute reached the trunk) or ``0``.\n"
     "    \"\"\"\n"
     "    seams = config.get(\"seams\") if isinstance(config, dict) else None\n"
     "    if isinstance(seams, dict) and \"trunk_equalize_bottom_rows\" in seams:\n"
     "        return int(seams[\"trunk_equalize_bottom_rows\"]), \"stamped (post-fix record)\"\n"
     "    argv = list(config.get(\"argv\") or []) if isinstance(config, dict) else []\n"
     "    n = 0\n"
     "    if \"--equalize-bottom-rows\" in argv:\n"
     "        n = int(argv[argv.index(\"--equalize-bottom-rows\") + 1])\n"
     "    if n > 0 and \"--image-hw\" in argv:\n"
     "        return 0, (f\"DROPPED: pre-fix record, argv --equalize-bottom-rows {n} with --image-hw \"\n"
     "                   f\"(D-REFCV6-EQUALIZE-DROPPED) -- the trunk trained with 0 rows\")\n"
     "    return n, \"pre-fix record without --image-hw: the attribute reached the trunk\"\n"
     "\n"
     "\n"
     "def _pin_trainer_cfg(cfg: v3.RefCV3Config, args) -> v3.RefCV3Config:\n",
     "E9 constants + legacy resolver")

open(P, "w", encoding="utf-8", newline="").write(s)
print("applied:", ", ".join(EDITS))
