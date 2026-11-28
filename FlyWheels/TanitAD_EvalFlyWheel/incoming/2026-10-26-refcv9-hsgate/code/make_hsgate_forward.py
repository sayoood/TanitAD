"""make_hsgate_forward.py -- builds ship_r8_hsgate_forward.py from ship_r8_sub/ship_r8_4c_forward.py (READ-ONLY, pinned by the 4c manifest) by anchored
replacements, each asserted to match EXACTLY ONCE, and writes the unified diff beside it (ship_r8_hsgate_forward.DIFF.txt).  PREREG_HSGATE PHASE 2 device.
The 4c copy's arms / controls are untouched; ONE switch is added: --hsgate (mode h2cn): FACTORED_DEPLOY8 (Amendment 5, the factored branch's code) is built on
the SAME rows beside H2CN_DEPLOY8, the REGISTERED gate (HSGATE_FROZEN.json == REGISTERED_HSGATE.sha256) picks per window -> HSGATE8, plus the identity arms
HSGATE8_G0 (g := 0) / HSGATE8_G1 (g := 1) and the controls K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE, K_FEED.
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub/ship_r8_4c_forward.py")
DST = HERE / "ship_r8_hsgate_forward.py"
DIFF = HERE / "ship_r8_hsgate_forward.DIFF.txt"

R = []


def rep(old, new, count=1):
    R.append((old, new, count))


rep('"""ship_r8_4c_forward.py -- SHIP-R8 AMENDMENT 4c: a COPY of',
    '"""ship_r8_hsgate_forward.py -- PREREG_HSGATE (2331e1f3...) PHASE 2: a COPY of ship_r8_sub/ship_r8_4c_forward.py (left untouched; built by\n'
    'make_hsgate_forward.py, diff ship_r8_hsgate_forward.DIFF.txt) with ONE switch added, --hsgate (mode h2cn): FACTORED_DEPLOY8 (Amendment 5, the factored\n'
    'branch\'s code verbatim) is built on the SAME rows beside H2CN_DEPLOY8; the REGISTERED gate (HSGATE_FROZEN.json, sha256 == REGISTERED_HSGATE.sha256, checked\n'
    'BEFORE the first eval-side output) picks per window -> HSGATE8; identity arms HSGATE8_G0 (g := 0) / HSGATE8_G1 (g := 1); controls K_REG_HSGATE,\n'
    'K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED (H2CN / FACTORED positions bit-equal the LANDED packages), K_GATE_SURF, K_TRACE, K_FEED.  --hsgate refuses\n'
    'any --out-root outside C:/Users/Admin/refcv8_pf/refcv9_hsgate and F:/refcv9_hsgate_scratch.\n\n'
    'ship_r8_4c_forward.py -- SHIP-R8 AMENDMENT 4c: a COPY of')
rep('HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\nimport ship_r8_sub_common as C  # noqa: E402\n',
    'HERE = Path(__file__).resolve().parent\n'
    'SUBDIR = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")                # READ-ONLY: the copied device\'s helpers are imported from there\n'
    'sys.path.insert(0, str(SUBDIR))\nsys.path.insert(0, str(HERE))\nimport ship_r8_sub_common as C  # noqa: E402\nimport hsgate_common as HG  # noqa: E402\n')
rep('COMMON_SHA = C.sha256_file(HERE / "ship_r8_sub_common.py")\n',
    'COMMON_SHA = C.sha256_file(SUBDIR / "ship_r8_sub_common.py")\n'
    'HSG_COMMON_SHA = C.sha256_file(HERE / "hsgate_common.py")\n'
    'REG_HSG = HG.WORK / "REGISTERED_HSGATE.sha256"                     # written by the Master Mind, NEVER here\n'
    'HSG_FROZEN, HSG_FIT_INFO = HG.WORK / "HSGATE_FROZEN.json", HG.WORK / "HSGATE_FIT_INFO.json"\n')
rep('"HC": "H2CN_CAP8", "HL": "H2CNL_DEPLOY8", "HM": "H2CNL_MUT8"}',
    '"HC": "H2CN_CAP8", "HL": "H2CNL_DEPLOY8", "HM": "H2CNL_MUT8", "HS": "HSGATE8", "Z0": "HSGATE8_G0", "Z1": "HSGATE8_G1"}')
rep('def pf_lane():',
    'def bit_eq(a, b) -> bool:\n    a, b = np.asarray(a), np.asarray(b)\n    return bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())\n\n\ndef pf_lane():')
rep('            h2cnl: bool = False, work_root: Path | None = None) -> int:',
    '            h2cnl: bool = False, work_root: Path | None = None, hsgate: bool = False) -> int:')
rep('        seals = {**seals, "4c": HL.check_amendment_4c_seal()}              # AMENDMENT 4c: refuses unless the sealed text is the file (only when --h2cnl)\n',
    '        seals = {**seals, "4c": HL.check_amendment_4c_seal()}              # AMENDMENT 4c: refuses unless the sealed text is the file (only when --h2cnl)\n'
    '    if hsgate:\n        seals = {**seals, "PREREG_HSGATE": HG.check_prereg()}              # PREREG_HSGATE: refuses unless the sealed text is the file\n')
rep('("_h2cnl" if h2cnl else "")', '("_h2cnl" if h2cnl else "") + ("_hsgate" if hsgate else "")')
rep('        arm_order = ["H2CN_DEPLOY8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1"] + (["H2CN_CAP8"] if cap8 else []) + \\\n'
    '                    (["H2CNL_DEPLOY8", "H2CNL_MUT8"] if h2cnl else [])\n',
    '        hsg, landed = {}, {}\n'
    '        if hsgate:\n'
    '            # PREREG_HSGATE s1: the gate is frozen and its sha256 REGISTERED (REGISTERED_HSGATE.sha256, written by the MM) BEFORE any SCORE-side output\n'
    '            reg_g = C.read_seal(REG_HSG)\n'
    '            got_g = C.sha256_file(HSG_FROZEN) if HSG_FROZEN.exists() else ""\n'
    '            if len(reg_g) != 64 or len(got_g) != 64:\n'
    '                raise SystemExit(f"[hsgate-fwd] INCONCLUSIVE gate registration: registered \'{reg_g}\' file \'{got_g}\' -- refusing")\n'
    '            if reg_g != got_g:\n'
    '                raise SystemExit(f"[hsgate-fwd] the gate file hashes to {got_g}, the REGISTERED gate is {reg_g} -- refusing")\n'
    '            g_rec = {"arm": "HSGATE8", "registered_sha256": reg_g, "registration_file": str(REG_HSG), "registration_file_mtime_epoch": os.path.getmtime(REG_HSG),\n'
    '                     "gate_file": str(HSG_FROZEN), "gate_file_sha256": got_g, "gate_file_mtime_epoch": os.path.getmtime(HSG_FROZEN),\n'
    '                     "passed_epoch": time.time(), "passed_iso": time.strftime("%FT%T%z"),\n'
    '                     "rule": "PREREG_HSGATE s1: the gate is frozen and its sha256 registered by the Master Mind before any SCORE-side output; this record is written before the first one"}\n'
    '            (work / "REGISTRATION_GATE_HSGATE8.json").write_text(json.dumps(g_rec, indent=1), encoding="utf-8")\n'
    '            L2h, HLh = HG.load_l2_common()\n'
    '            hsg = {"gate": json.loads(HSG_FROZEN.read_text(encoding="utf-8")), "rec": g_rec, "L2": L2h, "HL": HLh}\n'
    '            for s in seeds:\n'
    '                fac[s] = load_factored_inputs(s, True)                          # Amendment 5\'s scored inputs (L1 os-only bank + SELFSHARP trace)\n'
    '                if s not in exl:\n'
    '                    exl[s] = load_eval_extras(tree, s)\n'
    '                landed[s] = {"H8": np.load(str(HG.LANDED_H8).format(seed=s), allow_pickle=False), "FX": np.load(str(HG.LANDED_FX).format(seed=s), allow_pickle=False),\n'
    '                             "files": {"H8": str(HG.LANDED_H8).format(seed=s), "FX": str(HG.LANDED_FX).format(seed=s)}}\n'
    '        arm_order = ["H2CN_DEPLOY8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1"] + (["H2CN_CAP8"] if cap8 else []) + \\\n'
    '                    (["H2CNL_DEPLOY8", "H2CNL_MUT8"] if h2cnl else []) + (["HSGATE8", "FACTORED_DEPLOY8", "HSGATE8_G0", "HSGATE8_G1"] if hsgate else [])\n')
rep('"H2CNL_DEPLOY8": "HL", "H2CNL_MUT8": "HM"}',
    '"H2CNL_DEPLOY8": "HL", "H2CNL_MUT8": "HM", "HSGATE8": "HS", "HSGATE8_G0": "Z0", "HSGATE8_G1": "Z1"}')
rep('    stash, kv, align, store, extra, compl = {}, {}, {}, {}, {}, {}\n',
    '    stash, kv, align, store, extra, compl = {}, {}, {}, {}, {}, {}\n    gate_keys = {}\n')
rep('                ex["h2cnl"] = hl\n            store_extra = ',
    '                ex["h2cnl"] = hl\n'
    '            if hsgate:\n'
    '                # PREREG_HSGATE s1: FACTORED_DEPLOY8 exactly as the factored branch builds it (Amendment 5), on the SAME rows as H2CN_DEPLOY8; HSGATE8 picks per window\n'
    '                b = fac[seed]\n'
    '                keys = list(zip(np.asarray(S["clip"], np.int64).tolist(), np.asarray(S["ws"], np.int64).tolist()))\n'
    '                jb = np.array([b["b1"]["idx"].get(k, -1) for k in keys])\n'
    '                jt = np.array([b["ts"]["idx"].get(k, -1) for k in keys])\n'
    '                if (jb < 0).any() or (jt < 0).any():\n'
    '                    raise SystemExit(f"[hsgate-fwd] {int((jb < 0).sum())} / {int((jt < 0).sum())} {surf} windows have no os-only bank / SELFSHARP trace entry")\n'
    '                P_os8 = np.ascontiguousarray(b["b1"]["plan8"][jb]).astype(np.float32)\n'
    '                P_ss8 = np.ascontiguousarray(b["ts"]["plan8"][jt]).astype(np.float32)\n'
    '                P_fx = compose(RT, P_os8, P_ss8)\n'
    '                st["in8"]["FX"] = P_fx\n'
    '                st["FX"] = (RS.retime_plan(P_fx, prior8, T8, table, cast=True), RS.retime_plan(P_fx, prior8, T8, table, cast=False))\n'
    '                sFX = RT.retime_progress(RT.polyline_geometry(P_fx)["L"][:, 1:], RT.prior_arc(prior8), g8)\n'
    '                e_ = exl[seed]\n'
    '                v0_ctx = ctx["v0"][rows]\n'
    '                rjh, jH = join_eval_extras(e_, S["clip"], S["ws"], v0_ctx, plan8, 0)\n'
    '                rjhm, _jhm = join_eval_extras(e_, S["clip"], S["ws"], v0_ctx, plan8, e_["stride"])\n'
    '                if not rjh["PASS"]:\n'
    '                    raise SystemExit(f"[hsgate-fwd] K-JOIN (eval extras) failed on {surf} seed {seed}: {rjh}")\n'
    '                Xg, parts = HG.features(hsg["L2"], hsg["HL"], AL, e_[HG.LOGIT_KEY][jH], e_[HG.V0_KEY][jH], st["FX"][0][:, HG.S6], st["H8"][0][:, HG.S6])\n'
    '                gw = np.asarray(HG.gate_decide(hsg["gate"], Xg), bool)\n'
    '                gm = gw[:, None, None]\n'
    '                st["HS"] = (np.where(gm, st["FX"][0], st["H8"][0]), np.where(gm, st["FX"][1], st["H8"][1]))\n'
    '                st["in8"]["HS"] = np.where(gm, P_fx, plan8)\n'
    '                st["s8"]["HS"] = np.where(gw[:, None], sFX, s8)\n'
    '                st["Z0"], st["Z1"] = st["H8"], st["FX"]                          # the identity arms: g := 0 / g := 1 through the SAME device\n'
    '                st["in8"]["Z1"] = P_fx\n'
    '                st["s8"]["Z0"] = s8\n'
    '                st["gate"] = gw\n'
    '                lk = {}\n'
    '                for kk in ("H8", "FX"):\n'
    '                    z_ = landed[seed][kk]\n'
    '                    lk[kk] = {"rows_aligned": bool(np.array_equal(z_[f"{surf}_clip"], np.asarray(S["clip"], np.int64)) and np.array_equal(z_[f"{surf}_ws"], np.asarray(S["ws"], np.int64))),\n'
    '                              "pos8_bit_equal": bit_eq(z_[f"{surf}_pos8"], st[kk][0]), "pos64_bit_equal": bit_eq(z_[f"{surf}_pos64"], st[kk][1][:, slots]),\n'
    '                              "plan8_bit_equal": bit_eq(z_[f"{surf}_plan8"], st["in8"].get(kk, plan8)), "prior8_bit_equal": bit_eq(z_[f"{surf}_prior8"], prior8)}\n'
    '                mh = np.asarray(S["mask"], bool)\n'
    '                ex["hsgate"] = {"join": rjh, "join_mutation_next_window": rjhm, "ws_stride": e_["stride"], "extras_file": e_["file"], "extras_file_sha256": e_["sha256"],\n'
    '                                "landed": lk, "n_windows": int(len(gw)), "n_compliant": int(mh.sum()), "n_F_all": int(gw.sum()), "n_F_compliant": int(gw[mh].sum()),\n'
    '                                "F_rate_all_windows": float(gw.mean()), "F_rate_compliant": (float(gw[mh].mean()) if mh.any() else None),\n'
    '                                "d_mean": float(parts["d"].mean()), "features_finite": bool(np.isfinite(Xg).all()),\n'
    '                                "windows_positions_differ_HSGATE8_vs_H2CN_DEPLOY8": int((st["HS"][0] != st["H8"][0]).any((1, 2)).sum()),\n'
    '                                "windows_positions_differ_HSGATE8_vs_FACTORED_DEPLOY8": int((st["HS"][0] != st["FX"][0]).any((1, 2)).sum())}\n'
    '                gate_keys[(seed, surf)] = (np.asarray(S["clip"], np.int64), np.asarray(S["ws"], np.int64), gw)\n'
    '            store_extra = ')
rep('("H8", "HN", "D8", "OS", "FS", "FX", "HC", "HL", "HM") if k2 in st]',
    '("H8", "HN", "D8", "OS", "FS", "FX", "HC", "HL", "HM", "HS") if k2 in st]')
rep('            if k in ("OS", "FS", "FX"):\n', '            if k in ("OS", "FS", "FX", "Z1"):\n')
rep('            if k in ("H8", "HN", "HC", "HL", "HM"):\n',
    '            if k == "HS":\n'
    '                return diag_progress(RT, st["in8"]["HS"][rows], st["HS"][1][rows], st["HS"][0][rows], st["s8"]["HS"][rows],\n'
    '                                     "HSGATE8: per window the chosen arm\'s positions against THAT arm\'s 8-point polyline (FACTORED: the composed plan; H2CN: the model plan)")\n'
    '            if k in ("H8", "HN", "HC", "HL", "HM", "Z0"):\n')
rep('            need += ["K_REG_H2CNL", "K_REG_H2CNL_MUT", "K_FREEZE_H2CNL", "K_JOIN_EVAL"]\n    else:\n',
    '            need += ["K_REG_H2CNL", "K_REG_H2CNL_MUT", "K_FREEZE_H2CNL", "K_JOIN_EVAL"]\n'
    '        if hsgate:\n'
    '            # PREREG_HSGATE s2.  In `need`: a failure WITHHOLDS (no FORWARD file, no verdict, no bars).\n'
    '            fe = reg_rec["first_eval_output_epoch"]\n'
    '            gr = hsg["rec"]\n'
    '            controls["K_REG_HSGATE"] = bool(len(gr["registered_sha256"]) == 64 and gr["registered_sha256"] == gr["gate_file_sha256"] == C.sha256_file(HSG_FROZEN)\n'
    '                                            and fe is not None and gr["passed_epoch"] < fe and gr["registration_file_mtime_epoch"] < fe)\n'
    '            controls["K_REG_HSGATE_detail"] = {**gr, "first_eval_output_epoch": fe, "first_eval_output_iso": time.strftime("%FT%T%z", time.localtime(fe or 0))}\n'
    '            fit = json.loads(HSG_FIT_INFO.read_text(encoding="utf-8")) if HSG_FIT_INFO.exists() else {}\n'
    '            fcs = fit.get("controls") or {}\n'
    '            controls["K_FIT_HSGATE"] = bool(fit.get("ALL_PASS") is True and fit.get("smoke_partial_trace") is False and fcs.get("K-RELOAD") is True\n'
    '                                            and (fit.get("gate_file") or {}).get("sha256") == gr["gate_file_sha256"])\n'
    '            controls["K_FIT_HSGATE_detail"] = {"fit_info": str(HSG_FIT_INFO), "fit_info_sha256": (C.sha256_file(HSG_FIT_INFO) if HSG_FIT_INFO.exists() else None),\n'
    '                                               "ALL_PASS": fit.get("ALL_PASS"), **{k_: (v_.get("PASS") if isinstance(v_, dict) else v_) for k_, v_ in fcs.items()\n'
    '                                                                                    if k_ in ("K-FRESH-SS", "K-JOIN", "K-OOF_PASS", "K-PATH", "K-PRIOR", "K-DETERM", "K-RELOAD")}}\n'
    '            mm0, mm1, n0, n1, pid = [], [], 0, 0, {}\n'
    '            for s in seeds:\n'
    '                for surf in SURFS:\n'
    '                    st_ = stash[(s, surf)]\n'
    '                    pid[f"{surf}|s{s}"] = {"G0_pos32": bit_eq(st_["Z0"][0], st_["H8"][0]), "G0_pos64": bit_eq(st_["Z0"][1], st_["H8"][1]),\n'
    '                                           "G1_pos32": bit_eq(st_["Z1"][0], st_["FX"][0]), "G1_pos64": bit_eq(st_["Z1"][1], st_["FX"][1])}\n'
    '                    for stn in SETS:\n'
    '                        A_ = res["seeds"][str(s)]["surfaces"][surf]["sets"][stn]["arms"]\n'
    '                        mm0 += [f"s{s}|{surf}|{stn}{m}" for m in D8.exact_equal(A_["HSGATE8_G0"], A_["H2CN_DEPLOY8"])]\n'
    '                        mm1 += [f"s{s}|{surf}|{stn}{m}" for m in D8.exact_equal(A_["HSGATE8_G1"], A_["FACTORED_DEPLOY8"])]\n'
    '                        n0 += D8.count_leaves(A_["HSGATE8_G0"])\n'
    '                        n1 += D8.count_leaves(A_["HSGATE8_G1"])\n'
    '            bars_id = {}\n'
    '            for stn in SETS:\n'
    '                for a_, b_ in (("HSGATE8_G0", "H2CN_DEPLOY8"), ("HSGATE8_G1", "FACTORED_DEPLOY8")):\n'
    '                    ka, kb = f"{stn}|{a_}", f"{stn}|{b_}"\n'
    '                    if ka in res.get("bars", {}) and kb in res.get("bars", {}):\n'
    '                        bars_id[ka] = not D8.exact_equal(res["bars"][ka], res["bars"][kb])\n'
    '            controls["K_ID0"] = bool(not mm0 and n0 > 0 and all(v["G0_pos32"] and v["G0_pos64"] for v in pid.values())\n'
    '                                     and all(v for k_, v in bars_id.items() if "G0" in k_))\n'
    '            controls["K_ID1"] = bool(not mm1 and n1 > 0 and all(v["G1_pos32"] and v["G1_pos64"] for v in pid.values())\n'
    '                                     and all(v for k_, v in bars_id.items() if "G1" in k_))\n'
    '            controls["K_ID_detail"] = {"positions": pid, "G0_cells_n_leaves": n0, "G0_cells_n_mismatches": len(mm0), "G0_first_mismatches": mm0[:10],\n'
    '                                       "G1_cells_n_leaves": n1, "G1_cells_n_mismatches": len(mm1), "G1_first_mismatches": mm1[:10], "bars_equal": bars_id}\n'
    '            hx = {k: v["hsgate"] for k, v in extra.items() if isinstance(k, str)}\n'
    '            kls = json.loads(HG.FRESH_KLABSRC.read_text(encoding="utf-8"))\n'
    '            controls["K_EVALFEAT"] = bool(kls.get("key") == HG.LOGIT_KEY == HL.LOGIT_KEY and (kls.get("K_LABSRC") or {}).get("pass") is True\n'
    '                                          and exl[0]["sha256"] == (kls.get("K_LABSRC") or {}).get("extras_sha256")\n'
    '                                          and all(Path(exl[s]["file"]).resolve() == Path(EVAL_EXTRAS.format(tree=tree, seed=s)).resolve() for s in seeds)\n'
    '                                          and len(hx) == 4 and all(v["join"]["PASS"] and not v["join_mutation_next_window"]["PASS"]\n'
    '                                                                   and v["join_mutation_next_window"]["red_by_crosscheck_alone"] and v["features_finite"] for v in hx.values()))\n'
    '            controls["K_EVALFEAT_detail"] = {"klabsrc_file": str(HG.FRESH_KLABSRC), "klabsrc_sha256": C.sha256_file(HG.FRESH_KLABSRC), "klabsrc_key": kls.get("key"),\n'
    '                                             "klabsrc_extras_sha256_s0": (kls.get("K_LABSRC") or {}).get("extras_sha256"),\n'
    '                                             "eval_extras": {str(s): {"file": exl[s]["file"], "sha256": exl[s]["sha256"]} for s in seeds},\n'
    '                                             "keys_used": [HG.LOGIT_KEY, HG.V0_KEY],\n'
    '                                             "joins": {k: {"join": v["join"], "mutation_next_window": v["join_mutation_next_window"]} for k, v in hx.items()}}\n'
    '            controls["K_LANDED"] = bool(len(hx) == 4 and all(all(all(x.values()) for x in v["landed"].values()) for v in hx.values()))\n'
    '            controls["K_LANDED_detail"] = {k: v["landed"] for k, v in hx.items()}\n'
    '            gs = {}\n'
    '            for s in seeds:\n'
    '                c2, w2, g2 = gate_keys[(s, "S2")]\n'
    '                c6, w6, g6 = gate_keys[(s, "S6")]\n'
    '                k2 = {(int(a__), int(b__)): bool(c__) for a__, b__, c__ in zip(c2, w2, g2)}\n'
    '                miss = sum((int(a__), int(b__)) not in k2 for a__, b__ in zip(c6, w6))\n'
    '                dif = sum(k2.get((int(a__), int(b__))) != bool(c__) for a__, b__, c__ in zip(c6, w6, g6))\n'
    '                gs[str(s)] = {"n_S6": int(len(g6)), "S6_rows_missing_in_S2": int(miss), "decisions_differ": int(dif)}\n'
    '            controls["K_GATE_SURF"] = bool(all(v["S6_rows_missing_in_S2"] == 0 and v["decisions_differ"] == 0 for v in gs.values()))\n'
    '            controls["K_GATE_SURF_detail"] = gs\n'
    '            trh = {str(s): trace_checks(fac[s]["rec"], True) for s in seeds}\n'
    '            controls["K_TRACE"] = bool(all(all(v.values()) for v in trh.values()))\n'
    '            controls["K_TRACE_detail"] = {"checks": trh, "inputs": {str(s): fac[s]["rec"] for s in seeds}}\n'
    '            fdh = {str(s): feed_checks(s, True) for s in seeds}\n'
    '            controls["K_FEED_perframe_full"] = bool(all(r["perframe_full"] for v in fdh.values() for r in v.values()))\n'
    '            controls["K_FEED_detail"] = fdh\n'
    '            need += ["K_REG_HSGATE", "K_FIT_HSGATE", "K_ID0", "K_ID1", "K_EVALFEAT", "K_LANDED", "K_GATE_SURF", "K_TRACE", "K_FEED_perframe_full"]\n'
    '    else:\n')
rep('"tool": "ship_r8_4c_forward.py"', '"tool": "ship_r8_hsgate_forward.py"', count=2)
rep('                          f"SHIP-R8 AMENDMENT 4c: arm positions replaced; scored arm H2CNL_DEPLOY8, mutation arm H2CNL_MUT8 beside (H2CN_DEPLOY8 re-run beside them for the reproduction)" if h2cnl else\n',
    '                          f"SHIP-R8 AMENDMENT 4c: arm positions replaced; scored arm H2CNL_DEPLOY8, mutation arm H2CNL_MUT8 beside (H2CN_DEPLOY8 re-run beside them for the reproduction)" if h2cnl else\n'
    '                          f"PREREG_HSGATE PHASE 2: scored arm HSGATE8 (per-window gate between FACTORED_DEPLOY8 and H2CN_DEPLOY8); identity arms HSGATE8_G0 / HSGATE8_G1 beside" if hsgate else\n')
rep('        res["h2cnl_x_extra"] = HL.X_EXTRA\n',
    '        res["h2cnl_x_extra"] = HL.X_EXTRA\n'
    '    if hsgate:\n'
    '        res["hsgate"] = {"gate_file": str(HSG_FROZEN), "gate_sha256": hsg["rec"]["gate_file_sha256"], "registered": hsg["rec"]["registered_sha256"],\n'
    '                         "registration_file": str(REG_HSG), "prereg_sha256": seals["PREREG_HSGATE"]["sha256"], "hsgate_common_sha256": HSG_COMMON_SHA,\n'
    '                         "F_rate": {k: {kk: v["hsgate"][kk] for kk in ("n_windows", "n_compliant", "n_F_all", "n_F_compliant", "F_rate_all_windows", "F_rate_compliant")}\n'
    '                                    for k, v in extra.items() if isinstance(k, str)},\n'
    '                         "landed_inputs": {str(s): {k: {"file": f, "sha256": C.sha256_file(f)} for k, f in landed[s]["files"].items()} for s in seeds}}\n')
rep('"FACTORED_DEPLOY8", "OSONLY_DEPLOY8", "H2CN_CAP8", "H2CNL_DEPLOY8", "H2CNL_MUT8")]',
    '"FACTORED_DEPLOY8", "OSONLY_DEPLOY8", "H2CN_CAP8", "H2CNL_DEPLOY8", "H2CNL_MUT8", "HSGATE8")]')
rep('            if arm in ("H2CNL_DEPLOY8", "H2CNL_MUT8"):\n',
    '            if arm == "HSGATE8":\n'
    '                meta["hsgate_gate_sha256"] = hsg["rec"]["gate_file_sha256"]\n'
    '                meta["note_gate"] = "<S>_gate: the registered gate\'s per-window decision (True = FACTORED_DEPLOY8\'s positions, False = H2CN_DEPLOY8\'s)"\n'
    '                for surf_ in SURFS:\n'
    '                    arrs[f"{surf_}_gate"] = np.asarray(stash[(seed, surf_)]["gate"], bool)\n'
    '            if arm in ("H2CNL_DEPLOY8", "H2CNL_MUT8"):\n')
rep('    ap.add_argument("--work-root", default=None, help="where the transient RT.forward work dir lives (default: --out-root); removed at the end")\n',
    '    ap.add_argument("--work-root", default=None, help="where the transient RT.forward work dir lives (default: --out-root); removed at the end")\n'
    '    ap.add_argument("--hsgate", action="store_true", help="h2cn: PREREG_HSGATE -- also build FACTORED_DEPLOY8 + HSGATE8 (+ G0 / G1 identity arms) + their controls")\n')
rep('    return forward(a.mode, a.tree, seeds, a.scored, a.mutate, Path(a.out_root), Path(a.registered_stage), a.cap8, a.h2cnl, Path(a.work_root) if a.work_root else None)\n',
    '    if a.hsgate:\n'
    '        if a.mode != "h2cn" or a.cap8 or a.h2cnl or a.mutate:\n'
    '            raise SystemExit("--hsgate applies to --mode h2cn only, without --cap8 / --h2cnl / --mutate")\n'
    '        okroots = (HG.WORK.resolve(), HG.SCRATCH.resolve())\n'
    '        for p_ in [Path(a.out_root)] + ([Path(a.work_root)] if a.work_root else []):\n'
    '            rp = p_.resolve()\n'
    '            if not any(rp == r_ or r_ in rp.parents for r_ in okroots):\n'
    '                raise SystemExit(f"[hsgate-fwd] REFUSING: {p_} is outside {okroots}")\n'
    '    else:\n'
    '        raise SystemExit("[hsgate-fwd] this copy runs ONLY with --hsgate (the 4c reads live in ship_r8_sub/ and are never re-run from here)")\n'
    '    return forward(a.mode, a.tree, seeds, a.scored, a.mutate, Path(a.out_root), Path(a.registered_stage), a.cap8, a.h2cnl, Path(a.work_root) if a.work_root else None,\n'
    '                   a.hsgate)\n')


def main():
    src = SRC.read_text(encoding="utf-8")
    out = src
    for old, new, count in R:
        n = out.count(old)
        if n != count:
            raise SystemExit(f"anchor matched {n} times (want {count}): {old[:90]!r}")
        out = out.replace(old, new)
    DST.write_text(out, encoding="utf-8", newline="\n")
    d = difflib.unified_diff(src.splitlines(True), out.splitlines(True), fromfile="ship_r8_sub/ship_r8_4c_forward.py", tofile="ship_r8_hsgate_forward.py")
    DIFF.write_text("".join(d), encoding="utf-8", newline="\n")
    print(f"wrote {DST} ({len(out)} bytes) + {DIFF}; {len(R)} anchored replacements")


if __name__ == "__main__":
    main()
