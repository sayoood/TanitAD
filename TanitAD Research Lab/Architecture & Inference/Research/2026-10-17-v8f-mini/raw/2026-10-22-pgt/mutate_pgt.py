"""Mutation proof for PREREG_V8F_PGT's instruments: every mutation must turn stack/tests/test_v8f_pgt.py RED; the unmutated tree must be
GREEN before and after, and every mutated file is restored byte-identical (in a `finally`, then verified). Deletes nothing.

  python mutate_pgt.py <repo root> <python exe> [--set reach|reader|queue|queue-stages|chain|all] [--check-anchors]

  reach   the lever's path: the masks in v8f_mini._forward, the schedule, loss/forward, the trainer's config + loop + gate, built_levers
  reader  code/v8f_pgt_read.py: the verdict logic, the PRIMARY / M-LEAK / M-DK bars, the refusals
  queue   code/v8f_mini_box2_queue.cmd (gatepgt / trainpgt / evalpgt) and code/box2_ops/chain_v8f_pgt.sh
Anchors must occur EXACTLY ONCE (asserted); a mutant that is semantically equivalent to the original is not listed."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
WHICH = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "all"
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
CODE = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code"
READ, QUEUE, CHAIN = CODE / "v8f_pgt_read.py", CODE / "v8f_mini_box2_queue.cmd", CODE / "box2_ops" / "chain_v8f_pgt.sh"
TEST = "stack/tests/test_v8f_pgt.py"

REACH = [
    # ---- the masks (v8f_mini._forward)
    (MODEL, "S->T/O: a floor hard-coded at M2b's 0.2 survives p_gt = 0",
     "        if use_labels and p_gt > 0:\n            use = (torch.rand(B, generator=gen, device=dev) < p_gt)[:, None]",
     "        if use_labels:\n            use = (torch.rand(B, generator=gen, device=dev) < max(p_gt, 0.2))[:, None]"),
    (MODEL, "S->T/O: the draw ignores rc_ok (a GT checkpoint where it is invalid)", ' & b["lab_s"]["rc_ok"] & b["lab_s"]["has"][:, None]\n',
     ' & b["lab_s"]["has"][:, None]\n'),
    (MODEL, "S->T/O: never reads the GT checkpoint", '            ck_src = torch.where(use[..., None], b["lab_s"]["rc"], ck_s)\n',
     '            ck_src = ck_s\n'),
    (MODEL, "T->O: a floor hard-coded at M2b's 0.2 survives p_gt = 0",
     "        if use_labels and p_gt > 0:\n            use_t = (torch.rand(B, generator=gen, device=dev) < p_gt) & b[\"lab_t\"][\"has\"]",
     "        if use_labels:\n            use_t = (torch.rand(B, generator=gen, device=dev) < max(p_gt, 0.2)) & b[\"lab_t\"][\"has\"]"),
    (MODEL, "T->O: the draw ignores the label's `has` (GT code on an unlabelled window)",
     '< p_gt) & b["lab_t"]["has"]\n', '< p_gt)\n'),
    (MODEL, "T->O: never reads the GT code", 'code = torch.where(use_t[:, None], self._code_gt(b["lab_t"]), code)\n',
     'code = torch.where(use_t[:, None], code, code)\n'),
    (MODEL, "T->O: the draw ignores the generator", "            use_t = (torch.rand(B, generator=gen, device=dev) < p_gt)",
     "            use_t = (torch.rand(B, device=dev) < p_gt)"),
    (MODEL, "T->O: half the rate", "            use_t = (torch.rand(B, generator=gen, device=dev) < p_gt)",
     "            use_t = (torch.rand(B, generator=gen, device=dev) < 0.5 * p_gt)"),
    (MODEL, "forward ignores the p_gt it is given", "return self._forward(batch, self.p_gt if p_gt is None else p_gt, True, gen)",
     "return self._forward(batch, self.p_gt, True, gen)"),
    (MODEL, "loss drops the p_gt it is given", "        out = self.forward(batch, p_gt, gen)\n", "        out = self.forward(batch, None, gen)\n"),
    (MODEL, "built_levers misreports p_gt_start", '"p_gt_start": cfg.p_gt_start,', '"p_gt_start": 1.0,'),
    (MODEL, "built_levers misreports p_gt_end", '"p_gt_end": cfg.p_gt_end,', '"p_gt_end": 0.2,'),
    # ---- the schedule
    (MODEL, "the schedule does not decay", "    return float(p0 + (p1 - p0) * min(max(step / max(total, 1), 0.0), 1.0))",
     "    return float(p0)"),
    (MODEL, "the schedule runs backwards", "    return float(p0 + (p1 - p0) * min(max(step / max(total, 1), 0.0), 1.0))",
     "    return float(p1 + (p0 - p1) * min(max(step / max(total, 1), 0.0), 1.0))"),
    (MODEL, "the schedule is not clamped at the end value", "    return float(p0 + (p1 - p0) * min(max(step / max(total, 1), 0.0), 1.0))",
     "    return float(p0 + (p1 - p0) * max(step / max(total, 1), 0.0))"),
    # ---- the trainer: argv -> cfg -> loop -> masks, and the gate
    (TRAIN, "make_config drops --p-gt-start", "p_gt_start=a.p_gt_start, p_gt_end=a.p_gt_end,", "p_gt_start=1.0, p_gt_end=a.p_gt_end,"),
    (TRAIN, "make_config drops --p-gt-end", "p_gt_start=a.p_gt_start, p_gt_end=a.p_gt_end,", "p_gt_start=a.p_gt_start, p_gt_end=0.2,"),
    (TRAIN, "the loop swaps the schedule's endpoints", "p_gt = M.p_gt_schedule(step, total, cfg.p_gt_start, cfg.p_gt_end)",
     "p_gt = M.p_gt_schedule(step, total, cfg.p_gt_end, cfg.p_gt_start)"),
    (TRAIN, "the loop does not hand p_gt to the loss", "out = model.loss(b, p_gt=p_gt, gen=None)", "out = model.loss(b, gen=None)"),
    (TRAIN, "the gate does not compare p_gt_start / p_gt_end", 'for f in ("hist", "hmax", "d", "n_q", "nb", "p_gt_start", "p_gt_end", "w_ckpt",',
     'for f in ("hist", "hmax", "d", "n_q", "nb", "w_ckpt",'),
    (TRAIN, "M2b's default floor silently moves to 0.0", '    ap.add_argument("--p-gt-end", type=float, default=0.2)\n',
     '    ap.add_argument("--p-gt-end", type=float, default=0.0)\n'),
]
R_ = READ
READER = [
    # ---- the decision table (stage logic)
    (R_, "s0: CONTINUE and STOP swapped", '("CONTINUE" if go else "STOP")', '("STOP" if go else "CONTINUE")'),
    (R_, "final: SUCCESS needs only ONE passing seed", 'ok = all(rows[f"s{s}"]["judge"]["passed"] for s in seeds)', 'ok = any(rows[f"s{s}"]["judge"]["passed"] for s in seeds)'),
    (R_, "mechanism read as ANY seed instead of every seed", 'mech_all = all(rows[f"s{s}"]["judge"]["mechanism"] for s in seeds)',
     'mech_all = any(rows[f"s{s}"]["judge"]["mechanism"] for s in seeds)'),
    (R_, "final: the two FAIL labels swapped", '("SUCCESS" if ok else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM")), mech_all',
     '("SUCCESS" if ok else ("FAIL-NO-MECHANISM" if mech_all else "FAIL-MECHANISM-OK")), mech_all'),
    (R_, "s0: the two FAIL labels swapped", '(None if go else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM"))',
     '(None if go else ("FAIL-NO-MECHANISM" if mech_all else "FAIL-MECHANISM-OK"))'),
    (R_, "final: SUCCESS declared whatever the seeds read", 'return None, ("SUCCESS" if ok else', 'return None, ("SUCCESS" if True else'),
    # ---- the committed bars
    (R_, "PRIMARY bar is F, not 2F", 'abs(p["point"]) > 2 * F)', 'abs(p["point"]) > F)'),
    (R_, "PRIMARY bar has no size floor", 'abs(p["point"]) > 2 * F)', 'abs(p["point"]) > 0)'),
    (R_, "PRIMARY bar is 3F", 'abs(p["point"]) > 2 * F)', 'abs(p["point"]) > 3 * F)'),
    (R_, "PRIMARY does not need a separated interval", 'passed = bool(_is_neg_sep(p) and abs(p["point"]) > 2 * F)', 'passed = bool(p["point"] < 0 and abs(p["point"]) > 2 * F)'),
    (R_, "PRIMARY passes a WORSE arm (sign)", 'return bool(r["point"] < 0 and r["separated"])', 'return bool(r["point"] > 0 and r["separated"])'),
    (R_, "separated OR improving (not AND)", 'return bool(r["point"] < 0 and r["separated"])', 'return bool(r["point"] < 0 or r["separated"])'),
    (R_, "size bar of M-LEAK / M-DK is the full reference, not half", "HALF = 0.5", "HALF = 1.0"),
    (R_, "size bar of M-LEAK / M-DK is 0.6 of the reference", "HALF = 0.5", "HALF = 0.6"),
    (R_, "size bar of M-LEAK / M-DK is 0.4 of the reference", "HALF = 0.5", "HALF = 0.4"),
    (R_, "size bar of M-LEAK / M-DK is gone", "HALF = 0.5", "HALF = 0.0"),
    (R_, "mechanism is EITHER symptom", 'out["mechanism"] = bool(out["mleak"]["met"] and out["mdk"]["met"])', 'out["mechanism"] = bool(out["mleak"]["met"] or out["mdk"]["met"])'),
    (R_, "a non-positive reference symptom is silently scored on its size", "        if not ref > 0:\n", "        if False:\n"),
    (R_, "the judge reads seed 0's reference at every seed", 'row["judge"] = judge(row, base[s], F)', 'row["judge"] = judge(row, base[0], F)'),
    (R_, "F is not the banked M2b seed floor", 'F = abs(float(Zb_all[(ARM_B, 0)]["ade6"].mean()) - float(Zb_all[(ARM_B, 1)]["ade6"].mean()))',
     'F = abs(float(Zp[(ARM_B, 0)]["ade6"].mean()) - float(Zp[(ARM_P, 0)]["ade6"].mean()))'),
    (R_, "stage s0 reads both seeds", 'seeds = (0,) if a.stage == "s0" else (0, 1)', 'seeds = (0, 1)'),
    # ---- the contrasts
    (R_, "PRIMARY contrast sign flipped", '"primary": paired(pf, A["ade6"], Bm["ade6"], eid, n_boot)', '"primary": paired(pf, Bm["ade6"], A["ade6"], eid, n_boot)'),
    (R_, "M-LEAK contrast sign flipped", 'paired(pf, lam_a[lead], lam_b[lead], eid[lead], n_boot)', 'paired(pf, lam_b[lead], lam_a[lead], eid[lead], n_boot)'),
    (R_, "M-LEAK reads ALL windows, not the LEAD windows", 'paired(pf, lam_a[lead], lam_b[lead], eid[lead], n_boot)', 'paired(pf, lam_a, lam_b, eid, n_boot)'),
    (R_, "M-LEAK reference reads ALL windows", '"lambda_m2b_m": float(lam[lead].mean())', '"lambda_m2b_m": float(lam.mean())'),
    (R_, "LAMBDA is read at 4 s, not 2 s", "IDX_2S = 9 ", "IDX_2S = 19 "),
    (R_, "LAMBDA is the cross-track, not the along-track", 'return v["lat_pred"][:, IDX_2S, 0] - v["orc_pred"][:, IDX_2S, 0]', 'return v["lat_pred"][:, IDX_2S, 1] - v["orc_pred"][:, IDX_2S, 1]'),
    (R_, "LAMBDA sign flipped (oracle minus all-latcons)", 'return v["lat_pred"][:, IDX_2S, 0] - v["orc_pred"][:, IDX_2S, 0]', 'return v["orc_pred"][:, IDX_2S, 0] - v["lat_pred"][:, IDX_2S, 0]'),
    (R_, "LAMBDA reference CI built from the wrong arrays", 'paired(pf, Zm2b["lat_pred"][lead][:, IDX_2S, 0], Zm2b["orc_pred"][lead][:, IDX_2S, 0], eid[lead], n_boot)',
     'paired(pf, Zm2b["lat_pred"][lead][:, IDX_2S, 0], Zm2b["lat_pred"][lead][:, IDX_2S, 0], eid[lead], n_boot)'),
    (R_, "S5 threshold is 10 s, not 5 s", "TTC_THR_S = 5.0", "TTC_THR_S = 10.0"),
    (R_, "S5 threshold is 2 s, not 5 s", "TTC_THR_S = 5.0", "TTC_THR_S = 2.0"),
    (R_, "S5 contrast sign flipped", "paired(pf, (ta[m] < TTC_THR_S).astype(np.float64), (tb[m] < TTC_THR_S).astype(np.float64), eid[m], n_boot)",
     "paired(pf, (tb[m] < TTC_THR_S).astype(np.float64), (ta[m] < TTC_THR_S).astype(np.float64), eid[m], n_boot)"),
    (R_, "S5 reference is M2b against itself, not against P10", 'ex = s5_pair(pf, pw, lead, eid, Zm2b["dk"], "P10", n_boot)', 'ex = s5_pair(pf, pw, lead, eid, Zm2b["dk"], Zm2b["dk"], n_boot)'),
    (R_, "S5 reference sign flipped (P10 minus M2b)", 'ex = s5_pair(pf, pw, lead, eid, Zm2b["dk"], "P10", n_boot)', 'ex = s5_pair(pf, pw, lead, eid, "P10", Zm2b["dk"], n_boot)'),
    (R_, "S5 keeps the windows whose min-TTC is undefined (scored not at risk)", "    m = lead & np.isfinite(ta) & np.isfinite(tb)\n", "    m = lead\n"),
    (R_, "the companion S5 reads ALL windows, not the LEAD windows", "alt_all_lead=paired(pf, (ta[lead] < TTC_THR_S).astype(np.float64), (tb[lead] < TTC_THR_S).astype(np.float64), eid[lead], n_boot))",
     "alt_all_lead=paired(pf, (ta < TTC_THR_S).astype(np.float64), (tb < TTC_THR_S).astype(np.float64), eid, n_boot))"),
    (R_, "the companion S5 sign flipped", "alt_all_lead=paired(pf, (ta[lead] < TTC_THR_S).astype(np.float64), (tb[lead] < TTC_THR_S).astype(np.float64), eid[lead], n_boot))",
     "alt_all_lead=paired(pf, (tb[lead] < TTC_THR_S).astype(np.float64), (ta[lead] < TTC_THR_S).astype(np.float64), eid[lead], n_boot))"),
    (R_, "the dropped-window count is not reported", "n_dropped_no_lead_in_path=n_lead - int(m.sum())", "n_dropped_no_lead_in_path=0"),
    (R_, "the identity control does not cover the companion S5", ', "mdk_alt": r["mdk"]["delta"]["alt_all_lead"]}', '}'),
    # ---- the refusals of load_runs
    (R_, "refusal off: runs not on one clip list", 'if len({d["train_sha256"] for d in sm.values()}) != 1:', 'if False:'),
    (R_, "refusal off: the RA-1 holdout not withheld", 'if {d.get("excluded_sha256") for d in sm.values()} != {HOLDOUT_SHA}:', 'if False:'),
    (R_, "refusal off: n_clips", "if {d[\"n_clips\"] for d in sm.values()} != {N_TRAIN}:", "if False:"),
    (R_, "refusal off: a run is not done", 'if not all(d.get("done") is True for d in sm.values()):', 'if False:'),
    (R_, "refusal off: different step counts", 'if len({d["steps"] for d in sm.values()}) != 1:', 'if False:'),
    (R_, "refusal off: summary says another arm / seed", 'if d["arm"] != "M2b" or d["seed"] != s:', 'if False:'),
    (R_, "refusal off: the launch gate is not PASS", 'if sc.get("declared_vs_built") != "PASS":', 'if False:'),
    (R_, "refusal off: argv --arm is not M2b", 'if fl.get("--arm") != "M2b":', 'if False:'),
    (R_, "refusal off: M2b-pgt argv does not carry the dose", "                if not _same_float(fl.get(flag), want):\n", "                if False:\n"),
    (R_, "refusal off: built p_gt differs from the dose", "                elif not (_same_float(b0, PGT_START) and _same_float(b1, PGT_END)):\n", "                elif False:\n"),
    (R_, "refusal off: built p_gt unrecorded", "                if b0 is None or b1 is None:\n                    bad.append(f\"{arm}_s{s}: {w}.json levers record no built",
     "                if False:\n                    bad.append(f\"{arm}_s{s}: {w}.json levers record no built"),
    (R_, "built p_gt checked in summary.json only (selfcheck.json ignored)", "            for w, (b0, b1) in built.items():\n                if b0 is None",
     "            for w, (b0, b1) in [('summary', built['summary'])]:\n                if b0 is None"),
    (R_, "built p_gt checked in selfcheck.json only (summary.json ignored)", "            for w, (b0, b1) in built.items():\n                if b0 is None",
     "            for w, (b0, b1) in [('selfcheck', built['selfcheck'])]:\n                if b0 is None"),
    (R_, "refusal off: a second lever differs between the arms", "                if fp != fm:\n", "                if False:\n"),
    (R_, "the M2b-pgt train_log is optional", 'd["steps"], PGT_START, PGT_END, True)', 'd["steps"], PGT_START, PGT_END, False)'),
    (R_, "train_log tolerance is 1.0", "> 1e-9]", "> 1.0]"),
    (R_, "refusal off: an empty train_log", "    if not recs:\n", "    if False:\n"),
    (R_, "refusal off: an M2b run carries a lever flag", "                if flag in fl:\n", "                if False:\n"),
    (R_, "refusal off: an M2b run built with another schedule", "if (b0 is not None and not _same_float(b0, M2B_START)) or (b1 is not None and not _same_float(b1, M2B_END)):", "if False:"),
    # ---- the refusals of the evals and the world
    (R_, "refusal off: an eval on another grid (load_arm)", '        if not np.array_equal(np.asarray(zz["eid"]), eid):\n', "        if False:\n"),
    (R_, "refusal off: an arm on another grid (load_all)", "            elif not np.array_equal(Z[(arm, s)][\"eid\"], eid):\n", "            elif False:\n"),
    (R_, "refusal off: absmask eval is not an absmask run", '    if _mode(p["absj"]) != "absmask":\n', "    if False:\n"),
    (R_, "refusal off: an oracle eval is not an oracle run", '        if os.path.exists(pj) and _mode(pj) != "oracle":\n', "        if False:\n"),
    (R_, "refusal off: no tac_code on the absmask eval", '    if "tac_code" not in za.files:\n', "    if False:\n"),
    (R_, "the old-format GT oracle does not fall back to all-goal", '    if "tac_code" not in zo.files:                    #', '    if False:                    #'),
    (R_, "refusal off: no pred / ade6 in an eval", "            if k not in zz.files:\n", "            if False:\n"),
    (R_, "check_gt_codes: the GT constraint blocks are not compared across arms", "    WC.present_error({k: v for k, v in Z.items()}, seeds)\n", "    pass\n"),
    (R_, "check_gt_codes: the all-latcons oracle's GT outside latcons is unchecked", '        if not np.array_equal(v["lat_code"][v["has"]][:, keep]', '        if False and not np.array_equal(v["lat_code"][v["has"]][:, keep]'),
    (R_, "refusal off: the rebuilt eval grid is not the evals' grid", '    if not np.array_equal(np.asarray(G["eid"]), np.asarray(eid)):\n', "    if False:\n"),
    (R_, "refusal off: the lead block has K < 10", '    if blk10["_k"] < 10:\n', "    if False:\n"),
    # ---- the controls and the constants
    (R_, "known-value control off: LAMBDA / S5 excess", "            if abs(got - KNOWN[key][s]) > KNOWN_TOL[key]:\n", "            if False:\n"),
    (R_, "known-value control off: F", '    if abs(F - KNOWN["F"]) > KNOWN_TOL["F"]:\n', "    if False:\n"),
    (R_, "known-value tolerance is 1.0", 'KNOWN_TOL = {"lambda": 6e-4, "excess": 6e-5, "F": 6e-5}', 'KNOWN_TOL = {"lambda": 1.0, "excess": 1.0, "F": 1.0}'),
    (R_, "identity control off", '            if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):\n', "            if False:\n"),
    (R_, "identity control skips the bounds", 'if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and v["point"] == 0.0):', 'if not (v["delta"] == 0.0 and v["point"] == 0.0):'),
    (R_, "the committed dose in the reader is 0.5 -> 0.2", "PGT_START, PGT_END = 0.5, 0.0 ", "PGT_START, PGT_END = 0.5, 0.2 "),
    (R_, "the committed dose in the reader is 1.0 -> 0.0", "PGT_START, PGT_END = 0.5, 0.0 ", "PGT_START, PGT_END = 1.0, 0.0 "),
    (R_, "the reader's M2b dose is 1.0 -> 0.0", "M2B_START, M2B_END = 1.0, 0.2 ", "M2B_START, M2B_END = 1.0, 0.0 "),
    (R_, "the reader's schedule never decays", "    return float(p0 + (p1 - p0) * min(max(step / max(total, 1), 0.0), 1.0))", "    return float(p0)"),
    (R_, "the reader's schedule is not clamped", "    return float(p0 + (p1 - p0) * min(max(step / max(total, 1), 0.0), 1.0))", "    return float(p0 + (p1 - p0) * max(step / max(total, 1), 0.0))"),
    (R_, "the reader's holdout sha is another", 'HOLDOUT_SHA = "f6a1dee0', 'HOLDOUT_SHA = "f6a1dee1'),
    (R_, "the reader's clip count is another", "N_TRAIN = 4512", "N_TRAIN = 4500"),
    (R_, "the reader's banked F is another", '"F": 0.1207}', '"F": 0.1307}'),
    (R_, "the reader's banked M2b leak is another", '"lambda": {0: 0.2678, 1: 0.1667}', '"lambda": {0: 0.2778, 1: 0.1667}'),
    (R_, "the reader's banked S5 excess is another", '"excess": {0: 0.0442, 1: 0.0250}', '"excess": {0: 0.0442, 1: 0.0350}'),
    (R_, "identity control ignores delta", 'if not (v["delta"] == 0.0 and v["lo"] == 0.0', 'if not (v["lo"] == 0.0'),
    (R_, "identity control ignores the lower bound", 'v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"]', 'v["delta"] == 0.0 and v["hi"]'),
    (R_, "identity control ignores the point estimate", 'and v["hi"] == 0.0 and v["point"] == 0.0):', 'and v["hi"] == 0.0):'),
    (R_, "check-inputs does not compare the banked GT codes", "    check_gt_codes(Zb, (0, 1), WC)\n", "    pass\n"),
]
Q_, C_ = QUEUE, CHAIN
QUEUEM = [
    # ---- queue: gatepgt
    (Q_, "gatepgt drops the whole lever", '--arm M2b --p-gt-start %PGTS% --p-gt-end %PGTE% --out "%RUNSM%\\gate_M2b-pgt"', '--arm M2b --out "%RUNSM%\\gate_M2b-pgt"'),
    (Q_, "gatepgt drops --p-gt-end", '--arm M2b --p-gt-start %PGTS% --p-gt-end %PGTE% --out "%RUNSM%\\gate_M2b-pgt"', '--arm M2b --p-gt-start %PGTS% --out "%RUNSM%\\gate_M2b-pgt"'),
    (Q_, "gatepgt hard-codes the end at M2b's 0.2", '--arm M2b --p-gt-start %PGTS% --p-gt-end %PGTE% --out "%RUNSM%\\gate_M2b-pgt"', '--arm M2b --p-gt-start %PGTS% --p-gt-end 0.2 --out "%RUNSM%\\gate_M2b-pgt"'),
    (Q_, "gatepgt writes into M2b's own run directory", '--out "%RUNSM%\\gate_M2b-pgt" %A% --selfcheck-only"', '--out "%RUNSM%\\M2b_s0" %A% --selfcheck-only"'),
    (Q_, "gatepgt is not a selfcheck-only run", '%A% --selfcheck-only"\nif defined V8F_DRYRUN (echo %GP%', '%A%"\nif defined V8F_DRYRUN (echo %GP%'),
    (Q_, "gatepgt runs a different string than it prints", '%GP% > "%RUNSM%\\gate_M2b-pgt.log" 2>&1',
     '"%PY%" "%TRAIN%" --arm M2b --out "%RUNSM%\\gate_M2b-pgt" %A% --selfcheck-only > "%RUNSM%\\gate_M2b-pgt.log" 2>&1'),
    # ---- queue: trainpgt
    (Q_, "trainpgt drops --p-gt-start", '--arm M2b --p-gt-start %PGTS% --p-gt-end %PGTE% --seed %~2', '--arm M2b --p-gt-end %PGTE% --seed %~2'),
    (Q_, "trainpgt drops --p-gt-end", '--arm M2b --p-gt-start %PGTS% --p-gt-end %PGTE% --seed %~2', '--arm M2b --p-gt-start %PGTS% --seed %~2'),
    (Q_, "trainpgt hard-codes the end at M2b's 0.2", '--p-gt-end %PGTE% --seed %~2', '--p-gt-end 0.2 --seed %~2'),
    (Q_, "trainpgt hard-codes the start at 0.5", '--p-gt-start %PGTS% --p-gt-end %PGTE% --seed %~2', '--p-gt-start 0.5 --p-gt-end %PGTE% --seed %~2'),
    (Q_, "trainpgt swaps the endpoints", '--p-gt-start %PGTS% --p-gt-end %PGTE% --seed %~2', '--p-gt-start %PGTE% --p-gt-end %PGTS% --seed %~2'),
    (Q_, "trainpgt drops the grid", '--seed %~2 --grid pool4x10 --out "%OUT%" %A%"\nif defined V8F_DRYRUN (echo %TP%', '--seed %~2 --out "%OUT%" %A%"\nif defined V8F_DRYRUN (echo %TP%'),
    (Q_, "trainpgt runs a different string than it prints", '%TP% >> "%OUT%\\train.log" 2>&1',
     '"%PY%" "%TRAIN%" --arm M2b --seed %~2 --grid pool4x10 --out "%OUT%" %A% >> "%OUT%\\train.log" 2>&1'),
    (Q_, "trainpgt trains into M2b's own run directory", 'if "%~2"=="" (echo REFUSED: trainpgt needs a SEED & exit /b 2)\nset "OUT=%RUNSM%\\M2b-pgt_s%~2"',
     'if "%~2"=="" (echo REFUSED: trainpgt needs a SEED & exit /b 2)\nset "OUT=%RUNSM%\\M2b_s%~2"'),
    (Q_, "trainpgt accepts no seed", 'if "%~2"=="" (echo REFUSED: trainpgt needs a SEED & exit /b 2)\n', 'rem\n'),
    (Q_, "trainpgt does not notice a trainer that exits 0 without summary.json", 'if not exist "%OUT%\\summary.json" (echo [FAIL] train M2b-pgt_s%~2 exited 0 without summary.json & exit /b 1)\n', 'rem\n'),
    (Q_, "trainpgt does not notice a failing trainer", 'if errorlevel 1 (echo [FAIL] train M2b-pgt_s%~2 - see %OUT%\\train.log & exit /b 1)\n', 'rem\n'),
    # ---- queue: the dose, defined once
    (Q_, "the dose start moved", 'set "PGTS=0.5"', 'set "PGTS=0.6"'),
    (Q_, "the dose end moved", 'set "PGTE=0.0"', 'set "PGTE=0.1"'),
    (Q_, "the dose end is M2b's own", 'set "PGTE=0.0"', 'set "PGTE=0.2"'),
    (Q_, "the dose start is defined a second time", 'set "PGTE=0.0"\n', 'set "PGTE=0.0"\nset "PGTS=0.5"\n'),
    (Q_, "the dose end is defined a second time", 'set "PGTS=0.5"\n', 'set "PGTS=0.5"\nset "PGTE=0.0"\n'),
    (Q_, "the dispatch sends trainpgt to the base train stage", 'if /i "%~1"=="trainpgt" goto :trainpgt', 'if /i "%~1"=="trainpgt" goto :train'),
    (Q_, "the dispatch sends gatepgt to the base gate", 'if /i "%~1"=="gatepgt" goto :gatepgt', 'if /i "%~1"=="gatepgt" goto :gate'),
    (Q_, "the dispatch sends evalpgt to the WC eval", 'if /i "%~1"=="evalpgt" goto :evalpgt', 'if /i "%~1"=="evalpgt" goto :evalwc'),
    # ---- queue: evalpgt
    (Q_, "E0 drops --project", '--tag %TG% %EC% %ER% --project --families"', '--tag %TG% %EC% %ER% --families"'),
    (Q_, "E0 drops --families", '--tag %TG% %EC% %ER% --project --families"', '--tag %TG% %EC% %ER% --project"'),
    (Q_, "E1 is not an absmask run", '-tacabs --tac absmask %EC% %ER%"', '-tacabs --tac oracle %EC% %ER%"'),
    (Q_, "E2 is not an oracle run", '--tag %TG%-tacorc --tac oracle %EC% %ER%"', '--tag %TG%-tacorc --tac absmask %EC% %ER%"'),
    (Q_, "E3 drops the all-latcons group (a whole-GT oracle under the latcons name)", '--tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"', '--tac oracle %EC% %ER%"'),
    (Q_, "E3 asks for all-goal, not all-latcons", '--tac-groups all-latcons --tac-alpha 1', '--tac-groups all-goal --tac-alpha 1'),
    (Q_, "E3 drops --tac-alpha 1", '--tac-groups all-latcons --tac-alpha 1 %EC% %ER%"', '--tac-groups all-latcons %EC% %ER%"'),
    (Q_, "E0 evaluates the banked M2b checkpoint", '--ckpt "%OUT%\\ckpt_last.pt" --tag %TG% %EC%', '--ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --tag %TG% %EC%'),
    (Q_, "E3 evaluates the banked M2b checkpoint", '--ckpt "%OUT%\\ckpt_last.pt" --tag %TG%-tacorc-all-latcons', '--ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --tag %TG%-tacorc-all-latcons'),
    (Q_, "E2's tag is E3's (the oracle eval overwrites nothing and names the wrong file)", '--tag %TG%-tacorc --tac oracle %EC% %ER%"', '--tag %TG%-tacorc-all-latcons --tac oracle %EC% %ER%"'),
    (Q_, "the dry run omits E3", 'echo %E2% & echo %E3% & exit /b 0', 'echo %E2% & exit /b 0'),
    (Q_, "the dry run omits E1", 'echo %E0% & echo %E1% & echo %E2%', 'echo %E0% & echo %E2%'),
    (Q_, "the real run executes E1 where it prints E2", '%E2% >> "%EV%\\eval_%TG%-tacorc.log" 2>&1', '%E1% >> "%EV%\\eval_%TG%-tacorc.log" 2>&1'),
    (Q_, "the real run executes E2 where it prints E3", '%E3% >> "%EV%\\eval_%TG%-tacorc-all-latcons.log" 2>&1', '%E2% >> "%EV%\\eval_%TG%-tacorc-all-latcons.log" 2>&1'),
    (Q_, "the real run executes E0 where it prints E1", '%E1% >> "%EV%\\eval_%TG%-tacabs.log" 2>&1', '%E0% >> "%EV%\\eval_%TG%-tacabs.log" 2>&1'),
    (Q_, "evalpgt evaluates an unfinished run", 'if not exist "%OUT%\\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)\n', 'rem\n'),
    (Q_, "evalpgt does not notice a failing E3", 'if errorlevel 1 (echo [FAIL] eval %TG%-tacorc-all-latcons - see %EV%\\eval_%TG%-tacorc-all-latcons.log & exit /b 1)\n', 'rem\n'),
    (Q_, "evalpgt does not notice a failing E1", 'if errorlevel 1 (echo [FAIL] eval %TG%-tacabs - see %EV%\\eval_%TG%-tacabs.log & exit /b 1)\n', 'rem\n'),
    (Q_, "evalpgt skips E3 whenever E2's npz exists", 'if exist "%EV%\\eval_%TG%-tacorc-all-latcons.npz" (echo [skip]', 'if exist "%EV%\\eval_%TG%-tacorc.npz" (echo [skip]'),
    (Q_, "a finished E0 skips E1 as well", 'echo [skip] eval %TG% & goto :evalpgt_1)', 'echo [skip] eval %TG% & goto :evalpgt_2)'),
    (Q_, "a finished E1 skips E2 as well", 'echo [skip] eval %TG%-tacabs & goto :evalpgt_2)', 'echo [skip] eval %TG%-tacabs & goto :evalpgt_3)'),
    (Q_, "a finished E2 skips E3 as well", 'echo [skip] eval %TG%-tacorc & goto :evalpgt_3)', 'echo [skip] eval %TG%-tacorc & goto :evalpgt_4)'),
    (Q_, "evalpgt writes into M2b's run directory", 'set "OUT=%RUNSM%\\M2b-pgt_s%~2"\nset "TG=M2b-pgt_s%~2"', 'set "OUT=%RUNSM%\\M2b_s%~2"\nset "TG=M2b-pgt_s%~2"'),
    (Q_, "evalpgt tags the eval as M2b", 'set "TG=M2b-pgt_s%~2"', 'set "TG=M2b_s%~2"'),
    # ---- chain
    (C_, "the chain copies the dose as a literal start", "PGTS=$(sed -n 's/^set \"PGTS=\\(.*\\)\"$/\\1/p' $QT)", "PGTS=0.5"),
    (C_, "the chain copies the dose as a literal end", "PGTE=$(sed -n 's/^set \"PGTE=\\(.*\\)\"$/\\1/p' $QT)", "PGTE=0.0"),
    (C_, "the chain tolerates a dose defined twice (PGTE)", "[ \"$(grep -c '^set \"PGTE=' $QT)\" = 1 ] ||", "true ||"),
    (C_, "the chain tolerates a dose defined twice (PGTS)", "[ \"$(grep -c '^set \"PGTS=' $QT)\" = 1 ] &&", "true &&"),
    (C_, "the chain does not require the dose in the queue", "[ -n \"$PGTS\" ] && [ -n \"$PGTE\" ] ||", "true ||"),
    (C_, "the gate assertion on the built schedule is gone", 'if lv.get("p_gt_start") != p0 or lv.get("p_gt_end") != p1: bad.append(', 'if False: bad.append('),
    (C_, "the gate no longer refuses M2b's own schedule", 'if (lv.get("p_gt_start"), lv.get("p_gt_end")) == (1.0, 0.2): bad.append(', 'if False: bad.append('),
    (C_, "the gate no longer checks a second lever (w_tac / w_cons)", 'if (w.get("w_tac"), w.get("w_cons")) != (1.0, 0.5): bad.append(', 'if False: bad.append('),
    (C_, "the gate no longer checks the holdout", 'if i["excluded_sha256"] != hold: bad.append(', 'if False: bad.append('),
    (C_, "the gate no longer checks M2b's clip list", 'if i["train_sha256"] != m2b["train_sha256"]: bad.append(', 'if False: bad.append('),
    (C_, "the gate no longer checks M2b's step count", 'if sc["steps_total"] != steps: bad.append(', 'if False: bad.append('),
    (C_, "the gate no longer checks declared_vs_built", 'if sc.get("declared_vs_built") != "PASS": bad.append(', 'if False: bad.append('),
    (C_, "the gate assertions' failure no longer stops the chain", "<<'EOF' || stop \"gate M2b-pgt assertions\"", "<<'EOF' || true"),
    (C_, "the chain runs M2b's own gate, not gatepgt", "queue gatepgt 120 gatepgt", "queue gatepgt 120 gate"),
    (C_, "the chain trains with the base train stage", 'queue "t_M2b-pgt_s$s" 1500 trainpgt $s', 'queue "t_M2b-pgt_s$s" 1500 train M2b $s'),
    (C_, "the chain evaluates with the WC eval stage", 'queue "e_M2b-pgt_s$s" 600 evalpgt $s', 'queue "e_M2b-pgt_s$s" 600 evalwc $s'),
    (C_, "check-inputs failure no longer stops the chain", '|| stop "the reader cannot use the banked M2b side (check-inputs)"', '|| echo ignored'),
    (C_, "check-inputs is not run", 'PYTHONIOENCODING=utf-8 $PY "$RD/v8f_pgt_read.py" --check-inputs', 'PYTHONIOENCODING=utf-8 true "$RD/v8f_pgt_read.py" --check-inputs'),
    (C_, "the seed-0 CONTINUE gate is not enforced", '= CONTINUE ] || finish "$G1', '= CONTINUE ] || echo "$G1'),
    (C_, "the final read is the seed-0 read", "pgtread final pgt_read.json", "pgtread s0 pgt_read.json"),
    (C_, "the seed-0 read is the final read", "pgtread s0 pgt_read_s0.json", "pgtread final pgt_read_s0.json"),
    (C_, "a failing reader no longer stops the chain", '|| stop "pgt read $1"', '|| true'),
    (C_, "train_log.jsonl is not pulled", "for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do", "for f in selfcheck.json summary.json argv.json train.log ckpt_last.pt; do"),
    (C_, "the all-latcons eval is not pulled", 'for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do', 'for f in "" -tacabs -tacorc; do for x in json npz log; do'),
    (C_, "the absmask eval is not pulled", 'for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do', 'for f in "" -tacorc -tacorc-all-latcons; do for x in json npz log; do'),
    (C_, "the archive carries no B1 lead block", '"$PK" "$B1" || stop "archive"', '"$PK" || stop "archive"'),
    (C_, "the tip's built_levers record is not required (p_gt_start)", "grep -q '\"p_gt_start\": cfg.p_gt_start' || stop", "grep -q '\"p_gt_start\": cfg.p_gt_start' || echo"),
    (C_, "the tip's built_levers record is not required (p_gt_end)", "grep -q '\"p_gt_end\": cfg.p_gt_end' || stop", "grep -q '\"p_gt_end\": cfg.p_gt_end' || echo"),
    (C_, "the dry run omits evalpgt 1", 'for st in gatepgt "trainpgt 0" "trainpgt 1" "evalpgt 0" "evalpgt 1"; do', 'for st in gatepgt "trainpgt 0" "trainpgt 1" "evalpgt 0"; do'),
    (C_, "the dry run omits trainpgt 0", 'for st in gatepgt "trainpgt 0" "trainpgt 1"', 'for st in gatepgt "trainpgt 1"'),
    (C_, "the STOP marker is renamed", 'stop() { echo "ZZV8FMINI-STOP $*"', 'stop() { echo "ZZV8FMINI-HALT $*"'),
    (C_, "the DONE marker is renamed", 'echo "ZZV8FMINI-DONE $(date', 'echo "ZZV8FMINI-OK $(date'),
    (C_, "the busy marker is not honoured", "[ -f $B/BOX2_BUSY.txt ] && {", "[ -f $B/BOX2_BUSY_X.txt ] && {"),
    (C_, "the banked M2b train_log is not staged", 'cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null', 'true'),
    (C_, "the banked all-latcons oracle is not staged", "for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b_s$s-tacorc-all-latcons.npz; do", "for f in eval_M2b_s$s-tacorc-all-goal.npz; do"),
    (C_, "the wait for the predecessor is skipped", "until grep -q \"^$AFTER_DONE\" \"$AFTER_OUT\" 2>/dev/null; do", "until true; do"),
]
QONLY, CONLY = [m for m in QUEUEM if m[0] is QUEUE], [m for m in QUEUEM if m[0] is CHAIN]
MUT = {"reach": REACH, "reader": READER, "queue": QUEUEM, "queue-stages": QONLY, "chain": CONLY, "all": REACH + READER + QUEUEM}[WHICH]
if "--part" in sys.argv:                                   # `--part 2/3`: every 3rd mutant starting at the 2nd (run the parts in separate copies of the tree, in parallel)
    _k, _n = (int(x) for x in sys.argv[sys.argv.index("--part") + 1].split("/"))
    MUT = MUT[_k - 1::_n]
KSEL = {id(m): k for k, L in (("reader", READER), ("queue or chain", QUEUEM)) for m in L}      # a reader / queue mutant is tried on its own tests first


def run(k=None):
    cmd = [PY, "-m", "pytest", "-q", "-x", "-rf", "-p", "no:cacheprovider", TEST] + (["-k", k] if k else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
                            "PYTHONPATH": os.pathsep.join([str(ROOT / "stack"), str(ROOT / "taniteval")])})
    lines = r.stdout.strip().splitlines() or ["<no output>"]
    failed = [x.split("::", 1)[1].split(" - ")[0].strip() for x in lines if x.startswith("FAILED ") and "::" in x]
    return r.returncode, lines[-1] + (f"  [{', '.join(failed)}]" if failed else "")


def anchors(m, b):
    o, n = m[2].encode(), m[3].encode()
    if b'\r\n' in b:                                                         # .cmd files are CRLF on the tip; every other file is LF
        o, n = o.replace(b'\n', b'\r\n'), n.replace(b'\n', b'\r\n')
    assert b.count(o) == 1, f"{m[1]}: anchor found {b.count(o)} times in {m[0].name}"
    assert o != n, f"{m[1]}: the mutation changes nothing"
    return o, n


orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
for m in MUT:
    anchors(m, orig[m[0]])
print(f"anchors OK: {len(MUT)} mutants, every anchor occurs exactly once", flush=True)
if "--check-anchors" in sys.argv:
    sys.exit(0)
rc, last = run()
print(f"BASELINE rc={rc} {last}")
assert rc == 0, "the unmutated tree must be GREEN"
red = 0
for m in MUT:
    path, name = m[0], m[1]
    b = orig[path]
    o, n = anchors(m, b)
    path.write_bytes(b.replace(o, n))
    try:
        k = KSEL.get(id(m))
        rc, last = run(k)
        scope = f"[-k {k!r}] " if k else ""
        if rc == 0 and k:                                  # survived its own tests: the whole file decides
            rc, last = run()
            scope = "[whole file] "
    finally:
        path.write_bytes(b)
    red += rc != 0
    print(f"{'RED  ' if rc != 0 else 'GREEN'} {path.name}: {name} -> {scope}{last}", flush=True)
assert all(p.read_bytes() == orig[p] for p in orig), "a file was not restored"
rc, last = run()
print(f"AFTER rc={rc} {last}")
print(f"{red} / {len(MUT)} mutations RED")
sys.exit(0 if red == len(MUT) and rc == 0 else 1)
