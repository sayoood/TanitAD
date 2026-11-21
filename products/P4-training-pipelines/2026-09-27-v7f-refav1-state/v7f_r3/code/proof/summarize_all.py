"""Regenerate every raw/ summary from the run outputs (so no summary predates the final code)."""
from __future__ import annotations

import collections
import json
import re
import statistics as st
import xml.etree.ElementTree as ET

R = "C:/Users/Admin/v7f_r3"
O = f"{R}/proof/out"


def w(name: str, lines: list[str]) -> None:
    with open(f"{R}/raw/{name}", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"--- raw/{name}")
    print("\n".join(lines))


# 1. per-test-id outcome comparison ------------------------------------------------------------
def load_junit(p):
    out = {}
    for tc in ET.parse(p).getroot().iter("testcase"):
        tid = f"{tc.get('classname')}::{tc.get('name')}"
        stt = "passed"
        for ch in tc:
            if ch.tag in ("failure", "error"):
                stt = ch.tag
                break
            if ch.tag == "skipped":
                stt = "skipped"
        out[tid] = stt
    return out


a, b = load_junit(f"{R}/raw/junit_baseline_c36b6ddd.xml"), load_junit(f"{R}/raw/junit_v7f_r3_final.xml")
new = sorted(set(b) - set(a))
changed = sorted(t for t in set(a) & set(b) if a[t] != b[t])
fa = sorted(t for t in a if a[t] in ("failure", "error"))
fb = sorted(t for t in b if b[t] in ("failure", "error"))
L = ["Per-test-id comparison, 88 test files touching train_v6_staged / declared_vs_built / v7_labels /",
     "tac_goal_head (raw/test_files_touching.txt); baseline = untouched c36b6ddd copy, modified = v7f_r3 FINAL code.",
     "",
     f"baseline c36b6ddd: {dict(sorted(collections.Counter(a.values()).items()))} total {len(a)}",
     f"modified v7f_r3  : {dict(sorted(collections.Counter(b.values()).items()))} total {len(b)}",
     "",
     f"tests only in modified: {len(new)} | outside the new R3 file: "
     f"{[t for t in new if 'test_tactical_label_reach_v6' not in t]}",
     f"new tests outcomes: {dict(collections.Counter(b[t] for t in new))}",
     f"tests only in baseline: {sorted(set(a) - set(b))}",
     f"outcome CHANGED on the {len(set(a) & set(b))} common tests: {len(changed)}",
     *[f"    {a[t]} -> {b[t]} {t}" for t in changed],
     "",
     f"failures/errors identical set in both trees: {fa == fb} ({len(fa)})",
     *[f"    {a[t]} {t}" for t in fa],
     "",
     "test_runbook_commands in modified: "
     f"{dict(collections.Counter(b[t] for t in b if 'test_runbook_commands' in t))} | static-closure test: "
     f"{b.get('tests.test_runbook_commands::test_the_static_closure_matches_the_runtime_one')}"]
w("tests_outcome_compare.txt", L)

reasons = collections.Counter()
for tc in ET.parse(f"{R}/raw/junit_v7f_r3_final.xml").getroot().iter("testcase"):
    for ch in tc:
        if ch.tag in ("failure", "error"):
            msg = (ch.get("message") or "") + " " + (ch.text or "")
            m = re.search(r"(is MISSING at [^\n]{0,150}|S2LabelError[^\n]{0,170}|does not exist[^\n]{0,60}|"
                          r"FileNotFoundError[^\n]{0,170}|has sha256 [0-9a-f]{16}[^\n]{0,60}|"
                          r"launch source moved[^\n]{0,80}|AssertionError: WindowsPath\([^\n]{0,170})", msg)
            why = m.group(1) if m else msg.strip().splitlines()[0][:170]
            why = re.sub(r"C:.{0,3}Users.{0,3}Admin.{0,3}v7f_r3", "<TREE>", why)
            reasons[(tc.get("classname").split(".")[-1], why)] += 1
w("preexisting_failure_reasons.txt",
  ["The 18 failures/errors present IDENTICALLY in the untouched c36b6ddd tree and the modified tree, by reason",
   "(raw/junit_v7f_r3_final.xml; <TREE> = the test checkout's parent, which carries only stack/ + taniteval/):", ""]
  + [f"{n:3d}  {c:28s} {y[:200]}" for (c, y), n in sorted(reasons.items())]
  + ["", f"total: {sum(reasons.values())}"])

# 2. cross-process OFF proof ---------------------------------------------------------------------
pa = json.load(open(f"{O}/off_c36b6ddd.json"))
pb = json.load(open(f"{O}/off_v7f_r3.json"))
L = [f"A: {pa['trainer_file']} | tanitad {pa['tanitad_file']} | R3 symbol {pa['has_r3_symbol']}",
     f"B: {pb['trainer_file']} | tanitad {pb['tanitad_file']} | R3 symbol {pb['has_r3_symbol']}"]
n = 0
for k in sorted(pa["runs"]):
    same = pa["runs"][k] == pb["runs"][k]
    n += same
    r = pa["runs"][k]
    L.append(f"{k:28s} {'IDENTICAL' if same else 'DIFFERS'} loss={r['loss_hex']} terms={sorted(r['terms'])} "
             f"param_grads={r['n_param_grads']}")
L += ["", f"{n}/{len(pa['runs'])} runs bit-identical: loss (float.hex), every term (float.hex), log-key set, "
      "global RNG state after the step, digest of EVERY parameter gradient after backward, state_dict digest",
      "configs: default | --goal-multilabel ; batches: plain (incumbent keys) | labels (+ S2 + tac_* + all six "
      "TAC_LABEL_BATCH_KEYS from the REAL v7.2 join, 9 literal clips)"]
w("off_compare.txt", L)


# 3. seeded CLI OFF comparison -------------------------------------------------------------------
def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for kk, v in d.items():
            out |= flat(v, f"{p}.{kk}" if p else kk)
    else:
        out[p] = d
    return out


ra = json.load(open(f"{O}/cli_off/v7f_r3_base/S-S/dry_run.json"))
rb = json.load(open(f"{O}/cli_off_rep/base_again/dry_run.json"))
L = ["CLI flag-OFF dry-run (the real main(): preflight + dry_run), untouched c36b6ddd tree",
     "(C:/Users/Admin/v7f_r3_base = byte copy of the snapshot, raw/tree_identity_base_vs_snapshot.txt) vs the",
     "modified tree (FINAL code), both via code/proof/seeded_main.py (global torch/random seeded to 0 OUTSIDE",
     "the trainer). PRE-EXISTING, not this change: without that external seed two runs of the SAME untouched",
     "tree differ, because dry_run() never seeds the global RNG before build_stack_from_args (train() does).", "",
     f"control: untouched tree vs itself (seeded), S-S steps identical = {ra['steps'] == rb['steps']}", ""]
skip = (lambda kk: kk.startswith("provenance") or kk == "args.out")
for stg in ("S-W", "S-T", "S-S", "S-J"):
    x = json.load(open(f"{O}/cli_off/v7f_r3_base/{stg}/dry_run.json"))
    y = json.load(open(f"{O}/cli_off/v7f_r3/{stg}/dry_run.json"))
    L.append(f"== {stg}: dry_run.json steps IDENTICAL={x['steps'] == y['steps']} ({len(y['steps'])} rows; "
             f"per-step loss {[r['loss'] for r in y['steps']]}; terms {y['steps'][0]['terms']})")
    L.append("   dry_run.json top-level keys differing (excl. elapsed_s): "
             f"{sorted(kk for kk in set(x) | set(y) if kk != 'elapsed_s' and x.get(kk) != y.get(kk))}")
    ca = flat(json.load(open(f"{O}/cli_off/v7f_r3_base/{stg}/config.json")))
    cb = flat(json.load(open(f"{O}/cli_off/v7f_r3/{stg}/config.json")))
    L.append(f"   config.json keys ADDED  : {sorted(kk for kk in cb if kk not in ca and not skip(kk))}")
    L.append(f"   config.json keys REMOVED: {sorted(kk for kk in ca if kk not in cb and not skip(kk))}")
    L.append(f"   config.json values CHANGED: {sorted(kk for kk in ca if kk in cb and ca[kk] != cb[kk] and not skip(kk))}")
w("cli_off_dryrun_compare.txt", L)

# 4. CLI ON/OFF dry-run summary ----------------------------------------------------------------
L = []
for nm in ("dry_on_ST", "dry_on_SJ", "dry_off_ST"):
    r = json.load(open(f"{O}/{nm}/dry_run.json"))
    L.append(f"==== {nm}: gate {r['gate_verdict']}")
    t = r.get("tac_label_all")
    L.append("  tac_label_all report: " + (
        f"exercised {t['exercised']} | gdvb_v6 {t['gdvb_v6']} | negatives {t['negatives']} | trainable "
        f"{t['n_trainable']}/{t['n_total']} | masked {sorted(t['masked_why'])} | n_label_records "
        f"{t['n_label_records']} | on pos_weight cap {t['n_on_pos_weight_cap']}" if t else "(no tac_label_all key)"))
    for row in r["steps"]:
        keep = {kk: row[kk] for kk in ("step", "loss", "terms", "gnorm") if kk in row}
        keep |= {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in row.items()
                 if kk in ("tac_label_loss", "tac_lat_ce", "tac_lon_ce", "tac_goal_bce", "tac_speedband_l1_ms",
                           "tac_goal_n_supervised", "tac_n_valid", "seam_op", "t1_latent", "s1_latent",
                           "fan_mean_ade")}
        L.append("   " + json.dumps(keep))
    c = json.load(open(f"{O}/{nm}/config.json"))
    L.append(f"  config.json loss_weights_in_force.w_tac_label_all = {c['loss_weights_in_force'].get('w_tac_label_all')}")
w("cli_dry_runs_summary.txt", L)

# 5. real eval-label smoke ---------------------------------------------------------------------
r = json.load(open(f"{O}/smoke_real_eval_labels.json"))
L = []
for arm in ("train_on_measured", "train_off"):
    rows = r[arm]["rows"]
    L.append(f"== {arm} (w_tac_label_all={r[arm]['w_tac_label_all']}, in-band {r[arm]['n_in_band_windows']}/"
             f"{r[arm]['n_windows']})")
    for kk in ("loss", "tac_lat_ce", "tac_lon_ce", "tac_goal_bce", "tac_speedband_l1_ms", "t1_latent", "seam_op"):
        v = [x[kk] for x in rows if x.get(kk) is not None]
        if v:
            L.append(f"  {kk:22s} steps1-5 mean {st.mean(v[:5]):.4f} | steps26-30 mean {st.mean(v[-5:]):.4f} | n={len(v)}")
    L.append(f"  terms: {rows[0]['terms']}")
for nm in ("reach_measured", "reach_all"):
    x = r[nm]
    L.append(f"== {nm}: tokens reached {x['n_tokens_reached']}/22, tac_label_all {x['tac_label_all']:.4f}, "
             f"speed-band arg rows {x['speed_band_arg_rows_reached']}")
    for t, d in x["per_token"].items():
        L.append(f"   {t:28s} pos_sup={d['n_pos_supervised']:3d} neg_sup={d['n_neg_supervised']:3d} "
                 f"mask={d['class_mask']} reached={d['reached']}")
w("smoke_real_eval_labels_trajectory.txt", L)


# 6. real train() smoke ------------------------------------------------------------------------
def rows_of(p):
    return [d for d in (json.loads(l) for l in open(p, encoding="utf-8")) if "step" in d and "loss" in d]


keys = ("step", "loss", "terms", "t1_latent", "seam_op", "fan_mean_ade", "tac_label_loss", "tac_n_valid",
        "tac_lat_ce", "tac_lon_ce", "tac_goal_bce", "tac_goal_n_supervised", "tac_goal_n_pos",
        "tac_goal_pos_counts", "tac_speedband_l1_ms")
L = ["REAL train() S-T smoke, CPU, FINAL code. Real: the v6 train() loop, a 3-clip v2 cache copied from the local",
     "eval cache (one RED, one GREEN, one YELLOW traffic-light clip), the canonical v7.2 EVAL label blob via",
     "--s2-labels, the R3 join, the S-T freeze over a 2-step S-W checkpoint (--init-from/--prev-gate; the",
     "INCONCLUSIVE gate overridden WITH a stated reason). NOT real: model geometry (tiny ViT d32/depth1 at",
     "256x640), 6 steps, batch 4. Stamped overrides: --no-require-parity --exclude-eval-clips none",
     "--allow-eval-clips-in-train (eval clips as train data: a PLUMBING smoke, not an arm).", ""]
on = rows_of(f"{O}/train_smoke/ST_on/train_log.jsonl")
off = rows_of(f"{O}/train_smoke/ST_off/train_log.jsonl")
for arm, R_ in (("ST_on", on), ("ST_off", off)):
    L.append(f"==== {arm}: {len(R_)} logged steps")
    for x in R_:
        L.append("   " + json.dumps({kk: (round(x[kk], 5) if isinstance(x.get(kk), float) else x.get(kk))
                                     for kk in keys if kk in x}))
L.append("")
L.append("step-1 incumbent terms ON vs OFF (same init, same drawn windows, before any update):")
for kk in ("t1_latent", "seam_op", "fan_mean_ade"):
    L.append(f"   {kk}: ON {on[0].get(kk)} | OFF {off[0].get(kk)} | identical {on[0].get(kk) == off[0].get(kk)}")
seen = set()
for x in on:
    seen |= {t for t in (x.get("tac_goal_pos_counts") or {}) if t.startswith("TRAFFIC_LIGHT")}
L.append(f"traffic-light colours supervised as POSITIVES across the {len(on)} ON steps: {sorted(seen)}")
c = json.load(open(f"{O}/train_smoke/ST_on/config.json"))
t = c["tac_label_all"]
L.append("")
L.append(f"config.json['tac_label_all']: negatives {t['negatives']} | trainable {t['n_trainable']}/{t['n_total']} | "
         f"masked {sorted(t['masked_why'])} | join "
         f"{ {kk: v for kk, v in t['join'].items() if kk in ('n_windows', 'n_windows_in_band_tac', 'n_matched_episodes', 'n_joined_records', 'n_joined_records_with_speed_band_args')} }")
L.append(f"config.json['s2']['join']['tactical_consumer']: {c['s2']['join']['tactical_consumer']}")
L.append(f"config.json loss_weights_in_force.w_tac_label_all: {c['loss_weights_in_force']['w_tac_label_all']}")
c0 = json.load(open(f"{O}/train_smoke/ST_off/config.json"))
L.append(f"OFF config.json: has tac_label_all {'tac_label_all' in c0} | has s2 {'s2' in c0} | w in force "
         f"{c0['loss_weights_in_force']['w_tac_label_all']}")
w("train_smoke_ST_summary.txt", L)
