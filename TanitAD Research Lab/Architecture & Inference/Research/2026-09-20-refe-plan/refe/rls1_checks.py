#!/usr/bin/env python3
"""RL-S1 chain helpers (refe/rls1_pod.sh, eval/RLS1_POD_RUNBOOK.md): every guard that needs more than a grep lives HERE so that
refe/test_rls1_chain.py can drive it in both directions.  stdlib + numpy; the two eval/ modules are imported lazily.

  verify-file       size + md5 + sha256 of a checkpoint against its PIN (model_final.pt, SFT-4 arm B u400)      ZZRLS1_FILE_OK / _REFUSED
  verify-partition  the frozen partition: manifest sha256, partition_id, both TSVs vs the manifest, log-disjoint ZZRLS1_PARTITION_OK / _REFUSED
  val-logs          cut -f2 rl_val_scenes.tsv | sort -u  (the trainer's --heldout-logs)                       ZZRLS1_VALLOGS_OK
  tlane-check       teacher-lane side files that REACH the 7th output (key + ckpt_step + 64 finite distances)   ZZRLS1_TLANE_OK / _REFUSED
  tlane-exact       TL-1: the labels of one side are EXACT: one accepted row per dumped set, none foreign, none doubled, count == the
                    partition side's scene count                                                               ZZRLS1_TLANE_OK / _REFUSED
  updates-check     the planned optimizer updates (ceil(n/batch * epochs/accum), capped by --max-updates) EQUAL --max-updates  ZZRLS1_UPDATES_OK / _REFUSED
  join-gate         THE post-label gate: every dumped set has a PDM label AND an rl_terms record (100 %), the dump IS the
                    partition side, and G-REWARD-ID (g > 0 <=> DAC = 0) has ZERO disagreements                     ZZRLS1_JOIN_OK / _REFUSED
  monitor / replay  the kill rules K2 (NaN), K3, K4 (two consecutive 250-update checks), K5 (R2), K6 (hours) on the RL-val lookup

Why 100 %: a C run (no --epo) trains on EVERY PDM-labelled set of --onpolicy, while --epo restricts the index to sets that have an rl_terms
record (scorer_finetune.py, `tr_idx = [... in terms_train]`).  Unless the labeller covered every set, C and R are not set-identical and the
C-vs-R contrast is void.  The gate cannot see that from one argv, so the chain does.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "eval"

# ---- pins (MEASURED 2026-10-12 on the dev box: size + md5 + sha256 of the files in place; model_final's sha256 = MODEL_REGISTRY.md, summary.json)
PINS = {
    "model_final": {"size": 1288484600, "md5": "b54773f8c302b2ec8ff4c5971d1e1370",
                    "sha256": "4246a320be71ad0e315f8f298b7dadab3dbbba2c24f4fc30706e0129f5057e66"},
    "u400": {"size": 1288489838, "md5": "8e382eebb35b3e155862453617e5786e",
             "sha256": "8b4dba4818b5cc96c640fe892efe114037ace7cec9dda060c5b4cdd5fb92410d"},
}
CKPT_STEP = 10075                  # model_final.pt's step: the ckpt_step every dumped set must carry (raw/rl_pool/ckpt_last_vs_model_final.json)
# TL-1 (PI 2026-10-12, "paper-pure, smaller pool"): the same tool, the same seed, n_train 48,000 -> 16,000 (RL-val is byte-identical).  OLD id 272cfd357b81 / manifest
# 6a2b5aec... (48,000 / 3,000); NEW id below.  The reason and both ids are logged in eval/PREREG_RL_S1.md section 9.
PARTITION = {"partition_id": "9995a39aeeac3833dd3f679b75e883b056e3d4d235bb9dc986578a81a7a256fa",
             "manifest_sha256": "f8a1640cb61df48ea86e6afdb9d7c357bd6443073b367779585f5238bb723edb",
             "n_train": 16000, "n_val": 3000, "min_val_logs": 60}
OLD_PARTITION_ID = "272cfd357b81db535a1214fc344f71469219c17ac3c8203698d5c2b0f1befa49"
N_PROP = 64
LANE_CHANGE_FACTOR = 2.0           # K5
K4_ROUNDS = 2                      # K3 / K4: two consecutive checks


def say(marker, *rest):
    print(" ".join([marker, *[str(x) for x in rest]]), flush=True)


# ------------------------------------------------------------------------------------------------------------- files / partition
def digests(path, chunk=1 << 22):
    m, s, n = hashlib.md5(), hashlib.sha256(), 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            m.update(b)
            s.update(b)
            n += len(b)
    return n, m.hexdigest(), s.hexdigest()


def sha256_file(path):
    return digests(path)[2]


def verify_file(path, pin, pins=None):
    pins = PINS if pins is None else pins
    if pin not in pins:
        return False, f"unknown_pin:{pin}"
    if not os.path.isfile(path):
        return False, f"missing:{path}"
    want = pins[pin]
    size, md5, sha = digests(path)
    if size != want["size"]:
        return False, f"size_mismatch:{size}!={want['size']}"
    if md5 != want["md5"]:
        return False, f"md5_mismatch:{md5[:12]}!={want['md5'][:12]}"
    if sha != want["sha256"]:
        return False, f"sha256_mismatch:{sha[:12]}!={want['sha256'][:12]}"
    return True, f"{size} B md5 {md5[:12]} sha256 {sha[:12]}"


def read_scenes(tsv):
    """token<TAB>log with a header row -> list of (token, log)."""
    out = []
    with open(tsv, encoding="utf-8", newline="") as f:
        head = f.readline().rstrip("\r\n").split("\t")
        it, il = head.index("token"), head.index("log")
        for ln in f:
            ln = ln.rstrip("\r\n")
            if ln:
                p = ln.split("\t")
                out.append((p[it], p[il]))
    return out


def verify_partition(manifest, train, val, pins=None):
    pins = PARTITION if pins is None else pins
    for p in (manifest, train, val):
        if not os.path.isfile(p):
            return False, f"missing:{p}"
    if sha256_file(manifest) != pins["manifest_sha256"]:
        return False, "manifest_sha256_mismatch (not the frozen manifest: a re-partition must be logged in PREREG_RL_S1 and re-pinned here)"
    m = json.load(open(manifest, encoding="utf-8"))
    if m.get("verify") != "clean":
        return False, f"manifest_verify_not_clean:{m.get('verify')}"
    if m.get("partition_id") != pins["partition_id"]:
        return False, f"partition_id_mismatch:{str(m.get('partition_id'))[:12]}"
    for side, p in (("train", train), ("val", val)):
        if os.path.basename(p) != m[side]["file"]:
            return False, f"{side}_file_name:{os.path.basename(p)}!={m[side]['file']}"
        if sha256_file(p) != m[side]["sha256"]:
            return False, f"{side}_tsv_sha256_mismatch"
    tr, va = read_scenes(train), read_scenes(val)
    if (len(tr), len(va)) != (pins["n_train"], pins["n_val"]):
        return False, f"counts:{len(tr)}/{len(va)}!={pins['n_train']}/{pins['n_val']}"
    ltr, lva = {l for _, l in tr}, {l for _, l in va}
    if ltr & lva:
        return False, f"logs_on_both_sides:{len(ltr & lva)}"
    if len(lva) < pins["min_val_logs"]:
        return False, f"val_logs:{len(lva)}<{pins['min_val_logs']}"
    if {t for t, _ in tr} & {t for t, _ in va}:
        return False, "token_on_both_sides"
    return True, f"{m['partition_id'][:12]} train={len(tr)}/{len(ltr)} logs val={len(va)}/{len(lva)} logs"


def write_val_logs(val_tsv, out):
    logs = sorted({l for _, l in read_scenes(val_tsv)})
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(l + "\n" for l in logs))
    return len(logs)


# ------------------------------------------------------------------------------------------------------------- teacher lane
def tlane_check(dir_, tsv, ckpt_step, min_rows, n_prop=N_PROP):
    """Rows of <dir>/teacher_lane_r*.jsonl that train.OnPolicyBank._apply_teacher_lane would ACCEPT for the dumped sets (key in the partition side,
    ckpt_step equal, n_prop finite centre-line distances).  The trainer's G10 needs > 0 on both sides; the chain asks for it before any compute."""
    scenes = dict(read_scenes(tsv))
    good = bad = 0
    for fp in sorted(glob.glob(os.path.join(dir_, "teacher_lane_r*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                    k = tuple(d["key"])
                    props = d["props"]
                    ok = (len(k) == 4 and scenes.get(k[1]) == k[0] and int(k[3]) == 0 and int(d["ckpt_step"]) == ckpt_step and len(props) == n_prop
                          and all(p.get("center_line.CenterLine.info") is not None and math.isfinite(float(p["center_line.CenterLine.info"])) for p in props))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    ok = False
                good += ok
                bad += not ok
    if good < max(min_rows, 1):
        return False, f"usable_rows:{good}<{max(min_rows, 1)} (unusable {bad}) in {dir_}", good
    return True, f"{good} usable rows ({bad} unusable)", good


def _lane_row_state(d, ckpt_step, n_prop=N_PROP):
    """'ok' | 'bad' for a decoded teacher-lane row: the trainer's acceptance (4-part key, rank 0, ckpt_step, n_prop finite distances) PLUS the SFT-4 recipe marker."""
    try:
        k = tuple(d["key"])
        props = d["props"]
        ok = (len(k) == 4 and int(k[3]) == 0 and int(d["ckpt_step"]) == ckpt_step and len(props) == n_prop and d.get("repair") == "last_heading_hold"
              and d.get("kind") == "teacher_lane_labels"
              and all(p.get("center_line.CenterLine.info") is not None and math.isfinite(float(p["center_line.CenterLine.info"])) for p in props))
    except (KeyError, TypeError, ValueError, AttributeError):
        return "bad"
    return "ok" if ok else "bad"


def tlane_exact(dir_, sets_dir, tsv, ckpt_step=CKPT_STEP, n_prop=N_PROP, expect=None):
    """TL-1: -> (ok, reason, report).  The teacher-lane labels of one partition side are EXACT, never 'at least one usable row' (that was the pre-TL-1 floor):
      * the dump holds exactly the side's scenes (and `expect` of them when given);
      * every dumped set has exactly ONE accepted row, of the dump's own ckpt_step, carrying `repair: last_heading_hold`;
      * no accepted row for a set that was not dumped; no decodable-but-rejected row (a stale or foreign recipe in the directory); no key labelled twice.
    A torn (undecodable) line is a kill's fragment: counted, not fatal."""
    scenes = dict(read_scenes(tsv))
    dumped, dup_dump, stray, _ = read_dumped(sets_dir)
    rep = {"scenes": len(scenes), "dumped": len(dumped), "accepted": 0, "missing": 0, "foreign": 0, "doubled": 0, "rejected": 0, "torn": 0}
    seen = {}
    for fp in sorted(glob.glob(os.path.join(dir_, "teacher_lane_r*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    rep["torn"] += 1
                    continue
                if _lane_row_state(d, ckpt_step, n_prop) != "ok":
                    rep["rejected"] += 1
                    continue
                k = tuple(d["key"])
                k = (k[0], k[1], int(k[2]), int(k[3]))
                if k in seen:
                    rep["doubled"] += 1
                seen[k] = ckpt_step
    rep["accepted"] = len(seen)
    rep["foreign"] = sum(1 for k in seen if k not in dumped)
    rep["missing"] = sum(1 for k in dumped if k not in seen)
    bad = []
    if dup_dump or stray:
        bad.append(f"dump_not_clean(dup {dup_dump}, stray {stray[:2]})")
    if len(scenes) != len(dumped) or any(scenes.get(k[1]) != k[0] for k in dumped):
        bad.append(f"dump_is_not_the_side:{len(dumped)}/{len(scenes)}")
    if expect is not None and len(dumped) != expect:
        bad.append(f"dumped_count:{len(dumped)}!={expect}")
    if rep["missing"]:
        bad.append(f"missing:{rep['missing']}")
    if rep["foreign"]:
        bad.append(f"foreign:{rep['foreign']}")
    if rep["doubled"]:
        bad.append(f"doubled:{rep['doubled']}")
    if rep["rejected"]:
        bad.append(f"rejected_rows:{rep['rejected']}")
    if rep["accepted"] != len(scenes):
        bad.append(f"label_count:{rep['accepted']}!={len(scenes)}")
    if bad:
        return False, ";".join(bad) + f" in {dir_}", rep
    return True, f"exact {rep['accepted']}/{len(scenes)} (torn fragments {rep['torn']})", rep


def planned_updates(args_str, n_sets):
    """The trainer's own schedule (scorer_finetune.py:927-929) from an argv string: ceil(n/batch * epochs/accum), capped by --max-updates.  -> (total, max_updates)."""
    t = args_str.split()

    def val(flag, default):
        return type(default)(t[t.index(flag) + 1]) if flag in t else default
    batch, accum, epochs, mx = val("--batch", 8), val("--accum", 8), val("--epochs", 1.0), val("--max-updates", 0)
    total = int(math.ceil(n_sets / batch * epochs / accum))
    return (min(total, mx) if mx else total), mx


def updates_check(args_str, n_sets):
    """The pre-registered budget is --max-updates updates.  A shrunken pool with an unchanged --epochs ends EARLIER (TL-1: 16,000 sets at --epochs 2 = 500 updates, and the
    chain reads scores_update1000_*), silently: the trainer's `total = min(...)` never complains.  -> (ok, reason)."""
    total, mx = planned_updates(args_str, n_sets)
    if mx <= 0:
        return False, "no_max_updates_in_argv"
    if total != mx:
        return False, f"planned_updates:{total}!=max_updates:{mx} (n_sets {n_sets}; raise --epochs or lower --max-updates, the prereg budget is {mx} updates)"
    return True, f"{total} updates over {n_sets} sets"


# ------------------------------------------------------------------------------------------------------------- the join gate
def _rtj():
    sys.path.insert(0, str(EVAL_DIR))
    import rl_terms_join_identity as rtj
    return rtj


def read_dumped(sets_dir):
    """{key: ckpt_step} of the frozen-fan dump (rl_fan_dump.py: onpolicy_rlfan_*.jsonl).  A key twice, or a foreign file, is an error."""
    out, dup = {}, 0
    files = sorted(glob.glob(os.path.join(sets_dir, "onpolicy_rlfan_*.jsonl")))
    stray = [os.path.basename(f) for f in glob.glob(os.path.join(sets_dir, "onpolicy_*.jsonl")) if f not in files]
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                r = json.loads(ln)
                if r.get("kind") != "onpolicy_set":
                    continue
                k = (r["log_name"], r["token"], int(r["step"]), int(r.get("rank", 0)))
                dup += k in out
                out[k] = int(r["ckpt_step"])
    return out, dup, stray, len(files)


def join_gate(side, sets_dir, pdm_dir, terms_dir, tsv, ckpt_step=CKPT_STEP, n_prop=N_PROP, expect=None):
    """-> (ok, reason, report).  See the module docstring."""
    rtj = _rtj()
    rep = {"side": side}
    scenes = read_scenes(tsv)
    want = {t: l for t, l in scenes}
    dumped, dup, stray, nfiles = read_dumped(sets_dir)
    rep.update({"scenes_in_partition_side": len(scenes), "dumped_sets": len(dumped), "dump_files": nfiles, "duplicate_keys": dup})
    if nfiles == 0:
        return False, "no_dump_files", rep
    if stray or dup:
        return False, f"foreign_or_duplicate_sets:stray={stray[:3]} dup={dup}", rep
    if expect is not None and len(scenes) != expect:
        return False, f"partition_side_size:{len(scenes)}!={expect}", rep
    toks = {k[1] for k in dumped}
    if toks != set(want) or len(dumped) != len(want):
        return False, (f"dump_is_not_the_partition_side:dumped={len(dumped)} partition={len(want)} "
                       f"missing={len(set(want) - toks)} extra={len(toks - set(want))}"), rep
    wrong_log = [k for k in dumped if want[k[1]] != k[0]]
    if wrong_log or any(k[3] != 0 for k in dumped):
        return False, f"log_or_rank_disagrees:{len(wrong_log)}", rep
    if set(dumped.values()) != {ckpt_step}:
        return False, f"ckpt_step:{sorted(set(dumped.values()))[:3]}!={ckpt_step} (the fan must be model_final's)", rep
    pdm = rtj.load(rtj.files_of([pdm_dir], "pdm_"), "pdm_targets")
    rlt = rtj.load(rtj.files_of([terms_dir], "rl_terms_"), "rl_terms")
    keyed = [(k, ckpt_step) for k in dumped]
    miss_pdm = [k for k in keyed if k not in pdm]
    miss_rlt = [k for k in keyed if k not in rlt]
    rep.update({"pdm_labelled": len(keyed) - len(miss_pdm), "terms_labelled": len(keyed) - len(miss_rlt),
                "pdm_missing_first5": [list(k[0]) for k in miss_pdm[:5]], "terms_missing_first5": [list(k[0]) for k in miss_rlt[:5]]})
    if miss_pdm or miss_rlt:
        return False, (f"labels_below_100_percent:pdm {rep['pdm_labelled']}/{len(keyed)} terms {rep['terms_labelled']}/{len(keyed)} "
                       "(a C run trains on every PDM-labelled set, an --epo run only on sets with rl_terms: C and R would not be set-identical)"), rep
    bad_shape = [k for k in keyed if any(len(pdm[k].get(f, ())) != n_prop for f in ("nc", "dac", "ep", "ttc", "c", "ddc"))
                 or any(len(rlt[k].get(f, ())) != n_prop for f in ("d", "k", "lc_executed", "dither", "gmax"))
                 or not isinstance(rlt[k].get("label_fires"), bool)]
    if bad_shape:
        return False, f"record_shape:{len(bad_shape)} sets lack {n_prop}-wide arrays or a boolean label_fires", rep
    j = rtj.join({k: rlt[k] for k in keyed}, {k: pdm[k] for k in keyed})
    rep["identity"] = {x: j[x] for x in ("scenes_joined", "scenes_rlt_only", "scenes_pdm_only", "proposals", "confusion", "disagreements_n",
                                         "D_exactly_zero_on_dac1", "D_positive_on_dac0_share")}
    c = j["confusion"]
    if j["scenes_joined"] != len(keyed) or j["disagreements_n"] != 0 or c["pred_off_dac1"] or c["pred_ok_dac0"] or j["D_exactly_zero_on_dac1"] is False:
        return False, f"G_REWARD_ID_broken:disagreements={j['disagreements_n']} joined={j['scenes_joined']}/{len(keyed)} confusion={c}", rep
    return True, (f"{len(keyed)} sets, pdm 100 %, terms 100 %, G-REWARD-ID 0 disagreements over {j['proposals']} proposals"), rep


# ------------------------------------------------------------------------------------------------------------- the kill rules
def judge(arm, rec, hist, k3_enabled=True):
    """(code or None, why).  rec/hist items: {'tag','g3_ok','g5_ok','k4_breach','k5_breach'}.  K3 and K4 need TWO CONSECUTIVE breaches, K5 (R2 only) one."""
    prev = hist[-1] if hist else None
    k3 = lambda r: not (r["g3_ok"] and r["g5_ok"])                                # noqa: E731
    if k3_enabled and prev is not None and k3(rec) and k3(prev):
        return "K3", f"g3/g5 bound breached at {prev['tag']} and {rec['tag']}"
    if prev is not None and rec["k4_breach"] and prev["k4_breach"]:
        return "K4", f"RL-val PDMS below the control by more than 0.5 at {prev['tag']} and {rec['tag']}"
    if arm == "R2" and rec.get("k5_breach"):
        return "K5", f"R2 lane-change rate above {LANE_CHANGE_FACTOR} x the label base rate at {rec['tag']}"
    return None, ""


def lane_change_stats(picks_lc, fires):
    """picks_lc, fires: aligned bool lists over the RL-val sets (lc_executed at the pick; the scene-level label)."""
    n = len(picks_lc)
    rate, base = sum(picks_lc) / n, sum(fires) / n
    tp = sum(1 for a, b in zip(picks_lc, fires) if a and b)
    return {"n": n, "rate": rate, "base_rate": base, "n_lane_change": sum(picks_lc), "n_label_fires": sum(fires),
            "precision": tp / sum(picks_lc) if sum(picks_lc) else None, "recall": tp / sum(fires) if sum(fires) else None,
            "breach": bool(rate > LANE_CHANGE_FACTOR * base)}


class RlValChecker:
    """One RL-val lookup of an R arm against its matched control at one tag (in-process rl_val_eval, point values only: n_boot is small)."""

    def __init__(self, arm, refdir, pdm, terms, sets, tsv, rule="navsim_v1_lane", n_boot=200):
        sys.path.insert(0, str(EVAL_DIR))
        sys.path.insert(0, str(HERE))
        import rl_val_eval as RV
        import rl_reward as RW
        self.RV, self.arm, self.refdir, self.rule, self.n_boot = RV, arm, refdir, rule, n_boot
        self.pdm, _ = RV.load_pdm(pdm)
        self.terms, _ = RV.load_terms(terms)
        self.sets, self.scenes = sets, RV.read_scenes(tsv)
        self.cfg = RW.RewardCfg(lam_d=0.15, w_prog=0.5, guard=True, use_k=(arm == "R2"), use_l=(arm == "R2"))

    def __call__(self, tag, files):
        RV = self.RV
        ref = os.path.join(self.refdir, f"scores_{tag}_A.npz")
        arms = {"R": RV.load_scores(files["B"]), "C": RV.load_scores(ref)}
        common = set(arms["R"]) & set(arms["C"])
        paths = RV.load_paths(self.sets, common, {k: arms["R"][k]["ckpt"] for k in common})
        res = RV.run(arms, self.pdm, self.terms, paths, self.rule, self.cfg, self.scenes, True, "C", self.n_boot)
        g3 = res["arms"]["R"].get("g3") or {}
        pv = res["paired_vs_base"]["R"]
        picks, fires = [], []
        for k in sorted(common):
            t = self.terms.get(k)
            if t is None or t.get("lc") is None:
                continue
            i, _ = RV.select(arms["R"][k]["scores"], self.rule, arms["R"][k]["space"])
            picks.append(bool(t["lc"][i]))
            fires.append(bool(t["fires"]))
        lc = lane_change_stats(picks, fires) if picks else None
        return {"tag": tag, "g3_ok": bool(g3 and g3["pick_ok"] and g3["share_ok"]), "g3": g3 or None,
                "g5_ok": bool(pv["g5"]["NC_fall_ok"] and pv["g5"]["TTC_fall_ok"]), "g5": pv["g5"],
                "k4_breach": bool(pv["K4"]["breach"]), "pdms_delta_x100": pv["K4"]["pdms_delta"],
                "k5_breach": bool(lc["breach"]) if lc else False, "lane_change": lc, "n_sets": res["eligibility"]["sets_used"]}


def _read_rows(path):
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def _finite_row(r):
    return all(math.isfinite(float(v)) for k, v in r.items() if k in ("bceA", "lossB", "listnet", "epo", "epo_J", "epo_kl") and isinstance(v, (int, float)))


def monitor(arm, out_dir, check_fn, pid, alive_fn, kill_fn, sleep_fn=time.sleep, now_fn=time.time, poll=20.0, max_hours=6.0, start=None):
    """Beside a running scorer_finetune.py: after every `score_dump` event (the files are complete then) run the RL-val lookup and apply the kill rules.
    C arms are watched for K2 (non-finite loss) and K6 only; K3 / K4 / K5 are the R arms' (they compare R with its control).  A check that RAISES is
    logged (ZZRLS1_CHECK_ERROR) and does not kill: `replay` re-judges every tag after the run.  -> exit code 0 (clean) / 2 (killed)."""
    start = now_fn() if start is None else start
    seen, hist, k3_enabled, killed = set(), [], True, None
    sft = os.path.join(out_dir, "sft.jsonl")

    def kill(code, why):
        with open(os.path.join(out_dir, "KILLED.json"), "w", encoding="utf-8") as f:
            json.dump({"arm": arm, "code": code, "why": why, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        say("ZZRLS1_KILL", arm, code, why)
        kill_fn(pid)
        return code

    while True:
        was_alive = alive_fn(pid)
        for r in _read_rows(sft):
            if r.get("event") == "train" and not _finite_row(r):
                kill("K2", f"non-finite loss at update {r.get('update')}")
                return 2
            if r.get("event") != "score_dump" or r.get("tag") in seen:
                continue
            seen.add(r["tag"])
            if arm.startswith("C"):
                continue
            try:
                if r["tag"] == "step0":
                    rec = check_fn("step0", r["files"])
                    k3_enabled = bool(rec["g3_ok"])
                    say("ZZRLS1_K3_" + ("LIVE" if k3_enabled else "DISABLED_BASE_ALREADY_BREACHES_G3_ON_RL_VAL"), arm, json.dumps(rec.get("g3")))
                    continue
                rec = check_fn(r["tag"], r["files"])
            except Exception as e:                                       # noqa: BLE001
                say("ZZRLS1_CHECK_ERROR", arm, r.get("tag"), f"{type(e).__name__}: {str(e)[:200]}")
                continue
            code, why = judge(arm, rec, hist, k3_enabled)
            hist.append(rec)
            with open(os.path.join(out_dir, "kill_checks.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "k3_enabled": k3_enabled, "kill": code}) + "\n")
            say("ZZRLS1_CHECK", arm, r["tag"], f"g3_ok={rec['g3_ok']} g5_ok={rec['g5_ok']} k4_breach={rec['k4_breach']} k5_breach={rec['k5_breach']}")
            if code:
                kill(code, why)
                return 2
        if (now_fn() - start) / 3600.0 > max_hours:
            kill("K6", f"more than {max_hours} h on the A40")
            return 2
        if not was_alive:
            return 0
        sleep_fn(poll)


def replay(arm, out_dir, check_fn):
    """After the run: re-judge EVERY tag in order from sft.jsonl (the monitor may have errored or missed a check).  -> (code or None, why, records)."""
    rows = _read_rows(os.path.join(out_dir, "sft.jsonl"))
    if any(r.get("event") == "train" and not _finite_row(r) for r in rows):
        return "K2", "non-finite loss in sft.jsonl", []
    if arm.startswith("C"):
        return None, "control arm: K2 only", []
    hist, k3_enabled, recs = [], True, []
    for r in rows:
        if r.get("event") != "score_dump":
            continue
        rec = check_fn(r["tag"], r["files"])
        if r["tag"] == "step0":
            k3_enabled = bool(rec["g3_ok"])
            continue
        code, why = judge(arm, rec, hist, k3_enabled)
        hist.append(rec)
        recs.append({**rec, "k3_enabled": k3_enabled})
        if code:
            return code, why, recs
    return None, "", recs


# ------------------------------------------------------------------------------------------------------------- CLI
def _kill_group(pid):
    import signal
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(30)
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        pass


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                                # a zombie answers kill -0: it is over
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify-file"); p.add_argument("--path", required=True); p.add_argument("--pin", required=True)
    p = sub.add_parser("verify-partition"); [p.add_argument(f"--{x}", required=True) for x in ("manifest", "train", "val")]
    p = sub.add_parser("val-logs"); p.add_argument("--val", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("tlane-check"); p.add_argument("--dir", required=True); p.add_argument("--tsv", required=True)
    p.add_argument("--min-rows", type=int, default=1); p.add_argument("--ckpt-step", type=int, default=CKPT_STEP)
    p = sub.add_parser("tlane-exact"); p.add_argument("--dir", required=True); p.add_argument("--sets", required=True); p.add_argument("--tsv", required=True)
    p.add_argument("--ckpt-step", type=int, default=CKPT_STEP); p.add_argument("--expect", type=int, default=None); p.add_argument("--json", default=None)
    p = sub.add_parser("updates-check"); p.add_argument("--args", required=True); p.add_argument("--n-sets", type=int, required=True)
    p = sub.add_parser("join-gate"); p.add_argument("--side", required=True, choices=["train", "val"])
    for x in ("sets", "pdm", "terms", "tsv"):
        p.add_argument(f"--{x}", required=True)
    p.add_argument("--ckpt-step", type=int, default=CKPT_STEP); p.add_argument("--n-prop", type=int, default=N_PROP)
    p.add_argument("--expect", type=int, default=None); p.add_argument("--json", default=None)
    for name in ("monitor", "replay"):
        p = sub.add_parser(name)
        for x in ("arm", "out", "refdir", "pdm", "terms", "sets", "tsv"):
            p.add_argument(f"--{x}", required=True)
        p.add_argument("--rule", default="navsim_v1_lane")
        if name == "monitor":
            p.add_argument("--pid", type=int, required=True); p.add_argument("--poll", type=float, default=20.0)
            p.add_argument("--max-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    if a.cmd == "verify-file":
        ok, why = verify_file(a.path, a.pin)
        say("ZZRLS1_FILE_OK" if ok else "ZZRLS1_FILE_REFUSED", a.pin, why)
        return 0 if ok else 1
    if a.cmd == "verify-partition":
        ok, why = verify_partition(a.manifest, a.train, a.val)
        say("ZZRLS1_PARTITION_OK" if ok else "ZZRLS1_PARTITION_REFUSED", why)
        return 0 if ok else 1
    if a.cmd == "val-logs":
        say("ZZRLS1_VALLOGS_OK", write_val_logs(a.val, a.out))
        return 0
    if a.cmd == "tlane-check":
        ok, why, _ = tlane_check(a.dir, a.tsv, a.ckpt_step, a.min_rows)
        say("ZZRLS1_TLANE_OK" if ok else "ZZRLS1_TLANE_REFUSED", why)
        return 0 if ok else 1
    if a.cmd == "tlane-exact":
        ok, why, rep = tlane_exact(a.dir, a.sets, a.tsv, a.ckpt_step, N_PROP, a.expect)
        if a.json:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"ok": ok, "why": why, **rep}, f, indent=1)
        say("ZZRLS1_TLANE_OK" if ok else "ZZRLS1_TLANE_REFUSED", why)
        return 0 if ok else 1
    if a.cmd == "updates-check":
        ok, why = updates_check(a.args, a.n_sets)
        say("ZZRLS1_UPDATES_OK" if ok else "ZZRLS1_UPDATES_REFUSED", why)
        return 0 if ok else 1
    if a.cmd == "join-gate":
        ok, why, rep = join_gate(a.side, a.sets, a.pdm, a.terms, a.tsv, a.ckpt_step, a.n_prop, a.expect)
        if a.json:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"ok": ok, "why": why, **rep}, f, indent=1)
        say("ZZRLS1_JOIN_OK" if ok else "ZZRLS1_JOIN_REFUSED", a.side, why)
        return 0 if ok else 1
    chk = RlValChecker(a.arm, a.refdir, a.pdm, a.terms, a.sets, a.tsv, a.rule) if not a.arm.startswith("C") else (lambda *x: None)
    if a.cmd == "monitor":
        return monitor(a.arm, a.out, chk, a.pid, _alive, _kill_group, poll=a.poll, max_hours=a.max_hours)
    code, why, recs = replay(a.arm, a.out, chk)
    with open(os.path.join(a.out, "kill_replay.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"arm": a.arm, "kill": code, "why": why, "checks": recs}, f, indent=1)
    say("ZZRLS1_REPLAY_KILL" if code else "ZZRLS1_REPLAY_CLEAN", a.arm, code or "", why)
    return 2 if code else 0


if __name__ == "__main__":
    sys.exit(main())
