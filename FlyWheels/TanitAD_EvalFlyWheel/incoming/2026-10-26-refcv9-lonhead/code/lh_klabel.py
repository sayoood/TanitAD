"""lh_klabel.py -- PREREG_LONHEAD_C s1 K-LABEL: the per-window GT LON6 target, built by EXACTLY the function refcv8's trainer uses, and its
reproduction of the battery's GT lon label on eval139.  CPU only, 0 GPU, no model output is read.  NEW file (refcv9 LONHEAD agent, 2026-10-26).

THE TRAINER'S LON TARGET (refcv8 = `--refcv8` + `--r8-v9-labels[-eval]`, E:/refcv8_eval_kit/ckpt/config.json argv):
  stack/scripts/refc_v3_train.py:4664-4674   item["lon_v7"] = IGNORE if sid in tactical_excluded_sids else _j8["v9_lon"],
                                             _j8 = self.r8_join.item(sid, t + w - 1 + raw_offset, self._now_s(ep, t))       (:4646-4647)
  stack/scripts/refc_v3_train.py:9541-9546 / 9834-9838   r8train.load_v9_join(<release>, expect_md5, rc_variant, lat_variant)  (train / eval)
  stack/tanitad/train/refcv8_train.py:876-903   R8LabelJoin.item -> V9.row_for_now (clock checked to 1e-6 s) -> V9.window_targets(ids="v7")["lon"]
  stack/tanitad/data/v9_labels.py:155-186      window_targets: lon = rows["lon_v7id"][r]
  stack/scripts/refc_v3_train.py:4309-4315     now_s = grid_start_s + (t + w - 1 + raw_offset) * dt_s  (clock from the clip-clock sidecar, :4387-4410)
  stack/scripts/refc_v3_train.py:4255-4260     G3: an eval clip absent from the sidecar has its TACTICAL targets excluded (IGNORE)
The release row's `lon_v7id` is written by stack/scripts/build_v9_labels.py:clip_labels (:893-963, LON9_TO_7 :467); its FOLLOW evidence comes ONLY
from the `--lead` pass (:910-931): without it `fol_known` is False, so DECELERATE / ACCELERATE / KEEP become PARTIAL (lon_v7id = IGNORE,
allowed {cls, FOLLOW}) and FOLLOW is never emitted (:932-958).

CHECKS (all computed here; none is a model output):
  K-LABEL-A  the trainer's join function on the eval139 release (md5 6b5c7f20, the one the battery roll fed) reproduces the battery's banked
             `lon_v7` on EVERY eval window, both inference-seed dumps (exact integer equality, IGNORE included).
  K-LABEL-B  the FRESH kit's build recipe (`l1c_build_inputs.py step_v9`: build_v9_labels.py build WITHOUT --lead, L1C DEVIATIONS D4)
             applied to eval139 -- emulated EXACTLY from the release's own `lon_cls` (no-lead lon_v7id = lon_v7id iff lon_cls in {HOLD, CREEP,
             STOP}, else IGNORE: build_v9_labels.py:932-958 with lead=None) -- reproduces the battery label on every window?  This is what
             "train and eval targets agree" means for the fresh split AS BUILT.
  FRESH      the same join function on the fresh release (md5 cc06d03a) and the fresh dump's banked `lon_v7` (identity), + the class census.
No clip id is printed or written: clips are handled as sha12 / sid integers inside the process only; outputs are counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

TREE = Path(os.environ.get("R8_LAUNCH_TREE", "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58"))
for p in (str(TREE / "stack"),):
    if p not in sys.path:
        sys.path.insert(0, p)

EVAL = {"release": "E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz", "md5": "6b5c7f207cffc3b7eb3cd527fd433599",
        "sidecar": "E:/refcv8_eval_kit/data/refcv8_clip_clock_sidecar_ts33.jsonl",
        "dumps": {s: f"C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s{s}" for s in (0, 1)},
        "windows": "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/windows_s0.json"}
FRESH = {"release": "F:/l1c/kit_fresh/data/v9/v9_labels_eval.npz", "md5": "cc06d03a453f87f778e7555d5607a5aa",
         "sidecar": "F:/l1c/kit_fresh/data/clock_sidecar.jsonl", "dumps": {0: "F:/l1c/tree/step48852ema_fresh/dump_s0"},
         "windows": "F:/l1c/kit_fresh/windows_s0.json"}
RC_VARIANT, LAT_VARIANT = "A50", "a"            # trainer argparse default (--r8-rc-variant A50, refc_v3_train.py:12281); argv --r8-v9-lat-variant a
IGNORE = -100
LON6_V7 = (0, 1, 3, 4, 5, 7)
LON7 = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
LON9_EXACT_WITHOUT_LEAD = (0, 1, 2)           # HOLD, CREEP, STOP: assigned before the FOLLOW test, never partial (build_v9_labels.py:933-944)
RAW_OFFSET = 2                                 # n_stack 3 -> raw = provider + 2 (release row_key; refc_v3_train.py:4290-4294)


def sha12(cid: str) -> str:
    return hashlib.sha256(cid.encode("utf-8")).hexdigest()[:12]


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def trainer_lon_targets(kit: dict, seed: int, emulate_no_lead: bool = False, k_shift: int = 0, release: str | None = None, md5: str | None = None) -> dict:
    """Per-window `lon_v7` of the dump's windows (dump order) by the TRAINER'S function.  Returns arrays + the alignment record."""
    from tanitad.data.v2_dataset import stable_episode_id
    from tanitad.data import clip_clock as CC
    from tanitad.train import refcv8_train as r8train
    kit = dict(kit, release=(release or kit["release"]), md5=(md5 or kit["md5"]))
    join = r8train.load_v9_join(kit["release"], expect_md5=kit["md5"], rc_variant=RC_VARIANT, lat_variant=LAT_VARIANT)
    rel = join.rel
    table = CC.read_clip_clock_sidecar(kit["sidecar"]).table
    dump = Path(kit["dumps"][seed])
    ex = np.load(dump / "refcv6_extras.npz")
    jx = json.load(open(dump / "refcv6_extras.npz.json", encoding="utf-8"))
    idx2s12 = {int(k): str(v) for k, v in jx["clip_sha12_by_index"].items()}
    # sha12 -> sid from the RAW ids (the trainer's stable_episode_id), cross-checked against the release's own clip table
    wins = json.load(open(kit["windows"], encoding="utf-8"))
    s12_sid = {sha12(c): stable_episode_id(c) for c in wins.keys()}
    t_by_s12 = {sha12(c): sorted(int(t) for t in v) for c, v in wins.items()}
    z = np.load(kit["release"], allow_pickle=False)
    rel_s12 = {str(s): int(d) for s, d in zip(z["clip__sha12"], z["clip__sid"])}
    chk = {"windows_sha12_eq_dump_sha12": set(s12_sid) == set(idx2s12.values()),
           "release_sid_eq_stable_episode_id": all(rel_s12.get(s) == d for s, d in s12_sid.items()),
           "n_clips_dump": len(set(idx2s12.values())), "n_clips_release": len(rel_s12)}
    ci, ws, lab_b = ex["clip_index"].astype(np.int64), ex["ws"].astype(np.int64), ex["lon_v7"].astype(np.int64)
    # the dump's ws vs the windows json's t: ws = t + W - 1 (the window NOW in provider rows)
    dW = set()
    for c in np.unique(ci):
        s = idx2s12[int(c)]
        w_ = np.sort(ws[ci == c])
        tt = np.asarray(t_by_s12[s])
        if len(w_) == len(tt):
            dW |= set((w_ - tt).tolist())
        else:
            dW.add(("len_mismatch", int(len(w_)), int(len(tt))))
    chk["ws_minus_t"] = sorted(str(x) for x in dW)
    lon = np.full(len(ws), IGNORE, np.int64)
    lon_cls9 = np.full(len(ws), IGNORE, np.int64)
    g3_excl = np.zeros(len(ws), bool)
    n_refused = 0
    for i in range(len(ws)):
        s = idx2s12[int(ci[i])]
        sid = s12_sid[s]
        k = int(ws[i]) + RAW_OFFSET + int(k_shift)
        if sid not in table:                                      # G3: eval clip with no MEASURED clock -> tactical targets excluded
            g3_excl[i] = True
            continue
        g0, dt = table[sid]
        now_s = float(g0) + k * float(dt)                         # refc_v3_train.py:4315 (t + w - 1 == ws)
        if k_shift and not (0 <= k - rel.clips[sid][2] < rel.clips[sid][1]):
            n_refused += 1                                        # mutation arm: a shifted k past the clip end is a refused join
            continue
        try:
            it = join.item(sid, k, now_s)
        except Exception:      # noqa: BLE001  (a refused join would have killed the trainer at launch; counted, never guessed)
            n_refused += 1
            continue
        lon[i] = int(it["v9_lon"])
        r = join.row(sid, k, now_s)
        lon_cls9[i] = int(rel.rows["lon_cls"][r])
    out = {"lon": lon, "lon_battery": lab_b, "g3_excluded": g3_excl, "lon_cls9": lon_cls9, "clip_index": ci, "ws": ws,
           "checks": {**chk, "n_windows": int(len(ws)), "n_join_refused": n_refused, "n_g3_excluded_windows": int(g3_excl.sum()),
                      "n_g3_excluded_clips": int(len(set(ci[g3_excl].tolist())))},
           "release": {"file": kit["release"], "md5": md5_file(kit["release"]), "manifest_has_lead_clips": int(z["clip__has_lead"].sum()),
                       "n_clips": int(len(z["clip__has_lead"]))}}
    if emulate_no_lead:
        nl = np.where(np.isin(lon_cls9, LON9_EXACT_WITHOUT_LEAD), lon, IGNORE)
        nl[g3_excl] = IGNORE
        out["lon_no_lead"] = nl
    return out


def hist(a) -> dict:
    u, c = np.unique(np.asarray(a), return_counts=True)
    return {("IGNORE" if int(k) == IGNORE else f"{int(k)} {LON7[int(k)]}"): int(n) for k, n in zip(u, c)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="C:/Users/Admin/refcv8_pf/refcv9_lonhead/result/K_LABEL.json")
    a = ap.parse_args()
    t0 = time.time()
    res = {"tool": "lh_klabel.py", "script_sha256": sha256_file(__file__), "generated": time.strftime("%FT%T%z"),
           "prereg": {"file": "PREREG_LONHEAD_C.md", "sha256": sha256_file(Path(__file__).resolve().parent.parent / "PREREG_LONHEAD_C.md")},
           "function": "refc_v3_train.py:4646-4674 -> refcv8_train.R8LabelJoin.item (:876-903) -> v9_labels.window_targets(ids='v7')['lon'] (:155-186); "
                       "now_s refc_v3_train.py:4309-4315; G3 exclusion :4255-4260",
           "eval": {}, "fresh": {}}
    okA = True
    for s in (0, 1):
        r = trainer_lon_targets(EVAL, s, emulate_no_lead=True)
        eq = r["lon"] == r["lon_battery"]
        eqB = r["lon_no_lead"] == r["lon_battery"]
        lb = r["lon_battery"]
        lost = (~eqB) & np.isin(lb, LON6_V7)
        rec = {"checks": r["checks"], "release": r["release"],
               "K_LABEL_A": {"n_windows": int(len(eq)), "n_equal": int(eq.sum()), "exact": bool(eq.all()),
                             "battery_label_hist": hist(lb), "function_label_hist": hist(r["lon"])},
               "K_LABEL_B_fresh_recipe_no_lead": {"n_windows": int(len(eqB)), "n_equal": int(eqB.sum()), "exact": bool(eqB.all()),
                                                  "n_battery_LON6_labels_lost_or_changed": int(lost.sum()),
                                                  "n_battery_LON6_labels": int(np.isin(lb, LON6_V7).sum()),
                                                  "lost_by_battery_class": hist(lb[lost]), "no_lead_label_hist": hist(r["lon_no_lead"])}}
        okA = okA and rec["K_LABEL_A"]["exact"] and r["checks"]["windows_sha12_eq_dump_sha12"] and r["checks"]["release_sid_eq_stable_episode_id"] \
            and r["checks"]["n_join_refused"] == 0
        res["eval"][f"s{s}"] = rec
        print(f"[klabel] eval s{s}: A exact={rec['K_LABEL_A']['exact']} ({rec['K_LABEL_A']['n_equal']}/{len(eq)}); "
              f"B(no-lead recipe) exact={rec['K_LABEL_B_fresh_recipe_no_lead']['exact']} ({int(eqB.sum())}/{len(eqB)}); checks {r['checks']}", flush=True)
    muts = {}
    for sh in (1, 7):                                            # deliberate regressions: k off by one row, and ws read as the window START t
        m = trainer_lon_targets(EVAL, 0, k_shift=sh)
        e_ = m["lon"] == m["lon_battery"]
        muts[f"k_shift_+{sh}"] = {"n_equal": int(e_.sum()), "n_windows": int(len(e_)), "exact": bool(e_.all()), "must_be": False,
                                  "n_join_refused": m["checks"]["n_join_refused"]}
    res["K_LABEL_A_mutations"] = muts
    mut_red = all(not v["exact"] for v in muts.values())
    okA = okA and mut_red
    print(f"[klabel] mutations (must NOT be exact): {muts}", flush=True)
    if os.environ.get("LH_REBUILT_RELEASE"):                    # K-LABEL-C: a release REBUILT with this tree's builder (+ lead pass) on eval139
        rb, rbm = os.environ["LH_REBUILT_RELEASE"], md5_file(os.environ["LH_REBUILT_RELEASE"])
        rc = {}
        for s in (0, 1):
            r = trainer_lon_targets(EVAL, s, release=rb, md5=rbm)
            e_ = r["lon"] == r["lon_battery"]
            rc[f"s{s}"] = {"n_equal": int(e_.sum()), "n_windows": int(len(e_)), "exact": bool(e_.all()), "release_md5": rbm,
                           "n_join_refused": r["checks"]["n_join_refused"], "mismatch_battery_hist": hist(r["lon_battery"][~e_]),
                           "mismatch_rebuilt_hist": hist(r["lon"][~e_])}
        res["K_LABEL_C_rebuilt_release"] = {"release": rb, "builder": "this tree's build_v9_labels.py (the FRESH kit's builder, md5 e6b2a7d4) + "
                                            "lead-pass over the eval kit's b1 agent join", **rc,
                                            "exact_both_seeds": bool(all(rc[f"s{s}"]["exact"] for s in (0, 1)))}
        print(f"[klabel] C (rebuilt release {rbm}): {rc}", flush=True)
    if os.environ.get("LH_REBUILT_RELEASE_D"):                  # K-LABEL-D: the WHOLE raw pipeline (parquets -> b1 join -> lead pass -> build) on eval139
        rb, rbm = os.environ["LH_REBUILT_RELEASE_D"], md5_file(os.environ["LH_REBUILT_RELEASE_D"])
        rd = {}
        for s in (0, 1):
            r = trainer_lon_targets(EVAL, s, release=rb, md5=rbm)
            e_ = r["lon"] == r["lon_battery"]
            zr = np.load(rb, allow_pickle=False)
            s12_lead = {str(x) for x, h in zip(zr["clip__sha12"], zr["clip__has_lead"]) if int(h) == 1}
            jx = json.load(open(Path(EVAL["dumps"][s]) / "refcv6_extras.npz.json", encoding="utf-8"))["clip_sha12_by_index"]
            led = np.array([jx[str(int(c))] in s12_lead for c in r["clip_index"]])
            rd[f"s{s}"] = {"n_equal": int(e_.sum()), "n_windows": int(len(e_)), "exact": bool(e_.all()),
                           "n_windows_on_clips_with_lead": int(led.sum()), "n_equal_on_clips_with_lead": int((e_ & led).sum()),
                           "exact_on_clips_with_lead": bool(e_[led].all()), "n_clips_without_lead": int(len(set(r["clip_index"][~led].tolist()))),
                           "mismatch_battery_hist": hist(r["lon_battery"][~e_]), "mismatch_rebuilt_hist": hist(r["lon"][~e_])}
        res["K_LABEL_D_raw_pipeline"] = {"release": rb, "release_md5": rbm, **rd,
                                         "pipeline": "code/chain_label_pipeline.sh eval139 (obstacle_offline_b1eval parquets -> build_b1_agent_join.py -> lead-pass -> build)"}
        print(f"[klabel] D (raw pipeline {rbm}): {rd}", flush=True)
    if os.environ.get("LH_REBUILT_NOLEAD"):                     # validates the B emulation against the REAL builder run without --lead
        rn, rnm = os.environ["LH_REBUILT_NOLEAD"], md5_file(os.environ["LH_REBUILT_NOLEAD"])
        r = trainer_lon_targets(EVAL, 0, release=rn, md5=rnm)
        em = trainer_lon_targets(EVAL, 0, emulate_no_lead=True)["lon_no_lead"]
        e_ = r["lon"] == em
        eb = r["lon"] == r["lon_battery"]
        res["K_LABEL_B_validated_by_real_no_lead_build"] = {"release": rn, "release_md5": rnm, "n_windows": int(len(e_)),
                                                            "real_build_eq_emulation": int(e_.sum()), "exact": bool(e_.all()),
                                                            "real_no_lead_build_eq_battery": int(eb.sum()), "real_no_lead_label_hist": hist(r["lon"])}
        print(f"[klabel] B validation (real no-lead build vs emulation): {res['K_LABEL_B_validated_by_real_no_lead_build']}", flush=True)
    rf = trainer_lon_targets(FRESH, 0)
    eqf = rf["lon"] == rf["lon_battery"]
    lab = rf["lon"]
    res["fresh"] = {"checks": rf["checks"], "release": rf["release"],
                    "function_eq_fresh_dump_lon_v7": {"n_windows": int(len(eqf)), "n_equal": int(eqf.sum()), "exact": bool(eqf.all())},
                    "label_hist": hist(lab), "n_LON6_labelled": int(np.isin(lab, LON6_V7).sum()),
                    "n_cruise_group_labels (FOLLOW, CRUISE)": int(np.isin(lab, (0, 1)).sum()), "n_ACCELERATE_labels": int((lab == 7).sum()),
                    "n_decel_group_labels (BRAKE_TO, CREEP, HOLD)": int(np.isin(lab, (3, 4, 5)).sum())}
    print(f"[klabel] fresh: function == fresh dump lon_v7 {res['fresh']['function_eq_fresh_dump_lon_v7']}; hist {res['fresh']['label_hist']}", flush=True)
    if os.environ.get("LH_FRESH_DRY_RELEASE"):                  # the fixed recipe on the 5 fresh clips whose obstacle parquet is local (mechanics)
        rp = os.environ["LH_FRESH_DRY_RELEASE"]
        zr = np.load(rp, allow_pickle=False)
        s12_lead = {str(x) for x, h in zip(zr["clip__sha12"], zr["clip__has_lead"]) if int(h) == 1}
        jx = json.load(open(Path(FRESH["dumps"][0]) / "refcv6_extras.npz.json", encoding="utf-8"))["clip_sha12_by_index"]
        d5 = trainer_lon_targets(FRESH, 0, release=rp, md5=md5_file(rp))
        led = np.array([jx[str(int(c))] in s12_lead for c in d5["clip_index"]])
        res["fresh_fixed_recipe_dry_local_clips"] = {"release": rp, "release_md5": md5_file(rp), "n_clips_with_lead": len(s12_lead),
                                                     "n_windows_on_those_clips": int(led.sum()), "label_hist_with_lead": hist(d5["lon"][led]),
                                                     "label_hist_as_built_same_windows": hist(lab[led]),
                                                     "other_windows_identical_to_as_built": bool((d5["lon"][~led] == lab[~led]).all())}
        print(f"[klabel] fresh dry (fixed recipe, local clips): {res['fresh_fixed_recipe_dry_local_clips']}", flush=True)
    okB = all(res["eval"][f"s{s}"]["K_LABEL_B_fresh_recipe_no_lead"]["exact"] for s in (0, 1))
    okD = res.get("K_LABEL_D_raw_pipeline", {})
    okD = bool(okD) and all(okD[f"s{s}"]["exact"] for s in (0, 1))
    okC = res.get("K_LABEL_C_rebuilt_release", {}).get("exact_both_seeds")
    res["K_LABEL"] = {"A_trainer_function_reproduces_battery": bool(okA),
                      "B_fresh_recipe_reproduces_battery": bool(okB),
                      "C_fresh_builder_WITH_lead_pass_reproduces_battery": okC,
                      "PASS": bool(okA and okB),
                      "D_raw_pipeline_parquets_join_lead_build_reproduces_battery": okD,
                      "reading": ("PASS: train and eval targets agree" if (okA and okB) else
                                  "FAIL for the fresh kit AS BUILT: the trainer's function reproduces the battery label (A), but the FRESH release was built "
                                  "WITHOUT the lead pass, and that recipe does NOT reproduce the battery label on eval139 (B). "
                                  + ("The fixed recipe (this tree's builder + the lead pass, C; and the whole raw pipeline parquets -> join -> lead -> build, D) "
                                     "reproduces it EXACTLY, so the fresh targets need only the fresh clips' obstacle.offline parquets."
                                     if (okC and okD) else "The fixed recipe was not shown to reproduce it (C/D)."))}
    res["wall_s"] = round(time.time() - t0, 1)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    print(f"ZZKLABELZZ PASS={res['K_LABEL']['PASS']} A={okA} B={okB} -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
