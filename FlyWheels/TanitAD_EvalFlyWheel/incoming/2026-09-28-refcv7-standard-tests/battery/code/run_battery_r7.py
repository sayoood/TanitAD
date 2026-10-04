"""run_battery_r7.py -- the refcv7 four-family milestone battery, ONE command (SPEC.md §4).

    python run_battery_r7.py --ckpt D:/refcv7_eval_kit/ckpt/ckpt_5000.pt \
        --config D:/refcv7_eval_kit/ckpt/config.json --metrics <metrics copy> [--infer-seeds 0,1]

Ported from the refcv6 package's `run_battery.py` (2026-09-23). Steps (each banks before the next):
  0. SPEC sha256; the dev-box GPU LOCK is taken for the whole battery unless the caller holds it
     (`--lock-held`); every GPU stage is a CHILD process (the parent never holds a CUDA context).
  1. G0 (`g0_refcv7.py`, SPEC §2) when `--metrics` carries the checkpoint's eval row. Anything but
     PASS stops the battery (SPEC §2).
  2. Per inference seed: the T1 roll (`r7_roll` = refcv3_arm.run_dump fed like the trainer) on S2.
  3. Per seed: panel (+ STOP + banked baselines + refcv6@38k when banked, pairing VERIFIED),
     `analyze_refcv3`, cross-model paired families, tactical, S6.
  4. Inference-seed replicate, the bars (SPEC §3.5) with the NOT PROVEN rule, `battery_summary.json`.
Assert on the JSON artifacts, never on an exit code.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
from tanitad.eval import refcv7_loader as L  # noqa: E402  (the landed loader, blob a8eb3e7e)

L.bootstrap()
import numpy as np  # noqa: E402,F401
import torch  # noqa: E402
import r7_roll as RR  # noqa: E402
import r7_panel as P  # noqa: E402
import gpu_gate  # noqa: E402
import gpu_lock  # noqa: E402

KIT_EVAL = str(L.KIT / "data/refcv6-b1-416x1024-eval139")
KIT_LABELS = str(L.KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
#: SPEC §3.5 -- copied from SPEC_REFCV7 §3. `{s}` = the SAME inference seed's baseline.
BARS = [
    {"id": "BAR-R7-1", "surface": "S2", "a": "ha0_ext", "b": "os", "metric": "ade_m",
     "statement": "os - echo (ADE 0-2 s) < 0, separated, at both inference seeds"},
    {"id": "BAR-R7-2", "surface": "S2", "a": "b_refcv6_38k_s{s}", "b": "os", "metric": "ade_m",
     "statement": "os - refcv6@38k < 0, separated, on the same windows (same inference seed)"},
    {"id": "BAR-R7-3", "surface": "S6", "a": "ha0_ext", "b": "os", "metric": "ade_m",
     "statement": "(non-regression) at 6 s, os - echo < 0, separated"},
]
PAIRS = [("ha0_ext", "os", "os_minus_ha0ext"), ("ha", "os", "os_minus_ha"),
         ("ha0", "os", "os_minus_ha0__CV"), ("stop", "os", "os_minus_stop"),
         ("b_refcv6_38k_s0", "os", "os_minus_refcv6_38k_s0"),
         ("b_refcv6_38k_s1", "os", "os_minus_refcv6_38k_s1"),
         ("b_refcv4b", "os", "os_minus_refcv4b"), ("b_refcv5v2_s0", "os", "os_minus_refcv5v2_s0"),
         ("os", "os_navzero", "navzero_minus_os"), ("os", "os_navshuf", "navshuf_minus_os"),
         ("os", "os_navflip", "navflip_minus_os"),
         ("os", "os_vmaxzero", "vmaxzero_minus_os"), ("os", "os_filteroff", "filteroff_minus_os"),
         ("ha0_ext", "os_filteroff", "filteroff_minus_ha0ext"),
         ("ha0_ext", "b_refcv6_38k_s0", "refcv6_38k_s0_minus_ha0ext"),
         ("ha0_ext", "b_refcv6_38k_s1", "refcv6_38k_s1_minus_ha0ext"),
         ("ha0_ext", "b_refcv4b", "refcv4b_minus_ha0ext")]


def yield_gpu(a, stage: str, summary: dict, bank) -> None:
    """The evening yield rule (README; not a SPEC change): after a GPU stage that ends after 20:00,
    release the caller's lock, sleep >= 90 s, re-acquire (`gpu_yield.py`). No-op without --yield-job."""
    if not (a.lock_held and a.yield_job and a.yield_pid is not None):
        return
    r = subprocess.run([sys.executable, str(HERE / "gpu_yield.py"), "--job", a.yield_job, "--pid",
                        str(a.yield_pid), "--stage", stage, "--log", a.yield_log],
                       capture_output=True, text=True)
    try:
        rec = json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:                                      # noqa: BLE001 -- recorded
        rec = {"stage": stage, "unparsed": (r.stdout or "")[-300:], "stderr": (r.stderr or "")[-300:]}
    summary.setdefault("gpu_yields", []).append(rec)
    bank()
    if rec.get("reacquired") is False:
        raise SystemExit(f"[battery] could not re-acquire the GPU lock after yielding at {stage}")


def pair_name(a, b):
    return next((n for x, y, n in PAIRS if x == a and y == b), None)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gate_wait(out_json, min_free_ram_gb=6.0):
    """Between GPU stages: no OTHER python compute app on the card (the lock keeps new jobs out;
    this catches one that ignores it). ⛔ The parent never holds a CUDA context and passes its
    own pid, so it can never wait on itself (the refcv6 battery's 2026-09-24 self-deadlock)."""
    t0 = time.time()
    while True:
        g = gpu_gate.gate(self_pid=os.getpid(), min_free_ram_gb=min_free_ram_gb)
        if g["ok"]:
            json.dump(g, open(out_json, "w"), indent=1)
            return g
        print(f"[battery] GPU gate WAIT: {json.dumps(g)}", flush=True)
        if time.time() - t0 > 12 * 3600:
            json.dump(g, open(out_json, "w"), indent=1)
            raise SystemExit("[battery] GPU gate did not pass within 12 h")
        time.sleep(60)


def roll_one(ckpt, config, seed, dump_dir, windows, log, device="cuda", frame_memo=True, full=True):
    """`full` = every SPEC §3.2 arm (inference seed 0); otherwise `os` + the model-free controls only
    (SPEC AMENDMENT A3: the bars read only `os`; the perturbation / sensitivity arms are seed-0 reads)."""
    st = RR.install_patches(config_path=config, windows=windows, frame_memo=frame_memo,
                            obedience=full, vmaxzero=full, with_navflip=full, filteroff=full)
    R = RR.ra3()
    os.makedirs(dump_dir, exist_ok=True)
    argv = ["--ckpt", ckpt, "--config", config, "--episodes", KIT_EVAL, "--labels", KIT_LABELS,
            "--nav-source", "v72", "--grid", "2s", "--action-units", "steer",
            "--window-stride", "1", "--with-oracle-sel",
            "--dump-dir", dump_dir, "--dump-only", "--out", os.path.join(dump_dir, "_unused.json"),
            "--device", device, "--infer-seed", str(seed), "--seed", "0", "--arm", "refcv7",
            "--lru", "8"] + (["--with-navflip"] if full else ["--no-navshuf", "--no-navzero"])
    t0 = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    R.main(argv)
    man = json.load(open(os.path.join(dump_dir, "manifest.json"), encoding="utf-8"))
    ex_path = os.path.join(dump_dir, "refcv6_extras.npz")
    meta = RR.save_extras(st, ex_path, [e for e in _clip_ids_by_index(KIT_EVAL)])
    mr = st["model_record"]
    rec = {"seed": seed, "arm_set": "full (SPEC §3.2)" if full else "os + model-free controls (SPEC A3)",
           "dump_dir": dump_dir, "wall_s": round(time.time() - t0, 1),
           "n_windows": man["grid"]["n_windows"], "n_episodes": man["grid"]["n_episodes"],
           "skipped": man["grid"]["n_windows_skipped"], "extras": meta,
           "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                             if torch.cuda.is_initialized() else None),
           "model_record": {k: mr.get(k) for k in ("state_dict", "param_breakdown",
                                                   "anchor_file_vs_ckpt_buffers",
                                                   "trunk_memory_levers_built", "departures",
                                                   "declared_vs_built")},
           "window_restriction": st.get("window_restriction"),
           "label_clock": (st.get("dataset_record") or {}).get("label_clock"),
           "requires_grad_state": "loader (all False) -- see G0's requires_grad control",
           "frame_memo": (None if st.get("memo") is None else
                          {"hits": st["memo"].hits, "misses": st["memo"].misses})}
    del st
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return rec


def _clip_ids_by_index(kit_eval):
    from tanitad.data.v2_dataset import load_or_build_manifest
    return [str(c) for c in load_or_build_manifest(kit_eval, verbose=False)["clip_id"]]


def bar_verdicts(cells_by_seed: dict, s6_by_seed: dict, is_milestone: bool, floor_m,
                 pending: tuple = ()) -> list:
    """SPEC §3.5: PASS / FAILED / NOT PROVEN / NOT EVALUABLE per bar; every seed must pass."""
    out = []
    for bar in BARS:
        v = dict(bar)
        per = {}
        src = cells_by_seed if bar["surface"] == "S2" else s6_by_seed
        for s, cells in src.items():
            a = bar["a"].format(s=s)
            nm = pair_name(a, bar["b"])
            c = ((((cells or {}).get("pairs") or {}).get(nm) or {}).get("families") or {})
            c = (c.get("ADE") or {}).get(bar["metric"])
            if c is None:
                per[s] = {"status": "ABSENT", "pair": nm}
                continue
            sep_ok = (c["delta"] < 0) and (c["hi"] < 0)
            within = (floor_m is not None and abs(float(c["delta"])) <= 2.0 * float(floor_m))
            per[s] = {"pair": nm, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"],
                      "separated": c["separated"], "n_windows": c.get("n_windows"),
                      "n_episodes": c.get("n_episodes"), "pass_separated": bool(sep_ok),
                      "within_2x_inference_seed_floor": bool(within)}
        v["per_inference_seed"] = per
        v["inference_seed_floor_m"] = floor_m
        if not is_milestone:
            v["verdict"] = "NOT EVALUATED (pipeline validation checkpoint; SPEC §6)"
        elif bar["id"] in pending and any(x.get("status") == "ABSENT" for x in per.values()):
            v["verdict"] = ("PENDING (the refcv6@38k S2 rolls run LAST in the milestone chain; "
                            "SPEC A4)")
        elif len(per) < 2 or any(x.get("status") == "ABSENT" for x in per.values()):
            v["verdict"] = "NOT EVALUABLE (a cell is absent, or fewer than 2 inference seeds)"
        elif not all(x["pass_separated"] for x in per.values()):
            v["verdict"] = "FAILED"
        elif floor_m is None:
            v["verdict"] = "NOT PROVEN (no inference-seed floor)"
        elif any(x["within_2x_inference_seed_floor"] for x in per.values()):
            v["verdict"] = "NOT PROVEN (|delta| within 2x the inference-seed floor)"
        else:
            v["verdict"] = "PASS (single training seed; training-seed floor unmeasured)"
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="D:/refcv7_eval_kit/ckpt/config.json")
    ap.add_argument("--infer-seeds", default="0,1")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-root", default="D:/refcv7_eval_kit/battery",
                    help="WORK dir (raw clip ids live there); bank_tag.py banks a sanitized copy")
    ap.add_argument("--metrics", default=None, help="metrics.jsonl copy for G0")
    ap.add_argument("--no-g0", action="store_true")
    ap.add_argument("--g0-json", default=None, help="reuse an EXISTING G0 artifact (md5 must match)")
    ap.add_argument("--g0-seeds", default=",".join(str(i) for i in range(24)),
                    help="SPEC AMENDMENT A5: 24 inference seeds (0..23); seeds 0..7 also give the "
                         "as-registered and A2 verdicts")
    ap.add_argument("--min-milestone-step", type=int, default=5000)
    ap.add_argument("--skip-roll", action="store_true", help="re-analyse existing dumps")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lock-held", action="store_true", help="the caller holds the GPU lock")
    ap.add_argument("--yield-job", default=None, help="the CALLER's lock job: yield it after 20:00")
    ap.add_argument("--yield-pid", type=int, default=None)
    ap.add_argument("--yield-log", default="D:/refcv7_eval_kit/chain/gpu_yields.jsonl")
    ap.add_argument("--max-clips", type=int, default=0, help="SMOKE ONLY: restrict S2 to N clips")
    ap.add_argument("--max-windows-per-clip", type=int, default=0, help="SMOKE ONLY")
    # Master Mind ruling 2026-10-04 (G0-50,400): the REGISTERED A6 text is the record; the coded A6 judge
    # omitted A6 from the A2 low-support tuple. A coded A6 FAIL may be overridden ONLY by a corrected-judge
    # text verdict for the SAME checkpoint, re-verified here; both records are stamped on every output.
    ap.add_argument("--g0-text-verdict", default=None,
                    help="g0_A6_text.json (code/g0_rejudge_a6_text.py) for THIS checkpoint; STAMPED, verified")
    ap.add_argument("--g0-text-ruling", default=None, help="the ruling text the override rests on (required "
                                                            "with --g0-text-verdict)")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    step = int(ck.get("step"))
    del ck
    tag = a.tag or f"step{step}"
    root = Path(a.out_root) / tag
    root.mkdir(parents=True, exist_ok=True)
    spec = HERE.parent / "SPEC.md"
    summary = {"tool": "run_battery_r7.py", "tag": tag, "ckpt": a.ckpt,
               "ckpt_md5": L.md5_file(a.ckpt), "step": step, "config": a.config,
               "config_md5": L.md5_file(a.config),
               "spec_sha256": sha256(spec) if spec.exists() else None,
               "is_milestone": step >= a.min_milestone_step,
               "tanitad_file": __import__("tanitad").__file__,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": {}}

    def bank():
        json.dump(summary, open(root / "battery_summary.json", "w", encoding="utf-8"),
                  indent=1, default=str)

    bank()
    lock_job = f"refcv7-battery-{tag}"
    if not a.lock_held:
        lk = gpu_lock.acquire(lock_job, os.getpid(), 12 * 3600)
        summary["stages"]["gpu_lock"] = lk
        bank()
        if not lk.get("ok"):
            raise SystemExit("[battery] GPU lock not acquired within 12 h")
    else:
        summary["stages"]["gpu_lock"] = {"held_by_caller": gpu_lock.read_lock()}
    try:
        _run(a, summary, root, bank)
    finally:
        if not a.lock_held:
            summary["stages"]["gpu_lock_release"] = gpu_lock.release(lock_job)
        summary["parent_cuda_initialized"] = bool(torch.cuda.is_initialized())
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        bank()


def a7_stage_summary(g0: dict) -> dict:
    """The SPEC A7 keys of the battery summary's G0 stage, copied from the G0 artifact (nothing recomputed):
    the A7 verdict, the A7.2 population guard (n members, N_in, N_num, bound), the M5 probe, and the REPORTED-NEVER-
    GATING A7.3 pack banking and A7.4 seed-draw reports. Absent in a pre-A7 artifact -> every value None."""
    v7 = g0.get("verdict_A7") or {}
    gd, m5, a7 = v7.get("a7_lowsupport") or {}, v7.get("m5") or {}, g0.get("a7") or {}
    return {"G0_A7": v7.get("G0"),
            "reasons_A7": (v7.get("reasons") or [])[:20],
            "a7_registration": v7.get("registration"),
            "a7_lowsupport": {k: gd.get(k) for k in ("status", "n_members", "N_in", "N_num", "bound", "why")},
            "a7_m5": {k: m5.get(k) for k in ("status", "detected", "N_M5", "N_num", "bound", "blind_spot",
                                             "restoration_bit_exact", "n_gating_detection_out")},
            "a7_reports": {"packs": (a7.get("packs") or {}).get("status"),
                           "seed_group": (a7.get("seed_group") or {}).get("VERDICT")
                           or (a7.get("seed_group") or {}).get("status"),
                           "seed_group_F": ((a7.get("seed_group") or {}).get("test_full") or {}).get("F"),
                           "seed_group_p": ((a7.get("seed_group") or {}).get("test_full") or {}).get("p_one_sided"),
                           "seed_draw_correlation": [
                               {"earlier_step": p.get("earlier_step"), "n_cells": p.get("n_cells"),
                                "r_cell": p.get("r_cell")}
                               for p in ((a7.get("seed_draw_correlation") or {}).get("pairs") or [])]
                           or (a7.get("seed_draw_correlation") or {}).get("status")}}


def g0_text_override(a, g0: dict, ckpt_md5: str) -> dict:
    """Master Mind ruling 2026-10-04: a coded G0-A6 FAIL proceeds ONLY on a corrected-judge verdict of the
    REGISTERED A6 text for the SAME checkpoint. Everything is re-verified here on CONTENT -- the text file's
    own claims are not trusted: the corrected judge in use is re-run on the banked G0 artifact and must
    reproduce the file's PASS and its (empty) reason list. Any gap -> SystemExit (fail closed)."""
    if not a.g0_text_ruling:
        raise SystemExit("[battery] --g0-text-verdict needs --g0-text-ruling (the ruling it rests on)")
    p = Path(a.g0_text_verdict)
    if not p.exists():
        raise SystemExit(f"[battery] text verdict {p} does not exist -- STOP (SPEC §2)")
    t = json.load(open(p, encoding="utf-8"))
    v6t = t.get("verdict_A6_text") or {}
    bad = []
    if t.get("REFUSED"):
        bad.append(f"the text re-judge REFUSED: {t['REFUSED']}")
    if t.get("ckpt_md5") != ckpt_md5 or int(t.get("step", -1)) != int(g0.get("step", -2)):
        bad.append("text verdict is for another checkpoint / step")
    if (g0.get("verdict") or {}).get("amendment") != "A6":
        bad.append("the coded gate is not A6: the text override covers ONLY the A6-item-7 judge defect")
    if v6t.get("G0") != "PASS" or v6t.get("reasons"):
        bad.append(f"text verdict is {v6t.get('G0')} with reasons {str(v6t.get('reasons'))[:200]}")
    ctl = t.get("controls_other_verdicts_unchanged") or {}
    if not ctl or any(c.get("ok") is not True for c in ctl.values()):
        bad.append(f"the corrected judge's controls did not all reproduce the banked verdicts: {ctl}")
    if not (t.get("compare_with_readonly_reconstruction") or {}).get("agree"):
        bad.append("the text verdict does not agree with the read-only reconstruction")
    if not ((v6t.get("mutation_detection") or {}).get("m1") or {}).get("detected"):
        bad.append("M1 not detected under the text verdict")
    import g0_refcv7 as G                               # the CURRENT judge, re-run on the banked artifact
    if "A6" not in getattr(G, "A2_LOWSUPPORT_AMENDS", ()):
        bad.append("the judge in use does not carry the A6-item-7 fix")
    else:
        inrun = {k: r.get("inrun") for k, r in (g0["verdict"].get("terms") or {}).items()}
        by_seed = {int(s): {"row": v["row"], "buffers_unchanged": v.get("buffers_unchanged", True)}
                   for s, v in g0["by_seed"].items()}
        v_now = G.judge(inrun, by_seed, g0, amend="A6")
        if v_now["G0"] != "PASS" or v_now["reasons"] != v6t.get("reasons"):
            bad.append(f"the judge in use re-reads {v_now['G0']} ({v_now['reasons'][:3]}), not the file's PASS")
        mut_now = {m: d["n_terms_out"] for m, d in v_now["mutation_detection"].items()}
        mut_t = {m: d.get("n_terms_out") for m, d in (v6t.get("mutation_detection") or {}).items()}
        if mut_now != mut_t:
            bad.append(f"mutation counts differ: now {mut_now} vs file {mut_t}")
    if bad:
        raise SystemExit(f"[battery] G0 text override REFUSED: {bad}")
    return {"gate_source": "REGISTERED A6 TEXT, corrected judge (Master Mind ruling 2026-10-04)",
            "ruling": a.g0_text_ruling, "text_verdict_file": str(p), "text_verdict_sha256": sha256(p),
            "record_lines": t.get("record_lines"), "G0_A6_text": v6t.get("G0"),
            "mutation_detection_text": {m: d.get("n_terms_out") for m, d in
                                        (v6t.get("mutation_detection") or {}).items()},
            "re_verified_by_judge_in_use": True}


def _run(a, summary, root, bank):
    # ---- 1. G0 --------------------------------------------------------------------------- #
    if a.g0_json or not a.no_g0:
        g0_json = Path(a.g0_json) if a.g0_json else root / "g0.json"
        if not a.g0_json:
            if not a.metrics:
                raise SystemExit("[battery] G0 needs --metrics (or pass --no-g0 and say why)")
            gate_wait(str(root / "gate_g0.json"))
            cmd = [sys.executable, str(HERE / "g0_refcv7.py"), "--ckpt", a.ckpt, "--config",
                   a.config, "--metrics", a.metrics, "--seeds", a.g0_seeds, "--out", str(g0_json)]
            with open(root / "g0.log", "w", encoding="utf-8") as lf:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            yield_gpu(a, "g0", summary, bank)
        if not g0_json.exists():
            summary["stages"]["g0"] = {"G0": "NO ARTIFACT", "log": str(root / "g0.log")}
            bank()
            raise SystemExit("[battery] G0 wrote no JSON -- STOP (SPEC §2)")
        g0 = json.load(open(g0_json, encoding="utf-8"))
        if g0.get("ckpt_md5") != summary["ckpt_md5"]:
            raise SystemExit("[battery] the G0 artifact is for another checkpoint")
        v = g0["verdict"]
        summary["stages"]["g0"] = {
            "G0": v["G0"], "amendment": v.get("amendment"),
            "G0_as_registered": (g0.get("verdict_as_registered") or {}).get("G0"),
            "G0_A2": (g0.get("verdict_A2") or {}).get("G0"),
            "reasons_A2": ((g0.get("verdict_A2") or {}).get("reasons") or [])[:20],
            "n_seeds": len(g0.get("by_seed") or {}),
            "reasons_as_registered": ((g0.get("verdict_as_registered") or {}).get("reasons") or [])[:20],
            "reasons": v["reasons"][:40], "by_class_counts": v["by_class_counts"],
            "medians": v["medians"],
            "mutation_detection": {m: {k: d.get(k) for k in ("detected", "n_terms_out", "how")}
                                   for m, d in v["mutation_detection"].items()},
            "blind_spots_named": v.get("blind_spots_named"),
            "wrapper_clause": (g0.get("wrapper_control") or {}).get("clause"),
            "requires_grad_control": g0.get("requires_grad_control"), "json": str(g0_json),
            # SPEC A6 (DRAFT until registered): always reported beside A5; gates only if registered
            "G0_A5": (g0.get("verdict_A5") or {}).get("G0"),
            "reasons_A5": ((g0.get("verdict_A5") or {}).get("reasons") or [])[:20],
            "G0_A6": (g0.get("verdict_A6") or {}).get("G0"),
            "reasons_A6": ((g0.get("verdict_A6") or {}).get("reasons") or [])[:20],
            "a6_registration": (g0.get("verdict_A6") or {}).get("registration"),
            "a6_rescued": (g0.get("verdict_A6") or {}).get("a6_rescued"),
            "a6_threshold_terms": (g0.get("verdict_A6") or {}).get("a6_threshold_terms"),
            "a6_medians": (g0.get("verdict_A6") or {}).get("medians"),
            # SPEC A7 (registered 2026-10-04T17:11:57+02:00): reported AFTER A6; it is the gate for a G0 that started
            # after the registration file was written (g0["verdict"] is then the A7 verdict)
            **a7_stage_summary(g0)}
        bank()
        if v["G0"] != "PASS":
            if not a.g0_text_verdict:
                raise SystemExit(f"[battery] G0 = {v['G0']} -- STOP (SPEC §2)")
            ov = g0_text_override(a, g0, summary["ckpt_md5"])      # raises unless fully re-verified
            summary["stages"]["g0"]["text_override"] = ov
            summary["G0_RECORD"] = ov["record_lines"]
            print("[battery] G0 coded FAIL overridden by the REGISTERED A6 text verdict: "
                  + " | ".join(ov["record_lines"] or []), flush=True)
            bank()
    else:
        summary["stages"]["g0"] = {"G0": "SKIPPED (--no-g0)"}
        bank()
    # ---- 2. rolls ------------------------------------------------------------------------ #
    kit_ids = set(_clip_ids_by_index(KIT_EVAL))
    windows = RR.s2_window_list(P.BASELINES["b_refcv4b"], kit_ids)
    if a.max_clips:
        windows = {c: windows[c] for c in sorted(windows)[:a.max_clips]}
    if a.max_windows_per_clip:
        windows = {c: w[:a.max_windows_per_clip] for c, w in windows.items()}
    if a.max_clips or a.max_windows_per_clip:
        summary["SMOKE_RESTRICTED_S2"] = True
    summary["S2"] = {"n_clips": len(windows), "n_windows": sum(len(v) for v in windows.values()),
                     "source": P.BASELINES["b_refcv4b"]}
    seeds = [int(s) for s in a.infer_seeds.split(",") if s.strip()]
    rolls = {}
    for s in seeds:
        d = str(root / f"dump_s{s}")
        rj = root / f"roll_s{s}.json"
        if a.skip_roll and os.path.exists(os.path.join(d, "manifest.json")) and rj.exists():
            rolls[s] = json.load(open(rj, encoding="utf-8"))
            continue
        gate_wait(str(root / f"gate_roll_s{s}.json"))
        wj = root / f"windows_s{s}.json"
        json.dump(windows, open(wj, "w", encoding="utf-8"))
        cmd = [sys.executable, str(HERE / "roll_seed_r7.py"), "--ckpt", a.ckpt, "--config",
               a.config, "--seed", str(s), "--dump-dir", d, "--windows-json", str(wj),
               "--device", a.device, "--out-json", str(rj)]
        if s != seeds[0]:
            cmd.append("--os-only")                           # SPEC AMENDMENT A3
        with open(root / f"roll_s{s}.log", "w", encoding="utf-8") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if not rj.exists():
            summary["stages"][f"roll_s{s}"] = {"status": "NO ARTIFACT",
                                               "log": str(root / f"roll_s{s}.log")}
            bank()
            raise SystemExit(f"[battery] the seed-{s} roll wrote no record -- see roll_s{s}.log")
        rolls[s] = json.load(open(rj, encoding="utf-8"))
        summary["stages"][f"roll_s{s}"] = rolls[s]
        bank()
        yield_gpu(a, f"roll_s{s}", summary, bank)
    # ---- 3. panels ----------------------------------------------------------------------- #
    bases = P.baselines_available()
    summary["baselines_used"] = bases
    summary["baselines_missing"] = sorted(set(P.REFCV6_38K) - set(bases))
    cells, s6cells = {}, {}
    for s in seeds:
        d = rolls[s]["dump_dir"]
        pd = str(root / f"panel_s{s}")
        prec = P.build_panel_dump(d, pd, os.path.join(d, "refcv6_extras.npz"), baselines=bases)
        summary["stages"][f"panel_s{s}"] = {"pairing": prec["pairing"],
                                            "void_gates_3_4": prec["void_gates_3_4"],
                                            "arms": prec["arms"], "n_windows": prec["n_windows"]}
        bank()
        an = P.analyze(pd, str(root / f"analysis_s{s}.json"), KIT_LABELS, n_boot=a.n_boot)
        vg = P.void_gate_checks(pd, an)
        json.dump(vg, open(root / f"void_gates_s{s}.json", "w", encoding="utf-8"), indent=1)
        summary["stages"][f"void_gates_s{s}"] = vg
        cells[s] = P.cross_paired(pd, PAIRS, n_boot=a.n_boot)
        json.dump(cells[s], open(root / f"cross_paired_s{s}.json", "w", encoding="utf-8"), indent=1)
        man = json.load(open(os.path.join(pd, "manifest.json"), encoding="utf-8"))
        clip_eid = {int(e["episode_index"]): int(e["file_index"]) for e in man["episodes"]}
        tac = P.tactical_v6(os.path.join(d, "refcv6_extras.npz"), clip_eid, n_boot=a.n_boot)
        json.dump(tac, open(root / f"tactical_v6_s{s}.json", "w", encoding="utf-8"), indent=1)
        accp = P.acceptance(an, os.path.join(d, "refcv6_extras.npz"), clip_eid, pd,
                            n_boot=a.n_boot)
        accp["tzero"] = P.tzero_from_tactical(tac)
        accp["note"] = ("refcv6's FROZEN acceptance instruments read on refcv7 as INFORMATIVE "
                        "(no refcv7 bar)")
        json.dump(accp, open(root / f"acceptance_s{s}.json", "w", encoding="utf-8"), indent=1,
                  default=str)
        s6d = str(root / f"s6_s{s}")
        s6rec = P.build_s6_dump(d, s6d, baselines=bases)
        P.analyze(s6d, str(root / f"analysis_s6_s{s}.json"), KIT_LABELS, n_boot=a.n_boot)
        s6cells[s] = P.cross_paired(s6d, PAIRS, n_boot=a.n_boot)
        json.dump(s6cells[s], open(root / f"cross_paired_s6_s{s}.json", "w", encoding="utf-8"),
                  indent=1)
        summary["stages"][f"analysis_s{s}"] = {"s6": s6rec,
                                               "tflip": accp.get("tflip", {}).get("verdict"),
                                               "obedience": accp.get("obedience", {}).get("verdict")}
        bank()
    floor = None
    if len(seeds) >= 2:
        rep = P.seed_replicate(str(root / f"panel_s{seeds[0]}"), str(root / f"panel_s{seeds[1]}"),
                               n_boot=a.n_boot)
        json.dump(rep, open(root / "inference_seed_replicate.json", "w", encoding="utf-8"), indent=1)
        floor = abs(float(rep["families"]["ADE"]["ade_m"]["delta"]))
        summary["inference_seed_replicate"] = {"max_abs_path_diff_m": rep["max_abs_path_diff_m"],
                                               "ade": rep["families"]["ADE"]["ade_m"],
                                               "floor_m": floor}
        for s in seeds:
            ap_ = root / f"analysis_s{s}.json"
            an_ = json.load(open(ap_, encoding="utf-8"))
            an_["inference_seeds"] = seeds
            an_["seed_floor_m"] = floor
            an_["inference_seed_replicate"] = {"file": "inference_seed_replicate.json",
                                               "ade_m": rep["families"]["ADE"]["ade_m"],
                                               "max_abs_path_diff_m": rep["max_abs_path_diff_m"]}
            json.dump(an_, open(ap_, "w", encoding="utf-8"), indent=1, default=str)
    for s in seeds:
        cj = root / f"criteria_s{s}.json"
        with open(root / f"criteria_s{s}.log", "w", encoding="utf-8") as lf:
            subprocess.run([sys.executable, str(L.REPO / "tools" / "criteria_check.py"),
                            str(root / f"analysis_s{s}.json"), "--json", str(cj)],
                           stdout=lf, stderr=subprocess.STDOUT, cwd=str(L.REPO))
        if cj.exists():
            cr = json.load(open(cj, encoding="utf-8"))
            arts = cr.get("artifacts") or {}
            summary["stages"][f"criteria_s{s}"] = {
                "registry_version": cr.get("registry_version"),
                "per_artifact": {os.path.basename(str(k)): {x: v.get(x) for x in
                                                           ("scope", "tier", "n_violations",
                                                            "n_work_items")}
                                 for k, v in arts.items()}}
        bank()
    pend = ("BAR-R7-2",) if summary.get("baselines_missing") else ()
    summary["bars"] = bar_verdicts(cells, s6cells, summary["is_milestone"], floor, pending=pend)
    summary["strategic"] = {"status": "NOT APPLICABLE", "n": 0,
                            "reason": "the strategic layer is OFF (--no-strategic); its route loss "
                                      "is gated off in training, so the route head is untrained"}
    bank()
    print(json.dumps({"tag": summary["tag"],
                      "bars": [(b["id"], b["verdict"]) for b in summary["bars"]]}, indent=1))


if __name__ == "__main__":
    main()
