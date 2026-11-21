"""MM-E19 — the ONE-COMMAND read for the k=60 horizon arm.

Pre-registration: ``Project Steering/PREREG_MM_E19_K60_HORIZON.md``. This runner
PREPARES THE NUMBERS the committed criteria consume and emits ONE JSON.
⛔ EVERY criterion verdict field is EMPTY BY DESIGN — the ORCHESTRATOR applies
the pre-registered criteria; this runner never writes PASS/FAIL/HORIZON-*.

Stages (each banked incrementally into --outdir/raw/; a failed stage is recorded
and the rest continue):

  1. INSTALL     copy the pulled Thor ckpt (+config.json) into
                 ``<assets>/v7tiny_<arm>/`` — md5-stamped, source recorded.
  2. SPIKES      the §3c arrival-rate instrument from ``train_log.jsonl``:
                 rows with gnorm > 50 per 1,000 steps per 2,000-step window —
                 committed at step 7,800 BEFORE any read precisely so a sliding
                 median could not fake stability. Also the exact spike steps and
                 magnitudes, so the caveat's regime bounds can be recomputed
                 from the full 30k log rather than quoted stale.
  3. NORMS       the §3 SECONDARY read straight from the state dict (no model
                 build): ``‖to_scale_shift‖`` per layer (incumbent converged at
                 2.97 / 3.06 / 5.44) and ``‖act_emb.*‖``. Computed for the k=60
                 arm AND the incumbent from their local ckpts — the incumbent
                 norms double as a known-value control for the extraction.
                 Plus the ONE-VARIABLE check: the args diff vs the incumbent
                 (expected: o5_k, clip, paths — §3c made clip 0.5 a second
                 variable, which is why HORIZON-WORKS needs a clip-0.5 k=8
                 control).
  4. LATENTMOTION  the §3b ego-marginal read (E-DEC-59), ⛔ on BOTH PCA bands
                 0:8 AND 8:16 (a 0:8 null is what E-DEC-40 once over-read) and
                 at BOTH probe horizons k=4 AND k=60 (or probe-horizon
                 confounds arm-horizon), BOTH arms in each invocation.
  5. ACTDIV      the PRIMARY instrument (MM-E10 action-divergence): h1
                 ``ratio_action_over_scene`` for the k=60 arm and the incumbent
                 on the same device — the ≥10× criterion input. Banked Thor
                 incumbent reference: 0.005947. ``--no-actdiv`` skips.
  6. MERGE       one JSON: numbers, n, stamps, the caveat, and the criterion
                 fields (verdicts EMPTY).

Example (the real 30k read, after pulling the final ckpt from Thor):

  python mm_e19_read.py \
      --ckpt <pull>/ckpt.pt --config <pull>/config.json \
      --train-log <pull>/train_log.jsonl \
      --assets C:/Users/Admin/tanitad-caches/mm-e19-assets-20260901 \
      --stack  C:/Users/Admin/tanitad-wt/stack --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time

SP = pathlib.Path(__file__).resolve().parent

SPIKE_THRESH = 50.0     # the committed instrument: gnorm > 50
SPIKE_WINDOW = 2000     # per 2,000-step window, rate reported per 1,000 steps

CAVEAT = ("This arm trained through an intermittent-spiking regime (gnorm "
          "excursions, clip 0.5 containing the damage but not the cause), "
          "recorded in the prereg as spanning steps ~4,000-14,000 and peaking "
          "at 8,000-10,000 at the 14.4k amendment. ⛔ Any MM-E19 number travels "
          "with this caveat. The 'spike_arrival' block below recomputes the "
          "regime from the FULL log shipped with the read — quote THOSE bounds, "
          "not the interim ones.")

CRITERIA = {
    "primary_action_ratio": {
        "instrument": "actdiv (MM-E10/MM-E17), h1 ratio_action_over_scene",
        "incumbent_h1": 0.005947,
        "criterion": ("HORIZON-WORKS: h1 ratio rises >=10x (>= 0.0595) | "
                      "HORIZON-PARTIAL: rises, but <10x | HORIZON-INERT: "
                      "unchanged within noise | VOID: C0 != 0 or any "
                      "scene_spread ~ 0 (C160)"),
        "attribution_gate": ("⛔ HORIZON-WORKS additionally REQUIRES the "
                             "clip-0.5 k=8 control (prereg §3c): this arm is "
                             "o5_k 60 + clip 0.5 — TWO changes from the "
                             "incumbent."),
        "verdict": ""},
    "secondary_gains": {
        "instrument": ("state-dict norms (stage NORMS below) — Frobenius of the "
                       ".weight tensor, VERIFIED 2026-09-01 to be MM-E18's own "
                       "definition: it reproduces msg_e18.txt's step-30000 row "
                       "on o1ctrl30k/ckpt_step30000.pt (md5 eeb12bd11269…) to "
                       "all four decimals (2.9700/3.0605/5.4368, act_emb2 "
                       "9.4015)"),
        "criterion": ("||to_scale_shift|| at 30k should EXCEED the incumbent's "
                      "converged values and ||act_emb.2|| should stop declining. "
                      "⚠️ ATTRIBUTION DISCREPANCY IN THE PREREG, flagged for the "
                      "orchestrator: prereg §3 quotes 2.97/3.06/5.44 (and "
                      "9.5694→9.4015) as 'the incumbent's', but MM-E18's own "
                      "artifact (msg_e18.txt) measured those on o1ctrl30k, NOT "
                      "on postrain30k. The ACTUAL matched incumbent "
                      "postrain30k (final ckpt, md5 a58585883c27…) reads "
                      "to_scale_shift 3.8301/4.3603/4.2527, act_emb.2.weight "
                      "9.6288 — extracted with the same verified instrument, "
                      "in norms.incumbent below. Which baseline the criterion "
                      "binds to is the orchestrator's call. If the ratio rises "
                      "but the gains do NOT, the mechanism story is wrong — "
                      "report both."),
        "verdict": ""},
    "ego_marginal_band_0_8": {
        "instrument": "latentmotion (E-DEC-59) ego_marginal_over_drift, band 0:8",
        "banked_reference_k4_rdw8p30k": {"delta": -0.0006, "t": -0.48},
        "criterion": ("positive and material (t >> 2) at k=60 => horizon "
                      "hypothesis CONFIRMED on the data side; ~0 => actions do "
                      "not predict the latent transition at any horizon we can "
                      "reach. ⛔ a null in THIS band alone is NOT 'actions carry "
                      "nothing' — see band 8:16."),
        "verdict": ""},
    "ego_marginal_band_8_16": {
        "instrument": "latentmotion (E-DEC-59) ego_marginal_over_drift, band 8:16",
        "criterion": ("the band where E-DEC-40's band control found the ONLY "
                      "hint of action content (+0.0258, t 2.73) — a "
                      "low-variance subspace the top-8 projection is blind to "
                      "by construction. Both bands or the read is inadmissible."),
        "verdict": ""},
    "void_checks": {
        "criterion": ("actdiv C0_identity == 0 for every arm/horizon; no "
                      "scene_spread ~ 0; latentmotion constant control reads "
                      "0.0000; drift positive control clears its shuffle."),
        "verdict": ""},
    "anti_gates": {
        "criterion": ("⛔ o5_loss MUST NOT be compared across different o5_k — "
                      "a 60-step rollout is a harder task, not a worse score. "
                      "⛔ k=60 saw ~43% fewer windows per episode (319,002 "
                      "total) — same parity CORPUS, different window count. "
                      "T0 regression WITH action-ratio gain = intended trade, "
                      "escalates to T1; T0 regression with NO gain = dead arm."),
        "verdict": ""},
}


def md5_of(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_spike_arrival(train_log: pathlib.Path,
                        thresh: float = SPIKE_THRESH,
                        window: int = SPIKE_WINDOW) -> dict:
    """The §3c instrument, verbatim: spikes (gnorm > thresh) per 1,000 steps in
    per-``window`` buckets, plus the exact spike rows. Header rows and
    spectrum-only rows (no ``gnorm``) are skipped; on a resume, the LAST row
    per step wins."""
    rows = {}
    with open(train_log, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            s, g = d.get("step"), d.get("gnorm")
            if isinstance(s, (int, float)) and isinstance(g, (int, float)):
                rows[int(s)] = (float(g), d.get("o5_loss"), d.get("loss"))
    if not rows:
        return {"error": f"no (step, gnorm) rows in {train_log}"}
    steps = sorted(rows)
    max_step = steps[-1]
    spikes = [(s, rows[s][0]) for s in steps if rows[s][0] > thresh]
    wins = []
    lo = 0
    while lo < max_step:
        hi = lo + window
        g = [rows[s][0] for s in steps if lo < s <= hi]
        n_spk = sum(1 for v in g if v > thresh)
        wins.append({
            "steps": [lo, hi], "rows": len(g), "spikes_gt50": n_spk,
            "rate_per_1000": round(n_spk / (window / 1000.0), 2),
            "gnorm_median": round(statistics.median(g), 2) if g else None,
            "gnorm_max": round(max(g), 1) if g else None})
        lo = hi
    half = max_step / 2.0
    fh_rows = [s for s in steps if s <= half]
    sh_rows = [s for s in steps if s > half]

    def _rate(sub):
        if not sub:
            return None
        span = max(sub) - min(sub) + (steps[1] - steps[0] if len(steps) > 1 else 0)
        n = sum(1 for s in sub if rows[s][0] > thresh)
        return round(n / (span / 1000.0), 2) if span > 0 else None

    last_lo = max_step - window
    last_rows = [s for s in steps if s > last_lo]
    return {
        "instrument": f"rows with gnorm > {thresh:g} per 1,000 steps, "
                      f"per {window}-step window (prereg §3c, committed "
                      f"step 7,800)",
        "log_rows_used": len(steps), "max_step": max_step,
        "windows": wins,
        "spike_steps": [{"step": s, "gnorm": round(g, 1)} for s, g in spikes],
        "first_half_rate_per_1000": _rate(fh_rows),
        "second_half_rate_per_1000": _rate(sh_rows),
        "last_window_rate_per_1000": _rate(last_rows) if last_rows else None,
        "spiking_regime_bounds_steps": ([spikes[0][0], spikes[-1][0]]
                                        if spikes else None),
        "clean_steps_since_last_spike": (max_step - spikes[-1][0]
                                         if spikes else max_step),
    }


def weight_norms(ckpt_path: pathlib.Path) -> dict:
    """§3 SECONDARY read: ``to_scale_shift`` / ``act_emb`` tensor norms straight
    from the state dict — no model build, no forward."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("stack") or ck.get("model") or {}
    out = {"step": int(ck.get("step", -1)), "to_scale_shift": {}, "act_emb": {}}
    for k, v in sd.items():
        if not hasattr(v, "norm"):
            continue
        if "to_scale_shift" in k:
            out["to_scale_shift"][k] = round(float(v.float().norm()), 4)
        elif ".act_emb." in k or k.startswith("act_emb."):
            out["act_emb"][k] = round(float(v.float().norm()), 4)
    args = (ck.get("config") or {}).get("args") or {}
    out["args_subset"] = {k: args.get(k) for k in
                          ("o5_k", "clip", "lr", "steps", "horizons", "batch",
                           "o5_mode", "seed") if k in args}
    out["_args_full"] = args
    return out


def args_diff(a: dict, b: dict) -> dict:
    """ONE-VARIABLE check: keys whose values differ between two runs' args."""
    keys = sorted(set(a) | set(b))
    return {k: {"arm": a.get(k), "incumbent": b.get(k)}
            for k in keys if a.get(k) != b.get(k)}


def run_stage(cmd, log_path: pathlib.Path, env_extra=None) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    out = {"cmd": [str(c) for c in cmd], "rc": rc,
           "seconds": round(time.time() - t0, 1), "log": str(log_path)}
    if rc != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")
        out["log_tail"] = tail[-2000:]
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="MM-E19 one-command read")
    ap.add_argument("--ckpt", default="",
                    help="pulled Thor ckpt.pt for the k=60 arm; omit if already "
                         "installed at <assets>/v7tiny_<arm-name>/ckpt.pt")
    ap.add_argument("--config", default="",
                    help="pulled config.json (installed beside the ckpt)")
    ap.add_argument("--train-log", required=True,
                    help="pulled train_log.jsonl (arrival-rate stamp)")
    # ⛔ NO DEFAULT. This used to default to "k60clip05p30k", and on 2026-09-01
    # that silently mislabelled a DIFFERENT arm: the k=8 control was passed via
    # --ckpt, installed into <assets>/v7tiny_k60clip05p30k/ (overwriting the k60
    # checkpoint), and every table, log and JSON it produced was stamped
    # "k60clip05p30k". The numbers were right and the NAME was wrong, which is
    # the worse failure — the output looked like a k60 re-read that had moved
    # 2.07×. Only an md5 of the installed ckpt proved which arm it really was.
    # An identifier that can silently name the wrong artifact is not an
    # identifier; make the caller say it.
    ap.add_argument("--arm-name", required=True,
                    help="the arm this ckpt IS; stamped on every output")
    ap.add_argument("--incumbent", default="postrain30k")
    ap.add_argument("--replace-arm", action="store_true",
                    help="allow installing a ckpt over a DIFFERENT one already "
                         "held under this arm name (destroys that arm's assets)")
    ap.add_argument("--assets",
                    default=r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901",
                    help="asset home: v7tiny_<arm>/, sp2/cache/, helper modules")
    ap.add_argument("--stack", default=r"C:\Users\Admin\tanitad-wt\stack")
    ap.add_argument("--corpus", default="",
                    help="latentmotion corpus (default: <assets>/sp2/cache/"
                         "physicalai-val130-heldout)")
    ap.add_argument("--k-values", default="4,60",
                    help="BOTH by default — probe-horizon must not confound "
                         "arm-horizon (prereg §3b)")
    ap.add_argument("--bands", default="0:8,8:16",
                    help="BOTH by default — prereg: both bands or inadmissible")
    ap.add_argument("--nclips", type=int, default=80)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--no-actdiv", action="store_true",
                    help="skip the PRIMARY actdiv stage (dry-runs only)")
    ap.add_argument("--actdiv-corpus", default="",
                    help="default: <assets>/sp2/cache/physicalai-val-w120-256x640cyl")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args(argv)

    assets = pathlib.Path(a.assets)
    if not assets.is_dir():
        print(f"[REFUSED] assets dir {assets} does not exist", file=sys.stderr)
        return 2
    outdir = pathlib.Path(a.outdir) if a.outdir else SP / "mm_e19_out"
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    report = {"_prereg": "Project Steering/PREREG_MM_E19_K60_HORIZON.md",
              "_generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "_evidence_class": f"MEASURED (ours; dev-box {a.device})",
              "eval_tier": "T0-DIAGNOSTIC",
              "metric_families": {
                  "longitudinal": "REFUSED", "lateral": "REFUSED",
                  "tactical": "REFUSED", "strategic": "REFUSED",
                  "reason": ("T0-DIAGNOSTIC world-model probe panel (action "
                             "ratios, latent ego-marginals, weight norms) — "
                             "no driving metrics exist here to family-ise. "
                             "The four families bind driving evals "
                             "(EVAL_DOCTRINE); nothing in this file is a "
                             "driving-performance claim.")},
              "caveat_MUST_TRAVEL": CAVEAT,
              "criteria_FOR_ORCHESTRATOR_verdicts_EMPTY": CRITERIA,
              "stages": {}}

    # ---- 1. INSTALL ----------------------------------------------------------
    arm_dir = assets / f"v7tiny_{a.arm_name}"
    arm_dir.mkdir(exist_ok=True)
    inst = {"arm_dir": str(arm_dir)}
    if a.ckpt:
        src = pathlib.Path(a.ckpt)
        inst["source"] = str(src)
        inst["source_md5"] = md5_of(src)
        dst = arm_dir / "ckpt.pt"
        if dst.exists():
            inst["overwrote_md5"] = md5_of(dst)
            # ⛔ REFUSE to overwrite a DIFFERENT checkpoint under this arm name.
            # Recording "overwrote_md5" was not enough: on 2026-09-01 this line
            # replaced the k60 checkpoint with the k=8 control and the run went
            # on to produce a full, plausible, k60-labelled result set. The
            # destroyed arm was only recoverable because a copy happened to
            # survive in the pull directory.
            if inst["overwrote_md5"] != inst["source_md5"]:
                print(f"[REFUSED] {dst} already holds a DIFFERENT checkpoint "
                      f"({inst['overwrote_md5'][:12]}) than the one being "
                      f"installed ({inst['source_md5'][:12]}). Pass a correct "
                      f"--arm-name, or --replace-arm if you really mean to "
                      f"retire that arm's assets.", file=sys.stderr)
                if not a.replace_arm:
                    return 2
        shutil.copy2(src, dst)
        inst["installed_md5"] = md5_of(dst)
        if inst["installed_md5"] != inst["source_md5"]:
            print("[REFUSED] ckpt md5 changed during install", file=sys.stderr)
            return 2
        if a.config:
            shutil.copy2(a.config, arm_dir / "config.json")
            inst["config_md5"] = md5_of(arm_dir / "config.json")
    if not (arm_dir / "ckpt.pt").is_file():
        print(f"[REFUSED] no ckpt at {arm_dir / 'ckpt.pt'} and no --ckpt given",
              file=sys.stderr)
        return 2
    report["stages"]["install"] = inst

    # ---- 2. SPIKES -----------------------------------------------------------
    spike = parse_spike_arrival(pathlib.Path(a.train_log))
    report["spike_arrival"] = spike
    report["stages"]["spikes"] = {"train_log": str(a.train_log),
                                  "train_log_md5": md5_of(a.train_log)}

    # ---- 3. NORMS + one-variable check --------------------------------------
    arm_norms = weight_norms(arm_dir / "ckpt.pt")
    step = arm_norms["step"]
    interim = step < 30000
    report["arm"] = {"name": a.arm_name, "step": step,
                     "ckpt_md5": inst.get("installed_md5",
                                          md5_of(arm_dir / "ckpt.pt"))}
    report["_label"] = (f"INTERIM-DRYRUN step {step} — NEVER the read"
                        if interim else f"MM-E19 read at step {step}")
    report["interim_dryrun"] = interim
    inc_ck = assets / f"v7tiny_{a.incumbent}" / "ckpt.pt"
    inc_norms = weight_norms(inc_ck) if inc_ck.is_file() else {
        "error": f"no incumbent ckpt at {inc_ck}"}
    report["norms"] = {
        "arm": {k: v for k, v in arm_norms.items() if k != "_args_full"},
        "incumbent": {k: v for k, v in inc_norms.items() if k != "_args_full"},
        "references": {
            "norm_definition": "Frobenius norm of the .weight tensor",
            "known_value_control": ("VERIFIED 2026-09-01: this extractor "
                                    "reproduces msg_e18.txt's o1ctrl30k "
                                    "step-30000 row (2.9700/3.0605/5.4368, "
                                    "act_emb2 9.4015) to all four decimals "
                                    "from ckpt_step30000.pt md5 eeb12bd11269…"),
            "prereg_quoted_values": [2.97, 3.06, 5.44],
            "prereg_attribution": "'the incumbent' (prereg §3)",
            "measured_provenance": ("o1ctrl30k, msg_e18.txt 10-snapshot "
                                    "trajectory — NOT postrain30k; the prereg "
                                    "misattributes. The actual incumbent's "
                                    "norms are in 'incumbent' above."),
            "act_emb2_o1ctrl30k_trajectory": [9.5694, 9.4015]}}
    if "_args_full" in inc_norms:
        report["one_variable_check_args_diff"] = args_diff(
            arm_norms.get("_args_full", {}), inc_norms["_args_full"])

    # ---- 4. LATENTMOTION: both bands x both k, both arms ---------------------
    lm = SP / "latentmotion.py"
    corpus = a.corpus or str(assets / "sp2/cache/physicalai-val130-heldout")
    arms = f"{a.arm_name},{a.incumbent}"
    report["latentmotion"] = {"arms_requested": arms, "corpus": corpus,
                              "reads": {}}
    for k in [int(x) for x in a.k_values.split(",")]:
        for band in a.bands.split(","):
            tag = f"k{k}_band{band.replace(':', '_')}"
            out_json = raw / f"latentmotion_{tag}.json"
            st = run_stage(
                [py, str(lm), "--k", str(k), "--band", band, "--arms", arms,
                 "--assets", str(assets), "--stack", a.stack,
                 "--corpus", corpus, "--nclips", str(a.nclips),
                 "--device", a.device, "--out", str(out_json)],
                raw / f"latentmotion_{tag}.log")
            report["stages"][f"latentmotion_{tag}"] = st
            if st["rc"] == 0 and out_json.is_file():
                report["latentmotion"]["reads"][tag] = json.loads(
                    out_json.read_text(encoding="utf-8"))
            else:
                report["latentmotion"]["reads"][tag] = {
                    "error": f"stage failed rc={st['rc']} — see {st['log']}"}

    # ---- 5. ACTDIV (PRIMARY) -------------------------------------------------
    if a.no_actdiv:
        report["actdiv"] = {"skipped": "--no-actdiv (the PRIMARY >=10x "
                                       "criterion input is NOT in this file)"}
    else:
        adv = SP / "actdiv_local.py"
        adv_corpus = a.actdiv_corpus or str(
            assets / "sp2/cache/physicalai-val-w120-256x640cyl")
        out_json = raw / "actdiv_local.json"
        st = run_stage(
            [py, str(adv), "--corpus", adv_corpus, "--arms", arms,
             "--assets", str(assets), "--stack", a.stack,
             "--out", str(out_json)]
            + (["--device", a.device] if a.device else []),
            raw / "actdiv_local.log")
        report["stages"]["actdiv"] = st
        report["actdiv"] = (json.loads(out_json.read_text(encoding="utf-8"))
                            if st["rc"] == 0 and out_json.is_file() else
                            {"error": f"stage failed rc={st['rc']} — see "
                                      f"{st['log']}"})

    # ---- 6. n summary + MERGE ------------------------------------------------
    n = {}
    for tag, rd in report["latentmotion"]["reads"].items():
        for arm, av in (rd.get("arms") or {}).items():
            n[f"latentmotion_{tag}_{arm}_rows"] = av.get("n_rows")
    for arm, av in (report.get("actdiv", {}).get("arms") or {}).items():
        n[f"actdiv_{arm}_windows"] = av.get("n_windows")
    report["n"] = n

    suffix = "_INTERIM" if interim else ""
    merged = outdir / f"mm_e19_read_step{step}{suffix}.json"
    merged.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {merged}")
    print(f"   label: {report['_label']}")
    failed = [k for k, v in report["stages"].items()
              if isinstance(v, dict) and v.get("rc") not in (None, 0)]
    if failed:
        print(f"   ⛔ FAILED stages: {failed} — the merged JSON records them")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
