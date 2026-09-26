#!/usr/bin/env python3
"""refcv4b over the navtest scenes → a seam file the v1.1 scorer can read (TANITAD venv).

    # CPU smoke (≤ 20 tokens, no GPU):
    python code/run_bridge_navtest.py --arms A1_ego_cmd,A2_vision_pure --tokens raw/framebank_smoke_tokens.json \
        --inputs <export.json.gz> --bank <frame bank> --out raw/bridge_smoke --device cpu
    # the QUEUED full run — only through W1's GPU-gap launcher (PI 2026-09-19):
    python code/run_bridge_navtest.py --arms A1_ego_cmd --inputs … --bank … --out … --device auto --gpu-gap

⛔ NOTHING ABOUT THE MODEL IS RE-IMPLEMENTED. ``declare`` (the ego-input enforcement point),
``slot_sources``, ``pack_frames``, ``knots_to_navsim``, ``ARMS`` and the checkpoint loader are
E2's (`…/2026-09-19-navsim-refcv4b-bridge/code/tanitad_navsim_bridge.py`), imported by path. The
ONLY thing added here is a device parameter: E2's ``run_model`` hardcodes CPU, and the PI's
ruling is that our-model inference runs in a GPU GAP. ``tests/test_bridge_device.py`` pins that
this copy on CPU returns bit-identical trajectories to E2's own ``run_model``.

Frames come from W3's navtest bank (unique frames + index, sha256-verified per scene, KB1
bit-exact against E2's builder), never from loose .npy files.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import gzip
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[3]
E2_BRIDGE = REPO / ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/"
                    "code/tanitad_navsim_bridge.py")
DEF_CKPT = "D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt"
CKPT_MD5 = "99b573e8277d94a5e3bfbf630cb4d751"          # MODEL_REGISTRY.md §4.6


def load_e2_bridge():
    spec = importlib.util.spec_from_file_location("e2_bridge", E2_BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e2_bridge"] = mod
    spec.loader.exec_module(mod)
    return mod


def times_rel_t0(rec: dict) -> list:
    ts = [int(x) for x in rec["timestamps_us"]]
    return [(t - ts[-1]) / 1e6 for t in ts]


def run_model_dev(B, model, trainer_mod, rows_u8, decl: dict, steps: int, device: str) -> dict:
    """E2's ``tanitad_navsim_bridge.run_model`` with the device made explicit.

    VERBATIM except ``device``: ``model_inputs(..., device)`` and ``frames_to_device(..., device)``
    (E2's body passes "cpu" to both). Pinned equal on CPU by tests/test_bridge_device.py."""
    import torch
    mi = B.model_inputs(rows_u8, decl, device=device)
    fr = trainer_mod.frames_to_device(rows_u8[None], device)
    with torch.no_grad():
        out = model(fr, steps=steps, **mi["kwargs"])
    traj = out["traj"].float()[0].cpu().numpy().astype(np.float64)
    diag = {"sel_idx": int(out["sel_idx"][0])}
    if out.get("route_logits") is not None:
        diag["route_argmax"] = int(out["route_logits"][0].argmax(-1))
    return {"traj": traj, "diag": diag, "ego": mi["ego"], "nav": mi["nav"]}


class Bank:
    """W3's navtest frame bank: unique frames per shard + an index; sha256-checked per scene."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.index = json.load(open(self.root / "index.json", encoding="utf-8"))
        self._mm: dict = {}

    def get(self, token: str):
        e = self.index["tokens"][token]
        f = str(self.root / e["file"])
        if f not in self._mm:
            self._mm.clear()
            self._mm[f] = np.load(f, mmap_mode="r")
        arr = np.ascontiguousarray(self._mm[f][e["rows"]])
        sha = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        if sha != e["sha256"]:
            raise ValueError(f"{token}: bank sha {sha} != index {e['sha256']}")
        if arr.dtype != np.uint8 or arr.shape != (4, 256, 640, 3):
            raise ValueError(f"{token}: bank is {arr.dtype}{arr.shape}")
        if float(arr.mean()) < 1.0:
            raise ValueError(f"{token}: mean {arr.mean():.3f} — an all-black scene")
        src = np.load(self.root / "rigs" / f"{e['rig_key']}.src.npy")
        return arr, src >= 0, sha


# ═══ resume chunks, the RAM floor, and the niceness ══════════════════════════════════════════
# ⛔ MEASURED TWICE: a pass that banks only at the end loses everything it computed — navhard's
# official v2 runner scored all 5,912 scenarios and DIED in aggregation, and W3's own final cache
# pass was killed at 3,164 s leaving a 215-row CSV. So this one banks every CHUNK tokens and
# RESUMES from whatever is on disk, which also makes a partial pass SCORABLE instead of wasted.
CHUNK_DEFAULT = 200
RAM_FLOOR_MB = 3000                     # the W3 brief's binding floor
SEAM_KEYS = ("token", "fingerprint", "source", "poses", "knots", "device")


def _avail_mb():
    """System-wide AVAILABLE MiB, or None when it cannot be read (INCONCLUSIVE, never a zero)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1 << 20))
    except Exception:
        return None


def ram_ok(floor_mb: int = RAM_FLOOR_MB, sustain: int = 3, gap_s: float = 5.0, probe=_avail_mb):
    """(ok, last_sample, n_unreadable). ⚠️ ONE low sample is not pressure — MEASURED 2026-09-20:
    a 1,900 MB reading recovered to 3,413 MB seconds later and a single-sample guard stopped a
    healthy chain. Only ``sustain`` consecutive readings below the floor count. An unreadable
    probe fails OPEN and is COUNTED, because a number that could not be read is not a low number."""
    last, unreadable = None, 0
    for k in range(max(1, int(sustain))):
        a = probe()
        if a is None:
            unreadable += 1
            return True, last, unreadable
        last = a
        if a >= floor_mb:
            return True, a, unreadable
        if k + 1 < sustain:
            time.sleep(gap_s)
    return False, last, unreadable


def below_normal() -> str:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return "BELOW_NORMAL"
    except Exception as exc:                                    # pragma: no cover - platform
        return f"UNCHANGED ({type(exc).__name__})"


def chunk_write(cdir: Path, idx: int, cols: dict) -> Path:
    """Atomic: write a temp then ``os.replace``, so a kill mid-write cannot leave a half part."""
    p, tmp = cdir / f"part_{idx:05d}.npz", cdir / f"part_{idx:05d}.tmp.npz"
    np.savez(tmp, **cols)
    os.replace(tmp, p)
    return p


def chunk_scan(cdir: Path) -> dict:
    """token -> (part file, row). ⚠️ A part that will not READ is skipped with a printed warning
    and its tokens are recomputed — never counted as done, never silently dropped."""
    out = {}
    for f in sorted(Path(cdir).glob("part_*.npz")):
        try:
            with np.load(f, allow_pickle=False) as z:
                tk = [str(x) for x in z["token"].tolist()]
        except Exception as exc:
            print(f"  [chunk] unreadable {f.name} ({type(exc).__name__}) — ignored, recomputing",
                  flush=True)
            continue
        for i, t in enumerate(tk):
            out[t] = (f.name, i)
    return out


def chunk_assemble(cdir: Path, toks) -> dict:
    """The seam columns in the ORDER of ``toks``, read back out of the banked parts.
    ⛔ Refuses unless every requested token is present."""
    cdir = Path(cdir)
    idx = chunk_scan(cdir)
    missing = [t for t in toks if t not in idx]
    if missing:
        raise SystemExit(f"⛔ assemble: {len(missing)}/{len(toks)} tokens absent from {cdir} "
                         f"(first {missing[:3]})")
    cache, rows = {}, {k: [] for k in SEAM_KEYS}
    for t in toks:
        fn, i = idx[t]
        if fn not in cache:
            with np.load(cdir / fn, allow_pickle=False) as z:
                cache[fn] = {k: z[k] for k in SEAM_KEYS}
        for k in SEAM_KEYS:
            rows[k].append(cache[fn][k][i])
    return {k: (np.stack(rows[k]) if k in ("poses", "knots") else np.asarray(rows[k]))
            for k in SEAM_KEYS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="", help="JSON doc/list restricting the tokens")
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    ap.add_argument("--gpu-gap", action="store_true", help="acquire the device through W1's launcher")
    ap.add_argument("--accept-training-box-load", action="store_true",
                    help="CPU inference beside a LIVE training process (the named override of the "
                         "suite device gate); the accepted process is recorded in the manifest")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=CHUNK_DEFAULT,
                    help="bank a resume part every N tokens")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute every token even if parts are banked")
    ap.add_argument("--assemble-only", action="store_true",
                    help="build the seam from the banked parts; no model, no GPU")
    ap.add_argument("--min-avail-mb", type=int, default=RAM_FLOOR_MB)
    ap.add_argument("--max-wait-min", type=int, default=240,
                    help="how long to wait for RAM before exiting with the parts intact")
    ap.add_argument("--skip-md5", action="store_true")
    a = ap.parse_args(argv)
    B = load_e2_bridge()
    prio = below_normal()
    import torch
    torch.set_num_threads(int(a.threads))
    # ⛔ THE SUITE DEVICE GATE (orchestrator arbitration 2026-09-20) — ONE implementation, the
    # plugin's, which itself calls W1's shared gate when taniteval.bench.gpu_gap exports one:
    #   auto|cuda -> wait for a GPU gap; cpu -> only with no training alive, or with
    #   --accept-training-box-load, which records the process it accepted.
    sys.path.insert(0, str(REPO / "taniteval"))
    from taniteval.bench.plugins.navsim_v1 import device_decision
    # ⛔ --assemble-only reads banked parts off disk: no model, no tensor, no device. Asking the
    # device gate for permission to concatenate files is how a recovery path ends up REFUSED in
    # the one state it exists for (a busy box, a killed run, parts on disk).
    gate = ({"allowed": True, "requested": a.device, "device": "none", "decision": "NO_DEVICE",
             "reason": "--assemble-only: no model is loaded and no device is used"}
            if a.assemble_only else device_decision(a))
    if not gate.get("allowed", False):
        # ⚠️ W1's gate answers "may I run RIGHT NOW"; --gpu-gap hands the decision to W1's
        # launcher, so a no-gap refusal on auto|cuda is that launcher's business, not an error.
        # ⛔ MEASURED 2026-09-20 — and NOT what this comment used to claim: the launcher does NOT
        # block on `auto`. `gpu_gap.py:8-10, 230-256`: auto = ONE probe then "cpu" (NO_GAP_CPU);
        # cuda = probes every interval_s up to max_wait_s (default 8 h) and then ALSO returns
        # "cpu" (GAP_WAIT_TIMEOUT). Every other refusal (notably --device cpu beside a live
        # trainer without the override) is fatal here.
        if a.gpu_gap and str(gate.get("requested", a.device)) in ("auto", "cuda"):
            print("[bridge] no GPU gap right now — handing the decision to W1's launcher "
                  "(auto: ONE probe then CPU · cuda: waits up to max_wait_s then CPU): "
                  + str(gate.get("reason", ""))[:180], flush=True)
        else:
            print(json.dumps({"REFUSED": gate}, indent=1))
            return 3
    device = a.device
    gap = None
    if a.gpu_gap and not a.assemble_only:
        sys.path.insert(0, str(REPO / "taniteval"))
        from taniteval.bench.gpu_gap import GpuGapLauncher
        gap = GpuGapLauncher(requested=a.device)
        device = gap.acquire()                       # auto: one probe · cuda: waits, then CPU
        if device == "cpu":
            # ⛔ THE HOLE THIS CLOSES. Both fallbacks land on the CPU path without the CPU gate's
            # OWN condition ever being evaluated — and that condition is the arbitration's: CPU
            # inference beside a LIVE training process is REFUSED unless the operator names the
            # override. Re-ask the gate as if `--device cpu` had been typed, and refuse if it
            # says no. (A gap-less GPU is not a licence to load the training box.)
            cpu_args = copy.copy(a)
            cpu_args.device = "cpu"
            cpu_gate = device_decision(cpu_args)
            gate = dict(gate)
            gate["acquired_device"] = "cpu"
            gate["launcher_semantics"] = ("gpu_gap.py:230-256 — auto: ONE probe then cpu; "
                                          "cuda: probes to max_wait_s then cpu")
            gate["cpu_regate"] = cpu_gate
            if not cpu_gate.get("allowed", False):
                print(json.dumps({"REFUSED_ON_CPU_FALLBACK": cpu_gate}, indent=1))
                return 3
            print("[bridge] no gap -> CPU, and the CPU gate re-checked: "
                  + str(cpu_gate.get("reason", cpu_gate.get("decision", "")))[:160], flush=True)
    elif device == "auto":
        device = "cpu"
    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = json.load(gzip.open(a.inputs, "rt", encoding="utf-8"))["tokens"]
    if a.tokens:
        w = json.load(open(a.tokens, encoding="utf-8"))
        w = set(w["tokens"] if isinstance(w, dict) else w)
        doc = {t: r for t, r in doc.items() if t in w}
    bank = Bank(a.bank)
    toks = sorted(t for t in doc if t in bank.index["tokens"])
    missing = sorted(set(doc) - set(toks))
    if not toks:
        sys.exit("⛔ empty token set (no token of the export is in the bank) — refusing")
    md5 = None if (a.skip_md5 or a.assemble_only) else B.md5_file(a.ckpt)
    if md5 is not None and md5 != CKPT_MD5:
        sys.exit(f"⛔ checkpoint md5 {md5} != registry {CKPT_MD5}")
    t0 = time.time()
    if a.assemble_only:
        model, tr, prov = None, None, {"step": "ASSEMBLE_ONLY", "decoder_steps": 0, "window": 0}
        steps, W = 0, 0
    else:
        model, cfg, targs, prov, arm_mod = B.load_refcv4b(a.ckpt)
        if device != "cpu":
            model = model.to(device)
        tr = arm_mod.trainer()
        steps, W = int(prov["decoder_steps"]), int(prov["window"])
    print(f"[bridge] model step={prov['step']} window={W} steps={steps} device={device} "
          f"load {time.time() - t0:.1f}s", flush=True)
    summary = {"device_gate": gate,
               "model": {"ckpt": a.ckpt, "ckpt_md5": md5, "step": prov["step"], "device": device,
                         "registry": "MODEL_REGISTRY.md §4.6 refcv4b-b1-v72-40k",
                         "window_rows": W, "decoder_steps": steps, "torch": torch.__version__},
               "bank": a.bank, "n_tokens": len(toks), "n_export_tokens_without_bank": len(missing),
               "arms": {}}
    for arm in [x for x in a.arms.split(",") if x]:
        if arm not in B.ARMS:
            sys.exit(f"unknown arm {arm!r}; known {sorted(B.ARMS)}")
        spec = B.ARMS[arm]
        ta = time.time()
        cdir = out / f"chunks_{arm}"
        cdir.mkdir(parents=True, exist_ok=True)
        banked = {} if a.no_resume else chunk_scan(cdir)
        todo = [] if a.assemble_only else [t for t in toks if t not in banked]
        nxt = 1 + max([int(f.stem.split("_")[1]) for f in cdir.glob("part_*.npz")], default=-1)
        print(f"  [{arm}] banked {len(banked)} · to run {len(todo)} · parts -> {cdir.name}",
              flush=True)
        recs, buf = [], {k: [] for k in SEAM_KEYS}
        ram_waits, ram_unreadable, stalled = [], 0, False
        rowsf = open(out / f"rows_{arm}.jsonl", "a", encoding="utf-8")
        for i, tok in enumerate(todo):
            r = doc[tok]
            decl = B.declare(r["ego_statuses"], arm)
            fr, mask, sha = bank.get(tok)
            times = times_rel_t0(r)
            src_idx = B.slot_sources(times, "ST" if spec["frames"] == "BLIND" else spec["frames"], W)
            rows = B.pack_frames(fr, src_idx, None if spec["frames"] != "BLIND" else 128)
            res = run_model_dev(B, model, tr, rows, decl, steps, device)
            p = B.knots_to_navsim(res["traj"])
            rec = {"token": tok, "frame_sha16": sha, "declared_values": decl,
                   "ego_block": res["ego"], "nav": res["nav"], "diag": res["diag"],
                   "slot_sources": src_idx, "device": device}
            if len(recs) < 50:
                recs.append(rec)
            rowsf.write(json.dumps(rec, default=str) + "\n")
            rowsf.flush()
            buf["token"].append(tok)
            buf["fingerprint"].append(r["fingerprint"])
            buf["source"].append("refcv4b")
            buf["poses"].append(p.astype(np.float32))
            buf["knots"].append(res["traj"].astype(np.float32))
            buf["device"].append(device)
            if (i + 1) % 100 == 0:
                print(f"  [{arm}] {i + 1}/{len(todo)} {(time.time() - ta) / (i + 1):.2f} s/scene",
                      flush=True)
            if len(buf["token"]) >= int(a.chunk) or i == len(todo) - 1:
                chunk_write(cdir, nxt, {k: (np.stack(v) if k in ("poses", "knots")
                                            else np.asarray(v)) for k, v in buf.items()})
                print(f"  [{arm}] banked part_{nxt:05d} ({len(buf['token'])} tokens)", flush=True)
                nxt += 1
                buf = {k: [] for k in SEAM_KEYS}
                ok, avail, unread = ram_ok(int(a.min_avail_mb))
                ram_unreadable += unread
                t_wait = time.time()
                while not ok:
                    if (time.time() - t_wait) / 60.0 > float(a.max_wait_min):
                        print(f"  [{arm}] ⛔ RAM below {a.min_avail_mb} MB for "
                              f"{a.max_wait_min} min — exiting with {nxt} parts banked; rerun to "
                              f"resume", flush=True)
                        stalled = True
                        break
                    print(f"  [{arm}] RAM {avail} MB < {a.min_avail_mb} — waiting 60 s", flush=True)
                    time.sleep(60)
                    ok, avail, unread = ram_ok(int(a.min_avail_mb))
                    ram_unreadable += unread
                if not ok:
                    ram_waits.append(round((time.time() - t_wait) / 60.0, 1))
                    break
                if time.time() - t_wait > 30:
                    ram_waits.append(round((time.time() - t_wait) / 60.0, 1))
        rowsf.close()
        got = chunk_scan(cdir)
        have = [t for t in toks if t in got]
        if not have:
            sys.exit(f"⛔ {arm}: no banked token at all in {cdir}")
        cols = chunk_assemble(cdir, have)
        dev_hist = {d: int((cols["device"] == d).sum()) for d in sorted(set(cols["device"].tolist()))}
        npz = out / f"seam_{arm}.npz"
        np.savez(npz, token=cols["token"], fingerprint=cols["fingerprint"],
                 source=cols["source"], poses=cols["poses"], knots=cols["knots"],
                 sampling=np.asarray([8, 0.5]), arm=np.asarray(arm),
                 n_requested=np.asarray(len(toks)))
        man = {"arm": arm, "spec": spec, "declared_inputs": list(spec["declared"]),
               "frames_construction": spec["frames"], "n": len(have), "n_requested": len(toks),
               "partial": len(have) != len(toks), "stalled_on_ram": stalled,
               "device": device, "devices": dev_hist,
               # ⚠️ a seam whose rows were not all produced on ONE device is a MIXTURE: CPU and
               # CUDA differ in the last ulp on uint8 scaling (programme-measured), so the record
               # says so rather than implying one device.
               "mixed_device": len(dev_hist) > 1, "priority": prio,
               "resumed_from_parts": len(banked), "n_parts": len(list(cdir.glob("part_*.npz"))),
               "ram_floor_mb": int(a.min_avail_mb), "ram_wait_min": ram_waits,
               "ram_probe_unreadable": ram_unreadable,
               "wall_s": round(time.time() - ta, 1), "seam": str(npz),
               "rows_jsonl": str(out / f"rows_{arm}.jsonl"),
               "conversion": B.knots_to_navsim.__doc__, "per_token": recs,
               "device_gate": gate, "gpu_gap_events": (gap.events if gap else None)}
        json.dump(man, open(out / f"seam_{arm}.manifest.json", "w", encoding="utf-8"), indent=1,
                  default=str)
        summary["arms"][arm] = {"seam": str(npz), "n": len(have), "n_requested": len(toks),
                                "partial": len(have) != len(toks), "devices": dev_hist,
                                "s_per_scene": (round((time.time() - ta) / len(todo), 3)
                                                if todo else None)}
        print(json.dumps({arm: summary["arms"][arm]}), flush=True)
    json.dump(summary, open(out / "bridge_run_summary.json", "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
