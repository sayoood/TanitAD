"""Head-to-head VLM comparison — Cosmos-Reason1-7B vs Cosmos-Reason2-8B.

WHY THIS EXISTS. `vlm_route_labels.py` labels a corpus with ONE model. This
harness answers a different question: *which* model should label it. That
requires a controlled comparison, and a controlled comparison has requirements
the labeler does not:

  1. THE SAME WINDOWS. Both arms must see a byte-identical window list, or the
     paired statistics (McNemar, paired episode-cluster bootstrap) are invalid.
     The window list is therefore built ONCE into ``windows.json`` and every
     model run reads it back; it is never re-derived per model.
  2. THE SAME PROMPT. `vlmroute-2026-07-20-a`, imported verbatim from
     `vlm_route_labels`, for every arm. The prompt was authored for Reason2 —
     that is a REAL CONFOUND and it must be named, not fixed. If Reason1 fails
     to parse it, that IS the measurement (formatting failure vs reasoning
     failure), so this harness stores the RAW reply and classifies the failure
     instead of retrying or repairing it.
  3. THE RAW REPLY. The labeler keeps only `raw_len`; a failure taxonomy needs
     the text. Stored truncated (`--raw-chars`) on every record.
  4. THE KINEMATIC GROUND TRUTH INLINE. `route_from_future_v21` is evaluated on
     the pod and written into each record, so scoring runs anywhere with no pod
     and no 4.4 GB val build. (`vlm_compare_score.py` is pure-stdlib+numpy.)
  5. COST. Per-window wall clock, generated TOKENS (not characters), and peak
     VRAM — the throughput half of the verdict.

LOADER NOTE (not a prompt change — a correctness fix). `vlm_route_labels.Cosmos`
loads with `device_map=` (requires `accelerate`, absent on tanitad-eval) and
hard-prefers `Qwen3VLForConditionalGeneration`. Reason1-7B is `qwen2_5_vl` and
Reason2-8B is `qwen3_vl`; forcing one class onto both is wrong. This harness
resolves the class from the checkpoint config via `AutoModelForImageTextToText`
and loads CPU->cuda. Everything the model is SHOWN and ASKED is untouched.

Usage (tanitad-eval, after any latency benchmark has printed ALLDONE):
  PYTHONPATH=/root/vlm_compare python3 vlm_model_compare.py \
      --val /root/valdata/physicalai-val-0c5f7dac3b11 \
      --out /root/vlm_compare/out --episodes 0-39 --stride 40 \
      --model nvidia/Cosmos-Reason1-7B --tag reason1 --passes A
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import refb_labels as R                    # noqa: E402
import vlm_route_labels as VL              # noqa: E402

MAX_NEW_A, MAX_NEW_B = 1000, 2200


# ------------------------------------------------- the left/right order probe
def swap_route_order(prompt_a: str) -> tuple[str, str]:
    """Return (prompt with `right` listed before `left`, its NEW version stamp).

    A DIAGNOSTIC ARM, never the headline. Reason2 shows a systematic left bias
    (32 % accuracy on detected right turns vs 74 % on left). Two explanations
    are indistinguishable from the headline run: the model prefers the
    first-listed enum token, or it genuinely cannot see right turns. Swapping
    the order separates them — if the bias follows the ordering it is OUR
    prompt's fault and is fixable; if it stays left it is a model property.

    The transformation is two explicit, auditable substitutions and it stamps a
    DIFFERENT ``prompt_version``, so a swapped record can never be silently
    pooled with the identical-prompt arms.
    """
    a = " | ".join(VL.ROUTE_ENUM)                        # left | straight | ...
    swapped = list(VL.ROUTE_ENUM)
    i, j = swapped.index("left"), swapped.index("right")
    swapped[i], swapped[j] = swapped[j], swapped[i]
    b = " | ".join(swapped)
    if a not in prompt_a or "left / right  =" not in prompt_a:
        raise SystemExit("route-order probe: prompt text changed, refusing to "
                         "guess — update swap_route_order() deliberately")
    out = prompt_a.replace(a, b).replace("left / right  =", "right / left  =")
    return out, VL.PROMPT_VERSION + "-rswap"


# --------------------------------------------------------------------- model
class VLM:
    """One checkpoint, resolved from its own config. No `accelerate` needed."""

    def __init__(self, model_id: str, dtype=torch.bfloat16):
        from transformers import AutoConfig, AutoProcessor
        t0 = time.time()
        cfg = AutoConfig.from_pretrained(model_id)
        self.arch = getattr(cfg, "model_type", "?")
        self.proc = AutoProcessor.from_pretrained(model_id)
        try:
            from transformers import AutoModelForImageTextToText as Cls
        except ImportError:                  # loud, never a silent fallback
            from transformers import AutoModelForCausalLM as Cls
        m = Cls.from_pretrained(model_id, dtype=dtype)
        self.model = m.to("cuda").eval()
        self.model_id = model_id
        self.load_s = time.time() - t0
        torch.cuda.synchronize()
        self.weights_gb = torch.cuda.memory_allocated() / 2 ** 30
        print(f"[vlm] {model_id} arch={self.arch} loaded in {self.load_s:.0f}s "
              f"weights={self.weights_gb:.2f} GiB", flush=True)

    @torch.no_grad()
    def ask(self, images, text, max_new_tokens):
        """-> (reply_text, n_prompt_tokens, n_gen_tokens, seconds, truncated)."""
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": text})
        msgs = [{"role": "user", "content": content}]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        torch.cuda.synchronize()
        t0 = time.time()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        gen = out[0][n_in:]
        n_gen = int(gen.shape[0])
        return (self.proc.decode(gen, skip_special_tokens=True),
                n_in, n_gen, dt, n_gen >= max_new_tokens)


# ------------------------------------------------------------------ windows
def build_windows(val_dir: str, lo: int, hi: int, stride: int, t0: int) -> list:
    """The SHARED window list + inline kinematic v2.1 ground truth.

    Built once and reused by every arm — the arms must not re-derive it, or a
    skipped window in one arm silently unpairs the comparison.
    """
    files = sorted(glob.glob(os.path.join(val_dir, "ep_*.pt")))[lo:hi + 1]
    if not files:
        raise SystemExit(f"no ep_*.pt in {val_dir} for episodes {lo}-{hi}")
    rows = []
    for f in files:
        ep = os.path.basename(f).replace(".pt", "")
        d = torch.load(f, map_location="cpu", weights_only=False)
        frames, poses = d["frames_u8"], d["poses"].float()
        T = min(frames.shape[0], poses.shape[0])
        for t in range(t0, T, stride):
            hist, fut, _ = VL.pick_frames(frames, t, T)
            if not fut:
                continue          # nothing of the future to see -> nothing to ask
            k = R.route_from_future_v21(poses, t)
            rows.append({
                "episode": ep, "t": int(t), "clip_len": int(T),
                "n_hist_frames": len(hist), "n_future_frames": len(fut),
                "kin_v21": {
                    "route": R.ROUTE_V21_NAMES[k["route"]],
                    "valid": bool(k["valid"]), "ambiguous": bool(k["ambiguous"]),
                    "reason": k["reason"],
                    "net_dyaw_deg": round(k["net_dyaw"] * 180.0 / 3.141592653589793, 2),
                    "peak_kappa": round(k["peak_kappa"], 5),
                    "concentration": round(k["concentration"], 4),
                    "arc_m": round(k["arc_m"], 1), "h_steps": int(k["h_steps"]),
                }})
    return rows


# ------------------------------------------------------------- classification
def classify_A(raw: str, truncated: bool, err: str | None) -> dict:
    """Parse/enum outcome for one Pass-A reply. Never repairs — only labels."""
    if err is not None:
        return {"outcome": "exception", "ROUTE": None, "enum_ok": False,
                "raw_route": None, "js": {}}
    js = VL.extract_json(raw)
    if js is None:
        has_brace = "{" in raw and "}" in raw
        return {"outcome": "json_invalid" if has_brace else "no_json",
                "ROUTE": None, "enum_ok": False, "raw_route": None, "js": {}}
    if not isinstance(js, dict) or "ROUTE" not in js:
        return {"outcome": "missing_route_key", "ROUTE": None, "enum_ok": False,
                "raw_route": None, "js": js if isinstance(js, dict) else {}}
    raw_route = js.get("ROUTE")
    route, ok = VL.coerce(raw_route, VL.ROUTE_ENUM)
    return {"outcome": "ok" if ok else "enum_violation",
            "ROUTE": route, "enum_ok": bool(ok),
            "raw_route": raw_route if isinstance(raw_route, str) else str(raw_route),
            "js": js, "truncated": bool(truncated)}


PASS_A_SLOTS = ("ROUTE", "route_confidence", "route_evidence",
                "route_event_time_s", "road_geometry", "sees_junction_ahead")


def slot_fill(js: dict) -> dict:
    """Per-slot answered/not — 'unknown' and null both count as NOT answered.

    A model that abstains everywhere has a perfect violation rate and zero
    value; this is the metric that catches it."""
    out = {}
    for s in PASS_A_SLOTS:
        v = js.get(s, None)
        out[s] = not (v is None or (isinstance(v, str) and
                                    v.strip().lower() in ("unknown", "", "null")))
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser("vlm_model_compare")
    ap.add_argument("--val", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--out", default="/root/vlm_compare/out")
    ap.add_argument("--episodes", default="0-39")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="arm name, e.g. reason1")
    ap.add_argument("--passes", default="A", choices=["A", "B", "AB"])
    ap.add_argument("--limit-windows", type=int, default=0)
    ap.add_argument("--window-stride", type=int, default=1,
                    help="run every Nth manifest window — a DETERMINISTIC "
                         "sub-sample spread over all episodes, so a cheaper "
                         "Pass-B run stays episode-balanced instead of "
                         "truncating to a prefix of episodes")
    ap.add_argument("--raw-chars", type=int, default=4000)
    ap.add_argument("--route-order", default="as_written",
                    choices=["as_written", "right_first"],
                    help="DIAGNOSTIC arm: 'right_first' lists `right` before "
                         "`left` in the ROUTE enum and stamps a different "
                         "prompt_version. Never the headline arm.")
    ap.add_argument("--windows-only", action="store_true",
                    help="build windows.json and exit (no GPU, no model)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wpath = os.path.join(args.out, "windows.json")
    if os.path.exists(wpath):
        W = json.load(open(wpath))
        print(f"[win] reusing {wpath}: {len(W['windows'])} windows", flush=True)
    else:
        lo, hi = ((int(x) for x in args.episodes.split("-"))
                  if "-" in args.episodes else (int(args.episodes),) * 2)
        rows = build_windows(args.val, lo, hi, args.stride, args.t0)
        W = {"val": args.val, "episodes": args.episodes, "stride": args.stride,
             "t0": args.t0, "prompt_version": VL.PROMPT_VERSION,
             "hist_s": list(VL.HIST_S), "fut_s": list(VL.FUT_S),
             "n_windows": len(rows),
             "n_episodes": len({r["episode"] for r in rows}),
             "windows": rows}
        json.dump(W, open(wpath, "w"), indent=1)
        print(f"[win] built {wpath}: {len(rows)} windows over "
              f"{W['n_episodes']} episodes", flush=True)
    if args.windows_only:
        return

    dst_dir = os.path.join(args.out, args.tag)
    os.makedirs(dst_dir, exist_ok=True)
    prompt_a, prompt_b = VL.build_prompt("A"), None
    prompt_version = VL.PROMPT_VERSION
    if args.route_order == "right_first":
        prompt_a, prompt_version = swap_route_order(prompt_a)
        print(f"[probe] ROUTE order swapped -> prompt_version={prompt_version}",
              flush=True)
    if "B" in args.passes:
        prompt_b = VL.build_prompt("B")

    vlm = VLM(args.model)
    torch.cuda.reset_peak_memory_stats()
    ep_cache_name, ep_cache = None, None
    n_done, t_start = 0, time.time()

    for w in W["windows"][::max(1, args.window_stride)]:
        dst = os.path.join(dst_dir, f"{w['episode']}_t{w['t']:04d}.json")
        if os.path.exists(dst):
            continue
        if ep_cache_name != w["episode"]:
            ep_cache = torch.load(os.path.join(W["val"], w["episode"] + ".pt"),
                                  map_location="cpu", weights_only=False)
            ep_cache_name = w["episode"]
        frames, poses = ep_cache["frames_u8"], ep_cache["poses"].float()
        T = min(frames.shape[0], poses.shape[0])
        hist, fut, lines = VL.pick_frames(frames, w["t"], T)
        eb = VL.ego_block(poses, w["t"])
        imgs = [VL.to_pil(frames, i) for i in hist + fut]
        head = (f"{VL._RULES}\n\nFRAMES PROVIDED (in order):\n"
                + "\n".join(lines)
                + f"\n\nEGO MOTION NOW (measured, past only):\n"
                  f"{json.dumps(eb)}\n{eb['summary']}\n")
        rec = {k: w[k] for k in ("episode", "t", "clip_len", "n_hist_frames",
                                 "n_future_frames", "kin_v21")}
        rec.update(model=vlm.model_id, model_tag=args.tag, arch=vlm.arch,
                   prompt_version=prompt_version, provenance="vlm",
                   route_order=args.route_order,
                   ego_block=eb, n_images=len(imgs))

        if "A" in args.passes:
            raw, n_in, n_gen, dt, trunc, err = "", 0, 0, 0.0, False, None
            try:
                raw, n_in, n_gen, dt, trunc = vlm.ask(imgs, head + "\n" + prompt_a,
                                                      MAX_NEW_A)
            except Exception as e:                       # fail loud, keep going
                err = f"{type(e).__name__}: {e}"
                print(f"[err] {w['episode']} t={w['t']}: {err}", flush=True)
            c = classify_A(raw, trunc, err)
            js = c.pop("js")
            rec["pass_A"] = {
                "pass": "A", "future_track_given": False,
                "ROUTE": c["ROUTE"], "enum_ok": c["enum_ok"],
                "raw_route": c["raw_route"], "outcome": c["outcome"],
                "truncated": bool(trunc), "error": err,
                "route_confidence": js.get("route_confidence"),
                "route_evidence": js.get("route_evidence"),
                "route_event_time_s": js.get("route_event_time_s"),
                "road_geometry": VL.coerce(
                    js.get("road_geometry"),
                    VL.ENUMS_SCENARIO["road_geometry"])[0],
                "road_geometry_ok": VL.coerce(
                    js.get("road_geometry"),
                    VL.ENUMS_SCENARIO["road_geometry"])[1],
                "sees_junction_ahead": js.get("sees_junction_ahead"),
                "slot_filled": slot_fill(js),
                "n_prompt_tokens": n_in, "n_gen_tokens": n_gen,
                "gen_seconds": round(dt, 3), "raw_len": len(raw),
                "raw": raw[:args.raw_chars]}

        if "B" in args.passes:
            fts = VL.future_track_summary(poses, w["t"])
            raw, n_in, n_gen, dt, trunc, err = "", 0, 0, 0.0, False, None
            try:
                raw, n_in, n_gen, dt, trunc = vlm.ask(
                    imgs, head + "\nNUMERIC FUTURE EGO TRACK (measured):\n"
                    + fts["summary"] + "\n\n" + prompt_b, MAX_NEW_B)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[err B] {w['episode']} t={w['t']}: {err}", flush=True)
            js = VL.extract_json(raw) or {}
            rec["pass_B"] = {"pass": "B", "future_track_given": True,
                             "future_track": fts, "parsed": js, "error": err,
                             "truncated": bool(trunc),
                             "n_prompt_tokens": n_in, "n_gen_tokens": n_gen,
                             "gen_seconds": round(dt, 3), "raw_len": len(raw),
                             "raw": raw[:args.raw_chars]}

        rec["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        with open(dst, "w") as fh:
            json.dump(rec, fh, indent=1)
        n_done += 1
        if n_done % 10 == 0:
            el = time.time() - t_start
            print(f"[{args.tag} {n_done}] {w['episode']} t={w['t']} "
                  f"A={rec.get('pass_A', {}).get('ROUTE')} "
                  f"{el / n_done:.2f}s/window "
                  f"peak={rec['peak_vram_gib']:.1f}GiB", flush=True)
        if args.limit_windows and n_done >= args.limit_windows:
            break

    summary = {"tag": args.tag, "model": vlm.model_id, "arch": vlm.arch,
               "passes": args.passes, "prompt_version": prompt_version,
               "route_order": args.route_order,
               "n_windows_run": n_done,
               "load_seconds": round(vlm.load_s, 1),
               "weights_gib": round(vlm.weights_gb, 3),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
               "wall_seconds": round(time.time() - t_start, 1),
               "s_per_window": round((time.time() - t_start) / max(1, n_done), 3)}
    json.dump(summary, open(os.path.join(args.out, f"run_{args.tag}.json"), "w"),
              indent=1)
    print("VLM_MODEL_COMPARE_DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
