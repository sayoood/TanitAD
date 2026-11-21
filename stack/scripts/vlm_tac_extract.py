"""Production VLM extractor for the tactical/strategic label pipeline.

⭐ ONE RUNNER FOR EVERY MACHINE. The PI's sequencing is "start with the T4, then
continue with Thor or a pod". That makes portability and RESUME the design, not
features bolted on later: this script must be able to stop on a Colab VM at
clip 800 and continue on Thor at clip 801 with no bookkeeping by hand.

⛔ IT DECIDES NOTHING. The VM/pod does the one thing it is uniquely able to do —
hold Qwen3.5-9B and generate — and emits RAW TEXT. Parsing, the strict verdict
vocabulary, the admissible-set prior, the ego gate and the composition all run
on the dev box against the tested code (``vlm_tac_prompts.parse_verdict``,
``tac_str_labels.compose``). A second copy of those rules on the remote side
would be an untested implementation of the part that matters most.

⛔ IT NEEDS NO CREDENTIALS. Frames and prompts arrive pre-built in the payload,
so no HF token is ever written to a remote filesystem, argument or CLI log.

FOUR THINGS THIS ENCODES, EACH FROM A MEASURED FAILURE (2026-08-19):

1. **Resume is keyed on (clip_id, kind) already in the output**, and the output
   is appended + fsynced after EVERY generation. A run that produced N answers
   must never be able to yield 0. *(A write-once predecessor lost a full 68-min
   run to a client timeout; the VM's file still held the PREVIOUS run's bytes.)*

2. **Every generation is also streamed to stdout, base64-framed with its own
   length.** Three Colab VMs were reaped mid-run in one session — twice while
   holding results. The operator's log is the one store that cannot be reaped,
   and per-frame lengths make a dropped or truncated frame detectable instead
   of silently decoding to garbage.

3. **``n_new_tokens`` and ``hit_cap`` are recorded per generation.** At a 1200
   cap, 13 of 18 generations were cut off mid-``<think>`` with no verdict line.
   A strict parser abstains on those — correctly — and "the VLM abstained on
   90 % of clips" then reads as a claim about the MODEL when it is a claim
   about the OPERATOR SETTING. Truncation must be visible in the artifact.

4. **Batch size is adaptive and self-measuring.** The environment is too flaky
   to calibrate separately (three sessions died trying), so the run reports its
   own throughput and steps the batch down on OOM rather than dying.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback

KINDS = ("lon", "lane", "sign")

#: ⭐ ROUTE_TO answers "was this turn ROUTE-determined, and which way did the
#: route go" — a question that only arises where the ego actually turns. Gating
#: the sign call on the Alpamayo turn classes is therefore a scope rule, not a
#: sampling shortcut. MEASURED on the 4,729-clip taxonomy: 186 turns (3.9 %),
#: so the gate removes 4,543 generations — about a third of the whole corpus
#: run — without dropping a single clip where the token was reachable.
TURN_LANES = frozenset({"turn left", "turn right"})


def log(*a) -> None:
    print("[vlm]", *a, flush=True)


def emit(rec: dict) -> None:
    """Stream one finished generation to stdout, framed with its own length."""
    blob = base64.b64encode(json.dumps(rec).encode()).decode()
    print(f"@@G{len(blob)}@{blob}@@", flush=True)


def wants_sign(clip: dict, sign_on_turns: bool) -> bool:
    if not sign_on_turns:
        return True
    lane = (clip.get("alpamayo", {}).get("lane") or "").strip().lower()
    return lane in TURN_LANES


def load_done(path: str) -> set[tuple[str, str]]:
    """(clip_id, kind) pairs already banked. A partial last line — the run died
    mid-write — is skipped rather than crashing the resume."""
    done: set[tuple[str, str]] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "clip_id" in r and "kind" in r:
                done.add((r["clip_id"], r["kind"]))
    return done


def build_inputs(proc, prompts, imgs_per):
    """⚠️ left-padding is REQUIRED for decoder-only batched generation: with
    right padding the pads sit between the prompt and the first generated token
    and the model attends to them."""
    proc.tokenizer.padding_side = "left"
    texts, flat = [], []
    for prompt, imgs in zip(prompts, imgs_per):
        msgs = [{"role": "user", "content":
                 [{"type": "image", "image": im} for im in imgs]
                 + [{"type": "text", "text": prompt}]}]
        texts.append(proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True))
        flat.extend(imgs)
    return proc(text=texts, images=flat, padding=True, return_tensors="pt")


def ensure_deps() -> list[str]:
    """pip-install the 4-bit stack if missing. ⛔ OPT-IN, NEVER THE DEFAULT.

    A fresh Colab VM has neither unsloth nor bitsandbytes, so without this the
    run dies at load with "no loader succeeded" (MEASURED 2026-08-19 — the
    previous VM only worked because an earlier smoke script had installed them,
    which made the extractor look self-sufficient when it was not).

    ⚠️ WHY IT IS NOT AUTOMATIC. `pip install <anything>` can resolve torch from
    the default index and silently replace a working build with one the driver
    cannot run. MEASURED TWICE on pod4 (CLAUDE.md): `uv pip install -U
    accelerate` and `compressed-tensors` each landed torch 2.13.0+cu130 on a
    CUDA-12.8 driver, after which `torch.cuda.is_available()` was False and
    every GPU job on the pod died. Neither command names torch; it arrives
    through the dependency closure. On an EPHEMERAL VM that risk is acceptable
    and recoverable. On Thor or a pod — where the environment is curated and
    the two-venv rule applies — it is not, so those hosts must arrive with
    their env already correct and never pass this flag.

    Guarded accordingly: `--no-deps` so the closure cannot drag torch forward,
    and a REAL conv2d afterwards, because cuBLAS can work while cuDNN is
    broken and `import torch` proves neither.
    """
    import subprocess
    notes: list[str] = []
    for mod, spec in (("unsloth", "unsloth"), ("bitsandbytes", "bitsandbytes")):
        try:
            __import__(mod)
            continue
        except ImportError:
            pass
        log(f"installing {spec} (missing) ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "--no-deps", spec], capture_output=True, text=True, encoding="utf-8")
        notes.append(f"{spec}: rc={r.returncode} {r.stderr[-160:].strip()}")
    if notes:
        # unsloth pulls a real dependency set; install it WITHOUT touching torch
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "unsloth_zoo", "trl", "peft", "accelerate",
                            "--no-deps"], capture_output=True, text=True, encoding="utf-8")
        notes.append(f"unsloth deps: rc={r.returncode}")
    import torch
    ok = False
    try:
        x = torch.randn(1, 1, 8, 8, device="cuda")
        w = torch.randn(1, 1, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w)
        torch.cuda.synchronize()
        ok = True
    except Exception as e:                                       # noqa: BLE001
        notes.append(f"⛔ POST-INSTALL CUDA conv2d FAILED: {type(e).__name__}: {e}")
    log(f"deps: {notes or 'all present'} | cuda_conv2d_ok={ok} "
        f"| torch={torch.__version__}")
    if not ok:
        raise RuntimeError("post-install CUDA check failed: " + " | ".join(notes))
    return notes


def load_model(model_name: str):
    """The programme's proven ladder (``colab/s2_lab_lib.load_vlm``): unsloth
    4-bit first. ⛔ 4-bit is not an optimisation on a 16 GB T4, it is the FIT —
    bf16 silently offloads to CPU/disk and crawls. Which loader won is returned
    so a silent fallback to a slow path cannot hide in the artifact."""
    import torch
    errs: list[str] = []
    try:
        from unsloth import FastVisionModel
        model, proc = FastVisionModel.from_pretrained(model_name, load_in_4bit=True)
        FastVisionModel.for_inference(model)
        return model, proc, "unsloth.FastVisionModel(load_in_4bit=True)", errs
    except Exception as e:                                       # noqa: BLE001
        errs.append(f"unsloth: {type(e).__name__}: {str(e)[:160]}")
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    # ⛔ AutoModelForCausalLM is LAST on purpose: it loads the text-only class
    # and discards the vision tower, so generate() then rejects pixel_values /
    # image_grid_thw / mm_token_type_ids and EVERY generation fails.
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                 "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_name, quantization_config=bnb,
                                        device_map="auto", trust_remote_code=True)
            return model, proc, f"transformers.{name} + bnb nf4", errs
        except Exception as e:                                   # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
    raise RuntimeError("no loader succeeded: " + " | ".join(errs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Defaults are the Colab paths so `colab exec -f` works with no argv;
    # Thor/pod pass them explicitly.
    ap.add_argument("--payload", default="/content/payload.json")
    ap.add_argument("--out", default="/content/vlm_tac_raw.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--kinds", default="lon,lane,sign")
    ap.add_argument("--sign-on-turns", action="store_true", default=True)
    ap.add_argument("--all-sign", dest="sign_on_turns", action="store_false",
                    help="run the sign call on every clip (see TURN_LANES)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--install-deps", action="store_true",
                    help="pip-install unsloth/bitsandbytes if missing. "
                         "EPHEMERAL VMs ONLY — see ensure_deps().")
    # ⛔ NOT parse_args: under `colab exec -f` this runs INSIDE a Jupyter
    # kernel, whose sys.argv is the kernel's own
    # (`-f /root/.../kernel-xxxx.json`). parse_args would abort the whole run on
    # "unrecognized arguments" before a single generation. parse_known_args
    # keeps the defaults — which ARE the Colab paths — and the ignored tokens
    # are logged rather than swallowed.
    args, ignored = ap.parse_known_args(argv)
    if ignored:
        print(f"[vlm] ignoring non-argv tokens (kernel context): {ignored}",
              flush=True)

    import torch
    from PIL import Image

    with open(args.payload, encoding="utf-8") as f:
        pl = json.load(f)
    samp = pl["sampling"]
    kinds = tuple(k for k in args.kinds.split(",") if k in KINDS)

    done = load_done(args.out)
    jobs = []
    for c in pl["clips"]:
        for kind in kinds:
            if kind == "sign" and not wants_sign(c, args.sign_on_turns):
                continue
            if (c["clip_id"], kind) in done:
                continue
            jobs.append((c, kind))
    if args.limit:
        jobs = jobs[:args.limit]
    log(f"payload {len(pl['clips'])} clips | kinds {kinds} | "
        f"sign_on_turns={args.sign_on_turns}")
    log(f"already banked {len(done)} | TODO {len(jobs)} generations")
    if not jobs:
        log("nothing to do — the output is already complete for this payload")
        return 0

    if args.install_deps:
        ensure_deps()
    model, proc, loader, errs = load_model(args.model)
    model.eval()
    log(f"loader={loader} weights={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
        f"loader_errors={errs}")

    # decode each clip's frames once, not once per kind
    frames: dict[str, list] = {}

    def imgs_for(c):
        if c["clip_id"] not in frames:
            frames.clear()          # one clip's frames at a time; they are big
            frames[c["clip_id"]] = [
                Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
                for b in c["frames_b64"]]
        return frames[c["clip_id"]]

    batch = max(1, args.batch)
    n_ok = n_err = n_cap = 0
    t_start = time.time()
    tok_total = 0
    fh = open(args.out, "a", encoding="utf-8")
    i = 0
    while i < len(jobs):
        grp = jobs[i:i + batch]
        torch.cuda.reset_peak_memory_stats()
        try:
            inp = build_inputs(proc, [c["calls"][k]["no_ego"] for c, k in grp],
                               [imgs_for(c) for c, _ in grp]).to(model.device)
            g0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=args.cap, do_sample=True,
                    temperature=samp["temperature"], top_p=samp["top_p"],
                    top_k=samp["top_k"],
                    repetition_penalty=samp["repetition_penalty"],
                    pad_token_id=proc.tokenizer.pad_token_id)
            dt = time.time() - g0
            new = out[:, inp["input_ids"].shape[1]:]
            txts = proc.batch_decode(new, skip_special_tokens=True)
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            for j, (c, kind) in enumerate(grp):
                ntok = int((new[j] != proc.tokenizer.pad_token_id).sum())
                rec = {"clip_id": c["clip_id"], "kind": kind, "raw": txts[j],
                       "n_new_tokens": ntok, "hit_cap": ntok >= args.cap,
                       # ⚠️ `<think>` is opened by the chat template and stripped
                       # as a special token, so its ABSENCE proves nothing. The
                       # CLOSING tag is what shows the trace finished.
                       "closed_think": "</think>" in txts[j],
                       "batch_secs": round(dt, 1), "batch": len(grp),
                       "peak_gb": peak, "cap": args.cap, "loader": loader}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                emit(rec)
                tok_total += ntok
                n_ok += 1
                n_cap += int(rec["hit_cap"])
            fh.flush()
            os.fsync(fh.fileno())
            el = time.time() - t_start
            log(f"  [{n_ok}/{len(jobs)}] b={len(grp)} {dt:.0f}s peak={peak}GB "
                f"| {tok_total/el:.1f} tok/s | cap-hits {n_cap}/{n_ok} "
                f"| ETA {(len(jobs)-n_ok)*el/max(n_ok,1)/3600:.1f}h")
            i += batch
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            if batch > 1:
                batch = max(1, batch // 2)
                log(f"  OOM -> stepping batch down to {batch} and retrying")
                continue
            for c, kind in grp:
                rec = {"clip_id": c["clip_id"], "kind": kind,
                       "error": f"OutOfMemoryError: {str(e)[:200]}"}
                fh.write(json.dumps(rec) + "\n")
                emit(rec)
                n_err += 1
            fh.flush()
            os.fsync(fh.fileno())
            i += batch
        except Exception as e:                                   # noqa: BLE001
            for c, kind in grp:
                rec = {"clip_id": c["clip_id"], "kind": kind,
                       "error": f"{type(e).__name__}: {str(e)[:200]}",
                       "tb": traceback.format_exc()[-600:]}
                fh.write(json.dumps(rec) + "\n")
                emit(rec)
                n_err += 1
            fh.flush()
            os.fsync(fh.fileno())
            log(f"  ERROR on {len(grp)} job(s): {type(e).__name__}: {e}")
            i += batch
    fh.close()
    el = time.time() - t_start
    log(f"DONE ok={n_ok} err={n_err} cap_hits={n_cap} "
        f"wall={el/60:.1f}min {tok_total/max(el,1):.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
