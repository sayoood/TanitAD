#!/usr/bin/env python3
"""Post-training gate: name every registered parameter that NEVER received a gradient.

THE DEFECT CLASS
----------------
A head can be built, unit-tested, registered on the module, counted in the parameter
breakdown, and still never be reached by the loss.  It then sits at its init for the whole
run and is shipped.  In this programme that has now surfaced five times, always too late:
``tac_goal_tok_head`` (11,286 params, zero gradient for all 40,284 steps of refcv5-v2),
``core.decoder.offset_head`` (6,160 at random init in a published arm), ``scorer.goal_point``
(1,026, zero-init AND zero gradient, so it emits the constant origin), the >=GT truncation
whose caller had no ``--gt-bar`` flag, and a spec guard that existed only inside its own test.
Every one passed its own unit tests.  "Rollable" and "trained" are different claims.

THE METHOD IS ANALYTIC, WHICH IS WHY IT IS TRUSTWORTHY
-----------------------------------------------------
``torch.optim.Adam`` allocates per-parameter state LAZILY -- on the first ``step()`` at which
``p.grad is not None``.  A parameter that appears in ``opt["param_groups"][*]["params"]`` but
has NO entry in ``opt["state"]`` therefore never received a gradient.  This is a property of
the checkpoint, not a statistic: it needs no threshold, no seed, and no re-run.

THE PART THAT COULD BE WRONG, AND HOW IT IS PINNED
--------------------------------------------------
The optimizer's ``state_dict`` keys are POSITIONAL ids, not names.  Recovering id -> name is
the only inferential step in the gate, so it is made refutable rather than plausible:

  * candidates are the model ``state_dict``'s floating tensors minus a DECLARED buffer-name
    filter, and every excluded / unconsumed candidate is named in the JSON;
  * ids are embedded into the candidate list order-preservingly, with each id that HAS state
    required to match its candidate's shape EXACTLY;
  * the embedding is computed TWICE -- leftmost and rightmost.  An id's position is FORCED
    exactly when the two agree, because every feasible embedding places it at >= leftmost and
    <= rightmost.  Any disagreement is reported as AMBIGUOUS and the gate returns
    INCONCLUSIVE.  It refuses rather than guesses.

Three cross-checks, none of which is derived from the thing being measured, are recorded in
the JSON and must all pass:

  1. every id WITH state matches its aligned parameter's shape exactly;
  2. the mapped parameter total equals ``config.json``'s ``param_breakdown.total`` -- an
     INDEPENDENTLY AUTHORED reference this gate never touches;
  3. same-breath controls read TRAINED with a NONZERO ``exp_avg``, which excludes a
     global-zero artifact.  A count of zero from a probe that could not read anything is
     indistinguishable from a genuine zero, so the controls must read non-zero in the same
     breath as the untrained count.

THE VERDICT IS THE JSON FILE, NEVER THE EXIT CODE
-------------------------------------------------
Measured in this programme: a 25-minute ``timeout`` killed a pre-launch gate with zero output
and no JSON written, and its wrapper printed ``GATE_EXIT=0``.  The admissible evidence that
this gate did not run is the MISSING JSON.  The JSON is written atomically (tmp + replace) so
a death mid-write cannot leave a truncated artifact that reads as finished, and ``status`` is
only ever ``PASS`` when everything needed was actually read.  Exit codes are a convenience:

    0  PASS          -- no untrained parameters outside the allow-list
    1  FAIL          -- untrained parameters found that are not allow-listed
    2  INCONCLUSIVE  -- something could not be read, or the naming could not be pinned

ALLOW-LIST, NOT A SILENT PASS
-----------------------------
Some parameters are legitimately frozen (refcv5-v2 trains 8,634,120 of its parameters on
purpose).  A gate that cannot tell DELIBERATELY FROZEN from ACCIDENTALLY UNREACHED gets
switched off within a week, so the expected-frozen set is explicit and anything outside it
fails.  An allow-list entry that matches nothing is reported as stale (and fails under
``--strict-allowlist``), because an allow-list nobody prunes is how a gate rots.

USAGE
-----
    python stack/scripts/check_untrained_params.py CKPT --out gate.json \
        [--expect-frozen NAME]... [--expect-frozen-file f.json] \
        [--control NAME]... [--param-total-ref N | --config CONFIG | --no-param-total-ref] \
        [--strict-allowlist] [--md5]

``NAME`` matches a parameter exactly, or as a module prefix (``a.b`` covers ``a.b.weight``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import traceback

SCHEMA = "untrained-params-gate-v1"

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"

EXIT = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}

# Declared buffer-name filter.  These are registered buffers that live in a module's
# state_dict beside its parameters but are never handed to the optimizer.  The filter only
# narrows the search; a filter error surfaces as a shape mismatch or an ambiguous position,
# both of which return INCONCLUSIVE rather than a wrong name.
DEFAULT_BUFFER_PATTERNS = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "pos_emb_cache",
    ".anchors",
)

# Keys under which a checkpoint may carry the model / optimizer state dicts.  The gate
# refuses rather than guessing if none is present.
MODEL_KEYS = ("model", "model_state", "state_dict", "model_state_dict", "net")
OPT_KEYS = ("opt", "optimizer", "optim", "optimizer_state_dict", "opt_state")

# Per-parameter state entries that prove lazy allocation is in use.  SGD without momentum
# allocates NO state at all, which would make every parameter look untrained -- a silent
# false-positive generator.  The gate detects that and returns INCONCLUSIVE.
LAZY_STATE_KEYS = ("exp_avg", "momentum_buffer", "square_avg", "sum", "exp_inf")
SHAPE_STATE_KEYS = ("exp_avg", "momentum_buffer", "square_avg", "sum", "exp_inf")


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: str, payload: dict) -> None:
    """Write the verdict atomically, in UTF-8, ASCII-escaped.

    A crash mid-write must not leave a file that reads as finished; a cp1252 console must not
    be able to kill the artifact either, so nothing non-ASCII is ever emitted.
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, ensure_ascii=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _pick(d: dict, keys) -> "tuple[str | None, object]":
    for k in keys:
        if isinstance(d, dict) and k in d:
            return k, d[k]
    return None, None


def _matches(name: str, entry: str) -> bool:
    """Allow-list matching: exact parameter name, or a module prefix."""
    return name == entry or name.startswith(entry + ".")


def _embed(id_shapes, cand_shapes, reverse: bool):
    """Order-preserving embedding of the id sequence into the candidate sequence.

    ``id_shapes[k]`` is the shape recorded in the optimizer state for id k, or None when that
    id has no state (and therefore carries no shape constraint).  Returns a list mapping each
    id index to a candidate index, or None if no embedding exists in this direction.
    """
    n = len(id_shapes)
    m = len(cand_shapes)
    out = [None] * n
    order = range(n - 1, -1, -1) if reverse else range(n)
    step = -1 if reverse else 1
    ptr = m - 1 if reverse else 0
    for k in order:
        want = id_shapes[k]
        if want is not None:
            while 0 <= ptr < m and cand_shapes[ptr] != want:
                ptr += step
        if not (0 <= ptr < m):
            return None
        out[k] = ptr
        ptr += step
    return out


def run(args) -> dict:
    """Build the verdict payload.  Never raises for a data problem -- returns INCONCLUSIVE."""
    report: dict = {
        "schema": SCHEMA,
        "status": INCONCLUSIVE,
        "reasons": [],
        "generated_utc": _utcnow(),
        "argv": list(sys.argv[1:]),
        "checkpoint": {"path": os.path.abspath(args.ckpt)},
    }
    reasons = report["reasons"]

    def inconclusive(msg: str) -> dict:
        reasons.append(msg)
        report["status"] = INCONCLUSIVE
        return report

    # --- read the checkpoint -------------------------------------------------------------
    try:
        report["checkpoint"]["bytes"] = os.path.getsize(args.ckpt)
    except OSError as exc:
        return inconclusive("checkpoint not stat-able: %s" % exc)

    if args.md5:
        try:
            report["checkpoint"]["md5"] = _md5(args.ckpt)
        except OSError as exc:
            return inconclusive("checkpoint not readable for md5: %s" % exc)

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment problem
        return inconclusive("torch not importable: %s" % exc)

    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:
        return inconclusive("checkpoint not loadable: %s: %s" % (type(exc).__name__, exc))

    if not isinstance(ck, dict):
        return inconclusive("checkpoint is a %s, not a dict of state dicts" % type(ck).__name__)

    mkey, sd = _pick(ck, MODEL_KEYS)
    okey, opt = _pick(ck, OPT_KEYS)
    report["checkpoint"]["model_key"] = mkey
    report["checkpoint"]["optimizer_key"] = okey
    report["checkpoint"]["step"] = ck.get("step")
    if sd is None:
        return inconclusive("no model state dict under any of %s" % (MODEL_KEYS,))
    if opt is None:
        return inconclusive("no optimizer state dict under any of %s" % (OPT_KEYS,))
    if not isinstance(opt, dict) or "param_groups" not in opt or "state" not in opt:
        return inconclusive("optimizer state dict has no param_groups/state")

    st = opt["state"]
    ids = []
    for g in opt["param_groups"]:
        ids.extend(g["params"])
    n_reg, n_state = len(ids), len(st)
    report["optimizer"] = {
        "n_groups": len(opt["param_groups"]),
        "n_registered": n_reg,
        "n_with_state": n_state,
        "duplicate_ids": len(ids) - len(set(ids)),
    }

    if len(ids) != len(set(ids)):
        return inconclusive("a parameter id appears in more than one group; ids are ambiguous")

    # --- precondition: the optimizer must allocate state LAZILY --------------------------
    state_keys = sorted({k for e in st.values() if isinstance(e, dict) for k in e})
    report["optimizer"]["state_entry_keys"] = state_keys
    shape_key = next((k for k in SHAPE_STATE_KEYS if k in state_keys), None)
    report["optimizer"]["shape_source_key"] = shape_key
    lazy = bool(set(state_keys) & set(LAZY_STATE_KEYS))
    report["optimizer"]["lazy_state_supported"] = lazy
    if n_state == 0:
        return inconclusive(
            "optimizer state is EMPTY: this optimizer allocates no per-parameter state "
            "(e.g. plain SGD), so absence of state does not imply absence of gradient"
        )
    if not lazy or shape_key is None:
        return inconclusive(
            "optimizer state entries carry none of %s; the lazy-allocation argument does not "
            "apply to this optimizer and the analytic method is invalid here" % (LAZY_STATE_KEYS,)
        )
    if n_state > n_reg:
        return inconclusive("more state entries (%d) than registered params (%d)" % (n_state, n_reg))

    # --- candidate list ------------------------------------------------------------------
    buf_patterns = () if args.no_buffer_filter else tuple(DEFAULT_BUFFER_PATTERNS)
    cand, nonfloat = [], []
    for name, t in sd.items():
        if not hasattr(t, "is_floating_point"):
            continue
        if not t.is_floating_point():
            nonfloat.append(name)
            continue
        if any(p in name for p in buf_patterns):
            continue
        cand.append(name)
    cand_shapes = [tuple(sd[n].shape) for n in cand]

    id_shapes = []
    for pid in ids:
        e = st.get(pid)
        if isinstance(e, dict) and shape_key in e and hasattr(e[shape_key], "shape"):
            id_shapes.append(tuple(e[shape_key].shape))
        else:
            id_shapes.append(None)
    n_constrained = sum(1 for s in id_shapes if s is not None)

    report["alignment"] = {
        "method": "order-preserving shape embedding; leftmost+rightmost uniqueness",
        "n_state_dict_tensors": len(sd),
        "n_non_floating_excluded": len(nonfloat),
        "buffer_name_filter": list(buf_patterns),
        "n_candidates": len(cand),
        "slack": len(cand) - n_reg,
    }
    if len(cand) < n_reg:
        return inconclusive(
            "only %d candidate parameters for %d registered ids: the buffer filter excluded "
            "real parameters, or the model state dict does not match this optimizer"
            % (len(cand), n_reg)
        )

    left = _embed(id_shapes, cand_shapes, reverse=False)
    right = _embed(id_shapes, cand_shapes, reverse=True)
    if left is None or right is None:
        return inconclusive(
            "no order-preserving shape embedding exists (leftmost=%s, rightmost=%s): the "
            "optimizer ids cannot be aligned to this model state dict"
            % (left is not None, right is not None)
        )

    ambiguous = [k for k in range(n_reg) if left[k] != right[k]]
    report["alignment"]["forced"] = not ambiguous
    report["alignment"]["n_ambiguous"] = len(ambiguous)
    report["alignment"]["ambiguous"] = [
        {
            "index": k,
            "optimizer_id": ids[k],
            "leftmost": cand[left[k]],
            "rightmost": cand[right[k]],
            "has_state": id_shapes[k] is not None,
        }
        for k in ambiguous[:64]
    ]
    used = set(left)
    report["alignment"]["unconsumed_candidates"] = [
        {"name": cand[i], "shape": list(cand_shapes[i]), "numel": int(sd[cand[i]].numel())}
        for i in range(len(cand))
        if i not in used
    ]
    if ambiguous:
        return inconclusive(
            "id->name alignment is AMBIGUOUS at %d of %d positions; refusing to guess a name"
            % (len(ambiguous), n_reg)
        )

    names = [cand[left[k]] for k in range(n_reg)]

    # --- cross-check 1: every id WITH state matches its aligned shape exactly ------------
    bad_shape = [
        {"index": k, "optimizer_id": ids[k], "name": names[k],
         "state_shape": list(id_shapes[k]), "param_shape": list(cand_shapes[left[k]])}
        for k in range(n_reg)
        if id_shapes[k] is not None and cand_shapes[left[k]] != id_shapes[k]
    ]
    cc = report["cross_checks"] = {}
    cc["shape_match"] = {
        "matched": n_constrained - len(bad_shape),
        "total": n_constrained,
        "ok": not bad_shape,
        "mismatches": bad_shape[:32],
    }

    # --- cross-check 2: mapped total vs an independently authored reference --------------
    mapped_total = int(sum(int(sd[n].numel()) for n in names))
    ref, ref_src = None, None
    if args.param_total_ref is not None:
        ref, ref_src = int(args.param_total_ref), "--param-total-ref"
    elif not args.no_param_total_ref:
        cfg_path = args.config
        if cfg_path is None:
            guess = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.json")
            cfg_path = guess if os.path.exists(guess) else None
        if cfg_path is not None:
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                ref = int(cfg["param_breakdown"]["total"])
                ref_src = "%s:param_breakdown.total" % os.path.abspath(cfg_path)
            except Exception as exc:
                cc["param_total"] = {
                    "mapped": mapped_total, "reference": None, "source": cfg_path,
                    "ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                }
                return inconclusive(
                    "param_breakdown.total not readable from %s (%s); pass --param-total-ref "
                    "or --no-param-total-ref to say so explicitly" % (cfg_path, exc)
                )
    if ref is None and not args.no_param_total_ref:
        cc["param_total"] = {"mapped": mapped_total, "reference": None, "source": None, "ok": False}
        return inconclusive(
            "no independent parameter-total reference found (no config.json beside the "
            "checkpoint); pass --config/--param-total-ref, or --no-param-total-ref to waive it"
        )
    cc["param_total"] = {
        "mapped": mapped_total,
        "reference": ref,
        "source": ref_src,
        "waived": bool(args.no_param_total_ref and ref is None),
        "ok": (ref is None) or (mapped_total == ref),
    }

    # --- cross-check 3: same-breath controls that MUST read non-zero ---------------------
    def _abs_sum(pid):
        e = st.get(pid)
        if not isinstance(e, dict) or shape_key not in e:
            return None
        try:
            return float(e[shape_key].abs().sum())
        except Exception:
            return None

    nonzero = zero = 0
    for k in range(n_reg):
        if id_shapes[k] is None:
            continue
        v = _abs_sum(ids[k])
        if v is None:
            continue
        if v > 0.0:
            nonzero += 1
        else:
            zero += 1
    auto_ok = nonzero > 0
    cc["controls"] = {
        "auto": {
            "n_with_state": n_state,
            "n_nonzero_optimizer_moment": nonzero,
            "n_zero_optimizer_moment": zero,
            "moment_key": shape_key,
            "ok": auto_ok,
            "note": "at least one trained parameter must carry a NONZERO optimizer moment; a "
                    "count of zero untrained params from a probe that read nothing would "
                    "otherwise be indistinguishable from a genuine zero",
        },
        "named": [],
    }
    for want in args.control:
        hit = next((k for k in range(n_reg) if _matches(names[k], want) or want in names[k]), None)
        if hit is None:
            cc["controls"]["named"].append({"name": want, "found": False, "ok": False})
            continue
        v = _abs_sum(ids[hit])
        cc["controls"]["named"].append({
            "name": want,
            "found": True,
            "resolved": names[hit],
            "has_state": id_shapes[hit] is not None,
            "optimizer_moment_abs_sum": v,
            "ok": bool(id_shapes[hit] is not None and v is not None and v > 0.0),
        })

    # --- the finding ---------------------------------------------------------------------
    allow = list(args.expect_frozen)
    if args.expect_frozen_file:
        try:
            with open(args.expect_frozen_file, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            if isinstance(blob, dict):
                blob = blob.get("expect_frozen", [])
            allow.extend(str(x) for x in blob)
        except Exception as exc:
            return inconclusive(
                "expect-frozen file not readable: %s: %s" % (type(exc).__name__, exc))

    untrained, matched_entries = [], set()
    for k in range(n_reg):
        if id_shapes[k] is not None:
            continue
        name = names[k]
        t = sd[name]
        try:
            absmax = float(t.abs().max()) if t.numel() else 0.0
            n_nonzero = int((t != 0).sum())
        except Exception:
            absmax, n_nonzero = None, None
        entry = next((e for e in allow if _matches(name, e)), None)
        if entry is not None:
            matched_entries.add(entry)
        untrained.append({
            "name": name,
            "optimizer_id": ids[k],
            "shape": list(cand_shapes[left[k]]),
            "numel": int(t.numel()),
            "exactly_zero": (None if absmax is None else absmax == 0.0),
            "absmax": absmax,
            "n_nonzero": n_nonzero,
            "allowed": entry is not None,
            "allowlist_entry": entry,
        })

    unexpected = [u for u in untrained if not u["allowed"]]
    report["untrained"] = {
        "count_params": len(untrained),
        "count_numel": int(sum(u["numel"] for u in untrained)),
        "params": untrained,
    }
    report["allowlist"] = {
        "entries": allow,
        "unmatched": [e for e in allow if e not in matched_entries],
        "strict": bool(args.strict_allowlist),
    }
    report["unexpected_untrained"] = {
        "count_params": len(unexpected),
        "count_numel": int(sum(u["numel"] for u in unexpected)),
        "names": [u["name"] for u in unexpected],
    }

    # --- verdict -------------------------------------------------------------------------
    if not cc["shape_match"]["ok"]:
        return inconclusive(
            "%d of %d ids with optimizer state do NOT match their aligned parameter's shape; "
            "naming is not pinned" % (len(bad_shape), n_constrained))
    if not cc["param_total"]["ok"]:
        return inconclusive(
            "mapped parameter total %d disagrees with the independently authored reference %d "
            "(%s): the id->name mapping is wrong" % (mapped_total, ref, ref_src))
    if not auto_ok:
        return inconclusive(
            "same-breath control FAILED: no trained parameter carries a nonzero optimizer "
            "moment, so a zero untrained count would be a claim about this probe")
    bad_controls = [c for c in cc["controls"]["named"] if not c["ok"]]
    if bad_controls:
        return inconclusive(
            "named same-breath control(s) failed: %s" % ", ".join(c["name"] for c in bad_controls))

    if unexpected:
        report["status"] = FAIL
        reasons.append(
            "%d parameter(s) totalling %d never received a gradient and are NOT allow-listed: %s"
            % (len(unexpected), report["unexpected_untrained"]["count_numel"],
               ", ".join(u["name"] for u in unexpected)))
        return report

    if args.strict_allowlist and report["allowlist"]["unmatched"]:
        report["status"] = FAIL
        reasons.append(
            "allow-list is STALE under --strict-allowlist: %s matched nothing untrained"
            % ", ".join(report["allowlist"]["unmatched"]))
        return report

    report["status"] = PASS
    if untrained:
        reasons.append(
            "%d parameter(s) totalling %d never received a gradient; all are allow-listed"
            % (len(untrained), report["untrained"]["count_numel"]))
    else:
        reasons.append("every registered parameter has optimizer state")
    if report["allowlist"]["unmatched"]:
        reasons.append(
            "WARNING: allow-list entries matched nothing untrained (stale): %s"
            % ", ".join(report["allowlist"]["unmatched"]))
    return report


def _print_summary(r: dict) -> None:
    """ASCII only.  A cp1252 console must never be able to kill the summary."""
    out = sys.stdout
    print("=" * 78, file=out)
    print("UNTRAINED-PARAMETER GATE: %s" % r["status"], file=out)
    print("=" * 78, file=out)
    print("checkpoint : %s" % r["checkpoint"].get("path"), file=out)
    o = r.get("optimizer") or {}
    if o:
        print("registered : %s    with optimizer state: %s"
              % (o.get("n_registered"), o.get("n_with_state")), file=out)
    a = r.get("alignment") or {}
    if a:
        print("alignment  : %d candidates, slack %s, forced=%s, ambiguous=%s"
              % (a.get("n_candidates", -1), a.get("slack"), a.get("forced"),
                 a.get("n_ambiguous")), file=out)
    cc = r.get("cross_checks") or {}
    if cc:
        sm, pt, ct = cc.get("shape_match", {}), cc.get("param_total", {}), cc.get("controls", {})
        print("check 1/3  : shape match %s/%s  ok=%s"
              % (sm.get("matched"), sm.get("total"), sm.get("ok")), file=out)
        print("check 2/3  : mapped %s vs reference %s  ok=%s"
              % (pt.get("mapped"), pt.get("reference"), pt.get("ok")), file=out)
        auto = ct.get("auto", {})
        print("check 3/3  : %s/%s trained params carry a nonzero moment  ok=%s"
              % (auto.get("n_nonzero_optimizer_moment"), auto.get("n_with_state"),
                 auto.get("ok")), file=out)
        for c in ct.get("named", []):
            print("             control %-34s %s"
                  % (c.get("name"), "TRAINED ok" if c.get("ok") else "FAILED"), file=out)
    u = r.get("untrained")
    if u:
        print("-" * 78, file=out)
        print("NEVER RECEIVED A GRADIENT: %d parameter(s), %d values"
              % (u["count_params"], u["count_numel"]), file=out)
        for p in u["params"]:
            print("   %-46s %-14s numel=%-8d %s%s"
                  % (p["name"], str(tuple(p["shape"])), p["numel"],
                     "EXACTLY-ZERO " if p["exactly_zero"] else "",
                     "allowed(%s)" % p["allowlist_entry"] if p["allowed"] else "UNEXPECTED"),
                  file=out)
    print("-" * 78, file=out)
    for msg in r.get("reasons", []):
        print("* %s" % msg, file=out)
    print("=" * 78, file=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Name every registered parameter that never received a gradient.")
    ap.add_argument("ckpt", help="checkpoint path (loaded on CPU; the GPU is never touched)")
    ap.add_argument("--out", required=True,
                    help="JSON verdict path (REQUIRED: the JSON is the verdict, not the exit code)")
    ap.add_argument("--expect-frozen", action="append", default=[], metavar="NAME",
                    help="parameter or module prefix that is DELIBERATELY frozen (repeatable)")
    ap.add_argument("--expect-frozen-file", default=None,
                    help="JSON list, or {'expect_frozen': [...]}, of deliberately frozen names")
    ap.add_argument("--strict-allowlist", action="store_true",
                    help="FAIL when an allow-list entry matches nothing untrained (stale entry)")
    ap.add_argument("--control", action="append", default=[], metavar="NAME",
                    help="parameter that MUST read trained with a nonzero moment (repeatable)")
    ap.add_argument("--config", default=None,
                    help="config.json carrying param_breakdown.total "
                         "(default: config.json beside the checkpoint)")
    ap.add_argument("--param-total-ref", type=int, default=None,
                    help="independently authored total parameter count to cross-check against")
    ap.add_argument("--no-param-total-ref", action="store_true",
                    help="explicitly waive cross-check 2 (recorded in the JSON as waived)")
    ap.add_argument("--no-buffer-filter", action="store_true",
                    help="debug: disable the declared buffer-name filter")
    ap.add_argument("--md5", action="store_true", help="record the checkpoint's md5 in the JSON")
    ap.add_argument("--quiet", action="store_true", help="suppress the stdout summary")
    args = ap.parse_args(argv)

    try:
        report = run(args)
    except Exception:
        report = {
            "schema": SCHEMA,
            "status": INCONCLUSIVE,
            "reasons": ["gate raised an unexpected exception; see traceback"],
            "generated_utc": _utcnow(),
            "argv": list(sys.argv[1:]),
            "checkpoint": {"path": os.path.abspath(args.ckpt)},
            "traceback": traceback.format_exc(),
        }

    try:
        _write_json_atomic(args.out, report)
    except Exception as exc:
        print("GATE COULD NOT WRITE ITS VERDICT to %s: %s: %s"
              % (args.out, type(exc).__name__, exc), file=sys.stderr)
        return EXIT[INCONCLUSIVE]

    if not args.quiet:
        _print_summary(report)
    return EXIT[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
