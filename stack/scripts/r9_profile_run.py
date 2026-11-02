"""r9_profile_run.py -- per-phase step-time profiler for the refcv9 E1 perception rig (`refc_v3_train.py --r9-perc-rig`).

Runs the REAL trainer in-process (`refc_v3_train.main(argv)`), with module-level timing wrappers that each bracket the phase with
``torch.cuda.synchronize()``. It changes NO trainer code and nothing on disk in the tree; the wrappers live only in this process.
Phases per step: data (blocked in next(iter), GPU idle), h2d (frames_to_device), fwd_loss (compute_losses_v3 minus h2d: forward +
loss terms incl. Hungarian), bwd (zero_grad..clip entry), clip, opt, ema, tail (log row / ckpt / eval / anything else up to the next
next_train_batch).

usage: python r9_profile_run.py <out.json> -- <trainer argv...>
"""
import sys, os, time, json, atexit
here = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.environ.get("R9_SCRIPTS", here)        # the tree's stack/scripts (the trainer imports its siblings from there)
STOP_AFTER = int(os.environ.get("R9_PROF_STEPS", "0"))   # 0 = run the whole argv; N = dump + _exit after N closed steps
# import-safe (the lander imports every landed module): the argv is read only when it carries "--"
out_path = sys.argv[1] if "--" in sys.argv[1:] else None
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv[1:] else []
sys.path.insert(0, SCRIPTS)
import torch
import refc_v3_train as T

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

REC = []          # one dict per step
cur = {}
marks = {"t_prev_end": None}

def wrap_next(orig):
    def f(*a, **k):
        sync()
        t1 = time.perf_counter()
        if cur:                                   # close the previous step
            cur["tail"] = t1 - marks["t_last"]
            cur["total"] = t1 - cur["_t_start"]
            cur["phase"] = "prof" if STATE["prof"] is not None else STATE["phase"]
            REC.append(dict(cur)); cur.clear()
            n = len(REC)
            for pl in PLAN:
                if pl["at"] == n:
                    for nm in pl["set"]: apply_toggle(nm)
                    STATE["phase"] = pl["name"]
            if STATE["prof"] is not None:
                STATE["prof_left"] -= 1
                if STATE["prof_left"] <= 0:
                    STATE["prof"].stop()
                    tab = STATE["prof"].key_averages().table(sort_by="self_cuda_time_total", row_limit=45, max_name_column_width=70)
                    open(out_path + ".torchprof.txt", "a").write("== profile at step %d (phase %s)\n%s\n" % (n, STATE["phase"], tab))
                    STATE["prof"] = None
            if n in PROF_AT and STATE["prof"] is None:
                STATE["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                STATE["prof"].start(); STATE["prof_left"] = 2
            if STOP_AFTER and len(REC) >= STOP_AFTER:
                _dump(); sys.stdout.flush(); os._exit(0)
        t_in = time.perf_counter()
        r = orig(*a, **k)
        t2 = time.perf_counter()
        cur["_t_start"] = t1
        cur["data"] = t2 - t_in
        marks["t_last"] = t2
        return r
    return f

def wrap_loss(orig):
    def f(*a, **k):
        sync(); t0 = time.perf_counter()
        cur["_h2d_before"] = cur.get("h2d", 0.0)
        r = orig(*a, **k)
        sync(); t1 = time.perf_counter()
        cur["fwd_loss"] = (t1 - t0) - (cur.get("h2d", 0.0) - cur["_h2d_before"])
        marks["t_last"] = t1
        return r
    return f

def wrap_h2d(orig):
    def f(x, device):
        sync(); t0 = time.perf_counter()
        r = orig(x, device)
        sync(); t1 = time.perf_counter()
        cur["h2d"] = cur.get("h2d", 0.0) + (t1 - t0)
        cur["h2d_bytes"] = cur.get("h2d_bytes", 0) + int(x.numel() * x.element_size())
        return r
    return f

def wrap_clip(orig):
    def f(*a, **k):
        sync(); t0 = time.perf_counter()
        cur["bwd"] = t0 - marks["t_last"]        # zero_grad / detector / backward / probes since the loss returned
        r = orig(*a, **k)
        sync(); t1 = time.perf_counter()
        cur["clip"] = t1 - t0
        marks["t_last"] = t1
        return r
    return f

def wrap_step(orig):
    def f(*a, **k):
        sync(); t0 = time.perf_counter()
        r = orig(*a, **k)
        sync(); t1 = time.perf_counter()
        cur["opt"] = t1 - t0
        marks["t_last"] = t1
        return r
    return f

def wrap_ema(orig):
    def f(*a, **k):
        sync(); t0 = time.perf_counter()
        r = orig(*a, **k)
        sync(); t1 = time.perf_counter()
        cur["ema"] = t1 - t0
        marks["t_last"] = t1
        return r
    return f

T.next_train_batch = wrap_next(T.next_train_batch)
T.compute_losses_v3 = wrap_loss(T.compute_losses_v3)
T.frames_to_device = wrap_h2d(T.frames_to_device)
torch.nn.utils.clip_grad_norm_ = wrap_clip(torch.nn.utils.clip_grad_norm_)
_bo = T.build_optimizer
def _build_optimizer(model, args):
    opt = _bo(model, args)
    opt.step = wrap_step(opt.step)               # instance attribute shadows the bound method
    return opt
T.build_optimizer = _build_optimizer
T._wema.WeightEMA.update = wrap_ema(T._wema.WeightEMA.update)


# ---------------------------------------------------------------- detail (sync-timed components) ----------------------------
DETAIL = os.environ.get("R9_PROF_DETAIL", "0") == "1"
PLAN = json.loads(os.environ.get("R9_PROF_PLAN", "[]"))          # [{"at": n_closed_steps, "name": "...", "set": ["ckpt0", ...]}]
PROF_AT = [int(x) for x in os.environ.get("R9_PROF_TORCH", "").split(",") if x.strip()]   # closed-step indices where a 2-step torch profile starts
STATE = {"model": None, "phase": "base", "prof": None, "prof_left": 0}

def timed(label, fn):
    def f(*a, **k):
        sync(); t0 = time.perf_counter()
        r = fn(*a, **k)
        sync(); cur["f_" + label] = cur.get("f_" + label, 0.0) + (time.perf_counter() - t0)
        return r
    return f

def hook_module(mod, label):
    t = {}
    def pre(m, a):
        sync(); t["t0"] = time.perf_counter()
    def post(m, a, o):
        sync(); cur["f_" + label] = cur.get("f_" + label, 0.0) + (time.perf_counter() - t["t0"])
    mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)

def attach_detail(model):
    STATE["model"] = model
    enc = model.core.encoder
    enc.forward_features_s8 = timed("trunk_all", enc.forward_features_s8)
    enc.s8_from_normalised = timed("trunk_s8pass", enc.s8_from_normalised)
    enc._backbone = timed("trunk_backbone", enc._backbone)
    if getattr(model, "_map_hires", None) is not None: hook_module(model._map_hires, "maphires_mod")
    if getattr(model, "_perception", None) is not None: hook_module(model._perception, "percep_mod")
    if getattr(model.core, "agent_head", None) is not None: hook_module(model.core.agent_head, "agent_head")
    T._refc_agents.agent_losses = timed("loss_agent", T._refc_agents.agent_losses)
    T._perc.box3d_loss_row = timed("loss_box3d", T._perc.box3d_loss_row)
    T._map_hires_loss = timed("loss_maphires", T._map_hires_loss)
    T._det_metrics.window_packs = timed("det_packs", T._det_metrics.window_packs)

def apply_toggle(name):
    m = STATE["model"]
    if name == "ckpt0":
        m.core.encoder.memory_levers["chunk_ckpt"] = 0
    elif name == "mhckpt0":
        m._map_hires.cfg.grad_ckpt = False
    elif name == "nocompile":
        m.core.encoder._net_fn = None
    elif name == "cudnnbench":
        torch.backends.cudnn.benchmark = True
    elif name in ("evalfull", "evalnew"):          # the held-out eval dataset (main-process decode): full 28-row window vs the newest row only
        import gc
        eds = [o for o in gc.get_objects() if isinstance(o, T.V3Dataset) and len(o) == 23772]
        if not eds:
            raise SystemExit("no eval dataset (23772 windows) found")
        for o in eds:
            o.newest_row_only = (name == "evalnew")
    else:
        raise SystemExit("unknown toggle " + name)
    print("[r9prof] toggle applied: " + name, flush=True)

_save0 = torch.save
def _timed_save(*a, **k):
    t0 = time.perf_counter()
    r = _save0(*a, **k)
    cur["f_save"] = cur.get("f_save", 0.0) + (time.perf_counter() - t0)
    return r
torch.save = _timed_save

_bo2 = T.build_optimizer
def _build_optimizer2(model, args):
    if DETAIL:
        attach_detail(model)
    else:
        STATE["model"] = model
    return _bo2(model, args)
T.build_optimizer = _build_optimizer2

def _dump():
    keys = ["data", "h2d", "fwd_loss", "bwd", "clip", "opt", "ema", "tail", "total"]
    fkeys = sorted({k for r in REC for k in r if k.startswith("f_")})
    rows = [{**{k: round(r.get(k, 0.0), 4) for k in keys + fkeys}, "phase": r.get("phase", "base")} for r in REC]
    mem = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if torch.cuda.is_available() else None
    json.dump({"argv": argv, "n_steps": len(rows), "rows": rows, "peak_mem_gb": mem, "plan": PLAN}, open(out_path, "w"))
    print("[r9prof] n_steps %d  peak_mem %s GB (whole run)" % (len(rows), mem), flush=True)
    for ph in dict.fromkeys(r["phase"] for r in rows):
        sel = [r for r in rows if r["phase"] == ph][2:]            # drop the first 2 steps of a phase (compile / warm-up)
        if not sel:
            continue
        print("[r9prof] --- phase %s (n=%d)" % (ph, len(sel)), flush=True)
        for k in keys + fkeys:
            v = sorted(r[k] for r in sel)
            print("[r9prof] %-16s mean %7.3f  median %7.3f  min %7.3f  max %7.3f" % (k, sum(v) / len(v), v[len(v) // 2], v[0], v[-1]), flush=True)
atexit.register(lambda: (not REC) or _dump())

if __name__ == "__main__":
    T.main(argv)
