"""EGOWM read harness: IA-0 and the beside modules of ``ladder_read/lr_lib.py`` with the speed channel of an E-KIN / E-NOW checkpoint. A THIN layer: lr_lib is imported and never edited.

How speed enters the read (IMPL_CHOICES_EGOWM.md section 1 and 13):
  * the ENCODED states (Copy-Last, Oracle) are speed-free by construction (the stack's ``encode`` adds speed to the ACTION embeddings only). ``encode_clip`` is therefore called as lr_lib calls it, with the
    stack's ``_ego`` unset, which returns the speed-free embeddings the free run needs;
  * the free run ``stack.rollout(level, z0, actions, context=ctx, speed=[.., H])`` gets, per anchor and step, ``anchored_speeds(mode, v, acc, anchors, kmax, cum)`` -- the trainer's own function, the same one
    its k-step loss calls: E-NOW = the measured speed at the anchor frame, repeated; E-KIN = that speed integrated over the logged accelerations ``act[f, 0]`` of the frames the run covers, floored at 0;
  * under a CONTRAST (IA-3) the integrated accelerations are the contrasted ones (E-KIN follows the candidate action; E-NOW holds);
  * the measured speed is ``eval141_speed.npz`` (``eval_speed.py``): kin3[:, 0] of the native pose track. Row 0 of a clip (no backward pose) is filled with row 1 so that the anchor at frame 0 exists and the cell
    set equals the control's; the count is in the result (``speed_row0_filled``).
A model WITHOUT a speed stack is delegated to lr_lib's own functions unchanged (the control: R35H reads bit for bit its banked numbers).
Install by ``with speed_read(fr_speed) as ctl:``; the patch is process-local and undone on exit. Tier T0."""
import contextlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LADDER_READ_CODE_ROOT", "F:/TanitAD-scratch/egowm/tree")
THEORY = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(THEORY, "ladder_read"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402
import lr_lib as L  # noqa: E402
import eval_speed as ES  # noqa: E402
from tanitad.models.v8f_mini import anchored_speeds  # noqa: E402

FRAMES_SEED = {0: 0, 1: 1, 2: 0}      # the seed-2 replicate keeps the readers and zcache of M2b_s0 (IMPL_CHOICES_EGOWM.md section 3)
_STATE = {"speed": None, "cur": None, "orig_enc": None, "orig_roll": None, "n_roll": 0}


def attach_speed(fr, vd):
    """Register the measured speed of each clip of ``fr`` by the identity of its arrays (the callers pass ``fr['zmean'][ci]`` and ``fr['act'][ci]``). Returns the registry."""
    reg = {}
    for ci, c in enumerate(fr["clips"]):
        v = vd[c]
        if len(v) != len(fr["zmean"][ci]):
            raise RuntimeError(f"{c}: {len(v)} speeds vs {len(fr['zmean'][ci])} frames")
        reg[id(fr["zmean"][ci])] = (c, torch.as_tensor(v).float())
    return reg


def _encode_clip(model, zmean, act, contrast=None):
    out = _STATE["orig_enc"](model, zmean, act, contrast)
    if not hasattr(model.wm.stack, "spd"):
        return out
    reg = _STATE["speed"]
    if id(zmean) not in reg:
        raise RuntimeError("encode_clip got a clip with no registered measured speed (speed_read(...) was installed with another frame set)")
    c, v = reg[id(zmean)]
    a = torch.as_tensor(act).clone()
    if contrast is not None:
        ch, dv = contrast
        a[..., ch] += dv
    _STATE["cur"] = {"clip": c, "v": v[None], "acc": a[:-1, 0].float()[None]}     # acc of the transition f -> f+1 = act[f, 0] (the trainer's ego_of)
    return out


def _roll_cells(stack, level, z, act, anchors, kmax, ctx):
    if not hasattr(stack, "spd"):
        return _STATE["orig_roll"](stack, level, z, act, anchors, kmax, ctx)
    cur = _STATE["cur"]
    if cur is None:
        raise RuntimeError("a speed stack rolled out with no speed context (encode_clip was not called for this clip)")
    idx = torch.as_tensor(list(anchors), dtype=torch.long)
    pad = TF.pad(act, (0, 0, 0, kmax))
    pos = idx[:, None] + torch.arange(kmax)[None]
    sp = anchored_speeds(stack.ego_mode, cur["v"], cur["acc"], idx, kmax, int(stack.cum[level - 1]))
    _STATE["n_roll"] += 1
    return stack.rollout(level, z[:, idx], pad[:, pos], context=ctx, speed=sp)


@contextlib.contextmanager
def speed_read(fr_speed_registry):
    """Patch lr_lib's two entry points (encode_clip, roll_cells) for the duration; gather / beside_arm / ia3_arm / slope_arm look them up as module globals."""
    if _STATE["orig_enc"] is not None:
        raise RuntimeError("speed_read is already installed")
    _STATE.update(speed=fr_speed_registry, cur=None, orig_enc=L.encode_clip, orig_roll=L.roll_cells, n_roll=0)
    L.encode_clip, L.roll_cells = _encode_clip, _roll_cells
    try:
        yield _STATE
    finally:
        L.encode_clip, L.roll_cells = _STATE["orig_enc"], _STATE["orig_roll"]
        _STATE.update(speed=None, cur=None, orig_enc=None, orig_roll=None)


def run_dir_of(tag, pulls=None):
    return f"{pulls or L.PULLS}/egowm_{tag}"


def seed_of(name):
    return int(name.rsplit("_s", 1)[1])


def read_arm_speed(name, run_dir, smoke=False, expect_mode=None):
    """The ladder_read ``read_arm`` (same calls, same order) for an arm that may carry a speed stack. ``name`` e.g. E-KIN_s0 / E-KIN_rep_s2 / R35H_s0; ``run_dir`` its banked folder (argv.txt, ckpt_last.pt).
    ``expect_mode``: the mode the run's argv declared (kin / now / None = off); the BUILT stack's rule must read back the same, or the arm is not read."""
    import hashlib
    t0 = time.time()
    free = L.memory_guard()
    model, cfg, argv, ck = L.load_arm(run_dir)
    built_speed = model.wm.built_speed() if model.wm.speed != "off" else {"mode": "off"}
    declared = (argv.get("wm-speed") or ["off"])[0]
    if built_speed["mode"] != declared or (expect_mode or "off") != declared:
        raise L.LadderMismatch(f"speed mode: BUILT {built_speed['mode']}, argv {declared}, expected {expect_mode or 'off'}")
    built = L.built_values(model)
    checked = L.check_built_vs_argv(built, argv)
    fseed = FRAMES_SEED[seed_of(name)]
    fr = L.load_eval_frames(f"{L.SCR}/cache/frames_s{fseed}.npz")
    if L.adapter_sha(model) != fr["adapter_sha"]:
        raise L.LadderMismatch(f"frozen adapter of this run differs from the cached frames' adapter (frames seed {fseed})")
    if int(argv["seed"][0]) != seed_of(name):
        raise L.LadderMismatch(f"argv --seed {argv['seed']} vs the run name {name}")
    vd, n_fill = ES.load()
    reg = attach_speed(fr, vd)
    Y = L.build_label_frames(fr["clips"], fr["T"])
    with speed_read(reg):
        cells, states, cum, plan, built_ctx = L.gather(model, fr)
        W = L.IL.boot_weights(np.arange(len(fr["clips"])))
        worst = L.beside_copy_selfcheck(cells, W)
        if worst != 0.0:
            raise RuntimeError(f"Copy-Last against itself is not exactly 0.0 (max |value| {worst})")
        n_cells_checked = L.check_cells_against_states(cells, states, cum)
        if built_speed["mode"] != "off" and not all(np.isfinite(c["pred"]).all() for c in cells.values()):
            raise RuntimeError("a speed read produced a non-finite predicted latent")
        ia0 = L.ia0_arm(cells, plan, cum, W, "built", smoke)
        ia0_ctx1 = L.ia0_arm(cells, plan, cum, W, "ctx1", smoke)
        beside = L.beside_arm(model, fr, Y, states, cells, cum, plan, W, "built", smoke)
        n_speed_rolls = _STATE["n_roll"]
    if built_speed["mode"] != "off" and n_speed_rolls == 0:
        raise RuntimeError("a speed arm was read and no rollout carried a speed")
    return {"tier": "T0", "arm": name.rsplit("_s", 1)[0], "seed": seed_of(name), "name": name, "run_dir": run_dir, "smoke": bool(smoke),
            "argv_sha256": hashlib.sha256(open(f"{run_dir}/argv.txt", "rb").read()).hexdigest(), "ckpt_step": int(ck.get("step", -1)),
            "built": built, "argv_checked": checked, "adapter_sha256": L.adapter_sha(model), "copy_vs_itself_max_abs": worst, "cells_checked_against_states": n_cells_checked,
            "n_clips": len(fr["clips"]), "n_frames": int(sum(fr["T"])), "n_not_ok_frames": int(sum((~o).sum() for o in fr["ok"])),
            "cum_stride": cum, "ctx_built": built_ctx, "horizon_plan": {str(k): v for k, v in plan.items()},
            "ia0": ia0, "ia0_planning_context1": ia0_ctx1, "passes": L.level_passes(ia0), "beside": beside,
            "bootstrap": {"estimator": "episode-cluster bootstrap over the 141 eval clips", "B": L.B_BOOT, "seed": 0}, "reference_draws": 19 if smoke else L.R_REF,
            "free_gb_at_start": round(free, 2), "wall_s": round(time.time() - t0, 1),
            "speed": {"built": built_speed, "frames_seed": fseed, "speed_source": "egowm/read/eval141_speed.npz (kin3[:,0], native pose track)", "speed_row0_filled": n_fill,
                      "n_speed_rollouts": n_speed_rolls}}


def main(argv):
    """python lr_speed.py <name>:<run_dir>[:<mode>] ...  -> results/arm_<name>.json (resumable; --force; --smoke)."""
    smoke, force = "--smoke" in argv, "--force" in argv
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rc = 0
    for spec in [a for a in argv if not a.startswith("--")]:
        name, run_dir, *m = spec.split("|")
        out = os.path.join(HERE, "results", ("smoke_" if smoke else "") + f"arm_{name}.json")
        if os.path.exists(out) and not force:
            print(f"[{name}] done already, skipped", flush=True)
            continue
        try:
            r = read_arm_speed(name, run_dir, smoke, m[0] if m else None)
            json.dump(r, open(out + ".tmp", "w", encoding="utf-8"))
            os.replace(out + ".tmp", out)
            print(f"[{name}] read in {r['wall_s']} s -> {out}", flush=True)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            open(os.path.join(HERE, "results", f"FAILED_{name}.txt"), "w", encoding="utf-8").write(f"{type(e).__name__}: {str(e)[:2000]}\n")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
