"""Periodic fp16 weights-only snapshots of the LIVE v6F run — STEP-STAMPED.

⛔ WHY THIS EXISTS. `train_v6_staged` writes ONE rolling `ckpt.pt`. Every save
destroys the previous state, so the run's history is unrecoverable at any price:
"did the world model learn X between step A and B?" cannot be asked after the
fact. The 2026-08-16 slot probe got a two-point trajectory ONLY by accident — a
stale `weights_fp16.pt` left on HF happened to be step 2000.

⇒ 0.67 GB per snapshot, ~10 snapshots for the rest of a 30k run, ~7 GB total,
against 372 GB free. The asymmetry is absolute: the disk is recoverable, the
history is not.

⚠️ THOR IS TRAINING. This is deliberately gentle and CPU-only:
  * `mmap=True` — lazy, page-cache backed, reclaimable; resident cost is the
    ~0.67 GB OUTPUT, not the 3.53 GB input;
  * one tensor cast at a time, never the whole state_dict in fp32;
  * no CUDA context is ever created;
  * `nice`d by the launcher, and it sleeps between polls.
Measured safe once already: the trainer advanced 9000 -> 9250 across a snapshot.

⚠️ TORN READS ARE EXPECTED, NOT ERRORS. The trainer may be mid-write when we
poll. Any failure is caught, logged, and RETRIED next cycle — never fatal, and
never a partial file left behind (write to .tmp, fsync, atomic rename).
"""
import json, os, time, traceback

SRC = "/home/nvidia/experiments/v6F-SW-30k/ckpt.pt"
CFG = "/home/nvidia/experiments/v6F-SW-30k/config.json"
OUT = "/home/nvidia/ckpt_snaps"
EVERY = 2000          # steps between snapshots
POLL_S = 600          # 10 min; ~22 steps at 26.6 s/step, so we never miss a bin
LOG = "/home/nvidia/logs/thor_snap_periodic.log"


def say(msg):
    line = "[snap] %s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def current_step():
    import torch
    ck = torch.load(SRC, map_location="cpu", weights_only=False, mmap=True)
    return int(ck.get("step", -1)), ck


def snapshot(step, ck):
    import torch
    dst = os.path.join(OUT, "v6F_sw_step%06d.fp16.pt" % step)
    if os.path.exists(dst):
        return None
    cfg = ck.get("config") or json.load(open(CFG))
    sd = ck["stack"]
    out = {}
    for k in list(sd.keys()):
        v = sd[k]
        out[k] = (v.detach().to(torch.float16).clone()
                  if v.is_floating_point() else v.detach().clone())
    tmp = dst + ".tmp"
    torch.save({"model": out, "_meta": {"step": step, "config": cfg},
                "_fp16_weights_only": True}, tmp)
    with open(tmp, "rb") as f:      # durability before the rename
        os.fsync(f.fileno())
    os.replace(tmp, dst)            # atomic: a reader never sees a partial file
    return dst, os.path.getsize(dst)


def main():
    os.makedirs(OUT, exist_ok=True)
    say("start every=%d poll=%ds out=%s" % (EVERY, POLL_S, OUT))
    done = set()
    for fn in os.listdir(OUT):
        if fn.startswith("v6F_sw_step") and fn.endswith(".fp16.pt"):
            try:
                done.add(int(fn[len("v6F_sw_step"):-len(".fp16.pt")]) // EVERY)
            except ValueError:
                pass
    while True:
        try:
            step, ck = current_step()
            b = step // EVERY
            if step > 0 and b not in done:
                r = snapshot(step, ck)
                done.add(b)
                if r:
                    say("SNAP step=%d bytes=%d" % (step, r[1]))
            del ck
            if step >= 30000:
                say("reached 30000 — exiting")
                return
        except Exception as e:                                  # noqa: BLE001
            # a torn read while the trainer saves is NORMAL; retry next cycle
            say("retry after %r" % (e,))
            say(traceback.format_exc().splitlines()[-1])
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
