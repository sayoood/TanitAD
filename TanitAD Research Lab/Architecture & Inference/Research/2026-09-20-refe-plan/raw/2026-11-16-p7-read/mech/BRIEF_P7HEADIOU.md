# BRIEF: P7 aux-head IoU [0, 40) m + head-vs-scorer DAC AUC on the 1,058 C3 navtest tokens (dev-box GPU, shared lock). Agent: Sonnet. MM, 2026-11-16

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/headiou/`. Scratch: `F:/tanitad_scratch/p7headiou/`.
- **Read first:** `p7_next/P7_MECH_READ.md` and `p7_next/HANDOFF.md`, from the previous agent.
- `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Read code from the mirror (`C:/Users/Admin/tanitad-push`; `git fetch -q origin agent/arch-inf-20260803`; FETCH_HEAD), because the E: tree lags.

## Why
SPEC_P7 §5, NOT-PROVEN branch. P7 = P6 on PDMS, and the DAC-zero share did not move (+0.09 pp [−0.19, +0.39]). The head IoU is the discriminator:
- **IoU ≥ 0.80:** the head is good, but the scorer never reads it (feed off). The lever is the scorer feed.
- **IoU < 0.80:** the 1 m grid, or the `--w-field` arm (D-AUX-4).

## Instrument
`PK/eval/aux_navtest_forward.py` + `PK/eval/aux_s1_readout.py`: the full planner forward, then the readout.
- **Needed adapter:** the trunk comes from P7's `F:/TanitAD_p7read/run/model_final.pt` (sha256 `2613fc17…`), the head from its `aux_drv` entry. The P7 head is 2 m cells on a 40x32 grid; see P7_MECH_READ.
- **Write the adapter minimally**, in a NEW file in your hold. Do not edit the instrument.
- **Prove the adapter loads the right weights before the real run:**
  - every `aux_drv` tensor is byte-equal to the checkpoint's;
  - a control with the head zeroed must give IoU ~ the empty-prediction value;
  - the forward on 2 tokens must be identical when run twice.

## GPU (binding)
- Take the dev-box shared lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` exactly as the instrument / `stage2_sft.py` `take_lock` do: wait until the GPU is idle (< 3,500 MiB used); never run without the lock; release it on every exit path.
- Other sessions queue on this lock. One run of about 1 h is fine; do not hold it longer than the run.
- The memory note says the shared GPU has 8 GB.
- Check dev-box free commit ≥ 6 GB before starting the forward, and wait in a script while it is below that.
- The run takes ~40–55 min at 3–11 s/token, so launch it detached and wait with a background bash waiter, never in model turns.

## Deliverable
`P7_HEAD_IOU.md` with:
- IoU over [0, 40) m on the 1,058 C3 tokens: per-token mean, plus the log-cluster bootstrap CI (`taniteval/ci.py`), with n;
- the head-vs-scorer DAC AUC, if the readout gives it;
- the SPEC_P7 branch it triggers, against bar 0.80;
- the cheapest next-arm spec, in one paragraph;
- the evidence class, estimator and artifact path for each number.

Also write `HANDOFF.md` (≤ 2k tokens).

## ⛔ Do not
- Commit, land or push.
- Train.
- Touch the pod (C0 trains there) or box 2.
- Edit existing instruments or the frozen PREREG / SPEC files.
- Print or copy secrets.
- Delete outside your scratch (literal paths, logged).
- Exceed ~150k context.

**Report:** at most 300 words, plus a manifest.
