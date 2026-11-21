# BRIEF: make the P5-C read (C1 vs C0) a one-command, detached dev-box chain, proven by a tiny-subset dry run. Agent: Sonnet. MM, 2026-11-17

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/`. **Scratch:** `F:/tanitad_scratch/readchain/`. ⛔ Nothing on C:.
- **Code:** read it from the mirror `C:/Users/Admin/tanitad-push` (fetch `agent/arch-inf-20260803`, pin the sha). `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## Why
- **Timing:**
  - C0 finishes about 2026-11-17 23:30Z and is pushed to HF by a pod chain: `Sayood/tanitad-refe-p5c`, prefix `p5c/c0`, with `model_final_sha256_C0.txt` on the pod.
  - C1 then trains about 29 h and is pushed the same way (prefix `p5c/c1`).
- **The read is pre-registered** in `PK/raw/2026-11-11-p5c-readiness/PREREG_P5C.md` (sections 5, 6, 8, 11) plus Amendments A1-A3.3.
- **The tools are landed:** `PK/eval/p5c/` (`p5c_read_prep.py`, `p5c_seal.py`, `p5c_read.py`, `p5c_render.py`, `p5c_families.py`, `p5c_tactical.py`, `p5c_determinism.py`), with tests.
- **New since the freeze:** the PI un-paused the STRATEGIC family. It is REPORTED beside the read, not gating: `PK/eval/instruments/strategic_family.py` and `PK/raw/2026-11-16-p7-read/strategic/INTEGRATION.md`.
- **The goal:** the moment C1's files are on HF, one detached command runs the whole read with no model turns.

## Deliverables
1. **`RUNBOOK_P5C_READ.md`.** The exact ordered command sequence, from the tools' own docstrings and tests (never invented):
   - download both arms from HF and verify their sha256 against the pod's `model_final_sha256_<arm>.txt` (also pushed);
   - seal both in ONE manifest BEFORE any read (`p5c_seal.py`, PREREG section 11);
   - the navtest evaluation of each arm;
   - `p5c_read`, families, tactical, determinism, render;
   - `strategic_family.py read --arm C1=<seam> --base C0=<seam>`;
   - the verdict line, including whether C0r is triggered (C1 - C0 beyond B = 0.7031, as PREREG section 6 defines it).
   - For each step state: inputs, outputs, dev-box GPU or CPU, and ESTIMATED wall time. Cite the P7 read (`E:/Projects/TanitAD-artifacts/p7_full_read/CHAIN.log`) for timings.
2. **`run_p5c_read.cmd` + `p5c_read_chain.sh`.** A resumable chain with markers per stage, `<stage>.DONE` / `chain.FAILED`, launched via WMI with CreateFlags `0x08000200`. GPU stages take the dev-box shared lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` exactly as `stage2_sft.py` / the P7 read chain do. It refuses to start unless both arms' sha files exist.
3. **A dry run, proving the chain end to end:**
   - **Input:** P7's `model_final.pt` (`F:/TanitAD_p7read/run/model_final.pt`, sha256 `2613fc17…`) stands in for BOTH arms, on a tiny navtest subset (e.g. 64 tokens, if the tools allow a subset; if they do not, say so and dry-run every CPU stage on existing seams).
   - **Identity control:** C1 = C0 = the same checkpoint must read delta exactly 0, with CI [0, 0], and strategic self-vs-self exactly 0.
   - **Seal check:** the seal must refuse when a sha differs (a deliberate-regression arm).
   - Bank the dry-run outputs in the hold. Never present them as a P5-C result.
4. **`HANDOFF.md`** (≤ 2k tokens): what is ready, the one command to start, the total ESTIMATED dev-box GPU hours, and the open questions.

**Dev-box rules:**
- CPU unless the dry run needs a GPU stage; any GPU use takes the shared lock, and is released on every exit path.
- `CUDA_VISIBLE_DEVICES=-1` for CPU stages.
- Free-commit guard ≥ 6 GB (parse as an integer; the locale prints a decimal comma).
- Serial, with TMP in scratch.

## ⛔ Do not
- Use the pod (C0 trains there) or box 2.
- Download C0 or C1 (they are not final).
- Edit landed or pinned tools, or frozen PREREG files. If a tool needs a change, write it as a patch file in the hold and say why.
- Commit, land or push. Train.
- Print or copy secrets.
- Delete outside your scratch; delete your scratch at the end, logged.
- **Exceed 150k context. HARD:** if you near it, write `HANDOFF.md` and stop. Never wait on a long job while alive: launch it detached, write the handoff, end.

**Report:** at most 300 words, plus a manifest.
