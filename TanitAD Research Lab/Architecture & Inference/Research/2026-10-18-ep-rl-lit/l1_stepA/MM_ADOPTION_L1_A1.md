# L1 Amendment 1: ADOPTED by the Master Mind (2026-10-18, before any Step A outcome exists)

- **Adopted file:** `PREREG_L1_AMENDMENT_1_PROPOSED.md`, **sha256 `45f5c7917feef407b3ce18951e35f7abdcef5b7a9fb65e41e9cebd3d50fd72ff`** (LF bytes, as landed). The file is landed unchanged, so its title still reads "PROPOSED". This record is the adoption.
- **Parent:** `../PREREG_L1_SEARCH_ADMISSION.md`, sha256 `a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801`. It is not edited.
- **Adopted as written:** A1.1-A1.4 (the G0 ruling: G0 passes on the PDM-label manifest M-PDM; the M-SCENE overlap is a DECLARED EXPOSURE that limits what O1 licenses) and C1-C9.
- **A1.5** stays an option. It becomes BACKLOG `L1-UNSEEN-STRATUM`, not part of Step A.

**Why adopt rather than redesign.**
- No navtrain log is disjoint from the final-run bank: 1,191 of 1,192, MEASURED by the agent's streamed scan. I confirmed independently that this bank is REFe's main-run `--grow` training bank (`D-REFE-GROW-1`).
- A literal G0 is therefore unsatisfiable on navtrain. The only honest choices are to declare the exposure or to build a scene-disjoint stratum (A1.5, a multi-hour data job).
- **O2, the outcome that matters most for ranking levers, is robust to the exposure** (A1.3). O1 licenses Step B only, stamped SEEN-SCENE, and the navtest confirm stays the first unseen read.

**On C7 (outcome precedence).**
- The parent wrote O2 by point estimates and O3 by intervals, so the two overlap.
- `interval_aware` is fixed now, before any number exists. The literal reading is printed beside it, and every verdict line states both whenever they differ.

**Next (BACKLOG `L1-STEPA-RUN`).** One GPU forward, `make_stepA_bundle.py`, 0.25-0.4 GPU-h (ESTIMATED), run under the dev-box lock via `launch_stepA.py`. The rest is CPU, about 1 h. Order: after S0, and after R1 / EP-R1 if they claim the GPU first.
- Its first run is also the first test of `make_stepA_bundle.py` and `label_perturbations.py`; both are untested on data.
- G2b (the bundle forward reproduces the banked fan bit for bit) is the gate that catches a wrong driver before any A-quantity is read.
