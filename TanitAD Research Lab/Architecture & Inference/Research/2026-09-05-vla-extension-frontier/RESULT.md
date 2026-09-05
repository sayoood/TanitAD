# VLA extension frontier research — a language backbone for the tactical and strategic layers of REF-C v5

**status: IN PROGRESS — done: skeleton banked / next: architecture facts (§0), survey (§1), design options (§2), grounding (§3), hypotheses (§4), traps (§5), plan (B)**

- **Agent:** TanitAD Research Lab (Architecture & Inference), time-boxed ~4 h, zero GPU, no sub-agents.
- **Branch:** `agent/arch-inf-20260803`. **Date:** 2026-09-05.
- **Ask (PI, verbatim summary):** extend REF-C (+ planned gap-closing extensions) and the 4b architecture with a
  vision-language part that shares REF-C's embedding space, processes nav commands, question queries, images and ego
  data, emits a system-initiated chain of thought (text) explaining the behaviour — distilled from the Alpamayo CoT we
  extracted — grounded in the scene and consistent with the trajectory hypotheses (no hallucinated internal thoughts),
  able to process AND emit strategic and tactical goals, complementing the current strategic/tactical layers, at a
  300–500 ms tact, ≤ 1 B parameters (smaller better), on the Jetson Thor.
- **Companion plan:** `Project Steering/REFCV5_VLA_EXTENSION_PLAN.md` (deliverable B).
- **Evidence classes used:** PUBLISHED-PRIMARY (table/section cited, primary banked in the Library) ·
  PUBLISHED-SECONDARY (abstract/aggregator only — inadmissible for the registry) · MEASURED (ours + artifact path) ·
  ESTIMATED (derivation shown) · HYPOTHESIS.

## 0. What OUR architecture is (read, not re-derived)
_(pending)_

### 0.1 REF-C v4b interfaces the module must read and write
### 0.2 The Alpamayo CoT we hold — n, format, alignment
### 0.3 Thor as the latency target

## 1. Survey — primaries banked
### 1.1 Driving VLM/VLA line
### 1.2 Small-VLM line (≤ 1 B)
### 1.3 CoT-for-control and latency line
### 1.4 Grounding / faithfulness line

## 2. Design options for OUR module (priced)
### 2.a Pretrained small VLM + projector from `pooled`/`z_tac`/`ctx`
### 2.b From-scratch small decoder on our corpus + distilled CoT
### 2.c Hybrid — frozen small LM + LoRA, REF-C trunk as the ONLY vision encoder
### 2.d Comparison table

## 3. The grounding / consistency mechanism — the USP
### 3.1 Consistency losses text ↔ `g_tac` / selected anchor
### 3.2 Verification against the fan
### 3.3 Contradiction detection and grounding tokens (`obstacle.offline`)
### 3.4 EXPLANATION FAITHFULNESS — the fifth metric family, with controls

## 4. Own hypotheses — `H-VLA-*` (validation on the tiny rig / dev box, both outcomes committed)

## 5. What we must NOT do

## 6. Deliverable manifest
_(pending)_
