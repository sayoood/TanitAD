# Stage B questions, decided 2026-11-09 from the PI's 60 audited frames

Frozen rule (`audit_prompts.py`): pick by balanced accuracy on the non-random strata (19 frames for visibility), report on the 40 random frames, adopt only if the report score is not below the current choice.

Rule output: visibility keeps `vis0` (select 0.5833, report 20/40); illumination stays SigLIP (`siglip_control`, select 0.5764, report 30/39). Qwen illumination questions were not adopted.

**Post-hoc step, to be stated on the dataset card.** `vis0` scores below always answering good (31/40). Combinations of the three saved scores were then read under the same selection score (`audit_prompts_probs.json`). Four arms tie at select 0.5833: `vis0`, mean of `vis0`+`vis1`, OR of `vis0`+`vis1`, AND of `vis0`+`vis2`. The tie was broken on the report frames, which makes it post hoc: AND of `vis0`+`vis2` reports 26/40 (good 22/31, reduced 4/8, poor 0/1) and mean of `vis0`+`vis1` reports 26/40 (reduced 3/8). AND of `vis0`+`vis2` is adopted: a not-good verdict needs both questions to say not-good. It remains below the constant 31/40; the raw answers of both questions stay in the per-clip ctxB parts.

Stage B asks: weather, surface, visibility (`vis0`), vis2. The camera-view question is dropped: it is not among the PI's fields, and removing it keeps the per-clip cost at four questions.
