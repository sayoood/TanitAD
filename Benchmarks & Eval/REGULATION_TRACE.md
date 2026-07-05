# UN ADS Regulation (WP.29, June 2026) — Requirements-to-Evidence Trace

> Source: `Ressources/ECE-TRANS-WP.29-2026-139e.pdf` (UNR + GTR on ADS, adopted June 2026).
> Maintained by the Benchmarks & Eval agent. Each requirement maps to a TanitAD design element and an
> evidence artifact (gate result, log format, report generator).

| Req. area | Regulation asks | TanitAD design element | Evidence artifact | Status |
|---|---|---|---|---|
| Safety management system | audited lifecycle safety governance | continuation protocol + decision log + gate ladder discipline | repo history, DECISIONS.md | seeded |
| Credible testing / safety case | structured evidence of no unreasonable risk | instrument doctrine I1–I4 + falsifiable D-gates | CI + experiment records | seeded |
| DDT fallback / MRM | minimal-risk manoeuvre capability | Brain 4: FallbackMonitor + deterministic MRC hook | `stack/tanitad/models/fourbrain.py` + D8 gate | implemented (sim form) |
| In-service monitoring & reporting (ISMR) | continuous performance monitoring, incident detection + reports | H11 layered monitors (imagination error, checker, Mahalanobis); H12 report generation (Phase 2) | monitor logs; Phase 2 report generator | monitor #1 implemented |
| DSSAD data recording | safety-relevant event recording | latent event log (z, Δz, alarms) write-on-surprise | H10 memory interface (Phase 0/1) | designed |
| ODD monitoring | detect ODD exit | strategic-layer OOD (Mahalanobis/KL drift) | D8 harness | designed |

**Open analysis task (Benchmarks & Eval agent):** full requirement extraction from the PDF into this
table with paragraph references — currently seeded from summary-level knowledge (Deep Think 1 +
UNECE press material).
