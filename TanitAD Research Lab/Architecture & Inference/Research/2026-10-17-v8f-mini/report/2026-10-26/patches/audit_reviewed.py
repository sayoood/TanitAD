# Hand review of the numeric tokens that audit_v32.py could not match with the same sign in a source (sections B, C and D).
# Executed by audit_v32.py; every token of B, C and D must be a key here, and every key must still be in B, C or D.
REVIEWED.update({
    "0.7164": "B. Magnitude of the T0-ceiling contrast gt - own ADE6, which the source writes as -0.7164 (`RESULT_V8F_SPEEDREAD.md`:18 and :122). The abstract and 8.1 item 12 say the true cells"
              " 'would cut ADE6 by 0.7164'; the signed forms (-0.7164 / -0.6676) are in 7.24 and in 8.3, row 1x.",
    "0.6676": "B. The seed-1 partner of 0.7164 (`RESULT_V8F_SPEEDREAD.md`:18 and :122, signed -0.6676): the same magnitude wording.",
    "8.3": "C. The carried '(§8.3)' in the status line (v31 line 3); the new text adds none.",
    "7.10": "C. Section reference to MIXO / the SPEEDCAL-KINDX base (v31 section 7.10), not a source number.",
    "8.4": "C. Section reference ('§8.4 row 2'), not a source number.",
    "470": "C. '470-494 h' is v31's own range for the LoRA box-2 projection (v31 section 7.22, and the 470 h/seed in the 8.4 row 2); the RESULT states 494 (LORA Amendment 3). The report"
           " quotes both so that the figures it already carries stay traceable.",
    "7.20": "C. Section reference (DAG against the kinematic floor and the flat control), not a source number.",
    "7.18": "C. Section reference (O-RECOVER), not a source number.",
    "7.24": "D. The new section number (SPEEDREAD).",
    "7.25": "D. The new section number (DAGGER2).",
    "7.26": "D. The new section number (DAG-ORC).",
})

# Tokens of the prose of 7.24 / 7.25 / 7.26 that do not occur in that section's own sources (audit section H).
_REF = "Section reference inside the report (a cross-reference to another section of v32 or v31), not a source number."
SCOPE_OK.update({
    "7.24:7.24": _REF, "7.24:7.22": _REF, "7.24:7.10": _REF, "7.24:8.4": _REF, "7.24:7.26": _REF,
    "7.24:470": "Quoted from the report's own v31 text ('470-494 h' in 7.22, '470 h/seed' in row 2 of 8.4), next to the RESULT's 494 h, so that the two figures the report carries stay traceable.",
    "7.24:11": "Derived: the control table lists C1 to C11, which is 11 controls (19 flags = 8 per-seed controls x 2 seeds + C9, C10, C11), see `RESULT_V8F_SPEEDREAD.md`:3 and :54.",
    "7.24:0.7164": "Magnitude of the gt - own ADE6 contrast, written -0.7164 in the RESULT (:18, :122, :200); the sentence says the cells are 'worth' it.",
    "7.24:0.6676": "Magnitude of the gt - own ADE6 contrast, written -0.6676 in the RESULT (:18, :122, :200).",
    "7.25:7.25": _REF, "7.25:7.22": _REF, "7.25:7.23": _REF, "7.25:7.20": _REF, "7.25:7.18": _REF, "7.25:7.26": _REF,
    "7.26:7.26": _REF, "7.26:7.25": _REF, "7.26:7.18": _REF,
})
