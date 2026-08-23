# Colab burst compute — validated pattern (2026-07-09)

**Status: LIVE.** One-time OAuth done by Sayed on tanitad-pod2; credential persists there.
End-to-end validation: `colab run --gpu T4` = fresh T4 VM, torch 2.11 cu128, CUDA matmul,
auto-release — **33 s cold-to-done, $0** (free tier / Sayed's compute units).

## How any agent runs a GPU burst job

The CLI lives on **tanitad-pod2** (`/opt/colabcli`, py3.12 venv, symlinked `colab`). Windows
dev-machine agents drive it over ssh:

```bash
# 1. write your standalone script (self-contained: pip-installs its own deps if needed)
scp my_experiment.py tanitad-pod2:/tmp/
# 2. run it on a fresh GPU VM (T4 free-tier; L4/A100 burn compute units - ask before A100)
ssh tanitad-pod2 'colab run --gpu T4 /tmp/my_experiment.py'
# 3. results: print to stdout (captured) or upload artifacts from within the script
#    (e.g., to the private HF repo Sayood/tanitad-internal via a token passed as an arg)
```

Supported accelerators: `--gpu T4 | L4 | G4 | H100 | A100`, `--tpu v5e1 | v6e1`.
`--keep` + `colab exec -s <name>` for interactive follow-up; `colab sessions` to list,
`colab stop` to clean up.

## Rules (burst-compute etiquette, D-020 §4)

- T4 freely; L4 sparingly; **A100/H100 only with a RESOURCE_LEDGER row** (they consume
  Sayed's compute units fast).
- Scripts must be self-contained (fresh VM every time — no state survives).
- Record hardware, wall-clock, and cost in your research note (CNCE discipline).
- Repo access from the VM: clone the public GitHub repo; NEVER copy Keys.txt to a Colab VM.
