# Colab Pro as TanitAD compute — operating guide (2026-09-27)

**Account:** `fambouzouraa@gmail.com` holds the PI's Google AI plan (PI, 2026-09-27: *"fambouzouraa is the right
account, use it carefully and effectively to get the max benefit"*). The colab-cli OAuth token on the dev box
(`~/.config/colab-cli/token.json`, first granted 2026-08-19) belongs to it and refreshes silently. ⛔ Never print
it, never commit it, never re-authenticate without the PI — the paste-code sign-in (COLAB_CLI_MCP.md §4) is theirs.

## 1. Drive it

```powershell
& D:\Projects\TanitAD\colab\colab.ps1 new  -s <name> --gpu G4          # or L4
& D:\Projects\TanitAD\colab\colab.ps1 exec -s <name> --timeout 3600 -f D:\path\job.py
& D:\Projects\TanitAD\colab\colab.ps1 stop -s <name>
C:\Users\Admin\venvs\colab\Scripts\python.exe D:\Projects\TanitAD\colab\ccu_info.py   # needs the two env vars below
```

* `colab.ps1` sets `PYTHONUTF8=1` and `PYTHONPATH=colab\win_shims` (the `termios` import shim). Both are needed by the
  keep-alive daemon too, which `colab new` spawns as a fresh `python -m colab_cli.cli` — an in-process stub would
  not reach it, and the daemon would die silently (its output goes to DEVNULL).
* ⛔ **PowerShell only.** Git-Bash/MSYS rewrites `/content` into `C:/Program Files/Git/content` (RUNNER.md §9, trap 3).
* ⛔ **Read `RUNNER.md` §9 before driving** — five MEASURED traps; `colab run` RELEASES the VM; a 3600 s proxy-token
  expiry reads as "session lost" while the VM is alive (§9d).
* Parallel agents: each uses its own state file, `colab.ps1 --config <path> ...`, and its own session names.

## 2. What a compute unit buys — MEASURED 2026-09-27 (`gpu_survey.ps1` → `raw/2026-09-27-gpu-survey/`)

Price = ccu-info's `consumptionRateHourly` read while the VM was assigned (Colab's own meter). Throughput = an
8192² matmul, 20 iterations after warm-up (`gpu_smoke.py`). One sample each, one VM each: a scale, not a benchmark.

| `--gpu` | hardware | CU/h | bf16 TFLOPS | fp16 | fp32 | **bf16 TFLOPS per CU/h** | CPUs | RAM GB | disk GB | HF pull MB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| **G4** | RTX PRO 6000 Blackwell Server, **98 GB** | **8.9** | **390.9** | 298.5 | 73.7 | **43.9** | 48 | 189.9 | 201.6 | 108.7 |
| **L4** | L4, 23 GB | **1.54** | 60.1 | 56.7 | 11.0 | **39.0** | 12 | 56.9 | 202.2 | 109.9 |
| T4 | T4, 15 GB | 1.07 | 2.2 (no bf16) | 22.7 | 3.8 | 2.1 | 2 | 13.6 | 202.2 | 138.2 |
| A100 | — | — | — | — | — | — | — | — | — | — |

A100 is **eligible but was not allocatable**: the assign POST answered **503 Service Unavailable** after 60 s —
capacity, not entitlement (`colab.log` 2026-09-27 17:00:15). H100 is not eligible on this plan. Stack on every VM:
driver 580.82.07 (CUDA 13.0), Python 3.13.15, torch 2.11.0+cu128, cuDNN conv2d verified on each. `colab new` took
~15 s to a READY session. Balance before and after the survey: 200 (the survey burned ≈ 0.14 CU by rate × time).

## 3. Spending rules (to get the most out of 200 CU)

1. **G4 for anything GPU-bound whose wall-clock matters** — it is 6.5× an L4 and slightly *cheaper per FLOP*.
   200 CU ≈ **22 G4-hours**. **L4 for long, light, or CPU/IO-bound jobs** (≈ 130 L4-hours). **Never T4** —
   2 CPUs, no bf16, a twentieth of G4's value.
2. **A VM that is not computing is burning.** Every job ends in `stop` inside a `finally`, and is then checked on
   the SERVER: `ccu_info.py` must read `assignments: []` and `consumptionRateHourly: 0`. A stop's exit code is not
   the evidence (`gpu_survey.ps1` is the pattern).
3. **A job carries its wall-clock cap and persists after every unit of work** (§9 trap 4), so a timeout or a
   reclaimed VM costs one unit, not the run.
4. **Data never ships from the dev box.** Its uplink is ~1.2 MB/s (memory: `devbox-uplink-1p2-mbps`); a VM pulls from
   HF at **~110 MB/s**. Stage inputs on HF (public or the PI's private repos) or pull them from their public source.
5. **Price before launch:** CU = rate × hours, written into the job's brief; the orchestrator owns the budget and
   no agent spends it without a go.

## 4. What it can take now, and what it cannot yet

* **Now (public or HF-resident inputs):** Research-Lab GPU experiments and paper reverse-engineering; anything whose
  data is public (NAVSIM/OpenScene is published on HF, so a navtest scoring can pull its own inputs — the harness is
  CPU-heavy and G4 has 48 cores).
* **Data route (PI, 2026-09-27: "follow your recommendation regarding data transfer"):** dev box or pod → the
  PRIVATE HF dataset **`Sayood/tanitad-colab-relay`** → the VM (~110 MB/s pull). `hf_relay_push.py` refuses a push
  unless a quota readout ≤ 24 h old shows private used + push < 1 TB (HF bills above it; deletions free nothing until
  a super-squash), and writes the MANIFEST last; `hf_relay_pull.py` pulls only MANIFEST-listed files and fails on any
  md5 mismatch. MEASURED first push: `refe/snapshots/snap_epoch016.pt`, 76,052,289 B, md5 `eacd8e8f…` (= the eval
  pipeline's fetch), 77.4 s from the dev box (0.94 MB/s); private storage 184.599 GB → 184.675 GB
  (`raw/2026-09-27-hf-relay/`).
* **The pod leg is deferred to its first consumer.** The A40 pod has no HF login and no `huggingface_hub`
  (MEASURED 2026-09-27); nothing queued needs pod-only data (snapshots are on the dev box; agent C: no proxy-set
  staging). Arming it needs a package install on the pod and a token there — both the PI's.
* ⛔ **The token on a VM is the PI's to provision.** Moving the Keys.txt token to a VM was REFUSED by the permission
  system as data exfiltration (2026-09-27), even with the PI's instruction — so no agent does it, by any route. The
  PI's route: a Colab Secret `HF_TOKEN` on the plan account (Colab UI → key icon). ⛔ **MEASURED 2026-09-27: a
  CLI-provisioned (headless) session CANNOT read it** — `secret_probe.py` asked for it the way `userdata.get` does
  (`GetSecret` over the kernel's message channel) and got **no reply in 30.0 s**: the notebook FRONTEND serves
  secrets, and a CLI session has none (`userdata.get` itself would wait forever). The PI's secret exists and is fine;
  it is readable only from a browser-attached notebook (an MCP-paired tab, or `colab url -s <name> --open`).
  (`raw/2026-09-27-hf-relay/relay-proof.*`; CPU VM 0.08–0.16 CU/h, the proof cost ≈ 0.003 CU.)
* ⭐ **HEADLESS JOBS GET PRIVATE FILES BY SIGNED LINKS (PI, 2026-09-27: "Signed links") — MEASURED WORKING.** The
  dev box uses the token locally to ask HF's resolve endpoint for each file WITHOUT following the redirect; the
  redirect target is HF's own signed CDN link, which opens that one file for **60 min**. `hf_signed_links.py` writes
  the links + the MANIFEST's bytes/md5 into a job file OUTSIDE the repo (it refuses a repo path), `colab upload`
  carries it to `/content/relay_job.json`, the local copy is deleted, and `signed_pull.py` downloads with **no token
  on the VM**, verifies bytes + md5, and deletes the job file. Proof (`signed_proof.ps1`, CPU VM): `snap_epoch016.pt`
  76,052,289 B, md5 `eacd8e8f…` = the eval pipeline's fetch, 7.3 s at 10.4 MB/s from `us.aws.cdn.hf.co`; VM stopped,
  server idle; ≈ 0.001 CU. No link or token appears in any log (grep-verified). ⇒ **mint links right before the job
  pulls** (60-min validity), never for a job that starts later.
* ⭐ **REFe navtest evals on Colab — P0 DONE (2026-09-27, `navtest/RESULT_P0.md`, 0.41 CU): the ROUTE IS HYBRID.**
  The GPU seam runs on a Colab L4 and passes its pre-registered gates on snapshot 016 (G-S: 0 pick flips, poses ≤ 15 µm;
  G-P: PDMS 76.5243, Δ 0.0000; G-REP: sharded = single). The Linux NAVSIM harness FAILS its EXACT gates by 1e-13 to 1e-9
  (EP only): Windows vs Linux libm/BLAS/GEOS round differently and no setting fixes it. ⇒ the seam npz (65 KB) comes back
  and the dev box's OWN harness scores it: PDMS 76.5243, every discrete term equal. No new tolerance is adopted
  (orchestrator, 2026-09-27); an all-Colab route would need the PI to accept one. Measured: L4 seam ~1.2 s/token
  (GPU-bound; sharding one L4 buys nothing), S3 DB range fetch 90 MB/s, ~0.17 CU per snapshot. ⚠ The first L4 session's
  auto-stop failed (its cleanup deleted the file the stop needed) and was caught by the server-side check — fixed in
  `navtest/p0_session.ps1`, NOT yet proven live. P1 (wiring the per-snapshot driver) is open.
* **Colab MCP server: REGISTERED** at user scope (2026-09-27, bundled Claude Code 2.1.281:
  `claude mcp add colab --scope user -- C:\Users\Admin\venvs\colab\Scripts\colab-mcp.exe`; health check Connected).
  New sessions see one tool, `open_colab_browser_connection`; pairing needs the PI signed in to Colab in the default
  browser (MCP_SETUP.md).

## 5. Artifacts

| file | what |
|---|---|
| `colab.ps1` | the wrapper (env + shim) |
| `ccu_info.py` | read-only balance / burn / assignments / eligibility (0.6.0 has no `usage`) |
| `gpu_smoke.py` | first-contact check of a VM: GPU, real CUDA, matmul TFLOPS, HF pull |
| `gpu_survey.ps1` | provision → price → smoke → stop → server-side zero-assignment assert, per GPU |
| `raw/2026-09-27-gpu-survey/` | the survey's per-GPU JSON and CLI logs |
| `hf_relay_push.py` / `hf_relay_pull.py` | the private-HF relay: guarded push (quota arithmetic, MANIFEST last) / VM-side md5-verified pull |
| `raw/2026-09-27-hf-relay/` | the fresh quota readout (private 184.599 GB of 1 TB) and the first push's log |
| `USAGE_LOG.md` | one row per metered Colab job |
| `win_shims/termios.py` | the import shim (2026-08-16) |

History: `COLAB_CLI_MCP.md` (the 2026-08-16 evaluation, free-tier era) and `RUNNER.md` (the S2 label lab + §9 traps).
