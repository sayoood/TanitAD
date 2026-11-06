# Build note: the TanitAD-4B paper PDF (2026-11-06)

* **Engine.** Tectonic 0.17.0 (XeTeX), the official x86_64-pc-windows-msvc release from GitHub; zip sha256
  `f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`, matching the digest GitHub publishes. Installed at the PI's
  request ("Install tectonic") under `C:/Users/Admin/tools/tectonic-0.17.0/`.
* **Command.** `tectonic --keep-logs --outdir <dir> tanitad_4b_paper.tex`.
* **Result.** 7 pages, 0 overfull boxes, 0 undefined references or citations. The source also compiles under pdfLaTeX (Overleaf):
  no engine-specific package is used.
* **Visual check.** Every page was rendered to PNG with the Windows PDF API and read by the Master Mind.
* **Changes from the uncompiled source in `raw/2026-11-06-paper-and-pitch/`.**
  * Preamble: `amsmath` before `newtx`, `amssymb` dropped (`newtxmath` provides it), font expansion off (XeTeX cannot expand).
  * Four loss-list items had a tab character where `\textbf` belonged (an escape collapse in an earlier scripted edit); restored.
  * Fig. 1: the input boxes sat under the encoder and their arrows pointed backwards; the fallback arrows now all point into the
    fallback brain.
  * Fig. 2: the held-checkpoint label is clear of the S-to-O line.
  * Fig. 3: seed 0 drawn above seed 1, values beside their bars, legend above the axis. The DAgger-arm bar (1.17 / 0.99 m) is
    sourced in `FACTS_PITCH.md` from report v41 section 7.20.
  * Equations 3, 7, 10, 11 and 14 split over two lines; equation 8 unboxed; roman subscripts in the training loss.
  * Run-in headings set as `\paragraph`; references in footnote size.
* **No number changed.**

| file | sha256 |
|---|---|
| `tanitad_4b_paper.tex` | `ad88ed32dd8d337796d28b8618a8626a0208f4d4a98e863da9d484ad013d0674` |
| `tanitad_4b_paper.pdf` | `c3c0eb2b19ac0ef0c31e2b71d9a8cecfa08a86b8391f57f10b8bde2ff6fb2250` |
