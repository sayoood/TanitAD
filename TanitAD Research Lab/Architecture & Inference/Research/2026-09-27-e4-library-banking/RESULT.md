# E4: the box-head literature's 21 arXiv primaries are banked, and kb_add's metadata fetch is repaired

**PI, 2026-09-27 ~09:25:** approved banking the 21 arXiv ids listed in
`../2026-09-27-box-head-literature/RESULT.md` section 8 (12 priority-1, 9 priority-2).

## What was banked (MEASURED)
- All 21 are in `TanitAD Research Lab/Library/library.json` with real PDFs on disk: `%PDF-` header,
  > 50 KB each, 132,234,497 B in total. `tools/kb_add.py --verify` reads 573 entries, 0 orphans,
  0 problems. Every entry is tagged `box-head`, with a one-line note, and cited by the literature RESULT.
- Keys: 2104.01318 Efficient DETR, 2010.04159 Deformable DETR, 2110.06922 DETR3D, 1708.02002 Focal
  loss, 2012.05780 OneNet, 1702.05693 CityPersons, 2103.16237 MonoDLE, 2203.13310 MonoDETR,
  2203.01305 DN-DETR, 2203.03605 DINO, 2303.11926 StreamPETR, 2311.11722 Sparse4D v3, 2211.10581
  Sparse4D, 2104.10956 FCOS3D, 2006.11275 CenterPoint, 2206.07705 LET-3D-AP, 2203.07669 Progressive
  DETR, 1901.05555 Class-Balanced Loss, 1908.09492 CBGS, 2002.09437 Focal-loss calibration, 1405.0312 COCO.

## The defect found while banking (MEASURED) and its fix
- 17 of the 21 were banked with an EMPTY title, as `<id>_untitled.pdf`. Two older entries (1407.7644,
  2609.10464) were already untitled on the tip.
- The cause: `arxiv_meta` calls `http://export.arxiv.org/api/query`, which answered **HTTP 406 Not
  Acceptable** to this client, with or without a User-Agent or Accept header. The failure was
  best-effort-silent. The PDF host (arxiv.org) served the same papers fine.
- The fix, in `tools/kb_add.py`: `arxiv_meta` tries the export API first and falls back to the
  abstract page's `citation_*` meta tags (`parse_abs_page`, pure). The new
  `tools/tests/test_library_meta_fallback.py` has 5 offline tests, including the red arm "both
  sources fail -> the error stays visible".
- The repair re-fetched metadata for all 19 untitled entries: 19 fixed, 0 failed. The 17 new files
  were renamed to titled names. The 2 files already tracked on the tip keep their names (a rename
  would be a deletion) and got their titles in place.
- The existing `test_library.py::test_every_entry_has_the_fields_a_citation_needs` FAILED on the
  untitled entries before the repair and passes after it.
- `test_library_tracking.py::test_every_banked_pdf_is_in_head` fails until this landing lands the PDFs;
  it is the intended "land me" check.
