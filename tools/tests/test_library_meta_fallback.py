"""kb_add.arxiv_meta falls back to the arXiv abstract page when the export API gives no title.

MEASURED 2026-09-27: the export API answered HTTP 406 to this client (with or without a User-Agent),
and 17 of 21 banked papers got an EMPTY title while their PDFs downloaded fine. Offline tests only:
the network is monkeypatched."""
import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kb_add_under_test", HERE / "kb_add.py")
kb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kb)

PAGE = ('<meta name="citation_title" content="Efficient DETR:  Improving &amp; Testing" />\n'
        '<meta name="citation_author" content="Yao, Zhuyu" />\n'
        '<meta name="citation_author" content="Ai, Jiangbo" />\n'
        '<meta name="citation_date" content="2021/04/03" />\n'
        '<meta name="citation_abstract" content="The recently proposed ..." />\n')


def test_parse_abs_page_literals():
    m = kb.parse_abs_page(PAGE)
    assert m["title"] == "Efficient DETR: Improving & Testing"      # entity + whitespace
    assert m["authors"] == ["Yao, Zhuyu", "Ai, Jiangbo"]
    assert m["published"] == "2021-04-03"
    assert m["abstract"] == "The recently proposed ..."


def test_parse_abs_page_without_tags_is_empty_not_garbage():
    m = kb.parse_abs_page("<html><title>arXiv Query: search_query=</title></html>")
    assert m["title"] == "" and m["authors"] == []


def test_api_failure_falls_back_to_the_abstract_page(monkeypatch):
    monkeypatch.setattr(kb, "_api_meta", lambda aid: {"title": "", "abstract": "", "authors": [],
                                                      "published": "", "meta_error": "HTTPError: 406"})
    monkeypatch.setattr(kb, "_abs_page_meta", lambda aid: kb.parse_abs_page(PAGE))
    m = kb.arxiv_meta("2104.01318")
    assert m["title"].startswith("Efficient DETR") and m["meta_source"] == "abs_page"


def test_api_title_wins_and_the_page_is_not_fetched(monkeypatch):
    monkeypatch.setattr(kb, "_api_meta", lambda aid: {"title": "From API", "abstract": "", "authors": [],
                                                      "published": "2021-01-01"})
    def boom(aid):
        raise AssertionError("the abstract page must not be fetched when the API has a title")
    monkeypatch.setattr(kb, "_abs_page_meta", boom)
    assert kb.arxiv_meta("x")["title"] == "From API"


def test_RED_both_sources_fail_keeps_the_error_visible(monkeypatch):
    monkeypatch.setattr(kb, "_api_meta", lambda aid: {"title": "", "abstract": "", "authors": [],
                                                      "published": "", "meta_error": "HTTPError: 406"})
    def fail(aid):
        raise OSError("offline")
    monkeypatch.setattr(kb, "_abs_page_meta", fail)
    m = kb.arxiv_meta("x")
    assert m["title"] == ""
    assert "406" in m["meta_error"] and "abs page: OSError" in m["meta_error"]
