"""Chunk context headers (ingest.py) and lay-term query expansion (tools.py)."""

from agents.tools import expand_lay_terms
from ingest import chapter_titles, context_headers

AR_CHUNKS = [
    "Chapter 2\nPolicy for Leave, page 3\nChapter 3\nSpecial Leave Accrual, page 8",   # table of contents
    "Chapter 2\nPolicy for Leave\n2–1. Military requirement for leave\na. Leave is ...",
    "(c) Commanders will announce block leave ... Soldiers may carry forward up to 60 days",
    "3–5. One-time leave sell back\na. Soldiers may sell back ...",
]


def test_chapter_titles_skip_toc_page_numbers():
    assert chapter_titles(AR_CHUNKS) == {2: "Policy for Leave", 3: "Special Leave Accrual"}


def test_headers_name_document_chapter_and_headings():
    h = context_headers("ar_600_8_10.pdf", AR_CHUNKS)
    assert h[1].startswith("AR 600-8-10 Leaves and Passes") and "Chapter 2: Policy for Leave" in h[1]
    assert "Military requirement for leave" in h[1]
    # a chunk with no heading of its own inherits the chapter it continues
    assert "Chapter 2: Policy for Leave" in h[2]
    # paragraph number 3–5 puts the chunk in chapter 3
    assert "Chapter 3: Special Leave Accrual" in h[3] and "Chapter 2" not in h[3]
    assert "One-time leave sell back" in h[3]


def test_lay_terms_add_regulation_vocabulary():
    assert "advance leave" in expand_lay_terms("Can I take leave before I've earned it?")
    assert "sell back" in expand_lay_terms("Can I get paid for leave I lose?")
    assert expand_lay_terms("per diem in Columbus GA") == "per diem in Columbus GA"
