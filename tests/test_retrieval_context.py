"""Chunk context headers written by ingest.py."""

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



def test_committed_search_index_is_complete_and_slim():
    """vectorstore/ is committed, so keep it the size ingest.py produces."""
    import sqlite3
    from pathlib import Path
    import chromadb
    store = Path(__file__).resolve().parent.parent / "vectorstore"
    col = chromadb.PersistentClient(path=str(store)).get_collection("dutyline")
    metas = col.get(include=["metadatas"])["metadatas"]
    assert len(metas) > 2000
    assert all(m.get("context") for m in metas), "chunks lack context headers: re-run ingest.py"
    db = sqlite3.connect(store / "chroma.sqlite3")
    assert not db.execute("select 1 from sqlite_master where name like '%fulltext%'").fetchone(), \
        "Chroma's unused text-search index is back (+30 MB): re-run ingest.py"
    db.close()
