"""
Ingestion pipeline: downloads PDFs, parses, chunks semantically,
embeds with a local model, and stores in ChromaDB.

Run once to build the vector store:
    python ingest.py

Run with --verify to test retrieval after ingestion:
    python ingest.py --verify
"""

import gc
import os
import re
import shutil
import sqlite3
import sys
import logging
import warnings
import requests
import chromadb
from chromadb.config import Settings
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Suppress ChromaDB telemetry noise and duplicate-ID warnings
os.environ["ANONYMIZED_TELEMETRY"] = "false"
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*existing embedding ID.*")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR      = Path("data")
VECTORSTORE   = Path("vectorstore")
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"   # small, fast, offline after first run
COLLECTION    = "dutyline"

# These limits count WORDS (see estimate_tokens), not model tokens. Text here
# runs ~1.67 tokens per word, so most passages exceed bge-small's 512-token
# window and the embedding only sees their first part (keyword search and
# the LLM still see all of it). Re-chunking to 400 real tokens (4,960
# passages) was tested in Nov 2026 and scored WORSE on the gold queries:
# MRR 0.66 -> 0.55, Recall@4 0.75 -> 0.71. Don't "fix" this without re-running
# python -m evals.run --only retrieval on both versions.
CHUNK_MIN_TOKENS = 150
CHUNK_MAX_TOKENS = 600

SOURCES = [
    # Travel
    {
        "url":         "https://media.defense.gov/2022/jan/04/2002917147/-1/-1/0/jtr.pdf",
        "local_path":  DATA_DIR / "jtr" / "jtr.pdf",
        "domain":      "travel",
        "branch":      "defense_wide",
        "doc_type":    "regulation",
        "source_file": "jtr.pdf",
    },
    # Leave — Army
    {
        "url":         "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN32388-AR_600-8-10-001-WEB-1.pdf",
        "local_path":  DATA_DIR / "army_regs" / "ar_600_8_10.pdf",
        "domain":      "leave",
        "branch":      "army",
        "doc_type":    "regulation",
        "source_file": "ar_600_8_10.pdf",
    },
    # Leave — Navy
    {
        "url":         "https://www.mynavyhr.navy.mil/Portals/55/Reference/MILPERSMAN/1000/1050Leave.pdf",
        "local_path":  DATA_DIR / "navy_regs" / "milpersman_1050.pdf",
        "domain":      "leave",
        "branch":      "navy",
        "doc_type":    "regulation",
        "source_file": "milpersman_1050.pdf",
    },
    # Leave — Air Force
    {
        "url":         "https://static.e-publishing.af.mil/production/1/af_a1/publication/afi36-3003/afi36-3003.pdf",
        "local_path":  DATA_DIR / "af_regs" / "afi_36_3003.pdf",
        "domain":      "leave",
        "branch":      "airforce",
        "doc_type":    "regulation",
        "source_file": "afi_36_3003.pdf",
    },
    # Army evaluations (AR 623-3) are not in the library: the PDF was never
    # obtained, and the agent says so when asked.
    # Eval — Navy
    {
        "url":         "https://www.mynavyhr.navy.mil/Portals/55/Reference/instructions/BUPERS/BUPERSINST_1610.10F.pdf",
        "local_path":  DATA_DIR / "navy_regs" / "bupersinst_1610_10f.pdf",
        "domain":      "eval",
        "branch":      "navy",
        "doc_type":    "regulation",
        "source_file": "bupersinst_1610_10f.pdf",
    },
    # Eval — Air Force
    {
        "url":         "https://static.e-publishing.af.mil/production/1/af_a1/publication/afi36-2406/afi36-2406.pdf",
        "local_path":  DATA_DIR / "af_regs" / "afi_36_2406.pdf",
        "domain":      "eval",
        "branch":      "airforce",
        "doc_type":    "regulation",
        "source_file": "afi_36_2406.pdf",
    },
    # Pay — DoD wide
    {
        "url":         "https://comptroller.defense.gov/Portals/45/documents/fmr/Volume_07a.pdf",
        "local_path":  DATA_DIR / "dod_regs" / "dod_fmr_vol7a.pdf",
        "domain":      "regs",
        "branch":      "defense_wide",
        "doc_type":    "regulation",
        "source_file": "dod_fmr_vol7a.pdf",
    },
    # Marines — evals
    {
        "url":         "https://www.marines.mil/portals/1/Publications/MCO%201610.7.pdf",
        "local_path":  DATA_DIR / "marine_regs" / "mco_1610_7.pdf",
        "domain":      "eval",
        "branch":      "marines",
        "doc_type":    "regulation",
        "source_file": "mco_1610_7.pdf",
    },
]

# ---------------------------------------------------------------------------
# Step 1: Download PDFs
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [ok] {dest.name} already present ({dest.stat().st_size // 1024} KB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [missing] {dest.name}")
    print(f"            Download manually from: {url}")
    print(f"            Save to: {dest}")
    return False

# ---------------------------------------------------------------------------
# Step 2: Parse PDF -> raw text with page numbers
# ---------------------------------------------------------------------------

def parse_pdf(path: Path) -> list[dict]:
    """Returns list of {text, page} dicts, one per page."""
    import pdfplumber
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({"text": text.strip(), "page": i + 1})
    except Exception as e:
        print(f"  [error] parsing {path.name}: {e}")
    return pages

# ---------------------------------------------------------------------------
# Step 3: Semantic chunking
# ---------------------------------------------------------------------------

# Patterns that mark the start of a new section in military docs
SECTION_PATTERNS = [
    r"^\d{6}\.",            # JTR: 020101.  020102.
    r"^\d+-\d+\.",          # Army Regs: 2-1.  3-12.
    r"^[A-Z]\d+\.",         # AFI: A2.1.
    r"^Chapter\s+\d+",      # Chapter headings
    r"^CHAPTER\s+\d+",
    r"^SECTION\s+[IVXLC]+", # SECTION IV
    r"^[A-Z]{2,}\s+\d+",    # AR 600
]

SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.MULTILINE)


def estimate_tokens(text: str) -> int:
    return len(text.split())


def split_into_sections(pages: list[dict]) -> list[dict]:
    """
    Joins all pages, then splits at section header boundaries.
    Returns list of {text, section_title, page_start}.
    """
    # Join pages into one string, tracking page boundaries
    full_text = ""
    page_offsets = []
    for p in pages:
        page_offsets.append((len(full_text), p["page"]))
        full_text += p["text"] + "\n\n"

    # Find all section boundaries
    boundaries = [m.start() for m in SECTION_RE.finditer(full_text)]
    boundaries.append(len(full_text))  # sentinel

    if not boundaries[:-1]:
        # No section headers found — fall back to page-level chunks
        return [{"text": p["text"], "section_title": "", "page_start": p["page"]} for p in pages]

    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end   = boundaries[i + 1]
        chunk_text = full_text[start:end].strip()
        if not chunk_text:
            continue

        # Extract section title (first line)
        first_line = chunk_text.split("\n")[0].strip()

        # Find which page this section starts on
        page_num = pages[0]["page"]
        for offset, pg in page_offsets:
            if offset <= start:
                page_num = pg

        sections.append({
            "text":          chunk_text,
            "section_title": first_line[:120],
            "page_start":    page_num,
        })

    return sections


def chunk_sections(sections: list[dict]) -> list[dict]:
    """
    Enforces CHUNK_MIN and CHUNK_MAX token limits.
    Merges tiny sections into the next. Splits huge sections by sentence.
    """
    chunks = []
    buffer = None

    for sec in sections:
        tokens = estimate_tokens(sec["text"])

        # Too small — merge into buffer
        if tokens < CHUNK_MIN_TOKENS:
            if buffer is None:
                buffer = sec.copy()
            else:
                buffer["text"] += "\n\n" + sec["text"]
            continue

        # Flush buffer first
        if buffer is not None:
            buffer["text"] += "\n\n" + sec["text"]
            sec = buffer
            buffer = None
            tokens = estimate_tokens(sec["text"])

        # Too large — split at sentence boundaries
        if tokens > CHUNK_MAX_TOKENS:
            sentences = re.split(r"(?<=[.!?])\s+", sec["text"])
            current = ""
            for sent in sentences:
                if estimate_tokens(current + sent) > CHUNK_MAX_TOKENS and current:
                    chunks.append({**sec, "text": current.strip()})
                    current = sec["section_title"] + "\n" + sent  # carry header
                else:
                    current += " " + sent
            if current.strip():
                chunks.append({**sec, "text": current.strip()})
        else:
            chunks.append(sec)

    if buffer:
        chunks.append(buffer)

    return chunks

# ---------------------------------------------------------------------------
# Step 3b: Context headers
# ---------------------------------------------------------------------------
# A chunk cut from the middle of a regulation often never names the document,
# chapter or topic it belongs to ("Soldiers may carry forward up to 60 days"
# with no mention of leave policy or the Army). Each chunk gets a short header
# (document title, chapter, numbered headings inside it) that is embedded and
# keyword-indexed with the text. The stored text itself is unchanged.

DOC_TITLES = {
    "jtr.pdf":                 "Joint Travel Regulations (JTR), DoD travel and per diem",
    "ar_600_8_10.pdf":         "AR 600-8-10 Leaves and Passes, Army leave policy",
    "milpersman_1050.pdf":     "MILPERSMAN 1050, Navy leave policy",
    "afi_36_3003.pdf":         "DAFI 36-3003 Military Leave Program, Air Force and Space Force leave",
    "bupersinst_1610_10f.pdf": "BUPERSINST 1610.10F Navy Performance Evaluation System, Navy fitness reports and evaluations",
    "afi_36_2406.pdf":         "AFI 36-2406 Officer and Enlisted Evaluation Systems, Air Force performance reports",
    "dod_fmr_vol7a.pdf":       "DoD Financial Management Regulation Volume 7A, military pay",
    "mco_1610_7.pdf":          "MCO 1610.7 Performance Evaluation System, Marine Corps fitness reports",
}

_CHAPTER_LINE = re.compile(r"(?im)^\s*chapter\s+(\d{1,2})\b[\s—–:.-]*([^\n]*)")
_MINOR_WORDS = {"and", "of", "the", "for", "to", "in", "on", "a", "or", "by", "at"}
# Numbered paragraph headings: JTR 020301. / AR 3–5. / AFI 3.2.1. / FMR 3.2
_HEADING = re.compile(r"(?m)^\s*(\d{6}\.|\d{1,2}[–-]\d{1,3}\.|\d{1,2}(?:\.\d{1,2}){1,4}\.?)\s+"
                      r"([A-Z][^\n.]{2,70})")
# Documents whose paragraph numbers don't start with the chapter number
_UNCHAPTERED_NUMBERING = {"dod_fmr_vol7a.pdf"}


def _chapter_of(number: str) -> int:
    """Chapter a paragraph number belongs to: 020301. → 2, 3–5. → 3, 4.1.2. → 4."""
    return int(number[:2]) if re.fullmatch(r"\d{6}\.", number) else int(re.match(r"\d+", number).group())


def _heading_title(text: str) -> str | None:
    """A chapter title, or None for TOC lines, sentences and cross-references."""
    t = re.sub(r",\s*page\s*\d+.*$", "", text.strip())
    t = re.sub(r"\s*\d+\s*$", "", t)
    if not 4 <= len(t) <= 100 or "...." in t or not t[0].isupper():
        return None
    words = [w for w in t.split() if w[0].isalpha() and w.lower() not in _MINOR_WORDS]
    return t if words and all(w[0].isupper() for w in words) else None


def chapter_titles(texts: list[str]) -> dict[int, str]:
    """Chapter number → title, from the first clean heading for each chapter."""
    titles: dict[int, str] = {}
    for text in texts:
        for m in _CHAPTER_LINE.finditer(text):
            n = int(m.group(1))
            title = _heading_title(m.group(2))
            if title is None and not m.group(2).strip():
                title = _heading_title(text[m.end():].lstrip("\n").split("\n", 1)[0])
            if title and n not in titles:
                titles[n] = title
    return titles


def context_headers(source_file: str, texts: list[str]) -> list[str]:
    """One header per chunk, for one document's chunks in reading order."""
    titles = chapter_titles(texts)
    doc = DOC_TITLES.get(source_file, source_file)
    chapter = None
    headers = []
    for text in texts:
        headings = [(m.group(1), m.group(2).strip()) for m in _HEADING.finditer(text)]
        numbered = [] if source_file in _UNCHAPTERED_NUMBERING else \
            [_chapter_of(n) for n, _ in headings if _chapter_of(n) in titles]
        found = [int(m.group(1)) for m in _CHAPTER_LINE.finditer(text) if int(m.group(1)) in titles]
        # The chapter carried over from the previous chunk, then the chapters
        # this chunk's paragraph numbers (most reliable) or page headers name
        touched = list(dict.fromkeys(([chapter] if chapter is not None else []) + (numbered or found)))
        if len(touched) > 1 and headings and text.lstrip().startswith(headings[0][0]):
            touched = touched[1:]   # chunk opens with a new paragraph: nothing carried over
        chapter = touched[-1] if touched else None
        parts = [doc]
        for n in touched[-2:]:
            parts.append(f"Chapter {n}: {titles[n]}")
        heads = []
        for _, h in headings:
            if len(h.split()) <= 8 and h not in heads:
                heads.append(h)
        if heads:
            parts.append("; ".join(heads[:6]))
        headers.append(". ".join(parts))
    return headers


def embedding_text(header: str, text: str) -> str:
    return f"{header}\n\n{text}"

# ---------------------------------------------------------------------------
# Step 4 + 5: Embed and store in ChromaDB
# ---------------------------------------------------------------------------

def build_vectorstore(all_chunks: list[dict], model: SentenceTransformer, collection):
    texts = [c["text"] for c in all_chunks]
    print(f"  [embed] encoding {len(texts)} chunks ...")
    embeddings = model.encode([embedding_text(c["context"], c["text"]) for c in all_chunks],
                              batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    ids       = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in all_chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"  [ok] {len(texts)} chunks stored in ChromaDB")

# ---------------------------------------------------------------------------
# Step 6: Verification
# ---------------------------------------------------------------------------

VERIFY_QUERIES = [
    ("What is the per diem rate policy for TDY travel?",       "travel"),
    ("How many days of leave do soldiers accrue per month?",    "leave"),
    ("Is the GTC mandatory for TDY travel?",                   "travel"),
    ("What is the mileage reimbursement rate for POV travel?",  "travel"),
    ("What happens to leave when a soldier separates?",         "leave"),
]


def verify(collection, model: SentenceTransformer):
    print("\n--- Verification ---")
    for query, expected_domain in VERIFY_QUERIES:
        embedding = model.encode([query], normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=1,
            where={"domain": expected_domain},
        )
        if results["documents"] and results["documents"][0]:
            doc      = results["documents"][0][0]
            meta     = results["metadatas"][0][0]
            distance = results["distances"][0][0]
            print(f"\nQ: {query}")
            print(f"   source:  {meta.get('source_file')} | section: {meta.get('section_title', '')[:60]}")
            print(f"   score:   {1 - distance:.3f}")
            print(f"   snippet: {doc[:120].replace(chr(10), ' ')} ...")
        else:
            print(f"\nQ: {query}")
            print("   [no result returned]")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def open_collection(path: Path):
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})


def slim_store(path: Path):
    """Drop Chroma's built-in full-text search index and compact the database.

    Chroma keeps a second copy of every passage for its own text filtering,
    which this app never uses (keyword search is done in memory by
    agents/tools.py). Removing it cuts the store from ~54 MB to ~23 MB with
    identical search results. The store is read-only afterwards: re-run this
    script to change it (it always rebuilds from scratch).
    """
    gc.collect()   # let Chroma release its database connection
    db = sqlite3.connect(path / "chroma.sqlite3")
    db.execute("DROP TABLE IF EXISTS embedding_fulltext_search")
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"  [slim] store is now {sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) // 2**20} MB")


def main():
    run_verify_only = "--verify" in sys.argv

    # Suppress ChromaDB telemetry noise
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

    print("\n=== Loading Embedding Model ===")
    print(f"  model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # --verify: skip ingestion, just query existing store
    if run_verify_only:
        collection = open_collection(VECTORSTORE)
        count = collection.count()
        if count == 0:
            print("[!] Vector store is empty — run without --verify first.")
            sys.exit(1)
        print(f"  [ok] Store has {count} chunks — running verification only\n")
        verify(collection, model)
        return

    # --- Full ingestion ---
    print("\n=== Step 1: Checking PDFs ===")
    downloaded = []
    for src in SOURCES:
        ok = download_pdf(src["url"], src["local_path"])
        if ok:
            downloaded.append(src)

    if len(downloaded) < len(SOURCES):
        print("\n[!] Missing PDFs — the index would be incomplete. Download them manually:")
        for src in SOURCES:
            if src in downloaded:
                continue
            print(f"    {src['url']}")
            print(f"    -> save to: {src['local_path']}\n")
        print("Then re-run: python3 ingest.py")
        sys.exit(1)

    print("\n=== Step 2-3: Parsing and Chunking ===")
    all_chunks = []
    for src in downloaded:
        print(f"\n  Processing {src['source_file']} ...")
        pages    = parse_pdf(src["local_path"])
        sections = split_into_sections(pages)
        chunks   = chunk_sections(sections)

        for chunk in chunks:
            chunk["source_file"] = src["source_file"]
            chunk["source_url"]  = src["url"]
            chunk["domain"]      = src["domain"]
            chunk["branch"]      = src["branch"]
            chunk["doc_type"]    = src["doc_type"]

        for chunk, header in zip(chunks, context_headers(src["source_file"], [c["text"] for c in chunks])):
            chunk["context"] = header

        print(f"  [ok] {len(pages)} pages -> {len(sections)} sections -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    print("\n=== Step 4+5: Embedding and Storing ===")
    # Always a clean rebuild, and only now that every PDF parsed: the old
    # store is untouched if anything above failed.
    shutil.rmtree(VECTORSTORE, ignore_errors=True)
    collection = open_collection(VECTORSTORE)
    build_vectorstore(all_chunks, model, collection)
    total = collection.count()
    print(f"\n  Total chunks in store: {total}")
    del collection
    slim_store(VECTORSTORE)

    print("\n=== Done ===")
    print(f"Vector store saved to: {VECTORSTORE}/")
    print("Run 'python3 ingest.py --verify' to test retrieval.")


if __name__ == "__main__":
    main()
