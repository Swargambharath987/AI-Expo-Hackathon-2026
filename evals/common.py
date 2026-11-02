"""Shared paths, loading and text-normalisation helpers for the eval harness."""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

EVALS_DIR   = Path(__file__).resolve().parent
ROOT        = EVALS_DIR.parent
CASES_DIR   = EVALS_DIR / "cases"
REPORTS_DIR = EVALS_DIR / "reports"
GOLD_FILE   = EVALS_DIR / "retrieval_gold.yaml"
RUBRIC_FILE = EVALS_DIR / "judge_rubric.md"

# The requesting user for every agent case. A squad leader at Fort Liberty —
# deliberately distinct from every soldier named in the cases so that
# profile-vs-subject mix-ups are detectable.
PROFILE = {
    "name_last_first":  "Carter, James R.",
    "rank":             "SSG",
    "grade":            "E-6",
    "ssn_last4":        "4821",
    "dod_id":           "1234567890",
    "unit":             "B Co, 1-505 PIR, 82nd ABN DIV",
    "installation":     "Fort Liberty, NC",
    "uic":              "WABCD1",
    "supervisor_name":  "CPT Hayes, Daniel",
    "supervisor_title": "Company Commander",
}


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for p in sorted(CASES_DIR.glob("*.yaml")):
        for c in load_yaml(p) or []:
            c["_file"] = p.name
            cases.append(c)
    ids = [c["id"] for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate case ids: {sorted(dupes)}")
    return cases


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

DASHES = re.compile(r"[‐‑‒–—―−]")


def norm_dashes(s: str) -> str:
    return DASHES.sub("-", s)


NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    """Every number that appears in a string (commas stripped)."""
    out: set[float] = set()
    for m in NUM_RE.findall(text):
        try:
            out.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def walk_numbers(obj) -> set[float]:
    """Numbers in a JSON-like object: numeric leaves plus numbers inside strings."""
    out: set[float] = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 2))
    elif isinstance(obj, str):
        out |= numbers_in(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= walk_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= walk_numbers(v)
    return out


STATE_ABBR = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4,
    "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11,
    "november": 11, "dec": 12, "december": 12,
}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\.?,?\s*(\d{{4}})?\b", re.I)
DATE_MDY = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})?\b", re.I)
DATE_NUM = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def dates_in(text: str) -> list[tuple[int, int, int | None]]:
    """(month, day, year|None) for every date-looking span in text."""
    out = []
    for d, mon, y in DATE_DMY.findall(text):
        out.append((MONTHS[mon.lower()[:3]], int(d), int(y) if y else None))
    for mon, d, y in DATE_MDY.findall(text):
        out.append((MONTHS[mon.lower()[:3]], int(d), int(y) if y else None))
    for m, d, y in DATE_NUM.findall(text):
        yy = int(y) + (2000 if len(y) == 2 else 0)
        out.append((int(m), int(d), yy))
    for y, m, d in DATE_ISO.findall(text):
        out.append((int(m), int(d), int(y)))
    return [(m, d, y) for m, d, y in out if 1 <= m <= 12 and 1 <= d <= 31]


def to_date(md: tuple[int, int, int | None], default_year: int) -> date | None:
    m, d, y = md
    try:
        return date(y or default_year, m, d)
    except ValueError:
        return None


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""
a an the and or of to for in on at by with from as is are was were be been his her he she
they them their it its this that these those will shall would should can could may might
must do does did have has had not no yes i me my we our you your per via into than then
also each any all over under up down out about after before during until while within
""".split())


def tokens(text: str) -> list[str]:
    t = norm_dashes(text.lower())
    for full, ab in STATE_ABBR.items():
        t = re.sub(rf"\b{full}\b", f"{full} {ab}", t)
    # month names -> 3-letter abbreviation so "November" matches "Nov"
    return [w[:3] if w in MONTHS else w for w in TOKEN_RE.findall(t)]


def content_tokens(text: str) -> list[str]:
    return [w for w in tokens(text) if w not in STOPWORDS and (len(w) > 2 or w.isdigit())]


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def today_form_date() -> str:
    """The auto-date fill_form writes when no date is given."""
    return datetime.now().strftime("%d %b %Y").upper()


def add_days(d: date, n: int) -> date:
    return d + timedelta(days=n)


def jdump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes out of n (sound for small n and rates near 0/1)."""
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))
