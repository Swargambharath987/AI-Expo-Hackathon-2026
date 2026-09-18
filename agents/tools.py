"""
Tool definitions for the Duty Line ReAct agent.

Four tools the LLM can call:
  1. search_regulations  — ChromaDB semantic search over ingested PDFs
  2. get_per_diem        — offline GSA rate lookup from gsa_cache.json
  3. calculate_travel_cost — JTR-compliant cost breakdown, pure math
  4. fill_form           — populate DA/DD forms, PDF or structured fallback

Usage:
  from agents.tools import TOOL_SCHEMAS, execute_tool
  # Pass TOOL_SCHEMAS to LLM. Call execute_tool(name, args) with LLM's response.
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Silence ChromaDB telemetry noise at import time
# ---------------------------------------------------------------------------
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*existing embedding ID.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent.parent
VECTORSTORE   = ROOT / "vectorstore"
GSA_CACHE     = ROOT / "data" / "gsa_cache.json"
FORMS_DIR     = ROOT / "data" / "forms"
OUTPUT_DIR    = ROOT / "output"
PROFILE_FILE  = ROOT / "profile.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_MODEL       = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME   = "dutyline"
POV_MILEAGE_RATE  = 0.70        # $/mile FY2026 (JTR Chapter 2)
STANDARD_LODGING  = 110         # DoD CONUS standard rate
STANDARD_MIE      = 68          # DoD CONUS standard M&IE
FIRST_LAST_PCT    = 0.75        # JTR: 75% M&IE on first and last day

# Common installation-to-installation one-way distances (miles)
KNOWN_DISTANCES: dict[tuple[str, str], int] = {
    ("fort liberty",  "fort moore"):   370,
    ("fort moore",    "fort liberty"):  370,
    ("fort liberty",  "fort bragg"):     0,   # same base, renamed
    ("fort campbell", "fort liberty"):  560,
    ("fort bliss",    "fort cavazos"):  157,
    ("jblm",          "fort wainwright"): 1850,
    ("fort drum",     "fort liberty"):  620,
    ("fort stewart",  "fort moore"):    120,
    ("fort belvoir",  "fort liberty"):  390,
    ("pentagon",      "fort liberty"):  390,
    # Alaska
    ("jber",            "fort wainwright"):  360,
    ("fort wainwright", "jber"):              360,
    # Texas
    ("fort cavazos",    "fort sam houston"):  160,
    ("fort sam houston","fort cavazos"):       160,
    ("fort bliss",      "fort sam houston"):  450,
    ("fort sam houston","fort bliss"):         450,
    # East Coast
    ("fort meade",      "fort liberty"):      420,
    ("fort liberty",    "fort meade"):        420,
    ("fort belvoir",    "fort meade"):         35,
    ("fort meade",      "fort belvoir"):        35,
    ("west point",      "fort liberty"):      590,
    ("fort liberty",    "west point"):        590,
    # Southeast
    ("fort stewart",    "fort liberty"):      620,
    ("fort liberty",    "fort stewart"):      620,
    ("fort jackson",    "fort liberty"):      420,
    ("fort liberty",    "fort jackson"):      420,
    ("fort benning",    "fort stewart"):      120,  # fort benning = fort moore (old name)
    # Midwest / Central
    ("fort leonard wood","fort riley"):       310,
    ("fort riley",      "fort leonard wood"): 310,
    ("fort leavenworth","fort riley"):         70,
    ("fort riley",      "fort leavenworth"):   70,
    # West Coast
    ("fort irwin",      "camp pendleton"):    180,
    ("camp pendleton",  "fort irwin"):        180,
    ("jblm",            "fort lewis"):          0,  # same installation
    ("fort lewis",      "jblm"):               0,
    # Hawaii
    ("schofield barracks","tripler army medical center"): 15,
    # Pentagon / DC area
    ("pentagon",         "fort meade"):        30,
    ("fort meade",       "pentagon"):          30,
    ("pentagon",         "fort belvoir"):      15,
    ("fort belvoir",     "pentagon"):          15,
}

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (expensive — load only when first tool is called)
# ---------------------------------------------------------------------------
_embed_model  = None
_collection   = None
_gsa_cache    = None
_profile      = None


def _get_embed_and_collection():
    global _embed_model, _collection
    if _collection is not None:
        return _embed_model, _collection
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    _embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(
        path=str(VECTORSTORE),
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _embed_model, _collection


def _get_gsa_cache() -> dict:
    global _gsa_cache
    if _gsa_cache is not None:
        return _gsa_cache
    if not GSA_CACHE.exists():
        _gsa_cache = {}
        return _gsa_cache
    _gsa_cache = json.loads(GSA_CACHE.read_text())
    return _gsa_cache


def _get_profile() -> dict:
    global _profile
    if _profile is not None:
        return _profile
    if PROFILE_FILE.exists():
        _profile = json.loads(PROFILE_FILE.read_text())
    else:
        _profile = {}
    return _profile


# ---------------------------------------------------------------------------
# Tool 1 — search_regulations
# ---------------------------------------------------------------------------

def search_regulations(query: str, domain: str | None = None, n: int = 4) -> dict:
    """
    Search the local vector store for relevant regulation text.
    Returns top-n chunks with source citations.
    """
    try:
        model, collection = _get_embed_and_collection()
        if collection.count() == 0:
            return {
                "success": False,
                "error":   "Vector store is empty. Run ingest.py first.",
                "results": [],
            }

        embedding = model.encode([query], normalize_embeddings=True).tolist()
        kwargs: dict[str, Any] = {"query_embeddings": embedding, "n_results": min(n, collection.count())}
        if domain:
            kwargs["where"] = {"domain": domain}

        results = collection.query(**kwargs)
        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append({
                    "text":     doc,
                    "source":   meta.get("source_file", "unknown"),
                    "section":  meta.get("section_title", "")[:80],
                    "domain":   meta.get("domain", ""),
                    "score":    round(1 - dist, 3),
                })

        if not chunks:
            return {
                "success": True,
                "results": [],
                "note":    f"No results found for query in domain '{domain or 'all'}'. Try a broader query.",
            }

        return {"success": True, "results": chunks}

    except Exception as e:
        return {"success": False, "error": str(e), "results": []}


# ---------------------------------------------------------------------------
# Tool 2 — get_per_diem
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.lower().replace(" ", "_").replace(",", "").replace(".", "")


def get_per_diem(city: str, state: str = "", year: str = "2026") -> dict:
    """
    Look up per diem rates from the offline GSA cache.
    Accepts city name or military installation name.
    Falls back to DoD standard CONUS rate if city not found.
    """
    try:
        cache = _get_gsa_cache()
        rates_db = cache.get("rates", {})

        # Try direct key match: city_state
        direct_key = _normalize(f"{city}_{state}") if state else _normalize(city)
        if direct_key in rates_db:
            entry = rates_db[direct_key]
            return _format_per_diem(entry, "exact_match")

        # Try matching by installation name
        city_lower = city.lower()
        for key, entry in rates_db.items():
            inst = entry.get("installation", "").lower()
            if city_lower in inst or inst in city_lower:
                return _format_per_diem(entry, "installation_match")

        # Try partial city name match
        for key, entry in rates_db.items():
            if city_lower in entry.get("city", "").lower():
                return _format_per_diem(entry, "partial_match")

        # Fall back to standard rate
        standard = cache.get("_meta", {}).get("standard_rate", {})
        lodging = standard.get("lodging", STANDARD_LODGING)
        mie     = standard.get("mie",     STANDARD_MIE)
        return {
            "success":           True,
            "match_type":        "standard_conus_fallback",
            "city":              city,
            "state":             state,
            "installation":      "not in database",
            "lodging":           lodging,
            "mie":               mie,
            "mie_first_last":    round(mie * FIRST_LAST_PCT),
            "total_daily":       lodging + mie,
            "fiscal_year":       year,
            "note":              f"City '{city}' not in GSA cache. Using DoD standard CONUS rate.",
        }

    except Exception as e:
        return {
            "success": False, "error": str(e),
            "lodging": STANDARD_LODGING, "mie": STANDARD_MIE,
            "mie_first_last": round(STANDARD_MIE * FIRST_LAST_PCT),
        }


def _format_per_diem(entry: dict, match_type: str) -> dict:
    lodging = entry.get("lodging", STANDARD_LODGING)
    mie     = entry.get("mie",     STANDARD_MIE)
    return {
        "success":        True,
        "match_type":     match_type,
        "city":           entry.get("city", ""),
        "state":          entry.get("state", ""),
        "installation":   entry.get("installation", ""),
        "lodging":        lodging,
        "mie":            mie,
        "mie_first_last": entry.get("mie_first_last", round(mie * FIRST_LAST_PCT)),
        "total_daily":    lodging + mie,
        "fiscal_year":    entry.get("fiscal_year", "2026"),
        "source":         entry.get("source", "gsa_cache"),
    }


# ---------------------------------------------------------------------------
# Tool 3 — calculate_travel_cost
# ---------------------------------------------------------------------------

def calculate_travel_cost(
    destination_city:   str,
    destination_state:  str,
    travel_days:        int,
    travel_mode:        str       = "POV",
    one_way_miles:      float     = 0.0,
    origin:             str       = "",
) -> dict:
    """
    Compute full TDY travel cost applying JTR rules.

    JTR rules applied:
    - First and last travel day M&IE = 75% of daily rate
    - Mileage for POV at current GSA rate ($0.70/mile FY2026)
    - Lodging = travel_days - 1 nights (depart on last day)
    - Round-trip mileage for POV

    Args:
        destination_city:  city of TDY destination
        destination_state: 2-letter state abbreviation
        travel_days:       total calendar days of TDY (including travel days)
        travel_mode:       POV | RENTAL | GOVT | PLANE
        one_way_miles:     one-way distance in miles (required for POV)
        origin:            origin installation name (for distance lookup if miles=0)
    """
    try:
        travel_days = max(1, int(travel_days))

        # Get per diem rates
        rates = get_per_diem(destination_city, destination_state)
        lodging_rate      = rates["lodging"]
        mie_rate          = rates["mie"]
        mie_first_last    = rates["mie_first_last"]

        # Lodging: (days - 1) nights — soldier doesn't need lodging on departure day
        lodging_nights    = max(0, travel_days - 1)
        lodging_total     = lodging_nights * lodging_rate

        # M&IE: first and last day at 75%, middle days at full rate
        if travel_days == 1:
            meals_total = mie_first_last            # only one day, count as first/last
        elif travel_days == 2:
            meals_total = mie_first_last * 2        # both days are first/last
        else:
            middle_days = travel_days - 2
            meals_total = (mie_first_last * 2) + (mie_rate * middle_days)

        # Mileage
        mileage_total = 0.0
        mileage_note  = ""
        if travel_mode.upper() == "POV":
            if one_way_miles <= 0 and origin:
                # KNOWN_DISTANCES keys are plain lowercased strings (with
                # spaces), not the underscore-joined form _normalize()
                # produces for the GSA cache — match that format instead.
                o, d = origin.strip().lower(), destination_city.strip().lower()
                one_way_miles = KNOWN_DISTANCES.get((o, d)) or KNOWN_DISTANCES.get((d, o), 0)
            if one_way_miles > 0:
                round_trip_miles = one_way_miles * 2
                mileage_total    = round(round_trip_miles * POV_MILEAGE_RATE, 2)
                mileage_note     = (
                    f"{one_way_miles:.0f} miles × 2 (round trip) × ${POV_MILEAGE_RATE}/mile"
                )
            else:
                mileage_note = "POV mileage not calculated — provide one_way_miles"
        elif travel_mode.upper() == "PLANE":
            mileage_note = "Airfare cost not estimated — use actual ticket price"
        elif travel_mode.upper() in ("RENTAL", "GOVT"):
            mileage_note = f"{travel_mode} vehicle — mileage reimbursement not applicable"

        grand_total = round(lodging_total + meals_total + mileage_total, 2)

        return {
            "success":          True,
            "destination":      f"{destination_city}, {destination_state}",
            "travel_days":      travel_days,
            "travel_mode":      travel_mode,
            "per_diem_source":  rates.get("match_type", ""),
            "lodging": {
                "nights":       lodging_nights,
                "rate":         lodging_rate,
                "total":        round(lodging_total, 2),
            },
            "meals": {
                "days":         travel_days,
                "daily_rate":   mie_rate,
                "first_last_rate": mie_first_last,
                "total":        round(meals_total, 2),
                "jtr_note":     "First/last day at 75% per JTR",
            },
            "mileage": {
                "mode":         travel_mode,
                "one_way_miles": one_way_miles,
                "rate_per_mile": POV_MILEAGE_RATE if travel_mode.upper() == "POV" else 0,
                "total":        mileage_total,
                "note":         mileage_note,
            },
            "grand_total":      grand_total,
            "summary":          (
                f"Lodging: ${lodging_total:.2f} | "
                f"Meals: ${meals_total:.2f} | "
                f"Mileage: ${mileage_total:.2f} | "
                f"TOTAL: ${grand_total:.2f}"
            ),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool 4 — fill_form
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Form field structures
# ---------------------------------------------------------------------------

# AcroForm field map: our semantic keys → actual PDF AcroForm field names.
# Only forms with real AcroForm fields are listed here.
PDF_FIELD_MAP: dict[str, dict[str, str]] = {
    "DD_1610": {
        "name_last_first": "name",
        "ssn_last4":       "ssn",
        "unit":            "org_elem",
        "installation":    "pds",
        "departure_date":  "proc_date",
        "destination":     "itin",
        "purpose":         "tdy_purp",
        "travel_days":     "tdy_days",
        "cost_estimate":   "d_TOTAL",
        "per_diem_est":    "a_PER_DIEM",
        "remarks":         "remarks",
        "pos_title":       "pos_title",
        "auth_no":         "auth_no",
        "iss_date":        "iss_date",
    }
}

# Display field map: our semantic keys → human-readable labels for reportlab PDF.
# Also defines which fields appear (and in what order) for each form.
FORM_DISPLAY_FIELDS: dict[str, list] = {
    "DD_1610": [
        ("name_last_first", "Traveler Name"),
        ("rank",            "Grade/Rank"),
        ("unit",            "Organization"),
        ("installation",    "Permanent Duty Station"),
        ("departure_date",  "Proceed Date"),
        ("return_date",     "Return Date"),
        ("destination",     "Destination/Itinerary"),
        ("purpose",         "Purpose of Travel"),
        ("travel_days",     "TDY Days"),
        ("travel_mode",     "Mode of Transportation"),
        ("cost_estimate",   "Total Estimated Cost"),
        ("per_diem_est",    "Per Diem Estimate"),
        ("remarks",         "Remarks"),
        ("auth_no",         "Order/Authorization Number"),
    ],
    "DA_31": [
        ("name_last_first", "Name (Last, First, MI)"),
        ("rank",            "Rank"),
        ("grade",           "Grade"),
        ("ssn_last4",       "SSN (Last 4)"),
        ("dod_id",          "DoD ID"),
        ("unit",            "Organization/Unit"),
        ("installation",    "Installation"),
        ("leave_type",      "Type of Leave"),
        ("departure_date",  "Date Leave Begins"),
        ("return_date",     "Date Leave Ends"),
        ("num_days",        "Number of Days"),
        ("leave_address",   "Address During Leave"),
        ("phone",           "Contact Phone"),
        ("supervisor_name", "Approving Official"),
    ],
    "DA_4856": [
        ("name_last_first",  "Name of Soldier"),
        ("rank",             "Rank"),
        ("ssn_last4",        "SSN (Last 4)"),
        ("unit",             "Organization"),
        ("date",             "Date"),
        ("counseling_type",  "Purpose of Counseling"),
        ("key_points",       "Key Points of Discussion"),
        ("action_plan",      "Plan of Action"),
        ("leader_name",      "Counselor Name"),
        ("leader_rank",      "Counselor Rank/Title"),
        ("supervisor_title", "Counselor Position"),
    ],
    "DA_4187": [
        ("name_last_first",  "Name"),
        ("rank",             "Grade"),
        ("ssn_last4",        "SSN (Last 4)"),
        ("unit",             "Organization"),
        ("action_requested", "Type of Requested Personnel Action"),
        ("effective_date",   "Effective Date"),
        ("reason",           "Reason for Requested Action"),
    ],
}

# Human-readable full form titles used in the PDF header
_FORM_TITLES: dict[str, str] = {
    "DD_1610": "DD FORM 1610 — Request and Authorization for TDY Travel of Service Member",
    "DA_31":   "DA FORM 31 — Request and Authority for Leave",
    "DA_4856": "DA FORM 4856 — Developmental Counseling Form",
    "DA_4187": "DA FORM 4187 — Personnel Action",
}

# Backward-compatible alias so any existing callers of FORM_FIELDS still work
FORM_FIELDS: dict[str, dict[str, str]] = {
    name: {our: label for our, label in fields}
    for name, fields in FORM_DISPLAY_FIELDS.items()
}


# ---------------------------------------------------------------------------
# Helper — reportlab PDF generator
# ---------------------------------------------------------------------------

def _generate_reportlab_pdf(
    form_name: str,
    display_rows: list,
    missing_fields: list,
    output_path: Path,
) -> bool:
    """
    Generate a military-styled letter-size PDF for any DA/DD form.

    Args:
        form_name:      Form key, e.g. "DA_31"
        display_rows:   List of (label, value) tuples for filled fields
        missing_fields: List of display labels that had no value
        output_path:    Destination file path (Path object)

    Returns True on success, False on any exception.
    """
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.lib.enums import TA_CENTER

        # Brand colors
        OLIVE    = colors.HexColor("#445233")
        WHITE    = colors.white
        AMBER    = colors.HexColor("#b5881a")
        ROW_A    = colors.HexColor("#f5f5f0")
        ROW_B    = colors.white
        GRAY_TXT = colors.HexColor("#555555")

        title_text    = _FORM_TITLES.get(form_name, f"{form_name} — Military Form")
        subtitle_text = (
            f"Generated by Duty Line \u00b7 "
            f"{datetime.now().strftime('%d %b %Y %H:%M')} \u00b7 "
            "Review with your S1 before submission"
        )

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=LETTER,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()

        style_title = ParagraphStyle(
            "FormTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=WHITE,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
        style_subtitle = ParagraphStyle(
            "FormSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.HexColor("#ddddcc"),
            alignment=TA_CENTER,
        )
        style_label = ParagraphStyle(
            "FieldLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#222222"),
            leading=12,
        )
        style_value = ParagraphStyle(
            "FieldValue",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#111111"),
            leading=12,
        )
        style_missing_hdr = ParagraphStyle(
            "MissingHdr",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=AMBER,
            spaceBefore=10,
            spaceAfter=4,
        )
        style_missing_item = ParagraphStyle(
            "MissingItem",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=AMBER,
            leftIndent=12,
            leading=13,
        )
        style_footer = ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7,
            textColor=GRAY_TXT,
            alignment=TA_CENTER,
            spaceBefore=8,
        )

        usable_width = LETTER[0] - 1.5 * inch

        story = []

        # ---- Header banner ----
        header_table = Table(
            [[Paragraph(title_text, style_title)],
             [Paragraph(subtitle_text, style_subtitle)]],
            colWidths=[usable_width],
        )
        header_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), OLIVE),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.2 * inch))

        # ---- Field / Value table ----
        if display_rows:
            col_label = usable_width * 0.38
            col_value = usable_width * 0.62
            table_data = [
                [Paragraph("<b>Field</b>", style_label),
                 Paragraph("<b>Value</b>", style_label)],
            ]
            for label, value in display_rows:
                table_data.append([
                    Paragraph(label, style_label),
                    Paragraph(str(value), style_value),
                ])

            field_table = Table(table_data, colWidths=[col_label, col_value])
            ts = TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#d0d4c8")),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, 0), 9),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ])
            # Alternating row backgrounds (skip header row at index 0)
            for i in range(1, len(table_data)):
                bg = ROW_A if (i - 1) % 2 == 0 else ROW_B
                ts.add("BACKGROUND", (0, i), (-1, i), bg)
            field_table.setStyle(ts)
            story.append(field_table)

        # ---- Missing fields section (amber) ----
        if missing_fields:
            story.append(Spacer(1, 0.15 * inch))
            story.append(Paragraph(
                "MISSING FIELDS — Required before submission",
                style_missing_hdr,
            ))
            for label in missing_fields:
                story.append(Paragraph(f"\u25a1  {label}", style_missing_item))

        # ---- Footer disclaimer ----
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph(
            "This document was auto-generated by Duty Line for reference purposes. "
            "All information must be verified by the responsible S1/G1 before official submission. "
            "Do not use as a legally binding document without proper review and signature.",
            style_footer,
        ))

        doc.build(story)
        return True

    except Exception as rl_err:
        logging.error(f"reportlab PDF generation failed: {rl_err}")
        return False


# ---------------------------------------------------------------------------
# Tool 4 — fill_form
# ---------------------------------------------------------------------------

def fill_form(form_name: str, field_values: dict) -> dict:
    """
    Populate a military form with provided field values.
    Merges soldier profile for common fields (name, rank, unit, SSN-4).

    Always generates a reportlab PDF (reliable, works for all forms).
    For DD_1610: also attempts to fill the real AcroForm fields via pypdf
    and saves a second PDF with suffix _acroform.pdf if successful.

    Args:
        form_name:    One of: DD_1610, DA_31, DA_4856, DA_4187
        field_values: Dict of field names to values

    Returns dict with pdf_path (always set), filled_fields, missing_fields, summary.
    """
    try:
        form_name = form_name.upper().replace("-", "_").replace(" ", "_")
        if form_name not in FORM_DISPLAY_FIELDS:
            return {
                "success": False,
                "error":   (
                    f"Unknown form '{form_name}'. "
                    f"Supported: {list(FORM_DISPLAY_FIELDS.keys())}"
                ),
            }

        # Merge soldier profile into field_values (profile provides defaults)
        profile = _get_profile()
        merged  = {**profile, **field_values}   # field_values overrides profile

        # Auto-populate date if not provided
        if not merged.get("date"):
            merged["date"] = datetime.now().strftime("%d %b %Y").upper()

        display_schema = FORM_DISPLAY_FIELDS[form_name]

        # Build display_rows and track filled / missing
        display_rows:   list = []
        filled_dict:    dict = {}
        missing_labels: list = []

        for our_key, label in display_schema:
            val = merged.get(our_key)
            if val:
                display_rows.append((label, str(val)))
                filled_dict[label] = str(val)
            else:
                missing_labels.append(label)

        # ---- Primary output: reportlab PDF (always generated) ----
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"{form_name}_{ts}.pdf"
        rl_ok       = _generate_reportlab_pdf(form_name, display_rows, missing_labels, output_path)
        if not rl_ok:
            output_path = None   # rare edge case

        # ---- Secondary output: AcroForm fill for DD_1610 via pypdf ----
        acroform_path = None
        if form_name in PDF_FIELD_MAP:
            pdf_template = FORMS_DIR / f"{form_name.lower()}.pdf"
            if pdf_template.exists():
                acroform_map = PDF_FIELD_MAP[form_name]
                acro_filled: dict = {}
                for our_key, acro_field in acroform_map.items():
                    val = merged.get(our_key)
                    if val:
                        acro_filled[acro_field] = str(val)
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(pdf_template))
                    writer = pypdf.PdfWriter()
                    writer.append(reader)
                    pdf_fields = reader.get_fields()
                    if pdf_fields:
                        for page in writer.pages:
                            annotations = page.get("/Annots")
                            if annotations:
                                for annotation in annotations:
                                    obj = annotation.get_object()
                                    field_name = obj.get("/T")
                                    if field_name and field_name in acro_filled:
                                        obj.update({
                                            pypdf.generic.NameObject("/V"):
                                                pypdf.generic.create_string_object(
                                                    acro_filled[field_name]
                                                )
                                        })
                        acroform_path = OUTPUT_DIR / f"{form_name}_{ts}_acroform.pdf"
                        with open(acroform_path, "wb") as f:
                            writer.write(f)
                except Exception as pdf_err:
                    logging.warning(f"AcroForm fill failed for {form_name}: {pdf_err}")
                    acroform_path = None

        # ---- Text summary (always saved alongside PDF) ----
        lines = [
            f"{'='*50}",
            f"FORM: {form_name}",
            f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}",
            f"{'='*50}",
        ]
        for label, value in display_rows:
            lines.append(f"  {label:<40} {value}")
        if missing_labels:
            lines.append("")
            lines.append("MISSING (not provided or not in profile):")
            for m in missing_labels:
                lines.append(f"  [ ] {m}")
        summary_text = "\n".join(lines)

        txt_path = OUTPUT_DIR / f"{form_name}_{ts}.txt"
        txt_path.write_text(summary_text)

        result: dict = {
            "success":        True,
            "form_name":      form_name,
            "pdf_generated":  rl_ok,
            "pdf_path":       str(output_path) if output_path else None,
            "txt_path":       str(txt_path),
            "filled_fields":  filled_dict,
            "missing_fields": missing_labels,
            "summary":        summary_text,
        }
        if acroform_path:
            result["acroform_pdf_path"] = str(acroform_path)
            result["note"] = (
                "Reportlab PDF generated (primary). "
                "AcroForm PDF also saved — open in Adobe Acrobat for editable fields."
            )
        else:
            result["note"] = "Reportlab PDF generated successfully."

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool schemas — OpenAI function-calling format (gemma4:e4b / Claude / any OpenAI-compatible model)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": (
                "Search military regulations and policy documents for relevant text. "
                "Use for ANY question about rules, entitlements, eligibility, procedures, or policy. "
                "Available domains: 'travel' (JTR), 'leave' (AR 600-8-10), 'regs' (general ARs), 'eval' (AR 623-3). "
                "Always cite the source and section in your answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query, e.g. 'GTC mandatory for TDY' or 'leave accrual rate active duty'",
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["travel", "leave", "regs", "eval"],
                        "description": "Filter to a specific regulation domain. Omit to search all documents.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default 2, max 4)",
                        "default": 2,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_per_diem",
            "description": (
                "Look up GSA per diem rates (lodging and M&IE) for a TDY destination city. "
                "Works fully offline from pre-cached data. "
                "Returns lodging rate, M&IE rate, and first/last-day rate (75% per JTR). "
                "Use before calculate_travel_cost or when soldier asks about per diem for a specific location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Destination city or military installation name, e.g. 'Columbus' or 'Fort Moore'",
                    },
                    "state": {
                        "type": "string",
                        "description": "Two-letter state abbreviation, e.g. 'GA', 'NC', 'VA'",
                    },
                    "year": {
                        "type": "string",
                        "description": "Fiscal year for rates (default '2026')",
                        "default": "2026",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_travel_cost",
            "description": (
                "Calculate the full TDY travel cost breakdown applying JTR rules. "
                "Handles lodging, M&IE (with 75% first/last-day rule), and POV mileage. "
                "Use when soldier needs a cost estimate for a TDY trip."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "TDY destination city or installation name",
                    },
                    "destination_state": {
                        "type": "string",
                        "description": "Two-letter state abbreviation",
                    },
                    "travel_days": {
                        "type": "integer",
                        "description": "Total calendar days of TDY including travel days",
                    },
                    "travel_mode": {
                        "type": "string",
                        "enum": ["POV", "RENTAL", "GOVT", "PLANE"],
                        "description": "Mode of transportation: POV (personal vehicle), RENTAL, GOVT (government vehicle), PLANE",
                        "default": "POV",
                    },
                    "one_way_miles": {
                        "type": "number",
                        "description": "One-way distance in miles (required if travel_mode is POV)",
                        "default": 0,
                    },
                    "origin": {
                        "type": "string",
                        "description": "Origin installation name (used for distance lookup if one_way_miles not provided)",
                        "default": "",
                    },
                },
                "required": ["destination_city", "destination_state", "travel_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_form",
            "description": (
                "Populate a military form (DA or DD form) with the provided information. "
                "Automatically uses the soldier's profile for name, rank, unit, and SSN-4. "
                "Generates a filled PDF if the form template is available, otherwise a structured text summary. "
                "Use after collecting all needed information from the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "form_name": {
                        "type": "string",
                        "enum": ["DD_1610", "DA_31", "DA_4856", "DA_4187"],
                        "description": "Form to fill: DD_1610 (TDY request), DA_31 (leave request), DA_4856 (counseling), DA_4187 (personnel action)",
                    },
                    "field_values": {
                        "type": "object",
                        "description": (
                            "Dict of field values. Common fields: "
                            "departure_date, return_date, destination, purpose, travel_mode, cost_estimate (DD_1610); "
                            "leave_type, num_days, leave_address (DA_31); "
                            "counseling_type, key_points, action_plan, leader_name (DA_4856)"
                        ),
                    },
                },
                "required": ["form_name", "field_values"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher — called by the orchestrator with LLM's tool_call response
# ---------------------------------------------------------------------------

def execute_tool(name: str, arguments: dict) -> str:
    """
    Dispatch a tool call by name and return result as a string for LLM observation.
    Never raises — always returns something the LLM can reason about.
    """
    DISPATCH = {
        "search_regulations":    search_regulations,
        "get_per_diem":          get_per_diem,
        "calculate_travel_cost": calculate_travel_cost,
        "fill_form":             fill_form,
    }

    fn = DISPATCH.get(name)
    if fn is None:
        return json.dumps({"success": False, "error": f"Unknown tool '{name}'. Available: {list(DISPATCH)}"})

    try:
        result = fn(**arguments)
    except TypeError as e:
        return json.dumps({"success": False, "error": f"Bad arguments for '{name}': {e}"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Tool '{name}' failed: {e}"})

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Standalone test — python3 agents/tools.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("TOOL VERIFICATION")
    print("=" * 60)

    # Test 1: search_regulations
    print("\n[Tool 1] search_regulations")
    r = search_regulations("Is GTC mandatory for TDY travel?", domain="travel")
    if r["success"] and r["results"]:
        top = r["results"][0]
        print(f"  OK — score {top['score']} | {top['source']} | {top['section'][:50]}")
    else:
        print(f"  FAIL — {r.get('error', 'no results')}")

    # Test 2: get_per_diem (exact match)
    print("\n[Tool 2] get_per_diem — Columbus, GA (Fort Moore)")
    r = get_per_diem("Columbus", "GA")
    if r["success"]:
        print(f"  OK — lodging ${r['lodging']}/night | M&IE ${r['mie']}/day | match: {r['match_type']}")
    else:
        print(f"  FAIL — {r.get('error')}")

    # Test 2b: get_per_diem by installation name
    print("\n[Tool 2] get_per_diem — Fort Moore (installation name lookup)")
    r = get_per_diem("Fort Moore")
    if r["success"]:
        print(f"  OK — lodging ${r['lodging']}/night | M&IE ${r['mie']}/day | match: {r['match_type']}")
    else:
        print(f"  FAIL — {r.get('error')}")

    # Test 3: calculate_travel_cost (the SPC Rivera demo scenario)
    print("\n[Tool 3] calculate_travel_cost — Fort Liberty → Fort Moore, 5 days, POV")
    r = calculate_travel_cost(
        destination_city="Columbus",
        destination_state="GA",
        travel_days=5,
        travel_mode="POV",
        one_way_miles=370,
        origin="Fort Liberty",
    )
    if r["success"]:
        print(f"  OK — {r['summary']}")
        print(f"       Lodging: {r['lodging']['nights']} nights × ${r['lodging']['rate']} = ${r['lodging']['total']}")
        print(f"       Meals:   {r['meals']['days']} days (first/last at ${r['meals']['first_last_rate']}) = ${r['meals']['total']}")
        print(f"       Mileage: {r['mileage']['note']}")
    else:
        print(f"  FAIL — {r.get('error')}")

    # Test 4: fill_form (no PDF needed, text output)
    print("\n[Tool 4] fill_form — DA_31 (leave request)")
    r = fill_form("DA_31", {
        "name_last_first": "Rivera, Maria J.",
        "rank":            "SPC",
        "unit":            "1-503 IN, 82nd Airborne",
        "leave_type":      "Annual",
        "departure_date":  "10 JUL 2026",
        "return_date":     "20 JUL 2026",
        "num_days":        "10",
        "leave_address":   "123 Main St, Raleigh NC 27601",
    })
    if r["success"]:
        print(f"  OK — {len(r['filled_fields'])} fields filled | {len(r['missing_fields'])} missing")
        print(f"       Output: {r['txt_path']}")
        if r["missing_fields"]:
            print(f"       Missing: {r['missing_fields']}")
    else:
        print(f"  FAIL — {r.get('error')}")

    # Test 5: execute_tool dispatcher
    print("\n[Dispatcher] execute_tool")
    result_str = execute_tool("get_per_diem", {"city": "San Diego", "state": "CA"})
    result_obj = json.loads(result_str)
    if result_obj.get("success"):
        print(f"  OK — dispatcher works | lodging ${result_obj['lodging']}")
    else:
        print(f"  FAIL — {result_obj.get('error')}")

    # Test 6: bad tool name
    print("\n[Dispatcher] execute_tool — unknown tool name")
    result_str = execute_tool("make_coffee", {})
    result_obj = json.loads(result_str)
    print(f"  {'OK' if not result_obj['success'] else 'FAIL'} — returns error cleanly: {result_obj['error'][:60]}")

    print("\n" + "=" * 60)
    print("DONE")
