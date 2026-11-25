#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layout-aware extraction of financial contracts in PDF and generation of a
reproducible intermediate JSON plus XML-like context for LLMs.

Main design points:
- Preserve line geometry and text provenance.
- Avoid deleting body content that merely looks like a page number.
- Detect common legal section headings without confusing lowercase bullets.
- Reconstruct line text from spans using horizontal gaps instead of blindly
  inserting spaces between every span.
- Preserve the original content order inside each section through section items.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Falta PyMuPDF. Instala la dependencia con: pip install pymupdf"
    ) from exc


# Text normalization.
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
SPANISH_ALPHA = "A-Za-z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1"

# Page number detection. PAGE_NUMBER_RE deliberately still accepts bare numbers,
# but bare numbers are removed only when they are in page margins. Explicit
# labels such as Page/Pagina/Pag./Pag. with accent may be removed anywhere.
PAGE_LABEL_PATTERN = r"(?:page|p\.|pag\.|p\u00e1g\.|pagina\.?|p\u00e1gina\.?)"
PAGE_NUMBER_RE = re.compile(
    rf"^\s*(?:{PAGE_LABEL_PATTERN}\s*)?\d{{1,4}}\s*(?:(?:/|of|de)\s*\d{{1,4}})?\s*$",
    re.IGNORECASE,
)
PAGE_NUMBER_WITH_MARKER_RE = re.compile(
    rf"^\s*{PAGE_LABEL_PATTERN}\s*\d{{1,4}}\s*(?:(?:/|of|de)\s*\d{{1,4}})?\s*$",
    re.IGNORECASE,
)

# Legal heading detection. Numeric headings allow "1. Definitions" as well as
# "1 Definitions" and nested forms such as "1.2.3 Definitions".
SECTION_PREFIX_PATTERN = (
    r"(?:section|clause|article|part|schedule|appendix|annex|"
    r"cap[i\u00ed]tulo|cl[a\u00e1]usula|anexo)"
)
NUMBERED_SECTION_RE = re.compile(
    rf"^\s*(?:(?P<prefix>{SECTION_PREFIX_PATTERN})\s+)?"
    r"(?P<number>\d+(?:\.\d+)*\.?)\s+(?P<title>\S.*)$",
    re.IGNORECASE,
)
PREFIXED_SECTION_RE = re.compile(
    rf"^\s*(?P<prefix>{SECTION_PREFIX_PATTERN})\s+"
    r"(?P<number>(?:\d+(?:\.\d+)*\.?|[A-Z]{1,4}\.?|[IVXLCDM]{1,8}\.?))\s+"
    r"(?P<title>\S.*)$",
    re.IGNORECASE,
)
UPPER_ALPHA_SECTION_RE = re.compile(r"^\s*(?P<number>[A-Z]\.)\s+(?P<title>\S.*)$")
UPPER_ROMAN_SECTION_RE = re.compile(r"^\s*(?P<number>[IVXLCDM]{1,8}\.)\s+(?P<title>\S.*)$")

BULLET_SYMBOLS = "-*\u2022\u2023\u25aa\u25e6"
LIST_SYMBOL_BULLET_RE = re.compile(rf"^\s*[{re.escape(BULLET_SYMBOLS)}]\s+\S+")
PAREN_BULLET_RE = re.compile(r"^\s*\(?[a-z0-9]{1,3}\)\s+\S+")
NUMERIC_PAREN_BULLET_RE = re.compile(r"^\s*\d{1,3}\)\s+\S+")
LOWER_DOT_BULLET_RE = re.compile(r"^\s*(?:[a-z]|[ivxlcdm]{1,8})\.\s+\S+")
NUMERIC_DOT_BULLET_RE = re.compile(r"^\s*\d{1,3}\.\s+\S+")

FIELD_RE = re.compile(
    rf"^\s*([{SPANISH_ALPHA}][\w\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1 .,/()&%\u00b0\u00ba\u00aa\-]{{2,90}}):\s*(.{{0,800}})\s*$"
)
CONTENT_KEYS = ("paragraphs", "bullets", "fields", "tables")


@dataclass
class LineItem:
    """Extracted line with geometric provenance."""

    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    font: str
    block_no: int
    line_no: int
    span_count: int
    wide_gap_count: int
    cells: list[dict[str, Any]] = field(default_factory=list)
    column: int = 0
    removed: bool = False
    remove_reason: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class PageInfo:
    page: int
    width: float
    height: float
    raw_text_chars: int
    extracted_line_count: int
    layout: str = "unknown"
    column_boundary_x: float | None = None
    ocr_used: bool = False
    alerts: list[str] = field(default_factory=list)


@dataclass
class ExtractionConfig:
    header_footer_margin_ratio: float = 0.08
    repeated_line_min_ratio: float = 0.60
    min_chars_for_text_layer: int = 80
    two_column_gap_ratio: float = 0.17
    min_lines_for_column_detection: int = 8
    use_ocr_when_needed: bool = True
    ocr_language: str = "eng+spa"
    password: str = ""


def normalize_inline_text(text: str) -> str:
    """Normalize text while preserving leading/trailing spaces."""

    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = CONTROL_CHARS_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text)


def normalize_text(text: str) -> str:
    """Normalize text to a single stripped line/block."""

    return normalize_inline_text(text).strip()


def canonical_repeated_line(text: str) -> str:
    """Stable signature for repeated header/footer detection."""

    text = normalize_text(text).lower()
    text = re.sub(r"\b\d{1,4}\b", "#", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -|\u2022\t")


def median_or_zero(values: Iterable[float]) -> float:
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(statistics.median(values)) if values else 0.0


def union_bbox(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def span_gap_count(cells: list[dict[str, Any]], page_width: float) -> int:
    """Count wide horizontal gaps between spans in a line."""

    if len(cells) < 2:
        return 0
    ordered = sorted(cells, key=lambda cell: cell["bbox"][0])
    threshold = max(18.0, page_width * 0.035)
    gaps = 0
    for left, right in zip(ordered, ordered[1:]):
        gap = float(right["bbox"][0]) - float(left["bbox"][2])
        if gap >= threshold:
            gaps += 1
    return gaps


def join_spans_gap_aware(spans: list[dict[str, Any]]) -> str:
    """Join span texts without inserting artificial spaces inside words.

    PyMuPDF may split a visual word into several spans when font attributes
    change. Blindly doing " ".join(spans) can turn "interest" into
    "inter est". This function preserves span edge spaces when present and
    inserts a space only when the horizontal gap is large enough.
    """

    if not spans:
        return ""

    ordered = sorted(spans, key=lambda span: float(span.get("bbox", (0, 0, 0, 0))[0]))
    parts: list[str] = []
    previous_x1: float | None = None
    previous_size = 0.0

    for span in ordered:
        raw_text = normalize_inline_text(str(span.get("text", "")))
        if not raw_text or not raw_text.strip():
            continue

        bbox = tuple(float(v) for v in span.get("bbox", (0, 0, 0, 0)))
        x0, _, x1, _ = bbox
        size = float(span.get("size", 0.0) or 0.0)

        if parts and previous_x1 is not None:
            gap = x0 - previous_x1
            has_edge_space = parts[-1].endswith(" ") or raw_text.startswith(" ")
            starts_with_closing_punctuation = raw_text.lstrip()[:1] in ".,;:%)]}»”"
            previous_ends_open = parts[-1].rstrip().endswith(("(", "[", "{", "¿", "¡", "«", "“"))
            if (
                not has_edge_space
                and not starts_with_closing_punctuation
                and not previous_ends_open
                and gap > max(1.5, previous_size * 0.25)
            ):
                parts.append(" ")

        parts.append(raw_text)
        previous_x1 = x1
        previous_size = size

    return normalize_text("".join(parts))


def extract_page_text_dict(page: fitz.Page, use_ocr: bool, language: str) -> tuple[dict[str, Any], bool, list[str]]:
    """Return PyMuPDF text dictionary and OCR status."""

    alerts: list[str] = []
    if use_ocr:
        try:
            textpage = page.get_textpage_ocr(language=language, full=True)
            return page.get_text("dict", sort=False, textpage=textpage), True, alerts
        except Exception as exc:  # OCR depends on Tesseract and tessdata.
            alerts.append(f"OCR solicitado pero no disponible en esta pagina: {exc}")
    return page.get_text("dict", sort=False), False, alerts


def extract_lines_from_page(
    page: fitz.Page,
    page_number: int,
    config: ExtractionConfig,
    force_ocr: bool = False,
) -> tuple[list[LineItem], PageInfo]:
    """Extract lines, coordinates and typographic metadata from one page."""

    rect = page.rect
    raw_text = page.get_text("text") or ""
    should_ocr = force_ocr or (
        config.use_ocr_when_needed and len(normalize_text(raw_text)) < config.min_chars_for_text_layer
    )
    text_dict, ocr_used, alerts = extract_page_text_dict(page, should_ocr, config.ocr_language)

    lines: list[LineItem] = []
    for block_no, block in enumerate(text_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_no, line in enumerate(block.get("lines", [])):
            spans = [span for span in line.get("spans", []) if normalize_text(str(span.get("text", "")))]
            if not spans:
                continue
            cells: list[dict[str, Any]] = []
            span_sizes: list[float] = []
            span_fonts: list[str] = []
            span_boxes: list[tuple[float, float, float, float]] = []

            for span in spans:
                text = normalize_text(str(span.get("text", "")))
                if not text:
                    continue
                bbox = tuple(float(v) for v in span.get("bbox", (0, 0, 0, 0)))
                span_sizes.append(float(span.get("size", 0.0) or 0.0))
                span_fonts.append(str(span.get("font", "")))
                span_boxes.append(bbox)
                cells.append(
                    {
                        "text": text,
                        "bbox": list(bbox),
                        "font": str(span.get("font", "")),
                        "size": float(span.get("size", 0.0) or 0.0),
                    }
                )

            text = join_spans_gap_aware(spans)
            if not text:
                continue
            x0, y0, x1, y1 = union_bbox(span_boxes)
            font = Counter(span_fonts).most_common(1)[0][0] if span_fonts else ""
            font_size = median_or_zero(span_sizes)
            line_item = LineItem(
                page=page_number,
                text=text,
                x0=round(x0, 2),
                y0=round(y0, 2),
                x1=round(x1, 2),
                y1=round(y1, 2),
                font_size=round(font_size, 2),
                font=font,
                block_no=block_no,
                line_no=line_no,
                span_count=len(cells),
                wide_gap_count=span_gap_count(cells, float(rect.width)),
                cells=cells,
            )
            lines.append(line_item)

    page_info = PageInfo(
        page=page_number,
        width=round(float(rect.width), 2),
        height=round(float(rect.height), 2),
        raw_text_chars=len(normalize_text(raw_text)),
        extracted_line_count=len(lines),
        ocr_used=ocr_used,
        alerts=alerts,
    )
    if not lines:
        page_info.alerts.append("No se extrajeron lineas de texto en esta pagina.")
    return lines, page_info


def detect_columns_for_page(lines: list[LineItem], page_info: PageInfo, config: ExtractionConfig) -> None:
    """Detect one or two columns through horizontal center separation."""

    usable = [
        line
        for line in lines
        if len(line.text) >= 3
        and line.width >= page_info.width * 0.04
        and line.width <= page_info.width * 0.65
    ]
    if len(usable) < config.min_lines_for_column_detection:
        page_info.layout = "one_column_or_sparse"
        for line in lines:
            line.column = 1
        return

    centers = sorted(line.center_x for line in usable)
    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    if not gaps:
        page_info.layout = "one_column"
        for line in lines:
            line.column = 1
        return

    best_gap, best_idx = max(gaps, key=lambda item: item[0])
    boundary = (centers[best_idx] + centers[best_idx + 1]) / 2.0
    left_count = sum(center <= boundary for center in centers)
    right_count = len(centers) - left_count
    gap_is_large = best_gap >= page_info.width * config.two_column_gap_ratio
    enough_each_side = left_count >= 3 and right_count >= 3
    boundary_is_central = page_info.width * 0.30 <= boundary <= page_info.width * 0.70

    if gap_is_large and enough_each_side and boundary_is_central:
        page_info.layout = "two_column_heuristic"
        page_info.column_boundary_x = round(boundary, 2)
        for line in lines:
            if line.width > page_info.width * 0.70:
                line.column = 0  # wide element: title, full-width table or notice.
            else:
                line.column = 1 if line.center_x <= boundary else 2
    else:
        page_info.layout = "one_column"
        for line in lines:
            line.column = 1


def order_column_zone(zone_lines: list[LineItem]) -> list[LineItem]:
    """Order non-wide lines inside a vertical zone of a two-column page."""

    left = [line for line in zone_lines if line.column == 1]
    right = [line for line in zone_lines if line.column == 2]
    other = [line for line in zone_lines if line.column not in (1, 2)]
    ordered: list[LineItem] = []
    ordered.extend(sorted(left, key=lambda item: (item.y0, item.x0)))
    ordered.extend(sorted(right, key=lambda item: (item.y0, item.x0)))
    ordered.extend(sorted(other, key=lambda item: (item.y0, item.x0)))
    return ordered


def order_two_column_page(page_lines: list[LineItem]) -> list[LineItem]:
    """Order a two-column page while keeping full-width elements in place.

    The page is split into vertical zones delimited by wide lines. Each zone is
    read left column then right column, and every wide line is emitted at its
    visual vertical position. This avoids moving mid-page full-width tables or
    headings to the end of the page.
    """

    wide_lines = sorted([line for line in page_lines if line.column == 0], key=lambda item: (item.y0, item.x0))
    narrow_lines = [line for line in page_lines if line.column != 0]
    ordered: list[LineItem] = []
    cursor_y = float("-inf")
    tolerance = 1.0

    for wide_line in wide_lines:
        zone = [line for line in narrow_lines if cursor_y - tolerance <= line.y0 < wide_line.y0 - tolerance]
        ordered.extend(order_column_zone(zone))
        ordered.append(wide_line)
        cursor_y = max(cursor_y, wide_line.y1)

    remaining = [line for line in narrow_lines if line.y0 >= cursor_y - tolerance]
    ordered.extend(order_column_zone(remaining))
    return ordered


def reading_order(lines: list[LineItem], pages: dict[int, PageInfo]) -> list[LineItem]:
    """Order lines with a reproducible page/layout rule."""

    ordered: list[LineItem] = []
    for page_number in sorted({line.page for line in lines}):
        page_lines = [line for line in lines if line.page == page_number]
        page_info = pages[page_number]
        if page_info.layout == "two_column_heuristic":
            ordered.extend(order_two_column_page(page_lines))
        else:
            ordered.extend(sorted(page_lines, key=lambda item: (item.y0, item.x0)))
    return ordered


def header_footer_position(line: LineItem, page_info: PageInfo, config: ExtractionConfig) -> tuple[bool, bool]:
    margin = page_info.height * config.header_footer_margin_ratio
    in_top = line.y1 <= margin
    in_bottom = line.y0 >= page_info.height - margin
    return in_top, in_bottom


def is_page_number_line(line: LineItem, page_info: PageInfo, config: ExtractionConfig) -> bool:
    """Return True only for safe page-number candidates.

    Bare numbers are considered page numbers only in top/bottom margins.
    Explicitly labelled page numbers are considered safe anywhere.
    """

    if PAGE_NUMBER_RE.match(line.text) is None:
        return False
    if PAGE_NUMBER_WITH_MARKER_RE.match(line.text) is not None:
        return True
    in_top, in_bottom = header_footer_position(line, page_info, config)
    return in_top or in_bottom


def mark_repeated_headers_and_footers(
    lines: list[LineItem],
    pages: dict[int, PageInfo],
    config: ExtractionConfig,
) -> list[str]:
    """Mark repeated headers/footers without physically deleting lines."""

    per_page_signatures: dict[str, set[int]] = defaultdict(set)
    signature_to_text: dict[str, str] = {}

    for line in lines:
        page_info = pages[line.page]
        in_top, in_bottom = header_footer_position(line, page_info, config)
        page_number_like = is_page_number_line(line, page_info, config)
        if in_top or in_bottom or page_number_like:
            signature = canonical_repeated_line(line.text)
            if signature:
                per_page_signatures[signature].add(line.page)
                signature_to_text.setdefault(signature, line.text)

    page_count = max(1, len(pages))
    # A line cannot be considered "repeated" in a meaningful way if it appears
    # on only one page. This prevents one-page contracts from losing legitimate
    # top/bottom content merely because it is inside the margin band.
    min_repeat_pages = 2
    repeated = {
        signature
        for signature, page_numbers in per_page_signatures.items()
        if len(page_numbers) >= min_repeat_pages
        and len(page_numbers) / page_count >= config.repeated_line_min_ratio
        and len(signature) >= 2
    }

    removed_examples: list[str] = []
    for line in lines:
        page_info = pages[line.page]
        signature = canonical_repeated_line(line.text)
        in_top, in_bottom = header_footer_position(line, page_info, config)
        if is_page_number_line(line, page_info, config):
            line.removed = True
            line.remove_reason = "page_number"
        elif signature in repeated and (in_top or in_bottom):
            line.removed = True
            line.remove_reason = "repeated_header_footer"
            removed_examples.append(signature_to_text.get(signature, line.text))

    return sorted(set(removed_examples))[:20]


def first_alpha_is_upper(text: str) -> bool:
    for char in text:
        if char.isalpha():
            return char.upper() == char
    return False


def looks_like_sentence_after_section_number(text: str) -> bool:
    """Heuristic guard against treating numbered list sentences as headings."""

    lowered = f" {normalize_text(text).lower()} "
    sentence_markers = (
        " shall ",
        " must ",
        " may ",
        " will ",
        " agree ",
        " agrees ",
        " is ",
        " are ",
        " deberá ",
        " debera ",
        " debe ",
        " podrá ",
        " podra ",
        " será ",
        " sera ",
        " estará ",
        " estara ",
        " son ",
        " está ",
        " esta ",
    )
    words = re.findall(rf"[{SPANISH_ALPHA}]+", text)
    return len(words) >= 4 and any(marker in lowered for marker in sentence_markers)


def title_signals(text: str, line: LineItem, median_font_size: float) -> tuple[bool, bool, bool]:
    words = re.findall(rf"[{SPANISH_ALPHA}]+", text)
    letters = sum(1 for char in text if char.isalpha())
    uppercase_letters = sum(1 for char in text if char.isalpha() and char.upper() == char)
    uppercase_ratio = uppercase_letters / letters if letters else 0.0
    short_title = 0 < len(words) <= 14 and not text.rstrip().endswith(";")
    font_signal = median_font_size > 0 and line.font_size >= median_font_size + 1.25
    all_caps_signal = uppercase_ratio >= 0.72 and len(words) <= 12
    return short_title, font_signal, all_caps_signal


def obvious_lowercase_or_symbol_bullet(text: str) -> bool:
    return any(
        regex.match(text)
        for regex in (
            LIST_SYMBOL_BULLET_RE,
            PAREN_BULLET_RE,
            NUMERIC_PAREN_BULLET_RE,
            LOWER_DOT_BULLET_RE,
        )
    )


def heading_match_title(text: str) -> tuple[str, bool] | None:
    """Return heading title and whether the heading has an explicit prefix."""

    prefixed = PREFIXED_SECTION_RE.match(text)
    if prefixed:
        return prefixed.group("title"), True
    numbered = NUMBERED_SECTION_RE.match(text)
    if numbered:
        return numbered.group("title"), bool(numbered.group("prefix"))
    upper_alpha = UPPER_ALPHA_SECTION_RE.match(text)
    if upper_alpha:
        return upper_alpha.group("title"), False
    upper_roman = UPPER_ROMAN_SECTION_RE.match(text)
    if upper_roman:
        return upper_roman.group("title"), False
    return None


def looks_like_heading(line: LineItem, median_font_size: float) -> bool:
    text = line.text.strip()
    if len(text) < 3 or len(text) > 180:
        return False
    if FIELD_RE.match(text):
        return False
    if obvious_lowercase_or_symbol_bullet(text):
        return False

    short_title, font_signal, all_caps_signal = title_signals(text, line, median_font_size)

    upper_alpha = UPPER_ALPHA_SECTION_RE.match(text)
    upper_roman = UPPER_ROMAN_SECTION_RE.match(text)
    if upper_alpha or upper_roman:
        marker_match = upper_alpha or upper_roman
        heading_title = marker_match.group("title")
        title_short, _, title_all_caps = title_signals(heading_title, line, median_font_size)
        return title_short and (font_signal or title_all_caps)

    heading_match = heading_match_title(text)
    if heading_match is not None:
        heading_title, has_explicit_prefix = heading_match
        title_short, _, title_all_caps = title_signals(heading_title, line, median_font_size)
        if has_explicit_prefix:
            return title_short or font_signal or title_all_caps
        if title_short and (font_signal or title_all_caps):
            return True
        return (
            title_short
            and first_alpha_is_upper(heading_title)
            and not looks_like_sentence_after_section_number(heading_title)
        )

    if not short_title:
        return False
    return font_signal or all_caps_signal


def looks_like_bullet(line: LineItem, median_font_size: float) -> bool:
    text = line.text.strip()
    if obvious_lowercase_or_symbol_bullet(text):
        return True
    if NUMERIC_DOT_BULLET_RE.match(text) and not looks_like_heading(line, median_font_size):
        return True
    if (UPPER_ALPHA_SECTION_RE.match(text) or UPPER_ROMAN_SECTION_RE.match(text)) and not looks_like_heading(
        line, median_font_size
    ):
        return True
    return False


def looks_like_table_row(line: LineItem) -> bool:
    text = line.text
    if "|" in text or "\t" in text:
        return True
    if line.wide_gap_count >= 2 and line.span_count >= 3:
        return True
    numeric_tokens = re.findall(r"(?:\d+[.,]?\d*%?|[$\u20ac\u00a3]\s*\d+[.,]?\d*)", text)
    return line.wide_gap_count >= 1 and len(numeric_tokens) >= 2


def new_section(section_number: int, title: str, page: int) -> dict[str, Any]:
    return {
        "section_id": f"sec_{section_number:03d}",
        "title": title,
        "page_start": page,
        "page_end": page,
        "paragraphs": [],
        "bullets": [],
        "fields": [],
        "tables": [],
        "items": [],
    }


def section_has_content(section: dict[str, Any]) -> bool:
    return any(section.get(key) for key in CONTENT_KEYS)


def add_section_item(section: dict[str, Any], item_type: str, **refs: int) -> None:
    item: dict[str, Any] = {"type": item_type}
    item.update(refs)
    section.setdefault("items", []).append(item)


def merge_paragraph_line(section: dict[str, Any], line: LineItem) -> None:
    """Merge line into section paragraphs while preserving item order."""

    paragraphs = section["paragraphs"]
    text = line.text
    page_ref = {"page": line.page, "bbox": [line.x0, line.y0, line.x1, line.y1]}

    last_item = section.get("items", [])[-1] if section.get("items") else None
    last_item_is_current_paragraph = (
        bool(last_item)
        and last_item.get("type") == "paragraph"
        and last_item.get("paragraph_index") == len(paragraphs) - 1
    )

    if not paragraphs or not last_item_is_current_paragraph:
        paragraph_index = len(paragraphs)
        paragraphs.append({"text": text, "source_lines": [page_ref]})
        add_section_item(section, "paragraph", paragraph_index=paragraph_index)
        return

    previous = paragraphs[-1]
    previous_text = previous["text"]
    should_join = False
    if previous_text.endswith("-") and not previous_text.endswith(" -"):
        previous["text"] = previous_text[:-1] + text.lstrip()
        previous["source_lines"].append(page_ref)
        return
    if not re.search(r"[.!?;:]\s*$", previous_text) and not looks_like_table_row(line):
        should_join = True
    if text[:1].islower():
        should_join = True

    if should_join:
        previous["text"] = normalize_text(previous_text + " " + text)
        previous["source_lines"].append(page_ref)
    else:
        paragraph_index = len(paragraphs)
        paragraphs.append({"text": text, "source_lines": [page_ref]})
        add_section_item(section, "paragraph", paragraph_index=paragraph_index)


def add_table_row(section: dict[str, Any], line: LineItem) -> tuple[int, int]:
    last_item = section.get("items", [])[-1] if section.get("items") else None
    previous_item_was_table = bool(last_item) and last_item.get("type") == "table_row"
    previous_table = section["tables"][-1] if section["tables"] else None
    table_page_gap = bool(previous_table) and previous_table["page_end"] < line.page - 1
    if not previous_table or table_page_gap or not previous_item_was_table:
        table_id = f"{section['section_id']}_tbl_{len(section['tables']) + 1:03d}"
        section["tables"].append({"table_id": table_id, "page_start": line.page, "page_end": line.page, "rows": []})
    table_index = len(section["tables"]) - 1
    table = section["tables"][table_index]
    table["page_end"] = max(table["page_end"], line.page)
    cells = [cell["text"] for cell in line.cells] if line.cells else [line.text]
    row_index = len(table["rows"])
    table["rows"].append(
        {
            "page": line.page,
            "bbox": [line.x0, line.y0, line.x1, line.y1],
            "cells": cells,
            "raw_text": line.text,
        }
    )
    return table_index, row_index


def segment_lines_into_sections(lines: list[LineItem]) -> list[dict[str, Any]]:
    """Reconstruct sections, paragraphs, bullets, fields and tables."""

    usable_lines = [line for line in lines if not line.removed and line.text]
    median_font_size = median_or_zero(line.font_size for line in usable_lines)
    sections: list[dict[str, Any]] = []
    section_counter = 0
    current = new_section(section_counter, "Document start", usable_lines[0].page if usable_lines else 1)
    sections.append(current)

    for line in usable_lines:
        if looks_like_heading(line, median_font_size):
            if current["title"] == "Document start" and not section_has_content(current):
                current["title"] = line.text
                current["page_start"] = line.page
                current["page_end"] = line.page
            else:
                section_counter += 1
                current = new_section(section_counter, line.text, line.page)
                sections.append(current)
            continue

        current["page_end"] = max(current["page_end"], line.page)

        field_match = FIELD_RE.match(line.text)
        if field_match and len(field_match.group(1).split()) <= 12:
            field_obj = {
                "name": normalize_text(field_match.group(1)),
                "value": normalize_text(field_match.group(2)),
                "page": line.page,
                "bbox": [line.x0, line.y0, line.x1, line.y1],
            }
            field_index = len(current["fields"])
            current["fields"].append(field_obj)
            add_section_item(current, "field", field_index=field_index)
        elif looks_like_bullet(line, median_font_size):
            bullet_obj = {
                "text": line.text,
                "page": line.page,
                "bbox": [line.x0, line.y0, line.x1, line.y1],
            }
            bullet_index = len(current["bullets"])
            current["bullets"].append(bullet_obj)
            add_section_item(current, "bullet", bullet_index=bullet_index)
        elif looks_like_table_row(line):
            table_index, row_index = add_table_row(current, line)
            add_section_item(current, "table_row", table_index=table_index, row_index=row_index)
        else:
            merge_paragraph_line(current, line)

    clean_sections = []
    for section in sections:
        if section_has_content(section) or section["title"] != "Document start":
            clean_sections.append(section)
    return clean_sections


def guess_language(text: str) -> str:
    """Simple and auditable predominant-language guess."""

    lowered = f" {text.lower()} "
    spanish_markers = [" el ", " la ", " de ", " que ", " contrato ", " riesgo ", " obligaciones "]
    english_markers = [" the ", " and ", " of ", " agreement ", " risk ", " party ", " transaction "]
    es_score = sum(lowered.count(marker) for marker in spanish_markers)
    en_score = sum(lowered.count(marker) for marker in english_markers)
    if es_score > en_score * 1.25:
        return "es"
    if en_score > es_score * 1.25:
        return "en"
    if es_score or en_score:
        return "mixed_or_uncertain"
    return "unknown"


def safe_index(items: list[Any], index: Any) -> Any | None:
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(items):
        return items[idx]
    return None


def text_from_section_item(section: dict[str, Any], item: dict[str, Any]) -> str:
    item_type = item.get("type")
    if item_type == "paragraph":
        paragraph = safe_index(section.get("paragraphs", []), item.get("paragraph_index"))
        if not paragraph:
            return ""
        return paragraph.get("text", "")
    if item_type == "bullet":
        bullet = safe_index(section.get("bullets", []), item.get("bullet_index"))
        if not bullet:
            return ""
        return bullet.get("text", "")
    if item_type == "field":
        field_item = safe_index(section.get("fields", []), item.get("field_index"))
        if not field_item:
            return ""
        return f"{field_item.get('name', '')}: {field_item.get('value', '')}"
    if item_type == "table_row":
        tables = section.get("tables", [])
        table = safe_index(tables, item.get("table_index"))
        if not table:
            return ""
        row = safe_index(table.get("rows", []), item.get("row_index"))
        if not row:
            return ""
        return " | ".join(row.get("cells", []))
    return ""


def section_text(section: dict[str, Any]) -> str:
    chunks: list[str] = []
    title = normalize_text(section.get("title", ""))
    if title and title != "Document start":
        chunks.append(title)

    items = section.get("items", [])
    if items:
        for item in items:
            text = text_from_section_item(section, item)
            if text:
                chunks.append(text)
    else:  # Backward-compatible fallback.
        chunks.extend(paragraph["text"] for paragraph in section.get("paragraphs", []))
        chunks.extend(bullet["text"] for bullet in section.get("bullets", []))
        chunks.extend(f"{field['name']}: {field['value']}" for field in section.get("fields", []))
        for table in section.get("tables", []):
            for row in table.get("rows", []):
                chunks.append(" | ".join(row.get("cells", [])))
    return "\n".join(chunks)


def close_open_table(lines: list[str], indent: str, open_table_index: int | None) -> int | None:
    if open_table_index is not None:
        lines.append(f"{indent}</table>")
    return None


def build_document_context(document: dict[str, Any]) -> str:
    """Generate a safe XML-like block for prompt injection."""

    meta = document.get("metadata", {})
    quality = document.get("extraction_quality", {})
    attrs = {
        "id": document.get("document_id", "unknown"),
        "pages": str(meta.get("page_count", "unknown")),
        "language": document.get("language_guess", "unknown"),
        "born_digital": str(quality.get("born_digital_likely", False)).lower(),
    }
    attr_text = " ".join(f'{key}="{escape(value, quote=True)}"' for key, value in attrs.items())
    lines = [f"<document {attr_text}>"]
    lines.append("  <metadata>")
    lines.append(escape(json.dumps(meta, ensure_ascii=False, sort_keys=True), quote=False))
    lines.append("  </metadata>")
    lines.append("  <extraction_quality>")
    lines.append(escape(json.dumps(quality, ensure_ascii=False, sort_keys=True), quote=False))
    lines.append("  </extraction_quality>")
    lines.append("  <sections>")

    for section in document.get("sections", []):
        section_attrs = (
            f'id="{escape(section.get("section_id", ""), quote=True)}" '
            f'page_start="{section.get("page_start", "")}" '
            f'page_end="{section.get("page_end", "")}" '
            f'title="{escape(section.get("title", ""), quote=True)}"'
        )
        lines.append(f"    <section {section_attrs}>")
        items = section.get("items", [])
        open_table_index: int | None = None

        if items:
            for item in items:
                item_type = item.get("type")
                if item_type != "table_row":
                    open_table_index = close_open_table(lines, "      ", open_table_index)

                if item_type == "paragraph":
                    paragraph = safe_index(section.get("paragraphs", []), item.get("paragraph_index"))
                    if not paragraph:
                        continue
                    lines.append(f"      <p>{escape(paragraph.get('text', ''), quote=False)}</p>")
                elif item_type == "bullet":
                    bullet = safe_index(section.get("bullets", []), item.get("bullet_index"))
                    if not bullet:
                        continue
                    lines.append(
                        f"      <bullet page=\"{bullet.get('page', '')}\">"
                        f"{escape(bullet.get('text', ''), quote=False)}</bullet>"
                    )
                elif item_type == "field":
                    field_item = safe_index(section.get("fields", []), item.get("field_index"))
                    if not field_item:
                        continue
                    lines.append(
                        f"      <field name=\"{escape(field_item.get('name', ''), quote=True)}\" "
                        f"page=\"{field_item.get('page', '')}\">"
                        f"{escape(field_item.get('value', ''), quote=False)}</field>"
                    )
                elif item_type == "table_row":
                    table_index = item.get("table_index", -1)
                    row_index = item.get("row_index", -1)
                    table = safe_index(section.get("tables", []), table_index)
                    if not table:
                        continue
                    if open_table_index != table_index:
                        open_table_index = close_open_table(lines, "      ", open_table_index)
                        lines.append(
                            f"      <table id=\"{escape(table.get('table_id', ''), quote=True)}\" "
                            f"page_start=\"{table.get('page_start', '')}\" "
                            f"page_end=\"{table.get('page_end', '')}\">"
                        )
                        open_table_index = table_index
                    row = safe_index(table.get("rows", []), row_index)
                    if not row:
                        continue
                    lines.append(f"        <row page=\"{row.get('page', '')}\">")
                    for cell in row.get("cells", []):
                        lines.append(f"          <cell>{escape(cell, quote=False)}</cell>")
                    lines.append("        </row>")
            open_table_index = close_open_table(lines, "      ", open_table_index)
        else:  # Backward-compatible fallback for documents created by older code.
            for paragraph in section.get("paragraphs", []):
                lines.append(f"      <p>{escape(paragraph.get('text', ''), quote=False)}</p>")
            for bullet in section.get("bullets", []):
                lines.append(
                    f"      <bullet page=\"{bullet.get('page', '')}\">"
                    f"{escape(bullet.get('text', ''), quote=False)}</bullet>"
                )
            for field_item in section.get("fields", []):
                lines.append(
                    f"      <field name=\"{escape(field_item.get('name', ''), quote=True)}\" "
                    f"page=\"{field_item.get('page', '')}\">{escape(field_item.get('value', ''), quote=False)}</field>"
                )
            for table in section.get("tables", []):
                lines.append(
                    f"      <table id=\"{escape(table.get('table_id', ''), quote=True)}\" "
                    f"page_start=\"{table.get('page_start', '')}\" page_end=\"{table.get('page_end', '')}\">"
                )
                for row in table.get("rows", []):
                    lines.append(f"        <row page=\"{row.get('page', '')}\">")
                    for cell in row.get("cells", []):
                        lines.append(f"          <cell>{escape(cell, quote=False)}</cell>")
                    lines.append("        </row>")
                lines.append("      </table>")

        lines.append("    </section>")
    lines.append("  </sections>")
    lines.append("</document>")
    return "\n".join(lines)


def validate_input_pdf_path(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise SystemExit(f"No existe el archivo de entrada: {pdf_path}")
    if not pdf_path.is_file():
        raise SystemExit(f"La ruta de entrada no es un archivo: {pdf_path}")


def open_pdf_document(pdf_path: Path, config: ExtractionConfig) -> tuple[fitz.Document, dict[str, Any]]:
    validate_input_pdf_path(pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise SystemExit(f"No se pudo abrir el PDF '{pdf_path}': {exc}") from exc

    doc_state = {
        "is_pdf": bool(getattr(doc, "is_pdf", True)),
        "needs_pass": bool(getattr(doc, "needs_pass", False)),
        "authenticated": False,
    }

    if not doc_state["is_pdf"]:
        doc.close()
        raise SystemExit(f"El archivo no parece ser un PDF valido: {pdf_path}")

    if doc_state["needs_pass"]:
        if not config.password:
            doc.close()
            raise SystemExit(
                "El PDF esta cifrado. Proporciona la contrasena con --password."
            )
        try:
            authenticated = bool(doc.authenticate(config.password))
        except Exception as exc:
            doc.close()
            raise SystemExit(f"No se pudo autenticar el PDF cifrado: {exc}") from exc
        if not authenticated:
            doc.close()
            raise SystemExit("La contrasena proporcionada para el PDF cifrado no es valida.")
        doc_state["authenticated"] = True

    return doc, doc_state


def inspect_and_extract(pdf_path: Path, config: ExtractionConfig) -> dict[str, Any]:
    """Run the full extraction pipeline."""

    all_lines: list[LineItem] = []
    pages: dict[int, PageInfo] = {}

    doc, doc_state = open_pdf_document(pdf_path, config)
    try:
        metadata = dict(doc.metadata or {})
        metadata.update(
            {
                "source_file": pdf_path.name,
                "page_count": doc.page_count,
                "is_pdf": doc_state["is_pdf"],
                "needs_pass": doc_state["needs_pass"],
                "authenticated": doc_state["authenticated"],
            }
        )

        for index, page in enumerate(doc, start=1):
            lines, page_info = extract_lines_from_page(page, index, config)
            pages[index] = page_info
            all_lines.extend(lines)
    finally:
        doc.close()

    for page_number, page_info in pages.items():
        page_lines = [line for line in all_lines if line.page == page_number]
        detect_columns_for_page(page_lines, page_info, config)

    removed_header_footer_examples = mark_repeated_headers_and_footers(all_lines, pages, config)
    ordered_lines = reading_order(all_lines, pages)
    sections = segment_lines_into_sections(ordered_lines)
    full_text = "\n".join(section_text(section) for section in sections)

    pages_with_text = sum(1 for page in pages.values() if page.raw_text_chars >= config.min_chars_for_text_layer)
    total_pages = max(1, len(pages))
    alerts = [alert for page in pages.values() for alert in page.alerts]
    if pages and pages_with_text / total_pages < 0.80:
        alerts.append("Posible documento escaneado o con capa de texto pobre; conviene validar OCR.")
    if not pages:
        alerts.append("El PDF no contiene paginas.")

    extraction_quality = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "PyMuPDF",
        "tool_version": getattr(fitz, "version", None),
        "total_pages": len(pages),
        "pages_with_sufficient_text_layer": pages_with_text,
        "text_layer_ratio": round(pages_with_text / total_pages, 3),
        "born_digital_likely": bool(pages and pages_with_text / total_pages >= 0.80),
        "total_extracted_lines": len(all_lines),
        "removed_repeated_header_footer_examples": removed_header_footer_examples,
        "layout_counts": dict(Counter(page.layout for page in pages.values())),
        "ocr_pages": [page.page for page in pages.values() if page.ocr_used],
        "alerts": alerts,
        "limitations": [
            "La deteccion de columnas es heuristica y se basa en separaciones horizontales de lineas.",
            "La deteccion de tablas es heuristica y no reconstruye celdas fusionadas.",
            "La segmentacion por secciones usa senales tipograficas y expresiones regulares; requiere validacion manual para evaluacion final.",
        ],
    }

    return {
        "document_id": pdf_path.stem,
        "metadata": metadata,
        "language_guess": guess_language(full_text),
        "pages": [asdict(pages[number]) for number in sorted(pages)],
        "lines": [asdict(line) for line in ordered_lines if not line.removed],
        "removed_lines": [asdict(line) for line in ordered_lines if line.removed],
        "sections": sections,
        "extraction_quality": extraction_quality,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=False)
        file.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae contratos PDF a JSON intermedio y contexto XML-like para LLMs."
    )
    parser.add_argument("pdf", type=Path, nargs="?", help="Ruta del contrato PDF de entrada.")
    parser.add_argument("--out_json", type=Path, default=None, help="Ruta del JSON intermedio de salida.")
    parser.add_argument("--out_context", type=Path, default=None, help="Ruta del contexto XML-like de salida.")
    parser.add_argument(
        "--use_ocr_when_needed",
        action="store_true",
        help="Intenta OCR con PyMuPDF/Tesseract en paginas con poca capa de texto.",
    )
    parser.add_argument(
        "--ocr_language",
        default="eng+spa",
        help="Idiomas de Tesseract para OCR; por defecto eng+spa.",
    )
    parser.add_argument(
        "--min_chars_for_text_layer",
        type=int,
        default=80,
        help="Minimo de caracteres por pagina para considerar suficiente la capa de texto.",
    )
    parser.add_argument(
        "--password",
        default="",
        help="Contrasena del PDF, si esta cifrado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractionConfig(
        use_ocr_when_needed=args.use_ocr_when_needed,
        ocr_language=args.ocr_language,
        min_chars_for_text_layer=args.min_chars_for_text_layer,
        password=args.password,
    )

    _script_dir = Path(__file__).resolve().parent
    _default_input_dir = _script_dir / "Plantillas"
    _default_output_dir = _script_dir / "JSON_Docs"

    pdf = getattr(args, "pdf", None)
    out_json = getattr(args, "out_json", None)
    out_context = getattr(args, "out_context", None)

    if pdf is None:
        print(f"Carpeta de entrada por defecto: {_default_input_dir}")
        in_path_str = input("Introduce el nombre o ruta del archivo PDF a procesar: ").strip()
        if not in_path_str:
            raise SystemExit("Error: No se proporcionó un archivo de entrada.")
        candidate = Path(in_path_str)
        if candidate.is_absolute():
            pdf = candidate
        else:
            pdf = _default_input_dir / candidate
        if not pdf.exists() and not pdf.suffix:
            pdf = pdf.with_suffix(".pdf")

    if not pdf.exists():
        raise SystemExit(f"Error: No se encontró el archivo PDF: {pdf}")

    if not out_json:
        _default_output_dir.mkdir(parents=True, exist_ok=True)
        default_out = _default_output_dir / pdf.with_suffix(".json").name
        if not default_out.exists():
            out_json = default_out
        else:
            print(f"Aviso: Ya existe un archivo con ese nombre: {default_out}")
            out_path_str = input("Introduce un nombre alternativo para el JSON de salida: ").strip()
            if not out_path_str:
                raise SystemExit("Error: No se proporcionó un nombre alternativo.")
            candidate_out = Path(out_path_str)
            if not candidate_out.suffix:
                candidate_out = candidate_out.with_suffix(".json")
            if candidate_out.is_absolute():
                out_json = candidate_out
            else:
                out_json = _default_output_dir / candidate_out

    if not out_context:
        _default_output_dir.mkdir(parents=True, exist_ok=True)
        out_context = _default_output_dir / pdf.with_suffix(".txt").name

    document = inspect_and_extract(pdf, config)
    context = build_document_context(document)
    write_json(out_json, document)
    write_text(out_context, context)


if __name__ == "__main__":
    main()
