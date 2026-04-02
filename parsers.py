"""
PDF parsers for UWorld Step 1 performance data.

PDF types handled:
  - Summary PDF (uworlddata.pdf): overall stats, percentile, usage
  - Breakdown PDFs (uworlddata3, uworlddata4): subject/system/topic tables (USAGE/P-RANK format)
  - Granular PDF (uworlddata2): topic-level data (USED Q / TOTAL Q format, no P-RANK)
  - Test result PDFs: per-question results from a specific test
  - Question PDFs: individual question + explanation
"""
import re
from collections import defaultdict
from typing import Optional, Tuple, List

import pdfplumber
import pandas as pd


# ─── Summary PDF (uworlddata.pdf style) ──────────────────────────────────────

def parse_summary_pdf(pdf_path: str) -> dict:
    """Parse the overall summary stats PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    patterns = {
        "total_correct":          r"Total Correct\s+(\d+)",
        "total_incorrect":        r"Total Incorrect\s+(\d+)",
        "total_omitted":          r"Total Omitted\s+(\d+)",
        "used_questions":         r"Used Questions\s+(\d+)",
        "unused_questions":       r"Unused Questions\s+(\d+)",
        "total_questions":        r"Total Questions\s+(\d+)",
        "tests_created":          r"Tests Created\s+(\d+)",
        "tests_completed":        r"Tests Completed\s+(\d+)",
        "your_score_pct":         r"Your Score \(\d+\w+ rank\)\s+(\d+)%",
        "median_score_pct":       r"Median Score \(\d+\w+ rank\)\s+(\d+)%",
        "your_percentile":        r"Your Score \((\d+)\w+ rank\)",
        "your_avg_time":          r"Your Average Time Spent \(sec\)\s+(\d+)",
        "others_avg_time":        r"Other.s Average Time Spent \(sec\)\s+(\d+)",
        "correct_to_incorrect":   r"Correct to Incorrect\s+(\d+)",
        "incorrect_to_correct":   r"Incorrect to Correct\s+(\d+)",
        "incorrect_to_incorrect": r"Incorrect to Incorrect\s+(\d+)",
    }

    stats = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            stats[key] = int(m.group(1))

    name_m = re.search(r"^([A-Za-z][\w ]+)\nUserId", text, re.MULTILINE)
    if name_m:
        stats["user_name"] = name_m.group(1).strip()

    return stats


# ─── Breakdown PDFs (position-based parser) ───────────────────────────────────

# x-coordinate threshold: words with x < NAME_THRESH belong to the name column
_NAME_THRESH = 275

# Words to skip (headers, footers, metadata)
_SKIP_WORDS = {
    "NAME", "USAGE", "CORRECT", "INCORRECT", "OMITTED", "Q", "P-RANK",
    "about:blank", "UserId", ":", "Karina", "Dalal", "3/20/26,",
    "3/21/26,", "PM", "AM", "USED", "TOTAL",
}

# Data row patterns
# Pattern A: "8/254 4 (50%) 4 (50%) 0 (0%) 18th"  ← USAGE fraction + P-RANK
_DATA_A = re.compile(
    r"(\d+)/(\d+)\s+"
    r"(\d+)\s+\((\d+)%\)\s+"
    r"(\d+)\s+\((\d+)%\)\s+"
    r"(\d+)\s+\((\d+)%\)\s*"
    r"(\d+\w+|-)?\s*$"
)
# Pattern B: "16 25 5 (31%) 11 (69%) 0 (0%)"  ← USED Q / TOTAL Q (no fraction, no P-RANK)
_DATA_B = re.compile(
    r"(\d+)\s+(\d+)\s+"
    r"(\d+)\s+\((\d+)%\)\s+"
    r"(\d+)\s+\((\d+)%\)\s+"
    r"(\d+)\s+\((\d+)%\)\s*$"
)


def _parse_data(name: str, data_text: str) -> Optional[dict]:
    """Try to parse a data row from name + data column text."""
    name = name.strip()
    if not name:
        return None

    m = _DATA_A.match(data_text)
    if m:
        prank = m.group(9) or "-"
        return {
            "name":          name,
            "used":          int(m.group(1)),
            "total":         int(m.group(2)),
            "correct":       int(m.group(3)),
            "correct_pct":   int(m.group(4)),
            "incorrect":     int(m.group(5)),
            "incorrect_pct": int(m.group(6)),
            "omitted":       int(m.group(7)),
            "omitted_pct":   int(m.group(8)),
            "prank":         prank if prank != "-" else None,
            "has_prank":     prank != "-",
        }

    m = _DATA_B.match(data_text)
    if m:
        return {
            "name":          name,
            "used":          int(m.group(1)),
            "total":         int(m.group(2)),
            "correct":       int(m.group(3)),
            "correct_pct":   int(m.group(4)),
            "incorrect":     int(m.group(5)),
            "incorrect_pct": int(m.group(6)),
            "omitted":       int(m.group(7)),
            "omitted_pct":   int(m.group(8)),
            "prank":         None,
            "has_prank":     False,
        }

    return None


def _parse_breakdown_page(page) -> list:
    """
    Parse one page of a breakdown PDF using word x/y positions.

    Strategy:
    - Name column: x < 275
    - Data column: x >= 275
    - Data words are grouped by y-position (data rows).
    - For each data row at y=Yd, the name is assembled from all name-column
      words within ±20px of Yd (handles names that wrap above/below the data).
    """
    words = page.extract_words(keep_blank_chars=False)
    if not words:
        return []

    name_col = [(w["text"], w["x0"], w["top"]) for w in words
                if w["x0"] < _NAME_THRESH and w["text"] not in _SKIP_WORDS
                and not re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}", w["text"])
                and not re.match(r"^\d{1,2}:\d{2}$", w["text"])]

    data_col = [(w["text"], w["x0"], w["top"]) for w in words
                if w["x0"] >= _NAME_THRESH and w["text"] not in _SKIP_WORDS
                and not re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}", w["text"])
                and not re.match(r"^\d{1,2}:\d{2}$", w["text"])]

    # Group data words by y-position (snap to 5px grid)
    data_rows: dict = defaultdict(list)
    for text, x, y in data_col:
        y_key = round(y / 5) * 5
        data_rows[y_key].append((text, x))

    results = []
    for y_key in sorted(data_rows):
        # Build data string (sorted left to right)
        data_words_sorted = sorted(data_rows[y_key], key=lambda w: w[1])
        data_text = " ".join(w[0] for w in data_words_sorted)

        # Gather name words within ±20px
        nearby = [(t, x, y) for t, x, y in name_col if abs(y - y_key) <= 20]
        nearby.sort(key=lambda w: (w[2], w[1]))  # by y then x
        name_text = " ".join(t for t, x, y in nearby)

        row = _parse_data(name_text, data_text)
        if row:
            results.append(row)

    return results


def parse_breakdown_pdf(pdf_path: str) -> pd.DataFrame:
    """Parse a subject/system breakdown PDF into a DataFrame."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            rows.extend(_parse_breakdown_page(page))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_breakdown_pdfs(pdf_paths: List[str]) -> pd.DataFrame:
    """Load and deduplicate multiple breakdown PDFs."""
    dfs = []
    for path in pdf_paths:
        try:
            df = parse_breakdown_pdf(path)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    return combined.drop_duplicates(subset=["name", "used"]).reset_index(drop=True)


# ─── Test Result PDFs ─────────────────────────────────────────────────────────

_CHECK_CHARS = set("✓✔√")
_CROSS_CHARS = set("✗✘×")


def parse_test_result_pdf(pdf_path: str) -> Tuple[dict, pd.DataFrame]:
    """
    Parse a UWorld test result PDF.
    Returns (header_dict, questions_df).
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
        all_tables = [page.extract_tables() for page in pdf.pages]

    text = "\n".join(pages_text)

    # ── Header ────────────────────────────────────────────────────────────────
    header = {}
    m = re.search(r"Custom Test Id[:\s]+(\d+)", text)
    if m:
        header["test_id"] = m.group(1)

    m = re.search(r"Your Score\s+(\d+)%", text)
    if m:
        header["score_pct"] = int(m.group(1))
    else:
        m = re.search(r"(\d+)%\s*\nAvg[:\s]+(\d+)%", text)
        if m:
            header["score_pct"] = int(m.group(1))
            header["avg_pct"] = int(m.group(2))

    m = re.search(r"Avg[:\s]+(\d+)%", text)
    if m:
        header["avg_pct"] = int(m.group(1))

    m = re.search(r"Mode\s+(Tutored|Timed|Untimed)", text)
    if m:
        header["mode"] = m.group(1)

    # ── Questions via table extraction ────────────────────────────────────────
    questions = []

    for page_tables in all_tables:
        for table in (page_tables or []):
            for row in table:
                if not row or len(row) < 7:
                    continue
                first = str(row[0] or "").strip()
                if not first or first[0] not in (_CHECK_CHARS | _CROSS_CHARS):
                    continue
                is_correct = first[0] in _CHECK_CHARS

                try:
                    id_cell = str(row[1] or "").strip()
                    id_m = re.match(r"(\d+)\s*[-–]\s*(\d+)", id_cell)
                    if not id_m:
                        continue

                    def _int(cell):
                        m = re.search(r"(\d+)", str(cell or ""))
                        return int(m.group(1)) if m else 0

                    questions.append({
                        "correct":            is_correct,
                        "position":           int(id_m.group(1)),
                        "question_id":        int(id_m.group(2)),
                        "subject":            str(row[2] or "").strip(),
                        "system":             str(row[3] or "").strip(),
                        "category":           str(row[4] or "").strip(),
                        "topic":              str(row[5] or "").strip(),
                        "pct_correct_others": _int(row[6]),
                        "time_spent_sec":     _int(row[7]) if len(row) > 7 else 0,
                        "avg_time_spent_sec": _int(row[8]) if len(row) > 8 else 0,
                    })
                except (ValueError, IndexError):
                    continue

    # ── Fallback: regex on raw text ───────────────────────────────────────────
    if not questions:
        for line in text.split("\n"):
            line = line.strip()
            if not line or line[0] not in (_CHECK_CHARS | _CROSS_CHARS):
                continue
            is_correct = line[0] in _CHECK_CHARS
            rest = line[1:].strip()

            m = re.match(
                r"(\d+)\s*[-–]\s*(\d+)\s+"
                r"(\w[\w\s&,/]+?)\s{2,}"
                r"(.+?)\s{2,}"
                r"(.+?)\s{2,}"
                r"(.+?)\s{2,}"
                r"(\d+)%\s+"
                r"(\d+)\s*sec\s+"
                r"(\d+)\s*sec",
                rest,
            )
            if m:
                questions.append({
                    "correct":            is_correct,
                    "position":           int(m.group(1)),
                    "question_id":        int(m.group(2)),
                    "subject":            m.group(3).strip(),
                    "system":             m.group(4).strip(),
                    "category":           m.group(5).strip(),
                    "topic":              m.group(6).strip(),
                    "pct_correct_others": int(m.group(7)),
                    "time_spent_sec":     int(m.group(8)),
                    "avg_time_spent_sec": int(m.group(9)),
                })

    return header, pd.DataFrame(questions)


# ─── Individual Question PDFs ─────────────────────────────────────────────────

def parse_question_pdf(pdf_path: str) -> dict:
    """Parse a UWorld individual question + explanation PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    result = {"raw_text": text}

    m = re.search(r"Question Id[:\s]+(\d+)", text)
    if m:
        result["question_id"] = m.group(1)

    result["is_correct"] = "Incorrect" not in text

    m = re.search(r"Correct answer\s*\n?\s*([A-E])", text)
    if m:
        result["correct_answer"] = m.group(1)

    m = re.search(r"Explanation\s*\n(.*?)(?:Educational objective|$)", text, re.DOTALL)
    if m:
        result["explanation"] = m.group(1).strip()[:2000]

    m = re.search(r"Educational objective[:\s]*\n?(.*?)(?:\n\n|$)", text, re.DOTALL)
    if m:
        result["educational_objective"] = m.group(1).strip()[:500]

    return result
