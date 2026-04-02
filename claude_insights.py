"""
Gemini API integration for UWorld performance analysis.
"""
import google.generativeai as genai
import pandas as pd
from typing import Optional
import PIL.Image
import io


def _client(api_key: Optional[str]):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash-lite")


def _ask(model, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text


# ─── Image-based analysis ────────────────────────────────────────────────────

def analyze_image(image_bytes: bytes, api_key: Optional[str] = None) -> str:
    model = _client(api_key)
    img = PIL.Image.open(io.BytesIO(image_bytes))

    prompt = """You are a USMLE Step 1 expert tutor. A student has uploaded a screenshot of a UWorld question and/or explanation.

Look at the image carefully and provide:
1. **Core concept tested** (1 sentence — what's the high-yield fact?)
2. **Why students pick the wrong answer** — what's the common trap in this vignette?
3. **The buzzword/pattern to memorize** for similar future questions
4. **Mnemonic or memory hook** (2-3 sentences max)
5. **How Step 1 might vary this question** — alternate presentations to watch for

Be concise and punchy. Focus on retention."""

    response = model.generate_content([prompt, img])
    return response.text


def analyze_test_image(image_bytes: bytes, api_key: Optional[str] = None) -> str:
    model = _client(api_key)
    img = PIL.Image.open(io.BytesIO(image_bytes))

    prompt = """You are a USMLE Step 1 expert tutor. A student has uploaded a screenshot of their UWorld test results.

Look at the image carefully and provide:
1. **Pattern diagnosis**: what common threads link the wrong answers (topic clusters, concept gaps)?
2. **For each wrong topic visible**: the exact concept to review + a one-line First Aid / Pathoma cross-reference
3. **Timing feedback**: based on any timing data visible, is the student rushing or overthinking?
4. **3 Anki search terms** the student should look up to create cards from this test
5. **One encouraging but honest sentence** to close

Be specific, reference actual topic names visible in the image, and keep it concise."""

    response = model.generate_content([prompt, img])
    return response.text


# ─── Overall study plan ───────────────────────────────────────────────────────

def generate_study_plan(
    summary: dict,
    breakdown: pd.DataFrame,
    api_key: Optional[str] = None,
) -> str:
    model = _client(api_key)

    score = summary.get("your_score_pct", "?")
    percentile = summary.get("your_percentile", "?")
    median = summary.get("median_score_pct", 63)
    used = summary.get("used_questions", 0)
    total = summary.get("total_questions", 3659)
    your_time = summary.get("your_avg_time", "?")
    others_time = summary.get("others_avg_time", 63)
    c2i = summary.get("correct_to_incorrect", 0)
    i2c = summary.get("incorrect_to_correct", 0)
    i2i = summary.get("incorrect_to_incorrect", 0)

    # Top-level subjects (those with a P-RANK)
    subjects_df = breakdown[breakdown["has_prank"]].sort_values("correct_pct")
    subjects_str = subjects_df[["name", "used", "total", "correct_pct", "prank"]].to_string(index=False)

    # Worst individual topics (no P-RANK = granular category rows)
    categories_df = breakdown[~breakdown["has_prank"] & (breakdown["used"] >= 3)]
    worst_str = (
        categories_df.nlargest(15, "incorrect_pct")[
            ["name", "used", "total", "correct_pct", "incorrect_pct"]
        ].to_string(index=False)
        if not categories_df.empty
        else "No category-level data available."
    )

    prompt = f"""You are a USMLE Step 1 expert tutor. A student has shared their UWorld QBank performance data with you.

OVERALL STATS:
- Score: {score}% (you are at the {percentile}th percentile; class median is {median}%)
- QBank used: {used}/{total} questions ({round(used*100/max(total,1))}% complete)
- Average time per question: {your_time}s (class average: {others_time}s)
- Answer changes: Correct→Wrong={c2i}, Wrong→Correct={i2c}, Wrong→Still Wrong={i2i}

PERFORMANCE BY SUBJECT/SYSTEM (sorted worst → best):
{subjects_str}

TOP 15 WORST CATEGORIES (min 3 questions attempted):
{worst_str}

Please give:
1. **Honest 2-sentence assessment** of where this student stands and the urgency
2. **Top 5 priority topics to tackle NOW** — ranked by (impact × weakness), with one specific reason each
3. **Concrete weekly plan**: how many UWorld Qs/day, in what subject order, and how to use First Aid alongside
4. **Strategy for the 3 worst systems** — what specific subtopics to hit and any high-yield mnemonics
5. **One key behavioral insight** from the answer-change data and time-per-question stats

Be direct, specific, and use the actual topic names from the data. No fluff."""

    return _ask(model, prompt)


# ─── Test-specific analysis ───────────────────────────────────────────────────

def analyze_test(
    header: dict,
    questions: pd.DataFrame,
    api_key: Optional[str] = None,
) -> str:
    model = _client(api_key)

    wrong = questions[~questions["correct"]]
    right = questions[questions["correct"]]

    # Questions where student was >30% slower than class average
    slow = questions[
        questions["avg_time_spent_sec"] > 0
        & (questions["time_spent_sec"] > questions["avg_time_spent_sec"] * 1.3)
    ]

    wrong_str = (
        wrong[["subject", "system", "category", "topic", "pct_correct_others", "time_spent_sec"]]
        .to_string(index=False)
        if not wrong.empty
        else "All correct!"
    )

    slow_str = (
        slow[["topic", "time_spent_sec", "avg_time_spent_sec"]].to_string(index=False)
        if not slow.empty
        else "No significant timing issues."
    )

    prompt = f"""You are a USMLE Step 1 expert tutor reviewing a student's UWorld test.

TEST SUMMARY:
- Test ID: {header.get("test_id", "?")}
- Score: {header.get("score_pct", "?")}%  (class avg: {header.get("avg_pct", "?")}%)
- Mode: {header.get("mode", "?")}
- Results: {len(right)} correct, {len(wrong)} incorrect out of {len(questions)} total

INCORRECT QUESTIONS (topic | % others got right | student's time):
{wrong_str}

QUESTIONS WHERE STUDENT WAS SIGNIFICANTLY SLOWER THAN CLASS:
{slow_str}

Please give:
1. **Pattern diagnosis**: what common threads link the wrong answers (topic clusters, concept gaps)?
2. **For each wrong topic**: the exact concept to review + a one-line First Aid / Pathoma cross-reference
3. **Timing feedback**: is the student rushing or overthinking? What to adjust?
4. **3 Anki search terms** the student should look up to create cards from this test
5. **One encouraging but honest sentence** to close

Be specific, reference actual topic names, and keep it concise."""

    return _ask(model, prompt)


# ─── Individual question analysis ────────────────────────────────────────────

def analyze_question(question_data: dict, api_key: Optional[str] = None) -> str:
    model = _client(api_key)

    raw = question_data.get("raw_text", "")[:4000]
    status = "INCORRECT" if not question_data.get("is_correct", True) else "CORRECT"
    correct_ans = question_data.get("correct_answer", "?")

    prompt = f"""You are a USMLE Step 1 expert tutor. A student is reviewing a UWorld question they got {status}.
The correct answer was {correct_ans}.

Here is the full question + explanation text:
---
{raw}
---

Please give:
1. **Core concept tested** (1 sentence — what's the high-yield fact?)
2. **Why students pick the wrong answer** — what's the common trap in this vignette?
3. **The buzzword/pattern to memorize** for similar future questions
4. **Mnemonic or memory hook** (2-3 sentences max)
5. **How Step 1 might vary this question** — alternate presentations to watch for

Be concise and punchy. Focus on retention."""

    return _ask(model, prompt)
