"""
UWorld Step 1 Insights — Streamlit app

Run with:
    streamlit run app.py
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from parsers import (
    load_breakdown_pdfs,
    parse_breakdown_pdf,
    parse_question_pdf,
    parse_summary_pdf,
    parse_test_result_pdf,
)
from claude_insights import analyze_question, analyze_test, generate_study_plan

load_dotenv()

st.set_page_config(
    page_title="UWorld Step 1 Insights",
    page_icon="🩺",
    layout="wide",
)

# ─── Bundled PDF paths ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
BUNDLED_SUMMARY = DATA_DIR / "uworlddata.pdf"
BUNDLED_BREAKDOWNS = [
    DATA_DIR / "uworlddata3.pdf",
    DATA_DIR / "uworlddata4.pdf",
    DATA_DIR / "uworlddata2.pdf",
]


# ─── Auto-load bundled data on first run ─────────────────────────────────────
if "summary" not in st.session_state and BUNDLED_SUMMARY.exists():
    try:
        st.session_state["summary"] = parse_summary_pdf(str(BUNDLED_SUMMARY))
    except Exception:
        pass

if "breakdown" not in st.session_state:
    paths = [str(p) for p in BUNDLED_BREAKDOWNS if p.exists()]
    if paths:
        try:
            st.session_state["breakdown"] = load_breakdown_pdfs(paths)
        except Exception:
            pass


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 UWorld Insights")
    st.divider()

    # Load API key from Streamlit secrets or env var (never shown to user)
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

    st.subheader("Upload Custom Data")
    st.caption("Default data is pre-loaded. Upload here to replace it.")

    summary_file = st.file_uploader("Summary PDF", type="pdf", key="su")
    if summary_file:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(summary_file.read())
            try:
                st.session_state["summary"] = parse_summary_pdf(tf.name)
                st.success("Summary loaded!")
            except Exception as e:
                st.error(str(e))

    breakdown_files = st.file_uploader(
        "Breakdown PDFs", type="pdf", accept_multiple_files=True, key="bu"
    )
    if breakdown_files:
        paths = []
        for f in breakdown_files:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tf.write(f.read())
                paths.append(tf.name)
        try:
            st.session_state["breakdown"] = load_breakdown_pdfs(paths)
            st.success(f"Loaded {len(breakdown_files)} breakdown PDF(s)!")
        except Exception as e:
            st.error(str(e))


# ─── Gate: require data ───────────────────────────────────────────────────────
summary: dict = st.session_state.get("summary", {})
breakdown: pd.DataFrame = st.session_state.get("breakdown", pd.DataFrame())

if not summary and breakdown.empty:
    st.title("UWorld Step 1 Insights")
    st.info("No data could be loaded. Upload your UWorld PDF exports in the sidebar.")
    st.stop()


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📚 Subjects", "🤖 AI Study Plan", "📝 Test Review", "🔍 Question"]
)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Overall Performance")

    if not summary:
        st.info("Load your summary PDF to see overall stats.")
        st.stop()

    score      = summary.get("your_score_pct", 0)
    percentile = summary.get("your_percentile", 0)
    median     = summary.get("median_score_pct", 63)
    used       = summary.get("used_questions", 0)
    total      = summary.get("total_questions", 3659)
    your_time  = summary.get("your_avg_time", 0)
    other_time = summary.get("others_avg_time", 63)
    usage_pct  = round(used * 100 / total) if total else 0

    # ── Key metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Your Score",     f"{score}%")
    c2.metric("Percentile",     f"{percentile}th", help="Among all UWorld users")
    c3.metric("QBank Used",     f"{usage_pct}%",  f"{used}/{total} Qs")
    c4.metric("Avg Time/Q",     f"{your_time}s",  f"{your_time - other_time:+d}s vs others")

    # ── Time warning ──────────────────────────────────────────────────────────
    if your_time > other_time + 20:
        st.warning(
            f"⏱️ You're spending **{your_time - other_time}s more per question** than average. "
            "This can hurt on timed exams — practice reading stems faster."
        )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Score Breakdown")
        correct   = summary.get("total_correct", 0)
        incorrect = summary.get("total_incorrect", 0)
        omitted   = summary.get("total_omitted", 0)
        fig = go.Figure(go.Pie(
            labels=["Correct", "Incorrect", "Omitted"],
            values=[correct, incorrect, omitted],
            marker_colors=["#27ae60", "#e74c3c", "#95a5a6"],
            hole=0.45,
            textinfo="label+percent",
        ))
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=0, r=0),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Total: {correct + incorrect + omitted} questions attempted")

    with col_b:
        st.subheader("Answer Changes")
        c2i = summary.get("correct_to_incorrect", 0)
        i2c = summary.get("incorrect_to_correct", 0)
        i2i = summary.get("incorrect_to_incorrect", 0)
        fig = go.Figure(go.Bar(
            x=["Correct → Wrong", "Wrong → Correct", "Wrong → Wrong"],
            y=[c2i, i2c, i2i],
            marker_color=["#e74c3c", "#27ae60", "#f39c12"],
            text=[c2i, i2c, i2i],
            textposition="outside",
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=10), showlegend=False,
                          yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        net = i2c - c2i
        if c2i > i2c:
            st.warning(f"⚠️ Net -{{abs(net)}} from answer changes. **Trust your first instinct more.**")
        elif i2c > c2i:
            st.success(f"✅ Net +{net} from answer changes. Your review instincts are good.")
        else:
            st.info("Answer changes are neutral — about even.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Subject analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Subject & Topic Analysis")

    if breakdown.empty:
        st.info("Load your breakdown PDFs to see subject analysis.")
        st.stop()

    # Subjects = rows that have a P-RANK (top-level groupings)
    subjects = breakdown[breakdown["has_prank"]].copy()
    categories = breakdown[~breakdown["has_prank"]].copy()

    # ── Bar chart: % correct by subject ──────────────────────────────────────
    if not subjects.empty:
        fig = px.bar(
            subjects.sort_values("correct_pct"),
            x="correct_pct",
            y="name",
            orientation="h",
            color="correct_pct",
            color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
            range_color=[0, 100],
            labels={"correct_pct": "% Correct", "name": ""},
            title="Performance by Subject / System",
            text="correct_pct",
        )
        fig.add_vline(x=50, line_dash="dash", line_color="gray",
                      annotation_text="50%", annotation_position="top")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            height=max(350, len(subjects) * 32),
            coloraxis_showscale=False,
            margin=dict(l=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚨 Worst Topics")
        st.caption("Ranked by % incorrect (min 2 questions attempted)")
        worst = (
            categories[categories["used"] >= 2]
            .nlargest(15, "incorrect_pct")
            [["name", "used", "total", "correct_pct", "incorrect_pct"]]
            .copy()
        )
        worst.columns = ["Topic", "Done", "Total", "Correct %", "Incorrect %"]

        def _highlight(row):
            if row["Incorrect %"] >= 70:
                return ["background-color: #fde8e8"] * len(row)
            if row["Incorrect %"] >= 50:
                return ["background-color: #fef9e7"] * len(row)
            return [""] * len(row)

        if not worst.empty:
            st.dataframe(
                worst.style
                     .apply(_highlight, axis=1)
                     .format({"Correct %": "{}%", "Incorrect %": "{}%"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No category-level data available.")

    with col2:
        st.subheader("📊 Coverage Gaps")
        st.caption("Topics with the most unused questions (biggest opportunity areas)")
        gap = breakdown[breakdown["total"] >= 10].copy()
        gap["coverage_pct"] = (gap["used"] / gap["total"] * 100).round(1)
        gap["remaining"] = gap["total"] - gap["used"]
        gap = gap.nsmallest(12, "coverage_pct")[
            ["name", "used", "total", "coverage_pct", "remaining"]
        ]
        gap.columns = ["Topic", "Done", "Total", "Coverage %", "Remaining Qs"]
        st.dataframe(
            gap.style.format({"Coverage %": "{}%"}),
            use_container_width=True,
            hide_index=True,
        )

    # ── Perfect 0% topics ────────────────────────────────────────────────────
    zeros = breakdown[(breakdown["correct_pct"] == 0) & (breakdown["used"] >= 2)]
    if not zeros.empty:
        st.divider()
        st.subheader("🆘 Zero-Correct Topics (0% correct, ≥2 attempted)")
        st.dataframe(
            zeros[["name", "used", "total", "incorrect_pct"]]
                 .rename(columns={"name": "Topic", "used": "Done",
                                  "total": "Total", "incorrect_pct": "Incorrect %"})
                 .style.format({"Incorrect %": "{}%"}),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — AI Study Plan
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("AI-Powered Study Plan")

    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar to use AI features.")
        st.stop()

    if summary or not breakdown.empty:
        st.caption(
            "Gemini will analyze your full performance data and generate"
            "a prioritized, personalized study plan."
        )
        if st.button("Generate Study Plan", type="primary", key="gen_plan"):
            with st.spinner("Gemini is analyzing your data..."):
                try:
                    plan = generate_study_plan(summary, breakdown, api_key)
                    st.session_state["study_plan"] = plan
                except Exception as e:
                    st.error(f"API error: {e}")

        if "study_plan" in st.session_state:
            st.markdown(st.session_state["study_plan"])
    else:
        st.info("Load your PDFs first.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Test Review
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Test Review")
    st.caption(
        "Upload a PDF of a specific UWorld test result. "
        "**To export from UWorld:** open a completed test → Test Results → Print → Save as PDF"
    )

    test_file = st.file_uploader("Upload Test Result PDF", type="pdf", key="test_upload")

    if test_file:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(test_file.read())
            tmp_path = tf.name

        with st.spinner("Parsing test..."):
            try:
                h, qdf = parse_test_result_pdf(tmp_path)
                st.session_state["test_header"] = h
                st.session_state["test_qdf"] = qdf
            except Exception as e:
                st.error(f"Could not parse PDF: {e}")

    if "test_qdf" in st.session_state:
        h   = st.session_state["test_header"]
        qdf = st.session_state["test_qdf"]

        if qdf.empty:
            st.warning(
                "Could not extract question rows from this PDF. "
                "The format may differ — try exporting as PDF from UWorld's Print dialog."
            )
        else:
            # ── Metrics ───────────────────────────────────────────────────────
            n_right = len(qdf[qdf["correct"]])
            n_wrong = len(qdf[~qdf["correct"]])
            score   = h.get("score_pct", round(n_right * 100 / max(len(qdf), 1)))
            avg     = h.get("avg_pct", 61)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score",     f"{score}%",  f"{score - avg:+d}% vs class")
            c2.metric("Correct",   n_right)
            c3.metric("Incorrect", n_wrong)
            c4.metric("Test ID",   h.get("test_id", "—"))

            st.divider()

            # ── Question table ────────────────────────────────────────────────
            display = qdf.copy()
            display["Result"] = display["correct"].map({True: "✓", False: "✗"})
            if "time_spent_sec" in display.columns and "avg_time_spent_sec" in display.columns:
                display["Δ Time"] = (
                    display["time_spent_sec"] - display["avg_time_spent_sec"]
                ).apply(lambda x: f"{x:+d}s")

            cols = ["Result", "position", "question_id", "subject",
                    "system", "category", "topic", "pct_correct_others",
                    "time_spent_sec", "Δ Time"]
            cols = [c for c in cols if c in display.columns]
            rename = {
                "position": "#", "question_id": "Q ID",
                "subject": "Subject", "system": "System",
                "category": "Category", "topic": "Topic",
                "pct_correct_others": "% Others", "time_spent_sec": "Time (s)",
            }

            def _color_result(row):
                if row.get("Result") == "✗":
                    return ["background-color: #fde8e8"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display[cols].rename(columns=rename)
                             .style.apply(_color_result, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # ── AI analysis ───────────────────────────────────────────────────
            if api_key:
                if st.button("Analyze This Test with AI", type="primary", key="ai_test"):
                    with st.spinner("Gemini is reviewing your test..."):
                        try:
                            analysis = analyze_test(h, qdf, api_key)
                            st.session_state["test_analysis"] = analysis
                        except Exception as e:
                            st.error(f"API error: {e}")

                if "test_analysis" in st.session_state:
                    st.divider()
                    st.markdown(st.session_state["test_analysis"])
            else:
                st.info("Add your API key in the sidebar to get AI analysis of this test.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Question Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Question Analysis")
    st.caption(
        "Upload a PDF of an individual UWorld question + explanation. "
        "**To export:** open a question in review mode → Print → Save as PDF"
    )

    q_file = st.file_uploader("Upload Question PDF", type="pdf", key="q_upload")

    if q_file:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(q_file.read())
            tmp_path = tf.name

        with st.spinner("Parsing question..."):
            try:
                q_data = parse_question_pdf(tmp_path)
                st.session_state["q_data"] = q_data
            except Exception as e:
                st.error(f"Could not parse PDF: {e}")

    if "q_data" in st.session_state:
        q_data = st.session_state["q_data"]

        status = "Incorrect" if not q_data.get("is_correct", True) else "Correct"
        st.badge(status, color="red" if status == "Incorrect" else "green")

        if q_data.get("correct_answer"):
            st.write(f"**Correct answer:** {q_data['correct_answer']}")

        if q_data.get("educational_objective"):
            st.info(f"**Educational objective:** {q_data['educational_objective']}")

        with st.expander("Full extracted text"):
            st.text(q_data.get("raw_text", "")[:3000])

        if api_key:
            if st.button("Analyze with Gemini", type="primary", key="ai_q"):
                with st.spinner("Gemini is analyzing..."):
                    try:
                        analysis = analyze_question(q_data, api_key)
                        st.session_state["q_analysis"] = analysis
                    except Exception as e:
                        st.error(f"API error: {e}")

            if "q_analysis" in st.session_state:
                st.divider()
                st.markdown(st.session_state["q_analysis"])
        else:
            st.info("Add your API key in the sidebar to analyze this question.")
