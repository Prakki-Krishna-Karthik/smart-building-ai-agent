"""Production Streamlit presentation for controller-generated reports.

This module is intentionally a view layer: it reads an ``OptimizationResult``
JSON report and its CSV history, transforms them into display data, and renders
charts. It does not run EnergyPlus, call Ollama, mutate IDFs, or make control
decisions.
"""

from __future__ import annotations

from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
import html
import json
from pathlib import Path
import sys
import time
from typing import Any

# ``streamlit run src/dashboard/app.py`` places the script directory first on
# ``sys.path``. Add the repository root so the application can import the
# project's ``src`` package consistently on Windows and Unix-like systems.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config.config import settings
from src.agent.tools import ToolDispatcher, create_default_tool_dispatcher
from src.agent.tool_loop import AgenticDecisionEngine
from src.controllers.optimization_controller import OptimizationController
from src.simulator.energyplus import EnergyPlusRunner
from src.simulator.output_parser import EnergyPlusOutputParser
from src.simulator.stress_test import prepare_stressed_idf
from src.llm.ollama_client import AgentToolSelection, OllamaClient
from src.utils.logging import configure_logging


PLOT_TEMPLATE = "plotly_white"


def _number(value: Any) -> float | None:
    """Return a finite numeric value or ``None`` for missing report data."""
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


def _display(value: Any, suffix: str = "") -> str:
    """Format a report value for a KPI or table."""
    number = _number(value)
    if number is None:
        return "N/A"
    return f"{number:,.2f}{suffix}"


def _compact_energy(value: Any) -> str:
    """Format energy with readable precision without inventing units."""
    number = _number(value)
    if number is None:
        return "Unavailable"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:,.2f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:,.2f}K"
    return f"{number:,.2f}"


def _card(title: str, value: str, icon: str, tone: str = "neutral", detail: str = "") -> str:
    """Return a consistent dashboard card fragment."""
    return (
        f'<div class="metric-card {tone}"><div class="metric-head">'
        f'<span class="metric-icon">{html.escape(icon)}</span>'
        f'<span class="metric-title">{html.escape(title)}</span></div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-detail">{html.escape(detail)}</div></div>'
    )


def _latest_report() -> Path | None:
    """Find the most recent successful controller report.

    A failed run still writes a diagnostic report. It must remain available
    through an explicit path/upload, but should not silently replace the last
    successful dashboard view.
    """
    reports = list(settings.output_directory.glob("optimization_*/optimization_report.json"))
    reports.extend(settings.output_directory.glob("demo_*/optimization_report.json"))
    reports.extend(settings.output_directory.glob("dashboard_*/optimization_report.json"))
    reports.extend(settings.output_directory.glob("latest_e2e_trace*/optimization_report.json"))
    successful = []
    for path in reports:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not report.get("error") and report.get("baseline_energy") is not None:
            successful.append(path)
    return max(successful, key=lambda path: path.stat().st_mtime) if successful else None


@st.cache_data(show_spinner=False)
def _read_report(path: str) -> tuple[dict[str, Any], bytes]:
    """Read a controller JSON report."""
    report_path = Path(path)
    raw = report_path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


@st.cache_data(show_spinner=False)
def _read_csv(path: str | None) -> tuple[pd.DataFrame, bytes | None]:
    """Read a controller CSV summary when available."""
    if not path or not Path(path).is_file():
        return pd.DataFrame(), None
    raw = Path(path).read_bytes()
    return pd.read_csv(Path(path)), raw


def _state(report: dict[str, Any], name: str) -> dict[str, Any]:
    """Get a serialized BuildingState from an OptimizationResult report."""
    value = report.get(name)
    return value if isinstance(value, dict) else {}


def _energy(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("energy", {})
    return value if isinstance(value, dict) else {}


def _comfort(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("comfort", {})
    return value if isinstance(value, dict) else {}


@st.cache_data(show_spinner=False)
def _read_benchmark() -> tuple[pd.DataFrame, bytes | None]:
    """Read the optional repeat-run benchmark generated by the backend."""
    path = PROJECT_ROOT / "benchmark_results.csv"
    if not path.is_file():
        return pd.DataFrame(), None
    raw = path.read_bytes()
    return pd.read_csv(path), raw


def _benchmark_panel() -> None:
    """Render measured repeat-run performance when benchmark data exists."""
    benchmark, raw = _read_benchmark()
    if benchmark.empty:
        return
    savings = pd.to_numeric(benchmark.get("savings_percentage"), errors="coerce").dropna()
    success = benchmark.get("pipeline_success", pd.Series(dtype=bool)).astype(bool)
    avg_savings = float(savings.mean()) if not savings.empty else None
    success_rate = float(success.mean() * 100) if not success.empty else None
    total_applied = pd.to_numeric(benchmark.get("recommendations_applied"), errors="coerce").fillna(0).sum()
    st.markdown('<div class="section-card benchmark-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">REPEAT-RUN VALIDATION</div><div class="section-title">Autonomous benchmark</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Runs completed", str(len(benchmark)))
    cols[1].metric("Average savings", _display(avg_savings, "%"))
    cols[2].metric("Success rate", _display(success_rate, "%"))
    cols[3].metric("Recommendations applied", str(int(total_applied)))
    if not savings.empty:
        figure = px.line(benchmark, x="run", y="savings_percentage", markers=True, template=PLOT_TEMPLATE, title="Measured savings across consecutive real runs")
        figure.update_traces(line={"color": "#F37021", "width": 3}, marker={"size": 9})
        figure.update_layout(yaxis_title="Savings (%)", xaxis_title=None, margin={"t": 42, "b": 10, "l": 10, "r": 10}, height=300)
        st.plotly_chart(figure, use_container_width=True)
    st.download_button("Download benchmark CSV", raw, file_name="benchmark_results.csv", mime="text/csv", key="benchmark-download")
    st.markdown('</div>', unsafe_allow_html=True)


def _prediction_validation_panel(report: dict[str, Any]) -> None:
    """Display LLM estimates separately from measured EnergyPlus outcomes."""
    validation = report.get("prediction_validation", {})
    if not isinstance(validation, dict):
        return
    status = str(validation.get("prediction_status", "Not Evaluated"))
    if status == "Consistent":
        status_label, status_tone = "✅ Consistent", "success"
    elif status == "Failed":
        status_label, status_tone = "❌ Failed", "error"
    else:
        status_label, status_tone = "Not Evaluated", "info"
    st.markdown('<div class="section-card prediction-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">AI VS SIMULATION</div><div class="section-title">Prediction Validation</div>', unsafe_allow_html=True)
    llm_savings = _number(validation.get("llm_estimated_energy_savings_pct"))
    if llm_savings is None:
        raw_llm_change = _number(validation.get("llm_estimated_change_pct"))
        llm_savings = -raw_llm_change if raw_llm_change is not None else None
    measured_savings = _number(validation.get("measured_energy_savings_pct"))
    if measured_savings is None:
        raw_measured_change = _number(validation.get("measured_energy_change_pct"))
        measured_savings = -raw_measured_change if raw_measured_change is not None else None
    difference = _number(validation.get("difference_percentage_points"))
    if difference is None:
        difference = abs(llm_savings - measured_savings) if llm_savings is not None and measured_savings is not None else _number(validation.get("magnitude_error_pct"))
    direction = "Yes" if validation.get("direction_match") is True else "No" if validation.get("direction_match") is False else "N/A"
    left, middle, difference_col, direction_col = st.columns(4)
    left.metric("LLM Estimated Energy Savings", _display(llm_savings, "%"))
    middle.metric("Measured Energy Savings (EnergyPlus)", _display(measured_savings, "%"))
    difference_col.metric("Difference (percentage points)", _display(difference, " pp"))
    direction_col.metric("Direction Match (Yes/No)", direction)
    getattr(st, status_tone)(f"Prediction Status: {status_label}")
    detail = []
    if validation.get("reason"):
        detail.append(str(validation["reason"]))
    explanation = report.get("confidence_score_explanation")
    if explanation:
        detail.append(f"Confidence: {explanation}")
    if detail:
        st.caption(" • ".join(detail))
    st.markdown('</div>', unsafe_allow_html=True)


def _apply_theme(dark_mode: bool) -> None:
    """Apply a clean Honeywell-inspired presentation style."""
    background = "#101820" if dark_mode else "#f5f7fa"
    surface = "#182532" if dark_mode else "#ffffff"
    text = "#f7fafc" if dark_mode else "#17212b"
    muted = "#a8b6c2" if dark_mode else "#657482"
    border = "#2c3d4d" if dark_mode else "#e3e8ed"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {background}; color: {text}; }}
        [data-testid="stSidebar"] {{ background: {surface}; }}
        .block-container {{ padding-top: 2rem; max-width: 1500px; }}
        h1, h2, h3 {{ color: {text}; letter-spacing: -.02em; }}
        .subtitle {{ color: {muted}; margin-top: -12px; }}
        .brand {{ color: #F37021; font-weight: 800; letter-spacing: .08em; }}
        .hero {{ background: linear-gradient(115deg, {surface}, #223548); border: 1px solid {border}; border-radius: 14px; padding: 24px 28px; margin-bottom: 22px; }}
        .hero-title {{ font-size: 2rem; font-weight: 800; letter-spacing: .02em; color: {text}; }}
        .hero-subtitle {{ color: {muted}; margin-top: 5px; font-size: 1rem; }}
        .badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; align-items: center; height: 100%; }}
        .badge {{ border: 1px solid #16A34A; color: #16A34A; border-radius: 999px; padding: 7px 11px; font-size: .78rem; font-weight: 700; white-space: nowrap; background: rgba(22,163,74,.08); }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin: 14px 0 24px; }}
        .metric-card {{ background: {surface}; border: 1px solid {border}; border-radius: 11px; padding: 15px 16px; min-height: 112px; box-shadow: 0 5px 16px rgba(0,0,0,.06); }}
        .metric-card.savings {{ border-top: 3px solid #16A34A; }} .metric-card.info {{ border-top: 3px solid #2563EB; }} .metric-card.unavailable {{ opacity: .7; }}
        .metric-head {{ display:flex; gap:7px; align-items:center; color:{muted}; font-size:.78rem; font-weight:700; }} .metric-icon {{ font-size:1rem; }}
        .metric-value {{ color:{text}; font-size:1.45rem; font-weight:800; margin-top:12px; white-space:nowrap; }} .metric-detail {{ color:{muted}; font-size:.72rem; margin-top:5px; }}
        .section-card {{ background:{surface}; border:1px solid {border}; border-radius:14px; padding:20px; margin:12px 0; box-shadow:0 8px 24px rgba(0,0,0,.05); }}
        .section-kicker {{ color:#F37021; font-size:.72rem; font-weight:800; letter-spacing:.12em; }} .section-title {{ color:{text}; font-size:1.25rem; font-weight:800; margin:5px 0 15px; }} .benchmark-card {{ border-top:3px solid #F37021; }}
        .status-badge {{ display:inline-flex; align-items:center; gap:6px; border:1px solid rgba(22,163,74,.4); color:#16A34A; border-radius:999px; padding:6px 10px; font-size:.74rem; font-weight:750; white-space:nowrap; background:rgba(22,163,74,.08); }}
        .activity-tool {{ display:flex; align-items:center; gap:10px; padding:10px 0; border-bottom:1px solid {border}; }} .activity-tool:last-child {{ border-bottom:0; }}
        .check {{ color:#16A34A; font-size:1.1rem; }} .tool-name {{ font-weight:800; color:{text}; }} .tool-meta {{ color:{muted}; font-size:.8rem; }}
        .recommendation-card {{ background:{surface}; border:1px solid {border}; border-left:4px solid #F37021; border-radius:10px; padding:16px; margin:10px 0; }}
        .rec-label {{ color:{muted}; font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }} .rec-value {{ color:{text}; font-size:1.08rem; font-weight:800; }}
        @media (max-width: 1100px) {{ .metric-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
        @media (max-width: 650px) {{ .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .hero-title {{ font-size:1.45rem; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpis(report: dict[str, Any]) -> None:
    baseline = report.get("baseline_energy")
    optimized = report.get("optimized_energy")
    savings = report.get("percentage_energy_savings")
    comparison = report.get("comfort_comparison", {})
    cards = [
        _card("Baseline Energy", _compact_energy(baseline), "⚡", "neutral", "reported units"),
        _card("Optimized Energy", _compact_energy(optimized), "⚡", "neutral", "reported units"),
        _card("Energy Savings", _display(savings, "%"), "📉", "savings", "measured reduction"),
        _card("PMV", _display(comparison.get("optimized_pmv")), "🌡", "unavailable" if _number(comparison.get("optimized_pmv")) is None else "neutral", "thermal comfort"),
        _card("PPD", _display(comparison.get("optimized_ppd"), "%"), "👥", "unavailable" if _number(comparison.get("optimized_ppd")) is None else "neutral", "thermal comfort"),
        _card("Execution Time", _display(report.get("execution_time_seconds"), " s"), "⏱", "info", "controller runtime"),
    ]
    st.markdown('<div class="metric-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def _kpis_clean(report: dict[str, Any]) -> None:
    """Render KPI cards with portable icons and consistent formatting."""
    comparison = report.get("comfort_comparison", {})
    cards = [
        _card("Baseline Energy", _compact_energy(report.get("baseline_energy")), "\u26a1", "neutral", "reported units"),
        _card("Optimized Energy", _compact_energy(report.get("optimized_energy")), "\u26a1", "neutral", "reported units"),
        _card("Measured Energy Savings (EnergyPlus)", _display(report.get("percentage_energy_savings"), "%"), "\u2193", "savings", "simulation result"),
        _card("PMV", _display(comparison.get("optimized_pmv")), "\u25c6", "neutral", "thermal comfort"),
        _card("PPD", _display(comparison.get("optimized_ppd"), "%"), "\u25cf", "neutral", "thermal comfort"),
        _card("Execution Time", _display(report.get("execution_time_seconds"), " s"), "\u23f1", "info", "controller runtime"),
    ]
    st.markdown('<div class="metric-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def _hero_clean() -> None:
    """Render the Honeywell operations-center header and connection badges."""
    st.markdown(
        '<div class="hero"><div style="display:flex;justify-content:space-between;gap:24px;align-items:center;flex-wrap:wrap;">'
        '<div><div class="brand">HONEYWELL | BUILDING OPERATIONS CENTER</div><div class="hero-title">SMART BUILDING AI AGENT</div>'
        '<div class="hero-subtitle">Autonomous HVAC Energy Optimization using EnergyPlus + Agentic AI</div></div>'
        '<div class="badge-row"><span class="status-badge">● EnergyPlus Connected</span><span class="status-badge">● Ollama Connected</span><span class="status-badge">● Agent Ready</span></div>'
        '</div></div>', unsafe_allow_html=True,
    )


def _hero() -> None:
    """Render the operations-center hero header."""
    st.markdown(
        '<div class="hero"><div class="hero-title">SMART BUILDING AI AGENT</div>'
        '<div class="hero-subtitle">Autonomous HVAC Energy Optimization using EnergyPlus + Agentic AI</div></div>',
        unsafe_allow_html=True,
    )


def _energy_chart(report: dict[str, Any]) -> None:
    values = pd.DataFrame({"Scenario": ["Baseline", "Optimized"], "Energy": [report.get("baseline_energy"), report.get("optimized_energy")]})
    values = values.dropna()
    if values.empty:
        st.info("Energy comparison is not available in this report.")
        return
    figure = px.bar(values, x="Scenario", y="Energy", color="Scenario", template=PLOT_TEMPLATE, color_discrete_sequence=["#8996a3", "#F37021"])
    figure.update_traces(hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>")
    savings = _number(report.get("percentage_energy_savings"))
    optimized = _number(report.get("optimized_energy"))
    if savings is not None and optimized is not None:
        figure.add_annotation(x="Optimized", y=optimized, text=f"↓ {savings:.2f}%", showarrow=False, yshift=18, font={"color": "#16A34A", "size": 14})
    figure.update_layout(showlegend=False, yaxis_title="Energy (reported units)", xaxis_title=None, margin={"t": 30, "b": 20, "l": 10, "r": 10})
    st.plotly_chart(figure, use_container_width=True)


def _history_frame(report: dict[str, Any], csv_frame: pd.DataFrame) -> pd.DataFrame:
    if not csv_frame.empty:
        return csv_frame
    records = report.get("iteration_history", [])
    return pd.DataFrame(records) if isinstance(records, list) else pd.DataFrame()


def _overview(report: dict[str, Any], history: pd.DataFrame) -> None:
    _hero_clean()
    st.title("Dashboard Overview")
    st.markdown('<div class="subtitle">Autonomous building optimization control room</div>', unsafe_allow_html=True)
    _kpis_clean(report)
    baseline = _number(report.get("baseline_energy")); optimized = _number(report.get("optimized_energy")); savings = _number(report.get("percentage_energy_savings"))
    if baseline is not None and optimized is not None:
        st.markdown(f'<div class="section-card"><div class="rec-label">MEASURED ENERGY SAVED (ENERGYPLUS)</div><div class="metric-value">{savings:.2f}%</div><div class="metric-detail">↓ {baseline - optimized:,.2f} reported units &nbsp; | &nbsp; Baseline {baseline / 1_000_000_000:.2f}B → Optimized {optimized / 1_000_000_000:.2f}B</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Baseline vs optimized energy")
        _energy_chart(report)
    with right:
        st.subheader("Run status")
        successful = all(status == "completed" for status in history.get("status", pd.Series(dtype=str)).tolist()) if not history.empty else True
        st.success("Controller run completed" if successful else "Controller run contains failures")
        st.write(f"Recommendations applied: {len(report.get('recommendations_applied', []))}")
        st.write(f"Iterations recorded: {len(report.get('iteration_history', []))}")
        if report.get("error"):
            st.error(str(report["error"]))
    _benchmark_panel()
    _prediction_validation_panel(report)


def _building_metrics(report: dict[str, Any]) -> None:
    st.title("Building Metrics")
    baseline = _state(report, "baseline_state")
    optimized = _state(report, "optimized_state")
    thermal = optimized.get("thermal", {}) if optimized else {}
    temperatures = thermal.get("zone_temperatures", {}) if isinstance(thermal, dict) else {}
    if temperatures:
        frame = pd.DataFrame({"Zone": list(temperatures), "Temperature": list(temperatures.values())})
        optimized_zones = {item.get("zone") for item in report.get("recommendations_applied", []) if isinstance(item, dict)}
        frame["State"] = frame["Zone"].map(lambda zone: "Optimized" if zone in optimized_zones else "Zone")
        figure = px.bar(frame, x="Zone", y="Temperature", color="State", template=PLOT_TEMPLATE, color_discrete_map={"Zone": "#6f9fb0", "Optimized": "#F37021"}, text=frame["Temperature"].map(lambda value: f"{value:.2f}°C"))
        figure.update_layout(yaxis_title="Temperature (°C)", xaxis_title=None)
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("Zone temperature data is not available in this controller report.")
    average = sum(temperatures.values()) / len(temperatures) if temperatures else None
    columns = st.columns(4)
    columns[0].metric("Outdoor Temperature", _display(thermal.get("outdoor_temperature"), " °C"))
    columns[1].metric("Zones", str(len(optimized.get("zone_names", []))))
    columns[2].metric("Occupied Zones", str(len(optimized.get("occupied_zones", []))))
    columns[3].metric("Average Indoor", _display(average, " °C"))
    st.subheader("Zone details")
    st.dataframe(pd.DataFrame({"Zone": list(temperatures), "Indoor Temperature (°C)": list(temperatures.values())}), use_container_width=True, hide_index=True)


def _energy_optimization(report: dict[str, Any], history: pd.DataFrame) -> None:
    st.title("Energy Optimization")
    left, right = st.columns(2)
    with left:
        st.subheader("Scenario comparison")
        _energy_chart(report)
    with right:
        st.subheader("Optimized energy mix")
        components = _energy(_state(report, "optimized_state"))
        rows = [(label.replace("_", " ").title(), value) for label, value in components.items() if label != "total_electricity_consumption" and _number(value) is not None]
        if rows:
            figure = px.pie(pd.DataFrame(rows, columns=["Component", "Energy"]), names="Component", values="Energy", hole=.45, template=PLOT_TEMPLATE, color_discrete_sequence=["#e75b25", "#3aa6b9", "#f3a712", "#72808d"])
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Energy component data is not available.")
    if not history.empty and "energy" in history:
        figure = px.line(history, x="iteration", y="energy", markers=True, color="status", template=PLOT_TEMPLATE, title="Energy by iteration")
        figure.update_layout(yaxis_title="Energy (reported units)", xaxis_title="Iteration")
        st.plotly_chart(figure, use_container_width=True)


def _gauge(title: str, value: float | None, minimum: float, maximum: float, suffix: str = "") -> None:
    if value is None:
        st.info(f"{title} is not available.")
        return
    figure = go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": title}, number={"suffix": suffix}, gauge={"axis": {"range": [minimum, maximum]}, "bar": {"color": "#e75b25"}, "bgcolor": "#1b2a38"}))
    figure.update_layout(template=PLOT_TEMPLATE, height=260, margin={"t": 60, "b": 10, "l": 20, "r": 20})
    st.plotly_chart(figure, use_container_width=True)


def _thermal_comfort(report: dict[str, Any]) -> None:
    st.title("Thermal Comfort")
    comparison = report.get("comfort_comparison", {})
    left, right = st.columns(2)
    with left:
        pmv = _number(comparison.get("optimized_pmv"))
        st.markdown(f'<div class="section-card"><div class="rec-label">PMV</div><div class="metric-value">{_display(pmv)}</div><div class="metric-detail">{"Thermal comfort indicator" if pmv is not None else "This EnergyPlus example model does not report PMV variables."}</div></div>', unsafe_allow_html=True)
    with right:
        ppd = _number(comparison.get("optimized_ppd"))
        st.markdown(f'<div class="section-card"><div class="rec-label">PPD</div><div class="metric-value">{_display(ppd, "%")}</div><div class="metric-detail">{"Thermal comfort indicator" if ppd is not None else "This EnergyPlus example model does not report PPD variables."}</div></div>', unsafe_allow_html=True)
    measured_result = str(report.get("measured_comfort_result", "Not Evaluated"))
    st.markdown(
        f'<div class="section-card"><div class="rec-label">MEASURED COMFORT RESULT (ENERGYPLUS)</div>'
        f'<div class="metric-value">{html.escape(measured_result)}</div>'
        f'<div class="metric-detail">Based only on baseline versus optimized PMV and PPD.</div></div>',
        unsafe_allow_html=True,
    )
    validation = report.get("comfort_validation", {})
    if not isinstance(validation, dict):
        validation = {}
    agreement = validation.get("agreement")
    agreement_display = "✓ Yes" if agreement is True else "✗ No" if agreement is False else "— Not Evaluated"
    st.markdown("### Comfort Validation")
    validation_columns = st.columns(3)
    validation_columns[0].metric("LLM Prediction", str(validation.get("llm_prediction", "Not Available")))
    validation_columns[1].metric("Measured Result (EnergyPlus)", str(validation.get("measured_result", measured_result)))
    validation_columns[2].metric("Agreement", agreement_display)
    reason = str(validation.get("reason", "No comfort validation explanation is available."))
    st.caption(f"Reason: {reason}")
    values = pd.DataFrame({"Scenario": ["Baseline", "Optimized"], "PMV": [comparison.get("baseline_pmv"), comparison.get("optimized_pmv")], "PPD": [comparison.get("baseline_ppd"), comparison.get("optimized_ppd")]})
    values = values.dropna(subset=["PMV", "PPD"], how="all")
    if not values.empty:
        figure = px.line(values, x="Scenario", y=["PMV", "PPD"], markers=True, template=PLOT_TEMPLATE, title="Comfort comparison")
        st.plotly_chart(figure, use_container_width=True)


def _decision_analysis(report: dict[str, Any]) -> None:
    st.title("AI Decision Analysis")
    st.markdown(
        f"**Measured Comfort Result (EnergyPlus):** {html.escape(str(report.get('measured_comfort_result', 'Not Evaluated')))}",
        unsafe_allow_html=True,
    )
    actions = report.get("recommendations_applied", [])
    if not isinstance(actions, list) or not actions:
        st.info("No recommendations were applied in this controller run.")
        return
    frame = pd.DataFrame(actions)
    for action in actions:
        confidence = _number(action.get("confidence_score"))
        savings = _number(action.get("estimated_energy_savings_pct"))
        st.markdown(
            f'<div class="recommendation-card"><div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">'
            f'<div><div class="rec-label">Zone</div><div class="rec-value">{html.escape(str(action.get("zone", "Unavailable")))}</div></div>'
            f'<div><div class="rec-label">Control</div><div class="rec-value">{html.escape(str(action.get("parameter", "Unavailable")))}</div></div>'
            f'<div><div class="rec-label">Current</div><div class="rec-value">{_display(action.get("current"), "°C")}</div></div>'
            f'<div><div class="rec-label">Recommended</div><div class="rec-value">{_display(action.get("recommended"), "°C")}</div></div>'
            f'<div><div class="rec-label">LLM Estimated Energy Savings</div><div class="rec-value">{_display(savings, "%")}</div></div>'
            f'<div><div class="rec-label">AI Confidence</div><div class="rec-value">{_display(confidence * 100 if confidence is not None else None, "%")}</div></div>'
            f'<div><div class="rec-label">LLM Predicted Comfort Impact</div><div class="rec-value">{html.escape(str(action.get("llm_predicted_comfort_impact", action.get("estimated_comfort_impact", "Unavailable"))).replace("_", " ").title())}</div></div>'
            f'</div></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        frame["AI Confidence"] = frame["confidence_score"].map(lambda value: float(value) * 100 if _number(value) is not None else None)
        figure = px.bar(frame, x="zone", y="AI Confidence", color="priority", template=PLOT_TEMPLATE, title="AI Confidence")
        figure.update_layout(yaxis_title="Confidence (%)", xaxis_title=None)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = px.bar(frame, x="zone", y="estimated_energy_savings_pct", color="priority", template=PLOT_TEMPLATE, title="LLM Estimated Energy Savings")
        figure.update_layout(yaxis_title="Estimated savings (%)", xaxis_title=None)
        st.plotly_chart(figure, use_container_width=True)


def _simulation_history(history: pd.DataFrame) -> None:
    st.title("Simulation History")
    if history.empty:
        st.info("No iteration history is available.")
        return
    st.dataframe(history, use_container_width=True, hide_index=True)
    if "iteration" in history and "status" in history:
        figure = px.scatter(history, x="iteration", y="status", color="status", size="recommendations_applied" if "recommendations_applied" in history else None, text="kind" if "kind" in history else None, template=PLOT_TEMPLATE, title="Optimization timeline")
        figure.update_layout(xaxis_title="Iteration", yaxis_title=None)
        st.plotly_chart(figure, use_container_width=True)


def _system_logs() -> None:
    st.title("System Logs")
    path = settings.log_directory / "application.log"
    if not path.is_file():
        st.info("No application log has been generated yet.")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    st.download_button("Download system log", text.encode("utf-8"), file_name="application.log", mime="text/plain")
    st.code("\n".join(text.splitlines()[-300:]), language="text")


class _AgentActivity:
    """Dashboard-only recorder for LLM selections and tool results."""

    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []


class _DashboardOllamaClient:
    """Delegate Ollama operations while recording tool selections for the UI."""

    def __init__(self, client: OllamaClient, activity: _AgentActivity) -> None:
        self._client = client
        self._activity = activity

    def select_tool(self, *args: Any, **kwargs: Any) -> AgentToolSelection:
        selection = self._client.select_tool(*args, **kwargs)
        self._activity.decisions.append({
            "action": selection.action,
            "tool": selection.tool_name,
            "intent": selection.intent,
            "reasoning": selection.reasoning,
        })
        return selection

    def optimize_building(self, building_state: Any) -> Any:
        return self._client.optimize_building(building_state)


class _DashboardDispatcher:
    """Delegate the existing dispatcher while recording its typed results."""

    def __init__(self, dispatcher: ToolDispatcher, activity: _AgentActivity) -> None:
        self._dispatcher = dispatcher
        self.registry = dispatcher.registry
        self._activity = activity

    def trusted_resources(self) -> dict[str, Any]:
        return self._dispatcher.trusted_resources()

    def _record(self, result: Any) -> Any:
        self._activity.tool_calls.append({
            "call_id": result.call_id,
            "tool": result.tool_name,
            "success": result.success,
            "error": result.error,
            "data_keys": list(result.data),
        })
        return result

    def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return self._record(self._dispatcher.dispatch(*args, **kwargs))

    def dispatch_intent(self, *args: Any, **kwargs: Any) -> Any:
        return self._record(self._dispatcher.dispatch_intent(*args, **kwargs))


def _run_controller(activity: _AgentActivity) -> Any:
    """Run the existing controller using only configured demo resources."""
    if not settings.demo_idf_path or not settings.demo_weather_file:
        raise FileNotFoundError("DEMO_IDF_PATH and DEMO_WEATHER_FILE must be configured")
    if not settings.demo_idf_path.is_file():
        raise FileNotFoundError(f"Configured demo IDF does not exist: {settings.demo_idf_path}")
    if not settings.demo_weather_file.is_file():
        raise FileNotFoundError(f"Configured weather file does not exist: {settings.demo_weather_file}")
    configure_logging(settings.log_directory, settings.log_level)
    run_root = settings.output_directory / f"dashboard_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    input_idf = settings.demo_idf_path
    if settings.stress_test:
        input_idf, _ = prepare_stressed_idf(
            settings.demo_idf_path,
            run_root / "stressed_input.idf",
            enabled=True,
        )
    runner = EnergyPlusRunner()
    base_dispatcher = create_default_tool_dispatcher(runner)
    dispatcher = _DashboardDispatcher(base_dispatcher, activity)
    ollama = _DashboardOllamaClient(OllamaClient(), activity)
    engine = AgenticDecisionEngine(ollama, dispatcher, max_steps=8)
    controller = OptimizationController(
        runner,
        engine,
        parser=EnergyPlusOutputParser(),
        report_directory=run_root,
        tool_dispatcher=dispatcher,
    )
    return controller.run(input_idf, settings.demo_weather_file, run_root, iterations=1)


def _run_with_progress(activity: _AgentActivity) -> Any:
    """Run the controller in a worker while presenting dashboard progress."""
    stages = (
        "Validating EnergyPlus",
        "Running Baseline Simulation",
        "Parsing Outputs",
        "Agent Selecting Tools",
        "Applying Recommendation",
        "Running Optimized Simulation",
        "Comparing Results",
        "Generating Report",
    )
    progress = st.progress(0, text="Starting autonomous workflow")
    status = st.status("Autonomous optimization in progress", expanded=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future[Any] = executor.submit(_run_controller, activity)
        index = 0
        last_stage: str | None = None
        while not future.done():
            stage = stages[min(index, len(stages) - 1)]
            if stage != last_stage:
                status.write(f"⏳ {stage}")
                last_stage = stage
            else:
                status.update(label=f"⏳ {stage}", state="running", expanded=True)
            progress.progress(min(95, int((index + 1) / len(stages) * 100)), text=stage)
            index += 1
            time.sleep(1.0)
        result = future.result()
    for stage_index, stage in enumerate(stages, start=1):
        status.write(f"✓ {stage}")
        progress.progress(int(stage_index / len(stages) * 100), text=stage)
    status.update(label="Autonomous optimization complete", state="complete", expanded=False)
    return result


def _agent_activity(activity: _AgentActivity | None, report: dict[str, Any]) -> None:
    """Render dashboard-captured LLM selections, tool results, and outcome."""
    st.subheader("Agent Activity")
    if activity is None:
        st.info("No live agent activity is available for this uploaded report.")
        return
    with st.expander("Tool sequence selected by the LLM", expanded=True):
        decisions = [item for item in activity.decisions if item.get("action") == "tool"]
        if decisions:
            for index, decision in enumerate(decisions, start=1):
                st.write(f"{index}. `{decision.get('tool')}` — {decision.get('intent')}")
        else:
            st.info("The LLM returned a final recommendation without selecting an intermediate tool.")
    with st.expander("Tool execution results"):
        if activity.tool_calls:
            st.dataframe(pd.DataFrame(activity.tool_calls), use_container_width=True, hide_index=True)
        else:
            st.info("No tool results were captured.")
    with st.expander("Final recommendation", expanded=True):
        actions = report.get("recommendations_applied", [])
        if actions:
            st.dataframe(pd.DataFrame(actions), use_container_width=True, hide_index=True)
        else:
            st.info("No recommendation was applied.")


def render_dashboard() -> None:
    """Render the live autonomous workflow and previously generated reports."""
    st.set_page_config(page_title="Smart Building AI Agent", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")
    dark_mode = st.sidebar.toggle("Dark mode", value=True)
    _apply_theme(dark_mode)
    st.sidebar.markdown('<div class="brand">HONEYWELL | SMART BUILDING AI</div>', unsafe_allow_html=True)
    st.sidebar.caption("SYSTEM STATUS")
    st.sidebar.markdown("<span class='status-badge'>● EnergyPlus</span> <span class='status-badge'>● Ollama</span> <span class='status-badge'>● Agent</span>", unsafe_allow_html=True)
    st.sidebar.divider()
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    activity = st.session_state.get("agent_activity")
    active_report_path = st.session_state.get("active_report_path")
    if st.sidebar.button("▶ Run Optimization", type="primary", use_container_width=True):
        activity = _AgentActivity()
        st.session_state["agent_activity"] = activity
        try:
            result = _run_with_progress(activity)
            if result.error:
                st.error(f"Optimization finished with an error: {result.error}")
                st.info("The diagnostic report was preserved. The dashboard continues showing the last successful report.")
                active_report_path = st.session_state.get("active_report_path") or _latest_report()
            else:
                st.session_state["active_report_path"] = result.report_path
                st.session_state["active_csv_path"] = result.csv_summary_path
                active_report_path = result.report_path
            st.cache_data.clear()
            if not result.error:
                st.success("Optimization completed and the generated report was loaded.")
        except Exception as exc:
            st.error(f"Optimization failed: {type(exc).__name__}: {exc}")
            st.stop()

    default_report = Path(active_report_path) if active_report_path else _latest_report()
    uploaded = st.sidebar.file_uploader("Optimization JSON report", type=["json"])
    report_path = st.sidebar.text_input("Report path", value=str(default_report) if default_report else "")
    raw_report: bytes | None = None
    if uploaded is not None:
        report = json.loads(uploaded.getvalue().decode("utf-8"))
        raw_report = uploaded.getvalue()
        activity = None
    elif report_path and Path(report_path).is_file():
        report, raw_report = _read_report(report_path)
    else:
        st.title("Smart Building AI Agent")
        st.info("Click **Run Optimization** to execute the configured EnergyPlus/Ollama workflow, or upload an existing report.")
        return
    csv_path = report.get("csv_summary_path")
    if not isinstance(csv_path, str):
        csv_path = st.session_state.get("active_csv_path")
    history, raw_csv = _read_csv(csv_path if isinstance(csv_path, str) else None)
    st.sidebar.download_button("Download JSON report", raw_report, file_name="optimization_report.json", mime="application/json")
    if raw_csv is not None:
        st.sidebar.download_button("Download CSV summary", raw_csv, file_name="optimization_summary.csv", mime="text/csv")
    _agent_activity(activity, report)
    page = st.sidebar.radio("Navigation", ["Dashboard Overview", "Building Metrics", "Energy Optimization", "Thermal Comfort", "AI Decision Analysis", "Simulation History", "System Logs"])
    if page == "Dashboard Overview":
        _overview(report, history)
    elif page == "Building Metrics":
        _building_metrics(report)
    elif page == "Energy Optimization":
        _energy_optimization(report, history)
    elif page == "Thermal Comfort":
        _thermal_comfort(report)
    elif page == "AI Decision Analysis":
        _decision_analysis(report)
    elif page == "Simulation History":
        _simulation_history(history)
    else:
        _system_logs()


if __name__ == "__main__":
    render_dashboard()
