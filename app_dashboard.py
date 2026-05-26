import streamlit as st
import html
import json
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    genai = None

ESTADO_DASHBOARD = Path(".dashboard_state.json")
ARCHIVO_DATOS = Path("REPORTE_LIMPIO_FINAL.parquet")
APP_VERSION = "V1.11"

CONTEXTO_CONCEPTUAL_SEGURIDAD = """
Contexto conceptual fijo para interpretar el tablero:
- ENVIPE es la Encuesta Nacional de Victimización y Percepción sobre Seguridad Pública del INEGI. Su propósito es generar información sobre victimización, incidencia delictiva, cifra negra, características del delito, víctimas, percepción de seguridad pública y desempeño institucional.
- Percepción de inseguridad mide una apreciación subjetiva de la población; no equivale directamente a incidencia delictiva registrada ni a riesgo objetivo.
- Cifra negra representa delitos no denunciados o denunciados sin inicio de investigación/carpeta. Es un indicador de subregistro, no una medición directa de todos los delitos reales.
- Incidencia delictiva en este tablero se usa como medición cuantitativa de delitos/tasas disponibles en la base; puede ser general o específica por tipo de delito.
- Percepción, cifra negra e incidencia no son variables equivalentes. Sus cruces son exploratorios y descriptivos.
- Las correlaciones y dispersiones del tablero muestran asociaciones dentro de los filtros seleccionados; no prueban causalidad.
- Toda lectura depende de años, entidades, sexo, delito y cobertura de datos seleccionados.
"""

NOTA_METODOLOGICA = (
    "Nota metodológica: este tablero realiza cruces descriptivos entre percepción de inseguridad, "
    "cifra negra e incidencia delictiva. Las relaciones observadas no deben interpretarse como causalidad. "
    "Los resultados dependen de los años, entidades, sexo, delito y cobertura de datos seleccionados."
)


def cargar_estado_persistente():
    if not ESTADO_DASHBOARD.exists():
        return {}

    try:
        return json.loads(ESTADO_DASHBOARD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def guardar_estado_persistente(estado):
    try:
        ESTADO_DASHBOARD.write_text(
            json.dumps(
                estado,
                ensure_ascii=False,
                indent=2,
                default=lambda valor: valor.item() if hasattr(valor, "item") else str(valor),
            ),
            encoding="utf-8"
        )
    except OSError:
        pass


def borrar_estado_persistente():
    try:
        ESTADO_DASHBOARD.unlink(missing_ok=True)
    except OSError:
        pass


ESTADO_PERSISTENTE = cargar_estado_persistente()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Seguridad MX",
    layout="wide"
)

if st.query_params.get("reset_dashboard") == "1":
    borrar_estado_persistente()
    for clave in [
        "anios_globales",
        "estados_globales",
        "sexo_percepcion",
        "tab4_delito_master",
        "analisis_ia",
        "analisis_ia_contexto",
        "analisis_ia_contexto_actual",
        "chat_ia",
        "graficos_ia",
        "graficos_ia_specs",
        "artefactos_ia_specs",
        "pregunta_ia_pendiente",
        "ai_autoscroll_ready",
        "report_autoscroll_ready",
    ]:
        st.session_state.pop(clave, None)
    st.query_params.clear()
    st.rerun()

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --app-bg: var(--background-color, #ffffff);
        --app-text: var(--text-color, #0a0a0a);
        --app-muted: rgba(128, 128, 128, 0.86);
        --app-border: rgba(128, 128, 128, 0.30);
        --app-soft: var(--secondary-background-color, #f6f6f6);
        --app-panel: var(--background-color, #ffffff);
        --app-invert: var(--text-color, #0a0a0a);
        --app-invert-text: var(--background-color, #ffffff);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg: var(--background-color, #0a0a0a);
            --app-text: var(--text-color, #f5f5f5);
            --app-muted: #b7b7b7;
            --app-border: #303030;
            --app-soft: var(--secondary-background-color, #141414);
            --app-panel: var(--background-color, #0f0f0f);
            --app-invert: var(--text-color, #ffffff);
            --app-invert-text: var(--background-color, #0a0a0a);
        }
    }

    html,
    body,
    .stApp,
    main,
    section.main,
    div[data-testid="stAppViewContainer"],
    div[data-testid="stMain"],
    div[data-testid="stMainBlockContainer"] {
        scroll-behavior: smooth;
    }

    a[id] {
        scroll-margin-top: 4rem;
    }

    html, body, [class*="stApp"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--app-bg);
        color: var(--app-text);
    }

    ::selection {
        background: var(--app-invert) !important;
        color: var(--app-invert-text) !important;
    }

    ::-moz-selection {
        background: var(--app-invert) !important;
        color: var(--app-invert-text) !important;
    }

    div[data-testid="stAppViewContainer"],
    div[data-testid="stMain"],
    section[data-testid="stSidebar"],
    header[data-testid="stHeader"],
    .block-container {
        transition: filter 220ms ease, opacity 220ms ease, background-color 220ms ease;
    }

    body.app-refreshing-all div[data-testid="stAppViewContainer"],
    body.app-refreshing-all section[data-testid="stSidebar"],
    body.app-refreshing-all header[data-testid="stHeader"] {
        filter: blur(2.4px) brightness(0.76);
        opacity: 0.86;
    }

    .refresh-fade-overlay {
        position: fixed;
        inset: 0;
        z-index: 999996;
        pointer-events: none;
        opacity: 0;
        background: rgba(0, 0, 0, 0.12);
        backdrop-filter: blur(1.4px);
        -webkit-backdrop-filter: blur(1.4px);
        transition: opacity 220ms ease;
    }

    body.app-refreshing-all .refresh-fade-overlay {
        opacity: 1;
    }

    .mobile-blocker {
        display: none;
    }

    @media (max-width: 760px) and (orientation: portrait) {
        section[data-testid="stSidebar"],
        header[data-testid="stHeader"] {
            display: none !important;
        }

        .mobile-blocker {
            position: fixed;
            inset: 0;
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
            background: var(--app-bg);
            color: var(--app-text);
        }

        .mobile-blocker-card {
            width: min(100%, 25rem);
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: var(--app-panel);
            padding: 1.25rem;
        }

        .mobile-blocker-card strong {
            display: block;
            font-size: 1.25rem;
            margin-bottom: 0.55rem;
        }

        .mobile-blocker-card p {
            margin: 0;
            color: var(--app-muted);
            line-height: 1.5;
        }
    }

    .block-container {
        max-width: 1440px;
        padding: 2.25rem 3rem 3rem;
    }

    .sidebar-hotzone {
        position: fixed;
        top: 0;
        left: 0;
        width: 24px;
        height: 100vh;
        z-index: 999997;
        background: transparent;
        pointer-events: auto;
    }

    #MainMenu,
    footer,
    div[data-testid="stToolbar"],
    div[data-testid="stDeployButton"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    div[data-testid="stActionButton"],
    button[data-testid="stActionButton"],
    a[data-testid="stActionButton"] {
        display: none !important;
        visibility: hidden !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
        height: 0 !important;
    }

    div[data-testid="stSidebarCollapsedControl"],
    div[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapsedControl"],
    button[data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        position: fixed !important;
        top: 0.85rem !important;
        left: 0.85rem !important;
        z-index: 999998 !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--app-soft);
        border-right: 1px solid var(--app-border);
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] {
        min-height: 3.1rem !important;
        height: 3.1rem !important;
        padding: 0.5rem 0.75rem 0 !important;
        overflow: visible !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] > div {
        min-height: 2.25rem !important;
        height: 2.25rem !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding: 0 1rem 1.5rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }

    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="baseButton-headerNoPadding"],
    button[kind="headerNoPadding"],
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarCollapsedControl"],
    button[data-testid="collapsedControl"],
    div[data-testid="stSidebarCollapseButton"] button,
    div[data-testid="stSidebarCollapsedControl"] button,
    div[data-testid="collapsedControl"] button {
        width: 2.25rem;
        height: 2.25rem;
        border: 1px solid var(--app-border) !important;
        border-radius: 8px !important;
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        opacity: 1 !important;
        box-shadow: 0 1px 8px rgba(0, 0, 0, 0.06);
        position: relative !important;
        z-index: 50 !important;
    }

    div[data-testid="stSidebarHeader"],
    div[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarCollapsedControl"],
    div[data-testid="collapsedControl"] {
        align-items: flex-start !important;
        padding-top: 0.65rem !important;
    }

    button[data-testid="stBaseButton-headerNoPadding"] *,
    button[data-testid="baseButton-headerNoPadding"] *,
    button[kind="headerNoPadding"] *,
    button[data-testid="stSidebarCollapseButton"] *,
    button[data-testid="stSidebarCollapsedControl"] *,
    button[data-testid="collapsedControl"] *,
    div[data-testid="stSidebarCollapseButton"] button *,
    div[data-testid="stSidebarCollapsedControl"] button *,
    div[data-testid="collapsedControl"] button * {
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
        stroke: var(--app-text) !important;
        opacity: 1 !important;
    }

    button[data-testid="stBaseButton-headerNoPadding"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover,
    button[kind="headerNoPadding"]:hover,
    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="stSidebarCollapsedControl"]:hover,
    button[data-testid="collapsedControl"]:hover,
    div[data-testid="stSidebarCollapseButton"] button:hover,
    div[data-testid="stSidebarCollapsedControl"] button:hover,
    div[data-testid="collapsedControl"] button:hover {
        border-color: var(--app-invert) !important;
        background: var(--app-soft) !important;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--app-text);
        letter-spacing: 0;
        margin-top: 0.15rem;
        margin-bottom: 0.75rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--app-text);
    }

    h1 {
        font-size: clamp(2.1rem, 4vw, 4rem);
        line-height: 0.95;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }

    h2, h3 {
        font-weight: 700;
    }

    p, label, span, div {
        letter-spacing: 0;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: var(--app-muted);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    div[data-testid="stRadio"] label,
    div[data-testid="stSegmentedControl"] label {
        border-color: var(--app-border);
        background: var(--app-panel);
        color: var(--app-text);
        border-radius: 8px;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] *,
    div[data-testid="stRadio"] *,
    div[data-testid="stSegmentedControl"] * {
        color: var(--app-text);
        fill: var(--app-text);
    }

    [data-baseweb="tag"] {
        background: var(--app-invert) !important;
        background-color: var(--app-invert) !important;
        color: var(--app-invert-text) !important;
        border-radius: 999px;
    }

    [data-baseweb="tag"] * {
        color: var(--app-invert-text) !important;
        fill: var(--app-invert-text) !important;
        stroke: var(--app-invert-text) !important;
    }

    ul[role="listbox"],
    div[role="listbox"],
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] ul {
        background: var(--app-panel) !important;
        border: 1px solid var(--app-border);
        color: var(--app-text) !important;
    }

    li[role="option"],
    div[role="option"],
    div[data-baseweb="menu"] li,
    div[data-baseweb="menu"] div {
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
    }

    li[role="option"]:hover,
    div[role="option"]:hover,
    div[data-baseweb="menu"] li:hover {
        background: var(--app-soft) !important;
        color: var(--app-text) !important;
    }

    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] *,
    ul[role="listbox"] *,
    div[role="listbox"] * {
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid var(--app-border);
        background: var(--app-soft);
        color: var(--app-text);
    }

    div[data-testid="stAlert"] * {
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
        stroke: var(--app-text) !important;
    }

    details[data-testid="stExpander"],
    div[data-testid="stExpander"] {
        border: 1px solid var(--app-border) !important;
        border-radius: 8px !important;
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        overflow: hidden;
    }

    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary:hover,
    details[data-testid="stExpander"] summary:focus,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary:hover,
    div[data-testid="stExpander"] summary:focus,
    div[data-testid="stExpander"] button,
    div[data-testid="stExpander"] button:hover,
    div[data-testid="stExpander"] button:focus {
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        border-color: var(--app-border) !important;
        box-shadow: none !important;
        outline: none !important;
    }

    details[data-testid="stExpander"] *,
    div[data-testid="stExpander"] *,
    div[data-testid="stExpanderDetails"] *,
    div[data-testid="stExpander"] svg,
    div[data-testid="stExpander"] svg * {
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
        stroke: var(--app-text) !important;
    }

    div[data-testid="stExpanderDetails"],
    div[data-testid="stExpanderDetails"] > div {
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
    }

    div[data-testid="stCode"],
    div[data-testid="stCodeBlock"],
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] code,
    pre,
    code {
        background: var(--app-soft) !important;
        color: var(--app-text) !important;
        border-color: var(--app-border) !important;
    }

    div[data-testid="stCode"],
    div[data-testid="stCodeBlock"] {
        border: 1px solid var(--app-border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    div[data-testid="stCodeBlock"] button,
    div[data-testid="stCodeBlock"] button:hover,
    div[data-testid="stCodeBlock"] button:focus,
    div[data-testid="stCodeBlock"] [role="button"] {
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }

    div[data-testid="stCodeBlock"] button *,
    div[data-testid="stCodeBlock"] [role="button"] *,
    div[data-testid="stCodeBlock"] svg,
    div[data-testid="stCodeBlock"] svg * {
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
        stroke: var(--app-text) !important;
    }

    div[data-testid="stPlotlyChart"] {
        border: 1px solid var(--app-border);
        border-radius: 8px;
        padding: 0.75rem;
        background: var(--app-panel);
    }

    div[data-testid="stPlotlyChart"] .modebar,
    div[data-testid="stPlotlyChart"] .modebar-group {
        background: transparent !important;
    }

    div[data-testid="stPlotlyChart"] .modebar-btn {
        background: transparent !important;
        color: var(--app-text) !important;
        opacity: 1 !important;
    }

    div[data-testid="stPlotlyChart"] .modebar-btn svg,
    div[data-testid="stPlotlyChart"] .modebar-btn svg path {
        fill: var(--app-text) !important;
        color: var(--app-text) !important;
        opacity: 1 !important;
    }

    div[data-testid="stPlotlyChart"] .modebar-btn:hover {
        background: var(--app-soft) !important;
        border-radius: 6px !important;
    }

    hr {
        border-color: var(--app-border);
        margin: 2rem 0;
    }

    .app-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.78rem;
        line-height: 1;
        font-weight: 800;
        text-transform: uppercase;
        color: var(--app-invert-text);
        background: var(--app-invert);
        border-radius: 999px;
        padding: 0.48rem 0.74rem;
        margin: 0.2rem 0 0.8rem;
    }

    .app-subtitle {
        max-width: 820px;
        color: var(--app-muted);
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1.25rem;
    }

    .side-brand {
        border: 1px solid var(--app-border);
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        background: var(--app-panel);
        margin: -2.72rem 3.25rem 0.75rem 0;
        min-height: 2.25rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .side-brand-title {
        color: var(--app-text);
        font-weight: 800;
        font-size: 0.98rem;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }

    .side-brand-subtitle {
        color: var(--app-muted);
        font-size: 0.76rem;
        line-height: 1.35;
    }

    .side-nav {
        display: grid;
        gap: 0.45rem;
        margin-top: 0.2rem;
    }

    .side-nav a {
        display: block;
        border: 1px solid var(--app-border);
        border-radius: 8px;
        padding: 0.58rem 0.7rem;
        color: var(--app-text) !important;
        background: var(--app-panel);
        text-decoration: none;
        font-weight: 650;
        font-size: 0.9rem;
    }

    .side-nav a:hover {
        border-color: var(--app-invert);
        background: var(--app-soft);
    }

    .reset-link {
        display: block;
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #b91c1c;
        border-radius: 8px;
        padding: 0.72rem 0.78rem;
        color: #ffffff !important;
        background: #b91c1c;
        text-align: center;
        text-decoration: none;
        font-weight: 800;
        margin-top: 1.2rem;
    }

    .reset-link:hover {
        background: #991b1b;
        border-color: #991b1b;
        color: #ffffff !important;
    }

    .sidebar-version {
        border-top: 1px solid var(--app-border);
        color: var(--app-muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-top: 1.2rem;
        padding-top: 0.75rem;
    }

    .export-block {
        border-top: 1px solid var(--app-border);
        margin: -0.15rem 0 0;
        padding-top: 0.55rem;
    }

    .sidebar-heading {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: var(--app-text);
        font-size: 1.18rem;
        line-height: 1.15;
        font-weight: 800;
        margin: 0 0 0.72rem;
    }

    .sidebar-icon {
        width: 1.05rem;
        height: 1.05rem;
        flex: 0 0 auto;
        color: var(--app-text);
    }

    .sidebar-icon path,
    .sidebar-icon line,
    .sidebar-icon polyline,
    .sidebar-icon circle {
        stroke: currentColor;
    }

    .section-title {
        scroll-margin-top: 4rem;
        margin-top: 2.8rem;
        padding-top: 1.8rem;
        border-top: 1px solid var(--app-border);
    }

    .section-title h2 {
        font-size: clamp(1.75rem, 2.8vw, 2.55rem);
        line-height: 1.05;
        margin: 0 0 0.45rem;
    }

    .section-title p {
        color: var(--app-muted);
        font-size: 1rem;
        margin: 0 0 1.25rem;
    }

    .section-title.first {
        margin-top: 1.2rem;
        padding-top: 0;
        border-top: 0;
    }

    .corr-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border: 1px solid var(--app-border);
        border-radius: 8px;
        background: var(--app-panel);
        color: var(--app-text);
        font-size: 0.9rem;
    }

    .corr-table th,
    .corr-table td {
        padding: 0.78rem 0.9rem;
        border-bottom: 1px solid var(--app-border);
        border-right: 1px solid var(--app-border);
        text-align: right;
        color: var(--app-text);
        background: var(--app-panel);
    }

    .corr-table th {
        font-weight: 700;
        background: var(--app-soft);
        text-align: left;
    }

    .corr-table tr:last-child th,
    .corr-table tr:last-child td {
        border-bottom: 0;
    }

    .corr-table th:last-child,
    .corr-table td:last-child {
        border-right: 0;
    }

    .ai-report {
        max-width: 980px;
        color: var(--app-text);
    }

    .ai-report h1,
    .ai-report h2,
    .ai-report h3,
    .ai-report p,
    .ai-report li,
    .ai-report strong {
        color: var(--app-text) !important;
    }

    .ai-table-wrap {
        width: 100%;
        max-width: 100%;
        overflow: auto;
        border: 1px solid var(--app-border);
        border-radius: 8px;
        background: var(--app-panel);
        margin: 0.75rem 0 1rem;
    }

    .ai-table {
        width: 100%;
        min-width: 720px;
        border-collapse: collapse;
        color: var(--app-text);
        background: var(--app-panel);
        font-size: 0.92rem;
    }

    .ai-table th,
    .ai-table td {
        padding: 0.72rem 0.8rem;
        border-right: 1px solid var(--app-border);
        border-bottom: 1px solid var(--app-border);
        color: var(--app-text);
        background: var(--app-panel);
        text-align: right;
        white-space: nowrap;
    }

    .ai-table th {
        background: var(--app-soft);
        color: var(--app-muted);
        font-weight: 750;
        text-align: left;
    }

    .ai-table tr:nth-child(even) td {
        background: var(--app-soft);
    }

    .ai-table td:first-child,
    .ai-table th:first-child {
        text-align: left;
    }

    .ai-table tr:last-child td {
        border-bottom: 0;
    }

    .ai-table th:last-child,
    .ai-table td:last-child {
        border-right: 0;
    }

    div[data-testid="stPlotlyChart"] .main-svg text {
        fill: var(--app-text) !important;
    }

    div[data-testid="stPlotlyChart"] .gridlayer path {
        stroke: var(--app-border) !important;
    }

    div[data-testid="stPlotlyChart"] .zerolinelayer path,
    div[data-testid="stPlotlyChart"] .xlines-above path,
    div[data-testid="stPlotlyChart"] .ylines-above path {
        stroke: var(--app-border) !important;
    }

    div[data-testid="stChatMessage"] {
        background: transparent !important;
        color: var(--app-text) !important;
    }

    div[data-testid="stChatMessage"] *,
    div[data-testid="stChatMessageContent"] *,
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li {
        color: var(--app-text) !important;
    }

    div[data-testid="stChatMessage"] ul,
    div[data-testid="stChatMessage"] ol {
        color: var(--app-text) !important;
    }

    div[data-testid="stChatMessage"] svg,
    div[data-testid="stChatMessage"] svg * {
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
        stroke: var(--app-text) !important;
    }

    div[data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarAssistant"] {
        background: var(--app-invert) !important;
        color: var(--app-invert-text) !important;
    }

    div[role="dialog"],
    div[data-testid="stModal"],
    div[data-baseweb="modal"],
    div[data-baseweb="popover"] {
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
    }

    div[role="dialog"] *,
    div[data-testid="stModal"] *,
    div[data-baseweb="modal"] *,
    div[data-baseweb="popover"] * {
        color: var(--app-text) !important;
    }

    div[role="dialog"] button,
    div[data-testid="stModal"] button,
    div[data-baseweb="modal"] button,
    div[data-baseweb="popover"] button {
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
        opacity: 1 !important;
    }

    div[role="dialog"] button[kind="primary"],
    div[data-testid="stModal"] button[kind="primary"],
    div[data-baseweb="modal"] button[kind="primary"] {
        background: var(--app-invert) !important;
        color: var(--app-invert-text) !important;
    }

    div[role="dialog"] code,
    div[role="dialog"] pre,
    div[role="dialog"] span,
    div[role="dialog"] kbd,
    div[data-testid="stModal"] code,
    div[data-testid="stModal"] pre,
    div[data-testid="stModal"] span,
    div[data-testid="stModal"] kbd,
    div[data-baseweb="modal"] code,
    div[data-baseweb="modal"] pre,
    div[data-baseweb="modal"] span,
    div[data-baseweb="modal"] kbd,
    div[data-baseweb="popover"] code,
    div[data-baseweb="popover"] pre,
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] kbd {
        background: var(--app-soft) !important;
        color: var(--app-text) !important;
    }

    div[role="dialog"] code,
    div[role="dialog"] pre,
    div[role="dialog"] kbd,
    div[data-testid="stModal"] code,
    div[data-testid="stModal"] pre,
    div[data-testid="stModal"] kbd,
    div[data-baseweb="modal"] code,
    div[data-baseweb="modal"] pre,
    div[data-baseweb="modal"] kbd,
    div[data-baseweb="popover"] code,
    div[data-baseweb="popover"] pre,
    div[data-baseweb="popover"] kbd {
        border: 1px solid var(--app-border) !important;
        border-radius: 4px !important;
        padding: 0.08rem 0.25rem !important;
    }

    .chat-form {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="mobile-blocker">
        <div class="mobile-blocker-card">
            <strong>Vista No Disponible En Celular Vertical</strong>
            <p>Este tablero necesita una pantalla ancha para leer filtros, gráficas y análisis. Úsalo en computadora o en iPad/tablet en horizontal.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tema_streamlit = st.context.theme.get("type") if st.context.theme else None
MODO_OSCURO = tema_streamlit == "dark"

COLOR_PANEL = "#0f0f0f" if MODO_OSCURO else "#ffffff"
COLOR_TEXTO = "#f5f5f5" if MODO_OSCURO else "#0a0a0a"
COLOR_MUTED = "#b7b7b7" if MODO_OSCURO else "#5f6368"
COLOR_BORDE = "#303030" if MODO_OSCURO else "#dedede"
COLOR_GRID = "#303030" if MODO_OSCURO else "#eeeeee"
COLOR_ACENTO = "#8f8f8f"
COLOR_SECUNDARIO = "#bdbdbd"
COLOR_TERCIARIO = "#5f5f5f"
COLOR_NEUTRO = "#1c1c1c" if MODO_OSCURO else "#f4f4f4"
ESCALA_CORRELACION = (
    [[0.0, "#202020"], [0.5, "#585858"], [1.0, "#a8a8a8"]]
    if MODO_OSCURO
    else [[0.0, "#cfcfcf"], [0.5, "#f5f5f5"], [1.0, "#6f6f6f"]]
)

st.markdown(
    f"""
    <style>
    button,
    button[kind],
    div[data-testid="stBaseButton-secondary"] button {{
        border-color: var(--app-border) !important;
        background: var(--app-panel) !important;
    }}

    button:hover,
    button:focus,
    input:hover,
    input:focus,
    input:focus-visible,
    textarea:hover,
    textarea:focus,
    textarea:focus-visible,
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {{
        box-shadow: none !important;
        outline: none !important;
    }}

    button:hover,
    button:focus,
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:focus-within {{
        border-color: var(--app-invert) !important;
    }}

    [data-baseweb="tag"] {{
        background: var(--app-invert) !important;
        background-color: var(--app-invert) !important;
        color: var(--app-invert-text) !important;
    }}

    [data-baseweb="tag"] * {{
        color: var(--app-invert-text) !important;
        fill: var(--app-invert-text) !important;
        stroke: var(--app-invert-text) !important;
    }}

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] *,
    ul[role="listbox"],
    ul[role="listbox"] *,
    div[role="listbox"],
    div[role="listbox"] * {{
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
    }}

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {{
        background: var(--app-panel) !important;
        border-color: var(--app-border) !important;
    }}

    .ai-table-wrap,
    .ai-table,
    .ai-table th,
    .ai-table td {{
        transition: background-color 180ms ease, color 180ms ease, border-color 180ms ease;
    }}

    [data-testid="stNotificationContentError"],
    [data-testid="stNotificationContentWarning"] {{
        background: var(--app-soft) !important;
        color: var(--app-text) !important;
        border-color: var(--app-border) !important;
    }}

    [data-testid="stNotificationContentError"] *,
    [data-testid="stNotificationContentWarning"] *,
    div[data-testid="stAlert"] svg,
    div[data-testid="stAlert"] svg * {{
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
        stroke: var(--app-text) !important;
    }}

    div[data-testid="stForm"] {{
        border: 1px solid var(--app-border) !important;
        border-radius: 8px !important;
        background: var(--app-panel) !important;
        padding: 0.9rem !important;
    }}

    div[data-testid="stTextInput"] input {{
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        border-color: var(--app-border) !important;
        caret-color: var(--app-text) !important;
    }}

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextInput"] input:focus-visible,
    div[data-testid="stTextInput"] input:invalid,
    div[data-testid="stTextInput"] input[aria-invalid="true"],
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="input"][aria-invalid="true"],
    div[data-baseweb="input"][data-invalid="true"] {{
        border-color: var(--app-border) !important;
        box-shadow: none !important;
        outline: none !important;
    }}

    div[data-testid="stTextInput"] input::placeholder {{
        color: var(--app-muted) !important;
        opacity: 1 !important;
    }}

    div[data-testid="InputInstructions"] {{
        display: none !important;
    }}

    .chat-row {{
        display: grid;
        grid-template-columns: 2.4rem minmax(0, 1fr);
        gap: 0.8rem;
        align-items: start;
        margin: 1.1rem 0;
    }}

    .chat-avatar {{
        width: 2rem;
        height: 2rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--app-border);
        border-radius: 8px;
        background: var(--app-invert);
        color: var(--app-invert-text);
        font-size: 0.78rem;
        font-weight: 800;
        line-height: 1;
    }}

    .chat-avatar.assistant {{
        background: var(--app-soft);
        color: var(--app-text);
    }}

    .chat-content,
    .chat-content p,
    .chat-content li,
    .chat-content strong,
    .chat-content em {{
        color: var(--app-text) !important;
    }}

    .generated-chart-note {{
        border: 1px solid var(--app-border);
        border-radius: 8px;
        background: var(--app-soft);
        color: var(--app-text);
        padding: 0.75rem 0.9rem;
        margin: 1rem 0 0.5rem;
        font-size: 0.92rem;
        line-height: 1.45;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

px.defaults.template = "plotly_dark" if MODO_OSCURO else "plotly_white"
px.defaults.color_discrete_sequence = [
    COLOR_ACENTO, COLOR_SECUNDARIO, COLOR_TERCIARIO,
    "#8a8a8a", "#d6d6d6", "#737373"
]

PALETA_NEUTRA = [
    COLOR_ACENTO,
    COLOR_SECUNDARIO,
    COLOR_TERCIARIO,
]
PALETA_MUCHOS_ESTADOS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1",
    "#FDD835", "#6D4C41", "#3949AB", "#D81B60", "#7CB342", "#00897B",
    "#C0CA33", "#5E35B1", "#F4511E", "#039BE5", "#8D6E63", "#546E7A",
    "#AD1457", "#2E7D32", "#EF6C00", "#1565C0", "#9E9D24", "#4527A0",
    "#C62828", "#00695C", "#0277BD", "#558B2F", "#FFB300", "#6A1B9A",
    "#00838F", "#B71C1C", "#33691E"
]
PALETA_SEXO = {
    "Total": COLOR_ACENTO,
    "Hombres": COLOR_SECUNDARIO,
    "Mujeres": COLOR_TERCIARIO,
}


def paleta_entidades(df, columna="Entidad federativa"):
    cantidad = df[columna].nunique(dropna=True) if columna in df.columns else 0
    return PALETA_MUCHOS_ESTADOS if cantidad > 3 else PALETA_NEUTRA


def ajustar_legenda_larga(fig, df, columna="Entidad federativa"):
    cantidad = df[columna].nunique(dropna=True) if columna in df.columns else 0
    if cantidad > 12:
        fig.update_layout(
            legend=dict(
                font=dict(size=10, color=COLOR_TEXTO),
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            )
        )
    return fig


def hay_variacion_suficiente(df, columnas):
    if df.empty:
        return False
    return all(col in df.columns and df[col].nunique(dropna=True) > 1 for col in columnas)


def mostrar_no_disponible(mensaje):
    st.info(mensaje)


def aplicar_estilo_figura(fig, altura=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color=COLOR_TEXTO),
        title=dict(font=dict(size=19, color=COLOR_TEXTO)),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(color=COLOR_TEXTO)
        ),
        margin=dict(l=64, r=56, t=56, b=58)
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_BORDE,
        tickfont=dict(color=COLOR_MUTED),
        title_font=dict(color=COLOR_TEXTO),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_BORDE,
        tickfont=dict(color=COLOR_MUTED),
        title_font=dict(color=COLOR_TEXTO),
    )
    if altura:
        fig.update_layout(height=altura)
    return fig


def mostrar_tabla_correlacion(df):
    tabla_html = df.to_html(
        classes="corr-table",
        border=0,
        float_format=lambda valor: f"{valor:.3f}"
    )
    st.markdown(tabla_html, unsafe_allow_html=True)


def formato_lista_procedimiento(valores, max_items=12):
    valores = list(valores or [])
    if not valores:
        return "Sin selección"
    if len(valores) <= max_items:
        return ", ".join(map(str, valores))
    return ", ".join(map(str, valores[:max_items])) + f" ... (+{len(valores) - max_items})"


def detectar_operacion_calculo(texto):
    texto = (texto or "").lower()
    if "percentil 90" in texto or "p90" in texto:
        return "Percentil 90", "quantile(0.9)"
    if "percentil 75" in texto or "p75" in texto or "cuartil 3" in texto or "q3" in texto:
        return "Percentil 75", "quantile(0.75)"
    if "percentil 25" in texto or "p25" in texto or "cuartil 1" in texto or "q1" in texto:
        return "Percentil 25", "quantile(0.25)"
    if "rango" in texto:
        return "Rango", "max() - min()"
    if "iqr" in texto or "intercuart" in texto:
        return "Rango intercuartílico", "quantile(0.75) - quantile(0.25)"
    if "varianza" in texto:
        return "Varianza", "var()"
    if "mediana" in texto:
        return "Mediana", "median()"
    if "máximo" in texto or "maximo" in texto or "mayor" in texto:
        return "Máximo", "max()"
    if "mínimo" in texto or "minimo" in texto or "menor" in texto:
        return "Mínimo", "min()"
    if "suma" in texto or "total" in texto:
        return "Suma", "sum()"
    if "desviación" in texto or "desviacion" in texto:
        return "Desviación estándar", "std()"
    if "conteo" in texto or "cuenta" in texto:
        return "Conteo", "count()"
    if "correl" in texto or "coeficiente" in texto:
        return "Correlación", "corr()"
    return "Media", "mean()"


def filtros_procedimiento(anios_seleccionados, estados_seleccionados, sexo_percepcion, delito_master):
    return {
        "Años": formato_lista_procedimiento(sorted(anios_seleccionados)),
        "Entidades": formato_lista_procedimiento(estados_seleccionados),
        "Sexo ENVIPE": sexo_percepcion,
        "Delito transversal": delito_master or "No seleccionado",
    }


def mostrar_procedimiento(procedimiento, titulo="Ver procedimiento"):
    if not procedimiento:
        return

    with st.expander(titulo, expanded=False):
        st.markdown(f"**Fuente usada:** `{procedimiento.get('fuente', 'No disponible')}`")
        for clave, valor in procedimiento.get("filtros", {}).items():
            st.markdown(f"**{clave}:** {html.escape(str(valor))}")

        for clave in ["variable", "metricas", "operacion", "agrupacion", "tipo_grafico", "observaciones"]:
            valor = procedimiento.get(clave)
            if valor is not None:
                st.markdown(f"**{clave.replace('_', ' ').title()}:** {html.escape(str(valor))}")

        codigo = procedimiento.get("codigo")
        if codigo:
            st.code(codigo.strip(), language="python")


def construir_procedimiento_tabla_chat(
    pregunta,
    tipo,
    df_master,
    df_filtrado,
    anios_seleccionados,
    estados_seleccionados,
    sexo_percepcion,
    delito_master,
):
    texto = (pregunta or "").lower()
    usar_cruce = df_master is not None and not df_master.empty
    fuente = "df_master / cruce_360" if usar_cruce else "df_filtrado / base_filtrada"
    df_base = df_master.copy() if usar_cruce else df_filtrado[df_filtrado["Año"].isin(anios_seleccionados)].copy()

    anios_mencionados = [int(valor) for valor in re.findall(r"\b(?:201[0-9]|202[0-9])\b", texto)]
    if anios_mencionados and "Año" in df_base.columns:
        df_base = df_base[df_base["Año"].isin(anios_mencionados)]

    metricas_disponibles = [col for col in etiquetas_metricas_cruce() if col in df_base.columns]
    metricas = detectar_metricas_pedidas(texto, metricas_disponibles, minimo=1)
    if not metricas:
        metricas = metricas_disponibles
    grupos = detectar_grupos_pedidos(texto, df_base.columns)
    if not grupos:
        grupos = ["Entidad federativa", "Año"] if {"Entidad federativa", "Año"}.issubset(df_base.columns) else []

    filtros = filtros_procedimiento(anios_seleccionados, estados_seleccionados, sexo_percepcion, delito_master)

    if tipo == "calculo":
        operacion, metodo = detectar_operacion_calculo(texto)
        metrica = metricas[0] if metricas else metrica_pedida(texto)
        etiqueta = etiquetas_metricas_cruce().get(metrica, metrica)
        observaciones = int(df_base[metrica].dropna().count()) if metrica in df_base.columns else 0
        if metodo == "quantile(0.9)":
            agg_codigo = "lambda serie: serie.quantile(0.9)"
        elif metodo == "quantile(0.75)":
            agg_codigo = "lambda serie: serie.quantile(0.75)"
        elif metodo == "quantile(0.25)":
            agg_codigo = "lambda serie: serie.quantile(0.25)"
        elif metodo == "max() - min()":
            agg_codigo = "lambda serie: serie.max() - serie.min()"
        elif metodo == "quantile(0.75) - quantile(0.25)":
            agg_codigo = "lambda serie: serie.quantile(0.75) - serie.quantile(0.25)"
        elif metodo.endswith("()"):
            agg_codigo = f'"{metodo[:-2]}"'
        else:
            agg_codigo = '"mean"'
        if grupos:
            codigo = f"""
df_base = df_master.copy()  # o df_filtrado, según la fuente indicada
resultado = (
    df_base
    .groupby({grupos!r}, as_index=False)
    .agg(Valor=("{metrica}", {agg_codigo}),
         Observaciones_usadas=("{metrica}", "count"))
)
"""
        elif operacion == "Correlación":
            codigo = f"""
df_base = df_master.copy()  # o df_filtrado, según la fuente indicada
resultado = df_base[{metricas!r}].corr().round(4)
"""
        else:
            codigo = f"""
df_base = df_master.copy()  # o df_filtrado, según la fuente indicada
serie = df_base["{metrica}"].dropna()
resultado = serie.{metodo}
"""
        return {
            "fuente": fuente,
            "filtros": filtros,
            "variable": etiqueta,
            "operacion": operacion,
            "agrupacion": grupos if grupos else "Sin agrupación",
            "observaciones": observaciones,
            "codigo": codigo,
        }

    codigo = f"""
df_base = df_master.copy()  # o df_filtrado, según la fuente indicada
tabla = (
    df_base
    .groupby({grupos!r}, as_index=False)[{metricas!r}]
    .mean()
    .round(3)
)
"""
    if "correl" in texto:
        codigo = f"""
df_base = df_master.copy()  # o df_filtrado, según la fuente indicada
tabla = df_base[{metricas!r}].corr().round(4)
"""
    elif "descriptiva" in texto or "estadística" in texto or "estadistica" in texto or "resumen" in texto:
        codigo = f"""
df_base = df_master.copy()  # o df_filtrado, según la fuente indicada
tabla = df_base[{metricas!r}].describe().round(4)
"""
    elif "pivot" in texto or "pivote" in texto:
        metrica = metricas[0] if metricas else metrica_pedida(texto)
        codigo = f"""
df_base = df_master.copy()  # o df_filtrado, según la fuente indicada
tabla = df_base.pivot_table(
    index="Entidad federativa",
    columns="Año",
    values="{metrica}",
    aggfunc="mean",
).round(3)
"""

    return {
        "fuente": fuente,
        "filtros": filtros,
        "metricas": [etiquetas_metricas_cruce().get(m, m) for m in metricas],
        "agrupacion": grupos if grupos else "Sin agrupación",
        "observaciones": int(len(df_base)),
        "codigo": codigo,
    }


def construir_procedimiento_grafico_chat(
    mensaje,
    df_master,
    anios_seleccionados,
    estados_seleccionados,
    sexo_percepcion,
    delito_master,
):
    pregunta = mensaje.get("pregunta", "")
    spec = mensaje.get("spec") or {}
    texto = f"{pregunta} {spec.get('x', '')} {spec.get('y', '')} {spec.get('tipo', '')}"
    metricas = detectar_metricas_pedidas(texto, etiquetas_metricas_cruce().keys(), minimo=1)
    tipo = spec.get("tipo") or "derivado de la solicitud"
    dataset = spec.get("dataset") or ("cruce360" if df_master is not None and not df_master.empty else "base_filtrada")
    fuente = "df_master / cruce_360" if dataset in {"cruce360", "correlacion", "ranking_entidades"} else dataset

    codigo = f"""
df_base = df_master.copy()  # cruce usado por la gráfica cuando aplica
fig = crear_grafico_desde_pregunta(
    pregunta={pregunta!r},
    df_filtrado=df_filtrado,
    df_total=df_total,
    df_master=df_base,
    anios_seleccionados=anios_seleccionados,
    sexo_percepcion=sexo_percepcion,
    delito_master=delito_master,
)
"""
    if spec:
        codigo = f"""
spec = {json.dumps(spec, ensure_ascii=False, indent=4)}
fig = crear_grafico_desde_spec_ia(
    spec=spec,
    df_filtrado=df_filtrado,
    df_total=df_total,
    df_master=df_master,
    anios_seleccionados=anios_seleccionados,
    sexo_percepcion=sexo_percepcion,
    delito_master=delito_master,
)
"""

    return {
        "fuente": fuente,
        "filtros": filtros_procedimiento(anios_seleccionados, estados_seleccionados, sexo_percepcion, delito_master),
        "tipo_grafico": tipo,
        "metricas": [etiquetas_metricas_cruce().get(m, m) for m in metricas],
        "observaciones": int(len(df_master)) if df_master is not None else 0,
        "codigo": codigo,
    }


st.markdown(
    """
    <h1>Investigación: Seguridad en México</h1>
    <p class="app-subtitle">
        Análisis cruzado de incidencia delictiva, cifra negra y percepción de seguridad.
    </p>
    """,
    unsafe_allow_html=True
)

# --- CARGA DE DATOS ---
ENTIDADES_CORREGIDAS = {
    "CIUDAD DE MA\x83A\xa9XICO": "CIUDAD DE MEXICO",
    "CIUDAD DE MA\xa9XICO": "CIUDAD DE MEXICO",
    "MA\x83A\xa9XICO": "MEXICO",
    "MA\xa9XICO": "MEXICO",
    "MICHOACA\x83A\xa1N DE OCAMPO": "MICHOACAN DE OCAMPO",
    "MICHOACA\xa1N DE OCAMPO": "MICHOACAN DE OCAMPO",
    "NUEVO LEA\x83A\xb3N": "NUEVO LEON",
    "NUEVO LEA\xb3N": "NUEVO LEON",
    "QUERA\x83A\xa9TARO": "QUERETARO",
    "QUERA\xa9TARO": "QUERETARO",
    "SAN LUIS POTOSA\x83A\xad": "SAN LUIS POTOSI",
    "SAN LUIS POTOSA\xad": "SAN LUIS POTOSI",
    "YUCATA\x83A\xa1N": "YUCATAN",
    "YUCATA\xa1N": "YUCATAN",
}


def normalizar_entidades(df):
    df = df.copy()
    df["Entidad federativa"] = (
        df["Entidad federativa"]
        .astype("string")
        .str.strip()
        .replace(ENTIDADES_CORREGIDAS)
    )

    llave = ["Año", "Entidad federativa", "Sexo", "Seguridad"]
    columnas_numericas = df.select_dtypes(include="number").columns.difference(["Año"])

    return (
        df.groupby(llave, as_index=False, dropna=False)[columnas_numericas.tolist()]
        .mean()
    )


def listar_claves_secrets_streamlit():
    try:
        return sorted(str(clave) for clave in st.secrets.to_dict().keys())
    except Exception:
        try:
            return sorted(str(clave) for clave in st.secrets.keys())
        except Exception:
            return []


def guardar_diagnostico_gemini(fuente, claves_visibles, error_secrets=None):
    st.session_state["gemini_config_source"] = fuente
    st.session_state["gemini_secret_keys_visibles"] = claves_visibles
    if error_secrets:
        st.session_state["gemini_secret_error"] = str(error_secrets)
    else:
        st.session_state.pop("gemini_secret_error", None)


def obtener_gemini_api_key():
    claves_visibles = listar_claves_secrets_streamlit()
    error_secrets = None

    lectores_secrets = [
        ("st.secrets['GEMINI_API_KEY']", lambda: st.secrets["GEMINI_API_KEY"]),
        ("st.secrets.get('GEMINI_API_KEY')", lambda: st.secrets.get("GEMINI_API_KEY", "")),
        ("st.secrets.to_dict()", lambda: st.secrets.to_dict().get("GEMINI_API_KEY", "")),
    ]

    for fuente, lector in lectores_secrets:
        try:
            clave = str(lector() or "").strip()
            if clave:
                guardar_diagnostico_gemini(fuente, claves_visibles)
                return clave
        except Exception as error:
            error_secrets = error

    for nombre_variable in [
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
    ]:
        clave = str(os.environ.get(nombre_variable, "") or "").strip()
        if clave:
            guardar_diagnostico_gemini(
                f"variable de entorno {nombre_variable}",
                claves_visibles,
                error_secrets,
            )
            return clave

    guardar_diagnostico_gemini("no encontrada", claves_visibles, error_secrets)
    return ""


def diagnostico_gemini_secrets():
    fuente = st.session_state.get("gemini_config_source", "desconocida")
    claves_visibles = st.session_state.get("gemini_secret_keys_visibles", [])
    claves_txt = ", ".join(claves_visibles) if claves_visibles else "ninguna"
    partes = [
        f"Fuente detectada: {fuente}.",
        f"Secrets visibles para esta app: {claves_txt}.",
    ]
    error_secrets = st.session_state.get("gemini_secret_error")
    if error_secrets:
        partes.append(f"Error al leer secrets: {error_secrets}")
    return " ".join(partes)


def mensaje_error_gemini(api_key, detalle=None):
    if detalle:
        return detalle
    if genai is None:
        return (
            "No se pudo conectar con Gemini porque falta el paquete "
            "`google-generativeai` en el entorno donde corre Streamlit."
        )
    if not api_key:
        return (
            "No se encontró `GEMINI_API_KEY` en el runtime de esta app. "
            "En Streamlit Cloud confirma que el secret esté guardado en esta misma app, "
            "espera a que propague y reinicia el despliegue. "
            f"Diagnóstico: {diagnostico_gemini_secrets()}"
        )
    return (
        "Gemini recibió una API key, pero no pudo inicializar el modelo. "
        "Revisa que la key sea válida, que tenga acceso a Gemini API y que no tenga espacios extra."
    )


@st.cache_resource
def cargar_modelo_ia(api_key):
    if genai is None:
        return None, mensaje_error_gemini(api_key)

    if not api_key:
        return None, mensaje_error_gemini(api_key)

    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash"), None
    except Exception as error:
        return None, (
            "Gemini recibió una API key, pero no pudo inicializar el modelo. "
            f"Detalle técnico: {type(error).__name__}: {error}"
        )


def resumen_numerico(df, columnas):
    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas or df.empty:
        return "Sin datos suficientes."

    resumen = (
        df[columnas_validas]
        .describe()
        .loc[["mean", "min", "max"]]
        .round(3)
    )
    return resumen.to_string()


def limitar_tabla_texto(df, columnas=None, max_filas=12):
    if df is None or df.empty:
        return "Sin datos."

    if columnas:
        columnas = [col for col in columnas if col in df.columns]
        df = df[columnas]

    return df.head(max_filas).to_string(index=False)


def construir_contexto_datos_dashboard(
    df_filtrado,
    df_total,
    df_master,
    anios_seleccionados,
    sexo_percepcion,
    delito_master,
):
    partes = [
        "",
        CONTEXTO_CONCEPTUAL_SEGURIDAD.strip(),
        "",
        "Contexto de datos disponibles para responder preguntas especificas:",
        f"- Archivo de datos: {ARCHIVO_DATOS.name}",
        f"- Version de datos cargada (mtime ns): {ARCHIVO_DATOS.stat().st_mtime_ns if ARCHIVO_DATOS.exists() else 'desconocida'}",
        f"- Filas filtradas, todos los sexos: {len(df_filtrado):,}",
        f"- Filas filtradas, sexo Total: {len(df_total):,}",
    ]

    df_env = df_filtrado[
        (df_filtrado["Seguridad"] == "Inseguro") &
        (df_filtrado["Año"].isin(anios_seleccionados))
    ].copy()

    if not df_env.empty and "ENV_Estimaciones puntuales" in df_env.columns:
        cobertura_env = (
            df_env.assign(tiene_env=df_env["ENV_Estimaciones puntuales"].notna())
            .groupby(["Año", "Sexo"], as_index=False)["tiene_env"]
            .sum()
            .pivot(index="Año", columns="Sexo", values="tiene_env")
            .fillna(0)
            .astype(int)
        )
        partes.append("Cobertura ENVIPE por año y sexo (conteo de entidades con dato):")
        partes.append(cobertura_env.to_string())

        env_por_anio = (
            df_env[df_env["Sexo"] == sexo_percepcion]
            .groupby("Año")["ENV_Estimaciones puntuales"]
            .agg(["count", "mean", "min", "max"])
            .round(2)
        )
        partes.append(f"Resumen ENVIPE por año para sexo {sexo_percepcion}:")
        partes.append(env_por_anio.to_string())

        anios_sin_env = [
            anio for anio in sorted(anios_seleccionados)
            if anio not in env_por_anio.index or env_por_anio.loc[anio, "count"] == 0
        ]
        partes.append(f"Años seleccionados sin ENVIPE para {sexo_percepcion}: {anios_sin_env}")
    else:
        partes.append("No hay datos ENVIPE en los filtros actuales.")

    if df_master is not None and not df_master.empty:
        partes.append("")
        partes.append("Datos usados por Análisis Cruzado 360:")
        partes.append(f"- Delito transversal actual: {delito_master or 'No disponible'}")
        partes.append(f"- Filas cruzadas: {len(df_master):,}")
        partes.append("Muestra de datos cruzados:")
        partes.append(
            limitar_tabla_texto(
                df_master,
                [
                    "Entidad federativa", "Año", "Delito", "Percepcion",
                    "Cifra_Negra", "Incidencia_General", "Incidencia_Especifica"
                ],
                max_filas=14,
            )
        )

        partes.append("Resumen por año del cruce:")
        partes.append(
            df_master.groupby("Año")[
                ["Percepcion", "Cifra_Negra", "Incidencia_General", "Incidencia_Especifica"]
            ].mean().round(2).to_string()
        )

    partes.append("")
    partes.append(
        "Si el usuario pide una grafica nueva, puedes proponerla en texto; "
        "la app intentara generar una grafica validada si la solicitud usa variables disponibles."
    )

    return "\n".join(partes)


def construir_contexto_ia(
    df_filtrado,
    df_master,
    anios_seleccionados,
    estados_seleccionados,
    sexo_seleccionado,
    delito_master,
):
    entidades = sorted(df_filtrado["Entidad federativa"].dropna().unique().tolist())
    anios = sorted(df_filtrado["Año"].dropna().unique().tolist())

    contexto = [
        "Dashboard: Investigación: Seguridad en México.",
        "Variables: percepción de inseguridad ENVIPE, cifra negra, incidencia general e incidencia específica.",
        CONTEXTO_CONCEPTUAL_SEGURIDAD.strip(),
        "",
        "Filtros aplicados:",
        f"- Años seleccionados por el usuario: {sorted(anios_seleccionados)}",
        f"- Años disponibles después del filtro: {anios}",
        f"- Estados seleccionados: {estados_seleccionados}",
        f"- Entidades disponibles después del filtro: {entidades}",
        f"- Sexo usado en percepción ENVIPE: {sexo_seleccionado}",
        f"- Delito transversal seleccionado: {delito_master or 'No disponible'}",
        f"- Filas filtradas en base maestra: {len(df_filtrado):,}",
        "",
        "Resumen de percepción ENVIPE:",
    ]

    df_inseguro_ia = df_filtrado[
        (df_filtrado["Seguridad"] == "Inseguro") &
        (df_filtrado["Año"].isin(anios_seleccionados)) &
        (df_filtrado["Sexo"] == sexo_seleccionado)
    ]

    if "ENV_Estimaciones puntuales" in df_inseguro_ia.columns and not df_inseguro_ia.empty:
        percepcion_entidad = (
            df_inseguro_ia.groupby("Entidad federativa")["ENV_Estimaciones puntuales"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
        )
        contexto.append(resumen_numerico(df_inseguro_ia, ["ENV_Estimaciones puntuales"]))
        contexto.append("Promedio de percepción por entidad:")
        contexto.append(percepcion_entidad.to_string())
    else:
        contexto.append("Sin datos de percepción para los filtros.")

    contexto.append("")
    contexto.append("Resumen del análisis cruzado:")

    if df_master is not None and not df_master.empty:
        metricas = [
            "Percepcion",
            "Cifra_Negra",
            "Incidencia_General",
            "Incidencia_Especifica",
        ]
        contexto.append(f"Filas cruzadas disponibles: {len(df_master):,}")
        contexto.append(resumen_numerico(df_master, metricas))

        corr = df_master[metricas].corr().round(3)
        contexto.append("Matriz de correlación de las variables cruzadas:")
        contexto.append(corr.to_string())

        entidad_cruce = (
            df_master.groupby("Entidad federativa")[metricas]
            .mean()
            .round(2)
            .sort_values("Incidencia_Especifica", ascending=False)
        )
        contexto.append("Promedios cruzados por entidad:")
        contexto.append(entidad_cruce.to_string())
    else:
        contexto.append("No hubo intersección suficiente para el análisis cruzado.")

    return "\n".join(contexto)


def generar_analisis_ia(modelo, contexto):
    prompt = f"""
Eres analista de datos públicos de seguridad en México.
Redacta un análisis ejecutivo claro, sobrio y útil para un dashboard.

Usa exclusivamente el contexto estadístico proporcionado.
No inventes datos. Si una relación es débil, dilo explícitamente.
Evita lenguaje alarmista, recomendaciones legales y repeticiones metodológicas.

Reglas metodológicas obligatorias:
- No afirmes causalidad.
- No digas que una variable "provoca", "explica", "determina", "causa" o "demuestra" otra.
- Usa lenguaje prudente: "se observa", "sugiere", "podría estar asociado", "en los filtros seleccionados", "para este cruce".
- Distingue siempre entre percepción, cifra negra e incidencia: no son equivalentes.
- Si hay correlación, aclara que es descriptiva y depende del filtro actual.
- No califiques a una entidad como "más peligrosa" solo por una variable aislada.
- Señala cuando el número de observaciones, años o entidades sea limitado.
- Si el usuario busca una conclusión fuerte, matízala como hipótesis descriptiva, no como prueba.
- No uses fuentes externas ni supongas navegación web en vivo; usa solo el contexto conceptual y estadístico incluido.

Entrega:
1. Lectura rápida de los filtros.
2. Hallazgos principales.
3. Relación entre percepción, cifra negra e incidencia.
4. Nota de datos solo si hay un problema real de disponibilidad o variación.

Máximo 900 palabras. No cierres con listas largas de próximos cruces.

Contexto:
{contexto}
"""
    respuesta = modelo.generate_content(prompt)
    return getattr(respuesta, "text", "").strip()


def responder_chat_ia(modelo, contexto, analisis, historial, pregunta):
    historial_txt = "\n".join(
        f"{mensaje['role']}: {mensaje['content']}"
        for mensaje in historial[-8:]
        if mensaje.get("role") in {"user", "assistant"}
    )

    prompt = f"""
Eres analista de datos públicos de seguridad en México.
Responde en español, de forma clara, breve y directa, usando solo el contexto del dashboard,
el análisis ya generado y la conversación. El contexto incluye datos filtrados,
cobertura por año y muestras de las tablas usadas por las gráficas; úsalo para
responder preguntas específicas. Si el usuario pide algo que no se puede inferir
de estos datos, dilo en una frase.

Reglas de estilo:
- Máximo 2 párrafos o 4 viñetas.
- No repitas recomendaciones metodológicas salvo que el usuario las pida.
- No cierres siempre con ideas de próximos cruces.
- Si el usuario pide una gráfica o tabla, responde de forma concreta qué vas a visualizar o tabular.
- Si una petición no es estadísticamente válida literalmente, no termines en "no se puede": explica la limitación en una frase y ofrece la visualización más cercana que sí se puede construir con los datos.
- Si el usuario pide comparar estados para un solo año, usa ranking/barras de una variable elegida; si pide correlación entre estados con varios años, trátalo como correlación de series por entidad.
- Si el usuario pide incidencia delictiva junto con cifra negra general, usa Incidencia_General y Cifra_Negra TOTAL; no lo reduzcas al delito transversal seleccionado salvo que el usuario nombre ese delito.
- Si el usuario pide media, promedio, máximo, mínimo, suma, conteo o mediana, calcula con los datos del contexto y menciona que se mostrará como tabla descargable.
- No inventes datos ni cambies la variable solicitada.

Reglas metodológicas:
- Responde siempre con cautela analítica.
- No infieras causalidad.
- No presentes correlaciones como prueba.
- Si el usuario pide una conclusión fuerte, matízala.
- Usa frases como "con estos filtros", "en esta muestra", "para este cruce", "los datos sugieren".
- Distingue percepción, cifra negra e incidencia; no las trates como equivalentes.
- No digas que un estado es "más peligroso" por una sola variable. Puedes decir "presenta valores más altos en esta métrica".
- Si faltan años, entidades u observaciones, dilo de forma breve y concreta.
- No afirmes que puedes navegar la web; esta app te da el contexto conceptual y los datos filtrados.

Contexto estadístico:
{contexto}

Análisis generado:
{analisis}

Conversación previa:
{historial_txt}

Pregunta del usuario:
{pregunta}
"""
    respuesta = modelo.generate_content(prompt)
    return getattr(respuesta, "text", "").strip()


def pregunta_pide_grafico(pregunta):
    texto = pregunta.lower()
    claves = [
        "grafica", "gráfica", "grafico", "gráfico", "visualiza", "dibuja",
        "graficar", "grafíc", "plot", "hazlo", "muestralo", "muéstralo",
        "tendencia", "visualización", "visualizacion", "pastel", "pie",
        "donut", "dona", "barras", "barra", "correlacion", "correlación",
        "matriz", "heatmap", "histograma", "dispersión", "dispersion",
        "ranking", "comparacion", "comparación", "relacion", "relación",
        "contra", " vs ", "asociacion", "asociación", "caja", "boxplot",
        "violín", "violin", "area", "área", "treemap", "mapa de calor",
        "densidad"
    ]
    return any(clave in texto for clave in claves)


def respuesta_indica_grafico(respuesta):
    texto = (respuesta or "").lower()
    claves = [
        "voy a visualizar", "se visualizar", "se visualizará", "visualizaré",
        "voy a graficar", "se graficar", "gráfico", "grafico", "gráfica",
        "matriz de correlación", "dispersión", "scatter", "barras", "pastel",
        "heatmap", "mapa de correlación"
    ]
    return any(clave in texto for clave in claves)


def pregunta_continua_grafico(pregunta, historial):
    texto = (pregunta or "").lower().strip()
    if not texto:
        return False

    claves_cortas = [
        "general", "esos estados", "disponibles", "esa", "ese",
        "ahora", "lo mismo", "otra variable", "con otra variable"
    ]
    if not any(clave in texto for clave in claves_cortas):
        return False

    return any(
        mensaje.get("role") == "chart" or pregunta_pide_grafico(mensaje.get("content", ""))
        for mensaje in historial[-6:]
    )


def pregunta_pide_tabla(pregunta):
    texto = (pregunta or "").lower()
    claves = [
        "tabla", "tabulado", "cuadro", "listado", "filtra", "filtrame",
        "filtrame", "en tabla", "tabla de", "tabla con",
        "muéstrame los datos", "muestrame los datos",
        "descargar", "exportar", "csv", "pivot", "pivote", "resumen",
        "estadística", "estadistica", "descriptiva", "ordenado"
    ]
    return any(clave in texto for clave in claves)


def pregunta_pide_calculo(pregunta):
    texto = (pregunta or "").lower()
    claves = [
        "media", "promedio", "mediana", "máximo", "maximo", "mínimo",
        "minimo", "suma", "total", "desviación", "desviacion", "conteo",
        "cuenta", "calcula", "calculo", "cálculo", "varianza",
        "percentil", "cuartil", "rango", "iqr", "correlación",
        "correlacion", "coeficiente", "pendiente", "crecimiento",
        "cambio", "diferencia"
    ]
    return any(clave in texto for clave in claves)


def metrica_pedida(texto, default="Cifra_Negra"):
    texto = (texto or "").lower()
    if "percep" in texto or "inseguridad" in texto or "envipe" in texto:
        return "Percepcion"
    if "incidencia delictiva" in texto or "general" in texto or "ie_" in texto:
        return "Incidencia_General"
    if "especific" in texto or "delito" in texto or "amenaza" in texto:
        return "Incidencia_Especifica"
    if "cifra" in texto or "denuncia" in texto or "denunci" in texto:
        return "Cifra_Negra"
    return default


def etiquetas_metricas_cruce():
    return {
        "Percepcion": "Percepción de inseguridad (%)",
        "Cifra_Negra": "Cifra Negra (%)",
        "Incidencia_Especifica": "Incidencia Específica",
        "Incidencia_General": "Incidencia General",
    }


def detectar_metricas_pedidas(texto, disponibles=None, minimo=1):
    texto = (texto or "").lower()
    disponibles = (
        list(etiquetas_metricas_cruce().keys())
        if disponibles is None
        else list(disponibles)
    )
    candidatas = []
    patrones = [
        ("Percepcion", ["percep", "inseguridad", "envipe"]),
        ("Cifra_Negra", ["cifra", "negra", "denuncia", "denunci"]),
        ("Incidencia_General", ["incidencia general", "incidencia delictiva", "general", "tasa general", "ie_"]),
        ("Incidencia_Especifica", ["incidencia especifica", "incidencia específica", "especific", "tipo de delito", "amenaza"]),
    ]
    for metrica, claves in patrones:
        if metrica in disponibles and any(clave in texto for clave in claves):
            candidatas.append(metrica)
    if "todas" in texto or "todo" in texto or "variables" in texto or "métricas" in texto or "metricas" in texto:
        candidatas = [m for m in etiquetas_metricas_cruce() if m in disponibles]
    if len(candidatas) < minimo:
        for metrica in disponibles:
            if metrica not in candidatas:
                candidatas.append(metrica)
            if len(candidatas) >= minimo:
                break
    return candidatas


def detectar_grupos_pedidos(texto, columnas):
    texto = (texto or "").lower()
    grupos = []
    if any(clave in texto for clave in ["estado", "estados", "entidad", "entidades"]) and "Entidad federativa" in columnas:
        grupos.append("Entidad federativa")
    if any(clave in texto for clave in ["año", "anio", "años", "anios", "tiempo"]) and "Año" in columnas:
        grupos.append("Año")
    if "delito" in texto and "Delito" in columnas:
        grupos.append("Delito")
    if "sexo" in texto and "Sexo" in columnas:
        grupos.append("Sexo")
    if "seguridad" in texto and "Seguridad" in columnas:
        grupos.append("Seguridad")
    return grupos


def texto_pide_incidencia_y_cifra(texto):
    texto = (texto or "").lower()
    pide_incidencia = "incidencia" in texto or "delictiva" in texto
    pide_cifra = "cifra" in texto or "denuncia" in texto or "denunci" in texto
    return pide_incidencia and pide_cifra


def anios_mencionados_en_texto(texto):
    return [int(valor) for valor in re.findall(r"\b(?:201[0-9]|202[0-9])\b", texto or "")]


def pregunta_pide_todos_estados(pregunta):
    texto = (pregunta or "").lower()
    claves = [
        "todos los estados",
        "todas las entidades",
        "todos las entidades",
        "todos los estado",
        "todos los estados disponibles",
        "todos los estados de mexico",
        "todos los estados de méxico",
        "todo el pais por estado",
        "todo el país por estado",
        "las 32 entidades",
        "32 entidades",
    ]
    return any(clave in texto for clave in claves)


def quitar_agregado_nacional(df):
    if df is None or df.empty or "Entidad federativa" not in df.columns:
        return df
    return df[df["Entidad federativa"] != "ESTADOS UNIDOS MEXICANOS"].copy()


def construir_cruce_chat_desde_total(df_total_base, anios_uso, delito_master):
    if df_total_base is None or df_total_base.empty:
        return pd.DataFrame(), delito_master

    df_total_base = df_total_base[df_total_base["Año"].isin(anios_uso)].copy()

    cols_ie = [c for c in df_total_base.columns if c.startswith("IE_") and c.endswith("_Est")]
    cols_cn = [c for c in df_total_base.columns if c.startswith("CN_") and c.endswith("_Est")]
    cols_itd = [c for c in df_total_base.columns if c.startswith("ITD_") and c.endswith("_Est")]

    if not (cols_ie and cols_cn and cols_itd):
        return pd.DataFrame(), delito_master

    df_env_cross = df_total_base[
        df_total_base["Seguridad"] == "Inseguro"
    ][[
        "Entidad federativa",
        "Año",
        "ENV_Estimaciones puntuales",
    ]].drop_duplicates()
    df_env_cross = df_env_cross.rename(columns={"ENV_Estimaciones puntuales": "Percepcion"})

    df_ie_cross = df_total_base.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_ie,
        var_name="Ind",
        value_name="Incidencia_General",
    )
    df_ie_cross["Año"] = df_ie_cross["Ind"].str.extract(r"IE_(\d{4})").astype(int)
    df_ie_cross = df_ie_cross[
        df_ie_cross["Año"].isin(anios_uso)
    ][[
        "Entidad federativa",
        "Año",
        "Incidencia_General",
    ]].drop_duplicates()

    df_cn_cross = df_total_base.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_cn,
        var_name="Ind",
        value_name="Cifra_Negra",
    )
    df_cn_cross["Año"] = df_cn_cross["Ind"].str.extract(r"CN_(\d{4})").astype(int)
    df_cn_cross["Delito"] = df_cn_cross["Ind"].str.extract(r"CN_\d{4}_(.*)_Est")
    df_cn_cross = df_cn_cross[df_cn_cross["Año"].isin(anios_uso)][[
        "Entidad federativa",
        "Año",
        "Delito",
        "Cifra_Negra",
    ]].drop_duplicates()

    df_itd_cross = df_total_base.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_itd,
        var_name="Ind",
        value_name="Incidencia_Especifica",
    )
    df_itd_cross["Año"] = df_itd_cross["Ind"].str.extract(r"ITD_(\d{4})").astype(int)
    df_itd_cross["Delito"] = df_itd_cross["Ind"].str.extract(r"ITD_\d{4}_(.*)_Est")
    df_itd_cross = df_itd_cross[df_itd_cross["Año"].isin(anios_uso)][[
        "Entidad federativa",
        "Año",
        "Delito",
        "Incidencia_Especifica",
    ]].drop_duplicates()

    delitos_cross = sorted(set(df_cn_cross["Delito"].dropna()) & set(df_itd_cross["Delito"].dropna()))
    delito_uso = delito_master if delito_master in delitos_cross else (delitos_cross[0] if delitos_cross else None)
    if not delito_uso:
        return pd.DataFrame(), delito_master

    df_cn_fil = df_cn_cross[df_cn_cross["Delito"] == delito_uso]
    df_itd_fil = df_itd_cross[df_itd_cross["Delito"] == delito_uso]

    df_cruce = pd.merge(
        df_env_cross,
        df_ie_cross,
        on=["Entidad federativa", "Año"],
        how="inner",
    )
    df_cruce = pd.merge(
        df_cruce,
        df_cn_fil,
        on=["Entidad federativa", "Año"],
        how="inner",
    )
    df_cruce = pd.merge(
        df_cruce,
        df_itd_fil,
        on=["Entidad federativa", "Año", "Delito"],
        how="inner",
    )
    return df_cruce.dropna(), delito_uso


def preparar_datos_chat_por_pregunta(
    pregunta,
    df_filtrado,
    df_total,
    df_master,
    anios_seleccionados,
    delito_master,
):
    anios_uso = anios_mencionados_en_texto(pregunta) or list(anios_seleccionados)
    usar_todos_estados = pregunta_pide_todos_estados(pregunta)

    if usar_todos_estados and "df_maestro" in globals():
        df_filtrado_uso = quitar_agregado_nacional(
            df_maestro[df_maestro["Año"].isin(anios_uso)].copy()
        )
        df_total_uso = df_filtrado_uso[df_filtrado_uso["Sexo"] == "Total"].copy()
        df_master_uso, delito_uso = construir_cruce_chat_desde_total(
            df_total_uso,
            anios_uso,
            delito_master,
        )
    else:
        df_filtrado_uso = df_filtrado[df_filtrado["Año"].isin(anios_uso)].copy()
        df_total_uso = df_total[df_total["Año"].isin(anios_uso)].copy()
        df_master_uso = (
            df_master[df_master["Año"].isin(anios_uso)].copy()
            if df_master is not None and not df_master.empty and "Año" in df_master.columns
            else df_master
        )
        delito_uso = delito_master

    estados_uso = (
        sorted(df_filtrado_uso["Entidad federativa"].dropna().unique().tolist())
        if "Entidad federativa" in df_filtrado_uso.columns
        else []
    )
    return df_filtrado_uso, df_total_uso, df_master_uso, anios_uso, estados_uso, delito_uso


def preparar_cruce_agrupado(df_master, texto="", grupo_default=None, metricas=None):
    if df_master is None or df_master.empty:
        return pd.DataFrame(), [], []

    metricas = [m for m in (metricas or detectar_metricas_pedidas(texto, df_master.columns)) if m in df_master.columns]
    if not metricas:
        return pd.DataFrame(), [], []

    grupos = detectar_grupos_pedidos(texto, df_master.columns)
    if not grupos:
        grupos = grupo_default or (["Año", "Entidad federativa"] if {"Año", "Entidad federativa"}.issubset(df_master.columns) else [])

    if grupos:
        df_plot = df_master.groupby(grupos, as_index=False)[metricas].mean().round(4)
    else:
        df_plot = df_master[metricas].copy()

    return df_plot.dropna(how="all", subset=metricas), grupos, metricas


def grafico_correlacion_chat(df_master, titulo="Matriz De Correlación", metricas=None):
    if df_master is None or df_master.empty:
        return None, "No hay datos cruzados suficientes para calcular correlaciones."

    metricas = [
        metrica for metrica in (metricas or list(etiquetas_metricas_cruce().keys()))
        if metrica in df_master.columns
    ]
    if len(metricas) < 2:
        return None, "Selecciona al menos dos variables disponibles para calcular la correlación."
    if not hay_variacion_suficiente(df_master, metricas):
        return None, "No hay variación suficiente para una matriz de correlación."

    matriz = df_master[metricas].corr().round(3)
    etiquetas = etiquetas_metricas_cruce()
    matriz.index = [etiquetas[col] for col in matriz.index]
    matriz.columns = [etiquetas[col] for col in matriz.columns]

    anotaciones = []
    for fila, nombre_fila in enumerate(matriz.index):
        for columna, nombre_columna in enumerate(matriz.columns):
            valor = matriz.iloc[fila, columna]
            texto_color = (
                "#0a0a0a"
                if (MODO_OSCURO and valor >= 0.78) or (not MODO_OSCURO and valor >= 0.72)
                else COLOR_TEXTO
            )
            if not MODO_OSCURO and valor < 0.72:
                texto_color = "#0a0a0a"
            anotaciones.append(
                dict(
                    x=nombre_columna,
                    y=nombre_fila,
                    text=f"{valor:.2f}",
                    showarrow=False,
                    font=dict(color=texto_color, size=12),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=matriz.values,
            x=matriz.columns,
            y=matriz.index,
            zmin=-1,
            zmax=1,
            colorscale=ESCALA_CORRELACION,
            xgap=1,
            ygap=1,
            colorbar=dict(
                tickfont=dict(color=COLOR_TEXTO),
                title=dict(text="r", font=dict(color=COLOR_TEXTO)),
            ),
            hovertemplate="%{y}<br>%{x}<br>Correlación: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=titulo, annotations=anotaciones)
    aplicar_estilo_figura(fig, altura=480)
    fig.update_layout(margin=dict(l=170, r=92, t=66, b=112))
    fig.update_xaxes(automargin=True, tickangle=35, showgrid=False, zeroline=False)
    fig.update_yaxes(automargin=True, autorange="reversed", showgrid=False, zeroline=False)
    return fig, "Matriz de correlación generada con las variables cruzadas filtradas."


def grafico_pastel_denuncias_chat(df_master, df_total, anios_seleccionados, delito_master, titulo=None):
    promedio_cn = np.nan

    if df_master is not None and not df_master.empty and "Cifra_Negra" in df_master.columns:
        promedio_cn = df_master["Cifra_Negra"].mean()

    if not np.isfinite(promedio_cn):
        cols = [c for c in df_total.columns if c.startswith("CN_") and c.endswith("_Est")]
        if cols:
            df_cn = df_total.melt(
                id_vars=["Entidad federativa"],
                value_vars=cols,
                var_name="Indicador",
                value_name="Valor",
            )
            df_cn["Año"] = df_cn["Indicador"].str.extract(r"CN_(\d{4})").astype(int)
            df_cn["Delito"] = df_cn["Indicador"].str.extract(r"CN_\d{4}_(.*)_Est")
            df_cn = df_cn[df_cn["Año"].isin(anios_seleccionados)]
            if delito_master in set(df_cn["Delito"].dropna()):
                df_cn = df_cn[df_cn["Delito"] == delito_master]
            promedio_cn = df_cn["Valor"].mean()

    if not np.isfinite(promedio_cn):
        return None, "No hay cifra negra suficiente para calcular la proporción de denuncias."

    promedio_cn = float(np.clip(promedio_cn, 0, 100))
    df_pie = pd.DataFrame({
        "Estado Legal": ["No Denunciado (Cifra Negra)", "Denunciado Formalmente"],
        "Porcentaje": [promedio_cn, 100 - promedio_cn],
    })

    fig = px.pie(
        df_pie,
        names="Estado Legal",
        values="Porcentaje",
        hole=0.42,
        title=titulo or f"Proporción De Denuncias ({delito_master or 'Filtros Actuales'})",
        color="Estado Legal",
        color_discrete_map={
            "No Denunciado (Cifra Negra)": COLOR_ACENTO,
            "Denunciado Formalmente": COLOR_TERCIARIO,
        },
    )
    fig.update_traces(textposition="inside", textinfo="label+percent")
    aplicar_estilo_figura(fig)
    fig.update_layout(margin=dict(l=24, r=24, t=64, b=76))
    return fig, "Pastel generado con el promedio de cifra negra de los filtros actuales."


def grafico_ranking_entidades_chat(df_master, texto, titulo=None):
    if df_master is None or df_master.empty:
        return None, "No hay datos cruzados suficientes para hacer un ranking por entidad."

    metrica = metrica_pedida(texto)
    etiquetas = etiquetas_metricas_cruce()
    df_rank = (
        df_master.groupby("Entidad federativa", as_index=False)[metrica]
        .mean()
        .sort_values(metrica, ascending=False)
        .head(20)
    )

    if df_rank.empty or df_rank[metrica].nunique(dropna=True) < 2:
        return None, "No hay variación suficiente para comparar entidades."

    fig = px.bar(
        df_rank,
        x=metrica,
        y="Entidad federativa",
        orientation="h",
        title=titulo or f"Ranking Por Entidad: {etiquetas[metrica]}",
        labels={metrica: etiquetas[metrica]},
        color="Entidad federativa",
        color_discrete_sequence=paleta_entidades(df_rank),
    )
    aplicar_estilo_figura(fig, altura=max(460, 28 * len(df_rank) + 120))
    fig.update_layout(showlegend=False, margin=dict(l=170, r=42, t=64, b=54))
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig, "Barras generadas con promedios por entidad para los filtros actuales."


def grafico_correlacion_entidades_chat(df_master, texto, titulo=None):
    if df_master is None or df_master.empty:
        return None, "No hay datos cruzados suficientes para comparar entidades."

    metrica = metrica_pedida(texto)
    etiquetas = etiquetas_metricas_cruce()

    if "Año" not in df_master.columns or df_master["Año"].nunique() < 2:
        return grafico_ranking_entidades_chat(
            df_master,
            texto,
            titulo=titulo or f"Comparación Por Estado: {etiquetas.get(metrica, metrica)}",
        )

    pivote = df_master.pivot_table(
        index="Año",
        columns="Entidad federativa",
        values=metrica,
        aggfunc="mean",
    ).dropna(axis=1, thresh=2)

    if pivote.shape[1] < 2:
        return None, "No hay suficientes entidades con al menos dos años para correlacionar series estatales."

    matriz = pivote.corr().round(3)
    anotaciones = []
    for fila, nombre_fila in enumerate(matriz.index):
        for columna, nombre_columna in enumerate(matriz.columns):
            valor = matriz.iloc[fila, columna]
            anotaciones.append(
                dict(
                    x=nombre_columna,
                    y=nombre_fila,
                    text=f"{valor:.2f}",
                    showarrow=False,
                    font=dict(color=COLOR_TEXTO, size=11),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=matriz.values,
            x=matriz.columns,
            y=matriz.index,
            zmin=-1,
            zmax=1,
            colorscale=ESCALA_CORRELACION,
            xgap=1,
            ygap=1,
            colorbar=dict(
                tickfont=dict(color=COLOR_TEXTO),
                title=dict(text="r", font=dict(color=COLOR_TEXTO)),
            ),
            hovertemplate="%{y}<br>%{x}<br>Correlación: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=titulo or f"Correlación Entre Estados: {etiquetas.get(metrica, metrica)}",
        annotations=anotaciones,
    )
    aplicar_estilo_figura(fig, altura=max(480, 22 * len(matriz) + 180))
    fig.update_layout(margin=dict(l=150, r=92, t=66, b=120))
    fig.update_xaxes(automargin=True, tickangle=35, showgrid=False, zeroline=False)
    fig.update_yaxes(automargin=True, autorange="reversed", showgrid=False, zeroline=False)
    return fig, "Correlación generada entre series temporales estatales para la variable elegida."


def grafico_histograma_chat(df_master, texto, titulo=None):
    if df_master is None or df_master.empty:
        return None, "No hay datos cruzados suficientes para un histograma."

    metrica = metrica_pedida(texto)
    etiquetas = etiquetas_metricas_cruce()
    if df_master[metrica].nunique(dropna=True) < 2:
        return None, "No hay variación suficiente para un histograma."

    fig = px.histogram(
        df_master,
        x=metrica,
        nbins=18,
        title=titulo or f"Distribución: {etiquetas[metrica]}",
        labels={metrica: etiquetas[metrica]},
        color_discrete_sequence=[COLOR_ACENTO],
    )
    aplicar_estilo_figura(fig)
    return fig, "Histograma generado con los registros cruzados filtrados."


def grafico_sexo_envipe_chat(df_filtrado, anios_seleccionados, titulo=None):
    df_env = df_filtrado[
        (df_filtrado["Seguridad"] == "Inseguro") &
        (df_filtrado["Año"].isin(anios_seleccionados))
    ].copy()

    if df_env.empty or "ENV_Estimaciones puntuales" not in df_env.columns:
        return None, "No hay datos ENVIPE suficientes para comparar por sexo."

    df_sexo = (
        df_env.groupby(["Año", "Sexo"], as_index=False)["ENV_Estimaciones puntuales"]
        .mean()
        .dropna()
    )
    if df_sexo.empty or df_sexo["Sexo"].nunique() < 2:
        return None, "No hay suficientes categorías de sexo para comparar."

    fig = px.line(
        df_sexo,
        x="Año",
        y="ENV_Estimaciones puntuales",
        color="Sexo",
        markers=True,
        title=titulo or "Comparación ENVIPE Por Sexo",
        labels={"ENV_Estimaciones puntuales": "% de Inseguridad"},
        color_discrete_map=PALETA_SEXO,
    )
    fig.update_layout(yaxis_ticksuffix="%")
    aplicar_estilo_figura(fig)
    return fig, "Comparación por sexo generada con ENVIPE filtrado."


def grafico_lineas_incidencia_cifra_general_chat(df_total, anios_seleccionados, titulo=None):
    if df_total is None or df_total.empty:
        return None, "No hay datos filtrados suficientes para comparar incidencia y cifra negra general."

    cols_ie = [c for c in df_total.columns if c.startswith("IE_") and c.endswith("_Est")]
    cols_cn = [c for c in df_total.columns if c.startswith("CN_") and c.endswith("_TOTAL_Est")]

    if not cols_ie or not cols_cn:
        return None, "No hay columnas compatibles de incidencia general y cifra negra general."

    df_ie = df_total.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_ie,
        var_name="Indicador",
        value_name="Valor",
    )
    df_ie["Año"] = df_ie["Indicador"].str.extract(r"IE_(\d{4})").astype(int)
    df_ie = df_ie[df_ie["Año"].isin(anios_seleccionados)]
    df_ie["Variable"] = "Incidencia General"

    df_cn = df_total.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_cn,
        var_name="Indicador",
        value_name="Valor",
    )
    df_cn["Año"] = df_cn["Indicador"].str.extract(r"CN_(\d{4})").astype(int)
    df_cn = df_cn[df_cn["Año"].isin(anios_seleccionados)]
    df_cn["Variable"] = "Cifra Negra General"

    df_plot = (
        pd.concat([df_ie, df_cn], ignore_index=True)
        .dropna(subset=["Valor"])
        .groupby(["Entidad federativa", "Año", "Variable"], as_index=False)["Valor"]
        .mean()
    )

    if df_plot.empty or df_plot["Año"].nunique() < 2:
        return None, "No hay suficientes años para comparar incidencia general y cifra negra general."

    df_plot["Serie"] = df_plot["Entidad federativa"] + " | " + df_plot["Variable"]
    fig = px.line(
        df_plot.sort_values(["Entidad federativa", "Variable", "Año"]),
        x="Año",
        y="Valor",
        color="Entidad federativa",
        line_dash="Variable",
        markers=True,
        title=titulo or "Incidencia General Y Cifra Negra General Por Estado",
        labels={"Valor": "Valor", "Variable": "Variable"},
        color_discrete_sequence=paleta_entidades(df_plot),
    )
    aplicar_estilo_figura(fig)
    ajustar_legenda_larga(fig, df_plot)
    return fig, "Líneas generadas con incidencia general y cifra negra general, no por delito específico."


def grafico_relacion_cruce_chat(df_master, texto, titulo=None):
    if df_master is None or df_master.empty:
        return None, "No hay datos cruzados suficientes para visualizar esa relación."

    etiquetas = etiquetas_metricas_cruce()
    texto = texto.lower()

    if "cifra" in texto and ("percep" in texto or "inseguridad" in texto):
        x, y = "Percepcion", "Cifra_Negra"
    elif "general" in texto and ("percep" in texto or "inseguridad" in texto):
        x, y = "Percepcion", "Incidencia_General"
    elif "general" in texto and ("cifra" in texto or "denunci" in texto):
        x, y = "Incidencia_General", "Cifra_Negra"
    elif "especific" in texto and ("percep" in texto or "inseguridad" in texto):
        x, y = "Percepcion", "Incidencia_Especifica"
    elif "especific" in texto and ("cifra" in texto or "denunci" in texto):
        x, y = "Incidencia_Especifica", "Cifra_Negra"
    else:
        x, y = "Incidencia_Especifica", "Percepcion"

    if df_master[x].nunique(dropna=True) < 2 or df_master[y].nunique(dropna=True) < 2:
        return None, "No hay variación suficiente para graficar esa relación."

    r = df_master[[x, y]].corr().iloc[0, 1]
    fig = px.scatter(
        df_master,
        x=x,
        y=y,
        color="Entidad federativa",
        size="Cifra_Negra" if "Cifra_Negra" not in {x, y} else "Incidencia_General",
        hover_data=["Año", "Delito", "Incidencia_General"],
        title=titulo or f"{etiquetas[y]} vs {etiquetas[x]} | r = {r:.3f}",
        labels=etiquetas,
        color_discrete_sequence=paleta_entidades(df_master),
        opacity=0.78,
    )
    aplicar_estilo_figura(fig)
    ajustar_legenda_larga(fig, df_master)
    return fig, "Dispersión generada con el cruce 360 filtrado."


def grafico_generico_cruce_chat(df_master, texto, tipo="linea", titulo=None):
    if df_master is None or df_master.empty:
        return None, "No hay datos cruzados suficientes para generar esa visualización."

    etiquetas = etiquetas_metricas_cruce()
    tipo = (tipo or "linea").lower()
    metricas = detectar_metricas_pedidas(texto, df_master.columns, minimo=1)

    if tipo in {"dispersion", "scatter", "burbuja", "burbujas"}:
        return grafico_relacion_cruce_chat(df_master, texto, titulo=titulo)

    if tipo in {"correlacion", "correlación", "heatmap", "mapa_calor"}:
        return grafico_correlacion_chat(
            df_master,
            titulo=titulo or "Matriz De Correlación",
            metricas=metricas,
        )

    if tipo in {"pastel", "pie", "donut", "dona"}:
        metrica = metricas[0]
        df_pie, grupos, _ = preparar_cruce_agrupado(
            df_master,
            texto,
            grupo_default=["Entidad federativa"],
            metricas=[metrica],
        )
        if df_pie.empty or len(df_pie) < 2:
            return None, "No hay suficientes categorías para un gráfico de pastel."
        nombre = grupos[0] if grupos else "Entidad federativa"
        df_pie = df_pie.sort_values(metrica, ascending=False).head(12)
        fig = px.pie(
            df_pie,
            names=nombre,
            values=metrica,
            hole=0.42 if tipo in {"donut", "dona"} else 0,
            title=titulo or f"Composición De {etiquetas.get(metrica, metrica)}",
            color_discrete_sequence=paleta_entidades(df_pie, nombre) if nombre == "Entidad federativa" else PALETA_NEUTRA,
        )
        fig.update_traces(textposition="inside", textinfo="label+percent")
        aplicar_estilo_figura(fig)
        fig.update_layout(margin=dict(l=24, r=24, t=64, b=76))
        return fig, "Pastel generado con valores promedio de los filtros actuales."

    if tipo in {"box", "boxplot", "caja", "violin", "violín"}:
        metrica = metricas[0]
        categoria = "Entidad federativa" if "Entidad federativa" in df_master.columns else None
        if df_master[metrica].nunique(dropna=True) < 2:
            return None, "No hay variación suficiente para graficar distribución."
        if tipo in {"violin", "violín"}:
            fig = px.violin(
                df_master,
                x=categoria,
                y=metrica,
                color=categoria,
                box=True,
                points="all",
                title=titulo or f"Distribución De {etiquetas.get(metrica, metrica)}",
                labels=etiquetas,
                color_discrete_sequence=paleta_entidades(df_master) if categoria else PALETA_NEUTRA,
            )
        else:
            fig = px.box(
                df_master,
                x=categoria,
                y=metrica,
                color=categoria,
                points="all",
                title=titulo or f"Caja De {etiquetas.get(metrica, metrica)}",
                labels=etiquetas,
                color_discrete_sequence=paleta_entidades(df_master) if categoria else PALETA_NEUTRA,
            )
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_master)
        return fig, "Distribución generada con los registros cruzados filtrados."

    if tipo in {"area", "área"}:
        metrica = metricas[0]
        df_plot, grupos, _ = preparar_cruce_agrupado(
            df_master,
            texto,
            grupo_default=["Año", "Entidad federativa"],
            metricas=[metrica],
        )
        if df_plot.empty or "Año" not in df_plot.columns or df_plot["Año"].nunique() < 2:
            return None, "No hay suficientes años para un gráfico de área."
        color = "Entidad federativa" if "Entidad federativa" in df_plot.columns else None
        fig = px.area(
            df_plot.sort_values("Año"),
            x="Año",
            y=metrica,
            color=color,
            title=titulo or f"Evolución De {etiquetas.get(metrica, metrica)}",
            labels=etiquetas,
            color_discrete_sequence=paleta_entidades(df_plot) if color else PALETA_NEUTRA,
        )
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_plot)
        return fig, "Área generada con promedios por año."

    if tipo in {"barras", "barra", "bar", "ranking"}:
        metrica = metricas[0]
        df_plot, grupos, _ = preparar_cruce_agrupado(
            df_master,
            texto,
            grupo_default=["Entidad federativa"],
            metricas=[metrica],
        )
        if df_plot.empty or df_plot[metrica].nunique(dropna=True) < 2:
            return None, "No hay variación suficiente para barras."
        categoria = "Entidad federativa" if "Entidad federativa" in df_plot.columns else (grupos[0] if grupos else None)
        if categoria is None:
            return None, "No hay una dimensión categórica para barras."
        df_plot = df_plot.sort_values(metrica, ascending=False).head(30)
        fig = px.bar(
            df_plot,
            x=metrica,
            y=categoria,
            orientation="h",
            color=categoria if categoria == "Entidad federativa" else None,
            title=titulo or f"Barras De {etiquetas.get(metrica, metrica)}",
            labels={**etiquetas, categoria: categoria},
            color_discrete_sequence=paleta_entidades(df_plot, categoria) if categoria == "Entidad federativa" else PALETA_NEUTRA,
        )
        aplicar_estilo_figura(fig, altura=max(460, 28 * len(df_plot) + 120))
        fig.update_layout(showlegend=False, margin=dict(l=180, r=48, t=64, b=58))
        fig.update_yaxes(autorange="reversed", automargin=True)
        return fig, "Barras generadas con promedios de los filtros actuales."

    if tipo in {"histograma", "hist", "densidad"}:
        return grafico_histograma_chat(df_master, texto, titulo=titulo)

    # Línea por defecto. Si piden varias métricas, se pivotea para comparar.
    df_plot, grupos, metricas = preparar_cruce_agrupado(
        df_master,
        texto,
        grupo_default=["Año", "Entidad federativa"],
        metricas=metricas,
    )
    if df_plot.empty or "Año" not in df_plot.columns or df_plot["Año"].nunique() < 2:
        return None, "No hay suficientes años para una línea temporal."

    if len(metricas) > 1:
        id_vars = ["Año"]
        if "Entidad federativa" in df_plot.columns:
            id_vars.append("Entidad federativa")
        df_long = df_plot.groupby(id_vars, as_index=False)[metricas].mean().melt(
            id_vars=id_vars,
            value_vars=metricas,
            var_name="Variable",
            value_name="Valor",
        )
        df_long["Variable"] = df_long["Variable"].map(etiquetas).fillna(df_long["Variable"])
        color = "Entidad federativa" if "Entidad federativa" in df_long.columns else "Variable"
        fig = px.line(
            df_long,
            x="Año",
            y="Valor",
            color=color,
            line_dash="Variable" if color != "Variable" else None,
            markers=True,
            title=titulo or "Evolución Comparada De Variables",
            color_discrete_sequence=paleta_entidades(df_long) if color == "Entidad federativa" else PALETA_NEUTRA,
        )
    else:
        metrica = metricas[0]
        color = "Entidad federativa" if "Entidad federativa" in df_plot.columns else None
        fig = px.line(
            df_plot.sort_values("Año"),
            x="Año",
            y=metrica,
            color=color,
            markers=True,
            title=titulo or f"Evolución De {etiquetas.get(metrica, metrica)}",
            labels=etiquetas,
            color_discrete_sequence=paleta_entidades(df_plot) if color else PALETA_NEUTRA,
        )
        if metrica in {"Percepcion", "Cifra_Negra"}:
            fig.update_layout(yaxis_ticksuffix="%")

    aplicar_estilo_figura(fig)
    ajustar_legenda_larga(fig, df_plot)
    return fig, "Línea temporal generada con los datos cruzados filtrados."


def construir_tabla_chat(pregunta, df_master, df_filtrado, anios_seleccionados, delito_master):
    texto = (pregunta or "").lower()

    if df_master is not None and not df_master.empty:
        df_base = df_master.copy()
        origen = "cruce_360"
    else:
        df_base = df_filtrado[df_filtrado["Año"].isin(anios_seleccionados)].copy()
        origen = "base_filtrada"

    if df_base.empty:
        return None, "No hay datos suficientes para construir una tabla con los filtros actuales."

    anios_mencionados = [
        int(valor) for valor in re.findall(r"\b(?:201[0-9]|202[0-9])\b", texto)
    ]
    if anios_mencionados and "Año" in df_base.columns:
        df_base = df_base[df_base["Año"].isin(anios_mencionados)]
        if df_base.empty:
            return None, f"No hay datos para {', '.join(map(str, anios_mencionados))} con los filtros actuales."

    metricas = [col for col in detectar_metricas_pedidas(texto, df_base.columns) if col in df_base.columns]
    if not metricas:
        metricas = [col for col in etiquetas_metricas_cruce() if col in df_base.columns]
    dimensiones = [col for col in ["Entidad federativa", "Año", "Delito", "Sexo", "Seguridad"] if col in df_base.columns]

    grupo = detectar_grupos_pedidos(texto, df_base.columns)
    if not grupo:
        grupo = ["Entidad federativa", "Año"] if {"Entidad federativa", "Año"}.issubset(df_base.columns) else dimensiones[:2]

    if "correl" in texto and len(metricas) >= 2:
        tabla = df_base[metricas].corr().round(4).reset_index().rename(columns={"index": "Variable"})
        nota = "Tabla de correlación generada con las variables disponibles del cruce."
        return tabla, nota

    if "descriptiva" in texto or "estadística" in texto or "estadistica" in texto or "resumen" in texto:
        tabla = (
            df_base[metricas]
            .describe()
            .loc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
            .round(4)
            .reset_index()
            .rename(columns={"index": "Estadístico"})
        )
    elif "pivot" in texto or "pivote" in texto:
        if {"Entidad federativa", "Año"}.issubset(df_base.columns) and metricas:
            tabla = (
                df_base.pivot_table(
                    index="Entidad federativa",
                    columns="Año",
                    values=metricas[0],
                    aggfunc="mean",
                )
                .round(3)
                .reset_index()
            )
            tabla.columns = [str(col) for col in tabla.columns]
        else:
            tabla = df_base[dimensiones + metricas].head(200).copy()
    elif not metricas:
        columnas = dimensiones + [
            col for col in df_base.select_dtypes(include="number").columns
            if col not in {"Año"}
        ][:6]
        tabla = df_base[columnas].head(200).copy()
    else:
        tabla = (
            df_base.groupby(grupo, as_index=False)[metricas]
            .agg(["mean", "median", "min", "max", "count"] if "completa" in texto or "avanzada" in texto else "mean")
            .round(3)
        )
        if isinstance(tabla.columns, pd.MultiIndex):
            tabla.columns = [
                "_".join(str(parte) for parte in col if parte).strip("_")
                for col in tabla.columns.to_flat_index()
            ]
        tabla = tabla.sort_values(grupo)

    if "top" in texto or "mayor" in texto or "alt" in texto:
        metrica = metrica_pedida(texto, default=metricas[0] if metricas else None)
        if metrica in tabla.columns:
            tabla = tabla.sort_values(metrica, ascending=False)
    elif "menor" in texto or "baj" in texto:
        metrica = metrica_pedida(texto, default=metricas[0] if metricas else None)
        if metrica in tabla.columns:
            tabla = tabla.sort_values(metrica, ascending=True)

    tabla = tabla.head(200)
    nota = (
        f"Tabla generada con {len(tabla):,} filas desde {origen}"
        f"{f' para {delito_master}' if delito_master else ''}."
    )
    return tabla, nota


def construir_calculo_chat(pregunta, df_master, df_filtrado, anios_seleccionados):
    texto = (pregunta or "").lower()

    if df_master is not None and not df_master.empty:
        df_base = df_master.copy()
    else:
        df_base = df_filtrado[df_filtrado["Año"].isin(anios_seleccionados)].copy()

    if df_base.empty:
        return None, "No hay datos suficientes para calcular con los filtros actuales."

    anios_mencionados = [
        int(valor) for valor in re.findall(r"\b(?:201[0-9]|202[0-9])\b", texto)
    ]
    if anios_mencionados and "Año" in df_base.columns:
        df_base = df_base[df_base["Año"].isin(anios_mencionados)]
        if df_base.empty:
            return None, f"No hay datos para {', '.join(map(str, anios_mencionados))} con los filtros actuales."

    metricas_disponibles = [col for col in etiquetas_metricas_cruce() if col in df_base.columns]
    metricas_pedidas = detectar_metricas_pedidas(texto, metricas_disponibles, minimo=1)
    metrica = metricas_pedidas[0] if metricas_pedidas else metrica_pedida(texto, default="Cifra_Negra")
    if metrica not in df_base.columns:
        numericas = [col for col in df_base.select_dtypes(include="number").columns if col != "Año"]
        if not numericas:
            return None, "No encontré variables numéricas para calcular."
        metrica = numericas[0]

    if ("correl" in texto or "coeficiente" in texto) and len(metricas_pedidas) >= 2:
        variables = [m for m in metricas_pedidas if m in df_base.columns]
        tabla = df_base[variables].corr().round(4).reset_index().rename(columns={"index": "Variable"})
        return tabla, "Coeficientes de correlación calculados con los filtros actuales."

    serie = df_base[metrica].dropna()
    if serie.empty:
        return None, f"No hay valores disponibles para {metrica}."

    if "percentil 90" in texto or "p90" in texto:
        operacion = "Percentil 90"
        aggfunc = lambda serie: serie.quantile(0.9)
    elif "percentil 75" in texto or "p75" in texto or "cuartil 3" in texto or "q3" in texto:
        operacion = "Percentil 75"
        aggfunc = lambda serie: serie.quantile(0.75)
    elif "percentil 25" in texto or "p25" in texto or "cuartil 1" in texto or "q1" in texto:
        operacion = "Percentil 25"
        aggfunc = lambda serie: serie.quantile(0.25)
    elif "rango" in texto:
        operacion = "Rango"
        aggfunc = lambda serie: serie.max() - serie.min()
    elif "iqr" in texto or "intercuart" in texto:
        operacion = "Rango intercuartílico"
        aggfunc = lambda serie: serie.quantile(0.75) - serie.quantile(0.25)
    elif "varianza" in texto:
        operacion = "Varianza"
        aggfunc = "var"
    elif "mediana" in texto:
        operacion = "Mediana"
        aggfunc = "median"
    elif "máximo" in texto or "maximo" in texto or "mayor" in texto:
        operacion = "Máximo"
        aggfunc = "max"
    elif "mínimo" in texto or "minimo" in texto or "menor" in texto:
        operacion = "Mínimo"
        aggfunc = "min"
    elif "suma" in texto or "total" in texto:
        operacion = "Suma"
        aggfunc = "sum"
    elif "desviación" in texto or "desviacion" in texto:
        operacion = "Desviación estándar"
        aggfunc = "std"
    elif "conteo" in texto or "cuenta" in texto:
        operacion = "Conteo"
        aggfunc = "count"
    else:
        operacion = "Media"
        aggfunc = "mean"

    grupos = []
    if ("por estado" in texto or "por entidad" in texto or "por estados" in texto) and "Entidad federativa" in df_base.columns:
        grupos.append("Entidad federativa")
    if ("por año" in texto or "por anio" in texto or "por años" in texto) and "Año" in df_base.columns:
        grupos.append("Año")
    if "por delito" in texto and "Delito" in df_base.columns:
        grupos.append("Delito")
    if "por sexo" in texto and "Sexo" in df_base.columns:
        grupos.append("Sexo")

    if grupos:
        tabla = (
            df_base.groupby(grupos, as_index=False)
            .agg(Valor=(metrica, aggfunc), Observaciones_usadas=(metrica, "count"))
            .sort_values(grupos)
        )
        tabla.insert(len(grupos), "Cálculo", operacion)
        tabla.insert(len(grupos) + 1, "Variable", etiquetas_metricas_cruce().get(metrica, metrica))
        tabla["Valor"] = tabla["Valor"].round(4)
        nota = (
            f"{operacion} de {etiquetas_metricas_cruce().get(metrica, metrica)} "
            f"agrupada por {', '.join(grupos)}."
        )
        return tabla.head(200), nota

    if "mediana" in texto:
        operacion = "Mediana"
        valor = serie.median()
    elif "percentil 90" in texto or "p90" in texto:
        operacion = "Percentil 90"
        valor = serie.quantile(0.9)
    elif "percentil 75" in texto or "p75" in texto or "cuartil 3" in texto or "q3" in texto:
        operacion = "Percentil 75"
        valor = serie.quantile(0.75)
    elif "percentil 25" in texto or "p25" in texto or "cuartil 1" in texto or "q1" in texto:
        operacion = "Percentil 25"
        valor = serie.quantile(0.25)
    elif "rango" in texto:
        operacion = "Rango"
        valor = serie.max() - serie.min()
    elif "iqr" in texto or "intercuart" in texto:
        operacion = "Rango intercuartílico"
        valor = serie.quantile(0.75) - serie.quantile(0.25)
    elif "varianza" in texto:
        operacion = "Varianza"
        valor = serie.var()
    elif "máximo" in texto or "maximo" in texto or "mayor" in texto:
        operacion = "Máximo"
        valor = serie.max()
    elif "mínimo" in texto or "minimo" in texto or "menor" in texto:
        operacion = "Mínimo"
        valor = serie.min()
    elif "suma" in texto or "total" in texto:
        operacion = "Suma"
        valor = serie.sum()
    elif "desviación" in texto or "desviacion" in texto:
        operacion = "Desviación estándar"
        valor = serie.std()
    elif "conteo" in texto or "cuenta" in texto:
        operacion = "Conteo"
        valor = serie.count()
    else:
        operacion = "Media"
        valor = serie.mean()

    tabla = pd.DataFrame([{
        "Cálculo": operacion,
        "Variable": etiquetas_metricas_cruce().get(metrica, metrica),
        "Valor": round(float(valor), 4),
        "Observaciones usadas": int(serie.count()),
    }])
    nota = f"{operacion} calculada sobre {etiquetas_metricas_cruce().get(metrica, metrica)} con los filtros actuales."
    return tabla, nota


def crear_grafico_desde_pregunta(
    pregunta,
    df_filtrado,
    df_total,
    df_master,
    anios_seleccionados,
    sexo_percepcion,
    delito_master,
):
    if not pregunta_pide_grafico(pregunta):
        return None, None

    texto = pregunta.lower()

    if texto_pide_incidencia_y_cifra(texto) and any(
        clave in texto
        for clave in ["linea", "línea", "evolución", "evolucion", "compar", "años", "anios", "tiempo"]
    ):
        return grafico_lineas_incidencia_cifra_general_chat(
            df_total,
            anios_seleccionados,
        )

    if "correl" in texto or "matriz" in texto or "heatmap" in texto:
        if "estado" in texto or "entidad" in texto:
            return grafico_correlacion_entidades_chat(df_master, texto)
        metricas_corr = detectar_metricas_pedidas(texto, etiquetas_metricas_cruce().keys(), minimo=2)
        if "cifra" in texto and ("percep" in texto or "inseguridad" in texto or "envipe" in texto):
            metricas_corr = ["Percepcion", "Cifra_Negra"]
        return grafico_correlacion_chat(df_master, metricas=metricas_corr)

    if "caja" in texto or "boxplot" in texto:
        return grafico_generico_cruce_chat(df_master, texto, tipo="box")

    if "violin" in texto or "violín" in texto:
        return grafico_generico_cruce_chat(df_master, texto, tipo="violin")

    if "area" in texto or "área" in texto:
        return grafico_generico_cruce_chat(df_master, texto, tipo="area")

    if "relacion" in texto or "relación" in texto or " vs " in texto or "contra" in texto or "asociacion" in texto or "asociación" in texto:
        return grafico_relacion_cruce_chat(df_master, texto)

    if "pastel" in texto or "pie" in texto or "donut" in texto or "dona" in texto or "proporcion" in texto or "proporción" in texto:
        return grafico_pastel_denuncias_chat(
            df_master,
            df_total,
            anios_seleccionados,
            delito_master,
        )

    if "sexo" in texto and ("envipe" in texto or "percep" in texto or "inseguridad" in texto):
        return grafico_sexo_envipe_chat(df_filtrado, anios_seleccionados)

    if "histograma" in texto or "distribución" in texto or "distribucion" in texto:
        if "denuncia" in texto or "denunci" in texto or "pastel" in texto:
            return grafico_pastel_denuncias_chat(
                df_master,
                df_total,
                anios_seleccionados,
                delito_master,
            )
        return grafico_histograma_chat(df_master, texto)

    if "barra" in texto or "barras" in texto or "ranking" in texto or "top" in texto:
        if "sexo" in texto:
            return grafico_sexo_envipe_chat(df_filtrado, anios_seleccionados)
        return grafico_generico_cruce_chat(df_master, texto, tipo="barras")

    if "envipe" in texto or "percepcion" in texto or "percepción" in texto or "inseguridad" in texto:
        df_env = df_filtrado[
            (df_filtrado["Seguridad"] == "Inseguro") &
            (df_filtrado["Año"].isin(anios_seleccionados))
        ].copy()

        if df_env.empty:
            return None, "No hay datos ENVIPE suficientes para generar ese gráfico."

        if "cobertura" in texto or "dato" in texto or "2012" in texto or "falt" in texto:
            cobertura = (
                df_env.assign(Con_Dato=df_env["ENV_Estimaciones puntuales"].notna())
                .groupby(["Año", "Sexo"], as_index=False)["Con_Dato"]
                .sum()
            )
            if cobertura.empty:
                return None, "No hay cobertura ENVIPE para graficar."

            fig = px.bar(
                cobertura,
                x="Año",
                y="Con_Dato",
                color="Sexo",
                barmode="group",
                title="Cobertura ENVIPE Por Año Y Sexo",
                labels={"Con_Dato": "Entidades con dato"},
                color_discrete_map=PALETA_SEXO,
            )
            aplicar_estilo_figura(fig)
            return fig, "Gráfico generado desde el chat: cobertura ENVIPE por año y sexo."

        df_env = df_env[df_env["Sexo"] == sexo_percepcion]
        if df_env["Año"].nunique() < 2:
            return None, "No hay suficientes años para graficar la evolución ENVIPE."

        fig = px.line(
            df_env,
            x="Año",
            y="ENV_Estimaciones puntuales",
            color="Entidad federativa",
            markers=True,
            title=f"Evolución ENVIPE ({sexo_percepcion})",
            labels={"ENV_Estimaciones puntuales": "% de Inseguridad"},
            color_discrete_sequence=paleta_entidades(df_env),
        )
        fig.update_layout(yaxis_ticksuffix="%")
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_env)
        return fig, "Gráfico generado desde el chat: evolución de percepción ENVIPE."

    if "cifra negra" in texto or "denuncia" in texto or "denunci" in texto:
        cols = [c for c in df_total.columns if c.startswith("CN_") and c.endswith("_Est")]
        if not cols:
            return None, "No hay columnas de cifra negra disponibles para graficar."

        df_cn_chat = df_total.melt(
            id_vars=["Entidad federativa"],
            value_vars=cols,
            var_name="Indicador",
            value_name="Valor"
        )
        df_cn_chat["Año"] = df_cn_chat["Indicador"].str.extract(r"CN_(\d{4})").astype(int)
        df_cn_chat["Delito"] = df_cn_chat["Indicador"].str.extract(r"CN_\d{4}_(.*)_Est")
        df_cn_chat = df_cn_chat[df_cn_chat["Año"].isin(anios_seleccionados)]

        delito = delito_master if delito_master in set(df_cn_chat["Delito"].dropna()) else "TOTAL"
        df_cn_chat = df_cn_chat[df_cn_chat["Delito"] == delito]

        if df_cn_chat.empty or df_cn_chat["Año"].nunique() < 2:
            return None, "No hay suficientes datos de cifra negra para generar ese gráfico."

        fig = px.line(
            df_cn_chat,
            x="Año",
            y="Valor",
            color="Entidad federativa",
            markers=True,
            title=f"Cifra Negra: {delito}",
            labels={"Valor": "% Cifra Negra"},
            color_discrete_sequence=paleta_entidades(df_cn_chat),
        )
        fig.update_layout(yaxis_ticksuffix="%")
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_cn_chat)
        return fig, f"Gráfico generado desde el chat: cifra negra para {delito}."

    if "incidencia" in texto:
        if df_master is not None and not df_master.empty and ("amenaza" in texto or "delito" in texto or "especific" in texto):
            fig = px.scatter(
                df_master,
                x="Incidencia_Especifica",
                y="Cifra_Negra",
                size="Percepcion",
                color="Entidad federativa",
                hover_data=["Año", "Incidencia_General"],
                title=f"Incidencia Específica vs Cifra Negra ({delito_master})",
                color_discrete_sequence=paleta_entidades(df_master),
                opacity=0.78,
            )
            aplicar_estilo_figura(fig)
            ajustar_legenda_larga(fig, df_master)
            return fig, "Gráfico generado desde el chat: relación de incidencia específica, cifra negra y percepción."

        cols = [c for c in df_total.columns if c.startswith("IE_") and c.endswith("_Est")]
        if not cols:
            return None, "No hay columnas de incidencia general disponibles para graficar."

        df_ie_chat = df_total.melt(
            id_vars=["Entidad federativa"],
            value_vars=cols,
            var_name="Indicador",
            value_name="Tasa_Incidencia"
        )
        df_ie_chat["Año"] = df_ie_chat["Indicador"].str.extract(r"IE_(\d{4})").astype(int)
        df_ie_chat = df_ie_chat[df_ie_chat["Año"].isin(anios_seleccionados)]

        if df_ie_chat.empty or df_ie_chat["Año"].nunique() < 2:
            return None, "No hay suficientes datos de incidencia general para generar ese gráfico."

        fig = px.line(
            df_ie_chat,
            x="Año",
            y="Tasa_Incidencia",
            color="Entidad federativa",
            markers=True,
            title="Incidencia General Estatal",
            labels={"Tasa_Incidencia": "Tasa por 100k hab."},
            color_discrete_sequence=paleta_entidades(df_ie_chat),
        )
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_ie_chat)
        return fig, "Gráfico generado desde el chat: incidencia general por año."

    if any(clave in texto for clave in ["linea", "línea", "evolución", "evolucion", "comparar", "comparación", "comparacion"]):
        return grafico_generico_cruce_chat(df_master, texto, tipo="linea")

    return grafico_generico_cruce_chat(df_master, texto, tipo="linea")


def extraer_json_respuesta(texto):
    texto = (texto or "").strip()
    if texto.startswith("```"):
        texto = texto.strip("`")
        texto = texto.replace("json", "", 1).strip()

    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio == -1 or fin == -1 or fin <= inicio:
        return None

    try:
        return json.loads(texto[inicio:fin + 1])
    except json.JSONDecodeError:
        return None


def solicitar_especificacion_grafico_ia(modelo, pregunta, contexto):
    prompt = f"""
Eres un asistente de visualización para un dashboard de seguridad en México.
El usuario quiere una gráfica nueva. No escribas código Python ni JavaScript.
Devuelve SOLO un JSON válido con esta forma:

{{
  "dataset": "envipe|envipe_cobertura|sexo_envipe|cifra_negra|incidencia_general|incidencia_especifica|cruce360|correlacion|denuncias_pastel|ranking_entidades",
  "tipo": "linea|barras|dispersion|pastel|donut|correlacion|histograma|area|box|violin",
  "x": "Año|Entidad federativa|Incidencia_Especifica|Incidencia_General|Percepcion|Cifra_Negra",
  "y": "ENV_Estimaciones puntuales|Valor|Tasa_Incidencia|Percepcion|Cifra_Negra|Incidencia_Especifica|Incidencia_General|Con_Dato",
  "color": "Entidad federativa|Sexo|Año|ninguno",
  "tamano": "Percepcion|Cifra_Negra|Incidencia_Especifica|ninguno",
  "titulo": "título corto en español"
}}

Reglas:
- Usa "envipe_cobertura" si preguntan por años faltantes, disponibilidad, cobertura o si hay datos en 2012.
- Usa "envipe" para percepción/inseguridad por año o entidad.
- Usa "sexo_envipe" si piden comparar hombres, mujeres o sexo.
- Usa "cifra_negra" para denuncias, no denuncia, cifra negra.
- Usa "incidencia_general" para tasa general estatal.
- Usa "incidencia_especifica" para incidencia por delito seleccionado.
- Si piden incidencia delictiva e cifra negra en una misma línea/comparación temporal, usa "cruce360" con tipo "linea" o la vista validada más cercana; la app priorizará incidencia general y cifra negra TOTAL cuando aplique.
- Usa "cruce360" para relaciones/correlaciones entre percepción, cifra negra e incidencia.
- Usa "correlacion" si piden matriz, correlación o heatmap.
- Usa "denuncias_pastel" si piden pastel, dona, pie o proporción de denunciado/no denunciado.
- Usa "ranking_entidades" si piden barras, ranking, top o comparación por entidad.
- Usa "cruce360" con tipo "linea", "area", "box", "violin", "histograma", "barras" o "dispersion" para visualizaciones libres de las métricas cruzadas.
- Si no hay suficiente claridad, elige la vista más simple que conteste la pregunta.

Contexto disponible:
{contexto}

Pregunta:
{pregunta}
"""
    respuesta = modelo.generate_content(prompt)
    return extraer_json_respuesta(getattr(respuesta, "text", ""))


def crear_grafico_desde_spec_ia(
    spec,
    df_filtrado,
    df_total,
    df_master,
    anios_seleccionados,
    sexo_percepcion,
    delito_master,
):
    if not isinstance(spec, dict):
        return None, "No pude interpretar la especificación de gráfica que propuso la IA."

    dataset = spec.get("dataset")
    tipo = spec.get("tipo", "linea")
    titulo = spec.get("titulo") or "Gráfico Generado Por IA"
    color = spec.get("color") if spec.get("color") != "ninguno" else None
    tamano = spec.get("tamano") if spec.get("tamano") != "ninguno" else None

    if dataset == "correlacion" or tipo == "correlacion":
        variables = set(etiquetas_metricas_cruce())
        metricas_corr = [
            valor for valor in [spec.get("x"), spec.get("y"), tamano]
            if valor in variables
        ]
        return grafico_correlacion_chat(df_master, titulo, metricas=metricas_corr or None)

    if dataset == "denuncias_pastel" or tipo in {"pastel", "donut"}:
        return grafico_pastel_denuncias_chat(
            df_master,
            df_total,
            anios_seleccionados,
            delito_master,
            titulo=titulo,
        )

    if dataset == "sexo_envipe":
        return grafico_sexo_envipe_chat(
            df_filtrado,
            anios_seleccionados,
            titulo=titulo,
        )

    if dataset == "ranking_entidades":
        return grafico_ranking_entidades_chat(
            df_master,
            f"{spec.get('y', '')} {spec.get('x', '')} {titulo}",
            titulo=titulo,
        )

    if dataset == "cruce360" and tipo in {"linea", "barras", "dispersion", "histograma", "area", "box", "violin"}:
        return grafico_generico_cruce_chat(
            df_master,
            f"{spec.get('y', '')} {spec.get('x', '')} {spec.get('color', '')} {titulo}",
            tipo=tipo,
            titulo=titulo,
        )

    if tipo == "histograma":
        return grafico_histograma_chat(
            df_master,
            f"{spec.get('y', '')} {spec.get('x', '')} {titulo}",
            titulo=titulo,
        )

    if dataset in {"envipe", "envipe_cobertura"}:
        df_plot = df_filtrado[
            (df_filtrado["Seguridad"] == "Inseguro") &
            (df_filtrado["Año"].isin(anios_seleccionados))
        ].copy()

        if df_plot.empty:
            return None, "La IA pidió ENVIPE, pero no hay datos suficientes con los filtros actuales."

        if dataset == "envipe_cobertura":
            df_plot = (
                df_plot.assign(Con_Dato=df_plot["ENV_Estimaciones puntuales"].notna())
                .groupby(["Año", "Sexo"], as_index=False)["Con_Dato"]
                .sum()
            )
            fig = px.bar(
                df_plot,
                x="Año",
                y="Con_Dato",
                color="Sexo",
                barmode="group",
                title=titulo,
                labels={"Con_Dato": "Entidades con dato"},
                color_discrete_map=PALETA_SEXO,
            )
        else:
            df_plot = df_plot[df_plot["Sexo"] == sexo_percepcion]
            if df_plot["Año"].nunique() < 2:
                return None, "La IA pidió ENVIPE, pero no hay suficientes años para graficar."
            fig = px.line(
                df_plot,
                x="Año",
                y="ENV_Estimaciones puntuales",
                color="Entidad federativa",
                markers=True,
                title=titulo,
                labels={"ENV_Estimaciones puntuales": "% de Inseguridad"},
                color_discrete_sequence=paleta_entidades(df_plot),
            )
            fig.update_layout(yaxis_ticksuffix="%")
            ajustar_legenda_larga(fig, df_plot)

        aplicar_estilo_figura(fig)
        return fig, "Gráfico generado por IA con especificación validada: ENVIPE."

    if dataset == "cifra_negra":
        cols = [c for c in df_total.columns if c.startswith("CN_") and c.endswith("_Est")]
        if not cols:
            return None, "La IA pidió cifra negra, pero no hay columnas compatibles."

        df_plot = df_total.melt(
            id_vars=["Entidad federativa"],
            value_vars=cols,
            var_name="Indicador",
            value_name="Valor"
        )
        df_plot["Año"] = df_plot["Indicador"].str.extract(r"CN_(\d{4})").astype(int)
        df_plot["Delito"] = df_plot["Indicador"].str.extract(r"CN_\d{4}_(.*)_Est")
        df_plot = df_plot[df_plot["Año"].isin(anios_seleccionados)]
        delito = delito_master if delito_master in set(df_plot["Delito"].dropna()) else "TOTAL"
        df_plot = df_plot[df_plot["Delito"] == delito]

        if df_plot.empty or df_plot["Año"].nunique() < 2:
            return None, "La IA pidió cifra negra, pero no hay suficientes años para graficar."

        fig = px.line(
            df_plot,
            x="Año",
            y="Valor",
            color="Entidad federativa",
            markers=True,
            title=titulo,
            labels={"Valor": "% Cifra Negra"},
            color_discrete_sequence=paleta_entidades(df_plot),
        )
        fig.update_layout(yaxis_ticksuffix="%")
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_plot)
        return fig, "Gráfico generado por IA con especificación validada: cifra negra."

    if dataset == "incidencia_general":
        cols = [c for c in df_total.columns if c.startswith("IE_") and c.endswith("_Est")]
        if not cols:
            return None, "La IA pidió incidencia general, pero no hay columnas compatibles."

        df_plot = df_total.melt(
            id_vars=["Entidad federativa"],
            value_vars=cols,
            var_name="Indicador",
            value_name="Tasa_Incidencia"
        )
        df_plot["Año"] = df_plot["Indicador"].str.extract(r"IE_(\d{4})").astype(int)
        df_plot = df_plot[df_plot["Año"].isin(anios_seleccionados)]

        if df_plot.empty or df_plot["Año"].nunique() < 2:
            return None, "La IA pidió incidencia general, pero no hay suficientes años para graficar."

        fig = px.line(
            df_plot,
            x="Año",
            y="Tasa_Incidencia",
            color="Entidad federativa",
            markers=True,
            title=titulo,
            labels={"Tasa_Incidencia": "Tasa por 100k hab."},
            color_discrete_sequence=paleta_entidades(df_plot),
        )
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_plot)
        return fig, "Gráfico generado por IA con especificación validada: incidencia general."

    if dataset == "incidencia_especifica":
        if df_master is None or df_master.empty:
            return None, "La IA pidió incidencia específica, pero no hay cruce suficiente con los filtros actuales."

        if tipo == "barras":
            return grafico_ranking_entidades_chat(df_master, "incidencia especifica", titulo=titulo)

        df_plot = (
            df_master.groupby(["Año", "Entidad federativa"], as_index=False)["Incidencia_Especifica"]
            .mean()
            .dropna()
        )
        if df_plot.empty or df_plot["Año"].nunique() < 2:
            return None, "La IA pidió incidencia específica, pero no hay suficientes años para graficar."

        fig = px.line(
            df_plot,
            x="Año",
            y="Incidencia_Especifica",
            color="Entidad federativa",
            markers=True,
            title=titulo,
            labels={"Incidencia_Especifica": "Incidencia Específica"},
            color_discrete_sequence=paleta_entidades(df_plot),
        )
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_plot)
        return fig, "Gráfico generado por IA con especificación validada: incidencia específica."

    if dataset == "cruce360":
        if df_master is None or df_master.empty:
            return None, "La IA pidió el cruce 360, pero no hay intersección suficiente con los filtros actuales."

        variables = {
            "Percepcion",
            "Cifra_Negra",
            "Incidencia_Especifica",
            "Incidencia_General",
        }
        x = spec.get("x") if spec.get("x") in variables else "Incidencia_Especifica"
        y = spec.get("y") if spec.get("y") in variables else "Cifra_Negra"
        color = color if color in {"Entidad federativa", "Año"} else "Entidad federativa"
        tamano = tamano if tamano in variables else "Percepcion"

        if df_master[x].nunique() < 2 or df_master[y].nunique() < 2:
            return None, "La IA pidió un cruce, pero no hay variación suficiente en los ejes."

        fig = px.scatter(
            df_master,
            x=x,
            y=y,
            color=color,
            size=tamano,
            hover_data=["Entidad federativa", "Año"],
            title=titulo,
            color_discrete_sequence=paleta_entidades(df_master) if color == "Entidad federativa" else None,
            opacity=0.78,
        )
        aplicar_estilo_figura(fig)
        ajustar_legenda_larga(fig, df_master)
        return fig, "Gráfico generado por IA con especificación validada: cruce 360."

    return None, "La IA pidió un dataset que no está permitido en el tablero."


def mostrar_mensaje_chat(mensaje):
    es_usuario = mensaje["role"] == "user"
    etiqueta = "TU" if es_usuario else "IA"
    clase = "user" if es_usuario else "assistant"
    col_icono, col_texto = st.columns([0.035, 0.965])

    with col_icono:
        st.markdown(
            f'<div class="chat-avatar {clase}">{html.escape(etiqueta)}</div>',
            unsafe_allow_html=True
        )

    with col_texto:
        st.markdown('<div class="chat-content">', unsafe_allow_html=True)
        st.markdown(mensaje["content"])
        st.markdown("</div>", unsafe_allow_html=True)


def mostrar_grafico_chat(
    mensaje,
    df_filtrado,
    df_total,
    df_master,
    anios_seleccionados,
    estados_seleccionados,
    sexo_percepcion,
    delito_master,
    indice_mensaje=None,
):
    fig = None
    nota = mensaje.get("nota") or "Gráfico generado desde el chat."
    pregunta = mensaje.get("pregunta", "")
    (
        df_filtrado_uso,
        df_total_uso,
        df_master_uso,
        anios_uso,
        estados_uso,
        delito_uso,
    ) = preparar_datos_chat_por_pregunta(
        pregunta,
        df_filtrado=df_filtrado,
        df_total=df_total,
        df_master=df_master,
        anios_seleccionados=anios_seleccionados,
        delito_master=delito_master,
    )

    if mensaje.get("modo") == "spec":
        fig, nota_auto = crear_grafico_desde_spec_ia(
            spec=mensaje.get("spec"),
            df_filtrado=df_filtrado_uso,
            df_total=df_total_uso,
            df_master=df_master_uso,
            anios_seleccionados=anios_uso,
            sexo_percepcion=sexo_percepcion,
            delito_master=delito_uso,
        )
    else:
        fig, nota_auto = crear_grafico_desde_pregunta(
            pregunta=pregunta,
            df_filtrado=df_filtrado_uso,
            df_total=df_total_uso,
            df_master=df_master_uso,
            anios_seleccionados=anios_uso,
            sexo_percepcion=sexo_percepcion,
            delito_master=delito_uso,
        )

    nota = nota or nota_auto or "Gráfico generado desde el chat."
    if fig is None:
        st.info(nota_auto or "No pude reconstruir esta gráfica con los filtros actuales.")
        return

    if pregunta_pide_todos_estados(pregunta) and "Se usaron todos los estados disponibles" not in nota:
        nota = f"{nota} Se usaron todos los estados disponibles, no solo la selección lateral."

    st.markdown(
        f'<div class="generated-chart-note">{html.escape(nota)}</div>',
        unsafe_allow_html=True
    )
    key_base = json.dumps(
        {
            "role": "chart",
            "indice": indice_mensaje,
            "pregunta": pregunta,
            "modo": mensaje.get("modo"),
            "spec": mensaje.get("spec"),
        },
        sort_keys=True,
        default=str,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        key=f"chat_plot_{indice_mensaje}_{abs(hash(key_base)) % 100000000}",
    )
    mostrar_procedimiento(
        construir_procedimiento_grafico_chat(
            mensaje,
            df_master=df_master_uso,
            anios_seleccionados=anios_uso,
            estados_seleccionados=estados_uso or estados_seleccionados,
            sexo_percepcion=sexo_percepcion,
            delito_master=delito_uso,
        ),
        titulo="Ver procedimiento de la gráfica",
    )


def mostrar_tabla_chat(
    mensaje,
    df_master,
    df_filtrado,
    anios_seleccionados,
    estados_seleccionados,
    sexo_percepcion,
    delito_master,
    indice_mensaje=None,
):
    tipo = mensaje.get("tipo", "tabla")
    pregunta = mensaje.get("pregunta", "")
    (
        df_filtrado_uso,
        _df_total_uso,
        df_master_uso,
        anios_uso,
        estados_uso,
        delito_uso,
    ) = preparar_datos_chat_por_pregunta(
        pregunta,
        df_filtrado=df_filtrado,
        df_total=df_filtrado[df_filtrado["Sexo"] == "Total"].copy(),
        df_master=df_master,
        anios_seleccionados=anios_seleccionados,
        delito_master=delito_master,
    )

    if tipo == "calculo":
        tabla, nota = construir_calculo_chat(
            pregunta,
            df_master=df_master_uso,
            df_filtrado=df_filtrado_uso,
            anios_seleccionados=anios_uso,
        )
    else:
        tabla, nota = construir_tabla_chat(
            pregunta,
            df_master=df_master_uso,
            df_filtrado=df_filtrado_uso,
            anios_seleccionados=anios_uso,
            delito_master=delito_uso,
        )

    nota = mensaje.get("nota") or nota
    if tabla is None or tabla.empty:
        st.info(nota or "No pude construir la tabla con los filtros actuales.")
        return
    if pregunta_pide_todos_estados(pregunta) and "Se usaron todos los estados disponibles" not in nota:
        nota = f"{nota} Se usaron todos los estados disponibles, no solo la selección lateral."

    st.markdown(
        f'<div class="generated-chart-note">{html.escape(nota)}</div>',
        unsafe_allow_html=True
    )
    tabla_html = tabla.to_html(
        index=False,
        border=0,
        classes="ai-table",
        escape=True,
        na_rep="",
    )
    st.markdown(
        f'<div class="ai-table-wrap">{tabla_html}</div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Descargar tabla CSV",
        data=tabla.to_csv(index=False).encode("utf-8"),
        file_name=f"consulta_ia_{tipo}.csv",
        mime="text/csv",
        key=f"download_{tipo}_{indice_mensaje}_{abs(hash(json.dumps(mensaje, sort_keys=True, default=str))) % 100000000}",
    )
    mostrar_procedimiento(
        construir_procedimiento_tabla_chat(
            pregunta=pregunta,
            tipo=tipo,
            df_master=df_master_uso,
            df_filtrado=df_filtrado_uso,
            anios_seleccionados=anios_uso,
            estados_seleccionados=estados_uso or estados_seleccionados,
            sexo_percepcion=sexo_percepcion,
            delito_master=delito_uso,
        ),
        titulo="Ver procedimiento de la tabla" if tipo != "calculo" else "Ver procedimiento del cálculo",
    )


@st.cache_data
def cargar_datos(version_archivo):
    df = pd.read_parquet(ARCHIVO_DATOS)
    return normalizar_entidades(df)

try:
    df_maestro = cargar_datos(ARCHIVO_DATOS.stat().st_mtime_ns)
except FileNotFoundError:
    st.error(f"Archivo '{ARCHIVO_DATOS.name}' no encontrado.")
    st.stop()

# --- BARRA LATERAL (FILTROS GLOBALES) ---
st.sidebar.markdown(
    """
    <div class="side-brand">
        <div class="side-brand-title">Investigación</div>
        <div class="side-brand-subtitle">Seguridad en México</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-heading">
        <svg class="sidebar-icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="4" x2="20" y1="21" y2="21"></line>
            <line x1="4" x2="20" y1="14" y2="14"></line>
            <line x1="4" x2="20" y1="7" y2="7"></line>
            <line x1="10" x2="10" y1="3" y2="11"></line>
            <line x1="16" x2="16" y1="10" y2="18"></line>
        </svg>
        <span>Filtros Globales</span>
    </div>
    """,
    unsafe_allow_html=True
)

anios_disponibles = sorted(df_maestro["Año"].unique().tolist())
if "anios_globales" not in st.session_state:
    anios_guardados = ESTADO_PERSISTENTE.get("anios_globales", anios_disponibles)
    st.session_state["anios_globales"] = [
        anio for anio in anios_guardados
        if anio in anios_disponibles
    ] or anios_disponibles

anios_seleccionados = st.sidebar.multiselect(
    "Selecciona Año(s):",
    anios_disponibles,
    key="anios_globales"
)

estados_disponibles = sorted(df_maestro["Entidad federativa"].unique().tolist())
estados_default = (
    ["ESTADOS UNIDOS MEXICANOS"]
    if "ESTADOS UNIDOS MEXICANOS" in estados_disponibles
    else estados_disponibles[:3]
)
if "estados_globales" not in st.session_state:
    estados_guardados = ESTADO_PERSISTENTE.get("estados_globales", estados_default)
    st.session_state["estados_globales"] = [
        estado for estado in estados_guardados
        if estado in estados_disponibles
    ] or estados_default

estados_seleccionados = st.sidebar.multiselect(
    "Selecciona Estado(s):",
    estados_disponibles,
    key="estados_globales"
)

# Aplicar filtros base
df_filtrado = df_maestro.copy()

if estados_seleccionados:
    df_filtrado = df_filtrado[df_filtrado["Entidad federativa"].isin(estados_seleccionados)]

df_total = df_filtrado[df_filtrado["Sexo"] == "Total"].copy()

for clave_persistente in ["analisis_ia", "analisis_ia_contexto", "chat_ia", "graficos_ia_specs", "artefactos_ia_specs"]:
    if clave_persistente not in st.session_state and clave_persistente in ESTADO_PERSISTENTE:
        st.session_state[clave_persistente] = ESTADO_PERSISTENTE[clave_persistente]

if "graficos_ia" not in st.session_state:
    st.session_state["graficos_ia"] = []
if "graficos_ia_specs" not in st.session_state:
    st.session_state["graficos_ia_specs"] = []
if "artefactos_ia_specs" not in st.session_state:
    st.session_state["artefactos_ia_specs"] = []

st.sidebar.markdown(
    """
    <div class="export-block">
        <div class="sidebar-heading">
            <svg class="sidebar-icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" x2="12" y1="15" y2="3"></line>
            </svg>
            <span>Exportar</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.download_button(
    "Exportar datos filtrados",
    data=df_filtrado.to_csv(index=False).encode("utf-8"),
    file_name="seguridad_mx_filtrado.csv",
    mime="text/csv"
)

st.sidebar.markdown(
    """
    <div class="sidebar-heading">
        <svg class="sidebar-icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="8" x2="21" y1="6" y2="6"></line>
            <line x1="8" x2="21" y1="12" y2="12"></line>
            <line x1="8" x2="21" y1="18" y2="18"></line>
            <line x1="3" x2="3.01" y1="6" y2="6"></line>
            <line x1="3" x2="3.01" y1="12" y2="12"></line>
            <line x1="3" x2="3.01" y1="18" y2="18"></line>
        </svg>
        <span>Secciones</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"""
    <nav class="side-nav">
        <a href="#percepcion-envipe">Percepción ENVIPE</a>
        <a href="#cifra-negra">Cifra Negra</a>
        <a href="#incidencia-delictiva">Incidencia Delictiva</a>
        <a href="#analisis-cruzado-360">Análisis Cruzado 360</a>
        <a href="#analisis-ia">Análisis Con IA</a>
    </nav>
    <a class="reset-link" href="?reset_dashboard=1">Restablecer A Default</a>
    <div class="sidebar-version">Versión {APP_VERSION}</div>
    """,
    unsafe_allow_html=True
)

components.html(
    """
    <script>
    (() => {
        const doc = window.parent.document;
        const win = window.parent;
        const appliedThemes = new WeakMap();

        const parseRgb = (value) => {
            const match = String(value || "").match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/i);
            return match ? match.slice(1, 4).map(Number) : null;
        };

        const getVar = (name, fallback) => {
            const value = win.getComputedStyle(doc.documentElement).getPropertyValue(name).trim();
            return value || fallback;
        };

        const isDark = () => {
            const bg = parseRgb(getVar("--background-color", "")) ||
                parseRgb(win.getComputedStyle(doc.body).backgroundColor);
            if (!bg) return win.matchMedia("(prefers-color-scheme: dark)").matches;
            const [r, g, b] = bg;
            return ((r * 299 + g * 587 + b * 114) / 1000) < 128;
        };

        const syncPlotTheme = () => {
            const dark = isDark();
            const text = dark ? "#f5f5f5" : "#0a0a0a";
            const muted = dark ? "#b7b7b7" : "#5f6368";
            const grid = dark ? "#303030" : "#eeeeee";
            const border = dark ? "#303030" : "#dedede";
            const heatmapScale = dark
                ? [[0, "#202020"], [0.5, "#585858"], [1, "#a8a8a8"]]
                : [[0, "#cfcfcf"], [0.5, "#f5f5f5"], [1, "#6f6f6f"]];
            const themeName = dark ? "dark" : "light";
            const Plotly = win.Plotly;

            if (!Plotly) return;

            doc.querySelectorAll('[data-testid="stPlotlyChart"] .js-plotly-plot').forEach((chart) => {
                if (!chart || !chart.data) return;
                if (appliedThemes.get(chart) === themeName) return;
                appliedThemes.set(chart, themeName);
                Plotly.relayout(chart, {
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "font.color": text,
                    "title.font.color": text,
                    "legend.font.color": text,
                    "legend.bgcolor": "rgba(0,0,0,0)",
                    "xaxis.gridcolor": grid,
                    "xaxis.linecolor": border,
                    "xaxis.tickfont.color": muted,
                    "xaxis.title.font.color": text,
                    "yaxis.gridcolor": grid,
                    "yaxis.linecolor": border,
                    "yaxis.tickfont.color": muted,
                    "yaxis.title.font.color": text,
                    "coloraxis.colorbar.tickfont.color": muted,
                    "coloraxis.colorbar.title.font.color": text
                });

                const heatmapIndexes = chart.data
                    .map((trace, index) => trace.type === "heatmap" ? index : null)
                    .filter((index) => index !== null);

                if (heatmapIndexes.length) {
                    Plotly.restyle(chart, {
                        "colorscale": [heatmapScale]
                    }, heatmapIndexes);

                    const trace = chart.data[heatmapIndexes[0]];
                    if (trace && trace.x && trace.y && trace.z) {
                        const annotations = [];
                        trace.y.forEach((yValue, rowIndex) => {
                            trace.x.forEach((xValue, columnIndex) => {
                                const value = Number(trace.z[rowIndex][columnIndex]);
                                const annotationColor = dark
                                    ? (value >= 0.78 ? "#0a0a0a" : "#f5f5f5")
                                    : (value >= 0.72 ? "#ffffff" : "#0a0a0a");
                                annotations.push({
                                    x: xValue,
                                    y: yValue,
                                    text: Number.isFinite(value) ? value.toFixed(2) : "",
                                    showarrow: false,
                                    font: { color: annotationColor, size: 12 }
                                });
                            });
                        });
                        Plotly.relayout(chart, { annotations });
                    }
                }
            });
        };

        let timer;
        let refreshTimer;
        let sidebarOpenTimer;
        const markAiPositionNow = () => {
            win.sessionStorage.setItem("dashboardAiSendY", String(win.scrollY));
            win.sessionStorage.setItem("dashboardAiSendAt", String(Date.now()));
            win.sessionStorage.setItem("dashboardAiMoved", "0");
        };

        const markAiMoved = () => {
            const sentAt = Number(win.sessionStorage.getItem("dashboardAiSendAt") || 0);
            if (!sentAt || Date.now() - sentAt > 60000) return;
            win.sessionStorage.setItem("dashboardAiMoved", "1");
        };

        const getSidebarWidth = () => {
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar) return 0;
            const rect = sidebar.getBoundingClientRect();
            const styles = win.getComputedStyle(sidebar);
            if (styles.display === "none" || styles.visibility === "hidden") return 0;
            return rect.width;
        };

        const findSidebarOpenButton = () => {
            const selectors = [
                '[data-testid="stSidebarCollapsedControl"] button',
                'button[data-testid="stSidebarCollapsedControl"]',
                '[data-testid="collapsedControl"] button',
                'button[data-testid="collapsedControl"]',
                'button[kind="headerNoPadding"]',
                'button[data-testid="stBaseButton-headerNoPadding"]',
                'button[data-testid="baseButton-headerNoPadding"]'
            ];
            for (const selector of selectors) {
                const button = doc.querySelector(selector);
                if (button) return button;
            }
            return null;
        };

        const openSidebarFromEdge = () => {
            if (getSidebarWidth() > 80) return;
            win.clearTimeout(sidebarOpenTimer);
            sidebarOpenTimer = win.setTimeout(() => {
                if (getSidebarWidth() > 80) return;
                const button = findSidebarOpenButton();
                if (button) button.click();
            }, 90);
        };

        const ensureSidebarHotzone = () => {
            let hotzone = doc.querySelector(".sidebar-hotzone");
            if (!hotzone) {
                hotzone = doc.createElement("div");
                hotzone.className = "sidebar-hotzone";
                hotzone.setAttribute("aria-hidden", "true");
                Object.assign(hotzone.style, {
                    position: "fixed",
                    top: "0",
                    left: "0",
                    width: "24px",
                    height: "100vh",
                    zIndex: "999997",
                    background: "transparent",
                    pointerEvents: "auto"
                });
                doc.body.appendChild(hotzone);
                hotzone.addEventListener("mouseenter", openSidebarFromEdge);
                hotzone.addEventListener("mousemove", openSidebarFromEdge);
            }
        };

        const ensureRefreshOverlay = () => {
            let overlay = doc.querySelector(".refresh-fade-overlay");
            if (!overlay) {
                overlay = doc.createElement("div");
                overlay.className = "refresh-fade-overlay";
                overlay.setAttribute("aria-hidden", "true");
                doc.body.appendChild(overlay);
            }
        };

        const scheduleSync = () => {
            win.clearTimeout(timer);
            timer = win.setTimeout(syncPlotTheme, 120);
        };

        const animateRefresh = () => {
            ensureRefreshOverlay();
            doc.body.classList.add("app-refreshing-all");
            win.clearTimeout(refreshTimer);
            refreshTimer = win.setTimeout(() => {
                doc.body.classList.remove("app-refreshing-all");
            }, 900);
        };

        const markAiSendPosition = (event) => {
            const button = event.target.closest("button");
            if (!button) return;
            const text = (button.innerText || "").trim().toLowerCase();
            if (text !== "enviar") return;
            markAiPositionNow();
        };

        const markReportGeneratePosition = (event) => {
            const button = event.target.closest("button");
            if (!button) return;
            const text = (button.innerText || "").trim().toLowerCase();
            if (text !== "generar análisis con ia") return;
            win.sessionStorage.setItem("dashboardReportSendY", String(win.scrollY));
            win.sessionStorage.setItem("dashboardReportSendAt", String(Date.now()));
        };

        const maybeScrollToAiResponse = () => {
            const marker = doc.querySelector('[data-ai-autoscroll="1"]');
            if (!marker || marker.dataset.done === "1") return;

            const sentAt = Number(win.sessionStorage.getItem("dashboardAiSendAt") || 0);
            const recent = Date.now() - sentAt < 60000;
            const userStayed = win.sessionStorage.getItem("dashboardAiMoved") !== "1";

            if (recent && userStayed) {
                marker.dataset.done = "1";
                win.setTimeout(() => {
                    marker.scrollIntoView({ behavior: "smooth", block: "end" });
                    win.sessionStorage.setItem("dashboardAiMoved", "0");
                }, 160);
            }
        };

        const maybeScrollToReport = () => {
            const marker = doc.querySelector('[data-report-autoscroll="1"]');
            if (!marker || marker.dataset.done === "1") return;

            const sentAt = Number(win.sessionStorage.getItem("dashboardReportSendAt") || 0);
            const sentY = Number(win.sessionStorage.getItem("dashboardReportSendY") || 0);
            const recent = Date.now() - sentAt < 60000;
            const userStayed = Math.abs(win.scrollY - sentY) < 90;

            if (recent && userStayed) {
                marker.dataset.done = "1";
                win.setTimeout(() => {
                    marker.scrollIntoView({ behavior: "smooth", block: "start" });
                }, 160);
            }
        };

        doc.addEventListener("change", (event) => {
            if (event.target.closest('input, select, textarea, [data-baseweb="select"]')) {
                animateRefresh();
            }
        }, true);

        doc.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" || event.shiftKey) return;
            if (!event.target.closest('.chat-form input, .chat-form textarea')) return;
            markAiPositionNow();
        }, true);

        doc.addEventListener("click", (event) => {
            markAiSendPosition(event);
            markReportGeneratePosition(event);
            if (event.target.closest('button, [role="option"], [data-baseweb="tag"]')) {
                animateRefresh();
            }
        }, true);

        win.addEventListener("wheel", markAiMoved, { passive: true });
        win.addEventListener("touchmove", markAiMoved, { passive: true });
        win.addEventListener("keydown", (event) => {
            if (["ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", " "].includes(event.key)) {
                markAiMoved();
            }
        }, true);

        doc.addEventListener("mousemove", (event) => {
            if (event.clientX <= 16) openSidebarFromEdge();
        }, true);

        doc.addEventListener("click", (event) => {
            const link = event.target.closest('.side-nav a[href^="#"]');
            if (!link) return;
            const id = decodeURIComponent(link.getAttribute("href").slice(1));
            const target = doc.getElementById(id);
            if (!target) return;
            event.preventDefault();
            target.scrollIntoView({ behavior: "smooth", block: "start" });
            win.history.replaceState(null, "", `#${id}`);
        }, true);

        new MutationObserver(() => {
            scheduleSync();
            ensureSidebarHotzone();
            ensureRefreshOverlay();
            maybeScrollToAiResponse();
            maybeScrollToReport();
        }).observe(doc.body, {
            attributes: true,
            childList: true,
            subtree: true
        });

        win.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", scheduleSync);
        ensureSidebarHotzone();
        ensureRefreshOverlay();
        scheduleSync();
        maybeScrollToAiResponse();
        maybeScrollToReport();
    })();
    </script>
    """,
    height=0,
)

# --- TAB 1: PERCEPCIÓN ENVIPE ---
st.markdown("""
    <a id="percepcion-envipe"></a>
    <section class="section-title first">
        <h2>Percepción ENVIPE</h2>
        <p>Evolución de la percepción de inseguridad por entidad y sexo.</p>
    </section>
""", unsafe_allow_html=True)
st.subheader("Evolución de la Percepción de Inseguridad")

if "sexo_percepcion" not in st.session_state:
    sexo_guardado = ESTADO_PERSISTENTE.get("sexo_percepcion", "Total")
    st.session_state["sexo_percepcion"] = (
        sexo_guardado if sexo_guardado in ["Total", "Hombres", "Mujeres"] else "Total"
    )

sexo_percepcion = st.selectbox(
    "Sexo para percepción:",
    ["Total", "Hombres", "Mujeres"],
    key="sexo_percepcion"
)

df_inseguro = df_filtrado[
    (df_filtrado["Seguridad"] == "Inseguro") &
    (df_filtrado["Sexo"] == sexo_percepcion) &
    (df_filtrado["Año"].isin(anios_seleccionados))
]

if (
    not df_inseguro.empty and
    "ENV_Estimaciones puntuales" in df_inseguro.columns and
    df_inseguro["Año"].nunique() >= 2
):
    fig_envipe = px.line(
        df_inseguro,
        x="Año",
        y="ENV_Estimaciones puntuales",
        color="Entidad federativa",
        markers=True,
        title=f"Tendencia de Inseguridad Percibida ({sexo_percepcion})",
        labels={"ENV_Estimaciones puntuales": "% de Inseguridad"},
        color_discrete_sequence=paleta_entidades(df_inseguro),
    )

    fig_envipe.update_layout(yaxis_ticksuffix="%")
    aplicar_estilo_figura(fig_envipe)
    ajustar_legenda_larga(fig_envipe, df_inseguro)
    st.plotly_chart(fig_envipe, width="stretch", theme=None)
else:
    mostrar_no_disponible("No hay suficientes años o datos de ENVIPE para dibujar la evolución.")

df_sexo_env = df_maestro[
    (df_maestro["Seguridad"] == "Inseguro") &
    (df_maestro["Año"].isin(anios_seleccionados))
].copy()

if estados_seleccionados:
    df_sexo_env = df_sexo_env[df_sexo_env["Entidad federativa"].isin(estados_seleccionados)]

if not df_sexo_env.empty and "ENV_Estimaciones puntuales" in df_sexo_env.columns:
    st.markdown("### Comparación Por Sexo")
    st.caption("La variable de sexo se usa de forma más directa en ENVIPE; por eso este bloque compara Total, Hombres y Mujeres sin depender del selector lateral.")

    entidades_comparacion = df_sexo_env["Entidad federativa"].nunique()
    if entidades_comparacion <= 6:
        df_sexo_plot = (
            df_sexo_env.groupby(["Entidad federativa", "Sexo"], as_index=False)
            ["ENV_Estimaciones puntuales"]
            .mean()
        )
        fig_sexo = px.bar(
            df_sexo_plot,
            x="Entidad federativa",
            y="ENV_Estimaciones puntuales",
            color="Sexo",
            barmode="group",
            title="Promedio De Percepción De Inseguridad Por Sexo",
            labels={"ENV_Estimaciones puntuales": "% de Inseguridad"},
            color_discrete_map=PALETA_SEXO,
        )
    else:
        fig_sexo = px.box(
            df_sexo_env,
            x="Sexo",
            y="ENV_Estimaciones puntuales",
            color="Sexo",
            points="outliers",
            title="Distribución De Percepción De Inseguridad Por Sexo",
            labels={"ENV_Estimaciones puntuales": "% de Inseguridad"},
            color_discrete_map=PALETA_SEXO,
        )

    fig_sexo.update_layout(yaxis_ticksuffix="%")
    aplicar_estilo_figura(fig_sexo)
    st.plotly_chart(fig_sexo, width="stretch", theme=None)

# --- SECCION 2: CIFRA NEGRA ---
st.markdown("""
    <a id="cifra-negra"></a>
    <section class="section-title">
        <h2>Cifra Negra</h2>
        <p>Tendencias por tipo de delito y porcentaje de delitos no denunciados o no investigados.</p>
    </section>
""", unsafe_allow_html=True)
st.subheader("Evolución de la Cifra Negra por Tipo de Delito")

cols_cn = [
    c for c in df_maestro.columns
    if c.startswith("CN_") and c.endswith("_Est")
]

if cols_cn:
    df_cn = df_total.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_cn,
        var_name="Indicador",
        value_name="Valor"
    )

    df_cn["Año"] = df_cn["Indicador"].str.extract(r"CN_(\d{4})").astype(int)
    df_cn["Delito"] = df_cn["Indicador"].str.extract(r"CN_\d{4}_(.*)_Est")

    df_cn = df_cn.drop_duplicates(
        subset=["Entidad federativa", "Año", "Delito"]
    )

    df_cn = df_cn[df_cn["Año"].isin(anios_seleccionados)]

    delitos_cn = sorted(df_cn["Delito"].dropna().unique())

    if delitos_cn:
        delito_sel_cn = st.selectbox(
            "Clasificación del delito (Cifra Negra):",
            delitos_cn,
            key="tab2_delito"
        )

        df_cn_plot = df_cn[df_cn["Delito"] == delito_sel_cn].sort_values("Año")

        if not df_cn_plot.empty and df_cn_plot["Año"].nunique() >= 2:
            fig_cn = px.line(
                df_cn_plot,
                x="Año",
                y="Valor",
                color="Entidad federativa",
                markers=True,
                title=f"Tendencia de Cifra Negra: {delito_sel_cn}",
                labels={"Valor": "% Cifra Negra"},
                color_discrete_sequence=paleta_entidades(df_cn_plot),
            )

            fig_cn.update_layout(yaxis_ticksuffix="%")
            aplicar_estilo_figura(fig_cn)
            ajustar_legenda_larga(fig_cn, df_cn_plot)
            st.plotly_chart(fig_cn, width="stretch", theme=None)
        else:
            mostrar_no_disponible("No hay suficientes años o datos de cifra negra para el delito seleccionado.")
    else:
        mostrar_no_disponible("No se detectaron delitos de cifra negra.")
else:
    mostrar_no_disponible("No se encontraron columnas de Cifra Negra con formato CN_..._Est.")

# --- SECCION 3: INCIDENCIA DELICTIVA ---
st.markdown("""
    <a id="incidencia-delictiva"></a>
    <section class="section-title">
        <h2>Incidencia Delictiva</h2>
        <p>Comparación de incidencia general y específica por entidad, año y delito.</p>
    </section>
""", unsafe_allow_html=True)
st.subheader("Evolución de la Incidencia Delictiva")

nivel_incidencia = st.radio(
    "Selecciona el nivel de análisis:",
    ["Incidencia General Estatal (IE)", "Incidencia por Tipo de Delito (ITD)"]
)

if "IE" in nivel_incidencia:
    cols_ie = [
        c for c in df_maestro.columns
        if c.startswith("IE_") and c.endswith("_Est")
    ]

    if cols_ie:
        df_ie = df_total.melt(
            id_vars=["Entidad federativa"],
            value_vars=cols_ie,
            var_name="Indicador",
            value_name="Tasa_Incidencia"
        )

        df_ie["Año"] = df_ie["Indicador"].str.extract(r"IE_(\d{4})").astype(int)

        df_ie = df_ie.drop_duplicates(
            subset=["Entidad federativa", "Año"]
        ).sort_values("Año")

        df_ie = df_ie[df_ie["Año"].isin(anios_seleccionados)]

        if not df_ie.empty and df_ie["Año"].nunique() >= 2:
            fig_ie = px.line(
                df_ie,
                x="Año",
                y="Tasa_Incidencia",
                color="Entidad federativa",
                markers=True,
                title="Evolución de la Tasa de Incidencia General (IE)",
                labels={"Tasa_Incidencia": "Tasa por 100k hab."},
                color_discrete_sequence=paleta_entidades(df_ie),
            )

            aplicar_estilo_figura(fig_ie)
            ajustar_legenda_larga(fig_ie, df_ie)
            st.plotly_chart(fig_ie, width="stretch", theme=None)
        else:
            mostrar_no_disponible("No hay suficientes años o datos de incidencia general para los filtros seleccionados.")
    else:
        mostrar_no_disponible("No se encontraron columnas de Incidencia General con formato IE_..._Est.")

else:
    cols_itd = [
        c for c in df_maestro.columns
        if c.startswith("ITD_") and c.endswith("_Est")
    ]

    if cols_itd:
        df_itd = df_total.melt(
            id_vars=["Entidad federativa"],
            value_vars=cols_itd,
            var_name="Indicador",
            value_name="Tasa_Incidencia"
        )

        df_itd["Año"] = df_itd["Indicador"].str.extract(r"ITD_(\d{4})").astype(int)
        df_itd["Delito"] = df_itd["Indicador"].str.extract(r"ITD_\d{4}_(.*)_Est")

        df_itd = df_itd.drop_duplicates(
            subset=["Entidad federativa", "Año", "Delito"]
        )

        df_itd = df_itd[df_itd["Año"].isin(anios_seleccionados)]

        delitos_itd = sorted(df_itd["Delito"].dropna().unique())

        if delitos_itd:
            delito_sel_itd = st.selectbox(
                "Selecciona el Tipo de Delito (Incidencia):",
                delitos_itd,
                key="tab3_delito"
            )

            df_itd_plot = df_itd[df_itd["Delito"] == delito_sel_itd].sort_values("Año")

            variacion_itd_por_entidad = (
                df_itd_plot.groupby("Año")["Tasa_Incidencia"].nunique(dropna=True).max()
                if not df_itd_plot.empty
                else 0
            )

            if df_itd_plot.empty or df_itd_plot["Año"].nunique() < 2:
                mostrar_no_disponible("No hay suficientes años o datos para el delito seleccionado.")
            elif df_itd_plot["Entidad federativa"].nunique() > 1 and variacion_itd_por_entidad <= 1:
                mostrar_no_disponible(
                    "Este delito no muestra variación por entidad en la base para los filtros actuales; "
                    "por eso no se dibuja una comparación estatal."
                )
            else:
                fig_itd = px.line(
                    df_itd_plot,
                    x="Año",
                    y="Tasa_Incidencia",
                    color="Entidad federativa",
                    markers=True,
                    title=f"Incidencia Específica: {delito_sel_itd} (ITD)",
                    labels={"Tasa_Incidencia": "Tasa por 100k hab."},
                    color_discrete_sequence=paleta_entidades(df_itd_plot),
                )

                aplicar_estilo_figura(fig_itd)
                ajustar_legenda_larga(fig_itd, df_itd_plot)
                st.plotly_chart(fig_itd, width="stretch", theme=None)
        else:
            mostrar_no_disponible("No se detectaron delitos de incidencia específica.")
    else:
        mostrar_no_disponible("No se encontraron columnas de Incidencia por Tipo de Delito con formato ITD_..._Est.")

# --- SECCION 4: ANALISIS CRUZADO 360 ---
st.markdown("""
    <a id="analisis-cruzado-360"></a>
    <section class="section-title">
        <h2>Análisis Cruzado 360</h2>
        <p>Relación entre percepción de inseguridad, incidencia delictiva y cifra negra.</p>
    </section>
""", unsafe_allow_html=True)
st.info(NOTA_METODOLOGICA)
st.subheader("Análisis Cruzado Total: Percepción vs Incidencia vs Cifra Negra")
st.markdown("Cruzamos todas las variables disponibles por Entidad, Año y Tipo de Delito.")

# 1. Preparar Percepción
df_env_cross = df_total[
    df_total["Seguridad"] == "Inseguro"
][[
    "Entidad federativa",
    "Año",
    "ENV_Estimaciones puntuales"
]].drop_duplicates()

df_env_cross = df_env_cross.rename(
    columns={"ENV_Estimaciones puntuales": "Percepcion"}
)

# 2. Preparar Incidencia General
cols_ie = [
    c for c in df_maestro.columns
    if c.startswith("IE_") and c.endswith("_Est")
]

# 3. Preparar Cifra Negra
cols_cn = [
    c for c in df_maestro.columns
    if c.startswith("CN_") and c.endswith("_Est")
]

# 4. Preparar Incidencia Específica
cols_itd = [
    c for c in df_maestro.columns
    if c.startswith("ITD_") and c.endswith("_Est")
]

df_master = pd.DataFrame()
delito_master = None

if cols_ie and cols_cn and cols_itd:
    df_ie_cross = df_total.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_ie,
        var_name="Ind",
        value_name="Incidencia_General"
    )

    df_ie_cross["Año"] = df_ie_cross["Ind"].str.extract(r"IE_(\d{4})").astype(int)

    df_ie_cross = df_ie_cross[[
        "Entidad federativa",
        "Año",
        "Incidencia_General"
    ]].drop_duplicates()

    df_cn_cross = df_total.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_cn,
        var_name="Ind",
        value_name="Cifra_Negra"
    )

    df_cn_cross["Año"] = df_cn_cross["Ind"].str.extract(r"CN_(\d{4})").astype(int)
    df_cn_cross["Delito"] = df_cn_cross["Ind"].str.extract(r"CN_\d{4}_(.*)_Est")

    df_cn_cross = df_cn_cross[[
        "Entidad federativa",
        "Año",
        "Delito",
        "Cifra_Negra"
    ]].drop_duplicates()

    df_itd_cross = df_total.melt(
        id_vars=["Entidad federativa"],
        value_vars=cols_itd,
        var_name="Ind",
        value_name="Incidencia_Especifica"
    )

    df_itd_cross["Año"] = df_itd_cross["Ind"].str.extract(r"ITD_(\d{4})").astype(int)
    df_itd_cross["Delito"] = df_itd_cross["Ind"].str.extract(r"ITD_\d{4}_(.*)_Est")

    df_itd_cross = df_itd_cross[[
        "Entidad federativa",
        "Año",
        "Delito",
        "Incidencia_Especifica"
    ]].drop_duplicates()

    delitos_cross = sorted(
        list(
            set(df_cn_cross["Delito"].dropna()) &
            set(df_itd_cross["Delito"].dropna())
        )
    )

    if delitos_cross:
        if "tab4_delito_master" not in st.session_state:
            delito_guardado = ESTADO_PERSISTENTE.get("tab4_delito_master")
            if delito_guardado in delitos_cross:
                st.session_state["tab4_delito_master"] = delito_guardado

        delito_master = st.selectbox(
            "Selecciona un delito para el análisis transversal:",
            delitos_cross,
            key="tab4_delito_master"
        )

        df_cn_fil = df_cn_cross[
            (df_cn_cross["Delito"] == delito_master) &
            (df_cn_cross["Año"].isin(anios_seleccionados))
        ]

        df_itd_fil = df_itd_cross[
            (df_itd_cross["Delito"] == delito_master) &
            (df_itd_cross["Año"].isin(anios_seleccionados))
        ]

        df_master = pd.merge(
            df_env_cross,
            df_ie_cross,
            on=["Entidad federativa", "Año"],
            how="inner"
        )

        df_master = pd.merge(
            df_master,
            df_cn_fil,
            on=["Entidad federativa", "Año"],
            how="inner"
        )

        df_master = pd.merge(
            df_master,
            df_itd_fil,
            on=["Entidad federativa", "Año", "Delito"],
            how="inner"
        )

        df_master = df_master.dropna()

        if not df_master.empty:
            st.divider()

            # --- GRÁFICA 1: BURBUJAS ---
            st.markdown("### 1. El Panorama Completo (Burbujas)")
            st.caption(
                "Eje X: Incidencia Específica | "
                "Eje Y: Cifra Negra | "
                "Tamaño: Percepción de Inseguridad | "
                "Color: Entidad"
            )

            fig_bubble = px.scatter(
                df_master,
                x="Incidencia_Especifica",
                y="Cifra_Negra",
                size="Percepcion",
                color="Entidad federativa",
                hover_name="Entidad federativa",
                hover_data=["Año", "Incidencia_General"],
                title=f"Relación Multivariable - {delito_master}",
                labels={
                    "Incidencia_Especifica": "Incidencia Específica",
                    "Cifra_Negra": "Cifra Negra (%)",
                    "Percepcion": "Percepción De Inseguridad (%)",
                },
                color_discrete_sequence=paleta_entidades(df_master),
                opacity=0.78,
                size_max=30 if df_master["Entidad federativa"].nunique() > 3 else 40,
            )

            fig_bubble.update_layout(yaxis_ticksuffix="%")
            aplicar_estilo_figura(fig_bubble)
            fig_bubble.update_traces(marker=dict(line=dict(width=0.8, color=COLOR_PANEL)))
            fig_bubble.update_layout(margin=dict(l=68, r=34, t=62, b=72))
            ajustar_legenda_larga(fig_bubble, df_master)
            st.plotly_chart(fig_bubble, width="stretch", theme=None)

            st.divider()

            # --- ROW PARA GRÁFICAS 2 Y 3 ---
            col1, col2 = st.columns(2)

            with col1:
                # --- GRÁFICA 2: BARRAS AGRUPADAS ---
                st.markdown("### 2. Percepción vs Cifra Negra")

                df_barras = df_master.groupby("Entidad federativa")[
                    ["Percepcion", "Cifra_Negra"]
                ].mean().reset_index()

                df_barras_melt = df_barras.melt(
                    id_vars="Entidad federativa",
                    value_vars=["Percepcion", "Cifra_Negra"],
                    var_name="Métrica",
                    value_name="Porcentaje"
                )

                fig_bar = px.bar(
                    df_barras_melt,
                    x="Entidad federativa",
                    y="Porcentaje",
                    color="Métrica",
                    barmode="group",
                    title=f"Promedio en años seleccionados ({delito_master})",
                    color_discrete_sequence=[COLOR_ACENTO, COLOR_SECUNDARIO]
                )

                fig_bar.update_layout(yaxis_ticksuffix="%")
                aplicar_estilo_figura(fig_bar)
                fig_bar.update_layout(
                    margin=dict(l=58, r=26, t=62, b=90),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.22,
                        xanchor="left",
                        x=0,
                        font=dict(color=COLOR_TEXTO, size=11),
                    )
                )
                st.plotly_chart(fig_bar, width="stretch", theme=None)

            with col2:
                # --- GRÁFICA 3: PASTEL ---
                st.markdown("### 3. Proporción De Denuncias")

                promedio_cn = df_master["Cifra_Negra"].mean()
                promedio_denunciado = 100 - promedio_cn

                df_pastel = pd.DataFrame({
                    "Estado Legal": [
                        "No Denunciado (Cifra Negra)",
                        "Denunciado Formalmente"
                    ],
                    "Porcentaje": [
                        promedio_cn,
                        promedio_denunciado
                    ]
                })

                fig_pie = px.pie(
                    df_pastel,
                    names="Estado Legal",
                    values="Porcentaje",
                    title=f"Distribución General Nacional ({delito_master})",
                    hole=0.4,
                    color="Estado Legal",
                    color_discrete_map={
                        "No Denunciado (Cifra Negra)": COLOR_ACENTO,
                        "Denunciado Formalmente": COLOR_TERCIARIO
                    }
                )

                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent",
                    hovertemplate="%{label}<br>%{percent}<extra></extra>",
                )

                aplicar_estilo_figura(fig_pie)
                fig_pie.update_layout(
                    margin=dict(l=20, r=20, t=62, b=90),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.12,
                        xanchor="center",
                        x=0.5,
                        font=dict(color=COLOR_TEXTO, size=11),
                    )
                )
                st.plotly_chart(fig_pie, width="stretch", theme=None)

            st.divider()

            # --- GRÁFICA 4: LÍNEAS DE TENDENCIA COMPARATIVA ---
            st.markdown("### 4. Líneas De Tendencia: Cifra Negra vs Incidencia Específica")

            df_lineas = df_master.groupby("Año")[
                ["Cifra_Negra", "Incidencia_Especifica"]
            ].mean().reset_index()

            if df_lineas["Año"].nunique() < 2 or not hay_variacion_suficiente(df_lineas, ["Cifra_Negra"]):
                mostrar_no_disponible("No hay suficientes años o variación para dibujar las líneas de tendencia.")
                st.divider()
            else:
                fig_lines = go.Figure()

                fig_lines.add_trace(go.Scatter(
                    x=df_lineas["Año"],
                    y=df_lineas["Cifra_Negra"],
                    name="Cifra Negra (%)",
                    mode="lines+markers",
                    yaxis="y1",
                    line=dict(color=COLOR_ACENTO, width=3)
                ))

                fig_lines.add_trace(go.Scatter(
                    x=df_lineas["Año"],
                    y=df_lineas["Incidencia_Especifica"],
                    name="Incidencia Específica (Tasa)",
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(color=COLOR_SECUNDARIO, width=3)
                ))

                fig_lines.update_layout(
                    title=f"Evolución Promedio: {delito_master} (Estados Seleccionados)",
                    yaxis=dict(
                        title=dict(
                            text="Cifra Negra (%)",
                            font=dict(color=COLOR_ACENTO)
                        ),
                        ticksuffix="%",
                        tickfont=dict(color=COLOR_ACENTO)
                    ),
                    yaxis2=dict(
                        title=dict(
                            text="Tasa",
                            font=dict(color=COLOR_SECUNDARIO)
                        ),
                        tickfont=dict(color=COLOR_SECUNDARIO),
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(
                        x=0.01,
                        y=1.12,
                        orientation="h"
                    )
                )

                aplicar_estilo_figura(fig_lines)
                fig_lines.update_layout(margin=dict(l=88, r=140, t=80, b=60))
                fig_lines.update_yaxes(automargin=True, title_standoff=18)
                fig_lines.update_layout(
                    yaxis2=dict(
                        automargin=True,
                        title_standoff=28
                    )
                )
                st.plotly_chart(fig_lines, width="stretch", theme=None)

                st.divider()

            # --- GRÁFICA 5: CORRELACIONES CONFIGURABLES ---
            st.markdown("### 5. Correlaciones Configurables")

            metricas_correlacion = {
                "Percepcion": "Percepción de inseguridad (%)",
                "Cifra_Negra": "Cifra Negra (%)",
                "Incidencia_Especifica": "Incidencia Específica",
                "Incidencia_General": "Incidencia General"
            }

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

            with col_ctrl1:
                metodo_corr = st.selectbox(
                    "Método de correlación:",
                    ["pearson", "spearman"],
                    format_func=lambda x: "Pearson (lineal)" if x == "pearson" else "Spearman (rangos)",
                    key="tab4_metodo_corr"
                )

            with col_ctrl2:
                nivel_corr = st.selectbox(
                    "Nivel de comparación:",
                    ["Entidad-año", "Promedio por entidad"],
                    key="tab4_nivel_corr"
                )

            with col_ctrl3:
                variables_corr = st.multiselect(
                    "Variables en la matriz:",
                    list(metricas_correlacion.keys()),
                    default=["Percepcion", "Cifra_Negra", "Incidencia_Especifica"],
                    format_func=lambda x: metricas_correlacion[x],
                    key="tab4_variables_corr"
                )

            if nivel_corr == "Promedio por entidad":
                df_corr_base = (
                    df_master.groupby("Entidad federativa", as_index=False)
                    [list(metricas_correlacion.keys())]
                    .mean()
                )
                opciones_color = ["Entidad federativa"]
            else:
                df_corr_base = df_master.copy()
                opciones_color = ["Entidad federativa", "Año"]

            if len(variables_corr) >= 2:
                matriz_corr = df_corr_base[variables_corr].corr(method=metodo_corr)
                matriz_corr.index = [metricas_correlacion[v] for v in matriz_corr.index]
                matriz_corr.columns = [metricas_correlacion[v] for v in matriz_corr.columns]

                anotaciones_corr = []
                for fila, nombre_fila in enumerate(matriz_corr.index):
                    for columna, nombre_columna in enumerate(matriz_corr.columns):
                        valor = matriz_corr.iloc[fila, columna]
                        texto_color = (
                            "#0a0a0a"
                            if (MODO_OSCURO and valor >= 0.78) or (not MODO_OSCURO and valor >= 0.72)
                            else COLOR_TEXTO
                        )
                        if not MODO_OSCURO and valor < 0.72:
                            texto_color = "#0a0a0a"

                        anotaciones_corr.append(
                            dict(
                                x=nombre_columna,
                                y=nombre_fila,
                                text=f"{valor:.2f}",
                                showarrow=False,
                                font=dict(color=texto_color, size=12)
                            )
                        )

                fig_corr = go.Figure(
                    data=go.Heatmap(
                        z=matriz_corr.values,
                        x=matriz_corr.columns,
                        y=matriz_corr.index,
                        zmin=-1,
                        zmax=1,
                        colorscale=ESCALA_CORRELACION,
                        xgap=1,
                        ygap=1,
                        colorbar=dict(
                            tickfont=dict(color=COLOR_TEXTO),
                            title=dict(text="r", font=dict(color=COLOR_TEXTO))
                        ),
                        hovertemplate="%{y}<br>%{x}<br>Correlación: %{z:.3f}<extra></extra>",
                    )
                )
                fig_corr.update_layout(
                    title=f"Matriz de correlación ({metodo_corr})",
                    annotations=anotaciones_corr,
                )
                aplicar_estilo_figura(fig_corr, altura=460)
                fig_corr.update_layout(
                    margin=dict(l=150, r=90, t=66, b=104),
                    coloraxis_showscale=False,
                )
                fig_corr.update_xaxes(
                    automargin=True,
                    tickangle=35,
                    showgrid=False,
                    zeroline=False,
                )
                fig_corr.update_yaxes(
                    automargin=True,
                    autorange="reversed",
                    showgrid=False,
                    zeroline=False,
                )
                st.plotly_chart(fig_corr, width="stretch", theme=None)

                mostrar_tabla_correlacion(matriz_corr)
            else:
                st.info("Selecciona al menos dos variables para calcular la matriz.")

            st.markdown("### 6. Dispersión A La Medida")

            col_scatter1, col_scatter2, col_scatter3, col_scatter4 = st.columns(4)
            variables_disponibles = list(metricas_correlacion.keys())

            with col_scatter1:
                eje_x = st.selectbox(
                    "Eje X:",
                    variables_disponibles,
                    index=2,
                    format_func=lambda x: metricas_correlacion[x],
                    key="tab4_eje_x"
                )

            with col_scatter2:
                eje_y = st.selectbox(
                    "Eje Y:",
                    variables_disponibles,
                    index=1,
                    format_func=lambda x: metricas_correlacion[x],
                    key="tab4_eje_y"
                )

            with col_scatter3:
                tamano = st.selectbox(
                    "Tamaño:",
                    ["Ninguno"] + variables_disponibles,
                    index=1,
                    format_func=lambda x: x if x == "Ninguno" else metricas_correlacion[x],
                    key="tab4_tamano"
                )

            with col_scatter4:
                color_por = st.selectbox(
                    "Color:",
                    opciones_color,
                    key="tab4_color"
                )

            columnas_scatter = [eje_x, eje_y, color_por]
            if tamano != "Ninguno":
                columnas_scatter.append(tamano)

            if "Entidad federativa" not in columnas_scatter:
                columnas_scatter.append("Entidad federativa")
            if nivel_corr == "Entidad-año" and "Año" not in columnas_scatter:
                columnas_scatter.append("Año")

            columnas_scatter = list(dict.fromkeys(columnas_scatter))
            df_scatter = df_corr_base[columnas_scatter].dropna()

            if eje_x == eje_y:
                st.info("Elige variables distintas en los ejes para ver una relación útil.")
            elif df_scatter[eje_x].nunique() > 1 and df_scatter[eje_y].nunique() > 1:
                r_xy = df_scatter[[eje_x, eje_y]].corr(method=metodo_corr).iloc[0, 1]

                hover_data = ["Entidad federativa"]
                if "Año" in df_scatter.columns:
                    hover_data.append("Año")

                fig_scatter_corr = px.scatter(
                    df_scatter,
                    x=eje_x,
                    y=eje_y,
                    color=color_por,
                    size=None if tamano == "Ninguno" else tamano,
                    hover_data=hover_data,
                    labels=metricas_correlacion,
                    title=(
                        f"{metricas_correlacion[eje_y]} vs "
                        f"{metricas_correlacion[eje_x]} | r = {r_xy:.3f}"
                    ),
                    color_discrete_sequence=(
                        paleta_entidades(df_scatter)
                        if color_por == "Entidad federativa"
                        else None
                    ),
                    opacity=0.78,
                    size_max=28 if df_scatter["Entidad federativa"].nunique() > 3 else 35,
                )

                x_vals = df_scatter[eje_x].astype(float)
                y_vals = df_scatter[eje_y].astype(float)
                pendiente, intercepto = np.polyfit(x_vals, y_vals, 1)
                x_linea = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_linea = pendiente * x_linea + intercepto

                fig_scatter_corr.add_trace(go.Scatter(
                    x=x_linea,
                    y=y_linea,
                    mode="lines",
                    name="Tendencia lineal",
                    line=dict(color=COLOR_ACENTO, width=3, dash="dash")
                ))

                aplicar_estilo_figura(fig_scatter_corr)
                fig_scatter_corr.update_traces(
                    selector=dict(mode="markers"),
                    marker=dict(line=dict(width=0.8, color=COLOR_PANEL))
                )
                fig_scatter_corr.update_layout(margin=dict(l=68, r=34, t=62, b=72))
                if color_por == "Entidad federativa":
                    ajustar_legenda_larga(fig_scatter_corr, df_scatter)
                st.plotly_chart(fig_scatter_corr, width="stretch", theme=None)
            else:
                mostrar_no_disponible("No hay variación suficiente para calcular esta correlación.")

        else:
            mostrar_no_disponible(
                "No hay intersección de datos para los Años, Estados y Delito seleccionados. "
                "Intenta ampliar tus filtros en la barra lateral."
            )
    else:
        st.info("No se detectaron delitos compatibles para cruzar la información.")
else:
    mostrar_no_disponible(
        "Faltan columnas necesarias para el análisis cruzado. "
        "Revisa que existan columnas IE_..._Est, CN_..._Est e ITD_..._Est."
    )

# --- SECCION 5: ANALISIS CON IA ---
st.markdown("""
    <a id="analisis-ia"></a>
    <section class="section-title">
        <h2>Análisis Con IA</h2>
        <p>Lectura ejecutiva generada bajo demanda con los filtros y cruces actuales.</p>
    </section>
""", unsafe_allow_html=True)
st.info(NOTA_METODOLOGICA)

with st.expander("Contexto Conceptual Y Fuentes", expanded=False):
    st.markdown(
        """
        - **ENVIPE**: encuesta del INEGI sobre victimización, percepción de seguridad pública, cifra negra, características del delito, víctimas y desempeño institucional.
        - **Percepción de inseguridad**: apreciación subjetiva de la población; no equivale por sí sola a incidencia delictiva.
        - **Cifra negra**: delitos no denunciados o denunciados sin inicio de investigación/carpeta; indica subregistro.
        - **Incidencia delictiva**: medición cuantitativa de delitos/tasas disponibles en la base, general o por tipo de delito.

        Fuentes base: [Programa ENVIPE INEGI](https://www.inegi.org.mx/programas/envipe/2025/),
        [ENVIPE 2025 resultados](https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/ENVIPE/ENVIPE_25_RR.pdf)
        y [Marco conceptual ENVIPE](https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/nueva_estruc/889463920038.pdf).
        """
    )

st.subheader("Interpretación automática del tablero")
st.markdown(
    "Genera un análisis textual con los años, estados, sexo de percepción y delito transversal seleccionados."
)

if st.button("Generar análisis con IA", type="primary"):
    gemini_api_key = obtener_gemini_api_key()
    modelo_ia, error_gemini = cargar_modelo_ia(gemini_api_key)

    if modelo_ia is None:
        st.error(mensaje_error_gemini(gemini_api_key, error_gemini))
    else:
        contexto_ia = construir_contexto_ia(
            df_filtrado=df_filtrado,
            df_master=df_master,
            anios_seleccionados=anios_seleccionados,
            estados_seleccionados=estados_seleccionados,
            sexo_seleccionado=sexo_percepcion,
            delito_master=delito_master,
        )
        contexto_ia = "\n".join([
            contexto_ia,
            construir_contexto_datos_dashboard(
                df_filtrado=df_filtrado,
                df_total=df_total,
                df_master=df_master,
                anios_seleccionados=anios_seleccionados,
                sexo_percepcion=sexo_percepcion,
                delito_master=delito_master,
            )
        ])

        with st.spinner("Generando análisis con Gemini..."):
            try:
                st.session_state["analisis_ia_contexto"] = contexto_ia
                st.session_state["analisis_ia"] = generar_analisis_ia(
                    modelo_ia,
                    contexto_ia
                )
                st.session_state["chat_ia"] = []
                st.session_state["graficos_ia"] = []
                st.session_state["graficos_ia_specs"] = []
                st.session_state["artefactos_ia_specs"] = []
                st.session_state["report_autoscroll_ready"] = True
            except Exception as error:
                st.session_state.pop("analisis_ia", None)
                st.session_state.pop("analisis_ia_contexto", None)
                st.error(f"No se pudo generar el análisis: {error}")

contexto_chat_actual = "\n".join([
    construir_contexto_ia(
        df_filtrado=df_filtrado,
        df_master=df_master,
        anios_seleccionados=anios_seleccionados,
        estados_seleccionados=estados_seleccionados,
        sexo_seleccionado=sexo_percepcion,
        delito_master=delito_master,
    ),
    construir_contexto_datos_dashboard(
        df_filtrado=df_filtrado,
        df_total=df_total,
        df_master=df_master,
        anios_seleccionados=anios_seleccionados,
        sexo_percepcion=sexo_percepcion,
        delito_master=delito_master,
    )
])
st.session_state["analisis_ia_contexto_actual"] = contexto_chat_actual

if st.session_state.get("analisis_ia"):
    if st.session_state.pop("report_autoscroll_ready", False):
        st.markdown('<div data-report-autoscroll="1"></div>', unsafe_allow_html=True)

    st.markdown(st.session_state["analisis_ia"])

st.markdown("### Chat Sobre El Análisis")
st.caption("Pregunta sobre el análisis generado o sobre los datos filtrados del tablero.")

historial_chat = st.session_state.setdefault("chat_ia", [])
st.session_state.setdefault("graficos_ia", [])
graficos_chat_specs = st.session_state.setdefault("graficos_ia_specs", [])
artefactos_chat_specs = st.session_state.setdefault("artefactos_ia_specs", [])

if graficos_chat_specs and not any(mensaje.get("role") == "chart" for mensaje in historial_chat):
    for item_grafico in graficos_chat_specs:
        historial_chat.append({
            "role": "chart",
            "modo": item_grafico.get("modo", "pregunta"),
            "pregunta": item_grafico.get("pregunta", ""),
            "spec": item_grafico.get("spec"),
            "nota": item_grafico.get("nota") or "Gráfico restaurado desde el estado guardado.",
        })

if artefactos_chat_specs and not any(mensaje.get("role") == "table" for mensaje in historial_chat):
    for item_tabla in artefactos_chat_specs:
        historial_chat.append({
            "role": "table",
            "tipo": item_tabla.get("tipo", "tabla"),
            "pregunta": item_tabla.get("pregunta", ""),
            "nota": item_tabla.get("nota"),
        })

for indice_mensaje, mensaje in enumerate(historial_chat):
    if mensaje.get("role") == "chart":
        mostrar_grafico_chat(
            mensaje,
            df_filtrado=df_filtrado,
            df_total=df_total,
            df_master=df_master,
            anios_seleccionados=anios_seleccionados,
            estados_seleccionados=estados_seleccionados,
            sexo_percepcion=sexo_percepcion,
            delito_master=delito_master,
            indice_mensaje=indice_mensaje,
        )
    elif mensaje.get("role") == "table":
        mostrar_tabla_chat(
            mensaje,
            df_master=df_master,
            df_filtrado=df_filtrado,
            anios_seleccionados=anios_seleccionados,
            estados_seleccionados=estados_seleccionados,
            sexo_percepcion=sexo_percepcion,
            delito_master=delito_master,
            indice_mensaje=indice_mensaje,
        )
    else:
        mostrar_mensaje_chat(mensaje)

pregunta_pendiente = st.session_state.get("pregunta_ia_pendiente")
if pregunta_pendiente:
    with st.spinner("La IA está revisando los datos filtrados..."):
        mostrar_mensaje_chat({
            "role": "assistant",
            "content": "Analizando los datos actuales del tablero..."
        })

        gemini_api_key_chat = obtener_gemini_api_key()
        modelo_ia_chat, error_gemini_chat = cargar_modelo_ia(gemini_api_key_chat)

        if modelo_ia_chat is None:
            respuesta_chat = mensaje_error_gemini(gemini_api_key_chat, error_gemini_chat)
            historial_chat.append({"role": "assistant", "content": respuesta_chat})
            st.session_state.pop("pregunta_ia_pendiente", None)
            st.session_state["ai_autoscroll_ready"] = True
            st.rerun()
        else:
            (
                df_filtrado_chat,
                df_total_chat,
                df_master_chat,
                anios_chat,
                estados_chat,
                delito_chat,
            ) = preparar_datos_chat_por_pregunta(
                pregunta_pendiente,
                df_filtrado=df_filtrado,
                df_total=df_total,
                df_master=df_master,
                anios_seleccionados=anios_seleccionados,
                delito_master=delito_master,
            )
            contexto_respuesta_chat = st.session_state.get("analisis_ia_contexto_actual", "")
            if pregunta_pide_todos_estados(pregunta_pendiente) or anios_mencionados_en_texto(pregunta_pendiente):
                contexto_respuesta_chat = "\n".join([
                    construir_contexto_ia(
                        df_filtrado=df_filtrado_chat,
                        df_master=df_master_chat,
                        anios_seleccionados=anios_chat,
                        estados_seleccionados=estados_chat,
                        sexo_seleccionado=sexo_percepcion,
                        delito_master=delito_chat,
                    ),
                    construir_contexto_datos_dashboard(
                        df_filtrado=df_filtrado_chat,
                        df_total=df_total_chat,
                        df_master=df_master_chat,
                        anios_seleccionados=anios_chat,
                        sexo_percepcion=sexo_percepcion,
                        delito_master=delito_chat,
                    ),
                    (
                        "Instrucción de alcance: si la pregunta pide todos los estados, "
                        "responde usando todos los estados disponibles aunque la barra lateral tenga pocos seleccionados."
                    ),
                ])
            try:
                respuesta_chat = responder_chat_ia(
                    modelo=modelo_ia_chat,
                    contexto=contexto_respuesta_chat,
                    analisis=st.session_state.get(
                        "analisis_ia",
                        "Aún no se ha generado reporte ejecutivo; responde con el contexto estadístico actual."
                    ),
                    historial=historial_chat,
                    pregunta=pregunta_pendiente,
                )
            except Exception as error:
                respuesta_chat = f"No pude responder en este momento: {error}"

        if not str(respuesta_chat).strip():
            respuesta_chat = "No recibí una respuesta útil de la IA. Intenta reformularlo con la variable o gráfica que quieres ver."

        historial_chat.append({"role": "assistant", "content": respuesta_chat})

        debe_generar_grafico = (
            pregunta_pide_grafico(pregunta_pendiente) or
            respuesta_indica_grafico(respuesta_chat) or
            pregunta_continua_grafico(pregunta_pendiente, historial_chat)
        )
        debe_generar_tabla = pregunta_pide_tabla(pregunta_pendiente)
        debe_generar_calculo = pregunta_pide_calculo(pregunta_pendiente)

        fig_chat, nota_chat = crear_grafico_desde_pregunta(
            pregunta=pregunta_pendiente,
            df_filtrado=df_filtrado_chat,
            df_total=df_total_chat,
            df_master=df_master_chat,
            anios_seleccionados=anios_chat,
            sexo_percepcion=sexo_percepcion,
            delito_master=delito_chat,
        )
        if fig_chat is not None and pregunta_pide_todos_estados(pregunta_pendiente):
            nota_chat = (
                (nota_chat or "Gráfico generado desde el chat.")
                + " Se usaron todos los estados disponibles, no solo la selección lateral."
            )

        spec_grafico = None
        modo_grafico = "pregunta"
        if fig_chat is None and debe_generar_grafico and modelo_ia_chat is not None:
            try:
                pregunta_para_spec = "\n".join([
                    pregunta_pendiente,
                    f"Respuesta tentativa de la IA: {respuesta_chat}",
                ])
                spec_grafico = solicitar_especificacion_grafico_ia(
                    modelo=modelo_ia_chat,
                    pregunta=pregunta_para_spec,
                    contexto=contexto_respuesta_chat,
                )
                fig_chat, nota_chat = crear_grafico_desde_spec_ia(
                    spec=spec_grafico,
                    df_filtrado=df_filtrado_chat,
                    df_total=df_total_chat,
                    df_master=df_master_chat,
                    anios_seleccionados=anios_chat,
                    sexo_percepcion=sexo_percepcion,
                    delito_master=delito_chat,
                )
                if fig_chat is not None and pregunta_pide_todos_estados(pregunta_pendiente):
                    nota_chat = (
                        (nota_chat or "Gráfico generado por IA con especificación validada.")
                        + " Se usaron todos los estados disponibles, no solo la selección lateral."
                    )
                modo_grafico = "spec"
            except Exception as error:
                nota_chat = f"No pude convertir la solicitud en un gráfico válido: {error}"

        if fig_chat is not None:
            mensaje_grafico = {
                "role": "chart",
                "modo": modo_grafico,
                "pregunta": pregunta_pendiente,
                "spec": spec_grafico,
                "nota": nota_chat,
            }
            historial_chat.append(mensaje_grafico)
            graficos_chat_specs.append({
                "modo": modo_grafico,
                "pregunta": pregunta_pendiente,
                "spec": spec_grafico,
                "nota": nota_chat,
            })
        elif nota_chat and debe_generar_grafico:
            historial_chat.append({"role": "assistant", "content": nota_chat})

        if debe_generar_tabla or debe_generar_calculo:
            tipo_artefacto = "calculo" if debe_generar_calculo else "tabla"
            if tipo_artefacto == "calculo":
                tabla_ia, nota_tabla = construir_calculo_chat(
                    pregunta_pendiente,
                    df_master=df_master_chat,
                    df_filtrado=df_filtrado_chat,
                    anios_seleccionados=anios_chat,
                )
            else:
                tabla_ia, nota_tabla = construir_tabla_chat(
                    pregunta_pendiente,
                    df_master=df_master_chat,
                    df_filtrado=df_filtrado_chat,
                    anios_seleccionados=anios_chat,
                    delito_master=delito_chat,
                )
            if tabla_ia is not None and not tabla_ia.empty and pregunta_pide_todos_estados(pregunta_pendiente):
                nota_tabla = (
                    (nota_tabla or "Tabla generada desde el chat.")
                    + " Se usaron todos los estados disponibles, no solo la selección lateral."
                )

            if tabla_ia is not None and not tabla_ia.empty:
                mensaje_tabla = {
                    "role": "table",
                    "tipo": tipo_artefacto,
                    "pregunta": pregunta_pendiente,
                    "nota": nota_tabla,
                }
                historial_chat.append(mensaje_tabla)
                artefactos_chat_specs.append({
                    "tipo": tipo_artefacto,
                    "pregunta": pregunta_pendiente,
                    "nota": nota_tabla,
                })
            elif nota_tabla:
                historial_chat.append({"role": "assistant", "content": nota_tabla})

        st.session_state.pop("pregunta_ia_pendiente", None)
        st.session_state["ai_autoscroll_ready"] = True
        st.rerun()

if st.session_state.pop("ai_autoscroll_ready", False):
    st.markdown('<div data-ai-autoscroll="1"></div>', unsafe_allow_html=True)

st.markdown('<div class="chat-form">', unsafe_allow_html=True)
with st.form("chat_ia_form", clear_on_submit=True):
    pregunta_ia = st.text_input(
        "Pregunta para la IA",
        placeholder="Pregunta sobre el análisis o los datos filtrados",
        label_visibility="collapsed"
    )
    enviar_pregunta = st.form_submit_button("Enviar")
st.markdown("</div>", unsafe_allow_html=True)

if enviar_pregunta and pregunta_ia:
    historial_chat.append({"role": "user", "content": pregunta_ia})
    st.session_state["pregunta_ia_pendiente"] = pregunta_ia
    st.rerun()

guardar_estado_persistente({
    "anios_globales": list(st.session_state.get("anios_globales", anios_seleccionados)),
    "estados_globales": list(st.session_state.get("estados_globales", estados_seleccionados)),
    "sexo_percepcion": st.session_state.get("sexo_percepcion", sexo_percepcion),
    "tab4_delito_master": st.session_state.get("tab4_delito_master", delito_master),
    "analisis_ia": st.session_state.get("analisis_ia"),
    "analisis_ia_contexto": st.session_state.get("analisis_ia_contexto"),
    "chat_ia": st.session_state.get("chat_ia", []),
    "graficos_ia_specs": st.session_state.get("graficos_ia_specs", []),
    "artefactos_ia_specs": st.session_state.get("artefactos_ia_specs", []),
})
