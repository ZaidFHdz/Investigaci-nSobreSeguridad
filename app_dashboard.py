import streamlit as st
import base64
import html
import json
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

ARCHIVO_DATOS = Path("REPORTE_LIMPIO_FINAL.parquet")
APP_VERSION = "V1.10"
CLIENT_STATE_PARAM = "client_state"


def cargar_estado_persistente():
    estado_codificado = st.query_params.get(CLIENT_STATE_PARAM, "")
    if not estado_codificado:
        return {}

    try:
        padding = "=" * (-len(estado_codificado) % 4)
        estado_json = base64.urlsafe_b64decode(
            (estado_codificado + padding).encode("utf-8")
        ).decode("utf-8")
        estado = json.loads(estado_json)
        return estado if isinstance(estado, dict) else {}
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def codificar_estado_cliente(estado):
    estado_json = json.dumps(
        estado,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda valor: valor.item() if hasattr(valor, "item") else str(valor),
    )
    return base64.urlsafe_b64encode(estado_json.encode("utf-8")).decode("utf-8").rstrip("=")


def borrar_estado_persistente():
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
        "tab2_delito",
        "tab3_delito",
        "tab4_delito_master",
        "tab4_grafica1_x",
        "tab4_grafica1_y",
        "tab4_grafica1_tamano",
        "tab4_grafica1_color",
        "tab4_grafica2_metrica_a",
        "tab4_grafica2_metrica_b",
        "tab4_grafica3_metrica",
        "tab4_linea_a",
        "tab4_linea_b",
        "tab4_metodo_corr",
        "tab4_nivel_corr",
        "tab4_variables_corr",
        "tab4_eje_x",
        "tab4_eje_y",
        "tab4_tamano",
        "tab4_color",
        "analisis_ia",
        "analisis_ia_contexto",
        "analisis_ia_contexto_actual",
        "chat_ia",
        "graficos_ia",
        "graficos_ia_specs",
        "pregunta_ia_pendiente",
        "ai_autoscroll_ready",
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

    .desktop-only-blocker {
        display: none;
    }

    body.non-desktop-device .desktop-only-blocker {
        position: fixed;
        inset: 0;
        z-index: 1000000;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
        background: var(--app-bg);
        color: var(--app-text);
    }

    body.non-desktop-device .desktop-only-blocker-card {
        width: min(100%, 34rem);
        border: 1px solid var(--app-border);
        border-radius: 8px;
        background: var(--app-panel);
        padding: 1.4rem;
    }

    body.non-desktop-device .desktop-only-blocker-card strong {
        display: block;
        color: var(--app-text);
        font-size: 1.4rem;
        line-height: 1.15;
        margin-bottom: 0.65rem;
    }

    body.non-desktop-device .desktop-only-blocker-card p {
        margin: 0;
        color: var(--app-muted);
        line-height: 1.55;
    }

    body.non-desktop-device div[data-testid="stAppViewContainer"],
    body.non-desktop-device section[data-testid="stSidebar"],
    body.non-desktop-device header[data-testid="stHeader"] {
        display: none !important;
    }

    .block-container {
        max-width: 1440px;
        padding: 2.25rem 3rem 3rem;
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
        width: 2.25rem !important;
        height: 2.25rem !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 8px !important;
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        box-shadow: 0 1px 8px rgba(0, 0, 0, 0.06);
    }

    div[data-testid="stSidebarHeader"],
    div[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarCollapsedControl"],
    div[data-testid="collapsedControl"] {
        align-items: flex-start !important;
        min-height: 2.25rem !important;
        height: 2.25rem !important;
        padding-top: 0.65rem !important;
        overflow: visible !important;
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

    div[data-testid="stMarkdownContainer"] p,
    label,
    .app-subtitle,
    .generated-chart-note,
    div[data-testid="stCaptionContainer"],
    div[data-testid="stAlert"] {
        transition: transform 180ms ease, color 180ms ease, opacity 180ms ease;
    }

    div[data-testid="stMarkdownContainer"] p:hover,
    label:hover,
    .app-subtitle:hover,
    .generated-chart-note:hover,
    div[data-testid="stCaptionContainer"]:hover,
    div[data-testid="stAlert"]:hover {
        transform: translateY(-1px);
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
    div[role="listbox"] {
        background: var(--app-panel);
        border: 1px solid var(--app-border);
        color: var(--app-text);
    }

    div[data-baseweb="popover"]:has([aria-disabled="true"]),
    div[data-baseweb="popover"]:has([role="option"][aria-disabled="true"]),
    ul[role="listbox"]:has([aria-disabled="true"]) {
        display: none !important;
    }

    li[role="option"],
    div[role="option"] {
        background: var(--app-panel);
        color: var(--app-text);
    }

    li[role="option"]:hover,
    div[role="option"]:hover {
        background: var(--app-soft);
        color: var(--app-text);
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
        margin: 0.2rem 3.1rem 0.75rem 0;
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

    .side-brand,
    .sidebar-heading,
    .side-nav a,
    .reset-link,
    .section-title h2,
    div[data-testid="stPlotlyChart"],
    div[data-testid="stForm"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    button {
        transition:
            transform 180ms ease,
            border-color 180ms ease,
            background-color 180ms ease,
            opacity 180ms ease,
            box-shadow 180ms ease;
    }

    .side-brand:hover,
    .sidebar-heading:hover,
    .side-nav a:hover,
    .reset-link:hover,
    div[data-testid="stPlotlyChart"]:hover,
    button:hover {
        transform: translateY(-1.5px);
    }

    .section-title:hover h2 {
        transform: translateY(-2px);
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        transform: translateY(-1px);
    }

    @media (prefers-reduced-motion: reduce) {
        .side-brand,
        .sidebar-heading,
        .side-nav a,
        .reset-link,
        .section-title h2,
        div[data-testid="stPlotlyChart"],
        div[data-testid="stForm"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        button {
            transition: none !important;
        }

        .side-brand:hover,
        .sidebar-heading:hover,
        .side-nav a:hover,
        .reset-link:hover,
        .section-title:hover h2,
        div[data-testid="stPlotlyChart"]:hover,
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="input"] > div:hover,
        button:hover {
            transform: none !important;
        }
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

    .export-block {
        border-top: 1px solid var(--app-border);
        margin: -0.15rem 0 0;
        padding-top: 0.55rem;
    }

    .sidebar-version {
        border-top: 1px solid var(--app-border);
        color: var(--app-muted);
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 1.2rem;
        padding-top: 0.75rem;
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
    div[data-testid="stModal"] {
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
    }

    div[role="dialog"] *,
    div[data-testid="stModal"] * {
        color: var(--app-text) !important;
    }

    div[role="dialog"] button,
    div[data-testid="stModal"] button {
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
        opacity: 1 !important;
    }

    .chat-form {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="desktop-only-blocker">
        <div class="desktop-only-blocker-card">
            <strong>Disponible Solo En Computadora</strong>
            <p>Este tablero está diseñado para mouse, teclado y pantalla amplia. Para evitar problemas con filtros, gráficas y chat, ábrelo desde Windows, macOS, Linux o una computadora de escritorio/portátil.</p>
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

    section[data-testid="stSidebar"] {{
        min-width: 23rem !important;
        width: 23rem !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
        min-height: 4.25rem !important;
        height: auto !important;
        max-height: 15rem !important;
        align-items: flex-start !important;
        padding-top: 0.45rem !important;
        padding-bottom: 0.45rem !important;
        overflow-y: auto !important;
        scrollbar-width: thin;
        scrollbar-color: var(--app-muted) transparent;
    }}

    section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div::-webkit-scrollbar {{
        width: 8px;
    }}

    section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div::-webkit-scrollbar-thumb {{
        background: var(--app-muted);
        border-radius: 999px;
    }}

    section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div > div:first-child {{
        align-content: flex-start !important;
        overflow: visible !important;
    }}

    section[data-testid="stSidebar"] [data-baseweb="tag"] {{
        max-width: 18rem !important;
        margin-bottom: 0.25rem !important;
    }}

    section[data-testid="stSidebar"] [data-baseweb="tag"] span {{
        max-width: 15.5rem !important;
    }}

    button:hover,
    button:focus,
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {{
        border-color: var(--app-invert) !important;
        box-shadow: none !important;
        outline: none !important;
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

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {{
        background: var(--app-panel) !important;
        border-color: var(--app-border) !important;
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


def mostrar_nota_ceros(df, columnas, contexto="esta gráfica", umbral=0.28):
    if df is None or df.empty:
        return

    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        return

    total = 0
    ceros = 0
    for columna in columnas_validas:
        serie = pd.to_numeric(df[columna], errors="coerce").dropna()
        total += int(serie.size)
        ceros += int((serie == 0).sum())

    if total and ceros / total >= umbral:
        st.caption(
            f"Nota: {contexto} contiene muchos valores en 0. "
            "Puede indicar ausencia de datos, registros no disponibles o una categoría sin observaciones para los filtros actuales."
        )


def restaurar_widget_desde_cliente(clave, opciones=None, default=None):
    if clave in st.session_state:
        return

    valor = ESTADO_PERSISTENTE.get(clave, default)
    if valor is None:
        return

    if opciones is not None:
        if isinstance(valor, list):
            valor = [item for item in valor if item in opciones]
            if not valor:
                return
        elif valor not in opciones:
            return

    st.session_state[clave] = valor


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


def obtener_gemini_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


@st.cache_resource
def cargar_modelo_ia(api_key):
    if genai is None or not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return None


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
Evita lenguaje alarmista y evita recomendaciones legales.

Entrega:
1. Lectura general de los filtros seleccionados.
2. Hallazgos principales.
3. Relación entre percepción, cifra negra e incidencia.
4. Posibles interpretaciones con cautela metodológica.
5. Preguntas o próximos cruces recomendados.

Contexto:
{contexto}
"""
    respuesta = modelo.generate_content(prompt)
    return getattr(respuesta, "text", "").strip()


def responder_chat_ia(modelo, contexto, analisis, historial, pregunta):
    historial_txt = "\n".join(
        f"{mensaje['role']}: {mensaje['content']}"
        for mensaje in historial[-8:]
    )

    prompt = f"""
Eres analista de datos públicos de seguridad en México.
Responde en español, de forma clara y breve, usando solo el contexto del dashboard,
el análisis ya generado y la conversación. El contexto incluye datos filtrados,
cobertura por año y muestras de las tablas usadas por las gráficas; úsalo para
responder preguntas específicas. Si el usuario pide algo que no se puede inferir
de estos datos, dilo y sugiere qué filtro o cruce revisaría.

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
        "tendencia", "visualización", "visualizacion"
    ]
    return any(clave in texto for clave in claves)


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

    return None, "Puedo generar gráficos de ENVIPE, cifra negra, incidencia o el cruce 360. Intenta pedirlo con una de esas variables."


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
  "dataset": "envipe|envipe_cobertura|cifra_negra|incidencia_general|cruce360",
  "tipo": "linea|barras|dispersion",
  "x": "Año|Entidad federativa|Incidencia_Especifica|Incidencia_General|Percepcion|Cifra_Negra",
  "y": "ENV_Estimaciones puntuales|Valor|Tasa_Incidencia|Percepcion|Cifra_Negra|Incidencia_Especifica|Incidencia_General|Con_Dato",
  "color": "Entidad federativa|Sexo|Año|ninguno",
  "tamano": "Percepcion|Cifra_Negra|Incidencia_Especifica|ninguno",
  "titulo": "título corto en español"
}}

Reglas:
- Usa "envipe_cobertura" si preguntan por años faltantes, disponibilidad, cobertura o si hay datos en 2012.
- Usa "envipe" para percepción/inseguridad por año o entidad.
- Usa "cifra_negra" para denuncias, no denuncia, cifra negra.
- Usa "incidencia_general" para tasa general estatal.
- Usa "cruce360" para relaciones/correlaciones entre percepción, cifra negra e incidencia.
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

if "graficos_ia" not in st.session_state:
    st.session_state["graficos_ia"] = []
if "graficos_ia_specs" not in st.session_state:
    st.session_state["graficos_ia_specs"] = []

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
        const stateKey = "seguridadMxDashboardClientState";

        const isDesktopDevice = () => {
            const ua = win.navigator.userAgent || "";
            const mobileLike = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|Tablet/i.test(ua);
            const coarsePointer = win.matchMedia("(pointer: coarse)").matches;
            const noHover = win.matchMedia("(hover: none)").matches;
            return !(mobileLike || (coarsePointer && noHover));
        };

        const syncDesktopGate = () => {
            doc.body.classList.toggle("non-desktop-device", !isDesktopDevice());
        };

        const restoreClientState = () => {
            try {
                const url = new URL(win.location.href);
                if (url.searchParams.has("reset_dashboard") || url.searchParams.has("client_state")) return;
                const stored = JSON.parse(win.localStorage.getItem(stateKey) || "null");
                if (!stored || !stored.encoded) return;
                url.searchParams.set("client_state", stored.encoded);
                win.location.replace(url.toString());
            } catch (error) {
                return;
            }
        };

        restoreClientState();
        syncDesktopGate();

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
        const scheduleSync = () => {
            win.clearTimeout(timer);
            timer = win.setTimeout(syncPlotTheme, 120);
        };

        const markAiSendPosition = (event) => {
            const button = event.target.closest("button");
            if (!button) return;
            const text = (button.innerText || "").trim().toLowerCase();
            if (text !== "enviar") return;
            win.sessionStorage.setItem("dashboardAiSendY", String(win.scrollY));
            win.sessionStorage.setItem("dashboardAiSendAt", String(Date.now()));
        };

        const maybeScrollToAiResponse = () => {
            const marker = doc.querySelector('[data-ai-autoscroll="1"]');
            if (!marker || marker.dataset.done === "1") return;

            const sentAt = Number(win.sessionStorage.getItem("dashboardAiSendAt") || 0);
            const sentY = Number(win.sessionStorage.getItem("dashboardAiSendY") || 0);
            const recent = Date.now() - sentAt < 60000;
            const userStayed = Math.abs(win.scrollY - sentY) < 90;

            if (recent && userStayed) {
                marker.dataset.done = "1";
                win.setTimeout(() => {
                    marker.scrollIntoView({ behavior: "smooth", block: "end" });
                }, 160);
            }
        };

        doc.addEventListener("click", (event) => {
            markAiSendPosition(event);
            if (event.target.closest(".reset-link")) {
                win.localStorage.removeItem(stateKey);
            }
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

        doc.addEventListener("wheel", (event) => {
            const multiselect = event.target.closest('div[data-testid="stMultiSelect"]');
            if (!multiselect) return;
            const scrollBox = multiselect.querySelector('div[data-baseweb="select"] > div');
            if (!scrollBox || scrollBox.scrollHeight <= scrollBox.clientHeight + 2) return;
            scrollBox.scrollTop += event.deltaY;
            event.preventDefault();
            event.stopPropagation();
        }, { passive: false, capture: true });

        new MutationObserver(() => {
            doc.querySelectorAll('[title="streamlitApp"]').forEach((node) => node.removeAttribute("title"));
            syncDesktopGate();
            scheduleSync();
            maybeScrollToAiResponse();
        }).observe(doc.body, {
            attributes: true,
            childList: true,
            subtree: true
        });

        win.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", scheduleSync);
        win.matchMedia("(pointer: coarse)").addEventListener("change", syncDesktopGate);
        win.matchMedia("(hover: none)").addEventListener("change", syncDesktopGate);
        scheduleSync();
        maybeScrollToAiResponse();
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
    mostrar_nota_ceros(df_inseguro, ["ENV_Estimaciones puntuales"], "la evolución ENVIPE")
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
    mostrar_nota_ceros(df_sexo_env, ["ENV_Estimaciones puntuales"], "la comparación por sexo")

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
        restaurar_widget_desde_cliente("tab2_delito", delitos_cn)
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
            mostrar_nota_ceros(df_cn_plot, ["Valor"], "la tendencia de cifra negra")
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
            mostrar_nota_ceros(df_ie, ["Tasa_Incidencia"], "la incidencia general")
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
            restaurar_widget_desde_cliente("tab3_delito", delitos_itd)
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
                mostrar_nota_ceros(df_itd_plot, ["Tasa_Incidencia"], "la incidencia específica")
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
            metricas_cruce = {
                "Percepcion": "Percepción De Inseguridad (%)",
                "Cifra_Negra": "Cifra Negra (%)",
                "Incidencia_Especifica": "Incidencia Específica",
                "Incidencia_General": "Incidencia General",
            }
            st.divider()

            # --- GRÁFICA 1: BURBUJAS ---
            st.markdown("### 1. Panorama Personalizable (Burbujas)")
            opciones_burbuja = list(metricas_cruce.keys())
            opciones_color_burbuja = ["Entidad federativa", "Año", "Delito"]
            for clave, opciones in {
                "tab4_grafica1_x": opciones_burbuja,
                "tab4_grafica1_y": opciones_burbuja,
                "tab4_grafica1_tamano": opciones_burbuja,
                "tab4_grafica1_color": opciones_color_burbuja,
            }.items():
                restaurar_widget_desde_cliente(clave, opciones)

            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                burbuja_x = st.selectbox(
                    "Eje X:",
                    opciones_burbuja,
                    index=2,
                    format_func=lambda valor: metricas_cruce[valor],
                    key="tab4_grafica1_x",
                )
            with col_b2:
                burbuja_y = st.selectbox(
                    "Eje Y:",
                    opciones_burbuja,
                    index=1,
                    format_func=lambda valor: metricas_cruce[valor],
                    key="tab4_grafica1_y",
                )
            with col_b3:
                burbuja_tamano = st.selectbox(
                    "Tamaño:",
                    opciones_burbuja,
                    index=0,
                    format_func=lambda valor: metricas_cruce[valor],
                    key="tab4_grafica1_tamano",
                )
            with col_b4:
                burbuja_color = st.selectbox(
                    "Color:",
                    opciones_color_burbuja,
                    key="tab4_grafica1_color",
                )

            st.caption(
                f"Eje X: {metricas_cruce[burbuja_x]} | "
                f"Eje Y: {metricas_cruce[burbuja_y]} | "
                f"Tamaño: {metricas_cruce[burbuja_tamano]} | "
                f"Color: {burbuja_color}"
            )

            fig_bubble = px.scatter(
                df_master,
                x=burbuja_x,
                y=burbuja_y,
                size=burbuja_tamano,
                color=burbuja_color,
                hover_name="Entidad federativa",
                hover_data=["Año", "Incidencia_General"],
                title=(
                    f"{metricas_cruce[burbuja_y]} vs {metricas_cruce[burbuja_x]} "
                    f"- {delito_master}"
                ),
                labels=metricas_cruce,
                color_discrete_sequence=(
                    paleta_entidades(df_master)
                    if burbuja_color == "Entidad federativa"
                    else PALETA_NEUTRA
                ),
                opacity=0.78,
                size_max=30 if df_master["Entidad federativa"].nunique() > 3 else 40,
            )

            if burbuja_y in {"Percepcion", "Cifra_Negra"}:
                fig_bubble.update_layout(yaxis_ticksuffix="%")
            if burbuja_x in {"Percepcion", "Cifra_Negra"}:
                fig_bubble.update_layout(xaxis_ticksuffix="%")
            aplicar_estilo_figura(fig_bubble)
            fig_bubble.update_traces(marker=dict(line=dict(width=0.8, color=COLOR_PANEL)))
            fig_bubble.update_layout(margin=dict(l=68, r=34, t=62, b=72))
            if burbuja_color == "Entidad federativa":
                ajustar_legenda_larga(fig_bubble, df_master)
            st.plotly_chart(fig_bubble, width="stretch", theme=None)
            mostrar_nota_ceros(
                df_master,
                [burbuja_x, burbuja_y, burbuja_tamano],
                "el panorama personalizable"
            )

            st.divider()

            # --- ROW PARA GRÁFICAS 2 Y 3 ---
            col1, col2 = st.columns(2)

            with col1:
                # --- GRÁFICA 2: BARRAS AGRUPADAS ---
                st.markdown("### 2. Comparación Personalizable Por Entidad")
                for clave in ["tab4_grafica2_metrica_a", "tab4_grafica2_metrica_b"]:
                    restaurar_widget_desde_cliente(clave, list(metricas_cruce.keys()))

                col_g2a, col_g2b = st.columns(2)
                with col_g2a:
                    metrica_barra_a = st.selectbox(
                        "Métrica A:",
                        list(metricas_cruce.keys()),
                        index=0,
                        format_func=lambda valor: metricas_cruce[valor],
                        key="tab4_grafica2_metrica_a",
                    )
                with col_g2b:
                    metrica_barra_b = st.selectbox(
                        "Métrica B:",
                        list(metricas_cruce.keys()),
                        index=1,
                        format_func=lambda valor: metricas_cruce[valor],
                        key="tab4_grafica2_metrica_b",
                    )

                df_barras = df_master.groupby("Entidad federativa")[
                    [metrica_barra_a, metrica_barra_b]
                ].mean().reset_index()

                df_barras_melt = df_barras.melt(
                    id_vars="Entidad federativa",
                    value_vars=[metrica_barra_a, metrica_barra_b],
                    var_name="Métrica",
                    value_name="Valor"
                )
                df_barras_melt["Métrica"] = df_barras_melt["Métrica"].map(metricas_cruce)

                fig_bar = px.bar(
                    df_barras_melt,
                    x="Entidad federativa",
                    y="Valor",
                    color="Métrica",
                    barmode="group",
                    title=(
                        f"{metricas_cruce[metrica_barra_a]} vs "
                        f"{metricas_cruce[metrica_barra_b]} ({delito_master})"
                    ),
                    color_discrete_sequence=[COLOR_ACENTO, COLOR_SECUNDARIO]
                )

                if {metrica_barra_a, metrica_barra_b}.issubset({"Percepcion", "Cifra_Negra"}):
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
                mostrar_nota_ceros(
                    df_barras,
                    [metrica_barra_a, metrica_barra_b],
                    "la comparación por entidad"
                )

            with col2:
                # --- GRÁFICA 3: PASTEL ---
                st.markdown("### 3. Proporción Personalizable")
                opciones_pastel = ["Cifra_Negra", "Percepcion"]
                restaurar_widget_desde_cliente("tab4_grafica3_metrica", opciones_pastel)
                metrica_pastel = st.selectbox(
                    "Proporción basada en:",
                    opciones_pastel,
                    format_func=lambda valor: metricas_cruce[valor],
                    key="tab4_grafica3_metrica",
                )

                promedio_pastel = float(np.clip(df_master[metrica_pastel].mean(), 0, 100))
                complemento_pastel = 100 - promedio_pastel
                etiqueta_principal = (
                    "No Denunciado (Cifra Negra)"
                    if metrica_pastel == "Cifra_Negra"
                    else "Población Que Percibe Inseguridad"
                )
                etiqueta_complemento = (
                    "Denunciado Formalmente"
                    if metrica_pastel == "Cifra_Negra"
                    else "Resto De La Población"
                )

                df_pastel = pd.DataFrame({
                    "Estado Legal": [
                        etiqueta_principal,
                        etiqueta_complemento,
                    ],
                    "Porcentaje": [
                        promedio_pastel,
                        complemento_pastel,
                    ]
                })

                fig_pie = px.pie(
                    df_pastel,
                    names="Estado Legal",
                    values="Porcentaje",
                    title=f"{metricas_cruce[metrica_pastel]} Promedio ({delito_master})",
                    hole=0.4,
                    color="Estado Legal",
                    color_discrete_map={
                        etiqueta_principal: COLOR_ACENTO,
                        etiqueta_complemento: COLOR_TERCIARIO,
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
                mostrar_nota_ceros(
                    df_master,
                    [metrica_pastel],
                    "la proporción personalizable"
                )

            st.divider()

            # --- GRÁFICA 4: LÍNEAS DE TENDENCIA COMPARATIVA ---
            st.markdown("### 4. Tendencias Personalizables")
            for clave in ["tab4_linea_a", "tab4_linea_b"]:
                restaurar_widget_desde_cliente(clave, list(metricas_cruce.keys()))

            col_l1, col_l2 = st.columns(2)
            with col_l1:
                linea_a = st.selectbox(
                    "Línea izquierda:",
                    list(metricas_cruce.keys()),
                    index=1,
                    format_func=lambda valor: metricas_cruce[valor],
                    key="tab4_linea_a",
                )
            with col_l2:
                linea_b = st.selectbox(
                    "Línea derecha:",
                    list(metricas_cruce.keys()),
                    index=2,
                    format_func=lambda valor: metricas_cruce[valor],
                    key="tab4_linea_b",
                )

            df_lineas = df_master.groupby("Año")[
                [linea_a, linea_b]
            ].mean().reset_index()

            if df_lineas["Año"].nunique() < 2 or not hay_variacion_suficiente(df_lineas, [linea_a]):
                mostrar_no_disponible("No hay suficientes años o variación para dibujar las líneas de tendencia.")
                st.divider()
            else:
                fig_lines = go.Figure()

                fig_lines.add_trace(go.Scatter(
                    x=df_lineas["Año"],
                    y=df_lineas[linea_a],
                    name=metricas_cruce[linea_a],
                    mode="lines+markers",
                    yaxis="y1",
                    line=dict(color=COLOR_ACENTO, width=3)
                ))

                fig_lines.add_trace(go.Scatter(
                    x=df_lineas["Año"],
                    y=df_lineas[linea_b],
                    name=metricas_cruce[linea_b],
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(color=COLOR_SECUNDARIO, width=3)
                ))

                fig_lines.update_layout(
                    title=(
                        f"{metricas_cruce[linea_a]} vs {metricas_cruce[linea_b]} "
                        f"({delito_master})"
                    ),
                    yaxis=dict(
                        title=dict(
                            text=metricas_cruce[linea_a],
                            font=dict(color=COLOR_ACENTO)
                        ),
                        ticksuffix="%" if linea_a in {"Percepcion", "Cifra_Negra"} else "",
                        tickfont=dict(color=COLOR_ACENTO)
                    ),
                    yaxis2=dict(
                        title=dict(
                            text=metricas_cruce[linea_b],
                            font=dict(color=COLOR_SECUNDARIO)
                        ),
                        ticksuffix="%" if linea_b in {"Percepcion", "Cifra_Negra"} else "",
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
                mostrar_nota_ceros(
                    df_lineas,
                    [linea_a, linea_b],
                    "las tendencias personalizables"
                )

                st.divider()

            # --- GRÁFICA 5: CORRELACIONES CONFIGURABLES ---
            st.markdown("### 5. Matriz De Correlación Personalizable")

            metricas_correlacion = {
                "Percepcion": "Percepción de inseguridad (%)",
                "Cifra_Negra": "Cifra Negra (%)",
                "Incidencia_Especifica": "Incidencia Específica",
                "Incidencia_General": "Incidencia General"
            }

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            restaurar_widget_desde_cliente("tab4_metodo_corr", ["pearson", "spearman"])
            restaurar_widget_desde_cliente("tab4_nivel_corr", ["Entidad-año", "Promedio por entidad"])
            restaurar_widget_desde_cliente("tab4_variables_corr", list(metricas_correlacion.keys()))

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
                mostrar_nota_ceros(
                    df_corr_base,
                    variables_corr,
                    "la matriz de correlación"
                )

                mostrar_tabla_correlacion(matriz_corr)
            else:
                st.info("Selecciona al menos dos variables para calcular la matriz.")

            st.markdown("### 6. Dispersión Personalizable")

            col_scatter1, col_scatter2, col_scatter3, col_scatter4 = st.columns(4)
            variables_disponibles = list(metricas_correlacion.keys())
            restaurar_widget_desde_cliente("tab4_eje_x", variables_disponibles)
            restaurar_widget_desde_cliente("tab4_eje_y", variables_disponibles)
            restaurar_widget_desde_cliente("tab4_tamano", ["Ninguno"] + variables_disponibles)
            restaurar_widget_desde_cliente("tab4_color", opciones_color)

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
                columnas_ceros_scatter = [eje_x, eje_y]
                if tamano != "Ninguno":
                    columnas_ceros_scatter.append(tamano)
                mostrar_nota_ceros(
                    df_scatter,
                    columnas_ceros_scatter,
                    "la dispersión personalizable"
                )
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

st.subheader("Interpretación automática del tablero")
st.markdown(
    "Genera un análisis textual con los años, estados, sexo de percepción y delito transversal seleccionados."
)

if st.button("Generar análisis con IA", type="primary"):
    modelo_ia = cargar_modelo_ia(obtener_gemini_api_key())

    if modelo_ia is None:
        st.error(
            "No se pudo configurar Gemini. Revisa que google-generativeai esté instalado "
            "y que la API key sea válida."
        )
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
            except Exception as error:
                st.session_state.pop("analisis_ia", None)
                st.session_state.pop("analisis_ia_contexto", None)
                st.error(f"No se pudo generar el análisis: {error}")

if st.session_state.get("analisis_ia"):
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

    st.markdown(st.session_state["analisis_ia"])

    st.markdown("### Chat Sobre El Análisis")
    st.caption("Pregunta sobre el análisis generado o sobre los datos filtrados del tablero.")

    historial_chat = st.session_state.setdefault("chat_ia", [])
    graficos_chat = st.session_state.setdefault("graficos_ia", [])
    graficos_chat_specs = st.session_state.setdefault("graficos_ia_specs", [])

    if not graficos_chat and graficos_chat_specs:
        for item_grafico in graficos_chat_specs:
            fig_reconstruida = None
            nota_reconstruida = item_grafico.get("nota")
            if item_grafico.get("modo") == "spec":
                fig_reconstruida, nota_auto = crear_grafico_desde_spec_ia(
                    spec=item_grafico.get("spec"),
                    df_filtrado=df_filtrado,
                    df_total=df_total,
                    df_master=df_master,
                    anios_seleccionados=anios_seleccionados,
                    sexo_percepcion=sexo_percepcion,
                    delito_master=delito_master,
                )
                nota_reconstruida = nota_reconstruida or nota_auto
            else:
                fig_reconstruida, nota_auto = crear_grafico_desde_pregunta(
                    pregunta=item_grafico.get("pregunta", ""),
                    df_filtrado=df_filtrado,
                    df_total=df_total,
                    df_master=df_master,
                    anios_seleccionados=anios_seleccionados,
                    sexo_percepcion=sexo_percepcion,
                    delito_master=delito_master,
                )
                nota_reconstruida = nota_reconstruida or nota_auto

            if fig_reconstruida is not None:
                graficos_chat.append({
                    "nota": nota_reconstruida or "Gráfico restaurado desde el estado guardado.",
                    "fig": fig_reconstruida,
                })

    for mensaje in historial_chat:
        mostrar_mensaje_chat(mensaje)

    pregunta_pendiente = st.session_state.get("pregunta_ia_pendiente")
    if pregunta_pendiente:
        with st.spinner("La IA está revisando los datos filtrados..."):
            mostrar_mensaje_chat({
                "role": "assistant",
                "content": "Analizando los datos actuales del tablero..."
            })

            modelo_ia_chat = cargar_modelo_ia(obtener_gemini_api_key())

            if modelo_ia_chat is None:
                respuesta_chat = (
                    "No pude conectar con Gemini. Revisa la instalación de google-generativeai "
                    "y la API key."
                )
            else:
                try:
                    respuesta_chat = responder_chat_ia(
                        modelo=modelo_ia_chat,
                        contexto=st.session_state.get("analisis_ia_contexto_actual", ""),
                        analisis=st.session_state["analisis_ia"],
                        historial=historial_chat,
                        pregunta=pregunta_pendiente,
                    )
                except Exception as error:
                    respuesta_chat = f"No pude responder en este momento: {error}"

            historial_chat.append({"role": "assistant", "content": respuesta_chat})

            fig_chat, nota_chat = crear_grafico_desde_pregunta(
                pregunta=pregunta_pendiente,
                df_filtrado=df_filtrado,
                df_total=df_total,
                df_master=df_master,
                anios_seleccionados=anios_seleccionados,
                sexo_percepcion=sexo_percepcion,
                delito_master=delito_master,
            )

            spec_grafico = None
            modo_grafico = "pregunta"
            if fig_chat is None and pregunta_pide_grafico(pregunta_pendiente) and modelo_ia_chat is not None:
                try:
                    spec_grafico = solicitar_especificacion_grafico_ia(
                        modelo=modelo_ia_chat,
                        pregunta=pregunta_pendiente,
                        contexto=st.session_state.get("analisis_ia_contexto_actual", ""),
                    )
                    fig_chat, nota_chat = crear_grafico_desde_spec_ia(
                        spec=spec_grafico,
                        df_filtrado=df_filtrado,
                        df_total=df_total,
                        df_master=df_master,
                        anios_seleccionados=anios_seleccionados,
                        sexo_percepcion=sexo_percepcion,
                        delito_master=delito_master,
                    )
                    modo_grafico = "spec"
                except Exception as error:
                    nota_chat = f"No pude convertir la solicitud en un gráfico válido: {error}"

            if fig_chat is not None:
                graficos_chat.append({"nota": nota_chat, "fig": fig_chat})
                graficos_chat_specs.append({
                    "modo": modo_grafico,
                    "pregunta": pregunta_pendiente,
                    "spec": spec_grafico,
                    "nota": nota_chat,
                })
            elif nota_chat and pregunta_pide_grafico(pregunta_pendiente):
                historial_chat.append({"role": "assistant", "content": nota_chat})

            st.session_state.pop("pregunta_ia_pendiente", None)
            st.session_state["ai_autoscroll_ready"] = True
            st.rerun()

    for grafico in graficos_chat:
        st.markdown(
            f'<div class="generated-chart-note">{html.escape(grafico["nota"])}</div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(grafico["fig"], width="stretch", theme=None)

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

estado_cliente_actual = {
    "anios_globales": list(st.session_state.get("anios_globales", anios_seleccionados)),
    "estados_globales": list(st.session_state.get("estados_globales", estados_seleccionados)),
    "sexo_percepcion": st.session_state.get("sexo_percepcion", sexo_percepcion),
    "tab2_delito": st.session_state.get("tab2_delito"),
    "tab3_delito": st.session_state.get("tab3_delito"),
    "tab4_delito_master": st.session_state.get("tab4_delito_master", delito_master),
    "tab4_grafica1_x": st.session_state.get("tab4_grafica1_x"),
    "tab4_grafica1_y": st.session_state.get("tab4_grafica1_y"),
    "tab4_grafica1_tamano": st.session_state.get("tab4_grafica1_tamano"),
    "tab4_grafica1_color": st.session_state.get("tab4_grafica1_color"),
    "tab4_grafica2_metrica_a": st.session_state.get("tab4_grafica2_metrica_a"),
    "tab4_grafica2_metrica_b": st.session_state.get("tab4_grafica2_metrica_b"),
    "tab4_grafica3_metrica": st.session_state.get("tab4_grafica3_metrica"),
    "tab4_linea_a": st.session_state.get("tab4_linea_a"),
    "tab4_linea_b": st.session_state.get("tab4_linea_b"),
    "tab4_metodo_corr": st.session_state.get("tab4_metodo_corr"),
    "tab4_nivel_corr": st.session_state.get("tab4_nivel_corr"),
    "tab4_variables_corr": st.session_state.get("tab4_variables_corr"),
    "tab4_eje_x": st.session_state.get("tab4_eje_x"),
    "tab4_eje_y": st.session_state.get("tab4_eje_y"),
    "tab4_tamano": st.session_state.get("tab4_tamano"),
    "tab4_color": st.session_state.get("tab4_color"),
}
estado_cliente_actual = {
    clave: valor
    for clave, valor in estado_cliente_actual.items()
    if valor is not None
}
estado_cliente_codificado = codificar_estado_cliente(estado_cliente_actual)

components.html(
    f"""
    <script>
    (() => {{
        const win = window.parent;
        const key = "seguridadMxDashboardClientState";
        const encoded = {json.dumps(estado_cliente_codificado)};
        const state = {json.dumps(estado_cliente_actual, ensure_ascii=False, default=str)};
        try {{
            win.localStorage.setItem(key, JSON.stringify({{ encoded, state, updatedAt: Date.now() }}));
            const url = new URL(win.location.href);
            if (url.searchParams.get("client_state") !== encoded && !url.searchParams.has("reset_dashboard")) {{
                url.searchParams.set("client_state", encoded);
                win.history.replaceState(null, "", url.toString());
            }}
        }} catch (error) {{
            return;
        }}
    }})();
    </script>
    """,
    height=0,
)
