"""
Modulo principal de la aplicación Streamlit para el Asistente de FICs Internacionales de BTG Pactual Colombia.
"""
from app.modules.metrics_ui import render_metrics
from app.modules.chat_ui import render_chat
import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Asistente FICs Internacionales BTG Pactual", page_icon="💰", layout="wide")

CHAT_TITLE = "💰 Asistente de FICs Internacionales - BTG Pactual Colombia"
CHAT_SUBTITLE = (
    "Consulta los prospectos de los Fondos de Inversión Colectiva internacionales de BTG Pactual Colombia: "
    "objetivos, comisiones, clases de inversionistas, pacto de permanencia y riesgos. "
    "La información es de carácter informativo y no constituye asesoría de inversión."
)

modo = st.sidebar.radio("Selecciona una vista:", ["🤖 Chatbot", "📊 Métricas"])
st.sidebar.divider()

if modo == "🤖 Chatbot":
    render_chat(title=CHAT_TITLE, subtitle=CHAT_SUBTITLE)
else:
    render_metrics()