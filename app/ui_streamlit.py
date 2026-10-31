"""
Modulo principal de la aplicación Streamlit para el Asistente de FICs Internacionales de BTG Pactual Colombia.
"""
from app.modules.chat_ui import render_chat
import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Asistente FICs Internacionales BTG Pactual", page_icon="💰", layout="centered")

render_chat(
    title="💰 Asistente de FICs Internacionales - BTG Pactual Colombia",
    subtitle=(
        "Consulta los prospectos de los Fondos de Inversión Colectiva internacionales de BTG Pactual Colombia. "
        "La información es de carácter informativo y no constituye asesoría de inversión."
    ),
)
