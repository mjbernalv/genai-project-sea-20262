"""
Componente de chat reutilizable para ui_streamlit.py y main_interface.py.
"""
from app.model.tracking import CHAT_EXPERIMENT, init_tracking, log_user_feedback
from app.model.prompt_registry import DEFAULT_PROMPT, resolve_prompt_ref
from app.model.llm_provider import describe
from app.model.rag_pipeline import (
    build_chain,
    list_prompt_versions,
    load_vectorstore_from_disk,
    save_vectorstore
)

import streamlit as st
import mlflow

import json
import time
import os

# Constantes
EVAL_DATASET = "tests/eval_dataset.json"
FALLBACK_EXAMPLES = [
    "¿Cuál es el objetivo de inversión del fondo BTG Pactual Oro?",
    "¿Qué comisión cobra la Clase A del fondo BTG Pactual Renta Fija Internacional?",
    "¿Cuál es el pacto de permanencia del fondo BTG Pactual Liquidez Dólares?",
]

# Recursos cacheados
@st.cache_resource(show_spinner="Cargando índice vectorial...")
def get_vectordb():
    try:
        return load_vectorstore_from_disk()
    except FileNotFoundError:
        with st.spinner("Indexando los prospectos (solo la primera vez, puede tardar unos minutos)..."):
            save_vectorstore()
        return load_vectorstore_from_disk()


@st.cache_resource(show_spinner="Preparando el asistente...")
def get_chain(prompt_version: str, k: int):
    chain = build_chain(get_vectordb(), prompt_version=prompt_version, k=k)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]
    return chain, prompt_uri


@st.cache_data(ttl=30, show_spinner=False)
def resolve_prompt(ref: str) -> str:
    return resolve_prompt_ref(ref)


@st.cache_data(ttl=30, show_spinner=False)
def get_prompt_options():
    return list_prompt_versions()


@st.cache_data
def get_example_questions(max_n: int = 4):
    """
    Usa las preguntas del dataset de evaluación: si cambias de dominio, cambian solas.
    """
    try:
        with open(EVAL_DATASET, encoding="utf-8") as f:
            questions = [item["question"] for item in json.load(f)]
        return questions[:max_n] or FALLBACK_EXAMPLES
    except Exception:
        return FALLBACK_EXAMPLES


# Utilidades
def _format_sources(source_documents):
    """
    Agrupa fragmentos por (documento, página) y elimina duplicados.
    """
    refs = {}
    for doc in source_documents or []:
        meta = doc.metadata or {}
        name = os.path.basename(meta.get("source", "desconocido"))
        page = meta.get("page")
        page = page + 1 if isinstance(page, int) else None
        key = (name, page)
        if key not in refs:
            refs[key] = doc.page_content.strip()
    return [{"document": n, "page": p, "snippet": t} for (n, p), t in refs.items()]


def _log_feedback(msg, thumbs_up: bool):
    """
    Registra el feedback del usuario en MLflow. Devuelve True si se registró correctamente.
    """
    if not msg.get("trace_id"):
        st.warning("Esta respuesta no tiene traza asociada; no se pudo registrar el feedback.")
        return False
    try:
        log_user_feedback(msg["trace_id"], thumbs_up)
        return True
    except Exception as e:
        st.warning(f"No se pudo registrar el feedback en MLflow: {e}")
        return False


def _render_sources(sources):
    """
    Muestra los fragmentos recuperados como referencias numeradas.
    """
    if not sources:
        return
    with st.expander(f"📚 Referencias ({len(sources)})"):
        for i, ref in enumerate(sources, 1):
            page = f" — página {ref['page']}" if ref["page"] else ""
            st.markdown(f"**{i}. {ref['document']}**{page}")
            snippet = ref["snippet"]
            st.caption(snippet[:400] + ("…" if len(snippet) > 400 else ""))


def _render_feedback(idx, msg):
    """
    Muestra los botones de 👍/👎 y registra el feedback del usuario.
    """
    key = f"fb_{idx}"
    if msg.get("feedback") is not None:
        st.session_state.setdefault(key, msg["feedback"])
    st.feedback("thumbs", key=key, on_change=_on_feedback, args=(idx,))


def _on_feedback(idx):
    """
    Callback de los botones de 👍/👎: registra el feedback del usuario.
    """
    value = st.session_state.get(f"fb_{idx}")  # 1 = 👍, 0 = 👎
    msg = st.session_state.messages[idx]
    if value is not None and value != msg.get("feedback") and _log_feedback(msg, bool(value)):
        msg["feedback"] = value


def _new_conversation():
    """
    Callback del botón "Nueva conversación": limpia el historial.
    """
    st.session_state.messages = []


# Sidebar
def render_sidebar():
    """
    Muestra la barra lateral con la configuración de la app y el botón de nueva conversación.
    """
    info = describe()
    st.sidebar.markdown("### ⚙️ Configuración")
    st.sidebar.markdown(
        f"**LLM:** `{info['llm_provider']}` · `{info['llm_model']}`  \n"
        f"**Embeddings:** `{info['embeddings_provider']}` · `{info['embeddings_model'].split('/')[-1]}`"
    )

    options = get_prompt_options()
    default = os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT
    if default not in options:
        options = [default, *options]
    prompt_version = st.sidebar.selectbox(
        "Prompt (MLflow Prompt Registry)", options, index=options.index(default),
        format_func=lambda o: o.replace("prompts:/", ""),
        help="@alias o versión fija. Registra cambios con `python -m app.model.prompt_registry`",
    )
    k = st.sidebar.slider("Fragmentos a recuperar (k)", 1, 10, 4,
                          help="Cuántos fragmentos de los prospectos se envían al LLM como contexto")

    st.sidebar.button("🗑️ Nueva conversación", width="stretch",
                      on_click=_new_conversation)

    return {"prompt_version": prompt_version, "k": k}


# Chat
def render_chat(title, subtitle):
    """
    Muestra el chat con el LLM, el prompt y el retriever configurados en la barra lateral.
    """
    config = render_sidebar()

    st.title(title)
    st.caption(subtitle)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        chain, prompt_uri = get_chain(resolve_prompt(config["prompt_version"]), config["k"])
    except Exception as e:
        st.error(f"No se pudo preparar el asistente: {e}")
        st.stop()

    # Trazas en el experimento del chat, enlazadas a esta versión de la app
    init_tracking(CHAT_EXPERIMENT, prompt_uri, config["k"])
    st.sidebar.caption(f"Prompt resuelto: `{prompt_uri.replace('prompts:/', '')}`")

    # Preguntas de ejemplo (solo al inicio de la conversación)
    pending = None
    examples_box = st.empty()
    if not st.session_state.messages:
        with examples_box.container():
            st.markdown("##### 💡 Prueba con alguna de estas preguntas")
            cols = st.columns(2)
            for i, q in enumerate(get_example_questions()):
                if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
                    pending = q

    # Historial
    for idx, msg in enumerate(st.session_state.messages):
        _render_message(idx, msg)

    question = st.chat_input("¿Qué deseas consultar sobre los FICs internacionales?") or pending
    if not question:
        return

    examples_box.empty()

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Consultando los prospectos..."):
            history = [(m["question"], m["answer"]) for m in st.session_state.messages]
            start = time.perf_counter()
            try:
                result = chain.invoke({"question": question, "chat_history": history})
                trace_id = mlflow.get_last_active_trace_id(thread_local=True)
            except Exception as e:
                st.error(f"Error al consultar el modelo: {e}")
                st.stop()
            latency = time.perf_counter() - start

        msg = {
            "question": question,
            "answer": result["answer"],
            "sources": _format_sources(result.get("source_documents")),
            "latency": latency,
            "prompt_version": prompt_uri.replace("prompts:/", ""),
            "k": config["k"],
            "trace_id": trace_id,
        }
        st.session_state.messages.append(msg)
        _render_answer(len(st.session_state.messages) - 1, msg)


def _render_answer(idx, msg):
    """
    Muestra la respuesta del LLM, las referencias y los botones de feedback.
    """
    st.markdown(msg["answer"])
    _render_sources(msg["sources"])
    st.caption(f"⏱️ {msg['latency']:.1f} s · prompt `{msg['prompt_version']}` · k={msg['k']}")
    _render_feedback(idx, msg)


def _render_message(idx, msg):
    """
    Muestra un mensaje del historial: pregunta, respuesta, referencias y feedback.
    """
    with st.chat_message("user", avatar="👤"):
        st.markdown(msg["question"])
    with st.chat_message("assistant", avatar="🤖"):
        _render_answer(idx, msg)