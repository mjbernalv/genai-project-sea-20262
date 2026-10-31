"""
Modulo para inicializar el tracking de MLflow y registrar la versión activa de la app.
"""

from mlflow.entities import AssessmentSource, AssessmentSourceType
import mlflow

from app.model.llm_provider import describe

import os
import re

CHAT_EXPERIMENT = "chatbot_rrhh"
EVAL_EXPERIMENT = "rag_evaluation"

_autolog_enabled = False
_versions_with_params: set[str] = set()

def enable_tracing():
    """
    Activa el tracing automático de LangChain (una sola vez por proceso).
    """
    global _autolog_enabled
    if not _autolog_enabled:
        mlflow.langchain.autolog(silent=True)
        _autolog_enabled = True


def app_config(prompt_uri: str, k: int) -> dict:
    """
    Devuelve un diccionario con la configuración activa de la app, para loguear en MLflow.
    """
    return {**describe(), "prompt": prompt_uri, "k": str(k)}


def _prompt_label(prompt_uri: str) -> str:
    """
    prompts:/asistente_rrhh/2 → asistente_rrhh-v2 ; app/prompts/v1_x.txt → local-v1_x
    """
    if prompt_uri.startswith("prompts:/"):
        return prompt_uri.replace("prompts:/", "").replace("/", "-v")
    return "local-" + os.path.splitext(os.path.basename(prompt_uri))[0]


def app_version_name(prompt_uri: str, k: int) -> str:
    """
    Nombre legible y estable para la versión de la app.
    MLflow no permite '/', ':', '.', '%', comillas en el nombre: se reemplazan por '-'.
    """
    cfg = describe()
    raw = f"rag-{cfg['llm_provider']}-{cfg['llm_model']}-{_prompt_label(prompt_uri)}-k{k}"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", raw)


def set_app_version(prompt_uri: str, k: int) -> str:
    """
    Marca la versión activa de la app: las trazas generadas a partir de ahora
    quedan enlazadas a ella. Devuelve el model_id.
    """
    active = mlflow.set_active_model(name=app_version_name(prompt_uri, k))
    if active.model_id not in _versions_with_params:
        # Los parámetros de un LoggedModel no se pueden sobrescribir: solo la primera vez
        if not mlflow.get_logged_model(active.model_id).params:
            mlflow.log_model_params(app_config(prompt_uri, k), model_id=active.model_id)
        _versions_with_params.add(active.model_id)
    return active.model_id


def init_tracking(experiment: str, prompt_uri: str, k: int) -> str:
    """
    Inicializa el tracking de MLflow y marca la versión activa de la app.
    Devuelve el model_id de la versión activa.
    """
    mlflow.set_experiment(experiment)
    enable_tracing()
    return set_app_version(prompt_uri, k)


def log_user_feedback(trace_id: str, thumbs_up: bool, user: str = "streamlit_user",
                      comment: str | None = None):
    """
    Adjunta el 👍/👎 del usuario a la traza de la respuesta.
    """
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_feedback",
        value=thumbs_up,
        rationale=comment,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=user),
    )