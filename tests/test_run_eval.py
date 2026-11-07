# tests/test_run_eval.py
"""
Quality gate: la última evaluación de la configuración que usa la app (prompt de producción,
RETRIEVER_K y CHUNK_SIZE) debe superar los umbrales mínimos de los jueces.

Se filtra por configuración porque run_experiments.py deja en MLflow varias evaluaciones
(otros prompts, k y chunks), y la última que termina no es necesariamente la de producción.
"""
import mlflow
import pytest
import math
import os

EVAL_EXPERIMENT = "rag_evaluation"
PROMPT_NAME = os.getenv("PROMPT_NAME", "asistente_fics_btg")

# Umbrales mínimos (promedio 0..1 de cada juez). Ajusta según tu caso.
THRESHOLDS = {
    "menciona_fondo/mean": float(os.getenv("MIN_MENCIONA_FONDO", 0.9)),
    "claridad/mean": float(os.getenv("MIN_CLARIDAD", 0.8)),
    "sin_asesoria_personalizada/mean": float(os.getenv("MIN_SIN_ASESORIA", 0.8)),
    "tono_profesional/mean": float(os.getenv("MIN_TONO_PROFESIONAL", 0.9)),
}


def production_config():
    """
    (prompt, k, chunk_size) que usa la app: PROMPT_VERSION o el alias @production, RETRIEVER_K y CHUNK_SIZE.
    El prompt se resuelve a su versión (prompts:/nombre/N), que es como lo registra run_eval.
    """
    ref = os.getenv("PROMPT_VERSION") or f"prompts:/{PROMPT_NAME}@production"
    if ref.startswith("prompts:/") and "@" in ref:
        pv = mlflow.genai.load_prompt(ref, allow_missing=True, cache_ttl_seconds=0)
        ref = f"prompts:/{pv.name}/{pv.version}" if pv else None
    return ref, int(os.getenv("RETRIEVER_K") or 4), int(os.getenv("CHUNK_SIZE") or 512)


@pytest.fixture(scope="module")
def latest_eval_run():
    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    if experiment is None:
        pytest.skip("No hay evaluaciones. Ejecuta primero: python -m app.model.run_eval")
    prompt, k, chunk_size = production_config()
    if prompt is None:
        pytest.fail(f"No se encontró el alias @production del prompt '{PROMPT_NAME}'. "
                    "Ejecuta: python -m app.model.prompt_registry")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(f"status = 'FINISHED' and params.prompt = '{prompt}' "
                       f"and params.k = '{k}' and params.chunk_size = '{chunk_size}'"),
        order_by=["start_time DESC"], max_results=1,
    )
    if runs.empty:
        pytest.fail(f"No hay una evaluación terminada de la configuración de producción "
                    f"(prompt={prompt}, k={k}, chunk_size={chunk_size}). Ejecuta: python -m app.model.run_eval")
    return runs.iloc[0]


@pytest.mark.parametrize("metric,threshold", THRESHOLDS.items())
def test_umbral_minimo(latest_eval_run, metric, threshold):
    column = f"metrics.{metric}"
    assert column in latest_eval_run and not math.isnan(latest_eval_run[column]), \
        f"La evaluación no registró la métrica '{metric}'"
    value = latest_eval_run[column]
    print(f"{metric} = {value:.2f} (mínimo {threshold})")
    assert value >= threshold, f"{metric} = {value:.2f} está por debajo del umbral {threshold}"