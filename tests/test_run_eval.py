# tests/test_run_eval.py
"""
Quality gate: la última evaluación (python -m app.model.run_eval) debe superar
los umbrales mínimos de los jueces de MLflow.
"""
import mlflow
import pytest
import os

EVAL_EXPERIMENT = "rag_evaluation"

# Umbrales mínimos (promedio 0..1 de cada juez). Ajusta según tu caso.
THRESHOLDS = {
    "menciona_fondo/mean": float(os.getenv("MIN_MENCIONA_FONDO", 0.9)),
    "claridad/mean": float(os.getenv("MIN_CLARIDAD", 0.8)),
    "sin_asesoria_personalizada/mean": float(os.getenv("MIN_SIN_ASESORIA", 0.8)),
}

@pytest.fixture(scope="module")
def latest_eval_run():
    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    if experiment is None:
        pytest.skip("No hay evaluaciones. Ejecuta primero: python app/model/run_eval.py")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        pytest.skip("El experimento de evaluación no tiene ejecuciones")
    return runs.iloc[0]


@pytest.mark.parametrize("metric,threshold", THRESHOLDS.items())
def test_umbral_minimo(latest_eval_run, metric, threshold):
    column = f"metrics.{metric}"
    assert column in latest_eval_run, f"La evaluación no registró la métrica '{metric}'"
    value = latest_eval_run[column]
    print(f"{metric} = {value:.2f} (mínimo {threshold})")
    assert value >= threshold, f"{metric} = {value:.2f} está por debajo del umbral {threshold}"