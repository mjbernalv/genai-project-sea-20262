"""
Componente de dashboard para main_interface.py: compara dos versiones del prompt lado a lado
(plantilla + métricas) y muestra el costo (tokens y latencia) por versión de la app.
"""
from app.model.tracking import CHAT_EXPERIMENT, DEFAULT_CHUNK_SIZE, EVAL_EXPERIMENT
from app.model.prompt_registry import PROMPT_NAME

import streamlit as st
import pandas as pd
import mlflow

from mlflow import MlflowClient
import difflib
import os

VERSION_COL = "versión de la app"
COST_COLUMNS = ["model_id", "latencia_s", "tokens_entrada", "tokens_salida", "tokens_total"]

# Utilidades
def _metric_name(column):
    """
    "metrics.claridad/mean" → "claridad"
    """
    return column.replace("metrics.", "").replace("/mean", "")


def _version_label(version):
    """
    Etiqueta legible de una versión del prompt: "v2 · @production, @v2_resumido_directo".
    """
    aliases = ", ".join(f"@{a}" for a in version["aliases"])
    return f"v{version['version']}" + (f" · {aliases}" if aliases else "")


@st.cache_data(ttl=30, show_spinner=False)
def _prompt_versions():
    """
    Versiones registradas del prompt en MLflow (la más reciente primero), con su plantilla y alias.
    """
    try:
        found = list(MlflowClient().search_prompt_versions(PROMPT_NAME))
    except Exception:
        return []

    versions = []
    for v in sorted(found, key=lambda v: int(v.version), reverse=True):
        pv = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{v.version}")
        versions.append({
            "version": int(v.version),
            "uri": f"prompts:/{PROMPT_NAME}/{v.version}",
            "aliases": sorted(pv.aliases),
            "template": pv.to_single_brace_format(),
            "commit": pv.commit_message,
        })
    return versions


@st.cache_data(ttl=30, show_spinner=False)
def _eval_runs():
    """
    Ejecuciones de run_eval (la más reciente primero).
    """
    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    if experiment is None:
        return pd.DataFrame()
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])


@st.cache_data(ttl=30, show_spinner="Leyendo trazas...")
def _trace_table(experiment_name, run_id=None):
    """
    Una fila por traza de la app (latencia y tokens), con el nombre de la versión de la app que la generó.
    Solo se cuentan las trazas enlazadas a una versión de la app (mlflow.modelId).
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame(columns=[*COST_COLUMNS, VERSION_COL])

    filters = {"run_id": run_id} if run_id else {}
    traces = mlflow.search_traces(locations=[experiment.experiment_id], max_results=1000,
                                  return_type="list", **filters)
    rows = []
    for t in traces:
        model_id = (t.info.trace_metadata or {}).get("mlflow.modelId")
        if not model_id:
            continue
        usage = t.info.token_usage or {}
        duration_ms = t.info.execution_duration
        rows.append({
            "model_id": model_id,
            "latencia_s": duration_ms / 1000 if duration_ms is not None else None,
            "tokens_entrada": usage.get("input_tokens"),
            "tokens_salida": usage.get("output_tokens"),
            "tokens_total": usage.get("total_tokens"),
        })

    df = pd.DataFrame(rows, columns=COST_COLUMNS)
    df[COST_COLUMNS[1:]] = df[COST_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    names = {m.model_id: m.name for m in mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id], output_format="list")}
    df[VERSION_COL] = df["model_id"].map(lambda m: names.get(m, m))
    return df


def _summarize_costs(df):
    """
    Promedios por versión de la app: trazas, latencia (media y p95) y tokens (entrada, salida y total).
    """
    return df.groupby(VERSION_COL).agg(
        trazas=("latencia_s", "size"),
        latencia_media_s=("latencia_s", "mean"),
        latencia_p95_s=("latencia_s", lambda s: s.quantile(0.95)),
        tokens_entrada=("tokens_entrada", "mean"),
        tokens_salida=("tokens_salida", "mean"),
        tokens_total=("tokens_total", "mean"),
    )


def _version_results(uri, runs):
    """
    Métricas (promedio de cada juez) y costos de la última evaluación hecha con esa versión del prompt.
    Devuelve (run, metricas, costos); run es None si esa versión no se ha evaluado.
    """
    if runs.empty or "params.prompt" not in runs.columns:
        return None, {}, {}
    evaluated = runs[runs["params.prompt"] == uri]
    if evaluated.empty:
        return None, {}, {}

    base = evaluated[(evaluated["params.k"] == str(os.getenv("RETRIEVER_K") or 4))
                     & (evaluated["params.chunk_size"] == str(os.getenv("CHUNK_SIZE") or DEFAULT_CHUNK_SIZE))]
    evaluated = base if not base.empty else evaluated

    run = evaluated.iloc[0]
    metrics = {_metric_name(c): float(run[c]) for c in runs.columns
               if c.startswith("metrics.") and c.endswith("/mean") and pd.notna(run[c])}
    traces = _trace_table(EVAL_EXPERIMENT, run["run_id"])
    costs = {}
    if not traces.empty:
        costs = {"latencia media (s)": traces["latencia_s"].mean(),
                 "tokens totales (media)": traces["tokens_total"].mean()}
    return run, metrics, costs


# Secciones
def _render_prompt_comparison():
    """
    Compara dos versiones del prompt lado a lado: plantilla y métricas de su última evaluación.
    """
    st.subheader("🆚 Comparar versiones del prompt")
    versions = _prompt_versions()
    if len(versions) < 2:
        st.info("Necesitas al menos dos versiones registradas del prompt. "
                "Ejecuta `python -m app.model.prompt_registry`.")
        return

    by_uri = {v["uri"]: v for v in versions}
    uris = list(by_uri)
    runs = _eval_runs()

    # Por defecto, las dos versiones más recientes que ya tienen evaluación (A la anterior, B la nueva)
    evaluated = [u for u in uris if _version_results(u, runs)[0] is not None]
    default_b, default_a = (evaluated[0], evaluated[1]) if len(evaluated) >= 2 else (uris[0], uris[1])

    col_a, col_b = st.columns(2)
    uri_a = col_a.selectbox("Versión A", uris, index=uris.index(default_a),
                            format_func=lambda u: _version_label(by_uri[u]))
    uri_b = col_b.selectbox("Versión B", uris, index=uris.index(default_b),
                            format_func=lambda u: _version_label(by_uri[u]))
    if uri_a == uri_b:
        st.warning("Elegiste la misma versión en A y en B.")

    results = {}
    for column, uri in ((col_a, uri_a), (col_b, uri_b)):
        version = by_uri[uri]
        run, metrics, costs = _version_results(uri, runs)
        results[uri] = (metrics, costs)
        with column:
            st.markdown(f"#### {_version_label(version)}")
            if version["commit"]:
                st.caption(version["commit"])
            st.code(version["template"], language=None, wrap_lines=True)

            if run is None:
                st.info(f"Esta versión aún no se ha evaluado. Ejecuta "
                        f"`PROMPT_VERSION={uri} python -m app.model.run_eval`.")
                continue
            st.caption(f"Última evaluación: {run['tags.mlflow.runName']} · {run['start_time']:%Y-%m-%d %H:%M}")
            st.dataframe(
                pd.DataFrame({"promedio": metrics}), width="stretch",
                column_config={"promedio": st.column_config.ProgressColumn(
                    "promedio (0 a 1)", min_value=0, max_value=1, format="%.2f")},
            )
            if costs:
                st.dataframe(pd.DataFrame({"promedio": costs}).round(2), width="stretch")

    (metrics_a, costs_a), (metrics_b, costs_b) = results[uri_a], results[uri_b]
    if metrics_a and metrics_b:
        st.markdown("#### Diferencia (B − A)")
        names = [m for m in metrics_a if m in metrics_b]
        delta = pd.DataFrame({"A": [metrics_a[m] for m in names],
                              "B": [metrics_b[m] for m in names]}, index=names)
        delta["B − A"] = delta["B"] - delta["A"]
        st.dataframe(delta, width="stretch", column_config={
            "A": st.column_config.ProgressColumn("A", min_value=0, max_value=1, format="%.2f"),
            "B": st.column_config.ProgressColumn("B", min_value=0, max_value=1, format="%.2f"),
            "B − A": st.column_config.NumberColumn("B − A", format="%+.2f"),
        })

    with st.expander("Diferencias entre las plantillas"):
        diff = difflib.unified_diff(by_uri[uri_a]["template"].splitlines(), by_uri[uri_b]["template"].splitlines(),
                                    fromfile=_version_label(by_uri[uri_a]), tofile=_version_label(by_uri[uri_b]),
                                    lineterm="")
        st.code("\n".join(diff) or "Las plantillas son idénticas.", language="diff", wrap_lines=True)


def _render_costs():
    """
    Muestra el promedio de tokens y la latencia por versión de la app, leyendo las trazas de MLflow.
    """
    st.subheader("💸 Costos y latencia por versión de la app")
    source = st.radio("Origen de las trazas", ["Evaluación", "Chat"], horizontal=True,
                      help="Evaluación: preguntas de run_eval. Chat: conversaciones reales en la app.")
    df = _trace_table(EVAL_EXPERIMENT if source == "Evaluación" else CHAT_EXPERIMENT)
    if df.empty:
        st.info("Aún no hay trazas de " + ("evaluaciones. Ejecuta `python -m app.model.run_eval`."
                                          if source == "Evaluación" else "conversaciones."))
        return

    summary = _summarize_costs(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Trazas", len(df))
    c2.metric("Latencia media", f"{df['latencia_s'].mean():.1f} s")
    c3.metric("Tokens por consulta (media)",
              f"{df['tokens_total'].mean():,.0f}" if df["tokens_total"].notna().any() else "—")

    st.dataframe(summary.round(2), width="stretch", column_config={
        "trazas": st.column_config.NumberColumn("trazas"),
        "latencia_media_s": st.column_config.NumberColumn("latencia media (s)", format="%.2f"),
        "latencia_p95_s": st.column_config.NumberColumn("latencia p95 (s)", format="%.2f"),
        "tokens_entrada": st.column_config.NumberColumn("tokens entrada", format="%.0f"),
        "tokens_salida": st.column_config.NumberColumn("tokens salida", format="%.0f"),
        "tokens_total": st.column_config.NumberColumn("tokens total", format="%.0f"),
    })
    if df["tokens_total"].isna().all():
        st.warning("Las trazas no traen conteo de tokens: el proveedor del LLM no lo reporta a MLflow.")

    chart_latency, chart_tokens = st.columns(2)
    with chart_latency:
        st.caption("Latencia media por versión (s)")
        st.bar_chart(summary["latencia_media_s"])
    with chart_tokens:
        st.caption("Tokens medios por versión (entrada + salida)")
        st.bar_chart(summary[["tokens_entrada", "tokens_salida"]])


def render_dashboard():
    """
    Muestra el dashboard: comparación de prompts y costos, en la interfaz de Streamlit.
    """
    st.title("🆚 Prompts y costos")
    st.caption("Datos de MLflow. Compara versiones del prompt y revisa cuántos tokens y cuánto tiempo "
               "consume cada versión de la app.")

    tab_prompts, tab_costs = st.tabs(["🆚 Comparar prompts", "💸 Costos y latencia"])
    with tab_prompts:
        _render_prompt_comparison()
    with tab_costs:
        _render_costs()
