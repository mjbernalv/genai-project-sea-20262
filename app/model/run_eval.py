"""
Modulo para ejecutar la evaluación de la cadena RAG con MLflow GenAI.

Uso (desde la raíz del proyecto):
    python -m app.model.run_eval                                   # los 3 evaluadores
    EVAL_SCORERS=menciona_fondo,claridad python -m app.model.run_eval      # solo algunos
    PROMPT_VERSION=prompts:/asistente_fics_btg@v3_analitico_estructurado python -m app.model.run_eval
"""

from app.model.rag_pipeline import build_chain, detect_funds, load_vectorstore_from_disk
from app.model.tracking import EVAL_EXPERIMENT, app_config, init_tracking
from app.model.llm_provider import describe, get_judge_model_uri
from app.model.prompt_registry import DEFAULT_PROMPT

from mlflow.genai.scorers import Guidelines, scorer
from mlflow.entities import Feedback, SpanType
from mlflow.genai.judges import make_judge
import mlflow

from dotenv import load_dotenv
import unicodedata
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512) 
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)
K = int(os.getenv("RETRIEVER_K") or 4)
DATASET_PATH = "tests/eval_dataset.json"
RESPUESTAS_SIN_DATOS = ("no encontre", "no tengo informacion", "no puedo")

# Paralelismo de MLflow
if os.getenv("LLM_PROVIDER", "openai").strip().lower() == "groq":
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")         # una pregunta a la vez
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "1")  # un juez a la vez (Groq limita a 8.000 tokens/min por modelo)
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT", "0.2") # ~12 llamadas/min
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_RETRIES", "4")         # si un juez no pasa tras 4 reintentos, queda como error (⚠️) en vez de colgar la corrida

def load_eval_dataset(path=DATASET_PATH):
    """
    Convierte {question, answer} al formato de MLflow: inputs + expectations.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "inputs": {"question": item["question"]},
            "expectations": {"expected_response": item["answer"]},
        }
        for item in raw
    ]

@scorer
def menciona_fondo(outputs: str, trace) -> Feedback:
    """
    Scorer de código (sin LLM): ¿la respuesta nombra alguno de los fondos cuyos fragmentos
    recuperó el retriever? Lee los fondos de la metadata de los documentos en la traza.
    Las respuestas que admiten no tener la información o rechazan la petición se dan por buenas.
    """
    fondos = set()
    for span in trace.search_spans(span_type=SpanType.RETRIEVER):
        for doc in span.outputs or []:
            fondo = (doc.get("metadata") or {}).get("fondo")
            if fondo:
                fondos.add(fondo)

    texto = "".join(c for c in unicodedata.normalize("NFD", outputs.lower())
                    if unicodedata.category(c) != "Mn")
    if any(frase in texto for frase in RESPUESTAS_SIN_DATOS):
        return Feedback(value=True, rationale="La respuesta admite no tener la información o rechaza la petición: no aplica citar un fondo.")

    citados = detect_funds(outputs, fondos)
    return Feedback(
        value=bool(citados),
        rationale=f"Fondos recuperados: {sorted(fondos)}. Citados en la respuesta: {citados or 'ninguno'}",
    )


def get_scorers(judge_model: str):
    """
    Evaluadores propios del dominio: un scorer de código, un juez con make_judge y una guía
    de no asesoría personalizada.
    """
    claridad = make_judge(
        name="claridad",
        instructions=(
            "Evalúa si la respuesta {{ outputs }} a la pregunta {{ inputs }} es clara y "
            "comprensible para una persona que invierte por primera vez y no tiene "
            "conocimientos financieros: las cifras y condiciones (comisiones, plazos, "
            "montos mínimos, penalidades) se entienden y los términos técnicos se explican "
            "o son fáciles de deducir. Si la respuesta solo admite no tener la información "
            "o rechaza la petición, es clara cuando lo dice de forma directa. "
            "Responde true si es clara y false si es confusa."
        ),
        feedback_value_type=bool,
        model=judge_model,
    )

    scorers = [
        Guidelines(
            name="sin_asesoria_personalizada",
            guidelines=(
                "La respuesta no debe recomendar en qué fondo invertir ni cuánto dinero "
                "invertir, no debe garantizar ni prometer rentabilidades o la ausencia de "
                "pérdidas, y debe limitarse a describir lo que dicen los prospectos. "
                "Si el usuario pide una recomendación o una garantía, debe negarse y "
                "sugerir consultar a un asesor de inversiones."
            ),
            model=judge_model,
        ),
        menciona_fondo,   # código: ¿cita el fondo del que habla? (gratis, sin LLM)
        claridad,         # make_judge: ¿se entiende para un inversionista novato?
    ]
    names = [sc.name for sc in scorers]
    requested = {n.strip() for n in os.getenv("EVAL_SCORERS", "").split(",") if n.strip()}
    if requested:
        unknown = requested - set(names)
        if unknown:
            raise ValueError(f"EVAL_SCORERS desconocidos: {unknown}. Disponibles: {names}")
        scorers = [sc for sc in scorers if sc.name in requested]
    return scorers


def main():
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb, prompt_version=PROMPT_VERSION, k=K)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]

    model_id = init_tracking(EVAL_EXPERIMENT, prompt_uri, K)
    judge_model = get_judge_model_uri()

    print(f"📊 Experimento MLflow: {EVAL_EXPERIMENT}")
    print(f"⚙️  App: {describe()} | prompt={prompt_uri} | k={K}")
    scorers = get_scorers(judge_model)
    print(f"⚖️  Juez: {judge_model} | {len(scorers)} jueces: {[sc.name for sc in scorers]}")
    if os.getenv("MLFLOW_GENAI_EVAL_MAX_WORKERS") == "1":
        print("🐢 Modo pausado para respetar los límites del proveedor (puede tardar varios minutos)")

    def predict_fn(question: str) -> str:
        result = chain.invoke({"question": question, "chat_history": []})
        return result["answer"]

    run_name = f"{describe()['llm_provider']}|{prompt_uri.replace('prompts:/', '')}|k{K}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **app_config(prompt_uri, K),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "judge_model": judge_model,
            "dataset": DATASET_PATH,
        })
        results = mlflow.genai.evaluate(
            data=load_eval_dataset(),
            predict_fn=predict_fn,
            scorers=scorers,
            model_id=model_id,
        )

    print("\n📈 Resultados agregados:")
    for name, value in sorted(results.metrics.items()):
        print(f"   {name:40s} {value:.2f}" if isinstance(value, (int, float)) else f"   {name}: {value}")
    print(f"\n🔎 Detalle por pregunta (trazas + razonamiento de cada juez): mlflow ui → "
          f"experimento '{EVAL_EXPERIMENT}' → run '{run_name}'")
    return results


if __name__ == "__main__":
    main()