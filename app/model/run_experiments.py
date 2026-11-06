"""
Modulo para evaluar todas las combinaciones de prompt, k y tamaño de chunk con run_eval.
"""
from app.model.run_eval import CHUNK_OVERLAP, main as run_eval, get_scorers
from app.model.rag_pipeline import VECTOR_DIR, load_vectorstore_from_disk, save_vectorstore
from app.model.tracking import EVAL_EXPERIMENT
from app.model.llm_provider import get_llm_provider
from app.model.prompt_registry import PROMPT_NAME

import mlflow

import itertools
import argparse
import time
import os

DEFAULT_PROMPTS = (1, 2, 3)
DEFAULT_KS = (4, 8)
DEFAULT_CHUNKS = (512, 1024)
GROQ_PAUSE_SECONDS = 90


def default_pause() -> float:
    """
    Segundos de espera entre configuraciones: EXPERIMENT_PAUSE, o 90 con Groq (límite de tokens/min) y 0 con otros.
    """
    if os.getenv("EXPERIMENT_PAUSE"):
        return float(os.environ["EXPERIMENT_PAUSE"])
    return GROQ_PAUSE_SECONDS if get_llm_provider() == "groq" else 0


def index_dir(chunk_size: int) -> str:
    return f"{VECTOR_DIR}_c{chunk_size}"


def ensure_index(chunk_size: int):
    """
    Carga el índice de ese tamaño de chunk; si no existe, lo construye (y queda registrado en MLflow).
    """
    path = index_dir(chunk_size)
    if not os.path.exists(path):
        print(f"🏗️  Construyendo el índice con chunk_size={chunk_size} en '{path}'...")
        save_vectorstore(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP, persist_path=path)
    return load_vectorstore_from_disk(path)


def evaluated_configs() -> set[tuple[str, str, str]]:
    """
    Configuraciones (prompt, k, chunk_size) cuya evaluación terminó con las métricas de todos los jueces.
    """
    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    if experiment is None:
        return set()
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="status = 'FINISHED'")
    needed = [f"metrics.{sc.name}/mean" for sc in get_scorers("openai:/placeholder")]
    done = set()
    for _, run in runs.iterrows():
        if all(c in runs.columns and run[c] == run[c] for c in needed):  # run[c] == run[c]: descarta NaN
            done.add((run["params.prompt"], str(run["params.k"]), str(run["params.chunk_size"])))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=int, nargs="+", default=DEFAULT_PROMPTS, help="versiones del prompt")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS, help="valores de k")
    parser.add_argument("--chunks", type=int, nargs="+", default=DEFAULT_CHUNKS, help="tamaños de chunk")
    parser.add_argument("--pause", type=float, default=default_pause(), metavar="SEGUNDOS",
                        help="espera entre configuraciones, para no superar los límites del proveedor "
                             "(por defecto 90 con Groq, 0 con otros)")
    parser.add_argument("--force", action="store_true", help="repetir también las configuraciones ya evaluadas")
    parser.add_argument("--dry-run", action="store_true", help="mostrar el plan sin ejecutar nada")
    args = parser.parse_args()

    done = set() if args.force else evaluated_configs()
    plan = [(c, p, k) for c, p, k in itertools.product(args.chunks, args.prompts, args.ks)]
    uri = lambda p: f"prompts:/{PROMPT_NAME}/{p}"
    pending = [(c, p, k) for c, p, k in plan if (uri(p), str(k), str(c)) not in done]

    print(f"📋 {len(plan)} configuraciones, {len(plan) - len(pending)} ya evaluadas, {len(pending)} por ejecutar:")
    for c, p, k in plan:
        mark = "⏭️ " if (c, p, k) not in pending else "▶️ "
        print(f"   {mark} prompt v{p} | k={k} | chunk={c}")
    if args.dry_run or not pending:
        return

    failed = []
    started = time.time()
    first = True
    for chunk in dict.fromkeys(c for c, _, _ in pending):  # un índice por tamaño de chunk
        vectordb = ensure_index(chunk)
        for c, p, k in pending:
            if c != chunk:
                continue
            if not first and args.pause > 0:
                print(f"\n⏳ Pausa de {args.pause:.0f} s antes de la siguiente configuración (límites del proveedor)...")
                time.sleep(args.pause)
            first = False
            print(f"\n{'=' * 80}\n🧪 prompt v{p} | k={k} | chunk={c}  ({(time.time() - started) / 60:.0f} min desde el inicio)")
            try:
                run_eval(prompt_version=uri(p), k=k, chunk_size=c, vectordb=vectordb)
            except Exception as e:  # una corrida fallida (por ejemplo, límites del proveedor) no detiene las demás
                print(f"❌ Falló v{p} | k={k} | chunk={c}: {e!r}")
                failed.append((c, p, k))

    print(f"\n✅ Terminado en {(time.time() - started) / 60:.0f} min. Fallidas: {len(failed)}")
    for c, p, k in failed:
        print(f"   v{p} | k={k} | chunk={c}")
    if failed:
        print("Vuelve a lanzar el comando para reintentarlas: las ya evaluadas se omiten.")


if __name__ == "__main__":
    main()
