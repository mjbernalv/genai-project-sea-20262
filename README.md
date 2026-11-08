# Proyecto IA Generativa - Seminario de experiencias en analítica

Este repositorio contiene el trabajo sobre inteligencia artificial generativa realizado para la materia de *Seminario de experiencias en analítica* de la Maestría en ciencia de datos y analítica en el semestre 2026-2. 

**Por:**
- Alejandro Olivares
- Santiago Zapata
- María José Bernal

## Asistente de FICs internacionales de BTG Pactual Colombia

Aplicación RAG (Streamlit + LangChain + MLflow) que responde preguntas sobre los prospectos de los
Fondos de Inversión Colectiva internacionales de BTG Pactual Colombia (Acciones Brasil, Acciones
Internacionales, Acciones Mercados Emergentes, Brasil Soberano, Liquidez Dólares, Oro y Renta Fija
Internacional). Los PDF están en `data/pdfs/`.

**Ejecución** (desde la raíz del proyecto, con las variables de `.env.example` copiadas a `.env`):

```bash
uv sync                                          # o: pip install -r requirements.txt
python -c "from app.model.rag_pipeline import save_vectorstore; save_vectorstore()"   # opcional: la app indexa sola la primera vez
python -m app.model.prompt_registry              # registra los prompts de app/prompts/ en MLflow
streamlit run app/main_interface.py              # chat + métricas
python -m app.model.run_eval                     # evaluación con tests/eval_dataset.json
mlflow ui --port 5000                            # trazas, prompts y evaluaciones
```

- `app/prompts/`: versiones del prompt del asistente (cada `.txt` queda como un alias en MLflow).
- `tests/eval_dataset.json`: preguntas y respuestas esperadas para evaluar el RAG; las primeras también aparecen como preguntas de ejemplo en el chat.
- La primera vez se descarga el modelo de embeddings (`jinaai/jina-embeddings-v2-base-es`, ~0,6 GB) y se indexan los PDF (1-5 min).

