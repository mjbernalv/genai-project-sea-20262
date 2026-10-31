"""
Modulo para construir un pipeline RAG (retrieval-augmented generation) usando LangChain y MLflow
sobre los prospectos de los FIC internacionales de BTG Pactual Colombia.
Incluye funciones para:
- Cargar documentos PDF y dividirlos en chunks.
- Crear un vectorstore FAISS con embeddings.
- Registrar y cargar prompts desde MLflow.
- Construir un ConversationalRetrievalChain con un LLM y un retriever.
"""
from app.model.prompt_registry import DEFAULT_PROMPT, load_prompt_text, list_prompt_options
from app.model.llm_provider import get_llm, get_embeddings, get_vectorstore_dir, describe

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.globals import set_verbose

from dotenv import load_dotenv
from typing import Any
import unicodedata
import mlflow
import os
import re

set_verbose(True)
load_dotenv()

DATA_DIR = "data/pdfs"
VECTOR_DIR = get_vectorstore_dir()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)

# Cada página de los prospectos repite el mismo aviso legal: se elimina
_LEGAL_NOTICE = re.compile(
    r"Las\s+obligaciones\s+de\s+la\s+Sociedad\s+Administradora.*?portafolio\s+del\s+Fondo\.",
    re.DOTALL,
)
_FUND_NAME = re.compile(r"[“\"](BTG\s+PACTUAL[^”\"]+)[”\"]")
SECTION_SEPARATORS = [r"\n(?=\d{1,2}\. ?\d{2}\s)", r"\n\n", r"\n", r" ", r""]
MIN_CHUNK_CHARS = 40


def _fund_name(first_page_text: str, filename: str) -> str:
    """
    Nombre del fondo según la carátula del prospecto ("BTG PACTUAL ORO"); si no se
    encuentra, se deduce del nombre del archivo.
    """
    match = _FUND_NAME.search(first_page_text)
    if match:
        return " ".join(match.group(1).split())
    return os.path.splitext(filename)[0].replace("Prospecto FIC", "").strip()


def load_documents(path=DATA_DIR):
    """
    Carga todos los documentos PDF de la carpeta y devuelve una lista de Documentos de LangChain.
    Cada página queda etiquetada con el fondo al que pertenece (metadata["fondo"]).
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No existe la carpeta de prospectos '{path}'. Ejecuta la app desde la raíz del proyecto.")

    docs = []
    for file in sorted(os.listdir(path)):
        if file.endswith(".pdf"):
            pages = PyPDFLoader(os.path.join(path, file)).load()
            if not pages:
                continue
            fund = _fund_name(pages[0].page_content, file)
            for page in pages:
                page.metadata["fondo"] = fund
                if page.metadata.get("page", 0) > 0:
                    page.page_content = _LEGAL_NOTICE.sub("", page.page_content)
            docs.extend(pages)
    if not docs:
        raise FileNotFoundError(f"No se encontraron PDFs en '{path}'.")
    return docs


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Divide los documentos en chunks y antepone el nombre del fondo a cada uno
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SECTION_SEPARATORS,
        is_separator_regex=True,
    )
    chunks = []
    for chunk in splitter.split_documents(docs):
        if len(chunk.page_content.strip()) < MIN_CHUNK_CHARS:
            continue
        chunk.page_content = f"[Fondo: {chunk.metadata['fondo']}]\n{chunk.page_content}"
        chunks.append(chunk)
    return chunks


def save_vectorstore(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, persist_path=VECTOR_DIR):
    """
    Crea un vectorstore FAISS a partir de los documentos PDF y lo guarda en disco.
    """
    docs = load_documents()
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(persist_path)

    mlflow.set_experiment("vectorstore_tracking")
    with mlflow.start_run(run_name="vectorstore_build"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("n_chunks", len(chunks))
        mlflow.log_param("n_docs", len(docs))
        mlflow.log_params(describe())
        mlflow.set_tag("vectorstore", persist_path)


def load_vectorstore(chunk_size=512, chunk_overlap=50):
    """
    Construye un vectorstore FAISS en memoria a partir de los PDF.
    """
    docs = load_documents()
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)


def load_vectorstore_from_disk(persist_path=VECTOR_DIR):
    """
    Carga desde disco un vectorstore FAISS previamente guardado.
    """
    if not os.path.exists(persist_path):
        raise FileNotFoundError(
            f"No existe el índice en '{persist_path}'. Genéralo primero con: "
            'python -c "from app.model.rag_pipeline import save_vectorstore; save_vectorstore()"'
        )
    embeddings = get_embeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


def load_prompt(version=DEFAULT_PROMPT):
    """
    Carga el prompt desde el Prompt Registry de MLflow (prompts:/nombre@alias)
    o desde un archivo local de app/prompts/ (nombre sin .txt).
    La URI resuelta queda en prompt.metadata["prompt_uri"].
    """
    prompt_text, uri = load_prompt_text(version)
    return PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_text,
        metadata={"prompt_uri": uri},
    )


def list_prompt_versions():
    """
    Alias y versiones disponibles en el registro (o archivos locales si no hay registro).
    """
    return list_prompt_options()


def _normalize(text: str) -> str:
    """
    Minúsculas y sin tildes, para comparar nombres de fondos con la pregunta.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _fund_aliases(fund: str) -> set[str]:
    """
    Formas en que un usuario suele nombrar un fondo.
    """
    name = _normalize(fund).replace("btg pactual", "").strip()
    words = name.split()
    aliases = {name}
    if words[0] == "acciones" and len(words) >= 3:
        aliases.add(" ".join(words[1:]))
    elif len(words) >= 3:
        aliases.add(" ".join(words[:2]))
    return aliases


def detect_funds(question: str, funds) -> list[str]:
    """
    Fondos que la pregunta menciona por nombre (lista vacía si no menciona ninguno).
    """
    q = _normalize(question)
    return [f for f in sorted(funds)
            if any(re.search(rf"\b{re.escape(a)}\b", q) for a in _fund_aliases(f))]


class FundAwareRetriever(BaseRetriever):
    """
    Los prospectos tienen la misma estructura, así que las comisiones o plazos de un fondo se
    confunden fácilmente con los de otro. Si la pregunta nombra uno o varios fondos, la búsqueda
    se limita a sus fragmentos (k por cada fondo mencionado, para que las comparaciones queden
    equilibradas); si no nombra ninguno, busca en todos.
    """
    vectordb: Any
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        funds = {d.metadata.get("fondo") for d in self.vectordb.docstore._dict.values()} - {None}
        mentioned = detect_funds(query, funds)
        if not mentioned:
            return self.vectordb.similarity_search(query, k=self.k)

        docs = []
        for fund in mentioned:
            docs.extend(self.vectordb.similarity_search(
                query, k=self.k, fetch_k=self.vectordb.index.ntotal,
                filter=lambda meta, fund=fund: meta.get("fondo") == fund,
            ))
        return docs


def build_chain(vectordb, prompt_version=DEFAULT_PROMPT, k=4):
    """"
    Construye un ConversationalRetrievalChain con el LLM, el prompt y el retriever.
    """
    prompt = load_prompt(prompt_version)
    retriever = FundAwareRetriever(vectordb=vectordb, k=k)
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(temperature=0),
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True 
    )