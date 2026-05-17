import os
from pathlib import Path
import chromadb
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HAI_API_KEY"),
    base_url="https://factchat-cloud.mindlogic.ai/v1/gateway",
)

EMBEDDING_MODEL = "gemini-2.5-flash"

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_document(file_path: str, doc_id: str) -> int:
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def retrieve_context(question: str, top_k: int = 3) -> list[str]:
    results = collection.query(query_texts=[question], n_results=top_k)
    return results["documents"][0] if results["documents"] else []
