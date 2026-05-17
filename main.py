import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI

from rag import ingest_document, retrieve_context

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("HAI_API_KEY"),
    base_url="https://factchat-cloud.mindlogic.ai/v1/gateway",
)

FALLBACK_CHAIN = [
    "gpt-5-mini",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
]

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ChatRequest(BaseModel):
    message: str
    model: str | None = None
    use_rag: bool = False


@app.get("/")
def root():
    return {"status": "GatewayRAG is running"}


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id = file.filename.replace(".pdf", "")
    num_chunks = ingest_document(str(file_path), doc_id)

    return {
        "status": "ingested",
        "filename": file.filename,
        "chunks_created": num_chunks,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    user_message = req.message

    context_chunks = []
    if req.use_rag:
        context_chunks = retrieve_context(req.message, top_k=3)
        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            user_message = (
                f"Use the following context to answer the question. "
                f"If the answer isn't in the context, say so.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"QUESTION: {req.message}"
            )

    models_to_try = [req.model] if req.model else FALLBACK_CHAIN
    errors = []

    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_message}],
            )
            return {
                "reply": response.choices[0].message.content,
                "model_used": model_name,
                "fallbacks_attempted": errors,
                "rag_used": req.use_rag,
                "context_chunks_retrieved": len(context_chunks),
            }
        except Exception as e:
            errors.append({"model": model_name, "error": str(e)})
            continue

    raise HTTPException(
        status_code=502,
        detail={"message": "All models failed", "errors": errors},
    )
