from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import os
from pydantic import BaseModel
from document_processor import DocumentProcessor
from config import Config
from rag_engine_faiss import RAGEngineFAISS

app = FastAPI()

# Allow CORS (configurable via ALLOWED_ORIGINS env, comma-separated)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000, http://127.0.0.1:3000, http://localhost:5173, *")
origin_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = RAGEngineFAISS()

class AskRequest(BaseModel):
    question: str

def format_sources(docs: List):
    if not docs:
        return ""
    out = "**📚 Source Documents:**\n\n"
    for i, doc in enumerate(docs, 1):
        content_preview = (
            (doc.page_content[:150] + "...") if len(doc.page_content) > 150 else doc.page_content
        )
        source = doc.metadata.get("source", "Unknown") if getattr(doc, "metadata", None) else "Unknown"
        out += f"**Source {i}** ({source}):\n{content_preview}\n\n"
    return out

@app.post("/api/ask")
async def ask_question(body: AskRequest):
    answer, source_docs = rag_system.query(body.question)
    sources = format_sources(source_docs)
    return {"answer": answer, "sources": sources}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp location
        temp_dir = os.path.join(os.getcwd(), "tmp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        safe_filename = os.path.basename(file.filename or "uploaded_file")
        temp_path = os.path.join(temp_dir, safe_filename)

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        ok = rag_system.load_and_index_file(temp_path)

        try:
            os.remove(temp_path)
        except Exception:
            pass

        status_msg = "✅ File processed and indexed" if ok else "❌ Failed to process file"
        return {"status": status_msg}
    except Exception as e:
        # Return structured error to the client
        return JSONResponse(status_code=500, content={"status": "❌ Upload failed", "detail": str(e)})


@app.get("/api/health")
async def health_check():
    try:
        # simple check: ensure rag_system exists and basic methods are callable
        _ = rag_system.get_document_count() if hasattr(rag_system, "get_document_count") else None
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@app.get("/api/stats")
async def get_stats():
    stats = rag_system.get_collection_stats()
    # Convert to markdown string for the frontend
    stats_md = (
        f"📊 System Statistics\n\n"
        f"💾 Storage: {stats.get('storage_type', 'FAISS')}\n"
        f"📁 Index Path: {stats.get('index_path', 'N/A')}\n"
        f"📄 Total Documents: {stats.get('total_documents', '0')}\n"
        f"🔢 Vector Dimension: {stats.get('vector_dimension', 'Unknown')}\n"
        f"🤖 AI Model: {Config.LLM_MODEL}"
    )
    return {"stats": stats_md}
