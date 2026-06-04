import os
import uuid
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure stdout encodes correctly to prevent crashes on Windows terminal
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import local pipeline components
from document_processor import extract_text_from_file, chunk_text
from rag_pipeline import RAGPipeline

app = FastAPI(title="Interactive Document RAG Engine")

# Global pipeline instance
pipeline = None

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("VERCEL"):
    UPLOAD_DIR = "/tmp/uploads"
else:
    UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("[STARTUP] Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    print("[STARTUP] RAG Pipeline initialized.")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    search_mode: str = "hybrid"
    hybrid_weight: float = 0.5
    provider: str = "mock"
    api_key: str = ""
    model_name: str = ""

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    try:
        # 1. Retrieve matching chunks
        contexts = pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            search_mode=request.search_mode,
            hybrid_weight=request.hybrid_weight
        )
        
        # 2. Generate answer
        answer = pipeline.generate_answer(
            query=request.query,
            contexts=contexts,
            provider=request.provider,
            api_key=request.api_key,
            model_name=request.model_name
        )
        
        return {
            "query": request.query,
            "answer": answer,
            "contexts": contexts
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/api/upload")
async def handle_upload(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50)
):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    try:
        # Save temporary file
        file_id = str(uuid.uuid4())
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
        
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Parse and chunk text
        try:
            raw_text = extract_text_from_file(temp_file_path, safe_filename)
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail=str(e))
            
        chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
        
        if not chunks:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail="Document appears to be empty or contains no extractable text.")
            
        # Add to pipeline
        pipeline.add_document(
            doc_id=file_id,
            title=file.filename,
            filename=safe_filename,
            content=raw_text,
            chunks=chunks
        )
        
        return {
            "message": "Document processed and indexed successfully.",
            "doc_id": file_id,
            "filename": file.filename,
            "chunks_count": len(chunks)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    if not pipeline:
        return []
    
    docs_list = []
    for doc_id, doc in pipeline.documents.items():
        docs_list.append({
            "doc_id": doc_id,
            "title": doc["title"],
            "filename": doc["filename"],
            "chunks_count": len(doc["chunks"]),
            "char_count": len(doc["content"])
        })
    return docs_list

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    success = pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Clean file from filesystem if matches doc_id prefix
    try:
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith(doc_id):
                os.remove(os.path.join(UPLOAD_DIR, fname))
    except Exception as e:
        print(f"[WARNING] Failed to remove upload file: {e}")
        
    return {"message": "Document deleted successfully."}

@app.get("/api/settings")
async def get_settings():
    if not pipeline:
        return {"vector_search_available": False}
    return {
        "vector_search_available": pipeline.use_vector_search,
        "chunks_indexed": len(pipeline.chunks_flat),
        "documents_count": len(pipeline.documents)
    }

# ----------------- STATIC FRONTEND SERVING -----------------

@app.get("/")
async def serve_index():
    frontend_dir = os.path.join(os.path.dirname(BASE_DIR), "frontend")
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"message": "Frontend index.html not found"})

@app.get("/styles.css")
async def serve_css():
    frontend_dir = os.path.join(os.path.dirname(BASE_DIR), "frontend")
    css_path = os.path.join(frontend_dir, "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path)
    return JSONResponse(status_code=404, content={"message": "Frontend styles.css not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
