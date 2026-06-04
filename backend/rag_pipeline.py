import os
import json
import numpy as np
import requests
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Save/load file paths
if os.environ.get("VERCEL"):
    DB_DIR = "/tmp"
    DB_FILE = "/tmp/db.json"
else:
    DB_DIR = os.path.join(os.path.dirname(__file__), "data")
    DB_FILE = os.path.join(DB_DIR, "db.json")

class RAGPipeline:
    def __init__(self):
        self.documents = {}  # {doc_id: {title: ..., filename: ..., content: ..., chunks: [...]}}
        self.chunks_flat = []  # List of dict: {"doc_id": ..., "doc_title": ..., "content": ..., "index": ...}
        self.embeddings = None  # numpy array of chunk embeddings
        self.embedder = None
        self.use_vector_search = False
        
        # Load local database
        self.load_db()
        
        # Try loading local embedding model (sentence-transformers)
        self.init_embedder()
        
        # Setup scikit-learn TF-IDF vectorizer for lexical search
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.build_lexical_index()

    def init_embedder(self):
        """Attempts to load local sentence-transformers model. Falls back gracefully if failure occurs."""
        print("[INFO] Attempting to load HuggingFace embedding model (all-MiniLM-L6-v2)...")
        try:
            from sentence_transformers import SentenceTransformer
            # Set caching directory within scratch folder to avoid root permission issues
            if os.environ.get("VERCEL"):
                cache_dir = "/tmp/.cache"
            else:
                cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir
            
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
            self.use_vector_search = True
            print("[INFO] Local embedding model loaded successfully.")
            # Calculate embeddings for existing chunks if any
            self.rebuild_vector_index()
        except Exception as e:
            print(f"[WARNING] sentence-transformers failed to load: {e}. Falling back to lexical (TF-IDF) retrieval only.")
            self.use_vector_search = False

    def load_db(self):
        """Loads the document database from local JSON file."""
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                self.flatten_chunks()
                print(f"[INFO] Loaded {len(self.documents)} document(s) from database.")
            except Exception as e:
                print(f"[ERROR] Failed to load DB file: {e}")
                self.documents = {}
                self.chunks_flat = []
        else:
            os.makedirs(DB_DIR, exist_ok=True)
            self.documents = {}
            self.chunks_flat = []

    def save_db(self):
        """Saves the document database to local JSON file."""
        os.makedirs(DB_DIR, exist_ok=True)
        try:
            with open(DB_FILE, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            print("[INFO] Database saved successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to save DB file: {e}")

    def flatten_chunks(self):
        """Flattens all document chunks into a single list for indexing."""
        self.chunks_flat = []
        for doc_id, doc in self.documents.items():
            for idx, chunk in enumerate(doc.get("chunks", [])):
                self.chunks_flat.append({
                    "doc_id": doc_id,
                    "doc_title": doc.get("title", "Untitled"),
                    "content": chunk["content"],
                    "start_idx": chunk.get("start_idx", 0),
                    "end_idx": chunk.get("end_idx", 0),
                    "index": idx
                })

    def add_document(self, doc_id: str, title: str, filename: str, content: str, chunks: list):
        """Adds a document, flat-indexes its chunks, and rebuilds indices."""
        self.documents[doc_id] = {
            "title": title,
            "filename": filename,
            "content": content,
            "chunks": chunks
        }
        self.save_db()
        self.flatten_chunks()
        self.build_lexical_index()
        self.rebuild_vector_index()

    def delete_document(self, doc_id: str):
        """Deletes a document and rebuilds indices."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.save_db()
            self.flatten_chunks()
            self.build_lexical_index()
            self.rebuild_vector_index()
            return True
        return False

    def build_lexical_index(self):
        """Builds the TF-IDF index for lexical search fallback/hybrid search."""
        if not self.chunks_flat:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return
            
        try:
            corpus = [c["content"] for c in self.chunks_flat]
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            print("[INFO] Lexical TF-IDF search index built successfully.")
        except Exception as e:
            print(f"[ERROR] Lexical index build failed: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    def rebuild_vector_index(self):
        """Recomputes embeddings for all chunks using sentence-transformers."""
        if not self.use_vector_search or not self.embedder or not self.chunks_flat:
            self.embeddings = None
            return
            
        try:
            print(f"[INFO] Vectorizing {len(self.chunks_flat)} chunks...")
            corpus = [c["content"] for c in self.chunks_flat]
            self.embeddings = self.embedder.encode(corpus, convert_to_numpy=True)
            print("[INFO] Vector store embeddings generated.")
        except Exception as e:
            print(f"[ERROR] Vector index rebuild failed: {e}")
            self.embeddings = None

    def retrieve(self, query: str, top_k: int = 3, search_mode: str = "hybrid", hybrid_weight: float = 0.5) -> list:
        """Retrieves top_k relevant chunks using Vector, Lexical, or Hybrid search."""
        if not self.chunks_flat:
            return []
            
        top_k = min(top_k, len(self.chunks_flat))
        
        # 1. Lexical Similarity Score (TF-IDF)
        lexical_scores = np.zeros(len(self.chunks_flat))
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            try:
                query_vector = self.tfidf_vectorizer.transform([query])
                lexical_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            except Exception as e:
                print(f"[WARNING] Lexical search execution failed: {e}")
                
        # 2. Vector Similarity Score (Embeddings)
        vector_scores = np.zeros(len(self.chunks_flat))
        if self.use_vector_search and self.embeddings is not None:
            try:
                query_emb = self.embedder.encode([query], convert_to_numpy=True)
                vector_scores = cosine_similarity(query_emb, self.embeddings).flatten()
            except Exception as e:
                print(f"[WARNING] Vector search execution failed: {e}")
                
        # Normalize scores to 0-1 range for hybrid blending
        def normalize(scores):
            s_min, s_max = scores.min(), scores.max()
            if s_max - s_min > 0:
                return (scores - s_min) / (s_max - s_min)
            return scores
            
        norm_lexical = normalize(lexical_scores)
        norm_vector = normalize(vector_scores)
        
        # Combine scores based on search mode
        if search_mode == "vector" and self.use_vector_search:
            final_scores = vector_scores
        elif search_mode == "lexical" or not self.use_vector_search:
            final_scores = lexical_scores
        else:  # Hybrid search
            # Blended score
            final_scores = (hybrid_weight * norm_vector) + ((1 - hybrid_weight) * norm_lexical)
            
        # Get top-k indices
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(final_scores[idx])
            chunk = self.chunks_flat[idx]
            results.append({
                "doc_id": chunk["doc_id"],
                "doc_title": chunk["doc_title"],
                "content": chunk["content"],
                "start_idx": chunk["start_idx"],
                "end_idx": chunk["end_idx"],
                "chunk_index": chunk["index"],
                "score": round(score, 4),
                "vector_score": round(float(vector_scores[idx]), 4),
                "lexical_score": round(float(lexical_scores[idx]), 4)
            })
            
        return results

    def generate_answer(self, query: str, contexts: list, provider: str = "mock", api_key: str = "", model_name: str = "") -> str:
        """Generates an answer using the selected LLM provider and retrieved context."""
        if not contexts:
            return "I don't have any documents indexed. Please upload a document first."
            
        context_text = "\n\n".join([f"--- Source: {c['doc_title']} (Chunk #{c['chunk_index']}) ---\n{c['content']}" for c in contexts])
        
        system_prompt = (
            "You are a helpful, precise RAG assistant. Your goal is to answer the user's question accurately using ONLY the provided context.\n"
            "Guidelines:\n"
            "1. Base your answer strictly on the provided context. If the context does not contain the answer, say 'I don't have that information based on the uploaded documents.'\n"
            "2. Be concise, clear, and informative.\n"
            "3. Refer to specific sources/document names where appropriate.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )

        if provider == "gemini":
            if not api_key:
                return "Error: Gemini API Key is missing. Please provide it in the Settings sidebar."
            try:
                genai.configure(api_key=api_key)
                # Use default gemini-1.5-flash if model name is empty
                model_to_use = model_name if model_name else "gemini-1.5-flash"
                model = genai.GenerativeModel(model_to_use)
                response = model.generate_content(system_prompt)
                return response.text
            except Exception as e:
                return f"Error querying Gemini API: {str(e)}"
                
        elif provider == "ollama":
            # Default local port
            ollama_url = "http://localhost:11434/api/generate"
            model_to_use = model_name if model_name else "llama3"
            try:
                payload = {
                    "model": model_to_use,
                    "prompt": system_prompt,
                    "stream": False
                }
                res = requests.post(ollama_url, json=payload, timeout=20)
                if res.status_code == 200:
                    return res.json().get("response", "")
                else:
                    return f"Error from local Ollama endpoint (Status {res.status_code}): {res.text}"
            except Exception as e:
                return f"Failed to connect to local Ollama. Ensure Ollama is running and `{model_to_use}` is pulled. Error: {str(e)}"
                
        else:
            # Fallback to local / mock generator
            # Synthesizes a response strictly by assembling information from top chunks.
            # Very useful for zero-setup, offline runs!
            best_chunk = contexts[0]
            summary = (
                f"**[Offline Mode]** Based on the most relevant match in **{best_chunk['doc_title']}** (Confidence: {best_chunk['score']}):\n\n"
                f"{best_chunk['content']}\n\n"
                f"*Note: Run with Gemini API or Ollama in the settings sidebar for a fully synthesized LLM answer.*"
            )
            return summary
