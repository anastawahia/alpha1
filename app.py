# -----------------------------
# RAG pipeline with text files
# LlamaIndex (0.13.x) + FAISS + MiniLM + Ollama
# -----------------------------

from pathlib import Path
from typing import List, Optional, Callable, Dict
import json, hashlib, time
import os

# LlamaIndex - Core
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter

# LLM via Ollama
from llama_index.llms.ollama import Ollama

# Embeddings via HuggingFace (MiniLM)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# FAISS
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore


# =============================
# Paths
# =============================
BASE_DIR = Path().resolve()        # In Jupyter/VS Code: current directory
DATA_DIR = BASE_DIR / "data"
UNSTRUCTURED_DIR = DATA_DIR / "unstructured"  # Put PDF/TXT/DOCX/MD here
STORAGE_DIR = BASE_DIR / "storage"            # Storage will be saved here

# Create structured, unstructured, and storage directories
UNSTRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Use a single, consistent index id to avoid multiple indices in the same storage
INDEX_ID = "main"

# Manifest path
MANIFEST = STORAGE_DIR / "ingest_manifest.json"


# =============================
# Settings (LLM + Embeddings + Chunking) — GPU for Ollama
# =============================

# GPU parameters for Ollama:
OLLAMA_GPU_KW = {
    # Number of GPUs to use. Try 1 if you have a single GPU like RTX 4060
    "num_gpu": 1,
    # Optionally force number of layers to load on GPU (may vary per model)
    # "gpu_layer": 9999,
}

# Configure LLM settings with Ollama (GPU-enabled via additional_kwargs)
Settings.llm = Ollama(
    model="llama3.2:1b",
    request_timeout=90.0,   # Slightly reduced due to expected acceleration
    temperature=0.1,
    additional_kwargs=OLLAMA_GPU_KW,
)

# Function to set Ollama model dynamically (keeps GPU kwargs)
def set_model(name: str, request_timeout: float = 90.0, temperature: float = 0.1):
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama
    Settings.llm = Ollama(
        model=name,
        request_timeout=request_timeout,
        temperature=temperature,
        additional_kwargs=OLLAMA_GPU_KW,
    )


# Alternative dynamic set_model function
def set_model(name: str, request_timeout: float = 120.0, temperature: float = 0.1):
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama
    Settings.llm = Ollama(model=name, request_timeout=request_timeout, temperature=temperature)


# Embeddings on GPU (MiniLM) + sane batching
import torch

# Automatically choose CUDA if available, otherwise CPU
EMB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Option: increase number of CPU threads if falling back to CPU
if EMB_DEVICE == "cpu":
    try:
        torch.set_num_threads(max(1, torch.get_num_threads()))
    except Exception:
        pass

# Larger batches for speed (especially on GPU)
EMB_BATCH = 128 if EMB_DEVICE == "cuda" else 32

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=EMB_DEVICE,                
    embed_batch_size=EMB_BATCH,       
)

# Optional warm-up to avoid first-query latency
try:
    _ = Settings.embed_model.get_query_embedding("warmup for latency")
except Exception:
    pass

print(f"[Embeddings] device={EMB_DEVICE} batch={EMB_BATCH}")


# Text splitter
splitter = SentenceSplitter(chunk_size=700, chunk_overlap=70)


# =============================
# Manifest helpers (fingerprint + read/write)
# =============================

# Generate unique file fingerprint with SHA-256
def file_fingerprint(p: Path) -> str:
    """Stable fingerprint based on absolute path, size, and mtime."""
    h = hashlib.sha256()
    st = p.stat()
    h.update(str(p.resolve()).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()

# Load/Save manifest as JSON dictionary
def load_manifest() -> Dict[str, dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}

def save_manifest(m: Dict[str, dict]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================
# Unstructured loaders
# =============================
def load_unstructured_docs() -> List[Document]:
    """Load PDF/TXT/DOCX/MD files as unstructured docs.
    Robust to uppercase extensions (e.g., .PDF) by pre-building the file list.
    """
    if not UNSTRUCTURED_DIR.exists():
        return []

    allowed = {".pdf", ".txt", ".md", ".docx"}
    files = [p for p in UNSTRUCTURED_DIR.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    if not files:
        return []

    reader = SimpleDirectoryReader(
        input_files=[str(p) for p in files],
        filename_as_id=True,
    )
    docs = reader.load_data()

    out: List[Document] = []
    for d in docs:
        md = dict(d.metadata) if d.metadata else {}
        md.setdefault("type", "unstructured_doc")
        # Enrich metadata for pre-filtering
        try:
            file_path = md.get("file_path")
            src_path = Path(file_path) if file_path else None
        except Exception:
            src_path = None
        domain = None
        if src_path:
            parts = [p.lower() for p in src_path.parts]
            for cand in ("procedures","drawings","reports","specs","manuals","policies","excel","csv","unstructured"):
                if cand in parts:
                    domain = cand
                    break
        md["domain"] = domain or "unstructured"
        try:
            doc_type = (src_path.suffix.lower().lstrip(".") if src_path else (Path(md.get("source","")).suffix.lower().lstrip(".")))
        except Exception:
            doc_type = ""
        md["doc_type"] = doc_type or "txt"
        try:
            mtime = int((src_path.stat().st_mtime) if src_path else 0)
        except Exception:
            mtime = 0
        import time as _time
        md["time_bucket"] = _time.strftime("%Y-%m", _time.gmtime(mtime)) if mtime else "unknown"
        file_path = md.get("file_path")
        if file_path:
            md["source_path"] = str(file_path)
            md["source"] = Path(file_path).name
        else:
            src = md.get("source") or md.get("file_name") or md.get("filename") or md.get("id")
            if src:
                md["source"] = str(src)
        out.append(Document(text=d.text, metadata=md))
    return out


# =============================
# Build / Persist index
# =============================
def _resolve_source_to_path(md: dict) -> Optional[Path]:
    """Resolve a file path from document metadata. Prefer 'source_path'; fallback to searching by name."""
    sp = md.get("source_path")
    if sp and Path(sp).exists():
        return Path(sp)
    name = md.get("source")
    if not name:
        return None
    candidates = list(UNSTRUCTURED_DIR.rglob(name))
    return candidates[0] if candidates else None


def build_or_update_index(persist_dir: Path = STORAGE_DIR) -> VectorStoreIndex:
    """Build/Update a FAISS index from (Unstructured) incrementally using a manifest."""
    u_docs = load_unstructured_docs()
    all_docs = u_docs 
    if not all_docs:
        raise RuntimeError(
            "No documents available for indexing.\n"
            f"- Or PDF/TXT/DOCX/MD files in: {UNSTRUCTURED_DIR}\n"
        )

    print("Creating fresh storage...")

    fresh = True
    try:
        existing_index = _load_index(persist_dir)
        fresh = False
    except Exception:
        existing_index = None
        fresh = True

    if fresh:
        print("Creating fresh storage...")
        probe_vec = Settings.embed_model.get_query_embedding("dimension probe")
        embed_dim = len(probe_vec)
        print(f"Using embedding dimension: {embed_dim}")
        faiss_idx = faiss.IndexHNSWFlat(embed_dim, 32)
        faiss_idx.hnsw.efConstruction = 200
        faiss_idx.hnsw.efSearch = 64
        vs = FaissVectorStore(faiss_index=faiss_idx)
        storage_context = StorageContext.from_defaults(vector_store=vs)
        print("Storage initialized with empty FAISS-HNSW index.")
        print("Creating new index...")
        index = VectorStoreIndex.from_documents(
            all_docs,
            storage_context=storage_context,
            show_progress=True,
            transformations=[SentenceSplitter(chunk_size=700, chunk_overlap=70)],
        )
    else:
        print("Incremental update: loading existing index and inserting changed docs only...")
        index = existing_index
        manifest = load_manifest()
        changed_docs = []
        for d in all_docs:
            md = d.metadata or {}
            src = md.get("source_path") or md.get("file_path") or md.get("source") or ""
            try:
                pth = Path(src)
            except Exception:
                pth = None
            if pth and pth.exists():
                fp = file_fingerprint(pth)
                old = manifest.get(str(pth.resolve()))
                if (not old) or (old.get("fingerprint") != fp):
                    changed_docs.append(d)
        if changed_docs:
            print(f"Inserting {len(changed_docs)} changed/new docs...")
            index.insert_documents(changed_docs)
        else:
            print("No changes detected; index is up to date.")

    probe_vec = Settings.embed_model.get_query_embedding("dimension probe")
    embed_dim = len(probe_vec)
    print(f"Using embedding dimension: {embed_dim}")
    faiss_idx = faiss.IndexHNSWFlat(embed_dim, 32)
    faiss_idx.hnsw.efConstruction = 200
    faiss_idx.hnsw.efSearch = 64
    vs = FaissVectorStore(faiss_index=faiss_idx)
    storage_context = StorageContext.from_defaults(vector_store=vs)
    print("Storage initialized with empty FAISS index.")
    print("Creating new index...")

    index = VectorStoreIndex.from_documents(
        documents=all_docs,
        storage_context=storage_context,
        show_progress=True,
        use_async=False  
    )

    manifest = load_manifest()

    print(f"\nProcessing {len(all_docs)} documents...")

    for doc in all_docs:
        md = getattr(doc, "metadata", {}) or {}
        src_path = _resolve_source_to_path(md)
        if src_path and src_path.exists():
            key = str(src_path.resolve())
            manifest[key] = {
                "fingerprint": file_fingerprint(src_path),
                "last_indexed": int(time.time()),
            }

    save_manifest(manifest)
    storage_context.persist(persist_dir=str(persist_dir))
    print("Index built and persisted.")
    return index


# =============================
# Load from storage + Query
# =============================
def _load_index(persist_dir: Path = STORAGE_DIR) -> VectorStoreIndex:
    """Load index from storage using the fixed INDEX_ID."""
    if not (persist_dir.exists() and any(persist_dir.iterdir())):
        raise RuntimeError("No existing storage found. Run build_or_update_index() first.")

    vs = FaissVectorStore.from_persist_dir(str(persist_dir))
    storage_context = StorageContext.from_defaults(
        vector_store=vs,
        persist_dir=str(persist_dir),
    )
    return load_index_from_storage(storage_context)


def ensure_index(persist_dir: Path = STORAGE_DIR) -> VectorStoreIndex:
    """If no storage exists, build first; then return loaded index."""
    if not (persist_dir.exists() and any(persist_dir.iterdir())):
        build_or_update_index(persist_dir)
    return _load_index(persist_dir)


# =============================
# Query helpers
# =============================
def ask(query: str, top_k: int = 5) -> str:
    """
    Simple query function across all indexed data.
    - query: the query text
    - top_k: number of results to retrieve
    """
    idx = ensure_index(STORAGE_DIR)

    qe = idx.as_query_engine(similarity_top_k=top_k)

    try:
        resp = qe.query(query)
        return str(resp)
    except Exception as e:
        return f"Error during query: {str(e)}"


# =============================
# FAISS storage verification and index coverage check  
# =============================
def _faiss_ntotal_from_storage_dir(storage_dir: Path) -> Optional[int]:
    """Retrieve FAISS ntotal from storage (best-effort)."""
    try:
        vs = FaissVectorStore.from_persist_dir(str(storage_dir))
        for attr in ("faiss_index", "index", "_faiss_index"):
            obj = getattr(vs, attr, None)
            if obj is not None and hasattr(obj, "ntotal"):
                return int(obj.ntotal)
    except Exception:
        pass

    def _walk(o):
        if isinstance(o, dict):
            for v in o.values():
                p = _walk(v)
                if p:
                    return p
        elif isinstance(o, list):
            for v in o:
                p = _walk(v)
                if p:
                    return p
        elif isinstance(o, str):
            if o.lower().endswith((".index", ".faiss")):
                return o
        return None

    try:
        import json as _json
        candidates = list(storage_dir.glob("*vector_store*.json"))
        for js in candidates:
            data = _json.loads(js.read_text(encoding="utf-8"))
            rel = _walk(data)
            if rel:
                p = Path(rel)
                idx_path = p if p.is_absolute() else (storage_dir / rel)
                if idx_path.exists():
                    ix = faiss.read_index(str(idx_path))
                    return int(ix.ntotal)
    except Exception:
        pass
    return None


def check_storage() -> None:
    """Print a quick summary of FAISS vectors inside the persisted storage."""
    if not STORAGE_DIR.exists():
        print("WARNING: Storage folder does not exist.")
        return

    ntotal = _faiss_ntotal_from_storage_dir(STORAGE_DIR)
    if ntotal is None:
        print("WARNING: Could not read FAISS ntotal (index may be empty or attribute not exposed).")
    else:
        print("OK: Storage folder found.")
        print(f"Total vectors in FAISS index: {ntotal}")


def _iter_all_data_files() -> List[Path]:
    exts = {".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx", ".xls"}
    files = []
    if UNSTRUCTURED_DIR.exists():
        files.extend([p for p in UNSTRUCTURED_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return files


def verify_coverage(verbose: bool = True) -> dict:
    """Return a report showing missing/stale files against the manifest and FAISS vector count."""
    report = {
        "total_files_in_data": 0,
        "indexed_files": 0,
        "missing_in_storage": [],
        "stale_in_storage": [],
        "faiss_vectors": None,
    }

    files = _iter_all_data_files()
    report["total_files_in_data"] = len(files)

    manifest = load_manifest()
    ntotal = _faiss_ntotal_from_storage_dir(STORAGE_DIR)
    if ntotal is not None:
        report["faiss_vectors"] = ntotal

    indexed = 0
    for f in files:
        key = str(f.resolve())
        fp_now = file_fingerprint(f)
        entry = manifest.get(key)
        if not entry:
            report["missing_in_storage"].append(key)
        else:
            indexed += 1
            if entry.get("fingerprint") != fp_now:
                report["stale_in_storage"].append(key)

    report["indexed_files"] = indexed

    if verbose:
        print(f"Files in data: {report['total_files_in_data']}")
        print(f"Files recorded in storage (manifest hits): {report['indexed_files']}")
        print(f"FAISS vectors: {report['faiss_vectors']}")
        if report["missing_in_storage"]:
            print("\nMissing (not indexed yet):")
            for x in report["missing_in_storage"]:
                print(" -", x)
        if report["stale_in_storage"]:
            print("\nStale (changed on disk; needs re-index):")
            for x in report["stale_in_storage"]:
                print(" -", x)

    return report


# =============================
# Main (for direct run)
# =============================
if __name__ == "__main__":
    build_or_update_index(STORAGE_DIR)
    check_storage()
    verify_coverage(verbose=True)
    
    print("\nExample query:")
    print(ask("tell me about the boiler operation procedures"))
