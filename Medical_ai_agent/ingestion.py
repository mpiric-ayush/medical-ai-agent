"""
╔══════════════════════════════════════════════════════════════╗
║                MEDICAL AI AGENT - ingestion.py               ║
║    Document Parsing (PDF/Image OCR) + Pinecone Embedding     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import io
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List

from config import (
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV,
    PINECONE_INDEX, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# PDF / Image Parsing
# ──────────────────────────────────────────────────────────────

def parse_document(uploaded_file) -> Tuple[str, Dict[str, Any]]:
    """
    Parse uploaded file (PDF or image) to raw text.

    Strategy:
    - PDF  → pdfplumber (text layer) OR pytesseract (scanned pages)
    - Image → pytesseract OCR
    - Falls back gracefully if dependencies missing

    Returns: (raw_text, metadata_dict)
    """
    fname = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    metadata = {
        "filename": uploaded_file.name,
        "file_type": uploaded_file.type,
        "file_size_kb": len(file_bytes) // 1024,
        "pages": 1,
        "hash": hashlib.sha256(file_bytes).hexdigest()[:16]
    }

    # ── PDF Parsing ──────────────────────────
    if fname.endswith(".pdf"):
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                metadata["pages"] = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    if len(page_text.strip()) < 50:
                        # Scanned page - fall back to OCR
                        page_text = _ocr_pdf_page(page, page_num)
                    text_parts.append(f"[PAGE {page_num}]\n{page_text}")
            raw_text = "\n\n".join(text_parts)
            logger.info(f"PDF parsed: {metadata['pages']} pages, {len(raw_text)} chars")
            return raw_text, metadata
        except ImportError:
            logger.warning("pdfplumber not installed. Try: pip install pdfplumber")
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                metadata["pages"] = len(reader.pages)
                raw_text = "\n".join(p.extract_text() or "" for p in reader.pages)
                return raw_text, metadata
            except ImportError:
                pass
        return "[PDF parsing failed - install pdfplumber]", metadata

    # ── Image OCR ────────────────────────────
    elif any(fname.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]):
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(io.BytesIO(file_bytes))
            # Preprocess: convert to grayscale for better OCR
            if image.mode != "L":
                image = image.convert("L")
            # Use tesseract with medical document config
            custom_config = "--psm 3 --oem 3 -l eng"
            raw_text = pytesseract.image_to_string(image, config=custom_config)
            metadata["ocr_engine"] = "tesseract"
            logger.info(f"Image OCR complete: {len(raw_text)} chars")
            return raw_text, metadata
        except ImportError:
            logger.error("pytesseract not installed. Run: pip install pytesseract Pillow")
            return "[Image OCR failed - install pytesseract]", metadata
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return f"[OCR Error: {e}]", metadata

    # ── Unknown Format ────────────────────────
    else:
        try:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            return raw_text, metadata
        except Exception:
            return "[Unsupported file format]", metadata


def _ocr_pdf_page(page, page_num: int) -> str:
    """OCR a single pdfplumber page using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        img = page.to_image(resolution=200).original
        return pytesseract.image_to_string(img, config="--psm 6 -l eng")
    except Exception as e:
        return f"[OCR failed for page {page_num}: {e}]"


# ──────────────────────────────────────────────────────────────
# Text Chunking
# ──────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    Uses sentence-aware splitting to preserve medical context.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len
        )
        return splitter.split_text(text)
    except ImportError:
        # Simple fallback splitter
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks


# ──────────────────────────────────────────────────────────────
# Embedding + Pinecone Upsert
# ──────────────────────────────────────────────────────────────

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed a list of text strings using OpenAI ada-002."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Batch in groups of 100 (OpenAI limit)
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([r.embedding for r in response.data])
    return all_embeddings


def get_pinecone_index():
    """Initialize and return the Pinecone index."""
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,  # text-embedding-ada-002 dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        logger.info(f"Created Pinecone index: {PINECONE_INDEX}")

    return pc.Index(PINECONE_INDEX)


def index_document(raw_text: str, metadata: Dict, patient_id: str = "anonymous") -> int:
    """
    Chunk, embed, and upsert document into Pinecone.
    Returns the number of vectors upserted.
    """
    try:
        chunks = chunk_text(raw_text)
        if not chunks:
            return 0

        embeddings = get_embeddings(chunks)
        index = get_pinecone_index()

        doc_hash = metadata.get("hash", "unknown")
        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{doc_hash}_{i}",
                "values": emb,
                "metadata": {
                    "text": chunk[:500],     # Pinecone metadata limit
                    "patient_id": patient_id,
                    "filename": metadata.get("filename", "unknown"),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[i:i + batch_size])

        logger.info(f"Indexed {len(vectors)} vectors into Pinecone")
        return len(vectors)

    except Exception as e:
        logger.error(f"Pinecone indexing failed: {e}")
        return 0


# ──────────────────────────────────────────────────────────────
# Bulk Data Ingestion (for medical knowledge bases)
# ──────────────────────────────────────────────────────────────

def ingest_medical_knowledge_base(data_dir: str = "./data"):
    """
    Bulk ingest all medical knowledge base files.
    Run once to populate Pinecone with PubMed, DrugBank, Hetionet data.

    Usage:
        python -c "from ingestion import ingest_medical_knowledge_base; ingest_medical_knowledge_base()"
    """
    from pathlib import Path
    data_path = Path(data_dir)
    total_vectors = 0

    supported = [".txt", ".csv", ".pdf", ".json", ".tsv"]

    for file_path in data_path.rglob("*"):
        if file_path.suffix.lower() not in supported:
            continue

        logger.info(f"Ingesting: {file_path.name}")
        try:
            if file_path.suffix == ".csv" or file_path.suffix == ".tsv":
                text = _ingest_csv(file_path)
            elif file_path.suffix == ".json":
                text = _ingest_json(file_path)
            else:
                text = file_path.read_text(encoding="utf-8", errors="ignore")

            metadata = {
                "filename": file_path.name,
                "source": file_path.parent.name,
                "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            }
            count = index_document(text, metadata, patient_id="knowledge_base")
            total_vectors += count
            logger.info(f"  ✓ {count} vectors from {file_path.name}")
        except Exception as e:
            logger.error(f"  ✗ Failed {file_path.name}: {e}")

    logger.info(f"\n✅ Total vectors ingested: {total_vectors}")
    return total_vectors


def _ingest_csv(file_path: Path) -> str:
    """Convert CSV/TSV rows to text for embedding."""
    import csv
    sep = "\t" if file_path.suffix == ".tsv" else ","
    rows = []
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=sep)
        for row in reader:
            rows.append(" | ".join(f"{k}: {v}" for k, v in row.items() if v))
    return "\n".join(rows)


def _ingest_json(file_path: Path) -> str:
    """Flatten JSON to text for embedding."""
    import json
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return "\n".join(json.dumps(item) for item in data)
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    print(f"Starting bulk ingestion from: {data_dir}")
    count = ingest_medical_knowledge_base(data_dir)
    print(f"Done! {count} vectors indexed.")
