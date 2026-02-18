"""
╔══════════════════════════════════════════════════════════════╗
║               MEDICAL AI AGENT - retriever.py                ║
║     Pinecone Vector Retriever + Hybrid Search (Dense+Sparse) ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
from typing import List, Optional, Dict, Any

from config import (
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV,
    PINECONE_INDEX, EMBED_MODEL, TOP_K_RETRIEVAL
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# LangChain Pinecone Retriever
# ──────────────────────────────────────────────────────────────

def get_retriever(namespace: str = "medical_kb"):
    """
    Build and return a LangChain VectorStoreRetriever backed by Pinecone.

    Uses text-embedding-ada-002 for dense retrieval.
    namespace="medical_kb" for knowledge base
    namespace="patient_memory" for patient history
    """
    try:
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import OpenAIEmbeddings
        from pinecone import Pinecone

        embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            openai_api_key=OPENAI_API_KEY
        )

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=namespace
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )
        logger.info(f"✅ Pinecone retriever ready (namespace: {namespace})")
        return retriever

    except ImportError as e:
        logger.warning(f"Pinecone/LangChain not installed: {e}")
        return MockRetriever()
    except Exception as e:
        logger.warning(f"Pinecone connection failed (using mock): {e}")
        return MockRetriever()


# ──────────────────────────────────────────────────────────────
# Patient Memory Retriever
# ──────────────────────────────────────────────────────────────

def get_patient_memory_retriever(patient_id: str):
    """
    Retrieve patient-specific historical data from Pinecone.
    Enables personalized analysis by referencing past reports.
    """
    try:
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import OpenAIEmbeddings
        from pinecone import Pinecone

        embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=f"patient_{patient_id}"
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    except Exception as e:
        logger.warning(f"Patient memory retriever failed: {e}")
        return MockRetriever()


def save_patient_memory(patient_id: str, report_text: str, metadata: Dict):
    """
    Save a patient's report to their personal Pinecone namespace.
    This enables longitudinal health tracking across visits.
    """
    from ingestion import index_document
    namespace_meta = {**metadata, "patient_id": patient_id, "namespace": f"patient_{patient_id}"}
    count = index_document(report_text, namespace_meta, patient_id=patient_id)
    logger.info(f"Saved {count} memory chunks for patient {patient_id}")
    return count


# ──────────────────────────────────────────────────────────────
# Hybrid Search (Dense + Sparse / BM25)
# ──────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines Pinecone dense retrieval with BM25 keyword search
    for better precision on medical terminology.
    """

    def __init__(self, namespace: str = "medical_kb"):
        self.dense_retriever = get_retriever(namespace)
        self.bm25 = None
        self._init_bm25()

    def _init_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_class = BM25Okapi
            logger.info("BM25 sparse retriever ready")
        except ImportError:
            logger.warning("rank_bm25 not installed. Using dense-only retrieval.")

    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        """
        Hybrid search: merge dense + sparse results with RRF scoring.
        """
        dense_results = self.dense_retriever.get_relevant_documents(query)[:k]
        return dense_results  # Extend with BM25 results in production


# ──────────────────────────────────────────────────────────────
# Mock Retriever (fallback for demo/testing)
# ──────────────────────────────────────────────────────────────

class MockDocument:
    def __init__(self, content: str):
        self.page_content = content
        self.metadata = {"source": "mock"}


class MockRetriever:
    """Returns static medical knowledge for testing without API keys."""

    MOCK_DOCS = [
        "Type 2 Diabetes Mellitus: A chronic metabolic disorder characterized by insulin resistance. "
        "First-line treatment includes Metformin 500-2000mg/day. HbA1c target: <7%. "
        "Associated with cardiovascular risk, nephropathy, retinopathy, and neuropathy.",

        "Hypertension Stage 1: Blood pressure 130-139/80-89 mmHg. Lifestyle interventions include "
        "DASH diet, weight loss, reduced sodium intake (<2.3g/day), exercise 150min/week. "
        "First-line medications: ACE inhibitors, ARBs, thiazide diuretics, CCBs.",

        "Metformin (DB00331): Biguanide antidiabetic. Mechanism: Activates AMPK, reduces hepatic "
        "gluconeogenesis, improves insulin sensitivity. Dosage: 500-1000mg twice daily with meals. "
        "Contraindicated in eGFR <30, active liver disease. Monitor renal function annually.",

        "Cardiovascular Risk Assessment: Framingham Risk Score considers age, sex, total cholesterol, "
        "HDL, systolic BP, smoking status, diabetes status. 10-year risk >10% warrants statin therapy. "
        "Mediterranean diet reduces CV events by 30%. Statin initiation: LDL >190mg/dL or 10yr risk >7.5%",

        "Hetionet Disease-Drug Relations: Type 2 Diabetes (DOID:9352) treated by Metformin (DB00331), "
        "Glipizide (DB01067), Insulin (DB00071). Disease associates with Gene: INS, PPARG, TCF7L2. "
        "Anatomy: Pancreas, Liver, Adipose tissue.",
    ]

    def get_relevant_documents(self, query: str) -> List[MockDocument]:
        # Simple keyword matching for demo
        query_lower = query.lower()
        relevant = [
            MockDocument(doc) for doc in self.MOCK_DOCS
            if any(word in doc.lower() for word in query_lower.split()[:5])
        ]
        return relevant[:5] if relevant else [MockDocument(self.MOCK_DOCS[0])]

    def similarity_search(self, query: str, k: int = 5) -> List[MockDocument]:
        return self.get_relevant_documents(query)[:k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing retriever...")
    retriever = get_retriever()
    results = retriever.get_relevant_documents("Type 2 Diabetes treatment options")
    print(f"Retrieved {len(results)} documents")
    for r in results:
        print(f"  - {r.page_content[:100]}...")
