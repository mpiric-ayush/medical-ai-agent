import os
import json
import logging
from typing import Any, Dict, List, Optional

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class MedicalGraphEngine:
    """
    Wraps Neo4j graph database with LangChain GraphRAG.

    Hetionet Node Types:
      - Disease    : DB/DOID identifiers
      - Gene       : Entrez Gene IDs
      - Compound   : DrugBank IDs
      - Anatomy    : Uberon IDs
      - Symptom    : MeSH IDs
      - SideEffect : UMLS IDs
      - Pathway    : Reactome IDs

    Hetionet Edge Types:
      - treats, resembles, associates, binds,
      - causes, expresses, regulates, upregulates, downregulates,
      - localizes, palliates, participates_in
    """

    def __init__(self):
        self.driver = None
        self.graph_qa = None
        self._connect()

    def _connect(self):
        """Connect to Neo4j and initialize LangChain GraphCypherQAChain."""
        try:
            from langchain_community.graphs import Neo4jGraph
            from langchain.chains import GraphCypherQAChain
            from langchain_openai import ChatOpenAI

            self.graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD
            )

            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=OPENAI_API_KEY
            )

            self.graph_qa = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=self.graph,
                verbose=True,
                return_intermediate_steps=True,
                top_k=10
            )
            logger.info("✅ Connected to Neo4j GraphRAG")
        except Exception as e:
            logger.warning(f"Neo4j connection failed (demo mode): {e}")
            self.graph_qa = None

    def query(self, question: str) -> Dict[str, Any]:
        """
        Natural language query against the medical knowledge graph.
        Falls back to mock data if Neo4j unavailable.
        """
        if self.graph_qa:
            try:
                result = self.graph_qa.invoke({"query": question})
                return {
                    "answer": result.get("result", ""),
                    "cypher": result.get("intermediate_steps", [{}])[0].get("query", ""),
                    "data": result.get("intermediate_steps", [{}])[-1].get("context", [])
                }
            except Exception as e:
                logger.error(f"Graph query error: {e}")

        # Demo fallback
        return self._demo_query(question)

    def _demo_query(self, question: str) -> Dict[str, Any]:
        """Return demo graph data for testing without Neo4j."""
        return {
            "answer": f"[Demo Mode] Graph query for: {question}",
            "cypher": "MATCH (d:Disease)-[:TREATS]-(c:Compound) WHERE d.name CONTAINS $term RETURN d, c LIMIT 10",
            "data": [
                {"disease": "Type 2 Diabetes", "compound": "Metformin", "relation": "treats"},
                {"disease": "Hypertension", "compound": "Amlodipine", "relation": "treats"},
                {"gene": "INS", "disease": "Type 2 Diabetes", "relation": "associates"},
            ]
        }

    def get_disease_relations(self, disease_name: str) -> Dict[str, Any]:
        """Get all relationships for a specific disease."""
        q = f"""
        Find all compounds that treat {disease_name},
        all genes associated with {disease_name},
        and all symptoms of {disease_name}.
        """
        return self.query(q)

    def get_drug_interactions(self, drug_list: List[str]) -> Dict[str, Any]:
        """Check interactions between a list of drugs."""
        drugs_str = ", ".join(drug_list)
        q = f"""
        Check if there are any known interactions or contraindications
        between the following drugs: {drugs_str}.
        Include mechanism and severity.
        """
        return self.query(q)

    def get_risk_propagation(self, condition: str) -> List[Dict]:
        """Get disease progression/risk paths from knowledge graph."""
        q = f"""
        What diseases or conditions can {condition} progress to or cause?
        Return the risk pathway with probabilities if available.
        """
        result = self.query(q)
        return result.get("data", [])


def get_graph_engine() -> MedicalGraphEngine:
    """Singleton factory for the graph engine."""
    return MedicalGraphEngine()

def build_hetionet_graph(data_dir: str = "./data/hetionet"):
    """
    Load Hetionet nodes.tsv and edges.tsv into Neo4j.

    Download from: https://het.io/data/

    Files needed:
      - hetionet-v1.0-nodes.tsv
      - hetionet-v1.0-edges.tsv

    Usage:
        python graphrag_index.py
    """
    from pathlib import Path
    import csv

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        logger.error(f"Neo4j driver failed: {e}. Install: pip install neo4j")
        return

    nodes_file = Path(data_dir) / "hetionet-v1.0-nodes.tsv"
    edges_file = Path(data_dir) / "hetionet-v1.0-edges.tsv"

    if not nodes_file.exists():
        print(f"❌ Nodes file not found: {nodes_file}")
        print("Download from: https://github.com/hetio/hetionet/blob/main/hetnet/tsv/")
        return

    with driver.session() as session:
        # ── Load Nodes ──────────────────────
        print("Loading nodes...")
        with open(nodes_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            batch = []
            count = 0
            for row in reader:
                batch.append({"id": row["id"], "name": row["name"], "kind": row["kind"]})
                if len(batch) >= 500:
                    session.run("""
                    UNWIND $nodes AS n
                    CALL apoc.merge.node([n.kind], {id: n.id}, {name: n.name}) YIELD node
                    RETURN count(node)
                    """, nodes=batch)
                    count += len(batch)
                    batch = []
                    print(f"  ✓ {count} nodes loaded", end="\r")
            if batch:
                session.run("""
                UNWIND $nodes AS n
                MERGE (node {id: n.id, kind: n.kind})
                SET node.name = n.name
                """, nodes=batch)
            print(f"\n✅ Total nodes: {count + len(batch)}")

        # ── Load Edges ──────────────────────
        if edges_file.exists():
            print("Loading edges...")
            with open(edges_file, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                batch = []
                count = 0
                for row in reader:
                    batch.append({
                        "source": row["source"],
                        "target": row["target"],
                        "metaedge": row["metaedge"]
                    })
                    if len(batch) >= 500:
                        session.run("""
                        UNWIND $edges AS e
                        MATCH (s {id: e.source}), (t {id: e.target})
                        CALL apoc.merge.relationship(s, e.metaedge, {}, {}, t, {}) YIELD rel
                        RETURN count(rel)
                        """, edges=batch)
                        count += len(batch)
                        batch = []
                        print(f"  ✓ {count} edges loaded", end="\r")
            print(f"\n✅ Total edges: {count + len(batch)}")

    driver.close()
    print("\n🎉 Hetionet graph loaded into Neo4j!")


def build_pubmedkg_graph(tsv_path: str = "./data/pubmedkg/pubmedkg_subset.tsv"):
    """
    Load PubMed KG triples into Neo4j.

    Download from: https://pubmedkg.github.io

    TSV format: subject | predicate | object | pmid
    """
    from pathlib import Path
    import csv

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return

    tsv_file = Path(tsv_path)
    if not tsv_file.exists():
        print(f"❌ PubMed KG file not found: {tsv_file}")
        print("Download from: https://pubmedkg.github.io")
        return

    with driver.session() as session:
        with open(tsv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            batch = []
            count = 0
            for row in reader:
                batch.append({
                    "subject": row.get("subject", ""),
                    "predicate": row.get("predicate", "RELATES_TO"),
                    "object": row.get("object", ""),
                    "pmid": row.get("pmid", "")
                })
                if len(batch) >= 500:
                    session.run("""
                    UNWIND $triples AS t
                    MERGE (s:Entity {name: t.subject})
                    MERGE (o:Entity {name: t.object})
                    MERGE (s)-[:RELATES_TO {predicate: t.predicate, pmid: t.pmid}]->(o)
                    """, triples=batch)
                    count += len(batch)
                    batch = []
                    print(f"  ✓ {count} triples", end="\r")
            print(f"\n✅ PubMed KG loaded: {count} triples")

    driver.close()


if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build Medical Knowledge Graph in Neo4j")
    parser.add_argument("--source", choices=["hetionet", "pubmedkg", "all"], default="all")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    if args.source in ("hetionet", "all"):
        build_hetionet_graph(f"{args.data_dir}/hetionet")
    if args.source in ("pubmedkg", "all"):
        build_pubmedkg_graph(f"{args.data_dir}/pubmedkg/pubmedkg_subset.tsv")

    print("\n✅ Knowledge graph construction complete!")
