"""
╔══════════════════════════════════════════════════════════════╗
║                    MEDICAL AI AGENT - agents.py              ║
║          4-Agent CrewAI Pipeline: Diagnosis, Prognosis,      ║
║          Lifestyle, and Medication Agents                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
from typing import Any, Dict, List, Optional

from config import (
    OPENAI_API_KEY, LLM_MODEL, AGENT_TEMPERATURE,
    MAX_TOKENS, VERBOSE_AGENTS
)


# ──────────────────────────────────────────────────────────────
# Tool Definitions (used by all agents)
# ──────────────────────────────────────────────────────────────

def build_tools(retriever, graph_engine):
    """Build LangChain tools wrapping retriever + graph engine."""
    from langchain.tools import Tool

    def vector_search(query: str) -> str:
        """Search Pinecone vector store for similar medical knowledge."""
        try:
            results = retriever.similarity_search(query, k=5)
            return "\n---\n".join([r.page_content for r in results])
        except Exception as e:
            return f"Vector search error: {e}"

    def graph_query(query: str) -> str:
        """Query the Neo4j medical knowledge graph via GraphRAG."""
        try:
            result = graph_engine.query(query)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Graph query error: {e}"

    def pubmed_search(query: str) -> str:
        """Search PubMed for recent research on the topic."""
        try:
            # In production: use Entrez API / Biopython
            # For now returns stub - replace with real Entrez call
            from Bio import Entrez
            Entrez.email = os.getenv("ENTREZ_EMAIL", "test@test.com")
            handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
            record = Entrez.read(handle)
            return f"PubMed IDs found: {record['IdList']}"
        except Exception as e:
            return f"PubMed search unavailable (install biopython): {e}"

    def drugbank_lookup(drug_name: str) -> str:
        """Look up drug information from DrugBank index."""
        try:
            results = retriever.similarity_search(f"drug:{drug_name} mechanism dosage side effects", k=3)
            return "\n".join([r.page_content for r in results])
        except Exception as e:
            return f"Drug lookup error: {e}"

    return [
        Tool(name="medical_vector_search",    func=vector_search,   description="Search medical knowledge base for conditions, treatments, guidelines."),
        Tool(name="medical_graph_query",       func=graph_query,     description="Query Neo4j knowledge graph for disease-drug-gene-symptom relationships."),
        Tool(name="pubmed_research_search",    func=pubmed_search,   description="Search PubMed for the latest peer-reviewed research."),
        Tool(name="drugbank_medication_lookup",func=drugbank_lookup, description="Look up medication details, interactions, dosing from DrugBank."),
    ]


# ──────────────────────────────────────────────────────────────
# Agent Builder
# ──────────────────────────────────────────────────────────────

def build_agents(tools: List[Any]):
    """Instantiate all four CrewAI agents."""
    from crewai import Agent

    llm_kwargs = {"model": LLM_MODEL, "temperature": AGENT_TEMPERATURE}

    diagnosis_agent = Agent(
        role="Senior Medical Diagnostician",
        goal=(
            "Analyze the patient's medical report to identify all present or suspected "
            "conditions. Cross-reference findings with PubMed KG and Hetionet graph data. "
            "Provide evidence-based diagnoses with confidence scores and source citations."
        ),
        backstory=(
            "You are a board-certified physician with 20 years of clinical experience and "
            "deep expertise in interpreting lab results, imaging reports, and medical histories. "
            "You combine clinical intuition with rigorous knowledge graph analysis."
        ),
        tools=tools,
        verbose=VERBOSE_AGENTS,
        llm=llm_kwargs,
        max_iter=5,
    )

    prognosis_agent = Agent(
        role="Predictive Health Strategist",
        goal=(
            "Based on the diagnosed conditions, predict future health risks over the next "
            "3 months, 1 year, and 5 years. Use epidemiological data from MIMIC-III and "
            "Hetionet to assign probability scores to each risk."
        ),
        backstory=(
            "You are a specialist in predictive medicine and health outcomes research. "
            "You use statistical models, graph-based risk propagation, and clinical trial data "
            "to create accurate, actionable prognosis reports."
        ),
        tools=tools,
        verbose=VERBOSE_AGENTS,
        llm=llm_kwargs,
        max_iter=5,
    )

    lifestyle_agent = Agent(
        role="Lifestyle & Nutrition Specialist",
        goal=(
            "Create a comprehensive, personalized diet and exercise plan tailored to the "
            "patient's conditions. Include specific food recommendations, meal plans, "
            "exercise regimens, and sleep/stress management guidance backed by clinical evidence."
        ),
        backstory=(
            "You are a certified nutritionist and lifestyle medicine physician with expertise "
            "in therapeutic nutrition, sports medicine, and behavior change. You design "
            "evidence-based programs that patients can realistically follow."
        ),
        tools=tools,
        verbose=VERBOSE_AGENTS,
        llm=llm_kwargs,
        max_iter=4,
    )

    medication_agent = Agent(
        role="Clinical Pharmacologist",
        goal=(
            "Review the patient's current medications and identified conditions to recommend "
            "optimal medication protocols. Check for drug-drug interactions, contraindications, "
            "and provide dosing recommendations referenced to DrugBank and SIDER."
        ),
        backstory=(
            "You are a clinical pharmacologist with expertise in polypharmacy management, "
            "pharmacogenomics, and adverse drug event prevention. You ensure safe, effective "
            "medication regimens tailored to each patient's profile."
        ),
        tools=tools,
        verbose=VERBOSE_AGENTS,
        llm=llm_kwargs,
        max_iter=4,
    )

    return diagnosis_agent, prognosis_agent, lifestyle_agent, medication_agent


# ──────────────────────────────────────────────────────────────
# Task Builder
# ──────────────────────────────────────────────────────────────

def build_tasks(report_text: str, patient_profile: dict,
                diagnosis_agent, prognosis_agent, lifestyle_agent, medication_agent):
    """Build CrewAI tasks with structured output schemas."""
    from crewai import Task

    profile_str = json.dumps(patient_profile, indent=2)

    diagnosis_task = Task(
        description=f"""
Analyze the following medical report text and patient profile.
Extract and diagnose all identified medical conditions.

PATIENT PROFILE:
{profile_str}

MEDICAL REPORT TEXT:
{report_text}

Use the medical_vector_search and medical_graph_query tools to cross-reference findings.
Return a JSON object with this structure:
{{
  "summary": "2-3 paragraph clinical summary",
  "conditions": [
    {{
      "name": "Condition name",
      "icd10_code": "ICD-10 code",
      "severity": "low|medium|high",
      "confidence": 0-100,
      "description": "clinical description",
      "evidence": "lab values or findings supporting this",
      "source": "data source used"
    }}
  ],
  "risk_score": "X.X/10",
  "confidence": "XX.X%"
}}
""",
        agent=diagnosis_agent,
        expected_output="JSON with clinical summary and diagnosed conditions list"
    )

    prognosis_task = Task(
        description=f"""
Based on the diagnosed conditions from the previous analysis and patient profile below,
predict future health risks with probability scores.

PATIENT PROFILE:
{profile_str}

Query the knowledge graph for disease progression pathways and epidemiological risk data.
Return a JSON array of risks:
{{
  "risks": [
    {{
      "name": "Risk condition name",
      "probability": 0-100,
      "timeframe": "3-months|1-year|5-years",
      "description": "why this risk exists",
      "prevention": "preventive measures"
    }}
  ]
}}
""",
        agent=prognosis_agent,
        expected_output="JSON with health risk predictions and probability scores"
    )

    lifestyle_task = Task(
        description=f"""
Create a comprehensive, evidence-based diet and exercise plan for the patient
given their conditions and profile.

PATIENT PROFILE:
{profile_str}

Use pubmed_research_search to find the latest dietary and exercise guidelines.
Return a JSON object:
{{
  "diet": {{
    "recommended": ["food1", "food2"],
    "avoid": ["food1", "food2"],
    "meal_plan": "sample 3-day meal plan text",
    "macros": {{"protein": "Xg", "carbs": "Xg", "fat": "Xg", "calories": "XXXX kcal"}},
    "supplements": ["supplement1", "supplement2"]
  }},
  "exercises": [
    {{
      "icon": "emoji",
      "name": "Exercise name",
      "intensity": "low|moderate|high",
      "duration": "XX min",
      "frequency": "Nx/week",
      "description": "why this exercise benefits the patient"
    }}
  ],
  "sleep_recommendation": "7-8 hours, consistent schedule...",
  "stress_management": "Mindfulness, breathing exercises..."
}}
""",
        agent=lifestyle_agent,
        expected_output="JSON with diet plan, exercise regimen, and lifestyle recommendations"
    )

    medication_task = Task(
        description=f"""
Review the diagnosed conditions and create medication recommendations.
Check for interactions using drugbank_medication_lookup.

PATIENT PROFILE:
{profile_str}
CURRENT MEDICATIONS: {patient_profile.get('current_meds', 'None listed')}
ALLERGIES: {patient_profile.get('allergies', 'None listed')}

Return a JSON object:
{{
  "medications": [
    {{
      "name": "Drug name",
      "generic_name": "generic",
      "dosage": "XXmg frequency",
      "purpose": "why prescribed",
      "mechanism": "how it works",
      "side_effects": "common side effects",
      "interactions": "interactions to watch",
      "monitoring": "lab tests needed",
      "source": "DrugBank ID"
    }}
  ],
  "interactions_warning": ["any critical drug-drug interactions"],
  "stop_medications": ["any current meds that should be reviewed"]
}}
""",
        agent=medication_agent,
        expected_output="JSON with medication recommendations and interaction warnings"
    )

    return diagnosis_task, prognosis_task, lifestyle_task, medication_task


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

def run_medical_analysis(
    uploaded_file,
    use_graphrag: bool = True,
    use_pinecone: bool = True,
    patient_profile: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Main pipeline:
    1. Parse uploaded file → extract text
    2. Embed into Pinecone (if enabled)
    3. Build agents with tools
    4. Run CrewAI crew
    5. Return structured results dict
    """
    from crewai import Crew, Process
    from ingestion import parse_document, index_document
    from retriever import get_retriever
    from graphrag_index import get_graph_engine

    if patient_profile is None:
        patient_profile = {}

    # Step 1: Parse document
    raw_text, metadata = parse_document(uploaded_file)

    # Step 2: Index into Pinecone
    if use_pinecone:
        index_document(raw_text, metadata)

    # Step 3: Initialize retriever + graph
    retriever = get_retriever() if use_pinecone else None
    graph_engine = get_graph_engine() if use_graphrag else None

    # Step 4: Build tools + agents
    tools = build_tools(retriever, graph_engine)
    d_agent, p_agent, l_agent, m_agent = build_agents(tools)

    # Step 5: Build tasks
    d_task, p_task, l_task, m_task = build_tasks(
        raw_text, patient_profile, d_agent, p_agent, l_agent, m_agent
    )

    # Step 6: Run crew
    crew = Crew(
        agents=[d_agent, p_agent, l_agent, m_agent],
        tasks=[d_task, p_task, l_task, m_task],
        process=Process.sequential,
        verbose=VERBOSE_AGENTS,
    )

    crew_output = crew.kickoff()

    # Step 7: Parse and merge results
    results = {}
    try:
        results.update(json.loads(d_task.output.raw_output))
    except Exception:
        results["summary"] = str(d_task.output.raw_output)[:2000]
    try:
        results.update(json.loads(p_task.output.raw_output))
    except Exception:
        pass
    try:
        results.update(json.loads(l_task.output.raw_output))
    except Exception:
        pass
    try:
        results.update(json.loads(m_task.output.raw_output))
    except Exception:
        pass

    results["raw_text"] = raw_text[:3000]
    results["pages"] = metadata.get("pages", "N/A")
    return results
