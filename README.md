🏥 Medical AI Agent — Complete Build Guide
Senior ML Engineer's Full Walkthrough
Streamlit · GraphRAG · OpenAI Agents · Pinecone · Neo4j

🎯 What This System Does
Upload any medical report (PDF or image) and the system:

Parses it via OCR/PDF extraction
Embeds it into Pinecone vector store
Queries the Neo4j medical knowledge graph (Hetionet + PubMed KG)
Runs 4 specialized AI agents in sequence
Outputs across 6 structured tabs:
📋 Clinical Summary
🔬 Diagnosed Conditions (with ICD-10 codes + confidence scores)
📈 Future Risk Predictions (3mo / 1yr / 5yr)
🥗 Personalized Diet Plan
💊 Medication Recommendations (DrugBank-sourced)
🏃 Exercise & Lifestyle Plan
🏗️ Complete Architecture
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)                     │
│  Upload → Patient Profile → Config → Results Tabs → Export  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              INGESTION LAYER (ingestion.py)                  │
│  PDF: pdfplumber → text  |  Image: pytesseract → OCR        │
│  → chunk_text() → get_embeddings() → Pinecone.upsert()      │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌─────────────────────────┐
│  PINECONE MEMORY │   │   NEO4J KNOWLEDGE GRAPH  │
│  (retriever.py)  │   │   (graphrag_index.py)    │
│                  │   │                          │
│  Namespaces:     │   │  Hetionet v1.0:          │
│  - medical_kb    │   │  - 47K Nodes             │
│  - patient_{id}  │   │  - 2.2M Edges            │
│                  │   │  - Disease/Drug/Gene     │
│  ada-002 embeds  │   │  - PubMed KG triples     │
└──────────────────┘   └─────────────────────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  AGENT LAYER (agents.py)                     │
│                                                              │
│  [CrewAI Sequential Pipeline]                                │
│                                                              │
│  1. 🔬 Diagnosis Agent (GPT-4o)                             │
│     Tools: vector_search, graph_query, pubmed_search        │
│     Output: conditions[], summary, risk_score               │
│                                                              │
│  2. 📈 Prognosis Agent (GPT-4o)                             │
│     Tools: vector_search, graph_query                       │
│     Output: risks[] with probability 0-100                  │
│                                                              │
│  3. 🥗 Lifestyle Agent (GPT-4o)                             │
│     Tools: vector_search, pubmed_search                     │
│     Output: diet{}, exercises[]                             │
│                                                              │
│  4. 💊 Medication Agent (GPT-4o)                            │
│     Tools: drugbank_lookup, graph_query                     │
│     Output: medications[], interactions[]                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
📦 Medical Knowledge Datasets
🧬 1. Hetionet (Knowledge Graph)
What: Biomedical knowledge graph with Disease-Drug-Gene-Anatomy relations
Size: 47,031 nodes | 2,250,197 edges
Format: TSV (nodes + edges)
Download: https://github.com/hetio/hetionet/tree/main/hetnet/tsv
Direct files:
wget https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv.gz
wget https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.tsv.gz
Load with: python graphrag_index.py --source hetionet
📰 2. PubMed Knowledge Graph
What: 4.8M biomedical entity triples from PubMed abstracts
Format: TSV (subject | predicate | object | PMID)
Download: https://pubmedkg.github.io
Load with: python graphrag_index.py --source pubmedkg
🏥 3. MIMIC-III (Clinical Notes - Signup Required)
What: De-identified ICU patient records, 53K admissions
Format: CSV (NOTEEVENTS.csv, LABEVENTS.csv, DIAGNOSES_ICD.csv)
Access: https://physionet.org/content/mimiciii/1.4/ (Free, requires CITI training ~1hr)
Recommended subset: NOTEEVENTS.csv (clinical notes) + DIAGNOSES_ICD.csv
Use for: Training prognosis models, risk scoring benchmarks
💊 4. DrugBank (Medications)
What: 14,000+ drugs with mechanisms, interactions, side effects
Format: CSV / XML
Free download (subset): https://go.drugbank.com/releases/latest#open-data
Full access: https://go.drugbank.com (academic license free)
Files to use: drug_links.csv, drug_drug_interactions.csv
Load with: python ingestion.py ./data/drugbank
⚠️ 5. SIDER (Side Effects)
What: Drug side effects mined from FDA labels — 1,430 drugs, 5,868 side effects
Format: TSV
Download: http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz
Also: meddra_freq.tsv.gz (frequency data)
🖼️ 6. MedPix (Medical Images)
What: 59,000+ medical images with diagnoses
Access: https://medpix.nlm.nih.gov/home
Use for: Training/testing image report OCR and vision model analysis
📚 7. BioASQ (QA Benchmarks)
What: Biomedical question answering dataset — great for eval
Download: http://bioasq.org/participate/challenges
🔬 8. ClinicalTrials.gov
What: 450K+ clinical trial records
API: https://clinicaltrials.gov/api/query/full_studies?expr={query}&fmt=JSON
Use for: Prognosis agent research tool
🚀 Setup Guide (Step by Step)
Step 1: Clone & Install
git clone https://github.com/yourname/medical-ai-agent
cd medical-ai-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Tesseract OCR (for image reports)
# macOS:  brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
Step 2: Configure API Keys
cp .env.example .env
# Edit .env with your keys:
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENV=us-east-1
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=yourpassword
LANGSMITH_API_KEY=...          # Optional
TWILIO_ACCOUNT_SID=...         # Optional (risk alerts)
DEEPL_API_KEY=...              # Optional (translation)
Step 3: Download Medical Data
mkdir -p data/hetionet data/drugbank data/pubmedkg

# Hetionet
cd data/hetionet
wget https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv.gz
wget https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.tsv.gz
gunzip *.gz

# DrugBank open data
# Visit https://go.drugbank.com/releases/latest#open-data and download:
# - open_structures.sdf.zip → drug structures
# - drug_links.csv.zip → core drug info ← use this one

# SIDER
cd ../sider
wget http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz
gunzip *.gz
Step 4: Start Neo4j
# Option A: Docker (recommended)
docker run -d \
  --name neo4j-medical \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.20

# Option B: Neo4j AuraDB (cloud, free tier)
# https://neo4j.com/cloud/platform/aura-graph-database/
Step 5: Build Knowledge Graph
python graphrag_index.py --source all --data-dir ./data
# This loads Hetionet + PubMed KG into Neo4j
# Estimated time: 10-30 minutes depending on dataset size
Step 6: Index Medical Knowledge into Pinecone
python ingestion.py ./data
# This embeds all medical texts and upserts to Pinecone
# Estimated time: 15-60 minutes
Step 7: Run the App
streamlit run app.py
# Opens at http://localhost:8501
Step 8: Test with Sample Report
Download a sample from MIMIC-III demo set:

# Or use any medical lab report PDF you have
# Upload via the Streamlit UI and click "Analyze"
🌐 Deploy to Streamlit Cloud
Push code to GitHub (without .env — add to .gitignore)
Go to https://share.streamlit.io
Connect repo → set app.py as entry point
Add secrets in Streamlit Cloud settings:
[secrets]
OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "..."
Deploy → share the URL!
🆕 New Features & Ideas
Feature	Tool / Library	Status
🖼️ Image Report Analysis	GPT-4o Vision	Ready
🎤 Voice Report Input	OpenAI Whisper	Ready
📱 SMS/Email Risk Alerts	Twilio	Ready
🌍 Multi-Language Output	DeepL API	Ready
📊 ROUGE/BLEU Evaluation	rouge-score	Ready
🔗 Clinical Trials Search	ClinicalTrials.gov API	Easy add
🧬 Drug-Gene Interaction	PharmGKB API	Easy add
📅 Longitudinal Tracking	Pinecone patient namespaces	Ready
🏥 FHIR Integration	fhir.resources	Medium
🔐 HIPAA Compliant Storage	AWS S3 + KMS	Advanced
⚠️ Medical Disclaimer
This system is for educational and research purposes only.
It does NOT replace professional medical diagnosis, treatment, or advice.
Always consult a qualified healthcare provider for medical decisions.
Patient data should be handled in compliance with HIPAA/GDPR regulations.
