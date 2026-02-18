"""
╔══════════════════════════════════════════════════════════════╗
║               MEDICAL AI AGENT - config.py                   ║
║  All configuration, API keys, and model parameters           ║
╚══════════════════════════════════════════════════════════════╝

⚠️  NEVER commit this file with real API keys to version control!
    Use environment variables or a .env file with python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


# ──────────────────────────────────────────────────────────────
# API Keys (load from environment - NEVER hardcode)
# ──────────────────────────────────────────────────────────────

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY",   "sk-your-openai-api-key-here")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY",  "your-pinecone-api-key-here")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Optional: Claude fallback
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")  # Optional: LangSmith tracing

# Twilio (for risk alerts)
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN",    "")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_FROM_NUMBER",   "")

# DeepL (for multi-language output)
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")


# ──────────────────────────────────────────────────────────────
# Neo4j / Graph Database
# ──────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your-neo4j-password")

# AuraDB (Neo4j cloud) - use for production
# NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://xxxxxxxx.databases.neo4j.io")


# ──────────────────────────────────────────────────────────────
# Pinecone Vector Database
# ──────────────────────────────────────────────────────────────

PINECONE_ENV    = os.getenv("PINECONE_ENV",   "us-east-1")  # Serverless region
PINECONE_INDEX  = os.getenv("PINECONE_INDEX", "medical-ai-agent")


# ──────────────────────────────────────────────────────────────
# LLM Configuration
# ──────────────────────────────────────────────────────────────

# Primary LLM (can be overridden via Streamlit sidebar)
LLM_MODEL         = os.getenv("LLM_MODEL", "gpt-4o")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "4096"))

# Embedding model (OpenAI)
EMBED_MODEL       = "text-embedding-ada-002"
EMBED_DIMENSION   = 1536

# Claude (Anthropic) optional fallback
CLAUDE_MODEL      = "claude-3-5-sonnet-20241022"

# Vision model for image reports
VISION_MODEL      = "gpt-4o"


# ──────────────────────────────────────────────────────────────
# Retrieval Configuration
# ──────────────────────────────────────────────────────────────

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "8"))   # Vectors to retrieve
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",      "800")) # Characters per chunk
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP",   "100")) # Overlap between chunks


# ──────────────────────────────────────────────────────────────
# Agent Configuration
# ──────────────────────────────────────────────────────────────

VERBOSE_AGENTS  = os.getenv("VERBOSE_AGENTS", "false").lower() == "true"
MAX_AGENT_ITER  = int(os.getenv("MAX_AGENT_ITER", "5"))

# LangSmith tracing (optional monitoring)
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = "MedAI-Agent"


# ──────────────────────────────────────────────────────────────
# Data Paths
# ──────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CACHE_DIR       = BASE_DIR / ".cache"
EXPORT_DIR      = BASE_DIR / "exports"

# Ensure directories exist
for d in [DATA_DIR, CACHE_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Application Settings
# ──────────────────────────────────────────────────────────────

APP_TITLE       = "Medical AI Agent"
APP_VERSION     = "2.0.0"
APP_DESCRIPTION = (
    "AI-powered medical report analysis using GraphRAG, OpenAI Agents, and Pinecone."
    " For educational and research purposes only."
)

# Risk score thresholds
RISK_HIGH_THRESHOLD   = 70  # % probability = HIGH risk
RISK_MEDIUM_THRESHOLD = 40  # % probability = MEDIUM risk

# File upload limits
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = ["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"]


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

def validate_config():
    """Check that required API keys are set."""
    warnings = []
    if OPENAI_API_KEY.startswith("sk-your"):
        warnings.append("⚠️  OPENAI_API_KEY not set - agents will run in demo mode")
    if PINECONE_API_KEY == "your-pinecone-api-key-here":
        warnings.append("⚠️  PINECONE_API_KEY not set - using mock retriever")
    if NEO4J_PASSWORD == "your-neo4j-password":
        warnings.append("⚠️  NEO4J_PASSWORD not set - using demo graph data")
    return warnings


if __name__ == "__main__":
    warnings = validate_config()
    if warnings:
        print("Configuration Warnings:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("✅ All API keys configured correctly")
