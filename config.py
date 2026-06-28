# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "Adonis_docs"
VECTOR_SIZE = 3072
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1-mini"

BATCH_SIZE = 64

QDRANT_HOST = "qdrant_db"
QDRANT_PORT = 6333

JSON_PATH = "data/All_chunks_cleaned_V1.json"
