"""Configuration management for RAG Chat System."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DB_NAME = os.getenv("DB_NAME", "rag_chat_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Text Splitting Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval Configuration
    SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", "3"))

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if not cls.MONGODB_URI:
            raise ValueError("MONGODB_URI is required")

        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
