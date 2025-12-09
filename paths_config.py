"""
Centralized Path Configuration - All paths in one place

Directory Structure:
    /home/aiadmin/netovo_voicebot/
    ├── voicebot/           (AudioSocket voicebot)
    ├── rag/                (RAG pipeline & customer data)
    │   ├── customers/      (Customer PDFs and processed data)
    │   └── chroma_db/      (ChromaDB vector database)
    └── temp/               (Temporary files)
"""

import os
import platform
from pathlib import Path

# Detect OS and set default base directory
IS_WINDOWS = platform.system() == 'Windows'
DEFAULT_BASE_DIR = 'G:/home/aiadmin/netovo_voicebot' if IS_WINDOWS else '/home/aiadmin/netovo_voicebot'

# Load .env file from main netovo_voicebot directory
try:
    from dotenv import load_dotenv
    # Try to find .env in the base netovo_voicebot directory
    # On Linux: /home/aiadmin/netovo_voicebot/.env
    # On Windows dev: G:/home/aiadmin/netovo_voicebot/.env
    base_dir = Path(os.getenv('BASE_DIR', DEFAULT_BASE_DIR))
    env_path = base_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback: try Rag/.env for backward compatibility
        env_path_fallback = Path(__file__).parent / "Rag" / ".env"
        if env_path_fallback.exists():
            load_dotenv(env_path_fallback)
except ImportError:
    pass


class PathsConfig:
    """Centralized paths for RAG pipeline and voicebot"""

    # Base directory: /home/aiadmin/netovo_voicebot
    BASE_DIR = Path(os.getenv('BASE_DIR', DEFAULT_BASE_DIR))

    # RAG directory: /home/aiadmin/netovo_voicebot/rag
    RAG_DIR = BASE_DIR / os.getenv('RAG_SUBDIR', 'rag')

    # Voicebot directory: /home/aiadmin/netovo_voicebot/voicebot
    VOICEBOT_DIR = BASE_DIR / os.getenv('VOICEBOT_SUBDIR', 'voicebot')

    # ChromaDB: /home/aiadmin/netovo_voicebot/rag/chroma_db
    CHROMA_DB_PATH = RAG_DIR / os.getenv('CHROMA_DB_PATH', 'chroma_db').lstrip('./')

    # Customer data: /home/aiadmin/netovo_voicebot/rag/customers
    CUSTOMERS_DIR = RAG_DIR / "customers"

    # Temp directory: /home/aiadmin/netovo_voicebot/temp
    TEMP_DIR = Path(os.getenv('TEMP_DIR', BASE_DIR / 'temp'))

    # ChromaDB temp directory for temporary files (separate from main storage)
    # Uses BASE_DIR/temp/chroma_temp since /tmp may not exist on all systems
    CHROMA_TEMP_DIR = Path(os.getenv('CHROMA_TEMP_DIR', BASE_DIR / 'temp' / 'chroma_temp'))

    # Vosk model path from environment variable (no default path)
    VOSK_MODEL_PATH = Path(os.getenv('VOSK_MODEL_PATH', BASE_DIR / 'vosk-model-en-us-0.22'))

    @classmethod
    def get_customer_dir(cls, customer_id: str) -> Path:
        """Get customer base directory: customers/{customer_id}/"""
        return cls.CUSTOMERS_DIR / customer_id

    @classmethod
    def get_doc_dir(cls, customer_id: str, doc_name: str) -> Path:
        """Get document directory: customers/{customer_id}/{doc_name}/"""
        return cls.get_customer_dir(customer_id) / doc_name


if __name__ == "__main__":
    print("=" * 70)
    print("CENTRALIZED PATHS CONFIGURATION")
    print("=" * 70)
    print(f"BASE_DIR:         {PathsConfig.BASE_DIR}")
    print(f"RAG_DIR:          {PathsConfig.RAG_DIR}")
    print(f"VOICEBOT_DIR:     {PathsConfig.VOICEBOT_DIR}")
    print(f"CHROMA_DB_PATH:   {PathsConfig.CHROMA_DB_PATH}")
    print(f"CUSTOMERS_DIR:    {PathsConfig.CUSTOMERS_DIR}")
    print(f"TEMP_DIR:         {PathsConfig.TEMP_DIR}")
    print(f"CHROMA_TEMP_DIR:  {PathsConfig.CHROMA_TEMP_DIR}")
    print(f"VOSK_MODEL_PATH:  {PathsConfig.VOSK_MODEL_PATH}")
    print("=" * 70)
