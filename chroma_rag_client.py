#!/usr/bin/env python3
"""
ChromaDB RAG Client for AudioSocket Voicebot
Lightweight client for retrieving relevant documents from ChromaDB
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("chromadb not installed - RAG will be disabled")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed - RAG will be disabled")

logger = logging.getLogger(__name__)


class ChromaRAGClient:
    """
    Lightweight ChromaDB client for RAG retrieval in voicebot.

    Features:
    - Query customer-specific collections
    - Format context for LLM prompts
    - Handle missing collections gracefully
    - Filter low-quality results
    """

    def __init__(self, db_path: str, embedding_model: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize ChromaDB RAG client

        Args:
            db_path: Path to ChromaDB persistent storage
            embedding_model: Name of sentence-transformers model (must match indexing!)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model

        if not self.db_path.exists():
            logger.error(f"ChromaDB path does not exist: {db_path}")
            raise FileNotFoundError(f"ChromaDB path not found: {db_path}")

        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            logger.info(f"ChromaDB client initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

        # Initialize embedding model
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder = SentenceTransformer(embedding_model, device=device)

            # Warm up model (first inference is slower)
            _ = self.embedder.encode("warmup query", normalize_embeddings=True)

            # Validate embedding dimensions
            test_embedding = self.embedder.encode("test", normalize_embeddings=True)
            self.embedding_dim = len(test_embedding)

            logger.info(f"Embedding model loaded: {embedding_model} ({self.embedding_dim}D) on {device}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")

    def get_customer_collection(self, customer_id: str):
        """
        Get customer-specific collection from ChromaDB

        Args:
            customer_id: Customer identifier (e.g., "stuart_dean", "skisafe")

        Returns:
            ChromaDB collection object or None if not found
        """
        # Normalize customer_id format (lowercase, underscores)
        normalized_id = self._normalize_customer_id(customer_id)

        try:
            collection = self.client.get_collection(name=normalized_id)
            logger.info(f"Found collection: {normalized_id} ({collection.count()} chunks)")
            return collection
        except Exception as e:
            logger.warning(f"Collection not found for customer '{customer_id}': {e}")
            return None

    def retrieve(
        self,
        customer_id: str,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.3
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query

        Args:
            customer_id: Customer identifier
            query: User's question/query
            top_k: Number of top results to retrieve
            min_similarity: Minimum similarity score (0-1) to include result

        Returns:
            List of result dictionaries with keys:
            - text: Chunk text
            - metadata: Document metadata (doc_name, page_num, etc.)
            - similarity: Similarity score (0-1)
            - distance: Vector distance
        """
        # Get customer collection
        collection = self.get_customer_collection(customer_id)

        if collection is None:
            logger.error(f"No knowledge base found for customer: {customer_id}")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(
                query,
                convert_to_tensor=False,
                normalize_embeddings=True
            ).tolist()

            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # Format and filter results
            formatted_results = []

            if results and results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    chunk_id = results['ids'][0][i]
                    distance = results['distances'][0][i]
                    text = results['documents'][0][i] if results['documents'] else ''
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                    # Convert distance to similarity (for cosine: similarity = 1 - distance)
                    similarity = max(0, 1 - distance)

                    # Filter by minimum similarity
                    if similarity < min_similarity:
                        logger.debug(f"Skipping low-similarity chunk: {similarity:.3f} < {min_similarity}")
                        continue

                    # Filter out junk chunks
                    if not self._is_valid_chunk(text):
                        logger.debug(f"Skipping invalid chunk: {text[:50]}...")
                        continue

                    formatted_results.append({
                        'chunk_id': chunk_id,
                        'text': text,
                        'metadata': metadata,
                        'similarity': similarity,
                        'distance': distance
                    })

            logger.info(f"Retrieved {len(formatted_results)} relevant chunks (from {top_k} candidates)")

            # Log top result for debugging
            if formatted_results:
                top = formatted_results[0]
                logger.debug(f"Top result: similarity={top['similarity']:.3f}, doc={top['metadata'].get('doc_name', 'unknown')}, text={top['text'][:100]}...")

            return formatted_results

        except Exception as e:
            logger.error(f"RAG retrieval error: {e}", exc_info=True)
            return []

    def format_context(
        self,
        results: List[Dict],
        max_length: int = 2000,
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved documents for LLM prompt

        Args:
            results: List of retrieval results
            max_length: Maximum context length in characters
            include_metadata: Include document name and page numbers

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []
        current_length = 0

        for i, result in enumerate(results):
            text = result['text']
            metadata = result['metadata']
            similarity = result.get('similarity', 0)

            # Format chunk with metadata
            if include_metadata:
                doc_name = metadata.get('doc_name', 'Unknown')
                page_num = metadata.get('page_num', '?')

                chunk_formatted = f"""[Document: {doc_name}, Page {page_num}, Relevance: {similarity:.0%}]
{text}
"""
            else:
                chunk_formatted = text + "\n\n"

            # Check length limit
            if current_length + len(chunk_formatted) > max_length:
                logger.warning(f"Context limit reached, using only {i} of {len(results)} chunks")
                break

            context_parts.append(chunk_formatted)
            current_length += len(chunk_formatted)

        # Join all chunks
        context = "\n---\n".join(context_parts)

        logger.info(f"Formatted context: {len(context_parts)} chunks, {len(context)} chars")
        return context

    def _normalize_customer_id(self, customer_id: str) -> str:
        """
        Normalize customer ID to match collection naming

        Args:
            customer_id: Raw customer ID

        Returns:
            Normalized customer ID (lowercase, underscores)
        """
        return customer_id.lower().replace(" ", "_").replace("-", "_")

    def _is_valid_chunk(self, text: str) -> bool:
        """
        Check if chunk is valid (not boilerplate/junk)

        Args:
            text: Chunk text

        Returns:
            True if valid, False if should be filtered out
        """
        # Filter too short chunks
        if len(text.strip()) < 50:
            return False

        # Filter mostly numeric chunks (page numbers, etc.)
        if sum(c.isdigit() for c in text) / max(len(text), 1) > 0.5:
            return False

        # Filter boilerplate patterns (short chunks only)
        if len(text) < 100:
            boilerplate_patterns = [
                "page", "copyright", "all rights reserved",
                "confidential", "proprietary"
            ]
            text_lower = text.lower()
            if sum(pattern in text_lower for pattern in boilerplate_patterns) >= 2:
                return False

        return True

    def health_check(self, customer_id: str) -> Dict[str, any]:
        """
        Perform health check for customer's knowledge base

        Args:
            customer_id: Customer identifier

        Returns:
            Health check results dictionary
        """
        try:
            collection = self.get_customer_collection(customer_id)

            if collection is None:
                return {
                    'status': 'error',
                    'message': f'Collection not found for {customer_id}',
                    'chunks': 0
                }

            # Try a test query
            test_results = self.retrieve(customer_id, "test query", top_k=1)

            return {
                'status': 'ok',
                'customer_id': customer_id,
                'collection_name': collection.name,
                'total_chunks': collection.count(),
                'test_retrieval': 'success' if test_results else 'no_results',
                'embedding_model': self.embedding_model_name,
                'embedding_dim': self.embedding_dim
            }

        except Exception as e:
            logger.error(f"Health check failed for {customer_id}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }


# Singleton instance for efficient reuse
_rag_client_instance = None


def get_rag_client(db_path: str, embedding_model: str = "BAAI/bge-base-en-v1.5") -> ChromaRAGClient:
    """
    Get or create singleton RAG client instance

    Args:
        db_path: Path to ChromaDB
        embedding_model: Embedding model name

    Returns:
        ChromaRAGClient instance
    """
    global _rag_client_instance

    if _rag_client_instance is None:
        _rag_client_instance = ChromaRAGClient(db_path, embedding_model)
        logger.info("Created new RAG client instance")
    else:
        logger.debug("Reusing existing RAG client instance")

    return _rag_client_instance


if __name__ == "__main__":
    # Test the RAG client
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python chroma_rag_client.py <db_path> [customer_id] [query]")
        print("Example: python chroma_rag_client.py ./customers/skisafe/kb.db stuart_dean 'What is 3CX?'")
        sys.exit(1)

    db_path = sys.argv[1]
    customer_id = sys.argv[2] if len(sys.argv) > 2 else "stuart_dean"
    query = sys.argv[3] if len(sys.argv) > 3 else "What is VPN?"

    try:
        # Initialize client
        client = ChromaRAGClient(db_path)

        # Health check
        health = client.health_check(customer_id)
        print("\n=== HEALTH CHECK ===")
        print(f"Status: {health['status']}")
        print(f"Total chunks: {health.get('total_chunks', 0)}")
        print(f"Embedding model: {health.get('embedding_model', 'unknown')}")

        # Test retrieval
        print(f"\n=== RETRIEVAL TEST ===")
        print(f"Query: {query}")
        print(f"Customer: {customer_id}")

        results = client.retrieve(customer_id, query, top_k=3)

        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Similarity: {result['similarity']:.3f}")
                print(f"Document: {result['metadata'].get('doc_name', 'unknown')}")
                print(f"Page: {result['metadata'].get('page_num', '?')}")
                print(f"Text: {result['text'][:200]}...")

            # Format context
            print(f"\n=== FORMATTED CONTEXT ===")
            context = client.format_context(results)
            print(context[:500] + "..." if len(context) > 500 else context)
        else:
            print("No results found")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
