"""
Vector Store Service.

Manages embeddings and retrieval for RAG over:
1. CT Rules document
2. Past learnings (from PPTs)
3. Historical test results
"""

import os
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Vector store for RAG functionality.
    
    Uses ChromaDB for local vector storage and retrieval.
    """
    
    def __init__(self, persist_dir: str = None):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory to persist the database. 
                        If None, uses in-memory storage.
        """
        self.persist_dir = persist_dir
        
        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()
        
        # Create collections
        self._rules_collection = None
        self._learnings_collection = None
        self._results_collection = None
    
    @property
    def rules_collection(self):
        """Collection for CT Rules documents."""
        if self._rules_collection is None:
            self._rules_collection = self.client.get_or_create_collection(
                name="ct_rules",
                metadata={"description": "Creative Testing Rules and Guidelines"}
            )
        return self._rules_collection
    
    @property
    def learnings_collection(self):
        """Collection for past learnings from PPTs."""
        if self._learnings_collection is None:
            self._learnings_collection = self.client.get_or_create_collection(
                name="past_learnings",
                metadata={"description": "Historical learnings from creative testing"}
            )
        return self._learnings_collection
    
    @property
    def results_collection(self):
        """Collection for historical test results."""
        if self._results_collection is None:
            self._results_collection = self.client.get_or_create_collection(
                name="test_results",
                metadata={"description": "Historical creative testing results"}
            )
        return self._results_collection
    
    def add_rules_document(self, text: str, document_id: str = "ct_rules_v1", metadata: dict = None):
        """
        Add the CT Rules document to the vector store.
        
        Args:
            text: Full text of the rules document
            document_id: Unique ID for the document
            metadata: Optional metadata (version, date, etc.)
        """
        # Split into chunks (simple paragraph-based chunking)
        chunks = self._chunk_text(text, max_chunk_size=1000)
        
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {**(metadata or {}), "chunk_index": i, "source": "ct_rules"} 
            for i in range(len(chunks))
        ]
        
        # Remove existing document chunks
        try:
            existing = self.rules_collection.get(where={"source": "ct_rules"})
            if existing["ids"]:
                self.rules_collection.delete(ids=existing["ids"])
        except Exception:
            pass
        
        # Add new chunks
        self.rules_collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )
    
    def add_learning(
        self, 
        text: str, 
        source_file: str,
        brand: str = None,
        campaign: str = None,
        date: str = None,
    ):
        """
        Add a learning from a PPT or other document.
        
        Args:
            text: The learning text
            source_file: Source file name
            brand: Brand name
            campaign: Campaign name
            date: Date of the learning
        """
        doc_id = f"learning_{hash(text) % 10000000}"
        
        metadata = {
            "source_file": source_file,
            "source": "past_learnings",
        }
        if brand:
            metadata["brand"] = brand
        if campaign:
            metadata["campaign"] = campaign
        if date:
            metadata["date"] = date
        
        self.learnings_collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata],
        )
    
    def add_result_summary(
        self,
        campaign_id: str,
        summary: str,
        brand: str = None,
        pass_rate: float = None,
        date: str = None,
    ):
        """
        Add a test result summary for future reference.
        
        Args:
            campaign_id: Campaign ID
            summary: Summary text of results
            brand: Brand name
            pass_rate: Pass rate percentage
            date: Date of results
        """
        doc_id = f"result_{campaign_id}"
        
        metadata = {
            "campaign_id": campaign_id,
            "source": "test_results",
        }
        if brand:
            metadata["brand"] = brand
        if pass_rate is not None:
            metadata["pass_rate"] = pass_rate
        if date:
            metadata["date"] = date
        
        # Upsert (remove if exists, then add)
        try:
            self.results_collection.delete(ids=[doc_id])
        except Exception:
            pass
        
        self.results_collection.add(
            documents=[summary],
            ids=[doc_id],
            metadatas=[metadata],
        )
    
    def query_rules(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Query the rules collection.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of {text, metadata} dicts
        """
        results = self.rules_collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        
        return self._format_results(results)
    
    def query_learnings(
        self, 
        query: str, 
        n_results: int = 5,
        brand: str = None,
    ) -> list[dict]:
        """
        Query past learnings.
        
        Args:
            query: Search query
            n_results: Number of results to return
            brand: Optional brand filter
            
        Returns:
            List of {text, metadata} dicts
        """
        where = None
        if brand:
            where = {"brand": brand}
        
        results = self.learnings_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        
        return self._format_results(results)
    
    def query_results(
        self,
        query: str,
        n_results: int = 5,
        brand: str = None,
    ) -> list[dict]:
        """
        Query historical results.
        
        Args:
            query: Search query
            n_results: Number of results to return
            brand: Optional brand filter
            
        Returns:
            List of {text, metadata} dicts
        """
        where = None
        if brand:
            where = {"brand": brand}
        
        results = self.results_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        
        return self._format_results(results)
    
    def query_all(
        self,
        query: str,
        n_results: int = 3,
        brand: str = None,
    ) -> dict:
        """
        Query all collections and combine results.
        
        Args:
            query: Search query
            n_results: Number of results per collection
            brand: Optional brand filter
            
        Returns:
            Dict with 'rules', 'learnings', 'results' keys
        """
        return {
            "rules": self.query_rules(query, n_results),
            "learnings": self.query_learnings(query, n_results, brand),
            "results": self.query_results(query, n_results, brand),
        }
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000) -> list[str]:
        """Split text into chunks."""
        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _format_results(self, results: dict) -> list[dict]:
        """Format ChromaDB results into a cleaner structure."""
        formatted = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
        
        return formatted
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "rules_count": self.rules_collection.count(),
            "learnings_count": self.learnings_collection.count(),
            "results_count": self.results_collection.count(),
        }


# Global instance
_vector_store: Optional[VectorStore] = None


def get_vector_store(persist_dir: str = None) -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        # Default to a data directory in the project
        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
        _vector_store = VectorStore(persist_dir)
    return _vector_store
