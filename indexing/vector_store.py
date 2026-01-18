"""Vector store using ChromaDB."""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from indexing.chunking import CodeChunk
from indexing.embeddings import get_embedding_generator
from app.config import settings


class VectorStore:
    """Manage vector storage and retrieval using ChromaDB."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_db_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.embedding_generator = get_embedding_generator()
    
    def create_collection(self, repo_id: str, reset: bool = False) -> chromadb.Collection:
        """
        Create or get a collection for a repository.
        
        Args:
            repo_id: Repository ID
            reset: If True, delete existing collection
        
        Returns:
            ChromaDB collection
        """
        collection_name = f"repo_{repo_id}"
        
        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        return collection
    
    def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """
        Add code chunks to the vector store.
        
        Args:
            chunks: List of code chunks
        """
        if not chunks:
            return
        
        repo_id = chunks[0].repo_id
        collection = self.create_collection(repo_id)
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.chunk_text for chunk in chunks]
        metadatas = [
            {
                'file_path': chunk.file_path,
                'language': chunk.language,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'symbol_name': chunk.symbol_name or '',
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_generator.generate_embeddings(documents)
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(
        self,
        query: str,
        repo_id: str,
        n_results: int = 10,
        file_path_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar code chunks.
        
        Args:
            query: Search query
            repo_id: Repository ID
            n_results: Number of results to return
            file_path_filter: Optional file path to filter by
        
        Returns:
            List of search results with metadata
        """
        collection_name = f"repo_{repo_id}"
        
        try:
            collection = self.client.get_collection(collection_name)
        except Exception:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Build where clause if filtering
        where = None
        if file_path_filter:
            where = {"file_path": {"$contains": file_path_filter}}
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'chunk_id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                    'file_path': results['metadatas'][0][i]['file_path'],
                    'start_line': results['metadatas'][0][i]['start_line'],
                    'end_line': results['metadatas'][0][i]['end_line'],
                    'symbol_name': results['metadatas'][0][i].get('symbol_name', ''),
                })
        
        return formatted_results
    
    def get_chunk_by_id(self, chunk_id: str, repo_id: str) -> Optional[Dict]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            repo_id: Repository ID
        
        Returns:
            Chunk data or None
        """
        collection_name = f"repo_{repo_id}"
        
        try:
            collection = self.client.get_collection(collection_name)
            result = collection.get(ids=[chunk_id], include=['documents', 'metadatas'])
            
            if result['ids']:
                return {
                    'chunk_id': result['ids'][0],
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0],
                }
        except Exception:
            pass
        
        return None
    
    def delete_collection(self, repo_id: str) -> None:
        """
        Delete a collection.
        
        Args:
            repo_id: Repository ID
        """
        collection_name = f"repo_{repo_id}"
        try:
            self.client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")


# Global instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
