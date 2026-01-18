"""Metadata store using SQLite."""
import sqlite3
import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from indexing.chunking import CodeChunk
from indexing.loader import RepoMetadata
from app.config import settings


class MetadataStore:
    """Manage metadata storage using SQLite."""
    
    def __init__(self, db_path: Path = None):
        """
        Initialize the metadata store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or settings.metadata_db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create repos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repos (
                repo_id TEXT PRIMARY KEY,
                url TEXT,
                local_path TEXT NOT NULL,
                commit_hash TEXT,
                indexed_at TIMESTAMP,
                stats TEXT
            )
        """)
        
        # Create chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                language TEXT,
                start_line INTEGER,
                end_line INTEGER,
                symbol_name TEXT,
                chunk_text TEXT,
                FOREIGN KEY (repo_id) REFERENCES repos(repo_id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name)
        """)
        
        # Create full-text search virtual table
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                chunk_text,
                content=chunks,
                content_rowid=rowid
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def save_repo_metadata(self, metadata: RepoMetadata) -> None:
        """
        Save repository metadata.
        
        Args:
            metadata: Repository metadata
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO repos (repo_id, url, local_path, commit_hash, indexed_at, stats)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            metadata.repo_id,
            metadata.url,
            metadata.local_path,
            metadata.commit_hash,
            datetime.now().isoformat(),
            json.dumps(metadata.stats)
        ))
        
        conn.commit()
        conn.close()
    
    def get_repo_metadata(self, repo_id: str) -> Optional[Dict]:
        """
        Get repository metadata.
        
        Args:
            repo_id: Repository ID
        
        Returns:
            Repository metadata or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT repo_id, url, local_path, commit_hash, indexed_at, stats
            FROM repos
            WHERE repo_id = ?
        """, (repo_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'repo_id': row[0],
                'url': row[1],
                'local_path': row[2],
                'commit_hash': row[3],
                'indexed_at': row[4],
                'stats': json.loads(row[5]) if row[5] else {}
            }
        
        return None
    
    def list_repos(self) -> List[Dict]:
        """
        List all indexed repositories.
        
        Returns:
            List of repository metadata
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT repo_id, url, local_path, commit_hash, indexed_at, stats
            FROM repos
            ORDER BY indexed_at DESC
        """)
        
        repos = []
        for row in cursor.fetchall():
            repos.append({
                'repo_id': row[0],
                'url': row[1],
                'local_path': row[2],
                'commit_hash': row[3],
                'indexed_at': row[4],
                'stats': json.loads(row[5]) if row[5] else {}
            })
        
        conn.close()
        return repos
    
    def save_chunks(self, chunks: List[CodeChunk]) -> None:
        """
        Save code chunks.
        
        Args:
            chunks: List of code chunks
        """
        if not chunks:
            return
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Insert chunks
        chunk_data = [
            (
                chunk.chunk_id,
                chunk.repo_id,
                chunk.file_path,
                chunk.language,
                chunk.start_line,
                chunk.end_line,
                chunk.symbol_name,
                chunk.chunk_text
            )
            for chunk in chunks
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO chunks 
            (chunk_id, repo_id, file_path, language, start_line, end_line, symbol_name, chunk_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, chunk_data)
        
        # Update FTS index
        fts_data = [(chunk.chunk_id, chunk.chunk_text) for chunk in chunks]
        cursor.executemany("""
            INSERT OR REPLACE INTO chunks_fts (chunk_id, chunk_text)
            VALUES (?, ?)
        """, fts_data)
        
        conn.commit()
        conn.close()
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
        
        Returns:
            Chunk data or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_id, repo_id, file_path, language, start_line, end_line, symbol_name, chunk_text
            FROM chunks
            WHERE chunk_id = ?
        """, (chunk_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'chunk_id': row[0],
                'repo_id': row[1],
                'file_path': row[2],
                'language': row[3],
                'start_line': row[4],
                'end_line': row[5],
                'symbol_name': row[6],
                'chunk_text': row[7]
            }
        
        return None
    
    def search_chunks_lexical(self, repo_id: str, keyword: str, limit: int = 10) -> List[Dict]:
        """
        Search chunks using full-text search.
        
        Args:
            repo_id: Repository ID
            keyword: Search keyword
            limit: Maximum number of results
        
        Returns:
            List of matching chunks
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.chunk_id, c.repo_id, c.file_path, c.language, 
                   c.start_line, c.end_line, c.symbol_name, c.chunk_text,
                   fts.rank
            FROM chunks_fts fts
            JOIN chunks c ON fts.chunk_id = c.chunk_id
            WHERE fts.chunk_text MATCH ? AND c.repo_id = ?
            ORDER BY fts.rank
            LIMIT ?
        """, (keyword, repo_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'chunk_id': row[0],
                'repo_id': row[1],
                'file_path': row[2],
                'language': row[3],
                'start_line': row[4],
                'end_line': row[5],
                'symbol_name': row[6],
                'chunk_text': row[7],
                'score': -row[8]  # FTS rank is negative, convert to positive
            })
        
        conn.close()
        return results
    
    def delete_repo(self, repo_id: str) -> None:
        """
        Delete a repository and all its chunks.
        
        Args:
            repo_id: Repository ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Delete chunks FTS entries
        cursor.execute("""
            DELETE FROM chunks_fts 
            WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE repo_id = ?)
        """, (repo_id,))
        
        # Delete chunks
        cursor.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
        
        # Delete repo
        cursor.execute("DELETE FROM repos WHERE repo_id = ?", (repo_id,))
        
        conn.commit()
        conn.close()


# Global instance
_metadata_store = None


def get_metadata_store() -> MetadataStore:
    """Get or create the global metadata store instance."""
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store
