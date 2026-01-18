"""Code chunking with tree-sitter support."""
import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_java as tsjava
    import tree_sitter_go as tsgo
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from app.config import settings


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    chunk_id: str
    repo_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: Optional[str]
    chunk_text: str
    metadata: dict = field(default_factory=dict)


# Token encoder for measuring chunk sizes
encoding = None
if TIKTOKEN_AVAILABLE:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if TIKTOKEN_AVAILABLE and encoding:
        try:
            return len(encoding.encode(text))
        except Exception:
            pass
    # Fallback: rough estimate (words * 1.3)
    return int(len(text.split()) * 1.3)


def chunk_file(file_path: Path, repo_id: str, language: str) -> List[CodeChunk]:
    """
    Chunk a file into semantic units.
    
    Args:
        file_path: Path to the file
        repo_id: Repository ID
        language: Programming language
    
    Returns:
        List of code chunks
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    # Use tree-sitter for supported languages
    if TREE_SITTER_AVAILABLE and language in ['python', 'javascript', 'typescript', 'java', 'go']:
        chunks = chunk_with_tree_sitter(content, file_path, repo_id, language)
        if chunks:
            return chunks
    
    # Fallback to simple chunking
    return chunk_by_size(content, file_path, repo_id, language)


def chunk_with_tree_sitter(
    content: str,
    file_path: Path,
    repo_id: str,
    language: str
) -> List[CodeChunk]:
    """Chunk code using tree-sitter to identify functions/classes."""
    try:
        parser = Parser()
        
        # Set language
        if language == 'python':
            parser.set_language(Language(tspython.language()))
            query_string = """
            (function_definition) @function
            (class_definition) @class
            """
        elif language in ['javascript', 'typescript']:
            parser.set_language(Language(tsjavascript.language()))
            query_string = """
            (function_declaration) @function
            (class_declaration) @class
            (method_definition) @method
            """
        elif language == 'java':
            parser.set_language(Language(tsjava.language()))
            query_string = """
            (method_declaration) @method
            (class_declaration) @class
            """
        elif language == 'go':
            parser.set_language(Language(tsgo.language()))
            query_string = """
            (function_declaration) @function
            (method_declaration) @method
            """
        else:
            return []
        
        tree = parser.parse(bytes(content, 'utf8'))
        
        # Extract function/class boundaries
        chunks = []
        lines = content.split('\n')
        
        # Get all top-level definitions
        root = tree.root_node
        definitions = []
        
        for child in root.children:
            if child.type in ['function_definition', 'class_definition', 'function_declaration', 
                             'class_declaration', 'method_declaration']:
                start_line = child.start_point[0] + 1  # 1-indexed
                end_line = child.end_point[0] + 1
                
                # Get symbol name
                name_node = None
                for subchild in child.children:
                    if subchild.type in ['identifier', 'name']:
                        name_node = subchild
                        break
                
                symbol_name = name_node.text.decode('utf8') if name_node else None
                definitions.append((start_line, end_line, symbol_name))
        
        # Create chunks from definitions
        for i, (start, end, symbol) in enumerate(definitions):
            chunk_text = '\n'.join(lines[start-1:end])
            
            # Skip if too large, will be handled by size chunking
            if count_tokens(chunk_text) > settings.chunk_size * 1.5:
                continue
            
            chunk_id = f"{repo_id}:{file_path.name}:{start}:{end}"
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                repo_id=repo_id,
                file_path=str(file_path),
                language=language,
                start_line=start,
                end_line=end,
                symbol_name=symbol,
                chunk_text=chunk_text,
                metadata={'chunking_method': 'tree_sitter'}
            ))
        
        # If we got meaningful chunks, return them
        if chunks:
            return chunks
            
    except Exception as e:
        print(f"Tree-sitter parsing failed for {file_path}: {e}")
    
    return []


def chunk_by_size(
    content: str,
    file_path: Path,
    repo_id: str,
    language: str
) -> List[CodeChunk]:
    """
    Chunk code by size with overlap.
    
    Args:
        content: File content
        file_path: Path to file
        repo_id: Repository ID
        language: Programming language
    
    Returns:
        List of code chunks
    """
    lines = content.split('\n')
    chunks = []
    
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    
    current_chunk = []
    current_tokens = 0
    start_line = 1
    
    for i, line in enumerate(lines, 1):
        line_tokens = count_tokens(line)
        
        if current_tokens + line_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = '\n'.join(current_chunk)
            chunk_id = f"{repo_id}:{file_path.name}:{start_line}:{i-1}"
            
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                repo_id=repo_id,
                file_path=str(file_path),
                language=language,
                start_line=start_line,
                end_line=i - 1,
                symbol_name=None,
                chunk_text=chunk_text,
                metadata={'chunking_method': 'size_based'}
            ))
            
            # Create overlap
            overlap_lines = []
            overlap_tokens = 0
            for j in range(len(current_chunk) - 1, -1, -1):
                tokens = count_tokens(current_chunk[j])
                if overlap_tokens + tokens > overlap:
                    break
                overlap_lines.insert(0, current_chunk[j])
                overlap_tokens += tokens
            
            current_chunk = overlap_lines + [line]
            current_tokens = overlap_tokens + line_tokens
            start_line = i - len(overlap_lines) + 1
        else:
            current_chunk.append(line)
            current_tokens += line_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunk_id = f"{repo_id}:{file_path.name}:{start_line}:{len(lines)}"
        
        chunks.append(CodeChunk(
            chunk_id=chunk_id,
            repo_id=repo_id,
            file_path=str(file_path),
            language=language,
            start_line=start_line,
            end_line=len(lines),
            symbol_name=None,
            chunk_text=chunk_text,
            metadata={'chunking_method': 'size_based'}
        ))
    
    return chunks


def extract_imports(content: str, language: str) -> List[str]:
    """Extract import statements from code."""
    imports = []
    
    if language == 'python':
        pattern = r'^(?:from\s+[\w.]+\s+)?import\s+.+$'
    elif language in ['javascript', 'typescript']:
        pattern = r'^import\s+.+$'
    elif language == 'java':
        pattern = r'^import\s+[\w.]+;$'
    elif language == 'go':
        pattern = r'^import\s+.+$'
    else:
        return imports
    
    for line in content.split('\n'):
        if re.match(pattern, line.strip()):
            imports.append(line.strip())
    
    return imports
