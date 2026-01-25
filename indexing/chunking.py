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
from core.constants import MIN_CHUNK_SIZE_TOKENS, MAX_CONTEXT_LINES


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


def _extract_comments_and_docstring(
    lines: List[str],
    start_line: int,
    max_lines_back: int = None
) -> str:
    """
    Extract comments and docstrings before a definition.
    
    Args:
        lines: All lines of the file
        start_line: Starting line of the definition (1-indexed)
        max_lines_back: Maximum lines to look back (defaults to MAX_CONTEXT_LINES)
    
    Returns:
        Combined comments and docstrings
    """
    if max_lines_back is None:
        max_lines_back = MAX_CONTEXT_LINES
    
    context_lines = []
    # Look back up to max_lines_back lines
    lookback_start = max(0, start_line - max_lines_back - 1)
    
    for i in range(lookback_start, start_line - 1):
        if i < len(lines):
            line = lines[i].strip()
            # Include comment lines and docstrings
            if line.startswith('#') or line.startswith('"""') or line.startswith("'''") or line.startswith('//') or line.startswith('/*'):
                context_lines.append(lines[i])
            # Include empty lines between comments and definition
            elif not line and context_lines:
                context_lines.append(lines[i])
            # Stop if we hit non-comment code
            elif line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                # Check if it's a docstring continuation
                if '"""' in line or "'''" in line:
                    context_lines.append(lines[i])
                elif context_lines:
                    # We've hit actual code, stop collecting
                    break
    
    return '\n'.join(context_lines)


def _extract_function_signature(node, content: str, language: str) -> dict:
    """
    Extract function signature information (parameters, return type).
    
    Args:
        node: Tree-sitter node
        content: File content
        language: Programming language
    
    Returns:
        Dictionary with signature metadata
    """
    signature_info = {
        'parameters': [],
        'return_type': None
    }
    
    try:
        # Extract parameters based on language
        if language == 'python':
            # Look for parameters node
            for child in node.children:
                if child.type == 'parameters':
                    params_text = content[child.start_byte:child.end_byte].decode('utf8')
                    signature_info['parameters'] = [p.strip() for p in params_text.strip('()').split(',') if p.strip()]
                elif child.type == 'type' or child.type == 'return_type':
                    signature_info['return_type'] = content[child.start_byte:child.end_byte].decode('utf8')
        
        elif language in ['javascript', 'typescript']:
            for child in node.children:
                if child.type == 'formal_parameters':
                    params_text = content[child.start_byte:child.end_byte].decode('utf8')
                    signature_info['parameters'] = [p.strip() for p in params_text.strip('()').split(',') if p.strip()]
                elif child.type == 'type_annotation':
                    signature_info['return_type'] = content[child.start_byte:child.end_byte].decode('utf8')
        
        elif language == 'java':
            for child in node.children:
                if child.type == 'formal_parameters':
                    params_text = content[child.start_byte:child.end_byte].decode('utf8')
                    signature_info['parameters'] = [p.strip() for p in params_text.strip('()').split(',') if p.strip()]
                elif child.type == 'type':
                    signature_info['return_type'] = content[child.start_byte:child.end_byte].decode('utf8')
    except Exception:
        pass
    
    return signature_info


def chunk_with_tree_sitter(
    content: str,
    file_path: Path,
    repo_id: str,
    language: str
) -> List[CodeChunk]:
    """Chunk code using tree-sitter to identify functions/classes with enhanced context."""
    try:
        parser = Parser()
        
        # Set language
        if language == 'python':
            parser.set_language(Language(tspython.language()))
        elif language in ['javascript', 'typescript']:
            parser.set_language(Language(tsjavascript.language()))
        elif language == 'java':
            parser.set_language(Language(tsjava.language()))
        elif language == 'go':
            parser.set_language(Language(tsgo.language()))
        else:
            return []
        
        tree = parser.parse(bytes(content, 'utf8'))
        
        # Extract imports for metadata
        imports = extract_imports(content, language)
        
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
                
                # Extract signature info
                signature_info = _extract_function_signature(child, content.encode('utf8'), language)
                
                definitions.append((start_line, end_line, symbol_name, signature_info))
        
        # Create chunks from definitions with context
        for i, (start, end, symbol, sig_info) in enumerate(definitions):
            # Get the actual code
            chunk_code = '\n'.join(lines[start-1:end])
            
            # Extract comments and docstrings before the definition
            context = _extract_comments_and_docstring(lines, start)
            
            # Combine context and code
            if context:
                chunk_text = context + '\n' + chunk_code
            else:
                chunk_text = chunk_code
            
            # Skip if too large, will be handled by size chunking
            if count_tokens(chunk_text) > settings.chunk_size * 1.5:
                # Still create chunk but without context if it's just slightly over
                chunk_text = chunk_code
                if count_tokens(chunk_text) > settings.chunk_size * 1.5:
                    continue
            
            chunk_id = f"{repo_id}:{file_path.name}:{start}:{end}"
            
            # Build enhanced metadata
            metadata = {
                'chunking_method': 'tree_sitter',
                'imports': imports[:10],  # Store top 10 imports
                'has_context': bool(context),
                'parameters': sig_info.get('parameters', [])[:5],  # Store first 5 parameters
                'return_type': sig_info.get('return_type')
            }
            
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                repo_id=repo_id,
                file_path=str(file_path),
                language=language,
                start_line=start,
                end_line=end,
                symbol_name=symbol,
                chunk_text=chunk_text,
                metadata=metadata
            ))
        
        # Merge very small chunks with adjacent chunks
        chunks = _merge_small_chunks(chunks)
        
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
    
    # Extract imports once for all chunks
    imports = extract_imports(content, language)
    
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
                metadata={
                    'chunking_method': 'size_based',
                    'imports': imports[:10]  # Store top 10 imports
                }
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
            metadata={
                'chunking_method': 'size_based',
                'imports': imports[:10]  # Store top 10 imports
            }
        ))
    
    # Merge very small chunks for size-based chunking too
    chunks = _merge_small_chunks(chunks)
    
    return chunks


def _merge_small_chunks(chunks: List[CodeChunk], min_tokens: int = None) -> List[CodeChunk]:
    """
    Merge very small chunks with adjacent chunks.
    
    Args:
        chunks: List of chunks to process
        min_tokens: Minimum token count to keep a chunk separate (defaults to MIN_CHUNK_SIZE_TOKENS)
    
    Returns:
        List of merged chunks
    """
    if min_tokens is None:
        min_tokens = MIN_CHUNK_SIZE_TOKENS
    
    if not chunks:
        return chunks
    
    merged = []
    i = 0
    
    while i < len(chunks):
        current = chunks[i]
        current_tokens = count_tokens(current.chunk_text)
        
        # If chunk is too small, try to merge with next chunk
        if current_tokens < min_tokens and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            combined_text = current.chunk_text + '\n\n' + next_chunk.chunk_text
            combined_tokens = count_tokens(combined_text)
            
            # Only merge if combined chunk is still within reasonable size
            if combined_tokens <= settings.chunk_size * 1.2:
                # Create merged chunk
                merged_chunk = CodeChunk(
                    chunk_id=current.chunk_id,  # Keep first chunk's ID
                    repo_id=current.repo_id,
                    file_path=current.file_path,
                    language=current.language,
                    start_line=current.start_line,
                    end_line=next_chunk.end_line,
                    symbol_name=current.symbol_name or next_chunk.symbol_name,
                    chunk_text=combined_text,
                    metadata={
                        **current.metadata,
                        'merged': True,
                        'merged_chunks': 2,
                        'original_symbols': [current.symbol_name, next_chunk.symbol_name]
                    }
                )
                merged.append(merged_chunk)
                i += 2  # Skip both chunks
                continue
        
        # Keep chunk as-is
        merged.append(current)
        i += 1
    
    return merged


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
