import os
import sys
import hashlib
import logging
import uuid
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    OptimizersConfigDiff, HnswConfigDiff
)
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

log_file = Path('/app/logs/indexer.log')
if log_file.parent.exists():
    logging.getLogger().addHandler(logging.FileHandler(log_file))

logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
OPENCV_REPO_PATH = os.getenv('OPENCV_REPO_PATH', '/app/opencv_repo')
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'opencv_codebase')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')

FILE_EXTENSIONS = {'.cpp', '.hpp', '.h', '.c', '.cc', '.cxx', '.py', '.cu', '.cuh'}
SKIP_DIRS = {
    'build', '.git', '3rdparty', 'cmake', '__pycache__',
    'node_modules', '.vscode', '.idea', 'doc', 'data',
    'samples', 'test', 'tests', 'perf', 'android', 'ios',
    'platforms', '.cache', 'apps', 'java', 'js', 'objc'
}

MIN_CHUNK_LINES = 8
MAX_CHUNK_SIZE = 300
CONTEXT_LINES = 8
BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 32


@dataclass
class CodeChunk:
    id: str
    code: str
    chunk_type: str
    start_line: int
    end_line: int
    file_path: str
    module: str
    language: str
    complexity: str
    file_name: str


def detect_language(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    mapping = {
        '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
        '.hpp': 'cpp', '.h': 'cpp', '.c': 'cpp',
        '.cu': 'cpp', '.cuh': 'cpp',
        '.py': 'python'
    }
    return mapping.get(ext, 'unknown')


def extract_module(file_path: str, repo_root: str) -> str:
    try:
        rel_path = str(Path(file_path).relative_to(repo_root))
    except ValueError:
        rel_path = file_path

    parts = rel_path.replace('\\', '/').split('/')

    if 'modules' in parts:
        idx = parts.index('modules')
        if idx + 1 < len(parts):
            return parts[idx + 1]

    for known in ['core', 'imgproc', 'highgui', 'video', 'calib3d', 'features2d',
                  'objdetect', 'dnn', 'ml', 'flann', 'photo', 'stitching', 'videoio',
                  'imgcodecs', 'gapi', 'python', 'ts', 'world', 'cudaarithm', 'cudabgsegm',
                  'cudacodec', 'cudafeatures2d', 'cudafilters', 'cudaimgproc', 'cudalegacy',
                  'cudaobjdetect', 'cudaoptflow', 'cudastereo', 'cudawarping', 'cudev']:
        if known in parts:
            return known

    return 'core'


def calculate_complexity(code: str) -> str:
    lines = [l for l in code.split('\n')
             if l.strip() and not l.strip().startswith(('//', '#', '/*', '*', '///'))]
    line_count = len(lines)

    keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch',
                'throw', 'CV_Assert', 'CV_Error', 'CV_Check', 'parallel_for_']

    indicator_count = 0
    for line in lines:
        line_lower = f' {line} '
        for kw in keywords:
            if f' {kw} ' in line_lower or f' {kw}(' in line_lower or f' {kw}:' in line_lower:
                indicator_count += 1
                break

    if line_count < 15:
        return 'low'
    elif line_count < 60:
        return 'high' if indicator_count > line_count * 0.3 else 'medium'
    else:
        return 'high' if indicator_count > line_count * 0.2 else 'medium'


def generate_point_id(s: str) -> str:
    return str(uuid.UUID(hashlib.md5(s.encode()).hexdigest()))


class ASTChunker:
    def __init__(self):
        self._parsers: Dict = {}
        self._init_parsers()

    def _init_parsers(self):
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_cpp as tscpp
            import tree_sitter_python as tspy

            cpp_lang = Language(tscpp.language())
            py_lang = Language(tspy.language())

            self._parsers['cpp'] = Parser(cpp_lang)
            self._parsers['python'] = Parser(py_lang)

            logger.info("Tree-sitter parsers initialized")
        except ImportError as e:
            logger.warning(f"Tree-sitter not available: {e}")
        except Exception as e:
            logger.warning(f"Tree-sitter init failed: {e}")

    def chunk(self, source_code: str, language: str, file_path: str) -> List[CodeChunk]:
        if language in self._parsers and len(source_code) < 500000:
            try:
                chunks = self._ast_chunk(source_code, language, file_path)
                if chunks:
                    return chunks
            except Exception as e:
                logger.debug(f"AST chunking failed for {file_path}: {e}")

        return self._smart_chunk(source_code, file_path, language)

    def _ast_chunk(self, source_code: str, language: str, file_path: str) -> List[CodeChunk]:
        parser = self._parsers[language]
        source_bytes = source_code.encode('utf-8')
        tree = parser.parse(source_bytes)

        if language == 'cpp':
            terminal_types = {
                'function_definition',
                'class_specifier',
                'struct_specifier',
                'namespace_definition',
                'template_declaration',
                'enum_specifier'
            }
        else:
            terminal_types = {
                'function_definition',
                'class_definition',
                'decorated_definition'
            }

        chunks = []
        processed_ranges: Set[Tuple[int, int]] = set()
        lines = source_code.split('\n')
        total_lines = len(lines)

        def should_skip(start: int, end: int) -> bool:
            for ps, pe in processed_ranges:
                overlap = max(0, min(end, pe) - max(start, ps))
                size = end - start
                if size > 0 and overlap / size > 0.6:
                    return True
            return False

        def add_chunk(node, start: int, end: int):
            if should_skip(start, end):
                return

            chunk_lines = lines[start:end]
            code_text = '\n'.join(chunk_lines)

            non_empty = sum(1 for l in chunk_lines if l.strip())
            if non_empty < MIN_CHUNK_LINES:
                return

            processed_ranges.add((start, end))

            chunk_id = generate_point_id(f"{file_path}:{start}:{end}")

            chunks.append(CodeChunk(
                id=chunk_id,
                code=code_text,
                chunk_type=node.type,
                start_line=start + 1,
                end_line=end,
                file_path=file_path,
                module='',
                language=language,
                complexity='',
                file_name=Path(file_path).name
            ))

        def traverse(node, depth=0):
            if depth > 50:
                return

            if node.type in terminal_types:
                node_start = node.start_point[0]
                node_end = node.end_point[0] + 1

                start = max(0, node_start - CONTEXT_LINES)
                end = min(total_lines, node_end + CONTEXT_LINES)

                if end - start <= MAX_CHUNK_SIZE:
                    add_chunk(node, start, end)
                else:
                    add_chunk(node, node_start, min(node_start + MAX_CHUNK_SIZE, node_end))
                return

            for child in node.children:
                traverse(child, depth + 1)

        traverse(tree.root_node)

        return chunks

    def _smart_chunk(self, source_code: str, file_path: str, language: str) -> List[CodeChunk]:
        lines = source_code.split('\n')
        total_lines = len(lines)
        chunks = []

        boundaries = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            if language == 'cpp':
                if any(stripped.startswith(p) for p in ['class ', 'struct ', 'namespace ', 'template']):
                    boundaries.append((i, 'class'))
                elif '(' in stripped and '{' in stripped and not stripped.startswith(('if', 'for', 'while', 'switch')):
                    if any(t in stripped for t in ['void ', 'int ', 'bool ', 'float ', 'double ', 'Mat ', 'cv::']):
                        boundaries.append((i, 'function'))
                elif stripped.startswith('CV_EXPORTS') or stripped.startswith('CV_IMPL'):
                    boundaries.append((i, 'function'))
            elif language == 'python':
                if stripped.startswith('def ') or stripped.startswith('class '):
                    boundaries.append((i, 'function' if stripped.startswith('def') else 'class'))
                elif stripped.startswith('@'):
                    boundaries.append((i, 'decorator'))

        if boundaries:
            for idx, (start_line, chunk_type) in enumerate(boundaries):
                if idx + 1 < len(boundaries):
                    end_line = boundaries[idx + 1][0]
                else:
                    end_line = total_lines

                actual_start = max(0, start_line - CONTEXT_LINES)
                actual_end = min(total_lines, end_line + 2)

                if actual_end - actual_start > MAX_CHUNK_SIZE:
                    actual_end = actual_start + MAX_CHUNK_SIZE

                chunk_lines = lines[actual_start:actual_end]
                code_text = '\n'.join(chunk_lines)

                non_empty = sum(1 for l in chunk_lines if l.strip())
                if non_empty >= MIN_CHUNK_LINES:
                    chunk_id = generate_point_id(f"{file_path}:smart:{idx}")
                    chunks.append(CodeChunk(
                        id=chunk_id,
                        code=code_text,
                        chunk_type=chunk_type,
                        start_line=actual_start + 1,
                        end_line=actual_end,
                        file_path=file_path,
                        module='',
                        language=language,
                        complexity='',
                        file_name=Path(file_path).name
                    ))

        if not chunks:
            return self._fixed_chunk(source_code, file_path, language)

        return chunks

    def _fixed_chunk(self, source_code: str, file_path: str, language: str) -> List[CodeChunk]:
        lines = source_code.split('\n')
        total_lines = len(lines)
        chunks = []

        chunk_size = 80
        overlap = 15
        start = 0
        chunk_num = 0

        while start < total_lines:
            end = min(start + chunk_size, total_lines)
            chunk_lines = lines[start:end]
            code_text = '\n'.join(chunk_lines)

            non_empty = sum(1 for l in chunk_lines if l.strip())

            if non_empty >= MIN_CHUNK_LINES:
                chunk_id = generate_point_id(f"{file_path}:fixed:{chunk_num}")
                chunks.append(CodeChunk(
                    id=chunk_id,
                    code=code_text,
                    chunk_type='code_block',
                    start_line=start + 1,
                    end_line=end,
                    file_path=file_path,
                    module='',
                    language=language,
                    complexity='',
                    file_name=Path(file_path).name
                ))
                chunk_num += 1

            if end >= total_lines:
                break
            start = end - overlap

        return chunks


class OpenCVIndexer:
    def __init__(self):
        logger.info("=" * 70)
        logger.info("OpenCV Codebase Indexer v3.0 (Fixed)")
        logger.info("=" * 70)

        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            timeout=120,
            prefer_grpc=False,
            https=False
        )

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder='/app/cache/embeddings'
        )
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        self.chunker = ASTChunker()

        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'chunks_created': 0,
            'total_bytes': 0
        }

    def _setup_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == COLLECTION_NAME for c in collections)

            if exists:
                logger.info(f"Deleting existing collection: {COLLECTION_NAME}")
                self.client.delete_collection(COLLECTION_NAME)
                time.sleep(2)

            logger.info(f"Creating collection: {COLLECTION_NAME}")
            
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=False
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                    max_indexing_threads=0,
                    on_disk=False
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=0,
                    max_optimization_threads=2
                ),
                on_disk_payload=False
            )
            
            logger.info("Collection created with:")
            logger.info("  - indexing_threshold=0 (build index immediately)")
            logger.info("  - max_optimization_threads=2 (enable optimization)")
            logger.info("  - on_disk=False (keep in RAM for speed)")
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise

    def _collect_files(self, repo_path: Path) -> List[Path]:
        logger.info(f"Scanning repository: {repo_path}")
        files = []

        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

            for filename in filenames:
                ext = Path(filename).suffix.lower()
                if ext in FILE_EXTENSIONS:
                    files.append(Path(root) / filename)

        logger.info(f"Found {len(files)} source files")
        return files

    def _process_file(self, file_path: Path, repo_root: Path) -> List[CodeChunk]:
        try:
            stat = file_path.stat()
            if stat.st_size > 1024 * 1024:
                logger.debug(f"Skipping large file: {file_path} ({stat.st_size} bytes)")
                self.stats['files_skipped'] += 1
                return []

            content = file_path.read_text(encoding='utf-8', errors='ignore')

            if not content.strip() or len(content) < 100:
                self.stats['files_skipped'] += 1
                return []

            language = detect_language(file_path)

            try:
                rel_path = str(file_path.relative_to(repo_root))
            except ValueError:
                rel_path = str(file_path)

            module = extract_module(rel_path, str(repo_root))

            chunks = self.chunker.chunk(content, language, rel_path)

            for chunk in chunks:
                chunk.module = module
                chunk.complexity = calculate_complexity(chunk.code)

            self.stats['files_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['total_bytes'] += len(content)

            return chunks

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['files_skipped'] += 1
            return []

    def _create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def _wait_for_indexing(self, timeout: int = 300) -> bool:
        logger.info("Waiting for HNSW indexing to complete...")
        start_time = time.time()
        last_indexed = 0
        stall_count = 0
        
        while time.time() - start_time < timeout:
            try:
                info = self.client.get_collection(COLLECTION_NAME)
                indexed = info.indexed_vectors_count or 0
                total = info.points_count or 0
                status = info.status
                
                if indexed >= total and total > 0:
                    logger.info(f"Indexing complete: {indexed}/{total} vectors indexed")
                    return True
                
                if indexed == last_indexed:
                    stall_count += 1
                else:
                    stall_count = 0
                    last_indexed = indexed
                
                logger.info(f"Indexing progress: {indexed}/{total} (status: {status})")
                
                if stall_count >= 12:
                    logger.warning("Indexing appears stalled, attempting to trigger optimization...")
                    try:
                        self.client.update_collection(
                            collection_name=COLLECTION_NAME,
                            optimizers_config=OptimizersConfigDiff(
                                indexing_threshold=0,
                                max_optimization_threads=4
                            )
                        )
                        stall_count = 0
                    except Exception as e:
                        logger.warning(f"Failed to update collection: {e}")
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error checking indexing status: {e}")
                time.sleep(5)
        
        logger.warning(f"Indexing timeout after {timeout}s")
        return False

    def _test_search(self) -> bool:
        logger.info("Testing vector search...")
        try:
            test_vector = [0.1] * self.embedding_dim
            
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=test_vector,
                limit=5,
                with_payload=True
            )
            
            if results:
                logger.info(f"Search test PASSED: Retrieved {len(results)} results")
                logger.info(f"  Top result score: {results[0].score:.4f}")
                logger.info(f"  Top result file: {results[0].payload.get('file_path', 'unknown')}")
                return True
            else:
                logger.error("Search test FAILED: No results returned")
                return False
                
        except Exception as e:
            logger.error(f"Search test FAILED with error: {e}")
            return False

    def index(self, repo_path: str):
        repo_path = Path(repo_path)

        if not repo_path.exists():
            logger.error(f"Repository not found: {repo_path}")
            sys.exit(1)

        self._setup_collection()

        files = self._collect_files(repo_path)
        if not files:
            logger.error("No source files found!")
            sys.exit(1)

        logger.info("Processing files...")
        all_chunks: List[CodeChunk] = []

        for file_path in tqdm(files, desc="Parsing files"):
            chunks = self._process_file(file_path, repo_path)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {self.stats['files_processed']} files")

        if not all_chunks:
            logger.error("No chunks created!")
            sys.exit(1)

        logger.info("Generating embeddings...")
        codes = [c.code for c in all_chunks]

        all_embeddings = []
        for i in tqdm(range(0, len(codes), EMBEDDING_BATCH_SIZE), desc="Embedding"):
            batch = codes[i:i + EMBEDDING_BATCH_SIZE]
            embeddings = self._create_embeddings_batch(batch)
            all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated {len(embeddings)} embeddings")

        logger.info(f"Uploading to Qdrant (batch size: {BATCH_SIZE})...")

        for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Uploading"):
            batch = all_chunks[i:i + BATCH_SIZE]
            batch_embeddings = embeddings[i:i + BATCH_SIZE]

            points = [
                PointStruct(
                    id=chunk.id,
                    vector=emb.tolist(),
                    payload={
                        'code': chunk.code,
                        'file_path': chunk.file_path,
                        'file_name': chunk.file_name,
                        'module': chunk.module,
                        'chunk_type': chunk.chunk_type,
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line,
                        'language': chunk.language,
                        'complexity': chunk.complexity
                    }
                )
                for chunk, emb in zip(batch, batch_embeddings)
            ]

            try:
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True
                )
            except Exception as e:
                logger.error(f"Upload error at batch {i // BATCH_SIZE}: {e}")

        indexing_success = self._wait_for_indexing(timeout=300)
        
        search_success = self._test_search()
        
        info = self.client.get_collection(COLLECTION_NAME)

        logger.info("=" * 70)
        logger.info("INDEXING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Chunks created: {self.stats['chunks_created']}")
        logger.info(f"Total bytes: {self.stats['total_bytes']:,}")
        logger.info(f"Points in Qdrant: {info.points_count}")
        logger.info(f"Indexed vectors: {info.indexed_vectors_count}")
        logger.info(f"Collection status: {info.status}")
        logger.info(f"Optimizer status: {info.optimizer_status}")
        logger.info("=" * 70)

        if not indexing_success:
            logger.warning("WARNING: Indexing may not be complete!")
        
        if not search_success:
            logger.error("CRITICAL: Search is not working!")
            sys.exit(1)
        else:
            logger.info("SUCCESS: All vectors indexed and search is working!")


def main():
    try:
        indexer = OpenCVIndexer()
        indexer.index(OPENCV_REPO_PATH)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()