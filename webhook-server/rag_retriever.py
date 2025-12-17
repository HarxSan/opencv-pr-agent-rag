import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import re
import time
from collections import OrderedDict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from config import QdrantConfig, RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    code: str
    file_path: str
    module: str
    chunk_type: str
    start_line: int
    end_line: int
    language: str
    complexity: str
    relevance_score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LRUCache:
    def __init__(self, max_size: int = 500):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[List[CodeChunk]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, value: List[CodeChunk]):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value
    
    def clear(self):
        self._cache.clear()


class RAGRetriever:
    def __init__(self, qdrant_config: QdrantConfig, rag_config: RAGConfig):
        self.qdrant_config = qdrant_config
        self.rag_config = rag_config
        self._client: Optional[QdrantClient] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._cache = LRUCache(max_size=500)
        self._collection_verified = False  # Track if collection has been verified
        
        self._opencv_keywords = {
            'mat', 'umat', 'inputarray', 'outputarray', 'scalar',
            'point', 'rect', 'size', 'vec', 'matx', 'ptr',
            'cuda', 'gpu', 'ocl', 'opencl', 'simd', 'parallel',
            'filter', 'blur', 'edge', 'morph', 'threshold',
            'contour', 'feature', 'descriptor', 'match', 'transform',
            'video', 'capture', 'codec', 'frame', 'stream',
            'dnn', 'net', 'layer', 'blob', 'inference',
            'calib', 'stereo', 'disparity', 'homography', 'fundamental'
        }
    
    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            logger.info(f"Connecting to Qdrant at {self.qdrant_config.host}:{self.qdrant_config.port}")
            self._client = QdrantClient(
                host=self.qdrant_config.host,
                port=self.qdrant_config.port,
                api_key=self.qdrant_config.api_key,
                timeout=30,
                prefer_grpc=False,
                https=False
            )
            # Don't verify on initialization - do it lazily on first retrieval
        return self._client
    
    def _verify_collection_with_retry(self, max_retries: int = 3, retry_delay: int = 2) -> bool:
        """
        Verify collection exists and is ready, with retry logic.
        This handles the case where Qdrant is starting up and collection is still loading.
        """
        if self._collection_verified:
            return True
            
        for attempt in range(max_retries):
            try:
                collections = self.client.get_collections().collections
                collection_exists = any(c.name == self.qdrant_config.collection_name for c in collections)
                
                if not collection_exists:
                    logger.warning(
                        f"Collection '{self.qdrant_config.collection_name}' not found "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return False
                
                # Collection exists, check if it has points
                info = self.client.get_collection(self.qdrant_config.collection_name)
                points_count = info.points_count or 0
                
                if points_count == 0:
                    logger.warning(
                        f"Collection '{self.qdrant_config.collection_name}' exists but has 0 points "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return False
                
                logger.info(
                    f"Connected to collection '{self.qdrant_config.collection_name}' "
                    f"with {points_count} indexed chunks"
                )
                self._collection_verified = True
                return True
                
            except Exception as e:
                logger.warning(
                    f"Failed to verify Qdrant collection (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to verify collection after {max_retries} attempts")
                    return False
        
        return False
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.rag_config.embedding_model}")
            self._embedding_model = SentenceTransformer(
                self.rag_config.embedding_model,
                cache_folder=self.rag_config.cache_dir
            )
            logger.info("Embedding model loaded successfully")
        return self._embedding_model
    
    def _generate_cache_key(self, queries: List[str], module: Optional[str]) -> str:
        content = f"{','.join(sorted(queries))}:{module or 'all'}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _embed_queries(self, queries: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16
        )
        return [emb.tolist() for emb in embeddings]
    
    def _extract_search_queries(self, pr_title: str, pr_description: str,
                                changed_files: List[str], diff: str) -> List[str]:
        queries = []
        
        if pr_title:
            clean_title = re.sub(r'^\[.*?\]\s*', '', pr_title)
            clean_title = re.sub(r'^(fix|add|update|remove|refactor|improve|feature|bug|chore):\s*', '', clean_title, flags=re.IGNORECASE)
            if len(clean_title) > 5:
                queries.append(clean_title.lower().strip())
        
        module = self._extract_module(changed_files)
        if module and module != 'core':
            queries.append(f"{module} module implementation")
        
        keywords = self._extract_keywords(f"{pr_title} {pr_description}")
        opencv_kw = [kw for kw in keywords if kw.lower() in self._opencv_keywords]
        if opencv_kw:
            queries.append(" ".join(opencv_kw[:4]))
        
        patterns = self._extract_code_patterns(diff)
        for pattern in patterns[:3]:
            if pattern not in queries:
                queries.append(pattern)
        
        for file_path in changed_files[:5]:
            if any(file_path.endswith(ext) for ext in ['.cpp', '.hpp', '.h', '.c']):
                filename = file_path.split('/')[-1]
                name = re.sub(r'\.(cpp|hpp|h|c|cc)$', '', filename)
                name = re.sub(r'_impl$', '', name)
                if len(name) > 3 and name not in ['main', 'test', 'utils', 'common']:
                    queries.append(f"{name} implementation")
        
        unique_queries = list(dict.fromkeys(q for q in queries if q and len(q) > 3))
        final_queries = unique_queries[:8]
        
        logger.debug(f"Generated queries: {final_queries}")
        return final_queries
    
    def _extract_module(self, files: List[str]) -> Optional[str]:
        module_counts: Dict[str, int] = {}
        
        for file_path in files:
            if '/modules/' in file_path:
                parts = file_path.split('/modules/')
                if len(parts) > 1:
                    module = parts[1].split('/')[0]
                    module_counts[module] = module_counts.get(module, 0) + 1
        
        if module_counts:
            return max(module_counts, key=module_counts.get)
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'with', 'from', 'by', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'fix', 'add', 'update', 'change', 'remove', 'new',
            'bug', 'issue', 'error', 'problem', 'code', 'file', 'function',
            'class', 'method', 'variable', 'parameter', 'return', 'value',
            'pr', 'pull', 'request', 'merge', 'branch', 'commit'
        }
        
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        
        return list(dict.fromkeys(keywords))
    
    def _extract_code_patterns(self, diff: str) -> List[str]:
        patterns = []
        
        cv_classes = re.findall(r'cv::(\w+)', diff)
        if cv_classes:
            unique_cv = list(dict.fromkeys(cv_classes))[:4]
            for cls in unique_cv:
                if len(cls) > 2 and cls[0].isupper():
                    patterns.append(f"cv::{cls} usage")
        
        cv_funcs = re.findall(r'cv::(\w+)\s*\(', diff)
        if cv_funcs:
            unique_funcs = list(dict.fromkeys(cv_funcs))[:3]
            for func in unique_funcs:
                if len(func) > 3 and func[0].islower():
                    patterns.append(f"{func} function")
        
        macros = re.findall(r'(CV_\w+)', diff)
        if macros:
            unique_macros = list(dict.fromkeys(macros))[:2]
            for macro in unique_macros:
                patterns.append(f"{macro} macro")
        
        func_defs = re.findall(r'(?:void|int|bool|float|double|Mat|UMat|cv::\w+)\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:CV_\w+)?\s*[{;]', diff)
        if func_defs:
            for func in func_defs[:2]:
                if len(func) > 3 and func not in ['main', 'test']:
                    patterns.append(f"{func} implementation")
        
        return list(dict.fromkeys(patterns))
    
    def _calculate_rerank_score(self, query: str, chunk: CodeChunk, query_terms: set) -> float:
        score = chunk.relevance_score
        
        code_lower = chunk.code.lower()
        code_terms = set(self._extract_keywords(chunk.code))
        
        if query_terms and code_terms:
            overlap = len(query_terms & code_terms) / max(len(query_terms), 1)
            score += overlap * 0.12
        
        if chunk.complexity == 'medium':
            score += 0.08
        elif chunk.complexity == 'high':
            score += 0.04
        
        opencv_patterns = [
            ('cv::Mat', 0.03), ('CV_Assert', 0.02), ('InputArray', 0.03),
            ('OutputArray', 0.03), ('@brief', 0.02), ('CV_EXPORTS', 0.02),
            ('parallel_for_', 0.03), ('AutoBuffer', 0.02), ('UMat', 0.03)
        ]
        for pattern, boost in opencv_patterns:
            if pattern in chunk.code:
                score += boost
        
        if chunk.chunk_type in ['function_definition', 'class_specifier']:
            score += 0.05
        
        return min(score, 1.0)
    
    def _rerank_results(self, primary_query: str, results: List[CodeChunk]) -> List[CodeChunk]:
        if not results:
            return results
        
        query_terms = set(self._extract_keywords(primary_query))
        
        for chunk in results:
            chunk.relevance_score = self._calculate_rerank_score(
                primary_query, chunk, query_terms
            )
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def retrieve(self, pr_title: str, pr_description: str,
                 changed_files: List[str], diff: str) -> List[CodeChunk]:
        """
        Retrieve relevant code chunks with proper error handling for Qdrant unavailability.
        """
        # Verify collection is ready (with retry logic)
        if not self._verify_collection_with_retry():
            logger.error("RAG retrieval skipped: Qdrant collection not available")
            return []
        
        queries = self._extract_search_queries(pr_title, pr_description, changed_files, diff)
        if not queries:
            logger.warning("No search queries generated from PR context")
            return []
        
        module_filter = self._extract_module(changed_files)
        cache_key = self._generate_cache_key(queries, module_filter)
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for RAG query")
            return cached
        
        logger.info(f"Retrieving context with {len(queries)} queries, module={module_filter}")
        
        all_results: Dict[str, CodeChunk] = {}
        
        try:
            query_embeddings = self._embed_queries(queries)
        except Exception as e:
            logger.error(f"Failed to embed queries: {e}")
            return []
        
        for query, embedding in zip(queries, query_embeddings):
            try:
                filter_conditions = []
                if module_filter:
                    filter_conditions.append(
                        FieldCondition(key="module", match=MatchValue(value=module_filter))
                    )
                
                query_filter = Filter(must=filter_conditions) if filter_conditions else None
                
                search_results = self.client.search(
                    collection_name=self.qdrant_config.collection_name,
                    query_vector=embedding,
                    query_filter=query_filter,
                    limit=self.rag_config.top_k,
                    with_payload=True,
                    with_vectors=False
                )
                
                for result in search_results:
                    if result.score < self.rag_config.min_score:
                        continue
                    
                    code = result.payload.get('code', '')
                    if not code or len(code) < 50:
                        continue
                    
                    code_hash = hashlib.md5(code.encode()).hexdigest()[:16]
                    
                    if code_hash not in all_results:
                        all_results[code_hash] = CodeChunk(
                            code=code,
                            file_path=result.payload.get('file_path', ''),
                            module=result.payload.get('module', ''),
                            chunk_type=result.payload.get('chunk_type', ''),
                            start_line=result.payload.get('start_line', 0),
                            end_line=result.payload.get('end_line', 0),
                            language=result.payload.get('language', ''),
                            complexity=result.payload.get('complexity', 'unknown'),
                            relevance_score=float(result.score)
                        )
                    else:
                        existing = all_results[code_hash]
                        existing.relevance_score = max(existing.relevance_score, float(result.score))
                        
            except Exception as e:
                logger.error(f"Error searching for query '{query[:50]}...': {e}")
                continue
        
        if not module_filter and len(all_results) < 5:
            logger.info("Expanding search without module filter")
            for query, embedding in zip(queries[:3], query_embeddings[:3]):
                try:
                    results = self.client.search(
                        collection_name=self.qdrant_config.collection_name,
                        query_vector=embedding,
                        limit=self.rag_config.top_k // 2,
                        with_payload=True,
                        with_vectors=False
                    )
                    for r in results:
                        if r.score >= self.rag_config.min_score:
                            code = r.payload.get('code', '')
                            code_hash = hashlib.md5(code.encode()).hexdigest()[:16]
                            if code_hash not in all_results:
                                all_results[code_hash] = CodeChunk(
                                    code=code,
                                    file_path=r.payload.get('file_path', ''),
                                    module=r.payload.get('module', ''),
                                    chunk_type=r.payload.get('chunk_type', ''),
                                    start_line=r.payload.get('start_line', 0),
                                    end_line=r.payload.get('end_line', 0),
                                    language=r.payload.get('language', ''),
                                    complexity=r.payload.get('complexity', 'unknown'),
                                    relevance_score=float(r.score)
                                )
                except Exception as e:
                    logger.error(f"Error in expanded search: {e}")
        
        results = list(all_results.values())
        primary_query = queries[0] if queries else ""
        results = self._rerank_results(primary_query, results)
        
        final_results = results[:self.rag_config.top_k]
        
        self._cache.set(cache_key, final_results)
        
        logger.info(f"Retrieved {len(final_results)} unique context chunks")
        return final_results
    
    def format_context(self, chunks: List[CodeChunk], max_chars: int = None) -> str:
        if not chunks:
            return ""
        
        max_chars = max_chars or self.rag_config.max_context_chars
        
        header = [
            "=" * 70,
            "RELEVANT OPENCV CODEBASE CONTEXT",
            "Use this context to understand existing patterns and conventions.",
            "=" * 70,
            ""
        ]
        
        lines = header.copy()
        total_chars = sum(len(line) for line in lines)
        included_count = 0
        
        for idx, chunk in enumerate(chunks, 1):
            chunk_header = (
                f"--- [{idx}] {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line}) ---\n"
                f"Module: {chunk.module} | Type: {chunk.chunk_type} | "
                f"Complexity: {chunk.complexity} | Score: {chunk.relevance_score:.2f}\n\n"
            )
            
            code = chunk.code
            if len(code) > 1500:
                code = code[:1500] + f"\n... (truncated, {len(chunk.code)} chars)"
            
            chunk_text = chunk_header + code + "\n\n"
            
            if total_chars + len(chunk_text) > max_chars:
                if included_count == 0:
                    code = code[:max_chars - total_chars - len(chunk_header) - 100]
                    chunk_text = chunk_header + code + "\n... (truncated)\n\n"
                    lines.append(chunk_text)
                    included_count += 1
                else:
                    remaining = len(chunks) - included_count
                    if remaining > 0:
                        lines.append(f"... ({remaining} more context chunks omitted for brevity)")
                break
            
            lines.append(chunk_text)
            total_chars += len(chunk_text)
            included_count += 1
        
        result = "\n".join(lines)
        logger.info(f"Formatted context: {included_count} chunks, {len(result)} chars")
        return result
    
    def health_check(self) -> Tuple[bool, str]:
        """Health check with retry logic for startup scenarios"""
        try:
            if not self._verify_collection_with_retry(max_retries=1, retry_delay=1):
                return False, "Collection not available or empty"
            
            info = self.client.get_collection(self.qdrant_config.collection_name)
            return True, f"Qdrant OK: {info.points_count} points indexed"
        except Exception as e:
            return False, f"Qdrant error: {str(e)}"
    
    def clear_cache(self):
        self._cache.clear()
        logger.info("RAG cache cleared")