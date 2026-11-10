# Retrieval engines for enterprise-rag-evaluation-suite

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    query_id: str
    doc_ids: List[str]
    scores: List[float]

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, queries: List[str], top_k: int) -> List[RetrievalResult]:
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        pass

class DenseRetriever(BaseRetriever):
    """Dense vector retrieval using sentence transformers."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.index = None
        self.doc_ids = []
        self._initialize()
    
    def _initialize(self):
        from sentence_transformers import SentenceTransformer
        import faiss
        
        logger.info(f"Loading dense retriever model: {self.config.embedder}")
        self.model = SentenceTransformer(self.config.embedder)
        
        # Initialize FAISS index
        dimension = self.model.get_sentence_embedding_dimension()
        if self.config.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efSearch = self.config.ef_search
        else:
            self.index = faiss.IndexFlatIP(dimension)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        texts = [doc["text"] for doc in documents]
        self.doc_ids = [doc["id"] for doc in documents]
        
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.model.encode(texts, batch_size=self.config.batch_size, show_progress_bar=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        logger.info(f"Added {len(documents)} documents to index")
    
    def retrieve(self, queries: List[str], top_k: int) -> List[RetrievalResult]:
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning empty results")
            return [RetrievalResult(qid, [], []) for qid in range(len(queries))]
        
        logger.info(f"Retrieving top-{top_k} for {len(queries)} queries...")
        query_embeddings = self.model.encode(queries, batch_size=self.config.batch_size, show_progress_bar=True)
        faiss.normalize_L2(query_embeddings)
        
        scores, indices = self.index.search(query_embeddings, top_k)
        
        results = []
        for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            doc_ids = [self.doc_ids[idx] for idx in query_indices if idx < len(self.doc_ids)]
            doc_scores = [float(score) for score in query_scores[:len(doc_ids)]]
            results.append(RetrievalResult(
                query_id=str(i),
                doc_ids=doc_ids,
                scores=doc_scores
            ))
        
        return results

class SparseRetriever(BaseRetriever):
    """Sparse retrieval using BM25."""
    
    def __init__(self, config):
        self.config = config
        self.bm25 = None
        self.doc_ids = []
        self.doc_texts = []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi
        
        self.doc_ids = [doc["id"] for doc in documents]
        self.doc_texts = [doc["text"] for doc in documents]
        
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        tokenized_corpus = [text.split() for text in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built")
    
    def retrieve(self, queries: List[str], top_k: int) -> List[RetrievalResult]:
        if self.bm25 is None:
            logger.warning("BM25 index not built, returning empty results")
            return [RetrievalResult(str(i), [], []) for i in range(len(queries))]
        
        results = []
        for i, query in enumerate(queries):
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            doc_ids = [self.doc_ids[idx] for idx in top_indices]
            doc_scores = [float(scores[idx]) for idx in top_indices]
            
            results.append(RetrievalResult(
                query_id=str(i),
                doc_ids=doc_ids,
                scores=doc_scores
            ))
        
        return results

class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining dense and sparse with RRF."""
    
    def __init__(self, config):
        self.config = config
        self.dense_retriever = DenseRetriever(config)
        self.sparse_retriever = SparseRetriever(config)
        self.rrf_k = 60  # RRF constant
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)
    
    def retrieve(self, queries: List[str], top_k: int) -> List[RetrievalResult]:
        # Get results from both retrievers (retrieve more for fusion)
        dense_results = self.dense_retriever.retrieve(queries, top_k * 2)
        sparse_results = self.sparse_retriever.retrieve(queries, top_k * 2)
        
        # Reciprocal Rank Fusion
        fused_results = []
        for dense_res, sparse_res in zip(dense_results, sparse_results):
            doc_scores = {}
            
            # Add dense scores with RRF
            for rank, (doc_id, score) in enumerate(zip(dense_res.doc_ids, dense_res.scores)):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            
            # Add sparse scores with RRF
            for rank, (doc_id, score) in enumerate(zip(sparse_res.doc_ids, sparse_res.scores)):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            
            # Sort by fused score
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            fused_results.append(RetrievalResult(
                query_id=dense_res.query_id,
                doc_ids=[doc_id for doc_id, _ in sorted_docs],
                scores=[score for _, score in sorted_docs]
            ))
        
        return fused_results

def get_retriever(config) -> BaseRetriever:
    """Factory function to get retriever based on config."""
    if config.index_type == "bm25":
        return SparseRetriever(config)
    elif config.index_type == "hybrid":
        return HybridRetriever(config)
    else:
        return DenseRetriever(config)