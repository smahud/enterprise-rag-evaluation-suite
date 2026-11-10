# Reranking strategies for enterprise-rag-evaluation-suite

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class RerankingResult:
    query_id: str
    doc_ids: List[str]
    scores: List[float]

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, queries: List[str], retrieval_results: List[Any], top_k: int) -> List[RerankingResult]:
        pass

class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence transformers."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self._initialize()
    
    def _initialize(self):
        from sentence_transformers import CrossEncoder
        
        logger.info(f"Loading cross-encoder reranker: {self.config.model}")
        self.model = CrossEncoder(self.config.model, max_length=512)
        if self.config.use_fp16:
            self.model.model.half()
    
    def rerank(self, queries: List[str], retrieval_results: List[Any], top_k: int) -> List[RerankingResult]:
        if not retrieval_results:
            return []
        
        logger.info(f"Reranking {len(retrieval_results)} query results with cross-encoder...")
        
        results = []
        for i, (query, retrieval) in enumerate(zip(queries, retrieval_results)):
            if not retrieval.doc_ids:
                results.append(RerankingResult(query_id=retrieval.query_id, doc_ids=[], scores=[]))
                continue
            
            # Prepare pairs for cross-encoder
            pairs = [(query, doc_id) for doc_id in retrieval.doc_ids]
            
            # Get scores
            scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
            
            # Sort by score
            scored_docs = list(zip(retrieval.doc_ids, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k
            top_docs = scored_docs[:top_k]
            
            results.append(RerankingResult(
                query_id=retrieval.query_id,
                doc_ids=[doc_id for doc_id, _ in top_docs],
                scores=[float(score) for _, score in top_docs]
            ))
        
        return results

class LLMBasedReranker(BaseReranker):
    """LLM-based reranker using GPT or local models."""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self._initialize()
    
    def _initialize(self):
        import openai
        
        logger.info(f"Initializing LLM reranker with model: {self.config.model}")
        self.client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base
        )
    
    def rerank(self, queries: List[str], retrieval_results: List[Any], top_k: int) -> List[RerankingResult]:
        if not retrieval_results:
            return []
        
        logger.info(f"Reranking with LLM: {self.config.model}")
        
        results = []
        for query, retrieval in zip(queries, retrieval_results):
            if not retrieval.doc_ids:
                results.append(RerankingResult(query_id=retrieval.query_id, doc_ids=[], scores=[]))
                continue
            
            # Create reranking prompt
            docs_text = "\n\n".join([
                f"Document {j+1}:\n{doc_id}" 
                for j, doc_id in enumerate(retrieval.doc_ids)
            ])
            
            prompt = f"""Query: {query}

Documents:
{docs_text}

Rank the documents by relevance to the query. Return a JSON list of document indices (1-based) in order of relevance, e.g., [3, 1, 4, 2]."""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                
                content = response.choices[0].message.content.strip()
                # Parse JSON list
                import json
                ranking = json.loads(content)
                
                # Convert to 0-based indices
                ranked_doc_ids = [retrieval.doc_ids[idx - 1] for idx in ranking if 1 <= idx <= len(retrieval.doc_ids)]
                
                # Pad with remaining docs if needed
                remaining = [d for d in retrieval.doc_ids if d not in ranked_doc_ids]
                ranked_doc_ids.extend(remaining)
                
                results.append(RerankingResult(
                    query_id=retrieval.query_id,
                    doc_ids=ranked_doc_ids[:top_k],
                    scores=[1.0 - i * 0.1 for i in range(min(top_k, len(ranked_doc_ids)))]
                ))
                
            except Exception as e:
                logger.warning(f"LLM reranking failed for query {retrieval.query_id}: {e}")
                # Fallback to original order
                results.append(RerankingResult(
                    query_id=retrieval.query_id,
                    doc_ids=retrieval.doc_ids[:top_k],
                    scores=[1.0 - i * 0.1 for i in range(min(top_k, len(retrieval.doc_ids)))]
                ))
        
        return results

class HybridReranker(BaseReranker):
    """Hybrid reranker combining cross-encoder with reciprocal rank fusion."""
    
    def __init__(self, config):
        self.config = config
        self.cross_encoder = CrossEncoderReranker(config)
        self.rrf_k = 60
    
    def rerank(self, queries: List[str], retrieval_results: List[Any], top_k: int) -> List[RerankingResult]:
        # Get cross-encoder results
        ce_results = self.cross_encoder.rerank(queries, retrieval_results, top_k * 2)
        
        # Fuse with original retrieval order using RRF
        fused_results = []
        for ce_res, orig_res in zip(ce_results, retrieval_results):
            doc_scores = {}
            
            # Cross-encoder scores (already sorted)
            for rank, (doc_id, score) in enumerate(zip(ce_res.doc_ids, ce_res.scores)):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            
            # Original retrieval order
            for rank, doc_id in enumerate(orig_res.doc_ids):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            
            # Sort by fused score
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            fused_results.append(RerankingResult(
                query_id=orig_res.query_id,
                doc_ids=[doc_id for doc_id, _ in sorted_docs],
                scores=[score for _, score in sorted_docs]
            ))
        
        return fused_results

class AdaptiveReranker(BaseReranker):
    """Adaptive reranker that selects strategy based on query type."""
    
    def __init__(self, config):
        self.config = config
        self.cross_encoder = CrossEncoderReranker(config)
        self.llm_reranker = LLMBasedReranker(config) if config.get("api_key") else None
        self.hybrid = HybridReranker(config)
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for adaptive routing."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["compare", "versus", "vs", "difference", "contrast"]):
            return "comparison"
        elif any(word in query_lower for word in ["how", "why", "explain", "reason", "cause"]):
            return "explanation"
        elif any(word in query_lower for word in ["what is", "define", "definition", "meaning"]):
            return "factual"
        elif any(word in query_lower for word in ["list", "enumerate", "examples", "types"]):
            return "enumeration"
        else:
            return "general"
    
    def rerank(self, queries: List[str], retrieval_results: List[Any], top_k: int) -> List[RerankingResult]:
        results = []
        
        for query, retrieval in zip(queries, retrieval_results):
            query_type = self._classify_query(query)
            
            if query_type == "factual" and self.llm_reranker:
                # Use LLM for factual queries
                result = self.llm_reranker.rerank([query], [retrieval], top_k)[0]
            elif query_type == "explanation":
                # Use hybrid for explanations
                result = self.hybrid.rerank([query], [retrieval], top_k)[0]
            else:
                # Use cross-encoder for general/comparison/enumeration
                result = self.cross_encoder.rerank([query], [retrieval], top_k)[0]
            
            results.append(result)
        
        return results

def get_reranker(config) -> BaseReranker:
    """Factory function to get reranker based on config."""
    strategy = getattr(config, "strategy", "cross_encoder")
    
    if strategy == "llm":
        return LLMBasedReranker(config)
    elif strategy == "hybrid":
        return HybridReranker(config)
    elif strategy == "adaptive":
        return AdaptiveReranker(config)
    else:
        return CrossEncoderReranker(config)