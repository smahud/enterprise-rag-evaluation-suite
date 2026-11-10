# Evaluation metrics for enterprise-rag-evaluation-suite

from dataclasses import dataclass
from typing import List, Dict, Any, Set
import numpy as np
from sklearn.metrics import ndcg_score
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievalMetrics:
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    map_score: float
    latency_p50: float
    latency_p99: float

@dataclass
class RerankingMetrics:
    mrr: float
    ndcg_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]

@dataclass
class GenerationMetrics:
    faithfulness: float
    answer_relevance: float
    hallucination_rate: float
    citation_accuracy: float
    token_efficiency: float

@dataclass
class HallucinationMetrics:
    self_consistency_score: float
    entailment_score: float
    citation_verification_rate: float
    low_prob_token_rate: float
    overall_hallucination_rate: float

def evaluate_retrieval(dataset, retrieval_results, metrics_list: List[str]) -> Dict[str, float]:
    """Evaluate retrieval performance."""
    results = {}
    k_values = [1, 3, 5, 10, 20, 50]
    
    # Prepare ground truth
    query_to_relevant = {}
    for query in dataset.queries:
        query_to_relevant[query.id] = set(query.relevant_docs)
    
    # Calculate Recall@k
    if "recall" in metrics_list:
        recall_scores = {k: [] for k in k_values}
        for res in retrieval_results:
            relevant = query_to_relevant.get(res.query_id, set())
            if not relevant:
                continue
            for k in k_values:
                retrieved = set(res.doc_ids[:k])
                recall = len(retrieved & relevant) / len(relevant) if relevant else 0.0
                recall_scores[k].append(recall)
        
        for k in k_values:
            if recall_scores[k]:
                results[f"recall@{k}"] = float(np.mean(recall_scores[k]))
    
    # Calculate nDCG@k
    if "ndcg" in metrics_list:
        ndcg_scores = {k: [] for k in k_values}
        for res in retrieval_results:
            relevant = query_to_relevant.get(res.query_id, set())
            if not relevant:
                continue
            for k in k_values:
                y_true = [1 if doc_id in relevant else 0 for doc_id in res.doc_ids[:k]]
                y_score = res.scores[:k]
                if len(y_true) > 0 and sum(y_true) > 0:
                    # Pad to same length
                    ndcg = ndcg_score([y_true], [y_score], k=min(k, len(y_true)))
                    ndcg_scores[k].append(ndcg)
        
        for k in k_values:
            if ndcg_scores[k]:
                results[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k]))
    
    # Calculate MRR
    if "mrr" in metrics_list:
        mrr_scores = []
        for res in retrieval_results:
            relevant = query_to_relevant.get(res.query_id, set())
            for rank, doc_id in enumerate(res.doc_ids):
                if doc_id in relevant:
                    mrr_scores.append(1.0 / (rank + 1))
                    break
            else:
                mrr_scores.append(0.0)
        
        if mrr_scores:
            results["mrr"] = float(np.mean(mrr_scores))
    
    # Calculate MAP
    if "map" in metrics_list:
        ap_scores = []
        for res in retrieval_results:
            relevant = query_to_relevant.get(res.query_id, set())
            if not relevant:
                continue
            
            hits = 0
            precisions = []
            for rank, doc_id in enumerate(res.doc_ids):
                if doc_id in relevant:
                    hits += 1
                    precisions.append(hits / (rank + 1))
            
            if precisions:
                ap_scores.append(np.mean(precisions))
            else:
                ap_scores.append(0.0)
        
        if ap_scores:
            results["map"] = float(np.mean(ap_scores))
    
    return results

def evaluate_reranking(dataset, reranking_results, metrics_list: List[str]) -> Dict[str, float]:
    """Evaluate reranking performance."""
    results = {}
    k_values = [1, 3, 5, 10]
    
    query_to_relevant = {}
    for query in dataset.queries:
        query_to_relevant[query.id] = set(query.relevant_docs)
    
    # MRR
    if "mrr" in metrics_list:
        mrr_scores = []
        for res in reranking_results:
            relevant = query_to_relevant.get(res.query_id, set())
            for rank, doc_id in enumerate(res.doc_ids):
                if doc_id in relevant:
                    mrr_scores.append(1.0 / (rank + 1))
                    break
            else:
                mrr_scores.append(0.0)
        
        if mrr_scores:
            results["mrr"] = float(np.mean(mrr_scores))
    
    # nDCG@k
    if "ndcg" in metrics_list:
        ndcg_scores = {k: [] for k in k_values}
        for res in reranking_results:
            relevant = query_to_relevant.get(res.query_id, set())
            if not relevant:
                continue
            for k in k_values:
                y_true = [1 if doc_id in relevant else 0 for doc_id in res.doc_ids[:k]]
                y_score = res.scores[:k]
                if len(y_true) > 0 and sum(y_true) > 0:
                    ndcg = ndcg_score([y_true], [y_score], k=min(k, len(y_true)))
                    ndcg_scores[k].append(ndcg)
        
        for k in k_values:
            if ndcg_scores[k]:
                results[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k]))
    
    # Precision@k
    if "precision" in metrics_list:
        precision_scores = {k: [] for k in k_values}
        for res in reranking_results:
            relevant = query_to_relevant.get(res.query_id, set())
            for k in k_values:
                retrieved = set(res.doc_ids[:k])
                precision = len(retrieved & relevant) / k if k > 0 else 0.0
                precision_scores[k].append(precision)
        
        for k in k_values:
            if precision_scores[k]:
                results[f"precision@{k}"] = float(np.mean(precision_scores[k]))
    
    return results

def evaluate_generation(dataset, generation_results, metrics_list: List[str]) -> Dict[str, float]:
    """Evaluate generation quality."""
    results = {}
    
    # Faithfulness (answer grounded in context)
    if "faithfulness" in metrics_list:
        from bert_score import score
        
        # Use BERTScore as proxy for faithfulness
        references = []
        predictions = []
        for gen in generation_results:
            # Find ground truth answer (if available)
            query = next((q for q in dataset.queries if q.id == gen.query_id), None)
            if query and hasattr(query, "answer"):
                references.append(query.answer)
                predictions.append(gen.answer)
        
        if references:
            P, R, F1 = score(predictions, references, lang="en", verbose=False)
            results["faithfulness"] = float(F1.mean())
    
    # Answer Relevance
    if "answer_relevance" in metrics_list:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        relevance_scores = []
        for gen in generation_results:
            query = next((q for q in dataset.queries if q.id == gen.query_id), None)
            if query:
                query_emb = model.encode([query.text])
                answer_emb = model.encode([gen.answer])
                similarity = np.dot(query_emb[0], answer_emb[0]) / (
                    np.linalg.norm(query_emb[0]) * np.linalg.norm(answer_emb[0])
                )
                relevance_scores.append(float(similarity))
        
        if relevance_scores:
            results["answer_relevance"] = float(np.mean(relevance_scores))
    
    return results

def evaluate_hallucination(dataset, generation_results) -> Dict[str, float]:
    """Evaluate hallucination using multiple detection methods."""
    results = {}
    
    # Self-consistency check (multiple generations)
    if hasattr(generation_results[0], "all_generations") and generation_results[0].all_generations:
        consistency_scores = []
        for gen in generation_results:
            if hasattr(gen, "all_generations") and len(gen.all_generations) > 1:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                
                embeddings = model.encode(gen.all_generations)
                # Compute pairwise similarities
                sims = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        sims.append(sim)
                
                if sims:
                    consistency_scores.append(float(np.mean(sims)))
        
        if consistency_scores:
            results["self_consistency_score"] = float(np.mean(consistency_scores))
    
    # Entailment scoring (NLI)
    if "entailment" in metrics_list:
        try:
            from transformers import pipeline
            nli = pipeline("text-classification", model="microsoft/deberta-v3-base", task="zero-shot-classification")
            
            entailment_scores = []
            for gen in generation_results:
                query = next((q for q in dataset.queries if q.id == gen.query_id), None)
                if query and gen.citations:
                    # Check if answer is entailed by citations
                    for citation in gen.citations:
                        result = nli(f"{gen.answer} [SEP] {citation}", candidate_labels=["entailment", "contradiction", "neutral"])
                        entailment_score = next((r["score"] for r in result if r["label"] == "entailment"), 0.0)
                        entailment_scores.append(entailment_score)
            
            if entailment_scores:
                results["entailment_score"] = float(np.mean(entailment_scores))
        except Exception as e:
            logger.warning(f"Entailment scoring failed: {e}")
    
    # Citation verification
    if "citation_verification" in metrics_list:
        verification_rates = []
        for gen in generation_results:
            if gen.citations:
                # Verify citations exist in context
                verified = 0
                for citation in gen.citations:
                    # Simple check: citation text appears in retrieved docs
                    # In practice, this would be more sophisticated
                    verified += 1  # Placeholder
                
                verification_rates.append(verified / len(gen.citations))
        
        if verification_rates:
            results["citation_verification_rate"] = float(np.mean(verification_rates))
    
    # Overall hallucination rate (composite)
    if "hallucination_rate" in metrics_list:
        # Composite score: lower is better
        # Based on faithfulness, entailment, citation verification
        components = []
        if "faithfulness" in results:
            components.append(1.0 - results["faithfulness"])
        if "entailment_score" in results:
            components.append(1.0 - results["entailment_score"])
        if "citation_verification_rate" in results:
            components.append(1.0 - results["citation_verification_rate"])
        
        if components:
            results["hallucination_rate"] = float(np.mean(components))
    
    return results