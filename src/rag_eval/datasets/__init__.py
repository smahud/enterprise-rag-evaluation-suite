# Dataset loading for enterprise-rag-evaluation-suite

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Query:
    id: str
    text: str
    relevant_docs: List[str]  # List of relevant document IDs

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class Dataset:
    name: str
    queries: List[Query]
    corpus: List[Document]
    query_to_relevant: Dict[str, List[str]]  # query_id -> list of relevant doc_ids

def load_dataset(config) -> Dataset:
    """Load dataset based on configuration."""
    name = config.name.lower()
    
    if name == "msmarco":
        return load_msmarco(config)
    elif name == "nq":
        return load_nq(config)
    elif name == "hotpotqa":
        return load_hotpotqa(config)
    elif name == "fever":
        return load_fever(config)
    elif name == "custom":
        return load_custom(config)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def load_msmarco(config) -> Dataset:
    """Load MS MARCO dataset."""
    from datasets import load_dataset as hf_load_dataset
    
    logger.info("Loading MS MARCO dataset...")
    ds = hf_load_dataset("ms_marco", "v1.1", split=config.split, cache_dir=config.cache_dir)
    
    if config.max_samples:
        ds = ds.select(range(min(config.max_samples, len(ds))))
    
    queries = []
    query_to_relevant = {}
    doc_map = {}
    
    for item in ds:
        query_id = item["query_id"]
        query_text = item["query"]
        passages = item["passages"]
        
        query = Query(id=query_id, text=query_text, relevant_docs=[])
        queries.append(query)
        
        relevant_ids = []
        for passage in passages:
            if passage.get("is_selected", False):
                doc_id = passage.get("docid", f"doc_{len(doc_map)}")
                relevant_ids.append(doc_id)
                
                if doc_id not in doc_map:
                    doc_map[doc_id] = Document(
                        id=doc_id,
                        text=passage["passage_text"],
                        metadata={"source": "msmarco"}
                    )
        
        query_to_relevant[query_id] = relevant_ids
        query.relevant_docs = relevant_ids
    
    corpus = list(doc_map.values())
    
    logger.info(f"Loaded MS MARCO: {len(queries)} queries, {len(corpus)} documents")
    return Dataset(name="msmarco", queries=queries, corpus=corpus, query_to_relevant=query_to_relevant)

def load_nq(config) -> Dataset:
    """Load Natural Questions dataset."""
    from datasets import load_dataset as hf_load_dataset
    
    logger.info("Loading Natural Questions dataset...")
    ds = hf_load_dataset("natural_questions", split=config.split, cache_dir=config.cache_dir)
    
    if config.max_samples:
        ds = ds.select(range(min(config.max_samples, len(ds))))
    
    queries = []
    query_to_relevant = {}
    doc_map = {}
    
    for item in ds:
        query_id = item["id"]
        query_text = item["question"]["text"]
        
        query = Query(id=query_id, text=query_text, relevant_docs=[])
        queries.append(query)
        
        # Extract relevant documents from annotations
        relevant_ids = []
        for annotation in item.get("annotations", []):
            for passage in annotation.get("long_answer", []):
                doc_id = f"nq_doc_{len(doc_map)}"
                relevant_ids.append(doc_id)
                
                if doc_id not in doc_map:
                    doc_map[doc_id] = Document(
                        id=doc_id,
                        text=passage.get("text", ""),
                        metadata={"source": "nq"}
                    )
        
        query_to_relevant[query_id] = relevant_ids
        query.relevant_docs = relevant_ids
    
    corpus = list(doc_map.values())
    logger.info(f"Loaded NQ: {len(queries)} queries, {len(corpus)} documents")
    return Dataset(name="nq", queries=queries, corpus=corpus, query_to_relevant=query_to_relevant)

def load_hotpotqa(config) -> Dataset:
    """Load HotpotQA dataset."""
    from datasets import load_dataset as hf_load_dataset
    
    logger.info("Loading HotpotQA dataset...")
    ds = hf_load_dataset("hotpot_qa", "distractor", split=config.split, cache_dir=config.cache_dir)
    
    if config.max_samples:
        ds = ds.select(range(min(config.max_samples, len(ds))))
    
    queries = []
    query_to_relevant = {}
    doc_map = {}
    
    for item in ds:
        query_id = item["id"]
        query_text = item["question"]
        
        query = Query(id=query_id, text=query_text, relevant_docs=[])
        queries.append(query)
        
        relevant_ids = []
        for ctx in item.get("context", []):
            title = ctx[0]
            sentences = ctx[1]
            text = " ".join(sentences)
            
            doc_id = f"hotpot_doc_{len(doc_map)}"
            relevant_ids.append(doc_id)
            
            if doc_id not in doc_map:
                doc_map[doc_id] = Document(
                    id=doc_id,
                    text=text,
                    metadata={"title": title, "source": "hotpotqa"}
                )
        
        query_to_relevant[query_id] = relevant_ids
        query.relevant_docs = relevant_ids
    
    corpus = list(doc_map.values())
    logger.info(f"Loaded HotpotQA: {len(queries)} queries, {len(corpus)} documents")
    return Dataset(name="hotpotqa", queries=queries, corpus=corpus, query_to_relevant=query_to_relevant)

def load_fever(config) -> Dataset:
    """Load FEVER dataset."""
    from datasets import load_dataset as hf_load_dataset
    
    logger.info("Loading FEVER dataset...")
    ds = hf_load_dataset("fever", "v1.0", split=config.split, cache_dir=config.cache_dir)
    
    if config.max_samples:
        ds = ds.select(range(min(config.max_samples, len(ds))))
    
    queries = []
    query_to_relevant = {}
    doc_map = {}
    
    for item in ds:
        query_id = item["id"]
        query_text = item["claim"]
        
        query = Query(id=query_id, text=query_text, relevant_docs=[])
        queries.append(query)
        
        relevant_ids = []
        for evidence in item.get("evidence", []):
            for ev in evidence:
                doc_id = f"fever_doc_{len(doc_map)}"
                relevant_ids.append(doc_id)
                
                if doc_id not in doc_map:
                    doc_map[doc_id] = Document(
                        id=doc_id,
                        text=ev[2],  # evidence text
                        metadata={"page": ev[0], "sentence": ev[1], "source": "fever"}
                    )
        
        query_to_relevant[query_id] = relevant_ids
        query.relevant_docs = relevant_ids
    
    corpus = list(doc_map.values())
    logger.info(f"Loaded FEVER: {len(queries)} queries, {len(corpus)} documents")
    return Dataset(name="fever", queries=queries, corpus=corpus, query_to_relevant=query_to_relevant)

def load_custom(config) -> Dataset:
    """Load custom dataset from JSON files."""
    logger.info(f"Loading custom dataset from {config.cache_dir}...")
    
    queries_path = Path(config.cache_dir) / "queries.json"
    corpus_path = Path(config.cache_dir) / "corpus.json"
    qrels_path = Path(config.cache_dir) / "qrels.json"
    
    if not all(p.exists() for p in [queries_path, corpus_path, qrels_path]):
        raise FileNotFoundError(f"Custom dataset files not found in {config.cache_dir}")
    
    with open(queries_path) as f:
        queries_data = json.load(f)
    
    with open(corpus_path) as f:
        corpus_data = json.load(f)
    
    with open(qrels_path) as f:
        qrels_data = json.load(f)
    
    queries = [Query(**q) for q in queries_data]
    corpus = [Document(**d) for d in corpus_data]
    query_to_relevant = qrels_data
    
    for query in queries:
        query.relevant_docs = query_to_relevant.get(query.id, [])
    
    logger.info(f"Loaded custom dataset: {len(queries)} queries, {len(corpus)} documents")
    return Dataset(name="custom", queries=queries, corpus=corpus, query_to_relevant=query_to_relevant)

def download_dataset(name: str, cache_dir: str):
    """Download and cache a dataset."""
    from datasets import load_dataset as hf_load_dataset
    
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    if name.lower() == "msmarco":
        ds = hf_load_dataset("ms_marco", "v1.1", cache_dir=cache_dir)
        logger.info(f"Downloaded MS MARCO to {cache_dir}")
    elif name.lower() == "nq":
        ds = hf_load_dataset("natural_questions", cache_dir=cache_dir)
        logger.info(f"Downloaded NQ to {cache_dir}")
    elif name.lower() == "hotpotqa":
        ds = hf_load_dataset("hotpot_qa", "distractor", cache_dir=cache_dir)
        logger.info(f"Downloaded HotpotQA to {cache_dir}")
    elif name.lower() == "fever":
        ds = hf_load_dataset("fever", "v1.0", cache_dir=cache_dir)
        logger.info(f"Downloaded FEVER to {cache_dir}")
    else:
        raise ValueError(f"Unknown dataset: {name}")