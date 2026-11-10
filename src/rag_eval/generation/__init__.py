# Generation engines for enterprise-rag-evaluation-suite

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    query_id: str
    answer: str
    citations: List[str]
    tokens_used: int
    latency_ms: float

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, queries: List[str], reranking_results: List[Any]) -> List[GenerationResult]:
        pass

class OpenAIGenerator(BaseGenerator):
    """OpenAI-compatible generator (OpenAI, vLLM, etc.)."""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self._initialize()
    
    def _initialize(self):
        import openai
        
        logger.info(f"Initializing OpenAI generator with model: {self.config.model}")
        self.client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base
        )
    
    def generate(self, queries: List[str], reranking_results: List[Any]) -> List[GenerationResult]:
        import time
        
        results = []
        for query, reranking in zip(queries, reranking_results):
            start = time.time()
            
            # Build context from reranked documents
            context_docs = reranking.doc_ids[:5]  # Top 5 for context
            context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(context_docs)])
            
            prompt = f"""Based on the following context, answer the question. Cite sources using [Doc N] format.

Context:
{context}

Question: {query}

Answer:"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                )
                
                answer = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                
                # Extract citations
                import re
                citations = re.findall(r'\[Doc (\d+)\]', answer)
                citations = [context_docs[int(c)-1] for c in citations if int(c) <= len(context_docs)]
                
                results.append(GenerationResult(
                    query_id=reranking.query_id,
                    answer=answer,
                    citations=citations,
                    tokens_used=tokens_used,
                    latency_ms=(time.time() - start) * 1000
                ))
                
            except Exception as e:
                logger.error(f"Generation failed for query {reranking.query_id}: {e}")
                results.append(GenerationResult(
                    query_id=reranking.query_id,
                    answer=f"Error: {str(e)}",
                    citations=[],
                    tokens_used=0,
                    latency_ms=(time.time() - start) * 1000
                ))
        
        return results

class VLLMGenerator(BaseGenerator):
    """Local vLLM generator for open-source models."""
    
    def __init__(self, config):
        self.config = config
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        from vllm import LLM, SamplingParams
        
        logger.info(f"Initializing vLLM generator with model: {self.config.model}")
        self.llm = LLM(
            model=self.config.model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
    
    def generate(self, queries: List[str], reranking_results: List[Any]) -> List[GenerationResult]:
        import time
        
        prompts = []
        for query, reranking in zip(queries, reranking_results):
            context_docs = reranking.doc_ids[:5]
            context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(context_docs)])
            
            prompt = f"""Based on the following context, answer the question. Cite sources using [Doc N] format.

Context:
{context}

Question: {query}

Answer:"""
            prompts.append(prompt)
        
        start = time.time()
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for i, (output, reranking) in enumerate(zip(outputs, reranking_results)):
            answer = output.outputs[0].text
            tokens_used = len(output.outputs[0].token_ids)
            
            context_docs = reranking.doc_ids[:5]
            import re
            citations = re.findall(r'\[Doc (\d+)\]', answer)
            citations = [context_docs[int(c)-1] for c in citations if int(c) <= len(context_docs)]
            
            results.append(GenerationResult(
                query_id=reranking.query_id,
                answer=answer,
                citations=citations,
                tokens_used=tokens_used,
                latency_ms=(time.time() - start) * 1000 / len(prompts)
            ))
        
        return results

class LocalGenerator(BaseGenerator):
    """Local HuggingFace generator using transformers."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize()
    
    def _initialize(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading local model: {self.config.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, queries: List[str], reranking_results: List[Any]) -> List[GenerationResult]:
        import time
        import torch
        
        results = []
        for query, reranking in zip(queries, reranking_results):
            start = time.time()
            
            context_docs = reranking.doc_ids[:5]
            context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(context_docs)])
            
            prompt = f"""Based on the following context, answer the question. Cite sources using [Doc N] format.

Context:
{context}

Question: {query}

Answer:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            tokens_used = outputs[0].shape[1] - inputs.input_ids.shape[1]
            
            import re
            citations = re.findall(r'\[Doc (\d+)\]', answer)
            citations = [context_docs[int(c)-1] for c in citations if int(c) <= len(context_docs)]
            
            results.append(GenerationResult(
                query_id=reranking.query_id,
                answer=answer,
                citations=citations,
                tokens_used=tokens_used,
                latency_ms=(time.time() - start) * 1000
            ))
        
        return results

def get_generator(config) -> BaseGenerator:
    """Factory function to get generator based on config."""
    provider = getattr(config, "provider", "openai")
    
    if provider == "vllm":
        return VLLMGenerator(config)
    elif provider == "local":
        return LocalGenerator(config)
    else:
        return OpenAIGenerator(config)