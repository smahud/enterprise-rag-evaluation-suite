# Utilities for enterprise-rag-evaluation-suite

import logging
import functools
import time
from typing import Any, Callable, TypeVar
from pathlib import Path

F = TypeVar('F', bound=Callable[..., Any])

def setup_logging(level: str = "INFO", format: str = "json") -> logging.Logger:
    """Configure structured logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if format == "json":
        from pythonjsonlogger import jsonlogger
        handler = logging.StreamHandler()
        handler.setFormatter(jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            timestamp=True,
        ))
        logging.root.handlers = [handler]
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    logging.root.setLevel(log_level)
    return logging.getLogger(__name__)

def timer(func: F) -> F:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

class CacheManager:
    """Simple file-based cache manager."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Any:
        """Get cached value."""
        import pickle
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Cache read failed for {key}: {e}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        import pickle
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            self.logger.warning(f"Cache write failed for {key}: {e}")
    
    def delete(self, key: str) -> None:
        """Delete cached value."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
    
    def clear(self) -> None:
        """Clear all cache."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_json(path: str) -> Any:
    """Load JSON file."""
    import json
    with open(path) as f:
        return json.load(f)

def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)