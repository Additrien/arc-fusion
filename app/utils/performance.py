"""
Performance profiling utilities for Arc-Fusion RAG system.

Provides timing decorators and performance monitoring tools to identify bottlenecks
and optimize the multi-agent pipeline.
"""

import time
import asyncio
import functools
from typing import Any, Callable, Dict, Optional
from contextlib import asynccontextmanager, contextmanager
from app.utils.logger import get_logger

logger = get_logger('arc_fusion.performance')

# Global performance metrics storage
performance_metrics: Dict[str, Dict[str, Any]] = {}


def time_function(func_name: Optional[str] = None):
    """
    Decorator to time synchronous functions and log performance metrics.
    
    Args:
        func_name: Optional custom name for the function in logs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance
                logger.info(f"PERF: {name} completed in {execution_time:.3f}s")
                
                # Store metrics
                if name not in performance_metrics:
                    performance_metrics[name] = {
                        'calls': 0,
                        'total_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'avg_time': 0.0
                    }
                
                metrics = performance_metrics[name]
                metrics['calls'] += 1
                metrics['total_time'] += execution_time
                metrics['min_time'] = min(metrics['min_time'], execution_time)
                metrics['max_time'] = max(metrics['max_time'], execution_time)
                metrics['avg_time'] = metrics['total_time'] / metrics['calls']
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"PERF: {name} failed after {execution_time:.3f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator


def time_async_function(func_name: Optional[str] = None):
    """
    Decorator to time asynchronous functions and log performance metrics.
    
    Args:
        func_name: Optional custom name for the function in logs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance
                logger.info(f"PERF: {name} completed in {execution_time:.3f}s")
                
                # Store metrics
                if name not in performance_metrics:
                    performance_metrics[name] = {
                        'calls': 0,
                        'total_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'avg_time': 0.0
                    }
                
                metrics = performance_metrics[name]
                metrics['calls'] += 1
                metrics['total_time'] += execution_time
                metrics['min_time'] = min(metrics['min_time'], execution_time)
                metrics['max_time'] = max(metrics['max_time'], execution_time)
                metrics['avg_time'] = metrics['total_time'] / metrics['calls']
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"PERF: {name} failed after {execution_time:.3f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator


@contextmanager
def time_block(block_name: str):
    """
    Context manager to time code blocks.
    
    Args:
        block_name: Name of the code block for logging
    """
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger.info(f"PERF: {block_name} completed in {execution_time:.3f}s")


@asynccontextmanager
async def time_async_block(block_name: str):
    """
    Async context manager to time async code blocks.
    
    Args:
        block_name: Name of the code block for logging
    """
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger.info(f"PERF: {block_name} completed in {execution_time:.3f}s")


def get_performance_summary() -> Dict[str, Any]:
    """
    Get a summary of all performance metrics.
    
    Returns:
        Dictionary containing performance statistics
    """
    if not performance_metrics:
        return {"message": "No performance data collected"}
    
    summary = {
        "total_functions": len(performance_metrics),
        "functions": {}
    }
    
    total_calls = 0
    total_time = 0.0
    
    for name, metrics in performance_metrics.items():
        summary["functions"][name] = {
            "calls": metrics["calls"],
            "total_time": metrics["total_time"],
            "avg_time": metrics["avg_time"],
            "min_time": metrics["min_time"],
            "max_time": metrics["max_time"]
        }
        total_calls += metrics["calls"]
        total_time += metrics["total_time"]
    
    summary["overall"] = {
        "total_calls": total_calls,
        "total_time": total_time,
        "avg_time_per_call": total_time / total_calls if total_calls > 0 else 0
    }
    
    return summary


def log_performance_summary():
    """Log a summary of all performance metrics."""
    summary = get_performance_summary()
    
    if "message" in summary:
        logger.info("PERF SUMMARY: No performance data collected")
        return
    
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    # Sort functions by total time
    sorted_functions = sorted(
        summary["functions"].items(),
        key=lambda x: x[1]["total_time"],
        reverse=True
    )
    
    for name, metrics in sorted_functions:
        logger.info(f"{name}:")
        logger.info(f"  Calls: {metrics['calls']}")
        logger.info(f"  Total time: {metrics['total_time']:.3f}s")
        logger.info(f"  Avg time: {metrics['avg_time']:.3f}s")
        logger.info(f"  Min/Max: {metrics['min_time']:.3f}s / {metrics['max_time']:.3f}s")
        logger.info("")
    
    overall = summary["overall"]
    logger.info(f"OVERALL: {overall['total_calls']} calls, {overall['total_time']:.3f}s total, {overall['avg_time_per_call']:.3f}s avg")
    logger.info("=" * 60)


def clear_performance_metrics():
    """Clear all performance metrics."""
    global performance_metrics
    performance_metrics.clear()
    logger.info("PERF: Performance metrics cleared") 