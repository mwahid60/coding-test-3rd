"""
Simple logging utility - prints to terminal only
"""
import logging
import sys
from functools import wraps
from datetime import datetime


def setup_logging():
    """Setup basic logging to terminal"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger for a module"""
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls and results
    
    Usage:
        @log_function_call(logger)
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"▶️  Starting: {func_name}")
            try:
                result = await func(*args, **kwargs)
                logger.info(f"✅ Completed: {func_name}")
                return result
            except Exception as e:
                logger.error(f"❌ Failed: {func_name} - Error: {str(e)}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"▶️  Starting: {func_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"✅ Completed: {func_name}")
                return result
            except Exception as e:
                logger.error(f"❌ Failed: {func_name} - Error: {str(e)}")
                raise
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
