"""
Base classes and registries for error handling
"""

from typing import Callable, Dict, List
from fastapi import FastAPI


class ErrorRegistry:
    """Registry for organizing and managing error functions"""
    
    def __init__(self, error_class: str):
        self.error_class = error_class
        self.errors: Dict[str, Callable] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register(self, name: str, func: Callable, category: str = "default"):
        """Register an error function"""
        self.errors[name] = func
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)
    
    def get_all(self) -> Dict[str, Callable]:
        """Get all registered errors"""
        return self.errors.copy()
    
    def get_by_category(self, category: str) -> Dict[str, Callable]:
        """Get errors by category"""
        if category not in self.categories:
            return {}
        
        names = self.categories[category]
        return {name: self.errors[name] for name in names if name in self.errors}
    
    def list_categories(self) -> List[str]:
        """List all categories"""
        return list(self.categories.keys())
    
    def count_total(self) -> int:
        """Total number of registered errors"""
        return len(self.errors)
    
    def count_by_category(self) -> Dict[str, int]:
        """Count errors per category"""
        return {cat: len(names) for cat, names in self.categories.items()}


class BaseErrorHandler:
    """Base class for error handlers"""
    
    def __init__(self, error_class: str, category: str = ""):
        self.error_class = error_class
        self.category = category
        self.registry = ErrorRegistry(error_class)
    
    def register_error(self, name: str, func: Callable):
        """Register an error function"""
        self.registry.register(name, func, self.category)
    
    def get_error_function(self, name: str) -> Callable:
        """Get a specific error function"""
        return self.registry.errors.get(name)
    
    def execute_error(self, name: str, **kwargs):
        """Execute an error function"""
        func = self.get_error_function(name)
        if func:
            return func(**kwargs)
        raise ValueError(f"Error function '{name}' not found")
    
    def run_all(self) -> Dict:
        """Execute all registered errors"""
        results = {}
        for name, func in self.registry.get_all().items():
            try:
                func()
            except Exception as e:
                results[name] = str(type(e).__name__)
        return results
