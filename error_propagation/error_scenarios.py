"""
Error Scenario Helper Module
Provides utilities for creating multi-layer, multi-function error propagation scenarios.
"""

class ScenarioContext:
    """Maintains state across multiple function calls in a scenario"""
    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.data = {}
        self.state = "initialized"
        self.call_stack = []
    
    def push_call(self, func_name: str):
        """Track function calls"""
        self.call_stack.append(func_name)
    
    def set_data(self, key: str, value):
        """Store data between function calls"""
        self.data[key] = value
    
    def get_data(self, key: str):
        """Retrieve data"""
        if key not in self.data:
            raise KeyError(f"Data key '{key}' not found in scenario {self.scenario_id}")
        return self.data[key]
    
    def update_state(self, state: str):
        """Update scenario state"""
        self.state = state


class DataProcessor:
    """Base class for data processing with potential errors"""
    def __init__(self, name: str = "processor"):
        self.name = name
        self.validated = False
    
    def validate(self, data):
        """Validate input data"""
        if data is None:
            raise ValueError(f"{self.name}: Input data cannot be None")
        self.validated = True
        return data
    
    def process(self, data):
        """Process data - to be overridden"""
        if not self.validated:
            self.validate(data)
        return data


class ModelWrapper:
    """Wrapper for ML/AI models with state tracking"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False
        self.is_trained = False
        self.state_dict = {}
    
    def load(self):
        """Load model"""
        self.is_loaded = True
    
    def train(self):
        """Train model"""
        if not self.is_loaded:
            raise RuntimeError(f"{self.model_name}: Model must be loaded before training")
        self.is_trained = True
    
    def predict(self, data):
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name}: Model must be trained before prediction")
        return data


class PipelineManager:
    """Manages multi-step pipeline with error propagation"""
    def __init__(self):
        self.steps = []
        self.results = {}
        self.current_step = 0
    
    def add_step(self, step_name: str, function, *args, **kwargs):
        """Add step to pipeline"""
        self.steps.append({
            'name': step_name,
            'function': function,
            'args': args,
            'kwargs': kwargs
        })
    
    def execute(self):
        """Execute all steps sequentially"""
        for i, step in enumerate(self.steps):
            self.current_step = i
            try:
                result = step['function'](*step['args'], **step['kwargs'])
                self.results[step['name']] = result
            except Exception as e:
                raise RuntimeError(f"Pipeline failed at step '{step['name']}': {str(e)}")


class DataValidator:
    """Validates data at multiple stages"""
    @staticmethod
    def validate_shape(data, expected_shape):
        """Validate data shape"""
        import numpy as np
        if isinstance(data, np.ndarray):
            if data.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch: expected {expected_shape}, got {data.shape}"
                )
        return data
    
    @staticmethod
    def validate_dtype(data, expected_dtype):
        """Validate data type"""
        import numpy as np
        if isinstance(data, np.ndarray):
            if data.dtype != expected_dtype:
                raise TypeError(
                    f"Dtype mismatch: expected {expected_dtype}, got {data.dtype}"
                )
        return data
    
    @staticmethod
    def validate_not_null(data, field_name="data"):
        """Validate data is not null"""
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ValueError(f"{field_name} cannot be null or empty")
        return data


class ErrorPropagator:
    """Propagates errors through multiple layers"""
    def __init__(self):
        self.call_depth = 0
        self.max_depth = 5
    
    def wrap_call(self, func, *args, **kwargs):
        """Wrap function call to track depth"""
        self.call_depth += 1
        if self.call_depth > self.max_depth:
            raise RecursionError(f"Call depth exceeded {self.max_depth}")
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            raise RuntimeError(
                f"Error at depth {self.call_depth}: {str(e)}"
            ) from e
        finally:
            self.call_depth -= 1
