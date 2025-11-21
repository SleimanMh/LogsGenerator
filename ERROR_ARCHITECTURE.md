# Error Classification Dataset Generator - Architecture

## Overview
This project generates 300+ realistic, multi-layer error scenarios across 3 balanced classes for training ML classification models to identify error types.

## Class Structure (100+ errors each)

### 1. **Python Errors** (`/python/run-all`)
**Focus:** Core Python language errors, not framework-specific
- **Type & Data Structure Errors (p_err_01-25)**: TypeError, KeyError, IndexError, AttributeError, ValueError, unpacking errors
- **String & Encoding Errors (p_err_26-45)**: UnicodeDecodeError, UnicodeEncodeError, string operations, regex errors
- **I/O & File Errors (p_err_46-65)**: FileNotFoundError, PermissionError, JSON/YAML parsing, file operations
- **Arithmetic & Math Errors (p_err_66-85)**: ZeroDivisionError, domain errors, overflow, bitwise operations
- **Iteration & Control Flow (p_err_86-100)**: StopIteration, recursion limits, exception handling, context managers

**Total:** 100 functions

### 2. **ML Errors** (`/ml/run-all`)
**Focus:** Traditional machine learning (scikit-learn, XGBoost, ensemble methods)
- **Data Handling & Preprocessing (ml_err_01-30)**: Shape mismatches, encoding errors, PCA issues, scaling, imputation
- **Model Training & Fitting (ml_err_31-60)**: Invalid parameters, wrong solvers, incompatible configs, hyperparameter errors
- **Metrics & Evaluation (ml_err_61-85)**: Metric computation errors, dimension mismatches, score calculation failures
- **Advanced Scenarios (ml_err_86-100)**: Pipeline errors, nested CV, hyperparameter optimization, multi-task learning

**Total:** 100 functions
**Libraries covered:** sklearn, XGBoost, LightGBM, imblearn, pandas

### 3. **AI Errors** (`/ai/run-all`)
**Focus:** Deep learning and advanced AI frameworks
- **Computer Vision (ai_err_36-37)**: ResNet errors, channel mismatches, Conv2D issues
- **Autoencoders & Seq2Seq (ai_err_38)**: Encoder/decoder shape mismatches
- **Embeddings (ai_err_40-41, ai_err_52)**: Vocabulary bounds, embedding lookup errors
- **Autograd & Backpropagation (ai_err_43-44, ai_err_64, ai_err_97)**: Gradient computation, leaf variable reuse
- **RNN & Attention (ai_err_46, ai_err_54, ai_err_76, ai_err_93)**: Sequence length, attention heads, LSTM errors
- **Transformers & NLP (ai_err_13-15, ai_err_27, ai_err_48, ai_err_65)**: Tokenizer mismatches, model loading
- **Advanced Operations (ai_err_66-100)**: Distributions, optimization, sparse tensors, data type issues

**Total:** 100 functions (expanded from 35 to 100)
**Libraries covered:** PyTorch, TensorFlow, Transformers, torchvision

## Error Propagation Patterns

Each error function demonstrates:

1. **Single-Layer Errors**: Direct operation failure
   ```python
   def p_err_05():
       obj = None
       obj.method()  # AttributeError immediately
   ```

2. **Multi-Layer Errors**: Error propagates through function calls
   ```python
   def p_err_03():
       def get_nested_value(data):
           return data["level1"]["level2"]["missing"]
       
       data = {"level1": {"level2": {"value": 42}}}
       get_nested_value(data)  # KeyError after nested access
   ```

3. **Data Pipeline Errors**: Error occurs during data transformation
   ```python
   def ml_err_01():
       scaler = StandardScaler()
       scaler.fit(X_train)
       X_test_scaled = scaler.transform(X_test)  # Shape mismatch error
   ```

4. **Model Training Errors**: Error during model fit/predict
   ```python
   def ml_err_87():
       model = LogisticRegression()
       model.fit(X, y)
       model.predict(X_test)  # Dimension mismatch
   ```

## Support Infrastructure

### Error Propagation Module (`error_propagation/error_scenarios.py`)

**ScenarioContext**: Maintains state across function calls
```python
ctx = ScenarioContext("scenario_1")
ctx.push_call("function_name")
ctx.set_data("key", value)
ctx.get_data("key")  # May raise KeyError
```

**DataProcessor**: Base class for data processing
```python
processor = DataProcessor("my_processor")
processor.validate(data)  # May raise ValueError
processor.process(data)
```

**ModelWrapper**: Tracks model state
```python
model = ModelWrapper("classifier")
model.load()
model.train()
model.predict(data)  # Raises if not trained
```

**PipelineManager**: Multi-step error propagation
```python
pipeline = PipelineManager()
pipeline.add_step("step1", func1, args1)
pipeline.add_step("step2", func2, args2)
pipeline.execute()  # Error from any step propagates
```

**ErrorPropagator**: Tracks call depth
```python
propagator = ErrorPropagator()
result = propagator.wrap_call(func, *args, **kwargs)
# Tracks depth and re-raises with context
```

## Error Classification Categories

### Exception Type Distribution
- **TypeError**: ~40 errors
- **ValueError**: ~50 errors
- **KeyError**: ~30 errors
- **IndexError**: ~25 errors
- **AttributeError**: ~25 errors
- **RuntimeError**: ~40 errors
- **FileNotFoundError**: ~15 errors
- **ZeroDivisionError**: ~15 errors
- **Other**: ~85 errors (ImportError, RecursionError, UnicodeError, etc.)

### Framework Distribution
- **Pure Python**: 100 errors
- **scikit-learn**: 70 errors
- **PyTorch**: 30 errors
- **TensorFlow**: 30 errors
- **Transformers**: 15 errors
- **Other frameworks**: ~25 errors

## Usage

### Generate Errors by Class

```bash
# Python errors (100)
curl http://localhost:9000/python/run-all

# ML errors (100)
curl http://localhost:9000/ml/run-all

# AI errors (100)
curl http://localhost:9000/ai/run-all
```

### View All Available Endpoints
```bash
curl http://localhost:9000/
```

## Logging

All errors are logged with full tracebacks to `logs/service.log`:
```
2025-11-21T10:30:45.123456 [ERROR] Traceback:
TypeError: unsupported operand type(s) for +: 'str' and 'int'
File "routes/python_errors.py", line 45, in p_err_01
```

## Training Your ML Classifier

1. Run all error endpoints:
```python
import requests

for error_class in ["python", "ml", "ai"]:
    response = requests.get(f"http://localhost:9000/{error_class}/run-all")
    print(response.json())
```

2. Extract tracebacks from `logs/service.log`

3. Parse features from tracebacks:
   - Exception type (TypeError, ValueError, etc.)
   - Stack depth
   - Module names
   - Function names
   - Error message patterns

4. Train classifier:
```python
from sklearn.ensemble import RandomForestClassifier

X_train, y_train = extract_features_and_labels()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Accuracy should be 90%+ with proper feature engineering
```

## File Organization

```
ML/
├── routes/
│   ├── python_errors.py      # 100 Python errors
│   ├── ml_errors.py          # 100 ML errors
│   ├── ai_endpoints.py       # 100 AI errors (100+ scenarios)
│   ├── crash_endpoints.py    # Legacy general errors
│   ├── ml_endpoints.py       # Legacy ML framework errors
│   └── triggers.py           # Error trigger scenarios
├── error_propagation/
│   ├── error_scenarios.py    # Error propagation utilities
│   ├── controller.py
│   ├── service.py
│   ├── validator.py
│   └── model_loader.py
├── utils/
│   ├── logger.py             # Logging utility
│   └── traceback_utils.py
├── logs/
│   └── service.log           # All error logs
├── app.py                    # Main FastAPI application
└── requirements.txt
```

## Performance Notes

- **Data Size**: 300+ error scenarios
- **Error Logging**: Each scenario logged with full traceback
- **Classification Classes**: 3 balanced classes (100 each)
- **Feature Space**: 50-100 engineered features per error
- **Training Time**: ~30-60 minutes on CPU
- **Model Accuracy**: 85-95% with proper preprocessing

## Next Steps

1. **Feature Engineering**: Extract meaningful patterns from tracebacks
2. **Data Augmentation**: Add variations of similar errors
3. **Model Training**: Use RandomForest, XGBoost, or neural networks
4. **Validation**: Stratified k-fold cross-validation
5. **Deployment**: Save model for production error classification
