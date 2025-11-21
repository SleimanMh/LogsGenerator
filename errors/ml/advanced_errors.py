"""
ML Advanced & Complex Errors (ml_err_86 - ml_err_100)
Covers: Pipelines, nested CV, hyperparameter optimization, multi-task learning, imbalanced data
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Advanced Errors (86 - 100)
# ============================================================

def ml_err_86():
    """Pipeline with invalid test shape (predict error)"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    pipe.fit(X, y)

    # Wrong feature dimension → natural sklearn ValueError
    X_test = np.random.randn(10, 3)
    pipe.predict(X_test)


def ml_err_87():
    """Nested CV class distribution incompatible with StratifiedKFold"""
    import numpy as np
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    
    X = np.random.randn(20, 5)
    y = np.array([0]*19 + [1])  # highly imbalanced
    
    cv = StratifiedKFold(n_splits=10)
    cross_validate(LogisticRegression(), X, y, cv=cv)


def ml_err_88():
    """RandomizedSearchCV empty param_dist"""
    import numpy as np
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression
    
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    # Empty param_dist triggers natural ValueError
    rs = RandomizedSearchCV(LogisticRegression(), param_distributions={})
    rs.fit(X, y)


def ml_err_89():
    """Permutation importance invalid scorer"""
    import numpy as np
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    model = LogisticRegression().fit(X, y)

    permutation_importance(model, X, y, scoring='invalid_scorer')


def ml_err_90():
    """Partial dependence invalid feature index"""
    import numpy as np
    from sklearn.inspection import partial_dependence
    from sklearn.tree import DecisionTreeClassifier
    
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model = DecisionTreeClassifier().fit(X, y)
    
    # Feature index 100 does not exist
    partial_dependence(model, X, [100])


def ml_err_91():
    """Predict on unfitted model"""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    
    X = np.random.randn(10, 5)
    LogisticRegression().predict(X)   # Not fitted → natural sklearn error


def ml_err_92():
    """FeatureHasher invalid input type"""
    from sklearn.feature_extraction import FeatureHasher
    
    hasher = FeatureHasher(n_features=32, input_type='dict')

    # Passing a string causes natural type error
    hasher.transform("invalid_input")


def ml_err_93():
    """TF-IDF on numeric matrix"""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vec = TfidfVectorizer()
    X = np.random.randn(10, 5)   # numeric instead of text
    vec.fit(X)


def ml_err_94():
    """Word2Vec lookup missing token"""
    import gensim.models
    sentences = [["hello", "world"]]
    model = gensim.models.Word2Vec(sentences, vector_size=10)

    # Lookup non-existent token → KeyError naturally
    _ = model.wv["missing_token"]


def ml_err_95():
    """Cosine similarity with NaNs"""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    ratings = np.array([[5, 3, np.nan], [4, np.nan, 2]])

    # cosine_similarity naturally errors with NaNs
    cosine_similarity(ratings)


def ml_err_96():
    """NearestNeighbors too many neighbors"""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    
    X_train = np.random.randn(20, 5)

    # Asking for more neighbors than samples → ValueError
    nbrs = NearestNeighbors(n_neighbors=50)
    nbrs.fit(X_train)
    nbrs.kneighbors([[1, 2, 3, 4, 5]])


def ml_err_97():
    """SMOTE with only one class"""
    import numpy as np
    from imblearn.over_sampling import SMOTE
    
    X = np.random.randn(100, 5)
    y = np.zeros(100)  # one class only

    SMOTE().fit_resample(X, y)   # naturally errors


def ml_err_98():
    """Active learning uncertainty: invalid shape"""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    model = RandomForestClassifier().fit(X, y)

    # Predict_proba shape mismatch for reduction
    probs = model.predict_proba(X)
    wrong = probs[:, :1]         # shape (50,1)
    np.max(wrong, axis=1, keepdims=True)  # triggers shape logic error


def ml_err_99():
    """GridSearchCV wrong param_grid for estimator"""
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    
    X = np.random.randn(30, 5)
    y = np.random.randint(0, 2, 30)

    param_grid = {"unknown_param": [1, 2, 3]}   # invalid

    GridSearchCV(LogisticRegression(), param_grid).fit(X, y)


def ml_err_100():
    """Multi-task learning: inconsistent target dimensions"""
    import numpy as np
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import LinearRegression
    
    X = np.random.randn(50, 5)
    y1 = np.random.randn(50, 1)
    y2 = np.random.randn(49, 1)  # wrong size
    Y = np.hstack([y1, y2])      # dimension mismatch

    MultiOutputRegressor(LinearRegression()).fit(X, Y)


__all__ = [f"ml_err_{i:02d}" for i in range(86, 101)]
