"""
ML Data Handling & Preprocessing Errors (ml_err_01 - ml_err_30)
Covers: Shape mismatches, encoding, scaling, feature engineering, dimensionality reduction
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Data Handling & Preprocessing Errors (01 - 30)
# ============================================================

def ml_err_01():
    """Shape mismatch in preprocessing"""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(50, 3)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)


def ml_err_02():
    """Categorical encoding unseen value"""
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(['cat', 'dog', 'bird'])
    encoder.transform(['cat', 'elephant'])


def ml_err_03():
    """OneHotEncoder unseen category"""
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    enc = OneHotEncoder()
    X = np.array([['cat'], ['dog'], ['bird']])
    enc.fit(X)
    enc.transform([['elephant']])


def ml_err_04():
    """PCA more components than features"""
    import numpy as np
    from sklearn.decomposition import PCA
    X = np.random.randn(50, 10)
    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(X)


def ml_err_05():
    """Scaler transform before fit"""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X = np.random.randn(100, 5)
    scaler = StandardScaler()
    X_scaled = scaler.transform(X)


def ml_err_06():
    """MinMaxScaler with string data"""
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    df = pd.DataFrame({'col': ['a', 'b', 'c']})
    scaler = MinMaxScaler()
    scaler.fit_transform(df)


def ml_err_07():
    """MultiLabelBinarizer unseen class"""
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([[1, 2], [2, 3]])
    mlb.transform([[1, 4]])


def ml_err_08():
    """Polynomial features wrong shape"""
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    X = np.array([[1, 2], [3, 4]])
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_test = np.array([1, 2, 3])
    X_test_poly = poly.transform(X_test)


def ml_err_09():
    """RobustScaler on empty data"""
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    X = np.array([]).reshape(0, 5)
    scaler = RobustScaler()
    scaler.fit(X)


def ml_err_10():
    """Imputer incompatible strategy"""
    import pandas as pd
    from sklearn.impute import SimpleImputer
    df = pd.DataFrame({'col': ['a', 'b', 'c']})
    imputer = SimpleImputer(strategy='mean')
    imputer.fit_transform(df)


def ml_err_11():
    """Feature scaling NaN propagation"""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X = np.array([[1, 2], [np.nan, 4], [5, 6]])
    scaler = StandardScaler()
    scaler.fit(X)


def ml_err_12():
    """Cross-validation folds > samples"""
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    X = np.random.randn(5, 3)
    y = np.random.randn(5)
    cross_val_score(LinearRegression(), X, y, cv=10)


def ml_err_13():
    """Train-test split length mismatch"""
    import numpy as np
    from sklearn.model_selection import train_test_split
    X = np.random.randn(100, 5)
    y = np.random.randn(95)
    X_train, X_test, y_train, y_test = train_test_split(X, y)


def ml_err_14():
    """Pipeline wrong number of steps"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('scaler2', StandardScaler()),
        ('model', LinearRegression())
    ])


def ml_err_15():
    """GridSearchCV invalid params"""
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    param_grid = {'C': ['invalid', -5]}
    clf = LogisticRegression()
    grid = GridSearchCV(clf, param_grid)
    grid.fit(X, y)


def ml_err_16():
    """StratifiedKFold invalid class distribution"""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    X = np.random.randn(10, 5)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    skf = StratifiedKFold(n_splits=10)
    for train_idx, test_idx in skf.split(X, y):
        pass


def ml_err_17():
    """Feature scaling zero variance"""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X = np.array([[1, 5], [1, 5], [1, 5]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if np.any(np.isnan(X_scaled)):
        _ = 1 / 0


def ml_err_18():
    """Binarizer invalid threshold"""
    import numpy as np
    from sklearn.preprocessing import Binarizer
    X = np.random.randn(10, 5)
    bin = Binarizer(threshold="invalid")
    bin.transform(X)


def ml_err_19():
    """KBinsDiscretizer wrong n_bins"""
    import numpy as np
    from sklearn.preprocessing import KBinsDiscretizer
    X = np.random.randn(10, 5)
    kbd = KBinsDiscretizer(n_bins=100)
    kbd.fit_transform(X)


def ml_err_20():
    """Quantile Transformer extreme quantile"""
    import numpy as np
    from sklearn.preprocessing import QuantileTransformer
    X = np.random.randn(10, 5)
    qt = QuantileTransformer(output_distribution='uniform', subsample=5)
    qt.fit_transform(X)


def ml_err_21():
    """Normalizer with invalid norm"""
    import numpy as np
    from sklearn.preprocessing import Normalizer
    X = np.random.randn(10, 5)
    norm = Normalizer(norm='invalid')
    norm.transform(X)


def ml_err_22():
    """FunctionTransformer non-callable"""
    import numpy as np
    from sklearn.preprocessing import FunctionTransformer
    X = np.random.randn(10, 5)
    ft = FunctionTransformer(func="not_callable")
    ft.fit_transform(X)


def ml_err_23():
    """PowerTransformer invalid method"""
    import numpy as np
    from sklearn.preprocessing import PowerTransformer
    X = np.random.randn(10, 5)
    pt = PowerTransformer(method='invalid')
    pt.fit_transform(X)


def ml_err_24():
    """SplineTransformer wrong n_knots"""
    import numpy as np
    from sklearn.preprocessing import SplineTransformer
    X = np.random.randn(10, 1)
    st = SplineTransformer(n_knots=100)
    st.fit_transform(X)


def ml_err_25():
    """OrdinalEncoder with missing value"""
    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder
    X = np.array([[1], [2], [np.nan]])
    enc = OrdinalEncoder()
    enc.fit_transform(X)


def ml_err_26():
    """ColumnTransformer column name mismatch"""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    ct = ColumnTransformer([('scaler', StandardScaler(), ['missing'])])
    ct.fit_transform(df)


def ml_err_27():
    """SelectKBest invalid k value"""
    import numpy as np
    from sklearn.feature_selection import SelectKBest
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, 10)
    selector = SelectKBest(k=100)
    selector.fit_transform(X, y)


def ml_err_28():
    """VarianceThreshold threshold > max variance"""
    import numpy as np
    from sklearn.feature_selection import VarianceThreshold
    X = np.random.randn(10, 5)
    vt = VarianceThreshold(threshold=1000)
    vt.fit_transform(X)


def ml_err_29():
    """RFE invalid n_features_to_select"""
    import numpy as np
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    rfe = RFE(LinearRegression(), n_features_to_select=100)
    rfe.fit_transform(X, y)


def ml_err_30():
    """Sequential feature selection invalid n_features"""
    import numpy as np
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.linear_model import LogisticRegression
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    sfs = SequentialFeatureSelector(LogisticRegression(), n_features_to_select=100)
    sfs.fit(X, y)


# Registry
__all__ = [f'ml_err_{i:02d}' for i in range(1, 31)]
