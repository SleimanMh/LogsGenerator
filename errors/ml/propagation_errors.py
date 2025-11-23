"""
ML Multi-layer Error Propagation (ml_err_181 - ml_err_200)
Errors that propagate through nested function calls.
Each error cascades from one function to another through the call stack.
"""


# ml_err_181: Shape mismatch propagates from loader -> preprocessor -> model
def ml_err_181_load_data():
    """Load data with wrong shape."""
    import numpy as np
    return np.array([[1, 2], [3, 4]]), np.array([0])  # 2 samples, 1 label


def ml_err_181_preprocess(X, y):
    """Preprocess data."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X), y


def ml_err_181_train(X, y):
    """Train model - error propagates here."""
    from sklearn.linear_model import LogisticRegression
    LogisticRegression().fit(X, y)


def ml_err_181():
    """Shape mismatch: 2 samples, 1 label."""
    X, y = ml_err_181_load_data()
    X_scaled, y = ml_err_181_preprocess(X, y)
    ml_err_181_train(X_scaled, y)


# ml_err_182: NaN propagates from data -> training -> prediction
def ml_err_182_load_data():
    """Load data with NaN."""
    import numpy as np
    return np.array([[1, np.nan], [3, 4]]), np.array([0, 1])


def ml_err_182_train(X, y):
    """Train model with NaN."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    return model


def ml_err_182_predict(model, X):
    """Predict with NaN propagated."""
    return model.predict(X)


def ml_err_182():
    """NaN in features propagates through training."""
    X, y = ml_err_182_load_data()
    model = ml_err_182_train(X, y)
    ml_err_182_predict(model, X)


# ml_err_183: Unknown category propagates from encode -> fit -> transform
def ml_err_183_fit_encoder(data):
    """Fit encoder on known categories."""
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(['a', 'b', 'c'])
    return encoder


def ml_err_183_transform(encoder, data):
    """Transform unknown category - error propagates."""
    return encoder.transform(data)


def ml_err_183():
    """Unknown category error in encoding."""
    encoder = ml_err_183_fit_encoder(['a', 'b', 'c'])
    ml_err_183_transform(encoder, ['d'])


# ml_err_184: Feature dimension mismatch from training -> prediction
def ml_err_184_train_model(X, y):
    """Train on 2 features."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    return model


def ml_err_184_predict(model, X_test):
    """Predict with mismatched dimensions."""
    return model.predict(X_test)


def ml_err_184():
    """Dimension mismatch: trained on 2 features, predicted on 3."""
    model = ml_err_184_train_model([[1, 2], [3, 4]], [0, 1])
    ml_err_184_predict(model, [[1, 2, 3]])


# ml_err_185: Array dimension error from prepare -> fit
def ml_err_185_prepare_data():
    """Prepare 1D array instead of 2D."""
    import numpy as np
    return np.array([1, 2, 3]), np.array([0, 1])


def ml_err_185_fit_model(X, y):
    """Fit with 1D X - error propagates."""
    from sklearn.linear_model import LinearRegression
    LinearRegression().fit(X, y)


def ml_err_185():
    """1D array passed to model expecting 2D."""
    X, y = ml_err_185_prepare_data()
    ml_err_185_fit_model(X, y)


# ml_err_186: CV fold mismatch from validate -> split
def ml_err_186_create_cv(n_splits):
    """Create CV splitter with invalid splits."""
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=n_splits)


def ml_err_186_validate(cv, X, y):
    """Validate splits - error propagates."""
    return list(cv.split(X, y))


def ml_err_186():
    """100 splits for 3 samples."""
    cv = ml_err_186_create_cv(n_splits=100)
    ml_err_186_validate(cv, [[1], [2], [3]], [0, 1, 0])


# ml_err_187: Negative learning rate from init -> fit
def ml_err_187_create_model(lr):
    """Create model with invalid learning rate."""
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(learning_rate=lr)


def ml_err_187_fit(model, X, y):
    """Fit with invalid learning rate - error propagates."""
    model.fit(X, y)


def ml_err_187():
    """Negative learning rate during initialization."""
    model = ml_err_187_create_model(lr=-1)
    ml_err_187_fit(model, [[1, 2], [3, 4]], [0, 1])


# ml_err_188: Invalid PCA components from init -> fit
def ml_err_188_create_pca(n_comp):
    """Create PCA with invalid components."""
    from sklearn.decomposition import PCA
    return PCA(n_components=n_comp)


def ml_err_188_fit_pca(pca, X):
    """Fit with invalid components - error propagates."""
    pca.fit(X)


def ml_err_188():
    """Negative n_components."""
    pca = ml_err_188_create_pca(n_components=-1)
    ml_err_188_fit_pca(pca, [[1, 2], [3, 4]])


# ml_err_189: Single class from prepare -> fit
def ml_err_189_prepare_data():
    """Prepare single-class data."""
    return [[1], [2], [3]], [0, 0, 0]


def ml_err_189_fit_model(X, y):
    """Fit with single class - error propagates."""
    from sklearn.linear_model import LogisticRegression
    LogisticRegression().fit(X, y)


def ml_err_189():
    """All samples in single class."""
    X, y = ml_err_189_prepare_data()
    ml_err_189_fit_model(X, y)


# ml_err_190: Invalid kernel from init -> fit
def ml_err_190_create_svm(kernel):
    """Create SVM with invalid kernel."""
    from sklearn.svm import SVC
    return SVC(kernel=kernel)


def ml_err_190_fit_svm(svm, X, y):
    """Fit with invalid kernel - error propagates."""
    svm.fit(X, y)


def ml_err_190():
    """Invalid SVM kernel parameter."""
    svm = ml_err_190_create_svm(kernel='invalid')
    ml_err_190_fit_svm(svm, [[1, 2], [3, 4]], [0, 1])


# ml_err_191: Label mismatch in metric calculation
def ml_err_191_prepare_preds():
    """Prepare predictions with mismatched labels."""
    return [0, 1, 2], [0, 1, 3]


def ml_err_191_calculate_accuracy(y_true, y_pred):
    """Calculate accuracy with mismatched labels."""
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred, normalize=True)


def ml_err_191():
    """Mismatched shapes in metric calculation."""
    y_true, y_pred = ml_err_191_prepare_preds()
    ml_err_191_calculate_accuracy(y_true, y_pred)


# ml_err_192: Sample weight mismatch from prepare -> fit
def ml_err_192_prepare_weights():
    """Prepare mismatched sample weights."""
    return [[1, 2], [3, 4]], [0, 1], [1]  # 2 samples, 1 weight


def ml_err_192_fit_weighted(X, y, weights):
    """Fit with mismatched weights - error propagates."""
    from sklearn.linear_model import LogisticRegression
    LogisticRegression().fit(X, y, sample_weight=weights)


def ml_err_192():
    """Sample weight size mismatch."""
    X, y, weights = ml_err_192_prepare_weights()
    ml_err_192_fit_weighted(X, y, weights)


# ml_err_193: n_neighbors too large from init -> fit
def ml_err_193_create_knn(n_neighbors):
    """Create KNN with neighbors > samples."""
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def ml_err_193_fit_knn(knn, X, y):
    """Fit with too many neighbors - error propagates."""
    knn.fit(X, y)


def ml_err_193():
    """n_neighbors (10) > n_samples (3)."""
    knn = ml_err_193_create_knn(n_neighbors=10)
    ml_err_193_fit_knn(knn, [[1], [2], [3]], [0, 1, 0])


# ml_err_194: Negative max_depth from init -> fit
def ml_err_194_create_tree(max_depth):
    """Create decision tree with negative max_depth."""
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=max_depth)


def ml_err_194_fit_tree(tree, X, y):
    """Fit with negative max_depth - error propagates."""
    tree.fit(X, y)


def ml_err_194():
    """Negative max_depth parameter."""
    tree = ml_err_194_create_tree(max_depth=-1)
    ml_err_194_fit_tree(tree, [[1, 2], [3, 4]], [0, 1])


# ml_err_195: Negative n_estimators from init -> fit
def ml_err_195_create_forest(n_estimators):
    """Create random forest with negative n_estimators."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n_estimators)


def ml_err_195_fit_forest(forest, X, y):
    """Fit with negative n_estimators - error propagates."""
    forest.fit(X, y)


def ml_err_195():
    """Negative n_estimators parameter."""
    forest = ml_err_195_create_forest(n_estimators=-1)
    ml_err_195_fit_forest(forest, [[1, 2], [3, 4]], [0, 1])


# ml_err_196: Negative C parameter from init -> fit
def ml_err_196_create_lr(C):
    """Create LogisticRegression with negative C."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C)


def ml_err_196_fit_lr(lr, X, y):
    """Fit with negative C - error propagates."""
    lr.fit(X, y)


def ml_err_196():
    """Negative regularization parameter C."""
    lr = ml_err_196_create_lr(C=-0.5)
    ml_err_196_fit_lr(lr, [[1, 2], [3, 4]], [0, 1])


# ml_err_197: Invalid test_size from split -> validation
def ml_err_197_validate_split(test_size):
    """Validate test size and split."""
    from sklearn.model_selection import train_test_split
    return train_test_split([1, 2, 3], test_size=test_size)


def ml_err_197():
    """test_size > 1 (invalid)."""
    ml_err_197_validate_split(test_size=1.5)


# ml_err_198: Invalid random state from check -> generation
def ml_err_198_check_random(state):
    """Check random state and generate."""
    from sklearn.utils import check_random_state
    rs = check_random_state(state)
    return rs.rand()


def ml_err_198():
    """Invalid random state type."""
    ml_err_198_check_random(state='invalid_state')


# ml_err_199: CV splits > samples from init -> validation
def ml_err_199_validate_cv(cv_splits):
    """Validate CV splits exceed samples."""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    return cross_val_score(LogisticRegression(), [[1], [2]], [0, 1], cv=cv_splits)


def ml_err_199():
    """CV splits (10) > samples (2)."""
    ml_err_199_validate_cv(cv_splits=10)


# ml_err_200: Invalid scoring metric from validation
def ml_err_200_validate_score(scoring):
    """Validate scoring metric."""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    return cross_val_score(LogisticRegression(), [[1, 2], [3, 4]], [0, 1], scoring=scoring)


def ml_err_200():
    """Invalid scoring metric name."""
    ml_err_200_validate_score(scoring='invalid')


__all__ = [f'ml_err_{i}' for i in range(181, 201)]


__all__ = [f'ml_err_{i}' for i in range(181, 201)]
