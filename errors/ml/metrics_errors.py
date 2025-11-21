"""
ML Metrics & Evaluation Errors (ml_err_61 - ml_err_85)
Covers: Metric computation, dimension mismatches, invalid parameters, distance metrics
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Metrics & Evaluation Errors (61 - 85)
# ============================================================

def ml_err_61():
    """Accuracy score length mismatch"""
    from sklearn.metrics import accuracy_score
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2]
    score = accuracy_score(y_true, y_pred)


def ml_err_62():
    """Precision score invalid average"""
    from sklearn.metrics import precision_score
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 1]
    score = precision_score(y_true, y_pred, average='invalid')


def ml_err_63():
    """Recall score zero division"""
    from sklearn.metrics import recall_score
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    score = recall_score(y_true, y_pred)
    if score == 0:
        _ = 1 / (score - 1)


def ml_err_64():
    """F1 score undefined"""
    from sklearn.metrics import f1_score
    y_true = []
    y_pred = []
    score = f1_score(y_true, y_pred)


def ml_err_65():
    """ROC AUC invalid y_score"""
    from sklearn.metrics import roc_auc_score
    y_true = [0, 1, 1, 0]
    y_score = "invalid"
    roc_auc_score(y_true, y_score)


def ml_err_66():
    """Confusion matrix invalid format"""
    from sklearn.metrics import confusion_matrix
    y_true = [0, 1, 2]
    y_pred = [0, 1]
    cm = confusion_matrix(y_true, y_pred)


def ml_err_67():
    """Mean squared error invalid multioutput"""
    from sklearn.metrics import mean_squared_error
    y_true = [[1, 2], [3, 4]]
    y_pred = [[1, 2], [3, 4]]
    mse = mean_squared_error(y_true, y_pred, multioutput='invalid')


def ml_err_68():
    """Log loss invalid y_pred"""
    from sklearn.metrics import log_loss
    y_true = [0, 1, 1, 0]
    y_pred = [0.5, 1.5]
    loss = log_loss(y_true, y_pred)


def ml_err_69():
    """Silhouette score invalid n_samples"""
    import numpy as np
    from sklearn.metrics import silhouette_score
    X = np.random.randn(10, 5)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    score = silhouette_score(X, labels)
    if score > 0:
        _ = 1 / (1 - score)


def ml_err_70():
    """Calinski Harabasz score single cluster"""
    import numpy as np
    from sklearn.metrics import calinski_harabasz_score
    X = np.random.randn(10, 5)
    labels = np.zeros(10)
    score = calinski_harabasz_score(X, labels)


def ml_err_71():
    """Davies Bouldin score invalid"""
    import numpy as np
    from sklearn.metrics import davies_bouldin_score
    X = np.random.randn(10, 5)
    labels = np.zeros(10)
    score = davies_bouldin_score(X, labels)


def ml_err_72():
    """Matthews correlation coefficient mismatch"""
    from sklearn.metrics import matthews_corrcoef
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    mcc = matthews_corrcoef(y_true, y_pred)


def ml_err_73():
    """Hamming loss invalid input"""
    from sklearn.metrics import hamming_loss
    y_true = "invalid"
    y_pred = [0, 1, 1, 0]
    loss = hamming_loss(y_true, y_pred)


def ml_err_74():
    """Jaccard score invalid average"""
    from sklearn.metrics import jaccard_score
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 1]
    score = jaccard_score(y_true, y_pred, average='invalid')


def ml_err_75():
    """Hinge loss insufficient samples"""
    from sklearn.metrics import hinge_loss
    y_true = [1]
    y_pred = [1.5]
    loss = hinge_loss(y_true, y_pred)


def ml_err_76():
    """Pairwise distances invalid metric"""
    import numpy as np
    from sklearn.metrics.pairwise import pairwise_distances
    X = np.random.randn(10, 5)
    dist = pairwise_distances(X, metric='invalid_metric')


def ml_err_77():
    """Cosine distances incompatible shapes"""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_distances
    X = np.random.randn(10, 5)
    Y = np.random.randn(5, 3)
    dist = cosine_distances(X, Y)


def ml_err_78():
    """Euclidean distances dimension mismatch"""
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    X = np.random.randn(10, 5)
    Y = np.random.randn(5, 3)
    dist = euclidean_distances(X, Y)


def ml_err_79():
    """Manhattan distances shape incompatible"""
    import numpy as np
    from sklearn.metrics.pairwise import manhattan_distances
    X = np.random.randn(10, 5)
    Y = np.random.randn(8)
    dist = manhattan_distances(X, Y)


def ml_err_80():
    """Classification report empty predictions"""
    from sklearn.metrics import classification_report
    y_true = []
    y_pred = []
    report = classification_report(y_true, y_pred)


def ml_err_81():
    """AUC invalid format"""
    from sklearn.metrics import auc
    x = [1, 2, 3]
    y = [4, 5]
    a = auc(x, y)


def ml_err_82():
    """Precision recall curve invalid y_score"""
    from sklearn.metrics import precision_recall_curve
    y_true = [0, 1, 1, 0]
    y_score = "invalid"
    precision_recall_curve(y_true, y_score)


def ml_err_83():
    """RocCurveDisplay plot_roc_auc invalid"""
    from sklearn.metrics import RocCurveDisplay
    y_true = [0, 1, 1, 0]
    y_pred = "invalid"
    display = RocCurveDisplay.from_predictions(y_true, y_pred)


def ml_err_84():
    """ConfusionMatrixDisplay invalid"""
    from sklearn.metrics import ConfusionMatrixDisplay
    y_true = [0, 1]
    y_pred = [0, 1, 2]
    cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)


def ml_err_85():
    """Mutual information invalid"""
    from sklearn.metrics import mutual_info_score
    labels_true = [0, 0, 1, 1]
    labels_pred = [0, 1]
    mi = mutual_info_score(labels_true, labels_pred)


# Registry
__all__ = [f'ml_err_{i:02d}' for i in range(61, 86)]
