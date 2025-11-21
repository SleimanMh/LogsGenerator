"""
ML Model Training & Fitting Errors (ml_err_31 - ml_err_60)
Covers: Invalid parameters, incompatible configs, unfitted estimators, wrong solvers
"""

import traceback


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


# ============================================================
# Model Training & Fitting Errors (31 - 60)
# ============================================================

def ml_err_31():
    """Linear regression X y length mismatch"""
    import numpy as np
    from sklearn.linear_model import LinearRegression
    X = np.random.randn(100, 5)
    y = np.random.randn(95)
    LinearRegression().fit(X, y)


def ml_err_32():
    """Logistic regression invalid solver"""
    from sklearn.linear_model import LogisticRegression
    # Throws error when fitting, not when constructing
    clf = LogisticRegression(solver='invalid_solver')
    clf.fit([[1, 2]], [0])


def ml_err_33():
    """SVM invalid kernel"""
    from sklearn.svm import SVC
    clf = SVC(kernel='invalid_kernel')
    clf.fit([[1, 2]], [0])


def ml_err_34():
    """Decision tree max_depth type error"""
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth='not_int')
    clf.fit([[1, 2]], [0])


def ml_err_35():
    """Random forest n_estimators negative"""
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=-5)
    clf.fit([[1, 2]], [0])


def ml_err_36():
    """Gradient boosting invalid loss"""
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(loss='invalid_loss')
    clf.fit([[1, 2]], [0])


def ml_err_37():
    """KNeighbors n_neighbors > samples"""
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, 10)
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(X, y)


def ml_err_38():
    """Clustering invalid n_clusters"""
    import numpy as np
    from sklearn.cluster import KMeans
    X = np.random.randn(20, 5)
    KMeans(n_clusters=0).fit(X)


def ml_err_39():
    """GMM invalid n_components"""
    import numpy as np
    from sklearn.mixture import GaussianMixture
    X = np.random.randn(20, 5)
    GaussianMixture(n_components=-5).fit(X)


def ml_err_40():
    """IsolationForest invalid contamination"""
    import numpy as np
    from sklearn.ensemble import IsolationForest
    X = np.random.randn(20, 5)
    IsolationForest(contamination=1.5).fit(X)


def ml_err_41():
    """OneClassSVM invalid nu"""
    from sklearn.svm import OneClassSVM
    clf = OneClassSVM(nu=1.5)
    clf.fit([[1, 2], [2, 3]])


def ml_err_42():
    """Ridge regression alpha negative"""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=-1.0)
    model.fit([[1, 2]], [1])


def ml_err_43():
    """Lasso invalid max_iter"""
    from sklearn.linear_model import Lasso
    model = Lasso(max_iter='not_int')
    model.fit([[1, 2]], [1])


def ml_err_44():
    """ElasticNet invalid l1_ratio"""
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(l1_ratio=1.5)
    model.fit([[1, 2]], [1])


def ml_err_45():
    """BaggingClassifier base_estimator None"""
    from sklearn.ensemble import BaggingClassifier
    model = BaggingClassifier(base_estimator=None)
    model.fit([[1, 2]], [0])


def ml_err_46():
    """AdaBoost invalid algorithm"""
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(algorithm='invalid')
    clf.fit([[1, 2]], [0])


def ml_err_47():
    """VotingClassifier empty estimators"""
    from sklearn.ensemble import VotingClassifier
    vc = VotingClassifier(estimators=[])
    vc.fit([[1, 2]], [0])


def ml_err_48():
    """StackingClassifier invalid estimator"""
    import numpy as np
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, 20)
    StackingClassifier(
        estimators=[("lr", LogisticRegression()), ("bad", None)],
        final_estimator=LogisticRegression()
    ).fit(X, y)


def ml_err_49():
    """MultiOutput invalid target"""
    import numpy as np
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, 20)
    MultiOutputClassifier(LogisticRegression()).fit(X, y)


def ml_err_50():
    """NearestNeighbors invalid metric"""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric='invalid_metric')
    nn.fit([[1, 2], [3, 4]])


def ml_err_51():
    """Isotonic regression sample_weight mismatch"""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression()
    ir.fit(np.random.randn(50), np.random.randn(50), sample_weight=np.random.randn(40))


def ml_err_52():
    """PassiveAggressiveClassifier invalid loss"""
    from sklearn.linear_model import PassiveAggressiveClassifier
    clf = PassiveAggressiveClassifier(loss='invalid')
    clf.fit([[1, 2]], [0])


def ml_err_53():
    """SGDClassifier invalid penalty"""
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(penalty='invalid_penalty')
    clf.fit([[1, 2]], [0])


def ml_err_54():
    """MLPClassifier invalid hidden_layer_sizes"""
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes='invalid')
    clf.fit([[1, 2]], [0])


def ml_err_55():
    """ExtraTrees invalid n_estimators"""
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators='not_int')
    clf.fit([[1, 2]], [0])


def ml_err_56():
    """HistGradientBoostingClassifier invalid depth"""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    HistGradientBoostingClassifier(max_depth=0).fit(X, y)


def ml_err_57():
    """QDA invalid reg_param"""
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    clf = QuadraticDiscriminantAnalysis(reg_param=-1)
    clf.fit([[1, 2]], [0])


def ml_err_58():
    """MultinomialNB negative alpha"""
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=-1)
    clf.fit([[1, 2], [1, 3]], [0, 1])


def ml_err_59():
    """HuberRegressor invalid epsilon"""
    from sklearn.linear_model import HuberRegressor
    clf = HuberRegressor(epsilon='not_float')
    clf.fit([[1, 2], [3, 4]], [1, 0])


def ml_err_60():
    """TheilSen max_subpopulation negative"""
    from sklearn.linear_model import TheilSenRegressor
    model = TheilSenRegressor(max_subpopulation=-1)
    model.fit([[1, 2], [3, 4]], [1, 0])


__all__ = [f"ml_err_{i:02d}" for i in range(31, 61)]
