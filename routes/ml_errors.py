"""
ML Error Endpoints - 100+ Machine Learning Errors
Covers: scikit-learn, XGBoost, LightGBM, ensemble methods, preprocessing, 
validation, metrics, and traditional ML operations
"""

from fastapi import FastAPI
from utils.logger import log
import traceback
from error_propagation.error_scenarios import (
    ScenarioContext, DataProcessor, ModelWrapper, PipelineManager
)


def clean_traceback():
    tb = "".join(traceback.format_exc())
    return tb.replace("Traceback (most recent call last):", "").strip()


def register_ml_errors(app: FastAPI):

    # ============================================================
    # ML ERROR CLASS: Data Handling & Preprocessing (ml_err_01 - ml_err_30)
    # ============================================================

    def ml_err_01():
        """Shape mismatch in preprocessing"""
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 3)  # wrong features
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)

    def ml_err_02():
        """Categorical encoding unseen value"""
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(['cat', 'dog', 'bird'])
        encoder.transform(['cat', 'elephant'])  # unseen

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
        X_scaled = scaler.transform(X)  # not fit yet

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
        mlb.transform([[1, 4]])  # 4 not seen

    def ml_err_08():
        """Polynomial features wrong shape"""
        import numpy as np
        from sklearn.preprocessing import PolynomialFeatures
        X = np.array([[1, 2], [3, 4]])
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        X_test = np.array([1, 2, 3])  # wrong dims
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
        y = np.random.randn(95)  # mismatched
        X_train, X_test, y_train, y_test = train_test_split(X, y)

    def ml_err_14():
        """Pipeline wrong number of steps"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('scaler2', StandardScaler()),  # duplicate
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
            pass  # only 1 positive sample

    def ml_err_17():
        """Feature scaling zero variance"""
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        X = np.array([[1, 5], [1, 5], [1, 5]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Check for inf/nan
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
        import numpy as np
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

    # ============================================================
    # ML ERROR CLASS: Model Training & Fitting (ml_err_31 - ml_err_60)
    # ============================================================

    def ml_err_31():
        """Linear regression X y length mismatch"""
        import numpy as np
        from sklearn.linear_model import LinearRegression
        X = np.random.randn(100, 5)
        y = np.random.randn(95)
        model = LinearRegression()
        model.fit(X, y)

    def ml_err_32():
        """Logistic regression invalid solver"""
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(solver='invalid_solver')

    def ml_err_33():
        """SVM invalid kernel"""
        from sklearn.svm import SVC
        clf = SVC(kernel='invalid_kernel')

    def ml_err_34():
        """Decision tree max_depth type error"""
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth='not_int')

    def ml_err_35():
        """Random forest n_estimators negative"""
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=-5)

    def ml_err_36():
        """Gradient boosting invalid loss"""
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(loss='invalid_loss')

    def ml_err_37():
        """KNeighbors n_neighbors > samples"""
        import numpy as np
        from sklearn.neighbors import KNeighborsClassifier
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        knn = KNeighborsClassifier(n_neighbors=50)
        knn.fit(X, y)
        knn.predict(X[:1])

    def ml_err_38():
        """Clustering invalid n_clusters"""
        import numpy as np
        from sklearn.cluster import KMeans
        X = np.random.randn(50, 5)
        kmeans = KMeans(n_clusters=0)
        kmeans.fit(X)

    def ml_err_39():
        """GMM invalid n_components"""
        import numpy as np
        from sklearn.mixture import GaussianMixture
        X = np.random.randn(50, 5)
        gmm = GaussianMixture(n_components=-5)
        gmm.fit(X)

    def ml_err_40():
        """IsolationForest invalid contamination"""
        import numpy as np
        from sklearn.ensemble import IsolationForest
        X = np.random.randn(50, 5)
        iso = IsolationForest(contamination=1.5)
        iso.fit(X)

    def ml_err_41():
        """OneClassSVM invalid nu"""
        from sklearn.svm import OneClassSVM
        clf = OneClassSVM(nu=1.5)

    def ml_err_42():
        """Ridge regression alpha negative"""
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=-1.0)

    def ml_err_43():
        """Lasso invalid max_iter"""
        from sklearn.linear_model import Lasso
        model = Lasso(max_iter='not_int')

    def ml_err_44():
        """ElasticNet invalid l1_ratio"""
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(l1_ratio=1.5)

    def ml_err_45():
        """BaggingClassifier base_estimator None"""
        from sklearn.ensemble import BaggingClassifier
        bc = BaggingClassifier(base_estimator=None)

    def ml_err_46():
        """AdaBoost invalid algorithm"""
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(algorithm='invalid')

    def ml_err_47():
        """VotingClassifier empty estimators"""
        from sklearn.ensemble import VotingClassifier
        vc = VotingClassifier(estimators=[])

    def ml_err_48():
        """StackingClassifier invalid estimator"""
        import numpy as np
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        sc = StackingClassifier(
            estimators=[('lr', LogisticRegression()), ('invalid', None)],
            final_estimator=LogisticRegression()
        )
        sc.fit(X, y)

    def ml_err_49():
        """MultiOutput invalid target"""
        import numpy as np
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.linear_model import LogisticRegression
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        mo = MultiOutputClassifier(LogisticRegression())
        mo.fit(X, y)  # y should be 2D

    def ml_err_50():
        """NearestNeighbors invalid metric"""
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(metric='invalid_metric')

    def ml_err_51():
        """Isotonic regression sample_weight mismatch"""
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        X = np.random.randn(50, 1)
        y = np.random.randn(50)
        ir = IsotonicRegression()
        ir.fit(X, y, sample_weight=np.random.randn(40))

    def ml_err_52():
        """PassiveAggressiveClassifier invalid loss"""
        from sklearn.linear_model import PassiveAggressiveClassifier
        pac = PassiveAggressiveClassifier(loss='invalid')

    def ml_err_53():
        """SGDClassifier invalid penalty"""
        from sklearn.linear_model import SGDClassifier
        sgd = SGDClassifier(penalty='invalid_penalty')

    def ml_err_54():
        """MLPClassifier invalid hidden_layer_sizes"""
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes='invalid')

    def ml_err_55():
        """ExtraTreesClassifier invalid n_estimators"""
        from sklearn.ensemble import ExtraTreesClassifier
        etc = ExtraTreesClassifier(n_estimators='not_int')

    def ml_err_56():
        """HistGradientBoostingClassifier max_depth mismatch"""
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        hgb = HistGradientBoostingClassifier(max_depth=0)
        hgb.fit(X, y)

    def ml_err_57():
        """Quadratic Discriminant invalid reg_param"""
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        qda = QuadraticDiscriminantAnalysis(reg_param=-1.0)

    def ml_err_58():
        """Naive Bayes with negative alpha"""
        from sklearn.naive_bayes import MultinomialNB
        nb = MultinomialNB(alpha=-1.0)

    def ml_err_59():
        """Huber regressor invalid epsilon"""
        from sklearn.linear_model import HuberRegressor
        hr = HuberRegressor(epsilon='not_float')

    def ml_err_60():
        """TheilSen regressor max_subpopulation negative"""
        from sklearn.linear_model import TheilSenRegressor
        ts = TheilSenRegressor(max_subpopulation=-1)

    # ============================================================
    # ML ERROR CLASS: Metrics & Evaluation (ml_err_61 - ml_err_85)
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

    # ============================================================
    # ML ERROR CLASS: Advanced Scenarios (ml_err_86 - ml_err_100)
    # ============================================================

    def ml_err_86():
        """Pipeline with unfitted transformer"""
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
        X_test = np.random.randn(10, 3)  # wrong shape
        pipe.predict(X_test)

    def ml_err_87():
        """Nested CV with incompatible splits"""
        import numpy as np
        from sklearn.model_selection import cross_validate, StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        
        X = np.random.randn(20, 5)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        cv_outer = StratifiedKFold(n_splits=10)
        cv_results = cross_validate(LogisticRegression(), X, y, cv=cv_outer)

    def ml_err_88():
        """Random search with empty param_dist"""
        import numpy as np
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.linear_model import LogisticRegression
        
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        param_dist = {}
        rs = RandomizedSearchCV(LogisticRegression(), param_dist)
        rs.fit(X, y)

    def ml_err_89():
        """Permutation importance with incompatible scorer"""
        import numpy as np
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LogisticRegression
        
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        model = LogisticRegression()
        model.fit(X, y)
        
        perm_importance = permutation_importance(model, X, y, scoring='invalid_scorer')

    def ml_err_90():
        """Partial dependence plot invalid features"""
        import numpy as np
        from sklearn.inspection import partial_dependence
        from sklearn.tree import DecisionTreeClassifier
        
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        model = DecisionTreeClassifier()
        model.fit(X, y)
        
        pd = partial_dependence(model, X, features=[100])

    def ml_err_91():
        """SHAP values on unfitted model"""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        model = LogisticRegression()
        # missing fit
        try:
            # This would need shap library
            _ = model.predict(X)
        except:
            raise RuntimeError("Model not fitted")

    def ml_err_92():
        """Feature extraction invalid input"""
        import numpy as np
        from sklearn.feature_extraction import FeatureHasher
        
        hasher = FeatureHasher(n_features=128, input_type='dict')
        data = "invalid_input"
        transformed = hasher.transform(data)

    def ml_err_93():
        """Text vectorizer on numeric data"""
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vec = TfidfVectorizer()
        X = np.random.randn(10, 5)
        vec.fit(X)

    def ml_err_94():
        """Word2Vec incompatible vocabulary"""
        # This would need gensim, but simulating the error
        try:
            import gensim.models
            sentences = [['a', 'b'], ['c', 'd']]
            model = gensim.models.Word2Vec(sentences, vector_size=10)
            vec = model.wv['z']  # word not in vocab
        except:
            # Simulating error
            raise KeyError("word not in vocabulary")

    def ml_err_95():
        """Collaborative filtering missing rating"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Simulate user-item matrix with missing values
        ratings = np.array([[5, 3, np.nan], [4, np.nan, 2]])
        if np.any(np.isnan(ratings)):
            similarities = cosine_similarity(ratings)
            # This will produce NaN similarities

    def ml_err_96():
        """Recommender system cold start"""
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        
        # Training data
        X_train = np.random.randn(100, 10)
        
        # New user with no history
        X_new = np.random.randn(1, 10)
        
        nbrs = NearestNeighbors(n_neighbors=50)
        nbrs.fit(X_train)
        
        distances, indices = nbrs.kneighbors(X_new)
        if len(indices[0]) == 0:
            _ = 1 / 0

    def ml_err_97():
        """Imbalanced learning with no minority samples"""
        import numpy as np
        from imblearn.over_sampling import SMOTE
        
        X = np.random.randn(100, 5)
        y = np.zeros(100)  # all same class
        
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

    def ml_err_98():
        """Active learning uncertainty calculation"""
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        model = RandomForestClassifier()
        model.fit(X, y)
        
        # Get prediction probabilities
        probs = model.predict_proba(X)
        
        # Calculate uncertainty - can be NaN if all same class
        uncertainty = 1 - np.max(probs, axis=1)
        if np.all(uncertainty == 0):
            _ = 1 / np.min(uncertainty[uncertainty > 0])

    def ml_err_99():
        """Hyperparameter optimization callback error"""
        import numpy as np
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        param_grid = {'C': [0.1, 1, 10]}
        
        def callback(params):
            raise RuntimeError("Callback error")
        
        gs = GridSearchCV(LogisticRegression(), param_grid)
        gs.fit(X, y)

    def ml_err_100():
        """Multi-task learning dimension mismatch"""
        import numpy as np
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.linear_model import LinearRegression
        
        X = np.random.randn(50, 5)
        y_task1 = np.random.randn(50, 1)
        y_task2 = np.random.randn(40, 1)  # mismatched
        y_multi = np.hstack([y_task1, y_task2])
        
        mor = MultiOutputRegressor(LinearRegression())
        mor.fit(X, y_multi)

    # ============================================================
    # REGISTRATION ROUTES
    # ============================================================

    @app.get("/ml/run-all")
    def ml_run_all():
        """Execute all ML error scenarios"""
        functions = [
            ml_err_01, ml_err_02, ml_err_03, ml_err_04, ml_err_05,
            ml_err_06, ml_err_07, ml_err_08, ml_err_09, ml_err_10,
            ml_err_11, ml_err_12, ml_err_13, ml_err_14, ml_err_15,
            ml_err_16, ml_err_17, ml_err_18, ml_err_19, ml_err_20,
            ml_err_21, ml_err_22, ml_err_23, ml_err_24, ml_err_25,
            ml_err_26, ml_err_27, ml_err_28, ml_err_29, ml_err_30,
            ml_err_31, ml_err_32, ml_err_33, ml_err_34, ml_err_35,
            ml_err_36, ml_err_37, ml_err_38, ml_err_39, ml_err_40,
            ml_err_41, ml_err_42, ml_err_43, ml_err_44, ml_err_45,
            ml_err_46, ml_err_47, ml_err_48, ml_err_49, ml_err_50,
            ml_err_51, ml_err_52, ml_err_53, ml_err_54, ml_err_55,
            ml_err_56, ml_err_57, ml_err_58, ml_err_59, ml_err_60,
            ml_err_61, ml_err_62, ml_err_63, ml_err_64, ml_err_65,
            ml_err_66, ml_err_67, ml_err_68, ml_err_69, ml_err_70,
            ml_err_71, ml_err_72, ml_err_73, ml_err_74, ml_err_75,
            ml_err_76, ml_err_77, ml_err_78, ml_err_79, ml_err_80,
            ml_err_81, ml_err_82, ml_err_83, ml_err_84, ml_err_85,
            ml_err_86, ml_err_87, ml_err_88, ml_err_89, ml_err_90,
            ml_err_91, ml_err_92, ml_err_93, ml_err_94, ml_err_95,
            ml_err_96, ml_err_97, ml_err_98, ml_err_99, ml_err_100,
        ]

        for fn in functions:
            try:
                fn()
            except Exception:
                log(clean_traceback(), level="ERROR")

        return {"status": "done", "class": "ml", "executed": len(functions)}
