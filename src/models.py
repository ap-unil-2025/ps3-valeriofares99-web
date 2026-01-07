from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_logistic_regression(X, y):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)
    return model


def train_random_forest(X, y):
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def train_xgboost(X, y):
    model = XGBClassifier(
        eval_metric="logloss",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    return model


def train_knn(X, y):
    model = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    )
    model.fit(X, y)
    return model


def train_naive_bayes(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model


def train_svm(X, y):
    model = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)
    return model