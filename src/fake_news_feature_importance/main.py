from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from data_handling import DataHandler


def train_eval(
        estimator: BaseEstimator,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series) -> Tuple[float, float, RandomForestClassifier]:
    estimator.fit(X=x_train, y=y_train)
    preds_test = estimator.predict(X=x_test)
    accuracy_test = accuracy_score(y_true=y_test, y_pred=preds_test)
    f1_test = f1_score(y_true=y_test, y_pred=preds_test)
    return f1_test, accuracy_test, estimator


random_state = 1
max_instances = 5000  # available: 44898
monitoring_path = "../../monitoring"
data_path = "../../data/fake_news"

# feature engineering
data_handler = DataHandler(max_instances=max_instances, root=data_path)
x_train, y_train = data_handler.get_data_train()
x_test, y_test = data_handler.get_data_test()
data_train = pd.concat([x_train, y_train], axis=1)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(np.abs(data_train.corr()), annot=True)
plt.title("absolute correlation")
plt.tight_layout()
plt.savefig(monitoring_path + "/correlation")
plt.clf()


# normal training
random_forest = RandomForestClassifier(random_state=random_state)
f1_test, acc_test, _ = train_eval(
    estimator=random_forest,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test)
print("after normal training")
print("f1 test", f1_test)
print("accuracy test", acc_test)


# hyperparameter optimization
grid = {"n_estimators": [100, 200, 300], "min_samples_leaf": [1, 2, 4]}
grid_search = GridSearchCV(
    estimator=random_forest,
    param_grid=grid,
    cv=5,
    scoring="f1")
grid_search.fit(X=x_train, y=y_train)
best_params = grid_search.best_params_
random_forest = RandomForestClassifier(
    min_samples_leaf=best_params["min_samples_leaf"],
    n_estimators=best_params["n_estimators"],
    random_state=random_state)
f1_test, acc_test, _ = train_eval(
    estimator=random_forest,
    x_train=x_train,
    x_test=x_test,
    y_test=y_test,
    y_train=y_train)
print("after hyperparameter optimization")
print("f1 test", f1_test)
print("accuracy test", acc_test)


# ensembling
estimators = [
    ("rf", random_forest),  # reuse
    ("svc", SVC()),
    ("lr", LogisticRegression())
]
stacking_ensemble = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression())
f1_test, acc_test, _ = train_eval(
    estimator=stacking_ensemble,
    x_train=x_train,
    x_test=x_test,
    y_test=y_test,
    y_train=y_train)
print("after ensembling")
print("f1 test", f1_test)
print("accuracy test", acc_test)


# feature importance
feature_importances = pd.Series(random_forest.feature_importances_, index=x_train.columns).sort_values(ascending=False)
feature_importances.plot()
plt.title(f"random forest feature importance; test f1: {round(f1_test, 2)}")
plt.xticks(rotation=90)
plt.savefig(monitoring_path + "/feature_importance")
plt.clf()


# iteratively add features and calculate score
f1_scores = [0]
x_ticks = ["no features"]
features_used = []
for i in range(len(feature_importances)):
    x_ticks.append("add " + feature_importances.index[i] + "\n(importance " + str(round(feature_importances[i], 2)) + ")")
    features_used.append(feature_importances.index[i])
    random_forest.fit(X=x_train[features_used], y=y_train)
    preds_test = random_forest.predict(X=x_test[features_used])
    f1_scores.append(f1_score(y_true=y_test, y_pred=preds_test))

plt.plot(x_ticks, f1_scores)
plt.title("fake news detection with random forest")
plt.xticks(rotation=90)
plt.ylabel("test f score")
plt.xlabel("number of features used")
plt.savefig(monitoring_path + "/f1")

