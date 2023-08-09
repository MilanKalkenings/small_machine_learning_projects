import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import regex as re
import string


class DataHandler:
    def __init__(self, max_instances: int, root: str):
        x, y = self.load_x_y(root=root, max_instances=max_instances)
        x_clean = self.clean(x=x)
        x_final = self.engineer_features(x=x_clean)
        x_train, x_test, y_train, y_test = train_test_split(x_final, y, test_size=0.2)

        # fit scaler on train, transform on train and test (no data snooping)
        scaler = StandardScaler()
        scaler.fit(X=x_train)
        x_train.loc[:, :] = scaler.transform(x_train)
        x_test.loc[:, :] = scaler.transform(x_test)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def fraction_secure(numerator: float, denominator: float):
        if denominator == 0:
            return 0
        return numerator / denominator

    @staticmethod
    def clean(x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(["date", "subject"], axis=1)

    @staticmethod
    def x_y_split(
            data: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.Series]:
        x = data.drop("is_fake", axis=1)
        y = data["is_fake"]
        return x, y

    @staticmethod
    def words_from_text(text: str) -> List[str]:
        text_lower = text.lower()
        return ''.join(char for char in text_lower if char not in set(string.punctuation)).split()

    @staticmethod
    def count_char(text: str, char: str) -> int:
        return re.subn(fr"\{char}", '', text)[1]

    def get_data_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.x_train, self.y_train

    def get_data_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.x_test, self.y_test

    def engineer_features(self, x: pd.DataFrame) -> pd.DataFrame:
        # univariate transformations:
        cols_done_uni = []
        cols = ["text", "title"]
        for col in cols:
            lexical_diversity = x[col].apply(self.lexical_diversity)
            lexical_diversity.name = col + "_lexical_diversity"
            cols_done_uni.append(lexical_diversity)

            num_words = x[col].apply(self.num_words)
            num_words.name = col + "_num_words"
            cols_done_uni.append(num_words)

            num_exclamation = x[col].apply(self.count_char, args=["!"])
            num_exclamation.name = col + "_num_exclamation"
            cols_done_uni.append(num_exclamation)

            num_question = x[col].apply(self.count_char, args=["?"])
            num_question.name = col + "_num_question"
            cols_done_uni.append(num_question)
        data_uni = pd.concat(cols_done_uni, axis=1)

        # multivariate transformations:
        relative_title_length = data_uni["title_num_words"] / (data_uni["text_num_words"] + data_uni["title_num_words"])
        relative_title_length.name = "relative_title_length"
        return pd.concat([data_uni, relative_title_length], axis=1)

    def lexical_diversity(self, text: str) -> float:
        words = self.words_from_text(text=text)
        return self.fraction_secure(numerator=len(set(words)), denominator=len(words))

    def num_words(self, text: str) -> int:
        words = self.words_from_text(text=text)
        return len(words)

    def load_x_y(
            self,
            root: str,
            max_instances: int) \
            -> Tuple[pd.DataFrame, pd.Series]:
        fake_news = pd.read_csv(root + "/fake.csv")
        real_news = pd.read_csv(root + "/true.csv")
        fake_news["is_fake"] = 1
        real_news["is_fake"] = 0
        data = pd.concat([fake_news, real_news]).sample(frac=1).reset_index(drop=True).head(max_instances)
        x, y = self.x_y_split(data=data)
        return x, y






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


# stacking ensemble
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

