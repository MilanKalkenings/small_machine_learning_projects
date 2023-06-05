import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from data_handling import DataHandler

max_instances = 5000  # available: 44898
monitoring_path = "../../monitoring"
data_path = "../../data/fake_news"

data_handler = DataHandler(max_instances=max_instances, root=data_path)
x_train, y_train = data_handler.get_data_train()
x_test, y_test = data_handler.get_data_test()

# absolute correlation
data_train = pd.concat([x_train, y_train], axis=1)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(np.abs(data_train.corr()), annot=True)
plt.title("absolute correlation")
plt.tight_layout()
plt.savefig(monitoring_path + "/correlation")
plt.clf()  # clean

# sklearn random forest
random_forest = RandomForestClassifier(random_state=1)
random_forest.fit(X=x_train, y=y_train)
preds_test = random_forest.predict(X=x_test)

accuracy_test = accuracy_score(y_true=y_test, y_pred=preds_test)
f1_test = f1_score(y_true=y_test, y_pred=preds_test)
print("f1 test", f1_test, "accuracy test", accuracy_test)

feature_importances = pd.Series(random_forest.feature_importances_, index=x_train.columns).sort_values(ascending=False)
feature_importances.plot()
plt.title(f"random forest feature importance; test f1: {round(f1_test, 2)}")
plt.xticks(rotation=90)
plt.savefig(monitoring_path + "/feature_importance")
plt.clf()

f1_scores = []
features_used = []
for feature in feature_importances.index:
    features_used.append(feature)

    random_forest = RandomForestClassifier(random_state=1)
    random_forest.fit(X=x_train[features_used], y=y_train)
    preds_test = random_forest.predict(X=x_test[features_used])
    f1_scores.append(f1_score(y_true=y_test, y_pred=preds_test))

plt.plot(range(1, len(f1_scores) + 1), f1_scores)
plt.title("fake news detection with random forest")
plt.ylabel("test f score")
plt.xlabel("number of features used")
plt.savefig(monitoring_path + "/f1")

