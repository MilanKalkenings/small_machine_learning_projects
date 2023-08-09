import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import RandomForestRegressor
from rwanda_co2_catboost import CatBoostRegressor
import pickle
import optuna
import logging


class Encoder:
    def __init__(self):
        pass

    def fit(self, df: pd.DataFrame):
        self.feature_names = df.columns
        encoders = {}
        for feature_name in self.feature_names:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.fit(df[feature_name].values.reshape(-1, 1))
            encoders[feature_name] = encoder
        self.encoders = encoders

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        preprocessed = []
        for feature_name in self.feature_names:
            feature = df[feature_name].values.reshape(-1, 1)
            feature_encoded = self.encoders[feature_name].transform(feature)
            encoded_feature_names = []
            for i in range(len(feature_encoded[0])):
                encoded_feature_names.append(feature_name + "_" + str(i + 1))
            preprocessed.append(pd.DataFrame(data=feature_encoded,
                                             columns=encoded_feature_names))
        return pd.concat(preprocessed, axis=1)


class Discretizer:
    def __init__(self):
        self.percentiles = {}

    def fit(self, df_numeric: pd.DataFrame):
        for feature_name in df_numeric.columns:
            self.percentiles[feature_name] = [np.percentile(df_numeric[feature_name].dropna(), i * 2) for i in range(1, 51)]  # 2, 52 -> rmse=50

    def transform(self, df_numeric: pd.DataFrame) -> pd.DataFrame:
        transformed_features = np.zeros(shape=df_numeric.shape)
        for i, feature_name in enumerate(df_numeric.columns):
            for j, value in enumerate(df_numeric[feature_name].values):
                found = False
                for percentile in self.percentiles[feature_name]:
                    if np.isnan(value):  # create own category for nans
                        transformed_features[j, i] = self.percentiles[feature_name][0] - 1
                        found = True
                        break
                    elif value < percentile:
                        transformed_features[j, i] = percentile
                        found = True
                        break
                if not found:
                    transformed_features[j, i] = percentile  # biggest percentile
        return pd.DataFrame(data=transformed_features, columns=df_numeric.columns)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


class OptuneObjective:
    def __init__(self,
                 x_train: pd.DataFrame,
                 y_train: np.ndarray,
                 x_val: pd.DataFrame,
                 y_val: np.ndarray,
                 remove_at_least: int = 1,
                 patience: int = 3,
                 min_importance_range=(0.0001, 0.001),
                 lr_range=(0.01, 0.03),
                 depth_range=(4, 10)):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.remove_at_least = remove_at_least
        self.patience = patience

        # to tune:
        self.min_importance_range = min_importance_range
        self.lr_range = lr_range
        self.depth_range = depth_range

    def __call__(self, trial: optuna.Trial):
        """
        risks: if first trial is best trial, features are not saved. very unlikely with >10 trials
        """
        print("trial", trial.number)
        min_importance = trial.suggest_float(name="min_importance",
                                             low=self.min_importance_range[0],
                                             high=self.min_importance_range[1])
        model_class = trial.suggest_categorical(name="model_class", choices=["cb"]) #, "rf"])

        features_used = self.x_train.columns
        features_best = None
        rmse_best = np.inf
        patience_counter = 0
        for i in range(len(features_used)):
            if len(features_used) == 0:
                break
            print("features used:", len(features_used))
            if model_class == "cb":
                lr = trial.suggest_float(name="lr", low=self.lr_range[0], high=self.lr_range[1])
                model = CatBoostRegressor(learning_rate=lr, verbose=False, random_state=0)
            else:
                model = RandomForestRegressor(random_state=0)
            model.fit(self.x_train[features_used], self.y_train)
            pred = model.predict(self.x_val[features_used])
            rmse_val = rmse(y_true=self.y_val, y_pred=pred)
            if rmse_val < rmse_best:
                rmse_best = rmse_val
                features_best = features_used
                patience_counter = 0
                print("best val rmse:", rmse_best)
            else:
                patience_counter += 1
                # print("patience counter:", patience_counter)
                if patience_counter == self.patience:
                    # print("convergence")
                    break

            # decrease feature space
            imp_rel = model.feature_importances_ / model.feature_importances_.sum()
            importances = pd.DataFrame({"feature": features_used,
                                        "importance": imp_rel}).sort_values(by="importance", ascending=False).reset_index(drop=True)
            # print(importances)
            features_used = importances[importances["importance"] > min_importance]["feature"]
            if len(features_used) == len(importances["feature"]):
                features_used = importances["feature"][:-self.remove_at_least]
            min_importance = min_importance * 1.2

        # checkpoint if this trial is the best so far
        if (trial.number) > 0:
            if rmse_best < trial.study.best_value:
                print("checkpointing")
                while True:
                    try:
                        with open("../../monitoring/feature_importance/features_best.pkl", 'wb') as f:
                            pickle.dump(features_best, f)
                        break
                    except:
                        print("try again")
                        time.sleep(1)
        print("\n")
        return rmse_best


cat_cols = ["year", "week_no"]
df_train_val = pd.read_csv("C:/datasets/rwanda_co2/train.csv").sample(frac=1)
df_test = pd.read_csv("C:/datasets/rwanda_co2/test.csv")
ids_test = df_test["ID_LAT_LON_YEAR_WEEK"]
df_test.drop(labels=["ID_LAT_LON_YEAR_WEEK"], inplace=True, axis=1)
df_test.reset_index(inplace=True, drop=True)
df_train_val.drop(labels=["ID_LAT_LON_YEAR_WEEK"], inplace=True, axis=1)

df_train = df_train_val[:int((len(df_train_val) * 0.8))]
df_train.reset_index(inplace=True, drop=True)
df_val = df_train_val[-int((len(df_train_val) * 0.8)):]
df_val.reset_index(inplace=True, drop=True)

# fit_transform training data
y_train = df_train["emission"].values
x_train = df_train.drop(labels=["emission"], inplace=False, axis=1)
x_train_num = x_train.drop(labels=cat_cols, inplace=False, axis=1)
discretizer = Discretizer()
discretizer.fit(df_numeric=x_train_num)
x_train_num_discrete = discretizer.transform(df_numeric=x_train_num)
x_train_full = pd.concat([x_train_num_discrete, x_train[cat_cols]], axis=1)
encoder = Encoder()
encoder.fit(x_train_full)
x_train_onehot = encoder.transform(x_train_full).astype(int)
x_train_onehot = pd.concat([x_train_onehot, x_train_num.fillna(np.nanmin(x_train_num.values) -1000000)], axis=1)

# transform val data
y_val = df_val["emission"].values
x_val = df_val.drop(labels=["emission"], inplace=False, axis=1)
x_val_num = x_val.drop(labels=cat_cols, inplace=False, axis=1)
x_val_num_discrete = discretizer.transform(df_numeric=x_val_num)
x_val_full = pd.concat([x_val_num_discrete, x_val[cat_cols]], axis=1)
x_val_onehot = encoder.transform(x_val_full).astype(int)
x_val_onehot = pd.concat([x_val_onehot, x_val_num.fillna(-1)], axis=1)

# transform test data
x_test_num = df_test.drop(labels=cat_cols, inplace=False, axis=1)
x_test_num_discrete = discretizer.transform(df_numeric=x_test_num)
x_test_full = pd.concat([x_test_num_discrete, df_test[cat_cols]], axis=1)
x_test_onehot = encoder.transform(x_test_full).astype(int)
x_test_onehot = pd.concat([x_test_onehot, x_test_num.fillna(-1)], axis=1)

# logger settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("../../monitoring/feature_importance/log.txt", mode="w"))
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

# feature selection
optuna_objective = OptuneObjective(x_train=x_train_onehot, y_train=y_train, x_val=x_val_onehot, y_val=y_val)
sampler = optuna.samplers.TPESampler()
study = optuna.create_study()
study.optimize(func=optuna_objective, n_trials=100, n_jobs=1)
with open("../frameworks/study.pkl", 'wb') as f:
    pickle.dump(study, f)


# prepare submission
with open("../frameworks/study.pkl", 'rb') as f:
    study = pickle.load(f)
with open("../../monitoring/feature_importance/features_best.pkl", 'rb') as f:
    features_best = pickle.load(f)
ensemble_size = 30
preds = np.zeros(shape=len(df_test))
for i in range(ensemble_size):
    print("train & predict with ensemble member", i + 1)
    params_best = study.best_params
    if params_best["model_class"] == "cb":
        model_best = CatBoostRegressor(learning_rate=params_best["lr"], verbose=False, random_state=i)
    else:
        model_best = RandomForestRegressor(random_state=i)
    model_best.fit(X=pd.concat((x_train_onehot[features_best], x_val_onehot[features_best]), axis=0), y=np.concatenate((y_train, y_val)))  # retrain on train & val
    pred = model_best.predict(x_test_onehot[features_best])
    preds += pred
emission = pd.Series(data=(preds / ensemble_size), name="emission")
df = pd.DataFrame({"ID_LAT_LON_YEAR_WEEK": ids_test, "emission": emission})
print(df)
df.to_csv("submission.csv", index=False)


# final output:
# Trial 99 finished with value: 34.19925840808192 and parameters: {'min_importance': 0.00037848794022517685, 'model_class': 'cb', 'lr': 0.028945981936908564}. Best is trial 22 with value: 33.495643730215434.