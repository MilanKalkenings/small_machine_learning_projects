import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import copy
import pickle
from sklearn.metrics import mean_absolute_error
import optuna
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from typing import Tuple


class Block(nn.Module):
    def __init__(self, width: int):
        """
        mlp block with rezero-gated residual connection
        """
        super().__init__()
        self.linear = nn.Linear(in_features=width, out_features=width)
        self.relu = nn.GELU()  # GELU seems to outperform other activation functions here
        self.rezero = nn.Parameter(torch.zeros([1]))
        self.seq = nn.Sequential(self.linear, self.relu)

    def forward(self, x):
        return x + self.rezero * self.seq(x)


class NeuralNetwork(nn.Module):
    def __init__(self, boost: bool, depth: int, width: int, in_feats: int = 60):
        """
        neural network that can use simple end-to-end boosting mechanism
        :param boost: uses boosting mechanism if boost=True
        :param in_feats: number of input features
        """
        super().__init__()
        self.boost = boost

        self.first = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=width),
            nn.GELU())

        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.do = nn.Dropout(p=0.2)
        for _ in range(depth):
            self.blocks.append(Block(width=width))
            self.heads.append(nn.Linear(in_features=width, out_features=1))
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        x = self.first(x)
        if self.boost:
            individual_preds = []  # for ablation
        pred = 0
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if self.boost:
                head_pred = self.heads[i](self.do(x))
                pred += head_pred
                individual_preds.append(head_pred.detach())
        if not self.boost:
            pred = self.heads[-1](self.do(x))
        loss = self.loss(pred, y)
        if self.boost:
            return {"loss": loss, "pred": pred, "individual_preds": individual_preds}
        else:
            return {"loss": loss, "pred": pred}


def train_val_cb(lr: float, depth: int, iters: int) -> float:
    x_train, y_train, x_val, y_val = read_data()
    cb = CatBoostRegressor(
        learning_rate=lr,
        depth=depth,
        iterations=iters,
        verbose=False)
    cb.fit(X=x_train.numpy(), y=y_train.numpy())
    preds = cb.predict(x_val.numpy())
    return mean_absolute_error(y_true=y_val.numpy(), y_pred=preds)


def train_val_nn(
        boost: bool,
        depth: int,
        width: int,
        lr: float,
        weight_decay: float,
        nn_patience: int,
        max_epochs: int) -> Tuple[float, NeuralNetwork]:
    """
    :return: val mae, trained neural network
    """
    x_train, y_train, x_val, y_val = read_data()
    neural_network = NeuralNetwork(in_feats=len(x_train[0]), boost=boost, depth=depth, width=width)
    neural_network_best = copy.deepcopy(neural_network)
    optimizer = torch.optim.Adam(params=neural_network.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = torch.inf
    patience = nn_patience
    for epoch in range(max_epochs):
        loss = neural_network(x_train, y_train)["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            loss_val = neural_network(x_val, y_val)["loss"].item()
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience = nn_patience
                neural_network_best = copy.deepcopy(neural_network)
            else:
                patience -= 1
                if patience == 0:
                    print("patience broken at epoch", epoch + 1)
                    break
    return best_val_loss, neural_network_best


def read_data():
    """
    different train/val split in every call due to sample(frac=1) in first line
    """
    df = pd.read_csv("C:/Users/milan/datasets/house_prices.csv").sample(frac=1)
    df = df.dropna(axis=1).drop(["Id"], axis=1)
    df_train = df[:int(len(df) * 0.8)]
    df_val = df[int(len(df) * 0.8):]
    del df
    cat_cols = [col for col in df_train.columns if df_train[col].nunique() < 200]
    # preprocess data
    encoders = []
    for col in cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_train[col] = enc.fit_transform(df_train[col].values.reshape(-1, 1))
        df_val[col] = enc.transform(df_val[col].values.reshape(-1, 1))
        encoders.append(enc)
    scaler = StandardScaler()
    cols = df_train.columns
    df_train = pd.DataFrame(data=scaler.fit_transform(df_train), columns=cols)
    df_val = pd.DataFrame(data=scaler.transform(df_val), columns=cols)
    x_train = torch.tensor(df_train.drop(["SalePrice"], axis=1).values, dtype=torch.float)
    x_val = torch.tensor(df_val.drop(["SalePrice"], axis=1).values, dtype=torch.float)
    y_train = torch.tensor(df_train["SalePrice"].values, dtype=torch.float)
    y_train = torch.reshape(y_train, [len(y_train), 1])
    y_val = torch.tensor(df_val["SalePrice"].values, dtype=torch.float)
    y_val = torch.reshape(y_val, [len(y_val), 1])
    return x_train, y_train, x_val, y_val


class Objective:
    def __init__(
            self,
            n_trials: int = 100,
            max_seeds: int = 5,
            cb_lr_range: tuple = (0.01, 0.03),
            cb_depth_range: tuple = (4, 8),
            cb_iters_range: tuple = (800, 1200),
            nn_lr_range: tuple = (0.001, 0.5),
            nn_weight_decay_range: tuple = (0.05, 0.15),
            nn_depth_range: tuple = (4, 16),
            nn_width_range: tuple = (2, 8),
            nn_patience: int = 500,
            nn_max_epochs: int = 5000,
            max_loss: float = 0.2,
            boost: bool = True):
        """
        :param max_loss: threshold for pruning/aborting a trial
        :param boost: if True, boosted nn is used in objective_nn
        """
        self.n_trials = n_trials
        self.max_seeds = max_seeds
        self.cb_lr_range = cb_lr_range
        self.cb_depth_range = cb_depth_range
        self.cb_iters_range = cb_iters_range
        self.nn_lr_range = nn_lr_range
        self.nn_weight_decay_range = nn_weight_decay_range
        self.nn_depth_range = nn_depth_range
        self.nn_width_range = nn_width_range
        self.nn_patience = nn_patience
        self.nn_max_epochs = nn_max_epochs
        self.max_loss = max_loss
        self.boost = boost

    def set_boost(self, boost: bool):
        self.boost = boost

    def objective_cb(self, trial: optuna.Trial) -> float:
        """
        catboost objective
        :return: val mae (averaged over multiple seeds)
        """
        lr = trial.suggest_float(name="lr", low=self.cb_lr_range[0], high=self.cb_lr_range[1])  # default 0.03
        depth = trial.suggest_int(name="depth", low=self.cb_depth_range[0], high=self.cb_depth_range[1])  # default 6
        iters = trial.suggest_int(name="iters", low=self.cb_iters_range[0], high=self.cb_iters_range[1])  # default 1000
        val_losses = []
        for _ in range(self.max_seeds):
            val_losses.append(train_val_cb(lr=lr, depth=depth, iters=iters))
            if torch.mean(torch.tensor(val_losses)).item() > self.max_loss:
                break
        return torch.mean(torch.tensor(val_losses)).item()

    def objective_nn(self, trial: optuna.Trial) -> float:
        """
        optuna objective for neural network
        :return: val mae (averaged over multiple seeds)
        """
        lr = trial.suggest_float(name="lr", low=self.nn_lr_range[0], high=self.nn_lr_range[1])  # 0.02 works good
        weight_decay = trial.suggest_float(name="weight_decay", low=self.nn_weight_decay_range[0], high=self.nn_weight_decay_range[1])  # 0.1 works good
        depth = trial.suggest_int(name="depth", low=self.nn_depth_range[0], high=self.nn_depth_range[1])  # 8 works good
        width = trial.suggest_int(name="width", low=self.nn_width_range[0], high=self.nn_width_range[1])  # 4 works good
        best_val_losses = []
        for _ in range(self.max_seeds):
            best_val_losses.append(train_val_nn(
                boost=self.boost,
                depth=depth,
                width=width,
                lr=lr,
                weight_decay=weight_decay,
                nn_patience=self.nn_patience,
                max_epochs=self.nn_max_epochs)[0])
            if torch.mean(torch.tensor(best_val_losses)) > self.max_loss:
                break
        return torch.mean(torch.tensor(best_val_losses)).float()


##### determine hp #####
objective = Objective()
# nn-boost
study_nn_boost = optuna.create_study(sampler=optuna.samplers.TPESampler())
study_nn_boost.optimize(func=objective.objective_nn, n_trials=objective.n_trials)
pickle.dump(study_nn_boost, open("../../monitoring/simpleboost_tabnet/study_nn_boost.pkl", "wb"))
# cb baseline
study_cb = optuna.create_study(sampler=optuna.samplers.TPESampler())
study_cb.optimize(func=objective.objective_cb, n_trials=objective.n_trials)
pickle.dump(study_cb, open("../../monitoring/simpleboost_tabnet/study_cb.pkl", "wb"))
# nn baseline
objective.set_boost(boost=False)
study_nn = optuna.create_study(sampler=optuna.samplers.TPESampler())
study_nn.optimize(func=objective.objective_nn, n_trials=objective.n_trials)
pickle.dump(study_nn, open("../../monitoring/simpleboost_tabnet/study_nn.pkl", "wb"))
objective = Objective()
study_cb: optuna.Study = pickle.load(open("../../monitoring/simpleboost_tabnet/study_cb.pkl", "rb"))
cb_best_params = study_cb.best_params
study_nn_boost: optuna.Study = pickle.load(open("../../monitoring/simpleboost_tabnet/study_nn_boost.pkl", "rb"))
nn_boost_params = study_nn_boost.best_params
study_nn: optuna.Study = pickle.load(open("../../monitoring/simpleboost_tabnet/study_nn.pkl", "rb"))
nn_params = study_nn.best_params

##### benchmark #####
cb_val_losses = []
nn_boost_val_losses = []
nn_val_losses = []
for seed in range(10):
    print("seed", seed+1)
    cb_val_losses.append(train_val_cb(
        lr=cb_best_params["lr"],
        depth=cb_best_params["depth"],
        iters=cb_best_params["iters"]))
    nn_boost_val_losses.append(train_val_nn(
        boost=True,
        depth=nn_boost_params["depth"],
        width=nn_boost_params["width"],
        lr=nn_boost_params["lr"],
        weight_decay=nn_boost_params["weight_decay"],
        nn_patience=objective.nn_patience,
        max_epochs=objective.nn_max_epochs)[0])
    nn_val_losses.append(train_val_nn(
        boost=False,
        depth=nn_params["depth"],
        width=nn_params["width"],
        lr=nn_params["lr"],
        weight_decay=nn_params["weight_decay"],
        nn_patience=objective.nn_patience,
        max_epochs=objective.nn_max_epochs))
plt.boxplot([cb_val_losses, nn_boost_val_losses, nn_val_losses], labels=['catboost', 'boosted nn', "nn"])
plt.ylabel("mae (10 seeds)")
plt.title("competing against catboost on tabular data\nafter hpo with 100 optuna trials each")
plt.savefig("results.png")


##### ablation #####
_, nn_boost = train_val_nn(
        boost=True,
        depth=nn_boost_params["depth"],
        width=nn_boost_params["width"],
        lr=nn_boost_params["lr"],
        weight_decay=nn_boost_params["weight_decay"],
        nn_patience=objective.nn_patience,
        max_epochs=objective.nn_max_epochs)
x_train, y_train, _, _ = read_data()
individual_preds = nn_boost(x=x_train, y=y_train)["individual_preds"]
for head_id in range(len(individual_preds)):
    print(torch.mean(torch.abs(individual_preds[head_id])))
# > > > all heads have 0.065+/-0.005 -> contribution to ensemble pred is almost uniform across output heads
