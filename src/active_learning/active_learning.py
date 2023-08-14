import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from typing import List
from sklearn.cluster import KMeans
import random
import copy
from sklearn.preprocessing import LabelEncoder


def maxargs(values: np.ndarray, n: int):
    """
    :return: indices of the <n> biggest entries in <values>
    """
    return np.argsort(values)[::-1][:n]


def determine_start_indices_kmeans(x: np.ndarray, startsize: int) -> List[int]:
    """
    uses kmeans clustering to determine representative instances in <x>.

    creates <startsize> many clusters of <x>, and returns
    indices of instances in <x> closest to cluster centers.
    follows "exploration" paradigm,
    because instances in <x> closest to cluster
    centers approximately represent <x>.
    """
    kmeans = KMeans(n_clusters=startsize)  # startsize < endsize (x has endsize many entries)
    kmeans.fit(x)
    centers = kmeans.cluster_centers_
    indices = []
    for i in range(startsize):
        differences_to_center = np.linalg.norm(x - centers[i], axis=1)
        min_difference_id = np.argmin(differences_to_center)
        indices.append(int(min_difference_id))
    return list(set(indices))


def extend_random_indices_(x: np.ndarray, indices_used: List[int], extend_size: int, cls: RandomForestClassifier) -> List[int]:
    """
    extends <extend_size> many randomly chosen new indices to <indices_used>.
    inplace.
    """
    candidates = list(set(list(range(len(x)))) - set(indices_used))
    candidates = list(torch.tensor(candidates)[torch.randperm(n=len(candidates))])
    indices_used.extend(candidates[:extend_size])
    return indices_used


def extend_min_confidence_indices_(x: np.ndarray, indices_used: List[int], extend_size: int, cls: RandomForestClassifier) -> List[int]:
    """
    extends the <extend_size> many indices to <indices_used>
    that have the lowest prediction confidence and are not already in <indices_used>,
    when predicting with <cls> trained on <indices_used>.
    inplace.
    """
    indices_unused = list(set(range(len(x))) - set(indices_used))
    probas = cls.predict_proba(x[indices_unused])
    max_logits = np.max(probas, axis=1)
    lowest_confidence_indices_unused = maxargs(max_logits * (-1), n=extend_size)
    lowest_confidence_indices_global = torch.tensor(indices_unused)[lowest_confidence_indices_unused.copy()]
    return indices_used + list(lowest_confidence_indices_global)


def active_learning(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        stepsize: int,
        endsize: int,
        indices_start: List[int],
        labeling_policy,
        seed: int) -> List[float]:
    """
    :param stepsize: number of new instances added to (<x_train>, <y_train>) per active learning iteration
    :param endsize: number of (<x_train>, <y_train>) instances used in last active learning iteration
    :param indices_start: indices of (<x_train>, <y_train>) used as a starting point
    :param labeling_policy: function that determines indices of (<x_train>, <y_train>) in next iteration
    :param seed: seed for random forest
    :return: accuracy scores on (<x_val>, <y_val>)
    """
    accs = []
    indices_used = indices_start
    for _ in torch.arange(start=len(indices_start), step=stepsize, end=endsize):
        # print(len(indices_used))
        cls = RandomForestClassifier(random_state=seed, n_estimators=30)
        cls.fit(x_train[indices_used], y_train[indices_used])
        accs.append(accuracy_score(y_true=y_val, y_pred=cls.predict(x_val)))
        indices_used = labeling_policy(x=x_train, indices_used=indices_used, extent_size=stepsize, cls=cls)
    return accs


##### hyperparameters #####
stepsize = 25  # 5
startsize = 10  # 100
endsize = 200  # < 1718
val_size = 1000
n_seeds = 40


##### load data #####
# https://www.kaggle.com/datasets/amirhosseinmirzaie/pistachio-types-detection
df = pd.read_csv("pistachio.csv").sample(frac=1)
label_encoder = LabelEncoder()
df["Class"] = label_encoder.fit_transform(df["Class"])
df_train = df[:endsize]
df_val = df[endsize:endsize + val_size]
del df
inputs_train = df_train.drop(["Class"], axis=1).values
targets_train = df_train["Class"].values
del df_train
inputs_val = df_val.drop(["Class"], axis=1).values
targets_val = df_val["Class"].values
del df_val


##### active learning #####
indices_start = list(range(startsize))  # determine_start_indices_kmeans(x=inputs_train, startsize=startsize)
accs_random = []
accs_max_entropy = []
accs_min_confidence = []
for seed in range(n_seeds):
    random.seed(seed)
    print("seed", seed + 1)
    accs_random.append(active_learning(
        x_train=inputs_train,
        y_train=targets_train,
        x_val=inputs_val,
        y_val=targets_val,
        stepsize=stepsize,
        endsize=endsize,
        indices_start=copy.deepcopy(indices_start),
        labeling_policy=extend_random_indices_,
        seed=seed))
    accs_min_confidence.append(active_learning(
        x_train=inputs_train,
        y_train=targets_train,
        x_val=inputs_val,
        y_val=targets_val,
        stepsize=stepsize,
        endsize=endsize,
        indices_start=copy.deepcopy(indices_start),
        labeling_policy=extend_min_confidence_indices_,
        seed=seed))
accs_random = np.array(accs_random)
accs_max_entropy = np.array(accs_max_entropy)
accs_min_confidence = np.array(accs_min_confidence)


##### plot results #####
x_axis = torch.arange(start=startsize, step=stepsize, end=endsize)
mean_random = np.mean(accs_random, axis=0)
std_random = np.std(accs_random, axis=0)
plt.fill_between(x_axis, mean_random - std_random, mean_random + std_random, color="blue", label="random policy", alpha=0.3)

mean_min_confidence = np.mean(accs_min_confidence, axis=0)
std_min_confidence = np.std(accs_min_confidence, axis=0)
plt.fill_between(x_axis, mean_min_confidence - std_min_confidence, mean_min_confidence + std_min_confidence, color="green", label="min confidence policy", alpha=0.3)


plt.legend(loc="upper left")
plt.xlabel("training data points")
plt.ylabel(f"accuracy score (mean +/- std; {n_seeds} seeds)")
plt.title("active learning\nlabeling policy comparison")
plt.tight_layout()
plt.savefig("results.png")
