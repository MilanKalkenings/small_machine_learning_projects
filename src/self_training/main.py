import tensorflow as tf
from torch.optim import Adam
from torch.utils.data import SequentialSampler
import tensorflow_datasets as tfds
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "milankalkenings==0.1.32"])

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from milankalkenings import *


################################################################################################################
# code provided by ujjwal to download the data set


class CitrusLeaves(object):
    def __init__(self):
        self.ds = tfds.builder('citrus_leaves')
        self.ds.download_and_prepare()
        # In the following line we say that first 100 images of the training set are used as labeled dataset.
        self.ds_labeled = self.ds.as_dataset(split='train[:100]', shuffle_files=True)  # 100
        # In the following line, next 50 images are used as unlabeled set.
        self.ds_unlabeled = self.ds.as_dataset(split='train[100:151]', shuffle_files=False)  # 151
        # The remaining dataset is used for testing
        self.ds_test = self.ds.as_dataset(split='train[151:]', shuffle_files=False)

    @staticmethod
    def _map_fn(ds):
        # In this function we normalize the image in the range [-1.0, 1.0]
        image = ds['image']
        label = ds['label']
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, size=(32, 32))
        image = tf.divide(image, 255.0)
        image = tf.multiply(image, 2.0)
        image = tf.subtract(image, 1.0)
        return image, label

    def create_train_labeled_dataset(self):
        # This gives us a dictionary indexed by an integer from 0 to number of images.
        # For each key, the value is a tuple (image, label)
        ds = self.ds_labeled.map(lambda x: self._map_fn(x))
        ds = ds.batch(1)
        data_dict = dict()
        for i, (image, label) in enumerate(ds):
            data_dict[i] = (image, label)
        return data_dict

    def create_train_unlabeled_dataset(self):
        # This gives us a dictionary indexed by an integer from 0 to number of images.
        # For each key, the value is a tuple (image, label)
        ds = self.ds_unlabeled.map(lambda x: self._map_fn(x))
        ds = ds.batch(1)
        data_dict = dict()
        for i, (image, label) in enumerate(ds):
            data_dict[i] = (image, label)
        return data_dict

    def create_test_dataset(self):
        # This gives us a dictionary indexed by an integer from 0 to number of images.
        # For each key, the value is a tuple (image, label)
        ds = self.ds_test.map(lambda x: self._map_fn(x))
        ds = ds.batch(1)
        data_dict = dict()
        for i, (image, label) in enumerate(ds):
            data_dict[i] = (image, label)
        return data_dict
################################################################################################################


def to_torch(data_dict):
    imgs = []
    labels = []
    for data_point in data_dict.values():
        imgs.append(data_point[0].numpy())
        labels.append(data_point[1].numpy())
    imgs_torch = torch.permute(torch.squeeze(torch.tensor(np.array(imgs))), [0, 3, 1, 2])
    labels_torch = torch.squeeze(torch.tensor(np.array(labels)))
    return imgs_torch, labels_torch


def create_train_loaders(train_labeled_imgs: torch.Tensor,
                         train_labeled_labels: torch.Tensor,
                         train_unlabeled_imgs: torch.Tensor,
                         batch_size: int):
    train_labeled_dataset = StandardDataset(x=train_labeled_imgs, y=train_labeled_labels)
    train_unlabeled_dataset = StandardDataset(x=train_unlabeled_imgs, y=torch.zeros(size=[len(train_unlabeled_imgs)],
                                                                                    dtype=torch.long))
    loader_train_labeled = DataLoader(dataset=train_labeled_dataset, batch_size=batch_size)
    loader_train_unlabeled = DataLoader(dataset=train_unlabeled_dataset,
                                        batch_size=batch_size,
                                        sampler=SequentialSampler(data_source=train_unlabeled_dataset))
    return loader_train_labeled, loader_train_unlabeled


def create_loaders(train_labeled_imgs: torch.Tensor,
                   train_labeled_labels: torch.Tensor,
                   train_unlabeled_imgs: torch.Tensor,
                   val_imgs: torch.Tensor,
                   val_labels: torch.Tensor,
                   batch_size: int):
    loader_train_labeled, loader_train_unlabeled = create_train_loaders(train_labeled_imgs=train_labeled_imgs,
                                                                        train_labeled_labels=train_labeled_labels,
                                                                        train_unlabeled_imgs=train_unlabeled_imgs,
                                                                        batch_size=batch_size)
    val_dataset = StandardDataset(x=val_imgs, y=val_labels)
    loader_val = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            sampler=SequentialSampler(data_source=val_dataset))
    return loader_train_labeled, loader_train_unlabeled, loader_val


class Classifier(Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=4096, out_features=4)

        self.ce_loss = nn.CrossEntropyLoss()  # default: mean reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.flatten(x)
        scores = self.linear(x)
        loss = self.ce_loss(scores, y)
        return {"scores": scores, "loss": loss}


def modules_eq(module1: Module, module2: Module):
    """
    checks if two torch modules have the same parameters.
    """
    params1 = list(module1.parameters())
    params2 = list(module2.parameters())

    if len(params1) != len(params2):
        return False

    for i in range(len(params1)):
        if not torch.all(torch.eq(params1[i], params2[i])):
            return False

    return True


# main
batch_size = 256
n_epochs = 1
early_stopping_max_violations = 2
lrs = [0.001] * n_epochs
checkpoint_initial = "../../monitoring/self_learning/checkpoint_initial.pkl"
checkpoint_running = "../../monitoring/self_learning/checkpoint_running.pkl"
checkpoint_final = "../../monitoring/self_learning/checkpoint_final.pkl"

cl = CitrusLeaves()
train_labeled_imgs, train_labeled_labels = to_torch(cl.create_train_labeled_dataset())
train_unlabeled_imgs, _ = to_torch(cl.create_train_unlabeled_dataset())
val_imgs, val_labels = to_torch(cl.create_test_dataset())
loader_train_labeled, loader_train_unlabeled, loader_val = create_loaders(train_labeled_imgs=train_labeled_imgs,
                                                                          train_labeled_labels=train_labeled_labels,
                                                                          train_unlabeled_imgs=train_unlabeled_imgs,
                                                                          val_imgs=val_imgs,
                                                                          val_labels=val_labels,
                                                                          batch_size=batch_size)

# module_initial = Classifier()
# save_secure(module=module_initial, file=checkpoint_initial)
module_initial = load_secure(file=checkpoint_initial)

"""
# debug
trainer = Trainer(module=module_initial,
                  loader_train=loader_train_labeled,
                  loader_val=loader_val,
                  loader_test=loader_val,
                  optimizer_class=Adam,
                  checkpoint_running=checkpoint_running,
                  checkpoint_final=checkpoint_final)
trainer.determine_initial_lr(module=module_initial,
                             n_iters=50,
                             lr_candidates=[0.001, 0.0005, 0.0001, 0.00005, 0.00001],
                             batch_debug=next(iter(loader_train_labeled)),
                             save_file="../../monitoring/self_learning/initial_lr")
"""

val_losses = []
estimated_labels = []
train_accs = []
val_accs = []
for iteration in range(51):
    print("labeled training data points:", len(train_labeled_imgs))
    make_reproducible()
    module_initial = load_secure(file=checkpoint_initial)
    trainer = Trainer(module=module_initial,
                      loader_train=loader_train_labeled,
                      loader_val=loader_val,
                      loader_test=loader_val,
                      optimizer_class=Adam,
                      checkpoint_running=checkpoint_running,
                      checkpoint_final=checkpoint_final)
    module, _, _ = trainer.train(n_epochs=n_epochs,
                                 lrs=lrs,
                                 early_stopping_max_violations=early_stopping_max_violations)
    loss_sum = trainer.loss_epoch_eval(module=module, loader_eval=loader_val)
    val_losses.append(loss_sum)
    train_accs.append(trainer.acc_epoch_eval(module=module, loader_eval=loader_train_labeled))
    val_accs.append(trainer.acc_epoch_eval(module=module, loader_eval=loader_val))

    if len(train_unlabeled_imgs) > 0:
        # estimate labels
        scores = trainer.predict_epoch_scores_eval(module=module, loader_eval=loader_train_unlabeled)
        max_obs_id = torch.div(torch.argmax(input=torch.flatten(scores)), 4, rounding_mode="trunc")
        estimated_label = torch.argmax(scores[max_obs_id])

        # move observation with inferred label to the labeled training data
        train_labeled_imgs = torch.cat([train_labeled_imgs, train_unlabeled_imgs[max_obs_id:max_obs_id+1]], dim=0)
        train_labeled_labels = torch.cat([train_labeled_labels, torch.unsqueeze(estimated_label, dim=0)], dim=0)
        train_unlabeled_imgs = torch.cat([train_unlabeled_imgs[:max_obs_id], train_unlabeled_imgs[max_obs_id+1:]], dim=0)
        loader_train_labeled, loader_train_unlabeled = create_train_loaders(train_labeled_imgs=train_labeled_imgs,
                                                                            train_labeled_labels=train_labeled_labels,
                                                                            train_unlabeled_imgs=train_unlabeled_imgs,
                                                                            batch_size=batch_size)
        estimated_labels.append(estimated_label)
        print("estimated new label:", estimated_label)
        print("\n\n")

print("estimated labels:", estimated_labels)
print("train accs:", train_accs)
print("val accs:", val_accs)
lines_multiplot(lines=[val_losses],
                y_label="test loss",
                x_label="estimated training labels",
                title="self training",
                save_file="../../monitoring/self_learning/losses",
                multiplot_labels=["test loss"])

lines_multiplot(lines=[val_accs],
                y_label="test accuracy",
                x_label="estimated training labels",
                title="self training",
                save_file="../../monitoring/self_learning/accs",
                multiplot_labels=["test accuracy"])
