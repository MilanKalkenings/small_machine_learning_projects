import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from modules import Module


class TrainerSetup:
    def __init__(self):
        self.device = "cpu"
        self.monitor_n_losses = 100  # prints loss slope after this amount of training steps
        self.checkpoint_initial = "../monitoring/checkpoint_initial.pkl"
        self.checkpoint_running = "../monitoring/checkpoint_running.pkl"
        self.lrrt_n_batches = 100  # batches used in lrrt for learning rate determination
        self.lrrt_slope_desired = 0  # exclusive
        self.lrrt_max_decays = 10  # max number of candidate decays performed in lrrt
        self.lrrt_decay = 0.9
        self.lrrt_initial_candidates = np.array([1e-1, 1e-3, 1e-4, 1e-6])


class Trainer:
    def __init__(self, loader_train: DataLoader, loader_val: DataLoader, setup: TrainerSetup):
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.device = setup.device
        self.monitor_n_losses = setup.monitor_n_losses

        self.checkpoint_initial = setup.checkpoint_initial
        self.checkpoint_running = setup.checkpoint_running

        self.lrrt_n_batches = setup.lrrt_n_batches
        self.lrrt_slope_desired = setup.lrrt_slope_desired
        self.lrrt_max_decays = setup.lrrt_max_decays
        self.lrrt_decay = setup.lrrt_decay
        self.lrrt_initial_candidates = setup.lrrt_initial_candidates

    def forward_batch(self, module: Module, batch: list):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        return module(x=x, y=y)

    def train_batch(self, module: Module, adam: Adam, batch: list, freeze_pretrained_layers: bool):
        # freeze/unfreeze here: longer runtime, better code encapsulation
        if freeze_pretrained_layers:
            module.freeze_pretrained_layers()
        else:
            module.unfreeze_pretrained_layers()
        module.train()
        module.zero_grad()
        loss = self.forward_batch(module=module, batch=batch)["loss"]
        loss.backward()
        adam.step()
        return float(loss)

    def train_n_batches(self, module: Module, adam: Adam, n_batches: int, freeze_pretrained_layers: bool):
        losses = []
        for train_iter, batch in enumerate(self.loader_train):
            if train_iter == n_batches:
                break

            losses.append(self.train_batch(module=module, adam=adam, batch=batch,
                                           freeze_pretrained_layers=freeze_pretrained_layers))
            if (len(losses) % self.monitor_n_losses) == 0:
                losses_last = np.array(losses[-self.monitor_n_losses:])
                slope_last, _ = np.polyfit(x=np.arange(len(losses_last)), y=losses_last, deg=1)
                print("iter", train_iter + 1, "mean loss", losses_last.mean(), "loss slope", slope_last)
        slope_total, bias_total = np.polyfit(x=np.arange(len(losses)), y=losses, deg=1)
        return losses, slope_total, bias_total

    def loss_batch_eval(self, module: Module, batch: list):
        module.eval()
        with torch.no_grad():
            return float(self.forward_batch(module=module, batch=batch)["loss"])

    def loss_epoch_eval(self, module: Module, loader_eval: DataLoader):
        batch_losses = np.zeros(len(loader_eval))
        for batch_nr, batch in enumerate(loader_eval):
            batch_losses[batch_nr] = self.loss_batch_eval(module=module, batch=batch)
        return float(batch_losses.mean())

    def losses_epoch_eval(self, module: Module):
        loss_epoch_train = self.loss_epoch_eval(module=module, loader_eval=self.loader_train)
        loss_epoch_val = self.loss_epoch_eval(module=module, loader_eval=self.loader_val)
        return loss_epoch_train, loss_epoch_val

    def predict_class_labels_batch(self, module: Module, batch: list):
        scores = self.forward_batch(module=module, batch=batch)["scores"]
        return torch.argmax(scores, dim=1)

    def accuracy_epoch_eval(self, module: Module, loader_eval: DataLoader):
        module.eval()
        correct_sum = 0
        obs_sum = 0
        for batch in loader_eval:
            with torch.no_grad():
                x, y = batch
                y = y.to(self.device)

                preds = self.predict_class_labels_batch(module=module, batch=batch)
                correct_sum += (y == preds).sum()
                obs_sum += len(y)
        return float(correct_sum / obs_sum)

    def accuracies_epoch_eval(self, module: Module):
        accuracy_epoch_train = self.accuracy_epoch_eval(module=module, loader_eval=self.loader_train)
        accuracy_epoch_val = self.accuracy_epoch_eval(module=module, loader_eval=self.loader_val)
        return accuracy_epoch_train, accuracy_epoch_val

    def accuracies_losses_epoch_eval(self, module: Module):
        accuracy_train, accuracy_val = self.accuracies_epoch_eval(module=module)
        loss_train, loss_val = self.losses_epoch_eval(module=module)
        return accuracy_train, accuracy_val, loss_train, loss_val

    def lrrt(self, freeze_pretrained_layers: bool):
        """
        Learning Rate Range Test; basic idea:
        for each learning rate in a set of learning rate candidates:
            load a checkpoint
            train from the checkpoint on a small amount of batches
            determine the slope of the batch losses
            return the learning rate that creates the steepest negative slope

        modified to rerun with a decayed set of learning rate candidates
        until a max number of iterations or a certain slope is reached.

        :return: best learning rate, best loss slope
        """
        print("greedily searching lr using lrrt")
        slope_desired_found = False
        candidate_lrs = self.lrrt_initial_candidates
        lr_best_total = np.inf
        slope_best_total = np.inf
        for decay_it in range(self.lrrt_max_decays + 1):
            candidate_slopes = np.zeros(shape=len(candidate_lrs))
            for i, lr_candidate in enumerate(candidate_lrs):
                module = torch.load(self.checkpoint_running)
                adam = Adam(params=module.parameters(), lr=lr_candidate)
                candidate_slopes[i] = self.train_n_batches(module=module, adam=adam, n_batches=self.lrrt_n_batches,
                                                           freeze_pretrained_layers=freeze_pretrained_layers)[1]
            best_candidate_slope_id = np.argmin(candidate_slopes)
            best_candidate_slope = candidate_slopes[best_candidate_slope_id]
            best_candidate_lr = candidate_lrs[best_candidate_slope_id]
            if best_candidate_slope < slope_best_total:
                slope_best_total = best_candidate_slope
                lr_best_total = best_candidate_lr
            if slope_best_total < self.lrrt_slope_desired:
                slope_desired_found = True
                break
            else:
                print("decaying candidate lrs")
                candidate_lrs = candidate_lrs * self.lrrt_decay
        if not slope_desired_found:
            print("lr with desired loss slope", self.lrrt_slope_desired, "not found. using approx best lr instead")
        print("best loss slope", slope_best_total, "best lr", lr_best_total)
        return lr_best_total, slope_best_total

    def train_n_epochs_early_stop_initial_lrrt(self, n_epochs: int, freeze_pretrained_layers: bool):
        """
        determines the initial learning rate per epoch using lrrt.
        early stops (naively after one early stop violation)

        :param int n_epochs: number of training epochs after determining the initial learning rate with lrrt
        :param bool freeze_pretrained_layers:
        :return: early stopped trained module
        """
        module = torch.load(self.checkpoint_running)
        accuracy_train, accuracy_val, loss_train, loss_val_last = self.accuracies_losses_epoch_eval(module=module)
        print("initial acc train", accuracy_train, "initial acc val", accuracy_val)
        print("initial eval loss train", loss_train, "initial eval loss val", loss_val_last)
        for epoch in range(1, n_epochs + 1):
            print("training epoch", epoch)
            lr_best, _ = self.lrrt(freeze_pretrained_layers=freeze_pretrained_layers)
            module = torch.load(self.checkpoint_running).to(self.device)
            adam = Adam(params=module.parameters(), lr=lr_best)
            self.train_n_batches(module=module, adam=adam, n_batches=len(self.loader_train),
                                 freeze_pretrained_layers=freeze_pretrained_layers)

            accuracy_train, accuracy_val, loss_train, loss_val = self.accuracies_losses_epoch_eval(module=module)
            print("acc train", accuracy_train, "acc val", accuracy_val)
            print("eval loss train", loss_train, "eval loss val", loss_val)
            if loss_val < loss_val_last:
                torch.save(module, self.checkpoint_running)
                print("loss improvement achieved, running checkpoint updated")
                loss_val_last = loss_val
            else:
                print("no loss improvement achieved, early stopping")
                break
        return torch.load(self.checkpoint_running)
