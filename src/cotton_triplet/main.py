import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "milankalkenings==0.1.20"])
from data_handling import create_loaders, DebugData
from modules import ResNet
from milankalkenings.deep_learning import make_reproducible, Setup, Trainer, Visualizer, Debugger
import torch
from evaluation import evaluate
from torch.utils.data import DataLoader

###############################
# setup
make_reproducible(seed=3)

# general
embedding_size = 16
cls_focus = 0.4
triplet_loss_margin = 1
batch_size = 128  # 15 training batches
monitor_n_losses = 16  # 16 > 15, not displayed

# debug
n_iters_debug = 15
lrrt_max_decays = 20
lrrt_decay = 0.9
lrrt_n_batches = 5

# training
max_epochs = 50
overkill_initial_lr = 0.005
overkill_decay = 0.95
overkill_max_violations = 14
overkill_max_decays = 100

loader_train, loader_val, loader_test = create_loaders(batch_size=batch_size)
print("train batches", len(loader_train))
print("val batches", len(loader_val))
print("test batches", len(loader_test))
setup = Setup(loader_train=loader_train,
              loader_val=loader_val,
              loader_test=loader_test,
              lrrt_decay=lrrt_decay,
              lrrt_max_decays=lrrt_max_decays,
              lrrt_n_batches=lrrt_n_batches,
              overkill_decay=overkill_decay,
              overkill_initial_lr=overkill_initial_lr,
              overkill_max_violations=overkill_max_violations,
              overkill_max_decays=overkill_max_decays,
              monitor_n_losses=monitor_n_losses)


resnet = ResNet(alpha=cls_focus, triplet_loss_margin=triplet_loss_margin, embedding_size=embedding_size)
torch.save(resnet, setup.checkpoint_initial)
resnet = torch.load(setup.checkpoint_initial)
torch.save(resnet, setup.checkpoint_running)
trainer = Trainer(setup=setup)
debugger = Debugger(setup=setup)
visualizer = Visualizer()

###############################
# debug
batch_debug = next(iter(loader_train))
lr_candidates = torch.tensor([1e-3, 1e-4])
lr_candidates_str = [str(lr_candidate) for lr_candidate in lr_candidates]
losses_candidates = []
for lr_candidate in lr_candidates:
    resnet_debug = torch.load(setup.checkpoint_initial)
    _, losses_candidate = debugger.overfit_batch(module=resnet_debug,
                                                 batch_debug=batch_debug,
                                                 n_iters=n_iters_debug,
                                                 lr=lr_candidate)
    losses_candidates.append(losses_candidate)
visualizer.lines_subplot(lines=losses_candidates,
                         title="overfitting one batch",
                         y_label="loss",
                         x_label="training iteration",
                         file_name="overfitting_one_batch",
                         subplot_titles=lr_candidates_str)
resnet_debug = torch.load(setup.checkpoint_initial)
loader_debug = DataLoader(dataset=DebugData(batch_debug=batch_debug), batch_size=batch_size)
resnet_debug_trained, _ = debugger.overfit_batch(module=resnet_debug,
                                                 batch_debug=batch_debug,
                                                 n_iters=n_iters_debug * 2,
                                                 lrrt_loader=loader_train)
evaluate(module=resnet_debug_trained, loader_eval=loader_debug, name_fig="results_debug")

###############################
# fine tune
resnet, best_lrs, losses_train, losses_val = trainer.train_overkill(max_epochs=max_epochs, freeze_pretrained=False)
visualizer.lines_subplot(lines=[best_lrs],
                         title="lrrt learning rates",
                         subplot_titles=[""],
                         y_label="lr",
                         x_label="iteration",
                         file_name="lr_per_iter")
visualizer.lines_multiplot(lines=[losses_train, losses_val],
                           title="loss",
                           multiplot_labels=["training data", " validation data"],
                           y_label="loss", x_label="epoch",
                           file_name="loss_per_epoch")

###############################
# final evaluations
evaluate(module=resnet, loader_eval=loader_train, name_fig="train")
evaluate(module=resnet, loader_eval=loader_test, name_fig="test")

