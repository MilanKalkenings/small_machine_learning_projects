import subprocess
import sys
import os
import numpy as np
import torch
from evaluation import seg_acc, cls_acc
from data_handling import create_loaders_cls, create_loaders_seg
from modules import UnetDown, Unet
subprocess.check_call([sys.executable, "-m", "pip", "install", "milankalkenings==0.0.9"])
from milankalkenings.deep_learning import Module, TrainerSetup, Trainer, make_reproducible

make_reproducible()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# cls setup
loader_train_cls, loader_val_cls = create_loaders_cls(batch_size=32)
print("cls train batches:", len(loader_train_cls), "cls val batches:", len(loader_val_cls))
trainer_setup_cls = TrainerSetup()
trainer_setup_cls.checkpoint_initial = "../monitoring/checkpoint_initial_cls.pkl"
trainer_setup_cls.checkpoint_running = "../monitoring/checkpoint_running_cls.pkl"
trainer_setup_cls.lrrt_n_batches = 10  # lrrt on about 10% of an epoch
trainer_setup_cls.monitor_n_losses = 50
trainer_setup_cls.lrrt_max_decays = 3
train_epochs_max_cls = 50

# seg setup
loader_train_seg, loader_val_seg, class_weights_seg = create_loaders_seg(batch_size=4)
print("seg train batches:", len(loader_train_seg), "seg val batches:", len(loader_val_seg))
trainer_setup_seg = TrainerSetup()
trainer_setup_seg.checkpoint_initial = "../monitoring/checkpoint_initial_seg.pkl"
trainer_setup_seg.checkpoint_running = "../monitoring/checkpoint_running_seg.pkl"
trainer_setup_seg.lrrt_n_batches = 10  # lrrt on a whole epoch
trainer_setup_seg.monitor_n_losses = 11  # 11 > 10, not shown
trainer_setup_seg.lrrt_max_decays = 1
trainer_setup_seg.lrrt_initial_candidates = np.array([5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6])
train_epochs_max_seg = 1000

# stage 1: cls pretraining

# save the initial checkpoint once
unet_down = UnetDown()
torch.save(unet_down, trainer_setup_cls.checkpoint_initial)


unet_down = torch.load(trainer_setup_cls.checkpoint_initial)
torch.save(unet_down, trainer_setup_cls.checkpoint_running)
cls_acc_train_prior = cls_acc(module=unet_down, loader_eval=loader_train_cls)
cls_acc_val_prior = cls_acc(module=unet_down, loader_eval=loader_val_cls)
print("initial cls acc train", cls_acc_train_prior, "initial cls acc val", cls_acc_val_prior)

trainer_cls = Trainer(loader_train=loader_train_cls, loader_val=loader_val_cls, setup=trainer_setup_cls)
trainer_cls.train_n_epochs_early_stop_initial_lrrt(max_epochs=train_epochs_max_cls, freeze_pretrained_layers=False)

unet_down = torch.load(trainer_setup_cls.checkpoint_running)
cls_acc_train_post = cls_acc(module=unet_down, loader_eval=loader_train_cls)
cls_acc_val_post = cls_acc(module=unet_down, loader_eval=loader_val_cls)
print("trained cls acc train", cls_acc_train_post, "trained cls acc val", cls_acc_val_post)

pretrained_unet_down = torch.load(trainer_setup_cls.checkpoint_running)
torch.save(pretrained_unet_down, "../monitoring/checkpoint_pretrained_undet_down.pkl")

# seg fine tuning

# save the initial checkpoint once and replace its unet_down with cls pretrained component
unet = Unet(loss_class_weights=class_weights_seg)
unet.unet_down = torch.load("../monitoring/checkpoint_pretrained_undet_down.pkl")
torch.save(unet, trainer_setup_seg.checkpoint_initial)

print("\n\nseg fine tuning")
unet = torch.load(trainer_setup_seg.checkpoint_initial)
torch.save(unet, trainer_setup_seg.checkpoint_running)
seg_acc_train_initial = seg_acc(module=unet, loader_eval=loader_train_seg)
seg_acc_val_initial = seg_acc(module=unet, loader_eval=loader_val_seg)
print("initial seg acc train", seg_acc_train_initial, "initial seg acc val", seg_acc_val_initial)

print("adjusting the upwards part to the pretrained downwards part")
trainer_seg = Trainer(loader_train=loader_train_seg, loader_val=loader_val_seg, setup=trainer_setup_seg)
trainer_seg.train_n_epochs_early_stop_initial_lrrt(max_epochs=train_epochs_max_seg, freeze_pretrained_layers=True)

unet = torch.load(trainer_setup_seg.checkpoint_running)
seg_acc_train_adjusted = seg_acc(module=unet, loader_eval=loader_train_seg)
seg_acc_val_adjusted = seg_acc(module=unet, loader_eval=loader_val_seg)
print("adjusted seg acc train", seg_acc_train_adjusted, "adjusted seg acc val", seg_acc_val_adjusted)

print("\n\ntraining the whole model")
trainer_seg = Trainer(loader_train=loader_train_seg, loader_val=loader_val_seg, setup=trainer_setup_seg)
trainer_seg.train_n_epochs_early_stop_initial_lrrt(max_epochs=train_epochs_max_seg, freeze_pretrained_layers=False)

unet = torch.load(trainer_setup_seg.checkpoint_running)
seg_acc_train_final = seg_acc(module=unet, loader_eval=loader_train_seg)
seg_acc_val_final = seg_acc(module=unet, loader_eval=loader_val_seg)
print("final seg acc train", seg_acc_train_final, "final seg acc val", seg_acc_val_final)
torch.save(unet, "../monitoring/checkpoint_final.pkl")


print("alternatively without pretraining")
unet = Unet(loss_class_weights=class_weights_seg)  # random init, not from checkpoint with pretrained weights
torch.save(unet, trainer_setup_seg.checkpoint_initial)  # overwrite initial checkpoint with random init unet
torch.save(unet, trainer_setup_seg.checkpoint_running)  # overwrite running checkpoint with random init unet


print("\n\ntraining from scratch")
trainer_seg = Trainer(loader_train=loader_train_seg, loader_val=loader_val_seg, setup=trainer_setup_seg)
trainer_seg.train_n_epochs_early_stop_initial_lrrt(max_epochs=train_epochs_max_seg, freeze_pretrained_layers=False)

unet = torch.load(trainer_setup_seg.checkpoint_running)
seg_acc_train_final = seg_acc(module=unet, loader_eval=loader_train_seg)
seg_acc_val_final = seg_acc(module=unet, loader_eval=loader_val_seg)
print("seg acc train after training from scratch", seg_acc_train_final, "seg acc val after training from scratch", seg_acc_val_final)
torch.save(unet, "../monitoring/checkpoint_alt_final.pkl")


# final evaluation
loader_train_seg, loader_val_seg, _ = create_loaders_seg(batch_size=4)
unet_with_pretrain = torch.load("../monitoring/checkpoint_final.pkl")
unet_without_pretrain = torch.load("../monitoring/checkpoint_alt_final.pkl")

for p1, p2 in zip(unet_without_pretrain.parameters(), unet_with_pretrain.parameters()):
    if (p1 != p2).sum().item() != 0:
        print("Models have different parameters")
    else:
        print("Models have the same parameters")

acc_train_without_pretrain = seg_acc(module=unet_without_pretrain, loader_eval=loader_train_seg)
acc_val_without_pretrain = seg_acc(module=unet_without_pretrain, loader_eval=loader_val_seg)

acc_train_with_pretrain = seg_acc(module=unet_with_pretrain, loader_eval=loader_train_seg)
acc_val_with_pretrain = seg_acc(module=unet_with_pretrain, loader_eval=loader_val_seg)

print("acc_train_without_pretrain", acc_train_without_pretrain)
print("acc_val_without_pretrain", acc_val_without_pretrain)
print("acc_train_with_pretrain", acc_train_without_pretrain)
print("acc_val_with_pretrain", acc_val_without_pretrain)
