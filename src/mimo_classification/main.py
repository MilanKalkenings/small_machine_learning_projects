import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "milankalkenings==0.0.11"])
import torch
from modules import Mimo
from data_handling import cifar1_loaders
from milankalkenings.deep_learning import TrainerSetup, Trainer, make_reproducible
from evaluation import accuracy_inf
import matplotlib.pyplot as plt


make_reproducible()
n_subnets = 2
max_epochs = 50
loader_train, loader_inference_train, loader_inference_val = cifar1_loaders(n_subnets=n_subnets,
                                                                            batch_size=64,
                                                                            image_size=128,
                                                                            n_train_obs=32_000,
                                                                            n_val_obs=3_200)
trainer_setup = TrainerSetup()
trainer_setup.lrrt_max_decays = 0
trainer_setup.lrrt_n_batches = 250  # lrrt on whole epoch
trainer_setup.monitor_n_losses = 251  # > 250, not shown
trainer_setup.es_max_violations = 5
trainer = Trainer(loader_train=loader_train, loader_val=loader_inference_val, setup=trainer_setup)


# save initial checkpoints
mimo_initial = Mimo(n_subnets=n_subnets, n_classes=10)
torch.save(mimo_initial, trainer_setup.checkpoint_initial)
torch.save(mimo_initial, trainer_setup.checkpoint_running)


# determine the initial learning rate candidates by overfitting one batch
example_batch_train = next(iter(loader_train))

plt.figure(figsize=(30, 15))
plt.suptitle("overfitting one batch with various learning rates")
losses = []
lr_cands = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
for i, cand in enumerate(lr_cands):
    mimo = torch.load(trainer_setup.checkpoint_initial)
    adam = torch.optim.Adam(params=mimo.parameters(), lr=cand)
    _, losses_debug = trainer.overfit_one_train_batch(module=mimo,
                                                      batch=example_batch_train,
                                                      optimizer=adam,
                                                      n_iters=5,
                                                      freeze_pretrained_layers=False)
    plt.subplot(2, 5, i+1)
    plt.title("lr: " + str(cand))
    plt.ylabel("loss")
    plt.xlabel("training iteration")
    plt.plot(torch.arange(len(losses_debug)), losses_debug)
plt.tight_layout()
plt.savefig("../monitoring/initial_learning_rate_det.png")
trainer.lrrt_initial_candidates = torch.tensor([5e-4, 1e-4, 5e-5])

# initial evaluation
acc_inf_train_initial = accuracy_inf(mimo=mimo_initial, loader_inf_eval=loader_inference_train)
acc_inf_val_initial = accuracy_inf(mimo=mimo_initial, loader_inf_eval=loader_inference_val)
print("accuracy before training")
print("train:", acc_inf_train_initial, "val:", acc_inf_val_initial)

# training
print("\n\ntraining")
trainer.train_n_epochs_early_stop_initial_lrrt(max_epochs=max_epochs, freeze_pretrained_layers=False)

# final evaluation
mimo_final = torch.load(trainer_setup.checkpoint_final)
acc_inf_train_final = accuracy_inf(mimo=mimo_final, loader_inf_eval=loader_inference_train)
acc_inf_val_final = accuracy_inf(mimo=mimo_final, loader_inf_eval=loader_inference_val)
print("\n\naccuracy after training")
print("train:", acc_inf_train_final, "val:", acc_inf_val_final)


