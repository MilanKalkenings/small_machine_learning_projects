import torch
from data_handling import create_loaders_mnist
from modules import CNN
from milankalkenings.deep_learning import TrainerSetup, Trainer
from evaluation import acc_mc_dropout
import matplotlib.pyplot as plt


# experiment config
n_runs = 10
max_epochs = 30
n_forward_passes = 5

# loading data
loader_train, loader_val, loader_test = create_loaders_mnist(batch_size=64)
some_batch = next(iter(loader_train))
some_x, some_y = some_batch

# setups
## mc dropout
trainer_setup = TrainerSetup()
trainer_setup.monitor_n_losses = 400  # > 99, not shown in lrrt
trainer_setup.lrrt_n_batches = 99
trainer_setup.checkpoint_initial = "../monitoring/checkpoint_initial_mc.pkl"
trainer_setup.checkpoint_running = "../monitoring/checkpoint_running_mc.pkl"
trainer = Trainer(loader_train=loader_train, loader_val=loader_val, setup=trainer_setup)

## baseline
trainer_setup_baseline = TrainerSetup()
trainer_setup_baseline.monitor_n_losses = 400  # > 99, not shown in lrrt
trainer_setup_baseline.lrrt_n_batches = 99
trainer_setup_baseline.checkpoint_initial = "../monitoring/checkpoint_initial_baseline.pkl"
trainer_setup_baseline.checkpoint_running = "../monitoring/checkpoint_running_baseline.pkl"
trainer_baseline = Trainer(loader_train=loader_train, loader_val=loader_val, setup=trainer_setup_baseline)


# save initial checkpoints
## mc dropout
module_initial = CNN(depth=4, example_x=some_x, n_classes=10, mc_dropout=True)
torch.save(module_initial, trainer_setup.checkpoint_initial)
torch.save(module_initial, trainer_setup.checkpoint_running)

## baseline
module_initial_baseline = CNN(depth=4, example_x=some_x, n_classes=10, mc_dropout=False)
torch.save(module_initial_baseline, trainer_setup_baseline.checkpoint_initial)
torch.save(module_initial_baseline, trainer_setup_baseline.checkpoint_running)

# training and evaluation over n_runs runs
train_accs = torch.zeros(size=[n_forward_passes]).float()
test_accs = torch.zeros(size=[n_forward_passes]).float()
train_acc_baseline = torch.tensor([0]).float()
test_acc_baseline = torch.tensor([0]).float()
for run in range(1, n_runs + 1):
    # training
    ## mc dropout
    print("\n\nmc dropout training run", run)
    trainer.train_n_epochs_early_stop_initial_lrrt(max_epochs=max_epochs, freeze_pretrained_layers=False)

    ## baseline
    print("\n\nbaseline training run", run)
    trainer_baseline.train_n_epochs_early_stop_initial_lrrt(max_epochs=max_epochs, freeze_pretrained_layers=False)

    # evaluation
    ## mc dropout
    module_final = torch.load(trainer_setup.checkpoint_running)
    train_accs_run = []
    test_accs_run = []
    for i in range(n_forward_passes):
        train_accs_run.append(acc_mc_dropout(module=module_final, loader_eval=loader_train, n_forward_passes=i + 1))
        test_accs_run.append(acc_mc_dropout(module=module_final, loader_eval=loader_test, n_forward_passes=i + 1))
    train_accs += (torch.tensor(train_accs_run))
    test_accs += (torch.tensor(test_accs_run))

    ## baseline
    module_final_baseline = torch.load(trainer_setup_baseline.checkpoint_running)
    train_acc_baseline_run = acc_mc_dropout(module=module_final_baseline, loader_eval=loader_train, n_forward_passes=1)
    test_acc_baseline_run = acc_mc_dropout(module=module_final_baseline, loader_eval=loader_test, n_forward_passes=1)
    train_acc_baseline += torch.tensor(train_acc_baseline_run)
    test_acc_baseline += torch.tensor(test_acc_baseline_run)

    # reset running checkpoints for next run
    ## mc dropout
    module_initial = torch.load(trainer_setup.checkpoint_initial)
    torch.save(module_initial, trainer_setup.checkpoint_running)

    ## baseline
    module_initial_baseline = torch.load(trainer_setup.checkpoint_initial)
    torch.save(module_initial_baseline, trainer_setup_baseline.checkpoint_running)


# plot the results
plt.scatter(0, train_acc_baseline / n_runs, label="dropout train")
plt.scatter(0, test_acc_baseline / n_runs, label="dropout test")
plt.plot(range(len(train_accs)), train_accs / n_runs, label="mc dropout train")
plt.plot(range(len(test_accs)), test_accs / n_runs, label="mc dropout test")
plt.xticks(range(len(train_accs)))
plt.gca().set_xticklabels(range(1, len(train_accs) + 1))
plt.title("monte carlo dropout vs normal dropout")
plt.xlabel("forward passes")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("../monitoring/mcd_accuracies.png")
