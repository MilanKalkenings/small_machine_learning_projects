import torch
from data_handling import DataHandlerSetup, DataHandler
from training import TrainerSetup, Trainer
from reproducibility import make_reproducible

make_reproducible()
data_handler_setup = DataHandlerSetup()
data_handler = DataHandler(setup=data_handler_setup)
trainer_setup = TrainerSetup()
trainer = Trainer(setup=trainer_setup, loader_train=data_handler.loader_train, loader_val=data_handler.loader_val)

"""
# to create the initial checkpoint
module = CNN()
torch.save(module, trainer.checkpoint_initial)
torch.save(module, trainer.checkpoint_running)
"""

module = torch.load(trainer.checkpoint_initial)
torch.save(module, trainer.checkpoint_running)

print("adjust the head to the resnet body")
trainer.train_n_epochs_early_stop_initial_lrrt(n_epochs=20, freeze_pretrained_layers=True)

print("\n\n\ntraining the whole module")
trainer.train_n_epochs_early_stop_initial_lrrt(n_epochs=20, freeze_pretrained_layers=False)

