import subprocess
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders import AstarDataModule
from runner import NeuralAstarModule
from pytorch_lightning.loggers import WandbLogger
import wandb
import random

def main():
    subprocess.call('git clone -b main https://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@github.com/shreya-bhatt27/NeuralAstar-ported.git')
    subprocess.call('git clone https://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@github.com/shreya-bhatt27/dataset_astar.git')
    subprocess.call('wandb login eb94e0c9d64b72218420c8f40585ec650a663fa4')
    subprocess.call('cd NeuralAstar')
    subprocess.call('cd ../')
    subprocess.call('mkdir checkpoints')
    subprocess.call('cd NeuralAstar')
    checkpoint_callback = ModelCheckpoint(
         dirpath='../checkpoints',
         filename='{epoch}-{hmean:.2f}-{los_tot:.2f}',
         monitor='hmean',
         save_top_k = -1,
         mode = 'max',
    )
    pl.utilities.seed.seed_everything(1993, workers = True)
    logger = WandbLogger(project="Neural-astar-dropout-experiment")
    DataModule = AstarDataModule("../dataset_astar/data/mpd/all_064_moore_c16.npz")
    model = NeuralAstarModule(0.0, 'vgg16_bn', True, 'm+')
    trainer = pl.Trainer(gpus=1, log_every_n_steps=1,callbacks=[checkpoint_callback], max_epochs=400,logger=logger,num_sanity_val_steps=0,gradient_clip_val=40,weights_summary="full", deterministic=True)
    trainer.fit(model, DataModule)
    trainer.save_checkpoint("dropout.pth")
    wandb.save("dropout.pth")
    trainer.test(model, DataModule.test_dataloader())