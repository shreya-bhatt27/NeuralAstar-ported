import subprocess
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders import AstarDataModule
from runner import NeuralAstarModule
from pytorch_lightning.loggers import WandbLogger
import wandb
import random

def main(whether_wandb, wandb_login):
    print("HELLOOO FIRST")
    logger = None
    if whether_wandb == True:
        subprocess.call('wandb login '  + str(wandb_login))
        logger = WandbLogger(project="Neural-astar-dropout-experiment")
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
    print("HELLLLLOOOO_BEFORE")
    DataModule = AstarDataModule("../../data/mpd/multiple_bugtraps_032_moore_c8.npz")
    model = NeuralAstarModule(0.0, 'vgg16_bn', True, 'm+')
    trainer = pl.Trainer(gpus=1, log_every_n_steps=1,callbacks=[checkpoint_callback], max_epochs=400,logger=logger,num_sanity_val_steps=0,gradient_clip_val=40,weights_summary="full", deterministic=True)
    trainer.fit(model, DataModule)
    print("HELLLLLOOOO_AFTER")
    trainer.save_checkpoint("dropout.pth")
    if whether_wandb:
        wandb.save("dropout.pth")
    trainer.test(model, DataModule.test_dataloader())
    print("HELLO_LAST")

if __name__ == '__main__':
    main()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--whether_wandb', metavar='path', required=True)
    parser.add_argument('--wandb_login', metavar='path', required=True)
    args = parser.parse_args()
    main(workspace=args.workspace)
