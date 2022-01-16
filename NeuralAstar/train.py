from NeuralAstar.runner import NeuralAstarModule
from dataloaders import AstarDataModule
from runner import BBAstarModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(project="Neural-astar-cc-tl")

DataModule = AstarDataModule("dataset_astar/data/mpd/bugtrap_forest_032_moore_c8.npz")
model = NeuralAstarModule(0.5, 'vgg16_bn', False, 'm+')
trainer = pl.Trainer(gpus=1, log_every_n_steps=1, max_epochs=100,logger=logger,num_sanity_val_steps=0,gradient_clip_val=40,weights_summary="full" )
trainer.fit(model, DataModule)
trainer.test(model, DataModule.test_dataloader())