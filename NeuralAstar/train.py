import torch
import pytorch_lightning as pl
import sys
import os
from dataloaders import AstarDataModule
from data_utils.planner.newplanner import combine_planner
from data_utils.utils.mechanism import Mechanism, NorthEastWestSouth, Moore
from data_utils.utils.data import get_hard_medium_easy_masks
from data_utils.utils.metrics import compute_mean_metrics
from data_utils.utils._il_utils import dilate_opt_trajs

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
# write log callback and check validation loop

wandb_logger = WandbLogger(project="Neural-astar-2")


class NeuralAstarModule(pl.LightningModule):

  def __init__(self):
      super().__init__()
      self.mechanism = Moore()
      self.planner = combine_planner(self.mechanism)
      self.model = self.planner.model
      self.skip_exp_when_training = False
      self.astar_ref = self.planner.astar_ref
      self.output_exp_instead_of_rel_exp = False

  def forward(self, x):
      map_designs, start_maps, goal_maps = x
      y = self.planner.forward(map_designs, start_maps, goal_maps)
      return y

  def loss_fn(self, input, target):
      
      loss = torch.nn.L1Loss()
      output = loss(input.double(), target.double())
      return output

  def training_step(self, train_batch, batch_idx):
      self.planner.model.train()
      map_designs, goal_maps, opt_policies, opt_dists = train_batch

      if map_designs.dim() == 3:
                map_designs = map_designs.unsqueeze(1)

      start_maps = self.planner.create_start_maps(opt_dists, self.device)

      x = (map_designs, start_maps, goal_maps)
      outputs = self.forward(x)

      opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism)
      loss = self.loss_fn(outputs[0], opt_trajs)
      return loss


  def validation_step(self, val_batch, batch_idx):
      self.planner.model.eval()
      map_designs, goal_maps, opt_policies, opt_dists = val_batch

      if map_designs.dim() == 3:
                map_designs = map_designs.unsqueeze(1)

      wall_dist = torch.min(opt_dists)
      with torch.no_grad():
          save_file = True
          num_eval_points = 5 if save_file else 2  # save time for validation
          # Compute success and optimality
          masks = get_hard_medium_easy_masks(opt_dists, self.device, False, num_eval_points)

          masks = torch.concat(masks, axis=1)
          pred_dist_maps = torch.empty_like(opt_dists)
          pred_dist_maps[:] = torch.nan 
          loss_tot = 0.0
          rel_exps_maps = torch.empty_like(opt_dists)
          rel_exps_maps[:] = torch.nan
          for i in range(masks.shape[1]):
              start_maps = masks[:, i]
              x = map_designs, start_maps, goal_maps
              outputs = self.forward(x)
              opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism)
              opt_trajs_ = dilate_opt_trajs(opt_trajs.double(), map_designs.double(), self.mechanism)
              loss = self.loss_fn(outputs[0], opt_trajs)
              loss_tot += loss
              pred_dists = -(outputs[1].sum(dim=(1, 2, 3)) - 1)
              arrived = (outputs[1] * start_maps).sum(dim=(1, 2, 3))
              not_passed_through_obstacles = (outputs[1] *
                                            (1 - map_designs)).sum(
                                                dim=(1, 2, 3)) == 0
              arrived = arrived * not_passed_through_obstacles
              pred_dists = pred_dists * arrived + wall_dist * (1.0 - arrived)

            # relative number of expansions
              pred_exps = outputs[0].sum((1, 2, 3))
              if (self.model.training & self.skip_exp_when_training) is not True:
                  astar_outputs = self.astar_ref(map_designs, start_maps,
                                                 goal_maps, map_designs)
                  exps = astar_outputs[0].sum((1, 2, 3))
              else:
                  exps = pred_exps
              rel_exps = pred_exps / exps

              if self.output_exp_instead_of_rel_exp:
                  rel_exps = pred_exps

              pred_dist_maps[masks[:, i]] = pred_dists[:]
              rel_exps_maps[masks[:, i]] = rel_exps[:]
          loss_tot /= masks.shape[1]
          (masks, indices) = masks.max(axis=1)

          p_opt, p_suc, p_exp = compute_mean_metrics(
                pred_dist_maps,
                rel_exps_maps,
                opt_dists,
                masks,
            )

          self.log("p_opt", p_opt, prog_bar=True, logger=True)
          self.log("p_suc", p_suc, prog_bar=True, logger=True)
          self.log("p_exp", p_exp, prog_bar=True, logger=True)
          self.log("loss_tot" , loss_tot, prog_bar=True, logger=True)
          self.log("loss" , loss, logger=True)

  def configure_optimizers(self):
      optimizer = torch.optim.RMSprop(self.planner.model.parameters(), lr=1e-3)
      return optimizer


DataModule = AstarDataModule("../../planning-datasets/data/mpd/bugtrap_forest_032_moore_c8.npz")
model = NeuralAstarModule()
trainer = pl.Trainer(gpus=1, log_every_n_steps=1, max_epochs=15, logger=wandb_logger)
trainer.fit(model, DataModule)
