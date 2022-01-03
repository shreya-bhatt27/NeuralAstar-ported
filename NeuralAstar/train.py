import torch
import pytorch_lightning as pl
import sys
import os
import numpy as np
from dataloaders import AstarDataModule
from data_utils.planner.newplanner import combine_planner
from data_utils.utils.mechanism import Mechanism, NorthEastWestSouth, Moore
from data_utils.utils.data import get_hard_medium_easy_masks
from data_utils.utils.metrics import compute_mean_metrics

# write log callback and check validation loop

class NeuralAstarModule(pl.LightningModule):

  def __init__(self):
      self.mechanism = Moore()
      self.planner = combine_planner(self.mechanism)

  def forward(self, x):
      x = self.planner.forward(x)
      return x

  def loss_fn(input,target):
      loss = torch.nn.L1Loss(input,target)
      return loss

  def training_step(self, train_batch, batch_idx):
      self.planner.model.train()
      map_designs, goal_maps, opt_policies, opt_dists = train_batch

      map_designs_CPU = map_designs.data.numpy()
      goal_maps_CPU = goal_maps.data.numpy()
      opt_policies_CPU = opt_policies.data.numpy()
      opt_dists_CPU = opt_dists.data.numpy()
      map_designs = map_designs.to(self.device)
      goal_maps = goal_maps.to(self.device)
      opt_policies = opt_policies.to(self.device)
      opt_dists = opt_dists.to(self.device)

      if map_designs.dim() == 3:
                map_designs = map_designs.unsqueeze(1)
                map_designs_CPU = np.expand_dims(map_designs_CPU, axis=1)

      start_maps = self.planner.create_start_maps(opt_dists)

      x = (map_designs, start_maps, goal_maps)
      loss, outputs = self.forward(x)

      opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism)
      loss = self.loss_fn(outputs[0], opt_trajs)
      return loss, outputs

  def validation_step(self, val_batch, batch_idx):
      self.planner.model.eval()
      map_designs, goal_maps, opt_policies, opt_dists = val_batch

      map_designs_CPU = map_designs.data.numpy()
      goal_maps_CPU = goal_maps.data.numpy()
      opt_policies_CPU = opt_policies.data.numpy()
      opt_dists_CPU = opt_dists.data.numpy()
      map_designs = map_designs.to(self.device)
      goal_maps = goal_maps.to(self.device)
      opt_policies = opt_policies.to(self.device)
      opt_dists = opt_dists.to(self.device)

      if map_designs.dim() == 3:
                map_designs = map_designs.unsqueeze(1)
                map_designs_CPU = np.expand_dims(map_designs_CPU, axis=1)
      wall_dist = np.min(opt_dists_CPU)
      with torch.no_grad():
          save_file = True
          num_eval_points = 5 if save_file else 2  # save time for validation
          # Compute success and optimality
          masks = get_hard_medium_easy_masks(opt_dists, False,
                                           num_eval_points)
          masks = np.concatenate(masks, axis=1)
          pred_dist_maps = np.empty_like(opt_dists)
          pred_dist_maps[:] = np.NAN
          loss_tot = 0.0
          rel_exps_maps = np.empty_like(opt_dists)
          rel_exps_maps[:] = np.NAN
          for i in range(masks.shape[1]):
              start_maps = masks[:, i]
              loss, outputs = self.forward(map_designs, goal_maps, start_maps)
              loss_tot += loss
              pred_dists = -(outputs[1].sum(dim=(1, 2, 3)) - 1)
              arrived = (outputs[1] * start_maps).sum(dim=(1, 2, 3))
              not_passed_through_obstacles = (outputs[1] *
                                            (1 - map_designs)).sum(
                                                dim=(1, 2, 3)) == 0
              arrived = arrived * not_passed_through_obstacles
              pred_dists = pred_dists * arrived + wall_dist * (1.0 - arrived)
              pred_dists = pred_dists.cpu().data.numpy()

            # relative number of expansions
              pred_exps = outputs[0].cpu().data.numpy().sum((1, 2, 3))
              if (self.model.training & self.skip_exp_when_training) is not True:
                  astar_outputs = self.astar_ref(map_designs, start_maps,
                                                 goal_maps, map_designs)
                  exps = astar_outputs[0].cpu().data.numpy().sum((1, 2, 3))
              else:
                  exps = pred_exps
              rel_exps = pred_exps / exps

              if self.output_exp_instead_of_rel_exp:
                  rel_exps = pred_exps

              pred_dist_maps[masks[:, i]] = pred_dists[:]
              rel_exps_maps[masks[:, i]] = rel_exps[:]
          loss_tot /= masks.shape[1]

          p_opt, p_suc, p_exp = compute_mean_metrics(
                pred_dist_maps,
                rel_exps_maps,
                opt_dists,
                masks.max(axis=1),
            )

          self.log_values(loss_tot, p_opt, p_suc, p_exp, pred_dist_maps, rel_exps_maps, masks)


  def configure_optimizers(self):
      optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-3)
      return optimizer


DataModule = AstarDataModule()
model = NeuralAstarModule()
trainer = pl.Trainer()
trainer.fit(model, DataModule)
