import torch
import pytorch_lightning as pl
import sys
import os
import wandb
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.utils.data as data
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
from data_utils.utils.mechanism import Moore , NorthEastWestSouth
from data_utils.planner.newplanner import combine_planner
from data_utils.utils._il_utils import dilate_opt_trajs, get_hard_medium_easy_masks, compute_bsmean_cbound
from data_utils.utils.metrics import compute_mean_metrics
from data_utils.planner.bbastar import BBAstarPlanner
from dataloaders import AstarDataModule

class BBAstarModule(pl.LightningModule):

  def __init__(self):
      super().__init__()
      self.mechanism = Moore()
      self.planner = BBAstarPlanner(self.mechanism)
      self.model = self.planner.model
      self.skip_exp_when_training = False
      self.astar_ref = self.planner.astar_ref
      self.output_exp_instead_of_rel_exp = False
      

  def show_maze(self, image):
      image = image.cpu().detach().numpy().squeeze()
      plt.matshow(image)
      plt.show()

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
      opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism, self.device)
      opt_trajs_ = dilate_opt_trajs(opt_trajs.double(), map_designs.double(), self.mechanism)
      #self.show_maze(outputs[2][2][0])   
      loss = self.loss_fn(outputs[0], opt_trajs)
      self.log("train_loss", loss, prog_bar=True, logger=True)
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

          masks = torch.cat(masks, axis=1)
          pred_dist_maps = torch.empty_like(opt_dists)
          pred_dist_maps[:] = float('nan')
          loss_tot = 0.0
          rel_exps_maps = torch.empty_like(opt_dists)
          rel_exps_maps[:] = float('nan')
          for i in range(masks.shape[1]):
              start_maps = masks[:, i]
              x = map_designs, start_maps, goal_maps
              outputs = self.forward(x)
              opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism, self.device)
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
          hmean = (2*p_opt*p_exp)/(p_opt+p_exp)

          self.log("p_opt", p_opt, prog_bar=True, logger=True)
          self.log("p_suc", p_suc, prog_bar=True, logger=True)
          self.log("p_exp", p_exp, prog_bar=True, logger=True)
          self.log("loss_tot" , loss_tot, prog_bar=True, logger=True)
          self.log("loss" , loss, logger=True)
          self.log("hmean", hmean, logger=True, prog_bar=True)
          #trainer.save_checkpoint("recent_bbastar.pth")
          #wandb.save("recent_bbastar.pth")
            
  def test_step(self, test_batch, batch_idx):
      self.planner.model.eval()
      map_designs, goal_maps, opt_policies, opt_dists = test_batch

      if map_designs.dim() == 3:
                map_designs = map_designs.unsqueeze(1)

      wall_dist = torch.min(opt_dists)
      with torch.no_grad():
          save_file = True
          num_eval_points = 5 if save_file else 2  # save time for validation
          # Compute success and optimality
          masks = get_hard_medium_easy_masks(opt_dists, self.device, False, num_eval_points)

          masks = torch.cat(masks, axis=1)
          pred_dist_maps = torch.empty_like(opt_dists)
          pred_dist_maps[:] = float('nan')
          loss_tot = 0.0
          rel_exps_maps = torch.empty_like(opt_dists)
          rel_exps_maps[:] = float('nan')
          for i in range(masks.shape[1]):
              start_maps = masks[:, i]
              x = map_designs, start_maps, goal_maps
              outputs = self.forward(x)
              opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism, self.device)
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

          self.log("test_p_opt", p_opt, prog_bar=True, logger=True)
          self.log("test_p_suc", p_suc, prog_bar=True, logger=True)
          self.log("test_p_exp", p_exp, prog_bar=True, logger=True)
          self.log("test_loss_tot" , loss_tot, prog_bar=True, logger=True)
          self.log("test_loss" , loss, logger=True)

  def configure_optimizers(self):
      optimizer = torch.optim.RMSprop(self.planner.model.parameters(), lr=1e-3)
      return optimizer


class NeuralAstarModule(pl.LightningModule):

#   def __init__(self, g_ratio, encoder_backbone, dilate_gt , encoder_input):
#       super().__init__()
#       self.mechanism = Moore()
#       self.g_ratio = g_ratio
#       self.encoder_input = encoder_input
#       self.encoder_backbone = encoder_backbone
#       self.dilate_gt = dilate_gt
#       self.planner = combine_planner(self.mechanism, self.g_ratio, self.encoder_backbone, self.dilate_gt, self.encoder_input)
#       self.model = self.planner.model
#       self.skip_exp_when_training = False
#       self.astar_ref = self.planner.astar_ref
#       self.output_exp_instead_of_rel_exp = False
  def __init__(self, config):
      super().__init__()    
      self.mechanism = Moore()
      self.planner = combine_planner(self.mechanism, config)
      self.model = self.planner.model
      self.skip_exp_when_training = False
      self.astar_ref = self.planner.astar_ref
      self.output_exp_instead_of_rel_exp = False

  def show_maze(self, image):
      image = image.cpu().detach().numpy().squeeze()
      plt.matshow(image)
      plt.show()

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
      opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism, self.device)
      if self.dilate_gt:
          opt_trajs_ = dilate_opt_trajs(opt_trajs.double(), map_designs.double(), self.mechanism)
      else:
          opt_trajs = opt_trajs
      self.show_maze(outputs[2][2][0])   
      loss = self.loss_fn(outputs[0], opt_trajs)
      self.log("train_loss", loss, prog_bar=True, logger=True)
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

          masks = torch.cat(masks, axis=1)
          pred_dist_maps = torch.empty_like(opt_dists)
          pred_dist_maps[:] = float('nan')
          loss_tot = 0.0
          rel_exps_maps = torch.empty_like(opt_dists)
          rel_exps_maps[:] = float('nan')
          for i in range(masks.shape[1]):
              start_maps = masks[:, i]
              x = map_designs, start_maps, goal_maps
              outputs = self.forward(x)
              opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism, self.device)
              if self.dilate_gt:
                  opt_trajs_ = dilate_opt_trajs(opt_trajs.double(), map_designs.double(), self.mechanism)
              else:
                  opt_trajs = opt_trajs
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
          p_exp = 1 - p_exp
          hmean = (2*p_opt*p_exp)/(p_opt+p_exp)

          self.log("p_opt", p_opt, prog_bar=True, logger=True)
          self.log("p_suc", p_suc, prog_bar=True, logger=True)
          self.log("p_exp", p_exp, prog_bar=True, logger=True)
          self.log("loss_tot" , loss_tot, prog_bar=True, logger=True)
          self.log("loss" , loss, logger=True)
          #trainer.save_checkpoint("recent_bbastar.pth")
          self.log("hmean" , hmean, prog_bar=True, logger=True)
          #wandb.save("recent_bbastar.pth")
            
  def test_step(self, test_batch, batch_idx):
      self.planner.model.eval()
      map_designs, goal_maps, opt_policies, opt_dists = test_batch

      if map_designs.dim() == 3:
                map_designs = map_designs.unsqueeze(1)

      wall_dist = torch.min(opt_dists)
      with torch.no_grad():
          save_file = True
          num_eval_points = 5 if save_file else 2  # save time for validation
          # Compute success and optimality
          masks = get_hard_medium_easy_masks(opt_dists, self.device, False, num_eval_points)

          masks = torch.cat(masks, axis=1)
          pred_dist_maps = torch.empty_like(opt_dists)
          pred_dist_maps[:] = float('nan')
          loss_tot = 0.0
          rel_exps_maps = torch.empty_like(opt_dists)
          rel_exps_maps[:] = float('nan')
          for i in range(masks.shape[1]):
              start_maps = masks[:, i]
              x = map_designs, start_maps, goal_maps
              outputs = self.forward(x)
              opt_trajs = self.planner.get_opt_trajs(start_maps, goal_maps, opt_policies, self.mechanism, self.device)
              if self.dilate_gt:
                  opt_trajs_ = dilate_opt_trajs(opt_trajs.double(), map_designs.double(), self.mechanism)
              else:
                  opt_trajs = opt_trajs
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
          
          #print("pred_dist_maps_size", pred_dist_maps.size())
          path_len = pred_dist_maps.sum(dim=(1, 2, 3))
          print(path_len)
          path_len_opt = opt_trajs.sum(dim=(1,2,3))
          print(path_len_opt)
          path_len = torch.nan_to_num(path_len,1)
          path_len_opt = torch.nan_to_num(path_len_opt,1)
          path_opt_ratio = torch.div(path_len_opt,path_len)
          path_opt_ratio = path_opt_ratio.mean()
          #print("opt_traj_size", opt_trajs.size())
                    
          score = compute_bsmean_cbound(pred_dist_maps, rel_exps_maps, opt_dists, masks)
          p_opt_bsm = score[0][0]
          p_opt_lci = score[0][1]
          p_opt_uci = score[0][2]
          p_exp_bsm = score[1][0]
          p_exp_lci = score[1][1]
          p_exp_uci = score[1][2]
          hmean_bsm = score[2][0]
          hmean_lci = score[2][1]
          hmean_uci = score[2][2]

          self.log("path_opt_ratio", path_opt_ratio, prog_bar=True, logger=True)
          self.log("test_p_opt", p_opt, prog_bar=True, logger=True)
          self.log("test_p_suc_av", p_suc, prog_bar=True, logger=True)
          self.log("test_p_exp_av", p_exp, prog_bar=True, logger=True)
          self.log("test_p_opt_bsm", p_opt_bsm, prog_bar=True, logger=True)
          self.log("test_p_opt_lci", p_opt_lci, prog_bar=True, logger=True)
          self.log("test_p_opt_uci", p_opt_uci, prog_bar=True, logger=True)
          self.log("test_p_exp_bsm", p_exp_bsm, prog_bar=True, logger=True)
          self.log("test_p_exp_lci", p_exp_lci, prog_bar=True, logger=True)
          self.log("test_p_exp_uci", p_exp_uci, prog_bar=True, logger=True)
          self.log("test_hmean_bsm", hmean_bsm, prog_bar=True, logger=True)
          self.log("test_hmean_lci", hmean_lci, prog_bar=True, logger=True)
          self.log("test_hmean_uci", hmean_uci, prog_bar=True, logger=True)

          self.log("test_loss_tot" , loss_tot, prog_bar=True, logger=True)
          self.log("test_loss" , loss, logger=True)


  def configure_optimizers(self):
      optimizer = torch.optim.RMSprop(self.planner.model.parameters(), lr=1e-3)
      return optimizer
