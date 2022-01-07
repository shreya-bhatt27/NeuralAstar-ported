import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import numpy as np
import torch
import torch.utils.data as data

class MazeDataset(data.Dataset):
  def __init__(self, filename: str, dataset_type: str):
    super().__init__()
    assert filename.endswith("npz")  # Must be .npz format
    self.filename = filename
    self.dataset_type = dataset_type  # train, valid, test
    self.mazes, self.goal_maps, self.opt_policies, self.opt_dists = self._process(filename)
    self.num_actions = self.opt_policies.shape[1]
    self.num_orient = self.opt_policies.shape[2]

  def _process(self, filename: str):
    with np.load(filename) as f:
      dataset2idx = {"train": 0, "valid": 4, "test": 8}
      idx = dataset2idx[self.dataset_type]
      mazes = f["arr_" + str(idx)]
      goal_maps = f["arr_" + str(idx + 1)]
      opt_policies = f["arr_" + str(idx + 2)]
      opt_dists = f["arr_" + str(idx + 3)]

  # Set proper datatypes
    mazes = mazes.astype(np.float32)
    goal_maps = goal_maps.astype(np.float32)
    opt_policies = opt_policies.astype(np.float32)
    opt_dists = opt_dists.astype(np.float32)
    return mazes, goal_maps, opt_policies, opt_dists

  def __getitem__(self, index: int):
    maze = self.mazes[index]
    goal_map = self.goal_maps[index]
    opt_policy = self.opt_policies[index]
    opt_dist = self.opt_dists[index]

    return maze, goal_map, opt_policy, opt_dist

  def __len__(self):
    return self.mazes.shape[0]

class AstarDataModule(pl.LightningDataModule):
  def __init__(self, datafile):
    super().__init__()
    self.datafile = datafile
  def setup(self, stage): 
    print("setting up")    
  def train_dataloader(self):
    #self.setup('train')
    self.train_dataset = MazeDataset(self.datafile, "train")
    return torch.utils.data.DataLoader(self.train_dataset, batch_size=100, shuffle=True, num_workers=2)
  def val_dataloader(self):
    #self.setup('eval')
    self.val_dataset = MazeDataset(self.datafile, "valid")
    return torch.utils.data.DataLoader(self.val_dataset, batch_size=100, shuffle=False, num_workers=2)
  def test_dataloader(self):
    #self.setup('test')
    self.test_dataset = MazeDataset(self.datafile, "test")
    return torch.utils.data.DataLoader(self.test_dataset, batch_size=100, shuffle=False, num_workers=2)