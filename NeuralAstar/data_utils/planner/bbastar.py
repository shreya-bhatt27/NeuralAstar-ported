from newplanner import Unet
import torch
import torch.nn as nn
from data_utils.astar.differentiable_astar import DifferentiableAstar
from newplanner import combine_planner

class BBAstarFunc(torch.autograd.Function):
    """
    Implementation of Blackbox Backprop for A* Search
    """
    @staticmethod
    def forward(ctx,
                cost_maps,
                start_maps,
                goal_maps,
                obstacles_maps,
                histories,
                astar_instance,
                lambda_val=5.0):
        ctx.lambda_val = lambda_val
        ctx.astar = astar_instance
        ctx.save_for_backward(cost_maps, start_maps, goal_maps, obstacles_maps)
        ctx.histories = histories
        return ctx.histories

    @staticmethod
    def backward(ctx, grad_output):
        cost_maps, start_maps, goal_maps, obstacles_maps = ctx.saved_variables
        cost_prime = cost_maps + ctx.lambda_val * grad_output
        cost_prime[cost_prime < 0.0] = 0.0  # instead of max(weights, 0.0)
        with torch.no_grad():
            better_histories, better_paths = ctx.astar(cost_prime, start_maps,
                                                       goal_maps,
                                                       obstacles_maps)
        gradient = -(ctx.histories - better_histories) / ctx.lambda_val
        return gradient, None, None, None, None, None, None

class BBAstar(nn.Module):
    """
    Implementation of Black-box A* Search
    """
    def __init__(
        self,
        mechanism,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.astar = DifferentiableAstar(
            mechanism=mechanism,
            g_ratio=0.5,
            Tmax=0.25,
            detach_g=True,
        )
        self.bbastar = BBAstarFunc.apply
        self.encoder_input = 'm+'
        self.encoder_backbone = 'vgg16_bn'
        self.encoder_depth = 4
        self.encoder = self.encoder = Unet(len(self.encoder_input), self.encoder_backbone, self.encoder_depth)
        self.ignore_obstacles = True
        self.learn_obstacles = False
        self.bbastar_lambda = 20

        if self.learn_obstacles:
            print('WARNING: learn_obstacles has been set to True')

    def forward(self, map_designs, start_maps, goal_maps):
        inputs = map_designs
        if "+" in self.encoder_input:
            inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
        pred_cost_maps = self.encoder(
            inputs, map_designs if not self.ignore_obstacles else None)
        obstacles_maps = map_designs if not self.learn_obstacles else torch.ones_like(
            map_designs)

        with torch.no_grad():
            histories, paths = self.astar(pred_cost_maps, start_maps,
                                          goal_maps, obstacles_maps)

        histories = self.bbastar(pred_cost_maps, start_maps, goal_maps,
                                 obstacles_maps, histories, self.astar,
                                 self.bbastar_lambda)

        return histories, paths, pred_cost_maps

class BBAstarPlanner(combine_planner):
    def __init__(self,
                 mechanism,):
        super().__init__(mechanism)
        self.model = BBAstar(mechanism)
