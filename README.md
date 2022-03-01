# NeuralAstar-ported
# Introduction
This is the Pytorch Lightning implementation of the Neural A* Planner proposed in [Path Planning using Neural A* Search (ICML 2021)](https://arxiv.org/abs/2009.07476). 
The Neural A* Planner consists of an encoder and a Differentiable A* module. The Differentiable A* module is a reformulation of the canonical A* search to be differentiable. A problem instance consists of a map (a binary matrix indicating traversable and non-traversable regions) and start and goal node locations. The encoder encoder takes the problem instance as input and returns a guidance map. The guidance map is used in combination with different heuristics to generate a best path from the start point to the goal point by the Differentiable A* module. The loss calculated by comparing the final path obtained by the planner with the ground truth path is propagated through the A* search by the Differentiable A* module, and the model is trained to reduce the loss.
We have ported the code from the [original repository](https://github.com/omron-sinicx/neural-astar) in Pytorch to Pytorch Lightning.
# Usage
To create datafiles, run the following commands
```
git clone --recursive https://github.com/omron-sinicx/planning-datasets.git
cd planning-datasets
python3 -m venv venv
source venv/bin/activate
pip install -e .
sh 0_MP.sh
sh 1_TiledMP.sh
sh 2_CSM.sh
sh 3_SDD.sh
```

Alternatively, the data in the 'data' folder of this repository can also be used. 


# Results

# Acknowledgement




