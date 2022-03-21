# NeuralAstar-ported
# Introduction
This is the Pytorch Lightning implementation of the Neural A* Planner proposed in [Path Planning using Neural A* Search (ICML 2021)](https://arxiv.org/abs/2009.07476). 
The Neural A* Planner consists of an encoder and a Differentiable A* module (a reformulation of the canonical A* search to be differentiable).

A problem instance consists of a map (a binary matrix indicating traversable and non-traversable regions) and start and goal node locations. The encoder takes the problem instance as input and returns a guidance map which is used in combination with different heuristics to generate a best path from the start point to the goal point. The loss calculated by comparing the final path obtained by the planner with the ground truth path is propagated through the A* search by the Differentiable A* module, and the model is trained to reduce this loss.

We have ported the code from the [original repository](https://github.com/omron-sinicx/neural-astar) in Pytorch to Pytorch Lightning.
# Usage
1.Use Dockerfile to build the environment we have conducted our experiments in

2.Dataset Generation

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
The experiments can be recreated by running the notebooks in the run_experiments folder on Kaggle.

3.Run Instructions

To run the experiments first clone the repository and then navigate to the respective directories and run the following lines of code. To run a particular experiment, run the file named run<exp_name>.py
To install the required dependencies,
```
!pip install segmentation_models_pytorch
!pip install pytorch_lightning
!pip install wandb
!pip install bootstrapped
!pip install urllib3
```
Example: To run the Dropout experiment
```
!git clone https://github.com/Alrash/clone-anonymous4open
%cd clone-anonymous4open

#cloning our Neural A* implementation
!python pull.py --dir . --target https://anonymous.4open.science/r/NeuralAstar-ported-6EB0
    
!ls ../clone-anonymous4open/Dropout/
%cd ../clone-anonymous4open/Dropout/NeuralAstar

python3 rundropout.py
```
# Results

# Acknowledgement
