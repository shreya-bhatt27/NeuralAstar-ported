# NeuralAstar-ported
Pytorch Lightning implementation of Neural Astar planner (Unet++ branch)

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
