import subprocess
subprocess.call('git clone -b main https://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@github.com/shreya-bhatt27/NeuralAstar-ported.git')
subprocess.call('git clone https://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@github.com/shreya-bhatt27/dataset_astar.git')
subprocess.call('wandb login eb94e0c9d64b72218420c8f40585ec650a663fa4')

subprocess.call('%cd NeuralAstar-ported/"Neural Astar Ported"/NeuralAstar')

