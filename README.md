# Vortex for Alpha Zero
An agent for playing the game Vortex based on a Graph Network and DeepMind's AlphaZero algorithm.

Vortex is the game [Hex](https://en.wikipedia.org/wiki/Hex_(board_game)), generalised to a graph. In order to play Vortex effectively, the agent must balance several risks. Selecting vertices too closely together gives the opponent freedom to form their connection. However, selecting vertices with gaps that cannot be defended, provides the opponent with an opportunity to block the path.
Deep Reinforcement Learning agents such as AlphaZero have demonstrated super-human levels of skill in complex games. However, the techniques employed by AI game agents rarely generalise to real world challenges. In part this is may be due to a dependence on the artificially regular structure and dimensions of grid-based game boards. 

Vortex provides a novel challenge in this respect, due to both the irregular structure of the graph that it is played on, and the fact that the Vortex graph is generated differently for each game. Each graph is connected differently and may vary widely in scale, containing a different number of vertices and edges. A Vortex playing agent must learn strategic patterns without the scaffold of a regular spatial framework such as the grid structure of games like Chess and Go. Furthermore it must learn underlying patterns of vertex connectivity that generalise to random boards rather than fitting to a fixed structure.

## Installing dependencies

For GPU:

`conda env create -f environment.yml`

For CPU:

`conda env create -f environment_cpu.yml`

Training without a GPU is likely to be rather slow, however if this is intended, the config file used in training must be apdated to diable GPU as follows:

`"cuda": false`

## Training the agent

`python main.py configs/vortex-5-20-gpu.json`

## Playing against the trained agent in a GUI

`python hex_gui.py`