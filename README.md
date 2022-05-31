# IP Simple Pygame Fluid Sim
Utilising the Multi-Agent Reinforcement Learning with a simple particle based fluid simulator with the aim for accurate and fast calculation.

Implementing MADDPG to train particles to conform to physical laws by using the velocity vector field divergence. Rewards given for reducing divergence and lots of reward given if divergence is 0 (or very close). Each particle acts as its own agent with a shared policy.

State Space:
* Particle Velocity
* N Nearest particles distance, angle, and velocities (currently set to 5, is adjusted by self.k in multi_agent_main_game.py)

Action Space:
* Force in x direction
* Force in y direction

These two forces are used to adjust the velocity of the particle so that it avoids collisions etc

## Useage
* Adjust display_size.py to suite your monitor best
* Adjust any parameters in parameters.py for desired fluid properties
* Run torch_maddpg_run.py for training or run multi_agent_main_game.py to just see the particles

## Libraries
* Pygame
* Pytorch
* Scipy
* Itertools
* PIL
* OpenAI Gym
* Numpy
* Pickle
* Statistics
