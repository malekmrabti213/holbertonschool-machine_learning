import matplotlib.pyplot as plt
import gym
import numpy as np
import math
import tensorflow as tf
import reinforcement_learning as rl
env_name = 'Breakout-v0'
# env_name = 'SpaceInvaders-v0'
rl.checkpoint_base_dir = 'checkpoints_tutorial16/'
rl.update_paths(env_name=env_name)