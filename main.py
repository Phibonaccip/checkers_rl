from enum import Enum
from gym import Env, spaces
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig


class Piece(Enum):
    Empty = 0,
    Black = 1,
    BlackKing = 2,
    White = 3,
    WhiteKing = 4,


class CheckersEnv(Env):
    def __init__(self, config):
        self.observation_space = spaces.MultiDiscrete(
            np.array([5, 32], dtype=Piece)
        )

        # [[(0,0), (),() ...], [...]]
        self.action_space = spaces.MultiDiscrete(
            np.array([32, 32], dtype=Piece)
        )


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="CartPole-v1")
    .framework("torch")
    .build()
)

for i in range(10):
    result = algo.train()
    print("Iteration: " + i)

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
