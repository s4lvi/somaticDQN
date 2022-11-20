import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy

from sDQN import sDQN
from DQN import DQN
from igt import IGT