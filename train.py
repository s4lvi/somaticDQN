from DQN import DQN
from sDQN import sDQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from igt import IGT
import random
import numpy as np

def train():
    env = IGT()
    NUM_RUNS = 100


    dqnStats = np.ndarray([NUM_RUNS,2], dtype=np.float16)
    for i in range(NUM_RUNS):
        dqnDQNp = DQN(
            policy="MlpPolicy",
            learning_starts=0,
            env=env,
            seed=random.randint(0,10000)
        ).learn(total_timesteps=100)
        a, s = evaluate_policy(dqnDQNp, env, n_eval_episodes=10)
        dqnStats[i] = [a,s]
        env.reset()
    # print("DQN with MlpPolicy", np.mean(dqnStats, axis=0))
    
    dqn2Stats = np.ndarray([NUM_RUNS,2], dtype=np.float16)
    for i in range(NUM_RUNS):    
        dqnSDQNp = sDQN(
            policy="SDQNPolicy",
            learning_starts=0,
            env=env,
            seed=random.randint(0,10000),
        ).learn(total_timesteps=100)
        a, s = evaluate_policy(dqnSDQNp, env, n_eval_episodes=10)
        dqn2Stats[i] = [a,s]
        env.reset()
    print("DQN with MlpPolicy", np.mean(dqnStats, axis=0))
    print("DQN with SDQNPolicy", np.mean(dqn2Stats, axis=0))


if __name__ == "__main__":
    train()
