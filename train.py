from DQN import DQN
from sDQN import sDQN
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from igt import IGT
import random
def train():
    env = IGT()
    dqnDQNp = DQN(
        policy="MlpPolicy",
        batch_size=8,
        learning_rate=0.0001,
        learning_starts=0,
        env=env,
        seed=random.randint(0,10000),
    ).learn(total_timesteps=100)

    dqnSDQNp = DQN(
        policy="SDQNPolicy",
        batch_size=8,
        learning_rate=0.0001,
        learning_starts=0,
        env=env,
        seed=random.randint(0,10000),
    ).learn(total_timesteps=100)

    sdqnSDQNp = sDQN(
        policy="SDQNPolicy",
        batch_size=8,
        learning_rate=0.0001,
        learning_starts=0,
        env=env,
        seed=random.randint(0,10000),
    ).learn(total_timesteps=100)

    print(evaluate_policy(dqnDQNp, env, n_eval_episodes=500))
    print(evaluate_policy(dqnSDQNp, env, n_eval_episodes=500))

    obs = env.reset()

if __name__ == "__main__":
    train()
