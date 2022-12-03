import DQN
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from igt import IGT

def train():
    env = IGT()
    model = DQN(
        policy="MlpPolicy",
        batch_size=1,
        gamma=0.1,
        learning_starts=0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        env=env,
        seed=0,
    ).learn(total_timesteps=100)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)

    obs = env.reset()

if __name__ == "__main__":
    train()
