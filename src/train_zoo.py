from stable_baselines3 import PPO, TD3, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from pps import PredatorPreySwarmEnv
import supersuit as ss
import json

with open("config/train_params.json") as f:
    config = json.load(f)

# Create training environment
env = PredatorPreySwarmEnv(config)
max_ep_len = env._ep_len
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

# Create evaluation environment
eval_env = PredatorPreySwarmEnv(config)
eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
eval_env = ss.concat_vec_envs_v1(eval_env, 1, base_class="stable_baselines3")

# define the model
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# Define the evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=10_000,
                             deterministic=True, render=False)

# Train the model
# train_episodes = 200
# total_timesteps = train_episodes * max_ep_len
total_timesteps = 600_000 #TODO get better value
model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

# Save the model
model.save("models/DecPenPPO")