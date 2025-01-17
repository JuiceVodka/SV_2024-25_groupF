from stable_baselines3 import PPO, DDPG
from pps import PredatorPreySwarmEnv
import supersuit as ss

import json

with open("config/eval_params.json") as f:
    config = json.load(f)

# Create the environment
env = PredatorPreySwarmEnv(config)
max_ep_len = env._ep_len
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

# Load the saved model
model = PPO.load("models/newTask_simpleAvoidV3.zip")

# Use the model for inference
obs = env.reset()
images = []
for _ in range(max_ep_len):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    img = env.render(mode="rgb_array")
    images.append(img)


# Save images as a GIF
images = images[1:]
print(len(images))
import imageio
imageio.mimsave('isolate.gif', images, fps=30)