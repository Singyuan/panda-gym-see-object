import gymnasium as gym
import panda_gym
import time 

# env = gym.make("PandaPickAndPlace-v3", render_mode="human")
# env = gym.make("PandaPickAndPlaceShot-v3", render_mode="human")
env = gym.make("PandaPickAndPlace6d-v3", render_mode="human")

observation, info = env.reset()

for _ in range(10):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)
    # if (terminated or truncated):
    #     observation, info = env.reset()

env.close()