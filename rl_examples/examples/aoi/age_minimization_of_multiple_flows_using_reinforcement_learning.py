import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from envs.aoi_single_server_multiple_sources import AOISingleServerMultipleFlowsEnv

gym.register(
    id='AOISingleServerMultipleFlowsEnv-v0',
    entry_point='envs.aoi_single_server_multiple_sources:AOISingleServerMultipleFlowsEnv',
    kwargs={}
)

env = gym.make('AOISingleServerMultipleFlowsEnv-v0',
               n_flows=4, arrival_probs=0.2, serve_prob=1, serve_order="LIFO")


def get_maf_action(obs) -> int:
    # obs = [age1, q1, age2, q2, ..., ageN, qN]
    ages = obs[::2]
    max_age = np.max(ages)
    max_age_flows = np.argwhere(ages == max_age).flatten()
    action = np.random.choice(max_age_flows)
    return action


rewards = []
ages = []

obs, info = env.reset()
logger.debug(f"obs: {obs}")
for i in range(10000):
    action = get_maf_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    logger.debug(f"action: {action}, reward: {reward}, observation: {obs}")
    rewards.append(reward)
    ages.append(info["age"])

        #print(action, obs, reward, terminated, info)
    if terminated:
        break

ages = np.array(ages)
time_average_aoi = ages.cumsum(axis=0) / np.arange(1, len(ages) + 1).reshape((-1, 1))
f, ax = plt.subplots()
plt.plot(time_average_aoi)
plt.show()

print(0)