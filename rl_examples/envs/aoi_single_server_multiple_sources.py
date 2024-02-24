import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AOISingleServerMultipleFlowsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, n_flows: int = 4, arrival_probs: float | list[float] = 0.2, serve_prob: float = 1):
        self.n_flows = n_flows
        if isinstance(arrival_probs, float):
            self.arrival_probs = [arrival_probs] * n_flows
        else:
            assert (len(arrival_probs) ==
                    self.n_flows), "arrival_probs should be a list of length n_flows"
            self.arrival_probs = arrival_probs
        self.serve_prob = serve_prob

        # Observations are lists of number of packets of each flow in the queue
        self.observation_space = spaces.Box(
            low=0, shape=(self.n_flows,), dtype=int)

        # We have n_flows + 1 actions
        # First n_flows actions are to select the flow to serve
        # Last action is to not serve any flow
        self.action_space = spaces.Discrete(self.n_flows + 1)

    def _get_obs(self):
        return self._buffer_lengths

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._buffer_lengths = np.zeros(self.n_flows, dtype=int)
        n_arrivals = np.random.binomial(1, self.arrival_probs)
        self._buffer_lengths += n_arrivals

        observation = self._get_obs()
        info = self._get_info()

        # TODO: Render

        return observation, info

    def step(self, action: int):
        # sirvo lo que hay, se va al siguiente paso (actualizar métricas), que empieza con la llegada de paquetes
        flow_to_serve = action if action < self.n_flows else None
        if flow_to_serve is not None and self._buffer_lengths[flow_to_serve] > 0 and np.random.rand() < self.serve_prob:
            self._buffer_lengths[flow_to_serve] -= 1

        # TODO: Update statistics / metrics

        n_arrivals = np.random.binomial(1, self.arrival_probs)
        self._buffer_lengths += n_arrivals

        # TODO: Add terminated and reward function
        terminated = False
        reward = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
