from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
import numpy as np


@dataclass
class Packet:
    flow: int
    arrival_time: int
    served_time: int | None = None


class AOISingleServerMultipleFlowsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, n_flows: int = 4, arrival_probs: float | list[float] = 0.2, serve_prob: float = 1, serve_order: str = "LIFO"):
        self.n_flows = n_flows
        if isinstance(arrival_probs, float):
            self.arrival_probs = [arrival_probs] * n_flows
        else:
            assert (len(arrival_probs) ==
                    self.n_flows), "arrival_probs should be a list of length n_flows"
            self.arrival_probs = arrival_probs
        self.serve_prob = serve_prob
        assert serve_order in [
            "FIFO", "LIFO"], "serve_order should be either 'FIFO' or 'LIFO'"
        self.serve_order = serve_order

        # Observations are lists of number of packets of each flow in the queue
        self.observation_space = spaces.Box(
            low=0, high=2**63 - 2, shape=(self.n_flows,), dtype=int)

        # We have n_flows + 1 actions
        # First n_flows actions are to select the flow to serve
        # Last action is to not serve any flow
        self.action_space = spaces.Discrete(self.n_flows + 1)

    def _get_obs(self):
        obs = []
        age = self._get_age()
        for flow in range(self.n_flows):
            obs += [age[flow], self._q[flow]]
        return np.array(obs)

    def _get_info(self):
        info = {
            "t": self._t,
            "buffer": self._buffer,
            "q": self._q,
            "u": self._u,
            "age": self._get_age()
        }
        return info

    def _get_age(self):
        return self._t - self._u

    def _arrivals_process(self):
        for flow in range(self.n_flows):
            if np.random.binomial(1, self.arrival_probs[flow]) == 1:
                logger.debug(f"Flow {flow} packet arrived at time {self._t}")
                self._buffer[flow].append(Packet(flow, self._t))
                self._q[flow] += 1  # One more packet in the queue

    def _serve_process(self, action: int):
        flow = action if action < self.n_flows else None
        if flow is not None:
            logger.debug(f"Serving flow {flow} packet at time {self._t}")
            if self._q[flow] == 0:
                logger.debug(
                    f"No packet to serve for flow {flow} at time {self._t}")
            elif np.random.rand() < self.serve_prob:
                logger.debug(f"Flow {flow} packet served at time {self._t}")
                if self.serve_order == "FIFO":
                    packet = self._buffer[flow].pop(0)
                    self._q[flow] -= 1
                    self._u[flow] = max(self._u[flow], packet.arrival_time)
                else:
                    packet = self._buffer[flow].pop(-1)
                    # TODO: FIX: Freshest packet in the queue changes to timestamp of the last in the queue
                    self._q[flow] -= 1
                    self._u[flow] = max(self._u[flow], packet.arrival_time)
            else:
                logger.debug(
                    f"Flow {flow} packet not served at time {self._t}. Packet retained in the queue.")
        else:
            logger.debug(f"Not serving any flow at time {self._t}")

    def _get_reward(self):
        # Average age of information
        age = self._get_age()
        reward = -np.sum(age)
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._t = 0
        self._buffer = [[] for _ in range(self.n_flows)]
        self._q = np.zeros(self.n_flows, dtype=int)
        self._u = np.zeros(self.n_flows, dtype=int)

        # Initial arrivals (first packet of each flow)
        for flow in range(self.n_flows):
            logger.debug(f"Flow {flow} packet arrived at time {self._t}")
            self._buffer[flow].append(Packet(flow, self._t))
            self._q[flow] += 1  # One more packet in the queue
        observation = self._get_obs()
        info = self._get_info()

        # TODO: Render

        return observation, info

    def step(self, action: int):
        # take action at t_n, get reward and make transition
        self._serve_process(action)
        reward = self._get_reward()
        self._arrivals_process()

        # move to step t_(n+1) and get observation
        self._t += 1

        observation = self._get_obs()
        info = self._get_info()

        terminated = False

        return observation, reward, terminated, False, info
