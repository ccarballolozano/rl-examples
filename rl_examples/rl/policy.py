import numpy as np


class Policy:
    def __init__(self, state_space: list, action_space: list):
        self.state_space = state_space
        self.action_space = action_space
        self.n_states = len(state_space)
        self.n_actions = len(action_space)
        self.policy = np.zeros([self.n_states, self.n_actions])

    def __str__(self):
        return f"Policy: {self.policy}"

    def __getitem__(self, key):
        return self.policy[key]
    
    def sample_action(self, state: int):
        return np.random.choice(self.action_space, p=self.policy[state])

    def get_action(self, state: int):
        return np.argmax(self.policy[state])

    def get_action_prob(self, state: int, action: int):
        return self.policy[state, action]
