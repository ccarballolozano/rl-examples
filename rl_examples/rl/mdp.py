from loguru import logger
import numpy as np


class MDP:
    """
    Markov Decision Process (MDP) class.
    """

    def __init__(self, states: list, actions: list, transitions: list[tuple], rewards: list[tuple]):
        """
        Initializes a Markov Decision Process (MDP) with the given parameters.

        Parameters:
        - states (list): List of states in the MDP.
        - actions (list): List of actions in the MDP.
        - transitions (list): List of tuples (state, action, next_state, prob) representing the transition probabilities.
        - rewards (list): List of tuples (state, action, next_state, reward) representing the rewards.

        Returns:
        - None
        """
        # TODO: Allow for general states and actions, not just consecutive integers starting from 0.
        assert all(i == states[i] for i in range(len(states))), "States must be a list of consecutive integers starting from 0."
        self.states = states
        assert all(i == actions[i] for i in range(len(actions))), "Actions must be a list of consecutive integers starting from 0."
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.transitions = self._build_transition_matrix(transitions)
        self.rewards = self._build_reward_matrix(rewards)

    def _build_transition_matrix(self, transitions: list):
        """
        Builds the transition matrix from the given list of transitions.

        Parameters:
        - transitions (list): List of tuples (state, action, next_state, prob) representing the transition probabilities.

        Returns:
        - P (ndarray): Transition matrix.
        """
        logger.debug(f"Building transition matrix from tuples: {transitions}.")
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i, (state, action, next_state, prob) in enumerate(transitions):
            P[state, action, next_state] = prob
        return P

    def _build_reward_matrix(self, rewards: list):
        """
        Builds the reward matrix from the given list of rewards.

        Parameters:
        - rewards (list): List of tuples (state, action, next_state, reward) representing the rewards.

        Returns:
        - R (ndarray): Reward matrix.
        """
        logger.debug(f"Building reward matrix from tuples: {rewards}.")
        R = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i, (state, action, next_state, reward) in enumerate(rewards):
            R[state, action, next_state] = reward
        return R

    def is_terminal_state(self, state):
        """
        Checks if the given state is a terminal state.

        Parameters:
        - state: The state to check.

        Returns:
        - bool: True if the state is a terminal state, False otherwise.
        """
        if sum(self.transitions[state, :, state]) == 1:
            return True
        else:
            return False

    def sample(self, state: int, action: int) -> tuple[int, float]:
        """
        Samples the next state and reward given the current state and action.

        Parameters:
        - state (int): The current state.
        - action (int): The action to take.
        """
        next_state = np.random.choice(
            self.n_states, p=self.transitions[state, action])
        reward = self.rewards[state, action, next_state]
        return next_state, reward
