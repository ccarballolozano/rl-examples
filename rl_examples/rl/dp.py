from loguru import logger
import numpy as np

from rl.mdp import MDP


def value_iteration(mdp: MDP, discount_factor=.95, theta=1e-6):
    V = np.zeros(mdp.n_states)
    policy = np.zeros([mdp.n_states, mdp.n_actions])
    k = 0
    while True:
        delta = 0
        V_k = V.copy()  # Use the same value function to update during a complete iteration.
        for s in range(mdp.n_states):
            v = V[s]
            V[s] = max(
                [sum(
                    [mdp.transitions[s, a, s_next] * (mdp.rewards[s, a, s_next] + discount_factor * V_k[s_next])
                     for s_next in range(mdp.n_states)])
                 for a in range(mdp.n_actions)])
            delta = max(delta, abs(v - V[s]))
        k += 1
        logger.debug(f"Iteration {k}, Delta: {delta}, Value function: {V}")
        if delta < theta:
            break
    for s in range(mdp.n_states):
        a_best = np.argmax(
            [sum(
                [mdp.transitions[s, a, next_s] * (mdp.rewards[s, a, next_s] + discount_factor * V[next_s])
                 for next_s in range(mdp.n_states)])
             for a in range(mdp.n_actions)])
        policy[s, a_best] = 1
    return policy, V
